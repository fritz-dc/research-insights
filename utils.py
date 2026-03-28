from pathlib import Path
from datetime import datetime
import json
import os
import re
import unicodedata

import httpx
import pandas as pd
import yaml
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent

# ── Config & I/O ──────────────────────────────────────────────────────────────


def load_cfg(path=None):
    """Load params.yaml.

    Resolution order when path is omitted:
    1. {ROOT}/params.yaml
    2. {ROOT}/CONFIG/params.yaml

    This preserves compatibility with both the refactor package layout and the
    earlier repository layout.
    """
    if path is not None:
        cfg_path = Path(path)
    else:
        candidates = [
            ROOT / "params.yaml",
            ROOT / "CONFIG" / "params.yaml",
        ]
        cfg_path = next((p for p in candidates if p.exists()), candidates[0])

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def build_output_path(subdir: str, fname: str, groupby_field: str = None, run_date: str = None, root: Path = ROOT) -> Path:
    """Resolve an OUTPUTS path and create parent directories.

    Without groupby_field/run_date (notebooks 01, 01.5):
        OUTPUTS/{subdir}/{fname}
    With groupby_field and run_date (notebook 02):
        OUTPUTS/{groupby_field}/{run_date}/{subdir}/{fname}
    """
    root = Path(root)
    if groupby_field and run_date:
        p = root / "OUTPUTS" / groupby_field / run_date / subdir / fname
    else:
        p = root / "OUTPUTS" / subdir / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    return p



def outpath(subdir, fname, root=ROOT, groupby_field=None, run_date=None):
    """Backward-compatible alias for build_output_path()."""
    return build_output_path(
        subdir,
        fname,
        groupby_field=groupby_field,
        run_date=run_date,
        root=root,
    )



def get_run_date() -> str:
    """Return today's date as YYYY-MM-DD for output path nesting."""
    return datetime.now().strftime("%Y-%m-%d")



def ingest(path):
    """Load SQL extract CSV; parse comma-separated LISTAGG token string into list."""
    raw = pd.read_csv(path)
    date_cols = [c for c in ["posted_date", "funded_date"] if c in raw.columns]
    df = pd.read_csv(path, parse_dates=date_cols) if date_cols else raw
    df["tokens"] = (
        df["tokens"].fillna("").str.split(",").apply(lambda ts: [t.strip() for t in ts if t.strip()])
    )
    return df



def get_llm_client() -> OpenAI:
    """Build and return an OpenAI client with SSL verification disabled.

    verify=False is required for the proxy environment. This is intentional
    and not configurable — it reflects a fixed infrastructure constraint.
    Raises ValueError if OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))


# ── Token helpers ─────────────────────────────────────────────────────────────


def tokens_to_str(token_list):
    """Token list → space-joined string for sklearn."""
    return " ".join(token_list) if token_list is not None else ""



def flat_freq(df, col="tokens"):
    """Corpus-wide token frequency Series."""
    return pd.Series([t for ts in df[col] for t in ts]).value_counts()



def token_doc_freq(df):
    """Distinct project count per token — via explode."""
    return (
        df[["project_id", "tokens"]]
        .explode("tokens")
        .rename(columns={"tokens": "token"})
        .drop_duplicates()
        .groupby("token")["project_id"]
        .nunique()
        .rename("doc_count")
    )


# ── Analysis helpers ──────────────────────────────────────────────────────────


def make_vec(min_df, max_df, ngram_range):
    """Shared TF-IDF vectorizer factory."""
    return TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b[a-z][a-z_\-]*\b",
    )



def add_bin(df, bins):
    """Label each project with the first matching analyst-defined bin, else None."""
    df = df.copy()
    df["bin"] = None
    for b in bins:
        mask = (df["posted_date"] >= b["start"]) & (df["posted_date"] <= b["end"])
        df.loc[mask & df["bin"].isna(), "bin"] = b["name"]
    return df



def group_key(keys, group_cols):
    """Normalise groupby key to dict for scalar or tuple keys."""
    return dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))



def build_project_topic_bridge(
    weights_df: pd.DataFrame,
    groupby_field: str,
    threshold: float,
) -> pd.DataFrame:
    """Build project-topic bridge using topic_share threshold.

    topic_share = W[project, topic] / sum_k(W[project, k])
    where the sum is over ALL topics for that project within its group.

    topic_key format: '{groupby_field}={group_value}|topic={topic_id}'
    This is the exact format expected by get_project_ids() in 02.

    Args:
        weights_df: FULL DataFrame with columns [groupby_field, topic_id,
            project_id, weight, rank] covering every project-topic weight in
            each retained group before any top-k, representative-project, or
            analysis-only filtering.
        groupby_field: name of the groupby column (e.g. 'project_category')
        threshold: minimum topic_share for inclusion

    Returns:
        DataFrame with columns [topic_key, project_id, groupby_field, topic_id,
                                weight, topic_share]
    """
    required_cols = {groupby_field, "topic_id", "project_id", "weight"}
    missing = required_cols - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df missing required columns: {sorted(missing)}")

    totals = (
        weights_df.groupby([groupby_field, "project_id"])["weight"]
        .sum()
        .rename("total_weight")
        .reset_index()
    )

    merged = weights_df.merge(
        totals,
        on=[groupby_field, "project_id"],
        how="left",
        validate="many_to_one",
    )

    merged["topic_share"] = 0.0
    nonzero = merged["total_weight"].fillna(0) > 0
    merged.loc[nonzero, "topic_share"] = (
        merged.loc[nonzero, "weight"] / merged.loc[nonzero, "total_weight"]
    )

    linked = merged[merged["topic_share"] >= threshold].copy()
    linked["topic_key"] = (
        groupby_field
        + "="
        + linked[groupby_field].astype(str)
        + "|topic="
        + linked["topic_id"].astype(str)
    )

    return linked[
        ["topic_key", "project_id", groupby_field, "topic_id", "weight", "topic_share"]
    ]



def slugify_group_value(value: str, max_len: int = 64) -> str:
    """Convert an arbitrary group value to a safe filename component.

    Steps:
    1. Normalize unicode to ASCII (NFKD + encode/decode)
    2. Replace any non-alphanumeric character with underscore
    3. Collapse repeated underscores
    4. Strip leading/trailing underscores
    5. Truncate to max_len (default 64) to avoid filesystem limits

    Returns 'unknown' for empty or fully non-ASCII inputs.
    """
    value = str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w]", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    return value[:max_len] if value else "unknown"


# ── Quality report ────────────────────────────────────────────────────────────

HARD_STOPWORDS = {
    "the", "and", "for", "with", "are", "was", "were", "been", "this", "that", "from",
    "they", "them", "their", "will", "would", "should", "could", "have", "has", "had",
    "use", "using", "used", "make", "makes", "making", "get", "gets", "getting",
    "help", "helps", "project", "students", "student", "classroom", "learning",
    "school", "teacher", "teachers", "education", "grade", "grades", "materials",
    "supplies", "tools", "resources", "funded", "funding", "donors", "donor",
}


def quality_report(df, label, doc_freq=None, matrices=None, save_path=None):
    freq = flat_freq(df)
    stops = [t for t in freq.head(200).index if t in HARD_STOPWORDS]
    stats = {
        "checkpoint": label,
        "timestamp": datetime.now().isoformat(),
        "n_projects": len(df),
        "token_count_distribution": df["tokens"].apply(len).describe().to_dict(),
        "vocab": {
            "unique": int(len(freq)),
            "total": int(freq.sum()),
            "stopword_violations": stops,
        },
        "gates": {"no_stopwords": not stops, "violations": stops},
    }
    if doc_freq is not None and len(doc_freq):
        stats["doc_freq"] = {
            "retained": int(len(doc_freq)),
            "min": int(doc_freq.min()),
            "max": int(doc_freq.max()),
            "median": float(doc_freq.median()),
        }
    if matrices:
        stats["matrices"] = {
            k: {
                "shape": list(X.shape),
                "nnz": int(X.nnz),
                "sparsity": round(1 - X.nnz / (X.shape[0] * X.shape[1] or 1), 4),
            }
            for k, X in matrices.items()
        }
    tc = stats["token_count_distribution"]
    print(f"\n{'=' * 55}  [{label}]")
    print(f"  Projects : {stats['n_projects']:,}")
    print(f"  Tok/proj : min={tc['min']:.0f}  p50={tc['50%']:.0f}  max={tc['max']:.0f}")
    print(f"  Vocab    : {stats['vocab']['unique']:,} unique tokens")
    if matrices:
        for k, m in stats["matrices"].items():
            print(f"  {k:20s}: shape={m['shape']}  sparsity={m['sparsity']:.3f}")
    print(f"  Stopwords: {'PASS' if not stops else 'FAIL — ' + str(stops)}")
    print(f"{'=' * 55}\n")
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
    return stats


__all__ = [
    "ROOT",
    "load_cfg",
    "build_output_path",
    "outpath",
    "get_run_date",
    "ingest",
    "get_llm_client",
    "tokens_to_str",
    "flat_freq",
    "token_doc_freq",
    "make_vec",
    "add_bin",
    "group_key",
    "build_project_topic_bridge",
    "slugify_group_value",
    "quality_report",
]
