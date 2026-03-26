from pathlib import Path
from datetime import datetime
import json, yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent

# ── Config & I/O ──────────────────────────────────────────────────────────────

def load_cfg(path=None):
    with open(path or ROOT / "CONFIG/params.yaml") as f:
        return yaml.safe_load(f)

def outpath(subdir, fname, root=ROOT):
    p = root / "OUTPUTS" / subdir / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def ingest(path):
    """Load SQL extract CSV; parse comma-separated LISTAGG token string into list."""
    raw = pd.read_csv(path)
    date_cols = [c for c in ["posted_date", "funded_date"] if c in raw.columns]
    df = pd.read_csv(path, parse_dates=date_cols) if date_cols else raw
    df["tokens"] = (df["tokens"].fillna("").str.split(",")
                     .apply(lambda ts: [t.strip() for t in ts if t.strip()]))
    return df

# ── Token helpers ─────────────────────────────────────────────────────────────

def tokens_to_str(token_list):
    """Token list → space-joined string for sklearn."""
    return " ".join(token_list) if token_list is not None else ""

def flat_freq(df, col="tokens"):
    """Corpus-wide token frequency Series."""
    return pd.Series([t for ts in df[col] for t in ts]).value_counts()

def token_doc_freq(df):
    """Distinct project count per token — via explode."""
    return (df[["project_id", "tokens"]]
              .explode("tokens").rename(columns={"tokens": "token"})
              .drop_duplicates()
              .groupby("token")["project_id"].nunique()
              .rename("doc_count"))

# ── Analysis helpers ──────────────────────────────────────────────────────────

def make_vec(min_df, max_df, ngram_range):
    """Shared TF-IDF vectorizer factory."""
    return TfidfVectorizer(min_df=min_df, max_df=max_df,
                           ngram_range=ngram_range,
                           token_pattern=r"(?u)\b[a-z][a-z_\-]*\b")

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

# ── Quality report ────────────────────────────────────────────────────────────

HARD_STOPWORDS = {
    "the","and","for","with","are","was","were","been","this","that","from",
    "they","them","their","will","would","should","could","have","has","had",
    "use","using","used","make","makes","making","get","gets","getting",
    "help","helps","project","students","student","classroom","learning",
    "school","teacher","teachers","education","grade","grades","materials",
    "supplies","tools","resources","funded","funding","donors","donor",
}

def quality_report(df, label, doc_freq=None, matrices=None, save_path=None):
    freq  = flat_freq(df)
    stops = [t for t in freq.head(200).index if t in HARD_STOPWORDS]
    stats = {
        "checkpoint":               label,
        "timestamp":                datetime.now().isoformat(),
        "n_projects":               len(df),
        "token_count_distribution": df["tokens"].apply(len).describe().to_dict(),
        "vocab":  {"unique": int(len(freq)), "total": int(freq.sum()),
                   "stopword_violations": stops},
        "gates":  {"no_stopwords": not stops, "violations": stops},
    }
    if doc_freq is not None and len(doc_freq):
        stats["doc_freq"] = {"retained": int(len(doc_freq)),
                             "min": int(doc_freq.min()), "max": int(doc_freq.max()),
                             "median": float(doc_freq.median())}
    if matrices:
        stats["matrices"] = {
            k: {"shape": list(X.shape), "nnz": int(X.nnz),
                "sparsity": round(1 - X.nnz / (X.shape[0] * X.shape[1] or 1), 4)}
            for k, X in matrices.items()
        }
    tc = stats["token_count_distribution"]
    print(f"\n{'='*55}  [{label}]")
    print(f"  Projects : {stats['n_projects']:,}")
    print(f"  Tok/proj : min={tc['min']:.0f}  p50={tc['50%']:.0f}  max={tc['max']:.0f}")
    print(f"  Vocab    : {stats['vocab']['unique']:,} unique tokens")
    if matrices:
        for k, m in stats["matrices"].items():
            print(f"  {k:20s}: shape={m['shape']}  sparsity={m['sparsity']:.3f}")
    print(f"  Stopwords: {'PASS' if not stops else 'FAIL — ' + str(stops)}")
    print(f"{'='*55}\n")
    if save_path:
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
    return stats
