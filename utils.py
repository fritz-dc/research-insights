
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import re
import unicodedata
from typing import Any

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


def write_json(path: Path, payload: Any) -> Path:
    """Deterministic JSON writer used across manifests and structured outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    """Return true content MD5 for a file, or None if the file does not exist."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_meta(path: Path, label: str | None = None) -> dict[str, Any]:
    """Build provenance metadata for one artifact path."""
    path = Path(path)
    exists = path.exists()
    return {
        "label": label or path.name,
        "path": str(path),
        "exists": exists,
        "md5": compute_md5(path) if exists and path.is_file() else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def build_output_path(
    subdir: str,
    fname: str,
    groupby_field: str = None,
    run_date: str = None,
    root: Path = ROOT,
) -> Path:
    """Resolve a canonical OUTPUTS path and create parent directories.

    Without groupby_field/run_date (notebooks 01, 02):
        OUTPUTS/{subdir}/{fname}
    With groupby_field and run_date (legacy grouped outputs):
        OUTPUTS/{groupby_field}/{run_date}/{subdir}/{fname}
    """
    root = Path(root)
    if groupby_field and run_date:
        p = root / "OUTPUTS" / groupby_field / run_date / subdir / fname
    else:
        p = root / "OUTPUTS" / subdir / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def build_run_output_path(
    subdir: str,
    fname: str,
    groupby_field: str,
    run_date: str,
    run_id: str,
    root: Path = ROOT,
) -> Path:
    """Resolve a run-scoped OUTPUTS path for Notebook 03."""
    root = Path(root)
    p = root / "OUTPUTS" / "runs" / str(groupby_field) / str(run_date) / str(run_id) / subdir / fname
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


def canonicalize_filter_spec(filter_logic: str, filters: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return the canonical filter-spec object used for run hashing/logging."""
    filters = filters or []
    canonical_filters = []
    for item in filters:
        if not isinstance(item, dict):
            raise ValueError(f"Filter entries must be dicts, got {type(item)}")
        canonical_filters.append({k: item[k] for k in sorted(item.keys())})
    return {
        "schema_version": "v1",
        "filter_logic": filter_logic,
        "filters": canonical_filters,
    }


def get_filter_fields_key(filters: list[dict[str, Any]] | None) -> str:
    """Concatenate referenced filter fields, or 'none' when no filters are supplied."""
    filters = filters or []
    fields = sorted({str(f.get("field", "")).strip() for f in filters if str(f.get("field", "")).strip()})
    return "none" if not fields else "__".join(fields)


def get_run_id(groupby_field: str, filter_spec: dict[str, Any] | None = None) -> str:
    """Build a readable run id with scope hash."""
    scope = json.dumps(filter_spec or {"filters": [], "filter_logic": "and"}, sort_keys=True, separators=(",", ":"))
    scope_hash = hashlib.md5(scope.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{groupby_field}_{scope_hash}"


def validate_filter_spec(df: pd.DataFrame, filter_logic: str, filters: list[dict[str, Any]] | None) -> None:
    """Validate analysis filter configuration against the dataframe schema."""
    if filter_logic != "and":
        raise ValueError("analysis.filter_logic must be exactly 'and'")
    filters = filters or []
    for rule in filters:
        if "field" not in rule:
            raise ValueError(f"Filter rule missing 'field': {rule}")
        field = rule["field"]
        if field not in df.columns:
            raise ValueError(f"Filter field '{field}' not found in dataframe columns")
        op = rule.get("op")
        if op not in {"eq", "in", "range", "is_null", "not_null"}:
            raise ValueError(f"Unsupported filter op '{op}' in rule: {rule}")
        if op == "eq" and "value" not in rule:
            raise ValueError(f"eq filter missing 'value': {rule}")
        if op == "in":
            values = rule.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"in filter requires non-empty 'values': {rule}")
        if op == "range":
            if "min" not in rule and "max" not in rule:
                raise ValueError(f"range filter requires 'min' and/or 'max': {rule}")


def _coerce_bound_for_series(series: pd.Series, value: Any) -> Any:
    """Coerce range bounds for datetime columns only; forbid other implicit coercions."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value)
    return value


def apply_filters(
    df: pd.DataFrame,
    filter_logic: str,
    filters: list[dict[str, Any]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply validated analysis filters and return the filtered dataframe plus summary."""
    filters = filters or []
    validate_filter_spec(df, filter_logic, filters)
    if not filters:
        return df.copy(), {
            "filter_logic": filter_logic,
            "filters": [],
            "n_rules": 0,
            "input_row_count": int(len(df)),
            "output_row_count": int(len(df)),
            "dropped_row_count": 0,
            "retained_pct": 100.0 if len(df) else 0.0,
            "fields_checked": [],
            "no_rows_after_filter": False,
            "filter_fields_key": "none",
        }

    mask = pd.Series(True, index=df.index)
    for rule in filters:
        field = rule["field"]
        op = rule["op"]
        series = df[field]

        if op == "eq":
            rule_mask = series == rule["value"]
        elif op == "in":
            rule_mask = series.isin(rule["values"])
        elif op == "range":
            rule_mask = pd.Series(True, index=df.index)
            if "min" in rule:
                rule_mask &= series >= _coerce_bound_for_series(series, rule["min"])
            if "max" in rule:
                rule_mask &= series <= _coerce_bound_for_series(series, rule["max"])
        elif op == "is_null":
            rule_mask = series.isna()
        elif op == "not_null":
            rule_mask = series.notna()
        else:
            raise ValueError(f"Unsupported filter op '{op}'")
        mask &= rule_mask.fillna(False)

    out = df.loc[mask].copy()
    retained_pct = round((len(out) / len(df) * 100), 2) if len(df) else 0.0
    return out, {
        "filter_logic": filter_logic,
        "filters": filters,
        "n_rules": len(filters),
        "input_row_count": int(len(df)),
        "output_row_count": int(len(out)),
        "dropped_row_count": int(len(df) - len(out)),
        "retained_pct": retained_pct,
        "fields_checked": [rule["field"] for rule in filters],
        "no_rows_after_filter": out.empty,
        "filter_fields_key": get_filter_fields_key(filters),
    }


def ingest(path):
    """Load SQL extract CSV; parse comma-separated LISTAGG token string into list."""
    raw = pd.read_csv(path)
    date_cols = [c for c in ["posted_date", "funded_date"] if c in raw.columns]
    df = pd.read_csv(path, parse_dates=date_cols) if date_cols else raw
    df["tokens"] = (
        df["tokens"].fillna("").str.split(",").apply(lambda ts: [t.strip() for t in ts if t.strip()])
    )
    return df


def ensure_warning_file(path: Path) -> Path:
    """Create an empty JSONL warnings file if it does not already exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return path


def append_warning(
    path: Path,
    stage_name: str,
    code: str,
    message: str,
    severity: str = "warning",
    context: dict[str, Any] | None = None,
) -> None:
    """Append one structured warning record to a JSONL file."""
    ensure_warning_file(path)
    record = {
        "timestamp": datetime.now().isoformat(),
        "stage_name": stage_name,
        "severity": severity,
        "code": code,
        "message": message,
        "context": context or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def get_warning_count(path: Path) -> int:
    """Count non-empty lines in a JSONL warnings file."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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


def start_stage_manifest(
    stage_name: str,
    notebook_file: str,
    config_path: str = "params.yaml",
    run_id: str | None = None,
    group_by_field: str | None = None,
    filter_fields_key: str | None = None,
) -> dict[str, Any]:
    """Create the common stage-manifest skeleton shared by all notebooks."""
    config_meta = artifact_meta(ROOT / config_path, label="config")
    started_at = datetime.now().isoformat()
    return {
        "schema_version": "v1",
        "run_id": run_id,
        "group_by_field": group_by_field,
        "filter_fields_key": filter_fields_key,
        "stage_name": stage_name,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "duration_seconds": None,
        "notebook_file": notebook_file,
        "config_path": config_path,
        "config_md5": config_meta["md5"],
        "input_artifacts": [],
        "output_artifacts": [],
        "row_counts": {},
        "key_params": {},
        "warnings_count": 0,
        "warnings_path": None,
    }


def finalize_stage_manifest(
    manifest: dict[str, Any],
    output_path: Path,
    status: str,
    input_artifacts: list[dict[str, Any]] | None = None,
    output_artifacts: list[dict[str, Any]] | None = None,
    row_counts: dict[str, Any] | None = None,
    key_params: dict[str, Any] | None = None,
    warnings_path: Path | None = None,
) -> dict[str, Any]:
    """Finalize and persist a stage manifest."""
    completed_at = datetime.now()
    started_at = datetime.fromisoformat(manifest["started_at"])
    manifest["status"] = status
    manifest["completed_at"] = completed_at.isoformat()
    manifest["duration_seconds"] = round((completed_at - started_at).total_seconds(), 2)
    manifest["input_artifacts"] = input_artifacts or []
    manifest["output_artifacts"] = output_artifacts or []
    manifest["row_counts"] = row_counts or {}
    manifest["key_params"] = key_params or {}
    if warnings_path is not None:
        manifest["warnings_path"] = str(warnings_path)
        manifest["warnings_count"] = get_warning_count(warnings_path)
    write_json(Path(output_path), manifest)
    return manifest


def build_pipeline_manifest(
    output_path: Path,
    run_id: str,
    run_date: str,
    group_by_field: str,
    filter_spec_path: Path,
    filter_summary_path: Path,
    stage_manifest_paths: list[Path],
    warnings_01_path: Path,
    warnings_02_path: Path,
    warnings_03_path: Path,
    final_outputs: dict[str, str],
    config_path: str = "params.yaml",
    filter_fields_key: str | None = None,
    status: str = "success",
) -> dict[str, Any]:
    """Persist the Notebook 03 pipeline manifest."""
    config_meta = artifact_meta(ROOT / config_path, label="config")
    payload = {
        "schema_version": "v1",
        "run_id": run_id,
        "group_by_field": group_by_field,
        "filter_fields_key": filter_fields_key,
        "run_date": run_date,
        "status": status,
        "created_at": datetime.now().isoformat(),
        "config_path": config_path,
        "config_md5": config_meta["md5"],
        "filter_spec_path": str(filter_spec_path),
        "filter_summary_path": str(filter_summary_path),
        "stage_manifests": [str(Path(p)) for p in stage_manifest_paths],
        "warnings_01_path": str(warnings_01_path),
        "warnings_02_path": str(warnings_02_path),
        "warnings_03_path": str(warnings_03_path),
        "warnings_files": [str(warnings_01_path), str(warnings_02_path), str(warnings_03_path)],
        "final_outputs": final_outputs,
    }
    write_json(output_path, payload)
    return payload


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


def get_effective_support_volume_scale(
    df: pd.DataFrame,
    groupby_field: str,
    confidence_cfg: dict[str, Any],
) -> dict[str, float | int]:
    """Derive the runtime denominator for support-volume scoring.

    Uses:
        effective_scale = round(support_volume_scale_median_size * median_group_size)

    Returns a dict with:
        - scale_multiplier
        - median_group_size
        - effective_scale

    Guarantees effective_scale >= 1 to avoid divide-by-zero.
    """
    if "support_volume_scale_median_size" not in confidence_cfg:
        raise KeyError(
            "Missing analysis.confidence.support_volume_scale_median_size in params.yaml"
        )

    try:
        scale_multiplier = float(confidence_cfg["support_volume_scale_median_size"])
    except Exception as e:
        raise ValueError(
            "analysis.confidence.support_volume_scale_median_size must be numeric"
        ) from e

    if pd.isna(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError(
            "analysis.confidence.support_volume_scale_median_size must be > 0"
        )

    required_cols = {groupby_field, "project_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot derive support-volume scale; df missing required columns: {sorted(missing)}"
        )

    group_project_counts = (
        df[[groupby_field, "project_id"]]
        .dropna(subset=["project_id"])
        .drop_duplicates()
        .groupby(groupby_field, dropna=False)["project_id"]
        .nunique()
    )

    median_group_size = (
        float(group_project_counts.median()) if not group_project_counts.empty else 0.0
    )

    effective_scale = max(1, int(round(scale_multiplier * median_group_size)))

    return {
        "scale_multiplier": scale_multiplier,
        "median_group_size": median_group_size,
        "effective_scale": effective_scale,
    }


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
    """Build project-topic bridge using topic_share threshold."""
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
    """Convert an arbitrary group value to a safe filename component."""
    value = str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w]", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    return value[:max_len] if value else "unknown"


def load_essay_snippet_lookup(
    project_ids: list[Any],
    data_dir: Path | None = None,
    max_chars: int = 300,
) -> dict[Any, str]:
    """Lazy-load essay text snippets for a project_id set from DATA/project_essay*.csv."""
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "DATA"
    needed = set(project_ids)
    if not needed:
        return {}
    lookup: dict[Any, str] = {}
    essay_files = sorted(data_dir.glob("project_essay*.csv"))
    text_cols = ["essay", "essay_text", "full_text", "project_essay", "text"]
    for fpath in essay_files:
        try:
            cols = pd.read_csv(fpath, nrows=0).columns.tolist()
        except Exception:
            continue
        available_text_col = next((c for c in text_cols if c in cols), None)
        if not available_text_col or "project_id" not in cols:
            continue
        usecols = ["project_id", available_text_col]
        for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=200000):
            sub = chunk[chunk["project_id"].isin(needed - set(lookup.keys()))]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                text = str(row.get(available_text_col, "") or "")
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    lookup[row["project_id"]] = text[:max_chars]
            if len(lookup) == len(needed):
                return lookup
    return lookup


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
        write_json(save_path, stats)
    return stats


__all__ = [
    "ROOT",
    "load_cfg",
    "write_json",
    "compute_md5",
    "artifact_meta",
    "build_output_path",
    "build_run_output_path",
    "outpath",
    "get_run_date",
    "canonicalize_filter_spec",
    "get_filter_fields_key",
    "get_run_id",
    "validate_filter_spec",
    "apply_filters",
    "ingest",
    "ensure_warning_file",
    "append_warning",
    "get_warning_count",
    "get_llm_client",
    "start_stage_manifest",
    "finalize_stage_manifest",
    "build_pipeline_manifest",
    "tokens_to_str",
    "flat_freq",
    "token_doc_freq",
    "make_vec",
    "add_bin",
    "group_key",
    "build_project_topic_bridge",
    "slugify_group_value",
    "load_essay_snippet_lookup",
    "quality_report",
    "get_effective_support_volume_scale",
]
