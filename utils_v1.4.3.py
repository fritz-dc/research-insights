"""
utils.py — Classroom Compass shared utilities  (pipeline v1.0.0)

All functions used across the three pipeline notebooks live here. Notebooks
should contain only cell-level orchestration, parameter setup, and display
logic — no reusable logic of their own.

Section map (in order):
    1.  Imports & module-level constants
    2.  Config & I/O
    3.  Run identity & filter helpers
    4.  Stage & pipeline manifests
    5.  Warning file helpers
    6.  LLM client
    7.  Ingest
    8.  Token helpers
    9.  Consolidation helpers          (NB01 Steps 3-4)
    10. Analysis helpers
    11. Quality
    12. TF-IDF & NMF helpers           (NB03 Steps 1-4)
    13. Enrichment helpers             (NB02 Passes A-C)
    14. Topic labeling helpers         (NB03 Step 5)
    15. Synthesis helpers              (NB03 Step 5)
    16. Insight normalization & verification
    17. Dedup helpers
    18. Evidence & support tables
    19. Packaging & tiering
    20. DOCX report helpers
    21. __all__
"""

# ── 1. Imports & module-level constants ───────────────────────────────────────

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
import hashlib
import json
import os
import re
import time as _time
import unicodedata

import httpx
import numpy as np
import openai as _openai
import pandas as pd
import simplemma
import yaml
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from openai import OpenAI
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Root of the project tree; used to resolve all relative paths.
ROOT = Path(__file__).parent

# Pipeline version string; recorded in every stage manifest and the pipeline
# manifest so that any output artifact can be traced back to a specific release.
PIPELINE_VERSION = "1.0.0"

# Regex that flags likely simplemma truncation artifacts (stems ending in
# common consonant clusters where a vowel is expected). Used by
# build_consolidation_candidates() to mark flagged tokens for review.
_TRUNC_RE = re.compile(r"(at|iv|iz|az|ig|if|ic|olog)$")


# ── 2. Config & I/O ───────────────────────────────────────────────────────────


def _version_sort_key(path: Path) -> tuple:
    """Sort versioned files so higher explicit versions win; plain names sort last."""
    m = re.search(r"v(\d+)", path.stem)
    return (int(m.group(1)) if m else -1, path.name.lower())


def _resolve_params_path(base_dirs: list[Path]) -> Path | None:
    """Return the highest-versioned params*.yaml / params*.yml across base_dirs."""
    matches: list[Path] = []
    seen: set[str] = set()
    for base in base_dirs:
        for pattern in ("params*.yaml", "params*.yml"):
            for p in base.glob(pattern):
                if p.is_file():
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        matches.append(p)
    if not matches:
        return None
    return sorted(matches, key=_version_sort_key, reverse=True)[0]


def resolve_params_path() -> Path:
    """Return the resolved params file path using the same logic as load_cfg()."""
    env_path = os.environ.get("PIPELINE_PARAMS")
    if env_path:
        return Path(env_path)

    cfg_path = _resolve_params_path([ROOT, ROOT / "CONFIG"])
    if cfg_path is None:
        raise FileNotFoundError(
            "No params*.yaml or params*.yml file found in project root or CONFIG/"
        )
    return cfg_path


def load_cfg(path: str | Path | None = None) -> dict[str, Any]:
    """Load params.yaml (or any versioned params*.yaml) and return it as a nested dict.

    Resolution order when path is omitted:
        1. PIPELINE_PARAMS environment variable (if set)
        2. Highest-versioned params*.yaml / params*.yml in {ROOT}
        3. Highest-versioned params*.yaml / params*.yml in {ROOT}/CONFIG

    Set PIPELINE_PARAMS to an absolute path before launching Jupyter to
    use an alternate params file without modifying notebooks:
        export PIPELINE_PARAMS=/path/to/custom_params.yaml

    Args:
        path: Explicit path to a YAML file. When provided, skips the
              resolution order entirely (including the env variable).

    Returns:
        Parsed YAML content as a dict.
    """
    if path is not None:
        cfg_path = Path(path)
    else:
        env_path = os.environ.get("PIPELINE_PARAMS")
        if env_path:
            cfg_path = Path(env_path)
        else:
            cfg_path = _resolve_params_path([ROOT, ROOT / "CONFIG"])
            if cfg_path is None:
                raise FileNotFoundError(
                    "No params*.yaml or params*.yml file found in project root or CONFIG/"
                )

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: Any) -> Path:
    """Write payload to a JSON file, creating parent directories as needed.

    Uses deterministic serialisation (sorted keys, ensure_ascii=False,
    default=str for non-serialisable types) so manifests produce stable diffs.

    Args:
        path:    Destination file path.
        payload: JSON-serialisable object.

    Returns:
        The resolved Path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    """Return the MD5 hex digest of a file, or None if the file does not exist.

    Args:
        path:       File to hash.
        chunk_size: Read chunk size in bytes.

    Returns:
        Lowercase hex digest string, or None.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_meta(path: Path, label: str | None = None) -> dict[str, Any]:
    """Build a provenance metadata dict for one artifact path.

    Args:
        path:  Path to the artifact (may or may not exist yet).
        label: Human-readable label; defaults to path.name.

    Returns:
        Dict with keys: label, path, exists, md5, size_bytes.
    """
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
    groupby_field: str | None = None,
    run_date: str | None = None,
    root: Path = ROOT,
) -> Path:
    """Resolve a canonical OUTPUTS path and create parent directories.

    Without groupby_field / run_date (NB01, NB02):
        OUTPUTS/{subdir}/{fname}
    With both (legacy grouped outputs):
        OUTPUTS/{groupby_field}/{run_date}/{subdir}/{fname}

    Args:
        subdir:        Subdirectory under OUTPUTS (e.g. "prepared", "enrichment").
        fname:         File name including extension.
        groupby_field: Optional grouping column name for run-scoped paths.
        run_date:      Optional YYYY-MM-DD string for run-scoped paths.
        root:          Project root; defaults to the module ROOT constant.

    Returns:
        Resolved Path with parent directories created.
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
    run_id: str,
    root: Path = ROOT,
    strategic_loop_enabled: bool = False,
    strategic_area_id: str | None = None,
) -> Path:
    """Resolve a run-scoped OUTPUTS path for NB03.

    Non-strategic: OUTPUTS/runs/non_strategic/{groupby_field}/{run_id}/{subdir}/{fname}
    Strategic:     OUTPUTS/runs/strategic_area/{strategic_area_id}/{run_id}/{subdir}/{fname}

    Args:
        subdir:                 Subdirectory (e.g. "analysis", "insights").
        fname:                  File name including extension.
        groupby_field:          Grouping column name (used in non-strategic mode).
        run_id:                 Unique run identifier from get_run_id().
        root:                   Project root; defaults to ROOT.
        strategic_loop_enabled: Whether this is a strategic run.
        strategic_area_id:      Strategic area ID (used in strategic mode).

    Returns:
        Resolved Path with parent directories created.
    """
    root = Path(root)
    if strategic_loop_enabled:
        p = (
            root
            / "OUTPUTS"
            / "runs"
            / "strategic_area"
            / str(strategic_area_id)
            / str(run_id)
            / subdir
            / fname
        )
    else:
        p = (
            root
            / "OUTPUTS"
            / "runs"
            / "non_strategic"
            / str(groupby_field)
            / str(run_id)
            / subdir
            / fname
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def outpath(
    subdir: str,
    fname: str,
    root: Path = ROOT,
    groupby_field: str | None = None,
    run_date: str | None = None,
) -> Path:
    """Backward-compatible alias for build_output_path().

    Deprecated: use build_output_path() or build_run_output_path() directly.
    This alias will be removed in a future release.
    """
    return build_output_path(
        subdir, fname, groupby_field=groupby_field, run_date=run_date, root=root
    )


def get_run_date() -> str:
    """Return today's date as a YYYY-MM-DD string for output path nesting."""
    return datetime.now().strftime("%Y-%m-%d")


# ── 3. Run identity & filter helpers ─────────────────────────────────────────


def canonicalize_filter_spec(
    filter_logic: str,
    filters: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return a canonical, hashable filter-spec object used for run IDs and logging.

    Keys within each filter rule are sorted so that equivalent specs always
    produce the same JSON string regardless of dict insertion order.

    Args:
        filter_logic: Combination logic; only "and" is currently supported.
        filters:      List of filter rule dicts. None is treated as [].

    Returns:
        Dict with keys: schema_version, filter_logic, filters.
    """
    filters = filters or []
    canonical = []
    for item in filters:
        if not isinstance(item, dict):
            raise ValueError(f"Filter entries must be dicts, got {type(item)}")
        canonical.append({k: item[k] for k in sorted(item.keys())})
    return {"schema_version": "v1", "filter_logic": filter_logic, "filters": canonical}


def get_filter_fields_key(filters: list[dict[str, Any]] | None) -> str:
    """Return a stable string key summarising which fields are filtered on.

    Used to keep run outputs from different filter scopes from colliding.

    Args:
        filters: List of filter rule dicts. None or [] returns "none".

    Returns:
        Double-underscore-joined sorted field names, or "none".
    """
    filters = filters or []
    fields = sorted(
        {str(f.get("field", "")).strip() for f in filters if str(f.get("field", "")).strip()}
    )
    return "none" if not fields else "__".join(fields)


def get_run_id(
    groupby_field: str,
    filter_spec: dict[str, Any] | None = None,
) -> str:
    """Build a human-readable, sortable run ID with an 8-char scope hash.

    Format: {YYYYMMDD_HHMMSS}_{groupby_field}_{hash}

    Args:
        groupby_field: The analysis grouping column.
        filter_spec:   Canonical filter spec from canonicalize_filter_spec().
                       None is treated as an empty filter spec.

    Returns:
        Run ID string.
    """
    scope = json.dumps(
        filter_spec or {"filters": [], "filter_logic": "and"},
        sort_keys=True,
        separators=(",", ":"),
    )
    scope_hash = hashlib.md5(scope.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{groupby_field}_{scope_hash}"


def validate_filter_spec(
    df: pd.DataFrame,
    filter_logic: str,
    filters: list[dict[str, Any]] | None,
) -> None:
    """Validate analysis filter configuration against the loaded dataframe.

    Raises ValueError on the first problem found. Intended to be called
    at notebook startup so errors surface early, before any LLM calls.

    Args:
        df:           The dataframe the filters will be applied to.
        filter_logic: Must be exactly "and".
        filters:      List of filter rule dicts. None or [] passes validation.

    Raises:
        ValueError: On unsupported filter_logic, missing fields, bad ops,
                    or missing required rule keys.
    """
    if filter_logic != "and":
        raise ValueError("analysis.filter_logic must be exactly 'and'")
    for rule in filters or []:
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
        if op == "range" and "min" not in rule and "max" not in rule:
            raise ValueError(f"range filter requires 'min' and/or 'max': {rule}")


def _coerce_bound_for_series(series: pd.Series, value: Any) -> Any:
    """Coerce a range bound to match the dtype of the target series.

    Only datetime columns are auto-coerced; all other types are returned
    unchanged to avoid silent type mismatches.

    Args:
        series: The dataframe column the bound will be compared against.
        value:  The raw bound value from the filter rule.

    Returns:
        Coerced value (datetime for datetime columns, original value otherwise).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value)
    return value


def apply_filters(
    df: pd.DataFrame,
    filter_logic: str,
    filters: list[dict[str, Any]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply validated analysis filters and return the filtered dataframe plus a summary.

    Args:
        df:           Input dataframe.
        filter_logic: Combination logic; only "and" is supported.
        filters:      List of filter rule dicts. None or [] returns df unchanged.

    Returns:
        (filtered_df, summary_dict) where summary_dict contains row counts,
        retained percentage, fields checked, and filter_fields_key.
    """
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
        field, op, series = rule["field"], rule["op"], df[rule["field"]]
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
    retained_pct = round(len(out) / len(df) * 100, 2) if len(df) else 0.0
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


# ── 4. Stage & pipeline manifests ────────────────────────────────────────────


def start_stage_manifest(
    stage_name: str,
    notebook_file: str,
    config_path: str = "params.yaml",
    run_id: str | None = None,
    group_by_field: str | None = None,
    filter_fields_key: str | None = None,
) -> dict[str, Any]:
    """Create the common stage-manifest skeleton shared by all three notebooks.

    The manifest starts with status="running" and is finalised by a later call
    to finalize_stage_manifest().

    Args:
        stage_name:       Human-readable name for this pipeline stage.
        notebook_file:    Actual filename of the calling notebook (for provenance).
        config_path:      Path to params.yaml relative to ROOT.
        run_id:           Run ID from get_run_id(), or None for NB01/NB02.
        group_by_field:   Grouping column name, or None for NB01/NB02.
        filter_fields_key: Key from get_filter_fields_key(), or None.

    Returns:
        Manifest dict ready to be passed to finalize_stage_manifest().
    """
    config_meta = artifact_meta(ROOT / config_path, label="config")
    return {
        "schema_version": "v1",
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "group_by_field": group_by_field,
        "filter_fields_key": filter_fields_key,
        "stage_name": stage_name,
        "status": "running",
        "started_at": datetime.now().isoformat(),
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
    """Finalize a stage manifest, compute duration, and write it to disk.

    Args:
        manifest:          Dict returned by start_stage_manifest().
        output_path:       Where to write the final JSON manifest.
        status:            "success" or "failure".
        input_artifacts:   List of artifact_meta() dicts for inputs.
        output_artifacts:  List of artifact_meta() dicts for outputs.
        row_counts:        Dict of labelled row counts for the QA record.
        key_params:        Dict of parameter values recorded for the QA record.
        warnings_path:     Path to the JSONL warnings file for this stage.

    Returns:
        The completed manifest dict (also written to output_path).
    """
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
    """Persist the end-of-NB03 pipeline manifest covering all three stages.

    Args:
        output_path:           Where to write the pipeline manifest JSON.
        run_id:                Run ID from get_run_id().
        run_date:              YYYY-MM-DD string.
        group_by_field:        Grouping column name.
        filter_spec_path:      Path to the saved filter spec JSON.
        filter_summary_path:   Path to the saved filter summary JSON.
        stage_manifest_paths:  Paths to all three stage manifests.
        warnings_01_path:      NB01 warnings JSONL path.
        warnings_02_path:      NB02 warnings JSONL path.
        warnings_03_path:      NB03 warnings JSONL path.
        final_outputs:         Dict of labelled final output paths.
        config_path:           Path to params.yaml relative to ROOT.
        filter_fields_key:     Key from get_filter_fields_key().
        status:                "success" or "failure".

    Returns:
        The manifest dict (also written to output_path).
    """
    config_meta = artifact_meta(ROOT / config_path, label="config")
    payload = {
        "schema_version": "v1",
        "pipeline_version": PIPELINE_VERSION,
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
        # Individual warning paths kept for backward compatibility with consumers
        # that reference them by key. warnings_files is the canonical list form.
        "warnings_01_path": str(warnings_01_path),
        "warnings_02_path": str(warnings_02_path),
        "warnings_03_path": str(warnings_03_path),
        "warnings_files": [
            str(warnings_01_path),
            str(warnings_02_path),
            str(warnings_03_path),
        ],
        "final_outputs": final_outputs,
    }
    write_json(output_path, payload)
    return payload


# ── 5. Warning file helpers ───────────────────────────────────────────────────


def ensure_warning_file(path: Path) -> Path:
    """Create an empty JSONL warnings file if it does not already exist.

    Args:
        path: Target JSONL file path.

    Returns:
        The resolved Path.
    """
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
    """Append one structured warning record to a JSONL warnings file.

    Each record contains a UTC timestamp, stage name, severity, code,
    human-readable message, and an optional context dict for debugging.

    Args:
        path:       JSONL file path (created if missing).
        stage_name: Name of the pipeline stage emitting the warning.
        code:       Machine-readable warning code (e.g. "NMF_GROUP_SKIPPED").
        message:    Human-readable description.
        severity:   "warning" or "error".
        context:    Optional dict of additional debug fields.
    """
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
    """Count non-empty lines in a JSONL warnings file.

    Args:
        path: JSONL file path.

    Returns:
        Number of warning records, or 0 if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ── 6. LLM client ────────────────────────────────────────────────────────────


def get_llm_client() -> OpenAI:
    """Build and return an OpenAI client with SSL verification disabled.

    SSL verification is disabled to work with the DonorsChoose proxy
    environment. This is a fixed infrastructure requirement, not a
    configurable option.

    Returns:
        Authenticated OpenAI client.

    Raises:
        ValueError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))


_OPENAI_RETRYABLE_ERRORS = tuple(
    cls for cls in (
        getattr(_openai, "RateLimitError", None),
        getattr(_openai, "APITimeoutError", None),
        getattr(_openai, "APIStatusError", None),
    )
    if isinstance(cls, type)
)


def _extract_openai_error_body(exc: Exception) -> Any:
    """Best-effort extraction of an OpenAI/proxy error body."""
    body = getattr(exc, "body", None)
    if body is not None:
        return body
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.json()
        except Exception:
            return None
    return None


def _looks_like_flex_unavailable(exc: Exception, requested_service_tier: str) -> bool:
    """Return True when a Flex request should fall back to standard/default.

    This is intentionally narrow. We fall back only for Flex capacity/resource
    misses, not for every API failure or every malformed model response.
    """
    if requested_service_tier != "flex":
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    body = _extract_openai_error_body(exc)
    if isinstance(body, dict):
        err = body.get("error", body)
        code = str(err.get("code", "")).lower()
        typ = str(err.get("type", "")).lower()
        msg = str(err.get("message", "")).lower()
        if "resource_unavailable" in code or "resource unavailable" in msg:
            return True
        if "rate_limit" in typ and "flex" in msg and "unavailable" in msg:
            return True

    msg = str(exc).lower()
    return "resource unavailable" in msg and "flex" in msg


def chat_completion_with_tier_fallback(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
    response_format: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    **kwargs,
) -> Any:
    """Create one Chat Completion with immediate Flex fallback.

    The tier policy is per request:
      1. Try service_tier="flex" up to flex_attempts times.
      2. If Flex capacity is unavailable, retry the same request using
         fallback_service_tier.
      3. Do not exponential-backoff between Flex attempts unless explicitly
         configured through flex_retry_delay_seconds.

    Caller-level retry loops should handle JSON parse failures, schema
    validation failures, timeouts, and final warning behavior.
    """
    requested_service_tier = str(service_tier or "auto")
    fallback_service_tier = str(fallback_service_tier or "default")

    def _create(tier: str) -> Any:
        request_kwargs = {
            "model": model,
            "messages": messages,
            "service_tier": tier,
            "timeout": timeout_seconds,
            **kwargs,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        if max_completion_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_completion_tokens
        return client.chat.completions.create(**request_kwargs)

    if requested_service_tier != "flex":
        return _create(requested_service_tier)

    flex_attempts = max(0, int(flex_attempts))
    last_flex_error: Exception | None = None

    for attempt in range(flex_attempts):
        try:
            return _create("flex")
        except _OPENAI_RETRYABLE_ERRORS as e:
            if _looks_like_flex_unavailable(e, "flex"):
                last_flex_error = e
                if flex_retry_delay_seconds > 0 and attempt < flex_attempts - 1:
                    _time.sleep(flex_retry_delay_seconds)
                continue
            raise

    if last_flex_error is not None:
        return _create(fallback_service_tier)

    return _create(fallback_service_tier)


# ── 7. Ingest ─────────────────────────────────────────────────────────────────


def ingest(path: str | Path) -> pd.DataFrame:
    """Load a SQL extract CSV and parse the LISTAGG token string into a list.

    The SQL extract produces tokens as a comma-separated string (the result
    of a LISTAGG aggregation). This function splits that string into a proper
    Python list and parses known date columns.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with a 'tokens' column containing lists of strings.
    """
    raw = pd.read_csv(path)
    date_cols = [c for c in ["posted_date", "funded_date"] if c in raw.columns]
    df = pd.read_csv(path, parse_dates=date_cols) if date_cols else raw
    df["tokens"] = (
        df["tokens"]
        .fillna("")
        .str.split(",")
        .apply(lambda ts: [t.strip() for t in ts if t.strip()])
    )
    return df


# ── 8. Token helpers ─────────────────────────────────────────────────────────


def tokens_to_str(token_list: list[str] | None) -> str:
    """Join a token list into a space-separated string for sklearn vectorizers.

    Args:
        token_list: List of token strings, or None.

    Returns:
        Space-joined string, or "" for None / empty input.
    """
    if token_list is None:
        return ""
    return " ".join(token_list) if len(token_list) > 0 else ""


def flat_freq(df: pd.DataFrame, col: str = "tokens") -> pd.Series:
    """Compute corpus-wide token frequency across all projects.

    Args:
        df:  DataFrame with a column containing token lists.
        col: Name of the token-list column.

    Returns:
        Series of (token → count) sorted descending by count.
    """
    return pd.Series([t for ts in df[col] for t in ts]).value_counts()


def token_doc_freq(df: pd.DataFrame) -> pd.Series:
    """Compute the number of distinct projects containing each token.

    Uses explode rather than a loop, so it scales to large corpora.

    Args:
        df: DataFrame with 'project_id' and 'tokens' columns.

    Returns:
        Series of (token → distinct project count) named "doc_count".
    """
    return (
        df[["project_id", "tokens"]]
        .explode("tokens")
        .rename(columns={"tokens": "token"})
        .drop_duplicates()
        .groupby("token")["project_id"]
        .nunique()
        .rename("doc_count")
    )


def normalize_tokens(tokens: list[str], lang: str = "en") -> list[str]:
    """Apply morphological normalization to a token list via simplemma.

    Handles inflected forms correctly (drives→drive, organized→organize).
    Domain terms and proper nouns that simplemma does not recognise are
    returned unchanged; the consolidation map handles residual cases.

    Args:
        tokens: List of raw token strings.
        lang:   ISO 639-1 language code for simplemma. Defaults to "en".

    Returns:
        List of normalized token strings (same length as input).
    """
    return [simplemma.lemmatize(t, lang=lang) for t in tokens]


def dedupe_near_duplicate_projects(
    df: pd.DataFrame,
    *,
    block_field: str = "teacher_id",
    project_id_col: str = "project_id",
    token_col: str = "tokens",
    shingle_n: int = 5,
    containment_threshold: float = 0.90,
    min_tokens: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop near-duplicate project essays within a block, usually teacher_id.

    Uses ordered contiguous token shingles. A project is marked duplicate when
    its 5-token shingle set is at least containment_threshold contained in a
    previously kept project from the same block.

    Returns:
        filtered_df, audit_df
    """
    if block_field not in df.columns:
        audit = pd.DataFrame([{
            "status": "skipped",
            "reason": f"missing block_field: {block_field}",
        }])
        return df.copy(), audit

    def _shingles(tokens: Any) -> set[tuple[str, ...]]:
        toks = coerce_token_list(tokens)
        if len(toks) < max(shingle_n, min_tokens):
            return set()
        return {tuple(toks[i:i + shingle_n]) for i in range(len(toks) - shingle_n + 1)}

    def _containment(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))

    work = df.drop_duplicates(project_id_col).copy()
    work["_shingles"] = work[token_col].apply(_shingles)
    work["_token_count"] = work[token_col].apply(lambda x: len(coerce_token_list(x)))

    kept_ids = []
    removed_to_kept = {}
    audit_rows = []

    for block_value, g in work.groupby(block_field, dropna=False):
        reps: list[tuple[Any, set[tuple[str, ...]]]] = []

        # Stable order: earliest row wins within teacher block.
        for _, row in g.sort_values(project_id_col).iterrows():
            pid = row[project_id_col]
            shingles = row["_shingles"]

            if not shingles:
                kept_ids.append(pid)
                reps.append((pid, shingles))
                continue

            match_pid = None
            match_score = 0.0
            for rep_pid, rep_shingles in reps:
                score = _containment(shingles, rep_shingles)
                if score >= containment_threshold and score > match_score:
                    match_pid = rep_pid
                    match_score = score

            if match_pid is None:
                kept_ids.append(pid)
                reps.append((pid, shingles))
            else:
                removed_to_kept[pid] = match_pid
                audit_rows.append({
                    "project_id": pid,
                    "kept_project_id": match_pid,
                    "block_field": block_field,
                    "block_value": block_value,
                    "containment_score": round(float(match_score), 4),
                    "token_count": int(row["_token_count"]),
                    "method": f"{shingle_n}gram_containment",
                })

    filtered = df[df[project_id_col].isin(set(kept_ids))].copy()
    audit = pd.DataFrame(audit_rows)
    return filtered, audit


# ── 9. Consolidation helpers ─────────────────────────────────────────────────


def _auto_replacement(
    original: str,
    token_set: set[str],
    known_fixes: dict[str, str],
) -> str:
    """Return a replacement string for a token, or '' to leave it unchanged.

    Checks in order:
        1. known_fixes dict (simplemma truncation artifacts and other corrections).
        2. Singular-s stripping: map 'words' → 'word' when 'word' is in vocab.
        3. Singular-es stripping: map 'wishes' → 'wish' when 'wish' is in vocab.

    Args:
        original:    The token to check.
        token_set:   Set of all tokens in the top-N vocabulary window.
        known_fixes: Dict of {bad_stem: correct_form} loaded from
                     CONFIG/known_lemma_fixes.yaml.

    Returns:
        Replacement string, or '' if no replacement applies.
    """
    if original in known_fixes:
        return known_fixes[original]
    if original.endswith("s") and not original.endswith("ss") and len(original) > 3:
        singular = original[:-1]
        if singular in token_set:
            return singular
    if original.endswith("es") and len(original) > 4:
        singular = original[:-2]
        if singular in token_set:
            return singular
    return ""


def build_consolidation_candidates(
    df: pd.DataFrame,
    top_n: int = 1000,
    known_fixes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a consolidation candidate table from the top-N vocabulary tokens.

    For each of the top_n tokens by corpus frequency, auto-assigns a
    replacement (via _auto_replacement) and flags likely truncation artifacts
    (via _TRUNC_RE). The resulting CSV is the human-reviewable consolidation map
    used in NB01 Step 3.

    The known_fixes dict should be loaded from CONFIG/known_lemma_fixes.yaml
    (see params.yaml: preprocess.known_lemma_fixes_path). If not supplied,
    only plural-pair detection runs; KNOWN_FIXES corrections are skipped.

    Args:
        df:          DataFrame with a 'tokens' column (token lists).
        top_n:       Number of top-frequency tokens to include. Default 1000.
        known_fixes: Dict of {bad_stem: correct_form}. None treated as {}.

    Returns:
        DataFrame with columns: original, freq, flag_type, replacement, notes.
    """
    if known_fixes is None:
        known_fixes = {}

    vocab = flat_freq(df).head(top_n).reset_index()
    vocab.columns = ["original", "freq"]
    vocab["flag_type"] = vocab["original"].apply(
        lambda w: "truncation" if len(w) <= 8 and _TRUNC_RE.search(w) else ""
    )
    token_set = set(vocab["original"])
    vocab["replacement"] = vocab["original"].apply(
        lambda w: _auto_replacement(w, token_set, known_fixes)
    )
    vocab["notes"] = "auto"
    return vocab


# ── 10. Analysis helpers ──────────────────────────────────────────────────────


INJECTED_TOKEN_PREFIXES = ("__",)


def upweight_injected_tokens(
    X: Any,
    vec: TfidfVectorizer,
    weight: float,
    renormalize: bool = True,
) -> Any:
    """Multiply TF-IDF matrix columns corresponding to injected enrichment tokens.

    Injected tokens are analyst-curated semantic markers (e.g.
    __framing_calm__, __sensitive_context_food_insecurity__) that NB02 adds to
    project token lists based on essay content. Up-weighting them here makes
    them more influential in downstream NMF without changing which projects are
    in scope or which terms are in the vocabulary.

    For bigrams, any feature whose string contains a known injection prefix
    qualifies (catches "support __framing_calm__" as well as
    "__framing_calm__ space").

    Args:
        X:           Fitted TF-IDF sparse matrix from vec.fit_transform(docs).
        vec:         The vectorizer that produced X (used for feature names).
        weight:      Multiplier for injected-token columns. 1.0 = no change.
        renormalize: If True (default), re-L2-normalize rows after multiplying
                     so project-level magnitude is unchanged and only the
                     within-project weight balance shifts toward injected
                     tokens. If False, projects with more injected tokens
                     become more influential overall.

    Returns:
        The (possibly modified) sparse matrix. When weight == 1.0 or no
        injected-token columns are present, returns X unchanged.
    """
    if weight == 1.0:
        return X
    feat = vec.get_feature_names_out()
    injected_cols = [
        i for i, t in enumerate(feat)
        if any(p in t for p in INJECTED_TOKEN_PREFIXES)
    ]
    if not injected_cols:
        return X
    scaler = np.ones(X.shape[1])
    scaler[injected_cols] = weight
    X = X.multiply(scaler).tocsr()
    if renormalize:
        X = normalize(X, norm="l2", axis=1, copy=False)
    return X


def make_vec(
    min_df: int | float,
    max_df: int | float,
    ngram_range: tuple[int, int],
) -> TfidfVectorizer:
    """Instantiate a TF-IDF vectorizer with the pipeline's token pattern.

    The token pattern allows underscore and hyphen within tokens to preserve
    injected enrichment tokens (e.g. __framing_urgency__) and hyphenated terms.

    Args:
        min_df:       Minimum document frequency (int count or float fraction).
        max_df:       Maximum document frequency (int count or float fraction).
        ngram_range:  (min_n, max_n) tuple.

    Returns:
        Configured but unfitted TfidfVectorizer.
    """
    return TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b[a-z][a-z_\-]*\b",
    )


def add_bin(df: pd.DataFrame, bins: list[dict[str, Any]]) -> pd.DataFrame:
    """Label each project with the first matching analyst-defined date bin.

    Bins are matched in order against posted_date. Projects that match no
    bin receive None. Bins are defined in params.yaml under analysis.bins.

    Args:
        df:   DataFrame with a 'posted_date' column.
        bins: List of {name, start, end} dicts.

    Returns:
        Copy of df with an added 'bin' column.
    """
    df = df.copy()
    df["bin"] = None
    for b in bins:
        mask = (df["posted_date"] >= b["start"]) & (df["posted_date"] <= b["end"])
        df.loc[mask & df["bin"].isna(), "bin"] = b["name"]
    return df


def group_key(keys: Any, group_cols: list[str]) -> dict[str, Any]:
    """Normalise a groupby key (scalar or tuple) to a dict.

    pandas returns a scalar key for single-column groupby and a tuple for
    multi-column groupby. This function normalises both to a dict so
    downstream code does not need to branch.

    Args:
        keys:       The key returned by DataFrame.groupby().
        group_cols: List of column names used in the groupby.

    Returns:
        Dict mapping column name → key value.
    """
    return dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))


def build_project_topic_bridge(
    weights_df: pd.DataFrame,
    groupby_field: str,
    threshold: float,
) -> pd.DataFrame:
    """Build the project-topic bridge table from NMF weight outputs.

    For each project, computes topic_share = weight / sum(weights across topics
    for that project), then retains only rows where topic_share >= threshold.

    Args:
        weights_df:    DataFrame with columns: {groupby_field}, topic_id,
                       project_id, weight.
        groupby_field: The analysis grouping column.
        threshold:     Minimum topic_share for a project-topic link to be kept.
                       Sourced from analysis.topic_assignment_threshold.

    Returns:
        DataFrame with columns: topic_key, project_id, {groupby_field},
        topic_id, weight, topic_share.

    Raises:
        ValueError: If required columns are missing from weights_df.
    """
    required = {groupby_field, "topic_id", "project_id", "weight"}
    missing = required - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df missing required columns: {sorted(missing)}")

    totals = (
        weights_df.groupby([groupby_field, "project_id"])["weight"]
        .sum()
        .rename("total_weight")
        .reset_index()
    )
    merged = weights_df.merge(
        totals, on=[groupby_field, "project_id"], how="left", validate="many_to_one"
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
    """Convert a group value to a safe filesystem component for output paths.

    Strips accents, replaces non-word characters with underscores, collapses
    consecutive underscores, and truncates to max_len.

    Args:
        value:   Arbitrary group value string.
        max_len: Maximum length of the returned slug. Default 64.

    Returns:
        Safe lowercase slug, or "unknown" for empty/unrepresentable input.
    """
    value = str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:max_len] if value else "unknown"


def load_essay_snippet_lookup(
    project_ids: list[Any],
    data_dir: Path | None = None,
    max_chars: int = 300,
) -> dict[Any, str]:
    """Lazily load essay text snippets for a set of project IDs.

    Scans DATA/project_essay*.csv files in order, returning early once all
    requested IDs are found. Only loads the project_id and text columns to
    keep memory usage low.

    Args:
        project_ids: List of project IDs to look up.
        data_dir:    Directory containing project_essay*.csv files.
                     Defaults to {ROOT}/DATA.
        max_chars:   Maximum character length of each returned snippet.

    Returns:
        Dict mapping project_id → truncated essay text for found IDs.
        IDs not found in any file are absent from the dict.
    """
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "DATA"
    needed = set(project_ids)
    if not needed:
        return {}

    lookup: dict[Any, str] = {}
    essay_files = sorted(data_dir.glob("project_essay*.csv"))
    text_col_candidates = ["essay", "essay_text", "full_text", "project_essay", "text"]

    for fpath in essay_files:
        try:
            cols = pd.read_csv(fpath, nrows=0).columns.tolist()
        except Exception:
            continue
        text_col = next((c for c in text_col_candidates if c in cols), None)
        if not text_col or "project_id" not in cols:
            continue
        for chunk in pd.read_csv(fpath, usecols=["project_id", text_col], chunksize=200_000):
            sub = chunk[chunk["project_id"].isin(needed - set(lookup.keys()))]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                text = re.sub(r"\s+", " ", str(row.get(text_col, "") or "")).strip()
                if text:
                    lookup[row["project_id"]] = text[:max_chars]
            if len(lookup) == len(needed):
                return lookup
    return lookup



# ── 10A. Strategic loop preparation helpers ──────────────────────────────────

DEFAULT_PROJECT_CATEGORY_OTHER_VALUES: set[str] = {
    "",
    "missing",
    "awaiting classification",
    "awaiting classfication",  # common misspelling in source/category labels
    "awaiting_classification",
    "awaiting_classfication",
    "uncategorized",
    "unknown",
    "none",
    "nan",
}

DEFAULT_PROJECT_CATEGORY_BUCKET_RULES: dict[str, Any] = {
    "other_label": "All Other",
    "cumulative_share_keep": 0.90,
    "large_run_threshold": 50_000,
    "medium_run_threshold": 10_000,
    "min_count_large_run": 500,
    "min_count_medium_run": 250,
    "min_count_small_run": 100,
    "force_to_other_values": sorted(DEFAULT_PROJECT_CATEGORY_OTHER_VALUES),
}

DEFAULT_STRATEGIC_MIN_GROUP_PROJECTS = 200

DEFAULT_METADATA_LIFT_THRESHOLDS: dict[str, Any] = {
    "min_supporting_projects_for_lift": 50,
    "min_value_count_in_insight": 20,
    "min_value_count_in_run": 50,
    "min_cohens_h": 0.20,
    "min_abs_difference_pp": 2,
    "max_values_per_dimension": 3,
    "max_total_lift_facts_per_insight": 8,
}

DEFAULT_METADATA_LIFT_DIMENSIONS: list[str] = [
    "project_category_bucketed",
    "grade_band",
    "metro_type_at_time_of_posting",
    "fy25_historical_efs_status",
    "project_cost_bucket",
    "state_cluster",
    "funding_status",
    "posting_period",
    "school_is_low_income",
    "school_is_underserved_rural",
    "school_is_historically_underrepresented_race",
    "school_is_racially_predominant",
]

DEFAULT_STATE_CLUSTERS: dict[str, list[str]] = {
    "California": ["CA"],
    "New York": ["NY"],
    "Texas": ["TX"],
    "Nevada": ["NV"],
    "Florida": ["FL"],
    "Illinois": ["IL"],
    "Hawaii": ["HI"],
    "Pennsylvania": ["PA"],
    "North Carolina": ["NC"],
    "New Jersey": ["NJ"],
    "Arizona": ["AZ"],
    "Oklahoma": ["OK"],
    "Massachusetts": ["MA"],
    "Georgia": ["GA"],
    "New England": ["CT", "ME", "NH", "RI", "VT"],
    "Mid-Atlantic": ["DC", "DE", "MD"],
    "Appalachia": ["KY", "TN", "VA", "WV"],
    "Deep South": ["AL", "AR", "LA", "MS", "SC"],
    "Great Lakes": ["IN", "MI", "OH", "WI"],
    "Midwest Plains": ["IA", "KS", "MN", "MO", "ND", "NE", "SD"],
    "Mountain West": ["CO", "ID", "MT", "NM", "UT", "WY"],
    "Pacific Northwest": ["AK", "OR", "WA"],
}


def taxonomy_tag_to_bool_col(tag: str) -> str:
    """Return the boolean dataframe column name for a taxonomy tag.

    The NB02 enrichment output writes one boolean column per taxonomy tag using
    the convention tag_{taxonomy_tag}. This should be the primary membership
    source for strategic-area filtering.
    """
    tag = str(tag).strip()
    return tag if tag.startswith("tag_") else f"tag_{tag}"


def taxonomy_tag_to_injected_token(tag: str) -> str:
    """Return the injected token representation for a taxonomy tag."""
    tag = str(tag).strip()
    return tag if tag.startswith("__") and tag.endswith("__") else f"__{tag}__"


def coerce_token_list(value: Any) -> list[str]:
    """Coerce list-like token values into a clean list of strings.

    Handles real lists, tuples, numpy arrays, JSON/Python stringified lists,
    comma-delimited strings, whitespace-delimited strings, and nulls. This is
    intentionally defensive because parquet/CSV roundtrips can change list
    columns into strings.

    Examples:
        ["seat", "chair"]                    -> ["seat", "chair"]
        "['seat', 'chair']"                  -> ["seat", "chair"]
        '["seat", "chair"]'                  -> ["seat", "chair"]
        "seat, chair"                        -> ["seat", "chair"]
        "seat chair"                         -> ["seat", "chair"]
        "" / None / NaN                      -> []
    """
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [
            str(x).strip()
            for x in value
            if pd.notna(x) and str(x).strip()
        ]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        # Parse JSON/Python literal list strings produced by CSV roundtrips,
        # e.g. "['seat', 'chair']" or '["seat", "chair"]'.
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [
                        str(x).strip()
                        for x in parsed
                        if pd.notna(x) and str(x).strip()
                    ]
            except Exception:
                pass

            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [
                        str(x).strip()
                        for x in parsed
                        if pd.notna(x) and str(x).strip()
                    ]
            except Exception:
                pass

        # If the value looks comma-delimited, split on commas rather than
        # whitespace so multi-word terms such as "dry erase" stay intact.
        if "," in s:
            return [
                part.strip().strip("'\"")
                for part in s.split(",")
                if part.strip().strip("'\"")
            ]

        # Final fallback for true whitespace-delimited token strings.
        return [part.strip() for part in s.split() if part.strip()]

    return [str(value).strip()] if str(value).strip() else []


def _nested_get(cfg: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Return cfg[key1][key2]... if present, otherwise default."""
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_strategic_areas_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract strategic_areas from params-style config.

    Supports either a top-level strategic_areas block or a nested
    strategic_loop.strategic_areas block so the run design can be integrated
    into params.yaml without forcing one exact layout.
    """
    if isinstance(cfg.get("strategic_areas"), dict):
        return cfg["strategic_areas"]
    nested = _nested_get(cfg, ["strategic_loop", "strategic_areas"], {})
    return nested if isinstance(nested, dict) else {}


def get_strategic_run_plan_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the strategic run plan from params-style config.

    Supports top-level run_plan, top-level strategic_run_plan, or nested
    strategic_loop.run_plan. The returned dict is area_id -> split spec.
    """
    for path in (["run_plan"], ["strategic_run_plan"], ["strategic_loop", "run_plan"]):
        val = _nested_get(cfg, path, None)
        if isinstance(val, dict):
            return val
    return {}


def get_strategic_loop_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract optional strategic-loop defaults from params-style config."""
    defaults = {}
    for path in (["defaults"], ["strategic_loop", "defaults"]):
        val = _nested_get(cfg, path, None)
        if isinstance(val, dict):
            defaults.update(val)
    return defaults


def add_project_cost_bucket(
    df: pd.DataFrame,
    *,
    source_col: str = "total_cost",
    output_col: str = "project_cost_bucket",
) -> pd.DataFrame:
    """Add a coarse project-cost bucket from total project cost."""
    out = df.copy()
    if source_col not in out.columns:
        out[output_col] = pd.NA
        return out
    bins = [-np.inf, 250, 500, 1_000, 2_000, np.inf]
    labels = ["< $250", "$250–$499", "$500–$999", "$1,000–$1,999", "$2,000+"]
    out[output_col] = pd.cut(
        pd.to_numeric(out[source_col], errors="coerce"),
        bins=bins,
        labels=labels,
        right=False,
    ).astype("object")
    return out


def add_funding_status(
    df: pd.DataFrame,
    *,
    funded_date_col: str = "funded_date",
    expiration_date_col: str = "expiration_date",
    output_col: str = "funding_status",
) -> pd.DataFrame:
    """Add a three-bucket funding status for metadata lift and grouping.

    Values:
        funded            — funded_date is present.
        expired_unfunded  — not funded and expiration_date is before/today.
        live_unfunded     — not funded and not yet expired, or expiration missing.

    The three-bucket version avoids conflating active live projects with expired
    unfunded projects in metadata-lift prompts.
    """
    out = df.copy()
    if funded_date_col not in out.columns:
        out[output_col] = pd.NA
        return out

    funded = pd.to_datetime(out[funded_date_col], errors="coerce").notna()

    if expiration_date_col in out.columns:
        expiration = pd.to_datetime(out[expiration_date_col], errors="coerce", utc=True)
        expiration = expiration.dt.tz_localize(None)
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        expired = expiration.notna() & (expiration <= today)
    else:
        expired = pd.Series(False, index=out.index)

    out[output_col] = np.select(
        [funded, (~funded) & expired],
        ["funded", "expired_unfunded"],
        default="live_unfunded",
    )
    return out


def add_posting_period(
    df: pd.DataFrame,
    *,
    posted_year_quarter_col: str = "posted_year_quarter",
    posted_date_col: str = "posted_date",
    output_col: str = "posting_period",
) -> pd.DataFrame:
    """Add a stable posting-period field, preferring posted_year_quarter."""
    out = df.copy()
    if posted_year_quarter_col in out.columns:
        out[output_col] = out[posted_year_quarter_col]
    elif posted_date_col in out.columns:
        posted = pd.to_datetime(out[posted_date_col], errors="coerce")
        out[output_col] = posted.dt.to_period("Q").astype("string")
    else:
        out[output_col] = pd.NA
    return out


def add_state_cluster(
    df: pd.DataFrame,
    *,
    state_col: str = "state",
    output_col: str = "state_cluster",
    state_clusters: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add state_cluster from a two-letter state column."""
    out = df.copy()
    clusters = state_clusters or DEFAULT_STATE_CLUSTERS
    if state_col not in out.columns:
        out[output_col] = pd.NA
        return out
    state_to_cluster = {
        str(state).strip().upper(): cluster
        for cluster, states in clusters.items()
        for state in (states or [])
    }
    state_norm = out[state_col].astype("string").str.strip().str.upper()
    out[output_col] = state_norm.map(state_to_cluster).fillna("Unknown / Other")
    return out


def add_strategic_derived_fields(
    df: pd.DataFrame,
    *,
    state_clusters: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add all standard derived fields used by strategic-loop grouping/lift."""
    out = add_project_cost_bucket(df)
    out = add_funding_status(out)
    out = add_posting_period(out)
    out = add_state_cluster(out, state_clusters=state_clusters)
    return out


def _norm_category_value(value: Any) -> str:
    """Normalise category values for force-to-other comparisons."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _min_project_category_count(n_projects: int, bucket_rules: dict[str, Any]) -> int:
    """Return the run-size-specific project-category minimum count."""
    rules = {**DEFAULT_PROJECT_CATEGORY_BUCKET_RULES, **(bucket_rules or {})}
    if n_projects >= int(rules["large_run_threshold"]):
        return int(rules["min_count_large_run"])
    if n_projects >= int(rules["medium_run_threshold"]):
        return int(rules["min_count_medium_run"])
    return int(rules["min_count_small_run"])


def bucket_project_category_for_run(
    df: pd.DataFrame,
    *,
    source_col: str = "project_category",
    output_col: str = "project_category_bucketed",
    project_id_col: str = "project_id",
    bucket_rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add run-specific project_category_bucketed with an All Other tail.

    Rules:
      1. Missing / Awaiting Classification / Awaiting Classfication and other
         configured unclassified values are always forced to All Other.
      2. Among remaining categories, keep categories until cumulative project
         share reaches 90% within this run corpus.
      3. Also keep any category above the run-size-specific minimum count.
      4. Everything else becomes All Other.

    No hard maximum group cap is applied.
    """
    out = df.copy()
    rules = {**DEFAULT_PROJECT_CATEGORY_BUCKET_RULES, **(bucket_rules or {})}
    other_label = str(rules.get("other_label", "All Other"))
    force_values = {
        _norm_category_value(v)
        for v in rules.get("force_to_other_values", DEFAULT_PROJECT_CATEGORY_BUCKET_RULES["force_to_other_values"])
    }
    force_values |= DEFAULT_PROJECT_CATEGORY_OTHER_VALUES

    if source_col not in out.columns:
        out[output_col] = other_label
        return out

    raw = out[source_col].astype("object")
    norm = raw.apply(_norm_category_value)
    base = raw.astype("string").str.strip().fillna(other_label)
    forced_other = norm.isin(force_values)

    candidate = base.mask(forced_other, other_label)
    non_other = candidate[candidate != other_label]
    n_projects = int(out[project_id_col].nunique()) if project_id_col in out.columns else int(len(out))
    min_count = _min_project_category_count(n_projects, rules)

    if non_other.empty:
        out[output_col] = other_label
        return out

    vc = non_other.value_counts(dropna=False)
    n = int(vc.sum())
    cum_share_before = vc.cumsum().shift(fill_value=0) / max(n, 1)
    keep = set(vc.index[(cum_share_before < float(rules["cumulative_share_keep"])) | (vc >= min_count)])

    out[output_col] = candidate.where(candidate.isin(keep), other_label).astype("object")
    return out


def build_strategic_area_tag_matches(
    df: pd.DataFrame,
    strategic_areas: dict[str, Any],
    *,
    project_id_col: str = "project_id",
) -> pd.DataFrame:
    """Return one row per project × strategic area × matched taxonomy tag.

    Membership is based on NB02's boolean tag_{taxonomy_tag} columns. Raw-term
    backstops are intentionally not implemented here; current strategic areas
    are expected to be tag-driven. Missing tag columns are ignored so older
    enriched files can still run with a warning upstream if desired.
    """
    if project_id_col not in df.columns:
        raise ValueError(f"Missing project id column: {project_id_col}")

    available_cols = set(df.columns)
    chunks: list[pd.DataFrame] = []
    for area_id, spec in (strategic_areas or {}).items():
        if not isinstance(spec, dict):
            continue
        label = spec.get("label", area_id)
        include_tags = spec.get("include_taxonomy_tags", []) or []
        exclude_tags = spec.get("exclude_taxonomy_tags", []) or []
        include_pairs = [
            (str(tag), taxonomy_tag_to_bool_col(str(tag)))
            for tag in include_tags
            if taxonomy_tag_to_bool_col(str(tag)) in available_cols
        ]
        exclude_cols = [
            taxonomy_tag_to_bool_col(str(tag))
            for tag in exclude_tags
            if taxonomy_tag_to_bool_col(str(tag)) in available_cols
        ]
        if not include_pairs:
            continue
        exclude_mask = pd.Series(False, index=df.index)
        for col in exclude_cols:
            exclude_mask |= df[col].fillna(False).astype(bool)
        for tag, col in include_pairs:
            mask = df[col].fillna(False).astype(bool) & ~exclude_mask
            if not mask.any():
                continue
            chunks.append(pd.DataFrame({
                "project_id": df.loc[mask, project_id_col].values,
                "strategic_area_id": area_id,
                "strategic_area_label": label,
                "taxonomy_tag": tag,
                "tag_bool_column": col,
            }))
    if not chunks:
        return pd.DataFrame(columns=[
            "project_id", "strategic_area_id", "strategic_area_label",
            "taxonomy_tag", "tag_bool_column",
        ])
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates(["project_id", "strategic_area_id", "taxonomy_tag"])


def build_strategic_area_membership(tag_matches_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse project-area-tag matches to one row per project × area."""
    if tag_matches_df.empty:
        return pd.DataFrame(columns=[
            "project_id", "strategic_area_id", "strategic_area_label",
            "matched_taxonomy_tags_json", "matched_raw_terms_json", "match_source",
            "matched_taxonomy_tag_count", "matched_raw_term_count",
        ])
    grouped = (
        tag_matches_df
        .sort_values(["project_id", "strategic_area_id", "taxonomy_tag"])
        .groupby(["project_id", "strategic_area_id", "strategic_area_label"], dropna=False)
        .agg(matched_taxonomy_tags=("taxonomy_tag", lambda x: list(pd.unique(x))))
        .reset_index()
    )
    grouped["matched_taxonomy_tags_json"] = grouped["matched_taxonomy_tags"].apply(
        lambda vals: json.dumps([str(v) for v in vals], ensure_ascii=False)
    )
    grouped["matched_raw_terms_json"] = "[]"
    grouped["match_source"] = "injected_tag"
    grouped["matched_taxonomy_tag_count"] = grouped["matched_taxonomy_tags"].apply(len)
    grouped["matched_raw_term_count"] = 0
    return grouped.drop(columns=["matched_taxonomy_tags"])


def normalize_split_spec(split_spec: Any) -> dict[str, Any]:
    """Normalise a split spec into {split_id, groupby_fields, is_strategic_injected_tag}.

    Accepted inputs:
      - "project_category"
      - {"groupby_fields": ["project_category", "grade_band"], "label": "..."}
      - {"field": "grade_band"}

    Lists are treated as a combined split. Run-plan expansion should pass one
    split at a time; a second_splits list in YAML should be expanded by the
    caller unless it is intentionally wrapped in groupby_fields.
    """
    if isinstance(split_spec, str):
        fields = [split_spec]
        label = split_spec
    elif isinstance(split_spec, list):
        fields = [str(x) for x in split_spec]
        label = "_x_".join(fields)
    elif isinstance(split_spec, dict):
        raw_fields = (
            split_spec.get("groupby_fields")
            or split_spec.get("fields")
            or split_spec.get("field")
            or split_spec.get("grouping_type")
            or split_spec.get("split")
        )
        if raw_fields is None:
            raise ValueError(f"Split spec is missing groupby_fields/field: {split_spec}")
        fields = [str(raw_fields)] if isinstance(raw_fields, str) else [str(x) for x in raw_fields]
        label = str(split_spec.get("label") or split_spec.get("split_id") or "_x_".join(fields))
    else:
        raise ValueError(f"Unsupported split spec type: {type(split_spec)}")

    is_tag = len(fields) == 1 and fields[0] == "strategic_injected_tag"
    return {
        "split_id": label,
        "groupby_fields": fields,
        "is_strategic_injected_tag": is_tag,
    }


def resolve_groupby_source_column(field: str) -> str:
    """Map logical grouping names to physical dataframe columns."""
    mapping = {
        "project_category": "project_category_bucketed",
        "grade_band": "grade_band",
        "urbanity": "metro_type_at_time_of_posting",
        "efs": "fy25_historical_efs_status",
        "project_size": "project_cost_bucket",
        "state_cluster": "state_cluster",
        "strategic_injected_tag": "strategic_injected_tag",
    }
    return mapping.get(field, field)


def build_combined_group_column(
    df: pd.DataFrame,
    fields: list[str],
    *,
    output_col: str | None = None,
    separator: str = " | ",
    missing_label: str = "Missing",
) -> tuple[pd.DataFrame, str]:
    """Create a scalar group column from one or more logical grouping fields."""
    out = df.copy()
    source_cols = [resolve_groupby_source_column(f) for f in fields]
    missing = [c for c in source_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing grouping source columns for split {fields}: {missing}")
    out_col = output_col or "_x_".join(fields)
    parts = [out[c].fillna(missing_label).astype(str).str.strip() for c in source_cols]
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + separator + p
    out[out_col] = combined
    return out, out_col


def remove_injected_tokens_for_tags(
    tokens: Any,
    taxonomy_tags: list[str],
) -> list[str]:
    """Remove injected token forms for the supplied taxonomy tags from tokens."""
    remove = {taxonomy_tag_to_injected_token(tag) for tag in taxonomy_tags or []}
    return [tok for tok in coerce_token_list(tokens) if tok not in remove]


def remove_manual_exclude_tokens(
    tokens: list[str],
    exclude_tokens: list[str] | set[str],
) -> list[str]:
    exclude = set(str(t).strip() for t in exclude_tokens if str(t).strip())
    return [t for t in tokens if t not in exclude]
    

def filter_to_min_group_projects(
    df: pd.DataFrame,
    groupby_field: str,
    *,
    project_id_col: str = "project_id",
    min_projects: int = DEFAULT_STRATEGIC_MIN_GROUP_PROJECTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop grouping values with fewer than min_projects unique projects."""
    if groupby_field not in df.columns:
        raise ValueError(f"Missing groupby_field {groupby_field!r}")
    counts = (
        df.dropna(subset=[groupby_field])
        .groupby(groupby_field)[project_id_col]
        .nunique()
        .rename("project_count")
        .reset_index()
        .sort_values("project_count", ascending=False)
    )
    keep_values = set(counts.loc[counts["project_count"] >= min_projects, groupby_field])
    out = df[df[groupby_field].isin(keep_values)].copy()
    counts["kept_for_run"] = counts[groupby_field].isin(keep_values)
    return out, counts


def prepare_strategic_loop_run_dataframe(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    strategic_area_id: str,
    split_spec: Any,
    project_id_col: str = "project_id",
    token_col: str = "tokens",
    min_group_projects: int | None = None,
    project_category_bucket_rules: dict[str, Any] | None = None,
    state_clusters: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare the NB03 input dataframe for one strategic-area split run.

    This is the main surgical integration point for NB03. It assumes NB03 has
    already loaded the enriched parquet and already applied the existing NB03
    analysis filters. It does not apply date/corpus filters.

    Behavior:
      - strategic area membership uses tag_{taxonomy_tag} boolean columns.
      - strategic_injected_tag runs explode one row per project × matched tag,
        drop tag groups below min_group_projects, and remove area-defining
        injected tokens before TF-IDF/NMF.
      - non-tag runs keep injected tags and create a scalar groupby field,
        including combined fields such as project_category × grade_band.
      - project_category is bucketed within the current run corpus.
    """
    strategic_areas = get_strategic_areas_config(cfg)
    if strategic_area_id not in strategic_areas:
        raise ValueError(f"Unknown strategic_area_id {strategic_area_id!r}")
    area_spec = strategic_areas[strategic_area_id]
    include_tags = area_spec.get("include_taxonomy_tags", []) or []

    defaults = get_strategic_loop_defaults(cfg)
    min_projects = int(
        min_group_projects
        if min_group_projects is not None
        else defaults.get("min_projects_per_strategic_injected_tag_group", DEFAULT_STRATEGIC_MIN_GROUP_PROJECTS)
    )

    split = normalize_split_spec(split_spec)

    prepared = add_strategic_derived_fields(df, state_clusters=state_clusters)
    tag_matches = build_strategic_area_tag_matches(
        prepared, strategic_areas, project_id_col=project_id_col
    )
    area_tag_matches = tag_matches[tag_matches["strategic_area_id"] == strategic_area_id].copy()
    if area_tag_matches.empty:
        raise ValueError(f"No projects matched strategic area {strategic_area_id!r}")

    area_project_ids = set(area_tag_matches["project_id"].drop_duplicates())
    area_df = prepared[prepared[project_id_col].isin(area_project_ids)].copy()
    area_df = bucket_project_category_for_run(
        area_df,
        project_id_col=project_id_col,
        bucket_rules=project_category_bucket_rules,
    )

    if split["is_strategic_injected_tag"]:
        tag_counts = (
            area_tag_matches
            .groupby("taxonomy_tag")["project_id"]
            .nunique()
            .rename("project_count")
            .reset_index()
            .sort_values("project_count", ascending=False)
        )
        kept_tags = set(tag_counts.loc[tag_counts["project_count"] >= min_projects, "taxonomy_tag"])
        area_tag_matches = area_tag_matches[area_tag_matches["taxonomy_tag"].isin(kept_tags)].copy()
        if area_tag_matches.empty:
            raise ValueError(
                f"No strategic_injected_tag groups for {strategic_area_id!r} meet min_projects={min_projects}"
            )
        exploded = area_df.merge(
            area_tag_matches[["project_id", "taxonomy_tag"]],
            on="project_id",
            how="inner",
            validate="many_to_many",
        )
        exploded["strategic_injected_tag"] = exploded["taxonomy_tag"]
        if token_col in exploded.columns:
            exploded[token_col] = exploded[token_col].apply(
                lambda toks: remove_injected_tokens_for_tags(toks, include_tags)
            )
        run_df = exploded.copy()
        groupby_field = "strategic_injected_tag"
        group_counts = tag_counts.rename(columns={"taxonomy_tag": groupby_field})
        group_counts["kept_for_run"] = group_counts[groupby_field].isin(kept_tags)
    else:
        run_df, groupby_field = build_combined_group_column(
            area_df,
            split["groupby_fields"],
            output_col=split["split_id"],
        )
        run_df, group_counts = filter_to_min_group_projects(
            run_df,
            groupby_field,
            project_id_col=project_id_col,
            min_projects=min_projects,
        )
        if run_df.empty:
            raise ValueError(
                f"No groups for {strategic_area_id!r} split {split['split_id']!r} meet min_projects={min_projects}"
            )

    meta = {
        "strategic_area_id": strategic_area_id,
        "strategic_area_label": area_spec.get("label", strategic_area_id),
        "split_id": split["split_id"],
        "groupby_fields": split["groupby_fields"],
        "groupby_field": groupby_field,
        "is_strategic_injected_tag": bool(split["is_strategic_injected_tag"]),
        "min_group_projects": min_projects,
        "input_project_count": int(df[project_id_col].nunique()),
        "area_project_count": int(len(area_project_ids)),
        "run_row_count": int(len(run_df)),
        "run_project_count": int(run_df[project_id_col].nunique()),
        "group_count": int(run_df[groupby_field].nunique()),
        "group_counts": group_counts.to_dict(orient="records"),
        "removed_area_defining_injected_tokens": bool(split["is_strategic_injected_tag"]),
    }
    return run_df, meta


def expand_strategic_run_plan(run_plan_cfg: dict[str, Any]) -> pd.DataFrame:
    """Expand a compact strategic run plan into a flat run matrix.

    This helper intentionally does not apply defaults such as token handling;
    those are programmed into prepare_strategic_loop_run_dataframe().
    """
    plan = run_plan_cfg.get("run_plan", run_plan_cfg) if isinstance(run_plan_cfg, dict) else {}
    rows: list[dict[str, Any]] = []
    for area_id, spec in plan.items():
        if not isinstance(spec, dict):
            continue
        first = spec.get("first_split")
        if first is not None:
            norm = normalize_split_spec(first)
            rows.append({
                "strategic_area_id": area_id,
                "split_wave": "first_split",
                "split_id": norm["split_id"],
                "groupby_fields": norm["groupby_fields"],
                "sensitive_review": bool(spec.get("sensitive_review", False)),
            })
        for idx, item in enumerate(spec.get("second_splits", []) or [], start=1):
            norm = normalize_split_spec(item)
            rows.append({
                "strategic_area_id": area_id,
                "split_wave": f"second_split_{idx}",
                "split_id": norm["split_id"],
                "groupby_fields": norm["groupby_fields"],
                "sensitive_review": bool(spec.get("sensitive_review", False)),
            })
    return pd.DataFrame(rows)


def candidate_support_project_ids_from_source_topics(
    source_topics: list[Any],
    *,
    groupby_field: str,
    bridge_lookup: dict[str, pd.DataFrame],
) -> list[Any]:
    """Return unique supporting project IDs for candidate source topics."""
    ids: list[Any] = []
    for src in source_topics or []:
        if isinstance(src, dict):
            group = str(src.get("group", "")).strip()
            topic_id = src.get("topic_id", -1)
        elif isinstance(src, str) and "|" in src:
            group, topic_id = src.rsplit("|", 1)
            group = group.strip()
        else:
            continue
        try:
            topic_key = get_topic_key(groupby_field, group, int(float(topic_id)))
        except Exception:
            continue
        topic_rows = bridge_lookup.get(topic_key)
        if topic_rows is not None and not topic_rows.empty and "project_id" in topic_rows.columns:
            ids.extend(topic_rows["project_id"].tolist())
    return list(dict.fromkeys(ids))


def build_metadata_lift_context(
    *,
    run_df: pd.DataFrame,
    dimensions: list[str] | None = None,
    project_id_col: str = "project_id",
) -> dict[str, Any]:
    """Precompute run-level project metadata and baseline counts once per run.

    This is the efficient path for Step 7 metadata lift: run-level distinct
    project rows and run_counts_by_dimension are constant across candidate
    insights, so they should not be rebuilt for every candidate.
    """
    dims = [d for d in (dimensions or DEFAULT_METADATA_LIFT_DIMENSIONS) if d in run_df.columns]
    base_cols = [project_id_col] + dims
    project_df = run_df[base_cols].drop_duplicates(project_id_col).copy()
    run_counts_by_dimension = {
        dim: project_df.dropna(subset=[dim]).groupby(dim)[project_id_col].nunique()
        for dim in dims
    }
    return {
        "project_df": project_df,
        "run_n": int(project_df[project_id_col].nunique()),
        "dimensions": dims,
        "run_counts_by_dimension": run_counts_by_dimension,
        "project_id_col": project_id_col,
    }


def compute_metadata_lift(
    *,
    run_df: pd.DataFrame | None = None,
    support_project_ids: list[Any],
    dimensions: list[str] | None = None,
    project_id_col: str = "project_id",
    thresholds: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute significant metadata scope signals for a candidate insight.

    Compares the candidate support set against all unique projects in the
    current NB03 run corpus. Returns only rows passing the configured min-n,
    Cohen's-h effect-size, and absolute-difference thresholds.

    Pass a precomputed context from build_metadata_lift_context() to avoid
    rebuilding run-level distinct project metadata and run-count baselines for
    every candidate insight.

    The LLM prompt still receives human-readable share, percentage-point
    difference, and lift values; Cohen's h is used for selection and audit.
    """
    cfg = {**DEFAULT_METADATA_LIFT_THRESHOLDS, **(thresholds or {})}
    support_ids = set(support_project_ids or [])
    if len(support_ids) < int(cfg["min_supporting_projects_for_lift"]):
        return pd.DataFrame()

    if context is None:
        if run_df is None:
            raise ValueError("compute_metadata_lift requires either run_df or context")
        context = build_metadata_lift_context(
            run_df=run_df,
            dimensions=dimensions,
            project_id_col=project_id_col,
        )

    project_id_col = context["project_id_col"]
    project_df = context["project_df"]
    support_df = project_df[project_df[project_id_col].isin(support_ids)].copy()
    insight_n = int(support_df[project_id_col].nunique())
    if insight_n < int(cfg["min_supporting_projects_for_lift"]):
        return pd.DataFrame()

    run_n = int(context["run_n"])
    rows: list[dict[str, Any]] = []
    for dim in context["dimensions"]:
        run_counts = context["run_counts_by_dimension"].get(dim)
        if run_counts is None or run_counts.empty:
            continue
        insight_counts = support_df.dropna(subset=[dim]).groupby(dim)[project_id_col].nunique()
        if insight_counts.empty:
            continue

        for value, insight_count in insight_counts.items():
            run_count = int(run_counts.get(value, 0))
            insight_count = int(insight_count)

            if insight_count < int(cfg["min_value_count_in_insight"]):
                continue
            if run_count < int(cfg["min_value_count_in_run"]):
                continue

            insight_share = float(insight_count / insight_n) if insight_n else 0.0
            run_share = float(run_count / run_n) if run_n else 0.0
            if run_share <= 0:
                continue

            lift = insight_share / run_share
            diff_pp = (insight_share - run_share) * 100

            # Cohen's h is the primary effect-size gate: n-independent,
            # baseline-aware, and symmetric for over- and under-representation.
            # min_abs_difference_pp remains as a small backstop against
            # trivial-magnitude rows that happen to clear h.
            insight_share_clipped = float(np.clip(insight_share, 0.0, 1.0))
            run_share_clipped = float(np.clip(run_share, 0.0, 1.0))
            cohens_h = abs(
                2.0 * (
                    np.arcsin(np.sqrt(insight_share_clipped))
                    - np.arcsin(np.sqrt(run_share_clipped))
                )
            )

            if cohens_h < float(cfg["min_cohens_h"]):
                continue

            if abs(diff_pp) < float(cfg["min_abs_difference_pp"]):
                continue

            rows.append({
                "dimension": dim,
                "value": value,
                "insight_project_count": insight_count,
                "run_project_count": run_count,
                "insight_total_projects": insight_n,
                "run_total_projects": run_n,
                "insight_share": insight_share,
                "run_share": run_share,
                "difference_pp": diff_pp,
                "abs_difference_pp": abs(diff_pp),
                "lift": lift,
                "cohens_h": float(cohens_h),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        ["dimension", "cohens_h", "abs_difference_pp", "lift"],
        ascending=[True, False, False, False],
    )
    limited = []
    for _, g in out.groupby("dimension", sort=False):
        limited.append(g.head(int(cfg["max_values_per_dimension"])))
    out = pd.concat(limited, ignore_index=True) if limited else out
    out = out.sort_values(
        ["cohens_h", "abs_difference_pp", "lift"],
        ascending=[False, False, False],
    )
    return out.head(int(cfg["max_total_lift_facts_per_insight"])).reset_index(drop=True)

def _display_dimension_name(dim: str) -> str:
    """Human-readable labels for metadata lift prompt facts."""
    labels = {
        "project_category_bucketed": "Project category",
        "grade_band": "Grade band",
        "metro_type_at_time_of_posting": "Urbanity / metro",
        "fy25_historical_efs_status": "EFS",
        "project_cost_bucket": "Project size",
        "state_cluster": "State cluster",
        "funding_status": "Funding status",
        "posting_period": "Posting period",
        "school_is_low_income": "Low-income school flag",
        "school_is_underserved_rural": "Underserved rural flag",
        "school_is_historically_underrepresented_race": "Historically underrepresented race flag",
        "school_is_racially_predominant": "Racially predominant school flag",
    }
    return labels.get(dim, dim.replace("_", " ").title())


def format_metadata_lift_facts(lift_df: pd.DataFrame) -> str:
    """Format selected metadata scope signals for the Step 7 prompt.

    Selection is based on Cohen's h plus minimum-count and percentage-point
    gates. The formatted prompt text reports shares, percentage-point
    difference, and lift because those are easier for the LLM to use cleanly.
    """
    if lift_df is None or lift_df.empty:
        return ""
    lines = ["Optional metadata scope signals for this candidate insight:"]
    for _, row in lift_df.iterrows():
        dim = _display_dimension_name(str(row["dimension"]))
        value = str(row["value"])
        insight_pct = round(float(row["insight_share"]) * 100)
        run_pct = round(float(row["run_share"]) * 100)
        diff = float(row["difference_pp"])
        lift = float(row["lift"])
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"- {dim}: {value} is {insight_pct}% of supporting projects "
            f"vs {run_pct}% of this run ({sign}{diff:.0f}pp, {lift:.1f}x)."
        )
    lines.append(
        "Use these signals only to scope the insight. Do not use lift as causal evidence. "
        "Do not invent a new insight from lift. Do not imply the pattern is evenly "
        "distributed if the metadata strongly over-indexes in one segment."
    )
    return "\n".join(lines)


def metadata_lift_for_candidate_insight(
    insight: dict[str, Any],
    *,
    run_df: pd.DataFrame,
    groupby_field: str,
    bridge_lookup: dict[str, pd.DataFrame],
    dimensions: list[str] | None = None,
    project_id_col: str = "project_id",
    thresholds: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Compute and format Step 7 metadata lift for one candidate insight."""
    support_ids = candidate_support_project_ids_from_source_topics(
        insight.get("source_topics", []),
        groupby_field=groupby_field,
        bridge_lookup=bridge_lookup,
    )
    lift_df = compute_metadata_lift(
        run_df=run_df,
        support_project_ids=support_ids,
        dimensions=dimensions,
        project_id_col=project_id_col,
        thresholds=thresholds,
    )
    return lift_df, format_metadata_lift_facts(lift_df)

# ── 11. Quality ───────────────────────────────────────────────────────────────

# Module-level stopword set used as the default fallback in quality_report().
# These are terms that should never survive preprocessing in a clean corpus.
#
# NOTE: This constant is the live fallback until quality_report() is updated
# to load from params.yaml → quality.stopword_violation_list. Keep this set
# in sync with that YAML list. Once loading from params is wired in, this
# constant can be removed.
HARD_STOPWORDS: set[str] = {
    "the", "and", "for", "with", "are", "was", "were", "been", "this", "that",
    "from", "they", "them", "their", "will", "would", "should", "could", "have",
    "has", "had", "use", "using", "used", "make", "makes", "making", "get",
    "gets", "getting", "help", "helps", "project", "students", "student",
    "classroom", "learning", "school", "teacher", "teachers", "education",
    "grade", "grades", "materials", "supplies", "tools", "resources",
    "funded", "funding", "donors", "donor",
}


def quality_report(
    df: pd.DataFrame,
    label: str,
    doc_freq: pd.Series | None = None,
    matrices: dict[str, Any] | None = None,
    save_path: Path | None = None,
    stopwords: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Generate and print a corpus quality snapshot.

    Checks token distribution, vocabulary size, and stopword violations.
    Used at pipeline checkpoints (NB01 Step 6, NB03 Step 2) to gate progress
    before expensive LLM calls.

    Args:
        df:        DataFrame with a 'tokens' column.
        label:     Checkpoint label for display and JSON output (e.g. "cp1").
        doc_freq:  Optional Series of document frequencies for additional stats.
        matrices:  Optional dict of {name: scipy sparse matrix} to report on.
        save_path: Optional path to write the quality stats JSON.
        stopwords: Set or list of stopwords to check against. When None,
                   falls back to the HARD_STOPWORDS module constant. Intended
                   to be loaded from params.yaml → quality.stopword_violation_list
                   once that wiring is in place.

    Returns:
        Dict of quality statistics including stopword gate result.
    """
    active_stopwords = set(stopwords) if stopwords is not None else HARD_STOPWORDS
    freq = flat_freq(df)
    stops = [t for t in freq.head(200).index if t in active_stopwords]

    stats: dict[str, Any] = {
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


# ── 12. TF-IDF & NMF helpers ─────────────────────────────────────────────────


def cat_tfidf_slice(
    idx: Any,
    df_index: Any,
    X_full: Any,
    feat: Any,
    idf_vals: Any,
    top_n: int,
) -> pd.DataFrame:
    """Score one group slice against the rest of the corpus using a shared TF-IDF matrix.

    The vectorizer is fit once on the full corpus; this function only
    indexes into the resulting matrix, so there is no per-group refitting.

    Args:
        idx:       Index labels for the focal group's rows.
        df_index:  Full dataframe index (used to compute the complement set).
        X_full:    Full corpus TF-IDF sparse matrix.
        feat:      Feature name array from vectorizer.get_feature_names_out().
        idf_vals:  IDF value array from vectorizer.idf_.
        top_n:     Number of highest-TF-IDF terms to return.

    Returns:
        DataFrame with columns: token, tf, idf, tfidf, prevalence,
        contrast, project_count — sorted by tfidf descending.
    """
    rest_idx = df_index.difference(idx)
    X_cat = X_full[idx.tolist()]
    X_rest = X_full[rest_idx.tolist()]

    cat_prev = (X_cat > 0).mean(axis=0).A1
    rest_prev = (X_rest > 0).mean(axis=0).A1 if len(rest_idx) else np.zeros(len(feat))
    tf = X_cat.mean(axis=0).A1

    return pd.DataFrame({
        "token": feat,
        "tf": tf,
        "idf": idf_vals,
        "tfidf": tf * idf_vals,
        "prevalence": cat_prev,
        "contrast": cat_prev - rest_prev,
        "project_count": (X_cat > 0).sum(axis=0).A1.astype(int),
    }).nlargest(top_n, "tfidf")


def choose_n_components(
    n_docs: int,
    retained_vocab: int,
    base_n_components: int,
    slice_rules: dict[str, Any],
) -> int:
    """Choose an NMF topic count using corpus-size and vocabulary-size heuristics.

    Three caps are applied and the minimum is taken:
        doc_cap   — prevents too many topics relative to group size.
                    Uses slice_rules["min_projects_per_topic"] (default 15)
                    as the divisor: at least that many projects per topic.
        vocab_cap — prevents topics outnumbering the retained vocabulary.
        topic_cap — respects base_n_components or small_slice_topic_cap
                    when small-slice mode is active.

    Small-slice mode is activated by the notebook when the median group size
    falls below slice_rules["small_slice_cutoff"]; the notebook injects
    slice_rules["small_slice_mode"] = True before calling nmf_one().

    Args:
        n_docs:            Number of documents in this slice.
        retained_vocab:    Number of features retained by the TF-IDF vectorizer.
        base_n_components: Base topic count from params.yaml → nmf.n_components.
        slice_rules:       Dict from params.yaml → analysis.slice_rules.

    Returns:
        Final NMF n_components value (minimum 4).
    """
    min_ppt = slice_rules.get("min_projects_per_topic", 15)
    doc_cap = max(4, n_docs // min_ppt)
    vocab_cap = max(4, retained_vocab // 8)
    topic_cap = (
        slice_rules["small_slice_topic_cap"]
        if slice_rules.get("small_slice_mode", False)
        else base_n_components
    )
    return max(4, min(base_n_components, doc_cap, vocab_cap, topic_cap))


def nmf_one(
    docs: list[str],
    ct_cfg: dict[str, Any],
    cn_cfg: dict[str, Any],
    base_n_components: int,
    slice_rules: dict[str, Any],
) -> tuple[pd.DataFrame | None, Any | None, dict[str, Any]]:
    """Fit one NMF slice for a single group and return topics, weights, and metadata.

    Returns early (with None, None, skip_meta) when the slice does not meet
    minimum vocabulary or matrix density requirements. Skip reasons are
    recorded in the returned metadata dict for the warnings log.

    Args:
        docs:              List of token strings (one per project).
        ct_cfg:            TF-IDF config dict (params.yaml → tfidf).
        cn_cfg:            NMF config dict (params.yaml → nmf).
        base_n_components: Base topic count (params.yaml → nmf.n_components).
        slice_rules:       Slice eligibility rules (params.yaml → analysis.slice_rules).

    Returns:
        (topics_df, W, meta) where:
            topics_df  — DataFrame with topic_id, top_terms, top_weights columns,
                         or None on skip.
            W          — NMF weight matrix (n_docs × n_topics), or None on skip.
            meta       — Dict with n_components_used, retained_vocab,
                         nonzero_tfidf_nnz, and optionally skip_reason.
    """
    vec = make_vec(
        ct_cfg["min_df"],
        ct_cfg["max_df"],
        tuple(ct_cfg.get("ngram_range", [1, 1])),
    )
    X = vec.fit_transform(docs)
    X = upweight_injected_tokens(
        X, vec,
        weight=float(ct_cfg.get("injected_token_weight", 1.0)),
        renormalize=True,
    )
    retained_vocab = int(X.shape[1])
    nonzero_tfidf_nnz = int(X.nnz)

    if retained_vocab < slice_rules["min_retained_vocab"]:
        return None, None, {
            "skip_reason": "low_retained_vocab",
            "retained_vocab": retained_vocab,
            "nonzero_tfidf_nnz": nonzero_tfidf_nnz,
        }
    if nonzero_tfidf_nnz < slice_rules["min_tfidf_nnz"]:
        return None, None, {
            "skip_reason": "low_tfidf_nnz",
            "retained_vocab": retained_vocab,
            "nonzero_tfidf_nnz": nonzero_tfidf_nnz,
        }

    n_components_used = choose_n_components(
        n_docs=len(docs),
        retained_vocab=retained_vocab,
        base_n_components=base_n_components,
        slice_rules=slice_rules,
    )
    if retained_vocab < n_components_used:
        return None, None, {
            "skip_reason": "vocab_below_topic_count",
            "retained_vocab": retained_vocab,
            "nonzero_tfidf_nnz": nonzero_tfidf_nnz,
            "n_components_used": n_components_used,
        }

    model = NMF(
        n_components=n_components_used,
        random_state=cn_cfg["random_seed"],
        init="nndsvd",
        max_iter=cn_cfg["max_iter"],
    )
    W = model.fit_transform(X)
    vocab = vec.get_feature_names_out()

    rows = []
    for i, comp in enumerate(model.components_):
        idx = comp.argsort()[::-1][: cn_cfg["top_words"]]
        rows.append({
            "topic_id": i,
            "top_terms": vocab[idx].tolist(),
            "top_weights": comp[idx].tolist(),
        })

    return pd.DataFrame(rows), W, {
        "n_components_used": n_components_used,
        "retained_vocab": retained_vocab,
        "nonzero_tfidf_nnz": nonzero_tfidf_nnz,
    }


# ── 13. Enrichment helpers ────────────────────────────────────────────────────


def gate_cluster(
    cid: int,
    terms: list[str],
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    prompt_template: str,
    retries: int = 2,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run the LLM coherence gate for one enrichment cluster (NB02 Pass A).

    Calls the model once per cluster to classify it as inject / split / discard.
    On failure after all retries, returns a discard result with the error reason.

    The prompt_template must accept {cid} and {terms} format keys.
    Terms are capped at the first 20 before formatting.

    Args:
        cid:             Cluster ID (integer label from agglomerative clustering).
        terms:           List of vocabulary terms in this cluster.
        client:          OpenAI client from get_llm_client().
        model:           Model name (params.yaml → models.gate).
        system_prompt:   System prompt for the gating call.
        prompt_template: User prompt template with {cid} and {terms} placeholders.
        retries:         Maximum retry attempts on transient errors.

    Returns:
        Dict with keys: action, primary_category, subcategory, split_into, reasoning.
    """
    prompt = prompt_template.format(cid=cid, terms=terms[:20])
    for attempt in range(retries + 1):
        try:
            resp = chat_completion_with_tier_fallback(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                service_tier=service_tier,
                flex_attempts=flex_attempts,
                fallback_service_tier=fallback_service_tier,
                timeout_seconds=timeout_seconds,
                flex_retry_delay_seconds=flex_retry_delay_seconds,
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            if attempt < retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
            else:
                return {
                    "action": "discard",
                    "primary_category": None,
                    "subcategory": None,
                    "split_into": [],
                    "reasoning": f"API error after {retries} retries: {e}",
                }


def classify_batch(
    terms_batch: list[str],
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    prompt_template: str,
    taxonomy_ref: str,
    valid_categories: set[str],
    retries: int = 2,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
) -> dict[str, str | None]:
    """Classify a batch of vocabulary terms against the framing taxonomy (NB02 Pass B).

    Sends terms_batch to the model with the taxonomy reference embedded in the
    prompt. Filters returned categories against valid_categories so that model
    hallucinations never reach the injection map.

    The prompt_template must accept {taxonomy} and {terms} format keys.

    Args:
        terms_batch:      List of vocabulary terms to classify.
        client:           OpenAI client from get_llm_client().
        model:            Model name (params.yaml → models.classify).
        system_prompt:    System prompt for the classification call.
        prompt_template:  User prompt template with {taxonomy} and {terms} keys.
        taxonomy_ref:     Pre-formatted taxonomy reference string (built in NB02).
        valid_categories: Set of legal category names from the loaded taxonomy.
        retries:          Maximum retry attempts on transient errors.

    Returns:
        Dict mapping term → category string (or None for unclassified terms).
        Only entries with None or a valid category are included.
    """
    prompt = prompt_template.format(taxonomy=taxonomy_ref, terms=terms_batch)
    for attempt in range(retries + 1):
        try:
            resp = chat_completion_with_tier_fallback(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                service_tier=service_tier,
                flex_attempts=flex_attempts,
                fallback_service_tier=fallback_service_tier,
                timeout_seconds=timeout_seconds,
                flex_retry_delay_seconds=flex_retry_delay_seconds,
            )
            raw = json.loads(resp.choices[0].message.content.strip())
            return {k: v for k, v in raw.items() if v is None or v in valid_categories}
        except Exception as e:
            if attempt < retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
            else:
                print(f"    Batch error: {e}")
                return {t: None for t in terms_batch}


def inject_tokens(
    token_list: list[str],
    lookup: dict[str, list[str]],
) -> list[str]:
    """Append injection tokens to a project's token list (NB02 Pass C).

    For each token in token_list, looks up any enrichment tokens to inject
    via the lookup dict. Injected tokens are deduplicated (preserving first
    occurrence order) and appended after the original tokens.

    Original tokens are never modified — injection is strictly additive.

    Args:
        token_list: Original token list for one project.
        lookup:     Dict mapping source token → list of enrichment tokens
                    to inject (built from semantic_map and framing_map CSVs).

    Returns:
        Original token list with deduplicated enrichment tokens appended,
        or the unmodified original list if no lookup matches were found.
    """
    extra = []
    for t in token_list:
        if t in lookup:
            extra.extend(lookup[t])
    if not extra:
        return token_list
    return token_list + list(dict.fromkeys(extra))


# ── 14. Topic labeling helpers ────────────────────────────────────────────────


def _norm_group_value(value: Any) -> str:
    """Normalise a group value to a lowercase stripped string for comparisons."""
    return str(value or "").strip().casefold()


def _safe_topic_id(value: Any) -> int:
    """Coerce a topic_id to int, returning -1 on failure."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def build_input(
    t_row: Any,
    weights_df: pd.DataFrame,
    pid_text: pd.Series,
    groupby_field: str,
    n_representative: int,
    top_terms_in_prompt: int,
) -> dict[str, Any]:
    """Build one topic-labeling payload from a topic row and top representative projects.

    Selects the highest-weight projects for this topic to use as representative
    snippets, then formats unigrams, bigrams, and NMF terms separately to give
    the model the best signal for label generation.

    Args:
        t_row:               Row from topics_df for one topic.
        weights_df:          NMF weight bridge DataFrame.
        pid_text:            Series mapping project_id → essay snippet.
        groupby_field:       The analysis grouping column.
        n_representative:    Number of representative snippets to include.
        top_terms_in_prompt: Maximum number of terms to include per term type.

    Returns:
        Dict with keys: group_value, topic_id, bin_line, unigrams, bigrams,
        nmf_terms, snippets — ready to format into a prompt template.
    """
    terms = t_row["top_terms"]
    key_cols = [groupby_field] + (["bin"] if "bin" in t_row.index else [])
    mask = weights_df["topic_id"] == t_row["topic_id"]
    for col in key_cols:
        mask &= weights_df[col] == t_row[col]

    rep_pids = (
        weights_df[mask]
        .sort_values("weight", ascending=False)["project_id"]
        .tolist()[:n_representative]
    )

    n_uni = top_terms_in_prompt
    n_bi = max(2, top_terms_in_prompt // 2)
    n_nmf = top_terms_in_prompt
    return {
        "group_value": t_row[groupby_field],
        "topic_id": int(t_row["topic_id"]),
        "bin_line": (
            f"\nBin: {t_row['bin']}"
            if "bin" in t_row.index and pd.notna(t_row.get("bin"))
            else ""
        ),
        "unigrams": ", ".join([x for x in terms if " " not in x][:n_uni]),
        "bigrams": ", ".join([x for x in terms if " " in x][:n_bi]),
        "nmf_terms": ", ".join(terms[:n_nmf]),
        "snippets": "\n".join(f"- {pid_text.get(p, '')}" for p in rep_pids),
    }


def _make_label_error(
    inp: dict[str, Any],
    raw_text: str,
    code: str,
    model_labeling: str,
    groupby_field: str,
    error_text: str | None = None,
) -> dict[str, Any]:
    """Return a structured error object for topic label failures.

    Used by _label_with_retry() to produce a consistently-shaped error record
    that can be stored alongside successful label results and filtered later.

    Args:
        inp:           The labeling input dict from build_input().
        raw_text:      Raw model response text (may be empty or malformed).
        code:          Machine-readable error code.
        model_labeling: Model name used for the failed call.
        groupby_field: The analysis grouping column.
        error_text:    String representation of the exception, if any.

    Returns:
        Dict with parse_error=True and standard label-result shape.
    """
    return {
        "raw": raw_text,
        "parse_error": True,
        "error_code": code,
        "error": error_text,
        "model": model_labeling,
        "timestamp": datetime.now().isoformat(),
        groupby_field: inp["group_value"],
        "topic_id": inp["topic_id"],
    }


def _label_with_retry(
    inp: dict[str, Any],
    *,
    client: OpenAI,
    model_labeling: str,
    system_prompt: str,
    user_prompt_template: str,
    groupby_field: str,
    warnings_path: Path,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
    stage_name: str = "03_insights_generation",
) -> dict[str, Any]:
    """Call the labeling model with retry logic and structured error handling.

    Retries on JSON parse failures (backoff) and on rate-limit / timeout errors.
    Any other exception is treated as fatal for this topic (no retry).

    Args:
        inp:                  Labeling input dict from build_input().
        client:               OpenAI client.
        model_labeling:       Model name (params.yaml → models.labeling).
        system_prompt:        System prompt for the labeling call.
        user_prompt_template: User prompt template; formatted with inp as kwargs.
        groupby_field:        The analysis grouping column.
        warnings_path:        JSONL file for recording failures.
        max_retries:          Maximum retry attempts. Sourced from
                              params.yaml → llm.max_retries once wired.
        stage_name:           Stage name recorded in warning entries.

    Returns:
        Parsed label dict on success, or a _make_label_error() dict on failure.
    """
    text = ""
    for attempt in range(max_retries + 1):
        try:
            resp = chat_completion_with_tier_fallback(
                client=client,
                model=model_labeling,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_template.format(**inp)},
                ],
                service_tier=service_tier,
                flex_attempts=flex_attempts,
                fallback_service_tier=fallback_service_tier,
                timeout_seconds=timeout_seconds,
                flex_retry_delay_seconds=flex_retry_delay_seconds,
            )
            text = resp.choices[0].message.content.strip()
            obj = json.loads(text)
            obj[groupby_field] = str(inp["group_value"])
            obj["topic_id"] = int(inp["topic_id"])
            obj["model"] = model_labeling
            obj["timestamp"] = datetime.now().isoformat()
            return obj

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
                continue
            append_warning(
                warnings_path, stage_name, "LABELING_PARSE_FAILURE",
                f"JSON parse failure for {inp['group_value']} / topic {inp['topic_id']}",
                context={"group": inp["group_value"], "topic_id": inp["topic_id"], "error": str(e)},
            )
            return _make_label_error(inp, text, "LABELING_PARSE_FAILURE",
                                     model_labeling, groupby_field, str(e))

        except (_openai.RateLimitError, _openai.APITimeoutError) as e:
            if attempt < max_retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
            else:
                append_warning(
                    warnings_path, stage_name, "LABELING_API_FAILURE",
                    f"API failure after retries for {inp['group_value']} / topic {inp['topic_id']}",
                    context={"group": inp["group_value"], "topic_id": inp["topic_id"], "error": str(e)},
                )
                return _make_label_error(inp, text or str(e), "LABELING_API_FAILURE",
                                         model_labeling, groupby_field, str(e))

        except Exception as e:
            append_warning(
                warnings_path, stage_name, "LABELING_API_FAILURE",
                f"Unexpected error for {inp['group_value']} / topic {inp['topic_id']}",
                context={"group": inp["group_value"], "topic_id": inp["topic_id"], "error": str(e)},
            )
            return _make_label_error(inp, text or str(e), "LABELING_API_FAILURE",
                                     model_labeling, groupby_field, str(e))


# ── 15. Synthesis helpers ─────────────────────────────────────────────────────


def clean_label(text: Any) -> str:
    """Remove injected enrichment token markers from a label string.

    Strips patterns like __framing_urgency__ and __cat_marine_biology__ so
    that injected token names never appear in synthesis prompts or report text.

    Args:
        text: Any value; coerced to str before processing.

    Returns:
        Cleaned string with all __token_name__ patterns removed.
    """
    return re.sub(r"__[a-z_]+__\s*", "", str(text)).strip()


def build_topic_lines(
    df: pd.DataFrame,
    groupby_field: str,
    group: Any | None = None,
    top_terms_count: int = 4,
) -> str:
    """Render labeled topics into the line-oriented prompt format used in NB03.

    One line per topic, format:
        {group} | topic {id} | label: {label} | coherence: {flag} | [top_terms: ...] | description: {desc}

    Args:
        df:              labels_df from the labeling step.
        groupby_field:   The analysis grouping column.
        group:           If provided, filter to this group value only.
        top_terms_count: Maximum number of top terms to include per topic line.

    Returns:
        Newline-joined string of topic lines ready for prompt insertion.
    """
    if group is not None:
        df = df[df[groupby_field] == group]

    def _fmt_terms(val: Any, n: int) -> str:
        """Parse top_terms (list, JSON string, or CSV string) and return top n."""
        if n <= 0:
            return ""
        if isinstance(val, list):
            terms = [str(x).strip() for x in val if str(x).strip()]
        elif pd.isna(val):
            terms = []
        else:
            s = str(val).strip()
            if not s:
                terms = []
            else:
                try:
                    parsed = json.loads(s)
                    terms = [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else [x.strip() for x in s.split(",") if x.strip()]
                except Exception:
                    terms = [x.strip() for x in s.split(",") if x.strip()]
        terms = [clean_label(t) for t in terms[:n]]
        return ", ".join(t for t in terms if t)

    lines = []
    for _, row in df.iterrows():
        top_terms_str = _fmt_terms(row.get("top_terms"), top_terms_count)
        line = (
            f"  {row[groupby_field]} | topic {row.topic_id} | "
            f"label: {clean_label(row.proposed_label)} | "
            f"coherence: {row.coherence_flag} | "
        )
        # Optional strength signal so synthesis can weigh thin vs strong topics.
        # Present only when the labels_df has been augmented with
        # support_multiplier (run-median-normalized supporting-project count).
        support_mul = row.get("support_multiplier")
        if support_mul is not None and pd.notna(support_mul) and support_mul > 0:
            line += f"support: {float(support_mul):.1f}x | "
        if top_terms_str:
            line += f"top_terms: {top_terms_str} | "
        line += f"description: {clean_label(row.description)}"
        lines.append(line)
    return "\n".join(lines)


def build_unit_lines(
    unit_labels_df: pd.DataFrame,
    cluster_membership_df: pd.DataFrame | None,
    groupby_field: str,
    group: Any | None = None,
    top_terms_count: int = 4,
    max_variation_groups_shown: int = 4,
) -> str:
    """Render labeled units (clusters + singletons) into prompt lines.

    Sibling to build_topic_lines. Emits one line per unit so synthesis can
    reason about clusters as first-class evidence while still seeing singleton
    topics. Cluster lines carry span (n_topics, n_groups), variation_notes
    summary, and support_multiplier. Singleton lines carry the same shape as
    topic lines.

    Line formats:
      cluster:   cluster {cid} | label: {...} | coherence: {flag} | span: {N topics across M groups} |
                 support: 2.4x | variation: {group1: angle1; group2: angle2; ...; +K more} |
                 member_topics: <g>|<id>, <g>|<id>, ... | description: {...}
      singleton: {group} | topic {id} | label: {...} | coherence: {flag} |
                 support: 1.2x | top_terms: ... | description: {...}

    Args:
        unit_labels_df:           DataFrame of labeled units with unit_type column.
        cluster_membership_df:    Required when unit_labels_df contains cluster rows;
                                  used to look up member (group, topic_id) pairs.
                                  Pass None to skip cluster handling.
        groupby_field:            The analysis grouping column.
        group:                    If provided, filter to units containing this group.
                                  For singletons: matches the row's group field.
                                  For clusters: matches if any cluster member is in this group.
        top_terms_count:          Cap on top_terms shown per singleton line.
        max_variation_groups_shown: Cap on variation_notes entries shown per cluster line.

    Returns:
        Newline-joined string of unit lines.
    """
    if unit_labels_df.empty:
        return ""

    def _fmt_terms(val: Any, n: int) -> str:
        """Parse top_terms (list, JSON string, or CSV string) and return top n."""
        if n <= 0:
            return ""
        if isinstance(val, list):
            terms = [str(x).strip() for x in val if str(x).strip()]
        elif pd.isna(val):
            terms = []
        else:
            s = str(val).strip()
            if not s:
                terms = []
            else:
                try:
                    parsed = json.loads(s)
                    terms = (
                        [str(x).strip() for x in parsed if str(x).strip()]
                        if isinstance(parsed, list)
                        else [x.strip() for x in s.split(",") if x.strip()]
                    )
                except Exception:
                    terms = [x.strip() for x in s.split(",") if x.strip()]
        terms = [clean_label(t) for t in terms[:n]]
        return ", ".join(t for t in terms if t)

    # Pre-build a cluster_id -> member rows index so we can look up member
    # (group, topic_id) pairs without re-querying for each cluster line.
    cluster_members_by_id: dict[int, list[tuple[str, int]]] = {}
    if cluster_membership_df is not None and not cluster_membership_df.empty:
        for cid, sub in cluster_membership_df.groupby("cluster_id"):
            cluster_members_by_id[int(cid)] = [
                (str(r[groupby_field]), int(r["topic_id"]))
                for _, r in sub.iterrows()
            ]

    lines: list[str] = []
    for _, row in unit_labels_df.iterrows():
        unit_type = row.get("unit_type")

        # ── Cluster line ────────────────────────────────────────────────────
        if unit_type == "cluster":
            cid = int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else -1
            members = cluster_members_by_id.get(cid, [])

            # Optional group filter: include cluster only if any member is in
            # the requested group.
            if group is not None:
                if not any(g == group for g, _ in members):
                    continue

            # Span: counts come from the cluster's members.
            member_groups = sorted({g for g, _ in members})
            line = (
                f"  cluster {cid} | "
                f"label: {clean_label(row.get('proposed_label'))} | "
                f"coherence: {row.get('coherence_flag', '?')} | "
                f"span: {len(members)} topics across {len(member_groups)} groups | "
            )

            # Strength signal: same field as singleton lines.
            support_mul = row.get("support_multiplier")
            if support_mul is not None and pd.notna(support_mul) and support_mul > 0:
                line += f"support: {float(support_mul):.1f}x | "

            # Variation summary: pull the LLM-returned variation_notes if
            # present. Each is {group, distinctive_angle}. Skip
            # "no distinctive angle" entries to keep the line tight.
            var_notes = row.get("variation_notes")
            if isinstance(var_notes, list) and var_notes:
                informative = [
                    n for n in var_notes
                    if isinstance(n, dict)
                    and str(n.get("distinctive_angle", "")).strip().lower()
                    != "no distinctive angle"
                ]
                if informative:
                    shown = informative[:max_variation_groups_shown]
                    pieces = [
                        f"{n.get('group', '?')}: {clean_label(n.get('distinctive_angle', ''))}"
                        for n in shown
                    ]
                    extra = len(informative) - len(shown)
                    suffix = f"; +{extra} more" if extra > 0 else ""
                    line += f"variation: {'; '.join(pieces)}{suffix} | "

            # Member topics: synthesis must cite at this level, not at cluster level.
            if members:
                # Truncate if very long to keep the prompt readable.
                member_strs = [f"{g}|{tid}" for g, tid in members[:20]]
                if len(members) > 20:
                    member_strs.append(f"+{len(members) - 20} more")
                line += f"member_topics: {', '.join(member_strs)} | "

            line += f"description: {clean_label(row.get('description', ''))}"
            lines.append(line)
            continue

        # ── Singleton line ──────────────────────────────────────────────────
        # Singletons in unit_labels_df may not carry GROUPBY_FIELD/topic_id
        # directly; recover from the unit_id if needed. By construction in
        # Step 5, singleton unit_id is "singleton_{group}_{topic_id}", but
        # group values can contain underscores so we don't string-parse them.
        # Prefer the labels_df side (singleton_label_rows) which carries the
        # parsed fields. Here we work from whatever unit_labels_df has.
        s_group = row.get(groupby_field) or row.get("group")
        s_topic = row.get("topic_id")
        if pd.isna(s_group) or pd.isna(s_topic):
            # Cannot render without ids. Skip with no error -- this row will
            # appear elsewhere via labels_df if needed.
            continue

        if group is not None and str(s_group) != str(group):
            continue

        top_terms_str = _fmt_terms(row.get("top_terms"), top_terms_count)
        line = (
            f"  {s_group} | topic {int(s_topic)} | "
            f"label: {clean_label(row.get('proposed_label'))} | "
            f"coherence: {row.get('coherence_flag', '?')} | "
        )
        support_mul = row.get("support_multiplier")
        if support_mul is not None and pd.notna(support_mul) and support_mul > 0:
            line += f"support: {float(support_mul):.1f}x | "
        if top_terms_str:
            line += f"top_terms: {top_terms_str} | "
        line += f"description: {clean_label(row.get('description', ''))}"
        lines.append(line)

    return "\n".join(lines)


def build_per_group_prompt(
    group: Any,
    group_description: str,
    topic_lines_text: str,
    per_group_instructions: str,
) -> str:
    """Build the per-group synthesis prompt body.

    Args:
        group:                  Group value (e.g. category name).
        group_description:      Human-readable description from group_descriptions.
        topic_lines_text:       Output of build_topic_lines() for this group.
        per_group_instructions: Instruction block appended after the topic list.

    Returns:
        Formatted prompt string ready for _call_with_retry().
    """
    return f"""
Below is a list of NMF topics discovered from teacher project request essays on DonorsChoose
for a single group: {group} ({group_description}).
Each topic represents a cluster of essays with similar language, framing, and request patterns.
{topic_lines_text}

{per_group_instructions}
""".strip()


def _call_with_retry(
    prompt: str,
    *,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
) -> str | None:
    """Generic LLM caller with fixed-delay retry and tier fallback.

    Used for synthesis and cross-group analysis calls where the response
    is plain text (not JSON), so parse errors are not a retry trigger.

    Args:
        prompt:       User prompt string.
        client:       OpenAI client.
        model_name:   Model name.
        system_prompt: System prompt string.
        max_retries:  Maximum retry attempts on rate-limit or timeout errors.
                      Sourced from params.yaml → llm.max_retries once wired.

    Returns:
        Model response text stripped of leading/trailing whitespace,
        or None if a non-retryable error occurs.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = chat_completion_with_tier_fallback(
                client=client,
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                service_tier=service_tier,
                flex_attempts=flex_attempts,
                fallback_service_tier=fallback_service_tier,
                timeout_seconds=timeout_seconds,
                flex_retry_delay_seconds=flex_retry_delay_seconds,
            )
            return resp.choices[0].message.content.strip()
        except (_openai.RateLimitError, _openai.APITimeoutError):
            if attempt < max_retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
            else:
                raise
        except Exception as e:
            print(f"Non-retryable error: {e}")
            return None


def synthesize_one_group(
    group: Any,
    *,
    labels_df: pd.DataFrame,
    groupby_field: str,
    group_description: str,
    per_group_instructions: str,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    warnings_path: Path,
    outpath_func: Callable[[str, str], Path],
    synthesis_top_terms_count: int = 4,
    unit_labels_df: pd.DataFrame | None = None,
    cluster_membership_df: pd.DataFrame | None = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
    stage_name: str = "03_insights_generation",
) -> tuple[Any, str | None]:
    """Run one per-group synthesis pass and persist the raw text output.

    Builds the topic lines, formats the prompt, calls the model, and writes
    the result to a per-group text file. On failure, writes a warning and
    returns None for the result.

    Args:
        group:                    Group value (e.g. category name).
        labels_df:                Labeled topics DataFrame.
        groupby_field:            The analysis grouping column.
        group_description:        Prompt-friendly description for this group.
        per_group_instructions:   Instruction block for per-group synthesis.
        client:                   OpenAI client.
        model_name:               Model name (params.yaml → models.synthesis).
        system_prompt:            Synthesis system prompt.
        warnings_path:            JSONL file for recording failures.
        outpath_func:             Callable(subdir, fname) → Path for output files.
        synthesis_top_terms_count: Terms per topic line in the synthesis prompt.
        max_retries:              Maximum retry attempts.
        stage_name:               Stage name for warning records.

    Returns:
        (group, result_text) where result_text is None on failure.
    """
    if (
        unit_labels_df is not None
        and not unit_labels_df.empty
        and "unit_type" in unit_labels_df.columns
    ):
        topic_lines_text = build_unit_lines(
            unit_labels_df,
            cluster_membership_df if cluster_membership_df is not None else None,
            groupby_field,
            group=group,
            top_terms_count=synthesis_top_terms_count,
        )

        # Fallback for edge cases where unit filtering yields no lines for this group.
        if not str(topic_lines_text or "").strip():
            topic_lines_text = build_topic_lines(
                labels_df,
                groupby_field,
                group=group,
                top_terms_count=synthesis_top_terms_count,
            )
    else:
        topic_lines_text = build_topic_lines(
            labels_df,
            groupby_field,
            group=group,
            top_terms_count=synthesis_top_terms_count,
        )
    prompt = build_per_group_prompt(
        group=group,
        group_description=group_description,
        topic_lines_text=topic_lines_text,
        per_group_instructions=per_group_instructions,
    )
    try:
        result = _call_with_retry(
            prompt,
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            service_tier=service_tier,
            flex_attempts=flex_attempts,
            fallback_service_tier=fallback_service_tier,
            timeout_seconds=timeout_seconds,
            flex_retry_delay_seconds=flex_retry_delay_seconds,
        )
    except _OPENAI_RETRYABLE_ERRORS as e:
        # _call_with_retry re-raises retryable OpenAI/API errors after exhausting
        # logical retries; catch here so a single group failure takes the
        # graceful warning path rather than halting the notebook.
        append_warning(
            warnings_path, stage_name, "SYNTHESIS_GROUP_FAILED",
            f"Synthesis failed for group '{group}' after retries: {e}",
            context={"group": group, "error": str(e)},
        )
        return group, None
    if result is None:
        append_warning(
            warnings_path, stage_name, "SYNTHESIS_GROUP_FAILED",
            f"Synthesis failed for group '{group}'",
            context={"group": group},
        )
        return group, None

    slug = slugify_group_value(group)
    fpath = outpath_func("analysis", f"llm_synthesis_{slug}.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(result)
    return group, result


# ── 16. Insight normalization & verification ──────────────────────────────────


def strip_json_fences(text: str | None) -> str:
    """Remove optional ```json code fences from model output before json.loads().

    Args:
        text: Raw model response string, or None.

    Returns:
        Stripped string with fences removed.
    """
    text = (text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def normalize_source_topics(
    source_topics: list[Any] | None,
    required_group_values: list[Any],
) -> list[dict[str, Any]]:
    """Normalise model source_topics into [{'group': ..., 'topic_id': int}] records.

    Handles two input formats:
        - Dicts with group/topic_id keys (primary format from structured model output).
        - Pipe-delimited strings of the form "group_value|topic_id".

    Note: bare "Topic N" strings (without a group) are parsed for the topic_id
    but are then discarded because the subsequent group-membership check requires
    a non-empty group value present in required_group_values. They do not survive
    into the output list.

    Entries whose group value is not in required_group_values are always filtered
    out. Output is deduplicated while preserving first-seen order.

    Args:
        source_topics:         Raw source_topics list from model output.
        required_group_values: Allowed group values (from the source topics table).

    Returns:
        List of {group, topic_id} dicts, deduplicated in original order.
    """
    out = []
    for src in source_topics or []:
        group = tid = None
        if isinstance(src, dict):
            group = src.get("group", src.get("group_value", ""))
            tid = src.get("topic_id", src.get("topic", src.get("id", "")))
        elif isinstance(src, str):
            s = src.strip()
            if "|" in s:
                left, right = s.rsplit("|", 1)
                group, tid = left.strip(), right.strip()
            elif re.fullmatch(r"(?i)topic\s+\d+", s):
                tid = re.sub(r"(?i)topic\s+", "", s).strip()

        if group is not None:
            group = str(group).strip()
        if tid is not None:
            tid = str(tid).strip()
        if not tid:
            continue
        try:
            tid = int(float(tid))
        except Exception:
            continue
        if not group or group not in required_group_values:
            continue
        out.append({"group": group, "topic_id": tid})

    seen: set[tuple[str, int]] = set()
    deduped = []
    for item in out:
        key = (item["group"], item["topic_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped



def normalize_insight(
    insight: dict[str, Any],
    required_group_values: list[Any],
) -> dict[str, Any]:
    """Normalise one insight object, preserving both verified and claimed source_topics.

    Args:
        insight:               Raw insight dict from model JSON output.
        required_group_values: Allowed group values for source_topic filtering.

    Returns:
        Normalised dict with keys: title, finding, evidence_basis,
        scope_or_caveat, why_it_matters, source_topics, source_topics_claimed.

    Raises:
        ValueError: If insight is not a dict, or required fields are missing.
    """
    if not isinstance(insight, dict):
        raise ValueError(f"Insight must be a dict, got {type(insight)}")

    raw_source_topics = deepcopy(insight.get("source_topics", []))
    source_topics = normalize_source_topics(raw_source_topics, required_group_values)

    normalized = {
        "title": str(insight.get("title", "")).strip(),
        "finding": str(insight.get("finding", "")).strip(),
        "evidence_basis": str(insight.get("evidence_basis", "")).strip(),
        "scope_or_caveat": str(insight.get("scope_or_caveat", "")).strip(),
        "why_it_matters": str(insight.get("why_it_matters", "")).strip(),
        "source_topics": source_topics,
        "source_topics_claimed": raw_source_topics,
    }

    required_text_fields = ["title", "finding", "evidence_basis", "scope_or_caveat"]
    missing = [field for field in required_text_fields if not normalized[field]]
    if missing:
        raise ValueError(f"Insight missing required fields {missing}: {insight}")

    return normalized


def project_insight_for_saved_candidates(insight: dict[str, Any]) -> dict[str, Any]:
    """Reformat a saved candidate insight into the normalised pipeline shape.

    Used when loading previously saved candidates from a prior run back into
    the pipeline for re-processing. Converts flexible source_topic formats to
    the canonical group|topic_id string list.

    Args:
        insight: Candidate insight dict from a saved JSON file.

    Returns:
        Dict with keys: title, finding, evidence_basis, scope_or_caveat,
        why_it_matters, source_topics.
    """
    source_topics_out = []
    for src in insight.get("source_topics", []):
        if isinstance(src, dict):
            group = str(src.get("group", "")).strip()
            topic_id = int(float(src.get("topic_id", -1)))
            if group and topic_id >= 0:
                source_topics_out.append(f"{group}|{topic_id}")
        elif isinstance(src, str) and "|" in src:
            source_topics_out.append(src.strip())

    return {
        "title": str(insight.get("title", "")).strip(),
        "finding": str(insight.get("finding", "")).strip(),
        "evidence_basis": str(insight.get("evidence_basis", "")).strip(),
        "scope_or_caveat": str(insight.get("scope_or_caveat", "")).strip(),
        "why_it_matters": str(insight.get("why_it_matters", "")).strip(),
        "source_topics": source_topics_out,
    }


def verify_source_topics(
    insight: dict[str, Any],
    labels_df: pd.DataFrame,
    groupby_field: str,
    required_group_values: list[Any],
    *,
    client: OpenAI,
    model_verify: str,
    system_prompt: str,
    warnings_path: Path,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
    stage_name: str = "03_insights_generation",
) -> dict[str, Any]:
    """Verify that each claimed source topic directly supports the insight.

    Sends the insight title, finding, evidence_basis, and each claimed topic's
    label and description to the model. The model returns only the subset of
    topics that genuinely support the core claim. Topics dropped by the model
    are removed from insight["source_topics"] in-place.

    Returns the insight unchanged if it has no source_topics or if all API
    attempts fail (warnings are recorded but the pipeline continues).

    Args:
        insight:               Insight dict with source_topics list.
        labels_df:             Labeled topics DataFrame.
        groupby_field:         The analysis grouping column.
        required_group_values: Allowed group values.
        client:                OpenAI client.
        model_verify:          Model name (params.yaml → models.verify).
        system_prompt:         Verification system prompt.
        warnings_path:         JSONL file for recording failures.
        max_retries:           Maximum retry attempts.
        stage_name:            Stage name for warning records.

    Returns:
        The insight dict with source_topics updated to verified-only topics.
    """
    title = insight.get("title", "")
    finding = insight.get("finding", "")
    evidence_basis = insight.get("evidence_basis", "")
    source_topics = insight.get("source_topics", [])
    warning_id = (str(title).strip() or str(finding).strip())[:120]

    if not source_topics:
        return insight

    topic_lines = []
    for src in source_topics:
        if isinstance(src, dict):
            group = src.get("group", "")
            tid = str(src.get("topic_id", ""))
        elif isinstance(src, str) and "|" in src:
            group, tid = src.rsplit("|", 1)
        else:
            continue
        match = labels_df[
            (labels_df[groupby_field] == group)
            & (labels_df["topic_id"] == int(float(tid)))
        ]
        if not match.empty:
            row = match.iloc[0]
            topic_lines.append(
                f"  {group}|{tid} | label: {row['proposed_label']} | "
                f"description: {row['description']}"
            )

    if not topic_lines:
        return insight

    prompt = f"""
Insight title: {title}

Finding: {finding}

Evidence basis: {evidence_basis}

Claimed source topics:
{chr(10).join(topic_lines)}

Return JSON: {{"verified_topics": [{{"group": "...", "topic_id": <int>}}, ...]}}
""".strip()

    for attempt in range(max_retries + 1):
        try:
            resp = chat_completion_with_tier_fallback(
                client=client,
                model=model_verify,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                service_tier=service_tier,
                flex_attempts=flex_attempts,
                fallback_service_tier=fallback_service_tier,
                timeout_seconds=timeout_seconds,
                flex_retry_delay_seconds=flex_retry_delay_seconds,
            )
            result = json.loads(resp.choices[0].message.content)
            original_n = len(source_topics)
            insight["source_topics"] = normalize_source_topics(
                result.get("verified_topics", source_topics), required_group_values
            )
            verified_n = len(insight.get("source_topics", []))
            if verified_n != original_n:
                print(f"Adjusted source_topics: {title[:80]} | {original_n} -> {verified_n}")
            if verified_n == 0 and original_n > 0:
                print(f"WARNING: all source_topics removed: {title[:80]}")
            return insight
        except (_openai.RateLimitError, _openai.APITimeoutError) as e:
            if attempt < max_retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
            else:
                append_warning(
                    warnings_path,
                    stage_name,
                    "VERIFY_API_FAILURE",
                    f"Verification failed for '{warning_id or '[untitled]'}'",
                    context={"title": title or None, "error": str(e)},
                )
                return insight
        except Exception as e:
            append_warning(
                warnings_path,
                stage_name,
                "VERIFY_API_FAILURE",
                f"Verification failed for '{warning_id or '[untitled]'}'",
                context={"title": title or None, "error": str(e)},
            )
            return insight

def _verify_insight_list(
    items: list[dict[str, Any]],
    *,
    labels_df: pd.DataFrame,
    groupby_field: str,
    required_group_values: list[Any],
    client: OpenAI,
    model_verify: str,
    system_prompt: str,
    warnings_path: Path,
    min_source_topics_to_verify: int = 1,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run verify_source_topics() over a list of insights and collect summary stats.

    Insights with fewer than min_source_topics_to_verify claimed topics are
    passed through without verification (used to skip narrow by-group insights).

    Args:
        items:                       List of insight dicts.
        labels_df:                   Labeled topics DataFrame.
        groupby_field:               The analysis grouping column.
        required_group_values:       Allowed group values.
        client:                      OpenAI client.
        model_verify:                Verification model name.
        system_prompt:               Verification system prompt.
        warnings_path:               JSONL file for recording failures.
        min_source_topics_to_verify: Minimum claimed topic count to trigger
                                     verification. From analysis.verification
                                     .by_group_min_source_topics in params.yaml.

    Returns:
        (verified_items, stats_dict) where stats_dict records counts of
        changed and dropped-to-zero insights.
    """
    verified_items = []
    changed_count = dropped_to_zero_count = topics_before = topics_after = 0

    for insight in items:
        before_n = len(insight.get("source_topics", []))
        topics_before += before_n
        if before_n >= min_source_topics_to_verify:
            verified = verify_source_topics(
                insight,
                labels_df=labels_df,
                groupby_field=groupby_field,
                required_group_values=required_group_values,
                client=client,
                model_verify=model_verify,
                system_prompt=system_prompt,
                warnings_path=warnings_path,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                service_tier=service_tier,
                flex_attempts=flex_attempts,
                fallback_service_tier=fallback_service_tier,
                timeout_seconds=timeout_seconds,
                flex_retry_delay_seconds=flex_retry_delay_seconds,
            )
        else:
            verified = insight
        after_n = len(verified.get("source_topics", []))
        topics_after += after_n
        if after_n != before_n:
            changed_count += 1
        if before_n > 0 and after_n == 0:
            dropped_to_zero_count += 1
        verified_items.append(verified)

    return verified_items, {
        "insight_count": len(items),
        "changed_count": changed_count,
        "dropped_to_zero_count": dropped_to_zero_count,
        "topics_before": topics_before,
        "topics_after": topics_after,
    }


# ── 17. Dedup helpers ─────────────────────────────────────────────────────────


def _norm_text(s: Any) -> str:
    """Normalise a string for token-overlap comparison.

    Lowercases, strips punctuation, and collapses whitespace.
    """
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_set(s: Any) -> set[str]:
    """Return the set of words in a normalised string."""
    return set(_norm_text(s).split())


def _jaccard(a: Any, b: Any) -> float:
    """Compute Jaccard similarity between two token sets (or iterables).

    Returns 0.0 when either set is empty.
    """
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _topic_list(val: Any) -> list[str]:
    """Parse a source_topics value into a flat list of 'group|topic_id' strings.

    Handles list, JSON string, and comma-separated string formats. Used to
    build the verified_topics_list column for dedup comparisons.
    """
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


def _pair_kind(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Classify an insight pair into a dedup comparison category.

    Categories:
        key_vs_key    — both are cross-group (key_insights) insights.
        bg_vs_same_bg — both are by-group insights from the same category.
        None          — different-bucket by-group pair; never deduped.

    bg_vs_key pairs are intentionally excluded from dedup: cross-group and
    by-group insights can legitimately cover similar ground at different
    levels of specificity.

    Args:
        a: Row dict for the kept insight (must have 'section', 'category_bucket').
        b: Row dict for the candidate insight.

    Returns:
        Category string or None.
    """
    a_key = a["section"] == "key_insights"
    b_key = b["section"] == "key_insights"
    if a_key and b_key:
        return "key_vs_key"
    if not a_key and not b_key:
        return "bg_vs_same_bg" if a["category_bucket"] == b["category_bucket"] else None
    return None  # bg_vs_key — skip dedup


def _screen_pair(
    a: dict[str, Any],
    b: dict[str, Any],
) -> dict[str, Any] | None:
    """Pre-screen an insight pair for potential duplication before the dedupe rules.

    Computes topic, title, and text overlap and returns a metadata dict only
    for pairs that exceed the screening thresholds. Pairs below all thresholds
    are returned as None and never reach the deterministic dedupe logic.

    Screening thresholds (intentionally generous to avoid false negatives):
        topic_overlap  ≥ 0.30
        title_overlap  ≥ 0.45
        text_overlap   ≥ 0.40

    Args:
        a: Row dict for the kept insight.
        b: Row dict for the candidate insight.

    Returns:
        Dict with pair_kind, topic_overlap, title_overlap, text_overlap — or
        None if pair_kind is None or no threshold is met.
    """
    kind = _pair_kind(a, b)
    if kind is None:
        return None

    topic_overlap = _jaccard(a["verified_topics_list"], b["verified_topics_list"])
    title_overlap = _jaccard(a["title_tokens"], b["title_tokens"])
    text_overlap = _jaccard(a["text_tokens"], b["text_tokens"])

    if topic_overlap >= 0.30 or title_overlap >= 0.45 or text_overlap >= 0.40:
        return {
            "pair_kind": kind,
            "topic_overlap": topic_overlap,
            "title_overlap": title_overlap,
            "text_overlap": text_overlap,
        }
    return None


# ── 18. Evidence & support tables ────────────────────────────────────────────


def get_topic_key(groupby_field: str, group: Any, topic_id: Any) -> str:
    """Build the canonical topic_key string used in the bridge table.

    Format: {groupby_field}={group}|topic={topic_id}

    Args:
        groupby_field: The analysis grouping column name.
        group:         Group value.
        topic_id:      Integer topic ID.

    Returns:
        Topic key string.
    """
    return f"{groupby_field}={group}|topic={int(float(topic_id))}"


def iter_candidate_insights(
    data: dict[str, Any],
    output_group_key: str,
) -> Any:
    """Yield every synthesized insight as a flat candidate dict.

    Assigns sequential IDs: KI_001, KI_002, ... for key_insights and
    BG_001, BG_002, ... for by-group insights.

    Args:
        data:             insights_data dict with key_insights and by-group lists.
        output_group_key: Key for the by-group section (e.g. "by_group").

    Yields:
        Dicts with keys: insight_id, section, category_bucket, insight.
    """
    for idx, insight in enumerate(data.get("key_insights", []), start=1):
        yield {
            "insight_id": f"KI_{idx:03d}",
            "section": "key_insights",
            "category_bucket": None,
            "insight": insight,
        }
    group_idx = 1
    for group_value, items in data.get(output_group_key, {}).items():
        for insight in items:
            yield {
                "insight_id": f"BG_{group_idx:03d}",
                "section": output_group_key,
                "category_bucket": group_value,
                "insight": insight,
            }
            group_idx += 1


def _parse_topic_id(val: Any) -> int:
    """Parse a topic ID from either a plain integer or a 'Topic N' string."""
    s = str(val).strip()
    if s.lower().startswith("topic"):
        s = s.split()[-1]
    return int(s)


def build_bridge_lookup(
    project_topic_bridge_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Index the project-topic bridge table by topic_key for O(1) lookup.

    Args:
        project_topic_bridge_df: Output of build_project_topic_bridge().

    Returns:
        Dict mapping topic_key → DataFrame of {project_id, weight, topic_share}.
    """
    return {
        topic_key: grp[["project_id", "weight", "topic_share"]].copy()
        for topic_key, grp in project_topic_bridge_df.groupby("topic_key", observed=True)
    }


def build_label_index(
    labels_df: pd.DataFrame,
    groupby_field: str,
    warnings_path: Path | None = None,
    stage_name: str = "03_insights_generation",
) -> dict[tuple[str, int], Any]:
    """Build a (group, topic_id) → label row lookup dict from labels_df.

    Emits a warning for duplicate keys and keeps the first occurrence.

    Args:
        labels_df:     Labeled topics DataFrame from the labeling step.
        groupby_field: The analysis grouping column.
        warnings_path: Optional JSONL file for duplicate-key warnings.
        stage_name:    Stage name for warning records.

    Returns:
        Dict mapping (group_str, topic_id_int) → labels_df row.
    """
    out: dict[tuple[str, int], Any] = {}
    for _, row in labels_df.iterrows():
        key = (str(row[groupby_field]), int(row["topic_id"]))
        if key in out:
            if warnings_path is not None:
                append_warning(
                    warnings_path, stage_name, "DUPLICATE_TOPIC_LABEL_ROW",
                    f"Duplicate labels_df row for {key}; keeping first occurrence",
                    context={"group": key[0], "topic_id": key[1]},
                )
            continue
        out[key] = row
    return out


def summarize_insight_support(
    candidate: dict[str, Any],
    *,
    groupby_field: str,
    run_id: str,
    bridge_lookup: dict[str, pd.DataFrame],
    label_index: dict[tuple[str, int], Any],
    top_project_id_limit: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute support statistics for one candidate insight.

    For each verified source topic, retrieves its project rows from the bridge
    lookup and its label from the label index. Ranks supporting projects by
    combined topic_share across all source topics.

    Args:
        candidate:            Candidate dict from iter_candidate_insights().
        groupby_field:        The analysis grouping column.
        run_id:               Run ID for provenance.
        bridge_lookup:        From build_bridge_lookup().
        label_index:          From build_label_index().
        top_project_id_limit: Maximum number of project IDs to include in the
                              flat row. Intended to be sourced from
                              params.yaml → output.csv_max_ids_per_insight
                              once that wiring is in place.

    Returns:
        (flat_row, support_rows) where flat_row is one insight summary dict
        and support_rows is a list of per-topic support detail dicts.
    """
    insight = candidate["insight"]
    source_topics = insight.get("source_topics", [])
    claimed_topic_count = len(insight.get("source_topics_claimed", []))

    support_rows: list[dict[str, Any]] = []
    ranking_frames: list[pd.DataFrame] = []

    for src in source_topics:
        if not isinstance(src, dict):
            continue
        group = str(src.get("group", "")).strip()
        topic_id = int(float(src.get("topic_id", -1)))
        topic_key = get_topic_key(groupby_field, group, topic_id)
        topic_rows = bridge_lookup.get(
            topic_key, pd.DataFrame(columns=["project_id", "weight", "topic_share"])
        )
        label_row = label_index.get((group, topic_id))

        if not topic_rows.empty:
            ranking_frames.append(topic_rows[["project_id", "topic_share", "weight"]].copy())

        support_rows.append({
            "run_id": run_id,
            "insight_id": candidate["insight_id"],
            "section": candidate["section"],
            "category_bucket": candidate["category_bucket"],
            "group_by_field": groupby_field,
            "group_value": group,
            "topic_id": topic_id,
            "topic_label": label_row["proposed_label"] if label_row is not None else "[not found]",
            "topic_description": label_row["description"] if label_row is not None else "[not found]",
            "coherence_flag": label_row["coherence_flag"] if label_row is not None else "unknown",
            "supporting_project_count": int(topic_rows["project_id"].nunique()) if not topic_rows.empty else 0,
            "mean_topic_share": float(topic_rows["topic_share"].mean()) if not topic_rows.empty else 0.0,
            "median_topic_share": float(topic_rows["topic_share"].median()) if not topic_rows.empty else 0.0,
        })

    if ranking_frames:
        project_scores = (
            pd.concat(ranking_frames, ignore_index=True)
            .groupby("project_id", as_index=False)
            .agg(
                total_topic_share=("topic_share", "sum"),
                total_weight=("weight", "sum"),
            )
        )
        project_scores["project_id_numeric"] = pd.to_numeric(
            project_scores["project_id"], errors="coerce"
        )
        project_scores = project_scores.sort_values(
            ["total_topic_share", "total_weight", "project_id_numeric", "project_id"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        top_project_ids = project_scores["project_id"].tolist()[:top_project_id_limit]
        supporting_project_count = int(len(project_scores))
    else:
        top_project_ids = []
        supporting_project_count = 0

    verified_topic_count = len(support_rows)
    verification_ratio = (
        verified_topic_count / claimed_topic_count if claimed_topic_count > 0 else 0.0
    )
    mean_topic_share_all = (
        float(np.mean([r["mean_topic_share"] for r in support_rows])) if support_rows else 0.0
    )

    flat_row: dict[str, Any] = {
        "run_id": run_id,
        "insight_id": candidate["insight_id"],
        "section": candidate["section"],
        "category_bucket": candidate["category_bucket"],
        "title": insight.get("title", ""),
        "finding": insight.get("finding", ""),
        "evidence_basis": insight.get("evidence_basis", ""),
        "scope_or_caveat": insight.get("scope_or_caveat", ""),
        "why_it_matters": insight.get("why_it_matters", ""),
        "source_topics_verified": source_topics,
        "source_topics_claimed": insight.get("source_topics_claimed", []),
        "claimed_topic_count": int(claimed_topic_count),
        "verified_topic_count": int(verified_topic_count),
        "verification_ratio": float(verification_ratio),
        "supporting_project_count": int(supporting_project_count),
        "mean_topic_share_all_verified_topics": float(mean_topic_share_all),
        "top_project_ids": top_project_ids,
    }
    return flat_row, support_rows


def build_verified_insight_tables(
    insights_data: dict[str, Any],
    output_group_key: str,
    *,
    groupby_field: str,
    bridge_lookup: dict[str, pd.DataFrame],
    label_index: dict[tuple[str, int], Any],
    run_id: str | None = None,
    top_project_id_limit: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the flat insights table and per-topic support table for all candidates.

    Iterates every candidate insight via iter_candidate_insights() and calls
    summarize_insight_support() for each one.

    Args:
        insights_data:       Raw insights dict from the synthesis step.
        output_group_key:    By-group key (e.g. "by_group").
        groupby_field:       The analysis grouping column.
        bridge_lookup:       From build_bridge_lookup().
        label_index:         From build_label_index().
        run_id:              Run ID for provenance.
        top_project_id_limit: Maximum project IDs per insight row.

    Returns:
        (insights_flat_df, insight_topic_support_df)
    """
    flat_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []

    for candidate in iter_candidate_insights(insights_data, output_group_key):
        flat_row, topic_rows = summarize_insight_support(
            candidate,
            groupby_field=groupby_field,
            run_id=run_id,
            bridge_lookup=bridge_lookup,
            label_index=label_index,
            top_project_id_limit=top_project_id_limit,
        )
        flat_rows.append(flat_row)
        support_rows.extend(topic_rows)

    return pd.DataFrame(flat_rows), pd.DataFrame(support_rows)


# ── 19. Packaging & tiering ───────────────────────────────────────────────────


def apply_deterministic_packaging(
    insights_flat_df: pd.DataFrame,
    *,
    output_group_key: str,
    packaging_cfg: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Apply quality thresholds to accept insights into the report pool.

    Filters on three thresholds from params.yaml → analysis.packaging:
        min_verified_topic_count
        min_supporting_project_count
        min_mean_topic_share

    The main/appendix split is NOT performed here — that is handled downstream
    by assign_topline_sections_simple(). This function only determines which
    insights clear the quality bar.

    Args:
        insights_flat_df: Output of build_verified_insight_tables().
        output_group_key: By-group key (e.g. "by_group"). Unused directly
                          but kept in signature for caller compatibility.
        packaging_cfg:    Dict from params.yaml → analysis.packaging.

    Returns:
        Dict with a single key "accepted_df" containing the accepted insights,
        sorted by supporting_project_count descending.
    """
    pack_df = insights_flat_df.copy()
    pack_df["verified_topic_count"] = pack_df["verified_topic_count"].fillna(0).astype(int)
    pack_df["claimed_topic_count"] = pack_df["claimed_topic_count"].fillna(0).astype(int)
    pack_df["supporting_project_count"] = pack_df["supporting_project_count"].fillna(0).astype(int)
    pack_df["mean_topic_share_all_verified_topics"] = (
        pack_df["mean_topic_share_all_verified_topics"].fillna(0.0).astype(float)
    )
    if "verification_ratio" not in pack_df.columns:
        pack_df["verification_ratio"] = np.where(
            pack_df["claimed_topic_count"] > 0,
            pack_df["verified_topic_count"] / pack_df["claimed_topic_count"],
            0.0,
        )

    accepted_df = pack_df[
        (pack_df["verified_topic_count"] >= packaging_cfg["min_verified_topic_count"])
        & (pack_df["supporting_project_count"] >= packaging_cfg["min_supporting_project_count"])
        & (pack_df["mean_topic_share_all_verified_topics"] >= packaging_cfg["min_mean_topic_share"])
    ].copy()

    accepted_df = accepted_df.sort_values(
        ["supporting_project_count", "verified_topic_count",
         "mean_topic_share_all_verified_topics", "title"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return {"accepted_df": accepted_df}


def dedupe_packaged_insights(
    accepted_df: pd.DataFrame,
    *,
    dedupe_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove obvious duplicate insights from the accepted pool.

    Insights are processed in quality-rank order (highest mean_topic_share first).
    Each candidate is compared against all already-kept insights. A candidate is
    dropped if it is a key_vs_key or bg_vs_same_bg pair with a kept insight and
    meets the deterministic duplicate thresholds from params.yaml → analysis.dedupe.

    bg_vs_key pairs are never deduped (cross-group and by-group insights serve
    different purposes even when covering similar territory).

    Args:
        accepted_df:  Output of apply_deterministic_packaging()["accepted_df"].
        dedupe_cfg:   Dict from params.yaml → analysis.dedupe.

    Returns:
        (curated_df, audit_df) where audit_df records every dropped pair
        with overlap scores and reason.
    """
    working_df = accepted_df.copy()
    working_df["verified_topics_list"] = working_df["source_topics_verified"].apply(_topic_list)
    working_df["title_tokens"] = working_df["title"].apply(_token_set)
    working_df["text_tokens"] = (
        working_df["title"].fillna("") + " " + working_df["finding"].fillna("")
    ).apply(_token_set)

    working_df = working_df.sort_values(
        ["mean_topic_share_all_verified_topics", "supporting_project_count", "verification_ratio"],
        ascending=False,
    ).reset_index(drop=True)

    kept_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for _, row in working_df.iterrows():
        row_dict = row.to_dict()
        drop_current = False

        for kept in kept_rows:
            screen_meta = _screen_pair(row_dict, kept)
            if screen_meta is None or screen_meta["pair_kind"] == "bg_vs_key":
                continue

            topic_overlap = screen_meta["topic_overlap"]
            title_overlap = screen_meta["title_overlap"]
            text_overlap = screen_meta["text_overlap"]

            is_obvious_duplicate = (
                (topic_overlap >= dedupe_cfg["topic_overlap_high"])
                or (
                    topic_overlap >= dedupe_cfg["topic_overlap_min"]
                    and title_overlap >= dedupe_cfg["title_overlap_min"]
                    and text_overlap >= dedupe_cfg["text_overlap_min"]
                )
            )
            if is_obvious_duplicate:
                drop_current = True
                audit_rows.append({
                    "dropped_insight_id": row_dict["insight_id"],
                    "matched_kept_insight_id": kept["insight_id"],
                    "pair_kind": screen_meta["pair_kind"],
                    "topic_overlap": topic_overlap,
                    "title_overlap": title_overlap,
                    "text_overlap": text_overlap,
                    "reason": "deterministic_obvious_duplicate",
                })
                break

        if not drop_current:
            kept_rows.append(row_dict)

    return pd.DataFrame(kept_rows).copy(), pd.DataFrame(audit_rows).copy()


def assign_topline_sections_simple(
    curated_df: pd.DataFrame,
    *,
    output_group_key: str,
    main_cross_limit: int,
    main_min_verification_ratio: float = 0.0,
) -> pd.DataFrame:
    """Assign report_section and report_order to each curated insight.

    Main section:
        - Top N cross-category insights (by quality rank), up to main_cross_limit,
          optionally filtered by main_min_verification_ratio.
        - Top 1 by-group insight per category (by quality rank).
    Appendix:
        - All remaining accepted insights.

    Args:
        curated_df:                    Output of dedupe_packaged_insights()[0].
        output_group_key:              By-group section key (e.g. "by_group").
        main_cross_limit:              Max cross-category insights in the main section.
                                       From params.yaml → analysis.packaging.main_cross_limit.
        main_min_verification_ratio:   Minimum verification_ratio for main-section
                                       eligibility. From params.yaml →
                                       analysis.packaging.main_min_verification_ratio.
                                       Default 0.0 applies no filter (backward compatible).

    Returns:
        DataFrame with report_section ("main_cross", "main_by_group",
        "appendix_cross", "appendix_by_group") and report_order columns added.
    """
    rank_cols = [
        "supporting_project_count", "verified_topic_count",
        "mean_topic_share_all_verified_topics", "title",
    ]
    rank_asc = [False, False, False, True]
    working_df = curated_df.copy()

    # Cross-category main section — apply verification ratio filter if configured.
    cross_pool = working_df[working_df["section"] == "key_insights"]
    if main_min_verification_ratio > 0.0 and "verification_ratio" in cross_pool.columns:
        cross_pool = cross_pool[cross_pool["verification_ratio"] >= main_min_verification_ratio]

    main_cross_df = (
        cross_pool.sort_values(rank_cols, ascending=rank_asc)
        .head(main_cross_limit)
        .copy()
    )
    main_cross_df["report_section"] = "main_cross"
    main_cross_df["report_order"] = range(1, len(main_cross_df) + 1)

    # By-group main section — top 1 per category.
    by_group_pool = working_df[working_df["section"] == output_group_key]
    if main_min_verification_ratio > 0.0 and "verification_ratio" in by_group_pool.columns:
        by_group_pool = by_group_pool[by_group_pool["verification_ratio"] >= main_min_verification_ratio]

    main_by_group_df = (
        by_group_pool.sort_values(
            ["category_bucket", "supporting_project_count", "verified_topic_count",
             "mean_topic_share_all_verified_topics", "title"],
            ascending=[True, False, False, False, True],
        )
        .groupby("category_bucket", as_index=False, sort=True)
        .head(1)
        .copy()
    )
    main_by_group_df["report_section"] = "main_by_group"
    main_by_group_df["report_order"] = range(1, len(main_by_group_df) + 1)

    # Appendix — everything not in main.
    main_ids = set(main_cross_df["insight_id"]) | set(main_by_group_df["insight_id"])
    remainder_df = working_df[~working_df["insight_id"].isin(main_ids)].copy()

    appendix_cross_df = (
        remainder_df[remainder_df["section"] == "key_insights"]
        .sort_values(rank_cols, ascending=rank_asc)
        .copy()
    )
    appendix_cross_df["report_section"] = "appendix_cross"
    appendix_cross_df["report_order"] = range(1, len(appendix_cross_df) + 1)

    appendix_by_group_df = (
        remainder_df[remainder_df["section"] == output_group_key]
        .sort_values(
            ["category_bucket", "supporting_project_count", "verified_topic_count",
             "mean_topic_share_all_verified_topics", "title"],
            ascending=[True, False, False, False, True],
        )
        .copy()
    )
    appendix_by_group_df["report_section"] = "appendix_by_group"
    appendix_by_group_df["report_order"] = (
        appendix_by_group_df.groupby("category_bucket", sort=True).cumcount() + 1
    )

    return pd.concat(
        [main_cross_df, main_by_group_df, appendix_cross_df, appendix_by_group_df],
        ignore_index=True,
        sort=False,
    )


def build_structured_from_curated(
    curated_df: pd.DataFrame,
    *,
    output_group_key: str,
) -> dict[str, Any]:
    """Convert the curated insights DataFrame into the nested JSON structure for the report.

    Produces the insights_structured.json format consumed by build_packaged_report_docx().

    Args:
        curated_df:       Output of assign_topline_sections_simple().
        output_group_key: By-group section key (e.g. "by_group").

    Returns:
        Dict with keys: "key_insights" (list) and output_group_key (dict of
        category → list of insight dicts).
    """
    structured: dict[str, Any] = {"key_insights": [], output_group_key: {}}

    for _, row in curated_df.sort_values(["report_section", "report_order", "title"]).iterrows():
        looker_url = row.get("looker_url", "")
        if not isinstance(looker_url, str) or pd.isna(looker_url):
            looker_url = ""

        top_project_ids = row.get("top_project_ids", [])
        if not isinstance(top_project_ids, list):
            top_project_ids = []

        item: dict[str, Any] = {
            "insight_id": row["insight_id"],
            "title": row["title"],
            "finding": row["finding"],
            "evidence_basis": row["evidence_basis"],
            "scope_or_caveat": row["scope_or_caveat"],
            "why_it_matters": row["why_it_matters"],
            "source_topics": row["source_topics_verified"],
            "supporting_project_count": int(row["supporting_project_count"]),
            "verified_topic_count": int(row["verified_topic_count"]),
            "verification_ratio": float(row["verification_ratio"]),
            "mean_topic_share_all_verified_topics": float(row["mean_topic_share_all_verified_topics"]),
            "top_project_ids": top_project_ids,
            "looker_url": looker_url,
            "report_section": row["report_section"],
            "report_order": int(row["report_order"]),
            "warnings": [],
        }

        if row["section"] == "key_insights":
            structured["key_insights"].append(item)
        else:
            structured[output_group_key].setdefault(row["category_bucket"], []).append(item)

    return structured


# ── 20. DOCX report helpers ───────────────────────────────────────────────────


def build_looker_project_url(
    *,
    base_url: str,
    project_ids: list[Any],
    filter_field: str,
    fields: list[str] | str,
    limit: int = 500,
    max_ids: int = 100,
) -> str:
    """Build a Looker Explore URL pre-filtered to a capped list of project IDs.

    Deduplicates IDs (preserving rank order), caps at max_ids, and URL-encodes
    all parameters. Returns "" when no project IDs remain after normalisation.

    Args:
        base_url:     Looker Explore base URL (params.yaml → output.looker_base_url).
        project_ids:  Ranked list of project IDs to include in the filter.
        filter_field: Looker field to filter on (params.yaml → output.looker_filter_field).
        fields:       List of fields to include, or a comma-joined string.
        limit:        Looker row limit (params.yaml → output.looker_limit).
        max_ids:      Maximum IDs in the URL filter (params.yaml → output.looker_id_limit).

    Returns:
        Full URL string, or "" if no valid project IDs are provided.
    """
    def _normalize(value: Any) -> str:
        if pd.isna(value):
            return ""
        try:
            numeric = float(value)
            if numeric.is_integer():
                return str(int(numeric))
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    normalized = list(dict.fromkeys(pid for v in project_ids if (pid := _normalize(v))))[:max_ids]
    if not normalized:
        return ""

    if isinstance(fields, (list, tuple)):
        fields_param = ",".join(str(f).strip() for f in fields if str(f).strip())
    else:
        fields_param = str(fields).strip()
    if not fields_param:
        raise ValueError("build_looker_project_url requires at least one field")

    params = {
        "fields": fields_param,
        f"f[{filter_field}]": ",".join(normalized),
        "limit": str(limit),
    }
    return f"{base_url}?{urlencode(params)}"


def add_hyperlink(paragraph: Any, text: str, url: str) -> Any:
    """Add an external hyperlink run to a python-docx paragraph.

    python-docx does not natively support hyperlinks, so this function
    manipulates the underlying XML directly.

    Args:
        paragraph: python-docx Paragraph object.
        text:      Visible link text.
        url:       Target URL.

    Returns:
        The created hyperlink XML element.
    """
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    r_pr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.append(underline)
    new_run.append(r_pr)

    text_elem = OxmlElement("w:t")
    text_elem.text = text
    new_run.append(text_elem)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def add_heading(doc: Any, text: str, level: int) -> Any:
    """Add a black Arial heading to a python-docx Document.

    python-docx headings default to the theme colour; this function overrides
    that to ensure consistent black text in the report.

    Args:
        doc:   python-docx Document object.
        text:  Heading text.
        level: Heading level (1–4).

    Returns:
        The created Paragraph object.
    """
    p = doc.add_heading(text, level=level)
    p.runs[0].font.color.rgb = RGBColor(0, 0, 0)
    p.runs[0].font.name = "Arial"
    return p


def add_insight_meta_line(
    doc: Any,
    insight: dict[str, Any],
    font_size_pt: float = 8.5,
) -> None:
    """Add a compact italic meta line above an insight body in the DOCX.

    Displays supporting project count, verified source topic count, and
    average topic fit score. Controlled by params.yaml →
    output.report_include_meta_line.

    Args:
        doc:          python-docx Document object.
        insight:      Insight dict containing support statistics.
        font_size_pt: Font size for the meta line. Default 8.5pt.
    """
    mean_topic_fit = insight.get("mean_topic_share_all_verified_topics")
    fit_str = (
        f"{round(mean_topic_fit * 100):.0f}%"
        if isinstance(mean_topic_fit, (int, float))
        else "—"
    )
    text = (
        f"Supporting projects: {insight.get('supporting_project_count', '—')}  |  "
        f"Verified source topics: {insight.get('verified_topic_count', '—')}  |  "
        f"Average topic fit: {fit_str}"
    )
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(font_size_pt)
    p.paragraph_format.space_after = Pt(2)



def add_insight(
    doc: Any,
    insight: dict[str, Any],
    *,
    include_looker_link: bool = True,
    looker_link_text: str = "Project essays for this insight",
) -> None:
    """Add one accepted insight block to the DOCX report.

    Renders: bold title, Finding, Evidence basis, Scope / caveat,
    Why it matters, and (optionally) a Looker Explore hyperlink.

    Args:
        doc:                python-docx Document object.
        insight:            Insight dict from the structured output.
        include_looker_link: When True, appends a Looker link if looker_url
                             is present. Controlled by params.yaml →
                             output.report_include_looker_link.
        looker_link_text:   Visible hyperlink text. From params.yaml →
                            output.report_looker_link_text.
    """
    p = doc.add_paragraph()
    run = p.add_run(insight.get("title", ""))
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x3E, 0x00, 0xC9)

    for label, key in [
        ("Finding:", "finding"),
        ("Evidence basis:", "evidence_basis"),
        ("Scope / caveat:", "scope_or_caveat"),
        ("Why it matters:", "why_it_matters"),
    ]:
        value = str(insight.get(key, "") or "").strip()
        if not value:
            continue
        p = doc.add_paragraph()
        label_run = p.add_run(label + "  ")
        label_run.bold = True
        label_run.font.size = Pt(10)
        body_run = p.add_run(value)
        body_run.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(2)

    if include_looker_link and insight.get("looker_url"):
        p = doc.add_paragraph()
        label_run = p.add_run("Explore supporting projects: ")
        label_run.bold = True
        label_run.font.size = Pt(10)
        add_hyperlink(p, looker_link_text, insight["looker_url"])
        p.paragraph_format.space_after = Pt(2)

    doc.add_paragraph()
def build_packaged_report_docx(
    *,
    structured: dict[str, Any],
    output_path: Any,
    output_group_key: str = "by_group",
    report_cfg: dict[str, Any],
    project_count: int | None = None,
    run_id: str | None = None,
    normal_font_name: str = "Arial",
    normal_font_size_pt: float = 10.0,
    margin_inches: float = 1.0,
) -> None:
    """Build and save the final packaged DOCX report.

    Renders main and appendix sections, each containing cross-category and
    by-group subsections. Section labels, meta lines, and Looker links are
    all controlled by params.yaml → output.report_* keys passed through
    report_cfg.

    Args:
        structured:        Nested insights dict from build_structured_from_curated().
        output_path:       Destination file path for the DOCX.
        output_group_key:  By-group section key (e.g. "by_group").
        report_cfg:        Dict from params.yaml → output block.
        project_count:     Optional project count for the summary line.
        run_id:            Optional run ID for the summary line.
        normal_font_name:  Default body font. Default "Arial".
        normal_font_size_pt: Default body font size. Default 10pt.
        margin_inches:     Page margins. Default 1 inch.
    """
    title = report_cfg.get("report_title", "Report")
    main_label = report_cfg.get("report_main_section_label", "Main Insights")
    appendix_label = report_cfg.get("report_appendix_section_label", "Appendix")
    main_cross_label = report_cfg.get("report_main_cross_label", "Cross-Category Similarities")
    main_by_group_label = report_cfg.get("report_main_by_group_label", "Group-Specific Findings")
    appendix_cross_label = report_cfg.get("report_appendix_cross_label", "Additional Cross-Category Insights")
    appendix_by_group_label = report_cfg.get("report_appendix_by_group_label", "Additional Group-Specific Findings")
    incl_meta = report_cfg.get("report_include_meta_line", True)
    incl_summary = report_cfg.get("report_include_signal_summary", True)
    incl_looker = report_cfg.get("report_include_looker_link", True)
    looker_link_text = report_cfg.get("report_looker_link_text", "Project essays for this insight")

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(margin_inches)
    style = doc.styles["Normal"]
    style.font.name = normal_font_name
    style.font.size = Pt(normal_font_size_pt)

    # Report title
    p = doc.add_paragraph()
    run = p.add_run(title)
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(0x3E, 0x00, 0xC9)
    p.paragraph_format.space_after = Pt(6)

    # Optional summary line
    if incl_summary:
        all_items = structured.get("key_insights", []) + [
            i for items in structured.get(output_group_key, {}).values() for i in items
        ]
        parts = []
        if project_count is not None:
            parts.append(f"Projects in study: {project_count:,}")
        parts.append(f"Accepted insights: {len(all_items)}")
        if run_id is not None:
            parts.append(f"Run ID: {'_'.join(str(run_id).split('_')[:2])}")
        p = doc.add_paragraph()
        run = p.add_run("  |  ".join(parts))
        run.italic = True
        run.font.size = Pt(9)
        doc.add_paragraph()

    def _items_for_section(section_name: str):
        """Return (cross_list, by_group_dict) for the given report_section value."""
        cross = [i for i in structured.get("key_insights", []) if i.get("report_section") == section_name]
        by_group = {
            g: [i for i in items if i.get("report_section") == section_name]
            for g, items in structured.get(output_group_key, {}).items()
        }
        return cross, by_group

    def _render_section(section_title, cross_title, by_group_title, cross_key, by_group_key):
        """Render one report section (main or appendix) with cross and by-group subsections."""
        cross_main, bg_main = _items_for_section(cross_key)
        cross_app, bg_app = _items_for_section(by_group_key)
        cross = cross_main + cross_app
        merged: dict[str, list] = {}
        for g, items in {**bg_main, **bg_app}.items():
            merged[g] = merged.get(g, []) + items

        if not cross and not any(merged.values()):
            return

        p = doc.add_paragraph()
        run = p.add_run(section_title)
        run.bold = True
        run.underline = True
        run.font.name = "Arial"
        run.font.size = Pt(16)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)

        if cross:
            p = doc.add_paragraph()
            run = p.add_run(cross_title)
            run.bold = True
            run.font.size = Pt(14)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(10)
            for insight in sorted(cross, key=lambda x: (x.get("report_order", 9999), x.get("title", ""))):
                if incl_meta:
                    add_insight_meta_line(doc, insight)
                add_insight(doc, insight, include_looker_link=incl_looker, looker_link_text=looker_link_text)

        if any(merged.values()):
            p = doc.add_paragraph()
            run = p.add_run(by_group_title)
            run.bold = True
            run.font.size = Pt(14)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(10)
            for group, items in sorted(merged.items()):
                if not items:
                    continue
                p = doc.add_paragraph()
                run = p.add_run(str(group))
                run.bold = True
                run.font.size = Pt(12)
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(4)
                for insight in sorted(items, key=lambda x: (x.get("report_order", 9999), x.get("title", ""))):
                    if incl_meta:
                        add_insight_meta_line(doc, insight)
                    add_insight(doc, insight, include_looker_link=incl_looker, looker_link_text=looker_link_text)

    _render_section(main_label, main_cross_label, main_by_group_label, "main_cross", "main_by_group")
    _render_section(appendix_label, appendix_cross_label, appendix_by_group_label, "appendix_cross", "appendix_by_group")
    doc.save(output_path)


# ── 21. __all__ ───────────────────────────────────────────────────────────────

__all__ = [
    # Constants
    "ROOT",
    "PIPELINE_VERSION",
    # Config & I/O
    "resolve_params_path",
    "load_cfg",
    "write_json",
    "compute_md5",
    "artifact_meta",
    "build_output_path",
    "build_run_output_path",
    "outpath",           # deprecated alias — prefer build_output_path
    "get_run_date",
    # Run identity & filter helpers
    "canonicalize_filter_spec",
    "get_filter_fields_key",
    "get_run_id",
    "validate_filter_spec",
    "apply_filters",
    # Stage & pipeline manifests
    "start_stage_manifest",
    "finalize_stage_manifest",
    "build_pipeline_manifest",
    # Warning file helpers
    "ensure_warning_file",
    "append_warning",
    "get_warning_count",
    # LLM client
    "get_llm_client",
    "chat_completion_with_tier_fallback",
    # Ingest
    "ingest",
    # Token helpers
    "tokens_to_str",
    "flat_freq",
    "token_doc_freq",
    "normalize_tokens",
    "dedupe_near_duplicate_projects",
    # Consolidation helpers
    "build_consolidation_candidates",
    # Analysis helpers
    "make_vec",
    "add_bin",
    "group_key",
    "build_project_topic_bridge",
    "slugify_group_value",
    "load_essay_snippet_lookup",
    # Strategic loop preparation helpers
    "DEFAULT_PROJECT_CATEGORY_OTHER_VALUES",
    "DEFAULT_PROJECT_CATEGORY_BUCKET_RULES",
    "DEFAULT_STRATEGIC_MIN_GROUP_PROJECTS",
    "DEFAULT_METADATA_LIFT_THRESHOLDS",
    "DEFAULT_METADATA_LIFT_DIMENSIONS",
    "DEFAULT_STATE_CLUSTERS",
    "taxonomy_tag_to_bool_col",
    "taxonomy_tag_to_injected_token",
    "coerce_token_list",
    "get_strategic_areas_config",
    "get_strategic_run_plan_config",
    "get_strategic_loop_defaults",
    "add_project_cost_bucket",
    "add_funding_status",
    "add_posting_period",
    "add_state_cluster",
    "add_strategic_derived_fields",
    "bucket_project_category_for_run",
    "build_strategic_area_tag_matches",
    "build_strategic_area_membership",
    "normalize_split_spec",
    "resolve_groupby_source_column",
    "build_combined_group_column",
    "remove_injected_tokens_for_tags",
    "filter_to_min_group_projects",
    "prepare_strategic_loop_run_dataframe",
    "expand_strategic_run_plan",
    "candidate_support_project_ids_from_source_topics",
    "build_metadata_lift_context",
    "compute_metadata_lift",
    "format_metadata_lift_facts",
    "metadata_lift_for_candidate_insight",
    # Quality
    "HARD_STOPWORDS",
    "quality_report",
    # TF-IDF & NMF helpers
    "cat_tfidf_slice",
    "choose_n_components",
    "nmf_one",
    # Enrichment helpers
    "gate_cluster",
    "classify_batch",
    "inject_tokens",
    # Topic labeling helpers
    "_norm_group_value",
    "_safe_topic_id",
    "build_input",
    "_make_label_error",
    "_label_with_retry",
    # Synthesis helpers
    "clean_label",
    "build_topic_lines",
    "build_unit_lines",
    "build_per_group_prompt",
    "_call_with_retry",
    "synthesize_one_group",
    # Insight normalization & verification
    "strip_json_fences",
    "normalize_source_topics",
    "normalize_insight",
    "project_insight_for_saved_candidates",
    "verify_source_topics",
    "_verify_insight_list",
    # Dedup helpers
    "_norm_text",
    "_token_set",
    "_jaccard",
    "_topic_list",
    "_pair_kind",
    "_screen_pair",
    # Evidence & support tables
    "get_topic_key",
    "iter_candidate_insights",
    "_parse_topic_id",
    "build_bridge_lookup",
    "build_label_index",
    "summarize_insight_support",
    "build_verified_insight_tables",
    # Packaging & tiering
    "apply_deterministic_packaging",
    "dedupe_packaged_insights",
    "assign_topline_sections_simple",
    "build_structured_from_curated",
    # DOCX report helpers
    "add_hyperlink",
    "add_heading",
    "add_insight_meta_line",
    "add_insight",
    "build_packaged_report_docx",
    "build_looker_project_url",
]

def build_cluster_input(
    cluster_row: Any,
    membership_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    pid_text: pd.Series,
    groupby_field: str,
    n_representative_per_member: int = 1,
    top_terms_per_member: int = 8,
) -> dict[str, Any]:
    """Build one cluster-labeling payload from a cluster row and its members.

    Produces a structured input dict the cluster prompt can format directly.
    All list-like fields are coerced through coerce_token_list() so the
    function works whether the cluster row came from an in-memory DataFrame
    or was reloaded from CSV (where lists arrive as stringified literals).

    Args:
        cluster_row:                 Row from analysis_units_df where
                                     unit_type == "cluster".
        membership_df:               cluster_membership_df, filtered or full.
        weights_df:                  NMF weight bridge DataFrame.
        pid_text:                    Series mapping project_id → token snippet.
        groupby_field:               Analysis grouping column.
        n_representative_per_member: Representative project snippets per
                                     member group in the prompt. Default 1
                                     keeps prompts tight on large clusters.
        top_terms_per_member:        Cap on per-member top_terms shown.

    Returns:
        Dict with keys: cluster_id, unit_id, n_topics, n_groups, shared_core,
        concrete_shared_core, medoid_group, medoid_topic_id, medoid_top_terms,
        groups_present, members, members_block.
    """
    cid = int(cluster_row["cluster_id"])
    members = membership_df[membership_df["cluster_id"] == cid].copy()

    member_dicts: list[dict[str, Any]] = []
    for _, m in members.iterrows():
        group = m[groupby_field]
        topic_id = int(m["topic_id"])
        mask = (weights_df[groupby_field] == group) & (weights_df["topic_id"] == topic_id)
        rep_pids = (
            weights_df[mask]
            .sort_values("weight", ascending=False)["project_id"]
            .tolist()[:n_representative_per_member]
        )
        snippets = [pid_text.get(p, "") for p in rep_pids]
        member_dicts.append({
            "group": str(group),
            "topic_id": topic_id,
            "is_medoid": bool(m["is_medoid"]),
            "distinctive_terms": coerce_token_list(m.get("distinctive_terms")),
            "top_terms": coerce_token_list(m.get("top_terms"))[:top_terms_per_member],
            "snippets": [s for s in snippets if s],
        })

    # Render the per-member block once so the prompt template stays simple.
    lines: list[str] = []
    for md in member_dicts:
        flag = " (medoid)" if md["is_medoid"] else ""
        lines.append(f"Group: {md['group']}{flag}")
        distinct_str = ", ".join(md["distinctive_terms"]) or "(none)"
        lines.append(f"  Distinctive terms (in this member only): {distinct_str}")
        lines.append(f"  Top NMF terms: {', '.join(md['top_terms'])}")
        for s in md["snippets"]:
            lines.append(f"  - {s}")
        lines.append("")
    members_block = "\n".join(lines).rstrip()

    # Prefer the upstream unit_id when present; fall back to cluster_{id}.
    unit_id = str(cluster_row.get("unit_id") or f"cluster_{cid}")

    return {
        "cluster_id": cid,
        "unit_id": unit_id,
        "n_topics": int(cluster_row["n_topics"]),
        "n_groups": int(cluster_row["n_groups"]),
        "shared_core": coerce_token_list(cluster_row.get("shared_core")),
        "concrete_shared_core": coerce_token_list(cluster_row.get("concrete_shared_core")),
        "medoid_group": str(cluster_row.get("medoid_group", "")),
        "medoid_topic_id": int(cluster_row.get("medoid_topic_id", -1)),
        "medoid_top_terms": coerce_token_list(cluster_row.get("medoid_top_terms")),
        "groups_present": coerce_token_list(cluster_row.get("groups_present")),
        "members": member_dicts,
        "members_block": members_block,
    }


def _make_cluster_label_error(
    inp: dict[str, Any],
    raw_text: str,
    code: str,
    model_labeling: str,
    error_text: str | None = None,
) -> dict[str, Any]:
    """Return a structured error object for cluster-label failures.

    Parallel to _make_label_error() but with cluster-shaped identity fields.
    """
    return {
        "raw": raw_text,
        "parse_error": True,
        "error_code": code,
        "error": error_text,
        "model": model_labeling,
        "timestamp": datetime.now().isoformat(),
        "unit_type": "cluster",
        "unit_id": inp["unit_id"],
        "cluster_id": inp["cluster_id"],
    }


_VALID_COHERENCE_FLAGS = {"coherent", "mixed", "redundant", "unclear"}


def _label_cluster_with_retry(
    inp: dict[str, Any],
    *,
    client: OpenAI,
    model_labeling: str,
    system_prompt: str,
    user_prompt_template: str,
    warnings_path: Path,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    service_tier: str = "flex",
    flex_attempts: int = 2,
    fallback_service_tier: str = "default",
    timeout_seconds: float = 900.0,
    flex_retry_delay_seconds: float = 0.0,
    stage_name: str = "03_insights_generation",
) -> dict[str, Any]:
    """Call the labeling model for one cluster with retry and structured errors.

    Validation performed on each response:
    - variation_notes covers every group in groups_present (no missing)
    - variation_notes contains no groups not in groups_present (no unknown)
    - variation_notes has no duplicate group entries
    - every distinctive_angle is non-empty
    - coherence_flag is one of the four valid values

    On validation failure, the next attempt receives an explicit repair note
    describing exactly what was wrong, so the model has signal to correct.

    Returns:
        Parsed cluster-label dict on success, or a _make_cluster_label_error()
        dict on failure.
    """
    text = ""
    expected_groups = {str(g) for g in inp.get("groups_present", [])}
    repair_note = ""

    for attempt in range(max_retries + 1):
        try:
            user_content = user_prompt_template.format(**inp) + repair_note
            resp = chat_completion_with_tier_fallback(
                client=client,
                model=model_labeling,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                service_tier=service_tier,
                flex_attempts=flex_attempts,
                fallback_service_tier=fallback_service_tier,
                timeout_seconds=timeout_seconds,
                flex_retry_delay_seconds=flex_retry_delay_seconds,
            )
            text = resp.choices[0].message.content.strip()
            obj = json.loads(text)

            # Validate variation_notes coverage, duplicates, and angle emptiness.
            notes = obj.get("variation_notes", []) or []
            note_group_list = [
                str(n.get("group", "")) for n in notes if isinstance(n, dict)
            ]
            note_groups_set = set(note_group_list)
            unknown_groups = sorted(note_groups_set - expected_groups)
            missing_groups = sorted(expected_groups - note_groups_set)
            duplicate_groups = sorted(
                {g for g in note_group_list if note_group_list.count(g) > 1}
            )
            empty_angle_groups = sorted({
                str(n.get("group", ""))
                for n in notes
                if isinstance(n, dict)
                and not str(n.get("distinctive_angle", "")).strip()
            })

            # Validate coherence_flag.
            cflag = obj.get("coherence_flag")
            cflag_bad = cflag not in _VALID_COHERENCE_FLAGS

            validation_problems = (
                unknown_groups or missing_groups
                or duplicate_groups or empty_angle_groups
                or cflag_bad
            )

            if validation_problems:
                if attempt < max_retries:
                    # Build a targeted repair note for the next attempt.
                    repair_parts = ["Previous response failed validation."]
                    if missing_groups:
                        repair_parts.append(
                            f"You omitted these required groups from variation_notes: {missing_groups}."
                        )
                    if unknown_groups:
                        repair_parts.append(
                            f"You included groups that are not in Groups present and must be removed: {unknown_groups}."
                        )
                    if duplicate_groups:
                        repair_parts.append(
                            f"These groups appeared more than once and must appear exactly once: {duplicate_groups}."
                        )
                    if empty_angle_groups:
                        repair_parts.append(
                            f"These groups had empty distinctive_angle values; provide a real angle or the literal "
                            f"string 'no distinctive angle': {empty_angle_groups}."
                        )
                    if cflag_bad:
                        repair_parts.append(
                            f"coherence_flag must be one of {sorted(_VALID_COHERENCE_FLAGS)}; got {cflag!r}."
                        )
                    repair_parts.append(
                        f"Return variation_notes with exactly these groups, each appearing once: {sorted(expected_groups)}."
                    )
                    repair_note = "\n\n" + " ".join(repair_parts)
                    if retry_delay_seconds > 0:
                        _time.sleep(retry_delay_seconds)
                    continue

                # Final attempt failed validation; accept with a flag so
                # downstream can decide what to do.
                append_warning(
                    warnings_path, stage_name, "CLUSTER_LABELING_VALIDATION_FAILED",
                    f"Cluster {inp['cluster_id']} validation failed after {max_retries} retries",
                    context={
                        "cluster_id": inp["cluster_id"],
                        "unknown_groups": unknown_groups,
                        "missing_groups": missing_groups,
                        "duplicate_groups": duplicate_groups,
                        "empty_angle_groups": empty_angle_groups,
                        "coherence_flag": cflag,
                    },
                )
                obj["validation_warning"] = True

            obj["unit_type"] = "cluster"
            obj["unit_id"] = inp["unit_id"]
            obj["cluster_id"] = inp["cluster_id"]
            obj["model"] = model_labeling
            obj["timestamp"] = datetime.now().isoformat()
            return obj

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
                continue
            append_warning(
                warnings_path, stage_name, "CLUSTER_LABELING_PARSE_FAILURE",
                f"JSON parse failure for cluster {inp['cluster_id']}",
                context={"cluster_id": inp["cluster_id"], "error": str(e)},
            )
            return _make_cluster_label_error(
                inp, text, "CLUSTER_LABELING_PARSE_FAILURE", model_labeling, str(e)
            )

        except (_openai.RateLimitError, _openai.APITimeoutError) as e:
            if attempt < max_retries:
                if retry_delay_seconds > 0:
                    _time.sleep(retry_delay_seconds)
            else:
                append_warning(
                    warnings_path, stage_name, "CLUSTER_LABELING_API_FAILURE",
                    f"API failure after retries for cluster {inp['cluster_id']}",
                    context={"cluster_id": inp["cluster_id"], "error": str(e)},
                )
                return _make_cluster_label_error(
                    inp, text or str(e), "CLUSTER_LABELING_API_FAILURE",
                    model_labeling, str(e)
                )

        except Exception as e:
            append_warning(
                warnings_path, stage_name, "CLUSTER_LABELING_API_FAILURE",
                f"Unexpected error for cluster {inp['cluster_id']}",
                context={"cluster_id": inp["cluster_id"], "error": str(e)},
            )
            return _make_cluster_label_error(
                inp, text or str(e), "CLUSTER_LABELING_API_FAILURE",
                model_labeling, str(e)
            )