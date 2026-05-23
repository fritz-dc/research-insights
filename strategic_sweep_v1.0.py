#!/usr/bin/env python3
"""run_strategic_sweep.py

Run NB03 (insights generation) across every (strategic_area, split) combo
defined in CONFIG/splits.yaml, without manual editing of params.yaml or
re-running the notebook by hand.

What it does
------------
1. Discovers the latest 03_*.ipynb in the project root (by mtime).
2. Converts it once to a Python script in sweep_workdir/.
3. Reads CONFIG/splits.yaml and enumerates every (area, split) pair
   (first_split + each entry of second_splits). Combined-mapping splits
   (i.e. a dict with groupby_fields/label) are skipped with a warning.
4. Backs up the resolved params*.yaml file.
5. For each combo:
   - Updates strategic_loop.enabled / strategic_area_id / split IN PLACE
     in the params file, preserving comments, formatting, and every other
     key.
   - Runs the converted script in a fresh subprocess, streaming output
     live and to a per-combo log file.
   - Writes per-combo status to a JSONL log.
6. Restores the original params file on normal exit, on Ctrl-C, or on
   uncaught exception.

Combo semantics
---------------
Strict (area, split) enumeration as written in splits.yaml — no implicit
combinations, no skip-if-exists, always re-runs. Failures are logged and
the sweep continues.

Params lookup
-------------
Mirrors utils.resolve_params_path():
  1. $TREND_TRACKER_PARAMS env var if set
  2. highest-versioned params*.yaml in ROOT
  3. highest-versioned params*.yaml in ROOT/CONFIG

The subprocess is launched with TREND_TRACKER_PARAMS set to the same
resolved path, so the executed notebook reads the same file the runner
just edited.

Usage
-----
    python run_strategic_sweep.py                  # full sweep
    python run_strategic_sweep.py --dry-run        # validate combos + edits, no NB03
    python run_strategic_sweep.py --limit 2        # smoke test with first 2 combos
    python run_strategic_sweep.py --timeout 1800   # per-combo timeout, seconds

Outputs (under sweep_workdir/)
------------------------------
    nb03_executable_<ts>.py        the converted notebook
    params_backup_<ts>.yaml        pre-sweep snapshot of the params file
    sweep_log_<ts>.jsonl           per-combo records
    sweep_summary_<ts>.json        final status summary
    combo_logs_<ts>/<combo>.log    per-combo subprocess output

Notes
-----
- update_strategic_loop() uses targeted line-level string editing so YAML
  comments are preserved. It expects all three keys (enabled,
  strategic_area_id, split) to already exist under strategic_loop:.
- The runner runs combos serially (one subprocess at a time) by design.
- Tested only on the current shape of splits.yaml and params_v3.yaml.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import nbformat
import yaml
from nbconvert import PythonExporter


# ── Project layout ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
CONFIG_DIR = ROOT / "CONFIG"
SPLITS_YAML = CONFIG_DIR / "splits.yaml"
WORKDIR = ROOT / "sweep_workdir"


# ── 1. Notebook discovery + conversion ────────────────────────────────────────
def find_latest_nb03(root: Path) -> Path:
    """Pick the most-recently-modified 03_*.ipynb in the project root."""
    candidates = [p for p in root.glob("03_*.ipynb") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No 03_*.ipynb found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def convert_nb_to_py(nb_path: Path, out_path: Path) -> None:
    """Convert a notebook to a runnable .py via nbconvert's PythonExporter."""
    nb = nbformat.read(str(nb_path), as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)
    # Shim for IPython display() calls that nbconvert leaves in but doesn't import.
    shim = "from IPython.display import display\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(shim + source, encoding="utf-8")
    

# ── 2. Params resolution (mirrors utils.resolve_params_path) ──────────────────
def _version_sort_key(path: Path) -> tuple:
    m = re.search(r"v(\d+)", path.stem)
    return (int(m.group(1)) if m else -1, path.name.lower())


def resolve_params_path(root: Path) -> Path:
    """Find the same params file the notebook would load."""
    env_path = os.environ.get("TREND_TRACKER_PARAMS")
    if env_path:
        return Path(env_path)
    candidates: list[Path] = []
    seen: set[str] = set()
    for base in [root, root / "CONFIG"]:
        for pattern in ("params*.yaml", "params*.yml"):
            for p in base.glob(pattern):
                if not p.is_file():
                    continue
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No params*.yaml found in ROOT or CONFIG/")
    return sorted(candidates, key=_version_sort_key, reverse=True)[0]


# ── 3. Combo enumeration ──────────────────────────────────────────────────────
def enumerate_combos(splits_yaml_path: Path) -> tuple[list[tuple[str, str]], list[dict]]:
    """Return (combos, skipped) from splits.yaml run_plan.

    combos:   [(area_id, split_id), ...] preserving YAML order.
    skipped:  records for non-string splits (the combined dict form).
              These are not coded for per the current request.
    """
    with open(splits_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    run_plan = data.get("run_plan", {}) or {}

    combos: list[tuple[str, str]] = []
    skipped: list[dict] = []

    for area, spec in run_plan.items():
        spec = spec or {}
        first = spec.get("first_split")
        if isinstance(first, str):
            combos.append((area, first))
        elif first is not None:
            skipped.append({
                "area": area,
                "where": "first_split",
                "reason": "non_string_split_unsupported",
                "detail": repr(first)[:200],
            })

        for item in (spec.get("second_splits") or []):
            if isinstance(item, str):
                combos.append((area, item))
            else:
                skipped.append({
                    "area": area,
                    "where": "second_splits",
                    "reason": "non_string_split_unsupported",
                    "detail": repr(item)[:200],
                })

    return combos, skipped


# ── 4. In-place params editor (preserves comments) ────────────────────────────
_STRATEGIC_LOOP_HEADER_RE = re.compile(r"^strategic_loop\s*:\s*(?:#.*)?$")


def _replace_value_preserve_comment(line: str, new_value: str) -> str:
    """Replace the scalar value on a 'key: value  # comment' line.

    Leaves the key, indentation, surrounding whitespace, inline comment,
    and line ending untouched. If the line shape isn't recognized, returns
    the original line unchanged.
    """
    m = re.match(
        r"^(?P<head>\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*)"
        r"(?P<value>[^#\n]*?)"
        r"(?P<tail>\s*(?:#.*)?)"
        r"(?P<eol>\r?\n?)$",
        line,
    )
    if not m:
        return line
    return f"{m.group('head')}{new_value}{m.group('tail')}{m.group('eol')}"


def update_strategic_loop(
    text: str,
    *,
    area_id: str,
    split_id: str,
    enabled: bool = True,
) -> str:
    """Update strategic_loop.{enabled, strategic_area_id, split} in YAML text.

    Targeted line-level edit. All comments and unrelated keys are preserved
    byte-for-byte. Raises if strategic_loop isn't present or if any of the
    three target keys are missing from its body.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)

    # Locate the strategic_loop: top-level header (column 0).
    header_idx = None
    for i, line in enumerate(lines):
        if _STRATEGIC_LOOP_HEADER_RE.match(line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            "strategic_loop: top-level key not found in params file"
        )

    # Determine the block's child indent from the first non-blank, non-comment line.
    block_indent: int | None = None
    body_start = header_idx + 1
    for j in range(body_start, n):
        s = lines[j]
        stripped = s.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(s) - len(stripped)
        if indent == 0:
            break  # next top-level key without any body
        block_indent = indent
        break
    if block_indent is None:
        raise ValueError("strategic_loop: block has no child keys")

    targets = {
        "enabled": "true" if enabled else "false",
        "strategic_area_id": area_id,
        "split": split_id,
    }
    edited: set[str] = set()

    for j in range(body_start, n):
        s = lines[j]
        stripped = s.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(s) - len(stripped)
        if indent < block_indent:
            break  # exited the strategic_loop block
        if indent > block_indent:
            continue  # nested child of a child, leave alone
        key_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:", stripped)
        if not key_match:
            continue
        key = key_match.group(1)
        if key in targets and key not in edited:
            lines[j] = _replace_value_preserve_comment(s, targets[key])
            edited.add(key)
        if len(edited) == len(targets):
            break

    missing = set(targets) - edited
    if missing:
        raise ValueError(
            f"strategic_loop block is missing required keys: {sorted(missing)}. "
            "Add them to the params file before running the sweep."
        )
    return "".join(lines)


# ── 5. Per-combo subprocess runner (live-streamed) ────────────────────────────
def run_one_combo(
    executable_py: Path,
    *,
    env: dict,
    timeout_sec: int | None,
    combo_log_path: Path,
    indent: str = "    ",
) -> dict:
    """Run the converted NB03 script in a fresh subprocess.

    Streams subprocess stdout/stderr live to this process's stdout (with
    an indent prefix) and tees it to combo_log_path. Returns a status
    dict for the sweep log.
    """
    started = time.time()
    combo_log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        with open(combo_log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(executable_py)],
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(f"{indent}{line}")
                sys.stdout.flush()
                log_f.write(line)

            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return {
                    "status": "timeout",
                    "returncode": None,
                    "elapsed_sec": round(time.time() - started, 1),
                    "log_file": str(combo_log_path),
                }
        return {
            "status": "success" if proc.returncode == 0 else "failure",
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 1),
            "log_file": str(combo_log_path),
        }
    except Exception as e:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        return {
            "status": "error",
            "returncode": None,
            "elapsed_sec": round(time.time() - started, 1),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "log_file": str(combo_log_path),
        }


# ── 6. Orchestration ──────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate combo enumeration and exercise the params editor for each combo; do not execute NB03.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N combos (after enumeration order). For smoke testing.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-combo subprocess timeout in seconds. Default: no timeout.",
    )
    args = ap.parse_args()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = WORKDIR / f"sweep_log_{ts}.jsonl"
    summary_path = WORKDIR / f"sweep_summary_{ts}.json"
    executable_py = WORKDIR / f"nb03_executable_{ts}.py"
    combo_logs_dir = WORKDIR / f"combo_logs_{ts}"

    # 6.1 Find and convert the notebook.
    nb_path = find_latest_nb03(ROOT)
    print(f"[sweep] notebook        : {nb_path.name}")
    convert_nb_to_py(nb_path, executable_py)
    print(f"[sweep] converted to    : {executable_py.relative_to(ROOT)}")

    # 6.2 Resolve and back up the params file.
    params_path = resolve_params_path(ROOT)
    rel_params = (
        params_path.relative_to(ROOT) if params_path.is_relative_to(ROOT) else params_path
    )
    print(f"[sweep] params file     : {rel_params}")
    params_backup = WORKDIR / f"params_backup_{ts}{params_path.suffix}"
    shutil.copy2(params_path, params_backup)
    print(f"[sweep] params backed up: {params_backup.relative_to(ROOT)}")
    original_text = params_path.read_text(encoding="utf-8")

    # 6.3 Enumerate combos.
    combos, skipped = enumerate_combos(SPLITS_YAML)
    print(f"[sweep] combos to run   : {len(combos)}")
    if skipped:
        print(f"[sweep] combos skipped  : {len(skipped)} (non-string split entries)")
        for s in skipped:
            print(f"          - {s['area']} / {s['where']}: {s['detail']}")
    if args.limit:
        combos = combos[: args.limit]
        print(f"[sweep] limited to      : {len(combos)} (--limit)")

    # 6.4 Always-restore guard for Ctrl-C / SIGTERM.
    def _restore_and_exit(signum, _frame):
        print(f"\n[sweep] received signal {signum}; restoring params and exiting.")
        try:
            params_path.write_text(original_text, encoding="utf-8")
        except Exception as e:
            print(f"[sweep] WARNING: failed to restore params: {e}")
        sys.exit(130)

    signal.signal(signal.SIGINT, _restore_and_exit)
    try:
        signal.signal(signal.SIGTERM, _restore_and_exit)
    except (AttributeError, ValueError):
        pass  # SIGTERM may not be settable on some platforms

    # 6.5 Run the sweep.
    results: list[dict] = []
    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            for combo_idx, (area, split_id) in enumerate(combos, 1):
                combo_t0 = time.time()
                header = f"[{combo_idx:02d}/{len(combos)}] {area} × {split_id}"
                print(f"\n{header}")
                print("-" * len(header))

                # Edit params (raises if structure unexpected).
                try:
                    new_text = update_strategic_loop(
                        original_text,
                        area_id=area,
                        split_id=split_id,
                        enabled=True,
                    )
                except Exception as e:
                    record = {
                        "combo_idx": combo_idx,
                        "area": area,
                        "split": split_id,
                        "status": "params_edit_failed",
                        "error": f"{type(e).__name__}: {e}",
                        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                    print(f"  ! params edit failed: {e}")
                    results.append(record)
                    log_f.write(json.dumps(record) + "\n")
                    log_f.flush()
                    continue

                params_path.write_text(new_text, encoding="utf-8")

                if args.dry_run:
                    record = {
                        "combo_idx": combo_idx,
                        "area": area,
                        "split": split_id,
                        "status": "dry_run_ok",
                        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                    print(f"  ✓ params edited (dry run, NB03 not executed)")
                else:
                    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split_id)
                    combo_label = f"{combo_idx:02d}_{area}__{safe_split}"
                    combo_log_path = combo_logs_dir / f"{combo_label}.log"

                    sub_env = {**os.environ, "TREND_TRACKER_PARAMS": str(params_path)}
                    result = run_one_combo(
                        executable_py,
                        env=sub_env,
                        timeout_sec=args.timeout,
                        combo_log_path=combo_log_path,
                    )
                    record = {
                        "combo_idx": combo_idx,
                        "area": area,
                        "split": split_id,
                        "started_at": dt.datetime.fromtimestamp(combo_t0).isoformat(timespec="seconds"),
                        **result,
                    }
                    icon = {
                        "success": "✓",
                        "failure": "✗",
                        "timeout": "⌛",
                        "error": "!",
                    }.get(result["status"], "?")
                    print(
                        f"  {icon} {result['status']} "
                        f"(rc={result.get('returncode')}, {result.get('elapsed_sec', '?')}s)"
                    )

                results.append(record)
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
    finally:
        # Always restore the original params file.
        try:
            params_path.write_text(original_text, encoding="utf-8")
            print(f"\n[sweep] params restored from in-memory copy")
        except Exception as e:
            print(f"\n[sweep] WARNING: failed to restore params from memory: {e}")
            print(f"[sweep]          backup is at {params_backup}")

    # 6.6 Final summary.
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    summary = {
        "notebook": str(nb_path.relative_to(ROOT) if nb_path.is_relative_to(ROOT) else nb_path),
        "params_file": str(rel_params),
        "params_backup": str(params_backup.relative_to(ROOT)),
        "executable_py": str(executable_py.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)),
        "combos_attempted": len(results),
        "combos_skipped_non_string": skipped,
        "by_status": statuses,
        "started_at": ts,
        "finished_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[sweep] summary: {summary_path.relative_to(ROOT)}")
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
