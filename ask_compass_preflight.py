"""Ask Compass — Stage 1 preflight.

Contract freeze and verification. Run this to completion before any product
code is written. It answers, with evidence rather than inference:

  1. Is the frozen analytical utility the one we validated?      (PF-01, PF-02)
  2. Do the agent snapshot and the live report describe the
     same 1,047 insights?                                        (PF-03..PF-05)
  3. Will a backend insight_id join exactly one report card?     (PF-06, PF-07)
  4. Does a serving-configured chat satisfy every Step 0b
     invariant?                                                  (PF-08)
  5. Is the local Store C essay path ready for Call 3?           (PF-09)

Design rules:

  * No model calls. Preflight is offline except for constructing a chat
    object, which only reads a manifest. The single live smoke turn belongs
    to Stage 2 startup, not here.
  * Nothing is mutated. No file is written except the preflight report.
  * Fail-closed. Any FAIL means Stage 2 does not begin.
  * Every check reports observed values, not just pass/fail, so the report
    is a usable record rather than a green light.

Usage
-----
    python ask_compass_preflight.py
    python ask_compass_preflight.py --root "/path/to/Essay Prototype"
    python ask_compass_preflight.py --expect-utils-sha <sha256>

Exit codes: 0 all passed, 1 one or more failed, 2 preflight could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Expected values. These are the contract. Changing one means the prototype
# being served is not the prototype that was validated.
# --------------------------------------------------------------------------

EXPECTED_CODE_VERSION = "v11.3"
# Established by the 41/41 clean preflight run. These are the contract, and
# they are asserted by default: a bare `python ask_compass_preflight.py` must
# fail closed on drift. Use --rebaseline to deliberately establish new values.
EXPECTED_UTILS_SHA256 = (
    "70d73f919140b70839d3b5cee7440901ae1c284c09d439748590624a116074ec"
)
EXPECTED_SNAPSHOT_ID = "20260526_b5958b0bb9"
EXPECTED_REPORT_SHA256 = (
    "b5958b0bb9a02cfeb1b3e8ca53892d7e057a609719904060eff5596b9ca687a6"
)
EXPECTED_INSIGHT_COUNT = 1047
EXPECTED_SELF_AUDIT_CHECKS = 58
EXPECTED_SELF_AUDIT_SEQUENCES = 2

# Live report template renders cards as:
#   <article class="summary-card" id="card-${esc(ins.id)}" data-id="${esc(ins.id)}" ...>
# Both attributes carry ins.id, which is global_insight_id. data-id is also
# used by the existing filter code, so UX-39 adds data-insight-id as a
# purpose-named alias rather than overloading it.
CARD_ID_PATTERN = re.compile(
    r'id="card-\$\{esc\(ins\.id\)\}"\s+data-id="\$\{esc\(ins\.id\)\}"'
)

# Step 0b serving invariants.
#
# status() and NotebookConfig use different names for the same switch, and not
# every invariant is exposed by status() at all. Each entry lists the candidate
# key names in priority order; the first that resolves is asserted. This keeps
# the check robust to status() gaining or losing keys in a later build.
#
#   (label, candidate keys, required value, why it matters)
SERVING_INVARIANTS: list[tuple[str, tuple[str, ...], Any, str]] = [
    ("project_selection_llm_enabled", ("project_selection_llm_enabled",), True,
     "Call 3 off means every card renders without teacher projects while all "
     "acceptance tests still pass"),
    ("allow_essay_send_project_selection", ("allow_essay_send_project_selection",), True,
     "Call 3 cannot judge quote quality without essay text"),
    ("allow_essay_send_insight_selection", ("allow_essay_send_insight_selection",), False,
     "Essays must never reach insight selection; findings come from Compass only"),
    ("store_a_rewrite_query", ("store_a_rewrite_query",), False,
     "Query rewrite reintroduces the retrieval variance closed by REQ-22"),
    ("llm_enabled", ("llm_enabled", "use_llm"), True,
     "Serving process must call the model"),
    ("web_search_enabled", ("web_search_enabled", "use_web_search"), True,
     "External research is required for named-org queries"),
    ("store_c_vector_challenger_enabled", ("store_c_vector_challenger_enabled",), False,
     "Diagnostic challenger costs a call per insight and renders nothing"),
    ("snapshot_id", ("snapshot_id",), EXPECTED_SNAPSHOT_ID,
     "A different snapshot means different insight IDs and a broken card join"),
]

# AskCompassChat.from_setup reads exactly this path.
SETUP_MANIFEST_RELPATH = Path("OUTPUTS") / "ask_compass_agent" / "setup_manifest.json"


# --------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------

@dataclass
class Check:
    code: str
    name: str
    passed: bool
    detail: str = ""
    observed: Any = None
    expected: Any = None


@dataclass
class Preflight:
    root: Path
    checks: list[Check] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def add(self, code, name, passed, detail="", observed=None, expected=None):
        self.checks.append(Check(code, name, bool(passed), detail, observed, expected))
        return passed

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rebase(raw: str | None, manifest_root: str | None, root: Path) -> Path | None:
    """Resolve a manifest path against the current root.

    The setup manifest records absolute paths from the machine that built the
    snapshot. If the project folder is moved or copied, those paths no longer
    resolve and every path check fails for the wrong reason. Rebase onto the
    root actually in use, preferring the literal path when it exists.
    """
    if not raw:
        return None
    literal = Path(raw)
    if literal.exists():
        return literal
    if manifest_root:
        try:
            return root / literal.relative_to(Path(manifest_root))
        except ValueError:
            pass  # not under the recorded root; fall through
    return literal


# --------------------------------------------------------------------------
# PF-01  Frozen utility identity
# --------------------------------------------------------------------------

def pf01_utils_hash(pf: Preflight, utils_path: Path, expect_sha: str | None) -> None:
    """Assert the SHA256 of the frozen analytical utility.

    'Frozen' is only meaningful if it is enforced. expect_sha is the pinned
    constant unless --rebaseline passed None, so the default command fails
    closed when ask_compass_utils.py changes.
    """
    if not utils_path.exists():
        pf.add("PF-01", "ask_compass_utils.py present", False,
               f"not found at {utils_path}")
        return

    sha = sha256_file(utils_path)
    size = utils_path.stat().st_size
    pf.facts["utils_path"] = str(utils_path)
    pf.facts["utils_sha256"] = sha
    pf.facts["utils_bytes"] = size

    if expect_sha:
        pf.add("PF-01", "Frozen utility hash matches pinned value", sha == expect_sha,
               "ask_compass_utils.py has changed since it was validated; rerun "
               "the full evaluation before re-pinning" if sha != expect_sha else "",
               observed=sha, expected=expect_sha)
    else:
        # --rebaseline only. Records without asserting so a new validated
        # version can be established deliberately.
        pf.add("PF-01", "Frozen utility hash recorded (rebaseline)", True,
               "assertion skipped by --rebaseline; paste this into "
               "EXPECTED_UTILS_SHA256",
               observed=sha)


# --------------------------------------------------------------------------
# PF-02  Self-audit
# --------------------------------------------------------------------------

def pf02_self_audit(pf: Preflight, ac, expect_checks: int | None,
                    expect_sequences: int | None) -> None:
    """Import-time self-audit must pass and report the validated version.

    runtime_self_audit() returns a dict; it does not print a banner. The
    authoritative pass signal is an empty regression_failures list. Counts are
    recorded and asserted only when pinned, so a first run establishes the
    baseline instead of failing on an unknown expectation.
    """
    try:
        result = ac.runtime_self_audit()
    except Exception as exc:
        pf.add("PF-02", "Self-audit runs cleanly", False, f"{type(exc).__name__}: {exc}")
        return
    if not isinstance(result, dict):
        pf.add("PF-02", "Self-audit returns a result dict", False,
               observed=type(result).__name__)
        return

    failures = result.get("regression_failures") or []
    checks = result.get("regression_count")
    sequences = result.get("integration_count")
    version = result.get("code_version")

    # Live binding names, with line numbers, are the record that makes the
    # frozen monkey-patch layering auditable rather than assumed.
    pf.facts["self_audit"] = {
        k: result.get(k) for k in (
            "code_version", "runtime_contract", "run_turn_binding", "run_turn_line",
            "synthesis_binding", "research_binding", "interpret_binding",
            "hybrid_retrieve_binding", "vector_retrieve_binding",
            "status_binding", "stability_probe_binding",
            "regression_count", "integration_count")
    }

    pf.add("PF-02a", "Self-audit reports zero regression failures", not failures,
           observed=failures or "none")
    pf.add("PF-02b", "Code version is the validated build", version == EXPECTED_CODE_VERSION,
           observed=version, expected=EXPECTED_CODE_VERSION)

    if expect_checks is None:
        pf.add("PF-02c", "Regression check count recorded (rebaseline)", True,
               "assertion skipped by --rebaseline; paste into "
               "EXPECTED_SELF_AUDIT_CHECKS",
               observed=checks)
    else:
        pf.add("PF-02c", "Regression check count unchanged", checks == expect_checks,
               "a changed count means regressions were added or lost",
               observed=checks, expected=expect_checks)

    if expect_sequences is None:
        pf.add("PF-02d", "Integration check count recorded (rebaseline)", True,
               "assertion skipped by --rebaseline; paste into "
               "EXPECTED_SELF_AUDIT_SEQUENCES",
               observed=sequences)
    else:
        pf.add("PF-02d", "Integration check count unchanged", sequences == expect_sequences,
               observed=sequences, expected=expect_sequences)


# --------------------------------------------------------------------------
# PF-03..PF-05  Snapshot and report artifacts
# --------------------------------------------------------------------------

def pf03_setup_manifest(pf: Preflight, root: Path) -> dict | None:
    """Locate and validate the agent setup manifest."""
    # from_setup() reads exactly SETUP_MANIFEST_RELPATH; the others are
    # fallbacks for a relocated or hand-copied snapshot.
    candidates = [
        root / SETUP_MANIFEST_RELPATH,
        root / "OUTPUTS" / "ask_compass_prototype_handoff" / "setup_manifest.json",
        root / "setup_manifest.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pf.add("PF-03", "Setup manifest found", False,
               "looked in: " + "; ".join(str(p) for p in candidates))
        return None

    manifest = json.loads(path.read_text(encoding="utf-8"))
    pf.facts["setup_manifest_path"] = str(path)

    pf.add("PF-03a", "Setup manifest found", True, observed=str(path))
    pf.add("PF-03b", "Snapshot is the frozen prototype snapshot",
           manifest.get("snapshot_id") == EXPECTED_SNAPSHOT_ID,
           observed=manifest.get("snapshot_id"), expected=EXPECTED_SNAPSHOT_ID)

    registry = manifest.get("registry") or {}
    pf.add("PF-03c", "Registry record count matches report insight count",
           registry.get("record_count") == registry.get("report_insight_count")
           == EXPECTED_INSIGHT_COUNT,
           observed={"record_count": registry.get("record_count"),
                     "report_insight_count": registry.get("report_insight_count")},
           expected=EXPECTED_INSIGHT_COUNT)

    # Every declared path must exist. A missing store here surfaces as a
    # confusing runtime failure much later. Paths are rebased first so a
    # moved project folder does not fail for the wrong reason.
    manifest_root = manifest.get("root")
    pf.facts["manifest_root"] = manifest_root
    if manifest_root and Path(manifest_root) != root:
        pf.facts["root_rebased_from"] = manifest_root

    paths = manifest.get("paths") or {}
    missing = {}
    for key, raw in paths.items():
        resolved = rebase(raw, manifest_root, root)
        if resolved is None or not resolved.exists():
            missing[key] = str(resolved or raw)
    pf.add("PF-03d", "All declared snapshot paths exist", not missing,
           observed=missing or "all present")

    # Known, accepted condition. Surfaced deliberately so it is a recorded
    # decision rather than a silent assumption: over-index signals are weak
    # because the baseline corpus is newer than the frozen report.
    aligned = registry.get("baseline_aligned_to_report_snapshot")
    pf.facts["baseline_aligned_to_report_snapshot"] = aligned
    pf.facts["baseline_warning"] = registry.get("baseline_warning")
    pf.add("PF-03e", "Baseline alignment status recorded", True,
           "known and accepted: over-index signals are weak application signals"
           if aligned is False else "",
           observed=aligned)

    return manifest


def pf04_report_data(pf: Preflight, root: Path, manifest: dict | None) -> dict | None:
    """Verify the report data the agent reads is the byte-exact frozen copy."""
    path = None
    if manifest:
        candidate = rebase(manifest.get("report_json"), manifest.get("root"), root)
        if candidate and candidate.exists():
            path = candidate
    if path is None:
        found = sorted((root / "OUTPUTS" / "sweep_review" / "reports").glob(
            "classroom_compass_report_data_*.json"))
        path = found[-1] if found else None
    if path is None:
        pf.add("PF-04", "Report data JSON found", False,
               "no report_json in manifest and none under OUTPUTS/sweep_review/reports")
        return None

    sha = sha256_file(path)
    pf.facts["report_data_path"] = str(path)
    pf.facts["report_data_sha256"] = sha

    pf.add("PF-04a", "Report data JSON found", True, observed=str(path))
    pf.add("PF-04b", "Report data hash matches the frozen snapshot",
           sha == EXPECTED_REPORT_SHA256,
           "the agent snapshot and this report are different builds"
           if sha != EXPECTED_REPORT_SHA256 else "",
           observed=sha, expected=EXPECTED_REPORT_SHA256)

    if manifest and manifest.get("report_sha256"):
        pf.add("PF-04c", "Report data hash matches setup manifest",
               sha == manifest["report_sha256"],
               observed=sha, expected=manifest["report_sha256"])

    return json.loads(path.read_text(encoding="utf-8"))


def pf05_insight_ids(pf: Preflight, report: dict | None) -> list[str]:
    """The card join depends on id being global_insight_id, unique, non-blank."""
    if not report:
        pf.add("PF-05", "Report insight IDs validated", False, "no report data loaded")
        return []

    insights = report.get("insights") or []
    ids = [str(i.get("id") or "") for i in insights]
    gids = [str(i.get("global_insight_id") or "") for i in insights]

    pf.facts["report_insight_count"] = len(insights)
    pf.facts["report_meta_run_id"] = (report.get("meta") or {}).get("run_id")

    pf.add("PF-05a", "Report insight count as expected", len(insights) == EXPECTED_INSIGHT_COUNT,
           observed=len(insights), expected=EXPECTED_INSIGHT_COUNT)
    pf.add("PF-05b", "All insight IDs are non-blank", all(ids),
           observed=sum(1 for x in ids if not x))
    pf.add("PF-05c", "All insight IDs are unique", len(set(ids)) == len(ids),
           observed={"unique": len(set(ids)), "total": len(ids)})
    pf.add("PF-05d", "id equals global_insight_id for every insight", ids == gids,
           "the browser joins on id; the backend keys on global_insight_id",
           observed=sum(1 for a, b in zip(ids, gids) if a != b))
    return ids


# --------------------------------------------------------------------------
# PF-06  Live HTML card contract
# --------------------------------------------------------------------------

def pf06_card_contract(pf: Preflight, root: Path, report_ids: list[str]) -> None:
    """Confirm the live report renders a joinable card identifier.

    Reads the generated report rather than the template, because the template
    directory contains several versions and only one of them shipped.
    """
    found = sorted((root / "OUTPUTS" / "sweep_review" / "reports").glob(
        "classroom_compass_report_*.html"))
    if not found:
        pf.add("PF-06", "Live report HTML found", False,
               "no classroom_compass_report_*.html under OUTPUTS/sweep_review/reports")
        return

    path = found[-1]
    html = path.read_text(encoding="utf-8", errors="replace")
    pf.facts["live_report_html"] = str(path)
    pf.facts["live_report_html_mb"] = round(path.stat().st_size / (1 << 20), 1)

    pf.add("PF-06a", "Live report HTML found", True, observed=str(path))
    pf.add("PF-06b", "Card element carries ins.id in id and data-id",
           bool(CARD_ID_PATTERN.search(html)),
           "UX-39 joins on this attribute; if the pattern changed, the join breaks")
    pf.add("PF-06c", "Template placeholder was substituted",
           "__REPORT_DATA__" not in html,
           "an unsubstituted placeholder means this file is a template, not a report")
    pf.add("PF-06d", "data-insight-id not yet present", "data-insight-id" not in html,
           "informational: UX-39 adds this alias so the join does not overload data-id")

    # Spot-check that IDs from the payload actually appear in the rendered file.
    # Cheap substring test only; the payload is inlined so exact IDs must occur.
    if report_ids:
        sample = report_ids[:5] + report_ids[-5:]
        missing = [i for i in sample if i not in html]
        pf.add("PF-06e", "Sampled report IDs appear in the rendered HTML", not missing,
               observed=missing or "10/10 found")


# --------------------------------------------------------------------------
# PF-07  Registry to report join
# --------------------------------------------------------------------------

def pf07_registry_join(pf: Preflight, root: Path, manifest: dict | None,
                       report_ids: list[str]) -> None:
    """Require a one-to-one join between registry records and report cards.

    Reads the registry JSONL declared in the manifest rather than reaching into
    chat internals, so the check does not break when an attribute is renamed.
    A partial join is worse than none: filtering the report to a subset of the
    answer silently misrepresents the evidence.
    """
    if not manifest:
        pf.add("PF-07", "Registry available for join check", False, "no setup manifest")
        return

    reg_path = rebase((manifest.get("paths") or {}).get("registry"),
                      manifest.get("root"), root)
    if reg_path is None or not reg_path.exists():
        pf.add("PF-07", "Registry file found", False, observed=str(reg_path))
        return

    reg_ids: list[str] = []
    try:
        with reg_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                reg_ids.append(str(rec.get("insight_id") or rec.get("id") or ""))
    except Exception as exc:
        pf.add("PF-07", "Registry parses as JSONL", False, f"{type(exc).__name__}: {exc}",
               observed=str(reg_path))
        return

    pf.facts["registry_path"] = str(reg_path)
    pf.facts["registry_id_count"] = len(reg_ids)

    reg, rep = set(reg_ids), set(report_ids)
    pf.add("PF-07a", "Registry IDs are unique and non-blank",
           len(reg) == len(reg_ids) and all(reg_ids),
           observed={"unique": len(reg), "total": len(reg_ids),
                     "blank": sum(1 for x in reg_ids if not x)})
    pf.add("PF-07b", "Registry and report contain the same IDs", reg == rep,
           observed={"registry_only": sorted(reg - rep)[:5],
                     "report_only": sorted(rep - reg)[:5],
                     "registry": len(reg), "report": len(rep)})
    pf.add("PF-07c", "Join is one-to-one at the expected count",
           len(reg) == len(rep) == EXPECTED_INSIGHT_COUNT,
           observed={"registry": len(reg), "report": len(rep)},
           expected=EXPECTED_INSIGHT_COUNT)


# --------------------------------------------------------------------------
# PF-08  Serving configuration invariants
# --------------------------------------------------------------------------

def build_preflight_serving_config(ac, root: Path):
    """Draft of build_serving_config() for Stage 2.

    Deliberately explicit. NotebookConfig defaults are tuned for probe speed
    and have at least two values that must not reach serving:
    project_selector_mode='none' and store_a_rewrite_query=True.
    """
    return ac.NotebookConfig(
        root=root,
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        use_llm=True,
        use_web_search=True,
        use_store_a_vector_search=True,
        use_store_c_vector_challenger=False,   # diagnostic only, never rendered
        store_a_rewrite_query=False,           # REQ-22 variance contract
        project_selector_mode="local_tfidf",   # pre-filter; Call 3 still decides
        allow_essay_send_insight_selection=False,
        allow_essay_send_project_selection=True,
        model_candidate_count=25,
        project_prefilter_count=20,
        project_selection_min=3,
        project_selection_max=15,
        project_fallback_count=10,
        verify_ssl=False,                      # proven local proxy setting
        report_exact_anchor_supported=False,
    )


_MISSING = object()


def _resolve_invariant(keys: tuple[str, ...], status: dict, cfg) -> tuple[str, Any]:
    """First candidate key found in status(), else on the config object."""
    for key in keys:
        if key in status:
            return f"status.{key}", status[key]
    for key in keys:
        val = getattr(cfg, key, _MISSING)
        if val is not _MISSING:
            return f"config.{key}", val
    return "", _MISSING


def pf08_serving_invariants(pf: Preflight, ac, root: Path) -> None:
    """Instantiate a serving-configured chat and assert every Step 0b value.

    No model call. from_setup only reads the manifest, so this is fast.
    """
    try:
        cfg = build_preflight_serving_config(ac, root)
        chat = ac.AskCompassChat.from_setup(root, config=cfg)
        status = chat.status()
    except Exception as exc:
        pf.add("PF-08", "Serving-configured chat constructs", False,
               f"{type(exc).__name__}: {exc}")
        return

    pf.add("PF-08", "Serving-configured chat constructs", True)
    pf.facts["serving_status"] = dict(status)

    for label, keys, want, why in SERVING_INVARIANTS:
        source, value = _resolve_invariant(keys, status, cfg)
        if value is _MISSING:
            pf.add(f"PF-08:{label}", f"{label} is resolvable", False,
                   "invariant cannot be asserted at startup if neither status() "
                   "nor the config exposes it",
                   observed=f"tried: {', '.join(keys)}")
            continue
        pf.add(f"PF-08:{label}", f"{label} == {want!r} (via {source})",
               value == want, why, observed=value, expected=want)

    # The config object is asserted separately from the resolved status because
    # the two can diverge if a default is applied downstream.
    pf.add("PF-08:cfg_rewrite", "serving_config.store_a_rewrite_query is False",
           getattr(cfg, "store_a_rewrite_query", None) is False,
           observed=getattr(cfg, "store_a_rewrite_query", None), expected=False)
    pf.add("PF-08:cfg_selector", "serving_config.project_selector_mode is not 'none'",
           getattr(cfg, "project_selector_mode", None) != "none",
           "the close-out notebook sets 'none' for probe speed; it must not serve",
           observed=getattr(cfg, "project_selector_mode", None))


# --------------------------------------------------------------------------
# PF-09  Store C readiness
# --------------------------------------------------------------------------

def pf09_store_c(pf: Preflight, manifest: dict | None, root: Path) -> None:
    """Call 3 needs the local essay pool present and complete."""
    if not manifest:
        pf.add("PF-09", "Store C readiness", False, "no setup manifest")
        return

    sc = manifest.get("store_c_local") or {}
    pf.facts["store_c"] = {
        k: sc.get(k) for k in
        ("ready", "expected_unique_projects", "returned_unique_projects",
         "missing_unique_projects", "min_essays_per_insight", "max_essays_per_insight")
    }

    pf.add("PF-09a", "Store C reports ready", sc.get("ready") is True,
           observed=sc.get("ready"))
    pf.add("PF-09b", "No missing essay projects", sc.get("missing_unique_projects") == 0,
           observed=sc.get("missing_unique_projects"), expected=0)
    pf.add("PF-09c", "Essay pool depth supports the Call 3 pre-filter",
           (sc.get("min_essays_per_insight") or 0) >= 20,
           "pre-filter selects 20 from the approved pool",
           observed={"min": sc.get("min_essays_per_insight"),
                     "max": sc.get("max_essays_per_insight")})

    for key in ("bridge_path", "essay_lookup_path"):
        resolved = rebase(sc.get(key), manifest.get("root"), root)
        pf.add(f"PF-09:{key}", f"{key} exists",
               resolved is not None and resolved.exists(),
               observed=str(resolved) if resolved else None)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Ask Compass Stage 1 preflight")
    ap.add_argument("--root", default="/Users/matt.fritz/Desktop/Research Insights/Essay Prototype")
    ap.add_argument("--utils", default=None, help="path to ask_compass_utils.py")
    # Frozen values are asserted by default. Overrides exist for deliberately
    # validating a different build, not for routine use.
    ap.add_argument("--expect-utils-sha", default=EXPECTED_UTILS_SHA256,
                    help="override the pinned ask_compass_utils.py SHA256")
    ap.add_argument("--expect-checks", type=int, default=EXPECTED_SELF_AUDIT_CHECKS,
                    help="override the pinned regression_count")
    ap.add_argument("--expect-sequences", type=int, default=EXPECTED_SELF_AUDIT_SEQUENCES,
                    help="override the pinned integration_count")
    ap.add_argument("--rebaseline", action="store_true",
                    help="record the frozen values instead of asserting them. "
                         "Use only when deliberately establishing a new validated "
                         "version, after a full evaluation run.")
    ap.add_argument("--report", default=None, help="where to write preflight_report.json")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"FATAL: root does not exist: {root}")
        return 2

    pf = Preflight(root=root)
    pf.facts["run_at"] = datetime.now(timezone.utc).isoformat()
    pf.facts["root"] = str(root)

    # --rebaseline turns the three pinned assertions into recordings.
    expect_sha = None if args.rebaseline else args.expect_utils_sha
    expect_checks = None if args.rebaseline else args.expect_checks
    expect_sequences = None if args.rebaseline else args.expect_sequences
    pf.facts["rebaseline_mode"] = bool(args.rebaseline)
    if args.rebaseline:
        print("REBASELINE MODE — pinned values are recorded, not asserted.\n")

    utils_path = Path(args.utils) if args.utils else root / "ask_compass_utils.py"
    pf01_utils_hash(pf, utils_path, expect_sha)

    # Import after hashing so the recorded hash is of the file we import.
    sys.path.insert(0, str(utils_path.parent))
    try:
        import ask_compass_utils as ac  # noqa: E402
    except Exception:
        print("FATAL: could not import ask_compass_utils\n")
        traceback.print_exc()
        return 2

    pf02_self_audit(pf, ac, expect_checks, expect_sequences)
    manifest = pf03_setup_manifest(pf, root)
    report = pf04_report_data(pf, root, manifest)
    report_ids = pf05_insight_ids(pf, report)
    pf06_card_contract(pf, root, report_ids)
    pf07_registry_join(pf, root, manifest, report_ids)
    pf08_serving_invariants(pf, ac, root)
    pf09_store_c(pf, manifest, root)

    # ---------------------------------------------------------------- output
    width = 78
    print("=" * width)
    print("ASK COMPASS — STAGE 1 PREFLIGHT")
    print("=" * width)
    for c in pf.checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"[{mark}] {c.code:<18} {c.name}")
        if not c.passed or c.detail:
            if c.expected is not None:
                print(f"         expected: {c.expected}")
            if c.observed is not None:
                print(f"         observed: {c.observed}")
            if c.detail:
                print(f"         note:     {c.detail}")

    print("-" * width)
    failed = pf.failed
    print(f"{len(pf.checks) - len(failed)} passed, {len(failed)} failed")

    out = Path(args.report) if args.report else root / "OUTPUTS" / "ask_compass_prototype_handoff" / "preflight_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"checks": [asdict(c) for c in pf.checks], "facts": pf.facts},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"report written: {out}")

    if failed:
        print("\nPREFLIGHT FAILED — do not begin Stage 2.")
        for c in failed:
            print(f"  {c.code}: {c.name}")
        if any(c.code in ("PF-01", "PF-02c", "PF-02d") for c in failed):
            print("\nA pinned-contract check failed. The frozen utility is not "
                  "the validated build.\nIf the change is intentional, run the "
                  "full evaluation, then rerun with --rebaseline\nand update the "
                  "constants in this file and in ask_compass_server.py.")
        return 1

    print("\nPREFLIGHT PASSED — contract frozen, Stage 2 may begin.")
    if args.rebaseline:
        sa = pf.facts.get("self_audit") or {}
        print("\nRebaselined. Paste these into the constants at the top of "
              "this file and of ask_compass_server.py:")
        print(f'  EXPECTED_UTILS_SHA256 = "{pf.facts.get("utils_sha256")}"')
        print(f"  EXPECTED_SELF_AUDIT_CHECKS = {sa.get('regression_count')}")
        print(f"  EXPECTED_SELF_AUDIT_SEQUENCES = {sa.get('integration_count')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
