"""Ask Compass — local serving process.

A thin Flask wrapper around the frozen V11.3 analytical utility. It owns
transport, sessions and configuration; it owns no analytical logic.

Design rules
------------
* ask_compass_utils.py is frozen. This file imports it and calls its public
  surface (ask, reset, status) plus the module-level browser payload builder.
  It never calls private analytical helpers or reconstructs a model response.
* Startup is strict, runtime is graceful. A misconfigured process must not
  launch. A failed request must not corrupt session state or the report view.
* The browser never receives diagnostics. _diagnostics is retained
  server-side for the owner query log only.

Run
---
    python ask_compass_server.py
    python ask_compass_server.py --port 8000 --root "/path/to/Essay Prototype"
    python ask_compass_server.py --skip-smoke     # config work only, no model call

Exit codes: 0 clean shutdown, 2 startup gate failed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file

# ---------------------------------------------------------------------------
# Pinned contract. Verified by Stage 1 preflight on 41/41 checks.
# Changing any of these means the served prototype is not the validated one.
# ---------------------------------------------------------------------------

EXPECTED_UTILS_SHA256 = (
    "70d73f919140b70839d3b5cee7440901ae1c284c09d439748590624a116074ec"
)
EXPECTED_CODE_VERSION = "v11.3"
EXPECTED_SNAPSHOT_ID = "20260526_b5958b0bb9"
EXPECTED_REGRESSION_COUNT = 58
EXPECTED_INTEGRATION_COUNT = 2
EXPECTED_INSIGHT_COUNT = 1047

PROVEN_MODEL = "gpt-5.6-terra"
PROVEN_REASONING_EFFORT = "medium"

# Pinned smoke fixture. Chosen because it has the most repeated-run evidence
# of any query in the project: three clean runs, 3-4 supported insights each,
# guard mode assigned every time, no zero/non-zero flip.
SMOKE_QUERY = (
    "I am meeting with the Head of the Intel Foundation next week to explore "
    "growing our partnership. What should I show them, in particular Title 1 "
    "schools?"
)

MAX_QUERY_CHARS = 2000          # mirrors the HTML composer limit
MAX_SESSION_ID_CHARS = 128
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

# Step 0b invariants. Each entry lists candidate key names in priority order:
# status() first, then the config object. status() and NotebookConfig use
# different names for some of the same switches.
SERVING_INVARIANTS: list[tuple[str, tuple[str, ...], Any, str]] = [
    ("project_selection_llm_enabled", ("project_selection_llm_enabled",), True,
     "Call 3 off renders every card without teacher projects while all "
     "acceptance tests still pass"),
    ("allow_essay_send_project_selection", ("allow_essay_send_project_selection",), True,
     "Call 3 cannot judge quote quality without essay text"),
    ("allow_essay_send_insight_selection", ("allow_essay_send_insight_selection",), False,
     "Essays must never reach insight selection"),
    ("store_a_rewrite_query", ("store_a_rewrite_query",), False,
     "Query rewrite reintroduces the retrieval variance closed by REQ-22"),
    ("llm_enabled", ("llm_enabled", "use_llm"), True,
     "Serving process must call the model"),
    ("web_search_enabled", ("web_search_enabled", "use_web_search"), True,
     "External research is required for named-organization queries"),
    ("store_c_vector_challenger_enabled", ("store_c_vector_challenger_enabled",), False,
     "Diagnostic challenger costs a model call per insight and renders nothing"),
    ("snapshot_id", ("snapshot_id",), EXPECTED_SNAPSHOT_ID,
     "A different snapshot means different insight IDs and a broken card join"),
]

log = logging.getLogger("ask_compass")
_MISSING = object()


class StartupError(RuntimeError):
    """A startup gate failed. The process must not serve."""


# ---------------------------------------------------------------------------
# Serving configuration
# ---------------------------------------------------------------------------

def build_serving_config(ac, root: Path):
    """Production configuration, stated explicitly.

    Deliberately not derived from NotebookConfig defaults. Two of those
    defaults must never reach serving:

      project_selector_mode='none'   disables Call 3 silently
      store_a_rewrite_query=True     undoes the REQ-22 variance fix

    project_selector_mode='local_tfidf' is the *pre-filter* only. It narrows
    the approved 50-essay pool to 20; Call 3 still makes the final selection.
    """
    return ac.NotebookConfig(
        root=root,
        model=PROVEN_MODEL,
        reasoning_effort=PROVEN_REASONING_EFFORT,
        use_llm=True,
        use_web_search=True,
        use_store_a_vector_search=True,
        use_store_c_vector_challenger=False,
        store_a_rewrite_query=False,
        project_selector_mode="local_tfidf",
        allow_essay_send_insight_selection=False,
        allow_essay_send_project_selection=True,
        model_candidate_count=25,
        project_prefilter_count=20,
        project_selection_min=3,
        project_selection_max=15,
        project_fallback_count=10,
        verify_ssl=False,               # proven local proxy setting; do not change here
        report_exact_anchor_supported=False,
    )


def _resolve(keys: tuple[str, ...], status: dict, cfg) -> tuple[str, Any]:
    """First candidate key found in status(), else on the config object."""
    for key in keys:
        if key in status:
            return f"status.{key}", status[key]
    for key in keys:
        val = getattr(cfg, key, _MISSING)
        if val is not _MISSING:
            return f"config.{key}", val
    return "", _MISSING


# ---------------------------------------------------------------------------
# Startup gates
# ---------------------------------------------------------------------------

def gate_utils_identity(utils_path: Path, enforce: bool) -> str:
    """The frozen utility must be the file the preflight validated."""
    h = hashlib.sha256()
    with utils_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    if enforce and sha != EXPECTED_UTILS_SHA256:
        raise StartupError(
            "ask_compass_utils.py has changed since preflight.\n"
            f"  expected {EXPECTED_UTILS_SHA256}\n  observed {sha}\n"
            "Rerun ask_compass_preflight.py and re-pin before serving."
        )
    return sha


def gate_self_audit(ac) -> dict:
    """Import-time self-audit must pass with the pinned counts.

    The audit names every live monkey-patch binding with its line number. It
    is what makes freezing a 14k-line layered module safe rather than
    optimistic, so a clean pass is a precondition for serving.
    """
    audit = ac.runtime_self_audit()
    failures = audit.get("regression_failures") or []
    if failures:
        raise StartupError(f"Self-audit regression failures: {failures}")
    if audit.get("code_version") != EXPECTED_CODE_VERSION:
        raise StartupError(
            f"Code version {audit.get('code_version')!r}, expected "
            f"{EXPECTED_CODE_VERSION!r}"
        )
    for label, key, want in (
        ("regression", "regression_count", EXPECTED_REGRESSION_COUNT),
        ("integration", "integration_count", EXPECTED_INTEGRATION_COUNT),
    ):
        got = audit.get(key)
        if got != want:
            raise StartupError(
                f"{label} check count is {got}, expected {want}. Checks were "
                "added or lost; rerun preflight and re-pin."
            )
    return audit


def gate_serving_invariants(chat, cfg) -> dict:
    """Assert every Step 0b invariant before any user session exists."""
    status = chat.status()
    observed, problems = {}, []
    for label, keys, want, why in SERVING_INVARIANTS:
        source, value = _resolve(keys, status, cfg)
        observed[label] = value
        if value is _MISSING:
            problems.append(f"{label}: not exposed by status() or config")
        elif value != want:
            problems.append(f"{label}: {value!r}, expected {want!r} — {why}")

    # The config object is checked separately because it can diverge from the
    # resolved status if a default is applied downstream.
    if getattr(cfg, "store_a_rewrite_query", None) is not False:
        problems.append("serving_config.store_a_rewrite_query is not False")
    if getattr(cfg, "project_selector_mode", None) == "none":
        problems.append(
            "serving_config.project_selector_mode is 'none' — that is the "
            "close-out notebook setting and must not serve"
        )
    if problems:
        raise StartupError("Serving configuration invariants failed:\n  - "
                           + "\n  - ".join(problems))
    return observed


def gate_card_identity(root: Path, serve_override: Path | None = None) -> dict:
    """Confirm a backend insight_id will join exactly one report card.

    Verified against the generated report rather than a template, because the
    project contains several template versions and only one shipped.
    """
    outputs = root / "OUTPUTS"

    # The report data JSON the agent registry is keyed on comes from the cc_
    # path and always lands in sweep_review/reports.
    data_files = sorted((outputs / "sweep_review" / "reports").glob(
        "classroom_compass_report_data_*.json"))
    if not data_files:
        raise StartupError(
            f"No report data JSON under {outputs / 'sweep_review' / 'reports'}")

    # NB05 writes classroom_compass_<ts>.html into the *primary run's* reports
    # folder, which varies per run, so search the whole OUTPUTS tree. The cc_
    # path's classroom_compass_report_<ts>.html is excluded by name.
    html_files = sorted(
        (p for p in outputs.rglob("classroom_compass_*.html")
         if not p.name.startswith("classroom_compass_report_")),
        key=lambda p: p.stat().st_mtime)
    cc_html = sorted((outputs / "sweep_review" / "reports").glob(
        "classroom_compass_report_*.html"))

    if not html_files and not cc_html:
        raise StartupError(f"No report HTML anywhere under {outputs}")
    data_path = data_files[-1]

    # Among NB05 reports, prefer the newest one that actually carries the
    # panel. A stale report built from the old template would otherwise win
    # on mtime and silently serve a page with no Ask Compass.
    def has_panel(path: Path) -> bool:
        try:
            return "ASK-COMPASS" in path.read_text(
                encoding="utf-8", errors="ignore")[:400_000]
        except OSError:
            return False

    if serve_override:
        if not serve_override.exists():
            raise StartupError(f"--report not found: {serve_override}")
        html_path = serve_override
    else:
        with_panel = [p for p in html_files if has_panel(p)]
        html_path = (with_panel or html_files or cc_html)[-1]
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    insights = payload.get("insights") or []
    ids = [str(i.get("id") or "") for i in insights]

    if len(insights) != EXPECTED_INSIGHT_COUNT:
        raise StartupError(
            f"Report has {len(insights)} insights, expected {EXPECTED_INSIGHT_COUNT}"
        )
    if len(set(ids)) != len(ids) or not all(ids):
        raise StartupError("Report insight IDs are not unique and non-blank")
    if any(str(i.get("id")) != str(i.get("global_insight_id")) for i in insights):
        raise StartupError(
            "id != global_insight_id for at least one insight; the browser "
            "joins on id and the backend keys on global_insight_id"
        )
    served_is_nb05 = not html_path.name.startswith("classroom_compass_report_")
    return {
        "report_data_path": str(data_path),
        "report_html_path": str(html_path),
        "report_insight_count": len(insights),
        "served_builder": "nb05" if served_is_nb05 else "cc_",
        "ask_compass_in_report": has_panel(html_path),
        "nb05_reports_found": len(html_files),
    }


def gate_smoke_test(ac, root: Path, cfg) -> dict:
    """One live turn proving Call 3 produces browser-renderable content.

    Empty project_quotes is valid on any individual request, so configuration
    checks alone cannot distinguish "this insight had no good quotes" from
    "Call 3 is switched off". This is the only live-model work at startup.

    Runs on a disposable probe chat that is discarded, so its turn state never
    enters a user session.
    """
    probe = ac.AskCompassChat.from_setup(root, config=cfg)
    started = time.time()
    full = probe.ask(SMOKE_QUERY, debug=True)
    payload = ac._v6_browser_payload(probe, full)
    elapsed = round(time.time() - started, 1)

    selected = payload.get("selected_insights") or []
    with_summary = [s for s in selected if (s.get("project_selection_summary") or "").strip()]
    with_quotes = [s for s in selected if s.get("project_quotes")]
    quote_total = sum(len(s.get("project_quotes") or []) for s in selected)

    diag = (full.get("_diagnostics") or {}).get("project_selection") or {}
    llm_modes = [v.get("mode") for v in diag.values() if isinstance(v, dict)]
    llm_used = any(m == "llm_project_selection" for m in llm_modes)

    result = {
        "elapsed_seconds": elapsed,
        "selected_count": len(selected),
        "insights_with_summary": len(with_summary),
        "insights_with_quotes": len(with_quotes),
        "quote_total": quote_total,
        "project_selection_modes": sorted(set(m for m in llm_modes if m)),
    }

    problems = []
    if not selected:
        problems.append("smoke query selected zero insights")
    if not llm_used:
        problems.append(
            "no insight used llm_project_selection; a TF-IDF fallback alone "
            "does not prove Call 3 is active"
        )
    if not with_quotes:
        problems.append("no verified project quotes in the browser payload")
    if problems:
        raise StartupError(
            "Call 3 smoke test failed:\n  - " + "\n  - ".join(problems)
            + f"\n  observed: {result}\n"
            "Do not retry automatically. Inspect configuration, project data "
            "and API availability, then restart."
        )
    return result


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@contextmanager
def turn_state_guard(chat):
    """Roll back conversation state if a turn fails part-way through.

    The frozen ask() increments turn_number *before* running the turn, then
    commits session_state and history on success. So two things can leave a
    session advanced after a failed request:

      * run_turn() raises, and turn_number has already moved
      * ask() succeeds but payload construction raises, and session_state
        plus history have already been committed

    Either leaves a combination that never occurs in normal use, such as
    turn_number=2 with empty session_state. The next request then classifies
    as a follow-up with nothing to follow up on. The error contract says a
    failed request preserves prior state, so restore all three.

    BaseException, not Exception: an interrupt mid-turn must not leave the
    session inconsistent either.
    """
    snapshot_turn = chat.turn_number
    snapshot_state = copy.deepcopy(chat.session_state)
    snapshot_history_len = len(chat.history)
    try:
        yield
    except BaseException:
        chat.turn_number = snapshot_turn
        chat.session_state = snapshot_state
        del chat.history[snapshot_history_len:]
        raise


@dataclass
class Session:
    """One browser session owns exactly one chat, and therefore one
    session_state, turn_number, history and constraint ledger."""
    session_id: str
    chat: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    turns: int = 0


class SessionRegistry:
    """Process-local session store.

    No eviction in V1. This is a single-owner local prototype; an idle TTL
    could silently discard a constraint ledger mid-conversation, which is
    worse than holding a few objects. Reset and process restart are the
    cleanup mechanisms.
    """

    def __init__(self, ac, root: Path, cfg):
        self._ac, self._root, self._cfg = ac, root, cfg
        self._sessions: dict[str, Session] = {}
        self._guard = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        with self._guard:
            sess = self._sessions.get(session_id)
            if sess is None:
                chat = self._ac.AskCompassChat.from_setup(self._root, config=self._cfg)
                sess = Session(session_id=session_id, chat=chat)
                self._sessions[session_id] = sess
                log.info("session created: %s (total %d)", session_id, len(self._sessions))
            sess.last_seen = time.time()
            return sess

    def get(self, session_id: str) -> Session | None:
        with self._guard:
            return self._sessions.get(session_id)

    def count(self) -> int:
        with self._guard:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# Owner-side query log
# ---------------------------------------------------------------------------

class QueryLog:
    """Append-only JSONL, local to the owner.

    Separate from product telemetry. The unpredictable failure modes surface
    as patterns across many real queries rather than in any single one, and
    the gap notes are the only way to learn what users expect Compass to know.

    Quote text is not logged. Enable only for targeted local debugging.
    """

    def __init__(self, path: Path, log_quotes: bool = False):
        self.path = path
        self.log_quotes = log_quotes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, *, session_id, turn, query, full, payload, elapsed,
              code_version, snapshot_id, error_type=None) -> None:
        d = (full or {}).get("_diagnostics") or {}
        response = (payload or {}).get("response") or {}
        selected = (payload or {}).get("selected_insights") or []
        ps = d.get("project_selection") or {}

        coverage = [
            {
                "heading": s.get("heading"),
                "coverage": s.get("coverage"),
                "gap_type": s.get("gap_type"),
                "item_fits": [i.get("fit") for i in (s.get("items") or [])],
            }
            for s in (response.get("sections") or [])
        ]
        quotes_returned = sum(len(s.get("project_quotes") or []) for s in selected)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "code_version": code_version,
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "turn_number": turn,
            "query": query,
            "elapsed_seconds": elapsed,
            "error_type": error_type,
            "turn_scope": d.get("turn_scope"),
            "active_objectives": d.get("active_objectives"),
            "active_constraints": d.get("active_constraints"),
            "guard_mode": (d.get("under_selection_guard") or {}).get("guard_mode"),
            "expected_floor": (d.get("under_selection_guard") or {}).get("expected_floor"),
            "selected_count": len(selected),
            "selected_insight_ids": [s.get("insight_id") for s in selected],
            "coverage": coverage,
            "gap_note": response.get("gap_note"),
            "relaxation_note": response.get("relaxation_note"),
            "project_selection_modes": sorted(
                {v.get("mode") for v in ps.values() if isinstance(v, dict) and v.get("mode")}
            ),
            "project_selection_fallbacks": sum(
                1 for v in ps.values() if isinstance(v, dict) and v.get("fallback_used")
            ),
            "quotes_returned": quotes_returned,
        }
        if self.log_quotes:
            record["quotes"] = [
                {"insight_id": s.get("insight_id"), "quotes": s.get("project_quotes")}
                for s in selected
            ]

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

class BadRequest(ValueError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


def validate_ask(body: Any) -> tuple[str, str]:
    """Reject malformed input before any chat state is touched."""
    if not isinstance(body, dict):
        raise BadRequest("invalid_body", "Request body must be a JSON object.")

    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        raise BadRequest("missing_session_id", "session_id is required.")
    if not SESSION_ID_RE.match(session_id):
        raise BadRequest(
            "invalid_session_id",
            f"session_id must be 1-{MAX_SESSION_ID_CHARS} characters from "
            "[A-Za-z0-9_.:-]. Use an opaque identifier, never an email.",
        )

    query = str(body.get("query") or "").strip()
    if not query:
        raise BadRequest("empty_query", "query must not be blank.")
    if len(query) > MAX_QUERY_CHARS:
        raise BadRequest(
            "query_too_long",
            f"query is {len(query)} characters; the limit is {MAX_QUERY_CHARS}.",
        )

    # Model, reasoning effort, vector store, prompt and debug controls are
    # never accepted from the browser.
    rejected = sorted(set(body) - {"session_id", "query"})
    if rejected:
        raise BadRequest(
            "unsupported_fields",
            f"Unsupported fields: {', '.join(rejected)}. Only session_id and "
            "query are accepted.",
        )
    return session_id, query


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def create_app(ac, root: Path, cfg, health: dict, query_log: QueryLog) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    registry = SessionRegistry(ac, root, cfg)
    report_html = Path(health["card_identity"]["report_html_path"])

    def fail(status: int, error_type: str, message: str) -> tuple[Response, int]:
        """Uniform error shape. Never returns a partial payload: an incomplete
        selected_insights array would filter the report to the wrong cards."""
        return jsonify({"error_type": error_type, "message": message}), status

    # -- report, served same-origin so no CORS configuration is required -----
    @app.get("/")
    def serve_report():
        return send_file(report_html, mimetype="text/html")

    @app.get("/health")
    def health_route():
        return jsonify({**health, "sessions": registry.count()})

    @app.post("/reset")
    def reset_route():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return fail(400, "invalid_body", "Request body must be a JSON object.")
        session_id = str(body.get("session_id") or "").strip()
        if not SESSION_ID_RE.match(session_id or ""):
            return fail(400, "invalid_session_id", "A valid session_id is required.")

        sess = registry.get(session_id)
        if sess is None:
            # Idempotent: an unknown session is already clean. No model call.
            return jsonify({"ok": True, "existed": False})
        if not sess.lock.acquire(blocking=False):
            return fail(409, "session_busy", "A turn is in progress for this session.")
        try:
            sess.chat.reset()
            sess.turns = 0
        finally:
            sess.lock.release()
        return jsonify({"ok": True, "existed": True})

    @app.post("/ask")
    def ask_route():
        try:
            session_id, query = validate_ask(request.get_json(silent=True))
        except BadRequest as exc:
            return fail(400, exc.error_type, str(exc))

        sess = registry.get_or_create(session_id)

        # Serialize per session. Two concurrent turns would corrupt the
        # constraint ledger, which is the state most expensive to get right.
        if not sess.lock.acquire(blocking=False):
            return fail(409, "session_busy",
                        "A turn is already in progress for this session.")
        started = time.time()
        try:
            # The guard spans payload construction too: ask() has already
            # committed state by the time it returns.
            with turn_state_guard(sess.chat):
                full = sess.chat.ask(query, debug=True)
                payload = ac._v6_browser_payload(sess.chat, full)
            sess.turns += 1
            elapsed = round(time.time() - started, 2)
        except Exception as exc:
            elapsed = round(time.time() - started, 2)
            log.exception("turn failed for session %s", session_id)
            try:
                query_log.write(
                    session_id=session_id, turn=sess.turns + 1, query=query,
                    full=None, payload=None, elapsed=elapsed,
                    code_version=health["code_version"],
                    snapshot_id=health["snapshot_id"],
                    error_type=type(exc).__name__,
                )
            except Exception:
                log.exception("query log write failed")
            # Generic message to the browser; the trace stays in server logs.
            return fail(500, "turn_failed",
                        "The request could not be completed. Previous results "
                        "are unchanged.")
        finally:
            sess.lock.release()

        try:
            query_log.write(
                session_id=session_id, turn=sess.turns, query=query,
                full=full, payload=payload, elapsed=elapsed,
                code_version=health["code_version"],
                snapshot_id=health["snapshot_id"],
            )
        except Exception:
            # Logging must never fail a served turn.
            log.exception("query log write failed")

        log.info("turn ok session=%s turn=%d selected=%d %.1fs",
                 session_id, sess.turns,
                 len(payload.get("selected_insights") or []), elapsed)
        return jsonify(payload)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error_type": "not_found", "message": "Unknown route."}), 404

    @app.errorhandler(405)
    def bad_method(_):
        return jsonify({"error_type": "method_not_allowed",
                        "message": "Wrong HTTP method for this route."}), 405

    return app


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def run_startup(root: Path, *, skip_smoke: bool, enforce_hash: bool,
                log_quotes: bool, report: Path | None = None) -> tuple[Flask, dict]:
    utils_path = root / "ask_compass_utils.py"
    if not utils_path.exists():
        raise StartupError(f"ask_compass_utils.py not found at {utils_path}")

    print("gate 1/6  frozen utility identity")
    sha = gate_utils_identity(utils_path, enforce_hash)

    print("gate 2/6  import and self-audit")
    sys.path.insert(0, str(root))
    import ask_compass_utils as ac  # noqa: E402
    audit = gate_self_audit(ac)
    print(f"          {audit['code_version']} · {audit['regression_count']} regressions"
          f" · {audit['integration_count']} integration sequences")
    print(f"          run_turn -> {audit['run_turn_binding']} (line {audit['run_turn_line']})")

    print("gate 3/6  serving configuration")
    cfg = build_serving_config(ac, root)

    print("gate 4/6  probe chat and Step 0b invariants")
    probe = ac.AskCompassChat.from_setup(root, config=cfg)
    observed = gate_serving_invariants(probe, cfg)
    del probe

    print("gate 5/6  card identity contract")
    card = gate_card_identity(root, report)
    print(f"          {card['report_insight_count']} insights, "
          f"id == global_insight_id")
    print(f"          serving {Path(card['report_html_path']).name} "
          f"({card['served_builder']} builder)")
    if not card["ask_compass_in_report"]:
        print("          WARNING: the served report has no Ask Compass panel.")
        print(f"                   {card['nb05_reports_found']} NB05 report(s) "
              "found under OUTPUTS.")
        print("                   Rerun NB05 with report_template_v2.2.html, "
              "or pass --report.")

    if skip_smoke:
        print("gate 6/6  Call 3 smoke test SKIPPED (--skip-smoke)")
        smoke = {"skipped": True}
    else:
        print("gate 6/6  Call 3 smoke test (one live turn)")
        smoke = gate_smoke_test(ac, root, cfg)
        print(f"          {smoke['selected_count']} insights, "
              f"{smoke['quote_total']} verified quotes, {smoke['elapsed_seconds']}s")

    health = {
        "status": "ok",
        "code_version": audit.get("code_version"),
        "runtime_contract": audit.get("runtime_contract"),
        "snapshot_id": EXPECTED_SNAPSHOT_ID,
        "utils_sha256": sha,
        "regression_count": audit.get("regression_count"),
        "integration_count": audit.get("integration_count"),
        "serving_invariants": observed,
        "card_identity": card,
        "smoke": smoke,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    query_log = QueryLog(
        root / "OUTPUTS" / "ask_compass_agent" / "ask_compass_query_log.jsonl",
        log_quotes=log_quotes,
    )
    app = create_app(ac, root, cfg, health, query_log)
    return app, health



# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

def run_selftest() -> int:
    """Deterministic checks with a stub chat. No model calls, no snapshot.

    Covers the two behaviours that are cheap to break and expensive to notice:
    turn-state rollback on failure, and session isolation.
    """
    results: list[tuple[bool, str]] = []

    def check(ok: bool, name: str, detail: str = "") -> None:
        results.append((bool(ok), name))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

    class StubChat:
        """Mirrors the frozen ask(): turn_number advances before the work,
        session_state and history commit after it."""

        def __init__(self, name="s"):
            self.name = name
            self.turn_number = 0
            self.session_state: dict[str, Any] = {}
            self.history: list[dict] = []

        def reset(self):
            self.turn_number = 0
            self.session_state = {}
            self.history = []

        def ask(self, query, *, debug=False):
            self.turn_number += 1
            if "RAISE_IN_TURN" in query:
                raise RuntimeError("simulated run_turn failure")
            full = {"response": {}, "selected_insights": [], "_diagnostics": {}}
            self.session_state = {"turn": self.turn_number, "objectives": ["o1"]}
            self.history.append({"turn": self.turn_number, "query": query})
            return full

    # -- turn-state rollback ------------------------------------------------
    chat = StubChat()
    chat.ask("first turn")
    baseline = (chat.turn_number, copy.deepcopy(chat.session_state), len(chat.history))

    try:
        with turn_state_guard(chat):
            chat.ask("RAISE_IN_TURN")
    except RuntimeError:
        pass
    check(chat.turn_number == baseline[0], "rollback: turn_number restored",
          f"{chat.turn_number} == {baseline[0]}")
    check(chat.session_state == baseline[1], "rollback: session_state restored")
    check(len(chat.history) == baseline[2], "rollback: history not extended",
          f"{len(chat.history)} == {baseline[2]}")

    # failure *after* ask() succeeds, i.e. payload construction raises
    try:
        with turn_state_guard(chat):
            chat.ask("second turn")
            raise ValueError("simulated payload failure")
    except ValueError:
        pass
    check(chat.turn_number == baseline[0],
          "rollback: turn_number restored when payload build fails")
    check(chat.session_state == baseline[1],
          "rollback: committed session_state rolled back")
    check(len(chat.history) == baseline[2],
          "rollback: committed history entry removed")

    # session_state must be a deep copy, not a shared reference
    chat2 = StubChat()
    chat2.ask("turn")
    try:
        with turn_state_guard(chat2):
            chat2.session_state["objectives"].append("mutated")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check(chat2.session_state.get("objectives") == ["o1"],
          "rollback: nested mutation reverted (deep copy)",
          str(chat2.session_state.get("objectives")))

    # a successful turn must still commit
    chat3 = StubChat()
    with turn_state_guard(chat3):
        chat3.ask("ok")
    check(chat3.turn_number == 1 and len(chat3.history) == 1,
          "success path still commits state")

    # -- session isolation --------------------------------------------------
    class StubAC:
        class AskCompassChat:
            @staticmethod
            def from_setup(root, *, config=None):
                return StubChat()

    reg = SessionRegistry(StubAC(), Path("."), None)
    a1 = reg.get_or_create("A")
    b1 = reg.get_or_create("B")
    check(a1.chat is not b1.chat, "isolation: sessions get distinct chat objects")

    a1.chat.ask("A turn one")
    a1.chat.ask("A turn two")
    check(b1.chat.turn_number == 0 and b1.chat.session_state == {},
          "isolation: work in A does not touch B",
          f"B turn_number={b1.chat.turn_number}")

    b1.chat.ask("B turn one")
    check(a1.chat.turn_number == 2 and b1.chat.turn_number == 1,
          "isolation: independent turn counters",
          f"A={a1.chat.turn_number} B={b1.chat.turn_number}")

    a1.chat.reset()
    check(b1.chat.turn_number == 1, "isolation: reset in A does not reset B")
    check(a1.chat.turn_number == 0, "isolation: reset in A does clear A")

    a2 = reg.get_or_create("A")
    check(a2 is a1 and a2.chat is a1.chat,
          "isolation: re-requesting A returns the same session")
    check(reg.count() == 2, "isolation: registry holds exactly two sessions",
          str(reg.count()))

    # -- per-session lock ---------------------------------------------------
    check(a1.lock.acquire(blocking=False), "lock: acquires when free")
    check(not a1.lock.acquire(blocking=False), "lock: blocks a second acquire")
    a1.lock.release()
    check(b1.lock.acquire(blocking=False), "lock: sessions lock independently")
    b1.lock.release()

    failed = [n for ok, n in results if not ok]
    print("-" * 74)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    for n in failed:
        print(f"  FAIL {n}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask Compass local server")
    ap.add_argument("--root", default="/Users/matt.fritz/Desktop/Research Insights/Essay Prototype")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; keep local unless a pilot needs LAN")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--skip-smoke", action="store_true",
                    help="skip the live Call 3 turn; configuration work only")
    ap.add_argument("--no-hash-check", action="store_true",
                    help="allow a changed ask_compass_utils.py (development only)")
    ap.add_argument("--log-quotes", action="store_true",
                    help="include quote text in the owner query log")
    ap.add_argument("--report", default=None,
                    help="explicit report HTML to serve; defaults to the "
                         "newest NB05 output")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline state and session checks, then exit")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.selftest:
        print("=" * 74)
        print("ASK COMPASS — SERVER SELF-TEST (offline, no model calls)")
        print("=" * 74)
        return run_selftest()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"FATAL: root does not exist: {root}")
        return 2

    print("=" * 74)
    print("ASK COMPASS — SERVER STARTUP")
    print("=" * 74)
    try:
        app, health = run_startup(
            root,
            skip_smoke=args.skip_smoke,
            enforce_hash=not args.no_hash_check,
            log_quotes=args.log_quotes,
            report=Path(args.report).expanduser() if args.report else None,
        )
    except StartupError as exc:
        print("\n" + "=" * 74)
        print("STARTUP GATE FAILED — this is a configuration problem.")
        print("=" * 74)
        print(exc)
        return 2
    except Exception:
        print("\n" + "=" * 74)
        print("STARTUP FAILED — this is an environment problem "
              "(missing file, unavailable API, bad path).")
        print("=" * 74)
        traceback.print_exc()
        return 2

    print("-" * 74)
    print(f"all gates passed · serving on http://{args.host}:{args.port}")
    print(f"  report   http://{args.host}:{args.port}/")
    print(f"  health   http://{args.host}:{args.port}/health")
    print("-" * 74)

    # debug=False and use_reloader=False: the reloader would run every startup
    # gate twice, including a second live smoke turn.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
