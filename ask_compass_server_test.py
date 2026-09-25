"""Ask Compass — server integration test.

Run against an already-running server, from a second terminal:

    python ask_compass_server_test.py
    python ask_compass_server_test.py --base http://127.0.0.1:8000
    python ask_compass_server_test.py --quick     # skip the live turns

Live turns cost real model calls. --quick runs only the free checks
(health, validation, reset, routing) in a couple of seconds.

Writes /tmp/ask_compass_turn1.json so the payload shape can be inspected.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def call(base: str, path: str, body: dict | None = None, timeout: int = 300):
    """Returns (status_code, parsed_json_or_text)."""
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except urllib.error.URLError as e:
        return 0, str(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--quick", action="store_true",
                    help="skip live model turns")
    args = ap.parse_args()
    base = args.base

    print("=" * 72)
    print(f"ASK COMPASS — SERVER INTEGRATION TEST  ({base})")
    print("=" * 72)

    # ---------------------------------------------------------------- health
    status, health = call(base, "/health")
    if status == 0:
        print(f"\nCannot reach {base} — is the server running in another terminal?")
        print(f"  {health}")
        return 2
    check("GET /health returns 200", status == 200)
    if isinstance(health, dict):
        check("health has no diagnostics", "_diagnostics" not in json.dumps(health))
        check("health reports the pinned snapshot",
              health.get("snapshot_id") == "20260526_b5958b0bb9",
              str(health.get("snapshot_id")))
        check("health reports code version",
              health.get("code_version") == "v11.3", str(health.get("code_version")))
        smoke = health.get("smoke") or {}
        if smoke.get("skipped"):
            check("startup smoke ran", False, "server started with --skip-smoke")
        else:
            check("startup smoke produced quotes",
                  (smoke.get("quote_total") or 0) > 0,
                  f"{smoke.get('selected_count')} insights, "
                  f"{smoke.get('quote_total')} quotes, {smoke.get('elapsed_seconds')}s")

    # ------------------------------------------------------------ report page
    status, body = call(base, "/")
    check("GET / serves the report", status == 200 and isinstance(body, str)
          and "summary-card" in body)

    # ------------------------------------------------------------ validation
    bad = [
        ("blank query", {"session_id": "t", "query": "   "}, "empty_query"),
        ("missing session_id", {"query": "hello"}, "missing_session_id"),
        ("bad session_id chars", {"session_id": "a b/c", "query": "x"},
         "invalid_session_id"),
        ("over-length query", {"session_id": "t", "query": "x" * 2001},
         "query_too_long"),
        ("debug injection", {"session_id": "t", "query": "x", "debug": True},
         "unsupported_fields"),
        ("model injection", {"session_id": "t", "query": "x", "model": "gpt-4"},
         "unsupported_fields"),
    ]
    for name, payload, expect in bad:
        status, body = call(base, "/ask", payload)
        got = body.get("error_type") if isinstance(body, dict) else None
        check(f"rejects {name}", status == 400 and got == expect,
              f"{status} {got}")

    # ----------------------------------------------------------------- reset
    status, body = call(base, "/reset", {"session_id": "never-used-session"})
    check("reset on unknown session is idempotent",
          status == 200 and isinstance(body, dict) and body.get("existed") is False)
    status, body = call(base, "/reset", {})
    check("reset rejects a missing session_id", status == 400)

    # --------------------------------------------------------------- routing
    status, body = call(base, "/nope")
    check("unknown route returns a 404 error shape",
          status == 404 and isinstance(body, dict)
          and body.get("error_type") == "not_found")

    if args.quick:
        print("\n--quick: skipping live turns.")
        return summarize()

    # ------------------------------------------------------------ live turn 1
    print("\nLive turn 1 (about 1-2 minutes)...")
    status, t1 = call(base, "/ask", {
        "session_id": "itest",
        "query": "What should I show the Gates Foundation about math?",
    })
    if not check("turn 1 returns 200", status == 200, str(status)):
        print(json.dumps(t1, indent=2)[:800])
        return summarize()

    Path("/tmp/ask_compass_turn1.json").write_text(
        json.dumps(t1, indent=2, ensure_ascii=False), encoding="utf-8")

    check("payload keys are exactly the browser contract",
          set(t1) == {"response", "selected_insights", "looker_urls"},
          str(sorted(t1)))
    check("no diagnostics in the payload", "_diagnostics" not in json.dumps(t1))

    sel = t1.get("selected_insights") or []
    resp = t1.get("response") or {}
    check("at least one insight selected", len(sel) > 0, f"{len(sel)} selected")
    check("response carries search_summary", bool(resp.get("search_summary")))
    check("every item has a pitch_angle",
          all("pitch_angle" in i
              for s in (resp.get("sections") or []) for i in (s.get("items") or [])))
    check("no display_title in items",
          not any("display_title" in i
                  for s in (resp.get("sections") or []) for i in (s.get("items") or [])))

    quotes = sum(len(s.get("project_quotes") or []) for s in sel)
    check("teacher quotes present", quotes > 0, f"{quotes} quotes")
    check("every insight has both Looker URLs",
          all(set(v) == {"context", "top500"} for v in (t1.get("looker_urls") or {}).values()))

    # --------------------------------------------- turn 2, same session state
    print("\nLive turn 2, same session (about 1-2 minutes)...")
    status, t2 = call(base, "/ask", {
        "session_id": "itest",
        "query": "Make sure these tie to rural areas.",
    })
    check("turn 2 returns 200", status == 200, str(status))
    if status == 200:
        check("turn 2 returns insights",
              len(t2.get("selected_insights") or []) > 0,
              f"{len(t2.get('selected_insights') or [])} selected")

    # --------------------------------------------------------------- shapes
    print("\n" + "-" * 72)
    print("PROJECT_QUOTES SHAPE (first insight with quotes)")
    print("-" * 72)
    for s in sel:
        if s.get("project_quotes"):
            print("selected_insights[] keys:", sorted(s))
            print("summary:", (s.get("project_selection_summary") or "")[:200])
            print("quote[0]:", json.dumps(s["project_quotes"][0], indent=2)[:600])
            break

    print("\n" + "-" * 72)
    print("RESPONSE SHAPE")
    print("-" * 72)
    print("response keys:", sorted(resp))
    secs = resp.get("sections") or []
    if secs:
        print("section keys:", sorted(secs[0]))
        if secs[0].get("items"):
            print("item keys:  ", sorted(secs[0]["items"][0]))

    print("\nFull turn 1 payload saved to /tmp/ask_compass_turn1.json")
    return summarize()


def summarize() -> int:
    failed = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 72)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    for _, name, detail in failed:
        print(f"  FAIL {name}" + (f"  — {detail}" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
