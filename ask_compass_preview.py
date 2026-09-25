"""Ask Compass — fast template preview.

Renders report_template_v2_2.html against a slice of REAL NB05 payload data so
layout can be checked in seconds instead of regenerating a full report.

Critical detail, learned the hard way: this project contains TWO incompatible
report payloads.

  * NB05 payload  — lives only inside OUTPUTS/.../classroom_compass_<ts>.html
                    fields: tier, sep1_val, mean_topic_share, tvd,
                    group_display, is_cross, meta.group_tabs
  * cc_ payload   — OUTPUTS/.../classroom_compass_report_data_<ts>.json
                    fields: category_value_add_tier,
                    mean_topic_share_all_verified_topics, no tvd/sep/tier

report_template_v2.x consumes the NB05 shape. Feeding it the cc_ JSON renders
a page that looks broken in six separate ways for one single reason, so this
script takes the payload from NB05's own output and refuses to proceed if the
shape does not match what the template reads.

    python ask_compass_preview.py            # 12 insights, opens in a browser
    python ask_compass_preview.py -n 40
    python ask_compass_preview.py --no-open
    python ask_compass_preview.py --source OUTPUTS/.../classroom_compass_260526_1638.html
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

# Fields the template reads that only the NB05 payload provides. If any are
# absent the data is the wrong shape, and the preview would mislead.
REQUIRED_INSIGHT_FIELDS = [
    "id", "title", "tier", "group_display", "mean_topic_share", "tvd",
    "sep1_val", "sep1_lbl", "is_cross", "charts",
]
REQUIRED_META_FIELDS = ["group_tabs", "looker_id_limit"]


def find_nb05_report(root: Path) -> Path | None:
    """Newest NB05 output.

    NB05 writes classroom_compass_<ts>.html. The cc_ path writes
    classroom_compass_report_<ts>.html into the same folder, so the
    '_report_' files are excluded explicitly rather than by sort order.
    """
    outputs = root / "OUTPUTS"
    if not outputs.exists():
        return None
    # NB05 writes into the primary run's reports folder, which varies per run,
    # so the whole OUTPUTS tree is searched. Newest by modification time.
    def usable(path: Path) -> bool:
        """Skip reports that carry no inlined payload, so a stale or partial
        file cannot win on modification time alone."""
        try:
            return "const REPORT_DATA = " in path.read_text(
                encoding="utf-8", errors="ignore")[:400_000]
        except OSError:
            return False

    files = [p for p in outputs.rglob("classroom_compass_*.html")
             if not p.name.startswith("classroom_compass_report_")]
    files = [p for p in files if usable(p)]
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


def extract_payload(html_path: Path) -> dict:
    """Pull the inlined REPORT_DATA object out of a generated report."""
    text = html_path.read_text(encoding="utf-8")
    marker = "const REPORT_DATA = "
    i = text.find(marker)
    if i < 0:
        raise ValueError(f"no inlined REPORT_DATA in {html_path.name}")
    i += len(marker)
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return json.loads(text[i:j + 1])


def check_shape(payload: dict) -> list[str]:
    """Fail loudly on the wrong payload rather than rendering a broken page."""
    problems = []
    insights = payload.get("insights") or []
    if not insights:
        return ["payload contains no insights"]

    have = set(insights[0].keys())
    missing = [f for f in REQUIRED_INSIGHT_FIELDS if f not in have]
    if missing:
        problems.append("insight fields missing: " + ", ".join(missing))

    meta = payload.get("meta") or {}
    missing_meta = [f for f in REQUIRED_META_FIELDS if f not in meta]
    if missing_meta:
        problems.append("meta fields missing: " + ", ".join(missing_meta))

    # Positive identification of the wrong payload, so the message can say so.
    if "category_value_add_tier" in have or "global_insight_id" in have:
        problems.append(
            "this looks like the cc_build_report_payload JSON, not the NB05 "
            "payload; the template cannot render it")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--template", default="report_template_v2_2.html")
    ap.add_argument("--source", default=None,
                    help="an NB05-generated report HTML to take the payload from")
    ap.add_argument("-n", "--insights", type=int, default=12)
    ap.add_argument("--out", default="/tmp/ask_compass_preview.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    tpl = Path(args.template)
    if not tpl.is_absolute():
        tpl = root / tpl
    if not tpl.exists():
        print(f"Template not found: {tpl}")
        return 2

    src = Path(args.source) if args.source else find_nb05_report(root)
    if src is None or not src.exists():
        print("No NB05 report found anywhere under OUTPUTS.\n"
              "Expected a file named classroom_compass_<ts>.html (without "
              "'_report_').\nRun NB05 once, then rerun this preview, or pass "
              "--source explicitly.")
        return 2

    try:
        payload = extract_payload(src)
    except ValueError as exc:
        print(f"Could not read a payload from {src.name}: {exc}")
        return 2

    problems = check_shape(payload)
    if problems:
        print(f"PAYLOAD SHAPE MISMATCH — {src.name}")
        for p in problems:
            print(f"  - {p}")
        print("\nThe template reads NB05 fields (tier, sep1_val, tvd, "
              "group_display, meta.group_tabs).\nRendering anyway would give "
              "NaN values, a missing Groups row and cards that\nwill not open. "
              "Nothing written.")
        return 1

    # Spread the slice across groups so the preview shows variety rather than
    # a dozen cards from one batch.
    seen: dict[str, int] = {}
    picked = []
    for ins in payload["insights"]:
        key = str(ins.get("group_key") or ins.get("group_display") or "?")
        if seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        picked.append(ins)
        if len(picked) >= args.insights:
            break

    slim = dict(payload)
    slim["insights"] = picked
    slim["meta"] = dict(payload.get("meta") or {})
    # meta.group_tabs is deliberately left intact. Trimming it would hide the
    # Groups row and look exactly like the bug this script exists to prevent.

    chartjs = root / "chart.umd.min.js"
    html = (tpl.read_text(encoding="utf-8")
            .replace("__CHARTJS__",
                     chartjs.read_text(encoding="utf-8") if chartjs.exists()
                     else "/* chart.umd.min.js not found; charts disabled */")
            .replace("__FAVICON__", "data:,")
            .replace("__REPORT_DATA__",
                     json.dumps(slim, ensure_ascii=False, separators=(",", ":"),
                                default=str)))

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"template  {tpl.name}")
    print(f"payload   {src.name}  (NB05 shape verified)")
    print(f"insights  {len(picked)} of {len(payload['insights'])}, "
          f"across {len(seen)} groups")
    print(f"charts    {'included' if chartjs.exists() else 'MISSING chart.umd.min.js'}")
    print(f"written   {out}  ({len(html) / 1024:.0f} KB)")
    print("\nNote: the Ask Compass bar renders but cannot answer here. /ask is "
          "served\nby ask_compass_server.py against the full report.")

    if not args.no_open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
