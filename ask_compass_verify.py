"""Ask Compass — full verification sweep.

One command that runs every offline check across the whole prototype, so the
state of the build can be established without reading four separate outputs.

    python ask_compass_verify.py
    python ask_compass_verify.py --skip-preflight     # faster, config only
    python ask_compass_verify.py --server-base http://127.0.0.1:8000

What it covers:

  1. File inventory             every expected artifact present
  2. Pinned contract            utils hash and self-audit counts
  3. Stage 1 preflight          delegates to ask_compass_preflight.py
  4. Server self-test           delegates to ask_compass_server.py --selftest
  5. Template parity            regenerating v2.2 from v2.1 reproduces it
  6. Template integrity         every injected piece present, JS parses
  7. NB05 patch                 the three Ask Compass cells are in the notebook
  8. Served report              an NB05 report exists and carries the panel
  9. Constant parity            preflight and server agree on pinned values
 10. Live server (optional)     delegates to ask_compass_server_test.py --quick

No model calls. Steps 3, 4 and 10 shell out to the scripts that own those
checks rather than duplicating them, so there is one definition of each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_UTILS_SHA256 = (
    "70d73f919140b70839d3b5cee7440901ae1c284c09d439748590624a116074ec"
)
EXPECTED_CODE_VERSION = "v11.3"
EXPECTED_SNAPSHOT_ID = "20260526_b5958b0bb9"

EXPECTED_FILES = [
    ("ask_compass_utils.py", "frozen analytical utility"),
    ("ask_compass_preflight.py", "Stage 1 preflight"),
    ("ask_compass_server.py", "serving process"),
    ("ask_compass_server_test.py", "server integration test"),
    ("ask_compass_preview.py", "template preview"),
    ("build_template_v2_2.py", "template generator"),
    ("report_template_v2.2.html", "Ask Compass template"),
    ("report_template_v2.1.html", "base template"),
]

# Injected pieces that must survive in the built template.
TEMPLATE_MARKERS = [
    ('id="ask-compass"', "panel markup"),
    ("window.AskCompassReport", "report API"),
    ("acCited", "citation state"),
    ("|| !!acCited", "reorder guard"),
    ("ensureDetail(id);", "quote attachment without a click"),
    ("var PITCH_LABEL", "pitch label"),
    ("Background from outside Compass", "external caveat"),
    ("not in this report version", "partial-citation note"),
    ('id="ac-copy"', "copy control (UX-06/UX-15/AT-19)"),
    ("function copyAnswer", "copy handler"),
    ('id="ac-count"', "character counter (UX-05)"),
    ('id="ac-filter"', "filtered-by indicator (UX-17)"),
    ("Filtered by Ask Compass", "filter chip label"),
    ("<textarea id=\"ac-input\"", "multiline composer (UX-04/AT-02)"),
    ("!e.shiftKey", "Shift+Enter newline (AT-02)"),
    ("function paint()", "conversation transcript (UX-03)"),
    ("ac-past", "collapsed prior turns"),
]

# Pieces of the original template that must not be lost.
TEMPLATE_PRESERVED = [
    ("__REPORT_DATA__", "report data placeholder"),
    ("__CHARTJS__", "chart.js placeholder"),
    ("function detailHTML", "detailHTML"),
    ("function applyFilterDisplay", "applyFilterDisplay"),
    ("function expandCard", "expandCard"),
    ('id="card-grid"', "card grid"),
]

NB05_MARKERS = [
    ("global_insight_id", "global id join"),
    ("[ask-compass]", "ask-compass cells"),
    ("nb05_insight_id", "payload id swap"),
]

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n{title}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path, timeout: int = 600) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except FileNotFoundError as exc:
        return 127, str(exc)


def find_template(root: Path, stem: str) -> Path | None:
    """Tolerate either report_template_v2.2.html or ..._v2_2.html."""
    for name in (f"{stem}.html", f"{stem.replace('.', '_')}.html"):
        p = root / name
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/matt.fritz/Desktop/Research Insights/Essay Prototype")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--server-base", default=None,
                    help="if the server is running, also run its quick suite")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"FATAL: root does not exist: {root}")
        return 2

    print("=" * 74)
    print("ASK COMPASS — VERIFICATION SWEEP")
    print("=" * 74)
    print(f"root: {root}")

    # ------------------------------------------------------ 1. inventory
    section("1. File inventory")
    for name, label in EXPECTED_FILES:
        if name.startswith("report_template"):
            stem = name[:-5]
            p = find_template(root, stem)
            check(p is not None, f"{label} present",
                  p.name if p else f"neither {stem}.html nor "
                                   f"{stem.replace('.', '_')}.html")
        else:
            check((root / name).exists(), f"{label} present", name)

    # ------------------------------------------------- 2. pinned contract
    section("2. Pinned contract")
    utils = root / "ask_compass_utils.py"
    if utils.exists():
        got = sha256(utils)
        check(got == EXPECTED_UTILS_SHA256, "ask_compass_utils.py unchanged",
              "hash drift — rerun preflight --rebaseline if intentional"
              if got != EXPECTED_UTILS_SHA256 else got[:16] + "…")
    else:
        check(False, "ask_compass_utils.py unchanged", "file missing")

    # ------------------------------------------------------ 3. preflight
    if args.skip_preflight:
        section("3. Stage 1 preflight  (skipped)")
    else:
        section("3. Stage 1 preflight")
        code, out = run([sys.executable, "ask_compass_preflight.py"], root)
        m = re.search(r"(\d+) passed, (\d+) failed", out)
        check(code == 0, "preflight passes",
              m.group(0) if m else out.strip().splitlines()[-1][:70] if out else "")
        if code != 0:
            for line in out.splitlines():
                if line.strip().startswith("[FAIL]") or line.strip().startswith("PF-"):
                    print(f"         {line.strip()[:90]}")

    # ------------------------------------------------ 4. server self-test
    section("4. Server self-test")
    code, out = run([sys.executable, "ask_compass_server.py", "--selftest"], root)
    m = re.search(r"(\d+) passed, (\d+) failed", out)
    check(code == 0, "server self-test passes",
          m.group(0) if m else "see output")

    # -------------------------------------------------- 5. template parity
    section("5. Template parity")
    base = find_template(root, "report_template_v2.1")
    built = find_template(root, "report_template_v2.2")
    gen = root / "build_template_v2_2.py"
    if not (base and built and gen.exists()):
        check(False, "generator reproduces the template",
              "need v2.1, v2.2 and build_template_v2_2.py")
    else:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "regen.html"
            code, out = run([sys.executable, str(gen), "--src", str(base),
                             "--out", str(out_path)], root)
            if code != 0:
                check(False, "generator runs", out.strip()[-120:])
            else:
                same = (out_path.read_text(encoding="utf-8")
                        == built.read_text(encoding="utf-8"))
                check(same, "generator reproduces the template byte for byte",
                      "" if same else "v2.2 has hand edits the generator lacks")

    # ----------------------------------------------- 6. template integrity
    section("6. Template integrity")
    if not built:
        check(False, "template readable", "v2.2 not found")
    else:
        html = built.read_text(encoding="utf-8")
        missing = [lbl for needle, lbl in TEMPLATE_MARKERS if needle not in html]
        check(not missing, "all Ask Compass pieces present",
              "missing: " + ", ".join(missing) if missing else "")
        lost = [lbl for needle, lbl in TEMPLATE_PRESERVED if needle not in html]
        check(not lost, "original template intact",
              "lost: " + ", ".join(lost) if lost else "")

        # JS parse check, only if node is available.
        if shutil.which("node"):
            probe = (html.replace("__CHARTJS__", "//c")
                         .replace("__REPORT_DATA__", "{}")
                         .replace("__FAVICON__", "x"))
            blocks = [s for s in re.findall(r"<script[^>]*>(.*?)</script>",
                                            probe, re.S) if s.strip()]
            bad = []
            with tempfile.TemporaryDirectory() as td:
                for i, s in enumerate(blocks):
                    f = Path(td) / f"s{i}.js"
                    f.write_text(s, encoding="utf-8")
                    if run(["node", "--check", str(f)], root, 60)[0] != 0:
                        bad.append(i)
            check(not bad, f"all {len(blocks)} script blocks parse",
                  f"block(s) {bad} failed" if bad else "")
        else:
            print("  [SKIP] script syntax check — node not installed")

    # -------------------------------------------------------- 7. NB05 patch
    section("7. NB05 patch")
    nbs = sorted(root.glob("05_report_builder*.ipynb"))
    if not nbs:
        check(False, "NB05 notebook found", "no 05_report_builder*.ipynb")
    else:
        nb = nbs[-1]
        text = nb.read_text(encoding="utf-8")
        missing = [lbl for needle, lbl in NB05_MARKERS if needle not in text]
        check(not missing, f"{nb.name} carries the Ask Compass cells",
              "missing: " + ", ".join(missing) if missing else "")

    # ---------------------------------------------------- 8. served report
    section("8. Served report")
    outputs = root / "OUTPUTS"
    reports = [p for p in outputs.rglob("classroom_compass_*.html")
               if not p.name.startswith("classroom_compass_report_")] \
        if outputs.exists() else []
    if not reports:
        check(False, "an NB05 report exists", "none found under OUTPUTS")
    else:
        newest = sorted(reports, key=lambda p: p.stat().st_mtime)[-1]
        text = newest.read_text(encoding="utf-8", errors="ignore")
        check("ASK-COMPASS" in text, "newest NB05 report carries the panel",
              newest.name)

        # Cards are rendered client-side, so data-id never appears as a
        # literal value. What must be present is the card template string.
        # v2.1 renders the grid card with data-id="${ins.id}"; other
        # data-id forms exist elsewhere, so match the card one specifically.
        check('data-id="${ins.id}"' in text, "card template emits data-id")

        # The meaningful check is whether card ids join to the agent registry.
        # If they do not, the panel answers and highlights nothing.
        payload = None
        marker = "const REPORT_DATA = "
        i = text.find(marker)
        if i >= 0:
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
            try:
                payload = json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                payload = None

        if payload is None:
            check(False, "report payload readable", "no parsable REPORT_DATA")
        else:
            card_ids = {str(x.get("id", "")) for x in payload.get("insights", [])}
            check(bool(card_ids), "report carries insights", f"{len(card_ids)} cards")

            setup = root / "OUTPUTS" / "ask_compass_agent" / "setup_manifest.json"
            reg_path = None
            if setup.exists():
                man = json.loads(setup.read_text(encoding="utf-8"))
                raw = (man.get("paths") or {}).get("registry")
                if raw:
                    cand = Path(raw)
                    if not cand.exists() and man.get("root"):
                        try:
                            cand = root / cand.relative_to(Path(man["root"]))
                        except ValueError:
                            pass
                    reg_path = cand if cand.exists() else None

            if reg_path is None:
                check(False, "card ids join to the agent registry",
                      "registry not found; cannot verify highlighting")
            else:
                reg_ids = set()
                with reg_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            r = json.loads(line)
                            reg_ids.add(str(r.get("insight_id") or r.get("id") or ""))
                hit = card_ids & reg_ids
                # Some drift is expected: runs added after the snapshot was
                # frozen have no global id. Below 90% means the join is broken,
                # not drifting.
                ratio = len(hit) / max(len(reg_ids), 1)
                check(ratio >= 0.90, "card ids join to the agent registry",
                      f"{len(hit):,} of {len(reg_ids):,} registry insights "
                      f"({ratio:.1%})")

    # ------------------------------------------------- 9. constant parity
    section("9. Constant parity")
    pf = (root / "ask_compass_preflight.py")
    sv = (root / "ask_compass_server.py")
    if pf.exists() and sv.exists():
        a, b = pf.read_text(), sv.read_text()

        def const(text: str, name: str) -> str | None:
            m = re.search(name + r'\s*=\s*\(?\s*\n?\s*"([^"]+)"', text)
            return m.group(1) if m else None

        for name in ("EXPECTED_UTILS_SHA256", "EXPECTED_CODE_VERSION",
                     "EXPECTED_SNAPSHOT_ID"):
            va, vb = const(a, name), const(b, name)
            check(va is not None and va == vb, f"{name} agrees", str(va)[:24])
    else:
        check(False, "constant parity", "preflight or server missing")

    # ------------------------------------------------- 10. live server
    if args.server_base:
        section("10. Live server")
        code, out = run([sys.executable, "ask_compass_server_test.py",
                         "--base", args.server_base, "--quick"], root)
        m = re.search(r"(\d+) passed, (\d+) failed", out)
        check(code == 0, "server quick suite passes",
              m.group(0) if m else out.strip().splitlines()[-1][:70])
    else:
        section("10. Live server  (skipped — pass --server-base to include)")

    # ---------------------------------------------------------- summary
    failed = [r for r in results if not r[0]]
    print("\n" + "=" * 74)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        for _, name, detail in failed:
            print(f"  FAIL  {name}" + (f"  — {detail}" if detail else ""))
        print("\nVERIFICATION FAILED")
        return 1
    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
