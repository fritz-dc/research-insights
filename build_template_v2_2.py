"""Build report_template_v2.2.html = report_template_v2.1.html + Ask Compass.

This is a one-shot generator, not a runtime patcher. It produces a complete
template that NB05 picks up automatically (its _template_version sorter takes
the highest _vX.Y), so the report is *generated* with Ask Compass rather than
modified afterwards.

Every insertion is anchored on an exact, verified-unique string. If v2.1 ever
changes shape, this fails loudly instead of landing in the wrong place.

    python build_template_v2_2.py
    python build_template_v2_2.py --src report_template_v2_1.html --out report_template_v2_2.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARK = "ASK-COMPASS"

# ---------------------------------------------------------------------------
# 1. Stylesheet
#
# Scoped under #ask-compass or the .ac- prefix. The only rule that touches an
# existing element is .summary-card.ac-cited, which adds an outline and does
# not alter layout or box size.
# ---------------------------------------------------------------------------

CSS = """
/* ===== ASK-COMPASS ===== */
#ask-compass{
  --ac-line:var(--line,#e3e5ee); --ac-muted:var(--muted,#5b6072);
  --ac-ink:var(--ink,#14151c); --ac-accent:var(--accent,#3804c1);
  --ac-surface:var(--card,#fff); --ac-raised:var(--bg,#f7f8fc);
  max-width:1220px;margin:18px auto 0;padding:0 18px;color:var(--ac-ink);
}
#ask-compass .ac-panel{border:1px solid var(--ac-line);border-radius:12px;
  background:var(--ac-surface);overflow:hidden}
#ask-compass .ac-bar{display:flex;gap:9px;align-items:center;padding:12px 14px}
#ask-compass .ac-label{font-size:12px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ac-muted);white-space:nowrap}
#ask-compass .ac-bar{align-items:flex-start}
#ask-compass .ac-label{padding-top:10px}
#ask-compass .ac-input{flex:1;min-width:0;min-height:38px;max-height:132px;
  padding:9px 12px;font:inherit;font-size:13px;line-height:1.45;resize:none;
  overflow-y:auto;color:var(--ac-ink);background:var(--ac-raised);
  border:1px solid var(--ac-line);border-radius:8px}
#ask-compass .ac-count{display:none;font-size:11px;color:var(--ac-muted);
  padding:0 14px 8px;text-align:right}
#ask-compass .ac-count.on{display:block}
#ask-compass .ac-count.over{color:#8a5a00;font-weight:600}
#ask-compass .ac-filter{display:none;gap:8px;align-items:center;margin:0 14px 10px;
  padding:6px 10px;font-size:12px;color:var(--ac-muted);
  background:var(--ac-raised);border:1px solid var(--ac-line);border-radius:8px}
#ask-compass .ac-filter.on{display:flex}
#ask-compass .ac-filter-x{margin-left:auto;font:inherit;font-size:12px;
  color:var(--ac-accent);background:none;border:0;padding:0;cursor:pointer}
#ask-compass .ac-filter-x:hover{text-decoration:underline}
#ask-compass .ac-past{border-top:1px solid var(--ac-line);margin-top:14px}
#ask-compass .ac-past>summary{cursor:pointer;list-style:none;padding:10px 0 8px;
  font-size:13px;color:var(--ac-muted);display:flex;gap:8px;align-items:baseline}
#ask-compass .ac-past>summary::-webkit-details-marker{display:none}
#ask-compass .ac-past .ac-chev{font-size:8px;display:inline-block;
  color:var(--ac-muted);transform:rotate(-90deg);transition:transform .15s}
#ask-compass .ac-past[open] .ac-chev{transform:none}
#ask-compass .ac-past-q{color:var(--ac-ink);font-weight:600;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
#ask-compass .ac-past-body{padding:0 0 10px 18px;opacity:.85}
#ask-compass .ac-turn-q{font-size:12px;color:var(--ac-muted);margin:10px 0 2px}
#ask-compass .ac-input:focus{outline:2px solid var(--ac-accent);outline-offset:1px}
#ask-compass .ac-go{padding:9px 16px;font:inherit;font-size:13px;font-weight:600;
  color:#fff;background:var(--ac-accent);border:0;border-radius:8px;cursor:pointer}
#ask-compass .ac-quiet{padding:8px 12px;font:inherit;font-size:13px;
  color:var(--ac-muted);background:transparent;border:1px solid var(--ac-line);
  border-radius:8px;cursor:pointer}
#ask-compass button[disabled]{opacity:.5;cursor:default}
#ask-compass .ac-busy{display:none;flex-direction:column;gap:4px;
  padding:0 14px 12px;font-size:13px;color:var(--ac-muted)}
#ask-compass .ac-busy.on{display:flex}
#ask-compass .ac-step{display:flex;gap:8px;align-items:center;line-height:1.4}
#ask-compass .ac-step-done{opacity:.55}
#ask-compass .ac-step-mark{width:12px;height:12px;flex:0 0 12px;
  display:inline-flex;align-items:center;justify-content:center;font-size:10px;
  color:var(--ac-accent)}
#ask-compass .ac-dot{width:12px;height:12px;border:2px solid var(--ac-line);
  border-top-color:var(--ac-accent);border-radius:50%;animation:ac-spin .9s linear infinite}
@keyframes ac-spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){#ask-compass .ac-dot{animation:none}}
#ask-compass .ac-out{display:none;padding:2px 14px 16px;border-top:1px solid var(--ac-line)}
#ask-compass .ac-out.on{display:block}
#ask-compass .ac-h{font-size:17px;font-weight:700;margin:12px 0 5px}
#ask-compass .ac-sum{font-size:13.5px;line-height:1.55;margin:0 0 10px;color:var(--ac-ink)}
#ask-compass .ac-scope{display:flex;flex-wrap:wrap;gap:6px;align-items:center;
  font-size:12px;color:var(--ac-muted);margin:0 0 10px}
#ask-compass .ac-chip{padding:2px 8px;border:1px solid var(--ac-line);
  border-radius:999px;background:var(--ac-raised)}
#ask-compass .ac-ext{border:1px solid var(--ac-line);border-radius:8px;
  padding:10px 12px;margin:0 0 12px;background:var(--ac-raised);font-size:13px;line-height:1.5}
#ask-compass .ac-ext-t{font-size:11px;font-weight:700;letter-spacing:.05em;
  text-transform:uppercase;color:var(--ac-muted);display:block;margin-bottom:4px}
#ask-compass .ac-ext ul{margin:6px 0 0;padding-left:18px}
#ask-compass .ac-ext li{word-break:break-word}
#ask-compass .ac-ext-note{margin:6px 0 0;font-size:12px;color:var(--ac-muted);font-style:italic}
#ask-compass .ac-sec{margin:0 0 14px}
#ask-compass .ac-sec-h{display:flex;flex-wrap:wrap;gap:7px;align-items:center;
  margin:0 0 6px;font-size:13.5px;font-weight:700}
#ask-compass .ac-note-t{font-size:12.5px;color:var(--ac-muted);line-height:1.5;margin:0 0 8px}
#ask-compass .ac-tag{font-size:10px;font-weight:700;letter-spacing:.04em;
  text-transform:uppercase;padding:2px 7px;border-radius:5px;
  background:var(--ac-raised);color:var(--ac-muted);border:1px solid var(--ac-line)}
#ask-compass .ac-tag.covered,#ask-compass .ac-tag.direct{background:#e6f5ec;color:#1f6b3d;border-color:#bfe3cd}
#ask-compass .ac-tag.partial{background:#fff6e0;color:#8a5a00;border-color:#f0dcae}
#ask-compass .ac-tag.adjacent{background:#e9eefc;color:#2b3f8f;border-color:#cbd6f5}
#ask-compass .ac-tag.stretch{background:#fdf0f6;color:#8a2b62;border-color:#f2cfe2}
#ask-compass .ac-row{display:grid;grid-template-columns:84px 1fr;gap:10px;
  align-items:start;padding:7px 0;border-top:1px solid var(--ac-line)}
#ask-compass .ac-row:first-of-type{border-top:0}
#ask-compass .ac-row .ac-tag{justify-self:start;margin-top:2px}
#ask-compass .ac-row-empty{grid-column:1}
#ask-compass .ac-link{font:inherit;font-size:13.5px;font-weight:600;color:var(--ac-accent);
  background:none;border:0;padding:0;cursor:pointer;text-align:left}
#ask-compass .ac-link:hover{text-decoration:underline}
#ask-compass .ac-why{font-size:13px;line-height:1.5;color:var(--ac-ink);margin:3px 0 0}
#ask-compass .ac-why-k{font-weight:700;color:var(--ac-muted)}
#ask-compass .ac-gap{border-left:3px solid #f0dcae;padding:7px 0 7px 11px;
  margin:12px 0 0;font-size:13px;line-height:1.55}
#ask-compass .ac-gap-t{font-size:11px;font-weight:700;letter-spacing:.05em;
  text-transform:uppercase;color:var(--ac-muted);display:block;margin-bottom:3px}
#ask-compass .ac-err{padding:10px 12px;border-radius:8px;font-size:13px;
  background:#fff6e0;color:#8a5a00;border:1px solid #f0dcae}
/* Cards cited by the current answer. Outline only: no size or layout change. */
.summary-card.ac-cited{box-shadow:0 0 0 2px var(--accent,#3804c1)}
/* Teacher projects, appended to an expanded card by the panel. */
.ac-projects{margin:14px 0 0;padding:12px 14px;border:1px solid var(--line,#e3e5ee);
  border-radius:8px;background:var(--bg,#f7f8fc)}
.ac-projects .ac-p-label{font-size:11px;font-weight:700;letter-spacing:.05em;
  text-transform:uppercase;color:var(--muted,#5b6072);display:block;margin-bottom:6px}
.ac-projects .ac-p-sum{font-size:13px;line-height:1.55;margin:0 0 10px}
.ac-projects .ac-q{border-left:3px solid var(--line,#e3e5ee);padding:1px 0 1px 11px;
  margin:0 0 10px;font-size:13px;line-height:1.6;font-style:italic}
.ac-projects .ac-q:last-child{margin-bottom:0}
.ac-projects .ac-q-id{font-style:normal;color:var(--muted,#5b6072);white-space:nowrap}
/* ===== /ASK-COMPASS ===== */
"""

# ---------------------------------------------------------------------------
# 2. Panel markup, between the controls bar and the card grid
# ---------------------------------------------------------------------------

PANEL = """
<!-- ASK-COMPASS -->
<section id="ask-compass" aria-labelledby="ac-lbl">
  <div class="ac-panel">
    <div class="ac-bar">
      <span class="ac-label" id="ac-lbl">Ask Compass</span>
      <textarea id="ac-input" class="ac-input" rows="1" maxlength="2000"
                placeholder="What should I show the Gates Foundation about math?"
                aria-label="Ask a question about these insights"></textarea>
      <button id="ac-go" class="ac-go" type="button">Ask</button>
      <button id="ac-copy" class="ac-quiet" type="button"
              title="Copy the latest answer as plain text">Copy</button>
      <button id="ac-reset" class="ac-quiet" type="button">Reset</button>
    </div>
    <div id="ac-count" class="ac-count" aria-live="polite"></div>
    <div id="ac-filter" class="ac-filter" role="status">
      <span id="ac-filter-t">Filtered by Ask Compass</span>
      <button id="ac-filter-x" class="ac-filter-x" type="button">Show all insights</button>
    </div>
    <div id="ac-busy" class="ac-busy" role="status" aria-live="polite"></div>
    <div id="ac-out" class="ac-out" aria-live="polite"></div>
  </div>
</section>
<!-- /ASK-COMPASS -->
"""

# ---------------------------------------------------------------------------
# 3. Hooks inside the report IIFE
#
# v2.1 dims rather than hides, so the selection composes with the existing
# filter model: cited insights simply un-dim, everything else dims. Card
# visibility stays owned by applyFilterDisplay().
# ---------------------------------------------------------------------------

HOOK_STATE = ("let chartsRendered = {};",
              "let chartsRendered = {};\n"
              "let acCited = null;   /* ASK-COMPASS: Set of cited insight ids, or null */")

HOOK_DIM = (
    "  const dimmed = new Set();\n  D.insights.forEach(ins => {\n    let passes = true;",
    "  const dimmed = new Set();\n  D.insights.forEach(ins => {\n    let passes = true;\n"
    "    /* ASK-COMPASS: dim anything the current answer did not cite. */\n"
    "    if (acCited && !acCited.has(ins.id)) passes = false;")

# getDimmedIds() short-circuits to an empty Set when no filter is active, so
# the citation set has to count as active or it would never take effect.
HOOK_REORDER = (
    "  const anyActive = Object.values(filters).some(s => s.size > 0) || searchQ.trim();",
    "  const anyActive = Object.values(filters).some(s => s.size > 0) || searchQ.trim()\n"
    "    || !!acCited;   /* ASK-COMPASS: citations reorder cited cards to the top */")

HOOK_ACTIVE = (
    "  const anyF = Object.values(filters).some(s => s.size > 0);\n  if (!anyF && !q) return new Set();",
    "  const anyF = Object.values(filters).some(s => s.size > 0);\n"
    "  if (!anyF && !q && !acCited) return new Set();   /* ASK-COMPASS */")

HOOK_API = ("window.__copyLink = function (id) {", """/* ===== ASK-COMPASS: the only surface the panel may touch ===== */
window.AskCompassReport = {
  // Dim everything the answer did not cite, and outline what it did.
  // An empty list clears both, so a gap-only answer leaves the report intact.
  setCited: function (ids) {
    const list = Array.isArray(ids) ? ids.filter(Boolean) : [];
    document.querySelectorAll('.summary-card.ac-cited')
      .forEach(c => c.classList.remove('ac-cited'));
    // Only cite ids that exist in this report. If none match, leave the grid
    // alone rather than dimming every card, which is what an id-scheme
    // mismatch would otherwise look like.
    const found = [];
    list.forEach(id => {
      const card = document.querySelector('[data-id="' + id + '"]');
      if (card) { card.classList.add('ac-cited'); found.push(id); }
    });
    if (list.length && !found.length) {
      console.warn('Ask Compass: no insight id in this answer matches a ' +
                   'card in this report. Card highlighting is disabled for ' +
                   'this answer.', list.slice(0, 3));
    }
    acCited = found.length ? new Set(found) : null;
    applyFilterDisplay();
    return found.length;
  },
  // Expand a card and scroll to it, reusing the report's own logic.
  reveal: function (id) {
    const el = document.querySelector('[data-id="' + id + '"]');
    if (!el) return false;
    if (el.style.display === 'none') { activeGroup = '__all__'; applyFilterDisplay(); }
    if (expandedId !== id) expandCard(id, el, true);
    else el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return true;
  },
  // Append teacher quotes to an expanded card's detail panel. Called after
  // reveal so the detail DOM exists. Text nodes only, never innerHTML.
  setProjects: function (id, summary, quotes) {
    const detail = document.getElementById('detail-' + id);
    if (!detail) return false;
    // Build the detail DOM if the card has never been expanded, so
    // quotes attach to every cited card rather than only clicked ones.
    ensureDetail(id);
    const inner = detail.querySelector('.detail-inner') || detail;
    const old = inner.querySelector('.ac-projects');
    if (old) old.remove();
    if (!summary && !(quotes && quotes.length)) return false;
    const box = document.createElement('div');
    box.className = 'ac-projects';
    const lbl = document.createElement('span');
    lbl.className = 'ac-p-label';
    lbl.textContent = 'From teacher projects';
    box.appendChild(lbl);
    if (summary) {
      const p = document.createElement('p');
      p.className = 'ac-p-sum';
      p.textContent = summary;
      box.appendChild(p);
    }
    (quotes || []).forEach(q => {
      let text = (q.quote || '').trim();
      if (!text && Array.isArray(q.segments)) text = q.segments.join(' … ');
      if (!text) return;
      const d = document.createElement('div');
      d.className = 'ac-q';
      d.appendChild(document.createTextNode('\u201C' + text + '\u201D'));
      if (q.project_id) {
        d.appendChild(document.createTextNode(' '));
        const s = document.createElement('span');
        s.className = 'ac-q-id';
        s.textContent = '(' + q.project_id + ')';
        d.appendChild(s);
      }
      box.appendChild(d);
    });
    // Place after the prose, before charts, so it reads under
    // "Why it matters" and "Scope & caveats" at full width.
    const prose = inner.querySelector('.detail-prose');
    if (prose && prose.nextSibling) inner.insertBefore(box, prose.nextSibling);
    else inner.appendChild(box);
    if (detail.style.maxHeight && detail.style.maxHeight !== 'none') {
      detail.style.maxHeight = detail.scrollHeight + 'px';
    }
    return true;
  },
  // Clear the citation state and restore the report's own filters.
  clear: function () {
    acCited = null;
    document.querySelectorAll('.summary-card.ac-cited')
      .forEach(c => c.classList.remove('ac-cited'));
    document.querySelectorAll('.ac-projects').forEach(n => n.remove());
    const btn = document.getElementById('reset-btn');
    if (btn) btn.click(); else applyFilterDisplay();
  }
};
/* ===== /ASK-COMPASS ===== */

window.__copyLink = function (id) {""")

HOOKS = [("selection state", *HOOK_STATE),
         ("reorder guard", *HOOK_REORDER),
         ("dim guard", *HOOK_DIM),
         ("active-filter guard", *HOOK_ACTIVE),
         ("report API", *HOOK_API)]

# ---------------------------------------------------------------------------
# 4. Panel module
#
# Deliberately light. The chat answers "which cards should I look at and
# why"; everything else already lives on the card. No quotes in the chat
# (they attach to the card), external context expanded, one line per insight.
# ---------------------------------------------------------------------------

JS = r"""
<!-- ASK-COMPASS -->
<script>
(function () {
  'use strict';

  var SESSION_KEY = 'ask-compass-session';
  var els = {}, busy = false, timer = null, lastCited = [];

  // The backend does not stream and a turn runs 90-120s, so these are
  // time-based expectations, worded as activities rather than promises.
  var STAGES = [[0,'Reading your question…'],[6,'Checking external context…'],
                [22,'Searching approved insights…'],[48,'Selecting evidence…'],
                [78,'Pulling teacher project examples…'],
                [125,'Still working. This can take a few minutes…']];

  function el(t, c, x) {
    var n = document.createElement(t);
    if (c) n.className = c;
    if (x !== undefined && x !== null && x !== '') n.textContent = String(x);
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }

  function sid() {
    var v;
    try { v = sessionStorage.getItem(SESSION_KEY); } catch (e) { v = null; }
    if (!v) {
      v = 'ac-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 9);
      try { sessionStorage.setItem(SESSION_KEY, v); } catch (e) {}
    }
    return v;
  }

  function post(path, body) {
    return fetch(path, { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (d) {
          if (!r.ok) throw new Error(d.message || ('Request failed (' + r.status + ')'));
          return d;
        });
      });
  }

  function api() { return window.AskCompassReport || null; }

  // Progress is cumulative: finished steps stay visible and muted so a user
  // who looked away can still see what ran. Only the current step spins.
  var stageAt = -1;

  function paintStages(upto) {
    clear(els.busy);
    for (var i = 0; i <= upto; i++) {
      var row = el('div', 'ac-step' + (i < upto ? ' ac-step-done' : ''));
      var mark = el('span', 'ac-step-mark');
      if (i < upto) mark.textContent = '✓';          // check
      else mark.appendChild(el('span', 'ac-dot'));
      mark.setAttribute('aria-hidden', 'true');
      row.appendChild(mark);
      row.appendChild(el('span', null, STAGES[i][1]));
      els.busy.appendChild(row);
    }
  }

  function startBusy() {
    var t0 = Date.now();
    stageAt = 0;
    els.busy.classList.add('on');
    paintStages(0);
    timer = setInterval(function () {
      var s = (Date.now() - t0) / 1000, next = stageAt;
      for (var i = STAGES.length - 1; i >= 0; i--) {
        if (s >= STAGES[i][0]) { next = i; break; }
      }
      if (next !== stageAt) { stageAt = next; paintStages(stageAt); }
    }, 1000);
  }

  function stopBusy() {
    if (timer) clearInterval(timer);
    timer = null;
    els.busy.classList.remove('on');
    clear(els.busy);
    stageAt = -1;
  }
  function setBusy(on) {
    busy = on;
    els.go.disabled = on; els.reset.disabled = on;
    if (els.copy) els.copy.disabled = on || !turns.length;
    els.go.textContent = on ? 'Asking…' : 'Ask';
  }

  function tag(text, cls) { return el('span', 'ac-tag ' + (cls || ''), text); }

  // The search tool appends its own attribution parameter. The visible link
  // is cleaned; the href keeps the original so nothing is silently rewritten.
  function cleanUrl(u) {
    return String(u).replace(/([?&])utm_source=openai(&|$)/, function (m, a, b) {
      return b === '&' ? a : '';
    }).replace(/[?&]$/, '');
  }

  function renderExternal(ext) {
    if (!ext || !ext.used) return null;
    var box = el('div', 'ac-ext');
    box.appendChild(el('span', 'ac-ext-t', 'External context'));
    if (ext.summary) box.appendChild(el('div', null, ext.summary));
    var src = ext.sources || [];
    if (src.length) {
      var ul = el('ul');
      src.forEach(function (s) {
        var url = typeof s === 'string' ? s : (s && (s.url || s.href)) || '';
        if (!url) return;
        var a = el('a', null, cleanUrl(url));
        a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer';
        var li = el('li'); li.appendChild(a); ul.appendChild(li);
      });
      if (ul.childNodes.length) box.appendChild(ul);
    }
    box.appendChild(el('p', 'ac-ext-note',
      'Background from outside Compass. Not evidence for the findings below, ' +
      'and it does not establish that they align to it.'));
    return box;
  }

  // Label before the pitch line, so it reads as a suggested angle rather
  // than a summary of the finding. One constant, easy to reword.
  var PITCH_LABEL = 'Pitch: ';

  function renderRow(item) {
    var row = el('div', 'ac-row');

    // Tag occupies the fixed left column; empty when a fit is absent, so
    // every title still starts at the same left edge.
    if (item.fit) row.appendChild(tag(item.fit, item.fit));
    else row.appendChild(el('span', 'ac-row-empty'));

    var left = el('div');
    var btn = el('button', 'ac-link', item.title || item.insight_id);
    btn.type = 'button';
    btn.title = 'Open this insight below';
    btn.addEventListener('click', function () {
      var a = api();
      if (!a) return;
      a.reveal(item.insight_id);
      // Quotes are attached by render(); nothing to do on click.
    });
    left.appendChild(btn);
    if (item.pitch_angle) {
      var why = el('p', 'ac-why');
      why.appendChild(el('span', 'ac-why-k', PITCH_LABEL));
      why.appendChild(document.createTextNode(item.pitch_angle));
      left.appendChild(why);
    }
    row.appendChild(left);
    return row;
  }

  // Conversation history. Newest first: the latest answer renders expanded,
  // earlier ones collapse beneath it under their original question.
  var turns = [];

  function buildAnswer(payload) {
    var out = el('div', 'ac-turn');
    var resp = payload.response || {};
    var sel = payload.selected_insights || [];
    var byId = {};
    sel.forEach(function (s) { byId[s.insight_id] = s; });

    if (resp.title) out.appendChild(el('div', 'ac-h', resp.title));
    if (resp.search_summary) out.appendChild(el('p', 'ac-sum', resp.search_summary));

    var cons = resp.active_constraints;
    var list = Array.isArray(cons) ? cons : (cons && typeof cons === 'object'
      ? Object.keys(cons).map(function (k) {
          var v = cons[k];
          return k + ': ' + (Array.isArray(v) ? v.join(', ') : v);
        }) : []);
    if (list.length) {
      var sc = el('div', 'ac-scope');
      sc.appendChild(el('span', null, 'Active scope:'));
      list.forEach(function (c) { sc.appendChild(el('span', 'ac-chip', c)); });
      out.appendChild(sc);
    }

    var ext = renderExternal(resp.external_context);
    if (ext) out.appendChild(ext);

    (resp.sections || []).forEach(function (s) {
      var sec = el('section', 'ac-sec');
      var h = el('div', 'ac-sec-h');
      h.appendChild(el('span', null, s.heading || 'Insights'));
      var cov = { covered: 'covered', partial: 'partial',
                  no_supported_match: '' }[s.coverage];
      if (s.coverage) {
        h.appendChild(tag(s.coverage === 'no_supported_match'
          ? 'no supported match' : s.coverage, cov));
      }
      sec.appendChild(h);
      if (s.coverage_note) sec.appendChild(el('p', 'ac-note-t', s.coverage_note));
      (s.items || []).forEach(function (i) { sec.appendChild(renderRow(i)); });
      out.appendChild(sec);
    });

    [['Not covered', resp.gap_note], ['Worth relaxing', resp.relaxation_note]]
      .forEach(function (n) {
        if (!n[1]) return;
        var g = el('div', 'ac-gap');
        g.appendChild(el('span', 'ac-gap-t', n[0]));
        g.appendChild(document.createTextNode(String(n[1])));
        out.appendChild(g);
      });

    lastCited = sel.map(function (s) { return s.insight_id; });
    var a = api();
    if (a) {
      var shown = a.setCited(lastCited);
      // Attach quotes to every cited card up front. ensureDetail is
      // synchronous, so there is no click and no timing dependency.
      sel.forEach(function (s) {
        if (s.project_selection_summary || (s.project_quotes || []).length) {
          a.setProjects(s.insight_id, s.project_selection_summary,
                        s.project_quotes);
        }
      });
      // A cited insight with no card in this report build is dimmed away
      // with everything else: present in the answer, invisible below.
      // Usually means the report was rebuilt after the agent snapshot.
      if (lastCited.length && shown < lastCited.length) {
        out.appendChild(el('p', 'ac-note-t',
          (lastCited.length - shown) + ' of ' + lastCited.length +
          ' insights are not in this report version, so they are not ' +
          'highlighted below.'));
      }
      showFilterChip(shown);
    }
    return out;
  }

  // Repaint the whole transcript. Re-appending an existing element moves it,
  // so turn bodies are built once and reused rather than re-rendered.
  function paint() {
    clear(els.out);
    turns.forEach(function (t, i) {
      if (i === 0) {
        els.out.appendChild(el('p', 'ac-turn-q', t.query));
        els.out.appendChild(t.body);
        return;
      }
      var d = el('details', 'ac-past');
      var s = el('summary');
      // Matches the chevron the report uses on its own collapsible sections.
      s.appendChild(el('span', 'ac-chev', '▾'));
      s.appendChild(el('span', 'ac-past-q', t.query));
      d.appendChild(s);
      var wrap = el('div', 'ac-past-body');
      wrap.appendChild(t.body);
      d.appendChild(wrap);
      els.out.appendChild(d);
    });
    els.out.classList.toggle('on', turns.length > 0);
  }

  function showFilterChip(count) {
    if (!els.filter) return;
    var on = count > 0;
    els.filter.classList.toggle('on', on);
    if (on) {
      els.filterT.textContent = 'Filtered by Ask Compass — ' + count +
        (count === 1 ? ' insight' : ' insights');
    }
  }

  function fail(msg) {
    // Prior answers and card state are deliberately left untouched; the
    // error is prepended above the existing transcript.
    paint();
    els.out.insertBefore(el('div', 'ac-err', msg), els.out.firstChild);
    els.out.classList.add('on');
  }

  function ask() {
    if (busy) return;
    var q = (els.input.value || '').trim();
    if (!q) { els.input.focus(); return; }
    setBusy(true); startBusy();
    post('/ask', { session_id: sid(), query: q })
      .then(function (payload) {
        turns.unshift({ query: q, payload: payload, body: buildAnswer(payload) });
        paint();
        els.input.value = '';
        autoGrow();
        updateCount();
      })
      .catch(function (e) {
        fail(e.message || 'The request could not be completed. Previous results are unchanged.');
      })
      .then(function () { stopBusy(); setBusy(false); });
  }

  function reset() {
    if (busy) return;
    setBusy(true);
    post('/reset', { session_id: sid() })
      .catch(function () {})
      .then(function () {
        turns = [];
        clear(els.out); els.out.classList.remove('on');
        els.input.value = ''; lastCited = [];
        autoGrow(); updateCount(); showFilterChip(0);
        var a = api(); if (a) a.clear();
        setBusy(false);
      });
  }

  // ---- composer helpers -------------------------------------------------

  var COUNT_AT = 1600;   // spec: stay hidden until roughly here

  function updateCount() {
    if (!els.count) return;
    var n = (els.input.value || '').length;
    var on = n >= COUNT_AT;
    els.count.classList.toggle('on', on);
    els.count.classList.toggle('over', n >= 2000);
    if (on) els.count.textContent = n + ' / 2000';
  }

  function autoGrow() {
    els.input.style.height = 'auto';
    els.input.style.height = Math.min(els.input.scrollHeight, 132) + 'px';
  }

  // Copy the rendered latest answer as plain text: no JSON, no diagnostics,
  // no hidden metadata. innerText reads what is on screen.
  function copyAnswer() {
    if (!turns.length) return;
    var text = (turns[0].query ? turns[0].query + '\n\n' : '') +
               (turns[0].body.innerText || turns[0].body.textContent || '');
    var done = function () {
      els.copy.textContent = 'Copied';
      setTimeout(function () { els.copy.textContent = 'Copy'; }, 1600);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () {});
      return;
    }
    // Fallback for pages without clipboard permission.
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) {}
    document.body.removeChild(ta);
  }

  function init() {
    els.input = document.getElementById('ac-input');
    els.go = document.getElementById('ac-go');
    els.reset = document.getElementById('ac-reset');
    els.busy = document.getElementById('ac-busy');
    els.out = document.getElementById('ac-out');
    els.copy = document.getElementById('ac-copy');
    els.count = document.getElementById('ac-count');
    els.filter = document.getElementById('ac-filter');
    els.filterT = document.getElementById('ac-filter-t');
    if (!els.input || !els.go || !els.out) return;

    els.go.addEventListener('click', ask);
    els.reset.addEventListener('click', reset);
    if (els.copy) els.copy.addEventListener('click', copyAnswer);

    // Clear the card filter without discarding the answer.
    var fx = document.getElementById('ac-filter-x');
    if (fx) fx.addEventListener('click', function () {
      var a = api(); if (a) a.clear();
      showFilterChip(0);
    });

    els.input.addEventListener('input', function () {
      autoGrow(); updateCount();
    });
    els.input.addEventListener('keydown', function (e) {
      // Enter submits; Shift+Enter inserts a newline.
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
    });
    autoGrow();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
</script>
<!-- /ASK-COMPASS -->
"""


def build(src: str) -> tuple[str, list[str]]:
    out, applied = src, []

    if out.count("\n</style>") != 1:
        raise ValueError("expected exactly one '</style>' close")
    out = out.replace("\n</style>", CSS + "\n</style>", 1)
    applied.append("stylesheet")

    anchor = '<main id="card-grid"></main>'
    if out.count(anchor) != 1:
        raise ValueError(f"anchor not unique: {anchor}")
    out = out.replace(anchor, PANEL + anchor, 1)
    applied.append("panel markup")

    for name, a, b in HOOKS:
        if out.count(a) != 1:
            raise ValueError(f"anchor not unique for {name}: {a[:60]!r}")
        out = out.replace(a, b, 1)
        applied.append(f"hook: {name}")

    if out.count("</body>") != 1:
        raise ValueError("expected exactly one '</body>'")
    out = out.replace("</body>", JS + "</body>", 1)
    applied.append("panel module")
    return out, applied


def verify(src: str, built: str) -> list[str]:
    """Confirm the original survived and the additions landed once each."""
    problems = []

    # Removing every marked block must return the original byte for byte,
    # apart from the four in-place hook rewrites, which keep their anchors.
    stripped = built
    stripped = stripped.replace(CSS, "")
    stripped = stripped.replace(PANEL, "")
    stripped = stripped.replace(JS, "")
    for _, a, b in HOOKS:
        stripped = stripped.replace(b, a, 1)
    if stripped != src:
        problems.append(
            f"reversing the build does not reproduce the source "
            f"({len(stripped)} vs {len(src)} bytes)"
        )

    for needle, label in (
        ("__REPORT_DATA__", "report data placeholder"),
        ("__CHARTJS__", "chart.js placeholder"),
        ("__FAVICON__", "favicon placeholder"),
        ('id="card-grid"', "card grid"),
        ("function detailHTML", "detailHTML"),
        ("function applyFilterDisplay", "applyFilterDisplay"),
        ("function getDimmedIds", "getDimmedIds"),
        ("function expandCard", "expandCard"),
        ("window.__copyLink", "copy link"),
    ):
        if needle not in built:
            problems.append(f"lost from the original: {label}")

    for needle, label in (('id="ask-compass"', "panel"),
                          ("window.AskCompassReport", "report API"),
                          ("acCited", "citation state")):
        if built.count(needle) == 0:
            problems.append(f"not injected: {label}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="report_template_v2_1.html")
    ap.add_argument("--out", default="report_template_v2_2.html")
    args = ap.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        print(f"Not found: {src_path}")
        return 2
    src = src_path.read_text(encoding="utf-8")

    try:
        built, applied = build(src)
    except ValueError as exc:
        print(f"BUILD FAILED — {exc}")
        return 1

    problems = verify(src, built)
    for a in applied:
        print(f"  + {a}")
    print(f"\nsize  {len(src) / 1024:.0f} KB -> {len(built) / 1024:.0f} KB")
    if problems:
        print("\nverification")
        for p in problems:
            print(f"  FAIL  {p}")
        return 1
    print("verification: all checks passed")

    Path(args.out).write_text(built, encoding="utf-8")
    print(f"written  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
