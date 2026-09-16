/* Foreman console.
 *
 * No build step and no framework on purpose: the console is served straight off
 * disk by host/app.py, so there is nothing to compile, nothing to install and
 * one less thing that can fail in front of a judge. State arrives as server-sent
 * events and is rendered by plain functions.
 */
'use strict';

const $ = id => document.getElementById(id);
const SIGNAL = {pass: 'var(--pass)', fail: 'var(--fail)', unclear: 'var(--wait)'};

const LANG_LABEL = {en: 'English', ru: 'Русский', auto: 'Auto EN/RU'};
const DETECTED_LABEL = {en: 'English', ru: 'Russian'};
const LANG_KEY = 'foreman.speechLang';

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmtMs(ms){
  if (ms == null || !isFinite(ms)) return '—';
  return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';
}
function notify(text, kind){
  setText($('notice'), text || '');
  setAttr($('notice'), 'class', 'notice' + (kind ? ' ' + kind : ''));
}

/* State arrives about fifteen times a second. Writing a value that has not
   changed restarts CSS transitions and keeps the page permanently animating, so
   every write goes through a guard. */
function setText(el, value){ if (el.textContent !== value) el.textContent = value; }
function setHtml(el, value){ if (el.innerHTML !== value) el.innerHTML = value; }
function setAttr(el, name, value){ if (el.getAttribute(name) !== value) el.setAttribute(name, value); }
function setStyle(el, prop, value){ if (el.style.getPropertyValue(prop) !== value) el.style.setProperty(prop, value); }

/* ------------------------------------------------------------- speech language
 *
 * The selection is a decoding parameter sent with each recording. Changing it
 * loads nothing and restarts nothing: Whisper stays resident on the MLA, the
 * detector keeps running and the standard in force is untouched. It applies to
 * the next recording, and is locked while one is in progress so an in-flight
 * capture cannot change language halfway through.
 */
let speechLang = 'auto';
let recording = false;

function readStoredLang(){
  try {
    const v = localStorage.getItem(LANG_KEY);
    return (v === 'en' || v === 'ru' || v === 'auto') ? v : null;
  } catch (_) { return null; }   // private windows throw on access
}
function storeLang(v){
  try { localStorage.setItem(LANG_KEY, v); } catch (_) {}
}
function applyLang(v, persist){
  speechLang = v;
  $('speechlang').value = v;
  if (persist) storeLang(v);
}

function initLanguage(){
  const stored = readStoredLang();
  if (stored){ applyLang(stored, false); return; }
  applyLang('auto', false);
  const dlg = $('langdlg');
  // Only offer the dialog once; a browser without <dialog> just keeps the default.
  if (dlg && typeof dlg.showModal === 'function') dlg.showModal();
}

$('langdlg').addEventListener('click', e => {
  const btn = e.target.closest('button[data-lang]');
  if (!btn) return;
  applyLang(btn.dataset.lang, true);
  $('langdlg').close();
});
// Escape-closing the dialog still needs a stored choice, or it reopens next load.
$('langdlg').addEventListener('close', () => storeLang(speechLang));

$('speechlang').addEventListener('change', e => {
  if (recording){ e.target.value = speechLang; return; }
  applyLang(e.target.value, true);
  notify('Speech language set to ' + LANG_LABEL[speechLang] +
         '. It applies to your next recording.', 'good');
});

/* ------------------------------------------------------------------- routing
 *
 * Camera Check is a route, not a hidden panel. That is what makes the browser
 * Back button, a bookmark and a shared link all behave, and it is why the view
 * can no longer trap you: leaving is a normal navigation.
 */
const ROUTES = {'#/inspection': 'inspection', '#/camera': 'camera', '#/history': 'history'};
let view = null;
let liveTimer = null;

function currentRoute(){ return ROUTES[location.hash] || 'inspection'; }

async function applyRoute(){
  const next = currentRoute();
  if (next === view) return;
  view = next;
  const camera = view === 'camera';

  $('view-inspection').hidden = view !== 'inspection';
  $('view-camera').hidden     = view !== 'camera';
  $('view-history').hidden    = view !== 'history';
  $('standardbar').hidden = false;          // the standard stays visible in all
  for (const [id, name] of [['nav-inspection','inspection'], ['nav-camera','camera'],
                            ['nav-history','history']]) {
    $(id).setAttribute('aria-current', view === name ? 'page' : 'false');
  }
  document.title = {camera: 'Camera Check · Foreman',
                    history: 'History · Foreman'}[view] || 'Foreman';

  if (liveTimer){ clearInterval(liveTimer); liveTimer = null; }
  if (camera){
    livePoll();
    liveTimer = setInterval(livePoll, 200);
  }
  // Fetched once on entry, not polled: the audit trail only changes when an
  // inspection completes, and the live state stream already announces that.
  if (view === 'history') loadHistory();
  // Pause inspections in Camera Check only. The detector is never restarted and
  // the standard is never cleared - this is a debugging view, not a state change.
  try {
    await fetch('/api/mode', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({camera})
    });
  } catch (_) {}
}

window.addEventListener('hashchange', applyRoute);

document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  if ($('lightbox').hasAttribute('open')){ closeLightbox(); return; }
  if ($('langdlg').open) return;            // the dialog closes itself
  if (view === 'camera' || view === 'history') location.hash = '#/inspection';
});

/* ------------------------------------------------------------ manual capture
 *
 * "Inspect now" starts the evidence window at the button press, so what gets
 * judged is what the operator just did. The countdown is driven locally because
 * the edge blocks for the whole capture; the phase itself comes from the server,
 * so the bar can never claim progress the backend is not making.
 */
let captureTimer = null;

const CAP_ORDER = ['preparing', 'capturing', 'selecting', 'analyzing'];

/* Which step to show, derived from how long the manual inspection has been
   running. The host cannot observe the boundary between capture, selection and
   the model call - the edge does all three in one blocking request - so this is
   an honest approximation: capture is exact (the host set the clock), and
   selection is the brief moment after it, before the model dominates. */
/* `st` is the whole manual lifecycle: when preparation began, how long it runs,
   when capture began (0 until it does) and how long it runs. Preparation is a
   real phase with its own clock - the host reports both - so the console never
   has to guess, and never implies recording has started before it has. */
function capturePhaseAt(st){
  const nowS = Date.now() / 1000;
  if (!st.captureStart) return 'preparing';
  const elapsed = Math.max(0, nowS - st.captureStart);
  if (elapsed < st.captureS) return 'capturing';
  if (elapsed < st.captureS + 0.5) return 'selecting';
  return 'analyzing';
}

function remainingOf(startedAt, total){
  return Math.max(0, total - ((Date.now() / 1000) - startedAt));
}

function paintCapture(st){
  const phase = capturePhaseAt(st);
  const idx = CAP_ORDER.indexOf(phase);
  for (const li of $('capsteps').children){
    const at = CAP_ORDER.indexOf(li.dataset.phase);
    li.className = at < idx ? 'done' : (at === idx ? 'now' : '');
    if (li.dataset.phase === 'preparing'){
      const left = st.captureStart ? 0 : remainingOf(st.prepareStart, st.prepareS);
      setText(li, left > 0 ? 'Preparing ' + left.toFixed(1) : 'Prepared');
    }
    if (li.dataset.phase === 'capturing'){
      // Blank until capture actually begins, so a countdown never appears next
      // to "Preparing" and suggests frames are already being recorded.
      if (!st.captureStart){ setText(li, 'Capturing'); continue; }
      const left = remainingOf(st.captureStart, st.captureS);
      setText(li, left > 0 ? 'Capturing ' + left.toFixed(1) : 'Captured');
    }
  }
  // The bar tracks only the phase in progress, so preparation cannot look like
  // capture that is already part-way done.
  const pct = st.captureStart
    ? Math.min(100, (((Date.now() / 1000) - st.captureStart) / (st.captureS || 4)) * 100)
    : Math.min(100, (((Date.now() / 1000) - st.prepareStart) / (st.prepareS || 0.8)) * 100);
  setStyle($('capfill'), 'width', Math.max(0, pct).toFixed(1) + '%');
}

let captureState = null;

function setCapturePhase(active, st){
  const wrap = $('capture');
  wrap.hidden = !active;
  if (!active){
    if (captureTimer){ clearInterval(captureTimer); captureTimer = null; }
    captureState = null;
    setStyle($('capfill'), 'width', '0%');
    // Reset the steps too, so the next capture starts from a clean strip
    // instead of briefly showing the previous run's final state.
    for (const li of $('capsteps').children){
      setAttr(li, 'class', '');
      if (li.dataset.phase === 'preparing') setText(li, 'Preparing');
      if (li.dataset.phase === 'capturing') setText(li, 'Capturing');
    }
    return;
  }
  // The timer reads the latest state each tick, so the switch from preparing to
  // capturing lands as soon as the host reports it.
  captureState = st;
  paintCapture(captureState);
  if (!captureTimer) captureTimer = setInterval(() => {
    if (captureState) paintCapture(captureState);
  }, 100);
}

async function setMode(mode){
  try {
    const r = await fetch('/api/inspection_mode', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode})
    });
    if (!r.ok) notify('Could not change inspection mode.', 'bad');
  } catch (_) { notify('Could not reach the Foreman host.', 'bad'); }
}
$('mode-manual').onclick = () => setMode('manual');
$('mode-auto').onclick = () => setMode('auto');

/* After a manual inspection the operator may be scrolled down among the evidence
   or the history. Bring the verdict back into view once, when their own action
   produces a result - never in auto mode, which would hijack the page. */
let lastScrolledId = null;
let awaitingManualVerdict = false;
function revealVerdict(){
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  $('slab').scrollIntoView({behavior: reduce ? 'auto' : 'smooth', block: 'start'});
}

$('inspect-now').onclick = async () => {
  const btn = $('inspect-now');
  btn.disabled = true;
  try {
    const r = await fetch('/api/inspect_now', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'
    });
    if (!r.ok){
      const body = await r.json().catch(() => null);
      notify((body && body.detail) || 'Could not start an inspection.', 'bad');
    } else {
      notify('');
      awaitingManualVerdict = true;
    }
  } catch (_) {
    notify('Could not reach the Foreman host.', 'bad');
  }
  btn.disabled = false;
};

/* -------------------------------------------------------------------- render */

function chipsFor(parsed){
  if (!parsed) return '';
  const out = [];
  for (const c of (parsed.required || [])) out.push('<span class="chip req">' + escapeHtml(c) + ' required</span>');
  for (const c of (parsed.prohibited || [])) out.push('<span class="chip pro">' + escapeHtml(c) + ' prohibited</span>');
  // A relationship rule turns on a value, not on presence. "must be holding" and
  // "must not be holding" name the same objects, so without this the two rules
  // look identical in the console - which is exactly how the negation bug hid.
  if (parsed.relation && parsed.relation_object){
    const want = parsed.relation_expected ? 'required' : 'not allowed';
    out.push('<span class="chip ' + (parsed.relation_expected ? 'req' : 'pro') + '">' +
      escapeHtml(parsed.relation_subject + ' ' + parsed.relation + ' ' + parsed.relation_object) +
      ' ' + want + '</span>');
  }
  if (!parsed.grounded && parsed.raw){
    // Say so plainly: without a detector-supported object the verdict rests on
    // the language model alone, and the operator deserves to know that.
    out.push('<span class="chip warn">no detector grounding</span>');
  }
  // Name what the detector cannot check, rather than letting a partly-grounded
  // rule look fully checked. "pen" has no COCO class; claiming otherwise would
  // be exactly the kind of unearned confidence this system exists to avoid.
  for (const u of (parsed.unsupported || [])){
    out.push('<span class="chip warn" title="No detector class for this. Judged by the ' +
             'vision-language model only.">' + escapeHtml(u) + ' \u2014 not detectable</span>');
  }
  if (parsed.language) out.push('<span class="chip">' + (DETECTED_LABEL[parsed.language] || parsed.language) + '</span>');
  return out.join('');
}

function render(s){
  // The backend is named, never assumed. A local or synthetic result must not
  // be able to read as "Modalix connected" anywhere in the console.
  const BACKEND_LABEL = {modalix: 'Modalix connected', local: 'Local inference',
                         fake: 'Test harness'};
  setAttr($('dot'), 'class', 'dot ' + (s.connected ? (s.backend === 'modalix' ? 'up' : 'alt') : 'down'));
  setText($('health'), s.connected
    ? (BACKEND_LABEL[s.backend] || ('Connected (' + (s.backend || 'unknown') + ')'))
    : 'Edge unreachable');
  if (!s.connected && s.last_error) notify(s.last_error, 'bad');

  const rules = s.rules && s.rules.length ? s.rules : (s.standard ? [s.standard] : []);
  const std = $('standard');
  // With two rules the bar names both, numbered, so "Inspecting against" is
  // never a half-truth.
  setText(std, rules.length === 0 ? 'No standard set'
             : rules.length === 1 ? rules[0]
             : rules.map((t, i) => 'Rule ' + (i + 1) + ': ' + t).join('   '));
  std.classList.toggle('none', rules.length === 0);
  $('standardbar').classList.toggle('set', rules.length > 0);
  const parsedList = (s.parsed_rules && s.parsed_rules.length)
    ? s.parsed_rules : (s.parsed_standard ? [s.parsed_standard] : []);
  setHtml($('chips'), rules.length ? parsedList.map(chipsFor).join('') : '');

  // Re-render must never blank what the operator is looking at. The inputs are
  // only touched when the applied set actually changed - typing into a box
  // while a poll lands must not wipe the draft.
  const appliedNow = rules.join('\u0000');
  if (appliedNow !== appliedRules.join('\u0000')){
    if (rules.length === 0){
      appliedRules = [];
      if (document.activeElement !== $('typed')) $('typed').value = '';
      if (!$('rule2row').hidden) showRule2(false);
      markDraftState();
    } else {
      applyRulesToInputs(rules);
    }
  }

  setText($('n-pass'),  String(s.counters.passed));
  setText($('n-fail'),  String(s.counters.failed));
  setText($('n-total'), String(s.counters.total));

  const latest = s.recent[0] || null;
  const verdict = $('verdict'), reason = $('reason'), against = $('against');
  let signal = 'var(--faint)';
  let againstHtml = '';

  if (!s.connected){
    setText(verdict, 'Line down'); setAttr(verdict, 'class', 'verdict small');
    setText(reason, 'The Modalix edge agent is not answering. Check the Ethernet link, then restart it with scripts/run-edge.sh.');
    setAttr(reason, 'class', 'reason quiet');
    signal = 'var(--fail)';
  } else if (!s.standard){
    setText(verdict, 'Set the standard'); setAttr(verdict, 'class', 'verdict small');
    setText(reason, 'Press "State the standard" and say what a good item looks like. Whisper transcribes it on the DevKit.');
    setAttr(reason, 'class', 'reason quiet');
  } else if (s.paused && view !== 'camera'){
    setText(verdict, 'Inspections paused'); setAttr(verdict, 'class', 'verdict small');
    setText(reason, 'Camera Check is holding inspections. Return to this view to resume.');
    setAttr(reason, 'class', 'reason quiet');
    signal = 'var(--wait)';
  } else if (s.inspecting){
    const preparing = s.capture_phase === 'preparing';
    const manual = s.capture_phase === 'manual';
    const st = {prepareStart: s.prepare_started_at || 0, prepareS: s.prepare_s || 0.8,
                captureStart: manual ? (s.capture_started_at || 0) : 0,
                captureS: s.capture_s || 4};
    const heading = preparing ? 'Preparing'
      : manual
      ? {capturing: 'Capturing evidence', selecting: 'Selecting evidence',
         analyzing: 'Analyzing'}[capturePhaseAt(st)]
      : 'Judging';
    setText(verdict, heading); setAttr(verdict, 'class', 'verdict small');
    setText(reason, preparing
      ? 'Get into position. Nothing is being recorded yet - capture starts in '
        + (s.prepare_s || 0.8) + ' seconds.'
      : manual
      ? 'Recording ' + (s.capture_s || 4) + ' seconds from the moment preparation ended, '
        + 'then judging only those frames.'
      : 'The vision-language model is reading this item against your standard.');
    setAttr(reason, 'class', 'reason quiet');
    signal = 'var(--wait)';
  } else if (latest){
    setText(verdict, latest.verdict); setAttr(verdict, 'class', 'verdict');
    setText(reason, latest.reason || 'No reason returned.');
    setAttr(reason, 'class', 'reason');
    signal = SIGNAL[latest.verdict] || 'var(--faint)';
    againstHtml = 'Judged against <b>' + escapeHtml(latest.standard) + '</b>';
  } else if (s.warming_up){
    setText(verdict, 'Filling the window'); setAttr(verdict, 'class', 'verdict small');
    setText(reason, 'Collecting ' + (s.window_s || 3) + ' seconds of detector evidence before judging anything. '
      + (s.window_frames || 0) + ' frames so far.');
    setAttr(reason, 'class', 'reason quiet');
    signal = 'var(--wait)';
  } else if (s.gate_state === 'settling'){
    setText(verdict, 'Hold still'); setAttr(verdict, 'class', 'verdict small');
    setText(reason, 'Waiting for the item to settle before spending an inference.');
    setAttr(reason, 'class', 'reason quiet');
    signal = 'var(--wait)';
  } else {
    setText(verdict, 'Ready'); setAttr(verdict, 'class', 'verdict small');
    setText(reason, 'Place an item in view of the camera.');
    setAttr(reason, 'class', 'reason quiet');
  }
  setHtml(against, againstHtml);
  // Preparing and capturing are both manual phases; the strip shows either.
  const manualPhase = s.capture_phase === 'preparing' || s.capture_phase === 'manual';
  setCapturePhase(s.inspecting && manualPhase, {
    prepareStart: s.prepare_started_at || 0,
    prepareS: s.prepare_s || 0.8,
    captureStart: s.capture_phase === 'manual' ? (s.capture_started_at || 0) : 0,
    captureS: s.capture_s || 4,
  });

  const manualMode = (s.inspection_mode || 'manual') === 'manual';
  setAttr($('mode-manual'), 'aria-pressed', String(manualMode));
  setAttr($('mode-auto'), 'aria-pressed', String(!manualMode));
  $('inspect-now').hidden = !manualMode;
  // Exactly the three conditions that should ever block it.
  $('inspect-now').disabled = !s.connected || !s.standard || s.inspecting;

  // One scroll per inspection, and only for one the operator started.
  if (awaitingManualVerdict && !s.inspecting && latest && latest.id !== lastScrolledId){
    lastScrolledId = latest.id;
    awaitingManualVerdict = false;
    revealVerdict();
  }
  setStyle($('slab'), '--signal', signal);
  setStyle($('arming'), 'width', (s.gate_state === 'settling' ? s.gate_progress * 100 : 0) + '%');

  renderRuleResults(latest);
  renderMetrics(latest);
  renderFrames(latest);

  setText($('m-see'), s.connected && s.fps ? s.fps.toFixed(0) + ' fps' : '—');
  setText($('m-understand'), latest ? fmtMs(latest.metrics.inference_ms) : '—');
  setText($('m-act'), latest ? fmtMs(latest.metrics.end_to_end_ms) : '—');

  if (view === 'history' && s.counters.total !== lastVerdictCount){
    lastVerdictCount = s.counters.total;
    loadHistory();
  }
  setHtml($('history'), s.recent.slice(0, 40)
    .map(r => '<div class="tile ' + r.verdict + '" title="' +
              escapeHtml(r.verdict + ': ' + r.reason) + '"></div>').join(''));
}

/* The per-rule breakdown under the headline verdict. Shown only when there is
   more than one rule: a single-rule inspection already says everything in the
   headline, and repeating it would be noise. */
function renderRuleResults(latest){
  const box = $('rulesresult');
  const rules = (latest && latest.rules) || [];
  if (rules.length < 2){ box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = rules.map(r => {
    const v = String(r.verdict || 'unclear').toLowerCase();
    return '<li>' +
      '<span class="rn">Rule ' + escapeHtml(String(r.index || '')) + '</span>' +
      '<span class="rv ' + v + '">' + escapeHtml(v.toUpperCase()) + '</span>' +
      '<span class="rr">' + escapeHtml(r.reason || '') + '</span>' +
      '<span class="rt">' + escapeHtml(r.text || '') + '</span>' +
    '</li>';
  }).join('');
}

/* Measured evidence only. There is deliberately no "model confidence" here:
   the vision-language model does not report a calibrated one, so inventing a
   percentage would be the same class of error as the hallucination this whole
   grounding layer exists to catch. */
function renderMetrics(latest){
  const metrics = $('metrics'), basis = $('basis');
  metrics.innerHTML = ''; basis.textContent = '';
  if (!latest) return;

  const cards = [];
  for (const o of (latest.required_objects || [])){
    const pct = Math.round(o.presence_ratio * 100);
    cards.push({
      v: pct + '%', k: o.label + ' detector coverage',
      sub: o.frames_present + ' / ' + o.frames_total + ' frames',
      cls: o.frames_present === 0 ? 'absent' : (pct >= 60 ? 'full' : '')
    });
    // Coverage across time is what lets a flickering detection still ground a
    // verdict, so when it is doing that work it has to be visible.
    if (o.bucket_count){
      cards.push({
        v: o.buckets_supported + ' / ' + o.bucket_count,
        k: 'time segments covered',
        sub: o.spans_window ? 'seen throughout the window' : 'part of the window only',
        cls: o.spans_window ? 'full' : (o.buckets_supported === 0 ? 'absent' : '')
      });
    }
    if (o.frames_present > 0){
      cards.push({v: Math.round(o.conf_median * 100) + '%',
                  k: 'detector confidence', sub: 'median over the window'});
    }
  }
  for (const o of (latest.prohibited_objects || [])){
    const pct = Math.round(o.presence_ratio * 100);
    cards.push({v: pct + '%', k: o.label + ' present (prohibited)',
                sub: o.frames_present + ' / ' + o.frames_total + ' frames',
                cls: o.frames_present === 0 ? 'full' : 'absent'});
  }
  const v = latest.vlm || {};
  // Only call it agreement when the model actually judged each frame. A count of
  // images sent is not agreement, and labelling it so invited the reading that
  // "6 / 6" contradicted "56% present" - two different measurements of two
  // different things.
  const judged = (v.per_frame || []).filter(x => x !== null && x !== undefined).length;
  if (judged > 1){
    cards.push({v: Math.max(v.frames_supporting || 0, v.frames_contradicting || 0) + ' / ' + judged,
                k: 'model agreement', sub: 'frames the model judged alike'});
  }
  if (latest.window && latest.window.total_frames){
    cards.push({v: latest.window.total_frames, k: 'detector frames',
                sub: (latest.window.duration_s || 0).toFixed(1) + ' s window'});
  }
  cards.push({v: (latest.frames || []).length, k: 'evidence images',
              sub: 'shown below, spread across the window'});
  metrics.innerHTML = cards.map(c =>
    '<div class="metric ' + (c.cls || '') + '">' +
    '<div class="v">' + escapeHtml(c.v) + '</div>' +
    '<div class="k">' + escapeHtml(c.k) + '</div>' +
    (c.sub ? '<div class="sub">' + escapeHtml(c.sub) + '</div>' : '') + '</div>').join('');

  const why = {
    'detector-absent': 'Decided by the detector: a required object was never seen in the window. The language model cannot override this.',
    'detector-prohibited': 'Decided by the detector: a prohibited object was reliably present.',
    'intermittent-required': 'Decided by the detector: the required object appeared too inconsistently to judge.',
    'intermittent-prohibited': 'Decided by the detector: a prohibited object appeared intermittently.',
    'insufficient-window': 'Too little evidence had been collected to judge.',
    'vlm-contradictory': 'The language model read the frames differently from one another.',
    'vlm': 'Detector grounding was satisfied, so the language model judged the relationship.',
    'relationship-met': 'The relationship the standard asks about was observed to have the required value.',
    'relationship-violated': 'The relationship the standard asks about was observed to have the wrong value.',
    'relationship-unclear': 'The frames did not show the relationship clearly enough to decide.',
    'relationship-object-absent': 'Decided by the detector: the object was never in the window, so the relationship cannot hold.',
    'vlm-contradicts-detector': 'The model denied an object the detector had measured, so its reading was discarded.',
    'vlm-offtopic': 'The model answered about something this standard does not mention, so its reading was discarded.',
    'single-frame-fallback': 'Single-frame fallback: no temporal evidence and no detector grounding.'
  }[latest.decided_by] || '';
  basis.textContent = [why, ...(latest.notes || [])].filter(Boolean).join(' ');
}

let lastShots = [];
function renderFrames(latest){
  const framesEl = $('frames');
  const shots = (latest && latest.frames && latest.frames.length)
    ? latest.frames
    : (latest && latest.evidence_path
        ? [{path: latest.evidence_path, rel_ts: null, detections: []}] : []);
  lastShots = shots;
  const key = shots.map(f => f.path).join('|');
  if (framesEl.dataset.key !== key){
    framesEl.dataset.key = key;
    framesEl.innerHTML = shots.length ? shots.map((f, i) => {
      const src = '/api/evidence/' + String(f.path).replace('evidence/', '');
      const t = (f.rel_ts == null) ? ''
        : '<div class="t">+' + Number(f.rel_ts).toFixed(1) + ' s</div>';
      return '<button class="shot" type="button" data-i="' + i + '" title="Click to enlarge">' +
             '<img alt="Evidence frame judged by the model" src="' + src + '">' +
             '<div class="boxes">' + boxBlock(f.detections) + '</div>' + t + '</button>';
    }).join('')
      : '<p class="blank">The frames the vision-language model actually judged appear here, ' +
        'sampled across the evidence window.</p>';
  }
  $('evcap').textContent = shots.length > 1 ? shots.length + ' evidence frames' : 'Evidence frames';
  if (latest) $('evtime').textContent = new Date(latest.ts * 1000).toLocaleTimeString();
}

/* Boxes are normalised to the frame and every cell is exactly 16:9 like the
   1280x720 source, so percentage positioning lands 1:1 on the uncropped image. */
function boxBlock(detections){
  return (detections || []).map(d => {
    const [x1, y1, x2, y2] = d.bbox || [0, 0, 0, 0];
    const st = 'left:' + (x1 * 100) + '%;top:' + (y1 * 100) + '%;width:' +
               ((x2 - x1) * 100) + '%;height:' + ((y2 - y1) * 100) + '%';
    return '<div class="bx" style="' + st + '"><span>' +
           escapeHtml(d.label) + ' ' + Math.round((d.confidence || 0) * 100) + '%</span></div>';
  }).join('');
}

/* ----------------------------------------------------------------- lightbox */
function openLightbox(i){
  const f = lastShots[i];
  if (!f) return;
  $('lightbox-img').src = '/api/evidence/' + String(f.path).replace('evidence/', '');
  $('lightbox-boxes').innerHTML = boxBlock(f.detections);
  const dets = (f.detections || []).map(d =>
    escapeHtml(d.label) + ' ' + Math.round((d.confidence || 0) * 100) + '%').join('   ');
  $('lightbox-cap').textContent =
    (f.rel_ts != null ? '+' + Number(f.rel_ts).toFixed(1) + ' s   ' : '') + dets;
  $('lightbox').setAttribute('open', '');
}
function closeLightbox(){ $('lightbox').removeAttribute('open'); }

document.addEventListener('click', e => {
  const shot = e.target.closest && e.target.closest('.shot');
  if (shot){ openLightbox(Number(shot.dataset.i)); return; }
  if (e.target.closest && e.target.closest('.lightbox')) closeLightbox();
});

/* ------------------------------------------------------------------ history
 *
 * A read-only view over audit/inspections.jsonl. It loads on entry and after a
 * new verdict - never on a timer - because the audit trail only changes when an
 * inspection completes, and the state stream already tells us when that happens.
 */
let historyRows = [];
let lastVerdictCount = -1;
/* Which inspections are ticked. Export follows the selection when there is one
   and falls back to the whole history when there is not, so the controls mean
   "export what you are looking at" either way. */
let selected = new Set();

function verdictPill(v){
  const safe = ['pass','fail','unclear'].includes(v) ? v : 'unclear';
  return '<span class="vpill ' + safe + '">' + escapeHtml(safe.toUpperCase()) + '</span>';
}

function evidenceCell(r){
  const bits = [];
  if (r.required_summary) bits.push(escapeHtml(r.required_summary));
  if (r.prohibited_summary) bits.push('not allowed: ' + escapeHtml(r.prohibited_summary));
  if (!bits.length) bits.push('<span style="color:var(--faint)">no detector grounding</span>');
  return bits.join('<br>');
}

function latencyCell(r){
  const parts = [];
  if (r.vlm_ms != null) parts.push(fmtMs(r.vlm_ms));
  if (r.end_to_end_ms != null) parts.push('<span style="color:var(--faint)">' + fmtMs(r.end_to_end_ms) + ' total</span>');
  return parts.join('<br>') || '\u2014';
}

function renderHistory(rows){
  historyRows = rows;
  setText($('hist-count'), String(rows.length));
  $('hist-empty').hidden = rows.length > 0;
  // Drop ids that are no longer in the list, so the count can never claim more
  // than is actually selectable.
  const present = new Set(rows.map(r => r.id));
  for (const id of [...selected]) if (!present.has(id)) selected.delete(id);
  $('hist-body').innerHTML = rows.map((r, i) =>
    '<tr class="row' + (selected.has(r.id) ? ' picked' : '') + '" data-i="' + i +
      '" data-id="' + escapeHtml(r.id) + '" tabindex="0" aria-expanded="false">' +
      '<td class="pick"><input type="checkbox" aria-label="Select this inspection"' +
        (selected.has(r.id) ? ' checked' : '') + '></td>' +
      '<td class="when">' + escapeHtml((r.iso || '').replace('T', ' ').replace('+00:00', 'Z')) + '</td>' +
      '<td>' + verdictPill(r.verdict) +
        (r.decided_by ? '<span class="decided">' + escapeHtml(r.decided_by) + '</span>' : '') + '</td>' +
      '<td class="std">' + escapeHtml(r.standard || '') + '</td>' +
      '<td class="why">' + escapeHtml(r.reason || '') + '</td>' +
      '<td class="ev">' + evidenceCell(r) + '</td>' +
      '<td class="num">' + latencyCell(r) + '</td>' +
    '</tr>').join('');
  syncSelectionUi();
}

function toggleDetail(tr){
  const i = Number(tr.dataset.i);
  const r = historyRows[i];
  if (!r) return;
  const open = tr.nextElementSibling && tr.nextElementSibling.classList.contains('detail');
  if (open){ tr.nextElementSibling.remove(); tr.setAttribute('aria-expanded', 'false'); return; }
  const shots = (r.frames || []).map(f =>
    '<div class="shot"><img alt="Evidence frame from this inspection" src="/api/evidence/' +
    String(f.path).replace('evidence/', '') + '">' +
    '<div class="boxes">' + boxBlock(f.detections) + '</div>' +
    (f.rel_ts != null ? '<div class="t">+' + Number(f.rel_ts).toFixed(1) + ' s</div>' : '') +
    '</div>').join('');
  const meta = [
    r.window_frames != null ? '<b>' + r.window_frames + '</b> frames over <b>' +
      (r.window_s != null ? r.window_s.toFixed(1) : '?') + ' s</b>' : '',
    r.frames_judged ? '<b>' + r.frames_judged + '</b> frames judged by the model' : '',
    r.trigger_label ? 'triggered by <b>' + escapeHtml(r.trigger_label) + '</b>' : '',
    r.id ? 'id <b>' + escapeHtml(r.id) + '</b>' : ''
  ].filter(Boolean).join('<br>');
  // Expanding a two-rule inspection shows how each rule was decided, against
  // the one shared evidence set above.
  const perRule = (r.rules && r.rules.length > 1)
    ? '<ol class="rulesresult">' + r.rules.map(x => {
        const v = String(x.verdict || 'unclear').toLowerCase();
        return '<li><span class="rn">Rule ' + escapeHtml(String(x.index || '')) + '</span>' +
          '<span class="rv ' + v + '">' + escapeHtml(v.toUpperCase()) + '</span>' +
          '<span class="rr">' + escapeHtml(x.reason || '') + '</span>' +
          '<span class="rt">' + escapeHtml(x.text || '') + '</span></li>';
      }).join('') + '</ol>'
    : '';
  const row = document.createElement('tr');
  row.className = 'detail';
  row.innerHTML = '<td colspan="7"><div class="detail-in">' +
    (shots ? '<div class="detail-shots">' + shots + '</div>' : '') +
    '<div class="detail-meta">' + meta + perRule + '</div></div></td>';
  tr.after(row);
  tr.setAttribute('aria-expanded', 'true');
}

function exportHref(base){
  // No selection means "everything", which is what the whole-history export
  // always did. With a selection, only those ids travel.
  if (!selected.size) return base;
  return base + '?ids=' + encodeURIComponent([...selected].join(','));
}

function syncSelectionUi(){
  const n = selected.size;
  const label = $('sel-count');
  setText(label, n ? '  ·  ' + n + ' selected' : '');
  label.hidden = n === 0;
  $('sel-none').hidden = n === 0;
  setText($('sel-all'), n && n === historyRows.length ? 'Select none' : 'Select all');
  for (const [id, base] of [['export-csv', '/api/export.csv'],
                            ['export-json', '/api/export.json'],
                            ['export-zip', '/api/export.zip']]) {
    const a = $(id);
    setAttr(a, 'href', exportHref(base));
    setAttr(a, 'title', n ? 'Export the ' + n + ' selected inspection' + (n === 1 ? '' : 's')
                          : 'Export all recorded inspections');
  }
  const verb = n ? 'selected' : 'all';
  setText($('export-csv'), 'CSV');
  setText($('export-json'), 'JSON');
  setText($('export-zip'), 'Package');
  setAttr($('sel-all'), 'aria-label', 'Export ' + verb);
}

function toggleSelection(id, on){
  if (on) selected.add(id); else selected.delete(id);
  const tr = document.querySelector('.histtable tr.row[data-id="' + CSS.escape(id) + '"]');
  if (tr) tr.classList.toggle('picked', on);
  syncSelectionUi();
}

$('sel-all').addEventListener('click', () => {
  if (selected.size === historyRows.length) selected.clear();
  else historyRows.forEach(r => selected.add(r.id));
  renderHistory(historyRows);
});
$('sel-none').addEventListener('click', () => { selected.clear(); renderHistory(historyRows); });

document.addEventListener('change', e => {
  const box = e.target.closest && e.target.closest('.histtable td.pick input');
  if (!box) return;
  const tr = box.closest('tr.row');
  if (tr) toggleSelection(tr.dataset.id, box.checked);
});

async function loadHistory(){
  try {
    const r = await fetch('/api/history?limit=200');
    if (!r.ok) throw new Error(r.statusText);
    const body = await r.json();
    renderHistory(body.records || []);
  } catch (_) {
    notify('Could not read the audit trail.', 'bad');
  }
}

document.addEventListener('click', e => {
  if (e.target.closest && e.target.closest('.histtable td.pick')) return;  // ticking is not expanding
  const tr = e.target.closest && e.target.closest('.histtable tr.row');
  if (tr) toggleDetail(tr);
});
document.addEventListener('keydown', e => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  const tr = e.target.closest && e.target.closest('.histtable tr.row');
  if (tr){ e.preventDefault(); toggleDetail(tr); }
});

/* -------------------------------------------------------------- camera check */
function renderLive(d){
  const DASH = '\u2014';
  setText($('live-fps'),    d.fps ? d.fps.toFixed(1) : DASH);
  setText($('live-objs'),   String((d.detections || []).length));
  setText($('live-thresh'), d.gate_min_confidence != null
    ? Math.round(d.gate_min_confidence * 100) + '%' : DASH);
  setText($('live-win'),    d.window_frames != null ? String(d.window_frames) : DASH);
  setText($('liveage'),     d.age_s != null ? d.age_s.toFixed(1) + ' s ago' : '');
  setHtml($('liveboxes'),   boxBlock(d.detections));
  const rows = d.classes || [];
  setHtml($('classlist'), rows.length
    ? rows.map(c =>
        '<div class="classrow"><span class="n">' + escapeHtml(c.label) + '</span>' +
        '<span class="x">&times;' + c.count + '</span>' +
        '<span class="c">' + Math.round(c.confidence * 100) + '%</span></div>').join('')
    : '<p class="none">' + (d.connected
        ? 'Nothing detected above the threshold right now.'
        : 'Modalix is not reachable, so there is nothing to show.') + '</p>');
}

/* Debug threshold: a diagnostic overlay only.
 *
 * It calls a separate endpoint that re-runs the detector on the current frame.
 * Nothing it returns enters the evidence window, the gate or the grounding
 * policy, so moving this slider cannot change a PASS/FAIL. Sub-threshold boxes
 * are drawn dashed and amber so they can never be read as acted-on detections.
 */
let debugOn = false, debugLevel = 0.20;

$('dbg-on').addEventListener('change', e => {
  debugOn = e.target.checked;
  $('dbg-level').disabled = !debugOn;
  $('dbg-note').hidden = !debugOn;
  if (!debugOn) setHtml($('liveboxes'), '');
});
$('dbg-level').addEventListener('input', e => {
  debugLevel = Number(e.target.value) / 100;
  setText($('dbg-val'), e.target.value + '%');
});

async function debugPoll(){
  try {
    const r = await fetch('/api/live/debug?threshold=' + debugLevel);
    if (!r.ok){
      if (r.status === 501) notify('This backend does not provide debug detection.', 'bad');
      return null;
    }
    return await r.json();
  } catch (_) { return null; }
}

function renderDebug(d){
  const dets = d.detections || [];
  setHtml($('liveboxes'), dets.map(x => {
    const [x1, y1, x2, y2] = x.bbox || [0, 0, 0, 0];
    const st = 'left:' + (x1 * 100) + '%;top:' + (y1 * 100) + '%;width:' +
               ((x2 - x1) * 100) + '%;height:' + ((y2 - y1) * 100) + '%';
    const cls = x.below_threshold ? 'bx dbg' : 'bx';
    return '<div class="' + cls + '" style="' + st + '"><span>' +
           escapeHtml(x.label) + ' ' + Math.round((x.confidence || 0) * 100) + '%</span></div>';
  }).join(''));
  setHtml($('classlist'), dets.length
    ? dets.map(x => {
        const alt = (x.alternatives || []).slice(1, 3)
          .map(a => escapeHtml(a.label) + ' ' + Math.round(a.confidence * 100) + '%').join(', ');
        return '<div class="classrow' + (x.below_threshold ? ' dbg' : '') + '">' +
          '<span class="n">' + escapeHtml(x.label) + '</span>' +
          (alt ? '<span class="x">also: ' + alt + '</span>' : '') +
          '<span class="c">' + Math.round((x.confidence || 0) * 100) + '%</span></div>';
      }).join('')
    : '<p class="none">Nothing above ' + Math.round(d.threshold * 100) +
      '% in this frame. The working threshold is ' +
      Math.round(d.production_threshold * 100) + '%.</p>');
}

async function livePoll(){
  if (view !== 'camera') return;
  try {
    const r = await fetch('/api/live');
    if (r.ok) renderLive(await r.json());
  } catch (_) {}
  if (debugOn){
    const d = await debugPoll();
    if (d) renderDebug(d);
  }
  const img = $('liveimg');
  img.onerror = () => { $('liveempty').hidden = false; };
  img.onload  = () => { $('liveempty').hidden = true; };
  img.src = '/api/live/frame.jpg?t=' + Date.now();
}

/* ------------------------------------------------------------------- stream */
let es;
function connect(){
  es = new EventSource('/api/stream');
  es.onmessage = e => { try { render(JSON.parse(e.data)); } catch (_) {} };
  es.onerror = () => { es.close(); setTimeout(connect, 2000); };
}

/* ------------------------------------------------------------ the standard */
/* The applied rules, as the host confirmed them. The inputs hold DRAFT text and
   are the operator's to edit freely; this is what is actually in force. Keeping
   the two apart is what lets the applied text stay in the box - it used to be
   erased on apply, which left the operator with no record of the rule they had
   just set and nothing to edit for the next one. */
let appliedRules = [];

function ruleInputs(){
  return [$('typed'), $('typed2')];
}

function visibleRuleCount(){
  return $('rule2row').hidden ? 1 : 2;
}

/* Applied vs draft is shown, not guessed: a box matching what is in force reads
   as settled, one the operator has since edited reads as a draft. */
function markDraftState(){
  ruleInputs().forEach((el, i) => {
    const applied = appliedRules[i] || '';
    const val = el.value.trim();
    el.classList.toggle('applied', !!val && val === applied);
    el.classList.toggle('draft', !!val && val !== applied);
  });
}

function showRule2(show){
  $('rule2row').hidden = !show;
  $('rule2-add').hidden = show;
  if (!show) $('typed2').value = '';
  markDraftState();
}

async function setRules(rules){
  try {
    const r = await fetch('/api/standard', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({rules})
    });
    if (!r.ok){ notify('Could not set the standard: ' + await r.text(), 'bad'); return; }
    const body = await r.json().catch(() => null);
    // Echo back exactly what the host applied, so the box shows the rule in
    // force rather than whatever was typed.
    if (body && Array.isArray(body.rules)) applyRulesToInputs(body.rules);
    notify('');
  } catch (_) {
    notify('Could not reach the Foreman host.', 'bad');
  }
}

function applyRulesToInputs(rules){
  appliedRules = rules.slice(0, 2);
  const [a, b] = ruleInputs();
  a.value = appliedRules[0] || '';
  if (appliedRules.length > 1){
    showRule2(true);
    b.value = appliedRules[1];
  }
  markDraftState();
}

function setStandard(text){ return setRules([text]); }

$('save').onclick = () => {
  const values = ruleInputs()
    .slice(0, visibleRuleCount())
    .map(el => el.value.trim())
    .filter(Boolean);
  if (values.length) setRules(values);
};
ruleInputs().forEach(el => {
  el.addEventListener('input', markDraftState);
  // Editing the box changes nothing until Set standard is pressed.
  el.addEventListener('keydown', e => { if (e.key === 'Enter') $('save').click(); });
});
/* Voice fills the rule the operator last had selected, so "select Rule 2, speak"
   does what it looks like it does. Defaults to Rule 1. */
let voiceTarget = 'typed';
ruleInputs().forEach(el => el.addEventListener('focus', () => { voiceTarget = el.id; }));

$('rule2-add').onclick = () => { showRule2(true); $('typed2').focus(); };
$('rule2-remove').onclick = () => { showRule2(false); $('typed').focus(); };

$('clear').onclick = async () => {
  // Clear session really clears: the boxes empty, the second rule collapses and
  // nothing is left that could be mistaken for a rule still in force.
  appliedRules = [];
  $('typed').value = '';
  showRule2(false);
  markDraftState();
  notify('');
  try {
    await fetch('/api/session/clear', {method: 'POST'});
  } catch (_) {
    notify('Could not reach the Foreman host.', 'bad');
  }
};

/* ------------------------------------------------------------------- speech */
let recorder, chunks = [];

function setRecordingUi(on, label){
  recording = on;
  const btn = $('rec');
  btn.setAttribute('aria-pressed', String(on));
  btn.textContent = label;
  // Locking the selector while a capture is open means the language cannot
  // change under an in-flight recording.
  $('speechlang').disabled = on;
}

$('rec').onclick = async () => {
  const btn = $('rec');
  if (recorder && recorder.state === 'recording'){ recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    // Pin the language at the moment recording starts, so a mid-capture change
    // cannot apply to audio that was already being recorded.
    const langForThisTake = speechLang;
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      setRecordingUi(false, 'Transcribing on Modalix');
      btn.disabled = true;
      const fd = new FormData();
      fd.append('file', new Blob(chunks, {type: 'audio/webm'}), 'speech.webm');
      try {
        const r = await fetch('/api/standard/speak?language=' + encodeURIComponent(langForThisTake),
                              {method: 'POST', body: fd});
        const body = await r.json().catch(() => null);
        if (!r.ok){
          notify('Speech not accepted: ' + ((body && body.detail) || r.statusText), 'bad');
        } else if (body && body.accepted === false){
          // The standard in force is deliberately left alone here.
          notify('Speech unclear — please repeat. The current standard was kept.', 'bad');
        } else if (body){
          const detected = DETECTED_LABEL[body.language] || body.language;
          // Put the transcript in the field it was spoken for. The host applied
          // it as rule 1; if the operator was filling rule 2, the text lands
          // there as a draft and Set standard applies both together.
          const target = ($('rule2row').hidden || voiceTarget === 'typed') ? 'typed' : 'typed2';
          if (body.text) $(target).value = body.text;
          if (target === 'typed2'){
            // Rule 1 is unchanged; rule 2 is a draft until Set standard.
            markDraftState();
            notify('Heard (' + detected + '). Press Set standard to apply both rules.');
          } else {
            notify(langForThisTake === 'auto'
              ? 'Detected: ' + detected + '. Standard updated.'
              : 'Standard updated (' + detected + ').', 'good');
          }
        }
      } catch (_) {
        notify('Could not reach the DevKit for transcription.', 'bad');
      }
      btn.disabled = false;
      setRecordingUi(false, 'State the standard');
    };
    recorder.start();
    setRecordingUi(true, 'Stop and transcribe');
    notify('Recording — speak the standard in ' + LANG_LABEL[langForThisTake] + '.');
  } catch (_) {
    notify('No microphone access. Type the standard instead.', 'bad');
  }
};

/* --------------------------------------------------------------------- boot */
fetch('/api/config').then(r => r.json()).then(c => {
  $('insight').href = c.insight_url;
  if (!c.temporal){
    notify('Single-frame fallback mode: no temporal evidence and no detector grounding.', 'bad');
  }
}).catch(() => {});

initLanguage();
if (!location.hash) location.replace('#/inspection');
applyRoute();
connect();
