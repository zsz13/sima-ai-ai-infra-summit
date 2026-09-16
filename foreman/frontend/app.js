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
const ROUTES = {'#/inspection': 'inspection', '#/camera': 'camera'};
let view = null;
let liveTimer = null;

function currentRoute(){ return ROUTES[location.hash] || 'inspection'; }

async function applyRoute(){
  const next = currentRoute();
  if (next === view) return;
  view = next;
  const camera = view === 'camera';

  $('view-inspection').hidden = camera;
  $('view-camera').hidden = !camera;
  $('standardbar').hidden = false;          // the standard stays visible in both
  $('nav-inspection').setAttribute('aria-current', camera ? 'false' : 'page');
  $('nav-camera').setAttribute('aria-current', camera ? 'page' : 'false');
  document.title = camera ? 'Camera Check · Foreman' : 'Foreman';

  if (liveTimer){ clearInterval(liveTimer); liveTimer = null; }
  if (camera){
    livePoll();
    liveTimer = setInterval(livePoll, 200);
  }
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
  if (view === 'camera') location.hash = '#/inspection';
});

/* -------------------------------------------------------------------- render */

function chipsFor(parsed){
  if (!parsed) return '';
  const out = [];
  for (const c of (parsed.required || [])) out.push('<span class="chip req">' + escapeHtml(c) + ' required</span>');
  for (const c of (parsed.prohibited || [])) out.push('<span class="chip pro">' + escapeHtml(c) + ' prohibited</span>');
  if (!parsed.grounded && parsed.raw){
    // Say so plainly: without a detector-supported object the verdict rests on
    // the language model alone, and the operator deserves to know that.
    out.push('<span class="chip warn">no detector grounding</span>');
  }
  if (parsed.language) out.push('<span class="chip">' + (DETECTED_LABEL[parsed.language] || parsed.language) + '</span>');
  return out.join('');
}

function render(s){
  setAttr($('dot'), 'class', 'dot ' + (s.connected ? 'up' : 'down'));
  setText($('health'), s.connected ? 'Modalix connected' : 'Modalix unreachable');
  if (!s.connected && s.last_error) notify(s.last_error, 'bad');

  const std = $('standard');
  setText(std, s.standard || 'No standard set');
  std.classList.toggle('none', !s.standard);
  $('standardbar').classList.toggle('set', !!s.standard);
  setHtml($('chips'), s.standard ? chipsFor(s.parsed_standard) : '');

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
    setText(verdict, 'Judging'); setAttr(verdict, 'class', 'verdict small');
    setText(reason, 'The vision-language model is reading this item against your standard.');
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
  setStyle($('slab'), '--signal', signal);
  setStyle($('arming'), 'width', (s.gate_state === 'settling' ? s.gate_progress * 100 : 0) + '%');

  renderMetrics(latest);
  renderFrames(latest);

  setText($('m-see'), s.connected && s.fps ? s.fps.toFixed(0) + ' fps' : '—');
  setText($('m-understand'), latest ? fmtMs(latest.metrics.inference_ms) : '—');
  setText($('m-act'), latest ? fmtMs(latest.metrics.end_to_end_ms) : '—');

  setText($('histk'), s.counters.total ? 'Recent verdicts, newest first' : 'No verdicts yet');
  setHtml($('history'), s.recent.slice(0, 40)
    .map(r => '<div class="tile ' + r.verdict + '" title="' +
              escapeHtml(r.verdict + ': ' + r.reason) + '"></div>').join(''));
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
      v: pct + '%', k: o.label + ' present',
      sub: o.frames_present + ' / ' + o.frames_total + ' frames',
      cls: o.frames_present === 0 ? 'absent' : (pct >= 60 ? 'full' : '')
    });
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
  if (v.frames_judged){
    cards.push({v: Math.max(v.frames_supporting || 0, v.frames_contradicting || 0) + ' / ' + v.frames_judged,
                k: 'frames in agreement', sub: 'across the evidence window'});
  }
  if (latest.window && latest.window.total_frames){
    cards.push({v: latest.window.total_frames, k: 'frames of evidence',
                sub: (latest.window.duration_s || 0).toFixed(1) + ' s window'});
  }
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

async function livePoll(){
  if (view !== 'camera') return;
  try {
    const r = await fetch('/api/live');
    if (r.ok) renderLive(await r.json());
  } catch (_) {}
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
async function setStandard(text){
  try {
    const r = await fetch('/api/standard', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    if (!r.ok){ notify('Could not set the standard: ' + await r.text(), 'bad'); return; }
    notify('');
    $('typed').value = '';
  } catch (_) {
    notify('Could not reach the Foreman host.', 'bad');
  }
}
$('save').onclick = () => { const v = $('typed').value.trim(); if (v) setStandard(v); };
$('typed').onkeydown = e => { if (e.key === 'Enter') $('save').click(); };
$('clear').onclick = () => fetch('/api/session/clear', {method: 'POST'});

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
          notify(langForThisTake === 'auto'
            ? 'Detected: ' + detected + '. Standard updated.'
            : 'Standard updated (' + detected + ').', 'good');
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
