// ReadyMade Tracker web UI — talks ONLY to the bridge (POST /osc + GET /events). Builds itself from /rmt/query.
'use strict';
(() => {
  let PREFIX = '/rmt';
  const $ = (id) => document.getElementById(id);
  const STATE_COLOR = { RESTING: '#00dc00', MOVED: '#ff8c00', SETTLING: '#00dcdc', HIDDEN: '#a0a0a0' };
  const CLUSTER_COLORS = ['#50c8dc', '#e6a032', '#c878dc', '#78dc8c', '#dc8c78', '#a0dcdc'];
  // param panel: two-level CATEGORY → section grouping, in pipeline order (not alphabetical).
  const CATEGORIES = [
    ['INPUT', ['input', 'camera', 'mcu']],
    ['PREPROCESS', ['preprocess']],
    ['PROCESS', ['observe', 'classify', 'pose', 'identity', 'reid', 'efd', 'grouper', 'features', 'clusters']],
    ['OUTPUT', ['mjpeg', 'tuio2', 'oscout', 'rec']],
    ['CONTROL', ['oscin', 'webgui']],
  ];
  const OPEN_BY_DEFAULT = new Set(['input', 'camera']);   // #4 — input source open on load
  // friendlier section headers — the conf [section] key stays on hover + drives SAVE unchanged.
  const SECTION_LABELS = { observe: 'CONTOURS', classify: 'STATE', mcu: 'MCU' };
  // sections the webgui does NOT surface — pure .conf territory: the control-plane bind config you're
  // already connected THROUGH (oscin + this webgui), and perf (cv2_threads: machine tuning, set per rig
  // via conf/--cv2-threads). The params still exist in /query and /rmt/set for other clients;
  // this is a webgui-local curation, not a param-level hide.
  const HIDDEN_SECTIONS = new Set(['oscin', 'webgui', 'perf']);
  // a section can split its params into expandable sub-groups: [groupLabel, [keys]]. '' = ungrouped (top).
  const SUBGROUPS = {
    preprocess: [
      ['', ['preprocess', 'smooth_alpha']],                                        // master switch + EMA, ungrouped
      ['FLIP', ['flip']],
      ['CROP', ['crop_enabled', 'crop_top_x', 'crop_top_y', 'crop_width', 'crop_height']],
      ['MASK', ['mask_enabled', 'mask_cx', 'mask_cy', 'mask_radius']],
      ['FLATFIELD', ['flatfield_enabled', 'flatfield_path']],
      ['LENS', ['lens_enabled', 'lens_path']],
    ],
    observe: [
      ['', ['fg_threshold', 'blur', 'min_area', 'k']],                             // detection knobs, at top
      ['MOTION', ['motion_threshold', 'motion_frac']],
      ['REJECTION GATES', ['max_area', 'edge_ignore_margin']],
      ['MATCHING', ['reassoc_dist', 'assoc_size_tol', 'relocate_dist', 'reloc_shape_tol']],
    ],
    identity: [
      ['', ['pool_ttl_seconds', 'clear_frac']],
      ['EARLY GONE', ['away_quiet_seconds', 'table_quiet_frac']],
    ],
  };

  // ── bridge ────────────────────────────────────────────────────────────
  async function osc(address, ...args) {
    const r = await fetch('/osc', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address, args }) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return (await r.json()).messages || [];
  }
  const setDot = (cls, title) => { const d = $('conn'); d.className = 'dot ' + cls; d.title = title; };

  // a meaningful slider step per param: a 1 / 2 / 5 × 10ⁿ value near span/100 (≈ 0.1 … 0.001,
  // never a 9-digit float). decimalsFor gives how many places to SHOW/round the value to.
  function niceStep(lo, hi) {
    const span = Math.abs(hi - lo) || 1;
    const mag = Math.pow(10, Math.floor(Math.log10(span / 100)));
    const n = (span / 100) / mag;
    return (n >= 5 ? 5 : n >= 2 ? 2 : 1) * mag;
  }
  const decimalsFor = (step) => Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  // display a value at the param's meaningful precision: ints/text as-is, floats rounded to the
  // slider's step. Used BOTH on first render AND when the server acks a set (so the ack's raw
  // float — e.g. 0.8399999737739563 — never clobbers the shown 0.84).
  const fmtVal = (p, v) => (v == null || v === '' || isNaN(+v) || !p || p.type === 'int')
    ? String(v) : (+v).toFixed(decimalsFor(niceStep(p.lo, p.hi)));

  // ── control panel — built entirely from /rmt/query ──────────────────
  let PARAMS = [];                                    // last queried descriptors
  let ACTIVE = null;                                  // active category name
  let SOURCE = 'video';                               // input source — gates the camera panel + playback
  function applyTransportVis() {                      // pause/resume/step only make sense for a video source
    const pb = $('playback'); if (pb) pb.style.display = SOURCE === 'video' ? '' : 'none';
  }
  const dirty = new Set();                            // "section.key" changed since last save
  const timers = {};
  function markDirty(section, key) {
    dirty.add(section + '.' + key);
    $('save').classList.add('dirty');
  }
  function debouncedSet(section, key, value) {
    clearTimeout(timers[section + '.' + key]);
    timers[section + '.' + key] = setTimeout(async () => {
      try {
        const msgs = await osc(PREFIX + '/set', section, key, value);
        const ack = msgs.find((m) => m.address.endsWith('/ack'));
        if (ack) applyAck(section, key, ack.args);
      } catch (e) { console.warn('set failed', e); }
    }, 60);
  }
  function applyAck(section, key, a) {                // a = [section, key, ok|err, value|reason, clamped?]
    const row = document.querySelector(`[data-p="${section}.${key}"]`);
    const status = a[2], detail = a[3];
    const p = PARAMS.find((x) => x.section === section && x.key === key);
    if (row) {
      const note = row.querySelector('.note');
      if (status === 'err') { if (note) { note.textContent = detail; note.className = 'note err'; } return; }
      if (note) {
        note.textContent = !p || p.hot ? (a[4] == 1 ? 'clamped' : '') : '⟳';
        note.className = 'note' + (!p || p.hot ? '' : ' staged');
        note.title = !p || p.hot ? '' : 'staged — save + restart to apply';
      }
      const out = row.querySelector('.val'); if (out) out.textContent = fmtVal(p, detail);
    }
    if (status !== 'err') {
      if (p) p.current = detail;                        // keep PARAMS in sync so the crop/mask overlay follows edits
      if (section === 'preprocess') { draw(); updateCropToggleVis(); }   // live-redraw the ROI overlay + crop-toggle vis
      markDirty(section, key);
    }
  }

  function buildPanel(params) {
    PARAMS = params;
    const bySection = {};
    for (const p of params) (bySection[p.section] ||= []).push(p);
    const known = new Set(CATEGORIES.flatMap(([, s]) => s));
    const other = Object.keys(bySection).filter((s) => !known.has(s) && !HIDDEN_SECTIONS.has(s)).sort();
    const vis = (s) => bySection[s] && !HIDDEN_SECTIONS.has(s)   // .conf-only sections never surface here
      && (s !== 'camera' || SOURCE === 'camera');                 // hide camera unless source=camera
    const groups = (other.length ? [...CATEGORIES, ['OTHER', other]] : CATEGORIES)
      .map(([cat, secs]) => [cat, secs.filter(vis)]).filter(([, secs]) => secs.length);
    // tabs
    const tabs = $('tabs'); tabs.innerHTML = '';
    if (!groups.some(([c]) => c === ACTIVE)) ACTIVE = groups.length ? groups[0][0] : null;   // default INPUT
    for (const [cat] of groups) {
      const b = document.createElement('button');
      b.className = 'tab' + (cat === ACTIVE ? ' on' : ''); b.textContent = cat;
      b.onclick = () => { ACTIVE = cat; buildPanel(PARAMS); };
      tabs.appendChild(b);
    }
    // sections for the active tab
    const root = $('params'); root.innerHTML = '';
    const sections = (groups.find(([c]) => c === ACTIVE) || [null, []])[1];
    const single = sections.length === 1;             // one section → tab name IS the label; no collapsable
    for (const section of sections) {
      const body = SUBGROUPS[section] ? subgroupedBody(section, bySection[section])
                                      : rowsGrid(bySection[section]);
      if (single) {
        root.appendChild(body);
      } else {
        const sec = document.createElement('details'); sec.className = 'sec';
        sec.open = OPEN_BY_DEFAULT.has(section);       // #4 — collapsed except input source
        const summ = document.createElement('summary'); summ.textContent = SECTION_LABELS[section] || section;
        if (SECTION_LABELS[section]) summ.title = '[' + section + ']';   // renamed → show the conf key on hover
        sec.appendChild(summ);
        sec.appendChild(body);
        root.appendChild(sec);
      }
    }
    applyTransportVis();
  }

  function rowsGrid(params) {
    const rows = document.createElement('div'); rows.className = 'rows';   // one shared grid → aligned labels
    for (const p of params) if (!p.hidden) rows.appendChild(buildRow(p));   // hidden = off-panel
    return rows;
  }

  function subgroupedBody(section, params) {           // split a section into expandable sub-groups
    const byKey = {}; for (const p of params) byKey[p.key] = p;
    const box = document.createElement('div');
    const listed = new Set(SUBGROUPS[section].flatMap(([, keys]) => keys));
    for (const [glabel, keys] of SUBGROUPS[section]) {
      let gp = keys.map((k) => byKey[k]).filter(Boolean);
      if (section === 'preprocess' && glabel === 'CROP') gp = cropAsXYWH(byKey);
      if (!gp.length) continue;
      if (!glabel) { box.appendChild(rowsGrid(gp)); continue; }             // ungrouped (master switch), at top
      const d = document.createElement('details'); d.className = 'sec sub'; d.open = true;
      d.appendChild(Object.assign(document.createElement('summary'), { textContent: glabel }));
      d.appendChild(rowsGrid(gp));
      if (section === 'preprocess' && glabel === 'FLATFIELD') injectFlatfieldButtons(d);
      box.appendChild(d);
    }
    const leftover = params.filter((p) => !listed.has(p.key));               // a NEW param a stale grouping
    if (leftover.length) box.appendChild(rowsGrid(leftover));                // doesn't know must never vanish
    return box;
  }

  function injectFlatfieldButtons(groupEl) {           // icon-only capture / delete INLINE with the path field
    const input = groupEl.querySelector('[data-p="preprocess.flatfield_path"] input');
    if (!input) return;
    const wrap = document.createElement('span'); wrap.className = 'ctrl-ff';
    input.replaceWith(wrap); wrap.appendChild(input);   // input becomes flex-1 beside the two icon buttons
    const mk = (icon, title, fn) => { const b = document.createElement('button'); b.textContent = icon; b.title = title; b.onclick = () => { fn(); flash(b, '✓'); }; return b; };
    wrap.appendChild(mk('◉', 'capture: average 30 frames into a new flatfield (over an EMPTY table)', () => osc(PREFIX + '/flatfield', 'capture', 30).catch(() => {})));
    wrap.appendChild(mk('🗑', 'delete the flatfield', () => osc(PREFIX + '/flatfield', 'delete').catch(() => {})));
  }

  function cropAsXYWH(byKey) {                          // relabel the crop ROI params as X, Y, Width, Height
    const out = [];
    if (byKey.crop_enabled) out.push(byKey.crop_enabled);
    if (byKey.crop_top_x) out.push({ ...byKey.crop_top_x, label: 'X' });
    if (byKey.crop_top_y) out.push({ ...byKey.crop_top_y, label: 'Y' });
    if (byKey.crop_width) out.push({ ...byKey.crop_width, label: 'Width' });
    if (byKey.crop_height) out.push({ ...byKey.crop_height, label: 'Height' });
    return out;
  }

  function buildRow(p) {
    const row = document.createElement('div');
    row.className = 'row' + (p.hot ? '' : ' rebuild');
    row.dataset.p = p.section + '.' + p.key;
    const lab = document.createElement('label'); lab.textContent = p.label || p.key; lab.title = p.doc || '';
    row.appendChild(lab);
    const set = p.onSet || ((v) => debouncedSet(p.section, p.key, v));

    if (p.multi && p.enum) {                          // multiselect (features.enable) → checkbox group
      row.classList.add('multi');
      const box = document.createElement('div'); box.className = 'multi-box';
      const chosen = new Set((p.current || '').split(',').map((s) => s.trim()).filter(Boolean));
      for (const opt of p.enum) {
        const l = document.createElement('label');
        const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = chosen.has(opt);
        cb.onchange = () => { cb.checked ? chosen.add(opt) : chosen.delete(opt);
          set(p.enum.filter((o) => chosen.has(o)).join(',')); };
        l.appendChild(cb); l.appendChild(document.createTextNode(opt)); box.appendChild(l);
      }
      row.appendChild(box); appendNote(row, p); return row;
    }
    let control;
    if (p.enum && p.enum.length) {                    // enum → dropdown
      control = document.createElement('select');
      for (const opt of p.enum) { const o = document.createElement('option'); o.value = o.textContent = opt; control.appendChild(o); }
      control.value = p.current;
      control.onchange = () => { set(control.value);
        if (p.section === 'input' && p.key === 'source') { SOURCE = control.value; buildPanel(PARAMS); } };
    } else if (p.type === 'bool') {                   // bool → checkbox
      control = document.createElement('input'); control.type = 'checkbox';
      control.checked = p.current === '1' || p.current === 'True' || p.current === 'true';
      control.onchange = () => set(control.checked ? 1 : 0);
    } else if ((p.type === 'int' || p.type === 'float') && p.lo !== null && p.hi !== null
               && p.key !== 'port' && (p.hi - p.lo) <= 20000) {   // bounded, slidable → range
      control = document.createElement('input'); control.type = 'range';
      control.min = p.lo; control.max = p.hi;
      const stp = p.type === 'int' ? 1 : niceStep(p.lo, p.hi);   // meaningful precision per param
      const fmt = (v) => fmtVal(p, v);                           // no float noise, no 9-digit values (shared w/ ack)
      control.step = stp;
      control.value = p.current;
      const val = document.createElement('span'); val.className = 'val'; val.textContent = fmt(p.current);
      control.oninput = () => { val.textContent = fmt(control.value); set(+fmt(control.value)); };
      const ctrl = document.createElement('div'); ctrl.className = 'ctrl';   // one grid cell: slider + value
      ctrl.appendChild(control); ctrl.appendChild(val);
      row.appendChild(ctrl); appendNote(row, p); return row;
    } else {                                          // number (port / huge-range / unbounded) / string → text
      control = document.createElement('input');
      control.type = (p.type === 'int' || p.type === 'float') ? 'number' : 'text';
      if (control.type === 'number') {                // a real number field — exact entry (e.g. a port)
        if (p.lo !== null) control.min = p.lo;
        if (p.hi !== null) control.max = p.hi;
        if (p.type === 'int') control.step = 1;
      }
      control.value = p.current;
      control.onchange = () => {
        if (control.type !== 'number') { set(control.value); return; }
        const v = parseFloat(control.value);            // an emptied field is NaN → JSON null → the OSC
        if (!isNaN(v)) set(v);                          // encoder rejects it (a 502 per keystroke) — skip
      };
    }
    if (p.readonly) lockRo(control, p);               // structural / computed → show, but edit in readymade.conf
    row.appendChild(control); appendNote(row, p); return row;   // needs-restart rows stay EDITABLE (staged)
  }
  function lockRo(control, p) {                        // read-only param: show effective||current, strip edit/save
    const v = p.effective || p.current;
    control.classList.add('ro');
    control.title = (p.effective ? 'resolved at runtime — ' : '') + 'read-only (edit readymade.conf to change)';
    control.onchange = control.oninput = null;
    if (control.tagName === 'SELECT' || control.type === 'checkbox' || control.type === 'range') {
      control.disabled = true;
      if (control.type === 'checkbox') control.checked = v === '1' || v === 'true' || v === 'True';
      else control.value = v;
    } else { control.value = v; control.readOnly = true; }   // text / number stay copyable
  }
  function appendNote(row, p) {
    const n = document.createElement('span'); n.className = 'note';
    const isDirty = dirty.has(p.section + '.' + p.key);
    if (!p.hot && (p.dirty || isDirty)) { n.textContent = '⟳'; n.className = 'note staged'; n.title = 'staged — save + restart to apply'; }
    else if (!p.hot) { n.textContent = '⟳'; n.title = 'needs restart to change'; }
    row.appendChild(n);
  }

  async function query() {
    const el = $('params');
    const hadPanel = !!el.querySelector('.sec');                 // a built panel is showing
    if (!hadPanel) el.innerHTML = '<div class="muted">querying…</div>';
    let msgs;
    try { msgs = await osc(PREFIX + '/query'); }
    catch (e) {                                                  // keep the working panel on a hiccup —
      if (!hadPanel) el.innerHTML = '<div class="muted">query failed — is [oscin] enabled?</div>';
      throw e;                                                   //   only the never-loaded case shows text
    }
    const params = msgs.filter((m) => m.address.endsWith('/param')).map((m) => {
      const a = m.args;   // [section,key,type,hot,default,current,lo,hi,enum,doc,multi,dirty,effective,readonly,hidden]
      const num = (s) => (s === '' ? null : parseFloat(s));
      return { section: a[0], key: a[1], type: a[2], hot: a[3] == 1, default: a[4], current: a[5],
               lo: num(a[6]), hi: num(a[7]), enum: a[8] ? a[8].split(',') : null, doc: a[9],
               multi: a[10] == 1, dirty: a[11] == 1, effective: a[12] || null,
               readonly: a[13] == 1, hidden: a[14] == 1 };
    });
    if (!params.length) { $('params').innerHTML = '<div class="muted">no params (is [oscin] enabled?)</div>'; return; }
    SOURCE = (params.find((p) => p.section === 'input' && p.key === 'source') || {}).current || SOURCE;
    buildPanel(params);
    updateCropToggleVis();
  }
  const _truthy = (v) => v === true || v === 'true' || v === 'True' || v === '1';
  function updateCropToggleVis() {                     // the crop overlay toggle only matters when crop is on
    const p = PARAMS.find((x) => x.section === 'preprocess' && x.key === 'crop_enabled');
    const lbl = $('ov-crop').closest('label');
    if (lbl) lbl.style.display = (p && _truthy(p.current)) ? '' : 'none';
  }

  // ── save / restart / quit ─────────────────────────────────────────────
  function wireActions() {
    $('save').onclick = async () => {
      try {
        const msgs = await osc(PREFIX + '/save');
        const s = msgs.find((m) => m.address.endsWith('/saved'));
        if (s && s.args[0] >= 0) { dirty.clear(); $('save').classList.remove('dirty'); flash($('save'), 'saved ' + s.args[0]); }
        else flash($('save'), 'save failed');
      } catch (e) { flash($('save'), 'save failed'); }
    };
    $('restart').onclick = () => { if (confirm('Restart the tracker? Unsaved changes are lost — Save first.')) { osc(PREFIX + '/restart').catch(() => {}); awaitRestart(); } };
    $('quit').onclick = () => { if (confirm('Shut the tracker down?')) { osc(PREFIX + '/quit').catch(() => {}); setDot('err', 'quit'); banner('tracker shutting down…'); } };
    $('refresh').onclick = () => query().catch((e) => console.warn(e));
  }
  function flash(btn, txt) {                                     // re-entrant: restore the ORIGINAL label
    if (btn.dataset.orig === undefined) btn.dataset.orig = btn.textContent;
    btn.textContent = txt;
    clearTimeout(btn._flashT);
    btn._flashT = setTimeout(() => { btn.textContent = btn.dataset.orig; delete btn.dataset.orig; }, 1500);
  }
  function banner(txt) { let b = $('banner'); if (!b) { b = document.createElement('div'); b.id = 'banner'; b.style.cssText = 'position:fixed;top:0;left:0;right:0;padding:8px;text-align:center;background:#46708f;color:#fff;z-index:9;font-size:12px'; document.body.appendChild(b); } b.textContent = txt; }
  function awaitRestart() {                            // the bridge dies mid-restart → poll /info until it's back
    banner('restarting…'); setDot('err', 'restarting');
    let tries = 0;
    const t = setInterval(async () => {
      tries++;
      try { const r = await fetch('/info', { cache: 'no-store' }); if (r.ok) { clearInterval(t); location.reload(); } }
      catch (e) { if (tries > 60) { clearInterval(t); banner('restart timed out — reload manually'); } }
    }, 1000);
  }

  // ── transport / view controls ─────────────────────────────────────────
  let paused = false;                                            // synced from the server (head + ping)
  function setPaused(p) {
    paused = !!p;
    $('playpause').innerHTML = paused ? '▶' : '⏸';
    $('playpause').classList.toggle('on', paused);
  }
  function wireTransport() {
    document.querySelectorAll('[data-osc]').forEach((b) => b.onclick = () => {
      const arg = b.dataset.arg;
      osc(PREFIX + '/' + b.dataset.osc, ...(arg !== undefined ? [isNaN(arg) ? arg : +arg] : [])).catch(() => {});
    });
    $('playpause').onclick = () => {                             // ONE pause/resume toggle (like ● rec)
      setPaused(!paused);                                        // optimistic; the SSE ping confirms —
      osc(PREFIX + '/toggle').catch(() => {});                   //   a toggle from oscin/mcu/another tab
    };                                                           //   lands here too (server-synced)
    document.querySelectorAll('[data-view]').forEach((b) => b.onclick = () => {
      const leavingFeatures = featuresActive();
      setFeatures(false);                                          // a backend view exits the client inspector
      document.querySelectorAll('[data-view]').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      if (leavingFeatures) mjpegConnect();                         // the <img> was hidden in features → re-request the stream
      osc(PREFIX + '/view', b.dataset.view).catch(() => {});
    });
    $('btn-features').onclick = () => setFeatures(true);
    $('panel-toggle').onclick = () => {                          // hide/show the right param panel to clear space
      const c = document.body.classList.toggle('panel-collapsed');
      $('panel-toggle').textContent = c ? '⇤' : '⇥'; $('panel-toggle').classList.toggle('on', c);
      draw(); if (featuresActive()) getInspector().update(emitView(), scene.fdim);
    };
    ['ov-outlines', 'ov-bbox', 'ov-clusters', 'ov-groups', 'ov-ids', 'ov-rot', 'ov-mask', 'ov-crop', 'ov-margin', 'ov-gone', 'ov-legend'].forEach((id) => { $(id).onchange = draw; });   // redraw the overlay
  }
  // The "features" view is CLIENT-side (no backend /view): it swaps the video stage for the INLINE inspector
  // (EFD.createInspector — the EFD math + rendering live in efd.js, no separate page). Pause the transport,
  // study a shape, drag the EFD knobs — the reconstruction recomputes client-side from the frozen contour.
  let inspector = null;
  function getInspector() {
    if (!inspector) inspector = EFD.createInspector($('inspector'), { onParam: (k, v) => debouncedSet('efd', k, v) });
    return inspector;
  }
  const featuresActive = () => $('stage').classList.contains('features');
  // The FEATURES view follows the WIRE (owner decision): when the grouper fuses, the inspector
  // shows the emit view - compounds + unabsorbed objects - exactly what a client receives.
  // STABLE rows only: a SETTLING birth (or an unconfirmed mint) flickers for a few frames, and a
  // GONE object left the table - neither belongs in the band. A HIDDEN-but-PRESENT object (a hand
  // over it) KEEPS its chip: it is still part of the scene and of any compound (owner's rule).
  const stable = (o) => o.state !== 'SETTLING' && o.confirmed !== false
                        && (o.state !== 'HIDDEN' || o.present === true);
  const emitView = () => (scene.emit ? scene.objects.filter(o => scene.emit.includes(o.id))
                                     : scene.objects).filter(stable);
  function efdParam(key) {                                       // current efd.<key> from the last /query
    const p = PARAMS.find((x) => x.section === 'efd' && x.key === key);
    return p ? parseFloat(p.current) : NaN;
  }
  function setFeatures(on) {
    $('stage').classList.toggle('features', on);                 // CSS hides the video + overlay + shows #inspector
    $('btn-features').classList.toggle('on', on);
    if (on) {
      document.querySelectorAll('[data-view]').forEach((x) => x.classList.remove('on'));
      const insp = getInspector();
      insp.setParams(efdParam('bins'), efdParam('detail'));      // seed the knobs from the live [efd] backend values
      insp.update(emitView());
    }
  }

  // ── mjpeg video: (re)connect + auto-retry ─────────────────────────────
  // The stream drops when the tracker restarts / starts anew. The <img> used to hide itself on error and
  // never retry (forced a full page reload). Now it reconnects — on its own error AND whenever the SSE
  // re-opens (both die together on a restart), with a cache-buster so the browser actually re-requests.
  let MJPEG_PORT = 8080;
  function mjpegConnect() {
    const img = $('video'); img.style.display = '';
    img.src = `http://${location.hostname}:${MJPEG_PORT}/?t=${Date.now()}`;
  }

  // ── live overlay (SSE) ───────────────────────────────────────────────
  let scene = { objects: [] };
  let capFps = null, lastTs = null, sseWasUp = false;
  function connectSSE() {
    const es = new EventSource('/events');
    es.onopen = () => {                                          // re-open → restart video + REFRESH the
      setDot('ok', 'connected');                                 //   panel (a restart/reload from another
      if (sseWasUp) { mjpegConnect(); query().catch((e) => console.warn(e)); }   // surface changed params)
      sseWasUp = true;
    };
    es.onerror = () => setDot('err', 'disconnected');
    es.onmessage = (ev) => {
      let d; try { d = JSON.parse(ev.data); } catch { return; }
      if (d.paused !== undefined && d.paused !== null) setPaused(d.paused);   // head AND ping carry it
      if (d.ping) return;                                        // quiet-window ping: state only, no frame
      scene = d;
      $('objs').textContent = scene.objects.filter(o => !o.grp).length;   // the REAL scene count
      $('frame').textContent = scene.frame;
      updateStats(scene);
      updateRecBtn(scene.rec);
      draw();
      if (featuresActive()) getInspector().update(emitView(), scene.fdim);   // live-refresh the inspector
    };
  }
  function updateStats(s) {
    $('proc').textContent = s.proc_fps != null ? s.proc_fps.toFixed(0) : '--';
    if (s.view) $('view').textContent = s.view;
    if (s.ts != null && lastTs != null) {              // capture fps from consecutive ts deltas (EMA)
      const dt = s.ts - lastTs;
      if (dt > 1e-4 && dt < 5) { const inst = 1 / dt; capFps = capFps == null ? inst : capFps * 0.8 + inst * 0.2; $('cap').textContent = capFps.toFixed(0); }
    }
    lastTs = s.ts;
  }
  function updateRecBtn(rec) {
    const btn = $('rec-btn'); if (!btn) return;
    if (!rec) { btn.style.display = 'none'; return; }  // no rec field → [rec] not enabled → hide
    btn.style.display = '';
    if (rec.recording) { btn.textContent = `■ ${Math.round(rec.elapsed)}s`; btn.classList.add('recording'); }
    else { btn.textContent = '●'; btn.classList.remove('recording'); }
  }
  function drawLegend(ctx, w, h) {                     // small color key, bottom-left of the overlay
    const rows = [['RESTING', STATE_COLOR.RESTING], ['SETTLING', STATE_COLOR.SETTLING],
                  ['MOVED', STATE_COLOR.MOVED], ['HIDDEN (gone-soon)', STATE_COLOR.HIDDEN],
                  ['cluster', CLUSTER_COLORS[0]], ['group', '#c98bff']];
    const lh = 14, pad = 6, bw = 150, bh = rows.length * lh + pad * 2;
    const x0 = 8, y0 = h - bh - 8;
    ctx.fillStyle = 'rgba(10,10,10,0.72)'; ctx.fillRect(x0, y0, bw, bh);
    ctx.font = '10px monospace'; ctx.textBaseline = 'middle';
    rows.forEach(([name, col], i) => {
      const y = y0 + pad + i * lh + lh / 2;
      ctx.fillStyle = col; ctx.fillRect(x0 + pad, y - 4, 8, 8);
      ctx.fillStyle = '#ddd'; ctx.fillText(name, x0 + pad + 14, y);
    });
  }

  function draw() {
    if (!scene) return;
    const cv = $('overlay'), st = $('stage'), img = $('video');
    const w = st.clientWidth, h = st.clientHeight;
    if (cv.width !== w) cv.width = w; if (cv.height !== h) cv.height = h;
    const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, w, h);
    // Overlays are drawn CLIENT-side (toggleable) on the single-panel views. There are two: 'input' (the
    // full pre-crop frame — only crop/mask guides) and 'debug' (the cropped input masked to observe's fg;
    // object/cluster/id overlays go on top here). Only CONFIRMED objects are ever drawn — an un-confirmed
    // object (a fresh mint / a ghost that never persists) is NEVER shown; else the view floods with
    // hundreds of 3-frame ghost ids. Confirmed = persisted [classify].confirm_seconds.
    const view = scene.view || 'debug';
    const isInput = (view === 'input');                            // INPUT streams the FULL, pre-crop frame
    const showOutlines = $('ov-outlines').checked, showBbox = $('ov-bbox').checked,
          showClusters = $('ov-clusters').checked,
          showGroups = $('ov-groups').checked, showIds = $('ov-ids').checked, showRot = $('ov-rot').checked,
          showMask = $('ov-mask').checked, showCrop = $('ov-crop').checked, showMargin = $('ov-margin').checked,
          showGone = $('ov-gone').checked, showLegend = $('ov-legend').checked;
    if (showLegend) drawLegend(ctx, w, h);
    if (!showOutlines && !showBbox && !showClusters && !showGroups && !showIds && !showRot && !showMask && !showCrop && !showMargin) return;   // fully clean
    // map normalized 0..1 onto the video's ACTUAL displayed rect (object-fit:contain) — naturalWidth/
    // Height carry the MJPEG scale, so the overlay tracks the stream at any scale/aspect.
    let natW = img.naturalWidth, natH = img.naturalHeight;
    if (!(natW && natH)) {                       // no video (MJPEG off, or not loaded yet): take the aspect from
      const d = isInput ? scene.full_dim : scene.fdim;   // the FRAME dims the tracker sends — else the
      if (d && d[0] && d[1]) { natW = d[0]; natH = d[1]; }   // normalized coords stretch to the panel (circles → eggs)
    }
    let ox = 0, oy = 0, dw = w, dh = h;
    if (natW && natH) { const s = Math.min(w / natW, h / natH); dw = natW * s; dh = natH * s; ox = (w - dw) / 2; oy = (h - dh) / 2; }
    const X = (nx) => ox + nx * dw, Y = (ny) => oy + ny * dh;
    const cp = (k) => { const p = PARAMS.find((x) => x.section === 'preprocess' && x.key === k); const v = p ? parseFloat(p.current) : NaN; return isNaN(v) ? 0 : v; };
    // CROP ROI — a FULL-frame calibration guide, so only on the INPUT view (which now streams the uncropped
    // frame). crop_* are full-frame px, normalized by the FULL dims → the rectangle lands correctly.
    if (showCrop && isInput && scene.full_dim) {
      const [fw, fh] = scene.full_dim;
      const rw = cp('crop_width'), rh = cp('crop_height');           // 0 = to the frame edge
      const rx = X(cp('crop_top_x') / fw), ry = Y(cp('crop_top_y') / fh);
      const rex = X((rw ? cp('crop_top_x') + rw : fw) / fw), rey = Y((rh ? cp('crop_top_y') + rh : fh) / fh);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';                            // dim everything OUTSIDE the crop
      ctx.fillRect(ox, oy, dw, ry - oy); ctx.fillRect(ox, rey, dw, oy + dh - rey);
      ctx.fillRect(ox, ry, rx - ox, rey - ry); ctx.fillRect(rex, ry, ox + dw - rex, rey - ry);
      ctx.strokeStyle = '#e6c84a'; ctx.lineWidth = 1.5; ctx.setLineDash([7, 4]);   // yellow dashed ROI rect
      ctx.strokeRect(rx, ry, rex - rx, rey - ry); ctx.setLineDash([]);
    }
    // MASK circle — INPUT: full-frame coords (mask_* are full-frame px); other views: the backend sends it
    // pre-normalized to the cropped frame (scene.mask).
    if (showMask) {
      ctx.strokeStyle = '#ff5bc0'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      if (isInput && scene.full_dim && cp('mask_radius') > 0) {
        const [fw, fh] = scene.full_dim;
        ctx.beginPath(); ctx.arc(X(cp('mask_cx') / fw), Y(cp('mask_cy') / fh), cp('mask_radius') / fw * dw, 0, 7); ctx.stroke();
      } else if (!isInput && scene.mask) {
        const [mcx, mcy, mr] = scene.mask;
        ctx.beginPath(); ctx.arc(X(mcx), Y(mcy), mr * dw, 0, 7); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    // object overlays live in CROPPED coords → they only align on the cropped views; skip them on the full
    // INPUT view (which is the raw camera feed + the crop/mask calibration guides above).
    if (isInput) return;
    // the RIM gate ([observe].edge_ignore_margin): births inside the dotted band are rejected (and a
    // live blob reaching the frame edge flags its object `ignored`) — a calibration guide, like mask/crop.
    if (showMargin) {
      const p = PARAMS.find((x) => x.section === 'observe' && x.key === 'edge_ignore_margin');
      const m = p ? parseFloat(p.current) : NaN;
      if (!isNaN(m) && m > 0) {
        ctx.strokeStyle = '#9aa0a6'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
        ctx.strokeRect(X(m), Y(m), (1 - 2 * m) * dw, (1 - 2 * m) * dh);         // the frame-edge band
        if (scene.mask) {                                                        // the mask-rim band
          const [mcx, mcy, mr] = scene.mask;
          ctx.beginPath(); ctx.arc(X(mcx), Y(mcy), mr * (1 - m) * dw, 0, 7); ctx.stroke();
        }
        ctx.setLineDash([]);
      }
    }
    if (showClusters && scene.clusters) {                            // cluster circles + ID badges (a table now)
      for (const c of scene.clusters) {
        const col = CLUSTER_COLORS[(c.id - 1) % CLUSTER_COLORS.length];
        const [ccx, ccy, cr] = c.circle;               // min-enclosing circle (own centre + radius)
        const cx = X(ccx), cy = Y(ccy), r = cr * dw;
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.stroke();
        ctx.fillStyle = col; ctx.font = 'bold 13px monospace'; ctx.textAlign = 'center';
        ctx.fillText('C' + c.id, cx, cy - r - 5);      // cluster ID badge above the circle
        ctx.textAlign = 'left';
      }
    }
    for (const o of scene.objects) {
      if (o.grp) continue;                                        // compounds live in the FEATURES view; the
      //                                                             process canvas keeps the real objects +
      //                                                             the dashed groups overlay
      if (o.confirmed === false) continue;                        // draw ONLY confirmed objects (the debug overlay)
      if (!showGone && o.state === 'HIDDEN') continue;            // 'gone' unticked → hide lifted/covered husks
      const col = STATE_COLOR[o.state] || '#fff';
      if (showOutlines && o.contour && o.contour.length > 1) {
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
        o.contour.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.closePath(); ctx.stroke();
      }
      if (showBbox && o.bbox) {                       // axis-aligned bounds of the last observed blob
        ctx.strokeStyle = col; ctx.lineWidth = 1;
        ctx.strokeRect(X(o.bbox[0]), Y(o.bbox[1]), o.bbox[2] * dw, o.bbox[3] * dh);
      }
      if (showRot && o.theta != null) {               // orientation ARROW: θ=0 points UP, clockwise-positive —
        const cx = X(o.x), cy = Y(o.y), L = Math.max(12, 0.038 * dw);   // reads like a fiducial / clock hand
        const dx = Math.sin(o.theta), dy = -Math.cos(o.theta);          // 0 → up in the y-down frame
        const tx = cx + L * dx, ty = cy + L * dy, hb = 0.32 * L;        // tip + arrow-head size
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
        ctx.moveTo(cx, cy); ctx.lineTo(tx, ty);                         // the shaft
        ctx.moveTo(tx, ty); ctx.lineTo(tx - hb * (dx * 0.87 - dy * 0.5), ty - hb * (dy * 0.87 + dx * 0.5));
        ctx.moveTo(tx, ty); ctx.lineTo(tx - hb * (dx * 0.87 + dy * 0.5), ty - hb * (dy * 0.87 - dx * 0.5));
        ctx.stroke();                                                   // two head strokes, ±30° off the shaft
      }
      if (showIds) idBadge(ctx, '#' + (o.label != null ? o.label : o.id), X(o.x) + 8, Y(o.y) - 8, col);   // small recycled label
    }
    // COMPOUND super-objects (grouper) — only present when the scene is over [grouper].max_objects. Each is
    // the merged EFD blend of its fused members: dashed hull + translucent fill + connectors to the members
    // it swallowed + a G<id>·<n> badge. A distinct purple so it never reads as an object/cluster outline.
    if (showGroups && scene.groups) {
      const G = '#c98bff';
      const byId = {}; for (const o of scene.objects) byId[o.id] = o;
      for (const g of scene.groups) {
        if (g.n < 2) continue;                        // a singleton "group" IS its one object — the plain
        //                                               object rendering already shows it; no hull, no badge
        const gx = X(g.x), gy = Y(g.y);
        ctx.setLineDash([3, 3]); ctx.strokeStyle = G; ctx.lineWidth = 1;          // connectors to fused members
        for (const mid of g.members) { const m = byId[mid]; if (!m) continue; ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(X(m.x), Y(m.y)); ctx.stroke(); }
        if (g.contour && g.contour.length > 1) {                                   // the merged compound shape
          ctx.beginPath(); g.contour.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
          ctx.closePath(); ctx.setLineDash([8, 4]); ctx.lineWidth = 3; ctx.strokeStyle = G; ctx.stroke();
          ctx.fillStyle = 'rgba(201,139,255,0.12)'; ctx.fill();
        }
        ctx.setLineDash([]);
        idBadge(ctx, 'G' + g.id + '·' + g.n, gx + 8, gy - 8, G);                  // compound id · member count
      }
    }
  }

  // an ID BADGE matching compose.id_tag: a filled state-coloured chip, black border, white text on a
  // black halo — reads on any background. Used for object ids on the debug overlay.
  function idBadge(ctx, txt, x, y, col) {
    ctx.font = 'bold 13px monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    const pad = 4, tw = ctx.measureText(txt).width, th = 11;
    const x0 = x - pad, y0 = y - th - pad, w = tw + 2 * pad, h = th + 2 * pad;
    ctx.fillStyle = col; ctx.fillRect(x0, y0, w, h);                // state-colour chip
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5; ctx.strokeRect(x0, y0, w, h);   // black border
    ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(txt, x, y);        // black halo
    ctx.fillStyle = '#fff'; ctx.fillText(txt, x, y);               // white text
  }

  // ── zoom / pan / snapshot — inspect the views closer ──────────────────
  const zoom = { z: 1, x: 0, y: 0 };
  function applyZoom() {
    const t = `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.z})`;
    $('video').style.transform = t; $('overlay').style.transform = t;   // SAME transform → overlay stays aligned
    $('stage').classList.toggle('zoomed', zoom.z !== 1);
  }
  function setZoom(z) { zoom.z = Math.max(1, Math.min(8, z)); if (zoom.z === 1) { zoom.x = 0; zoom.y = 0; } applyZoom(); }
  function wireViewTools() {
    document.querySelectorAll('[data-zoom]').forEach((b) => b.onclick = () => {
      const a = b.dataset.zoom; setZoom(a === 'in' ? zoom.z * 1.3 : a === 'out' ? zoom.z / 1.3 : 1);  // fit = reset
    });
    let pan = null;                                                     // drag to pan while zoomed
    $('stage').addEventListener('mousedown', (e) => { if (zoom.z > 1 && !featuresActive()) { pan = { x: e.clientX - zoom.x, y: e.clientY - zoom.y }; e.preventDefault(); } });
    window.addEventListener('mousemove', (e) => { if (pan) { zoom.x = e.clientX - pan.x; zoom.y = e.clientY - pan.y; applyZoom(); } });
    window.addEventListener('mouseup', () => { pan = null; });
    $('stage').addEventListener('wheel', (e) => { if (!featuresActive()) { e.preventDefault(); setZoom(zoom.z * (e.deltaY < 0 ? 1.15 : 1 / 1.15)); } }, { passive: false });
    $('snap-btn').onclick = snapshot;
  }
  function snapshot() {                                                 // composite frame + overlay → PNG download
    const img = $('video'), ov = $('overlay');
    const c = document.createElement('canvas'); c.width = ov.width || 1280; c.height = ov.height || 720;
    const g = c.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, c.width, c.height);
    const natW = img.naturalWidth, natH = img.naturalHeight;
    if (natW && natH) {                                                 // the video at its object-fit:contain rect
      const s = Math.min(c.width / natW, c.height / natH), dw = natW * s, dh = natH * s;
      try { g.drawImage(img, (c.width - dw) / 2, (c.height - dh) / 2, dw, dh); } catch (e) { console.warn('snap video', e); }
    }
    g.drawImage(ov, 0, 0);                                              // the overlay on top (already frame-mapped)
    let url; try { url = c.toDataURL('image/png'); } catch (e) { flash($('snap-btn'), '⧉ blocked'); return; }  // taint
    const a = document.createElement('a'); a.href = url; a.download = `readymade-frame-${scene.frame || 0}.png`; a.click();
    flash($('snap-btn'), '⧉ saved');
  }

  // ── resizable panel ───────────────────────────────────────────────────
  function wireResize() {
    const rez = $('resizer'), panel = $('panel');
    let dragging = false;
    rez.addEventListener('mousedown', (e) => { dragging = true; rez.classList.add('drag'); document.body.style.cursor = 'col-resize'; e.preventDefault(); });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const wpx = Math.max(260, Math.min(window.innerWidth - e.clientX, window.innerWidth - 200));
      panel.style.flexBasis = wpx + 'px'; draw();
    });
    window.addEventListener('mouseup', () => { if (dragging) { dragging = false; rez.classList.remove('drag'); document.body.style.cursor = ''; } });
  }

  // ── boot ─────────────────────────────────────────────────────────────
  async function init() {
    try {
      const info = await (await fetch('/info')).json();
      PREFIX = info.prefix || '/rmt';
      MJPEG_PORT = info.mjpeg_port || 8080;
      $('video').crossOrigin = 'anonymous';            // mjpeg sends CORS → snapshot can draw the frame to a canvas
      $('video').onload = draw;                        // dims known → redraw so the overlay snaps onto it
      $('video').onerror = () => setTimeout(mjpegConnect, 1500);   // stream not up yet (starting/restarting) → retry
      mjpegConnect();
    } catch (e) { console.warn('/info failed', e); }
    wireTransport(); wireActions(); wireResize(); wireViewTools();
    connectSSE();
    try { await query(); } catch (e) { console.warn('query failed', e); setDot('err', 'query failed'); }
    window.addEventListener('resize', draw);
  }
  init();
})();
