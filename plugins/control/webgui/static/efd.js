// ReadyMade Tracker EFD + feature inspector — a self-contained component embedded in the MAIN view's "features" tab.
// TRUTH FROM THE WIRE, nothing derived locally: the outline IS the streamed contour (the tracker's own
// inverse-EFD reconstruction — wire_contour), the spectrum IS the /rmt/features/efd annotation's signed-bin
// triples, and the scalars ARE the feature annotations. Dragging the efd_detail knob sends /rmt/set; the
// tracker recomputes on its next tick and all three panels refresh from the SSE stream — the inspector
// shows exactly what every OSC client receives, never a client-side re-derivation.
'use strict';
window.EFD = (() => {
  const STATE_COLOR = { RESTING: '#00dc00', MOVED: '#ff8c00', SETTLING: '#00dcdc', HIDDEN: '#a0a0a0' };
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // ── the wire spectrum: /rmt/features/efd = flat (signed_bin, amplitude, phase) triples — one per
  // SURVIVING ± component of the tracker's per-component detail cut, amplitude ÷ the loudest single
  // bin (so the loudest = 1.0 = 0 dB), phase in radians. Parse → {pos/neg: Map(n→{amp,ph}), …}.
  const SELECT_FLOOR_DB = 100;   // == lib/efd._SELECT_FLOOR_DB (where the detail knob's cut line sits)
  function parseEfd(vec) {
    if (!vec || vec.length < 3) return null;
    const pos = new Map(), neg = new Map();
    let maxN = 1;
    for (let i = 0; i + 2 < vec.length; i += 3) {
      const b = Math.round(vec[i]);
      (b >= 0 ? pos : neg).set(Math.abs(b), { amp: vec[i + 1], ph: vec[i + 2] });
      maxN = Math.max(maxN, Math.abs(b));
    }
    return { pos, neg, maxN, count: pos.size + neg.size };
  }

  // ── canvas helpers ────────────────────────────────────────────────────
  function fit(c) {
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    return { g, w, h };
  }
  function bboxTx(pts, w, h, pad) {                     // fit a point set into a canvas box, preserving aspect
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const [x, y] of pts) { minx = Math.min(minx, x); miny = Math.min(miny, y); maxx = Math.max(maxx, x); maxy = Math.max(maxy, y); }
    const bw = maxx - minx || 1, bh = maxy - miny || 1;
    const s = Math.min((w - 2 * pad) / bw, (h - 2 * pad) / bh);
    const ox = (w - s * bw) / 2 - s * minx, oy = (h - s * bh) / 2 - s * miny;
    return ([x, y]) => [ox + s * x, oy + s * y];
  }
  function stroke(g, pts, tx, style, width) {
    g.beginPath();
    pts.forEach((p, i) => { const [x, y] = tx(p); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.closePath(); g.strokeStyle = style; g.lineWidth = width; g.stroke();
  }

  function drawOutline(canvas, pts) {
    // the object's WIRE outline — the tracker's inverse-EFD reconstruction at the current detail,
    // exactly what /rmt/contour and /tuio2/ocg carry. Nothing is rebuilt locally.
    const { g, w, h } = fit(canvas);
    if (!pts.length) return;
    stroke(g, pts, bboxTx(pts, w, h, 24), css('--warn'), 2);
  }

  const SPEC = { pos: '#e0503c', neg: '#4a90e0',            // +n red ▲ · −n blue ▼ — BRIGHT = on the wire
                 posOff: '#5a2d27', negOff: '#27384f' };    // DIM = cut by the detail knob (efd_full, SSE-only)
  const LOGFLOOR = 0.00001;                                // −100 dB — matches SELECT_FLOOR_DB so the cut lines
  //                                                          stay on-axis across the whole detail knob
  // The spectrum: one ± pair of bars per harmonic. BRIGHT bars come from the efd annotation's triples —
  // they SURVIVED the tracker's per-component detail cut and are exactly what the wire carries. DIM bars
  // come from efd_full (the complete spectrum, a webgui-SSE-only diagnostic) — the bins the cut REJECTED,
  // so the operator sees what the knob is discarding. Both share one scale: amplitude ÷ the loudest bin
  // (1.0 = 0 dB). The dashed ± lines show where the current detail slider puts the cut.
  function drawSpectrum(canvas, t, full, bins, detail, log) {
    const { g, w, h } = fit(canvas);
    const fullN = full && full[0] ? full[0].length : 1;
    const nBins = Math.max(bins, t ? t.maxN : 1, fullN);
    const padX = 24, padTop = 8, padBot = 15, bw = (w - 2 * padX) / nBins;
    const mid = padTop + (h - padTop - padBot) / 2, half = (h - padTop - padBot) / 2 - 2;
    const LF = Math.log10(LOGFLOOR);
    const hgt = (lv) => log ? Math.max(0, Math.min(1, (Math.log10(lv + 1e-9) - LF) / -LF)) : Math.max(0, Math.min(1, lv));
    if (full) {                                  // the COMPLETE spectrum, dim — rejected bins made visible
      for (let i = 0; i < fullN; i++) {
        const x = padX + i * bw + 1, bwe = Math.max(1, bw - 2);
        const p = full[0][i] || 0, m = full[1][i] || 0;
        if (p > 0) { g.fillStyle = SPEC.posOff; g.fillRect(x, mid - hgt(p) * half, bwe, hgt(p) * half); }
        if (m > 0) { g.fillStyle = SPEC.negOff; g.fillRect(x, mid, bwe, hgt(m) * half); }
      }
    }
    if (t) {                                     // the wire's survivors, bright, over their dim twins
      for (let i = 0; i < nBins; i++) {
        const n = i + 1, x = padX + i * bw + 1, bwe = Math.max(1, bw - 2);
        const p = t.pos.get(n), m = t.neg.get(n);
        if (p) { g.fillStyle = SPEC.pos; g.fillRect(x, mid - hgt(p.amp) * half, bwe, hgt(p.amp) * half); }
        if (m) { g.fillStyle = SPEC.neg; g.fillRect(x, mid, bwe, hgt(m.amp) * half); }
      }
    }
    const cy = hgt(Math.pow(10, -Math.min(1, Math.max(0, detail)) * SELECT_FLOOR_DB / 20)) * half;
    g.setLineDash([5, 3]); g.lineWidth = 1; g.strokeStyle = css('--warn');   // the detail cut, mirrored ±
    g.beginPath(); g.moveTo(padX, mid - cy); g.lineTo(w - padX, mid - cy);
    g.moveTo(padX, mid + cy); g.lineTo(w - padX, mid + cy); g.stroke(); g.setLineDash([]);
    g.strokeStyle = css('--border'); g.lineWidth = 1; g.beginPath(); g.moveTo(padX, mid); g.lineTo(w - padX, mid); g.stroke();   // zero axis
    const step = bw >= 20 ? 1 : Math.ceil(20 / bw);        // thin the labels out so they never overlap
    g.font = '9px monospace'; g.textAlign = 'center'; g.fillStyle = css('--dim');
    for (let i = 0; i < nBins; i += step) g.fillText(String(i + 1), padX + i * bw + bw / 2, h - 4);
  }

  function drawScalars(box, features) {
    const entries = Object.entries(features || {}).filter(([k, v]) => k !== 'efd' && Array.isArray(v) && v.length === 1);
    if (!entries.length) { box.innerHTML = '<div class="muted">no scalar features on the wire (enable them in [features])</div>'; return; }
    box.innerHTML = '';
    for (const [name, [v]] of entries) {
      const row = document.createElement('div'); row.className = 'scalar';
      const lab = document.createElement('span'); lab.className = 'sn'; lab.textContent = name;
      const bar = document.createElement('span'); bar.className = 'sb';
      const fill = document.createElement('i'); fill.style.width = (Math.max(0, Math.min(1, v)) * 100) + '%';
      const num = document.createElement('span'); num.className = 'sv'; num.textContent = v.toFixed(3);
      bar.appendChild(fill); row.append(lab, bar, num); box.appendChild(row);
    }
  }
  function drawChip(canvas, contour, color) {          // a mini silhouette of the object's actual shape
    const { g, w, h } = fit(canvas);
    if (!contour || contour.length < 3) return;
    const tx = bboxTx(contour, w, h, 3);
    g.beginPath();
    contour.forEach((p, i) => { const [x, y] = tx(p); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.closePath();
    g.fillStyle = color + '33'; g.fill();               // faint tint + a crisp outline
    g.strokeStyle = color; g.lineWidth = 1.5; g.stroke();
  }

  // ── the component: build into `root`, drive with update()/setParams() ──
  const TEMPLATE = `
    <div class="insp-band" data-band></div>
    <div class="insp-grid">
      <section class="insp-card"><h2>wire outline <small data-note></small></h2><canvas data-recon></canvas></section>
      <section class="insp-card"><h2>wire spectrum <small>/rmt/features/efd · |c₊ₙ| ▲ red · |c₋ₙ| ▼ blue (log dB) · bright = on the wire · dim = cut by detail · dashed = the cut line</small></h2><canvas data-spectrum></canvas></section>
      <section class="insp-card insp-scalars"><h2>scalar features</h2><div class="scalars" data-scalars></div></section>
    </div>
    <div class="insp-ctl">
      <label>efd_bins <input type="range" data-bins min="1" max="64" step="1" value="12"><span data-bins-v class="val">12</span></label>
      <label>efd_detail <input type="range" data-detail min="0" max="1" step="0.01" value="0.5"><span data-detail-v class="val">0.50</span></label>
      <label class="chk"><input type="checkbox" data-log checked> log</label>
      <span class="hint">knobs set [efd] on the tracker · bins stages a restart · detail is live (refreshes on the next tick)</span>
    </div>`;

  function createInspector(root, opts) {
    opts = opts || {};
    root.innerHTML = TEMPLATE;
    const q = (s) => root.querySelector(s);
    const band = q('[data-band]'), recon = q('[data-recon]'), spectrum = q('[data-spectrum]'),
          scalars = q('[data-scalars]'), note = q('[data-note]'),
          binsEl = q('[data-bins]'), detailEl = q('[data-detail]'), binsV = q('[data-bins-v]'), detailV = q('[data-detail-v]'),
          logEl = q('[data-log]');
    let objs = [], sel = null, bins = 12, detail = 0.5, log = true;   // log (dB) default — the reference spectrum
    let fdim = null;                          // [fw, fh] from the SSE head — un-normalize contours to PIXELS
    const toPx = (c) => (fdim && c) ? c.map((p) => [p[0] * fdim[0], p[1] * fdim[1]]) : c;
    // Caches so a live features view is CHEAP: the SSE pushes ~30 snapshots/s, but a resting object's
    // contour + annotations only change on a re-capture or a detail change. Signatures gate the redraws.
    let renderSig = null, bandSig = null;      // last-drawn signatures → skip redundant DOM/canvas churn
    let chips = new Map();                      // id → { cv, sig }: refresh a moved silhouette without a rebuild
    band.addEventListener('wheel', (e) => {              // vertical wheel → horizontal scroll (many objects)
      if (e.deltaY) { band.scrollLeft += e.deltaY; e.preventDefault(); }
    }, { passive: false });

    const contourSig = (o) => {                  // cheap contour identity: length + first/mid/last sampled points
      const c = o.contour;
      if (!c || c.length < 3) return o.id + ':none';
      const m = c[c.length >> 1], z = c[c.length - 1];
      return o.id + ':' + c.length + ':' + c[0][0] + ',' + c[0][1] + ':' + m[0] + ',' + m[1] + ':' + z[0] + ',' + z[1];
    };
    const featsSig = (feats) => {                // cheap VALUE signature — a detail change re-derives the
      if (!feats) return '-';                    // scalars/efd on the same contour; the values must repaint
      let s = 0, n = 0;
      for (const k in feats) { const v = feats[k]; if (Array.isArray(v) && v.length) { s += v[0] * 31 + v.length; n++; } }
      return n + ':' + s.toFixed(4);
    };

    function renderBand() {
      let sig = 'sel' + sel + ';';
      for (const o of objs) sig += o.id + o.state + '|';
      if (sig === bandSig) {                     // same set / states / selection → only redraw silhouettes that moved
        for (const o of objs) {
          const e = chips.get(o.id), cs = contourSig(o);
          if (e && cs !== e.sig) { drawChip(e.cv, toPx(o.contour), STATE_COLOR[o.state] || '#888'); e.sig = cs; }
        }
        return;
      }
      bandSig = sig; chips = new Map(); band.innerHTML = '';
      if (!objs.length) { band.innerHTML = '<span class="muted">no objects</span>'; return; }
      for (const o of objs) {
        const chip = document.createElement('button');
        chip.className = 'chip' + (o.id === sel ? ' on' : '');
        const col = STATE_COLOR[o.state] || '#888'; chip.style.setProperty('--c', col);
        const cv = document.createElement('canvas'); cv.className = 'chip-sil'; cv.width = 46; cv.height = 34;
        const meta = document.createElement('span'); meta.className = 'chip-meta';
        meta.innerHTML = `<span class="cid">#${o.id}</span><span class="cst">${o.state}</span>`;
        chip.append(cv, meta);
        chip.onclick = () => { sel = o.id; renderBand(); render(); };
        band.appendChild(chip);
        drawChip(cv, toPx(o.contour), col);             // the mini silhouette (after it's in the DOM → sized)
        chips.set(o.id, { cv, sig: contourSig(o) });
      }
    }
    function render() {
      if (sel === null && objs.length) sel = objs[0].id;
      const o = objs.find((x) => x.id === sel);
      if (!o || !o.contour || o.contour.length < 3) {
        if (renderSig !== 'none') { drawOutline(recon, []); drawSpectrum(spectrum, null, null, bins, detail, log); drawScalars(scalars, null); note.textContent = ''; renderSig = 'none'; }
        return;
      }
      const feats = o.ann && o.ann.features, csig = contourSig(o);
      const full = o.efd_full || null;           // the complete ± spectrum (SSE diagnostic) — may be absent
      const sig = csig + '|' + bins + '|' + detail + '|' + log + '|' + featsSig(feats);
      if (sig === renderSig) return;             // nothing that affects the view changed → skip the redraw
      renderSig = sig;
      drawOutline(recon, toPx(o.contour));       // the tracker's reconstruction — nothing rebuilt locally
      const t = parseEfd(feats && feats.efd);
      drawSpectrum(spectrum, t, full, bins, detail, log);
      drawScalars(scalars, feats);
      const total = full && full[0] ? 2 * full[0].length : null;
      note.textContent = t
        ? `${t.count}${total ? ' of ' + total : ''} components on the wire · cut −${Math.round(Math.min(1, Math.max(0, detail)) * 100)} dB`
        : 'efd not on the wire — add it to [features] enable';
    }
    // Skip all paint work when the inspector is HIDDEN (zero-size): leaving the "features" view sets
    // #inspector display:none, which fires the ResizeObserver with a 0×0 box — repainting into detached
    // zero canvases (and retriggering the observer) janks the browser.
    const paint = () => { if (!root.clientWidth || !root.clientHeight) return; renderBand(); render(); };
    const forceRepaint = () => { renderSig = null; bandSig = null; paint(); };   // resize/collapse → re-fit + redraw
    binsEl.oninput = () => { bins = +binsEl.value; binsV.textContent = bins; render(); if (opts.onParam) opts.onParam('bins', bins); };
    detailEl.oninput = () => { detail = +detailEl.value; detailV.textContent = detail.toFixed(2); render(); if (opts.onParam) opts.onParam('detail', detail); };
    logEl.onchange = () => { log = logEl.checked; render(); };   // log-scale the spectrum (see the small bins)
    if (window.ResizeObserver) new ResizeObserver(forceRepaint).observe(root);   // re-fit canvases on panel resize/collapse

    return {
      // Synchronous paint (no rAF — it doesn't fire in headless / backgrounded contexts). Cheap by design:
      // the signature gates above make a no-change frame just a couple of string compares.
      update(newObjs, dims) { objs = newObjs || []; if (dims) fdim = dims; paint(); },
      setParams(b, d) {
        if (b != null && !isNaN(b)) { bins = b; binsEl.value = b; binsV.textContent = b; }
        if (d != null && !isNaN(d)) { detail = d; detailEl.value = d; detailV.textContent = d.toFixed(2); }
        render();
      },
    };
  }

  return { createInspector, parseEfd, STATE_COLOR };
})();
