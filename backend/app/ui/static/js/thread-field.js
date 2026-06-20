/* Ariadne — fluid animated thread field.
 * Draws flowing bronze "threads of Ariadne" on a canvas behind the app.
 * Performance-aware: caps device pixel ratio, pauses when the tab is hidden,
 * and fully respects prefers-reduced-motion. No dependencies. */
(() => {
  const canvas = document.getElementById("threadCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  const reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Theme-aware palette pulled live from CSS variables.
  function palette() {
    const css = getComputedStyle(document.documentElement);
    const dark = document.documentElement.dataset.theme === "dark";
    const grab = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
    return {
      bronze: grab("--bronze", "#b98632"),
      bronze2: grab("--bronze-2", "#d6a756"),
      olive: grab("--olive", "#595c23"),
      dark,
    };
  }

  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(W * dpr));
    canvas.height = Math.max(1, Math.floor(H * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Each thread is a horizontal-ish silk strand made of a travelling sine sum.
  let threads = [];
  function buildThreads() {
    const count = W < 760 ? 5 : W < 1300 ? 7 : 9;
    threads = [];
    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      threads.push({
        baseY: H * (0.10 + t * 0.82) + (Math.random() - 0.5) * 38,
        amp: 26 + Math.random() * 60,           // vertical sway
        amp2: 10 + Math.random() * 26,           // secondary ripple
        k1: 0.0016 + Math.random() * 0.0016,     // wavelength 1
        k2: 0.0034 + Math.random() * 0.0030,     // wavelength 2
        speed: 0.10 + Math.random() * 0.16,      // drift speed
        speed2: 0.05 + Math.random() * 0.12,
        phase: Math.random() * Math.PI * 2,
        phase2: Math.random() * Math.PI * 2,
        width: 0.7 + Math.random() * 1.4,
        glow: i % 3 === 0,                        // a few brighter "guide" threads
        alpha: 0.10 + Math.random() * 0.16,
      });
    }
  }

  function hexA(hex, a) {
    const h = hex.replace("#", "");
    const n = h.length === 3
      ? h.split("").map(c => c + c).join("")
      : h.padEnd(6, "0").slice(0, 6);
    const r = parseInt(n.slice(0, 2), 16);
    const g = parseInt(n.slice(2, 4), 16);
    const b = parseInt(n.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  let pal = palette();
  let running = true;
  let t0 = performance.now();

  function drawThread(th, time, pal) {
    const step = 14;
    ctx.beginPath();
    let first = true;
    for (let x = -step; x <= W + step; x += step) {
      const y = th.baseY
        + Math.sin(x * th.k1 + time * th.speed + th.phase) * th.amp
        + Math.sin(x * th.k2 - time * th.speed2 + th.phase2) * th.amp2;
      if (first) { ctx.moveTo(x, y); first = false; }
      else ctx.lineTo(x, y);
    }
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    const c1 = pal.dark ? pal.bronze2 : pal.bronze;
    const c2 = pal.olive;
    grad.addColorStop(0, hexA(c1, 0));
    grad.addColorStop(0.22, hexA(c1, th.alpha));
    grad.addColorStop(0.5, hexA(c2, th.alpha * 0.9));
    grad.addColorStop(0.78, hexA(c1, th.alpha));
    grad.addColorStop(1, hexA(c1, 0));
    ctx.strokeStyle = grad;
    ctx.lineWidth = th.width;
    ctx.lineCap = "round";
    if (th.glow) {
      ctx.shadowColor = hexA(c1, pal.dark ? 0.5 : 0.32);
      ctx.shadowBlur = 12;
    } else {
      ctx.shadowBlur = 0;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // travelling "knot" of light along the guide threads
    if (th.glow) {
      const px = ((time * 26 * th.speed + th.phase * 120) % (W + 200)) - 100;
      const py = th.baseY
        + Math.sin(px * th.k1 + time * th.speed + th.phase) * th.amp
        + Math.sin(px * th.k2 - time * th.speed2 + th.phase2) * th.amp2;
      const r = 3.2;
      const g = ctx.createRadialGradient(px, py, 0, px, py, r * 5);
      g.addColorStop(0, hexA(c1, 0.7));
      g.addColorStop(1, hexA(c1, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(px, py, r * 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function frame(now) {
    if (!running) return;
    const time = (now - t0) / 1000;
    ctx.clearRect(0, 0, W, H);
    for (const th of threads) drawThread(th, time, pal);
    requestAnimationFrame(frame);
  }

  function renderStatic() {
    ctx.clearRect(0, 0, W, H);
    for (const th of threads) drawThread(th, 8, pal);
  }

  function start() {
    resize();
    buildThreads();
    pal = palette();
    if (reduceMotion) { renderStatic(); return; }
    running = true;
    t0 = performance.now();
    requestAnimationFrame(frame);
  }

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      resize();
      buildThreads();
      if (reduceMotion) renderStatic();
    }, 160);
  });

  document.addEventListener("visibilitychange", () => {
    if (reduceMotion) return;
    if (document.hidden) {
      running = false;
    } else if (!running) {
      running = true;
      requestAnimationFrame(frame);
    }
  });

  // React to theme changes (settings drawer toggling light/dark).
  const themeObserver = new MutationObserver(() => {
    pal = palette();
    if (reduceMotion) renderStatic();
  });
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  start();
})();
