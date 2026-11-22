// Интерактив «между / внутри» на страницах методов (перенесён из образца one_way_anova.html).
// SVG в исходном положении рисует генератор (tools/method_widgets.py); скрипт только двигает
// его элементы ползунками. Данные — из data-* у [data-widget="groups-split"]; встроенного кода
// нет (CSP script-src 'self'). P — регуляризованная неполная бета, как в образце. В Node
// (тест) экспортирует расчёт и раскладку, ничего не рисуя.
(function () {
  "use strict";
  var G = { W: 560, H: 330, L: 50, R: 20, T: 20, B: 40 };
  var DOT_STEP = 16, DOT_OFFSET = 40, MEAN_HALF = 55;

  // Регуляризованная неполная бета-функция (Numerical Recipes, betacf)
  function lgamma(x) {
    var c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    var y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t);
    var s = 1.000000000190015; for (var i = 0; i < c.length; i++) s += c[i] / ++y;
    return -t + Math.log(2.5066282746310005 * s / x);
  }
  function betacf(a, b, x) {
    var qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    d = 1 / (Math.abs(d) < 1e-30 ? 1e-30 : d); var h = d;
    for (var m = 1; m <= 200; m++) {
      var m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; d = 1 / (Math.abs(d) < 1e-30 ? 1e-30 : d);
      c = 1 + aa / c; c = Math.abs(c) < 1e-30 ? 1e-30 : c; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; d = 1 / (Math.abs(d) < 1e-30 ? 1e-30 : d);
      c = 1 + aa / c; c = Math.abs(c) < 1e-30 ? 1e-30 : c;
      var del = d * c; h *= del; if (Math.abs(del - 1) < 3e-12) break;
    }
    return h;
  }
  function ibeta(a, b, x) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    var bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function fP(F, d1, d2) { return ibeta(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)); }
  function fmtP(p) { return p < 0.001 ? "< 0.001" : p.toFixed(3); }
  function mean(a) { return a.reduce(function (s, v) { return s + v; }, 0) / a.length; }

  /** SS, MS, F и P при сдвиге средних (kb) и разбросе внутри групп (kw). */
  function stats(groups, df, kb, kw) {
    var all = []; groups.forEach(function (g) { all = all.concat(g.y); });
    var grand = mean(all), means0 = groups.map(function (g) { return mean(g.y); });
    var means = means0.map(function (m) { return grand + kb * (m - grand); });
    var ys = groups.map(function (g, i) { return g.y.map(function (v) { return means[i] + kw * (v - means0[i]); }); });
    var ssb = 0, ssw = 0;
    ys.forEach(function (a, i) {
      ssb += a.length * Math.pow(means[i] - grand, 2);
      a.forEach(function (v) { ssw += Math.pow(v - means[i], 2); });
    });
    var msb = ssb / df[0], msw = ssw / df[1], F = msb / msw;
    return { grand: grand, means: means, ys: ys, ssb: ssb, ssw: ssw, msb: msb, msw: msw, F: F, p: fP(F, df[0], df[1]) };
  }

  /** Координаты: общая средняя, средние групп, точки, отрезки у выделенной точки. */
  function layout(groups, df, yRange, hl, kb, kw) {
    var s = stats(groups, df, kb, kw), yMin = yRange[0], yMax = yRange[1];
    var sy = function (v) { return G.T + (yMax - Math.min(yMax, Math.max(yMin, v))) / (yMax - yMin) * (G.H - G.T - G.B); };
    var colX = function (i) { return G.L + (i + 0.5) * (G.W - G.L - G.R) / groups.length; };
    var hx = colX(hl[0]) - DOT_OFFSET + hl[1] * DOT_STEP;
    return {
      stats: s, grandY: sy(s.grand),
      means: s.means.map(function (m, i) { return { x1: colX(i) - MEAN_HALF, x2: colX(i) + MEAN_HALF, y: sy(m) }; }),
      dots: s.ys.map(function (a, i) { return a.map(function (v, j) { return { cx: colX(i) - DOT_OFFSET + j * DOT_STEP, cy: sy(v) }; }); }),
      segB: { x: hx + 12, y1: sy(s.grand), y2: sy(s.means[hl[0]]) },
      segW: { x: hx, y1: sy(s.means[hl[0]]), y2: sy(s.ys[hl[0]][hl[1]]) },
    };
  }

  var api = { stats: stats, layout: layout, fP: fP, fmtP: fmtP, geometry: G };
  if (typeof module === "object" && module.exports) { module.exports = api; return; }

  function set(el, attrs) { Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); }); }

  function init(root) {
    var groups = JSON.parse(root.dataset.groups), df = JSON.parse(root.dataset.df);
    var yRange = JSON.parse(root.dataset.y), hl = JSON.parse(root.dataset.highlight);
    var q = function (sel) { return root.querySelector(sel); };
    var kb = q(".gs-kb"), kw = q(".gs-kw");
    if (!kb || !kw) return;
    q(".gs-sliders").hidden = false; q(".gs-note").hidden = false;
    function draw() {
      var lay = layout(groups, df, yRange, hl, +kb.value, +kw.value), s = lay.stats;
      set(q(".gs-grand"), { y1: lay.grandY, y2: lay.grandY });
      root.querySelectorAll(".gs-mean").forEach(function (el) {
        var m = lay.means[+el.dataset.g]; set(el, { x1: m.x1, x2: m.x2, y1: m.y, y2: m.y });
      });
      root.querySelectorAll(".gs-dot").forEach(function (el) {
        var d = lay.dots[+el.dataset.g][+el.dataset.j]; set(el, { cx: d.cx, cy: d.cy });
      });
      set(q(".gs-seg-b"), { x1: lay.segB.x, x2: lay.segB.x, y1: lay.segB.y1, y2: lay.segB.y2 });
      set(q(".gs-seg-w"), { x1: lay.segW.x, x2: lay.segW.x, y1: lay.segW.y1, y2: lay.segW.y2 });
      q(".gs-ssb").textContent = s.ssb.toFixed(2);
      q(".gs-ssw").textContent = s.ssw.toFixed(2);
      q(".gs-f").textContent = s.F > 999 ? "> 999" : s.F.toFixed(1);
      q(".gs-p").textContent = fmtP(s.p);
      q(".gs-p").dataset.value = String(s.p);
      var top = Math.max(s.msb, s.msw, 1e-9);
      q(".gs-bar-b").style.width = (100 * s.msb / top) + "%";
      q(".gs-bar-w").style.width = Math.max(1, 100 * s.msw / top) + "%";
    }
    kb.addEventListener("input", draw);
    kw.addEventListener("input", draw);
    draw();
  }

  document.querySelectorAll('[data-widget="groups-split"]').forEach(init);
})();
