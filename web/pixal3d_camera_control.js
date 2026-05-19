var N = Object.defineProperty;
var H = (n, e, a) => e in n ? N(n, e, { enumerable: !0, configurable: !0, writable: !0, value: a }) : n[e] = a;
var m = (n, e, a) => H(n, typeof e != "symbol" ? e + "" : e, a);
import { app as P } from "/scripts/app.js";
const U = "Pixal3DCameraControl", z = "pixal3d-camera-control-styles", x = 430, L = 560, A = 650, M = 360, C = 520, D = L / x, O = 0.64;
function h(n, e, a) {
  return Math.max(e, Math.min(a, Number(n) || 0));
}
function F(n, e) {
  var a;
  return (a = n.widgets) == null ? void 0 : a.find((s) => s.name === e);
}
function I(n, e, a) {
  var i, r, o;
  const s = F(n, e);
  if (!s) return;
  const t = s.value;
  s.value = a, t !== a && ((i = s.callback) == null || i.call(s, a), (o = (r = P.graph) == null ? void 0 : r.setDirtyCanvas) == null || o.call(r, !0, !0));
}
function R(n) {
  var a;
  const e = (a = P.graph) == null ? void 0 : a.links;
  return !e || n == null ? null : Array.isArray(e) ? e.find((s) => s && s.id === n) || e[n] || null : e[n] || null;
}
function V(n, e) {
  var t, i, r;
  const a = (t = n.inputs) == null ? void 0 : t.find((o) => o.name === e), s = R(a == null ? void 0 : a.link);
  return s && ((r = (i = P.graph) == null ? void 0 : i.getNodeById) == null ? void 0 : r.call(i, s.origin_id)) || null;
}
function q(n) {
  if (!n) return "";
  let e = "", a = "input", s = "";
  if (typeof n == "string" ? e = n : typeof n == "object" && (e = n.filename || n.name || n.image || "", a = n.type || a, s = n.subfolder || ""), !e) return "";
  const t = new URLSearchParams();
  return t.set("filename", e), t.set("type", a), s && t.set("subfolder", s), `/view?${t.toString()}`;
}
function j(n) {
  var i, r;
  const e = V(n, "image"), a = (i = e == null ? void 0 : e.imgs) == null ? void 0 : i[0], s = (a == null ? void 0 : a.currentSrc) || (a == null ? void 0 : a.src);
  if (s) return s;
  const t = (r = e == null ? void 0 : e.widgets) == null ? void 0 : r.find((o) => o.name === "image");
  return q(t == null ? void 0 : t.value);
}
function $() {
  if (document.getElementById(z)) return;
  const n = document.createElement("style");
  n.id = z, n.textContent = `
        .pixal3d-camera-wrap {
            box-sizing: border-box;
            width: min(100%, var(--pixal3d-camera-widget-width, 410px));
            max-width: 100%;
            min-width: 0;
            margin: 0 auto;
            padding: 8px 10px 14px;
            color: #e8edf4;
            font: 12px/1.35 Arial, Helvetica, sans-serif;
            user-select: none;
            pointer-events: auto;
            overflow: hidden;
        }
        .pixal3d-camera-panel {
            box-sizing: border-box;
            width: 100%;
            max-width: 100%;
            background: #10141b;
            border: 1px solid #354052;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .pixal3d-camera-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 9px 10px;
            border-bottom: 1px solid #2d3746;
            background: #151b24;
        }
        .pixal3d-camera-title {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0;
        }
        .pixal3d-camera-readout {
            color: #aab7c8;
            font-size: 11px;
            white-space: nowrap;
        }
        .pixal3d-camera-canvas {
            box-sizing: border-box;
            display: block;
            width: 100%;
            height: var(--pixal3d-camera-canvas-height, 250px);
            max-width: 100%;
            background: #0c1016;
            cursor: crosshair;
        }
        .pixal3d-camera-controls {
            display: grid;
            gap: 10px;
            padding: 10px;
        }
        .pixal3d-camera-row {
            display: grid;
            grid-template-columns: 74px minmax(0, 1fr) 72px;
            align-items: center;
            gap: 8px;
        }
        .pixal3d-camera-row label {
            color: #b8c2d2;
            font-size: 11px;
        }
        .pixal3d-camera-row input[type="range"] {
            width: 100%;
            accent-color: #4fb4ff;
        }
        .pixal3d-camera-row input[type="number"] {
            box-sizing: border-box;
            width: 72px;
            padding: 5px 6px;
            border: 1px solid #3d4a5d;
            border-radius: 6px;
            background: #0d1219;
            color: #e8edf4;
            font-size: 11px;
        }
        .pixal3d-camera-view-toggle,
        .pixal3d-camera-actions {
            display: grid;
            gap: 6px;
        }
        .pixal3d-camera-view-toggle {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .pixal3d-camera-actions {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .pixal3d-camera-view-toggle button,
        .pixal3d-camera-actions button {
            padding: 7px 4px;
            border: 1px solid #3d4a5d;
            border-radius: 6px;
            background: #182231;
            color: #d7e3f4;
            font-size: 11px;
            cursor: pointer;
        }
        .pixal3d-camera-view-toggle button.is-active {
            border-color: #69c2ff;
            background: #243d5d;
            color: #ffffff;
        }
        .pixal3d-camera-view-toggle button:hover,
        .pixal3d-camera-actions button:hover {
            border-color: #4fb4ff;
            background: #203049;
        }
        .pixal3d-camera-hint {
            color: #8d9bae;
            font-size: 11px;
        }
    `, document.head.appendChild(n);
}
class G {
  constructor(e) {
    m(this, "node");
    m(this, "dragStart");
    m(this, "image");
    m(this, "imageUrl");
    m(this, "imageFailed", !1);
    m(this, "container", null);
    m(this, "canvas", null);
    m(this, "readout", null);
    m(this, "imageWatchTimer", null);
    m(this, "resizeObserver", null);
    m(this, "inputs", {});
    m(this, "fov", 49.134);
    m(this, "distance", 2);
    m(this, "scale", 1);
    m(this, "viewMode", "scene");
    this.node = e, this.dragStart = null, this.image = new Image(), this.image.decoding = "async", this.imageUrl = "", this.create(), this.syncFromWidgets(), this.updateLinkedImage(), this.startImageWatcher(), this.draw();
  }
  destroy() {
    var e;
    this.imageWatchTimer != null && (window.clearInterval(this.imageWatchTimer), this.imageWatchTimer = null), (e = this.resizeObserver) == null || e.disconnect(), this.resizeObserver = null, this.container = null, this.canvas = null;
  }
  create() {
    $(), this.container = document.createElement("div"), this.container.className = "pixal3d-camera-wrap", this.container.innerHTML = `
            <div class="pixal3d-camera-panel">
                <div class="pixal3d-camera-head">
                    <div class="pixal3d-camera-title">Manual Camera</div>
                    <div class="pixal3d-camera-readout"></div>
                </div>
                <canvas class="pixal3d-camera-canvas" width="390" height="250"></canvas>
                <div class="pixal3d-camera-controls">
                    <div class="pixal3d-camera-view-toggle">
                        <button data-view="scene" type="button">Scene</button>
                        <button data-view="pov" type="button">POV</button>
                        <button data-preset="default" type="button">Reset</button>
                    </div>
                    <div class="pixal3d-camera-row">
                        <label>FOV deg</label>
                        <input data-key="fov_degrees" type="range" min="5" max="140" step="0.1">
                        <input data-key="fov_degrees_num" type="number" min="5" max="140" step="0.1">
                    </div>
                    <div class="pixal3d-camera-row">
                        <label>Distance</label>
                        <input data-key="distance" type="range" min="0.1" max="20" step="0.01">
                        <input data-key="distance_num" type="number" min="0.1" max="20" step="0.01">
                    </div>
                    <div class="pixal3d-camera-row">
                        <label>Scale</label>
                        <input data-key="mesh_scale" type="range" min="0.05" max="10" step="0.05">
                        <input data-key="mesh_scale_num" type="number" min="0.05" max="10" step="0.05">
                    </div>
                    <div class="pixal3d-camera-actions">
                        <button data-preset="wide" type="button">Wide</button>
                        <button data-preset="close" type="button">Close</button>
                        <button data-preset="flat" type="button">Flat</button>
                    </div>
                    <div class="pixal3d-camera-hint">Scene shows the camera rig. POV matches the manual FOV, distance, and scale sent to Pixal3D. Wheel: distance.</div>
                </div>
            </div>
        `, this.canvas = this.container.querySelector("canvas"), this.readout = this.container.querySelector(".pixal3d-camera-readout"), this.inputs = {
      fov: this.container.querySelector('[data-key="fov_degrees"]'),
      fovNum: this.container.querySelector('[data-key="fov_degrees_num"]'),
      distance: this.container.querySelector('[data-key="distance"]'),
      distanceNum: this.container.querySelector('[data-key="distance_num"]'),
      scale: this.container.querySelector('[data-key="mesh_scale"]'),
      scaleNum: this.container.querySelector('[data-key="mesh_scale_num"]')
    };
    const e = (t) => this.setValues({ fov: h(t, 5, 140) }), a = (t) => this.setValues({ distance: h(t, 0.1, 20) }), s = (t) => this.setValues({ scale: h(t, 0.05, 10) });
    this.inputs.fov.addEventListener("input", (t) => e(t.target.value)), this.inputs.fovNum.addEventListener("change", (t) => e(t.target.value)), this.inputs.distance.addEventListener("input", (t) => a(t.target.value)), this.inputs.distanceNum.addEventListener("change", (t) => a(t.target.value)), this.inputs.scale.addEventListener("input", (t) => s(t.target.value)), this.inputs.scaleNum.addEventListener("change", (t) => s(t.target.value)), this.container.querySelectorAll("[data-preset]").forEach((t) => {
      const i = t;
      i.addEventListener("click", () => this.applyPreset(i.dataset.preset));
    }), this.container.querySelectorAll("[data-view]").forEach((t) => {
      const i = t;
      i.addEventListener("click", () => {
        this.viewMode = i.dataset.view === "pov" ? "pov" : "scene", this.updateControls(), this.draw();
      });
    }), this.container.addEventListener("wheel", (t) => {
      t.stopPropagation();
    }, { passive: !0 }), this.canvas.addEventListener("pointerdown", (t) => {
      var i, r;
      this.dragStart = {
        x: t.clientX,
        y: t.clientY,
        fov: this.fov,
        distance: this.distance
      }, (r = (i = this.canvas).setPointerCapture) == null || r.call(i, t.pointerId);
    }), this.canvas.addEventListener("pointermove", (t) => {
      if (!this.dragStart) return;
      const i = t.clientX - this.dragStart.x, r = t.clientY - this.dragStart.y;
      this.setValues({
        fov: h(this.dragStart.fov + i * 0.25, 5, 140),
        distance: h(this.dragStart.distance + r * 0.035, 0.1, 20)
      });
    }), this.canvas.addEventListener("pointerup", () => {
      this.dragStart = null;
    }), this.canvas.addEventListener("pointercancel", () => {
      this.dragStart = null;
    }), this.canvas.addEventListener("wheel", (t) => {
      t.preventDefault(), t.stopPropagation(), this.setValues({ distance: h(this.distance + Math.sign(t.deltaY) * 0.2, 0.1, 20) });
    }, { passive: !1 }), this.node.addDOMWidget("camera_ui", "div", this.container, { serialize: !1 }), "ResizeObserver" in window && (this.resizeObserver = new ResizeObserver(() => this.draw()), this.resizeObserver.observe(this.container)), window.requestAnimationFrame(() => this.draw());
  }
  syncLayout() {
    var o, d;
    if (!this.container || !this.canvas) return;
    const e = Math.max(280, Number((d = (o = this.node) == null ? void 0 : o.size) == null ? void 0 : d[0]) || x), a = Math.max(240, e - 22);
    this.container.style.setProperty("--pixal3d-camera-widget-width", `${a}px`);
    const s = this.container.querySelector(".pixal3d-camera-panel"), t = this.canvas.offsetWidth || (s == null ? void 0 : s.clientWidth) || a - 22, i = Math.max(220, Math.floor(t)), r = Math.max(180, Math.round(i * O));
    this.container.style.setProperty("--pixal3d-camera-canvas-height", `${r}px`), (this.canvas.width !== i || this.canvas.height !== r) && (this.canvas.width = i, this.canvas.height = r);
  }
  syncFromWidgets() {
    var e, a, s;
    this.fov = h(((e = F(this.node, "fov_degrees")) == null ? void 0 : e.value) ?? 49.134, 5, 140), this.distance = h(((a = F(this.node, "distance")) == null ? void 0 : a.value) ?? 2, 0.1, 20), this.scale = h(((s = F(this.node, "mesh_scale")) == null ? void 0 : s.value) ?? 1, 0.05, 10), this.updateControls();
  }
  setValues(e) {
    e.fov != null && (this.fov = h(e.fov, 5, 140)), e.distance != null && (this.distance = h(e.distance, 0.1, 20)), e.scale != null && (this.scale = h(e.scale, 0.05, 10)), I(this.node, "fov_degrees", Number(this.fov.toFixed(3))), I(this.node, "distance", Number(this.distance.toFixed(3))), I(this.node, "mesh_scale", Number(this.scale.toFixed(3))), this.updateControls(), this.draw();
  }
  applyPreset(e) {
    const a = {
      default: { fov: 49.134, distance: 2, scale: 1 },
      wide: { fov: 65, distance: 2.8, scale: 1 },
      close: { fov: 38, distance: 1.45, scale: 1 },
      flat: { fov: 24, distance: 3.2, scale: 1 }
    };
    this.setValues(a[e] || a.default);
  }
  updateControls() {
    var a;
    this.inputs.fov.value = String(this.fov), this.inputs.fovNum.value = this.fov.toFixed(3), this.inputs.distance.value = String(this.distance), this.inputs.distanceNum.value = this.distance.toFixed(3), this.inputs.scale.value = String(this.scale), this.inputs.scaleNum.value = this.scale.toFixed(3);
    const e = this.fov * Math.PI / 180;
    this.readout.textContent = `${this.viewMode.toUpperCase()} | ${e.toFixed(4)} rad`, (a = this.container) == null || a.querySelectorAll("[data-view]").forEach((s) => {
      const t = s;
      t.classList.toggle("is-active", t.dataset.view === this.viewMode);
    });
  }
  startImageWatcher() {
    this.imageWatchTimer == null && (this.imageWatchTimer = window.setInterval(() => {
      this.updateLinkedImage() && this.draw();
    }, 400));
  }
  updateLinkedImage() {
    const e = j(this.node);
    return e && e !== this.imageUrl ? (this.imageUrl = e, this.imageFailed = !1, this.image.onload = () => {
      this.imageFailed = !1, this.draw();
    }, this.image.onerror = () => {
      this.imageFailed = !0, this.draw();
    }, this.image.src = e, !0) : !e && this.imageUrl ? (this.imageUrl = "", this.imageFailed = !1, this.image.removeAttribute("src"), !0) : !1;
  }
  pathRoundRect(e, a, s, t, i, r) {
    e.beginPath(), e.roundRect ? e.roundRect(a, s, t, i, r) : e.rect(a, s, t, i);
  }
  drawFloorGrid(e, a, s, t) {
    e.save(), e.strokeStyle = "rgba(116, 135, 162, 0.18)", e.lineWidth = 1;
    for (let i = -8; i <= 8; i++) {
      const r = a / 2 + i * 18;
      e.beginPath(), e.moveTo(r, s), e.lineTo(a / 2 + i * 4, t), e.stroke();
    }
    for (let i = 0; i <= 9; i++) {
      const r = i / 9, o = t + (s - t) * (r * r);
      e.beginPath(), e.moveTo(0, o), e.lineTo(a, o), e.stroke();
    }
    e.restore();
  }
  drawImageCard(e, a, s, t, i, r = "image preview") {
    if (e.save(), e.fillStyle = "#17202c", e.strokeStyle = "#d8e7ff", e.lineWidth = 2, this.pathRoundRect(e, a, s, t, i, 8), e.fill(), e.stroke(), this.image.complete && this.image.naturalWidth > 0) {
      e.clip();
      const o = Math.max(t / this.image.naturalWidth, i / this.image.naturalHeight), d = this.image.naturalWidth * o, u = this.image.naturalHeight * o;
      e.drawImage(this.image, a + (t - d) / 2, s + (i - u) / 2, d, u);
    } else {
      e.fillStyle = "#758399", e.font = "12px Arial", e.textAlign = "center";
      const o = this.imageFailed ? "image unavailable" : this.imageUrl ? "image loading" : r;
      e.fillText(o, a + t / 2, s + i / 2 + 4);
    }
    e.restore();
  }
  drawScene(e, a, s) {
    const t = this.fov * Math.PI / 180, i = a / 2, r = 92, o = s - 30, d = h(o - this.distance * 42, 54, o - 46), u = h(1.1 / Math.max(this.distance, 0.35), 0.34, 1.45) * Math.sqrt(this.scale), p = 82 * u, l = 92 * u, c = h(Math.tan(t / 2) * this.distance * 42, 18, a / 2 - 24);
    this.drawFloorGrid(e, a, s, r), e.save(), e.fillStyle = "rgba(79, 180, 255, 0.15)", e.strokeStyle = "#4fb4ff", e.lineWidth = 2, e.beginPath(), e.moveTo(i, o), e.lineTo(i - c, d), e.lineTo(i + c, d), e.closePath(), e.fill(), e.stroke(), e.strokeStyle = "rgba(255,255,255,0.58)", e.setLineDash([5, 5]), e.beginPath(), e.moveTo(i, o), e.lineTo(i, d), e.stroke(), e.setLineDash([]), e.fillStyle = "rgba(255,207,90,0.18)", e.strokeStyle = "#ffcf5a", e.lineWidth = 2, e.beginPath(), e.arc(i, d, 12 + p * 0.25, 0, Math.PI * 2), e.fill(), e.stroke(), this.drawImageCard(e, i - p / 2, d - l / 2, p, l, "target"), e.fillStyle = "#ffcf5a", e.strokeStyle = "#18110a", e.lineWidth = 2, e.beginPath(), e.moveTo(i, o - 15), e.lineTo(i - 18, o + 15), e.lineTo(i + 18, o + 15), e.closePath(), e.fill(), e.stroke(), e.fillStyle = "#151b24", e.fillRect(i - 8, o - 2, 16, 10), e.fillStyle = "#dbe8fa", e.font = "12px Arial", e.textAlign = "left", e.fillText("Scene rig", 12, 20), e.fillText(`FOV ${this.fov.toFixed(1)} deg`, 12, 38), e.fillText(`distance ${this.distance.toFixed(2)}`, 12, 56), e.textAlign = "right", e.fillText(`view width ${(2 * this.distance * Math.tan(t / 2)).toFixed(2)}`, a - 12, 20), e.restore();
  }
  drawPov(e, a, s) {
    const t = this.fov * Math.PI / 180, i = 18, r = 16, o = a - 36, d = s - 42, u = i + o / 2, p = r + d / 2, l = !this.imageFailed && this.image.complete && this.image.naturalWidth > 0 ? this.image.naturalWidth / this.image.naturalHeight : 0.8, c = o / 2 / Math.tan(t / 2), f = 1 * this.scale, v = f / h(l, 0.55, 1.65), S = f * c / Math.max(this.distance, 0.05), y = v * c / Math.max(this.distance, 0.05), g = h(S, 12, o * 3.2), w = h(y, 12, d * 3.2), W = S / Math.max(o * 0.34, 1), T = 2 * this.distance * Math.tan(t / 2);
    e.save();
    const k = e.createLinearGradient(0, r, 0, r + d);
    k.addColorStop(0, "#18202b"), k.addColorStop(0.58, "#0f151d"), k.addColorStop(1, "#0a0e14"), e.fillStyle = k, this.pathRoundRect(e, i, r, o, d, 10), e.fill(), e.strokeStyle = "#8fcfff", e.lineWidth = 2, e.stroke(), e.clip(), e.strokeStyle = "rgba(255,255,255,0.10)", e.lineWidth = 1;
    const E = T / 2;
    for (let b = -4; b <= 4; b++) {
      const _ = u + b / 4 * c * (E / Math.max(this.distance, 0.05));
      e.beginPath(), e.moveTo(_, r), e.lineTo(_, r + d), e.stroke();
    }
    for (let b = 1; b < 3; b++)
      e.beginPath(), e.moveTo(i, r + d * b / 3), e.lineTo(i + o, r + d * b / 3), e.stroke();
    e.fillStyle = "rgba(79,180,255,0.10)", e.beginPath(), e.ellipse(u, p + w * 0.44, g * 0.58, w * 0.09, 0, 0, Math.PI * 2), e.fill(), this.drawImageCard(e, u - g / 2, p - w / 2, g, w, "camera view"), e.strokeStyle = "rgba(255,255,255,0.62)", e.setLineDash([4, 6]), e.beginPath(), e.moveTo(u - 15, p), e.lineTo(u + 15, p), e.moveTo(u, p - 15), e.lineTo(u, p + 15), e.stroke(), e.setLineDash([]), e.restore(), e.save(), e.fillStyle = "#dbe8fa", e.font = "12px Arial", e.textAlign = "left", e.fillText(`POV FOV ${this.fov.toFixed(1)} deg`, 12, s - 12), e.textAlign = "right", e.fillText(`view width ${T.toFixed(2)} | scale ${W.toFixed(2)}x`, a - 12, s - 12), e.restore();
  }
  draw() {
    if (!this.canvas) return;
    this.syncLayout(), this.updateLinkedImage();
    const e = this.canvas.getContext("2d");
    if (!e) return;
    const a = this.canvas.width, s = this.canvas.height;
    e.clearRect(0, 0, a, s), e.fillStyle = "#0c1016", e.fillRect(0, 0, a, s), this.viewMode === "pov" ? this.drawPov(e, a, s) : this.drawScene(e, a, s);
  }
}
P.registerExtension({
  name: "Pixal3D.CameraControl",
  async beforeRegisterNodeDef(n, e) {
    if (e.name !== U) return;
    const a = n.prototype.onNodeCreated, s = n.prototype.onConfigure, t = n.prototype.onConnectionsChange;
    function i(l) {
      for (const c of l.widgets || [])
        ["fov_degrees", "distance", "mesh_scale"].includes(c.name) && (c.hidden = !0, c.computeSize = () => [0, -4]);
    }
    function r(l) {
      return Math.max(C, Math.round(l * D));
    }
    function o(l) {
      var g, w, W;
      const c = Number((g = l.size) == null ? void 0 : g[0]) || 0, f = Number((w = l.size) == null ? void 0 : w[1]) || 0, v = Math.max(c || x, M), y = Math.abs(f - A) <= 1 ? L : Math.max(f || r(v), C);
      l.min_size = [M, C], l.minSize = [M, C], (v !== c || y !== f) && ((W = l.setSize) == null || W.call(l, [v, y]));
    }
    function d(l) {
      i(l), l.pixal3dCameraUI ? (l.pixal3dCameraUI.syncFromWidgets(), l.pixal3dCameraUI.draw()) : l.pixal3dCameraUI = new G(l), o(l);
    }
    n.prototype.onNodeCreated = function() {
      const l = a == null ? void 0 : a.apply(this, arguments);
      return this.serialize_widgets = !0, this.resizable = !0, this.resizeable = !0, d(this), l;
    }, n.prototype.onConfigure = function() {
      const l = s == null ? void 0 : s.apply(this, arguments);
      return setTimeout(() => d(this), 0), l;
    }, n.prototype.onConnectionsChange = function() {
      var c, f;
      const l = t == null ? void 0 : t.apply(this, arguments);
      return (c = this.pixal3dCameraUI) == null || c.updateLinkedImage(), (f = this.pixal3dCameraUI) == null || f.draw(), l;
    };
    const u = n.prototype.onResize;
    n.prototype.onResize = function() {
      var v, S, y, g;
      const l = u == null ? void 0 : u.apply(this, arguments), c = Math.max(M, Number((v = this.size) == null ? void 0 : v[0]) || x), f = r(c);
      return Math.abs((Number((S = this.size) == null ? void 0 : S[1]) || 0) - f) > 1 && ((y = this.setSize) == null || y.call(this, [c, f])), (g = this.pixal3dCameraUI) == null || g.draw(), l;
    }, n.prototype.computeSize = function() {
      return [x, L];
    };
    const p = n.prototype.onRemoved;
    n.prototype.onRemoved = function() {
      var l;
      return (l = this.pixal3dCameraUI) == null || l.destroy(), this.pixal3dCameraUI = null, p == null ? void 0 : p.apply(this, arguments);
    };
  }
});
