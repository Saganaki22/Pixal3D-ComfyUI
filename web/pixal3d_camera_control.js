var E = Object.defineProperty;
var U = (n, e, t) => e in n ? E(n, e, { enumerable: !0, configurable: !0, writable: !0, value: t }) : n[e] = t;
var h = (n, e, t) => U(n, typeof e != "symbol" ? e + "" : e, t);
import { app as y } from "/scripts/app.js";
const N = "Pixal3DCameraControl", T = "pixal3d-camera-control-styles", L = 430, P = 560;
function c(n, e, t) {
  return Math.max(e, Math.min(t, Number(n) || 0));
}
function b(n, e) {
  var t;
  return (t = n.widgets) == null ? void 0 : t.find((s) => s.name === e);
}
function S(n, e, t) {
  var i, r, l;
  const s = b(n, e);
  if (!s) return;
  const a = s.value;
  s.value = t, a !== t && ((i = s.callback) == null || i.call(s, t), (l = (r = y.graph) == null ? void 0 : r.setDirtyCanvas) == null || l.call(r, !0, !0));
}
function D(n) {
  var t;
  const e = (t = y.graph) == null ? void 0 : t.links;
  return !e || n == null ? null : Array.isArray(e) ? e.find((s) => s && s.id === n) || e[n] || null : e[n] || null;
}
function A(n, e) {
  var a, i, r;
  const t = (a = n.inputs) == null ? void 0 : a.find((l) => l.name === e), s = D(t == null ? void 0 : t.link);
  return s && ((r = (i = y.graph) == null ? void 0 : i.getNodeById) == null ? void 0 : r.call(i, s.origin_id)) || null;
}
function R(n) {
  if (!n) return "";
  let e = "", t = "input", s = "";
  if (typeof n == "string" ? e = n : typeof n == "object" && (e = n.filename || n.name || n.image || "", t = n.type || t, s = n.subfolder || ""), !e) return "";
  const a = new URLSearchParams();
  return a.set("filename", e), a.set("type", t), s && a.set("subfolder", s), `/view?${a.toString()}`;
}
function V(n) {
  var i, r;
  const e = A(n, "image"), t = (i = e == null ? void 0 : e.imgs) == null ? void 0 : i[0], s = (t == null ? void 0 : t.currentSrc) || (t == null ? void 0 : t.src);
  if (s) return s;
  const a = (r = e == null ? void 0 : e.widgets) == null ? void 0 : r.find((l) => l.name === "image");
  return R(a == null ? void 0 : a.value);
}
function H() {
  if (document.getElementById(T)) return;
  const n = document.createElement("style");
  n.id = T, n.textContent = `
        .pixal3d-camera-wrap {
            box-sizing: border-box;
            width: 100%;
            padding: 12px;
            color: #e8edf4;
            font: 12px/1.35 Arial, Helvetica, sans-serif;
            user-select: none;
            pointer-events: auto;
        }
        .pixal3d-camera-panel {
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
            display: block;
            width: 100%;
            height: 250px;
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
class j {
  constructor(e) {
    h(this, "node");
    h(this, "dragStart");
    h(this, "image");
    h(this, "imageUrl");
    h(this, "imageFailed", !1);
    h(this, "container", null);
    h(this, "canvas", null);
    h(this, "readout", null);
    h(this, "imageWatchTimer", null);
    h(this, "inputs", {});
    h(this, "fov", 49.134);
    h(this, "distance", 2);
    h(this, "scale", 1);
    h(this, "viewMode", "scene");
    this.node = e, this.dragStart = null, this.image = new Image(), this.image.decoding = "async", this.imageUrl = "", this.create(), this.syncFromWidgets(), this.updateLinkedImage(), this.startImageWatcher(), this.draw();
  }
  destroy() {
    this.imageWatchTimer != null && (window.clearInterval(this.imageWatchTimer), this.imageWatchTimer = null), this.container = null, this.canvas = null;
  }
  create() {
    H(), this.container = document.createElement("div"), this.container.className = "pixal3d-camera-wrap", this.container.innerHTML = `
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
    const e = (a) => this.setValues({ fov: c(a, 5, 140) }), t = (a) => this.setValues({ distance: c(a, 0.1, 20) }), s = (a) => this.setValues({ scale: c(a, 0.05, 10) });
    this.inputs.fov.addEventListener("input", (a) => e(a.target.value)), this.inputs.fovNum.addEventListener("change", (a) => e(a.target.value)), this.inputs.distance.addEventListener("input", (a) => t(a.target.value)), this.inputs.distanceNum.addEventListener("change", (a) => t(a.target.value)), this.inputs.scale.addEventListener("input", (a) => s(a.target.value)), this.inputs.scaleNum.addEventListener("change", (a) => s(a.target.value)), this.container.querySelectorAll("[data-preset]").forEach((a) => {
      const i = a;
      i.addEventListener("click", () => this.applyPreset(i.dataset.preset));
    }), this.container.querySelectorAll("[data-view]").forEach((a) => {
      const i = a;
      i.addEventListener("click", () => {
        this.viewMode = i.dataset.view === "pov" ? "pov" : "scene", this.updateControls(), this.draw();
      });
    }), this.canvas.addEventListener("pointerdown", (a) => {
      var i, r;
      this.dragStart = {
        x: a.clientX,
        y: a.clientY,
        fov: this.fov,
        distance: this.distance
      }, (r = (i = this.canvas).setPointerCapture) == null || r.call(i, a.pointerId);
    }), this.canvas.addEventListener("pointermove", (a) => {
      if (!this.dragStart) return;
      const i = a.clientX - this.dragStart.x, r = a.clientY - this.dragStart.y;
      this.setValues({
        fov: c(this.dragStart.fov + i * 0.25, 5, 140),
        distance: c(this.dragStart.distance + r * 0.035, 0.1, 20)
      });
    }), this.canvas.addEventListener("pointerup", () => {
      this.dragStart = null;
    }), this.canvas.addEventListener("pointercancel", () => {
      this.dragStart = null;
    }), this.canvas.addEventListener("wheel", (a) => {
      a.preventDefault(), this.setValues({ distance: c(this.distance + Math.sign(a.deltaY) * 0.2, 0.1, 20) });
    }, { passive: !1 }), this.node.addDOMWidget("camera_ui", "div", this.container, { serialize: !1 });
  }
  syncFromWidgets() {
    var e, t, s;
    this.fov = c(((e = b(this.node, "fov_degrees")) == null ? void 0 : e.value) ?? 49.134, 5, 140), this.distance = c(((t = b(this.node, "distance")) == null ? void 0 : t.value) ?? 2, 0.1, 20), this.scale = c(((s = b(this.node, "mesh_scale")) == null ? void 0 : s.value) ?? 1, 0.05, 10), this.updateControls();
  }
  setValues(e) {
    e.fov != null && (this.fov = c(e.fov, 5, 140)), e.distance != null && (this.distance = c(e.distance, 0.1, 20)), e.scale != null && (this.scale = c(e.scale, 0.05, 10)), S(this.node, "fov_degrees", Number(this.fov.toFixed(3))), S(this.node, "distance", Number(this.distance.toFixed(3))), S(this.node, "mesh_scale", Number(this.scale.toFixed(3))), this.updateControls(), this.draw();
  }
  applyPreset(e) {
    const t = {
      default: { fov: 49.134, distance: 2, scale: 1 },
      wide: { fov: 65, distance: 2.8, scale: 1 },
      close: { fov: 38, distance: 1.45, scale: 1 },
      flat: { fov: 24, distance: 3.2, scale: 1 }
    };
    this.setValues(t[e] || t.default);
  }
  updateControls() {
    var t;
    this.inputs.fov.value = String(this.fov), this.inputs.fovNum.value = this.fov.toFixed(3), this.inputs.distance.value = String(this.distance), this.inputs.distanceNum.value = this.distance.toFixed(3), this.inputs.scale.value = String(this.scale), this.inputs.scaleNum.value = this.scale.toFixed(3);
    const e = this.fov * Math.PI / 180;
    this.readout.textContent = `${this.viewMode.toUpperCase()} | ${e.toFixed(4)} rad`, (t = this.container) == null || t.querySelectorAll("[data-view]").forEach((s) => {
      const a = s;
      a.classList.toggle("is-active", a.dataset.view === this.viewMode);
    });
  }
  startImageWatcher() {
    this.imageWatchTimer == null && (this.imageWatchTimer = window.setInterval(() => {
      this.updateLinkedImage() && this.draw();
    }, 400));
  }
  updateLinkedImage() {
    const e = V(this.node);
    return e && e !== this.imageUrl ? (this.imageUrl = e, this.imageFailed = !1, this.image.onload = () => {
      this.imageFailed = !1, this.draw();
    }, this.image.onerror = () => {
      this.imageFailed = !0, this.draw();
    }, this.image.src = e, !0) : !e && this.imageUrl ? (this.imageUrl = "", this.imageFailed = !1, this.image.removeAttribute("src"), !0) : !1;
  }
  pathRoundRect(e, t, s, a, i, r) {
    e.beginPath(), e.roundRect ? e.roundRect(t, s, a, i, r) : e.rect(t, s, a, i);
  }
  drawFloorGrid(e, t, s, a) {
    e.save(), e.strokeStyle = "rgba(116, 135, 162, 0.18)", e.lineWidth = 1;
    for (let i = -8; i <= 8; i++) {
      const r = t / 2 + i * 18;
      e.beginPath(), e.moveTo(r, s), e.lineTo(t / 2 + i * 4, a), e.stroke();
    }
    for (let i = 0; i <= 9; i++) {
      const r = i / 9, l = a + (s - a) * (r * r);
      e.beginPath(), e.moveTo(0, l), e.lineTo(t, l), e.stroke();
    }
    e.restore();
  }
  drawImageCard(e, t, s, a, i, r = "image preview") {
    if (e.save(), e.fillStyle = "#17202c", e.strokeStyle = "#d8e7ff", e.lineWidth = 2, this.pathRoundRect(e, t, s, a, i, 8), e.fill(), e.stroke(), this.image.complete && this.image.naturalWidth > 0) {
      e.clip();
      const l = Math.max(a / this.image.naturalWidth, i / this.image.naturalHeight), o = this.image.naturalWidth * l, d = this.image.naturalHeight * l;
      e.drawImage(this.image, t + (a - o) / 2, s + (i - d) / 2, o, d);
    } else {
      e.fillStyle = "#758399", e.font = "12px Arial", e.textAlign = "center";
      const l = this.imageFailed ? "image unavailable" : this.imageUrl ? "image loading" : r;
      e.fillText(l, t + a / 2, s + i / 2 + 4);
    }
    e.restore();
  }
  drawScene(e, t, s) {
    const a = this.fov * Math.PI / 180, i = t / 2, r = 92, l = s - 30, o = c(l - this.distance * 42, 54, l - 46), d = c(1.1 / Math.max(this.distance, 0.35), 0.34, 1.45) * Math.sqrt(this.scale), u = 82 * d, f = 92 * d, m = c(Math.tan(a / 2) * this.distance * 42, 18, t / 2 - 24);
    this.drawFloorGrid(e, t, s, r), e.save(), e.fillStyle = "rgba(79, 180, 255, 0.15)", e.strokeStyle = "#4fb4ff", e.lineWidth = 2, e.beginPath(), e.moveTo(i, l), e.lineTo(i - m, o), e.lineTo(i + m, o), e.closePath(), e.fill(), e.stroke(), e.strokeStyle = "rgba(255,255,255,0.58)", e.setLineDash([5, 5]), e.beginPath(), e.moveTo(i, l), e.lineTo(i, o), e.stroke(), e.setLineDash([]), e.fillStyle = "rgba(255,207,90,0.18)", e.strokeStyle = "#ffcf5a", e.lineWidth = 2, e.beginPath(), e.arc(i, o, 12 + u * 0.25, 0, Math.PI * 2), e.fill(), e.stroke(), this.drawImageCard(e, i - u / 2, o - f / 2, u, f, "target"), e.fillStyle = "#ffcf5a", e.strokeStyle = "#18110a", e.lineWidth = 2, e.beginPath(), e.moveTo(i, l - 15), e.lineTo(i - 18, l + 15), e.lineTo(i + 18, l + 15), e.closePath(), e.fill(), e.stroke(), e.fillStyle = "#151b24", e.fillRect(i - 8, l - 2, 16, 10), e.fillStyle = "#dbe8fa", e.font = "12px Arial", e.textAlign = "left", e.fillText("Scene rig", 12, 20), e.fillText(`FOV ${this.fov.toFixed(1)} deg`, 12, 38), e.fillText(`distance ${this.distance.toFixed(2)}`, 12, 56), e.textAlign = "right", e.fillText(`view width ${(2 * this.distance * Math.tan(a / 2)).toFixed(2)}`, t - 12, 20), e.restore();
  }
  drawPov(e, t, s) {
    const a = this.fov * Math.PI / 180, i = 18, r = 16, l = t - 36, o = s - 42, d = i + l / 2, u = r + o / 2, f = !this.imageFailed && this.image.complete && this.image.naturalWidth > 0 ? this.image.naturalWidth / this.image.naturalHeight : 0.8, m = l / 2 / Math.tan(a / 2), k = 1 * this.scale, I = k / c(f, 0.55, 1.65), W = k * m / Math.max(this.distance, 0.05), M = I * m / Math.max(this.distance, 0.05), w = c(W, 12, l * 3.2), g = c(M, 12, o * 3.2), _ = W / Math.max(l * 0.34, 1), C = 2 * this.distance * Math.tan(a / 2);
    e.save();
    const v = e.createLinearGradient(0, r, 0, r + o);
    v.addColorStop(0, "#18202b"), v.addColorStop(0.58, "#0f151d"), v.addColorStop(1, "#0a0e14"), e.fillStyle = v, this.pathRoundRect(e, i, r, l, o, 10), e.fill(), e.strokeStyle = "#8fcfff", e.lineWidth = 2, e.stroke(), e.clip(), e.strokeStyle = "rgba(255,255,255,0.10)", e.lineWidth = 1;
    const x = C / 2;
    for (let p = -4; p <= 4; p++) {
      const F = d + p / 4 * m * (x / Math.max(this.distance, 0.05));
      e.beginPath(), e.moveTo(F, r), e.lineTo(F, r + o), e.stroke();
    }
    for (let p = 1; p < 3; p++)
      e.beginPath(), e.moveTo(i, r + o * p / 3), e.lineTo(i + l, r + o * p / 3), e.stroke();
    e.fillStyle = "rgba(79,180,255,0.10)", e.beginPath(), e.ellipse(d, u + g * 0.44, w * 0.58, g * 0.09, 0, 0, Math.PI * 2), e.fill(), this.drawImageCard(e, d - w / 2, u - g / 2, w, g, "camera view"), e.strokeStyle = "rgba(255,255,255,0.62)", e.setLineDash([4, 6]), e.beginPath(), e.moveTo(d - 15, u), e.lineTo(d + 15, u), e.moveTo(d, u - 15), e.lineTo(d, u + 15), e.stroke(), e.setLineDash([]), e.restore(), e.save(), e.fillStyle = "#dbe8fa", e.font = "12px Arial", e.textAlign = "left", e.fillText(`POV FOV ${this.fov.toFixed(1)} deg`, 12, s - 12), e.textAlign = "right", e.fillText(`view width ${C.toFixed(2)} | scale ${_.toFixed(2)}x`, t - 12, s - 12), e.restore();
  }
  draw() {
    if (!this.canvas) return;
    this.updateLinkedImage();
    const e = this.canvas.getContext("2d");
    if (!e) return;
    const t = this.canvas.width, s = this.canvas.height;
    e.clearRect(0, 0, t, s), e.fillStyle = "#0c1016", e.fillRect(0, 0, t, s), this.viewMode === "pov" ? this.drawPov(e, t, s) : this.drawScene(e, t, s);
  }
}
y.registerExtension({
  name: "Pixal3D.CameraControl",
  async beforeRegisterNodeDef(n, e) {
    if (e.name !== N) return;
    const t = n.prototype.onNodeCreated, s = n.prototype.onConfigure, a = n.prototype.onConnectionsChange;
    function i(o) {
      for (const d of o.widgets || [])
        ["fov_degrees", "distance", "mesh_scale"].includes(d.name) && (d.hidden = !0, d.computeSize = () => [0, -4]);
    }
    function r(o) {
      var d;
      i(o), o.pixal3dCameraUI ? (o.pixal3dCameraUI.syncFromWidgets(), o.pixal3dCameraUI.draw()) : o.pixal3dCameraUI = new j(o), (d = o.setSize) == null || d.call(o, [L, P]);
    }
    n.prototype.onNodeCreated = function() {
      const o = t == null ? void 0 : t.apply(this, arguments);
      return this.serialize_widgets = !0, this.resizable = !1, r(this), o;
    }, n.prototype.onConfigure = function() {
      const o = s == null ? void 0 : s.apply(this, arguments);
      return setTimeout(() => r(this), 0), o;
    }, n.prototype.onConnectionsChange = function() {
      var d, u;
      const o = a == null ? void 0 : a.apply(this, arguments);
      return (d = this.pixal3dCameraUI) == null || d.updateLinkedImage(), (u = this.pixal3dCameraUI) == null || u.draw(), o;
    }, n.prototype.computeSize = function() {
      return [L, P];
    };
    const l = n.prototype.onRemoved;
    n.prototype.onRemoved = function() {
      var o;
      return (o = this.pixal3dCameraUI) == null || o.destroy(), this.pixal3dCameraUI = null, l == null ? void 0 : l.apply(this, arguments);
    };
  }
});
