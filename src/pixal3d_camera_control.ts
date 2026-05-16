// ComfyUI serves this module at runtime from its frontend.
// @ts-ignore
import { app } from "/scripts/app.js";

const NODE_NAME = "Pixal3DCameraControl";
const STYLE_ID = "pixal3d-camera-control-styles";
const DEFAULT_WIDTH = 430;
const DEFAULT_HEIGHT = 560;
type ViewMode = "scene" | "pov";

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, Number(value) || 0));
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return;
    const oldValue = widget.value;
    widget.value = value;
    if (oldValue !== value) {
        widget.callback?.(value);
        app.graph?.setDirtyCanvas?.(true, true);
    }
}

function getGraphLink(linkId) {
    const links = app.graph?.links;
    if (!links || linkId == null) return null;
    if (Array.isArray(links)) {
        return links.find((link) => link && link.id === linkId) || links[linkId] || null;
    }
    return links[linkId] || null;
}

function getLinkedInputNode(node, inputName) {
    const input = node.inputs?.find((item) => item.name === inputName);
    const link = getGraphLink(input?.link);
    if (!link) return null;
    return app.graph?.getNodeById?.(link.origin_id) || null;
}

function imageValueToUrl(value) {
    if (!value) return "";
    let filename = "";
    let type = "input";
    let subfolder = "";

    if (typeof value === "string") {
        filename = value;
    } else if (typeof value === "object") {
        filename = value.filename || value.name || value.image || "";
        type = value.type || type;
        subfolder = value.subfolder || "";
    }

    if (!filename) return "";
    const params = new URLSearchParams();
    params.set("filename", filename);
    params.set("type", type);
    if (subfolder) params.set("subfolder", subfolder);
    return `/view?${params.toString()}`;
}

function getLoadImageUrl(node) {
    const source = getLinkedInputNode(node, "image");
    const previewImage = source?.imgs?.[0];
    const previewSrc = previewImage?.currentSrc || previewImage?.src;
    if (previewSrc) return previewSrc;

    const imageWidget = source?.widgets?.find((widget) => widget.name === "image");
    return imageValueToUrl(imageWidget?.value);
}

function injectCSS() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
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
    `;
    document.head.appendChild(style);
}

class Pixal3DCameraUI {
    node: any;
    dragStart: any;
    image: HTMLImageElement;
    imageUrl: string;
    imageFailed = false;
    container: HTMLElement | null = null;
    canvas: HTMLCanvasElement | null = null;
    readout: HTMLElement | null = null;
    imageWatchTimer: number | null = null;
    inputs: any = {};
    fov = 49.134;
    distance = 2.0;
    scale = 1.0;
    viewMode: ViewMode = "scene";

    constructor(node) {
        this.node = node;
        this.dragStart = null;
        this.image = new Image();
        this.image.decoding = "async";
        this.imageUrl = "";
        this.create();
        this.syncFromWidgets();
        this.updateLinkedImage();
        this.startImageWatcher();
        this.draw();
    }

    destroy() {
        if (this.imageWatchTimer != null) {
            window.clearInterval(this.imageWatchTimer);
            this.imageWatchTimer = null;
        }
        this.container = null;
        this.canvas = null;
    }

    create() {
        injectCSS();
        this.container = document.createElement("div");
        this.container.className = "pixal3d-camera-wrap";
        this.container.innerHTML = `
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
        `;

        this.canvas = this.container.querySelector("canvas");
        this.readout = this.container.querySelector(".pixal3d-camera-readout");
        this.inputs = {
            fov: this.container.querySelector('[data-key="fov_degrees"]'),
            fovNum: this.container.querySelector('[data-key="fov_degrees_num"]'),
            distance: this.container.querySelector('[data-key="distance"]'),
            distanceNum: this.container.querySelector('[data-key="distance_num"]'),
            scale: this.container.querySelector('[data-key="mesh_scale"]'),
            scaleNum: this.container.querySelector('[data-key="mesh_scale_num"]'),
        };

        const setFov = (value) => this.setValues({ fov: clamp(value, 5, 140) });
        const setDistance = (value) => this.setValues({ distance: clamp(value, 0.1, 20) });
        const setScale = (value) => this.setValues({ scale: clamp(value, 0.05, 10) });

        this.inputs.fov.addEventListener("input", (event) => setFov(event.target.value));
        this.inputs.fovNum.addEventListener("change", (event) => setFov(event.target.value));
        this.inputs.distance.addEventListener("input", (event) => setDistance(event.target.value));
        this.inputs.distanceNum.addEventListener("change", (event) => setDistance(event.target.value));
        this.inputs.scale.addEventListener("input", (event) => setScale(event.target.value));
        this.inputs.scaleNum.addEventListener("change", (event) => setScale(event.target.value));

        this.container.querySelectorAll("[data-preset]").forEach((button) => {
            const presetButton = button as HTMLElement;
            presetButton.addEventListener("click", () => this.applyPreset(presetButton.dataset.preset));
        });
        this.container.querySelectorAll("[data-view]").forEach((button) => {
            const viewButton = button as HTMLElement;
            viewButton.addEventListener("click", () => {
                this.viewMode = viewButton.dataset.view === "pov" ? "pov" : "scene";
                this.updateControls();
                this.draw();
            });
        });

        this.canvas.addEventListener("pointerdown", (event) => {
            this.dragStart = {
                x: event.clientX,
                y: event.clientY,
                fov: this.fov,
                distance: this.distance,
            };
            this.canvas.setPointerCapture?.(event.pointerId);
        });
        this.canvas.addEventListener("pointermove", (event) => {
            if (!this.dragStart) return;
            const dx = event.clientX - this.dragStart.x;
            const dy = event.clientY - this.dragStart.y;
            this.setValues({
                fov: clamp(this.dragStart.fov + dx * 0.25, 5, 140),
                distance: clamp(this.dragStart.distance + dy * 0.035, 0.1, 20),
            });
        });
        this.canvas.addEventListener("pointerup", () => {
            this.dragStart = null;
        });
        this.canvas.addEventListener("pointercancel", () => {
            this.dragStart = null;
        });
        this.canvas.addEventListener("wheel", (event) => {
            event.preventDefault();
            this.setValues({ distance: clamp(this.distance + Math.sign(event.deltaY) * 0.2, 0.1, 20) });
        }, { passive: false });

        this.node.addDOMWidget("camera_ui", "div", this.container, { serialize: false });
    }

    syncFromWidgets() {
        this.fov = clamp(getWidget(this.node, "fov_degrees")?.value ?? 49.134, 5, 140);
        this.distance = clamp(getWidget(this.node, "distance")?.value ?? 2.0, 0.1, 20);
        this.scale = clamp(getWidget(this.node, "mesh_scale")?.value ?? 1.0, 0.05, 10);
        this.updateControls();
    }

    setValues(values) {
        if (values.fov != null) this.fov = clamp(values.fov, 5, 140);
        if (values.distance != null) this.distance = clamp(values.distance, 0.1, 20);
        if (values.scale != null) this.scale = clamp(values.scale, 0.05, 10);
        setWidgetValue(this.node, "fov_degrees", Number(this.fov.toFixed(3)));
        setWidgetValue(this.node, "distance", Number(this.distance.toFixed(3)));
        setWidgetValue(this.node, "mesh_scale", Number(this.scale.toFixed(3)));
        this.updateControls();
        this.draw();
    }

    applyPreset(name) {
        const presets = {
            default: { fov: 49.134, distance: 2.0, scale: 1.0 },
            wide: { fov: 65.0, distance: 2.8, scale: 1.0 },
            close: { fov: 38.0, distance: 1.45, scale: 1.0 },
            flat: { fov: 24.0, distance: 3.2, scale: 1.0 },
        };
        this.setValues(presets[name] || presets.default);
    }

    updateControls() {
        this.inputs.fov.value = String(this.fov);
        this.inputs.fovNum.value = this.fov.toFixed(3);
        this.inputs.distance.value = String(this.distance);
        this.inputs.distanceNum.value = this.distance.toFixed(3);
        this.inputs.scale.value = String(this.scale);
        this.inputs.scaleNum.value = this.scale.toFixed(3);
        const radians = this.fov * Math.PI / 180;
        this.readout.textContent = `${this.viewMode.toUpperCase()} | ${radians.toFixed(4)} rad`;
        this.container?.querySelectorAll("[data-view]").forEach((button) => {
            const viewButton = button as HTMLElement;
            viewButton.classList.toggle("is-active", viewButton.dataset.view === this.viewMode);
        });
    }

    startImageWatcher() {
        if (this.imageWatchTimer != null) return;
        this.imageWatchTimer = window.setInterval(() => {
            if (this.updateLinkedImage()) {
                this.draw();
            }
        }, 400);
    }

    updateLinkedImage() {
        const url = getLoadImageUrl(this.node);
        if (url && url !== this.imageUrl) {
            this.imageUrl = url;
            this.imageFailed = false;
            this.image.onload = () => {
                this.imageFailed = false;
                this.draw();
            };
            this.image.onerror = () => {
                this.imageFailed = true;
                this.draw();
            };
            this.image.src = url;
            return true;
        }
        if (!url && this.imageUrl) {
            this.imageUrl = "";
            this.imageFailed = false;
            this.image.removeAttribute("src");
            return true;
        }
        return false;
    }

    pathRoundRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, width, height, radius);
        } else {
            ctx.rect(x, y, width, height);
        }
    }

    drawFloorGrid(ctx, width, height, horizon) {
        ctx.save();
        ctx.strokeStyle = "rgba(116, 135, 162, 0.18)";
        ctx.lineWidth = 1;
        for (let i = -8; i <= 8; i++) {
            const x = width / 2 + i * 18;
            ctx.beginPath();
            ctx.moveTo(x, height);
            ctx.lineTo(width / 2 + i * 4, horizon);
            ctx.stroke();
        }
        for (let i = 0; i <= 9; i++) {
            const t = i / 9;
            const y = horizon + (height - horizon) * (t * t);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        ctx.restore();
    }

    drawImageCard(ctx, x, y, width, height, label = "image preview") {
        ctx.save();
        ctx.fillStyle = "#17202c";
        ctx.strokeStyle = "#d8e7ff";
        ctx.lineWidth = 2;
        this.pathRoundRect(ctx, x, y, width, height, 8);
        ctx.fill();
        ctx.stroke();

        if (this.image.complete && this.image.naturalWidth > 0) {
            ctx.clip();
            const ratio = Math.max(width / this.image.naturalWidth, height / this.image.naturalHeight);
            const drawW = this.image.naturalWidth * ratio;
            const drawH = this.image.naturalHeight * ratio;
            ctx.drawImage(this.image, x + (width - drawW) / 2, y + (height - drawH) / 2, drawW, drawH);
        } else {
            ctx.fillStyle = "#758399";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            const text = this.imageFailed ? "image unavailable" : (this.imageUrl ? "image loading" : label);
            ctx.fillText(text, x + width / 2, y + height / 2 + 4);
        }
        ctx.restore();
    }

    drawScene(ctx, width, height) {
        const fovRad = this.fov * Math.PI / 180;
        const cx = width / 2;
        const horizon = 92;
        const cameraY = height - 30;
        const subjectY = clamp(cameraY - this.distance * 42, 54, cameraY - 46);
        const subjectScale = clamp(1.1 / Math.max(this.distance, 0.35), 0.34, 1.45) * Math.sqrt(this.scale);
        const subjectW = 82 * subjectScale;
        const subjectH = 92 * subjectScale;
        const halfWidth = clamp(Math.tan(fovRad / 2) * this.distance * 42, 18, width / 2 - 24);

        this.drawFloorGrid(ctx, width, height, horizon);

        ctx.save();
        ctx.fillStyle = "rgba(79, 180, 255, 0.15)";
        ctx.strokeStyle = "#4fb4ff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cameraY);
        ctx.lineTo(cx - halfWidth, subjectY);
        ctx.lineTo(cx + halfWidth, subjectY);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        ctx.strokeStyle = "rgba(255,255,255,0.58)";
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(cx, cameraY);
        ctx.lineTo(cx, subjectY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "rgba(255,207,90,0.18)";
        ctx.strokeStyle = "#ffcf5a";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, subjectY, 12 + subjectW * 0.25, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        this.drawImageCard(ctx, cx - subjectW / 2, subjectY - subjectH / 2, subjectW, subjectH, "target");

        ctx.fillStyle = "#ffcf5a";
        ctx.strokeStyle = "#18110a";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cameraY - 15);
        ctx.lineTo(cx - 18, cameraY + 15);
        ctx.lineTo(cx + 18, cameraY + 15);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#151b24";
        ctx.fillRect(cx - 8, cameraY - 2, 16, 10);

        ctx.fillStyle = "#dbe8fa";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText("Scene rig", 12, 20);
        ctx.fillText(`FOV ${this.fov.toFixed(1)} deg`, 12, 38);
        ctx.fillText(`distance ${this.distance.toFixed(2)}`, 12, 56);
        ctx.textAlign = "right";
        ctx.fillText(`view width ${(2 * this.distance * Math.tan(fovRad / 2)).toFixed(2)}`, width - 12, 20);
        ctx.restore();
    }

    drawPov(ctx, width, height) {
        const fovRad = this.fov * Math.PI / 180;
        const frameX = 18;
        const frameY = 16;
        const frameW = width - 36;
        const frameH = height - 42;
        const cx = frameX + frameW / 2;
        const cy = frameY + frameH / 2;
        const aspect = !this.imageFailed && this.image.complete && this.image.naturalWidth > 0
            ? this.image.naturalWidth / this.image.naturalHeight
            : 0.8;
        // Keep POV preview aligned with Pixal3D manual camera values.
        const focalPx = (frameW / 2) / Math.tan(fovRad / 2);
        const objectWorldWidth = 1.0 * this.scale;
        const objectWorldHeight = objectWorldWidth / clamp(aspect, 0.55, 1.65);
        const projectedW = objectWorldWidth * focalPx / Math.max(this.distance, 0.05);
        const projectedH = objectWorldHeight * focalPx / Math.max(this.distance, 0.05);
        const subjectW = clamp(projectedW, 12, frameW * 3.2);
        const subjectH = clamp(projectedH, 12, frameH * 3.2);
        const visibleScale = projectedW / Math.max(frameW * 0.34, 1);
        const viewWidthWorld = 2 * this.distance * Math.tan(fovRad / 2);

        ctx.save();
        const gradient = ctx.createLinearGradient(0, frameY, 0, frameY + frameH);
        gradient.addColorStop(0, "#18202b");
        gradient.addColorStop(0.58, "#0f151d");
        gradient.addColorStop(1, "#0a0e14");
        ctx.fillStyle = gradient;
        this.pathRoundRect(ctx, frameX, frameY, frameW, frameH, 10);
        ctx.fill();
        ctx.strokeStyle = "#8fcfff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.clip();

        ctx.strokeStyle = "rgba(255,255,255,0.10)";
        ctx.lineWidth = 1;
        const gridWorldHalf = viewWidthWorld / 2;
        for (let i = -4; i <= 4; i++) {
            const x = cx + (i / 4) * focalPx * (gridWorldHalf / Math.max(this.distance, 0.05));
            ctx.beginPath();
            ctx.moveTo(x, frameY);
            ctx.lineTo(x, frameY + frameH);
            ctx.stroke();
        }
        for (let i = 1; i < 3; i++) {
            ctx.beginPath();
            ctx.moveTo(frameX, frameY + frameH * i / 3);
            ctx.lineTo(frameX + frameW, frameY + frameH * i / 3);
            ctx.stroke();
        }

        ctx.fillStyle = "rgba(79,180,255,0.10)";
        ctx.beginPath();
        ctx.ellipse(cx, cy + subjectH * 0.44, subjectW * 0.58, subjectH * 0.09, 0, 0, Math.PI * 2);
        ctx.fill();

        this.drawImageCard(ctx, cx - subjectW / 2, cy - subjectH / 2, subjectW, subjectH, "camera view");

        ctx.strokeStyle = "rgba(255,255,255,0.62)";
        ctx.setLineDash([4, 6]);
        ctx.beginPath();
        ctx.moveTo(cx - 15, cy);
        ctx.lineTo(cx + 15, cy);
        ctx.moveTo(cx, cy - 15);
        ctx.lineTo(cx, cy + 15);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.restore();
        ctx.save();
        ctx.fillStyle = "#dbe8fa";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(`POV FOV ${this.fov.toFixed(1)} deg`, 12, height - 12);
        ctx.textAlign = "right";
        ctx.fillText(`view width ${viewWidthWorld.toFixed(2)} | scale ${visibleScale.toFixed(2)}x`, width - 12, height - 12);
        ctx.restore();
    }

    draw() {
        if (!this.canvas) return;
        this.updateLinkedImage();
        const ctx = this.canvas.getContext("2d");
        if (!ctx) return;
        const width = this.canvas.width;
        const height = this.canvas.height;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#0c1016";
        ctx.fillRect(0, 0, width, height);
        if (this.viewMode === "pov") {
            this.drawPov(ctx, width, height);
        } else {
            this.drawScene(ctx, width, height);
        }
    }
}

app.registerExtension({
    name: "Pixal3D.CameraControl",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onConfigure = nodeType.prototype.onConfigure;
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;

        function hideNativeWidgets(node) {
            for (const widget of node.widgets || []) {
                if (!["fov_degrees", "distance", "mesh_scale"].includes(widget.name)) continue;
                widget.hidden = true;
                widget.computeSize = () => [0, -4];
            }
        }

        function ensureUI(node) {
            hideNativeWidgets(node);
            if (!node.pixal3dCameraUI) {
                node.pixal3dCameraUI = new Pixal3DCameraUI(node);
            } else {
                node.pixal3dCameraUI.syncFromWidgets();
                node.pixal3dCameraUI.draw();
            }
            node.setSize?.([DEFAULT_WIDTH, DEFAULT_HEIGHT]);
        }

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.serialize_widgets = true;
            this.resizable = false;
            ensureUI(this);
            return result;
        };

        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => ensureUI(this), 0);
            return result;
        };

        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            this.pixal3dCameraUI?.updateLinkedImage();
            this.pixal3dCameraUI?.draw();
            return result;
        };

        nodeType.prototype.computeSize = function () {
            return [DEFAULT_WIDTH, DEFAULT_HEIGHT];
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.pixal3dCameraUI?.destroy();
            this.pixal3dCameraUI = null;
            return onRemoved?.apply(this, arguments);
        };
    },
});
