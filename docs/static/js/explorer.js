// Scene explorer: replays one EgoRecall test scene in time. The map shows the points observed so far, the trajectory
// and camera follow the recording, object boxes appear when an object is first seen, and each featured query can be
// opened at the moment it is asked, with its targets, anchors, and every method's answer. Arrows hanging above the
// scene point down at the targets and a chosen method's wrong picks, so they are easy to find among other objects.
//
// The scene's files are written by the maintainers' scene_assets.py; see its docstring for their layout.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

const RECENT_FRAMES = 12; // points first observed within 2 s are tinted orange
const SPEEDS = [1, 4, 16];
const COLOR = {
  background: 0xf4f6f9,
  inView: 0x344054,
  remembered: 0x98a2b3,
  camera: 0xe8671a,
  target: 0x22a55a,
  anchor: 0x8b5cf6,
  wrong: 0xe0483e,
  trajectoryOld: new THREE.Color(0xa9c6ea),
  trajectoryNew: new THREE.Color(0x1d5fb8),
};
const VERDICT = { correct: "correct", partial: "partly correct", wrong: "wrong object", empty: "no answer" };
const AXIS_NAMES = { history: "Observation history", egocentric: "Egocentric", allocentric: "Allocentric" };

// Marker arrows, in metres: the head's length and radius, and the shaft's. All arrows hang at one height, with their
// tips `gap` above the highest object box of the scene.
const ARROW = { head: 0.45, headRadius: 0.2, shaft: 0.8, shaftRadius: 0.06, gap: 0.2 };

// ---------- Small helpers ----------

function formatTime(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

// Number of sorted values less than or equal to x
function upperBound(sorted, x) {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] <= x) low = middle + 1;
    else high = middle;
  }
  return low;
}

// A small dot drawn in CSS for a method's verdict: filled, half-filled, red, or hollow
function verdictDot(verdict, title) {
  const dot = element("span", `verdict verdict-${verdict}`);
  dot.title = title;
  return dot;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.arrayBuffer();
}

// Fetch a whole file as a Blob, calling onProgress(received, total) after each chunk; total is the Content-Length, or 0
// when the server sends none
async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  return new Blob(chunks, { type: response.headers.get("Content-Type") ?? "" });
}

// A progress callback that fills a loading element's bar and adds the percentage to its label; without a known size,
// the bar slides instead
function showProgress(status, label) {
  const text = status.querySelector(".loading-text");
  const bar = status.querySelector(".progress");
  const fill = status.querySelector(".progress-fill");
  return (received, total) => {
    status.classList.toggle("is-indeterminate", !total);
    if (!total) return;
    const percent = Math.min(100, Math.round((100 * received) / total));
    text.textContent = `${label} ${percent}%`;
    fill.style.width = `${percent}%`;
    bar.setAttribute("aria-valuenow", String(percent));
  };
}

// Run a download again when it fails, for example because a connection dropped midway (forwarded ports and tunnels do
// this now and then): three attempts, one and then three seconds apart. onRetry is called before each new attempt, and
// the last failure names the file.
async function withRetries(file, download, onRetry) {
  const delays = [1000, 3000];
  for (let attempt = 0; ; attempt++) {
    try {
      return await download();
    } catch (error) {
      if (attempt === delays.length) throw new Error(`${file} did not download (${error.message})`);
      console.warn(`${file} did not download (${error.message}); trying again`);
      onRetry();
      await new Promise((resolve) => setTimeout(resolve, delays[attempt]));
    }
  }
}

// Replace a loading element's label and bar with the reason it failed and a button that reloads the page
function showFailure(status, message) {
  const button = element("button", "reload-button", "Reload the page");
  button.type = "button";
  button.addEventListener("click", () => location.reload());
  status.classList.remove("is-indeterminate");
  status.replaceChildren(element("span", "loading-text", message), button);
  status.hidden = false;
}

// ---------- Scene data ----------

// Box corners from an oriented box: rows of `axes` are the unit axis vectors, `size` the full side lengths
function boxCorners(object) {
  const [a, b, c, d, e, f, g, h, i] = object.axes;
  const axes = [[a, b, c], [d, e, f], [g, h, i]];
  const corners = [];
  for (let n = 0; n < 8; n++) {
    const signs = [n & 4 ? 1 : -1, n & 2 ? 1 : -1, n & 1 ? 1 : -1];
    const corner = [...object.center];
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < 3; j++) corner[j] += signs[k] * object.size[k] * 0.5 * axes[k][j];
    }
    corners.push(corner);
  }
  return corners;
}

// Height of a box's highest corner
function boxTop(object) {
  return Math.max(...boxCorners(object).map((corner) => corner[2]));
}

// The 12 box edges as flat segment endpoints, for LineSegmentsGeometry
function boxSegments(object) {
  const corners = boxCorners(object);
  const positions = [];
  for (let n = 0; n < 8; n++) {
    for (const bit of [1, 2, 4]) {
      const m = n ^ bit;
      if (m > n) positions.push(...corners[n], ...corners[m]);
    }
  }
  return positions;
}

function boxMatrix(object) {
  const [a, b, c, d, e, f, g, h, i] = object.axes;
  const [sx, sy, sz] = object.size;
  return new THREE.Matrix4().set(
    a * sx, d * sy, g * sz, object.center[0],
    b * sx, e * sy, h * sz, object.center[1],
    c * sx, f * sy, i * sz, object.center[2],
    0, 0, 0, 1,
  );
}

// Whether an object is in view at a frame, and the last frame it was seen at or before that frame
function objectState(object, frame) {
  if (object.first > frame) return { state: "unseen", lastSeen: null };
  let lastSeen = null;
  for (const [start, end] of object.segments) {
    if (start > frame) break;
    if (end >= frame) return { state: "inView", lastSeen: frame };
    lastSeen = end;
  }
  return { state: "remembered", lastSeen };
}

// ---------- The explorer ----------

class Explorer {
  constructor(root) {
    this.root = root;
    this.base = `static/scenes/${root.dataset.scene}/`;
    this.frame = 0;
    this.shownFrame = -1;
    this.playing = false;
    this.speed = 4;
    this.follow = false;
    this.selected = null;
    this.method = null;
    this.lineMaterials = [];
    this.highlightGroup = new THREE.Group();
    this.arrowGroup = new THREE.Group();
    this.needsRender = true;
    this.emptyPanel = root.querySelector("#query-panel").innerHTML;
  }

  // Open a query from elsewhere on the page, loading the scene first if needed
  open(idx) {
    this.root.scrollIntoView({ behavior: "smooth", block: "start" });
    if (this.queryById) this.select(idx);
    else {
      this.pendingQuery = idx;
      this.load();
    }
  }

  async load() {
    if (this.loading) return;
    this.loading = true;
    const status = this.root.querySelector("#viewer-status");
    const label = "Loading the scene…";
    const retrying = () => {
      status.querySelector(".loading-text").textContent = `${label} trying again`;
    };
    try {
      // The scene's files first, with the whole connection to themselves; the camera video downloads after them, while
      // the scene is built (bindVideo)
      this.meta = await withRetries("meta.json", () => fetchJson(this.base + "meta.json"), retrying);
      const [objects, queries, points, trajectory] = await Promise.all([
        withRetries("objects.json", () => fetchJson(this.base + "objects.json"), retrying),
        withRetries("queries.json", () => fetchJson(this.base + "queries.json"), retrying),
        // The points are nearly all of the scene's bytes, so their download stands for the scene's progress
        withRetries(
          "points.bin",
          async () => (await fetchWithProgress(this.base + "points.bin", showProgress(status, label))).arrayBuffer(),
          retrying,
        ),
        withRetries("trajectory.bin", () => fetchBuffer(this.base + "trajectory.bin"), retrying),
      ]);
      this.bindVideo();
      this.objects = objects;
      this.objectById = new Map(objects.map((object) => [object.id, object]));
      this.queries = queries;
      this.queryById = new Map(queries.map((query) => [query.idx, query]));
      this.trajectory = new Float32Array(trajectory);
      this.buildViewer(points);
      this.buildTimeline();
      this.buildQueryList();
      this.setFrame(0);
      status.hidden = true;
      this.animate();
      if (this.pendingQuery !== undefined) this.select(this.pendingQuery);
      const autoplay = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          autoplay.disconnect();
          if (!this.selected && this.frame === 0) this.setPlaying(true);
        }
      }, { threshold: 0.5 });
      autoplay.observe(this.root.querySelector("#viewer"));
    } catch (error) {
      console.error(error);
      const reason = /webgl/i.test(error.message)
        ? "the browser could not start WebGL, which draws the 3D view"
        : error.message.replace(/\.$/, "");
      showFailure(status, `The scene could not be loaded: ${reason}.`);
    }
  }

  // ---------- 3D view ----------

  buildViewer(pointBuffer) {
    const viewer = this.root.querySelector("#viewer");
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(COLOR.background);
    viewer.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.addEventListener("change", () => { this.needsRender = true; });

    this.materials = {
      camera: this.lineMaterial(COLOR.camera, 3),
      target: this.lineMaterial(COLOR.target, 3),
      anchor: this.lineMaterial(COLOR.anchor, 3),
      wrong: this.lineMaterial(COLOR.wrong, 3.5),
      correct: this.lineMaterial(COLOR.target, 6),
    };
    this.buildPoints(pointBuffer);
    this.buildTrajectory();
    this.buildBoxes();
    this.scene.add(this.highlightGroup);
    this.buildArrows();
    this.resetView();

    new ResizeObserver(() => this.resize()).observe(viewer);
    this.resize();
    this.bindPicking(viewer);
  }

  buildPoints(buffer) {
    const count = this.meta.points.count;
    const low = this.meta.points.min;
    const high = this.meta.points.max;
    const quantized = [0, 1, 2].map((axis) => new Uint16Array(buffer, axis * 2 * count, count));
    const firstSeen = new Uint16Array(buffer, 6 * count, count);
    const colors = new Uint8Array(buffer, 8 * count, 3 * count);

    // Undo the 16-bit quantization into interleaved positions
    const positions = new Float32Array(3 * count);
    for (let axis = 0; axis < 3; axis++) {
      const scale = (high[axis] - low[axis]) / 65535;
      const values = quantized[axis];
      for (let n = 0; n < count; n++) positions[3 * n + axis] = low[axis] + values[n] * scale;
    }
    this.firstSeen = firstSeen;

    // One set of attributes, drawn twice: observed points as a prefix of the sorted array, the rest faintly
    const attributes = {
      position: new THREE.BufferAttribute(positions, 3),
      color: new THREE.BufferAttribute(colors, 3, true),
      firstSeen: new THREE.BufferAttribute(new Float32Array(firstSeen), 1),
    };
    this.observedGeometry = new THREE.BufferGeometry();
    this.unobservedGeometry = new THREE.BufferGeometry();
    for (const [name, attribute] of Object.entries(attributes)) {
      this.observedGeometry.setAttribute(name, attribute);
      this.unobservedGeometry.setAttribute(name, attribute);
    }

    this.pointUniforms = {
      uFrame: { value: 0 },
      uRecent: { value: RECENT_FRAMES },
      uSize: { value: this.meta.points.voxel * 1.7 },
      uScale: { value: 500 },
    };
    const vertexShader = `
      attribute float firstSeen;
      attribute vec3 color;
      uniform float uFrame, uRecent, uSize, uScale;
      varying vec3 vColor;
      void main() {
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        gl_PointSize = clamp(uSize * uScale / -viewPosition.z, 1.5, 14.0);
        float age = uFrame - firstSeen;
        float fresh = age < uRecent ? 1.0 - age / uRecent : 0.0;
        vColor = mix(color, vec3(0.91, 0.40, 0.10), 0.65 * fresh);
      }`;
    const observedMaterial = new THREE.ShaderMaterial({
      uniforms: this.pointUniforms,
      vertexShader,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          vec2 offset = gl_PointCoord - 0.5;
          if (dot(offset, offset) > 0.25) discard;
          gl_FragColor = vec4(vColor, 1.0);
        }`,
    });
    const unobservedMaterial = new THREE.ShaderMaterial({
      uniforms: this.pointUniforms,
      vertexShader,
      fragmentShader: `
        void main() {
          vec2 offset = gl_PointCoord - 0.5;
          if (dot(offset, offset) > 0.25) discard;
          gl_FragColor = vec4(0.77, 0.80, 0.84, 0.35);
        }`,
      transparent: true,
      depthWrite: false,
    });
    this.observedPoints = new THREE.Points(this.observedGeometry, observedMaterial);
    this.unobservedPoints = new THREE.Points(this.unobservedGeometry, unobservedMaterial);
    this.observedPoints.frustumCulled = false;
    this.unobservedPoints.frustumCulled = false;
    this.unobservedPoints.visible = false;
    this.scene.add(this.unobservedPoints, this.observedPoints);
  }

  lineMaterial(color, width, options = {}) {
    const material = new LineMaterial({ color, linewidth: width, ...options });
    this.lineMaterials.push(material);
    return material;
  }

  cameraPose(frame) {
    const row = this.trajectory.subarray(7 * frame, 7 * frame + 7);
    return {
      position: new THREE.Vector3(row[0], row[1], row[2]),
      quaternion: new THREE.Quaternion(row[3], row[4], row[5], row[6]),
    };
  }

  buildTrajectory() {
    const frames = this.meta.num_frames;
    const positions = [];
    for (let frame = 0; frame < frames; frame++) positions.push(...this.trajectory.subarray(7 * frame, 7 * frame + 3));
    const geometry = new LineGeometry();
    geometry.setPositions(positions);
    geometry.setColors(new Array(positions.length).fill(0.5));
    this.trajectoryColors = geometry.attributes.instanceColorStart.data;
    this.trajectoryLine = new Line2(geometry, this.lineMaterial(0xffffff, 3, { vertexColors: true }));
    this.trajectoryLine.frustumCulled = false;
    this.scene.add(this.trajectoryLine);

    // Camera frustum in camera coordinates (x right, y down, z forward), placed by the pose at each frame
    const { width, height, fx, fy } = this.meta.camera;
    const depth = 0.35;
    const x = (depth * width) / (2 * fx);
    const y = (depth * height) / (2 * fy);
    const corners = [[-x, -y, depth], [x, -y, depth], [x, y, depth], [-x, y, depth]];
    const segments = [];
    for (let n = 0; n < 4; n++) segments.push(0, 0, 0, ...corners[n], ...corners[n], ...corners[(n + 1) % 4]);
    segments.push(-x * 0.4, -y, depth, 0, -y * 1.4, depth, 0, -y * 1.4, depth, x * 0.4, -y, depth); // "up" notch
    const frustumGeometry = new LineSegmentsGeometry();
    frustumGeometry.setPositions(segments);
    this.cameraFrustum = new LineSegments2(frustumGeometry, this.materials.camera);
    this.cameraFrustum.frustumCulled = false;
    this.scene.add(this.cameraFrustum);
  }

  // Arrows: the one height they all hang at, their shared geometry and materials, and lights that shade only them
  // (the points, boxes, and lines use unlit materials)
  buildArrows() {
    this.arrowTip = Math.max(...this.objects.map((object) => boxTop(object))) + ARROW.gap;
    const sky = new THREE.HemisphereLight(0xffffff, 0x8a94a6, 2.2);
    sky.position.set(0, 0, 1);
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(-3, -5, 8);
    this.scene.add(sky, sun, this.arrowGroup);

    // A downward arrow with its tip at the origin: three.js builds cones and cylinders along +y, so turn them to -z
    const head = new THREE.ConeGeometry(ARROW.headRadius, ARROW.head, 32);
    head.rotateX(-Math.PI / 2);
    head.translate(0, 0, ARROW.head / 2);
    const shaft = new THREE.CylinderGeometry(ARROW.shaftRadius, ARROW.shaftRadius, ARROW.shaft, 16);
    shaft.rotateX(Math.PI / 2);
    shaft.translate(0, 0, ARROW.head + ARROW.shaft / 2);
    this.arrowGeometries = [head, shaft];
    const arrowMaterial = (color) => new THREE.MeshLambertMaterial({
      color,
      emissive: new THREE.Color(color).multiplyScalar(0.25),
    });
    this.arrowMaterials = { target: arrowMaterial(COLOR.target), wrong: arrowMaterial(COLOR.wrong) };
  }

  // Hang an arrow above an object, pointing down at it
  addArrow(object, material) {
    const arrow = new THREE.Group();
    for (const geometry of this.arrowGeometries) arrow.add(new THREE.Mesh(geometry, material));
    arrow.position.set(object.center[0], object.center[1], this.arrowTip);
    this.arrowGroup.add(arrow);
  }

  // The corners of an object's box and the top of the arrow above it, for framing the view
  framePoints(object) {
    const corners = boxCorners(object).map((corner) => new THREE.Vector3(...corner));
    const arrowTop = this.arrowTip + ARROW.head + ARROW.shaft;
    return [...corners, new THREE.Vector3(object.center[0], object.center[1], arrowTop)];
  }

  buildBoxes() {
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    const pickGeometry = new THREE.BoxGeometry(1, 1, 1);
    const pickMaterial = new THREE.MeshBasicMaterial({ visible: false });
    this.boxMaterials = {
      inView: new THREE.LineBasicMaterial({ color: COLOR.inView }),
      remembered: new THREE.LineBasicMaterial({ color: COLOR.remembered, transparent: true, opacity: 0.55 }),
    };
    this.boxes = new Map();
    for (const object of this.objects) {
      const matrix = boxMatrix(object);
      const lines = new THREE.LineSegments(edges, this.boxMaterials.inView);
      const pick = new THREE.Mesh(pickGeometry, pickMaterial);
      for (const node of [lines, pick]) {
        node.matrixAutoUpdate = false;
        node.matrix.copy(matrix);
        node.visible = false;
        node.userData.object = object;
      }
      this.scene.add(lines, pick);
      this.boxes.set(object.id, { lines, pick, state: "unseen" });
    }
  }

  resetView() {
    const low = new THREE.Vector3(...this.meta.points.min);
    const high = new THREE.Vector3(...this.meta.points.max);
    const center = low.clone().add(high).multiplyScalar(0.5);
    const distance = low.distanceTo(high) * 0.95;
    const azimuth = THREE.MathUtils.degToRad(-65);
    const elevation = THREE.MathUtils.degToRad(58);
    this.camera.position.set(
      center.x + distance * Math.cos(elevation) * Math.cos(azimuth),
      center.y + distance * Math.cos(elevation) * Math.sin(azimuth),
      center.z + distance * Math.sin(elevation),
    );
    this.controls.target.copy(center);
    this.controls.update();
    this.needsRender = true;
  }

  resize() {
    const viewer = this.renderer.domElement.parentElement;
    const width = viewer.clientWidth;
    const height = viewer.clientHeight;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    const pixelHeight = height * this.renderer.getPixelRatio();
    this.pointUniforms.uScale.value = pixelHeight / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
    for (const material of this.lineMaterials) material.resolution.set(width, height);
    this.needsRender = true;
  }

  // Hovering over an object's box shows its label and when it was seen
  bindPicking(viewer) {
    const tooltip = viewer.querySelector("#viewer-tooltip");
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    viewer.addEventListener("pointermove", (event) => {
      const box = this.renderer.domElement.getBoundingClientRect();
      pointer.set(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
      raycaster.setFromCamera(pointer, this.camera);
      const candidates = [...this.boxes.values()].filter((entry) => entry.pick.visible).map((entry) => entry.pick);
      const hit = raycaster.intersectObjects(candidates, false)[0];
      if (!hit) {
        tooltip.hidden = true;
        return;
      }
      const object = hit.object.userData.object;
      const frame = Math.floor(this.frame);
      const { state, lastSeen } = objectState(object, frame);
      const fps = this.meta.fps;
      tooltip.textContent = state === "inView"
        ? `${object.label} · in view`
        : `${object.label} · last seen ${formatTime(lastSeen / fps)}, ${Math.round((frame - lastSeen) / fps)} s ago`;
      tooltip.style.left = `${event.clientX - box.left + 14}px`;
      tooltip.style.top = `${event.clientY - box.top + 14}px`;
      tooltip.hidden = false;
    });
    viewer.addEventListener("pointerleave", () => { tooltip.hidden = true; });
  }

  // ---------- Time ----------

  setFrame(value) {
    const last = this.meta.num_frames - 1;
    this.frame = Math.min(Math.max(value, 0), last);
    const frame = Math.floor(this.frame);
    if (frame === this.shownFrame) return;
    this.shownFrame = frame;
    if (this.selected && frame !== this.selected.frame) this.deselect();

    // Map: the points observed by this frame are a prefix of the sorted point array
    const observed = upperBound(this.firstSeen, frame);
    this.observedGeometry.setDrawRange(0, observed);
    this.unobservedGeometry.setDrawRange(observed, this.meta.points.count - observed);
    this.pointUniforms.uFrame.value = frame;

    // Trajectory up to this frame, lighter where older
    const colors = this.trajectoryColors.array;
    const old = COLOR.trajectoryOld;
    const recent = COLOR.trajectoryNew;
    for (let segment = 0; segment < frame; segment++) {
      for (const [offset, fraction] of [[0, segment / frame], [3, (segment + 1) / frame]]) {
        colors[6 * segment + offset] = old.r + (recent.r - old.r) * fraction;
        colors[6 * segment + offset + 1] = old.g + (recent.g - old.g) * fraction;
        colors[6 * segment + offset + 2] = old.b + (recent.b - old.b) * fraction;
      }
    }
    this.trajectoryColors.needsUpdate = true;
    this.trajectoryLine.geometry.instanceCount = frame;

    // Camera at this frame; in follow mode the view moves with it
    const pose = this.cameraPose(frame);
    if (this.follow && this.previousPose) {
      const shift = pose.position.clone().sub(this.previousPose.position);
      this.camera.position.add(shift);
      this.controls.target.add(shift);
    }
    this.previousPose = pose;
    this.cameraFrustum.position.copy(pose.position);
    this.cameraFrustum.quaternion.copy(pose.quaternion);

    // Object boxes: hidden until first seen, solid while in view, faded once remembered
    let seen = 0;
    let inView = 0;
    for (const object of this.objects) {
      const entry = this.boxes.get(object.id);
      const { state } = objectState(object, frame);
      if (state !== "unseen") seen++;
      if (state === "inView") inView++;
      if (state !== entry.state) {
        entry.state = state;
        entry.lines.visible = state !== "unseen";
        entry.pick.visible = state !== "unseen";
        if (state !== "unseen") entry.lines.material = this.boxMaterials[state];
      }
    }

    const seconds = frame / this.meta.fps;
    this.root.querySelector("#viewer-hud").textContent =
      `${formatTime(seconds)} · ${seen} objects seen · ${inView} in view`;
    this.root.querySelector("#explorer-time").textContent =
      `${formatTime(seconds)} / ${formatTime(last / this.meta.fps)}`;
    this.root.querySelector("#frame-time").textContent = formatTime(seconds);
    this.scrubber.value = String(frame);
    this.seekVideo(frame);
    this.needsRender = true;
  }

  animate() {
    let previous = performance.now();
    const tick = (now) => {
      const elapsed = Math.min((now - previous) / 1000, 0.1);
      previous = now;
      // Time moves only once the camera frames are ready, so the camera panel never shows a stale or empty frame; a
      // play request made earlier starts then
      if (this.playing && this.framesReady) {
        const next = this.frame + elapsed * this.meta.fps * this.speed;
        if (next >= this.meta.num_frames - 1) this.setPlaying(false);
        this.setFrame(next);
      }
      if (this.controls.update()) this.needsRender = true;
      if (this.needsRender) {
        this.renderer.render(this.scene, this.camera);
        this.needsRender = false;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  setPlaying(playing) {
    this.playing = playing;
    const button = this.root.querySelector("#explorer-play");
    button.textContent = playing ? "❚❚" : "▶";
    button.setAttribute("aria-label", playing ? "Pause" : "Play");
    if (playing && this.frame >= this.meta.num_frames - 1) this.setFrame(0);
  }

  // ---------- Egocentric video: seek to the shown frame, one seek at a time ----------

  bindVideo() {
    this.video = this.root.querySelector("#explorer-video");
    this.video.poster = this.base + this.meta.video.poster;
    this.video.addEventListener("seeked", () => {
      this.seeking = false;
      if (this.pendingFrame !== null) this.seekVideo(this.pendingFrame);
    });
    this.pendingFrame = null;

    // Fetch the whole video once, so seeking works on any web server, including ones without range requests. The camera
    // panel shows the download's progress until the first frame is decoded, and playback waits for it; if the video
    // cannot be loaded, the scene plays without it.
    const status = this.root.querySelector("#frame-status");
    const label = "Loading the camera frames…";
    const text = status.querySelector(".loading-text");
    text.textContent = label;
    const file = this.meta.video.file;
    withRetries(file, () => fetchWithProgress(this.base + file, showProgress(status, label)), () => {
      text.textContent = `${label} trying again`;
    })
      .then((blob) => {
        this.videoUrl = URL.createObjectURL(blob);
        this.video.addEventListener("loadeddata", () => {
          this.seeking = false;
          this.seekVideo(Math.floor(this.frame));
          this.framesReady = true;
          status.hidden = true;
        }, { once: true });
        this.video.src = this.videoUrl;
      })
      .catch((error) => {
        console.error(error);
        showFailure(status, `The camera frames could not be loaded: ${error.message}. The scene plays without them.`);
        this.framesReady = true;
      });
  }

  seekVideo(frame) {
    if (!this.videoUrl) return;
    if (this.seeking && performance.now() - this.seekStarted < 1000) {
      this.pendingFrame = frame;
      return;
    }
    this.pendingFrame = null;
    this.seeking = true;
    this.seekStarted = performance.now();
    this.video.currentTime = (frame + 0.5) / this.meta.fps;
  }

  // ---------- Timeline controls ----------

  buildTimeline() {
    this.scrubber = this.root.querySelector("#explorer-scrubber");
    this.scrubber.max = String(this.meta.num_frames - 1);
    this.scrubber.addEventListener("input", () => {
      this.setPlaying(false);
      this.setFrame(Number(this.scrubber.value));
    });
    this.root.querySelector("#explorer-play").addEventListener("click", () => this.setPlaying(!this.playing));

    const speeds = this.root.querySelector("#explorer-speeds");
    for (const speed of SPEEDS) {
      const button = element("button", speed === this.speed ? "is-active" : "", `${speed}×`);
      button.type = "button";
      button.addEventListener("click", () => {
        this.speed = speed;
        speeds.querySelectorAll("button").forEach((other) => other.classList.toggle("is-active", other === button));
      });
      speeds.append(button);
    }

    // Marks on the timeline for the featured queries
    const markers = this.root.querySelector("#explorer-markers");
    for (const query of this.queries) {
      const marker = element("button", "marker");
      marker.type = "button";
      marker.style.left = `${(100 * query.frame) / (this.meta.num_frames - 1)}%`;
      marker.title = `“${query.text}” at ${formatTime(query.frame / this.meta.fps)}`;
      marker.addEventListener("click", () => this.select(query.idx));
      markers.append(marker);
      query.marker = marker;
    }

    const tools = this.root.querySelector("#viewer-tools");
    tools.querySelector("[data-tool=follow]").addEventListener("click", (event) => {
      this.follow = !this.follow;
      event.currentTarget.classList.toggle("is-active", this.follow);
    });
    tools.querySelector("[data-tool=unobserved]").addEventListener("click", (event) => {
      this.unobservedPoints.visible = !this.unobservedPoints.visible;
      event.currentTarget.classList.toggle("is-active", this.unobservedPoints.visible);
      this.needsRender = true;
    });
    tools.querySelector("[data-tool=reset]").addEventListener("click", () => this.resetView());

    // Space plays or pauses and the arrow keys step one second while the explorer has focus
    this.root.addEventListener("keydown", (event) => {
      if (event.target.closest("input, select, button")) return;
      if (event.key === " ") {
        event.preventDefault();
        this.setPlaying(!this.playing);
      } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        this.setPlaying(false);
        this.setFrame(Math.floor(this.frame) + (event.key === "ArrowRight" ? 1 : -1) * this.meta.fps);
      }
    });
  }

  // ---------- Queries ----------

  buildQueryList() {
    const key = this.root.querySelector("#method-key");
    const entries = element("span", "key-entries");
    for (const [verdict, label] of Object.entries(VERDICT)) {
      const entry = element("span", "key-entry");
      entry.append(verdictDot(verdict, label), label);
      entries.append(entry);
    }
    const methods = this.meta.methods.map((method) => method.name).join(", ");
    key.append(element("span", "", `Answers in each row, left to right: ${methods}.`), entries);
    const list = this.root.querySelector("#query-list");
    for (const query of this.queries) list.append(this.queryRow(query));
  }

  queryRow(query) {
    const row = element("button", "query-row");
    row.type = "button";
    row.dataset.idx = String(query.idx);
    row.classList.toggle("is-selected", this.selected === query);
    const time = element("span", "query-time", formatTime(query.frame / this.meta.fps));
    const text = element("span", "query-text", `“${query.text}”`);
    const tags = element("span", "query-tags");
    for (const name of query.axes) {
      const dot = element("span", `axis-dot axis-${name}`);
      dot.title = AXIS_NAMES[name];
      tags.append(dot);
    }
    tags.append(element("span", "query-status", targetSummary(query, this.meta.fps)));
    const verdicts = element("span", "query-verdicts");
    for (const method of this.meta.methods) {
      const verdict = query.methods[method.key].verdict;
      verdicts.append(verdictDot(verdict, `${method.name}: ${VERDICT[verdict]}`));
    }
    row.append(time, text, tags, verdicts);
    row.addEventListener("click", () => this.select(query.idx));
    return row;
  }

  select(idx) {
    const query = this.queryById.get(idx);
    this.selected = query;
    this.method = null;
    this.setPlaying(false);
    this.setFrame(query.frame);
    this.root.querySelectorAll(".query-row").forEach((row) => {
      row.classList.toggle("is-selected", Number(row.dataset.idx) === idx);
    });
    this.root.querySelectorAll(".marker").forEach((marker) => marker.classList.remove("is-selected"));
    if (query.marker) query.marker.classList.add("is-selected");
    this.showQuery();
    this.drawHighlights();
    this.focusOn(query);
  }

  deselect() {
    this.selected = null;
    this.method = null;
    this.root.querySelectorAll(".query-row.is-selected, .marker.is-selected").forEach((node) => {
      node.classList.remove("is-selected");
    });
    this.root.querySelector("#query-panel").innerHTML = this.emptyPanel;
    this.drawHighlights();
  }

  // Frame the camera at the query time and the boxes and arrows of the query's targets, anchors, and any other given
  // objects, keeping the viewing direction
  focusOn(query, others = []) {
    const points = [this.cameraPose(query.frame).position];
    for (const oid of [...query.targets, ...query.anchors, ...others]) {
      points.push(...this.framePoints(this.objectById.get(oid)));
    }
    const bounds = new THREE.Box3().setFromPoints(points);
    const center = bounds.getCenter(new THREE.Vector3());
    const distance = Math.max(5, bounds.getSize(new THREE.Vector3()).length() * 1.6);
    const direction = this.camera.position.clone().sub(this.controls.target).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.needsRender = true;
  }

  drawHighlights() {
    for (const child of [...this.highlightGroup.children]) {
      child.geometry.dispose();
      this.highlightGroup.remove(child);
    }
    this.arrowGroup.clear();
    this.needsRender = true;
    const query = this.selected;
    if (!query) return;
    const addBox = (object, material) => {
      const geometry = new LineSegmentsGeometry();
      geometry.setPositions(boxSegments(object));
      this.highlightGroup.add(new LineSegments2(geometry, material));
    };
    for (const oid of query.anchors) addBox(this.objectById.get(oid), this.materials.anchor);
    for (const oid of query.targets) {
      addBox(this.objectById.get(oid), this.materials.target);
      this.addArrow(this.objectById.get(oid), this.arrowMaterials.target);
    }

    // The chosen method's picks: thicker green boxes when right, red boxes and arrows when wrong. Objects outside the
    // annotations, such as walls, have no box.
    if (this.method) {
      for (const oid of answerPicks(query.methods[this.method], query)) {
        const object = this.objectById.get(oid);
        if (!object) continue;
        if (query.targets.includes(oid)) addBox(object, this.materials.correct);
        else {
          addBox(object, this.materials.wrong);
          this.addArrow(object, this.arrowMaterials.wrong);
        }
      }
    }
  }

  // Refit the view when part of a pick's box or arrow is off screen
  showPicks(query) {
    const picks = answerPicks(query.methods[this.method], query).filter((oid) => this.objectById.has(oid));
    this.camera.updateMatrixWorld();
    const offScreen = picks.some((oid) => this.framePoints(this.objectById.get(oid)).some((point) => {
      point.project(this.camera);
      return Math.abs(point.x) > 0.95 || Math.abs(point.y) > 0.95 || point.z > 1;
    }));
    if (offScreen) this.focusOn(query, picks);
  }

  showQuery() {
    const query = this.selected;
    const panel = this.root.querySelector("#query-panel");
    panel.replaceChildren();
    const fps = this.meta.fps;

    panel.append(element("p", "panel-label", `Query ${query.idx} · asked at ${formatTime(query.frame / fps)}`));
    panel.append(element("p", "panel-text", `“${query.text}”`));
    panel.append(element("code", "panel-program", query.program));

    // Targets, with how long each remembered one has been out of view, and anchors
    const facts = element("ul", "panel-facts");
    const label = (oid) => this.objectById.get(oid).label;
    for (const oid of query.targets) {
      const item = element("li");
      item.append(element("span", "swatch swatch-target"));
      const lastSeen = query.last_seen[String(oid)];
      item.append(lastSeen === undefined
        ? `Target: ${label(oid)}, in view`
        : `Target: ${label(oid)}, out of view for ${Math.round((query.frame - lastSeen) / fps)} s`);
      facts.append(item);
    }
    for (const oid of query.anchors) {
      const item = element("li");
      item.append(element("span", "swatch swatch-anchor"), `Anchor: ${label(oid)}`);
      facts.append(item);
    }
    if (query.any_of) facts.append(element("li", "panel-note", "Any one of the targets is a correct answer."));
    panel.append(facts);

    // Each method's answer; choosing one draws its picks and describes them
    panel.append(element("p", "panel-label", "Method answers"));
    const chips = element("div", "method-chips");
    const prompt = element("p", "method-detail", "Choose a method to see the objects it picked.");
    let detail = prompt;
    for (const method of this.meta.methods) {
      const answer = query.methods[method.key];
      const chip = element("button", "method-chip");
      chip.type = "button";
      chip.append(verdictDot(answer.verdict, VERDICT[answer.verdict]), method.name);
      chip.title = `${method.family}: ${VERDICT[answer.verdict]}`;
      chip.addEventListener("click", () => {
        this.method = this.method === method.key ? null : method.key;
        chips.querySelectorAll(".method-chip").forEach((other) => {
          other.classList.toggle("is-active", other === chip && this.method);
        });
        const next = this.method ? describeAnswer(answer, query, this.objectById) : prompt;
        detail.replaceWith(next);
        detail = next;
        this.drawHighlights();
        if (this.method) this.showPicks(query);

        // Scroll the panel so the description is in sight
        const overflow = detail.getBoundingClientRect().bottom - panel.getBoundingClientRect().bottom + 14;
        if (overflow > 0) panel.scrollBy({ top: overflow, behavior: "smooth" });
      });
      chips.append(chip);
    }
    panel.append(chips, detail);
  }
}

function targetVisibility(query) {
  if (!query.hidden.length) return "visible";
  return query.visible.length ? "mixed" : "hidden";
}

function targetSummary(query, fps) {
  const kind = targetVisibility(query);
  if (kind === "visible") return "in view";
  if (kind === "mixed") return "partly in view";
  const stalest = Math.max(...Object.values(query.last_seen).map((frame) => query.frame - frame));
  return `out of view ${Math.round(stalest / fps)} s`;
}

// The objects a method's answer points at. Any-of queries are scored by the top-ranked pick alone; otherwise the
// predicted set, or the top-ranked guess when no object passed the method's score threshold.
function answerPicks(answer, query) {
  const top = answer.top1 === null ? [] : [answer.top1];
  if (query.any_of) return top;
  return answer.predicted.length ? answer.predicted : top;
}

// Labels such as "shoes" or "bottles" name several items and take "other" rather than "another"
function isPlural(label) {
  return /[^su]s$/.test(label);
}

// Name a pick so that a wrong object of a target's category reads as another one: "the target cabinet", "another
// cabinet, not the target", "the anchor box", or just the label of an object of another category
function pickPhrase(oid, query, objectById) {
  if (!objectById.has(oid)) return "an unannotated object or surface";
  const label = objectById.get(oid).label;
  const sameLabel = (oids) => oids.filter((other) => objectById.get(other).label === label).length;
  const role = (name, count) => {
    if (count === 1) return `the ${name} ${label}`;
    return isPlural(label) ? `${name} ${label}` : `${/^[aeiou]/.test(name) ? "an" : "a"} ${name} ${label}`;
  };
  if (query.targets.includes(oid)) return role("target", sameLabel(query.targets));
  if (query.anchors.includes(oid)) return role("anchor", sameLabel(query.anchors));
  const targets = sameLabel(query.targets);
  if (!targets) return label;
  return `${isPlural(label) ? "other" : "another"} ${label}, not ${targets === 1 ? "the" : "a"} target`;
}

// A method's answer in words, each pick marked right or wrong, with any missed targets and unmatched answers
function describeAnswer(answer, query, objectById) {
  const detail = element("div", "method-detail");
  const line = (...parts) => {
    const paragraph = element("p");
    paragraph.append(...parts);
    detail.append(paragraph);
  };
  const pick = (oid) => {
    const right = query.targets.includes(oid);
    const item = element("span", "pick", `${pickPhrase(oid, query, objectById)}\u00a0`);
    item.append(element("span", right ? "mark mark-right" : "mark mark-wrong", right ? "✓" : "✗"));
    return item;
  };

  if (query.any_of) {
    if (answer.top1 === null) line("No answer.");
    else line("Top-ranked pick: ", pick(answer.top1));
    return detail;
  }

  const { predicted } = answer;
  if (predicted.length === 1) line("Picked: ", pick(predicted[0]));
  else if (predicted.length > 1) {
    line("Picked:");
    const list = element("ul", "pick-list");
    for (const oid of predicted) {
      const item = element("li");
      item.append(pick(oid));
      list.append(item);
    }
    detail.append(list);
  } else if (answer.top1 !== null) {
    line("No object passed its score threshold.");
    line("Top-ranked guess: ", pick(answer.top1));
  } else if (!answer.unresolved) {
    line(answer.covered ? "No answer." : "No answer; no target was among its candidates.");
  }

  if (answer.unresolved === 1) {
    line(predicted.length
      ? "Plus one answer that matched no annotated object."
      : "Its answer matched no annotated object.");
  } else if (answer.unresolved) {
    line(predicted.length
      ? `Plus ${answer.unresolved} answers that matched no annotated object.`
      : `Its ${answer.unresolved} answers matched no annotated object.`);
  }
  const missed = query.targets.filter((oid) => !predicted.includes(oid));
  if (missed.length && missed.length < query.targets.length) {
    line(`Missed: ${missed.map((oid) => pickPhrase(oid, query, objectById)).join(", ")}.`);
  }
  return detail;
}

// ---------- Start when the explorer is about to scroll into view ----------

const root = document.getElementById("explorer");
if (root) {
  const explorer = new Explorer(root);
  document.querySelectorAll("[data-open-query]").forEach((node) => {
    const open = () => explorer.open(Number(node.dataset.openQuery));
    node.addEventListener("click", open);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      observer.disconnect();
      explorer.load();
    }
  }, { rootMargin: "600px 0px" });
  observer.observe(root);
}
