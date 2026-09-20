import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const envelope = {
  throttle_lo: 0.0, throttle_hi: 1.0,
  pitch_lo: -8.0, pitch_hi: 12.0,
  bank_abs: 25.0, rudder_abs: 1.0,
};

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

let ws = null;
let sessionId = null;
let epoch = 0;
let seq = 0;
let padIndex = null;
let reconnecting = false;
let profileHash = "default-v1";
let deadzoneFraction = 0.05;
let curve = "linear";
let lastApplied = { throttle: 0, pitch_deg: 0, bank_deg: 0, rudder: 0 };
let lastPending = "NONE";
let lastAuthority = "AUTO";
let snapUntil = 0;
let cameraMode = "chase";
let cameraReady = false;
let observerPosition = null;
let cameraTarget = null;
let deckMesh = null;

const keys = new Set();
const cmd = { throttle: 0.0, pitch_deg: 0.0, bank_deg: 0.0, rudder: 0.0 };

let renderer = null;
let scene = null;
let camera = null;
let vehicleMesh = null;
let seaMesh = null;
let seaParent = null;
let oceanMesh = null;
let lastVehiclePos = { x: 0, y: 0, z: 0 };
let lastHeadingRad = 0;
let lastAttitude = { heading_deg: 0, pitch_deg: 0, bank_deg: 0 };
let lastTelemetry = null;
let osdCanvas = null;
let osdCtx = null;
let osdDpr = 1;

let patchSize = 0;
let patchSpacing = 0;
let patchHeights = null;
let patchCentre = { x: 0, z: 0 };

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function setBadge(id, text, cls) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.className = "badge" + (cls ? " " + cls : "");
}

function appendEvent(obj) {
  const ev = $("events");
  const row = document.createElement("div");
  row.className = "row";
  const simt = obj.sim_time_s !== undefined ? obj.sim_time_s.toFixed(2) : "-";
  const kind = obj.kind || obj.type || "?";
  const time = document.createElement("span");
  time.className = "sim_t";
  time.textContent = simt;
  const label = document.createElement("span");
  label.className = "kind";
  label.textContent = kind;
  row.append(time, label);
  ev.prepend(row);
  while (ev.childElementCount > 60) ev.removeChild(ev.lastChild);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return r.json();
}

async function createSession() {
  const cap = await api("/api/capabilities");
  envelope.throttle_lo = cap.control_envelope.throttle_lo;
  envelope.throttle_hi = cap.control_envelope.throttle_hi;
  envelope.pitch_lo = cap.control_envelope.pitch_lo;
  envelope.pitch_hi = cap.control_envelope.pitch_hi;
  envelope.bank_abs = cap.control_envelope.bank_abs;
  envelope.rudder_abs = cap.control_envelope.rudder_abs;
  const r = await api("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seed: 42, scenario: "hybrid" }),
  });
  sessionId = r.session_id;
  epoch = r.epoch;
  setText("status", "session=" + sessionId.slice(0, 8));
  return sessionId;
}

function openWS() {
  if (!sessionId) {
    statusEl.innerHTML = `<span class="err">no session id</span>`;
    return;
  }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/sessions/${sessionId}`;
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    statusEl.innerHTML = "connected, awaiting hello";
  };
  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      handleBinaryFrame(new Uint8Array(ev.data));
      return;
    }
    let obj;
    try { obj = JSON.parse(ev.data); } catch (_) { return; }
    if (obj.type === "hello") {
      epoch = obj.epoch;
      if (obj.initial_telemetry) {
        lastTelemetry = obj.initial_telemetry;
        updateInstruments(obj.initial_telemetry);
        updateScene(obj.initial_telemetry);
        updateSticks(obj.initial_telemetry);
        drawOsd(obj.initial_telemetry);
      }
      statusEl.innerHTML = `<span class="ok">connected (epoch ${epoch})</span>`;
    } else if (obj.type === "telemetry") {
      lastTelemetry = obj;
      updateInstruments(obj);
      updateScene(obj);
      updateSticks(obj);
      drawOsd(obj);
    } else if (obj.type === "ack") {
      const id = String(obj.event_id || "");
      if (!id.startsWith("input-")) {
        appendEvent({
          sim_time_s: 0,
          kind: `ack:${id}:${obj.accepted ? "ok" : `no (${obj.reason || "拒否"})`}`,
        });
      }
    } else if (obj.type === "event") {
      appendEvent(obj);
    }
  };
  ws.onclose = () => {
    statusEl.innerHTML = `<span class="err">disconnected, retrying…</span>`;
    setTimeout(reconnect, 1500);
  };
  ws.onerror = () => {
    statusEl.innerHTML = `<span class="err">ws error, retrying…</span>`;
  };
}

function handleBinaryFrame(bytes) {
  let idx = -1;
  for (let i = 0; i < bytes.length; i += 1) {
    if (bytes[i] === 10) { idx = i; break; }
  }
  if (idx < 0) return;
  const headerText = new TextDecoder().decode(bytes.subarray(0, idx));
  let meta;
  try { meta = JSON.parse(headerText); } catch (_) { return; }
  if (meta.type !== "wave_patch" || meta.session_id !== sessionId || meta.epoch !== epoch) return;
  const bin = bytes.subarray(idx + 1);
  const view = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  if (bin.byteLength < 19) return;
  const version = view.getUint8(0);
  if (version !== 1) return;
  const size = view.getUint16(13, true);
  const spacing = view.getFloat32(15, true);
  const expected = 19 + size * size * 4;
  if (size < 2 || size > 257 || !Number.isFinite(spacing) || spacing <= 0 || bin.byteLength !== expected) return;
  const north = view.getFloat32(5, true);
  const east = view.getFloat32(9, true);
  if (!Number.isFinite(north) || !Number.isFinite(east)) return;
  // The 19-byte header and variable JSON prefix are not float-aligned.
  // DataView also makes the wire's little-endian order explicit.
  const heights = new Float32Array(size * size);
  for (let i = 0; i < heights.length; i += 1) {
    heights[i] = view.getFloat32(19 + i * 4, true);
    if (!Number.isFinite(heights[i])) return;
  }
  patchSize = size;
  patchSpacing = spacing;
  patchHeights = heights;
  patchCentre = { x: east, z: -north };
  updateSeaMesh();
}

function updateInstruments(t) {
  setBadge("lifecycle", t.lifecycle);
  setBadge("authority", t.authority, t.authority);
  setBadge("phase", t.phase);
  const pending = t.pending || "NONE";
  setBadge("pending", pending, pending === "OFFER_MANUAL" ? "OFFER" : "");
  lastPending = pending;
  lastAuthority = t.authority;
  if (t.applied_control) lastApplied = t.applied_control;
  setText("applied_values", `適用: thr ${lastApplied.throttle.toFixed(2)} · pit ${lastApplied.pitch_deg.toFixed(1)}° · bank ${lastApplied.bank_deg.toFixed(1)}° · rud ${lastApplied.rudder.toFixed(2)}`);
  const alt = t.position_neu_m.z;
  const vx = t.velocity_neu_m_s.x;
  const vz = t.velocity_neu_m_s.z;
  setText("i_alt", alt.toFixed(2));
  setText("i_vx", vx.toFixed(2));
  setText("i_vz", vz.toFixed(2));
  setText("i_bank", t.attitude.bank_deg.toFixed(1));
  setText("i_heading", t.attitude.heading_deg.toFixed(1));
  setText("i_keel", t.wave_clearance_m.toFixed(2));
  setText("i_simt", t.sim_time_s.toFixed(2));
  setText("i_tick", String(t.tick));
  $("btn_start").disabled = t.lifecycle !== "READY";
  $("btn_manual").disabled = t.lifecycle !== "READY";
  $("btn_pause").disabled = t.lifecycle !== "RUNNING";
  $("btn_resume").disabled = t.lifecycle !== "PAUSED";
  $("btn_land").disabled = t.lifecycle !== "RUNNING" || t.authority !== "HUMAN";
  $("btn_abort").disabled = ["FINISHED", "ABORTED"].includes(t.lifecycle);
  const takeBtn = $("btn_take");
  if (takeBtn) {
    takeBtn.disabled = t.lifecycle !== "RUNNING" || t.authority === "HUMAN";
    takeBtn.classList.toggle("primary", pending === "OFFER_MANUAL");
  }
}

function fillBar(id, frac) {
  const el = $(id);
  if (!el) return;
  const span = el.querySelector("span");
  if (span) span.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
}

function updateSticks(t) {
  const pitSpan = envelope.pitch_hi - envelope.pitch_lo || 20;
  fillBar("b_thr", cmd.throttle);
  fillBar("b_pit", (cmd.pitch_deg - envelope.pitch_lo) / pitSpan);
  fillBar("b_ban", 0.5 + 0.5 * (cmd.bank_deg / envelope.bank_abs));
  fillBar("b_rud", 0.5 + 0.5 * (cmd.rudder / envelope.rudder_abs));
  setText("v_thr", cmd.throttle.toFixed(2));
  setText("v_pit", cmd.pitch_deg.toFixed(1));
  setText("v_ban", cmd.bank_deg.toFixed(1));
  setText("v_rud", cmd.rudder.toFixed(2));
  const ms = $("match_status");
  if (!ms) return;
  if (t.lifecycle !== "RUNNING") {
    ms.textContent = t.lifecycle === "READY" ? "開始方法を選択してください" :
      t.lifecycle === "PAUSED" ? "一時停止中 — Resume で再開" : "セッション終了";
    ms.className = "hint";
    return;
  }
  if (t.authority === "HUMAN") {
    ms.textContent = "HUMAN 操縦中";
    ms.className = "hint match";
    return;
  }
  if (t.pending === "OFFER_MANUAL") {
    const dThr = Math.abs(cmd.throttle - lastApplied.throttle);
    const dPit = Math.abs(cmd.pitch_deg - lastApplied.pitch_deg);
    const dBan = Math.abs(cmd.bank_deg - lastApplied.bank_deg);
    const dRud = Math.abs(cmd.rudder - lastApplied.rudder);
    const ok = dThr <= 0.05 && dPit <= 2.0 && dBan <= 3.0 && dRud <= 0.1;
    ms.textContent = ok
      ? "スティック一致 — 0.5 秒保持で引継ぎ"
      : `合わせる: thr ${lastApplied.throttle.toFixed(2)}  pit ${lastApplied.pitch_deg.toFixed(1)}°  ban ${lastApplied.bank_deg.toFixed(1)}°`;
    ms.className = "hint " + (ok ? "match" : "nomatch");
    return;
  }
  ms.textContent = "AUTO: スティックは機体へ適用されません。Take Control で手動へ";
  ms.className = "hint";
}

function makeFlyingBoat() {
  const g = new THREE.Group();
  const hullMat = new THREE.MeshStandardMaterial({
    color: 0xc8963a, roughness: 0.5, metalness: 0.15,
  });
  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xd9b15a, roughness: 0.55,
  });
  const darkMat = new THREE.MeshStandardMaterial({
    color: 0x2c3540, roughness: 0.35, metalness: 0.45,
  });
  const hull = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.75, 4.4), hullMat);
  hull.position.y = 0.1;
  g.add(hull);
  const bow = new THREE.Mesh(new THREE.ConeGeometry(0.55, 1.4, 6), hullMat);
  bow.rotation.x = -Math.PI / 2;
  bow.position.set(0, 0.1, -2.6);
  g.add(bow);
  const wing = new THREE.Mesh(new THREE.BoxGeometry(15.0, 0.14, 1.6), wingMat);
  wing.position.set(0, 0.55, -0.2);
  g.add(wing);
  const floatL = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.22, 1.8), hullMat);
  floatL.position.set(-3.4, -0.15, 0.2);
  g.add(floatL);
  const floatR = floatL.clone();
  floatR.position.x = 3.4;
  g.add(floatR);
  const hstab = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.08, 0.55), wingMat);
  hstab.position.set(0, 0.45, 2.05);
  g.add(hstab);
  const vstab = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.05, 0.7), wingMat);
  vstab.position.set(0, 0.95, 2.05);
  g.add(vstab);
  const canopy = new THREE.Mesh(
    new THREE.SphereGeometry(0.42, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2),
    darkMat,
  );
  canopy.position.set(0, 0.55, -0.6);
  canopy.scale.set(1.1, 0.7, 1.3);
  g.add(canopy);
  return g;
}

function initScene() {
  const canvas = document.getElementById("scene");
  if (!canvas) return;
  const container = document.getElementById("scene-container");

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x8fb7d4);
  scene.fog = new THREE.Fog(0x8fb7d4, 120, 2200);

  const sizeScene = () => {
    const w = container.clientWidth || canvas.clientWidth || 600;
    const h = container.clientHeight || canvas.clientHeight || 400;
    renderer.setSize(w, h, false);
    if (camera) {
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    }
    sizeOsd();
  };

  camera = new THREE.PerspectiveCamera(70, 1, 0.5, 8000);
  camera.position.set(0, 16, 36);
  sizeScene();

  scene.add(new THREE.HemisphereLight(0xe8f2ff, 0x1a3040, 0.85));
  const dir = new THREE.DirectionalLight(0xfff4dd, 0.75);
  dir.position.set(40, 80, 20);
  scene.add(dir);

  const skyGeo = new THREE.SphereGeometry(2800, 32, 16);
  const skyMat = new THREE.MeshBasicMaterial({
    color: 0x8fb7d4, side: THREE.BackSide, fog: false,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));

  const sun = new THREE.Mesh(
    new THREE.SphereGeometry(28, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xfff1c2, fog: false }),
  );
  sun.position.set(200, 420, -900);
  scene.add(sun);

  const oceanGeo = new THREE.CircleGeometry(2200, 64);
  oceanGeo.rotateX(-Math.PI / 2);
  const oceanMat = new THREE.MeshStandardMaterial({
    color: 0x1a4d73, roughness: 0.55, metalness: 0.05,
  });
  oceanMesh = new THREE.Mesh(oceanGeo, oceanMat);
  oceanMesh.position.y = -0.15;
  scene.add(oceanMesh);

  vehicleMesh = makeFlyingBoat();
  scene.add(vehicleMesh);

  cameraTarget = new THREE.Vector3();
  let lastFrame = performance.now();
  const animate = (now) => {
    const dt = Math.min(0.1, (now - lastFrame) / 1000);
    lastFrame = now;
    updateCamera(dt);
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
  requestAnimationFrame(() => {
    sizeScene();
    renderer.render(scene, camera);
  });
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => {
      sizeScene();
      if (renderer && scene && camera) renderer.render(scene, camera);
    });
    ro.observe(container);
  }
}

function updateScene(t) {
  if (!renderer || !vehicleMesh) return;
  const pos = t.position_neu_m;
  const sceneX = pos.y;
  const sceneY = pos.z;
  const sceneZ = -pos.x;

  lastVehiclePos.x = sceneX;
  lastVehiclePos.y = sceneY;
  lastVehiclePos.z = sceneZ;
  vehicleMesh.position.set(sceneX, sceneY, sceneZ);

  const headingRad = THREE.MathUtils.degToRad(t.attitude.heading_deg);
  const pitchRad = THREE.MathUtils.degToRad(t.attitude.pitch_deg);
  const bankRad = THREE.MathUtils.degToRad(t.attitude.bank_deg);
  lastHeadingRad = headingRad;
  lastAttitude = {
    heading_deg: t.attitude.heading_deg,
    pitch_deg: t.attitude.pitch_deg,
    bank_deg: t.attitude.bank_deg,
  };
  vehicleMesh.rotation.set(pitchRad, -headingRad, -bankRad, "YXZ");

  if (oceanMesh) oceanMesh.position.set(sceneX, -0.15, sceneZ);

  if (!observerPosition) {
    observerPosition = new THREE.Vector3(sceneX + 22, 3.2, sceneZ + 25);
    // A fixed launch vessel: deck and rails make the operator's location legible.
    deckMesh = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({ color: 0x59646a });
    const deck = new THREE.Mesh(new THREE.BoxGeometry(8, 0.4, 14), material);
    deck.position.set(observerPosition.x, 1.2, observerPosition.z + 4);
    deckMesh.add(deck);
    for (const x of [-3.8, 3.8]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 14), material);
      rail.position.set(observerPosition.x + x, 2.3, observerPosition.z + 4);
      deckMesh.add(rail);
    }
    scene.add(deckMesh);
  }
}

function updateCamera(dt) {
  if (!vehicleMesh || !cameraTarget || !observerPosition) return;
  const pos = vehicleMesh.position;
  const fov = Number($("camera_zoom")?.value || 60);
  if (cameraMode === "fpv") {
    // Rigid nose camera: no smoothing so the horizon tracks attitude 1:1.
    const pose = fpvCameraPose(pos.x, pos.y, pos.z,
      lastAttitude.heading_deg, lastAttitude.pitch_deg, lastAttitude.bank_deg, FPV_OFFSET);
    camera.position.set(pose.position.x, pose.position.y, pose.position.z);
    camera.up.set(pose.up.x, pose.up.y, pose.up.z);
    camera.lookAt(pos.x + pose.forward.x * 100,
      pos.y + pose.forward.y * 100, pos.z + pose.forward.z * 100);
    cameraTarget.copy(pos);
    cameraReady = true;
    if (camera.fov !== fov) { camera.fov = fov; camera.updateProjectionMatrix(); }
    return;
  }
  camera.up.set(0, 1, 0);
  const desired = cameraMode === "observer" ? observerPosition.clone() :
    new THREE.Vector3(-Math.sin(lastHeadingRad) * 32, 12, Math.cos(lastHeadingRad) * 32).add(pos);
  const target = pos.clone();
  // Keep the aircraft in frame at every altitude; retain world-up during turns.
  const alpha = cameraReady ? 1 - Math.exp(-dt * 4) : 1;
  camera.position.lerp(desired, alpha);
  cameraTarget.lerp(target, alpha);
  camera.lookAt(cameraTarget);
  cameraReady = true;
  if (camera.fov !== fov) {
    camera.fov = fov;
    camera.updateProjectionMatrix();
  }
}

// ---- FPV nose camera + canvas OSD ----
// Canopy mount in model space: nose points along -Z, so offset is up and forward.
const FPV_OFFSET = { x: 0, y: 0.62, z: -0.9 };
const OSD_PITCH_PX_PER_DEG = 5;

// Pure: replicate vehicleMesh.rotation.set(pitchRad,-headingRad,-bankRad,"YXZ")
// i.e. R = Ry(h)·Rx(p)·Rz(b) with h=-heading, p=pitch, b=-bank. Returns world
// camera position, forward (-Z column) and up (+Y column) as plain vectors.
function fpvCameraPose(px, py, pz, headingDeg, pitchDeg, bankDeg, offset) {
  const h = -headingDeg * Math.PI / 180;
  const p = pitchDeg * Math.PI / 180;
  const b = -bankDeg * Math.PI / 180;
  const ch = Math.cos(h), sh = Math.sin(h);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cb = Math.cos(b), sb = Math.sin(b);
  const m00 = ch * cb + sh * sp * sb;
  const m01 = -ch * sb + sh * sp * cb;
  const m02 = sh * cp;
  const m10 = cp * sb;
  const m11 = cp * cb;
  const m12 = -sp;
  const m20 = -sh * cb + ch * sp * sb;
  const m21 = sh * sb + ch * sp * cb;
  const m22 = ch * cp;
  const ox = offset.x, oy = offset.y, oz = offset.z;
  return {
    position: {
      x: px + m00 * ox + m01 * oy + m02 * oz,
      y: py + m10 * ox + m11 * oy + m12 * oz,
      z: pz + m20 * ox + m21 * oy + m22 * oz,
    },
    forward: { x: -m02, y: -m12, z: -m22 },
    up: { x: m01, y: m11, z: m21 },
  };
}

// Pure: horizon geometry for the OSD. roll rotates the canvas so a right bank
// (bank>0) drops the right horizon; pitchOffset shifts the horizon down on nose-up.
function osdHorizonGeometry(w, h, pitchDeg, bankDeg) {
  const roll = -bankDeg * Math.PI / 180;
  const pitchOffset = pitchDeg * OSD_PITCH_PX_PER_DEG;
  const rungs = [];
  for (let p = -60; p <= 60; p += 10) {
    if (p === 0) continue;
    rungs.push({ pitch: p, dy: pitchOffset - p * OSD_PITCH_PX_PER_DEG });
  }
  return { cx: w / 2, cy: h / 2, roll, pitchOffset, rungs };
}

// Pure: vertical tape ticks. dy>0 places smaller values below centre.
function osdTicks(center, step, pxPerUnit, spanPx) {
  const half = spanPx / 2;
  const out = [];
  const start = Math.ceil((center - half) / step) * step;
  for (let v = start; v <= center + half + 1e-9; v += step) {
    out.push({ value: v, dy: (center - v) * pxPerUnit });
  }
  return out;
}

// Pure: heading strip ticks every 10°, labels wrapped to [0,360).
function osdHeadingTicks(headingDeg, pxPerDeg, spanPx) {
  const half = spanPx / 2;
  const out = [];
  const start = Math.ceil((headingDeg - half / pxPerDeg) / 10) * 10;
  for (let d = start; (d - headingDeg) * pxPerDeg <= half + 1e-9; d += 10) {
    out.push({ label: ((d % 360) + 360) % 360, dx: (d - headingDeg) * pxPerDeg });
  }
  return out;
}

// Pure: derive warning strings from a telemetry frame.
function osdWarnings(t) {
  const out = [];
  const dmg = t.damage || {};
  if (dmg.failed) out.push(`FAILURE ${dmg.failure_reason || ""}`.trim());
  if ((dmg.water_kg || 0) > 0.5) out.push(`FLOODING ${dmg.water_kg.toFixed(1)}kg`);
  if (Number.isFinite(t.wave_clearance_m) && t.wave_clearance_m < 0.35
      && t.position_neu_m && t.position_neu_m.z > 1.0) {
    out.push("LOW CLEARANCE");
  }
  if (["PAUSED", "FINISHED", "ABORTED"].includes(t.lifecycle)) out.push(t.lifecycle);
  for (const w of t.warnings || []) out.push(String(w));
  return out;
}

function initOsd() {
  osdCanvas = document.getElementById("osd");
  if (!osdCanvas) return;
  osdCtx = osdCanvas.getContext("2d");
  osdDpr = Math.min(window.devicePixelRatio || 1, 2);
  sizeOsd();
}

function sizeOsd() {
  if (!osdCanvas) return;
  const container = document.getElementById("scene-container");
  const w = (container && container.clientWidth) || osdCanvas.clientWidth || 600;
  const h = (container && container.clientHeight) || osdCanvas.clientHeight || 400;
  osdCanvas.width = Math.round(w * osdDpr);
  osdCanvas.height = Math.round(h * osdDpr);
  osdCanvas.style.width = w + "px";
  osdCanvas.style.height = h + "px";
  if (lastTelemetry) drawOsd(lastTelemetry);
}

function applyCameraMode(mode) {
  cameraMode = mode;
  cameraReady = false;
  if (vehicleMesh) vehicleMesh.visible = mode !== "fpv";
  if (osdCanvas) osdCanvas.style.display = mode === "fpv" ? "block" : "none";
  if (mode === "fpv" && lastTelemetry) drawOsd(lastTelemetry);
}

const OSD_GREEN = "rgba(120,255,150,0.9)";
const OSD_DIM = "rgba(120,255,150,0.55)";
const OSD_WARN = "rgba(255,120,90,0.95)";

function drawOsd(t) {
  if (!osdCtx || cameraMode !== "fpv" || !t) return;
  const w = osdCanvas.width / osdDpr;
  const h = osdCanvas.height / osdDpr;
  const ctx = osdCtx;
  ctx.save();
  ctx.setTransform(osdDpr, 0, 0, osdDpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "12px monospace";
  ctx.textBaseline = "middle";

  // Artificial horizon + pitch ladder + roll pointer.
  const g = osdHorizonGeometry(w, h, t.attitude.pitch_deg, t.attitude.bank_deg);
  const span = Math.hypot(w, h);
  ctx.save();
  ctx.translate(g.cx, g.cy);
  ctx.rotate(g.roll);
  ctx.strokeStyle = OSD_GREEN;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(-span / 2, g.pitchOffset);
  ctx.lineTo(span / 2, g.pitchOffset);
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.fillStyle = OSD_DIM;
  ctx.textAlign = "center";
  for (const r of g.rungs) {
    const wide = r.pitch % 30 === 0;
    const half = wide ? 34 : 20;
    ctx.beginPath();
    ctx.moveTo(-half, r.dy);
    ctx.lineTo(half, r.dy);
    ctx.stroke();
    if (wide) {
      ctx.fillText(String(r.pitch), half + 14, r.dy);
      ctx.fillText(String(r.pitch), -half - 14, r.dy);
    }
  }
  // Roll pointer at top of the rotated frame.
  const rp = Math.min(w, h) * 0.34;
  ctx.fillStyle = OSD_GREEN;
  ctx.beginPath();
  ctx.moveTo(0, -rp + 12);
  ctx.lineTo(-7, -rp);
  ctx.lineTo(7, -rp);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Fixed aircraft symbol at screen centre.
  ctx.strokeStyle = OSD_GREEN;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(g.cx - 26, g.cy);
  ctx.lineTo(g.cx - 8, g.cy);
  ctx.moveTo(g.cx + 8, g.cy);
  ctx.lineTo(g.cx + 26, g.cy);
  ctx.moveTo(g.cx, g.cy - 4);
  ctx.lineTo(g.cx, g.cy + 4);
  ctx.stroke();

  // Left airspeed tape, right altitude tape.
  drawTape(ctx, 70, g.cy, t.airspeed_m_s, 5, 4, h * 0.5, "m/s", 0);
  drawTape(ctx, w - 70, g.cy, t.position_neu_m.z, 5, 4, h * 0.5, "m", 1);

  // Top heading strip.
  drawHeadingStrip(ctx, g.cx, 30, t.attitude.heading_deg, w * 0.5);

  // Throttle bar (bottom-left) and keel clearance (bottom-centre).
  const thr = (t.applied_control && t.applied_control.throttle) || 0;
  const barH = h * 0.22;
  const bx = 24, by = h - 40 - barH;
  ctx.strokeStyle = OSD_DIM;
  ctx.lineWidth = 1;
  ctx.strokeRect(bx, by, 12, barH);
  ctx.fillStyle = OSD_GREEN;
  ctx.fillRect(bx, by + barH * (1 - thr), 12, barH * thr);
  ctx.textAlign = "left";
  ctx.fillStyle = OSD_DIM;
  ctx.fillText("THR", bx - 4, by - 12);

  ctx.textAlign = "center";
  ctx.fillStyle = OSD_GREEN;
  ctx.fillText(`KEEL ${t.wave_clearance_m.toFixed(2)} m`, g.cx, h - 24);

  // Phase / authority (top-left) and warnings (top-centre).
  ctx.textAlign = "left";
  ctx.fillStyle = OSD_DIM;
  ctx.fillText(`${t.lifecycle} · ${t.phase} · ${t.authority}`, 16, 54);
  const warns = osdWarnings(t);
  ctx.textAlign = "center";
  ctx.fillStyle = OSD_WARN;
  warns.slice(0, 4).forEach((msg, i) => ctx.fillText(msg, g.cx, 70 + i * 16));

  ctx.restore();
}

function drawTape(ctx, x, cy, center, step, pxPerUnit, spanPx, unit, digits) {
  const ticks = osdTicks(center, step, pxPerUnit, spanPx);
  ctx.strokeStyle = OSD_DIM;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, cy - spanPx / 2);
  ctx.lineTo(x, cy + spanPx / 2);
  ctx.stroke();
  ctx.fillStyle = OSD_GREEN;
  ctx.textAlign = "center";
  for (const tk of ticks) {
    const y = cy + tk.dy;
    const major = Math.round(tk.value) % (step * 2) === 0;
    ctx.beginPath();
    ctx.moveTo(x - (major ? 10 : 5), y);
    ctx.lineTo(x, y);
    ctx.stroke();
    if (major) ctx.fillText(tk.value.toFixed(digits), x + 16, y);
  }
  ctx.fillStyle = OSD_GREEN;
  ctx.fillRect(x - 22, cy - 9, 44, 18);
  ctx.fillStyle = "#04120a";
  ctx.fillText(center.toFixed(digits), x, cy);
  ctx.fillStyle = OSD_DIM;
  ctx.fillText(unit, x, cy + spanPx / 2 + 14);
}

function drawHeadingStrip(ctx, cx, y, headingDeg, spanPx) {
  const pxPerDeg = 2;
  const ticks = osdHeadingTicks(headingDeg, pxPerDeg, spanPx);
  ctx.strokeStyle = OSD_DIM;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - spanPx / 2, y);
  ctx.lineTo(cx + spanPx / 2, y);
  ctx.stroke();
  ctx.fillStyle = OSD_GREEN;
  ctx.textAlign = "center";
  for (const tk of ticks) {
    const x = cx + tk.dx;
    const major = tk.label % 30 === 0;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + (major ? 8 : 4));
    ctx.stroke();
    if (major) ctx.fillText(String(tk.label), x, y - 10);
  }
  ctx.fillStyle = OSD_GREEN;
  ctx.beginPath();
  ctx.moveTo(cx, y + 12);
  ctx.lineTo(cx - 5, y + 20);
  ctx.lineTo(cx + 5, y + 20);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#04120a";
  ctx.fillText(Math.round(((headingDeg % 360) + 360) % 360) + "°", cx, y + 34);
}

function updateSeaMesh() {
  if (!scene || !patchHeights || patchSize === 0) return;
  const extent = patchSpacing * (patchSize - 1);
  if (seaMesh && seaMesh.userData.size === patchSize && seaMesh.userData.spacing === patchSpacing) {
    const pos = seaMesh.geometry.attributes.position;
    for (let i = 0; i < patchSize; i += 1) {
      for (let j = 0; j < patchSize; j += 1) {
        pos.setY(i * patchSize + j, patchHeights[(patchSize - 1 - i) * patchSize + j]);
      }
    }
    pos.needsUpdate = true;
    seaMesh.geometry.computeVertexNormals();
    if (seaParent) seaParent.position.set(patchCentre.x, 0, patchCentre.z);
    return;
  }
  if (seaParent) scene.remove(seaParent);
  if (seaMesh) {
    seaMesh.geometry.dispose();
    seaMesh.material.dispose();
  }
  const geom = new THREE.PlaneGeometry(extent, extent, patchSize - 1, patchSize - 1);
  geom.rotateX(-Math.PI / 2);
  const pos = geom.attributes.position;
  for (let i = 0; i < patchSize; i += 1) {
    for (let j = 0; j < patchSize; j += 1) {
      pos.setY(i * patchSize + j, patchHeights[(patchSize - 1 - i) * patchSize + j]);
    }
  }
  pos.needsUpdate = true;
  geom.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x1f5a86, roughness: 0.38, metalness: 0.12,
    side: THREE.DoubleSide, polygonOffset: true,
    polygonOffsetFactor: -1, polygonOffsetUnits: -1,
  });
  seaMesh = new THREE.Mesh(geom, mat);
  seaMesh.userData.size = patchSize;
  seaMesh.userData.spacing = patchSpacing;
  seaParent = new THREE.Group();
  seaParent.add(seaMesh);
  seaParent.position.set(patchCentre.x, 0, patchCentre.z);
  scene.add(seaParent);
}

function sendEvent(kind, extra = {}) {
  if (!ws || ws.readyState !== 1) return;
  const id = `${kind}-${Date.now()}`;
  const msg = {
    v: 1, type: "event", event_id: id,
    kind, session_id: sessionId, epoch,
    sim_time_s: 0, payload: extra,
  };
  if (kind === "resume") msg.confirm = true;
  ws.send(JSON.stringify(msg));
}

function applyResponseCurve(x, kind) {
  if (kind === "gentle") return 0.5 * (1 - Math.cos(Math.PI * x));
  if (kind === "aggressive") return x * x * (3 - 2 * x);
  return x;
}

function calibrateAxis(raw, dz) {
  if (Math.abs(raw) < dz) return 0.0;
  const sign = raw < 0 ? -1 : 1;
  const mag = (Math.abs(raw) - dz) / (1 - dz);
  return sign * applyResponseCurve(Math.min(1, mag), curve);
}

function clampCmd() {
  cmd.throttle = Math.min(envelope.throttle_hi, Math.max(envelope.throttle_lo, cmd.throttle));
  cmd.pitch_deg = Math.min(envelope.pitch_hi, Math.max(envelope.pitch_lo, cmd.pitch_deg));
  cmd.bank_deg = Math.min(envelope.bank_abs, Math.max(-envelope.bank_abs, cmd.bank_deg));
  cmd.rudder = Math.min(envelope.rudder_abs, Math.max(-envelope.rudder_abs, cmd.rudder));
}

function readPad() {
  if (padIndex === null) return null;
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  const pad = pads[padIndex];
  if (!pad) return null;
  const leftY = (pad.axes[1] || 0) * ($("pitch_invert")?.checked ? -1 : 1);
  const leftX = pad.axes[0] || 0;
  const rightX = pad.axes[2] || 0;
  const thrRaw = pad.buttons[7] ? pad.buttons[7].value : 0;
  setText("pad_status", `${pad.id} / ${pad.mapping || "非標準配列"}`);
  setText("pad_raw", `LY ${leftY.toFixed(2)} · LX ${leftX.toFixed(2)} · RX ${rightX.toFixed(2)} · RT ${thrRaw.toFixed(2)}`);
  return {
    throttle: Math.min(1, Math.max(0, thrRaw)),
    pitch_deg: calibrateAxis(leftY, deadzoneFraction) *
      (leftY >= 0 ? envelope.pitch_hi : -envelope.pitch_lo),
    bank_deg: calibrateAxis(leftX, deadzoneFraction) * envelope.bank_abs,
    rudder: calibrateAxis(rightX, deadzoneFraction) * envelope.rudder_abs,
  };
}

function stepKeyboard(dt) {
  const pitRate = 18.0;
  const thrRate = 0.55;
  const banRate = 40.0;
  const rudRate = 1.6;
  if (keys.has("w")) cmd.pitch_deg += pitRate * dt;
  if (keys.has("s")) cmd.pitch_deg -= pitRate * dt;
  if (keys.has("e") || keys.has("arrowup")) cmd.throttle += thrRate * dt;
  if (keys.has("d") || keys.has("arrowdown")) cmd.throttle -= thrRate * dt;
  if (keys.has("a") || keys.has("arrowleft")) cmd.bank_deg -= banRate * dt;
  if (keys.has("f") || keys.has("arrowright")) cmd.bank_deg += banRate * dt;
  if (keys.has("q")) cmd.rudder -= rudRate * dt;
  if (keys.has("c")) cmd.rudder += rudRate * dt;
  if (!keys.has("a") && !keys.has("f") && !keys.has("arrowleft") && !keys.has("arrowright")) {
    cmd.bank_deg *= Math.max(0, 1 - 4 * dt);
    if (Math.abs(cmd.bank_deg) < 0.2) cmd.bank_deg = 0;
  }
  if (!keys.has("q") && !keys.has("c")) {
    cmd.rudder *= Math.max(0, 1 - 4 * dt);
    if (Math.abs(cmd.rudder) < 0.02) cmd.rudder = 0;
  }
  clampCmd();
}

function currentControl() {
  if (performance.now() < snapUntil && lastAuthority === "AUTO") {
    Object.assign(cmd, lastApplied);
    return {
      throttle: lastApplied.throttle,
      pitch_deg: lastApplied.pitch_deg,
      bank_deg: lastApplied.bank_deg,
      rudder: lastApplied.rudder,
    };
  }
  const pad = readPad();
  if (pad && $("input_source")?.value !== "keyboard") {
    cmd.throttle = pad.throttle;
    cmd.pitch_deg = pad.pitch_deg;
    cmd.bank_deg = pad.bank_deg;
    cmd.rudder = pad.rudder;
    clampCmd();
  }
  return {
    throttle: cmd.throttle,
    pitch_deg: cmd.pitch_deg,
    bank_deg: cmd.bank_deg,
    rudder: cmd.rudder,
  };
}

function sendInput() {
  if (!ws || ws.readyState !== 1) return;
  const ctrl = currentControl();
  seq += 1;
  ws.send(JSON.stringify({
    v: 1, type: "input", session_id: sessionId, epoch,
    seq, profile_hash: profileHash, control: ctrl,
  }));
}

function pollPad() {
  if (padIndex === null && navigator.getGamepads) {
    const pads = navigator.getGamepads();
    for (let i = 0; i < pads.length; i += 1) {
      if (pads[i]?.connected) { padIndex = i; break; }
    }
  }
}

async function reconnect() {
  if (reconnecting) return;
  reconnecting = true;
  try {
    // Keep the same flight and recording; the server issues a new epoch.
    openWS();
  } catch (e) {
    statusEl.innerHTML = `<span class="err">reconnect failed: ${e.message}</span>`;
    setTimeout(reconnect, 3000);
  } finally {
    reconnecting = false;
  }
}

window.addEventListener("gamepadconnected", (e) => {
  padIndex = e.gamepad.index;
  statusEl.innerHTML = `<span class="ok">pad connected: ${e.gamepad.id}</span>`;
});

window.addEventListener("gamepaddisconnected", (e) => {
  if (e.gamepad.index !== padIndex) return;
  padIndex = null;
  setText("pad_status", "未接続 — コントローラーのボタンを押してください");
  statusEl.innerHTML = `<span class="err">pad disconnected</span>`;
});

window.addEventListener("keydown", (e) => {
  if (e.target?.matches("input, select, textarea, [contenteditable]")) return;
  const k = e.key.toLowerCase();
  keys.add(k);
  if (["arrowup", "arrowdown", "arrowleft", "arrowright", " "].includes(k)) {
    e.preventDefault();
  }
});
window.addEventListener("keyup", (e) => {
  keys.delete(e.key.toLowerCase());
});

window.addEventListener("blur", () => keys.clear());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) keys.clear();
});

$("camera_mode").onchange = (e) => {
  applyCameraMode(e.target.value);
};
$("btn_start").onclick = () => sendEvent("start_takeoff");
$("btn_manual").onclick = () => sendEvent("start_manual");
$("btn_take").onclick = () => {
  cmd.throttle = lastApplied.throttle;
  cmd.pitch_deg = lastApplied.pitch_deg;
  cmd.bank_deg = lastApplied.bank_deg;
  cmd.rudder = lastApplied.rudder;
  snapUntil = performance.now() + 700;
  sendEvent("take_control");
};
$("btn_land").onclick = () => sendEvent("request_auto_land");
$("btn_pause").onclick = () => sendEvent("pause");
$("btn_resume").onclick = () => sendEvent("resume", { confirm: true });
$("btn_abort").onclick = () => sendEvent("abort");
$("dz").oninput = (e) => {
  deadzoneFraction = Number(e.target.value) / 100;
  $("dz_val").textContent = e.target.value + "%";
};
$("curve").onchange = (e) => { curve = e.target.value; };

(async () => {
  try {
    initScene();
    initOsd();
    await createSession();
    openWS();
  } catch (e) {
    statusEl.innerHTML = `<span class="err">${e.message}</span>`;
  }
})();

let lastPoll = performance.now();
setInterval(() => {
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastPoll) / 1000);
  lastPoll = now;
  stepKeyboard(dt);
  pollPad();
  sendInput();
}, 50);

setInterval(() => {
  if (!ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({
    v: 1, type: "ping", session_id: sessionId, epoch, sim_time_s: 0,
  }));
}, 8000);
