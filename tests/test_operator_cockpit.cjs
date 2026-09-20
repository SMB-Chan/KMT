// Run with: node --test tests/test_operator_cockpit.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/operator/cockpit.js'), 'utf8')
  .replace(/^import .*;\n/, '').split('window.addEventListener("gamepadconnected"')[0];
function cockpit() {
  const elements = new Map();
  const context = vm.createContext({ TextDecoder, document: { getElementById(id) {
    if (!elements.has(id)) elements.set(id, { classList: { toggle() {} } });
    return elements.get(id);
  } } });
  vm.runInContext(source + '\nsessionId = "test"; epoch = 1;', context);
  return { context, elements };
}
function message(padding = '') {
  const prefix = Buffer.from(JSON.stringify({ type: 'wave_patch', session_id: 'test', epoch: 1, padding }) + '\n');
  const bin = Buffer.alloc(19 + 16);
  bin.writeUInt8(1); bin.writeFloatLE(12, 5); bin.writeFloatLE(34, 9);
  bin.writeUInt16LE(2, 13); bin.writeFloatLE(5, 15);
  [1, -2, 3.5, 4].forEach((h, i) => bin.writeFloatLE(h, 19 + i * 4));
  return Buffer.concat([prefix, bin]);
}
test('wave decoding accepts all byte alignments and reads LE heights and centre', () => {
  for (let i = 0; i < 4; i++) {
    const { context } = cockpit();
    context.bytes = message('x'.repeat(i));
    vm.runInContext('handleBinaryFrame(bytes)', context);
    assert.equal(vm.runInContext('JSON.stringify(Array.from(patchHeights))', context), '[1,-2,3.5,4]');
    assert.equal(vm.runInContext('JSON.stringify(patchCentre)', context), '{"x":34,"z":-12}');
  }
});
test('invalid or stale waves do not replace the scene data', () => {
  for (const corrupt of ['truncated', 'nan', 'epoch']) {
    const { context } = cockpit();
    let bytes = message();
    if (corrupt === 'truncated') bytes = bytes.subarray(0, bytes.length - 1);
    if (corrupt === 'nan') bytes.writeFloatLE(NaN, bytes.length - 4);
    if (corrupt === 'epoch') vm.runInContext('epoch = 2', context);
    context.bytes = bytes;
    vm.runInContext('handleBinaryFrame(bytes)', context);
    assert.equal(vm.runInContext('patchHeights', context), null);
  }
});
test('running manual flight disables start and enables landing; pause enables resume', () => {
  const { context, elements } = cockpit();
  context.telemetry = { lifecycle: 'RUNNING', authority: 'HUMAN', phase: 'TAKEOFF', position_neu_m: { z: 5 }, velocity_neu_m_s: { x: 10, z: 0 }, attitude: { bank_deg: 0, heading_deg: 0 }, wave_clearance_m: 5, sim_time_s: 1, tick: 20 };
  vm.runInContext('updateInstruments(telemetry)', context);
  assert.equal(elements.get('btn_manual').disabled, true);
  assert.equal(elements.get('btn_land').disabled, false);
  context.telemetry.lifecycle = 'PAUSED';
  vm.runInContext('updateInstruments(telemetry)', context);
  assert.equal(elements.get('btn_resume').disabled, false);
  assert.equal(elements.get('btn_pause').disabled, true);
});
test('handover matching retains the latest automatic command after snap expires', () => {
  const { context } = cockpit();
  context.performance = { now: () => 100 };
  vm.runInContext('snapUntil = 200; lastApplied = { throttle: 0.42, pitch_deg: 3, bank_deg: 2, rudder: 0.1 }; currentControl()', context);
  context.performance.now = () => 300;
  assert.equal(vm.runInContext('JSON.stringify(currentControl())', context), '{"throttle":0.42,"pitch_deg":3,"bank_deg":2,"rudder":0.1}');
});
test('reconnect opens the existing session without creating a new flight', async () => {
  const { context } = cockpit();
  const urls = [];
  context.location = { protocol: 'http:', host: 'localhost:8766' };
  context.WebSocket = class { constructor(url) { urls.push(url); } };
  context.fetch = () => { throw Error('must not create a session'); };
  await vm.runInContext('reconnect()', context);
  assert.deepEqual(urls, ['ws://localhost:8766/ws/sessions/test']);
});
test('Xbox axes reach all three control channels with asymmetric pitch limits', () => {
  const { context } = cockpit();
  context.navigator = { getGamepads: () => [{ id: 'Xbox', mapping: 'standard', axes: [0.6, 1, -0.8], buttons: Array.from({length: 8}, () => ({value: 0.3})) }] };
  vm.runInContext('padIndex = 0', context);
  const ctrl = vm.runInContext('readPad()', context);
  assert.equal(ctrl.pitch_deg, 12);
  assert.ok(ctrl.bank_deg > 0);
  assert.ok(ctrl.rudder < 0);
  context.navigator.getGamepads = () => [{ id: 'Xbox', mapping: 'standard', axes: [-1, -1, 1], buttons: [] }];
  const other = vm.runInContext('readPad()', context);
  assert.equal(other.pitch_deg, -8);
  assert.equal(other.bank_deg, -25);
  assert.equal(other.rudder, 1);
});
test('FPV camera pose: level aircraft looks north from the canopy offset', () => {
  const { context } = cockpit();
  context.args = { px: 3, py: 10, pz: -7, h: 0, p: 0, b: 0 };
  const pose = vm.runInContext(
    'fpvCameraPose(args.px,args.py,args.pz,args.h,args.p,args.b,FPV_OFFSET)', context);
  assert.ok(Math.abs(pose.position.x - 3) < 1e-9);
  assert.ok(Math.abs(pose.position.y - (10 + 0.62)) < 1e-9);
  assert.ok(Math.abs(pose.position.z - (-7 - 0.9)) < 1e-9);
  assert.ok(Math.abs(pose.forward.x) < 1e-9);
  assert.ok(Math.abs(pose.forward.y) < 1e-9);
  assert.ok(Math.abs(pose.forward.z + 1) < 1e-9);
  assert.ok(Math.abs(pose.up.y - 1) < 1e-9);
});
test('FPV camera pose: heading 90 looks east, pitch tilts forward up', () => {
  const { context } = cockpit();
  context.a = { h: 90 };
  const east = vm.runInContext('fpvCameraPose(0,0,0,a.h,0,0,FPV_OFFSET).forward', context);
  assert.ok(Math.abs(east.x - 1) < 1e-9);
  assert.ok(Math.abs(east.z) < 1e-9);
  context.a = { p: 30 };
  const up = vm.runInContext('fpvCameraPose(0,0,0,0,a.p,0,FPV_OFFSET).forward', context);
  assert.ok(Math.abs(up.y - Math.sin(30 * Math.PI / 180)) < 1e-9);
});
test('FPV camera pose: forward and up stay orthonormal under combined attitude', () => {
  const { context } = cockpit();
  context.a = { h: 37, p: 12, b: -25 };
  const pose = vm.runInContext('fpvCameraPose(0,0,0,a.h,a.p,a.b,FPV_OFFSET)', context);
  const dot = pose.forward.x * pose.up.x + pose.forward.y * pose.up.y + pose.forward.z * pose.up.z;
  const nf = Math.hypot(pose.forward.x, pose.forward.y, pose.forward.z);
  const nu = Math.hypot(pose.up.x, pose.up.y, pose.up.z);
  assert.ok(Math.abs(dot) < 1e-9);
  assert.ok(Math.abs(nf - 1) < 1e-9);
  assert.ok(Math.abs(nu - 1) < 1e-9);
});
test('OSD warnings surface failure, flooding, low clearance, lifecycle, and server notes', () => {
  const { context } = cockpit();
  context.t = {
    damage: { failed: true, water_kg: 1.25, failure_reason: 'hull breach' },
    wave_clearance_m: 0.1, position_neu_m: { z: 8 }, lifecycle: 'PAUSED',
    warnings: ['stall'],
  };
  const w = vm.runInContext('JSON.stringify(osdWarnings(t))', context);
  assert.equal(w, JSON.stringify(['FAILURE hull breach', 'FLOODING 1.3kg', 'LOW CLEARANCE', 'PAUSED', 'stall']));
  context.t2 = {
    damage: { failed: false, water_kg: 0 }, wave_clearance_m: 5,
    position_neu_m: { z: 30 }, lifecycle: 'RUNNING', warnings: [],
  };
  assert.equal(vm.runInContext('JSON.stringify(osdWarnings(t2))', context), '[]');
  // Low clearance is suppressed while sitting on the water (z <= 1).
  context.t3 = {
    damage: {}, wave_clearance_m: 0.05, position_neu_m: { z: 0.3 },
    lifecycle: 'RUNNING', warnings: [],
  };
  assert.equal(vm.runInContext('JSON.stringify(osdWarnings(t3))', context), '[]');
});
test('OSD horizon centres at zero attitude and shifts 5px per degree of pitch', () => {
  const { context } = cockpit();
  context.a = { w: 800, h: 400 };
  const lvl = vm.runInContext('osdHorizonGeometry(a.w,a.h,0,0)', context);
  assert.equal(lvl.cx, 400);
  assert.equal(lvl.cy, 200);
  assert.ok(Math.abs(lvl.roll) < 1e-12);
  assert.ok(Math.abs(lvl.pitchOffset) < 1e-12);
  assert.ok(!lvl.rungs.some((r) => r.pitch === 0));
  const up = vm.runInContext('osdHorizonGeometry(a.w,a.h,10,0)', context);
  assert.equal(up.pitchOffset, 50);
  const bank = vm.runInContext('osdHorizonGeometry(a.w,a.h,0,20)', context);
  assert.ok(bank.roll < 0);
});
test('OSD tapes place the centre value at dy 0 and smaller values below', () => {
  const { context } = cockpit();
  const ticks = vm.runInContext('osdTicks(10,5,4,40)', context);
  const centre = ticks.find((t) => t.value === 10);
  assert.equal(centre.dy, 0);
  const below = ticks.find((t) => t.value === 0);
  assert.equal(below.dy, 40);
});
test('OSD heading strip wraps labels through north', () => {
  const { context } = cockpit();
  const ticks = vm.runInContext('osdHeadingTicks(355,2,40)', context);
  assert.ok(ticks.some((t) => t.label === 0));
  assert.ok(ticks.every((t) => t.label >= 0 && t.label < 360));
  assert.ok(ticks.every((t) => Math.abs(t.dx) <= 20 + 1e-9));
});
