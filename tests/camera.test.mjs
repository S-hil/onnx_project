import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { suggestThresholds } from "../dist/guidance.mjs";
import { readFile } from "node:fs/promises";
import { measure, classify } from "../dist/metrics.mjs";
const source = (
  await readFile(new URL("../dist/app.js", import.meta.url), "utf8")
).replace(/^import .*;\n/gm, "");
const flush = () => new Promise((resolve) => setImmediate(resolve));
function setup(getUserMedia) {
  const elements = new Map(),
    events = {};
  const context2d = {
    scale() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {},
    fillText() {},
    arc() {},
    fill() {},
    drawImage() {},
    getImageData() {
      return { data: new Uint8ClampedArray(320 * 240 * 4).fill(100) };
    },
  };
  const element = () => ({
    value: "",
    textContent: "",
    disabled: false,
    hidden: false,
    clientWidth: 600,
    currentTime: 0,
    videoWidth: 640,
    videoHeight: 480,
    readyState: 4,
    style: {},
    append() {},
    replaceChildren() {},
    getContext() {
      return context2d;
    },
    play: async () => {},
    pause() {},
    removeAttribute() {},
    addEventListener() {},
    removeEventListener() {},
  });
  const doc = {
    hidden: false,
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element,
    addEventListener(n, f) {
      events[n] = f;
    },
  };
  for (const [id, value] of Object.entries({
    brightness: "45",
    detail: "60",
    interval: "1",
  }))
    doc.getElementById(id).value = value;
  const sandbox = {
    document: doc,
    window: {
      devicePixelRatio: 1,
      addEventListener(n, f) {
        events[n] = f;
      },
    },
    navigator: { mediaDevices: { getUserMedia } },
    performance,
    Float64Array,
    Float32Array,
    Uint8ClampedArray,
    URL,
    Blob,
    console,
    setTimeout: () => 1,
    clearTimeout() {},
    suggestThresholds,
    measure,
    classify,
    initialize: async () => {},
    inferLearned: async (samples) => samples.map(() => [0.1, 0.2, 0.7]),
    infer: async (samples, b, d) =>
      samples.map((s) => 100 * Math.min(1, s.brightness / b, s.detail / d)),
  };
  vm.runInNewContext(source, sandbox);
  return { elements, events, doc };
}
function makeStream() {
  let stopped = 0;
  const track = {
    stop() {
      stopped++;
    },
    addEventListener() {},
  };
  return {
    stream: { getTracks: () => [track], getVideoTracks: () => [track] },
    stopped: () => stopped,
  };
}
test("live mode requests video only, measures a frame, and releases tracks on stop", async () => {
  const device = makeStream();
  let constraints;
  const env = setup(async (c) => {
    constraints = c;
    return device.stream;
  });
  await flush();
  await env.elements.get("camera").onclick();
  await flush();
  assert.equal(constraints.audio, false);
  assert.equal(env.elements.get("mode").textContent, "Live camera");
  assert.equal(env.elements.get("count").textContent, 1);
  assert.equal(env.elements.get("videoOverlay").hidden, false);
  assert.match(env.elements.get("overlayMetrics").textContent, /Brightness/);
  assert.match(env.elements.get("overlayCondition").textContent, /Low detail/);
  assert.match(env.elements.get("videoOverlay").className, /overlay-warning/);
  env.elements.get("stopCamera").onclick();
  assert.equal(env.elements.get("videoOverlay").hidden, true);
  assert.equal(device.stopped(), 1);
  assert.equal(env.elements.get("video").srcObject, null);
  assert.equal(env.elements.get("mode").textContent, "Camera stopped");
});
test("late permission result is released after cancellation", async () => {
  let resolve;
  const device = makeStream();
  const env = setup(() => new Promise((r) => (resolve = r)));
  await flush();
  const pending = env.elements.get("camera").onclick();
  env.elements.get("stopCamera").onclick();
  resolve(device.stream);
  await pending;
  assert.equal(device.stopped(), 1);
  assert.equal(env.elements.get("video").srcObject, undefined);
});
test("denied permission produces recovery guidance", async () => {
  const env = setup(async () => {
    throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
  });
  await flush();
  await env.elements.get("camera").onclick();
  assert.match(env.elements.get("status").textContent, /permission was denied/);
  assert.equal(env.elements.get("camera").disabled, false);
});
test("hidden page releases live camera", async () => {
  const device = makeStream();
  const env = setup(async () => device.stream);
  await flush();
  await env.elements.get("camera").onclick();
  env.doc.hidden = true;
  env.events.visibilitychange();
  assert.equal(device.stopped(), 1);
});
