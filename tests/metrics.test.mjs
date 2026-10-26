import test from "node:test";
import assert from "node:assert/strict";
import { measure, classify } from "../dist/metrics.mjs";
const pixels = (fn) =>
  Uint8ClampedArray.from(
    Array.from({ length: 400 }, (_, i) => [
      ...Array(3).fill(fn(i % 20, Math.floor(i / 20))),
      255,
    ]).flat(),
  );
test("uniform dark image has expected brightness and zero detail", () => {
  const m = measure(
    pixels(() => 10),
    20,
    20,
  );
  assert.ok(Math.abs(m.brightness - 10) < 1e-8);
  assert.ok(m.detail < 1e-8);
  assert.deepEqual(classify(m, 45, 60).flags, ["Dark", "Low detail"]);
});
test("high-contrast checkerboard preserves detail", () => {
  const m = measure(
    pixels((x, y) => ((x + y) % 2) * 255),
    20,
    20,
  );
  assert.ok(m.detail > 1000);
  assert.equal(classify(m, 45, 60).score, 100);
});
test("weakest normalized signal determines reference score", () => {
  assert.equal(classify({ brightness: 90, detail: 30 }, 45, 60).score, 50);
  assert.deepEqual(classify({ brightness: 45, detail: 60 }, 45, 60).flags, []);
});
