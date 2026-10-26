import test from "node:test";
import assert from "node:assert/strict";
import { suggestThresholds } from "../dist/guidance.mjs";
test("requires five measured reference samples", () =>
  assert.throws(() => suggestThresholds([]), /at least 5/));
test("uses median so one bright outlier does not dominate", () => {
  const a = Array.from({ length: 5 }, () => ({ brightness: 100, detail: 200 }));
  a[4] = { brightness: 255, detail: 9999 };
  assert.deepEqual(suggestThresholds(a), {
    brightness: 70,
    detail: 140,
    count: 5,
  });
});
test("threshold suggestions stay in slider bounds", () =>
  assert.deepEqual(
    suggestThresholds(
      Array.from({ length: 5 }, () => ({ brightness: 0, detail: 9999 })),
    ),
    { brightness: 5, detail: 300, count: 5 },
  ));
