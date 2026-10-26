let session = null;
export async function initialize() {
  if (!globalThis.ort)
    throw new Error(
      "ONNX Runtime assets are missing. Run npm ci and npm run prepare-runtime.",
    );
  ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(
    new URL("./models/visibility-baseline.onnx", import.meta.url).href,
    { executionProviders: ["wasm"] },
  );
  const scores = await infer([{ brightness: 90, detail: 30 }], 45, 60);
  if (Math.abs(scores[0] - 50) > 1e-4)
    throw new Error("ONNX startup parity check failed.");
}
export async function infer(samples, brightnessThreshold, detailThreshold) {
  if (!session) throw new Error("ONNX session is not ready.");
  if (!samples.length) return [];
  const input = Float32Array.from(
    samples.flatMap((s) => [s.brightness, s.detail]),
  );
  const result = await session.run({
    metrics: new ort.Tensor("float32", input, [samples.length, 2]),
    thresholds: new ort.Tensor(
      "float32",
      Float32Array.of(brightnessThreshold, detailThreshold),
      [2],
    ),
  });
  return Array.from(result.scores.data);
}
