import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const root = new URL("../", import.meta.url);
await mkdir(new URL("dist/vendor/", root), { recursive: true });
for (const name of [
  "ort.min.js",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]) {
  await copyFile(
    new URL(`node_modules/onnxruntime-web/dist/${name}`, root),
    new URL(`dist/vendor/${name}`, root),
  );
}
await copyFile(
  new URL("node_modules/onnxruntime-web/README.md", root),
  new URL("dist/vendor/ONNX-RUNTIME-README.md", root),
);
console.log("ONNX Runtime Web assets copied to dist/vendor.");
