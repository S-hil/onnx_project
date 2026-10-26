# Clarity — Live camera visibility telemetry with ONNX

An independent computer-vision portfolio prototype for investigating dark and low-detail video frames. Built to demonstrate camera acquisition, signal processing, ONNX deployment, runtime observability, and explicit evaluation boundaries relevant to surgical-vision engineering.

**Implemented:** webcam and file input, deterministic image metrics, actual ONNX Runtime Web scoring via WASM, timeline, configurable thresholds, CSV export, technical walkthrough, and tests.

**Not implemented:** trained surgical classifier, semantic obstruction recognition, endoscopic dataset evaluation, robotic control, or clinical validation. No Andromeda Surgical affiliation is implied.

## Quick start

Requires Node.js 20+ and Python 3.11+.

```sh
npm ci
npm start
```

Open http://127.0.0.1:8765. Dependencies are pinned in the lockfile. Installation copies runtime assets into `dist/vendor`; the runtime, model, and application are served locally. No external fonts, analytics, or inference endpoints are used.

- **Start live camera:** approve video access; no microphone or file upload is required. If the in-app browser blocks camera access, open the URL in Chrome or Safari.
- **Select video → Analyze video:** sample a browser-decodable recording.
- **Try synthetic demo:** use illustrative numeric measurements, clearly labeled synthetic. These are not measured frames, but their scores do execute through ONNX.
- Adjust thresholds to reclassify the report. Stop the camera before exporting a stable live report. CSV includes thresholds and provenance.

## Architecture

```text
Webcam: getUserMedia(video only)       File: HTMLVideoElement seek/decode
                  \                  /
                  Canvas 2D resize (width 320)
                            |
              Weighted RGB → Float64 grayscale
                            |
          Mean brightness + 4-neighbor Laplacian variance
                            |
              Float32 ONNX tensors: metrics [N,2]
                            |
             Div → ReduceMin → Clip → Mul (WASM)
                            |
          Scores [N] → timeline / threshold flags / CSV
```

### Image metrics

For gamma-encoded RGB pixel values, `Y′ = 0.2126R′ + 0.7152G′ + 0.0722B′` is a luma-like proxy, not calibrated physical luminance. Brightness `B` is mean Y′. Detail `D` is the variance of the 4-neighbor discrete Laplacian over interior pixels. Metric extraction is O(W×H) and preserves aspect ratio at a fixed width of 320 pixels.

### Actual ONNX graph

The included 518-byte `dist/models/visibility-baseline.onnx` is a deterministic graph, not a trained neural network. ONNX Runtime Web 1.20.1 runs it on the WASM execution provider with one thread.

Inputs:
- `metrics`: float32 `[N, 2]`, columns `[brightness, detail]`.
- `thresholds`: float32 `[2]`, strictly positive `[brightness_threshold, detail_threshold]`.

Output: `scores`, float32 `[N]`.

`score = 100 × clip(min(B/Tb, D/Td), 0, 1)`. JavaScript rounds for display and uses raw measurements to label dark / low-detail samples. Default thresholds 45 and 60 are uncalibrated starting points. The score is not a probability or clinical measure. Float32 ONNX execution can differ slightly from the Float64 reference near rounding boundaries.

The application blocks input if ONNX initialization fails; there is no silent JavaScript scoring fallback. Startup performs a known-answer check: `[90,30] / [45,60] → score 50`.

### Sampling, timing, and lifecycle

Sampling targets 0.5, 1, 2, or 5 seconds. Timeout scheduling follows processing, so actual spacing can exceed the target. Live timestamps use `performance.now`. Camera resolution is requested at an ideal 1280×720; actual dimensions are displayed. File timestamps refer to requested video seek positions and are not guaranteed frame indices.

Metric compute time covers resize, readback, and numeric measurements. ONNX timing displayed in the runtime bar covers batch inference and result transfer. Neither is end-to-end latency. The live report is rescored as a batch after each new sample or threshold change; inference timing therefore depends on retained sample count. Rendering runs on the main thread.

Stop camera, hide the page, or navigate away to stop all acquired tracks. A cancelled pending permission request cannot reopen acquisition: any late stream is immediately released. Live collection stops at 3,600 samples; file analysis caps work at 20,000 samples. Up to 200 flagged samples are rendered; CSV retains all samples. Raw camera frames are not recorded or sent to a server.

## Validation

```sh
npm test
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/export_model.py
.venv/bin/python tests/model_parity.py
```

Automated checks cover uniform dark frames, checkerboard detail, threshold rules, camera constraints and track cleanup with mocked browser APIs, permission denial, late permission cancellation, and hidden-page cleanup. Python ONNX Runtime parity tests compare 3,000 scores with a NumPy reference across three threshold pairs. Browser startup separately checks actual WASM execution. Hardware camera behavior still depends on host permissions and browser support; mocked lifecycle tests do not establish hardware compatibility. GitHub Actions runs the same automated checks.

## Proposed evaluation — no accuracy claims yet

Use licensed, independently labeled clips with clear, dark, blurred, and occluded events. Split by source recording or capture session before selecting thresholds. Freeze thresholds before testing held-out videos. Report condition-specific precision/recall, false alarms per minute, event detection delay, and results by lighting and device; separately profile p50/p95 metric and inference time.

A webcam is an accessible engineering demo, not a substitute for endoscopic validation. Sharp obstructions may score highly; clear smooth regions may look low-detail; noise, auto-exposure, compression, highlights, and motion can confound the metrics. Sparse sampling misses events. Temporal smoothing and semantic obstruction detection are future work.

## Interview walkthrough

1. Explain why an interpretable baseline precedes a learned classifier.
2. Start a webcam with a well-lit, detailed scene; inspect actual metric changes.
3. Reduce lighting, then show an out-of-focus nearby object. Discuss false positives.
4. Change thresholds and demonstrate reclassification through ONNX.
5. Stop the camera, export telemetry, and explain timing boundaries.
6. Discuss a future learned ONNX classifier with independent held-out evaluation.

## GitHub preparation

Upload the contents of this folder as the repository root. Include `dist/models/visibility-baseline.onnx`, source, scripts, tests, `package.json`, `package-lock.json`, and `.github`. Exclude `node_modules`, generated `dist/vendor`, private recordings, CSV exports, and virtual environments (covered by `.gitignore`). Run `npm ci` on a fresh clone to restore runtime files. No credentials are needed. A license has not been selected; choose one before inviting reuse. Third-party dependencies retain their own licenses.

GitHub Pages deployment would need a build step (`npm ci`) and publication of `dist`, including generated vendor assets. Camera access needs localhost or HTTPS. No repository has been created or uploaded by this demo.

References: [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), [getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia), [track.stop](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/stop).

## Live video overlay

During camera acquisition a pointer-transparent overlay shows the latest ONNX score, brightness, Laplacian detail, sample timestamp, target interval, and condition. Amber indicates a threshold flag; green indicates both signals meet thresholds. Updates follow analyzed samples, not every video frame. The overlay summarizes the whole image and does not localize obstructions. It hides when the camera stops and is excluded from frame measurements.

## Guided threshold selection

The interface explains brightness, detail, and the effect of each cutoff in plain language. Presets are explicitly illustrative. Clear-scene reference calibration takes the latest 5–10 measured samples, computes each median, uses 70% as its cutoff, then clamps to the slider ranges. The user must judge those samples to be clear. Synthetic data is rejected for calibration. These suggestions are not universal quality standards and should be checked against separate clear and degraded scenes.
