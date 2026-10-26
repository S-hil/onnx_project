import { measure, classify } from "./metrics.mjs";
import { initialize, infer, inferLearned } from "./inference.mjs";
import { suggestThresholds } from "./guidance.mjs";
let runtimeReady = false,
  renderVersion = 0;
const $ = (id) => document.getElementById(id),
  video = $("video");
let raw = [],
  rows = [],
  demo = false,
  busy = false,
  cancelled = false,
  objectUrl = null;
let stream = null,
  live = false,
  liveTimer = null,
  cameraPending = false,
  liveStarted = 0,
  sourceMode = "idle",
  generation = 0;
const time = (t) =>
  `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
function setBusy(value) {
  busy = value;
  $("analyze").disabled = value || live || cameraPending || !objectUrl;
  $("camera").disabled = value || live || cameraPending;
  $("demo").disabled = value;
  $("file").disabled = value;
  $("cancel").hidden = !value;
  $("progress").hidden = !value;
}
async function render() {
  const version = ++renderVersion;
  const next = raw.map((s) =>
    classify(s, +$("brightness").value, +$("detail").value),
  );
  $("export").disabled = true;
  if (next.length) {
    if (!runtimeReady) return;
    try {
      const started = performance.now();
      const scores = await infer(
        next,
        +$("brightness").value,
        +$("detail").value,
      );
      const learned = await inferLearned(next);
      if (version !== renderVersion) return;
      next.forEach((r, i) => {
        r.probabilities = learned[i];
        r.prediction = ["clear", "dark", "low_detail"][
          learned[i].indexOf(Math.max(...learned[i]))
        ];
      });
      next.forEach((r, i) => (r.score = Math.round(scores[i])));
      $("runtimeStatus").textContent =
        `ONNX active · WASM / CPU · ${next.length} rows · ${(performance.now() - started).toFixed(2)} ms batch inference · parity passed`;
    } catch (e) {
      $("runtimeStatus").textContent = `ONNX inference failed: ${e.message}`;
      stopCamera();
      return;
    }
  }
  rows = next;
  $("brightnessValue").value = $("brightness").value;
  $("detailValue").value = $("detail").value;
  $("score").textContent = rows.length
    ? Math.round(rows.reduce((a, b) => a + b.score, 0) / rows.length)
    : "—";
  const flagged = rows.filter((s) => s.flags.length);
  $("flagged").textContent = rows.length
    ? `${flagged.length} / ${rows.length}`
    : "—";
  $("count").textContent = rows.length || "—";
  $("source").textContent = demo
    ? "Synthetic data · not video measurements"
    : rows.length
      ? sourceMode === "camera"
        ? "Live camera samples"
        : "Measured video frames"
      : "Awaiting video";
  $("export").disabled = !rows.length;
  $("reviewcount").textContent = rows.length ? `(${flagged.length})` : "";
  $("moments").replaceChildren();
  if (!flagged.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = rows.length
      ? "No samples fall below the current thresholds."
      : "Flagged samples will appear here. Select a moment to seek to it.";
    $("moments").append(p);
  }
  for (const r of flagged.slice(0, 200)) {
    const el = document.createElement("div");
    el.className = "moment";
    const b = document.createElement("button");
    b.textContent = time(r.time);
    b.disabled = demo || sourceMode !== "video";
    b.onclick = () => {
      video.currentTime = r.time;
      video.pause();
      video.scrollIntoView({ behavior: "smooth", block: "center" });
    };
    const span = document.createElement("span");
    span.textContent = r.flags.join(" · ");
    const small = document.createElement("small");
    small.textContent = `Brightness ${r.brightness.toFixed(1)} / Detail ${r.detail.toFixed(1)}`;
    el.append(b, span, small);
    $("moments").append(el);
  }
  if (flagged.length > 200) {
    const p = document.createElement("p");
    p.textContent =
      "Showing first 200 flagged samples. Export CSV for all results.";
    $("moments").append(p);
  }
  updateTelemetry();
  draw();
}
function draw() {
  const c = $("chart"),
    w = c.clientWidth,
    h = 160,
    dpr = window.devicePixelRatio || 1;
  c.width = w * dpr;
  c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.font = "10px sans-serif";
  for (const n of [0, 50, 100]) {
    const y = 12 + (100 - n) * 1.2;
    ctx.strokeStyle = "#e7ede3";
    ctx.beginPath();
    ctx.moveTo(28, y);
    ctx.lineTo(w - 10, y);
    ctx.stroke();
    ctx.fillStyle = "#84917e";
    ctx.fillText(n, 0, y + 3);
  }
  if (!rows.length) return;
  const max = rows.at(-1).time || 1;
  const x = (t) => 30 + ((w - 45) * t) / max,
    y = (s) => 12 + (100 - s) * 1.2;
  ctx.beginPath();
  rows.forEach((r, i) =>
    i ? ctx.lineTo(x(r.time), y(r.score)) : ctx.moveTo(x(r.time), y(r.score)),
  );
  ctx.strokeStyle = "#48755a";
  ctx.lineWidth = 2;
  ctx.stroke();
  for (const r of rows.filter((s) => s.flags.length)) {
    ctx.beginPath();
    ctx.arc(x(r.time), y(r.score), 3, 0, Math.PI * 2);
    ctx.fillStyle = "#bf8a48";
    ctx.fill();
  }
  ctx.fillStyle = "#84917e";
  ctx.fillText("0:00", 28, 155);
  ctx.fillText(time(max), w - 40, 155);
  $("chartnote").textContent =
    `${demo ? "Synthetic demonstration" : sourceMode === "camera" ? "Live camera samples" : "Sampled video"} · Green: quality score · Amber: below threshold. Unsampled moments are not evaluated.`;
}
function waitEvent(target, name, action) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () =>
        end(new Error("Video decoding timed out. Try another file or format.")),
      15000,
    );
    function end(err) {
      clearTimeout(timer);
      target.removeEventListener(name, ok);
      target.removeEventListener("error", bad);
      err ? reject(err) : resolve();
    }
    const ok = () => end(),
      bad = () => end(new Error("This browser could not decode the video."));
    target.addEventListener(name, ok, { once: true });
    target.addEventListener("error", bad, { once: true });
    action();
  });
}
$("file").onchange = async () => {
  const f = $("file").files[0];
  if (!f) return;
  stopCamera();
  video.srcObject = null;
  video.muted = false;
  video.controls = true;
  sourceMode = "video";
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(f);
  raw = [];
  demo = false;
  render();
  $("empty").hidden = true;
  video.hidden = false;
  $("filename").textContent = f.name;
  setBusy(true);
  try {
    await waitEvent(video, "loadeddata", () => {
      video.src = objectUrl;
      video.load();
    });
    if (!Number.isFinite(video.duration) || video.duration <= 0)
      throw new Error("Video needs a finite, positive duration.");
    $("status").textContent =
      `Ready · ${time(video.duration)} · Choose Analyze video.`;
  } catch (e) {
    $("status").textContent = e.message;
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  } finally {
    setBusy(false);
  }
};
$("analyze").onclick = async () => {
  if (busy || !objectUrl) return;
  sourceMode = "video";
  video.hidden = false;
  $("empty").hidden = true;
  setBusy(true);
  cancelled = false;
  demo = false;
  raw = [];
  render();
  video.pause();
  video.controls = false;
  const prior = video.currentTime,
    canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = Math.max(
    3,
    Math.round((320 * video.videoHeight) / video.videoWidth),
  );
  const ctx = canvas.getContext("2d", { willReadFrequently: true }),
    step = +$("interval").value,
    total = Math.ceil(video.duration / step);
  if (total > 20000) {
    $("status").textContent =
      "Too many samples. Increase the sampling interval or choose a shorter clip.";
    video.controls = true;
    setBusy(false);
    return;
  }
  try {
    for (let i = 0; i < total; i++) {
      if (cancelled) break;
      const t = Math.min(i * step, Math.max(0, video.duration - 0.001));
      if (Math.abs(video.currentTime - t) > 0.00001)
        await waitEvent(video, "seeked", () => (video.currentTime = t));
      const started = performance.now();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const metrics = measure(
        ctx.getImageData(0, 0, canvas.width, canvas.height).data,
        canvas.width,
        canvas.height,
      );
      raw.push({ time: t, ...metrics, computeMs: performance.now() - started });
      $("progress").value = ((i + 1) / total) * 100;
      $("status").textContent = `Analyzing ${i + 1} / ${total} frames…`;
      await new Promise((r) => setTimeout(r, 0));
    }
    render();
    $("status").textContent = cancelled
      ? `Cancelled · ${raw.length} measured samples retained.`
      : `Complete · ${raw.length} frames analyzed. Adjust thresholds to explore the results.`;
  } catch (e) {
    render();
    $("status").textContent = `${e.message} ${raw.length} samples retained.`;
  } finally {
    video.currentTime = prior;
    video.controls = true;
    setBusy(false);
  }
};
$("cancel").onclick = () => (cancelled = true);
$("demo").onclick = () => {
  stopCamera();
  sourceMode = "synthetic";
  video.pause();
  video.hidden = true;
  $("empty").hidden = false;
  $("filename").textContent = "Synthetic telemetry · no video frames";
  demo = true;
  raw = Array.from({ length: 61 }, (_, i) => ({
    time: i,
    brightness: i >= 16 && i <= 23 ? 22 : 105 + 18 * Math.sin(i * 0.4),
    detail: i >= 38 && i <= 46 ? 12 : 150 + 50 * Math.cos(i * 0.6),
  }));
  render();
  $("status").textContent =
    "Synthetic demonstration: darkness at 0:16–0:23 and low detail at 0:38–0:46. Upload a video to analyze actual frames.";
};
for (const id of ["brightness", "detail"]) $(id).oninput = render;
$("export").onclick = () => {
  const csv = [
    "source,time_seconds,brightness,laplacian_variance,quality_score,brightness_threshold,detail_threshold,compute_ms,flags,learned_class,p_clear,p_dark,p_low_detail",
    ...rows.map(
      (r) =>
        `${sourceMode},${r.time},${r.brightness.toFixed(3)},${r.detail.toFixed(3)},${r.score},${$("brightness").value},${$("detail").value},${r.computeMs?.toFixed(3) ?? ""},${r.flags.join(";")},${r.prediction},${r.probabilities.join(",")}`,
    ),
  ].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" })),
    a = document.createElement("a");
  a.href = url;
  a.download = `clarity-${sourceMode}-report.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
function updateTelemetry() {
  const r = rows.at(-1);
  $("readingHelp").textContent = r
    ? `Latest sample: brightness ${r.brightness.toFixed(1)} vs cutoff ${$("brightness").value}; detail ${r.detail.toFixed(1)} vs cutoff ${$("detail").value}. ${r.flags.length ? "Review this sample: " + r.flags.join(" and ") + ". Try better lighting or a more focused, detailed scene; lower cutoffs only if you want to tolerate this condition." : "Both cutoffs are met. This does not rule out an obstruction or overexposure."}`
    : "Start the camera or analyze a video to explain the latest sample.";
  $("learnedReading").textContent = r?.probabilities
    ? `Trained model: ${r.prediction.replace("_", " ")}. Softmax: clear ${(r.probabilities[0] * 100).toFixed(1)}%, dark ${(r.probabilities[1] * 100).toFixed(1)}%, low detail ${(r.probabilities[2] * 100).toFixed(1)}%. Synthetic-trained; these outputs are not calibrated confidence or clinical probabilities.`
    : "Trained model: awaiting input. Synthetic training only.";
  $("overlayLearned").textContent = r?.prediction
    ? `Learned: ${r.prediction.replace("_", " ")} · synthetic-trained`
    : "Learned model: awaiting sample";
  $("videoOverlay").hidden = !live;
  $("videoOverlay").className =
    `video-overlay ${r?.flags.length ? "overlay-warning" : ""}`;
  $("overlayCondition").textContent = r
    ? r.flags.join(" + ") || "Above thresholds"
    : "Awaiting sample";
  $("overlayScore").textContent = r ? r.score : "—";
  $("overlayMetrics").textContent = r
    ? `Brightness ${r.brightness.toFixed(1)} · Detail ${r.detail.toFixed(1)}`
    : "Brightness — · Detail —";
  $("overlaySample").textContent = r
    ? `Sample ${time(r.time)} · ${$("interval").value}s target interval`
    : "Waiting for sample";
  $("mode").textContent = live
    ? "Live camera"
    : sourceMode === "camera"
      ? "Camera stopped"
      : sourceMode === "synthetic"
        ? "Synthetic"
        : sourceMode === "video"
          ? "Video file"
          : "Idle";
  $("currentBrightness").textContent = r ? r.brightness.toFixed(1) : "—";
  $("currentDetail").textContent = r ? r.detail.toFixed(1) : "—";
  $("latency").textContent =
    r?.computeMs !== undefined ? `${r.computeMs.toFixed(2)} ms` : "—";
  $("condition").textContent = r
    ? r.flags.join(" + ") || "Above thresholds"
    : "Awaiting input";
}
function stopCamera() {
  generation++;
  clearTimeout(liveTimer);
  live = false;
  cameraPending = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
  }
  $("stopCamera").hidden = true;
  $("camera").hidden = false;
  $("file").disabled = busy;
  $("demo").disabled = busy;
  $("camera").disabled = busy;
  $("analyze").disabled = busy || !objectUrl || sourceMode === "camera";
  if (sourceMode === "camera") {
    video.hidden = true;
    $("empty").hidden = false;
    $("status").textContent =
      `Camera stopped · ${raw.length} samples retained for export.`;
  }
  updateTelemetry();
}
$("stopCamera").onclick = stopCamera;
$("camera").onclick = async () => {
  if (busy || live || cameraPending || !runtimeReady) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    $("status").textContent =
      "Camera API unavailable. Open this localhost URL in Chrome or Safari, or use HTTPS for a hosted version.";
    return;
  }
  const token = ++generation;
  cameraPending = true;
  $("camera").disabled = true;
  $("camera").hidden = true;
  $("stopCamera").hidden = false;
  $("analyze").disabled = true;
  $("demo").disabled = true;
  $("file").disabled = true;
  $("status").textContent =
    "Waiting for camera permission. No microphone access is requested. Stop camera cancels this request in the app.";
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    if (token !== generation) {
      acquired.getTracks().forEach((t) => t.stop());
      return;
    }
    stream = acquired;
    sourceMode = "camera";
    demo = false;
    raw = [];
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    video.removeAttribute("src");
    video.srcObject = stream;
    video.muted = true;
    video.controls = false;
    video.hidden = false;
    $("empty").hidden = true;
    await video.play();
    if (token !== generation) return;
    cameraPending = false;
    live = true;
    liveStarted = performance.now();
    $("filename").textContent =
      `Live camera · ${video.videoWidth} × ${video.videoHeight}`;
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = Math.max(
      3,
      Math.round((320 * video.videoHeight) / video.videoWidth),
    );
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    stream.getVideoTracks().forEach((t) =>
      t.addEventListener("ended", () => {
        if (live) stopCamera();
      }),
    );
    function tick() {
      if (!live) return;
      try {
        if (video.readyState >= 2) {
          const start = performance.now();
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const metrics = measure(
            ctx.getImageData(0, 0, canvas.width, canvas.height).data,
            canvas.width,
            canvas.height,
          );
          raw.push({
            time: (performance.now() - liveStarted) / 1000,
            ...metrics,
            computeMs: performance.now() - start,
          });
          render();
          $("status").textContent =
            `Live · ${raw.length} samples · video only · ${$("interval").value}s target sampling interval`;
          if (raw.length >= 3600) {
            stopCamera();
            $("status").textContent =
              "Stopped at 3,600 samples. Export the report or start a new session.";
            return;
          }
        }
        liveTimer = setTimeout(tick, +$("interval").value * 1000);
      } catch (e) {
        stopCamera();
        $("status").textContent = `Camera processing stopped: ${e.message}`;
      }
    }
    tick();
  } catch (e) {
    if (token !== generation) return;
    stopCamera();
    const errors = {
      NotAllowedError:
        "Camera permission was denied or this embedded browser blocks camera access. Open the same localhost URL in Chrome or Safari and allow camera access.",
      NotFoundError: "No camera was found. Connect a webcam and retry.",
      NotReadableError:
        "Camera is unavailable or being used by another application.",
    };
    $("status").textContent =
      errors[e.name] || `Could not start camera: ${e.message}`;
  }
};
window.addEventListener("pagehide", stopCamera);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && (live || cameraPending)) stopCamera();
});
window.addEventListener("resize", draw);
render();
$("camera").disabled = true;
$("demo").disabled = true;
$("file").disabled = true;
initialize()
  .then(() => {
    runtimeReady = true;
    $("runtimeStatus").textContent =
      "ONNX active · 2 models · WASM / CPU · startup parity passed";
    $("camera").disabled = false;
    $("demo").disabled = false;
    $("file").disabled = false;
  })
  .catch((e) => {
    $("runtimeStatus").textContent = `ONNX unavailable: ${e.message}`;
    $("status").textContent =
      "Runtime initialization failed. Check local runtime assets and refresh.";
  });

function setThresholds(brightness, detail, message) {
  $("brightness").value = brightness;
  $("detail").value = detail;
  $("settingsHelp").textContent = message;
  render();
}
$("presetTolerant").onclick = () =>
  setThresholds(
    25,
    30,
    "Fewer warnings: brightness 25, detail 30. Darker or lower-detail images may pass. Illustrative preset.",
  );
$("presetDefault").onclick = () =>
  setThresholds(
    45,
    60,
    "Defaults restored: brightness 45, detail 60. Unvalidated starting values.",
  );
$("presetSensitive").onclick = () =>
  setThresholds(
    70,
    120,
    "More warnings: brightness 70, detail 120. Clear smooth scenes may trigger warnings. Illustrative preset.",
  );
$("calibrate").onclick = () => {
  try {
    if (sourceMode === "synthetic")
      throw new Error(
        "Synthetic values cannot calibrate your camera. Use measured camera or video samples.",
      );
    const t = suggestThresholds(raw);
    setThresholds(
      t.brightness,
      t.detail,
      `Reference applied from ${t.count} recent samples: brightness ${t.brightness}, detail ${t.detail}. Cutoffs are 70% of each median, constrained to slider ranges. Verify them on separate clear and degraded scenes.`,
    );
  } catch (e) {
    $("settingsHelp").textContent = e.message;
  }
};
