const percent = (v) => `${(v * 100).toFixed(1)}%`;
const root = document.getElementById("evaluationDetails");
function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  root.append(p);
}
function table(title, headers, rows) {
  const h = document.createElement("h3");
  h.textContent = title;
  const table = document.createElement("table"),
    caption = document.createElement("caption");
  caption.textContent = title;
  caption.className = "sr-only";
  table.append(caption);
  const head = document.createElement("thead"),
    tr = document.createElement("tr");
  headers.forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    tr.append(th);
  });
  head.append(tr);
  table.append(head);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((v, i) => {
      const cell = document.createElement(i === 0 ? "th" : "td");
      if (i === 0) cell.scope = "row";
      cell.textContent = String(v);
      tr.append(cell);
    });
    body.append(tr);
  });
  table.append(body);
  root.append(h, table);
}
try {
  const response = await fetch("./reports/evaluation.json");
  if (!response.ok) throw new Error("Evaluation artifact unavailable");
  const r = await response.json();
  document.getElementById("evaluationStatus").textContent =
    `540 procedural images / 180 base scenes · ${r.samples.train} train, ${r.samples.validation} validation, ${r.samples.test} test images. All variants of a base scene stay in one partition.`;
  const summary = document.getElementById("evaluationSummary");
  for (const [label, value] of [
    ["Learned test macro F1", percent(r.learned.macro_f1)],
    ["Tuned-rule test macro F1", percent(r.tuned_baseline.macro_f1)],
    ["Test scenes", String(r.scene_groups.test)],
  ]) {
    const card = document.createElement("div"),
      small = document.createElement("span"),
      strong = document.createElement("strong");
    small.textContent = label;
    strong.textContent = value;
    card.append(small, strong);
    summary.append(card);
  }
  paragraph(
    `Model: feature-based multiclass softmax regression, not an image CNN. Inputs are brightness and Laplacian variance. Embedded ONNX preprocessing applies log1p and training-only standardization. Full-batch gradient descent: ${r.training.epochs} epochs, learning rate ${r.training.learning_rate}, selected L2 ${r.training.selected_l2}. L2 was selected on validation macro F1; test data was excluded from fitting and selection.`,
  );
  table(
    "Held-out comparison",
    ["Method", "Accuracy", "Macro F1"],
    [
      [
        "Learned softmax",
        percent(r.learned.accuracy),
        percent(r.learned.macro_f1),
      ],
      [
        "Fixed rules (45 / 60)",
        percent(r.baseline.accuracy),
        percent(r.baseline.macro_f1),
      ],
      [
        `Validation-tuned rules (${r.tuned_baseline_selection.brightness} / ${r.tuned_baseline_selection.detail})`,
        percent(r.tuned_baseline.accuracy),
        percent(r.tuned_baseline.macro_f1),
      ],
    ],
  );
  paragraph(
    "The fixed baseline uses existing, untuned cutoffs: dark takes priority over low detail. Both the learned classifier and the tuned rules are selected on validation data only; rule selection searches 198 threshold pairs. Labels reflect applied transformations, not human judgments. Synthetic artifacts can make this task much easier than real-world visibility assessment.",
  );
  table(
    "Learned model by class",
    ["Class", "Precision", "Recall", "F1", "Test images"],
    r.learned.per_class.map((c) => [
      c.label,
      percent(c.precision),
      percent(c.recall),
      percent(c.f1),
      c.support,
    ]),
  );
  table(
    "Learned confusion matrix — rows true, columns predicted",
    ["True class", "Clear", "Dark", "Low detail"],
    r.learned.confusion_matrix.map((row, i) => [r.classes[i], ...row]),
  );
  paragraph(
    `Export parity: maximum absolute probability error ${r.parity_max_abs_error.toExponential(2)} between NumPy and ONNX Runtime on all held-out samples. ONNX model SHA-256: ${r.model_sha256}.`,
  );
  paragraph(
    `CPU microbenchmark: p50 ${r.timing.inference_p50_ms.toFixed(3)} ms, p95 ${r.timing.inference_p95_ms.toFixed(3)} ms; batch size 1, ${r.timing.warmup_runs} warmups, ${r.timing.measured_runs} runs. ${r.timing.platform}. This measures Python ONNX Runtime inference only, not browser WASM or end-to-end video latency.`,
  );
  const mistakes = r.errors.filter((e) => e.learned !== e.truth);
  table(
    "Learned error cases",
    ["Scene ID", "Applied label", "Prediction", "Brightness", "Detail"],
    mistakes.map((e) => [
      e.scene_id,
      e.truth,
      e.learned,
      e.brightness.toFixed(1),
      e.detail.toFixed(1),
    ]),
  );
  paragraph(
    "Limitations: only 36 held-out scene groups, procedural textures, grayscale imagery, single transformations, no real endoscopic labels, no temporal evaluation, no calibrated uncertainty, and no unknown-class detector. The classifier always chooses one of three labels even for out-of-distribution images. The original heuristic score and warning cutoffs remain separate from learned predictions.",
  );
} catch (e) {
  document.getElementById("evaluationStatus").textContent = e.message;
}
