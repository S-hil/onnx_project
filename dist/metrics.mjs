export function measure(data, width, height) {
  const gray = new Float64Array(width * height);
  let sum = 0;
  for (let i = 0; i < gray.length; i++) {
    gray[i] =
      0.2126 * data[i * 4] +
      0.7152 * data[i * 4 + 1] +
      0.0722 * data[i * 4 + 2];
    sum += gray[i];
  }
  let lapSum = 0,
    lapSq = 0,
    n = 0;
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const v =
        gray[i - 1] +
        gray[i + 1] +
        gray[i - width] +
        gray[i + width] -
        4 * gray[i];
      lapSum += v;
      lapSq += v * v;
      n++;
    }
  return {
    brightness: sum / gray.length,
    detail: n ? Math.max(0, lapSq / n - (lapSum / n) ** 2) : 0,
  };
}
export function classify(sample, brightnessThreshold, detailThreshold) {
  const flags = [];
  if (sample.brightness < brightnessThreshold) flags.push("Dark");
  if (sample.detail < detailThreshold) flags.push("Low detail");
  return {
    ...sample,
    flags,
    score: Math.round(
      100 *
        Math.min(
          1,
          sample.brightness / brightnessThreshold,
          sample.detail / detailThreshold,
        ),
    ),
  };
}
