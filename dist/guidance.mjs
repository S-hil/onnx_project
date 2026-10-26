export function suggestThresholds(samples) {
  if (samples.length < 5)
    throw new Error("Collect at least 5 measured samples first.");
  const recent = samples.slice(-10);
  const median = (key) => {
    const a = recent.map((s) => s[key]).sort((a, b) => a - b);
    const n = a.length;
    return n % 2 ? a[Math.floor(n / 2)] : (a[n / 2 - 1] + a[n / 2]) / 2;
  };
  const brightness = Math.max(
    5,
    Math.min(150, Math.round(0.7 * median("brightness"))),
  );
  const detail = Math.max(5, Math.min(300, Math.round(0.7 * median("detail"))));
  return { brightness, detail, count: recent.length };
}
