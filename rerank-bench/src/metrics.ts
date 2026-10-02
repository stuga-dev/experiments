/** Ranking metrics over one ordered candidate list, and the statistics over many. */

/** 1-based rank of the first gold id in `order`, or 0 when none is there. */
export function firstGoldRank(order: readonly string[], gold: ReadonlySet<string>): number {
  const i = order.findIndex((id) => gold.has(id));
  return i < 0 ? 0 : i + 1;
}

export const reciprocalRank = (rank: number) => (rank > 0 ? 1 / rank : 0);
export const hitAt = (rank: number, k: number) => (rank > 0 && rank <= k ? 1 : 0);

/** Binary-relevance nDCG@k: every gold id counts 1, ideal = all gold first. */
export function ndcgAt(order: readonly string[], gold: ReadonlySet<string>, k: number): number {
  let dcg = 0;
  order.slice(0, k).forEach((id, i) => {
    if (gold.has(id)) dcg += 1 / Math.log2(i + 2);
  });
  const inList = order.filter((id) => gold.has(id)).length;
  let ideal = 0;
  for (let i = 0; i < Math.min(inList, k); i++) ideal += 1 / Math.log2(i + 2);
  return ideal ? dcg / ideal : 0;
}

export function quantile(values: readonly number[], q: number): number {
  if (!values.length) return Number.NaN;
  const s = [...values].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
}

export const mean = (values: readonly number[]) => (values.length ? values.reduce((s, v) => s + v, 0) / values.length : Number.NaN);

/** A seeded PRNG, so a report reproduces its intervals. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Percentile bootstrap 95% interval of the mean, resampling questions. */
/** Percentile bootstrap interval of the mean at level 1 − alpha. */
export function bootstrapMean(values: readonly number[], iterations = 2000, seed = 7, alpha = 0.05): [number, number] {
  if (!values.length) return [Number.NaN, Number.NaN];
  const rand = mulberry32(seed);
  const means: number[] = [];
  for (let b = 0; b < iterations; b++) {
    let s = 0;
    for (let i = 0; i < values.length; i++) s += values[Math.floor(rand() * values.length)]!;
    means.push(s / values.length);
  }
  return [quantile(means, alpha / 2), quantile(means, 1 - alpha / 2)];
}
