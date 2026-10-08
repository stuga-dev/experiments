/** A seeded generator, so draws and reports are reproducible. */
export function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Paired bootstrap of b − a over the same questions: the mean difference and its 95% interval. */
export function paired(a: number[], b: number[], seed: number, draws = 2000): [number, number, number] {
  const d = a.map((x, i) => b[i]! - x);
  const rand = mulberry32(seed);
  const means: number[] = [];
  for (let k = 0; k < draws; k++) {
    let s = 0;
    for (let i = 0; i < d.length; i++) s += d[Math.floor(rand() * d.length)]!;
    means.push(s / d.length);
  }
  means.sort((x, y) => x - y);
  return [d.reduce((x, y) => x + y, 0) / d.length, means[Math.floor(0.025 * draws)]!, means[Math.ceil(0.975 * draws) - 1]!];
}
