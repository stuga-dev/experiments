/** Retrieval metrics over a ranked list of passage indexes. */

/** nDCG@k with graded gains, as pytrec_eval computes it for MTEB. */
export function ndcg(ranked: number[], gains: Map<number, number>, k = 10): number {
  let dcg = 0;
  ranked.slice(0, k).forEach((p, i) => {
    const g = gains.get(p) ?? 0;
    if (g > 0) dcg += g / Math.log2(i + 2);
  });
  const ideal = [...gains.values()].filter((g) => g > 0).sort((a, b) => b - a).slice(0, k);
  const idcg = ideal.reduce((a, g, i) => a + g / Math.log2(i + 2), 0);
  return idcg > 0 ? dcg / idcg : 0;
}

/** Share of the relevant passages within the first k. */
export function recall(ranked: number[], gains: Map<number, number>, k: number): number {
  const relevant = [...gains.entries()].filter(([, g]) => g > 0).map(([p]) => p);
  if (!relevant.length) return 0;
  const top = new Set(ranked.slice(0, k));
  return relevant.filter((p) => top.has(p)).length / relevant.length;
}

/** Whether the first passage is relevant. */
export function hit1(ranked: number[], gains: Map<number, number>): number {
  return ranked.length && (gains.get(ranked[0]!) ?? 0) > 0 ? 1 : 0;
}

/** SQL's rank(): one plus how many score strictly higher; equal scores share a position. */
export function sqlRank(scores: number[]): number[] {
  return scores.map((s) => 1 + scores.filter((t) => t > s).length);
}

/**
 * Stuga's retrieval fusion (packages/db/src/search.ts, askDocs): reciprocal rank with k = 60 and
 * equal weights, ordered by score, then by passage id. `keywordWeight` scales the keyword leg's
 * share, to try weightings Stuga does not use.
 */
export function rrf(
  keyword: { ids: number[]; scores: number[] },
  semantic: { ids: number[]; sims: number[] },
  pid: (i: number) => string,
  limit: number,
  keywordWeight = 1,
): number[] {
  const score = new Map<number, number>();
  const add = (ids: number[], ranks: number[], w: number) => ids.forEach((p, i) => score.set(p, (score.get(p) ?? 0) + w / (60 + ranks[i]!)));
  add(keyword.ids, sqlRank(keyword.scores), keywordWeight);
  add(semantic.ids, sqlRank(semantic.sims), 1);
  return [...score.entries()]
    .sort((a, b) => b[1] - a[1] || (pid(a[0]) < pid(b[0]) ? -1 : pid(a[0]) > pid(b[0]) ? 1 : 0))
    .slice(0, limit)
    .map(([p]) => p);
}
