/**
 * Score one model on the sets: dense retrieval alone (exact cosine), BM25 alone, and Stuga's
 * retrieval (both legs 96 deep, the vector leg cut at a distance, fused by RRF, 24 kept), per
 * question, into .cache/eval/<run>/<set>.json. Also keeps each question's relevant-passage distances
 * and the distances to 64 random passages, for calibrating cutoffs.
 *
 *   node src/eval.ts ollama:embeddinggemma-2:270m [miracl-ko,...,bright-so]
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { VARIANT, bm25, DEPTH } from "./bm25.ts";
import { cachePath } from "./embedder.ts";
import { hit1, ndcg, recall, rrf } from "./metrics.ts";
import { parseRun, runKey } from "./models.ts";
import { loadSet, SET_NAMES } from "./sets.ts";

/** What Stuga hands the reranker. */
export const CANDIDATES = 24;
/** Stuga's old default retrieval cutoff, and none (its default now). */
export const CUTOFFS = [0.9, 2] as const;
/** Keyword-leg weights tried in the fusion, without a cutoff: 1 is Stuga's. */
export const KEYWORD_WEIGHTS = [1, 0.75, 0.5, 0.25, 0] as const;
const TOP = 100;
const RANDOM = 64;

export interface QuestionScores {
  id: string;
  dense: { ndcg10: number; recall24: number; recall100: number; hit1: number };
  bm25: { ndcg10: number; recall24: number; hit1: number };
  /** By cutoff, and by keyword weight as `w<weight>`. */
  stuga: Record<string, { ndcg10: number; recall24: number; hit1: number; semantic: number }>;
  /** Cosine distance to each relevant passage. */
  gold: number[];
  /** Cosine distance to random passages of the set. */
  random: number[];
  /** Distance to the nearest passage that is not relevant. */
  nearestOther: number;
}

function readMeta(path: string): { dims: number; count: number; done: number[]; tokens: number; seconds: number } {
  return JSON.parse(readFileSync(`${path}.json`, "utf8"));
}

function readVectors(path: string): { dims: number; v: Float32Array; tokens: number; seconds: number } {
  const meta = readMeta(path);
  if (meta.done.length !== Math.ceil(meta.count / 64)) throw new Error(`${path}: not finished embedding`);
  const b = readFileSync(`${path}.f32`);
  const v = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  if (v.length !== meta.count * meta.dims) throw new Error(`${path}: ${v.length} values for ${meta.count} × ${meta.dims}`);
  for (let i = 0; i < meta.count; i++) {
    if (v.subarray(i * meta.dims, (i + 1) * meta.dims).every((x) => x === 0)) throw new Error(`${path}: row ${i} is empty`);
  }
  return { dims: meta.dims, v, tokens: meta.tokens, seconds: meta.seconds };
}

/** pytrec_eval's order for a scored list: by score, ties by passage id descending, as MMTEB scores. */
function trecOrder(ids: number[], score: (i: number) => number, pid: (i: number) => string): number[] {
  return [...ids].sort((a, b) => score(b) - score(a) || (pid(a) < pid(b) ? 1 : pid(a) > pid(b) ? -1 : 0));
}

/** A seeded generator, so the random passages are the same for every model. */
function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const run = parseRun(process.argv[2]!);
const names = (process.argv[3] ?? SET_NAMES.filter((s) => s.startsWith("miracl-")).join(",")).split(",");
for (const name of names) {
  const set = await loadSet(name);
  const out = `.cache/eval/${runKey(run)}/${set.name}${VARIANT ? `.${VARIANT}` : ""}.json`;
  if (existsSync(out) && !process.env.RESCORE) {
    console.log(`${set.name}: already scored`);
    continue;
  }
  const q = readVectors(cachePath(run, set.name, "query"));
  const p = readVectors(cachePath(run, set.name, "document"));
  const dims = p.dims;
  const n = set.passages.length;
  const ordOf = new Map(set.passages.map((d, i) => [d.id, i]));
  const keyword = await bm25(set);
  const pid = (i: number) => set.passages[i]!.id;
  const rand = mulberry32(42);
  /** `count` passages drawn at random that are not relevant to the question. */
  const randomOthers = (count: number, relevant: Map<number, number>, excluded: Set<number>) => {
    const out: number[] = [];
    while (out.length < count) {
      const i = Math.floor(rand() * n);
      if (!relevant.has(i) && !excluded.has(i)) out.push(i);
    }
    return out;
  };

  const scores: QuestionScores[] = [];
  const t0 = Date.now();
  // Doubles, so distinct similarities stay distinct for rank().
  const sims = new Float64Array(n);
  set.queries.forEach((query, qi) => {
    const qv = q.v.subarray(qi * dims, (qi + 1) * dims);
    const excluded = new Set([...(set.excluded.get(query.id) ?? [])].map((id) => ordOf.get(id)!));
    for (let i = 0; i < n; i++) {
      let s = 0;
      const off = i * dims;
      for (let k = 0; k < dims; k++) s += qv[k]! * p.v[off + k]!;
      sims[i] = s;
    }
    // The nearest TOP, by insertion into a short sorted list.
    const top: number[] = [];
    for (let i = 0; i < n; i++) {
      if ((top.length === TOP && sims[i]! <= sims[top[TOP - 1]!]!) || excluded.has(i)) continue;
      let j = top.length < TOP ? top.length : TOP - 1;
      top[j] = i;
      while (j > 0 && sims[top[j - 1]!]! < sims[i]!) {
        top[j] = top[j - 1]!;
        top[j - 1] = i;
        j--;
      }
    }
    const gains = new Map<number, number>();
    for (const [docId, g] of set.qrels.get(query.id) ?? []) {
      const o = ordOf.get(docId);
      if (o !== undefined && g > 0) gains.set(o, g);
    }
    const kw = keyword[query.id] ?? { ids: [], scores: [] };
    const kwScore = new Map(kw.ids.map((p, i) => [p, kw.scores[i]!]));
    const kwTrec = trecOrder(kw.ids, (i) => kwScore.get(i)!, pid);
    const denseTrec = trecOrder(top, (i) => sims[i]!, pid);
    const stuga: QuestionScores["stuga"] = {};
    for (const cutoff of CUTOFFS) {
      const near = top.slice(0, DEPTH).filter((i) => 1 - sims[i]! < cutoff);
      const fused = rrf(kw, { ids: near, sims: near.map((i) => sims[i]!) }, pid, CANDIDATES);
      stuga[String(cutoff)] = { ndcg10: ndcg(fused, gains), recall24: recall(fused, gains, CANDIDATES), hit1: hit1(fused, gains), semantic: near.length };
    }
    const near = top.slice(0, DEPTH);
    for (const w of KEYWORD_WEIGHTS) {
      const fused = rrf(kw, { ids: near, sims: near.map((i) => sims[i]!) }, pid, CANDIDATES, w);
      stuga[`w${w}`] = { ndcg10: ndcg(fused, gains), recall24: recall(fused, gains, CANDIDATES), hit1: hit1(fused, gains), semantic: near.length };
    }
    scores.push({
      id: query.id,
      dense: { ndcg10: ndcg(denseTrec, gains), recall24: recall(denseTrec, gains, 24), recall100: recall(denseTrec, gains, 100), hit1: hit1(denseTrec, gains) },
      bm25: { ndcg10: ndcg(kwTrec, gains), recall24: recall(kwTrec, gains, 24), hit1: hit1(kwTrec, gains) },
      stuga,
      gold: [...gains.keys()].map((i) => 1 - sims[i]!),
      random: randomOthers(RANDOM, gains, excluded).map((i) => 1 - sims[i]!),
      nearestOther: 1 - sims[top.find((i) => !gains.has(i)) ?? 0]!,
    });
  });
  const mean = (f: (s: QuestionScores) => number) => scores.reduce((a, s) => a + f(s), 0) / scores.length;
  const summary = {
    run: runKey(run),
    set: set.name,
    questions: scores.length,
    passages: n,
    dims,
    embed: { tokens: p.tokens + q.tokens, seconds: p.seconds + q.seconds },
    dense: { ndcg10: mean((s) => s.dense.ndcg10), recall24: mean((s) => s.dense.recall24), recall100: mean((s) => s.dense.recall100) },
    bm25: { ndcg10: mean((s) => s.bm25.ndcg10), recall24: mean((s) => s.bm25.recall24) },
    stuga: Object.fromEntries(CUTOFFS.map((c) => [c, { ndcg10: mean((s) => s.stuga[String(c)]!.ndcg10), recall24: mean((s) => s.stuga[String(c)]!.recall24) }])),
  };
  mkdirSync(`.cache/eval/${runKey(run)}`, { recursive: true });
  writeFileSync(out, JSON.stringify({ summary, scores }));
  console.log(`${set.name} (${((Date.now() - t0) / 1000).toFixed(0)}s):`, JSON.stringify({ dense: summary.dense, bm25: summary.bm25, stuga: summary.stuga }, (_, v) => (typeof v === "number" ? Math.round(v * 1000) / 1000 : v)));
}
