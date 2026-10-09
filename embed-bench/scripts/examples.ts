/**
 * Concrete examples for the EmbeddingGemma v1 vs v2 videos and posts, from the cached vectors and
 * BM25 hits that eval.ts scores. (a) to (d) are MIRACL English (questions and passages in English),
 * (e) is BRIGHT StackOverflow:
 *
 *   node scripts/examples.ts [results/2026-10-07/examples.json]
 *
 * It reads cutoffs.json from the output's folder (results/2026-10-07 when printing to stdout).
 *
 * Runs: v1 is EmbeddingGemma 300m (ollama/embeddinggemma:300m); v2 is EmbeddingGemma 2 at bf16
 * (ollama/embeddinggemma-2:270m-bf16-text), at Ollama's default nvfp4 tag (ollama/embeddinggemma-2:270m)
 * and that tag without prompts (+raw). Each MIRACL run's per-question ranking is rebuilt as eval.ts
 * builds it (exact cosine over every passage, the nearest 100, ties as eval.ts breaks them) and kept
 * under .cache/examples/; the rebuilt dense nDCG@10 and hit@1, hybrid nDCG@10 and random-pair
 * distances are checked against .cache/eval/<run>/miracl-en.json, and BRIGHT's relevant-passage
 * distances against .cache/eval/<run>/bright-so.json. The script stops on any difference.
 *
 * A rank is a passage's place in that model's ranking of the whole set (178,768 MIRACL passages;
 * BRIGHT's 107,081 less those it excludes for the question), ties by passage id descending as MMTEB
 * scores. "First relevant" is the best-placed passage the set judges relevant. MIRACL's annotators
 * judged only some passages; each one shown says whether it was judged relevant, judged not
 * relevant, or not judged (an unjudged passage counts as not relevant in every score).
 *
 * Selection rules. A question is "plain" when it has at most 9 words, no digits, and reads as an
 * ordinary question (starts with what, who, when, where, which, how, why, is, are, do, does, did, was
 * or were). Every count is over all 799 questions, plain or not. For (a) to (c), the plain questions a
 * rule admits are put in order: the rank the rule turns on nearest the pool's median rank first (so
 * not the most extreme), then the shorter question. They were then read in that order, and the first
 * one a viewer can judge at a glance is the example: the winning run's first passage states the
 * answer within the excerpt shown (its first 160 characters), the losing run's first passage plainly
 * does not, and the question needs no specialist knowledge. READ below lists, in order, each question
 * passed over and why; the script checks that they are exactly the ones before the example.
 *
 * (a) v1 vs v2: v1's first passage is relevant (rank 1) and v2 bf16's first relevant passage is at
 *     rank 3–10. The opposite count is v2 bf16 at rank 1 and v1 at 3–10. Each model's top 3 is shown,
 *     relevant or not.
 * (b) Hybrid rescues: v2 bf16's vectors alone place the first relevant passage below rank 10, and
 *     Stuga's hybrid (the vector leg's nearest 96 and BM25's 96, reciprocal rank fusion, k = 60,
 *     equal weights, 24 kept, as eval.ts computes it) places a relevant passage in its top 3. The
 *     opposite count is vectors alone in the top 3 and hybrid below 10. The matched words are the
 *     question's words whose pg_search tokens (Stuga's English-stemmed ICU column) the passage holds.
 * (c) Prompts matter: the nvfp4 tag with Stuga's prompts places a relevant passage first, and the same
 *     tag on raw text places its first relevant passage at rank 3–10; the opposite count both ways.
 * (d) Cutoff pair: one of a question's 64 random passages (eval.ts draws them with seed 42, so every
 *     model gets the same pairs) that v2 (the nvfp4 tag, as the cutoff episode uses) puts closer
 *     than 0.6 and v1 puts beyond its own 1% floor on MIRACL English (cutoffs.json random.p1, which is
 *     above 0.6). Of the plain questions, the pair taken shares no pg_search token between question
 *     and passage, and has both distances nearest each model's median random distance (so neither is
 *     extreme); it was read and kept.
 * (e) Far relevant passage, BRIGHT: a relevant passage (BRIGHT's gold: pages the accepted answer
 *     links) that v1 puts at or beyond its own 1% floor on BRIGHT (cutoffs.json bright-so
 *     random.p1). Questions with exactly one gold passage are used, so the passage is the one BRIGHT
 *     marks for that question rather than one of many chunks of a long page; they are read in order
 *     of question length, and the first is the example whose passage start shows its subject and
 *     whose link to the question a developer sees at once (READ lists the ones passed over).
 *
 * Postgres with pg_search (BENCH_DATABASE_URL, as bm25.ts) is read for tokenizing only. MIRACL text
 * (CC BY-SA 4.0) comes with each passage's Wikipedia title and page link; BRIGHT (CC BY 4.0) with its
 * dataset, revision and document id.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import postgres from "postgres";
import { bm25, DATABASE_URL, DEPTH } from "../src/bm25.ts";
import { mulberry32 } from "../src/bootstrap.ts";
import { cachePath } from "../src/embedder.ts";
import type { QuestionScores } from "../src/eval.ts";
import { ndcg, rrf } from "../src/metrics.ts";
import { parseRun, profileOf, runKey } from "../src/models.ts";
import { BRIGHT, MIRACL, loadSet } from "../src/sets.ts";
import { hfFile, readParquet } from "./parquet.ts";

const SET = "miracl-en";
const OUT = process.argv[2];
const RESULTS = OUT ? dirname(OUT) : "results/2026-10-07";
/** As eval.ts: the nearest kept per question, the fused list's length, random passages per question. */
const TOP = 100;
const CANDIDATES = 24;
const RANDOM = 64;
const CUTOFF = 0.6;
/** A passage this short (characters of text, title aside) is a scrap: a heading, a list's lead-in. */
const SHORT = 80;

const RUNS = {
  v1: "ollama:embeddinggemma:300m",
  v2: "ollama:embeddinggemma-2:270m-bf16-text",
  v2Default: "ollama:embeddinggemma-2:270m",
  v2Raw: "ollama:embeddinggemma-2:270m+raw",
} as const;
type RunName = keyof typeof RUNS;

/** The plain questions read before each example, in the order the rules put them, and why each was passed over. */
const READ = {
  v1VsV2: {
    chosen: "1613",
    passedOver: [
      ["796", "v2's first passage defines a buff as \"the opposite of a nerf\", which implies the answer"],
      ["996", "several people share the name; v2's first passage answers for another Laura Wright"],
      ["2775", "v1's first passage gives the year it was built (1849) only after its first 160 characters"],
      ["2294", "a medical term, and v2's first passage (its symptoms) is on topic, so the miss takes reading"],
      ["2597", "v1's first passage shows no area within its first 160 characters"],
      ["1480", "v2's first passage gives an alcohol content too (5–7%, for Canadian cider)"],
      ["2249", "v2's first passage gives a population too (the 2000 census)"],
      ["899", "v2's first passage gives his date of death and age too"],
    ],
  },
  hybridRescues: { chosen: "4271", passedOver: [] },
  promptsMatter: { chosen: "2586", passedOver: [] },
  brightFar: {
    chosen: "73",
    passedOver: [
      ["81", "the passage (pyproject.toml's [build-system] table) does not show at its start how it answers a question about pip's install order"],
      ["68", "the passage starts on Mappings in Python's data model and never mentions __call__ or special method lookup"],
      ["59", "the passage starts with the site's navigation and phone number"],
    ],
  },
} satisfies Record<string, { chosen: string; passedOver: [string, string][] }>;

interface Ranked {
  /** The nearest TOP passages, in eval.ts's order (similarity descending, ties by index ascending). */
  top: number[];
  topSims: number[];
  /** Relevant passage index → [rank over the whole set, cosine distance]. */
  relevant: Record<string, [number, number]>;
  /** Cosine distance to each of the question's random passages, in draw order. */
  random: number[];
}

const set = await loadSet(SET);
const n = set.passages.length;
const pid = (i: number) => set.passages[i]!.id;
const ordOf = new Map(set.passages.map((d, i) => [d.id, i]));
const gainsOf = set.queries.map((q) => {
  const gains = new Map<number, number>();
  for (const [docId, g] of set.qrels.get(q.id) ?? []) {
    const o = ordOf.get(docId);
    if (o !== undefined && g > 0) gains.set(o, g);
  }
  return gains;
});
// eval.ts draws every question's random passages from one generator, question by question.
const rand = mulberry32(42);
const randomIdx = gainsOf.map((gains) => {
  const out: number[] = [];
  while (out.length < RANDOM) {
    const i = Math.floor(rand() * n);
    if (!gains.has(i)) out.push(i);
  }
  return out;
});
const keyword = await bm25(set);

function readVectors(path: string): { dims: number; v: Float32Array } {
  const meta = JSON.parse(readFileSync(`${path}.json`, "utf8")) as { dims: number; count: number };
  const b = readFileSync(`${path}.f32`);
  const v = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  if (v.length !== meta.count * meta.dims) throw new Error(`${path}: ${v.length} values for ${meta.count} × ${meta.dims}`);
  return { dims: meta.dims, v };
}

/** Cosine similarity of question `qi` to every passage, in doubles as eval.ts sums them. */
function similarities(q: { dims: number; v: Float32Array }, p: { dims: number; v: Float32Array }, qi: number, out: Float64Array): Float64Array {
  const dims = p.dims;
  const qv = q.v.subarray(qi * dims, (qi + 1) * dims);
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    const off = i * dims;
    for (let k = 0; k < dims; k++) s += qv[k]! * p.v[off + k]!;
    out[i] = s;
  }
  return out;
}

/** Passage `r`'s place among `count` passages: one plus those nearer, ties by passage id descending. */
function rankOf(sims: Float64Array, r: number, id: (i: number) => string, skip?: Set<number>): number {
  let rank = 1;
  for (let i = 0; i < sims.length; i++) {
    if (i === r || skip?.has(i)) continue;
    if (sims[i]! > sims[r]! || (sims[i] === sims[r] && id(i) > id(r))) rank++;
  }
  return rank;
}

/** One run's rankings, rebuilt as eval.ts builds them, or read from .cache/examples/. */
function rankings(spec: string): Ranked[] {
  const run = parseRun(spec);
  const file = `.cache/examples/${runKey(run)}/${SET}.json`;
  if (existsSync(file) && !process.env.RESCORE) return JSON.parse(readFileSync(file, "utf8")) as Ranked[];
  const t0 = Date.now();
  const q = readVectors(cachePath(run, SET, "query"));
  const p = readVectors(cachePath(run, SET, "document"));
  const sims = new Float64Array(n);
  const out = set.queries.map((_, qi): Ranked => {
    similarities(q, p, qi, sims);
    const top: number[] = [];
    for (let i = 0; i < n; i++) {
      if (top.length === TOP && sims[i]! <= sims[top[TOP - 1]!]!) continue;
      let j = top.length < TOP ? top.length : TOP - 1;
      top[j] = i;
      while (j > 0 && sims[top[j - 1]!]! < sims[i]!) {
        top[j] = top[j - 1]!;
        top[j - 1] = i;
        j--;
      }
    }
    const relevant: Ranked["relevant"] = {};
    for (const r of gainsOf[qi]!.keys()) relevant[r] = [rankOf(sims, r, pid), 1 - sims[r]!];
    return { top, topSims: top.map((i) => sims[i]!), relevant, random: randomIdx[qi]!.map((i) => 1 - sims[i]!) };
  });
  mkdirSync(`.cache/examples/${runKey(run)}`, { recursive: true });
  writeFileSync(file, JSON.stringify(out));
  console.log(`${runKey(run)}: ranked in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  return out;
}

/** pytrec_eval's order, as eval.ts: similarity descending, ties by passage id descending. */
const trec = (r: Ranked) =>
  r.top
    .map((i, k) => [i, r.topSims[k]!] as const)
    .sort((a, b) => b[1] - a[1] || (pid(a[0]) < pid(b[0]) ? 1 : pid(a[0]) > pid(b[0]) ? -1 : 0))
    .map(([i]) => i);
/** Stuga's hybrid, as eval.ts: the vector leg's nearest 96 and BM25's 96, fused, 24 kept. */
const hybrid = (qi: number, r: Ranked) => {
  const kw = keyword[set.queries[qi]!.id] ?? { ids: [], scores: [] };
  return rrf(kw, { ids: r.top.slice(0, DEPTH), sims: r.topSims.slice(0, DEPTH) }, pid, CANDIDATES);
};

const ranked = Object.fromEntries(Object.entries(RUNS).map(([name, spec]) => [name, rankings(spec)])) as Record<RunName, Ranked[]>;
const firstRank = (r: Ranked) => Math.min(...Object.values(r.relevant).map(([rank]) => rank));

// Check the rebuild against what eval.ts scored.
for (const [name, spec] of Object.entries(RUNS) as [RunName, string][]) {
  const scored = (JSON.parse(readFileSync(`.cache/eval/${runKey(parseRun(spec))}/${SET}.json`, "utf8")) as { scores: QuestionScores[] }).scores;
  let off = 0;
  set.queries.forEach((query, qi) => {
    const s = scored[qi]!;
    const r = ranked[name][qi]!;
    if (s.id !== query.id) throw new Error(`${name}: question ${qi} is ${s.id}, not ${query.id}`);
    if (Math.abs(ndcg(trec(r), gainsOf[qi]!) - s.dense.ndcg10) > 1e-12) off++;
    if ((firstRank(r) === 1 ? 1 : 0) !== s.dense.hit1) off++;
    if (Math.abs(ndcg(hybrid(qi, r), gainsOf[qi]!) - s.stuga["2"]!.ndcg10) > 1e-12) off++;
    if (r.random.some((d, k) => d !== s.random[k])) off++;
  });
  if (off) throw new Error(`${name}: ${off} differences from .cache/eval`);
}
console.log("MIRACL rebuild matches .cache/eval for every run");

// Texts, with titles apart for attribution.
const corpus = await readParquet<{ id: string; title: string; text: string }>(await hfFile(MIRACL.dataset, MIRACL.revision, "en-corpus/dev-00000-of-00001.parquet"));
const byId = new Map(corpus.map((d) => [d.id, d]));
const excerpt = (t: string, max = 160) => {
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  return `${cut.slice(0, Math.max(cut.lastIndexOf(" "), max - 20)).replace(/[\s,;:]+$/, "")}…`;
};
const round = (x: number) => Math.round(x * 1e4) / 1e4;
/** How MIRACL's annotators judged passage `i` for question `qi`. */
const judged = (qi: number, i: number) => {
  const g = set.qrels.get(set.queries[qi]!.id)?.get(pid(i));
  return g === undefined ? "not judged" : g > 0 ? "relevant" : "judged not relevant";
};
const passage = (i: number, extra: Record<string, unknown> = {}, full = false) => {
  const d = byId.get(pid(i))!;
  const page = /^(\d+)#/.exec(d.id)?.[1];
  return { id: d.id, title: d.title, ...(full ? { text: d.text } : { excerpt: excerpt(d.text) }), wikipedia: page ? `https://en.wikipedia.org/?curid=${page}` : null, ...extra };
};

const STARTS = /^(what|who|when|where|which|how|why|is|are|do|does|did|was|were)\b/i;
const words = (t: string) => t.trim().split(/\s+/).length;
const plain = (qi: number) => {
  const t = set.queries[qi]!.text;
  return words(t) <= 9 && !/\d/.test(t) && STARTS.test(t);
};
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor((xs.length - 1) / 2)]!;
/** The plain questions of a pool, `value` nearest the pool's median first, then the shorter question. */
function order(pool: number[], value: (qi: number) => number): { order: number[]; median: number } {
  const m = median(pool.map(value));
  const sorted = pool.filter(plain).sort((a, b) => Math.abs(value(a) - m) - Math.abs(value(b) - m) || set.queries[a]!.text.length - set.queries[b]!.text.length || a - b);
  return { order: sorted, median: m };
}
/** The question READ chose, checking that the ones passed over are exactly those before it. */
function choose(sorted: number[], read: { chosen: string; passedOver: [string, string][] }, idOf: (qi: number) => string): number {
  const k = sorted.findIndex((qi) => idOf(qi) === read.chosen);
  if (k < 0) throw new Error(`${read.chosen} is not in its pool`);
  const before = sorted.slice(0, k).map(idOf);
  if (before.join() !== read.passedOver.map(([id]) => id).join()) throw new Error(`read before ${read.chosen}: ${before.join()}, not ${read.passedOver.map(([id]) => id).join()}`);
  return sorted[k]!;
}
const qid = (qi: number) => set.queries[qi]!.id;
const question = (qi: number) => ({ id: qid(qi), text: set.queries[qi]!.text });
const distanceIn = (r: Ranked, i: number) => {
  const k = r.top.indexOf(i);
  return k < 0 ? null : round(1 - r.topSims[k]!);
};
const top3 = (qi: number, r: Ranked, list = trec(r)) =>
  list.slice(0, 3).map((i, k) => passage(i, { rank: k + 1, relevant: gainsOf[qi]!.has(i), judged: judged(qi, i), distance: distanceIn(r, i) }));
const all = set.queries.map((_, qi) => qi);
const inRange = (x: number, lo: number, hi: number) => x >= lo && x <= hi;
const readNotes = (read: { passedOver: [string, string][] }) => read.passedOver.map(([id, why]) => ({ ...question(set.queries.findIndex((q) => q.id === id)), why }));

// (a) v1 vs v2.
const aPool = all.filter((qi) => firstRank(ranked.v1[qi]!) === 1 && inRange(firstRank(ranked.v2[qi]!), 3, 10));
const aOpposite = all.filter((qi) => firstRank(ranked.v2[qi]!) === 1 && inRange(firstRank(ranked.v1[qi]!), 3, 10));
const a = order(aPool, (qi) => firstRank(ranked.v2[qi]!));
const aQi = choose(a.order, READ.v1VsV2, qid);
const aV1First = trec(ranked.v1[aQi]!)[0]!;
const aV2First = trec(ranked.v2[aQi]!)[0]!;
const firstOf = (name: RunName) => all.filter((qi) => firstRank(ranked[name][qi]!) === 1).length;
const textLength = (i: number) => byId.get(pid(i))!.text.length;
const shortFirst = (name: RunName) => all.filter((qi) => textLength(trec(ranked[name][qi]!)[0]!) < SHORT).length;
const v1VsV2 = {
  rule: "v1 places a relevant passage first; v2 (bf16) places its first relevant passage at rank 3–10",
  count: aPool.length,
  plainCount: a.order.length,
  medianV2Rank: a.median,
  opposite: { rule: "v2 (bf16) places a relevant passage first; v1 places its first relevant passage at rank 3–10", count: aOpposite.length },
  bothFirst: all.filter((qi) => firstRank(ranked.v1[qi]!) === 1 && firstRank(ranked.v2[qi]!) === 1).length,
  v1FirstOnly: all.filter((qi) => firstRank(ranked.v1[qi]!) === 1 && firstRank(ranked.v2[qi]!) > 1).length,
  v2FirstOnly: all.filter((qi) => firstRank(ranked.v2[qi]!) === 1 && firstRank(ranked.v1[qi]!) > 1).length,
  firstIsRelevant: {
    note: "questions (of 799) whose first passage is one MIRACL judged relevant; higher is better",
    v1: firstOf("v1"),
    v2: firstOf("v2"),
    v2Default: firstOf("v2Default"),
    v2Raw: firstOf("v2Raw"),
  },
  poolV2FirstJudged: {
    note: "in the pool, how MIRACL's annotators judged v2's first passage (unjudged ones may still answer the question)",
    judgedNotRelevant: aPool.filter((qi) => judged(qi, trec(ranked.v2[qi]!)[0]!) === "judged not relevant").length,
    notJudged: aPool.filter((qi) => judged(qi, trec(ranked.v2[qi]!)[0]!) === "not judged").length,
  },
  shortFirstPassage: {
    note: `exploratory: questions (of 799) whose first passage has under ${SHORT} characters of text (a heading, a list's lead-in, a sentence fragment); lower is better`,
    corpusShare: round(corpus.filter((d) => d.text.length < SHORT).length / corpus.length),
    v1: shortFirst("v1"),
    v2: shortFirst("v2"),
    v2Default: shortFirst("v2Default"),
    v2Raw: shortFirst("v2Raw"),
  },
  example: {
    question: question(aQi),
    relevantCount: gainsOf[aQi]!.size,
    v1: { top3: top3(aQi, ranked.v1[aQi]!), firstRelevantRank: firstRank(ranked.v1[aQi]!) },
    v2: { top3: top3(aQi, ranked.v2[aQi]!), firstRelevantRank: firstRank(ranked.v2[aQi]!) },
    v1First: passage(aV1First, { judged: judged(aQi, aV1First), v1Rank: 1, v2Rank: ranked.v2[aQi]!.relevant[aV1First]![0], v1Distance: round(ranked.v1[aQi]!.relevant[aV1First]![1]), v2Distance: round(ranked.v2[aQi]!.relevant[aV1First]![1]) }, true),
    v2First: passage(aV2First, { judged: judged(aQi, aV2First), v2Rank: 1, v2Distance: distanceIn(ranked.v2[aQi]!, aV2First), v1Rank: trec(ranked.v1[aQi]!).indexOf(aV2First) + 1 || null, v1Distance: distanceIn(ranked.v1[aQi]!, aV2First) }, true),
  },
  passedOver: readNotes(READ.v1VsV2),
};

// pg_search's tokens, as Stuga's English column indexes and queries them.
const sql = postgres(DATABASE_URL, { max: 1, onnotice: () => {} });
const tokenCache = new Map<string, string[]>();
async function tokens(text: string): Promise<string[]> {
  const hit = tokenCache.get(text);
  if (hit) return hit;
  const [row] = await sql.unsafe<{ t: string[] | null }[]>(`SELECT ($1::pdb.icu('stemmer=english', 'stopwords_language=english'))::text[] AS t`, [text]);
  const t = row?.t ?? [];
  tokenCache.set(text, t);
  return t;
}
/** The question's words whose tokens the passage holds, as written in the question. */
async function matchedWords(q: string, i: number): Promise<string[]> {
  const inPassage = new Set(await tokens(set.passages[i]!.text));
  const out: string[] = [];
  for (const w of q.match(/[\p{L}\p{N}]+(?:['’][\p{L}]+)?/gu) ?? []) {
    const t = await tokens(w);
    if (t.length && t.some((x) => inPassage.has(x)) && !out.includes(w)) out.push(w);
  }
  return out;
}
const fusedRank = (list: number[], gains: Map<number, number>) => {
  const k = list.findIndex((i) => gains.has(i));
  return k < 0 ? null : k + 1;
};

// (b) Hybrid rescues, v2 at bf16.
const bRule = (r: Ranked[]) => all.filter((qi) => firstRank(r[qi]!) > 10 && (fusedRank(hybrid(qi, r[qi]!), gainsOf[qi]!) ?? Infinity) <= 3);
const bOpposite = (r: Ranked[]) => all.filter((qi) => firstRank(r[qi]!) <= 3 && (fusedRank(hybrid(qi, r[qi]!), gainsOf[qi]!) ?? Infinity) > 10);
const bPool = bRule(ranked.v2);
const b = order(bPool, (qi) => firstRank(ranked.v2[qi]!));
const bQi = choose(b.order, READ.hybridRescues, qid);
const bFused = hybrid(bQi, ranked.v2[bQi]!);
const rescued = bFused.find((i) => gainsOf[bQi]!.has(i))!;
const kw = keyword[qid(bQi)]!;
const kwRank = (i: number) => {
  const k = kw.ids.indexOf(i);
  return k < 0 ? null : k + 1;
};
const vectorRank = (r: Ranked, i: number) => r.relevant[i]?.[0] ?? (trec(r).indexOf(i) + 1 || null);
const sameQuestion = (name: RunName) => ({ vectorsFirstRelevantRank: firstRank(ranked[name][bQi]!), hybridFirstRelevantRank: fusedRank(hybrid(bQi, ranked[name][bQi]!), gainsOf[bQi]!) });
const hybridRescues = {
  rule: "v2 (bf16) vectors alone place the first relevant passage below rank 10; Stuga's hybrid places a relevant passage in its top 3",
  count: bPool.length,
  plainCount: b.order.length,
  medianVectorRank: b.median,
  opposite: { rule: "vectors alone place a relevant passage in the top 3; hybrid places its first relevant passage below rank 10", count: bOpposite(ranked.v2).length },
  note: "on MIRACL, fusing BM25 lowered nDCG@10 for most runs (README); the opposite count is the honest context for this example",
  v2Default: { count: bRule(ranked.v2Default).length, opposite: bOpposite(ranked.v2Default).length },
  v1: { count: bRule(ranked.v1).length, opposite: bOpposite(ranked.v1).length },
  example: {
    question: question(bQi),
    relevantCount: gainsOf[bQi]!.size,
    vectors: { top3: top3(bQi, ranked.v2[bQi]!), firstRelevantRank: firstRank(ranked.v2[bQi]!) },
    hybrid: {
      top3: bFused.slice(0, 3).map((i, k) => passage(i, { rank: k + 1, relevant: gainsOf[bQi]!.has(i), judged: judged(bQi, i), vectorRank: vectorRank(ranked.v2[bQi]!, i), keywordRank: kwRank(i) })),
      firstRelevantRank: fusedRank(bFused, gainsOf[bQi]!),
    },
    rescued: passage(rescued, { judged: judged(bQi, rescued), vectorRank: ranked.v2[bQi]!.relevant[rescued]![0], keywordRank: kwRank(rescued), hybridRank: bFused.indexOf(rescued) + 1, matchedWords: await matchedWords(set.queries[bQi]!.text, rescued) }, true),
    sameQuestion: { v2Default: sameQuestion("v2Default"), v1: sameQuestion("v1") },
  },
  passedOver: readNotes(READ.hybridRescues),
};

// (c) Prompts matter, the nvfp4 tag with and without prompts.
const profile = profileOf(parseRun(RUNS.v2Default));
const cPool = all.filter((qi) => firstRank(ranked.v2Default[qi]!) === 1 && inRange(firstRank(ranked.v2Raw[qi]!), 3, 10));
const cOpposite = all.filter((qi) => firstRank(ranked.v2Raw[qi]!) === 1 && inRange(firstRank(ranked.v2Default[qi]!), 3, 10));
const c = order(cPool, (qi) => firstRank(ranked.v2Raw[qi]!));
const cQi = choose(c.order, READ.promptsMatter, qid);
const cFirst = trec(ranked.v2Default[cQi]!)[0]!;
const cRawFirst = trec(ranked.v2Raw[cQi]!)[0]!;
const promptsMatter = {
  rule: "v2 (nvfp4 tag) with its prompts places a relevant passage first; on raw text its first relevant passage is at rank 3–10",
  count: cPool.length,
  plainCount: c.order.length,
  medianRawRank: c.median,
  opposite: { rule: "raw text places a relevant passage first; with its prompts the first relevant passage is at rank 3–10", count: cOpposite.length },
  promptsFirstOnly: all.filter((qi) => firstRank(ranked.v2Default[qi]!) === 1 && firstRank(ranked.v2Raw[qi]!) > 1).length,
  rawFirstOnly: all.filter((qi) => firstRank(ranked.v2Raw[qi]!) === 1 && firstRank(ranked.v2Default[qi]!) > 1).length,
  prompts: { query: profile.query, document: profile.document },
  example: {
    question: question(cQi),
    sent: {
      queryWithPrompt: `${profile.query}${set.queries[cQi]!.text}`,
      queryRaw: set.queries[cQi]!.text,
      passageWithPrompt: `${profile.document}${excerpt(set.passages[cFirst]!.text, 120)}`,
      passageRaw: excerpt(set.passages[cFirst]!.text, 120),
    },
    relevantCount: gainsOf[cQi]!.size,
    withPrompts: { top3: top3(cQi, ranked.v2Default[cQi]!), firstRelevantRank: firstRank(ranked.v2Default[cQi]!) },
    raw: { top3: top3(cQi, ranked.v2Raw[cQi]!), firstRelevantRank: firstRank(ranked.v2Raw[cQi]!) },
    promptsFirst: passage(cFirst, { judged: judged(cQi, cFirst), withPromptsRank: 1, rawRank: ranked.v2Raw[cQi]!.relevant[cFirst]![0] }, true),
    rawFirst: passage(cRawFirst, { judged: judged(cQi, cRawFirst), rawRank: 1, withPromptsRank: trec(ranked.v2Default[cQi]!).indexOf(cRawFirst) + 1 || null }, true),
  },
  passedOver: readNotes(READ.promptsMatter),
};

// (d) A random pair under 0.6 for v2 and beyond v1's own 1% floor.
type CutoffStats = { random: { p1: number; p5: number; p50: number }; fixed: Record<string, { keepsRelevant: number; letsRandom: number }>; levels: Record<string, { cutoff: number; keepsRelevant: number }> };
const cutoffs = JSON.parse(readFileSync(`${RESULTS}/cutoffs.json`, "utf8")) as Record<string, Record<string, CutoffStats>>;
const stat = (name: RunName, s = SET) => cutoffs[runKey(parseRun(RUNS[name]))]![s]!;
const floorV1 = stat("v1").random.p1;
if (!(floorV1 > CUTOFF)) throw new Error(`v1's floor ${floorV1} is not above ${CUTOFF}`);
const m1 = stat("v1").random.p50;
const m2 = stat("v2Default").random.p50;
const pairs = all.flatMap((qi) => randomIdx[qi]!.map((i, k) => ({ qi, i, v1: ranked.v1[qi]!.random[k]!, v2: ranked.v2Default[qi]!.random[k]! })));
const meets = pairs.filter((p) => p.v2 < CUTOFF && p.v1 > floorV1);
const below = (name: RunName) => ranked[name].flatMap((r) => r.random).filter((d) => d < CUTOFF).length / pairs.length;
const candidates = meets.filter((p) => plain(p.qi)).sort((x, y) => Math.abs(x.v1 - m1) + Math.abs(x.v2 - m2) - (Math.abs(y.v1 - m1) + Math.abs(y.v2 - m2)));
const unrelated: typeof candidates = [];
for (const p of candidates) {
  const q = new Set(await tokens(set.queries[p.qi]!.text));
  if (!(await tokens(set.passages[p.i]!.text)).some((t) => q.has(t))) unrelated.push(p);
  if (unrelated.length === 6) break;
}
const [dPair] = unrelated;
if (!dPair) throw new Error("no unrelated pair");
const cutoffPair = {
  rule: `one of a question's ${RANDOM} random passages: v2 (nvfp4 tag) closer than ${CUTOFF}, v1 beyond its own 1% floor on MIRACL English (${round(floorV1)})`,
  note: "cosine distance: lower is closer; 0 is as close as it gets",
  pairs: pairs.length,
  count: meets.length,
  share: round(meets.length / pairs.length),
  v1Over06AndV2Under06: round(pairs.filter((p) => p.v2 < CUTOFF && p.v1 > CUTOFF).length / pairs.length),
  under06: {
    note: `share of random (unrelated) pairs closer than ${CUTOFF}, which a ${CUTOFF} cutoff lets through; lower is better`,
    v1: { share: round(below("v1")), cutoffsJson: stat("v1").fixed[String(CUTOFF)]!.letsRandom },
    v2Default: { share: round(below("v2Default")), cutoffsJson: stat("v2Default").fixed[String(CUTOFF)]!.letsRandom },
    v2: { share: round(below("v2")), cutoffsJson: stat("v2").fixed[String(CUTOFF)]!.letsRandom },
  },
  medianRandomDistance: { v1: round(m1), v2Default: round(m2) },
  floor1pc: { v1: round(floorV1), v2Default: round(stat("v2Default").random.p1) },
  example: {
    question: question(dPair.qi),
    randomPassage: passage(dPair.i, { judged: judged(dPair.qi, dPair.i), v1Distance: round(dPair.v1), v2Distance: round(dPair.v2), v2Bf16Distance: round(ranked.v2[dPair.qi]!.random[randomIdx[dPair.qi]!.indexOf(dPair.i)]!) }, true),
    relevantPassages: [...gainsOf[dPair.qi]!.keys()].map((i) => passage(i, { v1Distance: round(ranked.v1[dPair.qi]!.relevant[i]![1]), v2Distance: round(ranked.v2Default[dPair.qi]!.relevant[i]![1]) })),
  },
  alternates: unrelated.slice(1).map((p) => ({ ...question(p.qi), passage: passage(p.i), v1Distance: round(p.v1), v2Distance: round(p.v2) })),
};
await sql.end();

// (e) A relevant BRIGHT passage at or beyond v1's own 1% floor there.
const bright = await loadSet("bright-so");
const bOrd = new Map(bright.passages.map((d, i) => [d.id, i]));
const bPid = (i: number) => bright.passages[i]!.id;
const goldOf = (qi: number) => [...(bright.qrels.get(bright.queries[qi]!.id)?.entries() ?? [])].filter(([id, g]) => g > 0 && bOrd.has(id)).map(([id]) => bOrd.get(id)!);
const brightScores = (name: RunName) => (JSON.parse(readFileSync(`.cache/eval/${runKey(parseRun(RUNS[name]))}/bright-so.json`, "utf8")) as { scores: QuestionScores[] }).scores;
const bScored = { v1: brightScores("v1"), v2Default: brightScores("v2Default") };
const bv1 = { q: readVectors(cachePath(parseRun(RUNS.v1), "bright-so", "query")), p: readVectors(cachePath(parseRun(RUNS.v1), "bright-so", "document")) };
const bFloor = stat("v1", "bright-so").random.p1;
const bRandom = bScored.v1.flatMap((s) => s.random).sort((x, y) => x - y);
const bSims = new Float64Array(bright.passages.length);
const gold = bright.queries.flatMap((query, qi) => {
  const s = bScored.v1[qi]!;
  if (s.id !== query.id || bScored.v2Default[qi]!.id !== query.id) throw new Error(`bright-so: question ${qi} is not ${query.id}`);
  similarities(bv1.q, bv1.p, qi, bSims);
  const excluded = new Set([...(bright.excluded.get(query.id) ?? [])].map((id) => bOrd.get(id)!));
  return goldOf(qi).map((i, k) => {
    const d = 1 - bSims[i]!;
    // eval.ts keeps each gold distance in the same order.
    if (d !== s.gold[k]) throw new Error(`bright-so ${query.id}: ${bPid(i)} at ${d}, eval.ts has ${s.gold[k]}`);
    return { qi, i, d, v2: bScored.v2Default[qi]!.gold[k]!, golds: s.gold.length, rank: d >= bFloor ? rankOf(bSims, i, bPid, excluded) : null };
  });
});
console.log("BRIGHT distances match .cache/eval");
const far = gold.filter((g) => g.d >= bFloor);
const farPool = far.filter((g) => g.golds === 1).sort((x, y) => bright.queries[x.qi]!.text.length - bright.queries[y.qi]!.text.length || x.qi - y.qi);
const eQi = choose(farPool.map((g) => g.qi), READ.brightFar, (qi) => bright.queries[qi]!.id);
const eGold = farPool.find((g) => g.qi === eQi)!;
const eText = bright.passages[eGold.i]!.text;
const brightFar = {
  rule: "a relevant BRIGHT passage (gold: a page the accepted answer links) that v1 puts at or beyond its own 1% floor on BRIGHT",
  note: "cosine distance: lower is closer. A floor is the distance only 1% of random (unrelated) question–passage pairs fall below; a cutoff there drops everything farther",
  floor1pc: { v1: round(bFloor), v2Default: round(stat("v2Default", "bright-so").random.p1) },
  v1RandomDistance: { p1: round(bFloor), p5: round(stat("v1", "bright-so").random.p5), p50: round(stat("v1", "bright-so").random.p50) },
  goldPassages: gold.length,
  questions: bright.queries.length,
  atOrBeyondFloor: far.length,
  shareAtOrBeyondFloor: round(far.length / gold.length),
  keptByFloor: { note: "share of relevant passages a cutoff at v1's floor keeps; higher is better", share: round(1 - far.length / gold.length), cutoffsJson: stat("v1", "bright-so").levels.balanced!.keepsRelevant },
  questionsWithOne: new Set(far.map((g) => g.qi)).size,
  questionsWithEveryGoldBeyond: bright.queries.filter((_, qi) => gold.some((g) => g.qi === qi) && gold.filter((g) => g.qi === qi).every((g) => g.d >= bFloor)).length,
  singleGoldQuestions: bright.queries.filter((_, qi) => goldOf(qi).length === 1).length,
  singleGoldAtOrBeyondFloor: farPool.length,
  farRanks: {
    note: "where v1 ranks the relevant passages at or beyond its floor, among all passages for the question: Stuga's vector leg takes the nearest 96 and hands 24 on",
    withinTop24: far.filter((g) => g.rank! <= CANDIDATES).length,
    withinTop96: far.filter((g) => g.rank! <= DEPTH).length,
    medianRank: median(far.map((g) => g.rank!)),
  },
  example: {
    question: { id: bright.queries[eQi]!.id, firstLine: bright.queries[eQi]!.text.split("\n")[0], text: bright.queries[eQi]!.text },
    passage: { id: bPid(eGold.i), start: excerpt(eText, 300), text: eText },
    v1: {
      distance: round(eGold.d),
      floor: round(bFloor),
      rank: eGold.rank,
      randomCloser: { note: "share of v1's random BRIGHT pairs nearer than this relevant passage", share: round(bRandom.filter((x) => x < eGold.d).length / bRandom.length) },
    },
    v2Default: { distance: round(eGold.v2), floor: round(stat("v2Default", "bright-so").random.p1), atOrBeyondFloor: eGold.v2 >= stat("v2Default", "bright-so").random.p1 },
  },
  passedOver: READ.brightFar.passedOver.map(([id, why]) => ({ id, firstLine: bright.queries.find((q) => q.id === id)!.text.split("\n")[0], why })),
  source: {
    ...BRIGHT,
    licence: "CC BY 4.0",
    attribution: "BRIGHT (xlangai/BRIGHT), StackOverflow subset: the question is a Stack Overflow post and the passage a chunk of a page its accepted answer links, as BRIGHT stores them",
  },
};

const result = {
  names: {
    v1: "EmbeddingGemma v1: Google's EmbeddingGemma, Ollama tag embeddinggemma:300m",
    v2: "EmbeddingGemma v2: EmbeddingGemma 2 at bf16, Ollama tag embeddinggemma-2:270m-bf16-text",
    v2Default: "EmbeddingGemma v2 at Ollama's default 4-bit (nvfp4) tag, embeddinggemma-2:270m",
    v2Raw: "the same default tag sent raw text, without its prompts",
  },
  set: SET,
  source: { ...MIRACL, split: "dev", questions: set.queries.length, passages: n },
  licence: "Question and passage text from MIRACL (MMTEB's MIRACLRetrievalHardNegatives), CC BY-SA 4.0; each passage is from the English Wikipedia article named in its title. brightFar quotes BRIGHT, CC BY 4.0.",
  runs: { v1: runKey(parseRun(RUNS.v1)), v2: runKey(parseRun(RUNS.v2)), v2Default: runKey(parseRun(RUNS.v2Default)), v2Raw: runKey(parseRun(RUNS.v2Raw)) },
  plainQuestions: all.filter(plain).length,
  v1VsV2,
  hybridRescues,
  promptsMatter,
  cutoffPair,
  brightFar,
};
const json = JSON.stringify(result, null, 2);
if (OUT) writeFileSync(OUT, `${json}\n`);
else console.log(json);
