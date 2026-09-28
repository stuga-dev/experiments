/**
 * results/bright-stackoverflow/<date>/analysis.json: why BRIGHT's first-stage order is hard to beat,
 * from the data and results already here. No model is called.
 *
 * - How long the passages are, against the 1,200 characters a judge sees.
 * - BM25 over each question's own 24 candidates, from whole passages and from what a judge sees.
 * - The two first stages over every question: which found a gold passage, which put one first.
 * - Per ranker, the questions where it kept, gained or lost a gold passage in first place.
 * - Given a second run whose judges read the excerpt (`--snippet excerpt`), each ranker's Hit@1
 *   under both on the same questions, and how often the excerpt differs from the start on MIRACL.
 *
 *   node scripts/analyze-bright.ts 2026-09-27 [2026-09-28-excerpt]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { loadCandidates, loadChunks, loadQuestions, setDir, type Chunk, type SetId } from "../src/corpus.ts";
import { excerpts } from "../src/excerpt.ts";
import { bootstrapMean, mean, quantile } from "../src/metrics.ts";
import { ACCOUNT_ERROR } from "../src/retry.ts";
import type { Row } from "../src/run.ts";
import { SNIPPET_CHARS } from "../src/stuga.ts";

const date: string = process.argv[2] ?? "";
const excerptRun = process.argv[3];
if (!date) throw new Error("usage: node scripts/analyze-bright.ts <date> [<excerpt run>]");
const HYBRID = "bright-stackoverflow";
const LEXICAL = "bright-stackoverflow-bm25";
/** Pyserini's defaults, as in build-bright.ts. */
const K1 = 0.9;
const B = 0.4;
const tokens = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];

function load(set: SetId) {
  const dir = setDir(set);
  const chunks = new Map(loadChunks(dir).map((c) => [c.id, c] as const));
  const questions = loadQuestions(dir);
  const candidates = new Map(loadCandidates(dir).map((c) => [c.question, c.ids] as const));
  const isGold = (qid: string, id: string) => questions.find((q) => q.id === qid)!.goldIds.includes(id);
  const firstGold = (qid: string) => candidates.get(qid)!.findIndex((id) => isGold(qid, id)) + 1;
  const rankable = questions.filter((q) => firstGold(q.id) > 0);
  return { chunks, questions, candidates, isGold, firstGold, rankable };
}

/** Hit@1 when BM25 orders each rankable question's candidates, reading `cut` characters of each. */
function bm25Hit1(set: ReturnType<typeof load>, cut: number): number {
  const docs = new Map([...set.chunks.values()].map((c: Chunk) => [c.id, tokens(c.text.slice(0, cut))] as const));
  const df = new Map<string, number>();
  for (const d of docs.values()) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1);
  const avg = [...docs.values()].reduce((a, d) => a + d.length, 0) / docs.size;
  const score = (query: string[], doc: string[]) => {
    const tf = new Map<string, number>();
    for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1);
    let s = 0;
    for (const t of new Set(query)) {
      const f = tf.get(t);
      if (!f) continue;
      const idf = Math.log(1 + (docs.size - df.get(t)! + 0.5) / (df.get(t)! + 0.5));
      s += (idf * f * (K1 + 1)) / (f + K1 * (1 - B + (B * doc.length) / avg));
    }
    return s;
  };
  const hits = set.rankable.filter((q) => {
    const query = tokens(q.q);
    const ids = set.candidates.get(q.id)!;
    // Ties keep the first stage's order.
    const best = ids.map((id, i) => ({ id, i, s: score(query, docs.get(id)!) })).sort((a, b) => b.s - a.s || a.i - b.i)[0]!;
    return set.isGold(q.id, best.id);
  });
  return hits.length / set.rankable.length;
}

/** A run's rows that count, as report.ts reads them: the latest per call, none our account refused. */
function rowsOf(set: SetId, run: string): Row[] {
  const latest = new Map<string, Row>();
  for (const line of readFileSync(`results/${set}/${run}/raw.jsonl`, "utf8").split("\n").filter(Boolean)) {
    const r = JSON.parse(line) as Row;
    latest.set(`${r.ranker}|${r.repeat}|${r.question}`, r);
  }
  return [...latest.values()].filter((r) => r.ranker !== "fusion" && !ACCOUNT_ERROR.test(r.error ?? ""));
}

/** Whether a row put a gold passage first; an unusable answer keeps the first stage's order. */
const hit = (data: ReturnType<typeof load>, r: Row) => data.isGold(r.question, (r.order ?? data.candidates.get(r.question)!)[0]!);

/** Per ranker, on the rankable questions: first place gold under both, only the ranker, only the first stage. */
function vsFirstStage(set: SetId, data: ReturnType<typeof load>, run = date) {
  const rows = rowsOf(set, run);
  const out: Record<string, { questions: number; kept: number; gained: number; lost: number }> = {};
  for (const r of rows) {
    if (data.firstGold(r.question) === 0) continue;
    const before = data.firstGold(r.question) === 1;
    const after = hit(data, r);
    const o = (out[r.ranker] ??= { questions: 0, kept: 0, gained: 0, lost: 0 });
    o.questions++;
    if (before && after) o.kept++;
    else if (after) o.gained++;
    else if (before) o.lost++;
  }
  return out;
}

/**
 * Per ranker in both runs, Hit@1 reading the start and reading the excerpt, on the questions both
 * answered, with the paired difference's interval, also Bonferroni-adjusted for the rankers compared.
 */
function excerptEffect(set: SetId, data: ReturnType<typeof load>, run: string) {
  const byRanker = (rows: Row[]) => {
    const m = new Map<string, Map<string, number>>();
    for (const r of rows) if (data.firstGold(r.question) > 0) (m.get(r.ranker) ?? m.set(r.ranker, new Map()).get(r.ranker)!).set(r.question, hit(data, r) ? 1 : 0);
    return m;
  };
  const [start, excerpt] = [byRanker(rowsOf(set, date)), byRanker(rowsOf(set, run))];
  const ids = [...excerpt.keys()].filter((id) => start.has(id));
  return Object.fromEntries(
    ids.map((id) => {
      const qs = [...excerpt.get(id)!.keys()].filter((q) => start.get(id)!.has(q));
      const diffs = qs.map((q) => excerpt.get(id)!.get(q)! - start.get(id)!.get(q)!);
      return [
        id,
        {
          questions: qs.length,
          start: mean(qs.map((q) => start.get(id)!.get(q)!)),
          excerpt: mean(qs.map((q) => excerpt.get(id)!.get(q)!)),
          diff: mean(diffs),
          ci: bootstrapMean(diffs),
          ciAdjusted: bootstrapMean(diffs, 20_000, 7, 0.05 / ids.length),
        },
      ];
    }),
  );
}

/** On MIRACL's candidates, the share whose excerpt differs from their start, of all and of the relevant ones. */
function miraclExcerpt() {
  const m = load("miracl");
  let all = 0;
  let moved = 0;
  let gold = 0;
  let goldMoved = 0;
  for (const q of m.questions) {
    const ids = m.candidates.get(q.id)!;
    const texts = ids.map((id) => m.chunks.get(id)!.text);
    excerpts(q.q, texts, SNIPPET_CHARS).forEach((e, i) => {
      const differs = e !== texts[i]!.slice(0, SNIPPET_CHARS).trim();
      all++;
      if (differs) moved++;
      if (m.isGold(q.id, ids[i]!)) {
        gold++;
        if (differs) goldMoved++;
      }
    });
  }
  return { candidates: moved / all, relevant: goldMoved / gold };
}

const hybrid = load(HYBRID);
const lexical = load(LEXICAL);
const median = (xs: number[]) => quantile(xs, 0.5);
const goldSeen = hybrid.rankable.flatMap((q) => hybrid.candidates.get(q.id)!.filter((id) => hybrid.isGold(q.id, id)).map((id) => hybrid.chunks.get(id)!.text.length));

const both = (f: (d: ReturnType<typeof load>, qid: string) => boolean) => {
  const counts = { both: 0, withEmbeddingsOnly: 0, bm25Only: 0 };
  for (const q of hybrid.questions) {
    const [h, l] = [f(hybrid, q.id), f(lexical, q.id)];
    if (h && l) counts.both++;
    else if (h) counts.withEmbeddingsOnly++;
    else if (l) counts.bm25Only++;
  }
  return counts;
};

const analysis = {
  snippetChars: SNIPPET_CHARS,
  passageChars: {
    miracl: median(loadChunks(setDir("miracl")).map((c) => c.text.length)),
    bright: median([...hybrid.chunks.values()].map((c) => c.text.length)),
  },
  /** Share of the gold passages among rankable questions' candidates that run past what a judge sees. */
  goldPastSnippet: goldSeen.filter((n) => n > SNIPPET_CHARS).length / goldSeen.length,
  bm25OnCandidates: Object.fromEntries(
    ([[HYBRID, hybrid], [LEXICAL, lexical]] as const).map(([set, d]) => [set, { questions: d.rankable.length, whole: bm25Hit1(d, Infinity), snippet: bm25Hit1(d, SNIPPET_CHARS) }]),
  ),
  firstStages: {
    questions: hybrid.questions.length,
    found: both((d, qid) => d.firstGold(qid) > 0),
    first: both((d, qid) => d.firstGold(qid) === 1),
    /** Of the questions only the first stage with embeddings found, those it put a gold passage first for. */
    firstAmongWithEmbeddingsOnly: hybrid.questions.filter((q) => lexical.firstGold(q.id) === 0 && hybrid.firstGold(q.id) === 1).length,
  },
  vsFirstStage: { [HYBRID]: vsFirstStage(HYBRID, hybrid), [LEXICAL]: vsFirstStage(LEXICAL, lexical) },
  ...(excerptRun
    ? {
        excerpt: {
          run: excerptRun,
          miraclDiffers: miraclExcerpt(),
          effect: { [HYBRID]: excerptEffect(HYBRID, hybrid, excerptRun), [LEXICAL]: excerptEffect(LEXICAL, lexical, excerptRun) },
          vsFirstStage: { [HYBRID]: vsFirstStage(HYBRID, hybrid, excerptRun), [LEXICAL]: vsFirstStage(LEXICAL, lexical, excerptRun) },
        },
      }
    : {}),
};

const out = `results/${HYBRID}/${date}/analysis.json`;
writeFileSync(out, `${JSON.stringify(analysis, null, 2)}\n`);
console.log(`wrote ${out}`);
