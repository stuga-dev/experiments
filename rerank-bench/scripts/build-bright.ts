/**
 * data/bright-stackoverflow/ and data/bright-stackoverflow-bm25/: BRIGHT's StackOverflow subset as
 * two reranking sets, every question with 24 candidates from the subset's whole corpus, retrieved
 * from the question alone as Stuga retrieves. With embeddings: BM25 and vector similarity,
 * 96 each, fused by reciprocal rank (k = 60). Without it: BM25's first 24. Vectors are OpenAI's
 * text-embedding-3-large at 1,024 dimensions, Stuga's default, cached under .cache/. Passages BRIGHT
 * excludes for a question are left out.
 *
 *   BENCH_KEYS=keys.json node --max-old-space-size=8192 scripts/build-bright.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { CANDIDATES, type CandidateSet, type Chunk, type Question } from "../src/corpus.ts";
import { readKeys } from "../src/lineup.ts";
import { embed, EMBED_MODEL } from "./embed.ts";
import { hfFile, readParquet } from "./parquet.ts";

const DATASET = "xlangai/BRIGHT";
const REVISION = "3066d29c9651a576c8aba4832d249807b181ecae";
const SUBSET = "stackoverflow";
/** Pyserini's defaults. */
const K1 = 0.9;
const B = 0.4;

const examples = await readParquet<{ id: string; query: string; gold_ids: string[]; excluded_ids: string[] }>(await hfFile(DATASET, REVISION, `examples/${SUBSET}-00000-of-00001.parquet`));
const docs = await readParquet<{ id: string; content: string }>(await hfFile(DATASET, REVISION, `documents/${SUBSET}-00000-of-00001.parquet`));
/** Each leg's depth and the fusion constant, as in Stuga's retrieval. */
const LEG = 96;
const RRF_K = 60;

const tokens = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];

// An inverted index: term → [doc index, term frequency] pairs.
const postings = new Map<string, number[]>();
const lengths = new Float64Array(docs.length);
docs.forEach((d, i) => {
  const tf = new Map<string, number>();
  for (const t of tokens(d.content)) tf.set(t, (tf.get(t) ?? 0) + 1);
  lengths[i] = [...tf.values()].reduce((a, b) => a + b, 0);
  for (const [t, n] of tf) {
    const list = postings.get(t);
    if (list) list.push(i, n);
    else postings.set(t, [i, n]);
  }
});
const avgLength = lengths.reduce((a, b) => a + b, 0) / docs.length;

function bm25(query: string): Float64Array {
  const scores = new Float64Array(docs.length);
  for (const t of new Set(tokens(query))) {
    const list = postings.get(t);
    if (!list) continue;
    const df = list.length / 2;
    const idf = Math.log(1 + (docs.length - df + 0.5) / (df + 0.5));
    for (let k = 0; k < list.length; k += 2) {
      const i = list[k]!;
      const tf = list[k + 1]!;
      scores[i] = scores[i]! + (idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * lengths[i]!) / avgLength));
    }
  }
  return scores;
}

const openai = readKeys().openai;
if (!openai) throw new Error("the vector leg needs an OpenAI key in BENCH_KEYS");
const cache = `.cache/embed/${EMBED_MODEL}/bright-${SUBSET}-${REVISION.slice(0, 8)}`;
console.log(`embedding ${docs.length} passages with ${EMBED_MODEL}`);
const docVectors = await embed(docs.map((d) => d.content), openai, `${cache}-documents.f32`);
const queryVectors = await embed(examples.map((e) => e.query), openai, `${cache}-queries.f32`);
const dot = (a: Float32Array, b: Float32Array) => {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k]! * b[k]!;
  return s;
};

/** `snowflake_docs/flatten_2_0.txt` → `snowflake_docs/flatten_2_0`: the file names the source and page. */
const titleOf = (id: string) => id.replace(/\.txt$/, "");

/** Two first stages over the same questions: BM25 with embeddings, as Stuga with Embeddings on, and BM25 alone. */
const stages = { "bright-stackoverflow": "hybrid", "bright-stackoverflow-bm25": "bm25" } as const;
const built = Object.fromEntries(Object.keys(stages).map((set) => [set, { corpus: new Map<string, Chunk>(), candidates: [] as CandidateSet[], rankable: 0 }]));
const questions: Question[] = [];
for (const [e, ex] of examples.entries()) {
  const excluded = new Set(ex.excluded_ids);
  const top96 = (scores: Float64Array) =>
    [...scores.keys()]
      .filter((i) => scores[i]! > 0 && !excluded.has(docs[i]!.id))
      .sort((a, b) => scores[b]! - scores[a]!)
      .slice(0, LEG);
  const lexical = top96(bm25(ex.query));
  const semantic = top96(Float64Array.from(docVectors, (v) => 1 + dot(v, queryVectors[e]!)));
  const fused = new Map<number, number>();
  for (const list of [lexical, semantic]) {
    list.forEach((i, rank) => fused.set(i, (fused.get(i) ?? 0) + 1 / (RRF_K + rank + 1)));
  }
  const tops = {
    hybrid: [...fused.keys()].sort((a, b) => fused.get(b)! - fused.get(a)!).slice(0, CANDIDATES),
    bm25: lexical.slice(0, CANDIDATES),
  };
  const id = `so-${ex.id}`;
  questions.push({ id, sample: SUBSET, q: ex.query, category: SUBSET, goldIds: [...ex.gold_ids] });
  for (const [set, stage] of Object.entries(stages)) {
    const b = built[set]!;
    const top = tops[stage];
    for (const i of top) {
      const d = docs[i]!;
      b.corpus.set(d.id, { id: d.id, sample: SUBSET, title: titleOf(d.id), section: "", text: d.content });
    }
    b.candidates.push({ question: id, ids: top.map((i) => docs[i]!.id) });
    if (top.some((i) => ex.gold_ids.includes(docs[i]!.id))) b.rankable++;
  }
}

const jsonl = (rows: unknown[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
for (const [set, b] of Object.entries(built)) {
  const out = `data/${set}`;
  mkdirSync(out, { recursive: true });
  writeFileSync(`${out}/corpus.jsonl`, jsonl([...b.corpus.values()]));
  writeFileSync(`${out}/questions.jsonl`, jsonl(questions));
  writeFileSync(`${out}/candidates.jsonl`, jsonl(b.candidates));
  console.log(`${set}: ${questions.length} questions (${b.rankable} with a gold passage among the ${CANDIDATES} candidates), ${b.corpus.size} passages → ${out}/`);
}
