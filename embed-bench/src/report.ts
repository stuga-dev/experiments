/**
 * Summarise every scored model: per language and averaged over languages as MMTEB averages them,
 * each model's difference from the reference by a paired bootstrap over the same questions
 * (resampled within each language), and the distance quantiles cutoffs are chosen from.
 *
 *   node src/report.ts [reference run key] [out dir]
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import type { QuestionScores } from "./eval.ts";
import { MIRACL_LANGUAGES } from "./sets.ts";
import { mulberry32 } from "./bootstrap.ts";

const REFERENCE = process.argv[2] ?? "ollama/embeddinggemma-2:270m";
const OUT = process.argv[3];
/** The languages every model was run on; others (Thai, for EmbeddingGemma 2) are left out of the averages. */
const CORE = (process.env.REPORT_SETS ?? "ko,zh,en").split(",");
const BOOTSTRAP = 2000;

type Metric = { name: string; of: (s: QuestionScores) => number };
const METRICS: Metric[] = [
  { name: "dense nDCG@10", of: (s) => s.dense.ndcg10 },
  { name: "dense Recall@100", of: (s) => s.dense.recall100 },
  { name: "Stuga nDCG@10", of: (s) => s.stuga["0.9"]!.ndcg10 },
  { name: "Stuga Recall@24", of: (s) => s.stuga["0.9"]!.recall24 },
];

/** run key → language → questions. */
const runs = new Map<string, Map<string, QuestionScores[]>>();
for (const backend of readdirSync(".cache/eval")) {
  for (const model of readdirSync(`.cache/eval/${backend}`)) {
    const byLang = new Map<string, QuestionScores[]>();
    for (const lang of MIRACL_LANGUAGES) {
      const f = `.cache/eval/${backend}/${model}/miracl-${lang}.json`;
      if (existsSync(f)) byLang.set(lang, (JSON.parse(readFileSync(f, "utf8")) as { scores: QuestionScores[] }).scores);
    }
    if (byLang.size) runs.set(`${backend}/${model}`, byLang);
  }
}

const mean = (xs: number[]) => xs.reduce((a, x) => a + x, 0) / xs.length;
const quantile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]!;
};


/** Macro average over the languages both runs have, and its paired bootstrap interval against the reference. */
function compare(run: Map<string, QuestionScores[]>, ref: Map<string, QuestionScores[]> | undefined, m: Metric, langs: string[]) {
  const value = mean(langs.map((l) => mean(run.get(l)!.map(m.of))));
  if (!ref) return { value };
  // Questions aligned by id, so each draw takes the same questions from both runs.
  const pairs = langs.map((l) => {
    const refById = new Map(ref.get(l)!.map((s) => [s.id, s]));
    return run.get(l)!.map((s) => m.of(s) - m.of(refById.get(s.id)!));
  });
  const diff = mean(pairs.map(mean));
  const rand = mulberry32(7);
  const draws: number[] = [];
  for (let b = 0; b < BOOTSTRAP; b++) {
    draws.push(mean(pairs.map((d) => mean(Array.from({ length: d.length }, () => d[Math.floor(rand() * d.length)]!)))));
  }
  return { value, diff, low: quantile(draws, 0.025), high: quantile(draws, 0.975) };
}

const reference = runs.get(REFERENCE);
const report: Record<string, unknown> = {};
const pct = (x: number) => (100 * x).toFixed(1).padStart(5);
for (const [key, byLang] of [...runs.entries()].sort()) {
  const langs = MIRACL_LANGUAGES.filter((l) => CORE.includes(l) && byLang.has(l) && (!reference || reference.has(l)));
  const ref = key === REFERENCE ? undefined : reference;
  const metrics = Object.fromEntries(METRICS.map((m) => [m.name, compare(byLang, ref, m, langs)]));
  const perLanguage = Object.fromEntries(
    langs.map((l) => [l, Object.fromEntries(METRICS.map((m) => [m.name, mean(byLang.get(l)!.map(m.of))]))]),
  );
  const bm25 = mean(langs.map((l) => mean(byLang.get(l)!.map((s) => s.bm25.ndcg10))));
  // Each question weighs the same: its median relevant distance, and all 64 of its random ones.
  const gold = langs.flatMap((l) => byLang.get(l)!.map((s) => quantile(s.gold, 0.5)));
  const random = langs.flatMap((l) => byLang.get(l)!.flatMap((s) => s.random));
  const distances = {
    relevant: { p50: quantile(gold, 0.5), p90: quantile(gold, 0.9), p95: quantile(gold, 0.95), p99: quantile(gold, 0.99) },
    random: { p01: quantile(random, 0.001), p1: quantile(random, 0.01), p5: quantile(random, 0.05), p50: quantile(random, 0.5) },
  };
  report[key] = { languages: langs, metrics, perLanguage, bm25NDCG10: bm25, distances };
  console.log(`\n${key} (${langs.join(" ")})`);
  for (const m of METRICS) {
    const r = metrics[m.name] as { value: number; diff?: number; low?: number; high?: number };
    const vs = r.diff === undefined ? "" : `  vs reference ${r.diff >= 0 ? "+" : ""}${(100 * r.diff).toFixed(1)} [${(100 * r.low!).toFixed(1)}, ${(100 * r.high!).toFixed(1)}]`;
    console.log(`  ${m.name.padEnd(18)} ${pct(r.value)}${vs}   ${langs.map((l) => `${l} ${pct(perLanguage[l]![m.name]!)}`).join("  ")}`);
  }
  console.log(`  BM25 nDCG@10 ${pct(bm25)}   distances: relevant p50 ${distances.relevant.p50.toFixed(3)} p95 ${distances.relevant.p95.toFixed(3)}; random p1 ${distances.random.p1.toFixed(3)} p50 ${distances.random.p50.toFixed(3)}`);
}
if (OUT) {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/summary.json`, JSON.stringify({ reference: REFERENCE, bootstrap: BOOTSTRAP, runs: report }, null, 2));
}
