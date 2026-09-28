/**
 * Summarise results/<set>/<date>/raw.jsonl into summary.json and a Markdown
 * leaderboard, which also goes into README.md between that set's leaderboard markers.
 *
 *   node src/report.ts results/miracl/2026-09-28 [--readme]
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { CANDIDATES, isGold, loadCandidates, loadChunks, loadQuestions, setDir, type Question, type SetId } from "./corpus.ts";
import { bootstrapMean, firstGoldRank, hitAt, mean, ndcgAt, quantile, reciprocalRank } from "./metrics.ts";
import { ACCOUNT_ERROR } from "./retry.ts";
import type { Row } from "./run.ts";
import type { Price, RankerKind } from "./types.ts";

const dir = process.argv.slice(2).find((a) => !a.startsWith("--"));
if (!dir) throw new Error("usage: node src/report.ts results/<set>/<date> [--readme]");
const meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8")) as {
  set: SetId;
  rankers: { id: string; label: string; kind: RankerKind; via: string; price: Price; repeats: number }[];
};
// A rerun appends; the latest row for each call is the one that counts. A call our own account
// refused measured nothing about the ranker and is left out until it runs again.
const latest = new Map<string, Row>();
for (const line of readFileSync(`${dir}/raw.jsonl`, "utf8").split("\n").filter(Boolean)) {
  const r = JSON.parse(line) as Row;
  latest.set(`${r.ranker}|${r.repeat}|${r.question}`, r);
}
const rows = [...latest.values()].filter((r) => !ACCOUNT_ERROR.test(r.error ?? ""));

const chunks = loadChunks(setDir(meta.set));
const questions = new Map(loadQuestions(setDir(meta.set)).map((q) => [q.id, q] as const));
const candidates = new Map(loadCandidates(setDir(meta.set)).map((c) => [c.question, c.ids] as const));
const goldOf = new Map<string, Set<string>>();
for (const q of questions.values()) goldOf.set(q.id, new Set(chunks.filter((c) => isGold(c, q)).map((c) => c.id)));

/** Questions with a gold passage among their candidates: the ones a reranker can get right. */
const rankable = [...questions.values()].filter((q) => (candidates.get(q.id) ?? []).some((id) => goldOf.get(q.id)!.has(id)));
const misses = [...questions.values()].filter((q) => !rankable.includes(q));

/** A call counts toward latency unless the vendor served most of its prompt from cache. */
const uncached = (r: Row) => !r.cachedTokens || r.cachedTokens < r.inputTokens / 2;

/** List price of one call: cached and cache-written input at their own rates where the price has them. */
function costPerSearch(r: Row, price: Price): number {
  if ("perSearch" in price) return price.perSearch ?? Number.NaN;
  const read = r.cachedTokens ?? 0;
  const write = r.cacheWriteTokens ?? 0;
  const fresh = r.inputTokens - read - write;
  return (fresh * price.input + read * (price.cacheRead ?? price.input) + write * (price.cacheWrite ?? price.input) + r.outputTokens * price.output) / 1e6;
}

/** Share of scored calls whose top score two passages shared. */
function tieRate(rows: Row[]): number {
  const scored = rows.filter((r) => r.scores && r.scores.length > 1);
  const tied = scored.filter((r) => {
    const s = [...r.scores!].sort((a, b) => b - a);
    return s[0] === s[1];
  });
  return scored.length ? tied.length / scored.length : 0;
}

interface Summary {
  id: string;
  label: string;
  kind: RankerKind;
  via: string;
  /** First public release of the model, YYYY-MM-DD; null for the fusion baseline. */
  released: string | null;
  models: string[];
  repeats: number;
  calls: number;
  failures: number;
  hit1: number;
  hit1Ci: [number, number];
  recall8: number;
  mrr: number;
  mrrCi: [number, number];
  ndcg10: number;
  latencyP50: number;
  latencyP95: number;
  /** NaN when the vendor publishes no price. */
  costPer1k: number;
  /** Mean tokens per search, over every call. */
  inputTokens: number;
  outputTokens: number;
  stability: number | null;
  /** Share of answered calls whose top score was shared, so fusion order broke the tie. */
  topTieRate: number;
  /** Share of calls that reported reasoning tokens. */
  reasoningRate: number;
  byCategory: Record<string, { n: number; hit1: number; mrr: number }>;
  /** Hit@1 over every question, a retrieval miss counting as a miss. */
  hit1All: number;
  /**
   * Hit@1 minus Jev's, and minus the first stage's, over the same questions: a paired bootstrap 95%
   * interval, and one Bonferroni-adjusted for every ranker compared with the same reference.
   */
  vsJev: Paired | null;
  vsFirstStage: Paired | null;
  /** LLM judges: of the usable answers whose coverage was recorded, the share that scored every candidate. */
  fullCoverage: number | null;
  coverageRecorded: number;
  /** Retried calls timed on their answering attempt only, from before end-to-end time was recorded. */
  untimedRetries: number;

}

// Release dates, keyed by ranker id; a variant ("-per-pair") shares its model's.
const models = JSON.parse(readFileSync("data/models.json", "utf8")) as Record<string, { released: string; source: string }>;
const releasedOf = (id: string) => models[id.replace(/-(per-pair|text)$/, "")]?.released ?? null;

type Paired = { diff: number; ci: [number, number]; ciAdjusted: [number, number] };

const summaries: Summary[] = [];
/** Per ranker, 1 or 0 for each rankable question it answered: the input to paired comparisons. */
const hitsOf = new Map<string, Map<string, number>>();
for (const rk of meta.rankers) {
  const mine = rows.filter((r) => r.ranker === rk.id);
  if (!mine.length) continue;
  const perQuestion = (q: Question, f: (r: Row) => number) => mean(mine.filter((r) => r.question === q.id).map(f));
  const rankOf = (r: Row) => firstGoldRank(r.order ?? candidates.get(r.question)!, goldOf.get(r.question)!);
  const qs = rankable.filter((q) => mine.some((r) => r.question === q.id));
  const hit1s = qs.map((q) => perQuestion(q, (r) => hitAt(rankOf(r), 1)));
  hitsOf.set(rk.id, new Map(qs.map((q, k) => [q.id, hit1s[k]!] as const)));
  const mrrs = qs.map((q) => perQuestion(q, (r) => reciprocalRank(rankOf(r))));
  const ok = mine.filter((r) => !r.error);
  // A judge that answers unusably is still waited for and billed: latency and cost count every call.
  const lat = mine.filter((r) => rk.kind !== "baseline" && uncached(r)).map((r) => r.totalMs ?? r.ms);
  const withCoverage = ok.filter((r) => r.covered != null);

  let stability: number | null = null;
  if (rk.repeats > 1) {
    const stable = qs.filter((q) => {
      const tops = mine.filter((r) => r.question === q.id).map((r) => r.order?.[0]);
      return tops.length > 1 && tops.every((t) => t === tops[0]);
    });
    stability = qs.length ? stable.length / qs.length : null;
  }

  const byCategory: Summary["byCategory"] = {};
  for (const cat of new Set(qs.map((q) => q.category))) {
    const cq = qs.filter((q) => q.category === cat);
    byCategory[cat] = {
      n: cq.length,
      hit1: mean(cq.map((q) => perQuestion(q, (r) => hitAt(rankOf(r), 1)))),
      mrr: mean(cq.map((q) => perQuestion(q, (r) => reciprocalRank(rankOf(r))))),
    };
  }


  summaries.push({
    id: rk.id,
    label: rk.label,
    kind: rk.kind,
    via: rk.via,
    released: releasedOf(rk.id),
    models: [...new Set(ok.map((r) => r.model))],
    repeats: rk.repeats,
    calls: mine.length,
    failures: mine.filter((r) => r.error).length,
    hit1: mean(hit1s),
    hit1Ci: bootstrapMean(hit1s),
    recall8: mean(qs.map((q) => perQuestion(q, (r) => hitAt(rankOf(r), 8)))),
    mrr: mean(mrrs),
    mrrCi: bootstrapMean(mrrs),
    ndcg10: mean(qs.map((q) => perQuestion(q, (r) => ndcgAt(r.order ?? candidates.get(r.question)!, goldOf.get(r.question)!, 10)))),
    latencyP50: lat.length ? quantile(lat, 0.5) : 0,
    latencyP95: lat.length ? quantile(lat, 0.95) : 0,
    costPer1k: mean(mine.map((r) => costPerSearch(r, rk.price))) * 1000,
    inputTokens: Math.round(mean(mine.map((r) => r.inputTokens))),
    outputTokens: Math.round(mean(mine.map((r) => r.outputTokens))),
    stability,
    topTieRate: tieRate(ok),
    reasoningRate: mine.filter((r) => (r.reasoningTokens ?? 0) > 0).length / mine.length,
    byCategory,
    hit1All: hit1s.reduce((a, b) => a + b, 0) / questions.size,
    vsJev: null,
    vsFirstStage: null,
    fullCoverage: withCoverage.length ? withCoverage.filter((r) => r.covered === candidates.get(r.question)!.length).length / withCoverage.length : null,
    coverageRecorded: withCoverage.length,
    untimedRetries: mine.filter((r) => r.attempts > 1 && r.totalMs == null).length,

  });
}
// Paired comparisons: the same questions, so question difficulty cancels out. Comparing many rankers
// with one reference, the adjusted interval holds for all of them at once.
function paired(id: string, ref: string, comparisons: number): Paired | null {
  const a = hitsOf.get(id);
  const b = hitsOf.get(ref);
  if (!a || !b || id === ref) return null;
  const diffs = [...a].filter(([q]) => b.has(q)).map(([q, h]) => h - b.get(q)!);
  if (!diffs.length) return null;
  return { diff: mean(diffs), ci: bootstrapMean(diffs), ciAdjusted: bootstrapMean(diffs, 20_000, 7, 0.05 / comparisons) };
}
const others = (ref: string) => summaries.filter((s) => s.id !== ref && s.kind !== "baseline" && hitsOf.has(ref)).length;
for (const s of summaries) {
  s.vsJev = paired(s.id, "jev", others("jev"));
  s.vsFirstStage = paired(s.id, "fusion", others("fusion"));
}

summaries.sort((a, b) => b.mrr - a.mrr);

const summary = {
  set: meta.set,
  dir,
  questions: questions.size,
  rankable: rankable.length,
  /** Share of questions with a gold passage among their candidates. */
  candidateRecall: rankable.length / questions.size,
  retrievalMisses: misses.map((q) => q.id),
  rankers: summaries,
};
writeFileSync(`${dir}/summary.json`, JSON.stringify(summary, null, 2) + "\n");

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
const usd = (v: number) => (Number.isNaN(v) ? "not listed" : v === 0 ? "—" : v < 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(v < 10 ? 1 : 0)}`);
const table = [
  `Ranked on ${rankable.length} questions whose answer was among the ${CANDIDATES} candidates${misses.length ? ` (${misses.length} retrieval misses left out)` : ""}. Intervals are bootstrap 95%.`,
  "",
  "| Ranker | Kind | Hit@1 | MRR | Recall@8 | p50 latency | p95 latency | $ / 1k searches | Unusable answers |",
  "|---|---|---|---|---|---|---|---|---|",
  ...summaries.map(
    (s) =>
      `| ${s.label} | ${s.kind} | ${pct(s.hit1)} (${pct(s.hit1Ci[0])}–${pct(s.hit1Ci[1])}) | ${s.mrr.toFixed(3)} | ${pct(s.recall8)} | ${s.kind === "baseline" ? "—" : ms(s.latencyP50)} | ${s.kind === "baseline" ? "—" : ms(s.latencyP95)} | ${usd(s.costPer1k)} | ${s.failures ? pct(s.failures / s.calls) : "0"} |`,
  ),
].join("\n");
writeFileSync(`${dir}/summary.md`, table + "\n");
console.log(table);

// Only the run the README describes updates its leaderboard.
if (process.argv.includes("--readme") && existsSync("README.md")) {
  const readme = readFileSync("README.md", "utf8");
  const start = `<!-- leaderboard:${meta.set}:start -->`;
  const end = `<!-- leaderboard:${meta.set}:end -->`;
  if (readme.includes(start) && readme.includes(end)) {
    writeFileSync("README.md", readme.replace(new RegExp(`${start}[\\s\\S]*${end}`), `${start}\n${table}\n${end}`));
  }
}
