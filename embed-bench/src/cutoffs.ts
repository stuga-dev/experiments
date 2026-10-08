/**
 * What a cosine-distance cutoff does to each model: the share of relevant passages it keeps and the
 * share of random passages it lets through, at two fixed cutoffs (0.6 and 0.9) and at each model's
 * own noise floor: the distance below which 0.1%, 1% or 5% of random
 * query-passage pairs fall. A floor is also set on the other sets and applied to each one, to see
 * whether it carries over between languages.
 *
 *   node src/cutoffs.ts [out dir]
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import type { QuestionScores } from "./eval.ts";

const SETS = ["miracl-ko", "miracl-zh", "miracl-en", "miracl-th", "bright-so"];
const FIXED = [0.6, 0.9];
const LEVELS = { strict: 0.001, balanced: 0.01, loose: 0.05 } as const;
const OUT = process.argv[2];

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]!;
const share = (xs: number[], cutoff: number) => xs.filter((x) => x < cutoff).length / xs.length;

interface SetData {
  /** Every relevant passage's distance, so a question with more relevant passages counts more, as recall does. */
  relevant: number[];
  random: number[];
}

const report: Record<string, unknown> = {};
for (const backend of readdirSync(".cache/eval")) {
  for (const model of readdirSync(`.cache/eval/${backend}`)) {
    const key = `${backend}/${model}`;
    const sets = new Map<string, SetData>();
    for (const set of SETS) {
      const f = `.cache/eval/${key}/${set}.json`;
      if (!existsSync(f)) continue;
      const scores = (JSON.parse(readFileSync(f, "utf8")) as { scores: QuestionScores[] }).scores;
      sets.set(set, {
        relevant: scores.flatMap((s) => s.gold).sort((a, b) => a - b),
        random: scores.flatMap((s) => s.random).sort((a, b) => a - b),
      });
    }
    if (!sets.size) continue;
    const perSet: Record<string, unknown> = {};
    for (const [set, d] of sets) {
      const floors = Object.fromEntries(Object.entries(LEVELS).map(([level, a]) => [level, quantile(d.random, a)]));
      // The floor from the other MIRACL sets of this model, applied here.
      const others = [...sets.entries()].filter(([s]) => s !== set && s.startsWith("miracl-")).flatMap(([, o]) => o.random).sort((a, b) => a - b);
      const transfer = others.length
        ? Object.fromEntries(Object.entries(LEVELS).map(([level, a]) => {
            const c = quantile(others, a);
            return [level, { cutoff: c, keepsRelevant: share(d.relevant, c), letsRandom: share(d.random, c) }];
          }))
        : null;
      perSet[set] = {
        relevant: { p5: quantile(d.relevant, 0.05), p50: quantile(d.relevant, 0.5), p95: quantile(d.relevant, 0.95) },
        random: { p01: quantile(d.random, 0.001), p1: quantile(d.random, 0.01), p5: quantile(d.random, 0.05), p50: quantile(d.random, 0.5) },
        fixed: Object.fromEntries(FIXED.map((c) => [c, { keepsRelevant: share(d.relevant, c), letsRandom: share(d.random, c) }])),
        levels: Object.fromEntries(Object.entries(floors).map(([level, c]) => [level, { cutoff: c, keepsRelevant: share(d.relevant, c) }])),
        transfer,
      };
    }
    report[key] = perSet;
    console.log(`\n${key}`);
    for (const [set, r] of Object.entries(perSet) as [string, any][]) {
      const pc = (x: number) => `${(100 * x).toFixed(1)}%`;
      console.log(
        `  ${set.padEnd(10)} relevant p50 ${r.relevant.p50.toFixed(3)}  random p1 ${r.random.p1.toFixed(3)} p50 ${r.random.p50.toFixed(3)}` +
          `  | 0.6 keeps ${pc(r.fixed["0.6"].keepsRelevant)} lets ${pc(r.fixed["0.6"].letsRandom)}` +
          `  | balanced ${r.levels.balanced.cutoff.toFixed(3)} keeps ${pc(r.levels.balanced.keepsRelevant)}` +
          (r.transfer ? `  | from others ${r.transfer.balanced.cutoff.toFixed(3)} keeps ${pc(r.transfer.balanced.keepsRelevant)} lets ${pc(r.transfer.balanced.letsRandom)}` : ""),
      );
    }
  }
}
if (OUT) {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/cutoffs.json`, JSON.stringify(report, null, 2));
}
