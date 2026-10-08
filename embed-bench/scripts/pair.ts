/**
 * Two runs on the same sets, question by question: dense and Stuga hybrid nDCG@10, the difference
 * (second minus first) and its paired bootstrap 95% interval.
 *
 *   node scripts/pair.ts ollama/embeddinggemma-2:270m ollama/embeddinggemma:300m cmteb-du,cmteb-ecom [out.json]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { paired } from "../src/bootstrap.ts";
import type { QuestionScores } from "../src/eval.ts";

const [a, b, sets, out] = process.argv.slice(2) as [string, string, string, string | undefined];
const read = (run: string, set: string) => (JSON.parse(readFileSync(`.cache/eval/${run}/${set}.json`, "utf8")) as { scores: QuestionScores[] }).scores;
const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;


const pc = (x: number) => (100 * x).toFixed(1);
const result: Record<string, unknown> = {};
for (const set of sets.split(",")) {
  const byId = new Map(read(a, set).map((q) => [q.id, q]));
  const qs = read(b, set).filter((q) => byId.has(q.id));
  const row: Record<string, unknown> = { questions: qs.length };
  for (const [name, of] of [["dense", (q: QuestionScores) => q.dense.ndcg10], ["hybrid", (q: QuestionScores) => q.stuga["2"]!.ndcg10]] as const) {
    const first = qs.map((q) => of(byId.get(q.id)!)), second = qs.map(of);
    const [diff, low, high] = paired(first, second, 11);
    row[name] = { [a]: mean(first), [b]: mean(second), diff, low, high };
    console.log(`${set.padEnd(14)} ${name.padEnd(7)} ${pc(mean(first))} → ${pc(mean(second))}  ${diff >= 0 ? "+" : ""}${pc(diff)} [${pc(low)}, ${pc(high)}]`);
  }
  result[set] = row;
}
if (out) writeFileSync(out, JSON.stringify({ first: a, second: b, bootstrap: 2000, sets: result }, null, 2));
