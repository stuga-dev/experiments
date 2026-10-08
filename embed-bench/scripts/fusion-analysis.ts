/**
 * The fusion questions, from the scored runs: how each keyword-leg weight does against Stuga's equal
 * weight, and for Chinese whether jieba beside ICU helps hybrid retrieval, at weight 1 and 0.5. Each
 * difference is a paired bootstrap over the same questions (95% interval).
 *
 *   node scripts/fusion-analysis.ts [out.json]
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { paired as bootstrap } from "../src/bootstrap.ts";
import type { QuestionScores } from "../src/eval.ts";

const WEIGHTS = ["w1", "w0.75", "w0.5", "w0.25", "w0"] as const;
const SETS = ["miracl-ko", "miracl-zh", "miracl-en", "miracl-th", "bright-so", "cmteb-ecom", "cmteb-medical", "cmteb-covid", "cmteb-du"];

function read(file: string): QuestionScores[] | null {
  return existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as { scores: QuestionScores[] }).scores : null;
}

const paired = (a: number[], b: number[]) => bootstrap(a, b, 11);

const pc = (x: number) => (100 * x).toFixed(1);
const ci = ([m, lo, hi]: [number, number, number]) => `${m >= 0 ? "+" : ""}${pc(m)} [${pc(lo)}, ${pc(hi)}]${lo > 0 ? " better" : hi < 0 ? " worse" : ""}`;
const out: Record<string, unknown> = {};

for (const backend of readdirSync(".cache/eval")) {
  for (const model of readdirSync(`.cache/eval/${backend}`)) {
    const run = `${backend}/${model}`;
    for (const set of SETS) {
      const s = read(`.cache/eval/${run}/${set}.json`);
      if (!s || !s[0]?.stuga.w1) continue;
      const metric = (key: string, m: "ndcg10" | "recall24") => s.map((q) => (key === "dense" ? q.dense[m] : q.stuga[key]![m]));
      const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
      const row: Record<string, unknown> = {
        weights: Object.fromEntries([...WEIGHTS, "dense"].map((w) => [w, { ndcg10: mean(metric(w, "ndcg10")), recall24: mean(metric(w, "recall24")) }])),
        half_vs_equal: { ndcg10: paired(metric("w1", "ndcg10"), metric("w0.5", "ndcg10")), recall24: paired(metric("w1", "recall24"), metric("w0.5", "recall24")) },
      };
      let jieba = "";
      const icu = read(`.cache/eval/${run}/${set}.icu.json`);
      if (icu && icu[0]?.stuga.w1 && icu.length === s.length) {
        const byId = new Map(icu.map((q) => [q.id, q]));
        const icuMetric = (key: string, m: "ndcg10" | "recall24") => s.map((q) => byId.get(q.id)!.stuga[key]![m]);
        const j = Object.fromEntries(
          (["w1", "w0.5"] as const).map((w) => [w, { ndcg10: paired(icuMetric(w, "ndcg10"), metric(w, "ndcg10")), recall24: paired(icuMetric(w, "recall24"), metric(w, "recall24")) }]),
        );
        row.jieba_vs_icu = j;
        jieba = `\n      jieba beside ICU, at w1: nDCG ${ci(j.w1!.ndcg10)} Recall ${ci(j.w1!.recall24)} | at w0.5: nDCG ${ci(j["w0.5"]!.ndcg10)} Recall ${ci(j["w0.5"]!.recall24)}`;
      }
      out[`${run} ${set}`] = row;
      const w = row.weights as Record<string, { ndcg10: number; recall24: number }>;
      const h = row.half_vs_equal as { ndcg10: [number, number, number]; recall24: [number, number, number] };
      console.log(
        `${model.slice(0, 22).padEnd(22)} ${set.padEnd(13)} ` +
          [...WEIGHTS, "dense"].map((k) => `${k} ${pc(w[k]!.ndcg10)}/${pc(w[k]!.recall24)}`).join("  ") +
          `\n      w0.5 vs w1: nDCG ${ci(h.ndcg10)} Recall ${ci(h.recall24)}${jieba}`,
      );
    }
  }
}
if (process.argv[2]) writeFileSync(process.argv[2], JSON.stringify(out, null, 2));
