/**
 * What turning a language on under Search languages does to the keyword leg, set by set: BM25 over
 * the general ICU column alone, with the language column beside it as Stuga queries them, and the
 * language column alone. For Chinese, also the column as Stuga built it before folding traditional
 * characters to simplified (tables ending _jieba, built with BM25_VARIANT=jieba node src/bm25-run.ts).
 *
 *   node scripts/bm25-languages.ts miracl-zh,cmteb-du,...
 */
import postgres from "postgres";
import { paired as bootstrap } from "../src/bootstrap.ts";
import { ndcg, recall } from "../src/metrics.ts";
import { loadSet } from "../src/sets.ts";

const sql = postgres(process.env.BENCH_DATABASE_URL ?? "postgres://bench@127.0.0.1:55441/bench", { max: 4, onnotice: () => {} });
const names = (process.argv[2] ?? "miracl-zh,miracl-ko,miracl-ja").split(",");
const out: Record<string, Record<string, { ndcg10: number; recall24: number; diff?: { ndcg10: number[]; recall24: number[] } }>> = {};

const paired = (a: number[], b: number[]) => bootstrap(a, b, 7);
for (const name of names) {
  const set = await loadSet(name);
  const lang = set.language;
  const ord = new Map(set.passages.map((p, i) => [p.id, i]));
  const table = `p_${name.replaceAll("-", "_")}`;
  const variants: Array<[string, string, string[]]> = [
    ["general only", table, ["chunk_text"]],
    ["general + language (Stuga)", table, ["chunk_text", `chunk_text_${lang}`]],
    ["language only", table, [`chunk_text_${lang}`]],
  ];
  if (lang === "zh") {
    variants.push(["general + jieba without folding", `${table}_jieba`, ["chunk_text", "chunk_text_zh"]]);
    variants.push(["jieba without folding only", `${table}_jieba`, ["chunk_text_zh"]]);
  }
  out[name] = {};
  const perQuestion: Record<string, { n: number[]; r: number[] }> = {};
  for (const [label, t, legs] of variants) {
    let n = 0;
    let r = 0;
    const pq = (perQuestion[label] = { n: [] as number[], r: [] as number[] });
    for (const q of set.queries) {
      const rows = await sql.unsafe<{ ord: number }[]>(
        `SELECT c.ord, pdb.score(c) AS s FROM ${t} c WHERE c.ord @@@ paradedb.disjunction_max(ARRAY[${legs.map((l) => `paradedb.match_disjunction('${l}', $1)`).join(", ")}]) ORDER BY s DESC LIMIT 96`,
        [q.text],
      );
      const gains = new Map<number, number>();
      for (const [id, g] of set.qrels.get(q.id) ?? []) if (g > 0 && ord.has(id)) gains.set(ord.get(id)!, g);
      const qn = ndcg(rows.map((x) => x.ord), gains);
      const qr = recall(rows.map((x) => x.ord), gains, 24);
      pq.n.push(qn);
      pq.r.push(qr);
      n += qn;
      r += qr;
    }
    out[name][label] = { ndcg10: n / set.queries.length, recall24: r / set.queries.length };
    const base = perQuestion["general only"]!;
    let vs = "";
    if (label !== "general only") {
      const dn = paired(base.n, pq.n);
      const dr = paired(base.r, pq.r);
      out[name][label]!.diff = { ndcg10: dn, recall24: dr };
      const f = (x: number[]) => `${x[0]! >= 0 ? "+" : ""}${(100 * x[0]!).toFixed(1)} [${(100 * x[1]!).toFixed(1)}, ${(100 * x[2]!).toFixed(1)}]`;
      vs = `   vs general only: nDCG ${f(dn)}  Recall ${f(dr)}`;
    }
    console.log(`${name.padEnd(13)} ${label.padEnd(34)} nDCG@10 ${((100 * n) / set.queries.length).toFixed(1)}  Recall@24 ${((100 * r) / set.queries.length).toFixed(1)}${vs}`);
  }
}
if (process.argv[3]) (await import("node:fs")).writeFileSync(process.argv[3], JSON.stringify(out, null, 2));
await sql.end();
