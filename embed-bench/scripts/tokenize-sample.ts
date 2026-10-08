/** How the general and the language tokenizer split a few questions and a relevant passage: node scripts/tokenize-sample.ts zh 3 */
import postgres from "postgres";
import { loadSet } from "../src/sets.ts";

const lang = process.argv[2] ?? "zh";
const n = Number(process.argv[3] ?? 3);
const sql = postgres(process.env.BENCH_DATABASE_URL ?? "postgres://bench@127.0.0.1:55441/bench", { onnotice: () => {} });
const lt = lang === "zh" ? "pdb.jieba" : lang === "ko" ? "pdb.lindera(korean)" : "pdb.lindera(japanese)";
const set = await loadSet(`miracl-${lang}`);
const byId = new Map(set.passages.map((p) => [p.id, p.text]));
for (const q of set.queries.slice(0, n)) {
  const gold = [...(set.qrels.get(q.id) ?? [])].find(([, g]) => g > 0)?.[0];
  const passage = (gold ? byId.get(gold) : "")?.slice(0, 120) ?? "";
  for (const [label, text] of [["Q", q.text], ["P", passage]] as const) {
    const [icu] = await sql.unsafe<{ t: string[] }[]>(`SELECT ($1::pdb.icu('stemmer=english', 'stopwords_language=english'))::text[] AS t`, [text]);
    const [lng] = await sql.unsafe<{ t: string[] }[]>(`SELECT ($1::${lt})::text[] AS t`, [text]);
    console.log(`${label}: ${text.replaceAll("\n", " ")}\n   icu  : ${icu!.t.join(" | ")}\n   ${lang}   : ${lng!.t.join(" | ")}`);
  }
  console.log();
}
await sql.end();
