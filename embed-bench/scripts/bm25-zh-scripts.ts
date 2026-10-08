/**
 * MIRACL's Chinese passages are Wikipedia text in both scripts, its questions simplified. BM25 with
 * and without jieba, for questions whose relevant passages are traditional or simplified, told
 * apart by characters that exist in one script only. node scripts/bm25-zh-scripts.ts
 */
import postgres from "postgres";
import { ndcg, recall } from "../src/metrics.ts";
import { loadSet } from "../src/sets.ts";

// Common characters written differently in the two scripts.
const TRADITIONAL = new Set("國這說為們時會來對學與個體發過還後們經從當實現關機說開長門問間點爲氣樣種處車東認語讓書邊頭將員場區應總數變動進產業義務項標記號歷設計");
const SIMPLIFIED = new Set("国这说为们时会来对学与个体发过还后经从当实现关机开长门问间点气样种处车东认语让书边头将员场区应总数变动进产业义务项标记号历设计");
const script = (t: string) => {
  let tr = 0;
  let si = 0;
  for (const c of t) {
    if (TRADITIONAL.has(c)) tr++;
    if (SIMPLIFIED.has(c)) si++;
  }
  return tr > si ? "traditional" : "simplified";
};

const sql = postgres(process.env.BENCH_DATABASE_URL ?? "postgres://bench@127.0.0.1:55441/bench", { max: 4, onnotice: () => {} });
const set = await loadSet("miracl-zh");
const ord = new Map(set.passages.map((p, i) => [p.id, i]));
const scores: Record<string, Record<string, { n: number; r: number; q: number }>> = {};
for (const legs of [["chunk_text"], ["chunk_text", "chunk_text_zh"]]) {
  const key = legs.join(" + ");
  for (const q of set.queries) {
    const gold = [...(set.qrels.get(q.id) ?? [])].filter(([, g]) => g > 0).map(([id]) => id);
    const kind = script(gold.map((id) => set.passages[ord.get(id)!]?.text ?? "").join(""));
    const rows = await sql.unsafe<{ ord: number }[]>(
      `SELECT c.ord, pdb.score(c) AS s FROM ${process.env.ZH_TABLE ?? "p_miracl_zh"} c WHERE c.ord @@@ paradedb.disjunction_max(ARRAY[${legs.map((l) => `paradedb.match_disjunction('${l}', $1)`).join(", ")}]) ORDER BY s DESC LIMIT 96`,
      [q.text],
    );
    const gains = new Map(gold.filter((id) => ord.has(id)).map((id) => [ord.get(id)!, 1]));
    const cell = ((scores[kind] ??= {})[key] ??= { n: 0, r: 0, q: 0 });
    cell.n += ndcg(rows.map((x) => x.ord), gains);
    cell.r += recall(rows.map((x) => x.ord), gains, 24);
    cell.q++;
  }
}
for (const [kind, byLegs] of Object.entries(scores)) {
  for (const [legs, c] of Object.entries(byLegs)) {
    console.log(`${kind.padEnd(12)} ${String(c.q).padStart(4)} questions  ${legs.padEnd(28)} nDCG@10 ${((100 * c.n) / c.q).toFixed(1)}  Recall@24 ${((100 * c.r) / c.q).toFixed(1)}`);
  }
}
await sql.end();
