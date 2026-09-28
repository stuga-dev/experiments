/**
 * data/miracl/: MIRACL as a reranking set, from MMTEB's MIRACLReranking dev split, which carries
 * MIRACL's queries and relevance judgments with a first-stage candidate list per query. Six
 * languages; each question keeps the first 24 candidates of its list, in the list's order. Questions
 * are drawn with a fixed seed from those with a relevant passage among their first 24.
 *
 *   node scripts/build-miracl.ts [--per-language 40]
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { CANDIDATES, type CandidateSet, type Chunk, type Question } from "../src/corpus.ts";
import { mulberry32 } from "../src/metrics.ts";
import { hfFile, readParquet } from "./parquet.ts";

const DATASET = "mteb/MIRACLReranking";
const REVISION = "ab6f54eff185a84bc1f6ab96b56bc7df87433228";
/** The languages of Stuga's privacy-laws sample. */
const LANGUAGES = ["en", "zh", "ja", "ko", "ar", "th"] as const;
const i = process.argv.indexOf("--per-language");
const PER_LANGUAGE = i >= 0 ? Number(process.argv[i + 1]) : 40;
const SEED = 20260928;

const corpus: Chunk[] = [];
const questions: Question[] = [];
const candidates: CandidateSet[] = [];
/** Per language: dev questions, and those with a relevant passage among their first 24 candidates. */
const eligibility: Record<string, { dev: number; eligible: number }> = {};

for (const lang of LANGUAGES) {
  const file = (part: string) => hfFile(DATASET, REVISION, `${lang}-${part}/dev-00000-of-00001.parquet`);
  const queries = await readParquet<{ _id: string; text: string }>(await file("queries"));
  const qrels = await readParquet<{ "query-id": string; "corpus-id": string; score: number | bigint }>(await file("qrels"));
  const lists = await readParquet<{ "query-id": string; "corpus-ids": string[] }>(await file("top_ranked"));
  const docs = new Map((await readParquet<{ _id: string; title: string; text: string }>(await file("corpus"))).map((d) => [d._id, d] as const));

  const relevant = new Set(qrels.filter((r) => Number(r.score) > 0).map((r) => `${r["query-id"]}|${r["corpus-id"]}`));
  const listOf = new Map(lists.map((l) => [l["query-id"], l["corpus-ids"].slice(0, CANDIDATES)] as const));
  const eligible = queries.filter((q) => listOf.get(q._id)?.some((id) => relevant.has(`${q._id}|${id}`)));

  // Fisher–Yates with a seeded generator: the same draw on every rebuild.
  const rand = mulberry32(SEED);
  const pool = [...eligible];
  for (let k = pool.length - 1; k > 0; k--) {
    const j = Math.floor(rand() * (k + 1));
    [pool[k], pool[j]] = [pool[j]!, pool[k]!];
  }
  for (const q of pool.slice(0, PER_LANGUAGE)) {
    const ids = listOf.get(q._id)!;
    const id = `${lang}-${q._id.replace(/^dev_query/, "")}`;
    for (const cid of ids) {
      const d = docs.get(cid);
      if (!d) throw new Error(`${lang}: no passage ${cid}`);
      corpus.push({ id: `${lang}/${cid}`, sample: lang, title: d.title, section: "", text: d.text });
    }
    questions.push({ id, sample: lang, q: q.text, category: lang, goldIds: ids.filter((cid) => relevant.has(`${q._id}|${cid}`)).map((cid) => `${lang}/${cid}`) });
    candidates.push({ question: id, ids: ids.map((cid) => `${lang}/${cid}`) });
  }
  eligibility[lang] = { dev: queries.length, eligible: eligible.length };
  console.log(`${lang}: ${Math.min(PER_LANGUAGE, eligible.length)} of ${eligible.length} eligible questions (${queries.length} in dev)`);
}

const out = "data/miracl";
mkdirSync(out, { recursive: true });
const jsonl = (rows: unknown[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
writeFileSync(`${out}/corpus.jsonl`, jsonl(corpus));
writeFileSync(`${out}/questions.jsonl`, jsonl(questions));
writeFileSync(`${out}/candidates.jsonl`, jsonl(candidates));
writeFileSync(`${out}/eligibility.json`, JSON.stringify(eligibility, null, 2) + "\n");
console.log(`${questions.length} questions, ${corpus.length} passages → ${out}/`);
