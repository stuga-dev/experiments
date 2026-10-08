/**
 * The test sets, read from pinned Hugging Face revisions into .cache/.
 *
 * miracl: MMTEB's MIRACLRetrievalHardNegatives, the dev split, per language: every judged passage
 * plus the hard negatives MMTEB pooled for its queries. A passage is embedded as Stuga embeds a
 * document's first chunk: its title, a blank line, its text.
 *
 * bright-so: BRIGHT's StackOverflow subset, its whole corpus (long passages from documentation and
 * forums), with the passages BRIGHT excludes for a question left out of that question's ranking.
 */
import { hfFile, readParquet } from "../scripts/parquet.ts";

export interface Passage {
  id: string;
  /** What is embedded and BM25-indexed. */
  text: string;
}

export interface Query {
  id: string;
  text: string;
}

export interface TestSet {
  name: string;
  passages: Passage[];
  queries: Query[];
  /** query id → passage id → graded relevance (> 0 is relevant). */
  qrels: Map<string, Map<string, number>>;
  /** query id → passages never ranked for it. */
  excluded: Map<string, Set<string>>;
  /** The language a node searching it has on, for the keyword leg. */
  language: string;
}

export const BRIGHT = { dataset: "xlangai/BRIGHT", revision: "3066d29c9651a576c8aba4832d249807b181ecae", subset: "stackoverflow" };

/**
 * Chinese retrieval sets of C-MTEB beside MIRACL's Wikipedia: web search (DuReader), e-commerce
 * listings, medical answers and COVID-19 news and notices, each at a pinned revision.
 */
export const CMTEB = {
  du: { dataset: "DuRetrieval", revision: "a1a333e290fe30b10f3f56498e3a0d911a693ced", qrels: "497b7bd1bbb25cb3757ff34d95a8be50a3de2279" },
  ecom: { dataset: "EcomRetrieval", revision: "687de13dc7294d6fd9be10c6945f9e8fec8166b9", qrels: "39c90699b034ec22ac45b3abf5b0bbb5ffd421f9" },
  medical: { dataset: "MedicalRetrieval", revision: "2039188fb5800a9803ba5048df7b76e6fb151fc6", qrels: "37b8efec53c54c3d9c6af212f6710b62ccdf895c" },
  covid: { dataset: "CovidRetrieval", revision: "1271c7809071a13532e05f25fb53511ffce77117", qrels: "a9f41b7cdf24785531d12417ce0d1157ed4b39ca" },
} as const;

/** Every set by name: miracl-<lang>, bright-so and cmteb-<name>. */
export const SET_NAMES = [...["ko", "zh", "th", "ja", "ar", "en"].map((l) => `miracl-${l}`), "bright-so", ...Object.keys(CMTEB).map((k) => `cmteb-${k}`)];

/** The parquet files of a Hugging Face dataset revision under data/, by their prefix. */
async function hfData(dataset: string, revision: string, prefix: string): Promise<string> {
  const res = await fetch(`https://huggingface.co/api/datasets/${dataset}/revision/${revision}`);
  if (!res.ok) throw new Error(`${dataset}: HTTP ${res.status}`);
  const files = ((await res.json()) as { siblings: { rfilename: string }[] }).siblings.map((s) => s.rfilename);
  const file = files.find((f) => f.startsWith(`data/${prefix}`) && f.endsWith(".parquet"));
  if (!file) throw new Error(`${dataset}: no data/${prefix}*.parquet`);
  return hfFile(dataset, revision, file);
}

export async function cmteb(key: keyof typeof CMTEB): Promise<TestSet> {
  const { dataset, revision, qrels: qrelsRevision } = CMTEB[key];
  const corpus = await readParquet<{ id: string; text: string }>(await hfData(`C-MTEB/${dataset}`, revision, "corpus"));
  const queries = await readParquet<{ id: string; text: string }>(await hfData(`C-MTEB/${dataset}`, revision, "queries"));
  const rows = await readParquet<{ qid: string; pid: string; score: bigint | number }>(await hfData(`C-MTEB/${dataset}-qrels`, qrelsRevision, "dev"));
  const qrels = new Map<string, Map<string, number>>();
  for (const r of rows) {
    const m = qrels.get(String(r.qid)) ?? new Map<string, number>();
    m.set(String(r.pid), Number(r.score));
    qrels.set(String(r.qid), m);
  }
  return {
    name: `cmteb-${key}`,
    passages: corpus.map((d) => ({ id: String(d.id), text: d.text })),
    queries: queries.map((q) => ({ id: String(q.id), text: q.text })).filter((q) => [...(qrels.get(q.id)?.values() ?? [])].some((s) => s > 0)),
    qrels,
    excluded: new Map(),
    language: "zh",
  };
}

export function loadSet(name: string): Promise<TestSet> {
  if (name === "bright-so") return brightStackoverflow();
  if (name.startsWith("cmteb-")) return cmteb(name.slice("cmteb-".length) as keyof typeof CMTEB);
  if (name.startsWith("miracl-")) return miracl(name.slice("miracl-".length));
  throw new Error(`unknown set ${name}`);
}

export async function brightStackoverflow(): Promise<TestSet> {
  const file = (part: string) => hfFile(BRIGHT.dataset, BRIGHT.revision, `${part}/${BRIGHT.subset}-00000-of-00001.parquet`);
  const docs = await readParquet<{ id: string; content: string }>(await file("documents"));
  const examples = await readParquet<{ id: string; query: string; gold_ids: string[]; excluded_ids: string[] }>(await file("examples"));
  return {
    name: "bright-so",
    passages: docs.map((d) => ({ id: d.id, text: d.content })),
    queries: examples.map((e) => ({ id: e.id, text: e.query })),
    qrels: new Map(examples.map((e) => [e.id, new Map(e.gold_ids.map((g) => [g, 1]))])),
    excluded: new Map(examples.map((e) => [e.id, new Set(e.excluded_ids.filter((x) => x !== "N/A"))])),
    language: "en",
  };
}

export const MIRACL = { dataset: "mteb/MIRACLRetrievalHardNegatives", revision: "332a9acb49f5e83d5397683f79d23e588f685916" };
/** Smallest first, so a run's first results come early. */
export const MIRACL_LANGUAGES = ["ko", "zh", "th", "ja", "ar", "en"] as const;

export async function miracl(lang: string): Promise<TestSet> {
  const file = (part: string) => hfFile(MIRACL.dataset, MIRACL.revision, `${lang}-${part}/dev-00000-of-00001.parquet`);
  const corpus = await readParquet<{ id: string; title: string; text: string }>(await file("corpus"));
  const queries = await readParquet<{ id: string; text: string }>(await file("queries"));
  const rows = await readParquet<{ "query-id": string; "corpus-id": string; score: bigint | number }>(await file("qrels"));
  const qrels = new Map<string, Map<string, number>>();
  for (const r of rows) {
    const m = qrels.get(r["query-id"]) ?? new Map<string, number>();
    m.set(r["corpus-id"], Number(r.score));
    qrels.set(r["query-id"], m);
  }
  return {
    name: `miracl-${lang}`,
    passages: corpus.map((d) => ({ id: d.id, text: d.title ? `${d.title}\n\n${d.text}` : d.text })),
    // Only questions with a relevant passage count, as MMTEB scores them.
    queries: queries.filter((q) => [...(qrels.get(q.id)?.values() ?? [])].some((s) => s > 0)),
    qrels,
    excluded: new Map(),
    language: lang,
  };
}
