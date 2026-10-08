/**
 * Stuga's keyword leg over a test set, in pg_search: the chunk index of
 * packages/db/src/schema/search-indexes.ts (English-stemmed ICU over all text, plus the set's
 * language column, gated to its script) and the leg's query from packages/db/src/search.ts
 * (`disjunction_max` of `match_disjunction` per column, by BM25 score, 96 deep), each copied
 * verbatim apart from table and column names. A node searching this set has the set's language
 * turned on; English is always on, and Thai has no tokenizer of its own in pg_search, so Thai
 * text gets only the ICU column, as on a node.
 *
 * Results are kept in .cache/bm25/<set>.json. BM25_VARIANT tries another keyword leg on the same
 * passages: `icu` queries the general column alone, and `jieba` the Chinese column as Stuga built it
 * before its tokenizer revision zh2 folded traditional characters to simplified (table
 * p_miracl_zh_jieba).
 */
export const VARIANT = process.env.BM25_VARIANT ?? "";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import postgres from "postgres";
import type { TestSet } from "./sets.ts";

export const DEPTH = 96;
export const DATABASE_URL = process.env.BENCH_DATABASE_URL ?? "postgres://bench@127.0.0.1:55441/bench";

const HAN = "㐀-䶿一-鿿";
const KANA = "ぁ-ヿ";
const ENGLISH = "'stemmer=english', 'stopwords_language=english'";

const LANGUAGE_COLUMNS: Record<string, { script: string; unless?: string; tokenizer: (alias: string) => string }> = {
  ar: { script: "؀-ۿ", tokenizer: (alias) => `pdb.icu('stemmer=arabic', 'alias=${alias}')` },
  ja: { script: KANA, tokenizer: (alias) => `pdb.lindera(japanese, 'alias=${alias}')` },
  ko: { script: "가-힣", tokenizer: (alias) => `pdb.lindera(korean, 'alias=${alias}')` },
  zh: {
    script: HAN,
    unless: KANA,
    tokenizer: (alias) => (VARIANT === "jieba" ? `pdb.jieba('alias=${alias}')` : `pdb.jieba('chinese_convert=t2s', 'alias=${alias}')`),
  },
};

function gated(lang: string, expr: string): string {
  const { script, unless } = LANGUAGE_COLUMNS[lang]!;
  const skip = unless ? ` AND (${expr}) !~ '[${unless}]'` : "";
  return `(CASE WHEN (${expr}) ~ '[${script}]'${skip} THEN (${expr}) ELSE '' END)`;
}

export interface Bm25Hits {
  /** query id → passage indexes, best first, with their scores. */
  [queryId: string]: { ids: number[]; scores: number[] };
}

export async function bm25(set: TestSet): Promise<Bm25Hits> {
  const out = `.cache/bm25/${set.name}${VARIANT ? `.${VARIANT}` : ""}.json`;
  if (existsSync(out)) return JSON.parse(readFileSync(out, "utf8")) as Bm25Hits;
  const sql = postgres(DATABASE_URL, { max: 4, onnotice: () => {} });
  const table = `p_${set.name.replaceAll("-", "_")}${VARIANT === "jieba" ? "_jieba" : ""}`;
  const langs = VARIANT !== "icu" && set.language in LANGUAGE_COLUMNS ? [set.language] : [];
  try {
    const [{ n } = { n: 0 }] = await sql<{ n: number }[]>`SELECT count(*)::int AS n FROM pg_class WHERE relname = ${table}`;
    if (n === 0) {
      // One transaction, so a failed build leaves no table behind to be mistaken for a finished one.
      await sql.begin(async (tx) => {
        await tx.unsafe(`CREATE TABLE ${table} (ord int NOT NULL, pid text NOT NULL, body text NOT NULL)`);
        for (let i = 0; i < set.passages.length; i += 5000) {
          // Postgres text holds no NUL, nor does a Stuga document; a few BRIGHT pages carry them.
          const rows = set.passages.slice(i, i + 5000).map((p, k) => ({ ord: i + k, pid: p.id, body: p.text.replaceAll("\0", "") }));
          await tx`INSERT INTO ${tx(table)} ${tx(rows)}`;
        }
        const extra = langs.map((l) => `,\n (${gated(l, "body")}::${LANGUAGE_COLUMNS[l]!.tokenizer(`chunk_text_${l}`)})`).join("");
        const t0 = Date.now();
        await tx.unsafe(`CREATE INDEX ${table}_bm25 ON ${table} USING bm25 (ord, ((body)::pdb.icu(${ENGLISH}, 'alias=chunk_text'))${extra}) WITH (key_field = 'ord')`);
        console.log(`  ${set.name}: indexed ${set.passages.length} passages in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
      });
    }
    const hits: Bm25Hits = {};
    for (const q of set.queries) {
      const legs = [`paradedb.match_disjunction('chunk_text', $1)`, ...langs.map((l) => `paradedb.match_disjunction('chunk_text_${l}', $1)`)].join(", ");
      // Passages a set excludes for the question are never ranked for it, so the leg reaches past them.
      const excluded = set.excluded.get(q.id) ?? new Set<string>();
      const rows = (
        await sql.unsafe<{ ord: number; s: number }[]>(
          `SELECT c.ord, pdb.score(c) AS s FROM ${table} c WHERE c.ord @@@ paradedb.disjunction_max(ARRAY[${legs}]) ORDER BY s DESC LIMIT ${DEPTH + excluded.size}`,
          [q.text],
        )
      )
        .filter((r) => !excluded.has(set.passages[r.ord]!.id))
        .slice(0, DEPTH);
      hits[q.id] = { ids: rows.map((r) => r.ord), scores: rows.map((r) => Number(r.s)) };
    }
    mkdirSync(".cache/bm25", { recursive: true });
    writeFileSync(out, JSON.stringify(hits));
    return hits;
  } finally {
    await sql.end();
  }
}
