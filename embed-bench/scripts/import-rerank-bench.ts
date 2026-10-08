/**
 * rerank-bench already embedded BRIGHT StackOverflow with text-embedding-3-large at 1,024
 * dimensions, the same texts this benchmark sends; copy those vectors into this cache rather than
 * pay for them again: node scripts/import-rerank-bench.ts ../rerank-bench
 */
import { copyFileSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { BATCH, cachePath } from "../src/embedder.ts";
import { loadSet } from "../src/sets.ts";

const from = `${process.argv[2]}/.cache/embed/text-embedding-3-large/bright-stackoverflow-3066d29c`;
const run = { backend: "openai" as const, model: "text-embedding-3-large", raw: false };
const set = await loadSet("bright-so");
for (const [role, part, count] of [["query", "queries", set.queries.length], ["document", "documents", set.passages.length]] as const) {
  const src = `${from}-${part}.f32`;
  if (statSync(src).size !== count * 1024 * 4) throw new Error(`${src}: not ${count} × 1024`);
  const to = cachePath(run, set.name, role);
  mkdirSync(to.slice(0, to.lastIndexOf("/")), { recursive: true });
  copyFileSync(src, `${to}.f32`);
  const done = Array.from({ length: Math.ceil(count / BATCH) }, (_, b) => b);
  writeFileSync(`${to}.json`, JSON.stringify({ dims: 1024, count, done, tokens: 0, seconds: 0, model: run.model, requested: 1024, prompt: "" }));
  console.log(`${set.name} ${part}: ${count} vectors`);
}
