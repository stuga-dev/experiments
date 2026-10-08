/**
 * Embed sets' passages and queries with one model.
 *
 *   node src/embed-run.ts ollama:embeddinggemma-2:270m [miracl-ko,miracl-zh,...,bright-so]
 *   node src/embed-run.ts ollama:embeddinggemma-2:270m+raw miracl-ko
 */
import { parseRun } from "./models.ts";
import { embedTexts } from "./embedder.ts";
import { loadSet, SET_NAMES } from "./sets.ts";

const run = parseRun(process.argv[2]!);
const names = (process.argv[3] ?? SET_NAMES.filter((s) => s.startsWith("miracl-")).join(",")).split(",");
for (const name of names) {
  const set = await loadSet(name);
  const t0 = Date.now();
  console.log(`${run.model}${run.raw ? " (raw)" : ""} ${set.name}: ${set.queries.length} queries, ${set.passages.length} passages`);
  await embedTexts(run, set.name, "query", set.queries.map((q) => q.text));
  const p = await embedTexts(run, set.name, "document", set.passages.map((d) => d.text));
  console.log(`  done ${set.name} in ${((Date.now() - t0) / 60_000).toFixed(1)} min, ${p.dims} dims`);
}
