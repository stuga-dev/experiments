/** Build each set's keyword leg: node src/bm25-run.ts [miracl-ko,bright-so,...] */
import { bm25 } from "./bm25.ts";
import { SET_NAMES, loadSet } from "./sets.ts";

for (const name of (process.argv[2] ?? SET_NAMES.join(",")).split(",")) {
  const set = await loadSet(name);
  const t0 = Date.now();
  const hits = await bm25(set);
  const empty = set.queries.filter((q) => !hits[q.id]?.ids.length).length;
  console.log(`${set.name}: ${set.queries.length} queries in ${((Date.now() - t0) / 1000).toFixed(0)}s, ${empty} with no keyword hit`);
}
