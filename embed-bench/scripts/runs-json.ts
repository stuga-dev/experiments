/** Gather the summary of every scored run and set into one file: node scripts/runs-json.ts results/<date>/runs.json */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";

const out: Record<string, Record<string, unknown>> = {};
for (const backend of readdirSync(".cache/eval")) {
  for (const model of readdirSync(`.cache/eval/${backend}`)) {
    for (const file of readdirSync(`.cache/eval/${backend}/${model}`).filter((f) => f.endsWith(".json")).sort()) {
      const { summary } = JSON.parse(readFileSync(`.cache/eval/${backend}/${model}/${file}`, "utf8")) as { summary: Record<string, unknown> };
      const keep = ["questions", "passages", "dims", "dense", "bm25", "stuga", "embed"];
      (out[`${backend}/${model}`] ??= {})[file.slice(0, -5)] = Object.fromEntries(keep.filter((k) => k in summary).map((k) => [k, summary[k]]));
    }
  }
}
writeFileSync(process.argv[2]!, JSON.stringify(out, null, 1));
console.log(Object.keys(out).length, "runs");
