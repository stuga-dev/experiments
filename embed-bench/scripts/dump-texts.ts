/** Write a set's queries and passages, in the benchmark's order, for scripts/st-embed.py: node scripts/dump-texts.ts ko */
import { mkdirSync, writeFileSync } from "node:fs";
import { miracl } from "../src/sets.ts";

const set = await miracl(process.argv[2]!);
mkdirSync(".cache/texts", { recursive: true });
writeFileSync(`.cache/texts/${set.name}-queries.json`, JSON.stringify(set.queries.map((q) => q.text)));
writeFileSync(`.cache/texts/${set.name}-passages.json`, JSON.stringify(set.passages.map((p) => p.text)));
console.log(`${set.name}: ${set.queries.length} queries, ${set.passages.length} passages`);
