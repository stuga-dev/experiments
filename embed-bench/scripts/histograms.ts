/**
 * The distance distributions behind cutoffs.json, for drawing: per run and set, how many relevant
 * pairs (each question to each of its relevant passages) and random pairs (each question to its 64
 * random passages) fall in each 0.02-wide bin of cosine distance, over the whole range 0 to 2. Read
 * from the `gold` and `random` fields of .cache/eval/<backend>/<model>/<set>.json, on the sets
 * cutoffs.ts reads, every relevant passage counted as cutoffs.ts counts it.
 *
 *   node scripts/histograms.ts [out.json]
 *
 * Bin i holds distances in [i·0.02, (i+1)·0.02); the last bin also holds 2. A distance a hair below 0
 * (a passage identical to its query, in float arithmetic) counts in bin 0.
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import type { QuestionScores } from "../src/eval.ts";

const SETS = ["miracl-ko", "miracl-zh", "miracl-en", "miracl-th", "bright-so"];
/** Bins per unit of distance: 50 gives 0.02-wide bins, 100 of them from 0 to 2. */
const PER_UNIT = 50;
const BINS = 2 * PER_UNIT;

interface Histogram {
  n: number;
  counts: number[];
}

function histogram(xs: number[]): Histogram {
  const counts = Array.from({ length: BINS }, () => 0);
  for (const x of xs) counts[Math.min(BINS - 1, Math.max(0, Math.floor(x * PER_UNIT)))]!++;
  return { n: xs.length, counts };
}

/** The bin holding the most pairs, as its range, for the console. */
function mode(h: Histogram): string {
  const i = h.counts.indexOf(Math.max(...h.counts));
  return `${(i / PER_UNIT).toFixed(2)}–${((i + 1) / PER_UNIT).toFixed(2)}`;
}

const runs: Record<string, Record<string, { relevant: Histogram; random: Histogram }>> = {};
for (const backend of readdirSync(".cache/eval")) {
  for (const model of readdirSync(`.cache/eval/${backend}`)) {
    const run = `${backend}/${model}`;
    for (const set of SETS) {
      const f = `.cache/eval/${run}/${set}.json`;
      if (!existsSync(f)) continue;
      const scores = (JSON.parse(readFileSync(f, "utf8")) as { scores: QuestionScores[] }).scores;
      const relevant = histogram(scores.flatMap((s) => s.gold));
      const random = histogram(scores.flatMap((s) => s.random));
      (runs[run] ??= {})[set] = { relevant, random };
      console.log(
        `${model.slice(0, 32).padEnd(32)} ${set.padEnd(10)} relevant ${String(relevant.n).padStart(5)} most at ${mode(relevant)}` +
          `  | random ${String(random.n).padStart(6)} most at ${mode(random)}`,
      );
    }
  }
}
// Each histogram's counts on one line, so the file reads by run and set.
const json = JSON.stringify({ binWidth: 1 / PER_UNIT, bins: BINS, from: 0, runs }, null, 2).replace(/\[\s*([\d,\s]+?)\s*\]/g, (_, xs: string) => `[${xs.split(/\s*,\s*/).join(", ")}]`);
if (process.argv[2]) writeFileSync(process.argv[2], `${json}\n`);
