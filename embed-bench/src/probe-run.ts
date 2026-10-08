/**
 * Does a node's own measurement mean what it says? Embed Stuga's probe (data/probe.json, dumped
 * from Stuga as it sends it) with a model, set each level as a node does, then apply those
 * distances to MIRACL: the share of its random query–passage pairs they let through should be the
 * level's share (0.1%, 1%, 5%), and the share of relevant passages they keep should be high.
 *
 *   node src/probe-run.ts ollama:embeddinggemma-2:270m [out dir]
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { embedTexts } from "./embedder.ts";
import type { QuestionScores } from "./eval.ts";
import { parseRun, runKey } from "./models.ts";
import { loadSet } from "./sets.ts";
import { type QueryStyle, queryStyle } from "./stuga-cutoff.ts";
import { quantile, strictFromTail } from "./stuga-stats.ts";

interface Probe {
  digest: string;
  queries: { text: string; lang: string; topic: string; style: QueryStyle }[];
  passages: { text: string; lang: string; topic: string }[];
}

const SHARE = { strict: 0.001, balanced: 0.01, loose: 0.05 } as const;
const SETS = ["miracl-ko", "miracl-zh", "miracl-en", "miracl-th"];
const run = parseRun(process.argv[2]!);
const OUT = process.argv[3];
const probe = JSON.parse(readFileSync("data/probe.json", "utf8")) as Probe;
const domain = (topic: string) => topic.slice(0, topic.indexOf("."));

const q = await embedTexts(run, "probe", "query", probe.queries.map((x) => x.text));
const p = await embedTexts(run, "probe", "document", probe.passages.map((x) => x.text));
const vec = (e: { dims: number; vectors: Float32Array }, i: number) => e.vectors.subarray(i * e.dims, (i + 1) * e.dims);
const dist = (a: Float32Array, b: Float32Array) => {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k]! * b[k]!;
  return 1 - s;
};

// The node's measurement: unrelated pairs are a query and a passage of another domain in its language.
const nulls: Record<QueryStyle, number[]> = { short: [], question: [] };
probe.queries.forEach((qq, qi) => {
  probe.passages.forEach((pp, pi) => {
    if (pp.lang === qq.lang && domain(pp.topic) !== domain(qq.topic)) nulls[qq.style].push(dist(vec(q, qi), vec(p, pi)));
  });
});
const levels = Object.fromEntries(
  (["short", "question"] as const).map((s) => {
    const n = [...nulls[s]].sort((a, b) => a - b);
    const t1 = quantile(n, SHARE.balanced);
    const t5 = quantile(n, SHARE.loose);
    return [s, { strict: strictFromTail(t1, t5, n[0]!), balanced: t1, loose: t5 }];
  }),
) as Record<QueryStyle, Record<keyof typeof SHARE, number>>;

/** Each probe language's own floors, for asking whether a floor per script would carry better. */
const langOf = (set: string) => set.slice(set.indexOf("-") + 1);
const perLanguage = (lang: string) =>
  Object.fromEntries(
    (["short", "question"] as const).map((s) => {
      const n: number[] = [];
      probe.queries.forEach((qq, qi) => {
        if (qq.lang !== lang || qq.style !== s) return;
        probe.passages.forEach((pp, pi) => {
          if (pp.lang === lang && domain(pp.topic) !== domain(qq.topic)) n.push(dist(vec(q, qi), vec(p, pi)));
        });
      });
      n.sort((a, b) => a - b);
      const t1 = quantile(n, SHARE.balanced);
      const t5 = quantile(n, SHARE.loose);
      return [s, { strict: strictFromTail(t1, t5, n[0]!), balanced: t1, loose: t5 }];
    }),
  ) as Record<QueryStyle, Record<keyof typeof SHARE, number>>;

const sets: Record<string, unknown> = {};
for (const name of SETS) {
  const f = `.cache/eval/${runKey(run)}/${name}.json`;
  if (!existsSync(f)) continue;
  const set = await loadSet(name);
  const styleOf = new Map(set.queries.map((x) => [x.id, queryStyle(x.text)]));
  const scores = (JSON.parse(readFileSync(f, "utf8")) as { scores: QuestionScores[] }).scores;
  const row: Record<string, unknown> = {};
  for (const level of Object.keys(SHARE) as (keyof typeof SHARE)[]) {
    let randomIn = 0;
    let random = 0;
    let goldIn = 0;
    let gold = 0;
    for (const s of scores) {
      const c = levels[styleOf.get(s.id) ?? "question"][level];
      randomIn += s.random.filter((d) => d < c).length;
      random += s.random.length;
      goldIn += s.gold.filter((d) => d < c).length;
      gold += s.gold.length;
    }
    const lets = randomIn / random;
    const lo = SHARE[level] / 2;
    const hi = SHARE[level] * 2;
    row[level] = { lets, keeps: goldIn / gold, within: lets >= lo && lets <= hi };
  }
  // The same, with the floors measured on the probe in this set's language, where the probe has it.
  const own = probe.queries.some((x) => x.lang === langOf(name)) ? perLanguage(langOf(name)) : null;
  if (own) {
    row.ownLanguage = Object.fromEntries(
      (Object.keys(SHARE) as (keyof typeof SHARE)[]).map((level) => {
        let randomIn = 0;
        let random = 0;
        let goldIn = 0;
        let gold = 0;
        for (const s of scores) {
          const c = own[styleOf.get(s.id) ?? "question"][level];
          randomIn += s.random.filter((d) => d < c).length;
          random += s.random.length;
          goldIn += s.gold.filter((d) => d < c).length;
          gold += s.gold.length;
        }
        return [level, { lets: randomIn / random, keeps: goldIn / gold }];
      }),
    );
  }
  row.questions = { short: scores.filter((s) => styleOf.get(s.id) === "short").length, question: scores.filter((s) => styleOf.get(s.id) === "question").length };
  sets[name] = row;
}

const pc = (x: number) => `${(100 * x).toFixed(2)}%`;
console.log(`${runKey(run)}: probe ${probe.digest.slice(0, 12)}, short ${JSON.stringify(levels.short)}, question ${JSON.stringify(levels.question)}`);
for (const [name, r] of Object.entries(sets) as [string, any][]) {
  console.log(
    `  ${name.padEnd(10)} ` +
      (Object.keys(SHARE) as (keyof typeof SHARE)[]).map((l) => `${l} lets ${pc(r[l].lets)}${r[l].within ? "" : " (out)"} keeps ${pc(r[l].keeps)}`).join(" | ") +
      (r.ownLanguage ? `\n             own-language floor: balanced lets ${pc(r.ownLanguage.balanced.lets)} keeps ${pc(r.ownLanguage.balanced.keeps)}, loose lets ${pc(r.ownLanguage.loose.lets)}` : ""),
  );
}
if (OUT) {
  mkdirSync(OUT, { recursive: true });
  const file = `${OUT}/probe-transfer.json`;
  const all = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : {};
  all[runKey(run)] = { probe: probe.digest, levels, sets };
  writeFileSync(file, JSON.stringify(all, null, 2));
}
