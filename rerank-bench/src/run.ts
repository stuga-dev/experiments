/**
 * Run rankers over every question's candidates in one test set and append one row per call to
 * results/<set>/<date>/raw.jsonl. Each ranker makes one request at a time after a
 * discarded warm-up; different rankers run side by side. A rerun skips rows
 * already written, so an interrupted run resumes.
 *
 *   BENCH_KEYS=keys.json node src/run.ts [--only a,b] [--limit N] [--repeats N] [--rankable-only] --set miracl|bright-stackoverflow [--out results/<set>/<date>]
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { arch, cpus, platform, totalmem } from "node:os";
import { isGold, loadCandidates, loadChunks, loadQuestions, setDir, setOf, type Chunk } from "./corpus.ts";
import { lineup, readKeys } from "./lineup.ts";
import { ACCOUNT_ERROR, isRetryable } from "./retry.ts";
import { orderByScores, seenBy, type Snippet } from "./stuga.ts";
import type { Passage, Ranker, RankResult } from "./types.ts";

export interface Row extends Omit<RankResult, "raw"> {
  ranker: string;
  repeat: number;
  question: string;
  /** Candidate ids in the order the ranker put them; fusion order when it failed. */
  order?: string[];
  raw?: string;
  at: string;
}

const arg = (name: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
};
const only = arg("only")?.split(",");
const limit = Number(arg("limit") ?? Infinity);
const set = setOf(arg("set"));
const outDir = arg("out") ?? `results/${set}/${new Date().toISOString().slice(0, 10)}`;

const chunks = new Map(loadChunks(setDir(set)).map((c) => [c.id, c] as const));
const candidates = new Map(loadCandidates(setDir(set)).map((c) => [c.question, c.ids] as const));
// A question whose answer retrieval missed scores nothing; --rankable-only saves its calls.
const questions = loadQuestions(setDir(set))
  .filter((q) => !process.argv.includes("--rankable-only") || (candidates.get(q.id) ?? []).some((id) => isGold(chunks.get(id)!, q)))
  .slice(0, limit);
const repeatsOverride = arg("repeats") ? Number(arg("repeats")) : undefined;
const snippet = (arg("snippet") ?? "excerpt") as Snippet;
if (snippet !== "excerpt" && snippet !== "start") throw new Error("--snippet is excerpt or start");
const rankers = lineup(readKeys())
  .filter((r) => !only || only.includes(r.id))
  .map((r) => (repeatsOverride ? { ...r, repeats: repeatsOverride } : r));

mkdirSync(outDir, { recursive: true });
const rawPath = `${outDir}/raw.jsonl`;
const done = new Set(
  existsSync(rawPath)
    ? readFileSync(rawPath, "utf8")
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l) as Row)
        .filter((r) => !ACCOUNT_ERROR.test(r.error ?? "") && !isRetryable(r.error ?? ""))
        .map((r) => `${r.ranker}|${r.repeat}|${r.question}`)
    : [],
);

// Runs of different rankers share a directory; each adds its rankers to meta.json.
const metaPath = `${outDir}/meta.json`;
const earlier = existsSync(metaPath) ? (JSON.parse(readFileSync(metaPath, "utf8")) as { started: string; rankers: { id: string }[] }) : null;
const described = rankers.map(({ id, label, kind, via, price, repeats, sent }) => ({ id, label, kind, via, price, repeats, ...(sent ? { sent } : {}) }));
writeFileSync(
  metaPath,
  JSON.stringify(
    {
      set,
      started: earlier?.started ?? new Date().toISOString(),
      snippet,
      node: process.version,
      machine: `${cpus()[0]?.model ?? "?"}, ${Math.round(totalmem() / 2 ** 30)} GB, ${platform()}/${arch()}`,
      piAi: JSON.parse(readFileSync("node_modules/@earendil-works/pi-ai/package.json", "utf8")).version,
      rankers: [...(earlier?.rankers ?? []).filter((r) => !described.some((d) => d.id === r.id)), ...described],
    },
    null,
    2,
  ),
);

/** A question's candidates as its judges see them. */
function passagesFor(q: { id: string; q: string }): Passage[] {
  const ids = candidates.get(q.id);
  if (!ids) throw new Error(`no candidates for ${q.id}`);
  const passages = ids.map((id) => {
    const c: Chunk | undefined = chunks.get(id);
    if (!c) throw new Error(`candidate ${id} is not in the corpus`);
    return { id: c.id, title: c.title, section: c.section, text: c.text };
  });
  return seenBy(q.q, passages, snippet);
}

async function lane(r: Ranker) {
  const first = questions[0];
  if (first) await r.rank(first.q, passagesFor(first)).catch(() => undefined);
  for (let repeat = 1; repeat <= r.repeats; repeat++) {
    for (const q of questions) {
      if (done.has(`${r.id}|${repeat}|${q.id}`)) continue;
      const passages = passagesFor(q);
      const res = await r.rank(q.q, passages);
      const idx = orderByScores(res.scores, passages.length);
      const row: Row = { ranker: r.id, repeat, question: q.id, ...res, order: idx.map((i) => passages[i]!.id), at: new Date().toISOString() };
      appendFileSync(rawPath, JSON.stringify(row) + "\n");
    }
    console.log(`${r.id}: repeat ${repeat}/${r.repeats} done`);
  }
}

await Promise.all(
  rankers.map((r) =>
    lane(r).catch((e) => {
      console.error(`${r.id} stopped: ${e instanceof Error ? e.message : String(e)}`);
    }),
  ),
);
console.log(`rows in ${rawPath}`);
