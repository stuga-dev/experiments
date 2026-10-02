/**
 * A few questions' top passages under a few rankers, for a write-up: each ranker's first eight
 * (what Stuga's Ask reads), where the answer landed among the 24, and how long the call took.
 * Uses each ranker's first run.
 *
 *   node scripts/export-examples.ts --questions pl-06,pl-34 --rankers fusion,jev --out examples.json
 */
import { readFileSync, writeFileSync } from "node:fs";
import { isGold, loadCandidates, loadChunks, loadQuestions, setDir, type SetId } from "../src/corpus.ts";
import type { Row } from "../src/run.ts";

const arg = (name: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
};
const questionIds = (arg("questions") ?? "").split(",").filter(Boolean);
const rankerIds = (arg("rankers") ?? "").split(",").filter(Boolean);
const out = arg("out") ?? "examples.json";
const dir = arg("results");
if (!dir) throw new Error("--results results/<set>/<date> is required");
const TOP = 8;

const meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8")) as { set: SetId; rankers: { id: string; label: string; kind: string }[] };
const data = setDir(meta.set);
const chunks = new Map(loadChunks(data).map((c) => [c.id, c] as const));
const candidates = new Map(loadCandidates(data).map((c) => [c.question, c.ids] as const));
const questions = new Map(loadQuestions(data).map((q) => [q.id, q] as const));
const rows = new Map<string, Row>();
for (const line of readFileSync(`${dir}/raw.jsonl`, "utf8").split("\n").filter(Boolean)) {
  const r = JSON.parse(line) as Row;
  rows.set(`${r.ranker}|${r.repeat}|${r.question}`, r);
}

/** The passage's opening prose: no heading, table or quote lines; links as their text. */
function snippet(text: string): string {
  const lines = text.split("\n").filter((l) => l.trim());
  const prose = lines.filter((l) => !/^\s*(#|\||>|:::)/.test(l));
  const body = (prose.length ? prose : lines).join(" ");
  return body
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/\\/g, "")
    .replace(/[*_`#|]/g, "")
    .replace(/-{3,}/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 180);
}

const examples = questionIds.map((qid) => {
  const q = questions.get(qid);
  const ids = candidates.get(qid);
  if (!q || !ids) throw new Error(`unknown question ${qid}`);
  const gold = ids.filter((id) => isGold(chunks.get(id)!, q));
  const judged = rankerIds.map((rid) => {
    const r = rows.get(`${rid}|1|${qid}`);
    const order = r?.order ?? ids;
    const scoreOf = (id: string) => (r?.scores ? r.scores[ids.indexOf(id)] ?? null : null);
    const rank = order.findIndex((id) => gold.includes(id)) + 1;
    const label = meta.rankers.find((m) => m.id === rid);
    return { id: rid, label: label?.label ?? rid, kind: label?.kind ?? "", ms: r?.ms ?? 0, answerRank: rank, top: order.slice(0, TOP).map((id) => ({ id, score: scoreOf(id) })) };
  });
  const shown = new Set([...gold, ...judged.flatMap((j) => j.top.map((t) => t.id))]);
  const passages = Object.fromEntries(
    [...shown].map((id) => {
      const c = chunks.get(id)!;
      const section = c.section.split(" > ").slice(1).join(" › ") || c.section;
      return [id, { title: c.title, section, text: snippet(c.text) }];
    }),
  );
  return { id: q.id, q: q.q, category: q.category, gold, passages, judged };
});

writeFileSync(out, JSON.stringify({ candidates: 24, top: TOP, examples }, null, 2) + "\n");
console.log(`${examples.length} examples → ${out}`);
