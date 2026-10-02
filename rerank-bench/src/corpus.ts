/** The corpus, questions and candidates the rankers are measured on, one directory per test set under data/. */
import { readFileSync } from "node:fs";
import type { Passage } from "./types.ts";

/**
 * Test sets, each `data/<set>/` with corpus.jsonl, questions.jsonl and candidates.jsonl, built by a
 * script in scripts/ from a pinned revision of a public dataset whose questions and relevance
 * judgments were made by people.
 */
export const SETS = ["miracl", "bright-stackoverflow", "bright-stackoverflow-bm25"] as const;
export type SetId = (typeof SETS)[number];
export const setDir = (set: SetId) => `data/${set}`;

export function setOf(value: string | undefined): SetId {
  const set = SETS.find((s) => s === value);
  if (!set) throw new Error(`--set must be one of ${SETS.join(", ")}`);
  return set;
}

/** Candidates per question: Stuga's CANDIDATE_LIMIT. */
export const CANDIDATES = 24;

export interface Chunk extends Passage {
  /** The part of the set it belongs to: a language, a site. */
  sample: string;
}

export interface Question {
  id: string;
  sample: string;
  q: string;
  /** The language or site, for results by category. */
  category: string;
  /** The passages judged relevant, by id. */
  goldIds: string[];
}

export interface CandidateSet {
  question: string;
  /** Passage ids in the first stage's order. */
  ids: string[];
}

function jsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as T);
}

export const loadChunks = (dir: string) => jsonl<Chunk>(`${dir}/corpus.jsonl`);
export const loadQuestions = (dir: string) => jsonl<Question>(`${dir}/questions.jsonl`);
export const loadCandidates = (dir: string) => jsonl<CandidateSet>(`${dir}/candidates.jsonl`);

export const isGold = (c: Chunk, q: Question) => q.goldIds.includes(c.id);
