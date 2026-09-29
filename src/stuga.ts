/**
 * What Stuga's judges see and send, copied from stuga-dev/stuga
 * packages/ai/src/retrieval/rerank.ts. Keep it verbatim: the point of the
 * benchmark is to measure these exact requests.
 */
import { excerpts } from "./excerpt.ts";
import type { Passage } from "./types.ts";

/** Characters of a passage a judge sees; the caller keeps the full passage. */
export const SNIPPET_CHARS = 1200;

/**
 * Which characters: the part of a long passage that best matches the query, as Stuga's judges read
 * it, or its first ones, as they read before.
 */
export type Snippet = "excerpt" | "start";

/** The passages as a judge sees them, their text cut to what it reads. */
export function seenBy(query: string, passages: Passage[], snippet: Snippet): Passage[] {
  const texts =
    snippet === "excerpt" ? excerpts(query, passages.map((p) => p.text), SNIPPET_CHARS) : passages.map((p) => p.text.slice(0, SNIPPET_CHARS).trim());
  return passages.map((p, i) => ({ ...p, text: texts[i]! }));
}

export function passageOf(c: Passage): { title: string; section: string; text: string } {
  return { title: c.title || "Untitled", section: c.section ?? "", text: c.text };
}

export const JUDGE_SYSTEM = `You are a search relevance judge. Given a user query and numbered
document snippets, rate how well EACH snippet helps answer or act on the query.
Score each 0-10 (10 = directly answers it; 0 = irrelevant). Judge only relevance,
not writing quality. Respond with ONLY a JSON array of {"i":<number>,"score":<0-10>}
for every snippet, no prose.`;

/**
 * The judge's output cap and reasoning, as Stuga sends them: up to 8,192 tokens (within the model's
 * own limit) and the lowest reasoning the model offers.
 */
export const JUDGE = { maxTokens: 8192, thinking: "off" } as const;

export function judgePrompt(query: string, candidates: Passage[]): string {
  const list = candidates
    .map((c, i) => {
      const p = passageOf(c);
      return `[${i}] ${p.section ? `${p.title} — ${p.section}` : p.title}\n${p.text}`;
    })
    .join("\n\n");
  return `Query: ${query}\n\nSnippets:\n${list}`;
}

/** The judge's JSON array as index → score, tolerating surrounding prose or fences. */
export function parseScores(text: string, n: number): number[] | null {
  const start = text.indexOf("[");
  const end = text.lastIndexOf("]");
  if (start < 0 || end <= start) return null;
  let arr: unknown;
  try {
    arr = JSON.parse(text.slice(start, end + 1));
  } catch {
    return null;
  }
  if (!Array.isArray(arr)) return null;
  const out = Array.from({ length: n }, () => 0);
  let any = false;
  for (const item of arr) {
    if (item && typeof item === "object" && "i" in item && "score" in item) {
      const i = Number((item as { i: unknown }).i);
      const s = Number((item as { score: unknown }).score);
      if (Number.isInteger(i) && i >= 0 && i < n && Number.isFinite(s)) {
        out[i] = Math.max(0, Math.min(10, s));
        any = true;
      }
    }
  }
  return any ? out : null;
}

/** `{"i":3,"score":7}` anywhere in the answer, and a `[3] 7` line. */
const SCORE_OBJECT = /\{\s*"i"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*(\d*\.?\d+)\s*\}/g;
const SCORE_LINE = /^\s*\[(\d+)\]\s*[:=-]?\s*(\d*\.?\d+)\s*$/gm;

/**
 * Scores a finished answer states without the array asked for: the objects one per line, a list
 * with a note after it, or `[i] score` lines. The first score for each passage counts, and at least
 * half the passages must have one, so a stray number in prose is never read as a ranking.
 */
export function statedScores(text: string, n: number): number[] | null {
  for (const pattern of [SCORE_OBJECT, SCORE_LINE]) {
    const out = Array.from({ length: n }, () => 0);
    const seen = new Set<number>();
    for (const [, index, score] of text.matchAll(pattern)) {
      const i = Number(index);
      if (i < n && !seen.has(i)) {
        seen.add(i);
        out[i] = Math.max(0, Math.min(10, Number(score)));
      }
    }
    if (seen.size * 2 >= n) return out;
  }
  return null;
}

/** One yes/no question per passage, as Stuga asks a System One model. */
export function systemOneQuestion(id: string) {
  return {
    type: "noul" as const,
    instructions: `Does passage ${id} help answer the query?`,
    criteria: {
      true: "The passage states the information the query asks for, or information needed to answer it",
      false: "The passage is only on a related topic, or is irrelevant to the query",
    },
  };
}

/**
 * Candidate order after a judge: by score, ties kept in fusion order. A failed
 * judge leaves the fusion order, as the node does.
 */
export function orderByScores(scores: number[] | null, n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  if (!scores) return idx;
  return idx.map((i) => ({ i, s: scores[i] ?? 0 })).sort((a, b) => b.s - a.s || a.i - b.i).map((x) => x.i);
}

/**
 * Not Stuga's: for the benchmark's audit, how many passages an answer gave a score, read the way
 * parseScores and statedScores read it.
 */
export function scoredCount(text: string, n: number): number {
  const start = text.indexOf("[");
  const end = text.lastIndexOf("]");
  if (start >= 0 && end > start) {
    try {
      const arr: unknown = JSON.parse(text.slice(start, end + 1));
      if (Array.isArray(arr)) {
        const seen = new Set<number>();
        for (const item of arr) {
          if (item && typeof item === "object" && "i" in item && "score" in item) {
            const i = Number((item as { i: unknown }).i);
            if (Number.isInteger(i) && i >= 0 && i < n && Number.isFinite(Number((item as { score: unknown }).score))) seen.add(i);
          }
        }
        if (seen.size) return seen.size;
      }
    } catch {
      // Not an array: read as statedScores does.
    }
  }
  for (const pattern of [SCORE_OBJECT, SCORE_LINE]) {
    const seen = new Set<number>();
    for (const [, index] of text.matchAll(pattern)) if (Number(index) < n) seen.add(Number(index));
    if (seen.size * 2 >= n) return seen.size;
  }
  return 0;
}
