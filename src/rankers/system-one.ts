/**
 * A System One classifier as the judge (TypeSafe's Jev): one request whose
 * state holds the query and every passage, with one yes/no question per
 * passage, answered as probabilities. `perPair` sends one request per passage
 * instead, all at once.
 */
import { withRetries } from "../retry.ts";
import { passageOf, systemOneQuestion } from "../stuga.ts";
import type { Passage, Ranker, RankResult } from "../types.ts";

interface Answer {
  model?: string;
  answers?: Record<string, { noul?: number }>;
  usage?: { input_tokens?: number; output_tokens?: number };
}

export interface SystemOneSpec {
  id: string;
  label: string;
  baseUrl: string;
  model: string;
  /** Empty for a local server that takes no key. */
  apiKey: string;
  /** Where it runs, for the results. */
  via?: string;
  /** $ per million input tokens. */
  inputPrice: number;
  mode: "batched" | "per-pair";
  repeats: number;
  /**
   * "text" asks one passage at a time with the state as plain text and no criteria: the shape an
   * encoder like Laya reads best. Stuga sends "json". Omitting `model` lets the server route.
   */
  shape?: "json" | "text";
}

async function ask(spec: SystemOneSpec, state: unknown, questions: Record<string, unknown>): Promise<Answer> {
  const res = await fetch(`${spec.baseUrl.replace(/\/+$/, "")}/systemone`, {
    method: "POST",
    headers: { ...(spec.apiKey ? { authorization: `Bearer ${spec.apiKey}` } : {}), "content-type": "application/json" },
    body: JSON.stringify({ ...(spec.model ? { model: spec.model } : {}), state, questions }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) throw new Error(`systemone ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return (await res.json()) as Answer;
}

function probability(a: Answer, id: string): number {
  const p = a.answers?.[id]?.noul;
  if (typeof p !== "number" || !Number.isFinite(p)) throw new Error(`systemone gave no answer for ${id}`);
  return p;
}

export function systemOne(spec: SystemOneSpec): Ranker {
  async function once(query: string, passages: Passage[]): Promise<RankResult> {
    const t0 = performance.now();
    try {
      if (spec.mode === "batched") {
        const ids = passages.map((_, i) => `p${i}`);
        const a = await ask(
          spec,
          { query, passages: Object.fromEntries(ids.map((id, i) => [id, passageOf(passages[i]!)])) },
          Object.fromEntries(ids.map((id) => [id, systemOneQuestion(id)])),
        );
        const ms = performance.now() - t0;
        return { scores: ids.map((id) => probability(a, id)), model: a.model ?? spec.model, inputTokens: a.usage?.input_tokens ?? 0, outputTokens: a.usage?.output_tokens ?? 0, ms, attempts: 1 };
      }
      const all = await Promise.all(
        passages.map((p) => {
          if (spec.shape === "text") {
            const q = passageOf(p);
            return ask(spec, `Query: ${query}\n\nPassage: ${q.title} — ${q.section}\n${q.text}`, { relevant: { type: "noul", instructions: "Does the passage answer the query?" } });
          }
          return ask(spec, { query, passage: passageOf(p) }, { relevant: { ...systemOneQuestion("relevant"), instructions: "Does the passage help answer the query?" } });
        }),
      );
      const ms = performance.now() - t0;
      return {
        scores: all.map((a) => probability(a, "relevant")),
        model: all[0]?.model ?? spec.model,
        inputTokens: all.reduce((s, a) => s + (a.usage?.input_tokens ?? 0), 0),
        outputTokens: all.reduce((s, a) => s + (a.usage?.output_tokens ?? 0), 0),
        ms,
        attempts: 1,
      };
    } catch (e) {
      return { scores: null, model: spec.model, inputTokens: 0, outputTokens: 0, ms: performance.now() - t0, attempts: 1, error: String(e) };
    }
  }

  return {
    id: spec.id,
    label: spec.label,
    kind: "classifier",
    via: spec.via ?? "TypeSafe API",
    price: { input: spec.inputPrice, output: 0 },
    repeats: spec.repeats,
    rank: (query, passages) => withRetries(() => once(query, passages)),
  };
}
