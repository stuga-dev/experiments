/**
 * Dedicated rerankers, which score each (query, document) pair: Cohere and
 * Amazon on Bedrock through InvokeModel, and any local server that speaks the
 * Cohere/Jina `/rerank` shape, such as llama.cpp's llama-server --rerank.
 * Each document is the passage a judge sees: title, section and the text it reads.
 */
import { withRetries } from "../retry.ts";
import { passageOf } from "../stuga.ts";
import type { Passage, Ranker, RankResult } from "../types.ts";

interface RerankAnswer {
  results?: { index: number; relevance_score: number }[];
  model?: string;
}

export function documentOf(p: Passage): string {
  const q = passageOf(p);
  return `${q.section ? `${q.title} — ${q.section}` : q.title}\n${q.text}`;
}

function toScores(a: RerankAnswer, n: number): number[] {
  const out = Array.from({ length: n }, () => Number.NEGATIVE_INFINITY);
  for (const r of a.results ?? []) if (r.index >= 0 && r.index < n) out[r.index] = r.relevance_score;
  if (out.some((s) => s === Number.NEGATIVE_INFINITY)) throw new Error("the reranker left documents unscored");
  return out;
}

export interface CrossEncoderSpec {
  id: string;
  label: string;
  via: string;
  /** The model the request names. */
  model: string;
  /** Returns the answer to one rerank request. */
  post: (query: string, documents: string[]) => Promise<RerankAnswer>;
  /** $ per search, 0 on this machine, null when no price is published. */
  perSearch: number | null;
  repeats: number;
}

export function crossEncoder(spec: CrossEncoderSpec): Ranker {
  async function once(query: string, passages: Passage[]): Promise<RankResult> {
    const t0 = performance.now();
    try {
      const a = await spec.post(query, passages.map(documentOf));
      const ms = performance.now() - t0;
      return { scores: toScores(a, passages.length), model: a.model ?? spec.model, inputTokens: 0, outputTokens: 0, ms, attempts: 1 };
    } catch (e) {
      return { scores: null, model: spec.model, inputTokens: 0, outputTokens: 0, ms: performance.now() - t0, attempts: 1, error: String(e) };
    }
  }
  return {
    id: spec.id,
    label: spec.label,
    kind: "reranker",
    via: spec.via,
    price: { perSearch: spec.perSearch },
    repeats: spec.repeats,
    rank: (query, passages) => withRetries(() => once(query, passages)),
  };
}

async function postJson(url: string, headers: Record<string, string>, body: unknown): Promise<RerankAnswer> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json", ...headers },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) throw new Error(`rerank ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return (await res.json()) as RerankAnswer;
}

/** A Bedrock reranker through InvokeModel with a Bedrock API key. */
export function bedrockRerank(modelId: string, region: string, bearerToken: string) {
  const url = `https://bedrock-runtime.${region}.amazonaws.com/model/${encodeURIComponent(modelId)}/invoke`;
  const cohere = modelId.startsWith("cohere.");
  return (query: string, documents: string[]) =>
    postJson(
      url,
      { authorization: `Bearer ${bearerToken}` },
      cohere ? { query, documents, top_n: documents.length, api_version: 2 } : { query, documents, top_n: documents.length },
    );
}

/** A local server with a Cohere-shaped `/v1/rerank`, e.g. llama-server --rerank. */
export function localRerank(baseUrl: string, model: string) {
  return (query: string, documents: string[]) =>
    postJson(`${baseUrl.replace(/\/+$/, "")}/v1/rerank`, {}, { model, query, documents, top_n: documents.length });
}
