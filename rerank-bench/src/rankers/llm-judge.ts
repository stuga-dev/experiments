/**
 * An LLM as the judge: Stuga's prompt and parser, sent through Pi's client for
 * the model's protocol with the options Stuga's completeText sends: a token cap,
 * and the lowest reasoning level the model offers.
 */
import { clampThinkingLevel, normalizeContext, type Api, type Model, type ProviderStreams } from "@earendil-works/pi-ai";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import { bedrockConverseStreamApi } from "@earendil-works/pi-ai/api/bedrock-converse-stream.lazy";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import { getBuiltinModel } from "@earendil-works/pi-ai/providers/all";
import { withRetries } from "../retry.ts";
import { JUDGE, JUDGE_SYSTEM, judgePrompt, parseScores, scoredCount, statedScores } from "../stuga.ts";
import type { Passage, Ranker, RankResult } from "../types.ts";

const APIS: Record<string, ProviderStreams> = {
  "anthropic-messages": anthropicMessagesApi(),
  "bedrock-converse-stream": bedrockConverseStreamApi(),
  "openai-completions": openAICompletionsApi(),
  "openai-responses": openAIResponsesApi(),
};

/** Stuga's per-attempt wait for response headers; a long streamed answer is not cut. */
const RESPONSE_START_TIMEOUT_MS = 60_000;

export interface JudgeAuth {
  /** The vendor's key; for Bedrock, a Bedrock API key sent as a bearer token. */
  apiKey: string;
  /** For Bedrock, `{ AWS_REGION }`. */
  env?: Record<string, string>;
}

export interface JudgeSpec {
  id: string;
  label: string;
  via: string;
  /** Pi catalog provider and model id. */
  provider: string;
  model: string;
  /** Overrides the catalog's base URL, e.g. a Bedrock region's runtime endpoint. */
  baseUrl?: string;
  /** For a model another service hosts: the id it answers to, and its prices, per million tokens. */
  hostedAs?: { model: string; cost: Model<Api>["cost"]; thinkingLevelMap?: Model<Api>["thinkingLevelMap"] };
  auth: JudgeAuth;
  repeats: number;
}

export function catalogModel(provider: string, id: string): Model<Api> {
  const m = getBuiltinModel(provider as never, id as never) as Model<Api> | undefined;
  if (!m) throw new Error(`Pi's catalog has no ${provider}/${id}`);
  return m;
}

export function llmJudge(spec: JudgeSpec): Ranker {
  const known = catalogModel(spec.provider, spec.model);
  const model: Model<Api> = {
    ...known,
    ...(spec.baseUrl ? { baseUrl: spec.baseUrl } : {}),
    ...(spec.hostedAs ? { id: spec.hostedAs.model, cost: spec.hostedAs.cost } : {}),
    ...(spec.hostedAs?.thinkingLevelMap ? { thinkingLevelMap: { ...known.thinkingLevelMap, ...spec.hostedAs.thinkingLevelMap } } : {}),
  };
  const found = APIS[model.api];
  if (!found) throw new Error(`no client for protocol ${model.api}`);
  const api: ProviderStreams = found;

  // As stuga's completeText sends it: a cap within the model's limit, and a reasoning level only
  // when the clamped level is not "off".
  const settings = JUDGE;
  const maxTokens = model.maxTokens > 0 ? Math.min(settings.maxTokens, model.maxTokens) : settings.maxTokens;
  const thinking = settings.thinking && clampThinkingLevel(model, settings.thinking);
  const reasoning = thinking && thinking !== "off" ? { reasoning: thinking } : {};

  async function once(query: string, passages: Passage[]): Promise<RankResult> {
    const context = normalizeContext({
      systemPrompt: JUDGE_SYSTEM,
      messages: [{ role: "user", content: judgePrompt(query, passages), timestamp: Date.now() }],
    });
    const t0 = performance.now();
    let message;
    try {
      message = await api
        .streamSimple(model, context, { ...spec.auth, maxTokens, ...reasoning, maxRetries: 0, timeoutMs: RESPONSE_START_TIMEOUT_MS } as never)
        .result();
    } catch (e) {
      return { scores: null, model: model.id, inputTokens: 0, outputTokens: 0, ms: performance.now() - t0, attempts: 1, error: String(e) };
    }
    const ms = performance.now() - t0;
    const u = message.usage;
    const base = {
      model: message.model ?? model.id,
      inputTokens: u.input + u.cacheRead + u.cacheWrite,
      cachedTokens: u.cacheRead,
      cacheWriteTokens: u.cacheWrite,
      outputTokens: u.output,
      ...(u.reasoning != null ? { reasoningTokens: u.reasoning } : {}),
      ms,
      attempts: 1,
    };
    if (message.stopReason === "error" || message.stopReason === "aborted") {
      return { ...base, scores: null, error: message.errorMessage ?? "the model request failed" };
    }
    const text = message.content.flatMap((b) => (b.type === "text" ? [b.text] : [])).join("");
    // As Stuga reads it: the array, else scores stated another way, unless the answer was cut off.
    const scores = parseScores(text, passages.length) ?? (message.stopReason === "length" ? null : statedScores(text, passages.length));
    const audit = { raw: text.slice(0, 4000), covered: scores ? scoredCount(text, passages.length) : 0 };
    if (!scores) {
      return { ...base, ...audit, scores: null, error: `unparseable answer (stop: ${message.stopReason})` };
    }
    return { ...base, ...audit, scores };
  }

  return {
    id: spec.id,
    label: spec.label,
    kind: "llm",
    via: spec.via,
    price: { input: model.cost.input, output: model.cost.output, cacheRead: model.cost.cacheRead, cacheWrite: model.cost.cacheWrite },
    repeats: spec.repeats,
    sent: { maxTokens, ...reasoning },
    rank: (query, passages) => withRetries(() => once(query, passages)),
  };
}
