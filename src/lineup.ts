/**
 * Every ranker in the benchmark. Models are the newest of each family that its
 * vendor served on the run date; `results/<set>/<date>/meta.json` records what answered.
 */
import { readFileSync } from "node:fs";
import { bedrockRerank, crossEncoder, localRerank } from "./rankers/cross-encoder.ts";
import { llmJudge } from "./rankers/llm-judge.ts";
import { systemOne } from "./rankers/system-one.ts";
import type { Ranker } from "./types.ts";

export interface Keys {
  openai?: string;
  typesafe?: string;
  bedrock?: { region: string; key: string };
}

/** Keys come from the JSON file named by BENCH_KEYS, never from the command line. */
export function readKeys(): Keys {
  const path = process.env.BENCH_KEYS;
  if (!path) throw new Error("set BENCH_KEYS to a JSON file with your provider keys (see README)");
  return JSON.parse(readFileSync(path, "utf8")) as Keys;
}

/** Bedrock bills rerankers per query of up to 100 documents: Cohere Rerank 3.5 at $2 per 1,000; its pricing page lists no price for Amazon Rerank 1.0. */
const COHERE_RERANK_PER_SEARCH = 0.002;
const JEV_INPUT_PRICE = 0.042;

/** llama-server --rerank instances on this machine, one per model. */
const LOCAL_RERANKERS = [
  { id: "qwen3-reranker-4b", label: "Qwen3-Reranker 4B (local)", url: process.env.LOCAL_RERANK_4B_URL ?? "http://127.0.0.1:8089" },
  { id: "qwen3-reranker-0-6b", label: "Qwen3-Reranker 0.6B (local)", url: process.env.LOCAL_RERANK_06B_URL ?? "http://127.0.0.1:8090" },
];

export function lineup(keys: Keys): Ranker[] {
  const out: Ranker[] = [];
  const fusion: Ranker = {
    id: "fusion",
    label: "No reranker (first-stage order)",
    kind: "baseline",
    via: "the set's first stage",
    price: { perSearch: 0 },
    repeats: 1,
    rank: async () => ({ scores: null, model: "first stage", inputTokens: 0, outputTokens: 0, ms: 0, attempts: 1 }),
  };
  out.push(fusion);

  if (keys.typesafe) {
    const base = { baseUrl: "https://api.typesafe.ai/v1", model: "jev-latest", apiKey: keys.typesafe, inputPrice: JEV_INPUT_PRICE, repeats: 1 };
    out.push(systemOne({ ...base, id: "jev", label: "Jev", mode: "batched" }));
    out.push(systemOne({ ...base, id: "jev-per-pair", label: "Jev, one request per passage", mode: "per-pair" }));
  }

  // GPT-6 goes through Bedrock when there is a Bedrock key, below.
  if (keys.openai && !keys.bedrock) {
    const auth = { apiKey: keys.openai };
    out.push(llmJudge({ id: "gpt-6-sol", label: "GPT-6 Sol", via: "OpenAI API", provider: "openai", model: "gpt-6-sol", auth, repeats: 1 }));
    out.push(llmJudge({ id: "gpt-6-luna", label: "GPT-6 Luna", via: "OpenAI API", provider: "openai", model: "gpt-6-luna", auth, repeats: 1 }));
  }

  if (keys.bedrock) {
    const { region, key } = keys.bedrock;
    const auth = { apiKey: key, env: { AWS_REGION: region } };
    const baseUrl = `https://bedrock-runtime.${region}.amazonaws.com`;
    const via = `Bedrock ${region}`;
    const bedrock = (id: string, label: string, model: string) =>
      llmJudge({ id, label, via, provider: "amazon-bedrock", model, baseUrl, auth, repeats: 1 });
    // GPT-6 through Bedrock's OpenAI-compatible Responses API and a US inference profile, so it gets
    // the request OpenAI's own API gets. Pi's catalog lists it for OpenAI only; the prices are Bedrock's
    // model-card rates for that profile, per million tokens (OpenAI's plus 10%).
    const gpt = (id: string, label: string, cost: { input: number; output: number; cacheRead: number; cacheWrite: number }) =>
      llmJudge({ id, label, via, provider: "openai", model: id, baseUrl: `${baseUrl}/openai/v1`, hostedAs: { model: `us.openai.${id}`, cost }, auth: { apiKey: key }, repeats: 1 });
    out.push(
      gpt("gpt-6-sol", "GPT-6 Sol", { input: 2.2, output: 11, cacheRead: 0.22, cacheWrite: 2.75 }),
      gpt("gpt-6-luna", "GPT-6 Luna", { input: 0.11, output: 0.55, cacheRead: 0.011, cacheWrite: 0.1375 }),
    );
    out.push(
      bedrock("claude-opus-5-5", "Claude Opus 5.5", "us.anthropic.claude-opus-5-5"),
      bedrock("claude-sonnet-5", "Claude Sonnet 5", "us.anthropic.claude-sonnet-5"),
      // Not in Pi's catalog yet; Bedrock serves it through the global profile only.
      llmJudge({
        id: "claude-sonnet-5-5",
        label: "Claude Sonnet 5.5",
        via,
        provider: "amazon-bedrock",
        model: "global.anthropic.claude-sonnet-5",
        baseUrl,
        hostedAs: { model: "global.anthropic.claude-sonnet-5-5", cost: { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 2.5 } },
        auth,
        repeats: 1,
      }),
      bedrock("claude-haiku-4-5", "Claude Haiku 4.5", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
      bedrock("grok-4-6", "Grok 4.6", "us.xai.grok-4.6"),
      bedrock("nova-2-lite", "Amazon Nova 2 Lite", "us.amazon.nova-2-lite-v1:0"),
      bedrock("deepseek-v3-2", "DeepSeek V3.2", "deepseek.v3.2"),
      bedrock("qwen3-235b", "Qwen3 235B A22B 2507", "qwen.qwen3-235b-a22b-2507-v1:0"),
      bedrock("mistral-large-3", "Mistral Large 3", "mistral.mistral-large-3-675b-instruct"),
      bedrock("llama-4-maverick", "Llama 4 Maverick", "us.meta.llama4-maverick-17b-instruct-v1:0"),
      bedrock("gpt-oss-120b", "gpt-oss-120b", "openai.gpt-oss-120b-1:0"),
      bedrock("glm-5", "GLM-5", "zai.glm-5"),
      bedrock("minimax-m2-5", "MiniMax M2.5", "minimax.minimax-m2.5"),
    );
    out.push(
      crossEncoder({ id: "cohere-rerank-3-5", label: "Cohere Rerank 3.5", via, model: "cohere.rerank-v3-5:0", post: bedrockRerank("cohere.rerank-v3-5:0", region, key), perSearch: COHERE_RERANK_PER_SEARCH, repeats: 1 }),
      crossEncoder({ id: "amazon-rerank-1", label: "Amazon Rerank 1.0", via, model: "amazon.rerank-v1:0", post: bedrockRerank("amazon.rerank-v1:0", region, key), perSearch: null, repeats: 1 }),
    );
  }

  // System One servers on this machine, one passage per request. Kev's README notes training on
  // states of up to 384 tokens; Laya is an encoder that reads its state as text, so it gets plain
  // text, no criteria, and picks its English or multilingual checkpoint itself.
  out.push(systemOne({ id: "kev-4b-per-pair", label: "Kev-4B (local)", baseUrl: process.env.KEV_URL ?? "http://127.0.0.1:8009/v1", model: "kev-4b", apiKey: "", inputPrice: 0, via: "this machine", repeats: 1, mode: "per-pair" }));
  out.push(systemOne({ id: "laya-text", label: "Laya (local)", baseUrl: process.env.LAYA_URL ?? "http://127.0.0.1:8010/v1", model: "", apiKey: "", inputPrice: 0, via: "this machine", repeats: 1, mode: "per-pair", shape: "text" }));
  for (const l of LOCAL_RERANKERS) {
    out.push(crossEncoder({ id: l.id, label: l.label, via: "llama.cpp on this machine", model: l.id, post: localRerank(l.url, l.id), perSearch: 0, repeats: 1 }));
  }
  return out;
}
