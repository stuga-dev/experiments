/**
 * The embedding models, and what Stuga sends each: the prompts from its profile table (or none, for
 * the no-prompt runs), and a width no wider than Stuga's default 1,024-dimension column, asked for
 * as Stuga asks for it.
 */
import { type EmbedProfile, embedProfile } from "./stuga-profile.ts";

/**
 * Ollama as a Stuga user runs it; hosted models as their vendors intend: OpenAI's API, Gemini's own
 * API with its retrieval task types, Cohere Embed v4 on Bedrock with its input types. `st` is a
 * reference: sentence-transformers at bf16, written by scripts/st-embed.py, for checking what
 * Ollama's quantization costs.
 */
export type Backend = "ollama" | "openai" | "gemini" | "bedrock" | "st";
const BACKENDS: readonly Backend[] = ["ollama", "openai", "gemini", "bedrock", "st"];

export interface ModelRun {
  backend: Backend;
  model: string;
  /** Without the model's prompts: what a client that sends raw text gets. */
  raw: boolean;
}

/** Stuga's default column. */
export const COLUMN = 1024;

export function profileOf(run: ModelRun): EmbedProfile {
  const p = embedProfile(run.model);
  return run.raw ? { ...p, query: "", document: "" } : p;
}

/** Stuga's `requestedDims` (packages/ai/src/retrieval/embed.ts), for its default column. */
export function requestedDims(run: ModelRun): number | null {
  const p = embedProfile(run.model);
  if (p.dims === null) return run.backend === "ollama" ? null : COLUMN;
  if (p.dims <= COLUMN) return null;
  return p.shortens ? (p.shortens.find((d) => d <= COLUMN) ?? null) : COLUMN;
}

/** A stable folder name under .cache/vectors/. */
export function runKey(run: ModelRun): string {
  return `${run.backend}/${run.model.replaceAll("/", "_")}${run.raw ? "+raw" : ""}`;
}

export function parseRun(spec: string): ModelRun {
  // backend:model[+raw], e.g. ollama:embeddinggemma-2:270m+raw
  const raw = spec.endsWith("+raw");
  const s = raw ? spec.slice(0, -4) : spec;
  const i = s.indexOf(":");
  const backend = s.slice(0, i) as Backend;
  if (!BACKENDS.includes(backend)) throw new Error(`unknown backend in ${spec}`);
  return { backend, model: s.slice(i + 1), raw };
}
