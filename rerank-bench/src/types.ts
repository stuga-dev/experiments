/** A retrieved passage, as the node stores it. */
export interface Passage {
  /** `<doc_id>#<chunk_index>`. */
  id: string;
  title: string;
  /** Heading path, `A > B > C`; empty when the passage sits above the first heading. */
  section: string;
  text: string;
}

/** What one ranker call produced for one question's candidates. */
export interface RankResult {
  /** A relevance score per candidate, in candidate order; null when the call failed or its answer was unusable. */
  scores: number[] | null;
  /** The model that answered, as the vendor reports it when it does. */
  model: string;
  inputTokens: number;
  /** Input tokens the vendor served from its prompt cache, a subset of `inputTokens`. */
  cachedTokens?: number;
  /** Input tokens the vendor wrote to its prompt cache, a subset of `inputTokens` billed at its own rate. */
  cacheWriteTokens?: number;
  outputTokens: number;
  /** Reasoning tokens, a subset of `outputTokens`, when the vendor reports them. */
  reasoningTokens?: number;
  /** Wall-clock of the attempt that answered, in milliseconds. */
  ms: number;
  /** Wall-clock from the first attempt to the result, retries and their waits included: what a user waits. */
  totalMs?: number;
  /** Attempts made, 1 when the first one answered. */
  attempts: number;
  /** Why the result is unusable: a request error, or an answer that did not parse. */
  error?: string;
  /** The text an LLM judge answered, kept for audit (first 4,000 characters). */
  raw?: string;
  /** How many passages an LLM judge's answer gave a score, as Stuga's parser read it. */
  covered?: number;
}

export type RankerKind = "baseline" | "classifier" | "llm" | "reranker";

/** Price per million tokens, or per search for a reranker billed per query; null when the vendor publishes none. */
export type Price = { input: number; output: number; cacheRead?: number; cacheWrite?: number } | { perSearch: number | null };

export interface Ranker {
  /** Stable key used in results. */
  id: string;
  /** Display name. */
  label: string;
  kind: RankerKind;
  /** Where it ran: a vendor API, Bedrock, or this machine. */
  via: string;
  price: Price;
  /** Runs over the whole question set; 1 for deterministic or expensive rankers. */
  repeats: number;
  /** What an LLM judge is sent besides the prompt: its output cap, and a reasoning level when one is set. */
  sent?: { maxTokens: number; reasoning?: string };
  rank(query: string, passages: Passage[]): Promise<RankResult>;
}
