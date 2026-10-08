/**
 * Embeds texts through a model's server and keeps the unit vectors under .cache/vectors/, saving
 * progress as it goes, so an interrupted run resumes and a finished one is never paid for twice.
 * Hosted APIs get several requests at once; Ollama one, which it serves fastest.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { type ModelRun, profileOf, requestedDims, runKey } from "./models.ts";

export const OLLAMA = process.env.OLLAMA_URL ?? "http://127.0.0.1:11435";
export const BATCH = 64;

/**
 * How hard each model may be driven: parallel batches, and a minute's tokens and requests kept just
 * under Bedrock's default quotas for us-west-2 (300,000 tokens and 2,000 requests a minute for
 * Cohere's models, 6,000 requests for Titan Text Embeddings V2, which takes one text per request).
 */
function limits(run: ModelRun): { parallel: number; tpm: number; rpm: number } {
  if (run.backend === "ollama" || run.backend === "st") return { parallel: 1, tpm: Infinity, rpm: Infinity };
  if (run.backend === "bedrock" && /titan/.test(run.model)) return { parallel: 8, tpm: Infinity, rpm: 5_700 };
  if (run.backend === "bedrock") return { parallel: 6, tpm: 270_000, rpm: 1_900 };
  return { parallel: 8, tpm: Infinity, rpm: Infinity };
}

/**
 * A minute's window of tokens and requests sent. A batch waits for room using a tokens-per-character
 * rate learned from the tokens Bedrock reports, which differs by script (Chinese is near one a character).
 */
class Throttle {
  private sent: { at: number; tokens: number; requests: number }[] = [];
  private tokensPerChar = 0.6;
  private readonly tpm: number;
  private readonly rpm: number;
  constructor(tpm: number, rpm: number) {
    this.tpm = tpm;
    this.rpm = rpm;
  }

  async room(chars: number, requests: number): Promise<void> {
    const estimate = Math.ceil(chars * this.tokensPerChar);
    for (;;) {
      const now = Date.now();
      this.sent = this.sent.filter((s) => now - s.at < 60_000);
      const tokens = this.sent.reduce((a, s) => a + s.tokens, 0);
      const reqs = this.sent.reduce((a, s) => a + s.requests, 0);
      if (tokens + estimate <= this.tpm && reqs + requests <= this.rpm) {
        this.sent.push({ at: now, tokens: estimate, requests });
        return;
      }
      await new Promise((r) => setTimeout(r, 250));
    }
  }

  learn(chars: number, tokens: number): void {
    if (chars > 0 && tokens > 0) this.tokensPerChar = 0.9 * this.tokensPerChar + 0.1 * (tokens / chars);
  }
}

export type Role = "query" | "document";

export interface Embedded {
  dims: number;
  /** n × dims, row-major, each row a unit vector. */
  vectors: Float32Array;
}

interface Meta {
  dims: number;
  count: number;
  done: number[];
  tokens: number;
  /** Summed request time: one worker's throughput is tokens / seconds. */
  seconds: number;
  model: string;
  requested: number | null;
  prompt: string;
}

export function cachePath(run: ModelRun, set: string, role: Role): string {
  return `.cache/vectors/${runKey(run)}/${set}-${role === "query" ? "queries" : "passages"}`;
}

export async function embedTexts(run: ModelRun, set: string, role: Role, texts: string[], log = console.log): Promise<Embedded> {
  const path = cachePath(run, set, role);
  const prompt = profileOf(run)[role];
  const requested = requestedDims(run);
  const batches = Math.ceil(texts.length / BATCH);
  let meta: Meta | null = existsSync(`${path}.json`) ? (JSON.parse(readFileSync(`${path}.json`, "utf8")) as Meta) : null;
  if (meta && (meta.count !== texts.length || meta.prompt !== prompt)) throw new Error(`${path}: cached for another input; delete it to rebuild`);
  let vectors = meta && existsSync(`${path}.f32`) ? readF32(`${path}.f32`) : null;
  // A missing or short vector file means the record of finished batches cannot be trusted.
  if (meta && (!vectors || vectors.length !== meta.count * meta.dims)) {
    meta = null;
    vectors = null;
  }
  const done = new Set(meta?.done ?? []);
  if (done.size === batches && vectors && meta) return { dims: meta.dims, vectors };

  const save = () => {
    mkdirSync(dirname(path), { recursive: true });
    // Written aside and renamed, so an interrupted save leaves the previous pair whole.
    writeFileSync(`${path}.f32.tmp`, Buffer.from(vectors!.buffer));
    renameSync(`${path}.f32.tmp`, `${path}.f32`);
    writeFileSync(`${path}.json.tmp`, JSON.stringify({ ...meta!, done: [...done].sort((a, b) => a - b) }));
    renameSync(`${path}.json.tmp`, `${path}.json`);
  };
  let lastLog = Date.now();
  const { parallel, tpm, rpm } = limits(run);
  const throttle = new Throttle(tpm, rpm);
  const embedBatch = async (b: number) => {
    const input = texts.slice(b * BATCH, (b + 1) * BATCH).map((t) => prompt + t);
    const chars = input.reduce((a, t) => a + t.length, 0);
    await throttle.room(chars, perText(run) ? input.length : 1);
    const started = performance.now();
    const res = await callWithRetry(run, input, role, requested);
    throttle.learn(chars, res.tokens);
    if (res.vectors.length !== input.length) throw new Error(`${run.model}: ${res.vectors.length} vectors for ${input.length} inputs`);
    const seconds = (performance.now() - started) / 1000;
    if (!meta || !vectors) {
      const dims = res.vectors[0]!.length;
      meta = { dims, count: texts.length, done: [], tokens: 0, seconds: 0, model: run.model, requested, prompt };
      vectors = new Float32Array(texts.length * dims);
    }
    res.vectors.forEach((v, i) => {
      if (v.length !== meta!.dims) throw new Error(`${run.model}: width changed from ${meta!.dims} to ${v.length}`);
      vectors!.set(unit(v), (b * BATCH + i) * meta!.dims);
    });
    meta.tokens += res.tokens;
    meta.seconds += seconds;
    done.add(b);
    if (done.size % 25 === 0) save();
    if (Date.now() - lastLog > 120_000) {
      lastLog = Date.now();
      log(`  ${set} ${role}: ${done.size}/${batches} batches, ${Math.round(meta.tokens / meta.seconds)} tokens/s per request stream`);
    }
  };

  const pending = Array.from({ length: batches }, (_, b) => b).filter((b) => !done.has(b));
  let next = 0;
  // The first batch alone when nothing is cached, so the width is known before workers share the buffer.
  if (!vectors && pending.length) await embedBatch(pending[next++]!);
  const worker = async () => {
    for (let k = next++; k < pending.length; k = next++) await embedBatch(pending[k]!);
  };
  await Promise.all(Array.from({ length: parallel }, worker));
  save();
  return { dims: meta!.dims, vectors: vectors! };
}

interface Batch {
  vectors: number[][];
  tokens: number;
}

/** Throttling is waited out however long it lasts; other failures are retried a few times. */
async function callWithRetry(run: ModelRun, input: string[], role: Role, dims: number | null): Promise<Batch> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await call(run, input, role, dims);
    } catch (e) {
      const throttled = e instanceof HttpError && e.status === 429;
      if (!throttled && (attempt >= 8 || (e instanceof HttpError && !e.retryable))) throw e;
      await new Promise((r) => setTimeout(r, Math.min(30_000, 1000 * 2 ** Math.min(attempt, 5))));
    }
  }
}

/** Titan Text Embeddings V2 takes one text a request. */
function perText(run: ModelRun): boolean {
  return run.backend === "bedrock" && /titan/.test(run.model);
}

/** At most `n` code points: Bedrock counts characters by code point. */
function codePoints(t: string, n: number): string {
  if (t.length <= n) return t;
  const cps = Array.from(t);
  return cps.length <= n ? t : cps.slice(0, n).join("");
}

/** Map over `items` with at most `n` calls in flight. */
async function pool<T, R>(items: T[], n: number, f: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = Array.from({ length: items.length });
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(n, items.length) }, async () => {
      for (let i = next++; i < items.length; i = next++) out[i] = await f(items[i]!);
    }),
  );
  return out;
}

class HttpError extends Error {
  readonly retryable: boolean;
  readonly status: number;
  constructor(message: string, retryable: boolean, status = 0) {
    super(message);
    this.retryable = retryable;
    this.status = status;
  }
}

/** Throttling, timeouts, server errors and Bedrock's 424 (a transient model error that says to try again) are retried. */
async function post(url: string, headers: Record<string, string>, body: unknown): Promise<{ json: unknown; res: Response }> {
  const res = await fetch(url, { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) });
  if (!res.ok) throw new HttpError(`${url.split("?")[0]} ${res.status}: ${(await res.text()).slice(0, 300)}`, res.status === 429 || res.status === 408 || res.status === 424 || res.status >= 500, res.status);
  return { json: await res.json(), res };
}

let keys: { openai?: string; gemini?: string; bedrock?: { key: string } } | null = null;
function key(name: "openai" | "gemini" | "bedrock"): string {
  keys ??= process.env.BENCH_KEYS ? JSON.parse(readFileSync(process.env.BENCH_KEYS, "utf8")) : {};
  const k = name === "bedrock" ? keys!.bedrock?.key : keys![name];
  if (!k) throw new HttpError(`no ${name} key in BENCH_KEYS`, false);
  return k;
}

async function call(run: ModelRun, input: string[], role: Role, dims: number | null): Promise<Batch> {
  switch (run.backend) {
    case "st":
      throw new HttpError("st vectors are written by scripts/st-embed.py", false);
    case "ollama":
      return ollamaFitting(run.model, input, dims);
    case "openai": {
      const { json } = await post("https://api.openai.com/v1/embeddings", { authorization: `Bearer ${key("openai")}` }, { model: run.model, input, dimensions: dims });
      const body = json as { data: { index: number; embedding: number[] }[]; usage: { prompt_tokens: number } };
      return { vectors: body.data.sort((a, b) => a.index - b.index).map((d) => d.embedding), tokens: body.usage.prompt_tokens };
    }
    case "gemini": {
      const taskType = role === "query" ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT";
      const { json } = await post(
        `https://generativelanguage.googleapis.com/v1beta/models/${run.model}:batchEmbedContents`,
        { "x-goog-api-key": key("gemini") },
        { requests: input.map((text) => ({ model: `models/${run.model}`, content: { parts: [{ text }] }, taskType, outputDimensionality: dims })) },
      );
      const body = json as { embeddings: { values: number[] }[] };
      // The batch endpoint reports no usage; tokens are counted from the price page's characters later.
      return { vectors: body.embeddings.map((e) => e.values), tokens: 0 };
    }
    case "bedrock":
      return bedrock(run.model, input, role, dims);
  }
}

async function ollama(model: string, input: string[], dims: number | null): Promise<Batch> {
  const { json } = await post(`${OLLAMA}/api/embed`, {}, dims === null ? { model, input } : { model, input, dimensions: dims });
  const body = json as { embeddings: number[][]; prompt_eval_count?: number };
  return { vectors: body.embeddings, tokens: body.prompt_eval_count ?? 0 };
}

/**
 * As Stuga's embed (packages/ai/src/retrieval/embed.ts, ollamaEmbedFitting): Ollama refuses some
 * texts as longer than the model's context instead of cutting them (bge-m3 on long Korean passages),
 * so a refused batch goes text by text, and a refused text is cut from its end until it fits.
 */
async function ollamaFitting(model: string, input: string[], dims: number | null): Promise<Batch> {
  const overflow = (e: unknown) => e instanceof HttpError && e.status === 413;
  try {
    return await ollama(model, input, dims);
  } catch (e) {
    if (!overflow(e)) throw e;
  }
  if (input.length > 1) {
    const parts: Batch[] = [];
    for (const text of input) parts.push(await ollamaFitting(model, [text], dims));
    return { vectors: parts.flatMap((p) => p.vectors), tokens: parts.reduce((a, p) => a + p.tokens, 0) };
  }
  let chars = [...input[0]!];
  for (let attempt = 0; attempt < 8 && chars.length > 1; attempt++) {
    chars = chars.slice(0, Math.floor(chars.length * 0.75));
    try {
      return await ollama(model, [chars.join("")], dims);
    } catch (e) {
      if (!overflow(e)) throw e;
    }
  }
  throw new HttpError(`ollama: ${model} refused a text as longer than its context, even cut to ${chars.length} characters`, false, 413);
}

const BEDROCK = "https://bedrock-runtime.us-west-2.amazonaws.com/model";

/**
 * Bedrock's text embedding models in us-west-2, each as its documentation asks: Cohere's with their
 * input types, Embed v4 at 1,024 dimensions through the US inference profile and cut at its end
 * (`truncate: RIGHT`); Embed Multilingual v3, which refuses a text over 2,048 characters rather than
 * cutting it, cut to that here first; Titan V2, one text a request, cut to its 50,000 characters and,
 * when it counts more than 8,192 tokens, cut in proportion and sent again. An empty text fails a
 * whole Cohere batch, so it goes as one space.
 */
async function bedrock(model: string, input: string[], role: Role, dims: number | null): Promise<Batch> {
  const headers = { authorization: `Bearer ${key("bedrock")}`, accept: "application/json" };
  const url = `${BEDROCK}/${encodeURIComponent(model)}/invoke`;
  const texts = input.map((t) => t || " ");
  if (/titan/.test(model)) {
    const one = async (text: string): Promise<{ v: number[]; tokens: number }> => {
      let t = codePoints(text, 50_000);
      for (;;) {
        try {
          const { json } = await post(url, headers, { inputText: t, dimensions: dims ?? 1024, normalize: true });
          const body = json as { embedding: number[]; inputTextTokenCount: number };
          return { v: body.embedding, tokens: body.inputTextTokenCount };
        } catch (e) {
          const over = e instanceof HttpError ? /request input token count: (\d+)/.exec(e.message) : null;
          if (!over) throw e;
          t = codePoints(t, Math.floor((Array.from(t).length * 8000) / Number(over[1])));
        }
      }
    };
    const out = await pool(texts, 8, (t) => callOne(one, t));
    return { vectors: out.map((o) => o.v), tokens: out.reduce((a, o) => a + o.tokens, 0) };
  }
  const v3 = /embed-multilingual-v3|embed-english-v3/.test(model);
  const body = v3
    ? { texts: texts.map((t) => codePoints(t, 2048)), input_type: role === "query" ? "search_query" : "search_document", truncate: "END" }
    : { texts, input_type: role === "query" ? "search_query" : "search_document", embedding_types: ["float"], output_dimension: dims, truncate: "RIGHT" };
  const { json, res } = await post(url, headers, body);
  const e = (json as { embeddings: number[][] | { float: number[][] } }).embeddings;
  return { vectors: Array.isArray(e) ? e : e.float, tokens: Number(res.headers.get("x-amzn-bedrock-input-token-count") ?? 0) };
}

/** One Titan text, with the same patience for throttling as a batch. */
async function callOne<R>(f: (t: string) => Promise<R>, t: string): Promise<R> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await f(t);
    } catch (e) {
      const throttled = e instanceof HttpError && e.status === 429;
      if (!throttled && (attempt >= 8 || (e instanceof HttpError && !e.retryable))) throw e;
      await new Promise((r) => setTimeout(r, Math.min(30_000, 1000 * 2 ** Math.min(attempt, 5))));
    }
  }
}

function readF32(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

function unit(v: number[]): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, (x) => x / n);
}
