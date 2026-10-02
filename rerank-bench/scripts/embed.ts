/**
 * Embeddings from OpenAI's text-embedding-3-large at 1,024 dimensions, Stuga's default. Requests keep
 * under the account's tokens-per-minute limit, and progress is saved under .cache/ as it goes, so an
 * interrupted run, or a rebuild, never pays for the same text twice.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export const EMBED_MODEL = "text-embedding-3-large";
export const EMBED_DIMS = 1024;
const BATCH = 96;
const PARALLEL = 16;
/** Below OpenAI's 1,000,000 tokens a minute for this model on the account that ran the benchmark. */
const TOKENS_PER_MINUTE = 850_000;

/** One unit vector per text, in input order. */
export async function embed(texts: string[], apiKey: string, cache: string): Promise<Float32Array[]> {
  const size = texts.length * EMBED_DIMS;
  const out = new Float32Array(size);
  const progress = `${cache}.done.json`;
  let done = new Set<number>();
  if (existsSync(cache)) {
    const buf = readFileSync(cache);
    if (buf.byteLength === size * 4) out.set(new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
    done = existsSync(progress) ? new Set(JSON.parse(readFileSync(progress, "utf8")) as number[]) : new Set(Array.from({ length: Math.ceil(texts.length / BATCH) }, (_, b) => b));
  }
  const batches = Math.ceil(texts.length / BATCH);
  const todo = Array.from({ length: batches }, (_, b) => b).filter((b) => !done.has(b));
  const save = () => {
    mkdirSync(dirname(cache), { recursive: true });
    writeFileSync(cache, Buffer.from(out.buffer));
    writeFileSync(progress, JSON.stringify([...done]));
  };
  // Tokens sent in the last minute, as (time, estimate) pairs; an estimate of a character in three
  // tokens errs high for prose and about right for code.
  const window: [number, number][] = [];
  async function room(estimate: number) {
    for (;;) {
      const now = Date.now();
      while (window.length && now - window[0]![0] > 60_000) window.shift();
      const used = window.reduce((a, [, n]) => a + n, 0);
      if (used + estimate <= TOKENS_PER_MINUTE) return window.push([now, estimate]);
      await sleep(60_000 - (now - window[0]![0]) + 50);
    }
  }
  let tokens = 0;
  let next = 0;
  async function worker() {
    for (let k = next++; k < todo.length; k = next++) {
      const b = todo[k]!;
      const batch = texts.slice(b * BATCH, (b + 1) * BATCH).map((t) => t || " ");
      await room(Math.ceil(batch.reduce((a, t) => a + t.length, 0) / 3));
      const res = await call(batch, apiKey);
      tokens += res.tokens;
      res.vectors.forEach((v, i) => out.set(unit(v), (b * BATCH + i) * EMBED_DIMS));
      done.add(b);
      if (done.size % 50 === 0) save();
      if (done.size % 100 === 0 || done.size === batches) console.log(`  ${done.size}/${batches} batches, ${tokens.toLocaleString("en")} tokens this run`);
    }
  }
  await Promise.all(Array.from({ length: PARALLEL }, worker));
  save();
  return texts.map((_, i) => out.subarray(i * EMBED_DIMS, (i + 1) * EMBED_DIMS));
}

async function call(input: string[], apiKey: string): Promise<{ vectors: number[][]; tokens: number }> {
  for (let attempt = 1; ; attempt++) {
    const res = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
      body: JSON.stringify({ model: EMBED_MODEL, input, dimensions: EMBED_DIMS }),
    }).catch((e: unknown) => e as Error);
    if (!(res instanceof Error) && res.ok) {
      const body = (await res.json()) as { data: { index: number; embedding: number[] }[]; usage: { prompt_tokens: number } };
      return { vectors: body.data.sort((a, b) => a.index - b.index).map((d) => d.embedding), tokens: body.usage.prompt_tokens };
    }
    const retryable = res instanceof Error || res.status === 429 || res.status >= 500;
    if (!retryable || attempt >= 20) throw new Error(res instanceof Error ? res.message : `embeddings ${res.status}: ${(await res.text()).slice(0, 120)}`);
    // Wait as long as the API asks (`6s`, `1m2s` or seconds), else back off.
    const asked = res instanceof Error ? null : (res.headers.get("x-ratelimit-reset-tokens") ?? res.headers.get("retry-after"));
    await sleep(asked ? duration(asked) + 250 : Math.min(30_000, 1000 * 2 ** attempt));
  }
}

function duration(s: string): number {
  if (/^\d+(\.\d+)?$/.test(s)) return Number(s) * 1000;
  let ms = 0;
  for (const [, n, unit] of s.matchAll(/(\d+(?:\.\d+)?)(ms|s|m|h)/g)) ms += Number(n) * { ms: 1, s: 1000, m: 60_000, h: 3_600_000 }[unit as "ms" | "s" | "m" | "h"];
  return ms || 1000;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function unit(v: number[]): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, (x) => x / n);
}
