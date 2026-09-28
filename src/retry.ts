/** Faults worth another attempt: throttling, overload, 5xx (Bedrock words its own), and the network. */
const RETRYABLE = /\b(408|409|429|5\d\d)\b|internal server error|service ?unavailable|throttl|rate.?limit|too many requests|overloaded|timed? ?out|ECONNRESET|ETIMEDOUT|EAI_AGAIN|socket hang up|fetch failed|network/i;

export function isRetryable(message: string): boolean {
  return RETRYABLE.test(message);
}

/**
 * A call refused by our own account, for billing or its rate limit, or failed by the vendor's or
 * this machine's infrastructure after every retry, measured nothing about the ranker: it runs again.
 */
export const ACCOUNT_ERROR = /no credits|insufficient_quota|billing|payment required|organization max RPM|rate_limit_reached_error|rate limit reached for .+ in organization/i;

export const MAX_ATTEMPTS = 4;

export async function backoff(attempt: number): Promise<void> {
  const ms = Math.min(8000, 1000 * 2 ** (attempt - 1)) * (0.5 + Math.random() / 2);
  await new Promise((r) => setTimeout(r, ms));
}

/**
 * Run `once` until it answers or fails for good. `ms` is the answering attempt alone; `totalMs` is
 * everything from the first attempt, waits included, which is what a user of the ranker waits.
 */
export async function withRetries<T extends { error?: string; ms: number; attempts: number; totalMs?: number }>(
  once: () => Promise<T>,
): Promise<T> {
  const t0 = performance.now();
  let last: T | undefined;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    last = { ...(await once()), attempts: attempt };
    if (!last.error || !isRetryable(last.error) || attempt === MAX_ATTEMPTS) return { ...last, totalMs: performance.now() - t0 };
    await backoff(attempt);
  }
  return { ...last!, totalMs: performance.now() - t0 };
}
