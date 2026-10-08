import assert from "node:assert/strict";
import { test } from "node:test";
import { mulberry32, paired } from "../src/bootstrap.ts";

test("the generator does not fall into a short cycle", () => {
  const rand = mulberry32(11);
  const seen = new Set<number>();
  for (let i = 0; i < 200_000; i++) seen.add(rand());
  assert.ok(seen.size > 199_000);
});

test("a constant difference has a zero-width interval at that difference", () => {
  assert.deepEqual(paired([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], 7).map((x) => +x.toFixed(9)), [0.1, 0.1, 0.1]);
});

test("the interval of a noisy difference covers its mean and is about as wide as the normal one", () => {
  const rand = mulberry32(3);
  const a = Array.from({ length: 1000 }, () => rand());
  const b = a.map((x) => x + 0.05 + (rand() - 0.5) * 0.4);
  const [mean, low, high] = paired(a, b, 11);
  assert.ok(low < mean && mean < high);
  const sd = Math.sqrt(b.reduce((s, x, i) => s + (x - a[i]! - mean) ** 2, 0) / (a.length - 1));
  const half = 1.96 * sd / Math.sqrt(a.length);
  assert.ok(Math.abs((high - low) / 2 - half) < 0.15 * half, `${(high - low) / 2} vs ${half}`);
});
