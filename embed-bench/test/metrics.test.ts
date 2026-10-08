import assert from "node:assert/strict";
import { test } from "node:test";
import { hit1, ndcg, recall, rrf, sqlRank } from "../src/metrics.ts";

test("nDCG@10 is 1 for a perfect list and discounts by log2 of the position", () => {
  const gains = new Map([[7, 1]]);
  assert.equal(ndcg([7, 1, 2], gains), 1);
  assert.equal(ndcg([1, 7], gains), 1 / Math.log2(3));
  assert.equal(ndcg([1, 2], gains), 0);
});

test("nDCG's ideal list counts only positive gains, at most k of them", () => {
  const gains = new Map([[1, 1], [2, 0], [3, 1]]);
  assert.equal(ndcg([1, 3], gains), 1);
  assert.ok(Math.abs(ndcg([1, 2, 3], gains) - (1 + 1 / 2) / (1 + 1 / Math.log2(3))) < 1e-12);
});

test("recall and hit@1", () => {
  const gains = new Map([[1, 1], [2, 1], [3, 0]]);
  assert.equal(recall([1, 3, 4], gains, 2), 0.5);
  assert.equal(recall([4, 2, 1], gains, 3), 1);
  assert.equal(hit1([3, 1], gains), 0);
  assert.equal(hit1([2], gains), 1);
});

test("rank() gives equal scores one position and skips after them", () => {
  assert.deepEqual(sqlRank([0.9, 0.5, 0.5, 0.1]), [1, 2, 2, 4]);
});

test("RRF sums 1/(60 + rank) over both legs and breaks ties by passage id", () => {
  const pid = (i: number) => `p${i}`;
  // 2 is second in both legs and beats 1 and 3, each first in one leg only.
  const out = rrf({ ids: [1, 2], scores: [9, 8] }, { ids: [3, 2], sims: [0.9, 0.8] }, pid, 3);
  assert.deepEqual(out, [2, 1, 3]);
  // Tied keyword scores share rank 1, so 4 and 5 tie and the smaller id comes first.
  assert.deepEqual(rrf({ ids: [5, 4], scores: [1, 1] }, { ids: [], sims: [] }, pid, 2), [4, 5]);
});
