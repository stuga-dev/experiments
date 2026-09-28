import { strict as assert } from "node:assert";
import { test } from "node:test";
import { bootstrapMean, firstGoldRank, hitAt, ndcgAt, quantile, reciprocalRank } from "../src/metrics.ts";
import { orderByScores, parseScores, statedScores } from "../src/stuga.ts";

test("first gold rank is 1-based, 0 when absent", () => {
  assert.equal(firstGoldRank(["a", "b", "c"], new Set(["c"])), 3);
  assert.equal(firstGoldRank(["a", "b"], new Set(["z"])), 0);
  assert.equal(reciprocalRank(0), 0);
  assert.equal(reciprocalRank(4), 0.25);
  assert.equal(hitAt(3, 3), 1);
  assert.equal(hitAt(4, 3), 0);
});

test("nDCG is 1 with every gold passage first, and counts only the top k", () => {
  assert.equal(ndcgAt(["g1", "g2", "x"], new Set(["g1", "g2"]), 10), 1);
  assert.ok(ndcgAt(["x", "g1"], new Set(["g1"]), 10) < 1);
  assert.equal(ndcgAt(["x", "g1"], new Set(["g1"]), 1), 0);
});

test("quantiles interpolate and the bootstrap brackets the mean", () => {
  assert.equal(quantile([1, 2, 3, 4], 0.5), 2.5);
  const [lo, hi] = bootstrapMean([0, 1, 0, 1, 1, 1, 0, 1]);
  assert.ok(lo <= 0.625 && hi >= 0.625);
});

test("the judge's answer parses as Stuga parses it", () => {
  assert.deepEqual(parseScores('```json\n[{"i":1,"score":9},{"i":0,"score":2}]\n```', 3), [2, 9, 0]);
  assert.deepEqual(parseScores('[{"i":0,"score":14}]', 1), [10]);
  assert.equal(parseScores("[{\"i\":0,", 2), null);
  assert.equal(parseScores("no array", 2), null);
});

test("ties keep fusion order, and a failed judge keeps it all", () => {
  assert.deepEqual(orderByScores([5, 9, 5], 3), [1, 0, 2]);
  assert.deepEqual(orderByScores(null, 3), [0, 1, 2]);
});

test("stated scores: objects without the array, [i] lines, and not a lone index", () => {
  assert.deepEqual(statedScores('{"i":0,"score":1}\n{"i":1,"score":9}', 2), [1, 9]);
  assert.deepEqual(statedScores("[0] 2\n[1] 7\n[2] 5", 3), [2, 7, 5]);
  assert.deepEqual(statedScores('[{"i":0,"score":3},{"i":1,"score":8}]\nNote: [0] is related.', 2), [3, 8]);
  assert.equal(statedScores("[2]", 4), null);
  assert.equal(statedScores('{"i":0,"score":5}', 4), null);
});
