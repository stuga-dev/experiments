import { strict as assert } from "node:assert";
import { test } from "node:test";
import { isGold, setOf, type Chunk, type Question } from "../src/corpus.ts";

const chunk = (id: string): Chunk => ({ id, sample: "ja", title: "T", section: "", text: "x" });

test("gold is the judged-relevant passages, by id", () => {
  const q: Question = { id: "ja-1", sample: "ja", q: "?", category: "ja", goldIds: ["ja/a"] };
  assert.equal(isGold(chunk("ja/a"), q), true);
  assert.equal(isGold(chunk("ja/b"), q), false);
});

test("--set takes a known set only", () => {
  assert.equal(setOf("miracl"), "miracl");
  assert.throws(() => setOf("stuga-samples"));
  assert.throws(() => setOf(undefined));
});
