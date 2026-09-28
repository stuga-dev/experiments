# BRIGHT, StackOverflow, BM25 only

The same questions, passages and licences as [bright-stackoverflow](../bright-stackoverflow/SOURCES.md),
with a first stage that has no vectors: each question's candidates are the 24 passages BM25
(k1 = 0.9, b = 0.4) ranks highest for the question. It stands for Stuga with **Embeddings**
off. `scripts/build-bright.ts` builds both sets in one pass.
