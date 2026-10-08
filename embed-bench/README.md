# embed-bench

EmbeddingGemma 2, which Google released on 2026-10-06, against EmbeddingGemma 300m and eight other
embedding models on text retrieval, searched the way [Stuga](https://github.com/stuga-dev/stuga)
searches: each model gets the prompts and width a Stuga node asks for (on Bedrock, the input types
its documentation asks for), and the keyword search and fusion are copied from Stuga's code. The
author declares that Stuga has no relationship with Ollama or with any vendor whose model is tested
here, and paid list price for every hosted call.

## Findings

- **On these MIRACL hard-negative subsets, using Stuga's text-search configuration, EmbeddingGemma 2
  scores below EmbeddingGemma 300m.** Dense nDCG@10 over Korean, Chinese and English, both at bf16:
  55.5 against 61.1 (Chinese 54.7 against 63.0, English 50.1 against 55.8, Korean 61.7 against
  64.4); 54.1 for `embeddinggemma-2:270m`, the tested Ollama default tag, which is 4-bit. Fused with
  BM25, about a point apart (53.4 against 54.4). bge-m3 has the highest dense score among these
  runs, 63.9. Google's stated gains cover multimodal retrieval and code search
  ([launch post](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2)).
  Neither is tested here.
- **The Korean bf16 scores are close across backends:** 61.7 through Ollama, 62.4 through
  sentence-transformers running Google's weights. This does not isolate the cause of the gap.
- **BRIGHT StackOverflow (exploratory, 117 questions):** EmbeddingGemma 2 against 300m, 10.4 against
  16.4 dense and 17.2 against 19.2 fused.
- **No single similarity cutoff fits every model.** On MIRACL, a cosine distance of 0.6 lets nearly
  every random passage through for EmbeddingGemma 2 and keeps only 72–87% of relevant ones for
  Titan Text Embeddings V2. Stuga's Balanced level, measured for each model, kept 96–100% of
  relevant passages for all nine models checked, and let through 0.14–9.68% of random passages
  against its 1% aim (within 0.5–2% on 11 of 28 model and language pairs).

<!-- Sources, all in results/2026-10-07/: MIRACL nDCG@10 averages and per-language values from summary.json
(metrics."dense nDCG@10", perLanguage, metrics."Stuga nDCG@10"); Korean sentence-transformers vs Ollama bf16
from summary.json perLanguage.ko; BRIGHT from runs.json ("bright-so" dense.ndcg10 and stuga."2".ndcg10);
0.6 cutoff from cutoffs.json (fixed."0.6"); Balanced from probe-transfer.json (sets.*.balanced, within).
Google's claims from the model card at huggingface.co/google/embeddinggemma-2. -->

## Test sets

| Set | Questions | Passages | Source at pinned revision | Run on |
|---|---|---|---|---|
| MIRACL Korean | 213 | 43,293 | `mteb/MIRACLRetrievalHardNegatives` `332a9ac`, dev | all Ollama and Bedrock runs, sentence-transformers |
| MIRACL Chinese | 393 | 81,309 | the same | all Ollama and Bedrock runs |
| MIRACL English | 799 | 178,768 | the same | all Ollama and Bedrock runs |
| MIRACL Thai | 733 | 116,649 | the same | EmbeddingGemma 2 (nvfp4) |
| BRIGHT StackOverflow | 117 | 107,081 | `xlangai/BRIGHT` `3066d29` | 8 runs |
| C-MTEB DuRetrieval (web search) | 2,000 | 100,001 | `C-MTEB/DuRetrieval` `a1a333e` | Titan V2, EmbeddingGemma 2 (nvfp4) |
| C-MTEB EcomRetrieval | 1,000 | 100,902 | `C-MTEB/EcomRetrieval` `687de13` | the same two |
| C-MTEB MedicalRetrieval | 1,000 | 100,999 | `C-MTEB/MedicalRetrieval` `2039188` | the same two |
| C-MTEB CovidRetrieval | 949 | 100,001 | `C-MTEB/CovidRetrieval` `1271c78` | the same two |

<!-- Questions and passages: results/2026-10-07/runs.json (questions, passages). Revisions: src/sets.ts. -->

[src/sets.ts](src/sets.ts) downloads each set on first use into `.cache/hf/` and pins its full
revision (the C-MTEB relevance judgments at revisions of their own).

- **MIRACL hard negatives:** MMTEB's version of MIRACL, per language. Native speakers wrote the
  questions over Wikipedia and judged the passages. The corpus holds every judged passage plus the
  hard negatives MMTEB pooled for each question from several retrievers, BM25 among them. Only
  questions with a relevant passage count, as MMTEB scores them. A passage is embedded the way Stuga
  embeds a document's first chunk: its title, a blank line, its text.
  - Caveat: the pool, built partly from BM25's top hits, fills the corpus with passages BM25 ranks
    high, and an unjudged passage counts as not relevant. That may work against a keyword leg, and
    so against fusion; this experiment does not isolate the effect.
- **BRIGHT StackOverflow:** the subset's whole corpus of long documentation and forum passages.
  Gold passages are the pages that accepted answers link to. A passage BRIGHT excludes for a question
  is never ranked for it.
  - Caveat: 117 questions, so a difference of a few points may be noise.
- **C-MTEB:** four Chinese retrieval sets beside MIRACL's Wikipedia: web search, e-commerce listings,
  medical answers, COVID-19 news and notices. Dev judgments.
  - Caveat: run for two models only, as a check on the fusion and Chinese-tokenizer results.

## Lineup

| Model | Run as | Precision | Width | Sets |
|---|---|---|---|---|
| EmbeddingGemma 2 | Ollama `embeddinggemma-2:270m` | nvfp4, Ollama's default | 768 | MIRACL ko zh en th, BRIGHT, C-MTEB |
| EmbeddingGemma 2 | Ollama `embeddinggemma-2:270m-bf16-text` | bf16, by its tag name | 768 | MIRACL ko zh en, BRIGHT |
| EmbeddingGemma 2 | Ollama `embeddinggemma-2:270m-mxfp8-text` | mxfp8 | 768 | MIRACL ko zh en |
| EmbeddingGemma 2, no prompts | Ollama `embeddinggemma-2:270m`, raw text | nvfp4 | 768 | MIRACL ko zh en |
| EmbeddingGemma 2 | sentence-transformers `google/embeddinggemma-2`, Mac GPU | bf16 | 768 | MIRACL ko |
| EmbeddingGemma 300m | Ollama `embeddinggemma:300m` | BF16 | 768 | MIRACL ko zh en, BRIGHT |
| Qwen3-Embedding 0.6B | Ollama `qwen3-embedding:0.6b` | Q8_0 | 1,024 | MIRACL ko zh en |
| bge-m3 (dense vectors) | Ollama `bge-m3` | F16 | 1,024 | MIRACL ko zh en, BRIGHT |
| nomic-embed-text-v2-moe | Ollama `nomic-embed-text-v2-moe` (512-token context) | F16 | 768 | MIRACL ko zh en |
| Snowflake Arctic Embed 2 | Ollama `snowflake-arctic-embed2` | F16 | 1,024 | MIRACL ko zh en |
| Amazon Titan Text Embeddings V2 | Bedrock `amazon.titan-embed-text-v2:0` | hosted | 1,024 | MIRACL ko zh en, BRIGHT, C-MTEB |
| Cohere Embed Multilingual v3 | Bedrock `cohere.embed-multilingual-v3` | hosted | 1,024 | MIRACL ko zh en, BRIGHT |
| Cohere Embed v4 | Bedrock `us.cohere.embed-v4:0` | hosted | 1,024 | MIRACL ko zh en, BRIGHT |
| OpenAI text-embedding-3-large | OpenAI API | hosted | 1,024 | BRIGHT |

<!-- Width: results/2026-10-07/runs.json (dims). Precision: as each Ollama 0.40.0 tag reports it (the bf16 tag only by its name). -->

- **Ollama:** version 0.40.0 on one Mac (Apple M6, 32 GB), one request of 64 texts at a time, each
  tag at the quantization it ships with. Of the tags tested, only EmbeddingGemma 2's default is
  4-bit. When Ollama refuses a text as longer than the model's context (bge-m3 on long Korean
  passages), the batch goes again text by text, and a refused text is cut until it fits, as Stuga
  does.
- **Prompts:** each model gets the query and document prompts Stuga sends it,
  [src/stuga-profile.ts](src/stuga-profile.ts), copied verbatim from Stuga. EmbeddingGemma's are
  `task: search result | query: ` and `title: none | text: `. The no-prompt run sends the raw text,
  as an app that adds no prompts does.
- **Width:** as Stuga asks for its 1,024-dimension column. A narrower model returns its own width;
  a hosted model is asked for 1,024.
- **Bedrock**, us-west-2, each model as its documentation asks:
  - Cohere Embed v4 through the US inference profile, input types `search_query` and
    `search_document`, cut at the end (`truncate: RIGHT`).
  - Cohere Embed Multilingual v3, the same input types. It refuses a text over 2,048 characters, so
    texts are cut to that first.
  - Titan Text Embeddings V2, which takes no input type: one text a request, normalized, cut to
    50,000 characters, and cut further when it counts over 8,192 tokens.
- **OpenAI:** BRIGHT only, with the 1,024-dimension vectors rerank-bench made of the same texts
  ([scripts/import-rerank-bench.ts](scripts/import-rerank-bench.ts)).
- **Precision checks:** EmbeddingGemma 2 at bf16 and mxfp8 through Ollama's other tags, and through
  sentence-transformers at bf16 with the same prompts, cut at 2,048 tokens as Ollama cuts it
  ([scripts/st-embed.py](scripts/st-embed.py)).

## What is measured

- **Dense:** exact cosine similarity over every passage. nDCG@10 scored as MMTEB scores it
  (our TypeScript implementation of pytrec_eval's nDCG formula; scoring ties use descending passage
  IDs), Recall@24 and Recall@100.
- **BM25:** Stuga's keyword leg in pg_search 0.25.9 on Postgres 18 ([src/bm25.ts](src/bm25.ts)). The
  index and the query are copied from Stuga's SQL, only the table and column names changed:
  - an English-stemmed ICU column over all text;
  - the set's language column beside it, applied only to text in that script: Lindera for Korean,
    jieba for Chinese, with traditional characters folded to simplified (`chinese_convert=t2s`);
  - `disjunction_max` over a `match_disjunction` per column, 96 deep;
  - Thai has no pg_search tokenizer of its own, so Thai gets the ICU column only, as on a node.
- **Stuga hybrid** (Hybrid in the tables): the 96 nearest passages and BM25's 96, fused by
  reciprocal rank (k = 60, equal weights, ties shared as SQL's `rank()` shares them), 24 kept: what
  Stuga hands its reranker. The reranker is off. nDCG@10 and Recall@24.
  - Scored with no distance cutoff and at 0.9, the retrieval cutoff Stuga used before it measured
    each model. The two agree on every run (`runs.json`, keys `"2"` and `"0.9"`).
- **Comparisons:** MIRACL figures are macro averages over Korean, Chinese and English. Each model's
  difference from EmbeddingGemma 2 (nvfp4) is paired: the same questions, 2,000 bootstrap draws
  resampled within each language, 95% interval. An interval that includes zero means the test
  cannot tell the two apart, not that they are equivalent. Intervals are exploratory and not
  adjusted for multiple comparisons.
- **Cutoffs (exploratory):** for every question, the cosine distance to each relevant passage and to
  64 random passages, drawn with a fixed seed so every model gets the same ones
  ([src/eval.ts](src/eval.ts)). [src/cutoffs.ts](src/cutoffs.ts) sets fixed distances of 0.6 and
  0.9 against each model's own noise floor: the distance below which 0.1%, 1% or 5% of random pairs
  fall.
- **Stuga's own measurement:** [data/probe.json](data/probe.json) holds the texts a Stuga node embeds
  to measure its model: 768 searches and questions and 512 passages, on 32 topics in 8 languages.
  [src/probe-run.ts](src/probe-run.ts) embeds them, sets **Strict**, **Balanced** and **Loose** as a
  node does ([src/stuga-stats.ts](src/stuga-stats.ts) and [src/stuga-cutoff.ts](src/stuga-cutoff.ts),
  copied verbatim), and applies them to MIRACL. A level should let through its share of random
  passages (1 in 1,000, 100 or 20) and keep nearly every relevant one.
- **Fusion weights (exploratory):** the keyword leg at weights 1, 0.75, 0.5, 0.25 and 0 against the
  vector leg's 1, with no cutoff. Half against equal is a paired bootstrap
  ([scripts/fusion-analysis.ts](scripts/fusion-analysis.ts)).
- **Chinese keyword leg (exploratory):** ICU alone, ICU with jieba (Stuga's), jieba alone, and jieba
  without the folding of traditional to simplified, on MIRACL Chinese and the four C-MTEB sets
  ([scripts/bm25-languages.ts](scripts/bm25-languages.ts)). MIRACL's Chinese passages are Wikipedia
  text in both scripts, and its questions are simplified.

## Results

### MIRACL: Korean, Chinese, English

| Model | Run as | Dense nDCG@10 | vs EmbeddingGemma 2 | ko | zh | en | Dense Recall@100 | Hybrid nDCG@10 | Hybrid Recall@24 |
|---|---|---|---|---|---|---|---|---|---|
| bge-m3 | Ollama F16 | 63.9 | +9.8 [8.2, 11.5] | 70.2 | 63.7 | 57.8 | 97.2 | 56.3 | 84.8 |
| Cohere Embed Multilingual v3 | Bedrock | 61.6 | +7.5 [5.9, 9.1] | 66.1 | 60.9 | 57.7 | 96.9 | 56.0 | 83.6 |
| nomic-embed-text-v2-moe | Ollama F16 | 61.2 | +7.1 [5.3, 9.0] | 66.6 | 61.5 | 55.5 | 97.1 | 53.3 | 82.0 |
| EmbeddingGemma 300m | Ollama BF16 | 61.1 | +7.0 [5.6, 8.4] | 64.4 | 63.0 | 55.8 | 97.2 | 54.4 | 83.0 |
| Snowflake Arctic Embed 2 | Ollama F16 | 60.6 | +6.5 [5.0, 8.1] | 66.2 | 61.3 | 54.3 | 96.1 | 54.5 | 82.7 |
| Cohere Embed v4 | Bedrock | 60.2 | +6.2 [4.6, 7.8] | 62.3 | 60.9 | 57.5 | 95.8 | 54.7 | 81.9 |
| Qwen3-Embedding 0.6B | Ollama Q8_0 | 56.7 | +2.7 [1.0, 4.3] | 57.6 | 58.7 | 53.9 | 94.6 | 53.2 | 79.5 |
| EmbeddingGemma 2 | Ollama bf16 | 55.5 | +1.4 [0.6, 2.1] | 61.7 | 54.7 | 50.1 | 96.3 | 53.4 | 81.7 |
| EmbeddingGemma 2 | Ollama mxfp8 | 55.4 | +1.3 [0.5, 2.1] | 61.7 | 54.7 | 49.7 | 96.2 | 53.3 | 81.7 |
| EmbeddingGemma 2 | Ollama nvfp4 (default) | 54.1 | reference | 59.7 | 53.5 | 49.0 | 95.8 | 53.0 | 81.7 |
| Amazon Titan Text Embeddings V2 | Bedrock | 51.3 | −2.7 [−4.4, −1.0] | 55.8 | 49.2 | 49.0 | 93.5 | 51.8 | 79.5 |
| EmbeddingGemma 2, no prompts | Ollama nvfp4 | 50.5 | −3.6 [−4.7, −2.4] | 59.3 | 49.3 | 42.8 | 94.2 | 51.6 | 80.2 |
| BM25 alone | pg_search | 27.9 | | 34.3 | 19.6 | 29.7 | | | |

From `results/2026-10-07/summary.json` (`summary.txt`); BM25 per language from `runs.json`. The ko,
zh and en columns are dense nDCG@10.

- Without its prompts, EmbeddingGemma 2 loses 3.6 points dense.
- Fused with BM25, nDCG@10 falls for every run except the two weakest dense runs (Titan V2 and
  EmbeddingGemma 2 without prompts), and the spread narrows from 50.5–63.9 to 51.6–56.3. The
  pooling caveat under Test sets applies.

### EmbeddingGemma 2 by precision, Korean

| Build | Dense nDCG@10 | Dense Recall@24 | Dense Recall@100 | Hybrid nDCG@10 |
|---|---|---|---|---|
| sentence-transformers, bf16 | 62.4 | 87.9 | 97.4 | 61.9 |
| Ollama bf16 | 61.7 | 88.0 | 97.4 | 61.9 |
| Ollama mxfp8 | 61.7 | 86.6 | 97.4 | 61.9 |
| Ollama nvfp4 (default) | 59.7 | 85.5 | 96.2 | 61.4 |
| EmbeddingGemma 300m, Ollama BF16 | 64.4 | 87.4 | 96.9 | 62.0 |

From `results/2026-10-07/runs.json` (`miracl-ko`). Over Korean, Chinese and English, bf16 scores
+1.4 [0.6, 2.1] and mxfp8 +1.3 [0.5, 2.1] over nvfp4, dense (`summary.json`).

### BRIGHT StackOverflow (exploratory)

| Model | Run as | Dense nDCG@10 | Dense Recall@24 | Dense Recall@100 | Hybrid nDCG@10 | Hybrid Recall@24 |
|---|---|---|---|---|---|---|
| EmbeddingGemma 300m | Ollama BF16 | 16.4 | 35.5 | 61.0 | 19.2 | 38.6 |
| Cohere Embed Multilingual v3 | Bedrock | 15.9 | 30.6 | 52.4 | 16.3 | 36.4 |
| Amazon Titan Text Embeddings V2 | Bedrock | 12.8 | 31.9 | 53.0 | 18.4 | 35.0 |
| OpenAI text-embedding-3-large | OpenAI | 12.7 | 28.3 | 54.4 | 18.0 | 34.2 |
| Cohere Embed v4 | Bedrock | 12.7 | 27.9 | 48.1 | 16.7 | 32.8 |
| EmbeddingGemma 2 | Ollama nvfp4 (default) | 10.5 | 22.9 | 47.0 | 17.0 | 33.4 |
| EmbeddingGemma 2 | Ollama bf16 | 10.4 | 24.8 | 46.9 | 17.2 | 33.7 |
| bge-m3 | Ollama F16 | 10.2 | 24.9 | 44.7 | 14.3 | 30.3 |
| BM25 alone | pg_search | 14.0 | 24.9 | | | |

From `results/2026-10-07/runs.json` (`bright-so`). Fusion raises nDCG@10 for every model here. The
paired intervals in `summary.json` cover MIRACL only.

### Cutoffs (exploratory)

A fixed cutoff against the floor Stuga measures for each model, on MIRACL Korean, Chinese and
English (ranges over the three):

| Model | 0.6 keeps relevant | 0.6 lets random | Balanced distance, questions | Keeps relevant | Lets random |
|---|---|---|---|---|---|
| EmbeddingGemma 2 (nvfp4) | 100% | 100% | 0.351 | 99.3–100% | 0.48–1.35% |
| EmbeddingGemma 300m | 95.6–99.0% | 0.1–0.2% | 0.751 | 99.8–100% | 0.37–1.80% |
| Qwen3-Embedding 0.6B | 98.2–99.9% | 0.2–1.5% | 0.628 | 99.5–100% | 0.38–2.41% |
| bge-m3 | 99.7–100% | 0.7–2.0% | 0.534 | 98.8–99.6% | 0.18–0.49% |
| nomic-embed-text-v2-moe | 97.2–98.7% | 0.1–0.5% | 0.665 | 99.2–99.9% | 0.22–1.91% |
| Snowflake Arctic Embed 2 | 95.4–97.4% | 0.1–0.4% | 0.677 | 98.9–99.8% | 0.16–1.22% |
| Amazon Titan Text Embeddings V2 | 72.0–86.7% | 0.0–0.1% | 0.782 | 96.3–98.8% | 0.21–0.65% |
| Cohere Embed Multilingual v3 | 99.7–100% | 1.0–29.5% | 0.508 | 98.6–100% | 0.14–3.03% |
| Cohere Embed v4 | 96.1–98.4% | 0.1–0.7% | 0.738 | 99.7–100% | 0.45–9.68% |

The 0.6 columns are from `results/2026-10-07/cutoffs.json` (`fixed`, `cutoffs.txt`); the measured
columns from `probe-transfer.json` (`levels.question.balanced`, `sets.*.balanced`).

- Balanced is meant to let 1% through. In English every model let through less than half of that;
  in Korean up to 9.68% (Cohere Embed v4). It landed within half to double of 1% on 11 of 28 model
  and language pairs, Thai included for EmbeddingGemma 2 (`probe-transfer.json`, `within`).
- Every MIRACL query counts as a question under Stuga's rule (`questions.short` is 0), so only the
  question floor was tested.
- 0.9 keeps 99.5–100% of relevant passages and lets through from 4.6% of random ones (Titan V2,
  English) to all of them (EmbeddingGemma 2) (`cutoffs.json`, `fixed."0.9"`).

On BRIGHT StackOverflow, many relevant passages sit as far away as random ones:

| Model | 0.6 keeps relevant | 0.6 lets random | 1% floor fitted on this BRIGHT sample keeps relevant |
|---|---|---|---|
| Amazon Titan Text Embeddings V2 | 22.2% | 0.0% | 70.3% |
| Cohere Embed v4 | 45.6% | 0.1% | 75.1% |
| OpenAI text-embedding-3-large | 46.4% | 0.2% | 74.3% |
| EmbeddingGemma 300m | 84.1% | 3.2% | 66.9% |
| Cohere Embed Multilingual v3 | 97.5% | 14.1% | 74.1% |
| bge-m3 | 99.6% | 46.9% | 62.3% |
| EmbeddingGemma 2 (nvfp4) | 100% | 100% | 62.6% |

From `results/2026-10-07/cutoffs.json` (`bright-so`: `fixed."0.6"`, `levels.balanced`). Stuga's Ask
and agents' `retrieve` take no distance cutoff; the search box uses the measured floor.

### Fusion weight (exploratory)

Half the keyword weight (0.5) against Stuga's equal weight, nDCG@10, paired 95% interval, one row
per run and set:

| Sets | Rows | Half better | Half worse | Cannot tell |
|---|---|---|---|---|
| MIRACL ko, zh, en, th | 38 | 38 | 0 | 0 |
| BRIGHT StackOverflow | 8 | 0 | 5 | 3 |
| C-MTEB, Titan V2 and EmbeddingGemma 2 | 8 | 5 | 2 | 1 |

From `results/2026-10-07/fusion.json` (`half_vs_equal`, `fusion.txt`).

- On C-MTEB, half the weight also lowers Recall@24 where BM25 is strong: Titan V2 e-commerce −2.7
  [−4.2, −1.1], Titan V2 COVID −2.3 [−4.4, −1.4], EmbeddingGemma 2 COVID −1.0 [−2.6, −0.2].
- MIRACL's negatives were pooled partly from BM25, which may favour a lighter keyword leg there;
  this experiment does not isolate the effect. Stuga keeps equal weights.

| C-MTEB set | BM25 | Titan V2 dense | Titan V2 hybrid | EmbeddingGemma 2 dense | EmbeddingGemma 2 hybrid |
|---|---|---|---|---|---|
| DuRetrieval | 61.3 | 73.4 | 71.9 | 75.5 | 71.9 |
| EcomRetrieval | 48.4 | 48.5 | 54.7 | 59.9 | 57.3 |
| MedicalRetrieval | 31.4 | 43.4 | 43.5 | 43.1 | 42.6 |
| CovidRetrieval | 78.8 | 64.2 | 75.7 | 75.8 | 80.8 |

nDCG@10 from `results/2026-10-07/runs.json` (`cmteb-*`); EmbeddingGemma 2 is the nvfp4 build.

### Chinese keyword leg (exploratory)

BM25 alone, nDCG@10:

| Keyword leg | MIRACL zh | DuRetrieval | EcomRetrieval | MedicalRetrieval | CovidRetrieval |
|---|---|---|---|---|---|
| ICU alone | 20.5 | 61.5 | 48.8 | 32.5 | 79.8 |
| ICU + jieba, folded (Stuga) | 19.6 | 61.3 | 48.2 | 31.5 | 78.8 |
| jieba, folded, alone | 19.2 | 59.0 | 38.7 | 28.9 | 78.5 |
| ICU + jieba, not folded | 16.2 | 61.4 | 47.5 | 31.4 | 78.8 |
| jieba, not folded, alone | 15.0 | 59.0 | 37.8 | 28.9 | 78.5 |

From `results/2026-10-07/bm25-zh.json`, which ranks ties in the order Postgres returns them;
`runs.json` breaks ties by passage id, as MMTEB does, so the two differ by up to 0.2.

- Without folding, jieba beside ICU costs MIRACL Chinese 4.3 [2.0, 6.0] against ICU alone; folded,
  the two cannot be told apart (−0.9 [−3.9, 1.8]). On the four simplified C-MTEB sets, folded jieba
  beside ICU is within about a point of ICU alone (`bm25-zh.json`).
- In the hybrid, MIRACL Chinese with EmbeddingGemma 2 scores 46.0 with folding, 43.3 with ICU alone
  and 41.7 without folding; with Titan V2, 45.3, 43.1 and 41.3 (`runs.json`, `miracl-zh`,
  `miracl-zh.icu`, `miracl-zh.jieba`). Folded jieba beside ICU against ICU alone: +2.7 [0.3, 5.0]
  for EmbeddingGemma 2, +2.2 [−0.3, 4.4] for Titan V2 (`fusion.json`, `jieba_vs_icu`).

## Caveats

- One Mac (Apple M6, 32 GB) and Ollama 0.40.0. Each Ollama model runs at the quantization its tag
  ships with, and another build of the same model can score differently: EmbeddingGemma 2 gains 1.4
  at bf16.
- Bedrock's models are as served on 2026-10-07; OpenAI's BRIGHT vectors are rerank-bench's.
- MIRACL's hard negatives were pooled partly from BM25, which may work against fusion there; C-MTEB,
  the check on that, ran for two models only.
- BRIGHT has 117 questions, and its model comparisons carry no paired intervals. Long BRIGHT
  passages are cut where each server cuts them; Cohere Embed Multilingual v3 reads the first 2,048
  characters.
- Text retrieval only. BRIGHT StackOverflow asks technical questions of documentation and forum
  passages; it is not a code-retrieval set.
- EmbeddingGemma 2 reads at most 2,048 tokens of a text here, through Ollama and in the
  sentence-transformers check alike; its 8K-token context is not evaluated.
- The sentence-transformers check covers Korean only, and Thai ran for EmbeddingGemma 2 only.
- Stuga's own measurement was checked on MIRACL Korean, Chinese and English (and Thai for
  EmbeddingGemma 2), with question-style queries only.
- The reranker is off: these are first-stage results.

## Run it

Node 26 and pnpm 12, Postgres 18 with pg_search 0.25.9, and Ollama listening on 127.0.0.1:11435.

```sh
cd experiments/embed-bench
pnpm install
mkdir -p results/<date>
cat > keys.json <<'EOF'
{ "openai": "…", "bedrock": { "key": "…" } }
EOF
OLLAMA_HOST=127.0.0.1:11435 ollama pull embeddinggemma-2:270m
node src/embed-run.ts ollama:embeddinggemma-2:270m miracl-ko,miracl-zh,miracl-en,bright-so
node src/embed-run.ts ollama:embeddinggemma-2:270m+raw miracl-ko,miracl-zh,miracl-en      # without prompts
BENCH_KEYS=keys.json node src/embed-run.ts bedrock:us.cohere.embed-v4:0 miracl-ko,miracl-zh,miracl-en,bright-so
node src/eval.ts ollama:embeddinggemma-2:270m miracl-ko,miracl-zh,miracl-en,bright-so     # builds the BM25 leg too
node src/eval.ts ollama:embeddinggemma-2:270m+raw miracl-ko,miracl-zh,miracl-en
node src/eval.ts bedrock:us.cohere.embed-v4:0 miracl-ko,miracl-zh,miracl-en,bright-so      # each run, from its cached vectors
node src/report.ts ollama/embeddinggemma-2:270m results/<date> > results/<date>/summary.txt
node src/cutoffs.ts results/<date> > results/<date>/cutoffs.txt
node scripts/fusion-analysis.ts results/<date>/fusion.json > results/<date>/fusion.txt
node src/probe-run.ts ollama:embeddinggemma-2:270m results/<date>
node scripts/runs-json.ts results/<date>/runs.json
```

- **Keys** come from the file `BENCH_KEYS` names, never from the command line. Bedrock takes an API
  key for us-west-2.
- **Postgres:** `BENCH_DATABASE_URL` (default `postgres://bench@127.0.0.1:55441/bench`), a database
  with `CREATE EXTENSION pg_search`. The first run of a set builds its table and BM25 index.
- **Ollama:** `OLLAMA_URL` points elsewhere (default `http://127.0.0.1:11435`).
- **OpenAI on BRIGHT:** `node scripts/import-rerank-bench.ts ../rerank-bench` copies rerank-bench's
  vectors, or `BENCH_KEYS=keys.json node src/embed-run.ts openai:text-embedding-3-large bright-so`
  embeds them again.
- **Chinese keyword variants:** `BM25_VARIANT=icu` (ICU alone) or `BM25_VARIANT=jieba` (no folding)
  before `node src/bm25-run.ts <sets>` and `node src/eval.ts <run> <sets>`, then
  `node scripts/bm25-languages.ts miracl-zh,cmteb-du,cmteb-ecom,cmteb-medical,cmteb-covid results/<date>/bm25-zh.json`.
- **sentence-transformers reference**, in a Python environment with sentence-transformers and torch:

  ```sh
  node scripts/dump-texts.ts ko
  st-env/bin/python -I scripts/st-embed.py google/embeddinggemma-2 miracl-ko .
  node src/eval.ts st:google/embeddinggemma-2 miracl-ko
  ```

- `runs.json` gathers the `summary` of every `.cache/eval/<run>/<set>.json`
  ([scripts/runs-json.ts](scripts/runs-json.ts)).
- Vectors are cached under `.cache/` (about 19 GB for this run) and an interrupted run resumes; they
  are not committed. `pnpm test` checks the metrics.

## Licence

Code MIT, including the files copied from Stuga and Stuga's probe texts in `data/probe.json`. Data:
MMTEB's MIRACL hard negatives are CC BY-SA 4.0 and BRIGHT is CC BY 4.0; the C-MTEB sets state no
licence on Hugging Face. No set's text is committed here, only scores.
