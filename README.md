# rerank-bench

Rerankers compared on identical candidates: a System One classifier (TypeSafe's Jev), LLMs used as
relevance judges, and dedicated rerankers, hosted and local. They are tested on two public
benchmarks whose questions and relevance judgments were made by people: MIRACL in six languages,
and BRIGHT's StackOverflow questions. Each ranker reorders 24 candidates per question and sees
them the way [Stuga](https://github.com/stuga-dev/stuga)'s judge does. The write-up is on the
[Stuga blog](https://stuga.dev/blog/reranking-with-a-classifier).

## Leaderboards

### MIRACL: English, Chinese, Japanese, Korean, Arabic, Thai

<!-- leaderboard:miracl:start -->
Ranked on 240 questions whose answer was among the 24 candidates. Intervals are bootstrap 95%.

| Ranker | Kind | Hit@1 | MRR | Recall@8 | p50 latency | p95 latency | $ / 1k searches | Unusable answers |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Reranker 4B (local) | reranker | 81% (75%–86%) | 0.886 | 100% | 5.3 s | 8.4 s | — | 0 |
| Claude Opus 5.5 | llm | 81% (76%–86%) | 0.882 | 99% | 3.5 s | 4.7 s | $37 | 0 |
| Claude Sonnet 5.5 | llm | 80% (75%–85%) | 0.872 | 99% | 2.0 s | 3.8 s | $20 | 0 |
| Cohere Rerank 3.5 | reranker | 78% (73%–83%) | 0.870 | 98% | 163 ms | 218 ms | $2.0 | 0 |
| Claude Sonnet 5 | llm | 80% (75%–85%) | 0.868 | 98% | 3.7 s | 10.3 s | $20 | 0 |
| Claude Haiku 4.5 | llm | 79% (74%–84%) | 0.866 | 98% | 1.9 s | 4.0 s | $9.3 | 0 |
| Jev, one request per passage | classifier | 78% (73%–83%) | 0.862 | 98% | 244 ms | 350 ms | $0.64 | 0 |
| GPT-6.1 Sol | llm | 77% (72%–82%) | 0.857 | 99% | 2.2 s | 4.8 s | $14 | 0 |
| Jev | classifier | 77% (72%–82%) | 0.854 | 99% | 131 ms | 185 ms | $0.37 | 0 |
| GLM-5 | llm | 77% (72%–83%) | 0.853 | 98% | 2.7 s | 9.5 s | $6.1 | 0 |
| Grok 4.6 | llm | 76% (70%–81%) | 0.849 | 99% | 11.9 s | 40.9 s | $20 | 0 |
| Amazon Rerank 1.0 | reranker | 77% (71%–82%) | 0.849 | 97% | 435 ms | 1.1 s | not listed | 0 |
| GPT-6 Sol | llm | 75% (70%–80%) | 0.847 | 99% | 2.3 s | 2.9 s | $11 | 0 |
| Qwen3-Reranker 0.6B (local) | reranker | 75% (69%–80%) | 0.841 | 99% | 987 ms | 1.6 s | — | 0 |
| Qwen3 235B A22B 2507 | llm | 75% (69%–80%) | 0.832 | 97% | 3.6 s | 4.9 s | $1.2 | 0 |
| gpt-oss-120b | llm | 74% (68%–79%) | 0.830 | 96% | 9.7 s | 16.7 s | $1.2 | 0 |
| Llama 4 Maverick | llm | 73% (67%–78%) | 0.823 | 98% | 945 ms | 1.1 s | $1.2 | 0 |
| DeepSeek V3.2 | llm | 73% (67%–78%) | 0.817 | 97% | 5.9 s | 8.1 s | $3.2 | 3% |
| GPT-6 Luna | llm | 73% (67%–78%) | 0.804 | 93% | 1.5 s | 2.1 s | $0.52 | 11% |
| MiniMax M2.5 | llm | 71% (66%–77%) | 0.804 | 94% | 8.6 s | 69.5 s | $4.4 | 10% |
| Mistral Large 3 | llm | 66% (60%–72%) | 0.776 | 95% | 7.0 s | 156.8 s | $3.8 | 9% |
| Amazon Nova 2 Lite | llm | 62% (56%–68%) | 0.758 | 98% | 1.3 s | 1.8 s | $2.2 | 0 |
| Kev-4B (local) | classifier | 64% (58%–70%) | 0.752 | 94% | 5.4 s | 7.2 s | — | 0 |
| Laya (local) | classifier | 49% (43%–55%) | 0.646 | 90% | 646 ms | 1.3 s | — | 0 |
| No reranker (first-stage order) | baseline | 36% (30%–43%) | 0.515 | 80% | — | — | — | 0 |
<!-- leaderboard:miracl:end -->

### BRIGHT: StackOverflow

The same questions under two first stages: BM25 with embeddings (Stuga with **Embeddings** on), and BM25 alone.

#### BM25 + embeddings

<!-- leaderboard:bright-stackoverflow:start -->
Ranked on 61 questions whose answer was among the 24 candidates (56 retrieval misses left out). Intervals are bootstrap 95%.

| Ranker | Kind | Hit@1 | MRR | Recall@8 | p50 latency | p95 latency | $ / 1k searches | Unusable answers |
|---|---|---|---|---|---|---|---|---|
| GPT-6 Luna | llm | 43% (30%–56%) | 0.561 | 82% | 1.7 s | 2.1 s | $0.92 | 0 |
| Claude Opus 5.5 | llm | 39% (26%–51%) | 0.548 | 87% | 4.1 s | 5.2 s | $61 | 0 |
| Claude Sonnet 5.5 | llm | 38% (25%–49%) | 0.543 | 90% | 2.0 s | 3.6 s | $28 | 0 |
| Amazon Nova 2 Lite | llm | 36% (25%–48%) | 0.511 | 79% | 1.3 s | 1.5 s | $3.0 | 0 |
| GLM-5 | llm | 36% (25%–48%) | 0.509 | 79% | 2.7 s | 5.3 s | $7.3 | 0 |
| GPT-6.1 Sol | llm | 33% (21%–44%) | 0.505 | 80% | 2.2 s | 4.1 s | $20 | 0 |
| Llama 4 Maverick | llm | 36% (23%–48%) | 0.504 | 77% | 1.0 s | 1.2 s | $1.8 | 0 |
| Claude Sonnet 5 | llm | 34% (23%–46%) | 0.499 | 77% | 3.6 s | 12.1 s | $32 | 0 |
| No reranker (first-stage order) | baseline | 36% (25%–48%) | 0.495 | 72% | — | — | — | 0 |
| GPT-6 Sol | llm | 28% (16%–39%) | 0.489 | 85% | 2.5 s | 2.8 s | $18 | 0 |
| MiniMax M2.5 | llm | 34% (23%–46%) | 0.488 | 85% | 8.1 s | 12.3 s | $3.1 | 0 |
| Grok 4.6 | llm | 31% (20%–43%) | 0.476 | 79% | 12.0 s | 18.9 s | $27 | 0 |
| Claude Haiku 4.5 | llm | 33% (21%–44%) | 0.468 | 79% | 2.0 s | 2.1 s | $13 | 0 |
| Kev-4B (local) | classifier | 31% (20%–43%) | 0.467 | 69% | 9.3 s | 17.0 s | — | 0 |
| Jev, one request per passage | classifier | 31% (20%–43%) | 0.466 | 74% | 182 ms | 376 ms | $0.97 | 0 |
| Qwen3 235B A22B 2507 | llm | 33% (20%–44%) | 0.464 | 77% | 3.3 s | 4.6 s | $1.7 | 0 |
| Jev | classifier | 31% (20%–43%) | 0.459 | 79% | 130 ms | 175 ms | $0.40 | 0 |
| Mistral Large 3 | llm | 30% (18%–41%) | 0.453 | 79% | 7.1 s | 144.9 s | $5.1 | 8% |
| gpt-oss-120b | llm | 25% (13%–36%) | 0.444 | 79% | 7.6 s | 13.7 s | $1.5 | 0 |
| Qwen3-Reranker 4B (local) | reranker | 23% (13%–34%) | 0.411 | 82% | 11.3 s | 24.9 s | — | 0 |
| DeepSeek V3.2 | llm | 25% (13%–36%) | 0.408 | 69% | 6.1 s | 8.4 s | $4.8 | 3% |
| Laya (local) | classifier | 23% (13%–34%) | 0.375 | 61% | 2.6 s | 2.8 s | — | 0 |
| Qwen3-Reranker 0.6B (local) | reranker | 21% (11%–33%) | 0.365 | 64% | 2.1 s | 4.9 s | — | 0 |
| Cohere Rerank 3.5 | reranker | 13% (5%–21%) | 0.304 | 59% | 243 ms | 374 ms | $2.0 | 0 |
| Amazon Rerank 1.0 | reranker | 16% (8%–26%) | 0.295 | 48% | 590 ms | 728 ms | not listed | 0 |
<!-- leaderboard:bright-stackoverflow:end -->

#### BM25 only

<!-- leaderboard:bright-stackoverflow-bm25:start -->
Ranked on 47 questions whose answer was among the 24 candidates (70 retrieval misses left out). Intervals are bootstrap 95%.

| Ranker | Kind | Hit@1 | MRR | Recall@8 | p50 latency | p95 latency | $ / 1k searches | Unusable answers |
|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | llm | 57% (43%–72%) | 0.727 | 98% | 4.0 s | 5.4 s | $71 | 0 |
| GPT-6 Luna | llm | 55% (40%–70%) | 0.686 | 94% | 1.8 s | 2.2 s | $1.2 | 2% |
| Claude Sonnet 5 | llm | 53% (38%–68%) | 0.673 | 91% | 3.6 s | 7.2 s | $36 | 0 |
| Claude Sonnet 5.5 | llm | 51% (36%–66%) | 0.670 | 98% | 2.0 s | 3.7 s | $32 | 0 |
| GPT-6.1 Sol | llm | 51% (36%–66%) | 0.669 | 96% | 2.4 s | 4.5 s | $24 | 0 |
| GLM-5 | llm | 49% (34%–64%) | 0.654 | 96% | 3.0 s | 6.6 s | $8.7 | 0 |
| GPT-6 Sol | llm | 49% (34%–64%) | 0.652 | 98% | 2.6 s | 3.1 s | $21 | 0 |
| Grok 4.6 | llm | 47% (32%–62%) | 0.644 | 94% | 11.2 s | 19.8 s | $29 | 0 |
| Qwen3 235B A22B 2507 | llm | 51% (36%–66%) | 0.628 | 94% | 4.0 s | 4.9 s | $2.0 | 0 |
| Llama 4 Maverick | llm | 47% (32%–62%) | 0.623 | 94% | 1.1 s | 1.3 s | $2.1 | 0 |
| MiniMax M2.5 | llm | 47% (32%–62%) | 0.619 | 91% | 7.6 s | 9.4 s | $3.5 | 0 |
| gpt-oss-120b | llm | 40% (26%–55%) | 0.606 | 94% | 7.7 s | 14.3 s | $1.7 | 0 |
| Amazon Nova 2 Lite | llm | 47% (32%–62%) | 0.589 | 81% | 1.3 s | 1.8 s | $3.3 | 2% |
| Jev | classifier | 40% (26%–55%) | 0.589 | 91% | 148 ms | 253 ms | $0.46 | 0 |
| Claude Haiku 4.5 | llm | 40% (26%–55%) | 0.585 | 91% | 2.1 s | 2.2 s | $14 | 0 |
| Qwen3-Reranker 4B (local) | reranker | 40% (28%–55%) | 0.583 | 96% | 13.1 s | 28.4 s | — | 0 |
| Jev, one request per passage | classifier | 40% (28%–55%) | 0.576 | 91% | 205 ms | 355 ms | $1.1 | 0 |
| DeepSeek V3.2 | llm | 38% (26%–53%) | 0.570 | 91% | 6.0 s | 8.5 s | $5.6 | 2% |
| Mistral Large 3 | llm | 38% (23%–53%) | 0.554 | 89% | 7.0 s | 149.4 s | $5.8 | 9% |
| Kev-4B (local) | classifier | 38% (23%–53%) | 0.553 | 94% | 10.3 s | 18.7 s | — | 0 |
| No reranker (first-stage order) | baseline | 40% (26%–53%) | 0.537 | 81% | — | — | — | 0 |
| Qwen3-Reranker 0.6B (local) | reranker | 36% (21%–51%) | 0.534 | 91% | 2.5 s | 6.4 s | — | 0 |
| Amazon Rerank 1.0 | reranker | 36% (23%–51%) | 0.532 | 83% | 620 ms | 766 ms | not listed | 0 |
| Cohere Rerank 3.5 | reranker | 36% (23%–51%) | 0.528 | 94% | 262 ms | 421 ms | $2.0 | 0 |
| Laya (local) | classifier | 28% (15%–40%) | 0.416 | 72% | 2.8 s | 2.9 s | — | 0 |
<!-- leaderboard:bright-stackoverflow-bm25:end -->


`results/<set>/<date>/` has every call (`raw.jsonl`), what ran where and what each judge was sent
(`meta.json`), and the numbers above (`summary.json`). Release dates, with their sources, are in
[data/models.json](data/models.json).

`results/bright-stackoverflow/<date>/analysis.json`, from `scripts/analyze-bright.ts` with no model
calls, looks at why BRIGHT's first-stage order is hard to beat: passage length against the 1,200
characters a judge sees, BM25 over each question's candidates from whole passages and from those
characters, the two first stages over every question, and where each ranker kept, gained or lost a
gold passage in first place. `results/<set>/2026-09-28-excerpt/` holds every ranker on both BRIGHT
sets reading the 1,200 characters that best match the question instead.

## What is measured

Every question has 24 candidates, and every ranker ranks exactly the same list. A question whose
gold passages are all missing from its candidates is a retrieval miss: it is reported and left out
of the scores.

- **MIRACL** ([data/miracl/](data/miracl/SOURCES.md)): 240 questions over Wikipedia, 40 in each of
  English, Chinese, Japanese, Korean, Arabic and Thai.
  - Native speakers wrote the questions and judged the passages.
  - Each question's candidates are the first 24 of MMTEB's MIRACLReranking list, in that list's
    order.
- **BRIGHT StackOverflow** ([data/bright-stackoverflow/](data/bright-stackoverflow/SOURCES.md)):
  117 real StackOverflow questions.
  - Their gold passages are the documentation pages that accepted answers link to.
  - The candidates are retrieved from the question alone over the subset's 107,081 passages, in
    Stuga's two configurations: with embeddings, BM25 and `text-embedding-3-large`
    similarity fused by reciprocal rank; without it
    ([data/bright-stackoverflow-bm25/](data/bright-stackoverflow-bm25/SOURCES.md)), BM25 alone.
- **Judges:** every ranker sees the same text per passage: its title, heading path and 1,200
  characters of its text. The leaderboards above used the first 1,200 (`--snippet start`). Stuga's
  judges now read the 1,200 that best match the query ([src/excerpt.ts](src/excerpt.ts), copied from
  Stuga; `--snippet excerpt`, the default). `meta.json` says which a run used.
  - LLMs get Stuga's prompt and settings ([src/stuga.ts](src/stuga.ts)): score each passage 0–10
    as a JSON array, at the lowest reasoning the model offers (`low` for GPT-6.1 Sol, `none` for GPT-6 Sol), in up to 8,192 output tokens. An
    answer that does not parse leaves the first-stage order, as in Stuga. `meta.json` records what
    each judge was sent.
  - Jev gets Stuga's System One request, the query and all 24 passages as state with one yes/no
    question per passage, and also one request per passage.
  - Kev and Laya, System One servers on this machine, get one passage per request; Laya's state is
    plain text.
  - Rerankers score each (query, passage) pair.
- **Metrics:**
  - Hit@1, MRR and Recall@8 (Stuga's Ask reads eight passages) on the questions whose candidates
    hold a gold passage, with bootstrap 95% intervals over questions.
  - Comparisons between rankers are paired: the difference on the same questions, with a paired
    bootstrap interval, against Jev (`vsJev`) and against the first-stage order (`vsFirstStage`) in
    `summary.json`. An interval that includes zero means the test cannot tell the two apart, not
    that they are equivalent.
  - Candidate recall (the share of questions with a gold passage among the candidates) and Hit@1
    over all questions, a retrieval miss counting as a miss.
  - LLM judges: each BRIGHT row keeps the answer's text and how many candidates it scored
    (`covered`); MIRACL rows keep the text of unusable answers only.
  - Latency is the wall-clock from the client, retries and their waits included (`totalMs`), one
    request at a time per ranker. A call whose prompt the vendor served mostly from cache is left
    out of latency.
  - Cost is measured tokens at list price, cached and cache-written input at their own rates, or the
    per-query price.

## Run it

Node 26 and pnpm 12.

```sh
pnpm install
cat > keys.json <<'EOF'
{ "openai": "…", "typesafe": "…", "bedrock": { "region": "us-west-2", "key": "…" } }
EOF
BENCH_KEYS=keys.json node src/run.ts --set miracl --only jev,gpt-6-sol --limit 5   # a quick look
BENCH_KEYS=keys.json node src/run.ts --set miracl --rankable-only                  # everything with a key
BENCH_KEYS=keys.json node src/run.ts --set miracl --snippet start                  # the first 1,200 characters
node src/report.ts results/miracl/<date> --readme
```

Keys come from the file `BENCH_KEYS` names, never from the command line. A ranker without a key is
left out. The local rerankers expect `llama-server --rerank` on ports 8089 (Qwen3-Reranker 4B)
and 8090 (0.6B):

```sh
llama-server -m Qwen3-Reranker-4B.Q8_0.gguf --rerank --port 8089 -ngl 999 -fa on -c 32768 --parallel 4 -b 4096 -ub 4096
```

Kev and Laya serve the System One API locally; `src/lineup.ts` expects them on ports 8009 and 8010:

```sh
# github.com/jaredpalmer/kev
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
# pip install "laya[serve]"; set the host, or it listens on every interface
LAYA_HOST=127.0.0.1 LAYA_PORT=8010 LAYA_DEVICE=mps laya-serve
```

Every set's data and every call's result are committed (see Metrics for which rows keep the
answer's text). The BRIGHT embeddings (about 440 MB)
are not: the committed candidate lists are all a reproduction needs. `node scripts/build-miracl.ts` and
`BENCH_KEYS=keys.json node --max-old-space-size=8192 scripts/build-bright.ts` rebuild the sets from
pinned dataset revisions; the second embeds BRIGHT's corpus with OpenAI.

## Add a ranker

A ranker is `rank(query, passages) → scores`, one score per passage (see [src/types.ts](src/types.ts)).
Add it to [src/lineup.ts](src/lineup.ts), run it, and open a pull request with its `results/`.
A server that speaks the System One API (`POST /v1/systemone`) needs no code: point `systemOne()`
at its base URL.

## Licence

Code MIT. Data: each set's `SOURCES.md`. MIRACL passages are CC BY-SA 4.0 and BRIGHT is CC BY 4.0.
