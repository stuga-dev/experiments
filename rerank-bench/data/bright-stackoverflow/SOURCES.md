# BRIGHT, StackOverflow

BRIGHT [1] is a benchmark for retrieval that needs reasoning. In its StackOverflow subset:

- The questions are real StackOverflow posts, often with code.
- The gold passages are pages of documentation that an accepted answer links to, confirmed by
  annotators.

This set uses [xlangai/BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) at revision
`3066d29c9651a576c8aba4832d249807b181ecae`: all 117 questions and the subset's 107,081 passages.

`scripts/build-bright.ts` builds each question's 24 candidates the way Stuga retrieves, from the
question alone:

- **Two lists of 96**, fused by reciprocal rank (k = 60):
  - BM25 (k1 = 0.9, b = 0.4), BRIGHT's own baseline;
  - cosine similarity of OpenAI's `text-embedding-3-large` at 1,024 dimensions, Stuga's default.
- **Excluded passages:** those BRIGHT excludes for a question are left out.
- **Retrieval misses:** a question whose gold passages are all missing from its 24 is reported as a
  miss.

The embeddings are not committed: at 1,024 floats for each of 107,081 passages they come to about
440 MB, and reproducing the rankings needs only the committed candidate lists. Rebuilding the lists
embeds the corpus again, about 46 million tokens, and needs an OpenAI key in `BENCH_KEYS`.

## Licence

BRIGHT is CC BY 4.0. The documentation passages keep their sources' terms, as BRIGHT redistributes
them.

[1] H. Su et al. BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
arXiv:2407.12883, 2024.
