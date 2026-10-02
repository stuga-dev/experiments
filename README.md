# Experiments

Reproducible tests behind Stuga's videos and posts: the code, its captured output, and what it
shows. Each folder runs on its own.

| Folder | Shows | Video |
|---|---|---|
| [rerank-bench](rerank-bench/) | 23 rerankers on the same candidates from MIRACL and BRIGHT StackOverflow: hit rate, latency and cost. | Claude Sonnet 5.5 vs Jev |
| [bm25-score-leak](bm25-score-leak/) | Raw BM25 scores over a shared index let a searcher estimate how many documents they cannot read contain a word. | Under the hood #1 |
| [vector-search-empty](vector-search-empty/) | HNSW search with a permission filter can return nothing when readable matches exist, and how iterative scans and an exact scan fix it. | Under the hood #2 |

Code MIT. Data under each source's licence; names in the SQL tests are made up.
