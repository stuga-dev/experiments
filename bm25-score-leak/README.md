# BM25 scores count documents you cannot read

BM25 weighs a word by how rare it is across the whole index. A permission filter decides which rows
come back; it does not change that count. So a searcher who plants words of known count in a note of
their own can read, from the scores on that note, how many hidden documents hold another word.

- `scores.sql`: the scores, and the count worked out from them (6 documents, 1 readable).
- `order.sql`: with scores hidden, the order of two results still moves with hidden documents; a bound from this one comparison.

Postgres 18 with pg_search 0.25.9 (`shared_preload_libraries = 'pg_search'`):

```sh
psql -d throwaway -f scores.sql
psql -d throwaway -f order.sql
```

Each file ends with the output it gave on 2026-10-01 and 2026-10-02.
