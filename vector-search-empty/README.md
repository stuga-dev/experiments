# Vector search with permissions comes back empty

pgvector's HNSW scan picks its nearest candidates before a permission filter runs. When the passages
nearest the question belong to documents the searcher cannot read, the filter leaves nothing.
Iterative scans keep walking, within two budgets: `hnsw.max_scan_tuples` and
`hnsw.scan_mem_multiplier` (a multiple of `work_mem`).

- `setup.sql`: 100,000 synthetic passages (384 dimensions); 3,000 unreadable ones crowd the query, 5 readable ones sit further out.
- `queries.sql`: plain HNSW (0), `ef_search` 1,000 (0), iterative scan (5), and where a distance cutoff belongs.
- `limit.sql`: 30,000 more unreadable passages: the iterative scan returns 0 until both budgets are raised.
- `stuga-probe.integration.test.ts`: the same crowd through Stuga's own search, before and after its fix.

Postgres 18 with pgvector 0.8.6:

```sh
psql -d throwaway -f setup.sql
psql -d throwaway -f queries.sql
psql -d throwaway -f limit.sql
```

Timings come from one Apple-silicon machine and vary by hardware; counts do not. Each file ends with
its output from 2026-10-02.
