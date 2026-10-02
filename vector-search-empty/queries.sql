-- Liv's semantic search, three ways. Run after setup.sql.
\set ON_ERROR_STOP 1
\timing on

\echo '--- 0. The truth, by exact scan: passages Liv can read within the cutoff (distance < 0.6)'
BEGIN;
SET LOCAL enable_indexscan = off;
SELECT topic, count(*) FROM chunks c, query
 WHERE c.readers && ARRAY['liv', 'everyone'] AND (c.embedding <=> query.q) < 0.6 GROUP BY topic;
COMMIT;

\echo '--- 1. HNSW as it comes: nearest 10 that Liv can read'
SELECT c.topic, round((c.embedding <=> query.q)::numeric, 3) AS distance
  FROM chunks c, query
 WHERE c.readers && ARRAY['liv', 'everyone']
 ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 10;

\echo '--- 2. Iterative scan, cutoff outside the ordered scan'
BEGIN;
SET LOCAL hnsw.iterative_scan = strict_order;
SELECT topic, distance FROM (
  SELECT c.topic, round((c.embedding <=> query.q)::numeric, 3) AS distance
    FROM chunks c, query
   WHERE c.readers && ARRAY['liv', 'everyone']
   ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 10) near
 WHERE distance < 0.6;
COMMIT;

\echo '--- 3. Iterative scan, cutoff inside the ordered scan'
BEGIN;
SET LOCAL hnsw.iterative_scan = strict_order;
SELECT c.topic, round((c.embedding <=> query.q)::numeric, 3) AS distance
  FROM chunks c, query
 WHERE c.readers && ARRAY['liv', 'everyone'] AND (c.embedding <=> query.q) < 0.6
 ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 10;
COMMIT;

\echo '--- 4. Without iterative scan, ef_search at its maximum (1000)'
BEGIN; SET LOCAL hnsw.ef_search = 1000;
SELECT count(*) FROM (SELECT c.id FROM chunks c WHERE c.readers && ARRAY['liv','everyone']
  ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 10) x;
COMMIT;
\echo '--- 5. ef_search 320, LIMIT 320 as in the Stuga search box, cutoff outside'
BEGIN; SET LOCAL hnsw.iterative_scan = strict_order; SET LOCAL hnsw.ef_search = 320;
SELECT count(*) FROM (SELECT c.id, c.embedding <=> (SELECT q FROM query) AS d FROM chunks c
  WHERE c.readers && ARRAY['liv','everyone'] ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 320) n WHERE d < 0.6;
COMMIT;
\echo '--- 6. ef_search 320, LIMIT 320 as in the Stuga search box, cutoff inside'
BEGIN; SET LOCAL hnsw.iterative_scan = strict_order; SET LOCAL hnsw.ef_search = 320;
SELECT count(*) FROM (SELECT c.id FROM chunks c
  WHERE c.readers && ARRAY['liv','everyone'] AND (c.embedding <=> (SELECT q FROM query)) < 0.6
  ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 320) n;
COMMIT;
SHOW hnsw.max_scan_tuples;

-- Output on Postgres 18.6 with pgvector 0.8.6 (2026-10-02; Apple silicon, timings vary by machine):
--   0. exact scan: team reorg notes | 5
--   1. HNSW as it comes (ef_search 40): 0 rows
--   2. iterative scan, cutoff outside: the 5 team reorg notes, distance 0.360 to 0.407, about 5 ms
--   3. iterative scan, cutoff inside: the same 5, about 32 ms (walks on until a scan budget stops it)
--   4. no iterative scan, ef_search 1000: 0 rows
--   5. ef_search 320, LIMIT 320, cutoff outside: 5, about 31 ms
--   6. ef_search 320, LIMIT 320, cutoff inside: 5, about 54 ms
-- Queries 5 and 6 are from a second run the same day, on the same machine, with Stuga's search-box
-- setting (LIMIT 320); in that run query 2 took about 6 ms and query 3 about 52 ms.
-- HR's 3,000 passages are all nearer the query than any of Liv's. In this test, a scan that stops
-- at ef_search candidates (at most 1,000) and then filters keeps none.
