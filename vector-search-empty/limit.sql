-- The iterative scan has budgets of its own: hnsw.max_scan_tuples (20,000 by default) and its scan
-- memory (hnsw.scan_mem_multiplier x work_mem). A big enough crowd of unreadable passages near the
-- query can use them up before the scan reaches the readable ones.
-- Run after setup.sql. Adds 30,000 HR passages: 33,000 in all.
\set ON_ERROR_STOP 1
\timing on
INSERT INTO chunks (topic, readers, embedding)
SELECT 'hr restructuring', ARRAY['hr'], near(0.6) FROM generate_series(1, 30000);

\echo '--- 7. Iterative scan (strict order), cutoff outside, 33,000 HR passages in the way'
BEGIN;
SET LOCAL hnsw.iterative_scan = strict_order;
SELECT count(*) FROM (
  SELECT c.embedding <=> (SELECT q FROM query) AS d FROM chunks c
   WHERE c.readers && ARRAY['liv', 'everyone']
   ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 10) near WHERE d < 0.6;
COMMIT;

\echo '--- 8. The same with hnsw.max_scan_tuples raised to 100,000'
BEGIN;
SET LOCAL hnsw.iterative_scan = strict_order;
SET LOCAL hnsw.max_scan_tuples = 100000;
SELECT count(*) FROM (
  SELECT c.embedding <=> (SELECT q FROM query) AS d FROM chunks c
   WHERE c.readers && ARRAY['liv', 'everyone']
   ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 10) near WHERE d < 0.6;
COMMIT;

\echo '--- 8b. And its memory, hnsw.scan_mem_multiplier (a multiple of work_mem, 4MB here), raised to 8'
BEGIN;
SET LOCAL hnsw.iterative_scan = strict_order;
SET LOCAL hnsw.max_scan_tuples = 100000;
SET LOCAL hnsw.scan_mem_multiplier = 8;
SELECT count(*) FROM (
  SELECT c.embedding <=> (SELECT q FROM query) AS d FROM chunks c
   WHERE c.readers && ARRAY['liv', 'everyone']
   ORDER BY c.embedding <=> (SELECT q FROM query) LIMIT 10) near WHERE d < 0.6;
COMMIT;

\echo '--- 9. Exact scan, for the truth'
BEGIN;
SET LOCAL enable_indexscan = off;
SELECT count(*) FROM chunks c, query
 WHERE c.readers && ARRAY['liv', 'everyone'] AND (c.embedding <=> query.q) < 0.6;
COMMIT;

\echo '--- 10. Where the first passage Liv can read sits, by exact distance'
BEGIN;
SET LOCAL enable_indexscan = off;
SELECT min(r) FROM (SELECT readers, row_number() OVER (ORDER BY embedding <=> (SELECT q FROM query)) r FROM chunks) x
 WHERE readers && ARRAY['liv', 'everyone'];
COMMIT;

-- Output on Postgres 18.6 with pgvector 0.8.6 (2026-10-02):
--   7. iterative scan, defaults: 0
--   8. max_scan_tuples 100,000: 0 (the scan's memory, 1 x work_mem = 4MB, stops it first)
--   8b. max_scan_tuples 100,000 and scan_mem_multiplier 8: 5, about 68 ms
--   9. exact scan: 5, about 23 ms
--   10. first readable passage: 33,001st nearest
