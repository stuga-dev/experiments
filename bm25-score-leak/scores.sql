-- BM25 scores count documents the searcher cannot read.
-- Postgres 18 with pg_search 0.25.9; run with psql -f on a throwaway database.
-- Each query is the plain shape an app runs: a match plus a permission filter, nothing else.
-- (pg_search folds extra predicates on the indexed column into the score, so keep them out.)
\set ON_ERROR_STOP 1
CREATE EXTENSION IF NOT EXISTS pg_search CASCADE;
DROP TABLE IF EXISTS docs;
CREATE TABLE docs (id serial PRIMARY KEY, body text, readers text[]);
CREATE INDEX docs_bm25 ON docs USING bm25 (id, body) WITH (key_field = 'id');

-- 200 ordinary documents everyone can read.
INSERT INTO docs (body, readers)
SELECT 'weekly update on the office move, budget and hiring plan number ' || g, ARRAY['everyone']
FROM generate_series(1, 200) g;

-- Liv's probe note (id 201): layoff and two made-up words, once each. 'marker2' is also in a second note of hers,
-- so she knows it is in exactly two documents: a reference marker.
INSERT INTO docs (body, readers) VALUES ('layoff marker1 marker2', ARRAY['liv']), ('marker2', ARRAY['liv']);

CREATE TEMP TABLE probe (step text, word text, score numeric);
CREATE FUNCTION pg_temp.probe(step text) RETURNS void LANGUAGE sql AS $$
  -- Liv's search for each word; the probe note is picked outside the search (OFFSET 0 fences it).
  INSERT INTO probe
  SELECT step, word, round(score::numeric, 2) FROM (
    SELECT 'layoff' AS word, id, pdb.score(id) AS score FROM docs
     WHERE body @@@ 'layoff' AND readers && ARRAY['liv', 'everyone'] OFFSET 0) a
   WHERE id = 201
  UNION ALL
  SELECT step, word, round(score::numeric, 2) FROM (
    SELECT 'marker1' AS word, id, pdb.score(id) AS score FROM docs
     WHERE body @@@ 'marker1' AND readers && ARRAY['liv', 'everyone'] OFFSET 0) b
   WHERE id = 201
  UNION ALL
  SELECT step, word, round(score::numeric, 2) FROM (
    SELECT 'marker2' AS word, id, pdb.score(id) AS score FROM docs
     WHERE body @@@ 'marker2' AND readers && ARRAY['liv', 'everyone'] OFFSET 0) c
   WHERE id = 201
$$;

SELECT pg_temp.probe('1. no hidden document');

-- HR writes in a folder Liv cannot open.
INSERT INTO docs (body, readers) VALUES ('q2 layoff: restructuring plan, roles affected in q2', ARRAY['hr']);
SELECT pg_temp.probe('2. one hidden document');

INSERT INTO docs (body, readers) SELECT 'layoff notes ' || g, ARRAY['hr'] FROM generate_series(1, 4) g;
SELECT pg_temp.probe('3. five hidden documents');

-- What Liv sees: her own note, every time.
SELECT step, word, score FROM probe ORDER BY step, word;

-- What she can work out. BM25's IDF is ln(1 + (N - n + 0.5) / (n + 0.5)); in one note the
-- other factors are equal, so score ratios are IDF ratios. marker2 (n = 2) against marker1
-- (n = 1) gives N, and layoff against marker1 then gives n.
WITH s AS (
  SELECT max(score) FILTER (WHERE word = 'layoff') k, max(score) FILTER (WHERE word = 'marker1') p,
         max(score) FILTER (WHERE word = 'marker2') w
  FROM probe WHERE step LIKE '3.%'
), nn AS (
  SELECT k, p, (SELECT n FROM generate_series(3, 100000) n
                ORDER BY abs(ln(1 + (n - 1.5) / 2.5) / ln(1 + (n - 0.5) / 1.5) - w / p) LIMIT 1) AS big_n
  FROM s
)
SELECT big_n AS documents_in_index,
       (SELECT n FROM generate_series(1, 1000) n
        ORDER BY abs(ln(1 + (big_n - n + 0.5) / (n + 0.5)) / ln(1 + (big_n - 0.5) / 1.5) - k / p) LIMIT 1)
         AS documents_holding_layoff,
       (SELECT count(*) FROM docs WHERE body @@@ 'layoff' AND readers && ARRAY['liv', 'everyone'])
         AS layoff_documents_liv_can_read
FROM nn;

-- Output on Postgres 18.6 with pg_search 0.25.9 (2026-10-02):
--
--            step           |  word   | score
--  --------------------------+---------+-------
--   1. no hidden document    | layoff  |  7.07
--   1. no hidden document    | marker1 |  7.07
--   1. no hidden document    | marker2 |  6.34
--   2. one hidden document   | layoff  |  6.34
--   2. one hidden document   | marker1 |  7.08
--   2. one hidden document   | marker2 |  6.34
--   3. five hidden documents | layoff  |  4.98
--   3. five hidden documents | marker1 |  7.09
--   3. five hidden documents | marker2 |  6.35
--
--   documents_in_index | documents_holding_layoff | layoff_documents_liv_can_read
--  --------------------+--------------------------+-------------------------------
--                  199 |                        6 |                             1
--
-- The index really holds 207 documents; from scores rounded to two decimals the fit lands on
-- 199, close enough that layoff's count still comes out at exactly 6.
