-- With raw scores hidden, the order of results still moves with documents the searcher cannot read.
-- Postgres 18 with pg_search 0.25.9; run with psql -f on a throwaway database.
\set ON_ERROR_STOP 1
CREATE EXTENSION IF NOT EXISTS pg_search CASCADE;
DROP TABLE IF EXISTS docs;
CREATE TABLE docs (id serial PRIMARY KEY, body text, readers text[]);
CREATE INDEX docs_bm25 ON docs USING bm25 (id, body) WITH (key_field = 'id');

INSERT INTO docs (body, readers)
SELECT 'weekly update on the office move, budget and hiring plan number ' || g, ARRAY['everyone']
FROM generate_series(1, 200) g;

-- Two notes of Liv's, alike in length: one holds layoff, the other marker2.
-- marker2 is also in a third note of hers, so it is in exactly two documents.
INSERT INTO docs (body, readers) VALUES
  ('note layoff', ARRAY['liv']),   -- id 201
  ('note marker2', ARRAY['liv']),   -- id 202
  ('marker2', ARRAY['liv']);        -- id 203

-- Liv's search: either word, readable documents only, best first. She sees the order, not the scores.
CREATE TEMP TABLE seen (step text, pos int, note text);
CREATE FUNCTION pg_temp.look(step text) RETURNS void LANGUAGE sql AS $$
  INSERT INTO seen
  SELECT step, row_number() OVER (ORDER BY score DESC, id), body FROM (
    SELECT id, body, pdb.score(id) AS score FROM docs
     WHERE body @@@ 'layoff OR marker2' AND readers && ARRAY['liv', 'everyone'] OFFSET 0) r
   WHERE id IN (201, 202)
$$;

SELECT pg_temp.look('1. no hidden document');
INSERT INTO docs (body, readers) VALUES ('q2 layoff: restructuring plan', ARRAY['hr']);
SELECT pg_temp.look('2. one hidden document');
INSERT INTO docs (body, readers) VALUES ('layoff notes 2', ARRAY['hr']);
SELECT pg_temp.look('3. two hidden documents');

SELECT step, pos, note FROM seen ORDER BY step, pos;

-- Output on Postgres 18.6 with pg_search 0.25.9 (2026-10-02):
--
--            step          | pos |     note
--  -------------------------+-----+--------------
--   1. no hidden document   |   1 | note layoff
--   1. no hidden document   |   2 | note marker2
--   2. one hidden document  |   1 | note layoff
--   2. one hidden document  |   2 | note marker2
--   3. two hidden documents |   1 | note marker2
--   3. two hidden documents |   2 | note layoff
--
-- Step 2 is a tie (layoff and marker2 each in two documents), broken here by id; the video shows
-- steps 1 and 3 only. Layoff's note above marker2's: layoff is in at most two documents.
-- Below it: layoff is in more than two, so at least two are hidden. A bound from this one comparison.
