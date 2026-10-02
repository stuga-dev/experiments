-- Filtered vector search on pgvector's HNSW index: a test corpus.
-- Postgres 18 with pgvector 0.8.6. Vectors are synthetic (384 dimensions); what matters is where
-- they sit: HR's passages crowd the query's neighbourhood, Liv's few relevant ones sit further out.
\set ON_ERROR_STOP 1
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS chunks, center, query;
CREATE TABLE chunks (id serial PRIMARY KEY, topic text, readers text[], embedding vector(384));
SELECT setseed(0.42);

CREATE TABLE center AS SELECT array_agg(random() - 0.5) AS c FROM generate_series(1, 384);
CREATE OR REPLACE FUNCTION near(spread float8) RETURNS vector LANGUAGE sql AS $$
  SELECT (SELECT array_agg(c[i] + spread * (random() - 0.5) ORDER BY i)
            FROM generate_series(1, 384) i)::vector FROM center
$$;
CREATE OR REPLACE FUNCTION anywhere() RETURNS vector LANGUAGE sql AS $$
  SELECT (SELECT array_agg(random() - 0.5) FROM generate_series(1, 384))::vector
$$;

-- 97,000 passages on other topics, readable by everyone.
INSERT INTO chunks (topic, readers, embedding)
SELECT 'other', ARRAY['everyone'], anywhere() FROM generate_series(1, 97000);
-- 3,000 HR passages on the restructuring, close to the query. Liv cannot read them.
INSERT INTO chunks (topic, readers, embedding)
SELECT 'hr restructuring', ARRAY['hr'], near(0.6) FROM generate_series(1, 3000);
-- 5 passages of Liv's team on the same subject, a little further out.
INSERT INTO chunks (topic, readers, embedding)
SELECT 'team reorg notes', ARRAY['liv'], near(1.2) FROM generate_series(1, 5);

CREATE TABLE query AS SELECT near(0.3) AS q;
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
ANALYZE chunks;
