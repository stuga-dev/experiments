// Stuga's own search against a crowd of unreadable passages. Copy into stuga/packages/db/src/ and run:
//   TEST_DATABASE_URL=postgres://… pnpm --filter @stuga/db exec vitest run src/stuga-probe.integration.test.ts
// Use a throwaway database with pgvector and a preloaded pg_search: the probe runs TRUNCATE docs CASCADE.
// One workspace: Bob's CROWD chunks sit nearest the query; Liv reads 20,000 far chunks and 5 near.
// The data stays in the database between runs (drop table probe_query to rebuild), so the same data
// can be searched before and after a change to search.ts.
import { describe, it, beforeAll, afterAll } from "vitest";
import { createClient, closeClients } from "./client.js";
import { seedWorkspaces } from "./testing/fixtures.js";
import { initSearchSchema } from "./testing/search-schema.js";
import { createDoc } from "./docs.js";
import { searchDocs, askDocs } from "./search.js";
import type { Sql } from "./client.js";
import { EMBEDDING_DIMS } from "@stuga/protocol/domain/limits";

const URL = process.env.TEST_DATABASE_URL!;
const CROWD = Number(process.env.CROWD ?? 33000);

describe("Stuga's search behind a crowd", () => {
  let sql: Sql;
  beforeAll(async () => {
    sql = createClient(URL);
    const [{ ready }] = await sql<{ ready: boolean }[]>`SELECT to_regclass('probe_query') IS NOT NULL AS ready`;
    if (ready) return;
    await initSearchSchema(sql);
    await sql`TRUNCATE docs CASCADE`;
    await seedWorkspaces(sql, "ws");
    await createDoc(sql, { docId: "bobs", workspaceId: "ws", owner: "user:bob", title: "Bob", aclPrincipals: ["user:bob"] });
    await createDoc(sql, { docId: "far", workspaceId: "ws", owner: "user:liv", title: "Far", aclPrincipals: ["user:liv"] });
    await createDoc(sql, { docId: "near", workspaceId: "ws", owner: "user:liv", title: "Near", aclPrincipals: ["user:liv"] });
    await sql.unsafe(`
      CREATE TEMP TABLE center AS SELECT array_agg(random() - 0.5) AS c FROM generate_series(1, ${EMBEDDING_DIMS});
      INSERT INTO doc_chunks (doc_id, workspace_id, chunk_index, content, embedding)
      SELECT 'bobs', 'ws', g, 'crowd ' || g,
             (SELECT array_agg(c[i] + 0.6 * (random() - 0.5) ORDER BY i) FROM generate_series(1, ${EMBEDDING_DIMS}) i)::vector
      FROM generate_series(1, ${CROWD}) g, center;
      -- WHERE g > 0 ties each far vector to its row; uncorrelated, Postgres would draw one and reuse it.
      INSERT INTO doc_chunks (doc_id, workspace_id, chunk_index, content, embedding)
      SELECT 'far', 'ws', g, 'far ' || g, (SELECT array_agg(random() - 0.5) FROM generate_series(1, ${EMBEDDING_DIMS}) WHERE g > 0)::vector
      FROM generate_series(1, 20000) g;
      INSERT INTO doc_chunks (doc_id, workspace_id, chunk_index, content, embedding)
      SELECT 'near', 'ws', g, 'team note ' || g,
             (SELECT array_agg(c[i] + 1.2 * (random() - 0.5) ORDER BY i) FROM generate_series(1, ${EMBEDDING_DIMS}) i)::vector
      FROM generate_series(1, 5) g, center;
      CREATE TABLE probe_query AS SELECT (SELECT array_agg(c[i] + 0.3 * (random() - 0.5) ORDER BY i) FROM generate_series(1, ${EMBEDDING_DIMS}) i) AS q FROM center;
      ANALYZE doc_chunks; ANALYZE docs;`);
  }, 1_800_000);
  afterAll(async () => { await closeClients(); });

  it("searches as Liv", async () => {
    const [{ q }] = await sql<{ q: number[] }[]>`SELECT q FROM probe_query`;
    const input = { embeddingDims: EMBEDDING_DIMS, maxDistance: 0.9, workspaceId: "ws", principals: ["user:liv"], query: "zzqq", queryEmbedding: q };
    const t0 = performance.now();
    const docs = await searchDocs(sql, input);
    const t1 = performance.now();
    // Ask retrieves its candidates with limit 24 (services/node/src/retrieval/retrieve.ts, CANDIDATE_LIMIT),
    // then reranks them and prefers at most 3 per document, backfilling to its top 8; this measures the retrieval.
    const passages = await askDocs(sql, { ...input, limit: 24 });
    const t2 = performance.now();
    process.stdout.write("PROBE " + JSON.stringify({ crowd: CROWD, docs: docs.map((r) => r.doc_id), docsMs: Math.round(t1 - t0),
      nearPassages: passages.filter((p) => p.doc_id === "near").length, askMs: Math.round(t2 - t1) }) + "\n");
  }, 600_000);
});

// Output on Postgres 18.6, pgvector 0.8.6, 2026-10-02 (Apple silicon), far vectors drawn per row,
// passages retrieved with Ask's limit of 24 (before Ask reranks them and prefers at most 3 per document, backfilling to its top 8):
//   before the fix (stuga 0c0a0b3, release 0.1.10 plus a README change):
//     {"crowd":33000,"docs":["far"],"docsMs":52,"nearPassages":0,"askMs":31}
//   after the fix, released in 0.1.11 (second run; the first, on a cold cache, took 922 ms for search):
//     {"crowd":33000,"docs":["near","far"],"docsMs":94,"nearPassages":5,"askMs":67}
//   (far's nearest chunk is at cosine distance 0.887, inside the probe's 0.9 cutoff; Bob's crowd at 0.147)
//   semanticScan, the count taken first: about 1 ms ("index": Liv reads more than 5,000 chunks)
