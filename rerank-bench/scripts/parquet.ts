/** Download a Hugging Face dataset file once into .cache/ and read a parquet file as row objects. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { parquetReadObjects } from "hyparquet";

/** `dataset` at a pinned `revision`, so a rebuild reads the same bytes. */
export async function hfFile(dataset: string, revision: string, path: string): Promise<string> {
  const local = `.cache/hf/${dataset}/${revision}/${path}`;
  if (!existsSync(local)) {
    const res = await fetch(`https://huggingface.co/datasets/${dataset}/resolve/${revision}/${path}`);
    if (!res.ok) throw new Error(`${dataset}/${path}: HTTP ${res.status}`);
    mkdirSync(dirname(local), { recursive: true });
    writeFileSync(local, Buffer.from(await res.arrayBuffer()));
  }
  return local;
}

export async function readParquet<T>(path: string): Promise<T[]> {
  const b = readFileSync(path);
  const rows = await parquetReadObjects({ file: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer });
  return rows as T[];
}
