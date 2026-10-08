"""
The reference vectors: a model through sentence-transformers at bf16 on the Mac's GPU, with the
same prompts Stuga sends, in embed-bench's cache format, so src/eval.ts scores them as `st:<model>`.

    node scripts/dump-texts.ts ko
    st-env/bin/python -I scripts/st-embed.py google/embeddinggemma-2 miracl-ko <bench dir>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

model_id, set_name, bench = sys.argv[1], sys.argv[2], sys.argv[3]
model = SentenceTransformer(model_id, device="mps", model_kwargs={"torch_dtype": torch.bfloat16})
# Ollama runs this model with a 2,048-token context and cuts longer input there; the reference does too.
model.max_seq_length = int(os.environ.get("MAX_SEQ", "2048"))
out_dir = os.path.join(bench, ".cache/vectors/st", model_id.replace("/", "_"))
os.makedirs(out_dir, exist_ok=True)
for role, part, prompt in (("query", "queries", "query"), ("document", "passages", "document")):
    texts = json.load(open(os.path.join(bench, f".cache/texts/{set_name}-{part}.json")))
    started = time.time()
    v = model.encode(texts, prompt_name=prompt, batch_size=int(os.environ.get("BATCH", "8")), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    seconds = time.time() - started
    v = v.astype(np.float32)
    path = os.path.join(out_dir, f"{set_name}-{part}")
    v.tofile(path + ".f32")
    json.dump(
        {"dims": v.shape[1], "count": len(texts), "done": list(range((len(texts) + 63) // 64)), "tokens": 0, "seconds": seconds,
         "model": model_id, "requested": None, "prompt": model.prompts[prompt]},
        open(path + ".json", "w"),
    )
    print(f"{set_name} {part}: {len(texts)} in {seconds:.0f}s, {v.shape[1]} dims, prompt {model.prompts[prompt]!r}")
