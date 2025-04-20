import modal
from pathlib import Path
import os
import torch
from tqdm.auto import tqdm
import numpy as np
import tiktoken
import datasets
import random
from collections import deque
import math
import glob
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from tqdm.auto import tqdm


image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasets",
    "huggingface_hub",
    "tiktoken",
    "tqdm",
    "torch"
)
app = modal.App("tokenize")

@app.function(
    secrets=[modal.Secret.from_name("HF-SECRET")],
    image=image,
    timeout=60*30
)
def tokenize_dclm(worker_id: int = 0):
    # first download the parquet for this worker
    import huggingface_hub
    huggingface_hub.snapshot_download(
        'TaylorAI/dclm_subset_1pct',
        repo_type="dataset",
        local_dir="/data",
        allow_patterns=[f"*{worker_id}.parquet.zst"]
    )
    encoding = tiktoken.encoding_for_model("gpt-4")
    big_path = Path("all_tokens.bin")
    big_path.unlink(missing_ok=True)
    eot_token = encoding._special_tokens['<|endoftext|>']
    buf_cap: int = 1 << 20
    buffer = []
    total = 0
    with open(big_path, "ab") as f:
        dataset = ds.dataset("/data", format="parquet")
        for batch in tqdm(dataset.to_batches(columns=["text"], batch_size=1_024), desc="Tokenizing"):
            texts = [val.as_py() for val in batch.column("text")]
            encoded = encoding.encode_ordinary_batch(texts)
            for tokens in encoded:
                buffer.extend(tokens)
                buffer.append(eot_token)
            if len(buffer) >= buf_cap:
                np.array(buffer, dtype="int32").tofile(f)
                total += len(buffer)
                buffer.clear()

        if buffer:
            np.array(buffer, dtype="int32").tofile(f)
            total += len(buffer)

    print(f"Total tokens written: {total:,}")
    huggingface_hub.upload_file(
        path_or_fileobj="all_tokens.bin",
        repo_id="andersonbcdefg/dclm-pretokenized",
        path_in_repo=f"tokens_shard_{worker_id}.bin",
        repo_type="dataset"
    )
