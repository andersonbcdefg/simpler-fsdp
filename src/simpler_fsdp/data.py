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
encoding = tiktoken.encoding_for_model("gpt-4")

def tokenize_parquet_dir(
    data_dir: str,
    world_size: int,
    max_tokens_per_shard: int = 2**22,
    buf_cap: int = 1 << 20
):
    big_path = Path("all_tokens.bin")
    big_path.unlink(missing_ok=True)
    eot_token = encoding._special_tokens['<|endoftext|>']

    buffer = []
    total = 0
    with open(big_path, "ab") as f:
        dataset = ds.dataset(data_dir, format="parquet")

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

def tokenize_all_streaming(
    world_size: int,             # W
    buf_cap: int = 1 << 20,       # flush every 1 M tokens
    dataset_name: str = "pszemraj/simple_wikipedia",
):
    big_path = Path("all_tokens.bin")
    big_path.unlink(missing_ok=True)
    eot_token = encoding._special_tokens['<|endoftext|>']

    # 0. load the data
    texts = datasets.load_dataset(dataset_name, split='train').select_columns(['text'])

    # 1. stream into big file
    with open(big_path, 'ab') as f:
        tokens = texts.map(
            lambda batch: {'tokens': encoding.encode_ordinary_batch(batch['text'])},
            batched=True,
            batch_size=1000,
        )
        tokens.map(
            lambda batch: np.array([tok for lst in batch['tokens'] for tok in [*lst, eot_token]]).tofile(f),
            batched=True,
            batch_size=1000
        )

def shard_big_file(
    big_file_path: str,
    output_dir: str,
    world_size: int,
    max_tokens_per_shard: int = 2**22,   # e.g. 2**22
    remove_after_sharding: bool = False
):
    big_path = Path(big_file_path)
    # truncate to same # tokens per worker and multiple of 128
    mod_factor = world_size * 128 # each worker gets multiple of 128 tokens
    total_tokens = big_path.stat().st_size // 4
    usable = total_tokens - (total_tokens % mod_factor)
    if usable < total_tokens: # drop the tail
        with open(big_path, "r+b") as f:
            f.truncate(usable * 4)

    # determine shards per worker
    tokens_per_worker = usable // world_size
    if tokens_per_worker % max_tokens_per_shard == 0:
        shards_per_worker = tokens_per_worker // max_tokens_per_shard
        shard_sizes = [max_tokens_per_shard] * shards_per_worker
    else:
        shards_per_worker = tokens_per_worker // max_tokens_per_shard + 1
    shard_sizes = [max_tokens_per_shard] * (shards_per_worker - 1) + [tokens_per_worker % max_tokens_per_shard]

    print(f"→ {usable:,} tokens kept of {total_tokens:,} "
          f"({shards_per_worker * world_size} shards)")

    # 2.  Slice the big file → shard files (no RAM blow‑up)
    mmap = np.memmap(big_path, dtype="int32", mode="r")
    shard_idx = 0
    start = 0
    for i in range(shards_per_worker):
        for j in range(world_size):
            ntok = shard_sizes[i]
            end = start + ntok
            mmap[start : end].tofile(f"{output_dir}/data_{shard_idx}.bin")
            shard_idx += 1
            start = end

    mmap._mmap.close() # pyright: ignore
    if remove_after_sharding:
        big_path.unlink() # keep disk tidy

def data_loader_fast(
    data_dir: str,
    batch_size: int,
    seq_len: int,
    world_size: int,   # W
    rank: int,         # this worker’s id
    seed: int = 0
):
    # discover my shard set
    shards = sorted(glob.glob(f"{data_dir}/*.bin"))
    shards = [
        p for p in shards
        if int(p.split('_')[-1].split('.')[0]) % world_size == rank
    ]
    if not shards:
        raise RuntimeError(f"rank {rank} has no shards — check world_size/rank")

    stride   = seq_len + 1
    rng_ep   = torch.Generator().manual_seed(seed)   # epoch‑level rng
    arange_  = torch.arange(stride, dtype=torch.int64)
    epochs_so_far = 0

    while True:                                      # epoch loop
        order = torch.randperm(len(shards), generator=rng_ep)
        for i in order.tolist():                     # shard loop (shuffled)
            path   = shards[i]
            tokens = torch.from_file(
                path,
                dtype=torch.int32,
                size=os.path.getsize(path)//4
            ).to(torch.int64)

            n_seqs = (tokens.numel() - 1) // seq_len
            rng_shard = torch.Generator().manual_seed(seed + epochs_so_far)

            perm = torch.randperm(n_seqs, generator=rng_shard)
            for s in range(0, n_seqs - batch_size + 1, batch_size):
                starts = perm[s:s + batch_size] * seq_len
                idx    = starts[:, None] + arange_
                batch  = tokens[idx]

                buf = torch.empty_like(batch, pin_memory=True)
                buf.copy_(batch, non_blocking=True)

                inputs  = buf[:, :-1].to(rank, non_blocking=True)
                targets = buf[:, 1:].to(rank, non_blocking=True)
                yield inputs, targets

def download_from_hub(repo_id: str, local_dir: str):
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    import huggingface_hub
    huggingface_hub.snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=local_dir
    )

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <world_size> [--hf <dataset_name>] | [--parquet <dir_path>]")
        sys.exit(1)

    world_size = int(sys.argv[1])

    if len(sys.argv) == 4 and sys.argv[2] == "--hf":
        dataset_name = sys.argv[3]
        tokenize_all_streaming(world_size, dataset_name=dataset_name)

    elif len(sys.argv) == 4 and sys.argv[2] == "--parquet":
        data_dir = sys.argv[3]
        tokenize_parquet_dir(data_dir, world_size)

    else:
        print("Usage: python script.py <world_size> [--hf <dataset_name>] | [--parquet <dir_path>]")
        sys.exit(1)
