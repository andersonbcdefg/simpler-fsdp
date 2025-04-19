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
encoding = tiktoken.encoding_for_model("gpt-4")

def tokenize_all_streaming(
    world_size: int,             # W
    max_tokens_per_shard: int = 2**22,   # e.g. 2**22
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
            mmap[start : end].tofile(f"final_data_{shard_idx}.bin")
            shard_idx += 1
            start = end

    mmap._mmap.close() # pyright: ignore
    big_path.unlink() # keep disk tidy

def data_loader_fast(
    batch_size: int,
    seq_len: int,
    world_size: int,   # W
    rank: int,         # this worker’s id  (0 ≤ rank < world_size)
    seed: int = 0
):
    # discover my shard set
    shards = sorted(glob.glob("final_data_*.bin"))
    shards = [p for p in shards
              if int(p.split('_')[-1].split('.')[0]) % world_size == rank]
    if not shards:
        raise RuntimeError("rank has no shards — check world_size/rank")

    stride   = seq_len + 1
    rng_ep   = torch.Generator().manual_seed(seed)   # epoch‑level rng
    arange_  = torch.arange(stride, dtype=torch.int64)

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
            rng_shard = torch.Generator().manual_seed(
                seed ^ int.from_bytes(path.encode(), "little")
            )

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


# def data_loader_fast(
#     batch_size: int,
#     seq_len: int,
#     device_id: int,
#     seed: int = 0
# ):
#     """Exactly‑once shuffle, pure‑torch implementation."""
#     path   = f"final_data_{device_id}.bin"
#     tokens = torch.from_file(
#         path,
#         dtype=torch.int32,
#         size=os.path.getsize(path) // 4         # bytes → int32 count
#     ).to(torch.int64)                           # easier math

#     stride  = seq_len + 1
#     n_seqs  = (tokens.numel() - 1) // seq_len
#     rng     = torch.Generator().manual_seed(seed)
#     arange_ = torch.arange(stride, dtype=torch.int64)

#     while True:                                 # epoch loop
#         perm = torch.randperm(n_seqs, generator=rng)           # ~2 MB
#         for s in range(0, n_seqs - batch_size + 1, batch_size):
#             starts = perm[s:s + batch_size] * seq_len          # (B,)
#             idx    = starts[:, None] + arange_                 # (B, stride)
#             batch  = tokens[idx]                               # CPU gather

#             buf = torch.empty_like(batch, pin_memory=True)     # staging
#             buf.copy_(batch, non_blocking=True)

#             inputs  = buf[:, :-1].to(device_id, non_blocking=True)
#             targets = buf[:, 1:].to(device_id, non_blocking=True)
#             yield inputs, targets

if __name__ == "__main__":
    import sys
    num_shards = int(sys.argv[1])
    tokenize_all_streaming(num_shards)
