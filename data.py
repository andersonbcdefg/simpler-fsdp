import os
import torch
from tqdm.auto import tqdm
import numpy as np
import tiktoken
import datasets
import random
from collections import deque
encoding = tiktoken.encoding_for_model("gpt-4")

def generate_data():
    ds = datasets.load_dataset('pszemraj/simple_wikipedia', split='train')
    texts = [x['text'] for x in ds] # pyright: ignore
    random.seed(42)
    random.shuffle(texts)
    with open('data.txt', 'w') as file:
        for text in texts:
            file.write(text.replace('\n', ' ') + '\n')

def tokenize_all(num_shards: int = 1):
    all_tokens = []
    with open('data.txt', 'r') as file:
        for line in tqdm(file, total=226_000, desc="Processing lines"):
            tokens = encoding.encode(line.strip())
            all_tokens.extend(tokens)
    print("Total tokens:", len(all_tokens))
    tokens_per_shard = len(all_tokens) // num_shards
    shards = [all_tokens[i:i+tokens_per_shard] for i in range(0, len(all_tokens), tokens_per_shard)]
    for i, shard in enumerate(shards):
        np.array(shard, dtype='int32').tofile(f'final_data_{i}.bin')

def data_loader_fast(
    batch_size: int,
    seq_len: int,
    device_id: int,
    seed: int = 0
):
    """Exactly‑once shuffle, pure‑torch implementation."""
    path   = f"final_data_{device_id}.bin"
    tokens = torch.from_file(
        path,
        dtype=torch.int32,
        size=os.path.getsize(path) // 4         # bytes → int32 count
    ).to(torch.int64)                           # easier math

    stride  = seq_len + 1
    n_seqs  = (tokens.numel() - 1) // seq_len
    rng     = torch.Generator().manual_seed(seed)
    arange_ = torch.arange(stride, dtype=torch.int64)

    while True:                                 # epoch loop
        perm = torch.randperm(n_seqs, generator=rng)           # ~2 MB
        for s in range(0, n_seqs - batch_size + 1, batch_size):
            starts = perm[s:s + batch_size] * seq_len          # (B,)
            idx    = starts[:, None] + arange_                 # (B, stride)
            batch  = tokens[idx]                               # CPU gather

            buf = torch.empty_like(batch, pin_memory=True)     # staging
            buf.copy_(batch, non_blocking=True)

            inputs  = buf[:, :-1].to(device_id, non_blocking=True)
            targets = buf[:, 1:].to(device_id, non_blocking=True)
            yield inputs, targets

if __name__ == "__main__":
    import sys
    num_shards = int(sys.argv[1])
    generate_data()
    tokenize_all(num_shards)
