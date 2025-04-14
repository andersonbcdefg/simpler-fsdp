import torch
from tqdm.auto import tqdm
import numpy as np
import tiktoken
import datasets
import random
from collections import deque
from torch.utils.data import Dataset, DataLoader
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

class TokenFileDataset(Dataset):
    def __init__(self, seq_len, device_id):
        self.seq_len = seq_len
        self.device_id = device_id
        self.filename = f'final_data_{device_id}.bin'
        self.tokens = np.memmap(self.filename, dtype=np.uint32, mode="r")
        self.num_tokens = len(self.tokens)
        self.num_seqs = (self.num_tokens - 1) // self.seq_len # need 1 extra token at the end to predict

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1 # return seq_len + 1 tokens for targets
        return torch.from_numpy(self.tokens[start:end]).long()

def data_loader_fast(
    batch_size,
    seq_len,
    device_id: int = 0,
    buffer_size: int = 1024
):
    dataset = TokenFileDataset(seq_len, device_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )
    while True:
        for chunk in dataloader:
            inputs, targets = chunk[:, :-1], chunk[:, 1:]
            yield inputs.to(device_id, non_blocking=True), targets.to(device_id, non_blocking=True)

if __name__ == "__main__":
    import sys
    num_shards = int(sys.argv[1])
    generate_data()
    tokenize_all(num_shards)
