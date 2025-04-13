import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import data_loader

class MLP(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(Attention, self).__init__()
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, L, self.num_heads, 3 * self.head_dim)
            .transpose(1, 2)
            .chunk(3, dim=-1) # B, H, L, Dh
        )
        attn_out_BHLD = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(attn_out_BHLD.transpose(1, 2).reshape(B, L, D))

class Block(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(Block, self).__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.attn = Attention(model_dim, num_heads)
        self.mlp = MLP(model_dim, model_dim * 4)

    def forward(self, x): # using parallel block setup
        normed = self.norm(x)
        attn_out = self.attn(normed)
        mlp_out = self.mlp(normed)
        return x + attn_out + mlp_out


class Transformer(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.w_embs = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads) for _ in range(num_layers)])
        self.classifier = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        x = self.w_embs(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


BATCH_SIZE = 16
SEQ_LEN = 128
TOTAL_STEPS = 1000
WARMUP_STEPS = 50

def train():
    timestamp = time.time()
    model = Transformer(100_352, 128, 4, 3).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / WARMUP_STEPS if step < WARMUP_STEPS else (TOTAL_STEPS - step) / (TOTAL_STEPS - WARMUP_STEPS)
    )
    print("training model with", sum(p.numel() for p in model.parameters()), "parameters")
    steps_so_far = 0
    with open(f"runs/{timestamp}.txt", "w") as f:
        with tqdm(total=TOTAL_STEPS) as pbar:
            for batch in data_loader(BATCH_SIZE * (SEQ_LEN + 1)):
                tokens = torch.from_numpy(batch).view(BATCH_SIZE, SEQ_LEN + 1).long().to("mps")
                logits = model(tokens[:, :-1])
                targets = tokens[:, 1:]
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                pbar.set_description(f"loss: {loss.item():.1f}, lr: {scheduler.get_last_lr()[0]:.1e}")
                f.write(f"{loss.item():.4f}\n")
                steps_so_far += 1
                pbar.update(1)
                if steps_so_far >= TOTAL_STEPS:
                    break


train()
