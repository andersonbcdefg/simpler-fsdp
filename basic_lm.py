import time
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import data_loader
from dataclasses import dataclass, field, asdict

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
        return x

@torch.compile
def linear_cross_entropy(embs, classifier, targets):
    logits = classifier(embs)
    return F.cross_entropy(logits, targets)

@dataclass
class Config:
    vocab_size: int = 100_352
    model_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    batch_size: int = 32
    seq_len: int = 128
    learning_rate: float = 1e-4
    total_steps: int = 1_000
    warmup_steps: int = 50

def train(config: Config | None = None):
    if config is None:
        config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = time.time()
    model = Transformer(
        config.vocab_size,
        config.model_dim,
        config.num_heads,
        config.num_layers
    ).to(device)
    # model.forward = torch.compile(model.forward)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / config.warmup_steps if step < config.warmup_steps else (config.total_steps - step) / (config.total_steps - config.warmup_steps)
    )
    scaler = torch.amp.grad_scaler.GradScaler()
    print("training model with", sum(p.numel() for p in model.parameters()), "parameters")
    steps_so_far = 0
    with open(f"runs/{timestamp}.txt", "w") as f:
        with tqdm(total=config.total_steps) as pbar:
            for inputs, targets in data_loader(config.batch_size, config.seq_len):
                with torch.autocast(device_type=device, enabled=device=="cuda"):
                    embs = model(inputs.to(device))
                loss = linear_cross_entropy(
                    embs.view(-1, embs.shape[-1]),
                    model.classifier,
                    targets.reshape(-1).to(device)
                )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                pbar.set_description(f"loss: {loss.item():.1f}, lr: {scheduler.get_last_lr()[0]:.1e}")
                f.write(f"{loss.item():.4f}\n")
                steps_so_far += 1
                pbar.update(1)
                if steps_so_far >= config.total_steps:
                    break


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple Transformer language model")

    # Add arguments for each field in Config
    config_fields = {field.name: field.type for field in Config.__dataclass_fields__.values()}

    for name, field_type in config_fields.items():
        parser.add_argument(
            f"--{name}",
            type=field_type,
            help=f"Override the default value for {name}"
        )

    return parser.parse_args()

def create_config_from_args(args):
    # Start with default config
    config = Config()

    # Override with args if provided
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            setattr(config, key, value)

    return config

if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(args)
    train(config)
