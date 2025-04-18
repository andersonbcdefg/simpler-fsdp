import time
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from cut_cross_entropy import linear_cross_entropy # pyright: ignore

class MLP(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(Attention, self).__init__()
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

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
        self.classifier = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, x, targets = None, dtype = None):
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        x = self.w_embs(x)
        for block in self.blocks:
            x = block(x)
        if targets is not None:
            return linear_cross_entropy(
                x.to(dtype),
                self.classifier.weight.to(dtype),
                targets
            )
        else:
            return x

@dataclass
class Config:
    vocab_size: int = 100_352
    model_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    batch_size: int = 32
    accumulation_steps: int = 4
    seq_len: int = 128
    learning_rate: float = 1e-4
    total_steps: int = 1_000
    warmup_steps: int = 50
    use_bf16: bool = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple Transformer language model")

    for name, field_obj in Config.__dataclass_fields__.values():
        field_type   = field_obj.type
        default_val  = field_obj.default

        if field_type == bool:
            # --flag / --no-flag form (Python ≥ 3.9)
            parser.add_argument(
                f"--{name}",
                action=argparse.BooleanOptionalAction,
                default=default_val,
                help=f"Toggle {name} (default: {default_val})",
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=field_type,
                default=default_val,
                help=f"Override the default value for {name}",
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
