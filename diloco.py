import os
import time
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import data_loader
from dataclasses import dataclass, field, asdict
from model import Transformer, Config, linear_cross_entropy, create_config_from_args
from contextlib import nullcontext
# NEW! for ddp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_ddp(config: Config | None = None):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    print(f"Start running basic DDP example on rank {rank}, device_id {device_id}.")

    if config is None:
        config = Config()

    assert config.batch_size % world_size == 0, "Batch size must be divisible by world size"
    timestamp = time.time()
    model = Transformer(
        config.vocab_size,
        config.model_dim,
        config.num_heads,
        config.num_layers
    ).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    # model.forward = torch.compile(model.forward)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / config.warmup_steps if step < config.warmup_steps else (config.total_steps - step) / (config.total_steps - config.warmup_steps)
    )
    scaler = torch.amp.grad_scaler.GradScaler()
    print("training model with", sum(p.numel() for p in model.parameters()), "parameters")
    losses = []
    steps_so_far = 0
    pbar = tqdm(total=config.total_steps) if device_id == 0 else None
    for inputs, targets in data_loader(config.batch_size // world_size, config.seq_len, device_id=device_id):
        with torch.autocast(device_type="cuda"):
            embs = ddp_model(inputs.to(device_id))
            loss = linear_cross_entropy(
                embs.view(-1, embs.shape[-1]),
                ddp_model.module.classifier,
                targets.reshape(-1).to(device_id)
            )
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        steps_so_far += 1
        if pbar and steps_so_far % 25 == 0:
            pbar.set_description(f"loss: {loss.item():.1f}, lr: {scheduler.get_last_lr()[0]:.1e}")
            pbar.update(steps_so_far - pbar.n)
        if steps_so_far >= config.total_steps:
            break

    if device_id == 0:
        with open(f"runs/{timestamp}.txt", "w") as f:
            for loss in losses:
                f.write(f"{loss:.4f}\n")
    dist.destroy_process_group()


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

if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(args)
    train_ddp(config)
