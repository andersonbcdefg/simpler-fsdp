import os
import time
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import data_loader, data_loader_fast
from dataclasses import dataclass, field, asdict
from model import Transformer, Config, linear_cross_entropy, parse_args, create_config_from_args
from contextlib import nullcontext
# NEW! for ddp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DiLoCo specific parameters
SYNC_EVERY = 32
OUTER_LR = 0.6

def get_global_params(outer_optimizer: torch.optim.Optimizer):
    return [
        param.data.detach().clone().to("cpu")
        for group in outer_optimizer.param_groups
        for param in group["params"]
    ]

def train_diloco(config: Config | None = None):
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
    # since no DDP wrapper, manually sync initial model weights
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    outer_optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / config.warmup_steps if step < config.warmup_steps else (config.total_steps - step) / (config.total_steps - config.warmup_steps)
    )
    scaler = torch.amp.grad_scaler.GradScaler()
    m_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"training model with {m_params:.2f}M parameters")

    # these are stored on CPU
    global_params = get_global_params(outer_optimizer)

    losses = []
    steps_so_far = 0
    pbar = tqdm(total=config.total_steps) if device_id == 0 else None
    for inputs, targets in data_loader_fast(
        config.batch_size // world_size,
        config.seq_len,
        device_id=device_id
    ):
        with torch.autocast(device_type="cuda"):
            embs = model(inputs.to(device_id))
            loss = linear_cross_entropy(
                embs.view(-1, embs.shape[-1]),
                model.classifier,
                targets.reshape(-1).to(device_id)
            )
        losses.append(loss.item())
        scaler.scale(loss).backward()

        if steps_so_far % config.accumulation_steps == config.accumulation_steps - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # decide whether to do diloco sync
            optimizer_steps_so_far = steps_so_far // config.accumulation_steps
            if optimizer_steps_so_far % SYNC_EVERY == SYNC_EVERY - 1:
                # do the thing!
                if device_id == 0:
                    print(f"perform outer step at step {steps_so_far}")
                local_params = [
                    param
                    for group in optimizer.param_groups
                    for param in group["params"]
                ]
                for global_param, local_param in zip(global_params, local_params):
                    global_param_on_device = global_param.data.to(local_param.device)
                    # calculate delta, set the local gradient to move TOWARDS new params avg.
                    local_param.grad = global_param_on_device - local_param.data
                    dist.all_reduce(tensor=local_param.grad, op=dist.ReduceOp.AVG)
                    # update local to be global
                    local_param.data = global_param_on_device

                outer_optimizer.step()
                outer_optimizer.zero_grad()
                global_params = get_global_params(outer_optimizer)

        scheduler.step()

        steps_so_far += 1
        if pbar and steps_so_far % 5 == 0:
            pbar.set_description(f"loss: {loss.item():.1f}, lr: {scheduler.get_last_lr()[0]:.1e}")
            pbar.update(steps_so_far - pbar.n)
        if steps_so_far >= config.total_steps:
            break

    if device_id == 0:
        with open(f"runs/{timestamp}.txt", "w") as f:
            for loss in losses:
                f.write(f"{loss:.4f}\n")
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(args)
    train_diloco(config)
