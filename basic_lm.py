import time
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import data_loader
from dataclasses import dataclass, field, asdict
from model import Transformer, Config, linear_cross_entropy, parse_args, create_config_from_args

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
                if steps_so_far % config.accumulation_steps == config.accumulation_steps - 1:
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

if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(args)
    train(config)
