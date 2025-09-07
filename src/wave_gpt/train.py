from datetime import datetime

import schedulefree
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from wave_gpt.model import GPT


def split_data(
    data: tuple[int, ...], train_part: float
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = int(len(data) * train_part)
    result = data[:n], data[n:]
    return result


def get_batch(
    data: tuple[int, ...],
    batch_size: int,
    block_size: int,
    device: str,
    use_random_coeff: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        tuple(
            torch.tensor(
                data[i : i + block_size],
                dtype=torch.int64,
            )
            for i in ix
        )
    )
    y = torch.stack(
        tuple(
            torch.tensor(
                data[i + 1 : i + block_size + 1],
                dtype=torch.int64,
            )
            for i in ix
        )
    )
    x, y = x.to(device=device), y.to(device=device)
    if use_random_coeff:
        coeff = (torch.randint(low=1, high=10, size=(1,)) / 10).to(device=device)
        x = (x * coeff).to(torch.int64)
        y = (y * coeff).to(torch.int64)
    return x, y


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    train_data: tuple[int, ...],
    val_data: tuple[int, ...],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    use_random_coeff: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device.type
    train = torch.Tensor()
    val = train
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = train_data if split == "train" else val_data
            X, Y = get_batch(
                data=data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
                use_random_coeff=use_random_coeff,
            )
            _, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean()
        if split == "train":
            train = mean_loss
        else:
            val = mean_loss
    return train, val


def train(
    mut_model: GPT,
    encoded_data: tuple[int, ...],
    learning_rate: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    max_iters: int = 5_000,
    eval_interval: int = 5_000,
    eval_iters: int = 200,
    batch_size: int = 16,
    train_part: float = 0.9,
    use_tqdm: bool = True,
    use_random_coeff: bool = False,
) -> tuple[float, int]:
    device = next(mut_model.parameters()).device.type
    train_data, val_data = split_data(encoded_data, train_part)
    optimizer = schedulefree.AdamWScheduleFree(
        mut_model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    start = datetime.now()
    if use_tqdm:
        pbar = tqdm(range(max_iters))
    else:
        pbar = range(max_iters)
    val_loss = torch.inf
    mut_model.train()
    optimizer.train()
    for iter in pbar:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            mut_model.eval()
            optimizer.eval()
            train_loss, val_loss = estimate_loss(
                model=mut_model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=mut_model.block_size,
                eval_iters=eval_iters,
                use_random_coeff=use_random_coeff,
            )
            mut_model.train()
            optimizer.train()
            status = (
                f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )
            if isinstance(pbar, tqdm):
                pbar.set_description(status)
            else:
                print(status)
        xb, yb = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=mut_model.block_size,
            device=device,
            use_random_coeff=use_random_coeff,
        )
        _, loss = mut_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_time_s = (datetime.now() - start).seconds
    mut_model.eval()
    result = float(val_loss), train_time_s
    return result
