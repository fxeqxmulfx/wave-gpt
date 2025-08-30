from datetime import datetime

import schedulefree
import torch
import torch.nn as nn
from tqdm import tqdm

from model import GPT


def split_data(
    data: tuple[int, ...], train_part: float = 0.9
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = int(len(data) * train_part)
    result = data[:n], data[n:]
    return result


def get_batch(
    data: tuple[int, ...],
    batch_size: int,
    block_size: int,
    device: str,
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
    x, y = x.to(device), y.to(device)
    return x, y


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    train_data: tuple[int, ...],
    val_data: tuple[int, ...],
    batch_size: int,
    block_size: int,
    eval_iters: int,
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
    embedding: tuple[int, ...],
    vocab_size: int,
    device: str,
    n_embd: int,
    block_size: int,
    n_head: int,
    n_layer: int,
    learning_rate: float,
    betas: tuple[float, float],
    weight_decay: float,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    batch_size: int,
    rmsnorm_eps: float,
    rope_theta: float,
) -> tuple[GPT, float, int]:
    train_data, val_data = split_data(embedding)
    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        rmsnorm_eps=rmsnorm_eps,
        rope_theta=rope_theta,
    )
    torch.set_float32_matmul_precision("high")
    model.to(device).compile(mode="max-autotune-no-cudagraphs")
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    start = datetime.now()
    pbar = tqdm(range(max_iters))
    val_loss = torch.inf
    model.train()
    optimizer.train()
    for iter in pbar:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            model.eval()
            optimizer.eval()
            train_loss, val_loss = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
            )
            model.train()
            optimizer.train()
            pbar.set_description(
                f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )
        xb, yb = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_time = (datetime.now() - start).seconds
    model.eval()
    result = model, float(val_loss), train_time
    return result
