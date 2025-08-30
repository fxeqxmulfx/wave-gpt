from datetime import datetime
from typing import Iterable

import torch
import torch.nn as nn
from tqdm import tqdm

from model import GPT


def load_data() -> str:
    with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        result = f.read()
    return result


def get_chars(text: str) -> str:
    result = "".join(sorted(set(text)))
    return result


class Encoder:
    __slots__ = ("stoi", "vocab_size")

    def __init__(self, chars: str) -> None:
        stoi = {ch: i for i, ch in enumerate(chars)}
        self.stoi = stoi
        self.vocab_size = len(stoi.keys())

    def encode(self, text: str) -> tuple[int, ...]:
        stoi = self.stoi
        result = tuple(stoi[ch] for ch in text)
        return result


class Decoder:
    __slots__ = "itos"

    def __init__(self, chars: str) -> None:
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def decode(self, tokens: Iterable[int]) -> str:
        itos = self.itos
        result = "".join(itos[ch] for ch in tokens)
        return result


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


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: tuple[int, ...],
    val_data: tuple[int, ...],
    batch_size: int,
    block_size: int,
    eval_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
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
    model.train()
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
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    batch_size: int,
) -> tuple[GPT, float, int]:
    train_data, val_data = split_data(embedding)
    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
    )
    torch.set_float32_matmul_precision("high")
    model.to(device).compile(mode="max-autotune-no-cudagraphs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = datetime.now()
    pbar = tqdm(range(max_iters))
    val_loss = torch.inf
    for iter in pbar:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss, val_loss = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
            )
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
    result = model, float(val_loss), train_time
    return result


# INIT: val_loss=1.82 train_time=188
# ADD torch.compile: val_loss=1.83 train_time=49.67
# UPD eval_interval = 5_000: val_loss=1.82 train_time=40.67
# ADD flash-attention: val_loss=1.8 train_time=26.67
# UPD bias=False in lm_head: val_loss=1.82 train_time=28.67


def main() -> None:
    n_embd = 64
    block_size = 32
    n_head = 4
    n_layer = 4
    learning_rate = 1e-3
    max_iters = 5_000
    eval_interval = 5_000
    eval_iters = 200
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_data()
    chars = get_chars(text)
    encoder = Encoder(chars)
    decoder = Decoder(chars)
    encoded_text = encoder.encode(text)
    assert decoder.decode(encoded_text) == text
    n = 3
    val_loss_array = torch.zeros((n,))
    train_time_array = torch.zeros((n,))
    model, val_loss, train_time = train(
        embedding=encoded_text,
        vocab_size=encoder.vocab_size,
        device=device,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        learning_rate=learning_rate,
        max_iters=max_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        batch_size=batch_size,
    )
    val_loss_array[0] = val_loss
    train_time_array[0] = train_time
    for i in range(1, n):
        _, val_loss, train_time = train(
            embedding=encoded_text,
            vocab_size=encoder.vocab_size,
            device=device,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            learning_rate=learning_rate,
            max_iters=max_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            batch_size=batch_size,
        )
        val_loss_array[i] = val_loss
        train_time_array[i] = train_time
    val_loss = float(torch.mean(val_loss_array))
    train_time = float(torch.mean(train_time_array))
    print(f"val_loss={round(val_loss, 2)} train_time={round(train_time, 2)}")
    context = torch.zeros((1, 1), dtype=torch.int64, device=device)
    print(decoder.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))


if __name__ == "__main__":
    main()
