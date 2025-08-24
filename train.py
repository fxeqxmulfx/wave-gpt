from datetime import datetime
from typing import Iterable

import torch
import torch.nn as nn

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


n_embd = 64
block_size = 32
n_head = 4
n_layer = 4
learning_rate = 1e-3
max_iters = 5_000
eval_interval = 100
eval_iters = 200
batch_size = 16

# INIT: val_loss==1.82 train_time==188


def main() -> None:
    torch.manual_seed(42)
    text = load_data()
    chars = get_chars(text)
    encoder = Encoder(chars)
    decoder = Decoder(chars)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded_text = encoder.encode(text)
    assert decoder.decode(encoded_text) == text
    train_data, val_data = split_data(encoded_text)
    model = GPT(
        vocab_size=encoder.vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
    )
    model.to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = datetime.now()
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss, val_loss = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
            )
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        xb, yb = get_batch(
            data=train_data,
            batch_size=batch_size,
            device=device,
        )
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f"Train time {(datetime.now() - start).seconds}")
    context = torch.zeros((1, 1), dtype=torch.int64, device=device)
    print(decoder.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))


if __name__ == "__main__":
    main()
