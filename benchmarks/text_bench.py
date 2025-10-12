from typing import Iterable

import torch

from wave_gpt.model import GPT
from wave_gpt.train import train


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


# INIT: val_loss=1.82 train_time=188
# ADD torch.compile: val_loss=1.83 train_time=49.67
# UPD eval_interval = 5_000: val_loss=1.82 train_time=40.67
# ADD flash-attention: val_loss=1.8 train_time=26.67
# UPD bias=False in lm_head: val_loss=1.82 train_time=28.67
# ADD RMSNorm: val_loss=1.82 train_time=28.0
# ADD RoPE: val_loss=1.76 train_time=28.67
# ADD SwiGLU: val_loss=1.74 train_time=29.67
# ADD AdamWScheduleFree: val_loss=1.68 train_time=28.0


def main() -> None:
    torch.set_float32_matmul_precision("high")
    n_embd = 64
    block_size = 32
    n_head = 4
    n_layer = 4
    learning_rate = 1e-2
    betas = (0.9, 0.95)
    weight_decay = 0.1
    max_iters = 5_000
    eval_interval = 5_000
    eval_iters = 200
    batch_size = 16
    rmsnorm_eps = 1e-5
    rope_theta = 10000
    swiglu_alpha = 1.702
    swiglu_limit = 7.0
    train_part = 0.9
    temperature = 1
    top_p = 0.95
    use_checkpoint = True
    use_tqdm = True
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
    model: GPT | None = None
    for i in range(n):
        model = GPT(
            vocab_size=encoder.vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            rmsnorm_eps=rmsnorm_eps,
            rope_theta=rope_theta,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            top_p=top_p,
            temperature=temperature,
            use_checkpoint=use_checkpoint,
        )
        model.to(device=device).compile(mode="max-autotune-no-cudagraphs")
        val_loss, train_time = train(
            mut_model=model,
            encoded_data=encoded_text,
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            max_iters=max_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            batch_size=batch_size,
            train_part=train_part,
            use_tqdm=use_tqdm,
        )
        val_loss_array[i] = val_loss
        train_time_array[i] = train_time
    val_loss = float(torch.mean(val_loss_array))
    train_time = float(torch.mean(train_time_array))
    print(f"val_loss={round(val_loss, 2)} train_time={round(train_time, 2)}")
    context = torch.zeros((1, 1), dtype=torch.int64, device=device)
    if model is not None:
        print(decoder.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))


if __name__ == "__main__":
    main()
