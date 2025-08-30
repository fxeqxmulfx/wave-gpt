import torch

from train import train


def encode(
    inp: torch.Tensor, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert vocab_size % 2 == 0
    _vocab_size = vocab_size - 2
    diff = torch.diff(inp, dim=0)
    max_diff, _ = torch.max(torch.abs(diff), dim=0)
    scale = _vocab_size / max_diff / 2
    scaled_diff = diff * scale
    residual = scaled_diff - scaled_diff.to(torch.int64)
    residual = torch.diff(
        torch.floor(torch.cumsum(residual, dim=0)).to(torch.int64),
        dim=0,
        prepend=torch.zeros(residual[:1].shape, dtype=torch.int64),
    )
    result = scaled_diff.to(torch.int64) + _vocab_size // 2 + residual
    return inp[:1], scale, result


def decode(
    start: torch.Tensor, scale: torch.Tensor, inp: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    assert vocab_size % 2 == 0
    _vocab_size = vocab_size - 2
    diff = (inp - _vocab_size // 2) / scale
    result = torch.concat((start, diff), dim=0).cumsum(dim=0)
    return result


def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    result = torch.mean(torch.abs(x - y), dim=0)
    return float(torch.mean(result))


# ADD wave encoder decoder: val_loss=0.18 train_time=27.0 MAE=0.0061
# FIX wave encoder decoder: val_loss=0.19 train_time=27.0 MAE=0.0041


def main() -> None:
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
    vocab_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = torch.arange(1_000_000)
    inp = torch.stack(
        (
            torch.sin(idx),
            torch.cos(idx),
        ),
        dim=1,
    )
    start, scale, embedding = encode(inp, vocab_size)
    torch.testing.assert_close(
        decode(start, scale, embedding, vocab_size), inp, rtol=1, atol=0.0076
    )
    n = 3
    val_loss_array = torch.zeros((n,))
    train_time_array = torch.zeros((n,))
    _embedding = tuple(embedding.reshape(-1).tolist())
    model, val_loss, train_time = train(
        embedding=_embedding,
        vocab_size=vocab_size,
        device=device,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        learning_rate=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        max_iters=max_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        batch_size=batch_size,
        rmsnorm_eps=rmsnorm_eps,
        rope_theta=rope_theta,
    )
    val_loss_array[0] = val_loss
    train_time_array[0] = train_time
    for i in range(1, n):
        _, val_loss, train_time = train(
            embedding=tuple(embedding.reshape(-1).tolist()),
            vocab_size=vocab_size,
            device=device,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            max_iters=max_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            batch_size=batch_size,
            rmsnorm_eps=rmsnorm_eps,
            rope_theta=rope_theta,
        )
        val_loss_array[i] = val_loss
        train_time_array[i] = train_time
    val_loss = float(torch.mean(val_loss_array))
    train_time = float(torch.mean(train_time_array))
    context = torch.tensor((_embedding[:8],), dtype=torch.int64, device=device)
    decoded = decode(
        start,
        scale,
        model.generate(context, max_new_tokens=22)[0]
        .reshape(-1, embedding.shape[1])
        .to(device="cpu"),
        vocab_size=vocab_size,
    )
    print(
        f"val_loss={round(val_loss, 2)} train_time={round(train_time, 2)} MAE={round(mae(inp[:16], decoded), 4)}"
    )


if __name__ == "__main__":
    main()
