import torch

from model import GPT
from train import train

from wave_decoder_encoder import encode, decode


def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    result = torch.mean(torch.abs(x - y), dim=0)
    return float(torch.mean(result))


# ADD wave encoder decoder: val_loss=0.18 train_time=27.0 MAE=0.0061
# FIX wave encoder decoder: val_loss=0.19 train_time=27.0 MAE=0.0041
# UPD use round instead floor: val_loss=0.19 train_time=26.67 MAE=0.0016
# FIX shift in wave encoder decoder: val_loss=0.19 train_time=28.0 MAE=0.0016
# CLR train: val_loss=0.19 train_time=34.67 MAE=0.0027


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
    start, scale, encoded_data = encode(inp, vocab_size)
    n = 3
    val_loss_array = torch.zeros((n,))
    train_time_array = torch.zeros((n,))
    mae_loss_array = torch.zeros((n,))
    _encoded_data = tuple(encoded_data.reshape(-1).tolist())
    for i in range(n):
        model = GPT(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            rmsnorm_eps=rmsnorm_eps,
            rope_theta=rope_theta,
        )
        model.to(device).compile(mode="max-autotune-no-cudagraphs")
        val_loss, train_time = train(
            mut_model=model,
            encoded_data=tuple(encoded_data.reshape(-1).tolist()),
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            max_iters=max_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            batch_size=batch_size,
        )
        context = torch.tensor((_encoded_data[:8],), dtype=torch.int64, device=device)
        decoded = decode(
            start,
            scale,
            model.generate(context, max_new_tokens=22)[0]
            .reshape(-1, encoded_data.shape[1])
            .to(device="cpu"),
            vocab_size=vocab_size,
        )
        val_loss_array[i] = val_loss
        train_time_array[i] = train_time
        mae_loss_array[i] = mae(inp[:16], decoded)
    val_loss = float(torch.mean(val_loss_array))
    train_time = float(torch.mean(train_time_array))
    mae_loss = float(torch.mean(mae_loss_array))
    print(
        f"val_loss={round(val_loss, 2)} train_time={round(train_time, 2)} MAE={round(mae_loss, 4)}"
    )


if __name__ == "__main__":
    main()
