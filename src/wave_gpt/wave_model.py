import pandas as pd
import torch

from wave_gpt.model import GPT
from wave_gpt.train import train
from wave_gpt.wave_encoder_decoder import decode, encode


class WaveGPT:
    __slots__ = "model"

    def __init__(
        self,
        device: str,
        vocab_size: int,
        n_embd: int = 64,
        block_size: int = 32,
        n_head: int = 4,
        n_layer: int = 4,
        rmsnorm_eps: float = 1e-5,
        rope_theta: float = 10000,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        temperature: float = 1,
        top_p: float = 0.95,
    ) -> None:
        model = GPT(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            rmsnorm_eps=rmsnorm_eps,
            rope_theta=rope_theta,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            temperature=temperature,
            top_p=top_p,
        )
        model.to(device=device).compile(mode="max-autotune-no-cudagraphs")
        model.eval()
        self.model = model

    def train(
        self,
        df: pd.DataFrame,
        order_of_derivative: int,
        domain_of_definition: torch.Tensor,
        learning_rate: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        max_iters: int = 5_000,
        eval_interval: int = 5_000,
        eval_iters: int = 200,
        batch_size: int = 16,
        train_part: float = 0.9,
        use_tqdm: bool = True,
    ) -> tuple[float, int]:
        _, _, encoded_data = encode(
            inp=torch.from_numpy(df.to_numpy()),
            vocab_size=self.model.vocab_size,
            domain_of_definition=domain_of_definition,
            order_of_derivative=order_of_derivative,
        )
        encoded_data = tuple(encoded_data.reshape(-1).tolist())
        val_loss, train_time_s = train(
            mut_model=self.model,
            encoded_data=encoded_data,
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
        result = val_loss, train_time_s
        return result

    @torch.inference_mode()
    def predict(
        self,
        df: pd.DataFrame,
        order_of_derivative: int,
        domain_of_definition: torch.Tensor,
        max_new_points: int,
    ) -> pd.DataFrame:
        columns = df.shape[1]
        assert (df.shape[0] + max_new_points - 1) * columns <= self.model.block_size
        vocab_size = self.model.vocab_size
        device = next(self.model.parameters()).device.type
        start, scale, encoded_data = encode(
            inp=torch.from_numpy(df.to_numpy()),
            vocab_size=vocab_size,
            domain_of_definition=domain_of_definition,
            order_of_derivative=order_of_derivative,
        )
        encoded_data = encoded_data.reshape(-1)
        context = encoded_data.unsqueeze(0).to(device=device)
        generated = (
            self.model.generate(
                context,
                max_new_tokens=max_new_points * columns,
            )[0]
            .reshape(-1, columns)
            .to(device="cpu")
        )
        decoded = decode(
            start=start,
            scale=scale,
            inp=generated,
            vocab_size=vocab_size,
            order_of_derivative=order_of_derivative,
        )
        result = pd.DataFrame(decoded.numpy(), columns=df.columns)
        return result

    def save(self, path: str = "wave-gpt.bin") -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "wave-gpt.bin") -> None:
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
