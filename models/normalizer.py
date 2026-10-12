import torch
import torch.nn as nn

class RunningObsNormalizer(nn.Module):
    """
    Running mean/std observation normalizer with clipping.
    Statistics are registered buffers, so they are saved and restored with the state_dict
    of any module that holds the normalizer.
    Statistics change only through update(), never in forward(), so they stay fixed
    while a batch of data is collected and used for the PPO update.
    """
    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))
        self.clip = clip
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """
        Merges the statistics of a batch (N, dim) into the running ones (parallel Welford update).
        """
        n = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + n
        delta = batch_mean - self.mean
        m2 = self.var * self.count + batch_var * n + delta.pow(2) * self.count * n / total
        self.mean += delta * n / total
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps), -self.clip, self.clip)
