from typing import Optional
import torch

from FasterGSCudaBackend import _C


def relocation_adjustment(
    old_opacities: torch.Tensor,
    old_scales: torch.Tensor,
    n_samples_per_primitive: torch.Tensor,
) -> 'tuple[torch.Tensor, torch.Tensor]':
    return _C.relocation_adjustment(old_opacities, old_scales, n_samples_per_primitive)


def add_noise(
    raw_scales: torch.Tensor,
    raw_rotations: torch.Tensor,
    raw_opacities: torch.Tensor,
    means: torch.Tensor,
    current_lr: float,
    noise_scale: Optional[torch.Tensor] = None,
    random_samples: Optional[torch.Tensor] = None,
) -> None:
    """MCMC exploration noise, sampled in the kernel (philox, seeded from torch's CUDA generator).

    `noise_scale` (n_primitives,) multiplies each primitive's sample; `random_samples`
    (n_primitives, 3) replaces the in-kernel sampling with caller-supplied standard normals.
    """
    _C.add_noise(raw_scales, raw_rotations, raw_opacities, means, current_lr, noise_scale, random_samples)
