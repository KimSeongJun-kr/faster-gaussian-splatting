from typing import NamedTuple, Any, Optional
import torch
from torch.autograd.function import once_differentiable

from FasterGSCudaBackend import _C


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor  # affine transformation from model/world space to view space
    cam_position: torch.Tensor  # camera position in world space
    bg_color: torch.Tensor  # background color in RGB format
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int  # width of the image plane in pixels
    height: int  # height of the image plane in pixels
    focal_x: float  # focal length in x direction in pixels
    focal_y: float  # focal length in y direction in pixels
    center_x: float  # x coordinate of the image center in pixels (positive -> right)
    center_y: float  # y coordinate of the image center in pixels (positive -> down)
    near_plane: float  # near clipping plane distance
    far_plane: float  # far clipping plane distance
    proper_antialiasing: bool  # whether to use proper antialiasing

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.bg_color,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.proper_antialiasing,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        w2c: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
        color_gate: Optional[torch.Tensor],
        color_gated: Optional[torch.Tensor],
    ) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        (
            image, alpha, depth,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            # the explicit (differentiable) w2c replaces the settings copy,
            # which occupies the first as_tuple() slot
            w2c,
            *rasterizer_settings.as_tuple()[1:],
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            depth,
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
            w2c,
        )
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        # Constants of the backward pass (never differentiated): keep the
        # references on ctx rather than in saved_tensors.
        ctx.color_gate = color_gate
        ctx.color_gated = color_gated
        return image, alpha, depth

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
        grad_alpha: torch.Tensor,
        grad_depth: torch.Tensor,
    ) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, Optional[torch.Tensor], None, None, None]':
        # materialize_grads is enabled by default, so unused outputs arrive as
        # zero-filled tensors; the None guards are purely defensive.
        if grad_alpha is None:
            grad_alpha = torch.zeros_like(grad_image[:1])
        if grad_depth is None:
            grad_depth = torch.zeros_like(grad_image[:1])
        *saved, w2c = ctx.saved_tensors
        compute_w2c_grad = ctx.needs_input_grad[7]
        empty = grad_image.new_empty(0)
        color_gate = ctx.color_gate if ctx.color_gate is not None else empty
        color_gated = ctx.color_gated if ctx.color_gated is not None else empty.to(torch.uint8)
        (
            grad_means, grad_scales, grad_rotations, grad_opacities,
            grad_sh_coefficients_0, grad_sh_coefficients_rest, grad_w2c
        ) = _C.backward(
            ctx.densification_info,
            grad_image,
            grad_alpha,
            grad_depth,
            *saved,
            # w2c sits in its usual positional slot (first as_tuple() entry);
            # compute_w2c_grad is appended after buffer_state
            w2c,
            *ctx.rasterizer_settings.as_tuple()[1:],
            *ctx.buffer_state,
            compute_w2c_grad,
            color_gate,
            color_gated,
        )
        return (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            None,  # densification_info
            grad_w2c if compute_w2c_grad else None,
            None,  # rasterizer_settings
            None,  # color_gate
            None,  # color_gated
        )


def diff_rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    w2c: Optional[torch.Tensor] = None,
    color_gate: Optional[torch.Tensor] = None,
    color_gated: Optional[torch.Tensor] = None,
) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
    """Returns (image (3,H,W), alpha (1,H,W), accumulated depth (1,H,W)).

    The depth map is alpha-weighted accumulated depth; divide by alpha
    (clamped) to obtain the expected depth.

    Pass `w2c` to make the view matrix a differentiable input (camera pose
    gradients); it takes precedence over `rasterizer_settings.w2c`. The
    returned gradient covers the geometric path only (mean projection incl.
    the Jacobian-through-depth term) — covariance-through-W and SH
    view-direction contributions are omitted.

    `color_gate` ((H,W) float in [0,1]) with `color_gated` ((N,) uint8, 1 =
    obeys the gate) scales the colour gradient (sh0 and sh_rest) that each
    flagged primitive receives from each pixel. Every other gradient path
    (opacity, mean, scale, rotation, depth, w2c) is untouched. Both None =
    no gate (the backward is bit-for-bit the ungated kernel path).
    """
    if (color_gate is None) != (color_gated is None):
        raise ValueError("color_gate and color_gated must be given together")
    if color_gate is not None:
        color_gate = color_gate.reshape(-1).contiguous().float()
        color_gated = color_gated.reshape(-1).contiguous().to(torch.uint8)
    if w2c is None:
        w2c = rasterizer_settings.w2c
    return _Rasterize.apply(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        densification_info,
        w2c.contiguous(),
        rasterizer_settings,
        color_gate,
        color_gated,
    )


def rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    to_chw: bool,
    clamp_output: bool = True,
) -> torch.Tensor:
    return _C.inference(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
        to_chw,
        clamp_output,
    )


def update_pruning_scores(
    scores: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> torch.Tensor:
    return _C.pruning_scores(
        scores,
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
    )
