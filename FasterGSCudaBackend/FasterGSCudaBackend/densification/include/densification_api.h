#pragma once

#include <torch/extension.h>
#include <optional>
#include <tuple>

namespace faster_gs::densification {

    std::tuple<torch::Tensor, torch::Tensor>
    relocation_wrapper(
        const torch::Tensor& old_opacities,
        const torch::Tensor& old_scales,
        const torch::Tensor& n_samples_per_primitive);

    void
    add_noise_wrapper(
        const torch::Tensor& raw_scales,
        const torch::Tensor& raw_rotations,
        const torch::Tensor& raw_opacities,
        torch::Tensor& means,
        const float current_lr,
        const std::optional<torch::Tensor>& noise_scale,
        const std::optional<torch::Tensor>& random_samples);

}
