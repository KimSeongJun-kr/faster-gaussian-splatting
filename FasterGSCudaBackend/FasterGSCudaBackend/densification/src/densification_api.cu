#include "densification_api.h"
#include "mcmc.h"
#include "densification_config.h"
#include "helper_math.h"
#include "torch_utils.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <mutex>
#include <optional>
#include <tuple>
#include <cstdint>

std::tuple<torch::Tensor, torch::Tensor>
faster_gs::densification::relocation_wrapper(
    const torch::Tensor& old_opacities,
    const torch::Tensor& old_scales,
    const torch::Tensor& n_samples_per_primitive)
{
    const int n_primitives = old_opacities.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor new_opacities = torch::empty({n_primitives, 1}, float_options);
    torch::Tensor new_scales = torch::empty({n_primitives, 3}, float_options);

    relocation_adjustment(
        old_opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(old_scales.contiguous().data_ptr<float>()),
        n_samples_per_primitive.contiguous().data_ptr<int64_t>(),
        new_opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(new_scales.data_ptr<float>()),
        n_primitives
    );

    return std::make_tuple(new_opacities, new_scales);
}

void
faster_gs::densification::add_noise_wrapper(
    const torch::Tensor& raw_scales,
    const torch::Tensor& raw_rotations,
    const torch::Tensor& raw_opacities,
    torch::Tensor& means,
    const float current_lr,
    const std::optional<torch::Tensor>& noise_scale,
    const std::optional<torch::Tensor>& random_samples)
{
    // tensors must be contiguous CUDA float tensors
    CHECK_INPUT(config::debug, raw_scales, "raw_scales");
    CHECK_INPUT(config::debug, raw_rotations, "raw_rotations");
    CHECK_INPUT(config::debug, raw_opacities, "raw_opacities");
    CHECK_INPUT(config::debug, means, "means");
    const int n_primitives = raw_scales.size(0);
    // optional per-primitive noise scale (n_primitives,) and caller-supplied standard normal samples (n_primitives, 3)
    const bool has_scale = noise_scale.has_value();
    const bool has_samples = random_samples.has_value();
    if (has_scale) {
        CHECK_INPUT(config::debug, noise_scale.value(), "noise_scale");
        TORCH_CHECK(noise_scale->numel() == n_primitives, "noise_scale must hold one value per primitive");
    }
    if (has_samples) {
        CHECK_INPUT(config::debug, random_samples.value(), "random_samples");
        TORCH_CHECK(random_samples->numel() == 3 * n_primitives, "random_samples must be of shape (n_primitives, 3)");
    }

    // draw philox seed and offset from torch's cuda generator (only when sampling in the kernel)
    uint64_t seed = 0, offset = 0;
    if (!has_samples) {
        auto* generator = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(generator->mutex_);
        std::tie(seed, offset) = generator->philox_engine_inputs(4);
    }

    add_noise(
        reinterpret_cast<float3*>(raw_scales.data_ptr<float>()),
        reinterpret_cast<float4*>(raw_rotations.data_ptr<float>()),
        raw_opacities.data_ptr<float>(),
        has_scale ? noise_scale->data_ptr<float>() : nullptr,
        has_samples ? reinterpret_cast<const float3*>(random_samples->data_ptr<float>()) : nullptr,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        n_primitives,
        current_lr,
        seed,
        offset
    );

}
