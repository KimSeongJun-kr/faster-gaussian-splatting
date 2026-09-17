#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "sh_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::backward {

    __global__ void preprocess_backward_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const uint* __restrict__ primitive_n_touched_tiles,
        const float2* __restrict__ grad_mean2d,
        const float* __restrict__ grad_conic,
        const float* __restrict__ grad_primitive_depth,
        float3* __restrict__ grad_means,
        float3* __restrict__ grad_scales,
        float4* __restrict__ grad_rotations,
        float* __restrict__ grad_opacities,
        float3* __restrict__ grad_sh_coefficients_0,
        float3* __restrict__ grad_sh_coefficients_rest,
        float* __restrict__ densification_info,
        const int densification_rows,
        float* __restrict__ grad_w2c,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const bool proper_antialiasing)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // sh evaluation backward
        const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
            sh_coefficients_rest, grad_sh_coefficients_0, grad_sh_coefficients_rest,
            mean3d, cam_position[0], primitive_idx,
            active_sh_bases, total_sh_bases
        );

        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        // compute 3d covariance from scale and rotation
        const float3 raw_scale = scales[primitive_idx];
        const float3 variance = expf(2.0f * raw_scale);
        const float4 raw_rotation = rotations[primitive_idx];
        float quaternion_norm_sq = 1.0f;
        const mat3x3 R = convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq);
        const mat3x3 RSS = {
            R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
            R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
            R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
        };
        const mat3x3_triu cov3d {
            RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
            RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
            RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
            RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
            RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
            RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
        };

        // ewa splatting gradient helpers
        const float clip_left = (-0.15f * width - center_x) / focal_x;
        const float clip_right = (1.15f * width - center_x) / focal_x;
        const float clip_top = (-0.15f * height - center_y) / focal_y;
        const float clip_bottom = (1.15f * height - center_y) / focal_y;
        const float x_clipped = clamp(x, clip_left, clip_right);
        const float y_clipped = clamp(y, clip_top, clip_bottom);
        const float j11 = focal_x / depth;
        const float j13 = -j11 * x_clipped;
        const float j22 = focal_y / depth;
        const float j23 = -j22 * y_clipped;
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z
        );
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z
        );
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
        );
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
        );

        // 2d covariance gradient
        const float a_raw = dot(jwc_r1, jw_r1), b = dot(jwc_r1, jw_r2), c_raw = dot(jwc_r2, jw_r2);
        const float kernel_size = proper_antialiasing ? config::dilation_proper_antialiasing : config::dilation;
        const float a = a_raw + kernel_size, c = c_raw + kernel_size;
        const float aa = a * a, bb = b * b, cc = c * c;
        const float ac = a * c, ab = a * b, bc = b * c;
        const float determinant = ac - bb;
        const float determinant_sq = determinant * determinant;
        const float determinant_rcp_sq = 1.0f / determinant_sq;
        const float3 dL_dconic = make_float3(
            grad_conic[primitive_idx],
            grad_conic[n_primitives + primitive_idx],
            grad_conic[2 * n_primitives + primitive_idx]
        );
        float3 dL_dcov2d = determinant_rcp_sq * make_float3(
            2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
            bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
            2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z
        );

        // account for proper antialiasing
        if (proper_antialiasing) {
            const float opacity = sigmoid(opacities[primitive_idx]);
            const float dL_dopacity_conv_factor = grad_opacities[primitive_idx];
            const float determinant_raw = a_raw * c_raw - bb;
            const float radicand = fmaxf(determinant_raw / determinant, 0.0f);
            const float conv_factor = sqrtf(radicand);
            const float dL_dopacity = dL_dopacity_conv_factor * conv_factor * opacity * (1.0f - opacity);
            grad_opacities[primitive_idx] = dL_dopacity;
            // the remaining part works but causes exploding gradients that lead to lots of degenerate Gaussians
            if constexpr (!config::detach_dilation_proper_antialiasing_from_cov2d) {
                // based on https://github.com/nerfstudio-project/gsplat/blob/65042cc501d1cdbefaf1d6f61a9a47575eec8c71/gsplat/cuda/include/Utils.cuh#L390
                const float3 conic = make_float3(
                    c / determinant,
                    -b / determinant,
                    a / determinant
                );
                const float determinant_conic = conic.x * conic.z - conic.y * conic.y;
                const float dL_dradicand = dL_dopacity_conv_factor * opacity / fmaxf(2.0f * conv_factor, 1e-6f);
                const float one_minus_radicand = 1.0f - radicand;
                dL_dcov2d.x += dL_dradicand * (one_minus_radicand * conic.x - kernel_size * determinant_conic);
                dL_dcov2d.y += dL_dradicand * 2.0f * one_minus_radicand * conic.y;
                dL_dcov2d.z += dL_dradicand * (one_minus_radicand * conic.z - kernel_size * determinant_conic);
            }
        }

        // 3d covariance gradient
        const mat3x3_triu dL_dcov3d = {
            jw_r1.x * jw_r1.x * dL_dcov2d.x + 2.0f * jw_r1.x * jw_r2.x * dL_dcov2d.y + jw_r2.x * jw_r2.x * dL_dcov2d.z,
            jw_r1.x * jw_r1.y * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.y * dL_dcov2d.z,
            jw_r1.x * jw_r1.z * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.z * dL_dcov2d.z,
            jw_r1.y * jw_r1.y * dL_dcov2d.x + 2.0f * jw_r1.y * jw_r2.y * dL_dcov2d.y + jw_r2.y * jw_r2.y * dL_dcov2d.z,
            jw_r1.y * jw_r1.z * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + jw_r2.y * jw_r2.z * dL_dcov2d.z,
            jw_r1.z * jw_r1.z * dL_dcov2d.x + 2.0f * jw_r1.z * jw_r2.z * dL_dcov2d.y + jw_r2.z * jw_r2.z * dL_dcov2d.z,
        };

        // gradient of J * W
        const float3 dL_djw_r1 = 2.0f * make_float3(
            jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
            jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
            jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y
        );
        const float3 dL_djw_r2 = 2.0f * make_float3(
            jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
            jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
            jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z
        );

        // gradient of non-zero entries in J
        const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
        const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
        const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
        const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

        // load gradient of 2d mean
        const float2 dL_dmean2d = grad_mean2d[primitive_idx];

        // for adaptive density control
        if (densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;
            float2 g = dL_dmean2d;  // plain: cancelled sum over pixels
            if (densification_rows >= 4) {
                // absgrad: component-wise |grad| sums written by blend_backward_cu
                // into rows 2/3 (norm taken here, once, after the global sum —
                // matches gsplat's ‖(Σ|dx|, Σ|dy|)‖ convention).
                g = make_float2(densification_info[2 * n_primitives + primitive_idx],
                                densification_info[3 * n_primitives + primitive_idx]);
                // zero-after-read: rows 2/3 are per-backward scratch. Every
                // primitive blend wrote has n_touched_tiles > 0, so this kernel
                // is guaranteed to visit it — clearing here keeps callers that
                // reuse one info tensor across backwards from inflating row 1
                // with norms of running sums.
                densification_info[2 * n_primitives + primitive_idx] = 0.0f;
                densification_info[3 * n_primitives + primitive_idx] = 0.0f;
            }
            const float2 dL_dmean2d_ndc = 0.5f * make_float2(
                g.x * width,
                g.y * height
            );
            densification_info[n_primitives + primitive_idx] += length(dL_dmean2d_ndc);
        }

        // mean3d camera space gradient from mean2d
        float3 dL_dmean3d_cam = make_float3(
            j11 * dL_dmean2d.x,
            j22 * dL_dmean2d.y,
            -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y
        );

        // add mean3d camera space gradient from J while accounting for clipping
        const bool valid_x = x >= clip_left && x <= clip_right;
        const bool valid_y = y >= clip_top && y <= clip_bottom;
        if (valid_x) dL_dmean3d_cam.x -= j11 * dL_dj13 / depth;
        if (valid_y) dL_dmean3d_cam.y -= j22 * dL_dj23 / depth;
        const float factor_x = 1.0f + static_cast<float>(valid_x);
        const float factor_y = 1.0f + static_cast<float>(valid_y);
        dL_dmean3d_cam.z += (j11 * (factor_x * x_clipped * dL_dj13 - dL_dj11) + j22 * (factor_y * y_clipped * dL_dj23 - dL_dj22)) / depth;

        // The primitive's rendered depth IS its camera-space z (see `depth` above),
        // so the blend pass's dL/d(primitive depth) lands here unchanged. Without
        // this term d(rendered depth)/d(camera-z) is silently zero: the loss still
        // falls (through alpha) but depth supervision cannot move geometry or pose.
        // Verified 2026-09-17 against finite differences — see
        // docs/ablation_study/tools/depth_gradient_gate.py in the dynsplat repo.
        if (grad_primitive_depth != nullptr) {
            dL_dmean3d_cam.z += grad_primitive_depth[primitive_idx];
        }

        // camera pose gradient: mean3d_cam = W * mean3d + t with [W | t] = the top three
        // w2c rows, so d/dW[row] = dL_dmean3d_cam[row] * [mean3d, 1]. The covariance
        // path through W and the SH view-direction path are omitted (geometry-only
        // approximation). All threads surviving the guard above reach this uniformly,
        // so the coalesced group is well defined; non-finite entries contribute zero
        // instead of leaving the reduction.
        if (grad_w2c != nullptr) {
            const float dL_dmean_cam[3] = {dL_dmean3d_cam.x, dL_dmean3d_cam.y, dL_dmean3d_cam.z};
            const float mean_hom[4] = {mean3d.x, mean3d.y, mean3d.z, 1.0f};
            const bool finite =
                isfinite(dL_dmean_cam[0]) && isfinite(dL_dmean_cam[1]) && isfinite(dL_dmean_cam[2]) &&
                isfinite(mean3d.x) && isfinite(mean3d.y) && isfinite(mean3d.z);
            const cg::coalesced_group active_threads = cg::coalesced_threads();
            #pragma unroll
            for (int row = 0; row < 3; row++) {
                #pragma unroll
                for (int col = 0; col < 4; col++) {
                    const float contribution = finite ? dL_dmean_cam[row] * mean_hom[col] : 0.0f;
                    const float group_sum = cg::reduce(active_threads, contribution, cg::plus<float>());
                    if (active_threads.thread_rank() == 0) atomicAdd(&grad_w2c[4 * row + col], group_sum);
                }
            }
        }

        // 3d mean gradient from splatting
        const float3 dL_dmean3d_from_splatting = make_float3(
            w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
            w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
            w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z
        );

        // write total 3d mean gradient
        const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
        grad_means[primitive_idx] = dL_dmean3d;

        // scale gradient
        const float3 dL_dvariance = make_float3(
            R.m11 * R.m11 * dL_dcov3d.m11 + R.m21 * R.m21 * dL_dcov3d.m22 + R.m31 * R.m31 * dL_dcov3d.m33 +
                2.0f * (R.m11 * R.m21 * dL_dcov3d.m12 + R.m11 * R.m31 * dL_dcov3d.m13 + R.m21 * R.m31 * dL_dcov3d.m23),
            R.m12 * R.m12 * dL_dcov3d.m11 + R.m22 * R.m22 * dL_dcov3d.m22 + R.m32 * R.m32 * dL_dcov3d.m33 +
                2.0f * (R.m12 * R.m22 * dL_dcov3d.m12 + R.m12 * R.m32 * dL_dcov3d.m13 + R.m22 * R.m32 * dL_dcov3d.m23),
            R.m13 * R.m13 * dL_dcov3d.m11 + R.m23 * R.m23 * dL_dcov3d.m22 + R.m33 * R.m33 * dL_dcov3d.m33 +
                2.0f * (R.m13 * R.m23 * dL_dcov3d.m12 + R.m13 * R.m33 * dL_dcov3d.m13 + R.m23 * R.m33 * dL_dcov3d.m23)
        );
        const float3 dL_dscale = 2.0f * variance * dL_dvariance;
        grad_scales[primitive_idx] = dL_dscale;

        // rotation gradient
        const mat3x3 dL_dR = {
            2.0f * (RSS.m11 * dL_dcov3d.m11 + RSS.m21 * dL_dcov3d.m12 + RSS.m31 * dL_dcov3d.m13),
            2.0f * (RSS.m12 * dL_dcov3d.m11 + RSS.m22 * dL_dcov3d.m12 + RSS.m32 * dL_dcov3d.m13),
            2.0f * (RSS.m13 * dL_dcov3d.m11 + RSS.m23 * dL_dcov3d.m12 + RSS.m33 * dL_dcov3d.m13),
            2.0f * (RSS.m11 * dL_dcov3d.m12 + RSS.m21 * dL_dcov3d.m22 + RSS.m31 * dL_dcov3d.m23),
            2.0f * (RSS.m12 * dL_dcov3d.m12 + RSS.m22 * dL_dcov3d.m22 + RSS.m32 * dL_dcov3d.m23),
            2.0f * (RSS.m13 * dL_dcov3d.m12 + RSS.m23 * dL_dcov3d.m22 + RSS.m33 * dL_dcov3d.m23),
            2.0f * (RSS.m11 * dL_dcov3d.m13 + RSS.m21 * dL_dcov3d.m23 + RSS.m31 * dL_dcov3d.m33),
            2.0f * (RSS.m12 * dL_dcov3d.m13 + RSS.m22 * dL_dcov3d.m23 + RSS.m32 * dL_dcov3d.m33),
            2.0f * (RSS.m13 * dL_dcov3d.m13 + RSS.m23 * dL_dcov3d.m23 + RSS.m33 * dL_dcov3d.m33)
        };
        const float4 dL_drotation = convert_quaternion_to_rotation_matrix_backward(raw_rotation, dL_dR);
        grad_rotations[primitive_idx] = dL_drotation;

    }

    // based on https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404
    __global__ void blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float* __restrict__ primitive_depth,
        const float3* __restrict__ bg_color,
        const float* __restrict__ grad_image,
        const float* __restrict__ grad_alpha_map,
        const float* __restrict__ grad_depth_map,
        const float* __restrict__ image,
        const float* __restrict__ depth_map,
        const float* __restrict__ tile_final_transmittances,
        const uint* __restrict__ tile_max_n_processed,
        const uint* __restrict__ tile_n_processed,
        const uint* __restrict__ bucket_tile_index,
        const float4* __restrict__ bucket_color_transmittance,
        const float* __restrict__ bucket_depth,
        float2* __restrict__ grad_mean2d,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_primitive_depth,
        float* __restrict__ grad_opacity,
        float3* __restrict__ grad_sh_coefficients_0,
        float* __restrict__ densification_info,
        const int densification_rows,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool proper_antialiasing)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        const uint bucket_idx = block.group_index().x;
        const uint lane_idx = warp.thread_rank();

        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        if (tile_bucket_idx * 32 >= tile_max_n_processed[tile_idx]) return;

        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

        // load gaussian data
        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};
        float3 conic = {0.0f, 0.0f, 0.0f};
        float opacity = 0.0f;
        float3 color = {0.0f, 0.0f, 0.0f};
        float3 color_grad_factor = {0.0f, 0.0f, 0.0f};
        float depth = 0.0f;
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
            conic = make_float3(conic_opacity);
            opacity = conic_opacity.w;
            const float3 color_unclamped = primitive_color[primitive_idx];
            color = fmaxf(color_unclamped, 0.0f);
            if (color_unclamped.x >= 0.0f) color_grad_factor.x = 1.0f;
            if (color_unclamped.y >= 0.0f) color_grad_factor.y = 1.0f;
            if (color_unclamped.z >= 0.0f) color_grad_factor.z = 1.0f;
            depth = primitive_depth[primitive_idx];
        }

        // helpers
        const float3 background = bg_color[0];
        const uint n_pixels = width * height;

        // gradient accumulation
        float2 dL_dmean2d_accum = {0.0f, 0.0f};
        // absgrad (AbsGS): per-pixel |dL/dmean2d| accumulated component-wise, so
        // opposing-sign contributions within a view cannot cancel. Lane owns the
        // primitive here, so this register already holds the bucket-local sum —
        // no warp reduction needed (unlike gsplat's pixel-per-thread layout).
        float2 dL_dmean2d_abs_accum = {0.0f, 0.0f};
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
        float dL_dopacity_accum = 0.0f;
        float dL_ddepth_accum = 0.0f;
        float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

        // tile metadata
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        uint last_contributor;
        float3 color_pixel_after;
        float transmittance;
        float3 grad_color_pixel;
        float grad_alpha_common;
        float depth_pixel_after;
        float grad_depth_pixel;

        bucket_color_transmittance += bucket_idx * config::block_size_blend;
        bucket_depth += bucket_idx * config::block_size_blend;
        __shared__ uint collected_last_contributor[32];
        __shared__ float4 collected_color_pixel_after_transmittance[32];
        __shared__ float4 collected_grad_info_pixel[32];
        __shared__ float collected_depth_pixel_after[32];
        __shared__ float collected_grad_depth_pixel[32];

        // iterate over all pixels in the tile
        #pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                if (local_idx < config::block_size_blend) {
                    const float4 color_transmittance = bucket_color_transmittance[local_idx];
                    const float start_depth = bucket_depth[local_idx];
                    const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                    const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                    // final values from forward pass before background blend and the respective gradients
                    float3 color_pixel_w_bg, grad_color_pixel;
                    float grad_alpha_pixel = 0.0f, depth_pixel = 0.0f, grad_depth_pixel_tmp = 0.0f;
                    if (pixel_coords.x < width && pixel_coords.y < height) {
                        color_pixel_w_bg = make_float3(
                            image[pixel_idx],
                            image[n_pixels + pixel_idx],
                            image[2 * n_pixels + pixel_idx]
                        );
                        grad_color_pixel = make_float3(
                            grad_image[pixel_idx],
                            grad_image[n_pixels + pixel_idx],
                            grad_image[2 * n_pixels + pixel_idx]
                        );
                        grad_alpha_pixel = grad_alpha_map[pixel_idx];
                        depth_pixel = depth_map[pixel_idx];
                        grad_depth_pixel_tmp = grad_depth_map[pixel_idx];
                    }
                    const float final_transmittance = tile_final_transmittances[pixel_idx];
                    collected_color_pixel_after_transmittance[lane_idx] = make_float4(
                        color_pixel_w_bg - final_transmittance * background - make_float3(color_transmittance),
                        color_transmittance.w
                    );
                    collected_grad_info_pixel[lane_idx] = make_float4(
                        grad_color_pixel,
                        final_transmittance * (grad_alpha_pixel - dot(grad_color_pixel, background))
                    );
                    collected_last_contributor[lane_idx] = tile_n_processed[pixel_idx];
                    collected_depth_pixel_after[lane_idx] = depth_pixel - start_depth;
                    collected_grad_depth_pixel[lane_idx] = grad_depth_pixel_tmp;
                }
                warp.sync();
            }

            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
                color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
                color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
                transmittance = warp.shfl_up(transmittance, 1);
                grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
                grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
                grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
                grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
                depth_pixel_after = warp.shfl_up(depth_pixel_after, 1);
                grad_depth_pixel = warp.shfl_up(grad_depth_pixel, 1);
            }

            // which pixel index should this thread deal with?
            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            // leader thread loads values from shared memory into registers
            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int current_shmem_index = i % 32;
                last_contributor = collected_last_contributor[current_shmem_index];
                const float4 color_pixel_after_transmittance = collected_color_pixel_after_transmittance[current_shmem_index];
                color_pixel_after = make_float3(color_pixel_after_transmittance);
                transmittance = color_pixel_after_transmittance.w;
                const float4 grad_info_pixel = collected_grad_info_pixel[current_shmem_index];
                grad_color_pixel = make_float3(grad_info_pixel);
                grad_alpha_common = grad_info_pixel.w;
                depth_pixel_after = collected_depth_pixel_after[current_shmem_index];
                grad_depth_pixel = collected_grad_depth_pixel[current_shmem_index];
            }

            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip) continue;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
            const float gaussian = expf(fminf(exponent, 0.0f));
            if (!config::original_opacity_interpretation && gaussian < config::min_alpha_threshold) continue;
            const float alpha = opacity * gaussian;
            if (config::original_opacity_interpretation && alpha < config::min_alpha_threshold) continue;

            const float blending_weight = transmittance * alpha;

            // color gradient
            const float3 dL_dcolor = blending_weight * grad_color_pixel * color_grad_factor;
            dL_dcolor_accum += dL_dcolor;

            color_pixel_after -= blending_weight * color;
            depth_pixel_after -= blending_weight * depth;

            // depth gradient w.r.t. the primitive's own depth. depth_acc(pixel) =
            // sum_i T_i * alpha_i * z_i, so d/dz_i is exactly the blending weight.
            // Upstream (LichtFeld-Studio PR #278 / gsplat) drops this; dropping it
            // makes d(rendered depth)/d(camera-z) zero, which silently disables
            // depth-driven geometry and pose optimisation.
            if (grad_primitive_depth != nullptr) {
                dL_ddepth_accum += blending_weight * grad_depth_pixel;
            }

            // alpha gradient
            const float one_minus_alpha = 1.0f - alpha;
            const float one_minus_alpha_rcp = 1.0f / fmaxf(one_minus_alpha, config::one_minus_alpha_eps);
            const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_after * one_minus_alpha_rcp, grad_color_pixel);
            const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
            const float dL_dalpha_from_depth = (transmittance * depth - depth_pixel_after * one_minus_alpha_rcp) * grad_depth_pixel;
            const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha + dL_dalpha_from_depth;
            // opacity gradient
            const float dL_dopacity = gaussian * dL_dalpha;
            dL_dopacity_accum += dL_dopacity;

            // conic and mean2d gradient
            const float gaussian_grad_helper = -alpha * dL_dalpha;
            const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(
                delta.x * delta.x,
                delta.x * delta.y,
                delta.y * delta.y
            );
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                conic.x * delta.x + conic.y * delta.y,
                conic.y * delta.x + conic.z * delta.y
            );
            dL_dmean2d_accum += dL_dmean2d;
            if (densification_rows >= 4) {
                dL_dmean2d_abs_accum += make_float2(fabsf(dL_dmean2d.x), fabsf(dL_dmean2d.y));
            }

            transmittance *= one_minus_alpha;
        }

        // finally add the gradients using atomics
        if (valid_primitive) {
            atomicAdd(&grad_mean2d[primitive_idx].x, dL_dmean2d_accum.x);
            atomicAdd(&grad_mean2d[primitive_idx].y, dL_dmean2d_accum.y);
            if (densification_rows >= 4) {
                atomicAdd(&densification_info[2 * n_primitives + primitive_idx], dL_dmean2d_abs_accum.x);
                atomicAdd(&densification_info[3 * n_primitives + primitive_idx], dL_dmean2d_abs_accum.y);
            }
            if (grad_primitive_depth != nullptr) {
                atomicAdd(&grad_primitive_depth[primitive_idx], dL_ddepth_accum);
            }
            atomicAdd(&grad_conic[primitive_idx], dL_dconic_accum.x);
            atomicAdd(&grad_conic[n_primitives + primitive_idx], dL_dconic_accum.y);
            atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], dL_dconic_accum.z);
            const float dL_dopacity = proper_antialiasing ? dL_dopacity_accum : opacity * (1.0f - opacity) * dL_dopacity_accum;
            atomicAdd(&grad_opacity[primitive_idx], dL_dopacity);
            atomicAdd(&grad_sh_coefficients_0[primitive_idx].x, dL_dcolor_accum.x);
            atomicAdd(&grad_sh_coefficients_0[primitive_idx].y, dL_dcolor_accum.y);
            atomicAdd(&grad_sh_coefficients_0[primitive_idx].z, dL_dcolor_accum.z);
        }
    }

}
