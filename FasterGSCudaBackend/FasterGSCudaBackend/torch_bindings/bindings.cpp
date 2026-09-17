#include <torch/extension.h>
#include "rasterization_api.h"
#include "adam.h"
#include "filter3d.h"
#include "densification_api.h"

namespace rasterization_api = faster_gs::rasterization;
namespace adam_api = faster_gs::adam;
namespace filter3d_api = faster_gs::filter3d;
namespace densification_api = faster_gs::densification;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("inference", &rasterization_api::inference_wrapper);
    m.def("pruning_scores", &rasterization_api::pruning_scores_wrapper);
    m.def("adam_step", &adam_api::adam_step_wrapper);
    m.def("update_3d_filter", &filter3d_api::update_3d_filter_wrapper);
    m.def("relocation_adjustment", &densification_api::relocation_wrapper);
    m.def("add_noise", &densification_api::add_noise_wrapper);
    // Version gate: a (4, N) densification_info silently degrades to plain
    // gradients on a backend built before absgrad support (row 0/1 offsets
    // coincide), so callers probe this instead of render-testing.
    m.def("absgrad_supported", []() { return true; });
    // Same idea for the per-pixel colour-gradient gate (backward's trailing
    // color_gate/color_gated tensors): an older backend would reject the
    // extra positional arguments at call time, so callers probe this first.
    m.def("color_gate_supported", []() { return true; });
}
