#pragma once
// decode_psos.hpp — compile the decode DAG's kernels into a DecodeStepPsos table (beta).
//
// Each Kernel kind maps to one runtime-compiled PSO from raw_metal/kernels/*.metal. Many
// kinds share a PSO (all Qmv* -> affine_qmv_fast; Rms/FfnRms/QNorm/KNorm/FinalRms ->
// rms_single_row; Rope/RopeK -> rope_neox_decode; Residual/LayerOut -> residual_add), so we
// compile each distinct (file, entrypoint) ONCE and fan it out to every kind that uses it.
//
// Activation dtype T = bf16 (delta's M=1 ports: core_out/activation dtype = bfloat); the
// recurrent/conv GDN state is fp32 inside the kernel. lm_head + embed are 4-bit g64.

#include <string>

#include "decode_step.hpp"
#include "mtl4_context.hpp"

namespace pie_metal_driver::raw_metal {

// Compile every decode kernel from `kernels_dir` (the raw_metal/kernels directory) into
// `out`, indexed by Kernel kind. `with_argmax` controls whether the optional device-argmax
// PSO is compiled (Argmax is not yet ported; default off → it is left invalid). Returns
// false if any compile fails (the failing kernel name is written to `*err` if non-null).
bool load_decode_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      DecodeStepPsos& out,
                      bool with_argmax = false,
                      std::string* err = nullptr,
                      bool fuse_residual = false);

}  // namespace pie_metal_driver::raw_metal
