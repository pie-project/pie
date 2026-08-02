// Explicit instantiation of every FA2 kernel head_dim=128 needs.
//
// One translation unit per head_dim, because cicc and ptxas are
// single-threaded per TU; see attention_flashinfer_common.cuh. Adding or
// removing a head_dim means editing src/kernels.def and adding/removing the
// matching file here -- CMakeLists.txt checks the two agree.
#include "ops/attention_flashinfer_common.cuh"

namespace pie_cuda_driver::ops {

template struct AttnHd<128>;

}  // namespace pie_cuda_driver::ops
