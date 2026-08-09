//===-- merge_states.cuh ------------------------------------------*- CUDA -*-===//
//
// The cascade merge lattice's NVRTC root: three element types and nothing
// else.
//
// # What this file is
//
// `#include <flashinfer/attention/cascade.cuh>` plus three `using`
// declarations. It holds no `__global__` of ours, no launcher, no `<<<>>>`
// and no host function -- the three `__global__`s it exists to instantiate
// are upstream's
// (`flashinfer::MergeStatesKernel`,
// `flashinfer::MergeStatesLargeNumIndexSetsKernel`,
// `flashinfer::PersistentVariableLengthMergeStatesKernel`) and the host
// arithmetic that used to surround them -- `MergeStates` at
// `cascade.cuh:637-668` and `VariableLengthMergeStates` at `:686-736` -- is
// `driver-cuda/src/fire/merge_states.rs` in Rust.
//
// `attn/fa2.cuh` beside this file is the same shape for the same reason and
// its header carries the long form of the argument. This one is shorter
// because the three kernels take ordinary scalars and pointers rather than a
// params struct, so there is no aggregate whose layout has to be mirrored and
// no `__device__` echo of a `sizeof` to export.
//
// # Why this is `csrc/src/cascade/` and not `csrc/src/attn/`
//
// By role it belongs beside `fa2.cuh`. It is not there because
// `csrc/src/attn/**` is owned by a concurrent agent for the whole of this
// session, and a new file in a directory somebody else is rewriting is a
// merge nobody asked for. The move is one `git mv`, one `sig.file` string and
// one `Unit::name`; the three are checked against each other by
// `tests/layers.rs`' `a_row_lives_in_the_unit_that_compiles_it`, so the
// rename cannot be done half way.
//
// # The vendored copy, which is the whole point
//
// The include below resolves against the CARRIED set --
// `csrc/vendor/flashinfer/attention/cascade.cuh`, walked into `carried.rs`
// and handed to NVRTC as `includeNames[]`. It is not the CPM checkout: no
// `-I` anywhere in this repository ever put `csrc/vendor` in front of a C++
// compiler, and the launcher that used to read this text read
// `${flashinfer_SOURCE_DIR}` instead. The two copies are the same upstream
// bytes, which is what kept the distinction invisible while both existed.
// Only one of them exists now, and it is this one.
//
// The unit therefore demands `Headers::LibraryAndVendor` (`unit.rs`'s
// `DEMANDS`), exactly as `attn/fa2_*` does and for the same reason.

#include <cuda_bf16.h>
#include <cstdint>

#include <flashinfer/attention/cascade.cuh>

namespace pie_cuda_driver::kernels::cascade {

// The element types. pie's attention output is bf16 and its partial
// log-sum-exps are `float` -- `S`, `s_merged` and `tmp_s` are `float*` in
// upstream's signatures and are not an axis -- so `DTypeIn` and `DTypeO` are
// the same type here and there is no dtype axis in this lattice.
//
// `attn/fa2.cuh`'s `DTypeO` is the same `__nv_bfloat16`, and it has to be:
// the buffers these kernels merge are the ones an FA2 split fire wrote.
using DTypeIn = __nv_bfloat16;
using DTypeO = __nv_bfloat16;

// `IdType` for the variable-length kernel's `indptr`. `std::int32_t`, matching
// `plan::prefill`'s `merge_indptr: Vec<i32>` and `plan::decode`'s `o_indptr`,
// which are the two arrays that are ever passed here.
using IdType = std::int32_t;

}  // namespace pie_cuda_driver::kernels::cascade
