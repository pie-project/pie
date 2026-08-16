//! Differential test: this port against the `scheduler.cuh` it replaces.
//!
//! # Why bytes and not behaviour
//!
//! A work partitioner has no observable behaviour short of the kernel running.
//! Its output is a `PlanInfo` struct and a block of `int32_t`s that a
//! `__global__` reads without checking — so the only honest question to ask a
//! port of it is whether it produces **the same bytes** as the C++ it replaces,
//! over inputs that look like real batches. Anything weaker (does it look
//! balanced? do the indices stay in range?) passes for a planner that assigns
//! every request to CTA 0.
//!
//! So this test compiles the real `flashinfer/attention/scheduler.cuh` with
//! `nvcc`, runs the same four entry points on the same inputs, and compares the
//! `ToVector()` of the `PlanInfo` and every byte of the staged upload.
//!
//! # How the C++ runs without a GPU
//!
//! `scheduler.cuh` is host code, but it makes four CUDA runtime calls:
//! `cudaGetDevice`, `cudaDeviceGetAttribute`, `cudaMemcpyAsync` and
//! `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. The harness `#define`s all
//! four to host functions **before** including the header, so the device facts
//! become injected constants and the H2D copy becomes a `memcpy` into a host
//! buffer. That is what lets the same case run against a chosen SM count and
//! compute capability rather than whatever card the test machine has — the
//! sweep covers an L40S, an H100 and a deliberately tiny 8-SM device without
//! any of them being present.
//!
//! `nvcc` is still needed to *parse* the header (it is a `.cuh` full of
//! `__device__` declarations in the tree it includes). It never launches
//! anything: the run takes about 10 ms and touches no driver.
//!
//! # One process per case
//!
//! Two of the four planners divide by `batch_size` without checking it, so the
//! empty batch is a `SIGFPE` in the C++ — not an exception, not an error code.
//! Running each case in its own process turns that into an observation ("the
//! C++ died here, and the port refuses instead") rather than a dead test
//! binary. It costs a fork per case and buys the ability to put the empty batch
//! in the sweep at all.
//!
//! # If nvcc is not there
//!
//! [`differential_parity`] is skipped, loudly, and [`hand_derived_fallback`]
//! runs instead — a much weaker result, and labelled as one: it checks a
//! handful of hand-computed schedules, which proves the port is self-consistent
//! and proves nothing about parity. The count of cases and the pass rate are
//! printed either way; run with `--nocapture` to see them.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use kernels_cuda::attn::plan::{decode, mla, prefill, sm90, Device, Workspace};

// ---------------------------------------------------------------------------
// the harness
// ---------------------------------------------------------------------------

/// The C++ driver, compiled once and run once per case.
///
/// Kept as a string rather than a checked-in `.cu` so that the file this test
/// compiles cannot drift from the test that describes it, and so the crate
/// ships no C++ of its own — which is the point of the exercise.
///
/// It is deliberately small: read a spec of whitespace-separated integers, call
/// one of the four entry points **our own code calls** (`DecodePlan`,
/// `PrefillPlan`, `PrefillSM90Plan`, `MLAPlan`, plus the two
/// `...WorkspaceSize` twins), and dump the `PlanInfo` and the staged bytes. No
/// judgement lives here; the comparison is all on the Rust side.
const HARNESS: &str = r##"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
static int g_num_sm = 132, g_cc_major = 9, g_cc_minor = 0, g_blocks_per_sm = 1;
static size_t g_memcpy_bytes = 0;

static cudaError_t fake_get_device(int* d) { *d = 0; return cudaSuccess; }
static cudaError_t fake_get_attribute(int* v, cudaDeviceAttr a, int) {
  switch (a) {
    case cudaDevAttrMultiProcessorCount: *v = g_num_sm; break;
    case cudaDevAttrComputeCapabilityMajor: *v = g_cc_major; break;
    case cudaDevAttrComputeCapabilityMinor: *v = g_cc_minor; break;
    default: *v = 0;
  }
  return cudaSuccess;
}
static cudaError_t fake_memcpy(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
  g_memcpy_bytes = n;
  memcpy(dst, src, n);
  return cudaSuccess;
}
// The kernel argument is dropped, so it is never ODR-used and never needs a
// device image.
static cudaError_t fake_occupancy(int* n, int, size_t) { *n = g_blocks_per_sm; return cudaSuccess; }

#define cudaGetDevice(p) fake_get_device(p)
#define cudaDeviceGetAttribute(v, a, d) fake_get_attribute((v), (a), (d))
#define cudaMemcpyAsync(d, s, n, k, st) fake_memcpy((d), (s), (n), (k), (st))
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor(n, kern, t, s) fake_occupancy((n), (t), (s))

// The fakes must be in place before the *first* flashinfer header, not just
// before `scheduler.cuh`: `utils.cuh` defines `GetCudaComputeCapability()`,
// `pos_enc.cuh` pulls it in, and a compute capability read from the real card
// picks a different `cta_tile_q` -- which is exactly the divergence this
// caught on the Turing cases before the includes were reordered.
#include <flashinfer/pos_enc.cuh>

// The decode work estimator takes the address of the decode kernel to ask the
// occupancy API about it. We fake the occupancy API, but the address still has
// to name something, so here is an empty kernel with the right template
// signature. It is never launched and never compiled for a device.
namespace flashinfer {
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {}
}  // namespace flashinfer

#include <flashinfer/attention/scheduler.cuh>

struct HP {
  using DTypeQ = __nv_bfloat16;
  using DTypeKV = __nv_bfloat16;
  using DTypeO = __nv_bfloat16;
  using IdType = int32_t;
};
struct HV {};

using flashinfer::PosEncodingMode;

static std::vector<long long> g_spec;
static size_t g_pos = 0;
static long long next_int() {
  if (g_pos >= g_spec.size()) { fprintf(stderr, "spec ran out\n"); exit(3); }
  return g_spec[g_pos++];
}
static std::vector<int32_t> next_ints(size_t n) {
  std::vector<int32_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = (int32_t)next_int();
  return v;
}

static FILE* g_out = nullptr;
static void put_u64(unsigned long long v) { fwrite(&v, 8, 1, g_out); }
static void dump_plan(const std::vector<int64_t>& info, size_t int_bytes, const void* buf) {
  put_u64(0);
  put_u64(info.size());
  for (int64_t v : info) put_u64((unsigned long long)v);
  put_u64(int_bytes);
  put_u64(int_bytes);
  fwrite(buf, 1, int_bytes, g_out);
}
static void dump_sizes(size_t f, size_t i) { put_u64(1); put_u64(f); put_u64(i); }
static void dump_refused() { put_u64(2); }

// The two workspaces. `int_buffer` is the H2D destination and `locked` is the
// page-locked source; upstream writes the second and copies it to the first, so
// dumping the first is dumping what would have reached the GPU. Both are zeroed
// so that the allocator's alignment padding compares equal to the port's
// zero-filled staging buffer rather than to whatever the heap held.
static std::vector<unsigned char> g_int_buf, g_locked, g_float_buf;
static size_t g_float_bytes = 0;

template <uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
static cudaError_t decode_plan(flashinfer::DecodePlanInfo& info, int32_t* indptr,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t page_size,
                               bool graph) {
  auto est = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      GROUP_SIZE, HEAD_DIM, PosEncodingMode::kNone, HV, HP>;
  return flashinfer::DecodePlan<HEAD_DIM, PosEncodingMode::kNone, HV, HP>(
      g_float_buf.data(), g_float_bytes, g_int_buf.data(), g_locked.data(), g_int_buf.size(),
      info, indptr, batch_size, num_qo_heads, page_size, graph, nullptr, est);
}

template <uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
static cudaError_t decode_size(size_t& fb, size_t& ib, int32_t* indptr, uint32_t batch_size,
                               uint32_t num_qo_heads, uint32_t page_size, bool graph) {
  auto est = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      GROUP_SIZE, HEAD_DIM, PosEncodingMode::kNone, HV, HP>;
  return flashinfer::DecodePlanWorkspaceSize<HEAD_DIM, PosEncodingMode::kNone, HV, HP>(
      fb, ib, indptr, batch_size, num_qo_heads, page_size, graph, nullptr, est);
}

#define DISPATCH_GROUP(HD, GS, ...)                                     \
  switch (GS) {                                                         \
    case 1: { constexpr uint32_t G = 1; __VA_ARGS__; break; }           \
    case 2: { constexpr uint32_t G = 2; __VA_ARGS__; break; }           \
    case 4: { constexpr uint32_t G = 4; __VA_ARGS__; break; }           \
    case 8: { constexpr uint32_t G = 8; __VA_ARGS__; break; }           \
    default: fprintf(stderr, "unhandled group size %u\n", (unsigned)GS); exit(4); \
  }

#define DISPATCH_HEAD_DIM(HD, ...)                                      \
  switch (HD) {                                                         \
    case 64: { constexpr uint32_t D = 64; __VA_ARGS__; break; }         \
    case 128: { constexpr uint32_t D = 128; __VA_ARGS__; break; }       \
    case 256: { constexpr uint32_t D = 256; __VA_ARGS__; break; }       \
    case 512: { constexpr uint32_t D = 512; __VA_ARGS__; break; }       \
    default: fprintf(stderr, "unhandled head dim %u\n", (unsigned)HD); exit(4); \
  }

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s <spec> <out>\n", argv[0]); return 2; }
  FILE* in = fopen(argv[1], "r");
  if (!in) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  long long v;
  while (fscanf(in, "%lld", &v) == 1) g_spec.push_back(v);
  fclose(in);
  g_out = fopen(argv[2], "wb");
  if (!g_out) { fprintf(stderr, "cannot write %s\n", argv[2]); return 2; }

  const int kind = (int)next_int();
  const int mode = (int)next_int();  // 0 = plan, 1 = workspace size
  g_num_sm = (int)next_int();
  g_cc_major = (int)next_int();
  g_cc_minor = (int)next_int();
  g_blocks_per_sm = (int)next_int();
  const size_t float_bytes = (size_t)next_int();
  g_float_bytes = float_bytes;
  const size_t int_bytes = (size_t)next_int();
  // The float workspace is never dereferenced -- the float allocator only ever
  // returns offsets -- so a declared size of 1 GiB is honoured with a 64-byte
  // allocation. malloc gives it 16-byte alignment, which is the one property
  // the offsets depend on.
  g_float_buf.assign(64, 0);
  g_int_buf.assign(int_bytes, 0);
  g_locked.assign(int_bytes, 0);

  try {
    if (kind == 0) {
      const uint32_t head_dim = (uint32_t)next_int();
      const uint32_t group_size = (uint32_t)next_int();
      const uint32_t num_qo_heads = (uint32_t)next_int();
      const uint32_t page_size = (uint32_t)next_int();
      const bool graph = next_int() != 0;
      const uint32_t batch_size = (uint32_t)next_int();
      auto indptr = next_ints(batch_size + 1);
      if (mode == 0) {
        flashinfer::DecodePlanInfo info;
        DISPATCH_HEAD_DIM(head_dim, DISPATCH_GROUP(D, group_size, {
          decode_plan<D, G>(info, indptr.data(), batch_size, num_qo_heads, page_size, graph);
        }))
        dump_plan(info.ToVector(), g_memcpy_bytes, g_int_buf.data());
      } else {
        size_t fb = 0, ib = 0;
        DISPATCH_HEAD_DIM(head_dim, DISPATCH_GROUP(D, group_size, {
          decode_size<D, G>(fb, ib, indptr.data(), batch_size, num_qo_heads, page_size, graph);
        }))
        dump_sizes(fb, ib);
      }
    } else if (kind == 1) {
      const uint32_t total_num_rows = (uint32_t)next_int();
      const uint32_t batch_size = (uint32_t)next_int();
      const uint32_t num_qo_heads = (uint32_t)next_int();
      const uint32_t num_kv_heads = (uint32_t)next_int();
      const uint32_t head_dim_qk = (uint32_t)next_int();
      const uint32_t head_dim_vo = (uint32_t)next_int();
      const uint32_t page_size = (uint32_t)next_int();
      const bool graph = next_int() != 0;
      const uint32_t sizeof_dtype_o = (uint32_t)next_int();
      const int32_t window_left = (int32_t)next_int();
      const int32_t fixed_split_size = (int32_t)next_int();
      const bool disable_split_kv = next_int() != 0;
      const int64_t num_colocated_ctas = (int64_t)next_int();
      auto qo = next_ints(batch_size + 1);
      auto kv = next_ints(batch_size + 1);
      if (mode == 0) {
        flashinfer::PrefillPlanInfo info;
        flashinfer::PrefillPlan<int32_t>(
            g_float_buf.data(), g_float_bytes, g_int_buf.data(), g_locked.data(),
            g_int_buf.size(), info, qo.data(), kv.data(), total_num_rows, batch_size, num_qo_heads,
            num_kv_heads, head_dim_qk, head_dim_vo, page_size, graph, sizeof_dtype_o, window_left,
            fixed_split_size, disable_split_kv, num_colocated_ctas, nullptr);
        dump_plan(info.ToVector(), g_memcpy_bytes, g_int_buf.data());
      } else {
        size_t fb = 0, ib = 0;
        flashinfer::PrefillPlanWorkspaceSize<int32_t>(
            fb, ib, qo.data(), kv.data(), total_num_rows, batch_size, num_qo_heads, num_kv_heads,
            head_dim_qk, head_dim_vo, page_size, graph, sizeof_dtype_o, window_left,
            fixed_split_size, disable_split_kv, num_colocated_ctas, nullptr);
        dump_sizes(fb, ib);
      }
    } else if (kind == 2) {
      const uint32_t total_num_rows = (uint32_t)next_int();
      const uint32_t batch_size = (uint32_t)next_int();
      const uint32_t num_qo_heads = (uint32_t)next_int();
      const uint32_t num_kv_heads = (uint32_t)next_int();
      const uint32_t head_dim_qk = (uint32_t)next_int();
      const uint32_t head_dim_vo = (uint32_t)next_int();
      const uint32_t page_size = (uint32_t)next_int();
      const bool causal = next_int() != 0;
      const bool graph = next_int() != 0;
      const uint32_t sizeof_dtype_o = (uint32_t)next_int();
      auto qo = next_ints(batch_size + 1);
      auto kv = next_ints(batch_size + 1);
      auto kv_len = next_ints(batch_size);
      flashinfer::PrefillPlanSM90Info info;
      flashinfer::PrefillSM90Plan<int32_t>(
          g_float_buf.data(), g_float_bytes, g_int_buf.data(), g_locked.data(),
          g_int_buf.size(), info, qo.data(), kv.data(), kv_len.data(), total_num_rows, batch_size,
          num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size, causal, graph,
          sizeof_dtype_o, nullptr);
      dump_plan(info.ToVector(), g_memcpy_bytes, g_int_buf.data());
    } else if (kind == 3) {
      const uint32_t batch_size = (uint32_t)next_int();
      const uint32_t num_heads = (uint32_t)next_int();
      const uint32_t head_dim_o = (uint32_t)next_int();
      const bool causal = next_int() != 0;
      auto qo = next_ints(batch_size + 1);
      auto kv = next_ints(batch_size + 1);
      auto kv_len = next_ints(batch_size);
      flashinfer::MLAPlanInfo info;
      flashinfer::MLAPlan<int32_t>(g_float_buf.data(), g_float_bytes, g_int_buf.data(),
                                   g_locked.data(), g_int_buf.size(), info, qo.data(), kv.data(),
                                   kv_len.data(), batch_size, num_heads, head_dim_o, causal,
                                   nullptr);
      dump_plan(info.ToVector(), g_memcpy_bytes, g_int_buf.data());
    } else {
      fprintf(stderr, "unknown kind %d\n", kind);
      return 2;
    }
  } catch (const std::exception& e) {
    // FLASHINFER_ERROR. Which one it was is not compared -- the port's error
    // taxonomy is deliberately finer than upstream's single throw.
    fseek(g_out, 0, SEEK_SET);
    dump_refused();
  }
  fclose(g_out);
  return 0;
}
"##;

/// Where FlashInfer's headers are, if they are anywhere.
///
/// The archive crate's build script vendored them under its `OUT_DIR`; the
/// hash in that path changes with the build, and there may be more than one,
/// so this globs rather than hard-codes. `PIE_FLASHINFER_INCLUDE` overrides
/// it.
///
/// That crate is deleted, so the walk below can only find what an older build
/// left under `target/`. The `kernels-cuda-` prefix is NOT what tells the two
/// apart and never was — it matched this crate's own build directories too,
/// back when it had a build script — the `out/kernels-cuda/build/` component
/// is.
fn flashinfer_src() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("PIE_FLASHINFER_INCLUDE") {
        let dir = PathBuf::from(dir);
        return dir.join("include/flashinfer/attention/scheduler.cuh").exists().then_some(dir);
    }
    let target = workspace_target()?;
    for profile in std::fs::read_dir(&target).ok()? {
        let build = profile.ok()?.path().join("build");
        let Ok(entries) = std::fs::read_dir(&build) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if !name.to_string_lossy().starts_with("kernels-cuda-") {
                continue;
            }
            let src = entry
                .path()
                .join("out/kernels-cuda/build/_deps/flashinfer-src");
            if src.join("include/flashinfer/attention/scheduler.cuh").exists() {
                return Some(src);
            }
        }
    }
    None
}

/// The workspace `target/` directory, walked up from this test's scratch dir.
fn workspace_target() -> Option<PathBuf> {
    let mut dir: &Path = Path::new(env!("CARGO_TARGET_TMPDIR"));
    while let Some(parent) = dir.parent() {
        if parent.file_name().is_some_and(|n| n == "target") {
            return Some(parent.to_path_buf());
        }
        dir = parent;
    }
    None
}

/// `nvcc`, wherever it is.
fn nvcc() -> Option<PathBuf> {
    for candidate in ["nvcc", "/usr/local/cuda/bin/nvcc"] {
        let path = PathBuf::from(candidate);
        if Command::new(&path).arg("--version").output().is_ok_and(|o| o.status.success()) {
            return Some(path);
        }
    }
    None
}

/// This process's scratch directory for the differential harness.
///
/// PER-PROCESS, and that is the whole point. It used to be a fixed
/// `$OUT_DIR/plan-diff`, which is fine for one test run and wrong for two: the
/// harness binary is written by `build_harness` and executed 638 times by
/// `run_case`, so a second concurrent run relinks the file the first one is
/// mid-`exec` on. That failure was caught in the act — every failure was
/// `Permission denied` on exec, with **zero numeric disagreements**, so the
/// test was reporting a scheduling accident as a parity break.
///
/// The `case{index}.spec` and `.bin` files share this directory and are the
/// worse half of the same bug: two runs at the same case index write the same
/// path, and that corrupts SILENTLY — a spec written by one process read back
/// by another compares clean or dirty for no reason either of them recorded.
///
/// The root is `CARGO_TARGET_TMPDIR` and not `OUT_DIR`, because this crate no
/// longer has a build script and `OUT_DIR` therefore does not exist. Cargo
/// sets this one for integration tests specifically so they have somewhere
/// inside `target/` to write, so the "nothing outside `target/`" property is
/// unchanged.
fn scratch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("plan-diff-{}", std::process::id()))
}

/// Compile the harness, or say why not.
///
/// The scratch directory is this test's own `CARGO_TARGET_TMPDIR`. Nothing is
/// written outside `target/`.
fn build_harness() -> Result<PathBuf, String> {
    let out = scratch_dir();
    // A recycled pid would leave someone else's harness here. Ours is the only
    // process that can be using this path, so clearing it is safe and a stale
    // binary from a previous life is not.
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    let exe = out.join("plan_harness");
    let src = out.join("plan_harness.cu");
    std::fs::write(&src, HARNESS).map_err(|e| format!("cannot write the harness: {e}"))?;

    let Some(nvcc) = nvcc() else { return Err("nvcc is not on PATH or in /usr/local/cuda".into()) };
    let Some(flashinfer) = flashinfer_src() else {
        return Err("FlashInfer's headers are not vendored under target/ -- \
                    build kernels-cuda once, or set PIE_FLASHINFER_INCLUDE"
            .into());
    };

    // `-std=c++20` and the bundled CCCL: `fastdiv.cuh` pulls `cuda::std::`
    // headers that the system CCCL in /usr/local/cuda/include does not satisfy.
    let output = Command::new(&nvcc)
        .args(["-std=c++20", "-O0", "-w"])
        .arg("-I")
        .arg(flashinfer.join("include"))
        .arg("-I")
        .arg(flashinfer.join("3rdparty/cccl/libcudacxx/include"))
        .arg("-o")
        .arg(&exe)
        .arg(&src)
        .output()
        .map_err(|e| format!("could not run {}: {e}", nvcc.display()))?;
    if !output.status.success() {
        return Err(format!(
            "nvcc refused the harness:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

// ---------------------------------------------------------------------------
// cases
// ---------------------------------------------------------------------------

/// What the C++ produced for one case, or how it failed.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// `PlanInfo::ToVector()`, the reported sizes, and the staged bytes.
    Plan { info: Vec<i64>, int_bytes: u64, upload: Vec<u8> },
    /// The sizing pass's two numbers.
    Sizes { float_bytes: u64, int_bytes: u64 },
    /// `FLASHINFER_ERROR` — a refusal, whatever it said.
    Refused,
    /// The process died on a signal. Only the empty batch does this.
    Died(String),
}

/// One differential case: a name, the spec the harness reads, and the closure
/// that runs the port on the same inputs.
struct Case {
    name: String,
    spec: String,
    rust: Box<dyn Fn() -> Outcome>,
}

/// The `Device` a case runs against, as the harness's three injected facts.
#[derive(Clone, Copy)]
struct Fake {
    num_sm: u32,
    cc_major: i32,
    cc_minor: i32,
    blocks_per_sm: u32,
}

impl Fake {
    const fn device(self) -> Device {
        Device::new(self.num_sm, self.cc_major)
    }
    fn header(self, kind: u32, mode: u32, ws: Workspace) -> String {
        format!(
            "{kind} {mode} {} {} {} {} {} {}",
            self.num_sm, self.cc_major, self.cc_minor, self.blocks_per_sm, ws.float_bytes,
            ws.int_bytes
        )
    }
}

/// L40S: what the profiling in `new-horizon.md` was measured on.
const L40S: Fake = Fake { num_sm: 142, cc_major: 8, cc_minor: 9, blocks_per_sm: 2 };
/// H100: the only device the SM90 planner is ever asked about.
const H100: Fake = Fake { num_sm: 132, cc_major: 9, cc_minor: 0, blocks_per_sm: 1 };
/// A device with fewer SMs than a batch has requests, which is where the
/// splitters stop splitting and the schedulers start doubling up.
const TINY: Fake = Fake { num_sm: 8, cc_major: 8, cc_minor: 0, blocks_per_sm: 1 };
/// Turing, the one compute capability that takes the other branch of
/// `FA2DetermineCtaTileQ`.
const TURING: Fake = Fake { num_sm: 68, cc_major: 7, cc_minor: 5, blocks_per_sm: 3 };

fn ints(values: &[i32]) -> String {
    let mut s = String::new();
    for v in values {
        let _ = write!(s, " {v}");
    }
    s
}

/// A workspace big enough that no planner in the sweep refuses.
const BIG: Workspace = Workspace::new(1 << 30, 1 << 24);

// --- decode ---------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_case(
    name: &str,
    fake: Fake,
    ws: Workspace,
    kv_indptr: Vec<i32>,
    num_qo_heads: u32,
    gqa_group_size: u32,
    page_size: u32,
    head_dim: u32,
    graph: bool,
    sizing: bool,
) -> Case {
    let batch_size = kv_indptr.len() as u32 - 1;
    let spec = format!(
        "{} {head_dim} {gqa_group_size} {num_qo_heads} {page_size} {} {batch_size}{}",
        fake.header(0, u32::from(sizing), ws),
        u32::from(graph),
        ints(&kv_indptr)
    );
    let max_grid_size = fake.blocks_per_sm * fake.num_sm;
    let rust = move || {
        let req = decode::Request {
            kv_indptr: &kv_indptr,
            batch_size,
            num_qo_heads,
            gqa_group_size,
            page_size,
            head_dim,
            enable_cuda_graph: graph,
        };
        if sizing {
            match decode::workspace_size(&req, max_grid_size) {
                Ok(s) => Outcome::Sizes {
                    float_bytes: s.float_bytes as u64,
                    int_bytes: s.int_bytes as u64,
                },
                Err(_) => Outcome::Refused,
            }
        } else {
            match decode::plan(&req, max_grid_size, ws) {
                Ok(p) => Outcome::Plan {
                    info: p.info.to_vector().to_vec(),
                    int_bytes: p.int_bytes as u64,
                    upload: p.int_upload,
                },
                Err(_) => Outcome::Refused,
            }
        }
    };
    Case { name: name.into(), spec, rust: Box::new(rust) }
}

// --- prefill --------------------------------------------------------------

#[derive(Clone)]
struct PrefillShape {
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim_vo: u32,
    page_size: u32,
    graph: bool,
    window_left: i32,
    fixed_split_size: i32,
    disable_split_kv: bool,
    num_colocated_ctas: i64,
}

impl PrefillShape {
    fn new(qo_indptr: Vec<i32>, kv_indptr: Vec<i32>) -> Self {
        Self {
            qo_indptr,
            kv_indptr,
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim_vo: 128,
            page_size: 16,
            graph: false,
            window_left: -1,
            fixed_split_size: -1,
            disable_split_kv: false,
            num_colocated_ctas: 0,
        }
    }
}

fn prefill_case(name: &str, fake: Fake, ws: Workspace, shape: PrefillShape, sizing: bool) -> Case {
    let batch_size = shape.qo_indptr.len() as u32 - 1;
    let total_num_rows = *shape.qo_indptr.last().expect("qo_indptr is never empty") as u32;
    let spec = format!(
        "{} {total_num_rows} {batch_size} {} {} 128 {} {} {} 2 {} {} {} {}{}{}",
        fake.header(1, u32::from(sizing), ws),
        shape.num_qo_heads,
        shape.num_kv_heads,
        shape.head_dim_vo,
        shape.page_size,
        u32::from(shape.graph),
        shape.window_left,
        shape.fixed_split_size,
        u32::from(shape.disable_split_kv),
        shape.num_colocated_ctas,
        ints(&shape.qo_indptr),
        ints(&shape.kv_indptr),
    );
    let device = fake.device();
    let rust = move || {
        let req = prefill::Request {
            qo_indptr: &shape.qo_indptr,
            kv_indptr: &shape.kv_indptr,
            total_num_rows,
            batch_size,
            num_qo_heads: shape.num_qo_heads,
            num_kv_heads: shape.num_kv_heads,
            head_dim_qk: 128,
            head_dim_vo: shape.head_dim_vo,
            page_size: shape.page_size,
            enable_cuda_graph: shape.graph,
            sizeof_dtype_o: 2,
            window_left: shape.window_left,
            fixed_split_size: shape.fixed_split_size,
            disable_split_kv: shape.disable_split_kv,
            num_colocated_ctas: shape.num_colocated_ctas,
        };
        if sizing {
            match prefill::workspace_size(&req, &device) {
                Ok(s) => Outcome::Sizes {
                    float_bytes: s.float_bytes as u64,
                    int_bytes: s.int_bytes as u64,
                },
                Err(_) => Outcome::Refused,
            }
        } else {
            match prefill::plan(&req, &device, ws) {
                Ok(p) => Outcome::Plan {
                    info: p.info.to_vector().to_vec(),
                    int_bytes: p.int_bytes as u64,
                    upload: p.int_upload,
                },
                Err(_) => Outcome::Refused,
            }
        }
    };
    Case { name: name.into(), spec, rust: Box::new(rust) }
}

// --- sm90 -----------------------------------------------------------------

#[derive(Clone)]
struct Sm90Shape {
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    kv_len_arr: Vec<i32>,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim_vo: u32,
    causal: bool,
    graph: bool,
}

impl Sm90Shape {
    fn new(qo_indptr: Vec<i32>, kv_len_arr: Vec<i32>) -> Self {
        let mut kv_indptr = vec![0i32];
        for len in &kv_len_arr {
            kv_indptr.push(kv_indptr.last().expect("seeded with a zero") + len);
        }
        Self {
            qo_indptr,
            kv_indptr,
            kv_len_arr,
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim_vo: 128,
            causal: true,
            graph: false,
        }
    }
}

fn sm90_case(name: &str, fake: Fake, ws: Workspace, shape: Sm90Shape) -> Case {
    let batch_size = shape.kv_len_arr.len() as u32;
    let total_num_rows = *shape.qo_indptr.last().expect("qo_indptr is never empty") as u32;
    let spec = format!(
        "{} {total_num_rows} {batch_size} {} {} 128 {} 1 {} {} 2{}{}{}",
        fake.header(2, 0, ws),
        shape.num_qo_heads,
        shape.num_kv_heads,
        shape.head_dim_vo,
        u32::from(shape.causal),
        u32::from(shape.graph),
        ints(&shape.qo_indptr),
        ints(&shape.kv_indptr),
        ints(&shape.kv_len_arr),
    );
    let device = fake.device();
    let rust = move || {
        let req = sm90::Request {
            qo_indptr: &shape.qo_indptr,
            kv_indptr: &shape.kv_indptr,
            kv_len_arr: &shape.kv_len_arr,
            total_num_rows,
            batch_size,
            num_qo_heads: shape.num_qo_heads,
            num_kv_heads: shape.num_kv_heads,
            head_dim_qk: 128,
            head_dim_vo: shape.head_dim_vo,
            page_size: 1,
            causal: shape.causal,
            enable_cuda_graph: shape.graph,
            sizeof_dtype_o: 2,
        };
        match sm90::plan(&req, &device, ws) {
            Ok(p) => Outcome::Plan {
                info: p.info.to_vector().to_vec(),
                int_bytes: p.int_bytes as u64,
                upload: p.int_upload,
            },
            Err(_) => Outcome::Refused,
        }
    };
    Case { name: name.into(), spec, rust: Box::new(rust) }
}

// --- mla ------------------------------------------------------------------

fn mla_case(
    name: &str,
    fake: Fake,
    ws: Workspace,
    qo_indptr: Vec<i32>,
    kv_len_arr: Vec<i32>,
    num_heads: u32,
    causal: bool,
) -> Case {
    let batch_size = kv_len_arr.len() as u32;
    let mut kv_indptr = vec![0i32];
    for len in &kv_len_arr {
        kv_indptr.push(kv_indptr.last().expect("seeded with a zero") + len);
    }
    let spec = format!(
        "{} {batch_size} {num_heads} 512 {}{}{}{}",
        fake.header(3, 0, ws),
        u32::from(causal),
        ints(&qo_indptr),
        ints(&kv_indptr),
        ints(&kv_len_arr),
    );
    let device = fake.device();
    let rust = move || {
        let req = mla::Request {
            qo_indptr: &qo_indptr,
            kv_indptr: &kv_indptr,
            kv_len_arr: &kv_len_arr,
            batch_size,
            num_heads,
            head_dim_o: 512,
            causal,
        };
        match mla::plan(&req, &device, ws) {
            Ok(p) => Outcome::Plan {
                info: p.info.to_vector().to_vec(),
                int_bytes: p.int_bytes as u64,
                upload: p.int_upload,
            },
            Err(_) => Outcome::Refused,
        }
    };
    Case { name: name.into(), spec, rust: Box::new(rust) }
}

// ---------------------------------------------------------------------------
// the sweep
// ---------------------------------------------------------------------------

/// An `indptr` from a list of spans.
fn indptr_of(spans: &[i32]) -> Vec<i32> {
    let mut indptr = vec![0i32];
    for span in spans {
        indptr.push(indptr.last().expect("seeded with a zero") + span);
    }
    indptr
}

/// Page counts that land exactly on, and exactly one off, every boundary the
/// decode partitioner has.
///
/// The chunk floor is `max(128 / page_size, 1)` — 8 pages at `page_size = 16` —
/// and the search then divides each request's page count by a chunk size. So
/// the interesting counts are the floor and its multiples, each ±1: a request
/// of 8 pages is one chunk and a request of 9 is two, which moves every
/// subsequent `o_indptr` entry. An off-by-one in a partitioner lives here and
/// nowhere else.
const PAGE_EDGES: [i32; 12] = [7, 8, 9, 15, 16, 17, 1, 2, 31, 32, 33, 8];

/// Token counts on and one off the FA2/FA3/MLA QO tile widths (16, 64, 128,
/// 192) — the other family of boundaries, on the query side.
const TILE_EDGES: [i32; 12] = [15, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 193];

/// Every case the differential test runs.
///
/// The sweep is adversarial on purpose. Batch size 1 and 512; uniform and
/// 100:1 skewed; one 128k-token sequence beside thirty 1-token ones; lengths on
/// and one off page boundaries; head counts that divide the CTA count (8 KV
/// heads into 132 SMs does not, 4 into 8 does); a workspace one byte too small;
/// and the empty batch, which kills two of the four planners outright.
#[allow(clippy::too_many_lines)]
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();

    // --- decode ---
    let uniform_64: Vec<i32> = (0..=64).map(|i| i * 10).collect();
    let uniform_512: Vec<i32> = (0..=512).map(|i| i * 7).collect();
    let one_long_many_short: Vec<i32> = {
        let mut v = vec![0i32, 8192];
        for i in 0..31 {
            v.push(8192 + i + 1);
        }
        v
    };
    let skewed: Vec<i32> = {
        let mut v = vec![0i32];
        for i in 0..16 {
            let last = *v.last().expect("seeded");
            v.push(last + if i % 4 == 0 { 4096 } else { 3 });
        }
        v
    };
    for (dev_name, fake) in [("l40s", L40S), ("h100", H100), ("tiny", TINY), ("turing", TURING)] {
        for (shape_name, indptr) in [
            ("batch1", vec![0i32, 137]),
            ("batch1-empty-kv", vec![0i32, 0]),
            ("uniform64", uniform_64.clone()),
            ("uniform512", uniform_512.clone()),
            ("one-long-many-short", one_long_many_short.clone()),
            ("skewed", skewed.clone()),
            ("empty-batch", vec![0i32]),
            ("page-edges", indptr_of(&PAGE_EDGES)),
        ] {
            for (heads_name, (qo, gqa)) in [("gqa4", (32u32, 4u32)), ("mha", (8, 1)), ("gqa8", (64, 8))]
            {
                for graph in [false, true] {
                    for sizing in [false, true] {
                        cases.push(decode_case(
                            &format!(
                                "decode/{dev_name}/{shape_name}/{heads_name}/graph={graph}/sizing={sizing}"
                            ),
                            fake,
                            BIG,
                            indptr.clone(),
                            qo,
                            gqa,
                            16,
                            128,
                            graph,
                            sizing,
                        ));
                    }
                }
            }
        }
    }
    // A workspace so small the planner must refuse.
    cases.push(decode_case(
        "decode/l40s/starved-workspace",
        L40S,
        Workspace::new(1 << 20, 8),
        uniform_64.clone(),
        32,
        4,
        16,
        128,
        false,
        false,
    ));
    // Page size 1 (no paging) and 32 (large pages) move the chunk floor.
    for page_size in [1u32, 32] {
        cases.push(decode_case(
            &format!("decode/l40s/page{page_size}"),
            L40S,
            BIG,
            uniform_64.clone(),
            32,
            4,
            page_size,
            128,
            false,
            false,
        ));
    }
    // A 512-wide head dim only moves the float carve, which is why it is here.
    cases.push(decode_case(
        "decode/l40s/head512",
        L40S,
        BIG,
        one_long_many_short.clone(),
        32,
        4,
        16,
        512,
        false,
        false,
    ));

    // --- prefill ---
    let long_prefill = PrefillShape::new(vec![0, 4096], vec![0, 256]);
    let decode_shaped = PrefillShape::new((0..=64).collect(), (0..=64).map(|i| i * 10).collect());
    let mixed = PrefillShape::new(
        vec![0, 4096, 4097, 4098, 4099, 8195],
        vec![0, 256, 257, 258, 259, 771],
    );
    let one_off_page = PrefillShape::new(vec![0, 255, 511, 768, 1025], vec![0, 16, 32, 48, 65]);
    let tile_edges = PrefillShape::new(indptr_of(&TILE_EDGES), indptr_of(&PAGE_EDGES));
    let big_batch =
        PrefillShape::new((0..=512).map(|i| i * 3).collect(), (0..=512).map(|i| i * 2).collect());
    for (dev_name, fake) in [("l40s", L40S), ("h100", H100), ("tiny", TINY), ("turing", TURING)] {
        for (shape_name, shape) in [
            ("long", long_prefill.clone()),
            ("decode-shaped", decode_shaped.clone()),
            ("mixed", mixed.clone()),
            ("one-off-page", one_off_page.clone()),
            ("tile-edges", tile_edges.clone()),
            ("big-batch", big_batch.clone()),
            ("batch1-empty", PrefillShape::new(vec![0, 1], vec![0, 0])),
            ("empty-batch", PrefillShape::new(vec![0], vec![0])),
        ] {
            for sizing in [false, true] {
                cases.push(prefill_case(
                    &format!("prefill/{dev_name}/{shape_name}/sizing={sizing}"),
                    fake,
                    BIG,
                    shape.clone(),
                    sizing,
                ));
            }
        }
    }
    // The four inputs that bypass or bound the binary search.
    for (name, mutate) in [
        ("graph", Box::new(|s: &mut PrefillShape| s.graph = true) as Box<dyn Fn(&mut PrefillShape)>),
        ("window512", Box::new(|s: &mut PrefillShape| s.window_left = 512)),
        ("window0", Box::new(|s: &mut PrefillShape| s.window_left = 0)),
        ("fixed-split-4", Box::new(|s: &mut PrefillShape| s.fixed_split_size = 4)),
        ("fixed-split-1", Box::new(|s: &mut PrefillShape| s.fixed_split_size = 1)),
        ("no-split", Box::new(|s: &mut PrefillShape| s.disable_split_kv = true)),
        ("colocated-200", Box::new(|s: &mut PrefillShape| s.num_colocated_ctas = 200)),
        ("colocated-1000", Box::new(|s: &mut PrefillShape| s.num_colocated_ctas = 1000)),
        ("page1", Box::new(|s: &mut PrefillShape| s.page_size = 1)),
        ("head64", Box::new(|s: &mut PrefillShape| s.head_dim_vo = 64)),
        ("head256", Box::new(|s: &mut PrefillShape| s.head_dim_vo = 256)),
        ("head512", Box::new(|s: &mut PrefillShape| s.head_dim_vo = 512)),
        ("mha", Box::new(|s: &mut PrefillShape| s.num_kv_heads = 32)),
        ("gqa16", Box::new(|s: &mut PrefillShape| s.num_kv_heads = 2)),
    ] {
        for (shape_name, base) in
            [("long", &long_prefill), ("mixed", &mixed), ("decode-shaped", &decode_shaped)]
        {
            let mut shape = base.clone();
            mutate(&mut shape);
            cases.push(prefill_case(
                &format!("prefill/l40s/{shape_name}/{name}"),
                L40S,
                BIG,
                shape,
                false,
            ));
        }
    }
    cases.push(prefill_case(
        "prefill/l40s/starved-workspace",
        L40S,
        Workspace::new(1 << 30, 8),
        mixed.clone(),
        false,
    ));
    cases.push(prefill_case(
        "prefill/l40s/starved-float",
        L40S,
        Workspace::new(1024, 1 << 24),
        decode_shaped.clone(),
        false,
    ));

    // --- sm90 ---
    let sm90_uniform = Sm90Shape::new((0..=32).map(|i| i * 512).collect(), vec![4096; 32]);
    let sm90_skewed = Sm90Shape::new(
        vec![0, 8192, 8193, 8194, 8195, 8196],
        vec![131_072, 7, 7, 7, 7],
    );
    let sm90_ties = Sm90Shape::new((0..=16).map(|i| i * 256).collect(), vec![1024; 16]);
    let sm90_batch1 = Sm90Shape::new(vec![0, 4096], vec![4096]);
    let sm90_short = Sm90Shape::new((0..=48).collect(), vec![1024; 48]);
    let sm90_empty = Sm90Shape::new(vec![0], vec![]);
    let sm90_zero_len = Sm90Shape::new(vec![0, 0, 4], vec![0, 4]);
    // 128 is the FA3 tile width exactly; 127 and 129 are the two ways to miss it.
    let sm90_edges = Sm90Shape::new(indptr_of(&TILE_EDGES), TILE_EDGES.to_vec());
    for (dev_name, fake) in [("h100", H100), ("tiny", TINY), ("l40s", L40S)] {
        for (shape_name, shape) in [
            ("uniform", sm90_uniform.clone()),
            ("skewed", sm90_skewed.clone()),
            ("ties", sm90_ties.clone()),
            ("batch1", sm90_batch1.clone()),
            ("short", sm90_short.clone()),
            ("empty-batch", sm90_empty.clone()),
            ("zero-len", sm90_zero_len.clone()),
            ("tile-edges", sm90_edges.clone()),
        ] {
            cases.push(sm90_case(
                &format!("sm90/{dev_name}/{shape_name}"),
                fake,
                BIG,
                shape,
            ));
        }
    }
    for (name, mutate) in [
        ("dense", Box::new(|s: &mut Sm90Shape| s.causal = false) as Box<dyn Fn(&mut Sm90Shape)>),
        ("graph", Box::new(|s: &mut Sm90Shape| s.graph = true)),
        ("head64", Box::new(|s: &mut Sm90Shape| s.head_dim_vo = 64)),
        ("mha", Box::new(|s: &mut Sm90Shape| s.num_kv_heads = 32)),
        ("one-head", Box::new(|s: &mut Sm90Shape| { s.num_qo_heads = 1; s.num_kv_heads = 1; })),
        ("many-heads", Box::new(|s: &mut Sm90Shape| { s.num_qo_heads = 128; s.num_kv_heads = 8; })),
    ] {
        for (shape_name, base) in
            [("uniform", &sm90_uniform), ("skewed", &sm90_skewed), ("short", &sm90_short)]
        {
            let mut shape = base.clone();
            mutate(&mut shape);
            cases.push(sm90_case(&format!("sm90/h100/{shape_name}/{name}"), H100, BIG, shape));
        }
    }
    // Over 4096 works per head: the same-schedule-for-all-heads fallback.
    cases.push(sm90_case(
        "sm90/h100/huge-batch",
        H100,
        BIG,
        Sm90Shape::new((0..=5000).collect(), vec![64; 5000]),
    ));
    cases.push(sm90_case(
        "sm90/h100/starved-workspace",
        H100,
        Workspace::new(0, 64),
        sm90_uniform.clone(),
    ));

    // --- mla ---
    let mla_decode: (Vec<i32>, Vec<i32>) = ((0..=32).collect(), vec![2048; 32]);
    let mla_prefill: (Vec<i32>, Vec<i32>) =
        (vec![0, 512, 1024, 1536], vec![512, 1024, 1536]);
    let mla_one_long: (Vec<i32>, Vec<i32>) = {
        let mut qo = vec![0i32];
        let mut lens = vec![131_072i32];
        qo.push(1);
        for i in 0..31 {
            qo.push(2 + i);
            lens.push(7);
        }
        (qo, lens)
    };
    let mla_batch1: (Vec<i32>, Vec<i32>) = (vec![0, 1], vec![4096]);
    let mla_zero: (Vec<i32>, Vec<i32>) = (vec![0, 1, 2], vec![0, 0]);
    let mla_big: (Vec<i32>, Vec<i32>) = ((0..=256).collect(), vec![1024; 256]);
    // KV lengths on and one off the chunk limit's 32/64/128/192/256 steps.
    let mla_edges: (Vec<i32>, Vec<i32>) =
        ((0..=12).collect(), vec![31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]);
    for (dev_name, fake) in [("h100", H100), ("tiny", TINY), ("l40s", L40S)] {
        for (shape_name, (qo, lens)) in [
            ("decode", mla_decode.clone()),
            ("prefill", mla_prefill.clone()),
            ("one-long-many-short", mla_one_long.clone()),
            ("batch1", mla_batch1.clone()),
            ("zero-len", mla_zero.clone()),
            ("big", mla_big.clone()),
            ("chunk-edges", mla_edges.clone()),
            ("empty-batch", (vec![0], vec![])),
        ] {
            for causal in [true, false] {
                for num_heads in [128u32, 16] {
                    cases.push(mla_case(
                        &format!("mla/{dev_name}/{shape_name}/heads{num_heads}/causal={causal}"),
                        fake,
                        BIG,
                        qo.clone(),
                        lens.clone(),
                        num_heads,
                        causal,
                    ));
                }
            }
        }
    }
    cases.push(mla_case(
        "mla/h100/starved-workspace",
        H100,
        Workspace::new(1 << 26, 1024),
        mla_decode.0.clone(),
        mla_decode.1.clone(),
        128,
        true,
    ));
    cases.push(mla_case(
        "mla/h100/starved-float",
        H100,
        Workspace::new(1024, 1 << 24),
        mla_decode.0.clone(),
        mla_decode.1.clone(),
        128,
        true,
    ));

    cases
}

// ---------------------------------------------------------------------------
// running
// ---------------------------------------------------------------------------

/// Run the C++ harness on one case and decode what it wrote.
fn run_cpp(exe: &Path, dir: &Path, index: usize, spec: &str) -> Result<Outcome, String> {
    let spec_path = dir.join(format!("case{index}.spec"));
    let out_path = dir.join(format!("case{index}.bin"));
    std::fs::write(&spec_path, spec).map_err(|e| format!("cannot write the spec: {e}"))?;
    let _ = std::fs::remove_file(&out_path);

    let output = Command::new(exe)
        .arg(&spec_path)
        .arg(&out_path)
        .output()
        .map_err(|e| format!("cannot run the harness: {e}"))?;
    if !output.status.success() {
        let how = output.status.code().map_or_else(
            || format!("signal {}", signal_of(&output.status)),
            |c| format!("exit {c}"),
        );
        return Ok(Outcome::Died(how));
    }

    let bytes = std::fs::read(&out_path).map_err(|e| format!("the harness wrote nothing: {e}"))?;
    decode_dump(&bytes).ok_or_else(|| "the harness wrote a malformed dump".to_string())
}

#[cfg(unix)]
fn signal_of(status: &std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt as _;
    status.signal().unwrap_or(0)
}

#[cfg(not(unix))]
fn signal_of(_status: &std::process::ExitStatus) -> i32 {
    0
}

/// The harness's dump format: a tag, then the fields for that tag.
fn decode_dump(bytes: &[u8]) -> Option<Outcome> {
    let mut cursor = 0usize;
    let u64_at = |cursor: &mut usize| -> Option<u64> {
        let end = *cursor + 8;
        let v = u64::from_le_bytes(bytes.get(*cursor..end)?.try_into().ok()?);
        *cursor = end;
        Some(v)
    };
    match u64_at(&mut cursor)? {
        0 => {
            let n = u64_at(&mut cursor)? as usize;
            let mut info = Vec::with_capacity(n);
            for _ in 0..n {
                info.push(u64_at(&mut cursor)? as i64);
            }
            let int_bytes = u64_at(&mut cursor)?;
            let n_upload = u64_at(&mut cursor)? as usize;
            let upload = bytes.get(cursor..cursor + n_upload)?.to_vec();
            Some(Outcome::Plan { info, int_bytes, upload })
        }
        1 => {
            let float_bytes = u64_at(&mut cursor)?;
            let int_bytes = u64_at(&mut cursor)?;
            Some(Outcome::Sizes { float_bytes, int_bytes })
        }
        2 => Some(Outcome::Refused),
        _ => None,
    }
}

/// Why two outcomes differ, in the fewest words that locate it.
fn diff(cpp: &Outcome, rust: &Outcome) -> Option<String> {
    match (cpp, rust) {
        (
            Outcome::Plan { info: a, int_bytes: ab, upload: au },
            Outcome::Plan { info: b, int_bytes: bb, upload: bu },
        ) => {
            if a != b {
                let at = a
                    .iter()
                    .zip(b)
                    .position(|(x, y)| x != y)
                    .map_or_else(|| "length".to_string(), |i| format!("field {i}"));
                return Some(format!("PlanInfo differs at {at}: cpp {a:?} rust {b:?}"));
            }
            if ab != bb {
                return Some(format!("int_bytes: cpp {ab} rust {bb}"));
            }
            if au.len() != bu.len() {
                return Some(format!("upload length: cpp {} rust {}", au.len(), bu.len()));
            }
            au.iter().zip(bu).position(|(x, y)| x != y).map(|i| {
                format!(
                    "upload byte {i} (i32 index {}): cpp {:?} rust {:?}",
                    i / 4,
                    &au[i & !3..(i & !3) + 4],
                    &bu[i & !3..(i & !3) + 4]
                )
            })
        }
        (a, b) if a == b => None,
        // The empty batch: the C++ divides by zero and dies, the port refuses.
        (Outcome::Died(_), Outcome::Refused) => None,
        (a, b) => Some(format!("cpp {a:?} rust {b:?}")),
    }
}

/// The gate: every case, both implementations, byte for byte.
///
/// Prints the case count and the pass rate, and fails the moment either number
/// is not what it should be. Run with `--nocapture` to see the tally on a pass.
#[test]
fn differential_parity() {
    let exe = match build_harness() {
        Ok(exe) => exe,
        Err(why) => {
            println!(
                "SKIPPED differential_parity: {why}\n\
                 This is a weaker result than parity. `hand_derived_fallback` still ran."
            );
            return;
        }
    };
    let dir = scratch_dir();
    let cases = cases();
    let total = cases.len();
    let mut failures: Vec<String> = Vec::new();
    let (mut planned, mut sized, mut refused, mut died, mut bytes) = (0usize, 0, 0, 0, 0usize);

    for (index, case) in cases.iter().enumerate() {
        let cpp = match run_cpp(&exe, &dir, index, &case.spec) {
            Ok(outcome) => outcome,
            Err(why) => {
                failures.push(format!("{}: harness error: {why}", case.name));
                continue;
            }
        };
        match &cpp {
            Outcome::Plan { upload, .. } => {
                planned += 1;
                bytes += upload.len();
            }
            Outcome::Sizes { .. } => sized += 1,
            Outcome::Refused => refused += 1,
            Outcome::Died(_) => died += 1,
        }
        let rust = (case.rust)();
        if let Some(why) = diff(&cpp, &rust) {
            failures.push(format!("{}: {why}\n  spec: {}", case.name, case.spec));
        }
    }

    let passed = total - failures.len();
    println!(
        "differential parity against scheduler.cuh: {passed}/{total} cases \
         ({:.1}%). {planned} produced a plan ({bytes} bytes of schedule compared byte for \
         byte), {sized} a workspace size, {refused} a refusal on both sides, and {died} \
         killed the C++ outright (the empty batch, which divides by batch_size) where the \
         port refuses instead.",
        100.0 * passed as f64 / total as f64
    );
    // A harness that refused everything would otherwise pass this test with a
    // perfect score, which is the failure mode of every differential test ever
    // written.
    assert!(
        planned > total / 2 && bytes > 1 << 20,
        "the sweep did not actually plan anything: {planned} plans, {bytes} bytes"
    );
    assert!(
        failures.is_empty(),
        "{} of {total} cases disagree with the C++:\n{}",
        failures.len(),
        failures.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
    );
    // Only on the way out clean. A failing run leaves its harness and its
    // per-case specs on disk, which is the one time anyone wants to look at
    // them; the assertions above have already returned by then.
    let _ = std::fs::remove_dir_all(&dir);
}

/// Hand-derived expectations, for when nvcc is not there.
///
/// **This is not parity.** It is a small set of schedules worked out by hand
/// from the C++ source, and it will keep passing if the port and the hand
/// derivation are wrong in the same way. It exists so that a machine with no
/// CUDA toolkit still fails loudly on an obvious regression, not so that the
/// differential test can be skipped.
#[test]
fn hand_derived_fallback() {
    // 64 requests of 10 pages each, 8 KV heads, a 32-CTA grid: 64 * 8 = 512
    // CTAs wanted against 32 available, so no split, and every request is one
    // work item whose chunk size is the longest request.
    let indptr: Vec<i32> = (0..=64).map(|i| i * 10).collect();
    let req = decode::Request {
        kv_indptr: &indptr,
        batch_size: 64,
        num_qo_heads: 32,
        gqa_group_size: 4,
        page_size: 16,
        head_dim: 128,
        enable_cuda_graph: false,
    };
    let est = decode::estimate(&req, 32).expect("a full grid is not a refusal");
    assert!(!est.split_kv);
    assert_eq!((est.new_batch_size, est.kv_chunk_size_in_pages, est.gdy), (64, 10, 8));

    // One 4096-page request on a 1024-CTA grid: 4096/32 = 128 chunks x 8 heads
    // is exactly 1024, so 32 pages is the smallest chunk that fits.
    let lone = [0i32, 4096];
    let est = decode::estimate(&decode::Request { kv_indptr: &lone, batch_size: 1, ..req }, 1024)
        .expect("a lone request is not a refusal");
    assert!(est.split_kv);
    assert_eq!((est.new_batch_size, est.kv_chunk_size_in_pages), (128, 32));

    // FA2's tile width, worked out from `FA2DetermineCtaTileQ`: a 4096-token
    // prefill with GQA 4 packs to 16384 rows, so 128-row tiles, so 128 tiles.
    let qo = [0i32, 4096];
    let kv = [0i32, 256];
    let split = prefill::split_qo_kv_indptr(
        &prefill::Request {
            qo_indptr: &qo,
            kv_indptr: &kv,
            total_num_rows: 4096,
            batch_size: 1,
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim_qk: 128,
            head_dim_vo: 128,
            page_size: 16,
            enable_cuda_graph: false,
            sizeof_dtype_o: 2,
            window_left: -1,
            fixed_split_size: -1,
            disable_split_kv: false,
            num_colocated_ctas: 0,
        },
        35,
        8,
    )
    .expect("a lone prefill is not a refusal");
    assert_eq!((split.cta_tile_q, split.new_batch_size), (128, 128));
    assert_eq!(split.o_indptr, vec![0, 4096]);

    // MLA's cluster shape: 128 heads x 1 token is 128 packed rows, above 64, so
    // two-CTA clusters and 128-row tiles.
    let mla_qo: Vec<i32> = (0..=8).collect();
    let mla_kv: Vec<i32> = (0..=8).map(|i| i * 1024).collect();
    let lens = vec![1024i32; 8];
    let sched = mla::schedule(
        &mla::Request {
            qo_indptr: &mla_qo,
            kv_indptr: &mla_kv,
            kv_len_arr: &lens,
            batch_size: 8,
            num_heads: 128,
            head_dim_o: 512,
            causal: true,
        },
        &Device::new(132, 9),
    )
    .expect("an MLA decode batch is not a refusal");
    assert_eq!((sched.cluster_size, sched.cluster_tile_q, sched.num_clusters), (2, 128, 66));

    println!(
        "hand-derived fallback: 4 schedules checked against values derived from the C++ by \
         hand. THIS IS NOT PARITY -- see differential_parity for the real gate."
    );
}

/// The empty batch, which is the one place this port deliberately differs.
///
/// Recorded as a test rather than a comment because it is the only behavioural
/// deviation in the module and it should fail loudly if someone "fixes" it into
/// silently returning an empty plan.
#[test]
fn the_empty_batch_is_refused_where_the_cpp_divides_by_zero() {
    let empty = [0i32];
    let none: [i32; 0] = [];
    assert!(prefill::split_qo_kv_indptr(
        &prefill::Request {
            qo_indptr: &empty,
            kv_indptr: &empty,
            total_num_rows: 0,
            batch_size: 0,
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim_qk: 128,
            head_dim_vo: 128,
            page_size: 16,
            enable_cuda_graph: false,
            sizeof_dtype_o: 2,
            window_left: -1,
            fixed_split_size: -1,
            disable_split_kv: false,
            num_colocated_ctas: 0,
        },
        35,
        8,
    )
    .is_err());
    assert!(mla::schedule(
        &mla::Request {
            qo_indptr: &empty,
            kv_indptr: &empty,
            kv_len_arr: &none,
            batch_size: 0,
            num_heads: 128,
            head_dim_o: 512,
            causal: true,
        },
        &Device::new(132, 9),
    )
    .is_err());

    // Decode and SM90 do not divide by the batch size, and must survive it.
    let plan = decode::plan(
        &decode::Request {
            kv_indptr: &empty,
            batch_size: 0,
            num_qo_heads: 32,
            gqa_group_size: 4,
            page_size: 16,
            head_dim: 128,
            enable_cuda_graph: false,
        },
        1024,
        Workspace::new(1 << 20, 1 << 20),
    )
    .expect("decode survives an empty batch");
    assert_eq!(plan.info.padded_batch_size, 0);
    assert!(sm90::schedule(
        &sm90::Request {
            qo_indptr: &empty,
            kv_indptr: &empty,
            kv_len_arr: &none,
            total_num_rows: 0,
            batch_size: 0,
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim_qk: 128,
            head_dim_vo: 128,
            page_size: 1,
            causal: true,
            enable_cuda_graph: false,
            sizeof_dtype_o: 2,
        },
        &Device::new(132, 9),
    )
    .is_ok());
}
