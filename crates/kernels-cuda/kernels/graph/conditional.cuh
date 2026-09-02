#pragma once

/// **THE ONE DEVICE FUNCTION A CUDA GRAPH'S CONTROL FLOW IS WRITTEN IN.**
///
/// `cudaGraphSetConditional` is the only way a graph's `IF`/`SWITCH` node
/// learns whether to take its body: the predicate is not a host branch and
/// not a node parameter, it is a store a KERNEL makes into a handle the
/// driver minted while the parent graph was being recorded. That is the whole
/// reason a conditional is worth more than the `cudaGraphNodeSetEnabled` the
/// fold reaches for — the decision rides inside the graph, so a replay under a
/// composition the recording fire never saw still decides correctly.
///
/// # The symbol, and the link stage it does NOT need
///
/// The toolkit declares it `extern __device__ __cudart_builtin__ void
/// cudaGraphSetConditional(unsigned long long, unsigned int)` in
/// `cuda_device_runtime_api.h` and defines it NOWHERE — not in any header, and
/// not in `libcudadevrt.a` (the archive was extracted and searched;
/// `.wiki/driver/new-horizon.md` §62.3). A call to an extern device function
/// with no definition in any linkable library is resolved by the DRIVER at
/// module load, whichever frontend emitted the call, and NVRTC and nvcc share
/// `cicc`. So this unit compiles like every other one here: whole-program, no
/// `--relocatable-device-code`, no `cuLink` against the device runtime. The
/// declaration is restated below rather than `#include`d because the toolkit
/// header is not in this plane's carried closure and the signature is two
/// integers.
extern "C" __device__ void cudaGraphSetConditional(
    unsigned long long handle,
    unsigned int value);

namespace pie::graph {

/// **SET ONE CONDITIONAL HANDLE FROM A ROW COUNT THE DEVICE ALREADY HOLDS.**
///
/// `indptr` is a window's rebased row CSR — `[lanes + 1]` `i32`, the vector
/// `engine_cuda::window::Window` stages once per distinct window per fire — so
/// `indptr[lanes]` is that window's row count and `indptr[lanes] != 0` is
/// exactly the zero-row rule `model_exec::fire::walk` applies on the host
/// (decision #3). One thread, one load, one store: the predicate is the
/// artifact's own semantics read from the device rather than restated.
///
/// `arm` is what makes this kernel WARMABLE. A `cudaGraphSetConditional` call
/// outside a conditional graph's launch has no handle to store into, so the
/// eager pass that compiles this unit and loads its module — which must happen
/// BEFORE the capture, since a JIT inside `cudaStreamBeginCapture` is host
/// work the thread-local mode refuses — fires it with `arm = 0` and reaches
/// the early return. The captured launch fires it with `arm = 1`.
///
/// A null `indptr` is the window this fire has no vector for, and it reads as
/// `absent`: the caller states what an absent window means rather than having
/// this kernel guess, because "no rows" and "no table" are not the same
/// sentence.
///
/// # **THE `win` GUARD, AND WHY THE LANE COUNT COULD NOT STAY AN IMMEDIATE**
///
/// `set_conditional_byte` below says the rule this kernel had to learn: an
/// immediate is a NODE PARAMETER, frozen at capture, which is the host branch
/// a conditional exists to replace. `indptr` was a pointer for exactly that
/// reason and `lanes` was not — and `lanes` is a LANE COUNT, which is the one
/// number a `record::BodyKey` deliberately does not fix (it fixes each present
/// class's row RUNG, a bound on the lane count and not the lane count). So a
/// body captured at one split replayed its predicate at another: the setter
/// read `indptr[lanes_at_capture]` out of a CSR holding `lanes_at_fire + 1`
/// real bounds and a zeroed tail, read the zero as "this region has no rows",
/// and SKIPPED THE BODY. The region never ran, its output rectangle kept the
/// bytes the fire before it left there, and two identical fires answered
/// differently — coherent, deterministic, and with no fault anywhere.
///
/// `win` is the region's staged live-geometry seat, whose word 2 is this
/// fire's own lane count for this window (`engine_cuda::window::Windows::live`
/// writes `[rows, row_offset, lanes, lane_offset]`, and
/// `engine_cuda::window::Cursor::count_of` hands the address). It is the same
/// `win` guard every seated point in this tree takes, for the same reason and
/// with the same null: a fire that staged no seat — which is every fire off
/// the bodies path, and those are never captured — falls back to the
/// immediate, which is its own lane count and always was.
__global__ void set_conditional(
    unsigned long long handle,
    const int* __restrict__ indptr,
    int lanes,
    unsigned int absent,
    int arm,
    const unsigned int* __restrict__ win)
{
    if (arm == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // THE FIRE'S LANE COUNT, NOT THE CAPTURE'S. `win[2]` is this window's
    // live lanes, staged per fire at an address the graph may bake.
    const int at = (win != nullptr) ? static_cast<int>(win[2]) : lanes;

    unsigned int value = absent;
    if (indptr != nullptr && at >= 0) {
        value = (indptr[at] != 0) ? 1u : 0u;
    }
    cudaGraphSetConditional(handle, value);
}

/// The same store, from a value the CALLER states rather than one the device
/// reads — the synthetic form a gate drives both arms of, and the form a
/// predicate that is not a row count would take.
///
/// `live` is a device byte: non-zero takes the body. It is a POINTER and not
/// an immediate on purpose — an immediate would be a node parameter, frozen at
/// capture, which is the host branch a conditional exists to replace.
__global__ void set_conditional_byte(
    unsigned long long handle,
    const unsigned char* __restrict__ live,
    unsigned int absent,
    int arm)
{
    if (arm == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const unsigned int value = (live == nullptr) ? absent : ((*live != 0) ? 1u : 0u);
    cudaGraphSetConditional(handle, value);
}

}

namespace pie::graph {

/// **THE SWITCH FORM, AND IT STORES AN INDEX RATHER THAN A BOOL.**
///
/// An `IF` handle takes 0 or 1 and the node has one body. A `SWITCH` handle
/// takes an ARM INDEX in `0..arms` and the node has `arms` of them, so the
/// question the predicate answers stops being "does this region run" and
/// becomes "which of these regions does". The driver's own rule for an index
/// at or past `arms` is that NO body runs, which is what makes the whole
/// group's empty case expressible: nothing has to be stored for nothing to
/// happen.
///
/// **ONE LAUNCH PER ARM, AND THE STORE IS CONDITIONAL ON THE ARM BEING LIVE.**
/// A group's arms are `arms` separate regions with `arms` separate windows,
/// and there is no one vector holding all their counts — building one would be
/// a per-fire device buffer nothing else needs. So the recorder launches this
/// once per arm, each with that arm's own boundary vector, and each stores
/// only if its own window has rows. That composes because P3 proves what makes
/// a SWITCH legal in the first place: **at most one arm is demanded by any
/// composition the budgets admit** (`switch_groups`, which only ever groups at
/// `max_lanes == 1`). Under that proof at most one of these launches stores,
/// so the order they land in cannot matter — and if the proof were ever wrong
/// the last one would win, which is a wrong answer rather than a corrupt
/// graph.
///
/// A null `indptr` stores nothing: this arm does not claim the group. The
/// recorder never passes one — it refuses a SWITCH whose arm cannot state a
/// row count, because "take it anyway" is the safe direction for an `IF` and
/// there is no such direction here — so the null is the gate's spelling of an
/// arm that stands down.
///
/// **AND THE `win` GUARD IS `set_conditional`'s, FOR ITS REASON.** A SWITCH
/// group only forms at `max_lanes == 1` today (`switch_groups`), so the stale
/// immediate that broke the `IF` could not reach this arm — the lane count is
/// one at capture and one at every replay. The seat is taken anyway: what
/// makes the number safe here is a property of the GROUPING RULE, not of this
/// kernel, and a predicate whose correctness rests on a bound somebody else
/// may widen is a predicate that will be wrong quietly.
__global__ void set_switch(
    unsigned long long handle,
    unsigned int arm,
    const int* __restrict__ indptr,
    int lanes,
    int armed,
    const unsigned int* __restrict__ win)
{
    if (armed == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int at = (win != nullptr) ? static_cast<int>(win[2]) : lanes;
    if (indptr == nullptr || at < 0) return;
    if (indptr[at] == 0) return;

    cudaGraphSetConditional(handle, arm);
}

}
