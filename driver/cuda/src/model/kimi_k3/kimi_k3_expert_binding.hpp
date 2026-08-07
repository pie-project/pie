#pragma once

// Which form of routed experts a Kimi-K3 MoE layer binds to.
//
// This is the whole CPU-observable contract of `bind_kimi_k3`'s MoE half --
// both forms optional, never neither, arity checked against the config -- and
// it is a header of its own because that is what makes it assertable. The
// decision itself is arithmetic on four facts, but the values those facts are
// read from are not constructible without a device: `LoadedModel`'s state is
// private with `load()` from a checkpoint as its only populated path, and
// `GroupStreamCache`'s constructor allocates a slab and creates a CUDA event.
// A test that had to build either would be a GPU test, and this decision is
// not a GPU decision.
//
// So: no CUDA includes here, and none in the translation unit that defines it.

#include <cstdint>
#include <string>

namespace pie_cuda_driver::model {

/// True when this layer's experts are streamed rather than stacked.
///
/// Throws when the contract published a form the driver cannot serve:
///
///   * neither form -- a streaming contract publishes a group *instead of*
///     the stacks, so the absence of both means the contract published
///     nothing this layer's MoE can read. Before the group existed this was
///     a bare "per-expert MXFP4 GEMV is not implemented"; the streamed
///     per-expert loop is now that implementation, so the refusal narrows to
///     the case where there is genuinely nothing to loop over.
///   * half the stacks -- `gate_up` without `down` or the reverse. Not a
///     streamable state and not a stackable one; falling through to the
///     group would silently drop the half that was published.
///   * a group whose arity disagrees with `num_experts` -- the forward
///     indexes it by expert id straight out of the router, so a short group
///     is an out-of-range page-in and a long one is a bank the router can
///     never reach the tail of.
bool kimi_k3_use_streamed_experts(
    bool gate_up_stack_present,
    bool down_stack_present,
    bool group_found,
    std::uint32_t group_arity,
    int num_experts,
    int layer,
    const std::string& group_name);

}  // namespace pie_cuda_driver::model
