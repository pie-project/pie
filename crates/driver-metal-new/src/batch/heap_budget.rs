//! The engine's heap budget: how much a family needs on top of its
//! weights, answered BEFORE the context exists, because it sizes it —
//! and the max-rows question answered by bisecting the budget itself.
//!
//! The budget sums the same pure pieces the engine will stage: the KV
//! region at the family's own layout, and the pool from the SAME
//! colouring at the SAME padded row counts. A guessed per-row price
//! goes stale — the C++ ships one that charged every prefill row a
//! `vocab × 2` logits slice, true before `RowGather` and not since —
//! and bisecting the real function cannot, because when the DAG
//! changes the function it bisects changes with it.

use crate::tuning::Tuning;

use super::dispatch_gemma4::gemma4_pool_elems;
use super::dispatch_gptoss::{gptoss_pool_elems, gptoss_qmm_pool_rows};
use super::dispatch_llama::{llama_pool_elems, llama_qmm_pool_rows};
use super::gemma4::Gemma4Geometry;
use super::geometry::DecodeGeometry;
use super::gptoss::GptOssGeometry;
use super::llama::LlamaGeometry;

/// The fewest rows a paged fire is allowed to budget down to.
pub const PAGED_MIN_FORWARD_TOKENS: u32 = 64;
/// The most rows any fire may carry, budget permitting.
pub const PAGED_MAX_FORWARD_TOKENS: u32 = 4096;

/// KV + pool + logits + constants, with slack — deliberately generous:
/// this is a budget, and a context that is too small fails at
/// allocation with no diagnosis of which one ran out.
const BASE_SLACK: u64 = 256 << 20;

fn clamp_sampled(rows: u32, requests: u32) -> u32 {
    requests.max(1).min(rows.max(1))
}

/// The llama family's extra-heap bytes for a fire of up to `max_tokens`
/// rows sampling up to `max_requests`.
#[must_use]
pub fn llama_extra_heap_bytes(
    g: &LlamaGeometry,
    tuning: &Tuning,
    max_ctx: u32,
    max_tokens: u32,
    max_requests: u32,
) -> u64 {
    let mut bytes = BASE_SLACK;
    // The KV region: per layer, both sides, plus one page's slack per
    // layer per side — the engine pages, so max_ctx rounds up.
    let per_layer = u64::from(g.n_kv_heads) * (u64::from(max_ctx) + 32) * u64::from(g.head_dim) * 2;
    bytes += u64::from(g.n_layers) * 2 * per_layer;
    let rows = llama_qmm_pool_rows(max_tokens);
    let sampled = llama_qmm_pool_rows(clamp_sampled(max_tokens, max_requests));
    for e in llama_pool_elems(g, tuning, rows, sampled) {
        bytes += e * 2;
    }
    bytes += u64::from(sampled) * u64::from(g.vocab) * 2; // the logits slot
    bytes
}

/// The gpt-oss family's extra-heap bytes.
#[must_use]
pub fn gptoss_extra_heap_bytes(
    g: &GptOssGeometry,
    tuning: &Tuning,
    max_ctx: u32,
    max_tokens: u32,
    max_requests: u32,
) -> u64 {
    let mut bytes = BASE_SLACK;
    let per_layer = u64::from(g.n_kv_heads) * (u64::from(max_ctx) + 32) * u64::from(g.head_dim) * 2;
    bytes += u64::from(g.n_layers) * 2 * per_layer;
    let rows = gptoss_qmm_pool_rows(max_tokens);
    let sampled = gptoss_qmm_pool_rows(clamp_sampled(max_tokens, max_requests));
    for e in gptoss_pool_elems(g, tuning, rows, sampled) {
        bytes += e * 2;
    }
    bytes += u64::from(sampled) * u64::from(g.vocab) * 2;
    bytes
}

/// The gemma4 family's extra-heap bytes. The KV term is per OWNING
/// layer at its own width — the same shape `stage_gemma4_kv` allocates.
#[must_use]
pub fn gemma4_extra_heap_bytes(
    g: &Gemma4Geometry,
    tuning: &Tuning,
    max_ctx: u32,
    max_tokens: u32,
    max_requests: u32,
) -> u64 {
    let mut bytes = BASE_SLACK;
    for layer in 0..g.n_layers {
        if g.is_kv_shared(layer) {
            continue;
        }
        bytes += 2
            * u64::from(g.n_kv_heads_of(layer))
            * (u64::from(max_ctx) + 32)
            * u64::from(g.head_dim_of(layer))
            * 2;
    }
    // The dense projections tile through the shared rules, so the pool
    // padding is the shared one.
    let rows = llama_qmm_pool_rows(max_tokens);
    let sampled = llama_qmm_pool_rows(clamp_sampled(max_tokens, max_requests));
    for e in gemma4_pool_elems(g, tuning, rows, sampled) {
        bytes += e * 2;
    }
    bytes += u64::from(sampled) * u64::from(g.vocab) * 2;
    bytes
}

/// The most rows one fire can afford under `budget_bytes`, by bisecting
/// the budget function itself.
///
/// Spent on the DIFFERENCE from a one-row fire, not the total: the KV
/// region and the base slack dominate the budget and do not scale with
/// rows, so comparing the total against a pool-sized budget makes every
/// model answer "one row" — which is how the C++'s first came out, at
/// the floor. Monotone in rows, so the bisection is exact rather than
/// an estimate — and stays exact when the DAG changes, because the
/// function it bisects changes with it.
#[must_use]
pub fn max_forward_tokens_for_budget(
    extra_heap_bytes: impl Fn(u32) -> u64,
    budget_bytes: u64,
) -> u32 {
    let floor = extra_heap_bytes(1);
    let rows_cost = |rows: u32| extra_heap_bytes(rows).saturating_sub(floor);
    if rows_cost(PAGED_MAX_FORWARD_TOKENS) <= budget_bytes {
        return PAGED_MAX_FORWARD_TOKENS;
    }
    let (mut lo, mut hi) = (1u32, PAGED_MAX_FORWARD_TOKENS);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if rows_cost(mid) <= budget_bytes {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo.max(PAGED_MIN_FORWARD_TOKENS)
}

/// Which tensors are worth streaming — mapped over a page-aligned pack
/// instead of copied into the heap — or `None` when nothing is.
///
/// A ROUTED expert bank, and nothing else: one pattern, no family
/// argument. The name is the whole test — a routed layer publishes an
/// `experts.` path whatever family it belongs to (`mlp.experts.` on
/// llama and gpt-oss, `experts.switch_glu.` on gemma4, which hangs its
/// bank off the layer), and a dense layer has no experts to match.
/// Asking the family was asking a proxy for the layer shape, and the
/// llama family — the only one with both shapes — is exactly where the
/// proxy broke: its clause named `mlp.gate_proj`, not a substring of
/// `mlp.experts.gate_proj`, so Qwen3-MoE streamed nothing while gpt-oss
/// streamed 10.75 GB off the same access pattern.
///
/// `.bias` is where MAPPING and PAGING part, and the difference is
/// correctness: mapping leaves the bias resident (three orders of
/// magnitude smaller than the weights, no index changes), but paging
/// RENUMBERS — `expert_ids` stops meaning "expert 57" and starts
/// meaning "the slot 57 was copied into", and the routed matvec offsets
/// the bias with that same buffer. A resident bias table indexed by a
/// slot number returns some other expert's bias beside this expert's
/// weights: fluent wrong tokens, not an error. So under paging the bias
/// joins the slab as one more band sharing the slot number.
#[must_use]
pub fn stream_predicate(
    stream_routed_experts: bool,
    slab_paging: bool,
) -> Option<impl Fn(&str) -> bool> {
    if !stream_routed_experts {
        return None;
    }
    Some(move |name: &str| {
        if name.ends_with(".bias") && !slab_paging {
            return false;
        }
        name.contains("experts.")
    })
}

// ─────────────────── the recurrent-state slot budget ────────────────────────

/// Slots the driver will reserve at most, and the request count it advertises.
///
/// The two are one constant on purpose: `context.cpp` derives the advertised
/// `max_forward_requests` from the same call that sizes the slots. Reserving
/// **fewer** than advertised would hang rather than queue — a request admitted
/// against an advertised slot that does not exist has nowhere to go — so the
/// two move together or not at all. Fewer admitted requests queue; a device
/// that is overrun does not.
pub const MAX_RS_SLOTS: u32 = 64;

/// The floor under the recurrent-state budget, before the working set is
/// consulted.
pub const RS_SLOT_BUDGET_FLOOR: u64 = 1536 << 20;

/// Bytes one recurrent-state slot costs: every GDN layer's conv pair plus its
/// recurrent state.
///
/// The conv state is counted **twice** because it is a ping-pong pair —
/// `ConvState` and `ConvStateOut` are distinct buffers and a slot owns both.
/// A slot sized for one of them is a slot that cannot take a step.
#[must_use]
pub fn rs_slot_bytes(geometry: &DecodeGeometry) -> u64 {
    let gdn_layers = (0..geometry.n_layers)
        .filter(|&layer| !geometry.is_full_attn(layer))
        .count() as u64;
    gdn_layers * (2 * geometry.gdn_conv_stride_bytes() + geometry.gdn_recurrent_stride_bytes())
}

/// The budget: a tenth of the device's working set, but never below the floor.
///
/// A working set of zero is a device that would not say, so the floor stands
/// alone rather than being invented from nothing.
#[must_use]
pub fn rs_slot_budget_bytes(device_working_set: u64) -> u64 {
    RS_SLOT_BUDGET_FLOOR.max(device_working_set / 10)
}

/// How many slots fit, given the budget and what the caller asked for.
///
/// # `requested` is a ceiling, not a floor
///
/// It used to be applied as `max(slots, requested)`, and that made the budget
/// **decorative**: `requested` is the advertised request count, a constant, so
/// every checkpoint reserved all 64 slots at whatever they cost. Qwen3.6-27B's
/// slots are 170 MiB, which is 10.6 GiB of recurrent state — 5.2 GiB over this
/// device.
///
/// What that looked like is worth keeping, because it is why the bug was hard:
/// the over-allocation returned `kIOGPUCommandBufferCallbackErrorOutOfMemory`
/// from the command queue, which arrived as **a command buffer that never
/// ran**. Every PTIR lane's status still held its zero fill, and the runtime
/// read those zeros back as every lane faulting — deterministic above
/// concurrency 4, and naming neither memory nor the batch it came from.
///
/// # The floor of one
///
/// A geometry whose slot does not fit the budget still gets one slot, because
/// a decoder with no slot cannot run at all. So the returned count can cost
/// more than the budget by design, and the budget is a bound on *how many*
/// rather than a promise about the total.
#[must_use]
pub fn rs_slots_for_budget(geometry: &DecodeGeometry, budget_bytes: u64, requested: u32) -> u32 {
    let per_slot = rs_slot_bytes(geometry);
    // No linear-attention layers means no state to slot: the count is whatever
    // the caller needs, at no cost.
    if per_slot == 0 {
        return requested.max(1);
    }
    let affordable = budget_bytes / per_slot;
    let slots = affordable
        .min(u64::from(MAX_RS_SLOTS))
        .min(u64::from(requested.max(1)))
        .max(1);
    u32::try_from(slots).unwrap_or(MAX_RS_SLOTS)
}

#[cfg(test)]
mod tests {
    /// A hybrid geometry: 36 layers, full attention every 6th, so 30 GDN.
    fn hybrid() -> DecodeGeometry {
        DecodeGeometry {
            n_layers: 36,
            full_attn_interval: 6,
            gdn_conv_dim: 4096,
            gdn_conv_k: 4,
            gdn_v_heads: 32,
            gdn_v_dim: 128,
            gdn_k_dim: 128,
            ..DecodeGeometry::default()
        }
    }

    #[test]
    fn a_slot_counts_the_conv_pair_twice_because_it_is_a_ping_pong() {
        let g = hybrid();
        let gdn = 30u64;
        assert_eq!(
            rs_slot_bytes(&g),
            gdn * (2 * g.gdn_conv_stride_bytes() + g.gdn_recurrent_stride_bytes()),
            "a slot sized for one half of the pair cannot take a step"
        );
    }

    #[test]
    fn the_request_count_is_a_ceiling_so_the_budget_is_not_decorative() {
        // The shipped bug: applied as `max(slots, requested)`, every checkpoint
        // reserved all 64 slots at whatever they cost, and Qwen3.6-27B's are
        // 170 MiB — 10.6 GiB, 5.2 GiB over the device.
        let g = hybrid();
        let per_slot = rs_slot_bytes(&g);
        // A budget that affords exactly three.
        let budget = per_slot * 3;
        assert_eq!(
            rs_slots_for_budget(&g, budget, MAX_RS_SLOTS),
            3,
            "asking for 64 does not conjure 64"
        );
    }

    #[test]
    fn asking_for_fewer_than_the_budget_affords_gets_fewer() {
        let g = hybrid();
        let budget = rs_slot_bytes(&g) * 40;
        assert_eq!(
            rs_slots_for_budget(&g, budget, 8),
            8,
            "the ask is the bound"
        );
        assert_eq!(
            rs_slots_for_budget(&g, budget, MAX_RS_SLOTS),
            40,
            "and the budget is the other one"
        );
    }

    #[test]
    fn the_reserved_count_never_exceeds_what_the_driver_advertises() {
        let g = hybrid();
        let vast = rs_slot_bytes(&g) * 10_000;
        assert_eq!(rs_slots_for_budget(&g, vast, 10_000), MAX_RS_SLOTS);
    }

    #[test]
    fn a_geometry_that_cannot_afford_one_slot_still_gets_one() {
        // Deliberate: a decoder with no slot cannot run at all, so the budget
        // bounds HOW MANY rather than promising a total.
        let g = hybrid();
        assert_eq!(rs_slots_for_budget(&g, 0, MAX_RS_SLOTS), 1);
        assert_eq!(rs_slots_for_budget(&g, 1, MAX_RS_SLOTS), 1);
    }

    #[test]
    fn a_geometry_with_no_gdn_layers_costs_nothing_and_gets_what_it_asks() {
        let dense = DecodeGeometry {
            n_layers: 32,
            full_attn_interval: 1, // every layer is full attention
            ..hybrid()
        };
        assert_eq!(rs_slot_bytes(&dense), 0);
        assert_eq!(rs_slots_for_budget(&dense, 0, 12), 12);
        assert_eq!(rs_slots_for_budget(&dense, 0, 0), 1, "never zero");
    }

    #[test]
    fn the_budget_is_a_tenth_of_the_working_set_but_never_under_the_floor() {
        assert_eq!(rs_slot_budget_bytes(0), RS_SLOT_BUDGET_FLOOR);
        assert_eq!(
            rs_slot_budget_bytes(8 << 30),
            RS_SLOT_BUDGET_FLOOR,
            "a tenth of 8 GiB is under the floor"
        );
        assert_eq!(rs_slot_budget_bytes(64 << 30), (64u64 << 30) / 10);
    }

    use super::*;

    #[test]
    fn the_bisection_spends_the_difference_and_lands_exactly() {
        // A synthetic monotone cost: floor 1 GiB, 1 MiB per row past one.
        let cost = |rows: u32| (1u64 << 30) + u64::from(rows.saturating_sub(1)) * (1 << 20);
        // A 100 MiB budget affords exactly 101 rows — the floor is NOT
        // charged against it, which is the bug the C++ names (every
        // model answering "one row").
        assert_eq!(max_forward_tokens_for_budget(cost, 100 << 20), 101);
        // A vast budget caps at the ceiling; a starved one floors.
        assert_eq!(
            max_forward_tokens_for_budget(cost, u64::MAX / 2),
            PAGED_MAX_FORWARD_TOKENS
        );
        assert_eq!(
            max_forward_tokens_for_budget(cost, 0),
            PAGED_MIN_FORWARD_TOKENS
        );
    }

    #[test]
    fn the_real_budgets_are_monotone_and_family_shaped() {
        let tuning = Tuning::default();
        let g = LlamaGeometry::default();
        let at = |rows| llama_extra_heap_bytes(&g, &tuning, 4096, rows, rows);
        assert!(at(64) < at(512) && at(512) < at(4096));
        // gemma4's KV term counts OWNING layers at their own widths: the
        // E2B has 15 owners, and doubling the shared tail must not move
        // the KV term.
        let g4 = Gemma4Geometry::default();
        let base = gemma4_extra_heap_bytes(&g4, &tuning, 4096, 64, 1);
        let mut more_shared = g4.clone();
        more_shared.num_kv_shared_layers += 5;
        let fewer_owners = gemma4_extra_heap_bytes(&more_shared, &tuning, 4096, 64, 1);
        assert!(fewer_owners < base, "fewer owners is less KV");
    }

    #[test]
    fn the_stream_predicate_is_one_pattern_and_the_bias_splits_on_paging() {
        assert!(stream_predicate(false, false).is_none());
        let mapped = stream_predicate(true, false).unwrap();
        // Every family's bank matches, whatever directory it hangs off.
        assert!(mapped("layers.0.mlp.experts.gate_proj.weight"));
        assert!(mapped("layers.0.experts.switch_glu.gate_proj.weight"));
        // The llama proxy's near misses stay unstreamed.
        assert!(!mapped("layers.0.mlp.gate_proj.weight"));
        assert!(!mapped("layers.0.router.per_expert_scale"));
        // Mapped, the bias stays resident; paged, it joins the slab.
        assert!(!mapped("layers.0.mlp.experts.gate_proj.bias"));
        let paged = stream_predicate(true, true).unwrap();
        assert!(paged("layers.0.mlp.experts.gate_proj.bias"));
    }
}
