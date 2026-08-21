//! Arena assignment and value alias ownership.

use super::*;

/// Where each SSA value's bytes live.
#[derive(Debug, Clone)]
pub struct Buffers {
    /// Byte offset per value, or [`Buffers::NAMED`] for backend-bound values.
    pub offset: Vec<usize>,
    /// Peak bytes.
    pub bytes: usize,
    /// Values exposed by seams; they cannot be recycled under hidden aliases.
    pub pinned: Vec<ValueId>,
    /// The epilogue emits only sampled rows; gather uses scratch when sampling a subset.
    pub n_requests: u32,
    pub epilogue_gather: usize,
    pub epilogue_norm: usize,
}

impl Buffers {
    /// `offset[v]` for a value whose bytes are a named buffer's.
    pub const NAMED: usize = usize::MAX;

    #[allow(clippy::too_many_lines)]
    pub fn assign(plan: &ForwardPlan, rows: &[Row]) -> Buffers {
        let n_tokens = rows.len();
        let n_requests = rows.iter().filter(|r| r.samples).count().max(1);

        // Whether the correction runs at all. The adapter seam is stated by
        // the model text and so is present on every fire of the family, but
        // the correction is guarded on the fire's rows -- and the pin below
        // takes a per-layer buffer out of the arena, which a fire that runs no
        // correction should not pay for.
        let any_lora = rows.iter().any(|r| r.lora);
        let mut pinned: Vec<ValueId> = Vec::new();
        // CUDA pins the exit seam; Metal keeps it in the arena.
        let arena_exit: Vec<ValueId> =
            if matches!(Backend::of_family(&plan.family), Some(Backend::Metal)) {
                plan.seams
                    .iter()
                    .filter(|s| s.seam == model_ir::seam::OUT.name)
                    .flat_map(|s| s.values.iter().copied())
                    .collect()
            } else {
                Vec::new()
            };
        for stmt in &plan.seams {
            // BEFORE the split below, because an attachment that states its
            // values takes the early return and the operand it does NOT state
            // is exactly the one at issue.
            if any_lora {
                pinned.extend(seam_reads_beyond_inputs(plan, stmt));
            }
            if !stmt.values.is_empty() {
                pinned.extend(
                    stmt.values
                        .iter()
                        .copied()
                        .filter(|v| !arena_exit.contains(v)),
                );
                continue;
            }
            let Some(at) = stmt.op else { continue };
            for probe in [at as usize, at as usize + 1] {
                if let Some(op) = plan.ops.get(probe)
                    && matches!(op.kind, OpKind::HookSite { .. } | OpKind::Launch { .. })
                {
                    pinned.extend(op.inputs.iter().copied());
                    break;
                }
            }
        }
        // Alias every value sharing an exposed buffer so named readers see final contents.
        {
            let owner = alias_owners(plan);
            let roots: std::collections::BTreeSet<ValueId> =
                pinned.iter().map(|&v| owner[v as usize]).collect();
            // Walk every value and pin aliases of exposed roots.
            for (v, root) in owner.iter().enumerate() {
                if roots.contains(root) {
                    pinned.push(v as ValueId);
                }
            }
        }
        pinned.sort_unstable();
        pinned.dedup();

        // A `Select` output is a window into its operand, not a new allocation.
        let owner = alias_owners(plan);

        // Last use is folded to the alias owner, so shared bytes free once.
        let mut last_use = vec![0usize; plan.values.len()];
        for (i, op) in plan.ops.iter().enumerate() {
            for &v in op.inputs.iter().chain(op.outputs.iter()) {
                let Some(&own) = owner.get(v as usize) else {
                    continue;
                };
                if let Some(slot) = last_use.get_mut(own as usize) {
                    *slot = (*slot).max(i);
                }
            }
        }
        for v in 0..plan.values.len() {
            last_use[v] = last_use[owner[v] as usize];
        }
        for &v in &arena_exit {
            last_use[owner[v as usize] as usize] = usize::MAX;
            last_use[v as usize] = usize::MAX;
        }

        let mut offset = vec![Self::NAMED; plan.values.len()];
        let mut size = vec![0usize; plan.values.len()];
        let mut free: Vec<(usize, usize)> = Vec::new();
        let mut used = 0usize;
        let mut live: Vec<ValueId> = Vec::new();

        for (i, op) in plan.ops.iter().enumerate() {
            live.retain(|&v| {
                if last_use[v as usize] >= i {
                    return true;
                }
                insert_free(&mut free, (offset[v as usize], size[v as usize]));
                false
            });
            // A `Select` is a window into its operand.
            if let OpKind::Select { index } = op.kind {
                let src = op.inputs[0];
                let out = op.outputs[0];
                let want = value_bytes(plan, out, n_tokens, n_requests);
                if offset[src as usize] == Self::NAMED {
                    offset[out as usize] = Self::NAMED;
                } else {
                    offset[out as usize] = offset[src as usize] + index as usize * want;
                }
                size[out as usize] = want;
                continue;
            }
            // In-place outputs reuse the input offset named by the kernel or semantic pair.
            {
                // OWNED for the launch arm and borrowed for the semantic one:
                // a launch's pairs are DERIVED from the row's `Source::Alias`
                // marks now, so there is no `&'static` list to hand back.
                let pairs = match &op.kind {
                    OpKind::Launch { kernel, .. } => {
                        model_ir::kernels::in_place_pairs(plan, kernel)
                    }
                    other => model_ir::kernels::semantic_in_place(other).to_vec(),
                };
                let mut aliased = false;
                for &(o, i) in &pairs {
                    // A pair outside arity is allowed; rows state the widest form.
                    if let (Some(&src), Some(&out)) =
                        (op.inputs.get(i as usize), op.outputs.get(o as usize))
                    {
                        offset[out as usize] = offset[src as usize];
                        size[out as usize] = value_bytes(plan, out, n_tokens, n_requests);
                        aliased = true;
                    }
                }
                if aliased {
                    for (o, &v) in op.outputs.iter().enumerate() {
                        if pairs.iter().any(|&(oi, _)| oi as usize == o) {
                            continue;
                        }
                        if pinned.binary_search(&v).is_ok() || is_raised(plan, v) {
                            offset[v as usize] = Self::NAMED;
                            continue;
                        }
                        let want = value_bytes(plan, v, n_tokens, n_requests);
                        let at = take_block(&mut free, &mut used, want);
                        offset[v as usize] = at;
                        size[v as usize] = want;
                        live.push(v);
                    }
                    continue;
                }
            }
            for &v in &op.outputs {
                // A RAISE IS NOT AN ACTIVATION AND TAKES NO ARENA BLOCK. It
                // reaches this loop because `OpKind::Prep` has an output now,
                // and without the guard `value_bytes` would size it from the
                // empty shape stored beside it and `take_block` would hand
                // back a real offset for zero bytes -- after which `slot`
                // reads `Arg::Arena` and the raise is a rectangle at a place
                // in the activation arena. It is `NAMED` for the reason every
                // other `NAMED` value is: the backend holds it.
                if pinned.binary_search(&v).is_ok() || is_raised(plan, v) {
                    offset[v as usize] = Self::NAMED;
                    continue;
                }
                let want = value_bytes(plan, v, n_tokens, n_requests);
                let at = take_block(&mut free, &mut used, want);
                offset[v as usize] = at;
                size[v as usize] = want;
                live.push(v);
            }
        }
        let mut epilogue_gather = Self::NAMED;
        let mut epilogue_norm = Self::NAMED;
        for op in &plan.ops {
            if !matches!(op.kind, OpKind::LmHead { .. }) {
                continue;
            }
            let Some(&input) = op.inputs.first() else {
                continue;
            };
            let width = value_bytes(plan, input, 1, 1);
            let sampled = rows.iter().filter(|r| r.samples).count().max(1);
            let want = width * sampled;
            if want == 0 {
                continue;
            }
            epilogue_gather = take_block(&mut free, &mut used, want);
            epilogue_norm = take_block(&mut free, &mut used, want);
            break;
        }

        Buffers {
            n_requests: u32::try_from(n_requests).unwrap_or(u32::MAX),

            offset,
            bytes: used,
            pinned,
            epilogue_gather,
            epilogue_norm,
        }
    }
}

/// Best fit, splitting any remainder back into the free list.
fn take_block(free: &mut Vec<(usize, usize)>, used: &mut usize, want: usize) -> usize {
    match free
        .iter()
        .enumerate()
        .filter(|(_, block)| block.1 >= want)
        .min_by_key(|(_, block)| block.1)
        .map(|(i, _)| i)
    {
        Some(f) => {
            let (off, size_of) = free.remove(f);
            let tail = (off + want).div_ceil(256) * 256;
            if tail < off + size_of {
                insert_free(free, (tail, off + size_of - tail));
            }
            off
        }
        None => {
            // Arena offsets are 256-byte aligned and bump-only when no free block fits.
            let at = used.div_ceil(256) * 256;
            *used = at + want;
            at
        }
    }
}

/// Returned blocks are merged with adjacent free blocks.
fn insert_free(free: &mut Vec<(usize, usize)>, block: (usize, usize)) {
    let (at, len) = block;
    if len == 0 {
        return;
    }
    let i = free.partition_point(|&(off, _)| off < at);
    free.insert(i, (at, len));
    if i + 1 < free.len() && free[i].0 + free[i].1 == free[i + 1].0 {
        let (_, next_len) = free.remove(i + 1);
        free[i].1 += next_len;
    }
    if i > 0 && free[i - 1].0 + free[i - 1].1 == free[i].0 {
        let (_, this_len) = free.remove(i);
        free[i - 1].1 += this_len;
    }
}

/// A `Select` output is a window into its operand, not a new allocation.
pub(crate) fn alias_owners(plan: &ForwardPlan) -> Vec<ValueId> {
    let mut owner: Vec<ValueId> = (0..plan.values.len() as ValueId).collect();

    fn find(owner: &mut [ValueId], v: ValueId) -> ValueId {
        let mut v = v;
        while owner[v as usize] != v {
            let up = owner[v as usize];
            owner[v as usize] = owner[up as usize];
            v = owner[v as usize];
        }
        v
    }

    for op in &plan.ops {
        let joined: Vec<(ValueId, ValueId)> = match &op.kind {
            OpKind::Select { .. } => match (op.inputs.first(), op.outputs.first()) {
                (Some(&src), Some(&out)) => vec![(src, out)],
                _ => Vec::new(),
            },
            OpKind::Launch { kernel, .. } => model_ir::kernels::in_place_pairs(plan, kernel)
                .into_iter()
                .filter_map(|(o, i)| {
                    Some((*op.inputs.get(i as usize)?, *op.outputs.get(o as usize)?))
                })
                .collect(),
            // Semantic in-place ops use the same `(output, input)` pairs.
            other => model_ir::kernels::semantic_in_place(other)
                .iter()
                .filter_map(|&(o, i)| {
                    Some((*op.inputs.get(i as usize)?, *op.outputs.get(o as usize)?))
                })
                .collect(),
        };
        for (src, out) in joined {
            if src as usize >= owner.len() || out as usize >= owner.len() {
                continue;
            }
            let (a, b) = (find(&mut owner, src), find(&mut owner, out));
            if a != b {
                // The earlier value keeps the allocation.
                let (keep, drop) = if a <= b { (a, b) } else { (b, a) };
                owner[drop as usize] = keep;
            }
        }
    }
    for v in 0..owner.len() {
        owner[v] = find(&mut owner, v as ValueId);
    }
    owner
}

pub fn value_bytes(plan: &ForwardPlan, v: ValueId, n_tokens: usize, n_requests: usize) -> usize {
    let Some(info) = plan.values.get(v as usize) else {
        return 0;
    };
    shape_elements(&info.shape, n_tokens, n_requests) * dtype_bytes(info.dtype) as usize
}

/// `param_extents` replace stated constants with fire-sized extents.
#[must_use]
pub fn shape_elements(shape: &model_ir::trace::Shape, n_tokens: usize, n_requests: usize) -> usize {
    let mut elements = 1usize;
    for dim in shape.0.iter().copied() {
        elements *= match dim {
            Dim::Tokens => n_tokens,
            Dim::Requests => n_requests,
            Dim::Const(c) => c as usize,
            Dim::MoeAlignedRoutes {
                top_k,
                experts,
                block,
            } => Dim::moe_aligned_rows(n_tokens as u32, top_k, experts, block) as usize,
        };
    }
    elements
}

/// Bytes per element.
#[must_use]
pub const fn dtype_bytes(d: DType) -> u32 {
    match d {
        DType::BF16 | DType::F16 => 2,
        DType::F32 | DType::I32 => 4,
    }
}

/// Whether this value is a raise rather than a tensor.
///
/// Asked of the VALUE and not of the op that produced it. A raise can only come
/// from an `OpKind::Prep` today, and keying the rule on that would be a second
/// place for the invariant to live -- one that goes quietly wrong the first
/// time anything else publishes one.
fn is_raised(plan: &ForwardPlan, v: ValueId) -> bool {
    plan.values
        .get(v as usize)
        .is_some_and(model_ir::trace::ValueInfo::is_raised)
}

/// The operands an attachment reads that its STATEMENT does not name.
///
/// A seam statement names the values the attachment REWRITES, because that is
/// what the seam signature is about: [`model_ir::seam::ATTN_QV`] sees `q` and
/// `v` and its correction adds a delta into both, in place. So the correction
/// op's `inputs` are `[q, v]` and the pinning above, which reads exactly that
/// list, keeps exactly those two out of the recycler.
///
/// The correction reads a THIRD operand, and no statement anywhere mentions
/// it: `x`, the projection input, which is what the low-rank `A` is applied
/// to. The backend recovers it the only way it can be recovered -- off the
/// qkv (or q) projection's own input -- and binds it as a foreign value at
/// dispatch time, which is a place the lowering has already finished by.
///
/// # What that cost
///
/// The recycler is right about `x` on the evidence it has. Nothing in the op
/// list reads `x` after the projection consumes it, so its block goes back on
/// the free list at that op, and the very next op takes it: on qwen3-0.6b,
/// `attn::split_qkv_bf16` writes the K projection into arena offset 10240,
/// which is the block the normed projection input was still sitting in, and
/// rope and attention write over it again after that. By the time the
/// correction ran -- which is AFTER the split, because it needs q and v as
/// separate buffers -- `x` was three writers stale.
///
/// The adapter was therefore applied to whatever the arena last held. That is
/// not a crash and it is not even obviously wrong output: the delta is a
/// plausible-looking matrix product of the wrong operand, and at
/// `adapter_scale: 0.0` the `B` factor multiplies it away entirely, so every
/// zero-adapter check passed. What it looked like from outside was that a
/// nonzero adapter was not REPRODUCIBLE -- three runs of `lora-probe` at
/// `adapter_scale: 0.5` on one build answered " a fictional series of novels
/// an", " capital of capital of capital o" and " a country that is a countr"
/// -- because attention output, unlike the forward, is not bit-stable across
/// fires.
///
/// # Why pinning and not a copy
///
/// The backend could snapshot `x` aside before the split overwrites it, and
/// that was the first fix drafted. It puts a per-layer memcpy in the walk, a
/// stash buffer in the frame, and a launch-index-keyed flag in the dispatch
/// plan, to work around a fact the lowering already has a word for. `pinned`
/// means "a reader exists that the op list does not show", which is precisely
/// and only what is true here. A pinned value leaves the arena and becomes a
/// named buffer, which is per-value and so per-layer -- and that incidentally
/// answers the other half of the symptom, that `x` arrived at the dispatch
/// arm as the SAME address on all 28 layers.
fn seam_reads_beyond_inputs(
    plan: &ForwardPlan,
    stmt: &model_ir::trace::SeamStatement,
) -> Vec<ValueId> {
    if stmt.seam != model_ir::seam::ATTN_QV.name {
        return Vec::new();
    }
    let Some(probe) = stmt.op.map(|at| at as usize) else {
        return Vec::new();
    };
    // Backwards to the projection whose input the correction re-reads. The
    // same rule the backend harvests it by, so the two cannot disagree about
    // which value `x` is.
    plan.ops[..probe]
        .iter()
        .rev()
        .find_map(|op| match &op.kind {
            // The projection is a Launch now; its weight rides the
            // positional weights list.
            OpKind::Launch { weights, .. }
                if weights
                    .first()
                    .is_some_and(|w| w.ends_with(".qkv") || w.ends_with(".q_proj")) =>
            {
                op.inputs.first().copied()
            }
            _ => None,
        })
        .into_iter()
        .collect()
}
