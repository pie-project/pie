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
                    OpKind::Launch { kernel, .. } => model_ir::kernels::in_place_pairs(plan, kernel),
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
                        if pinned.binary_search(&v).is_ok() {
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
                if pinned.binary_search(&v).is_ok() {
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
