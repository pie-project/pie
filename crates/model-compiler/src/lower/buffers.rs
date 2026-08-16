//! BUFFER ASSIGNMENT — the arena, and who owns which offset.

use super::*;

// ── Buffer assignment ──────────────────────────────────────────────────

/// Where each SSA value's bytes live.
///
/// A PINNED value is not allocated here at all: its bytes ARE the named
/// buffer's, and which buffer that is is the backend's binding. So
/// `offset[v] == NAMED` says "ask the backend", and such values are
/// excluded from [`Buffers::bytes`].
///
/// The DSL carries no buffer notion — `rmsnorm(x: &Val) -> Val` — so
/// choosing one is a backend job, and it is the job both CUDA executors
/// did as family convention. Doing it here, once, from the values' own
/// extents and liveness, is what lets an arm ask by value id and stay
/// family-blind.
///
/// A layer-unrolled plan names 28 distinct "normed activation" values
/// whose live ranges never overlap, so liveness reuse keeps the whole
/// frame inside a handful of buffers' worth of arena.
///
/// PINS are the exception: values that machinery OUTSIDE the traced ops
/// reaches by name — the query a hook observes, the normed activation an
/// adapter's host setup captures, the logits the sampler reads. The seam
/// signatures declare exactly that set (`sees`), so pins are derivable
/// rather than a per-family table. One empirical warning, paid for once:
/// a pin must be declared BY CONSUMER, not producer. A lowered trace may
/// state its attention as a stated-kernel `Launch` rather than a
/// semantic `Attention` op, so "the value o_proj reads lives in the
/// attention output buffer" is the sentence that holds under both
/// spellings.
#[derive(Debug, Clone)]
pub struct Buffers {
    /// Byte offset into the frame's activation arena, per value id, or
    /// [`Buffers::NAMED`] for a pinned value the backend binds by name.
    pub offset: Vec<usize>,
    /// Peak bytes.
    pub bytes: usize,
    /// Value ids a seam statement exposes, which therefore may not be
    /// recycled under a name outside machinery cannot follow.
    pub pinned: Vec<ValueId>,
    /// The epilogue's two intermediates, as byte offsets into the same
    /// arena — [`Buffers::NAMED`] when this fire needs neither.
    ///
    /// These are the ONLY buffers here that belong to no traced value,
    /// and the reason is that they belong to no traced STATEMENT either.
    /// One `LmHead` lowers to a row gather, a norm and a GEMM, and
    /// whether the gather runs at all is a fact about the FIRE's rows
    /// (`Row::samples`), not about the text — so the text cannot name
    /// what sits between them, and the lowering has to.
    ///
    /// Every CUDA executor reached for a workspace field here
    /// (`ws.norm_y`, `ws.norm_x`), each with its own apologetic comment,
    /// because the flat list handed all three rectangles the same
    /// operand run: `(activation, logits)`, which is true of the GEMM
    /// and of neither of the others.
    /// Rows the fire samples — see [`Lowered::n_requests`].
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
        // `Dim::Requests` sizes the epilogue's values, and the number it
        // means is THE ROWS THIS FIRE READS OUT — one per request normally,
        // more where a request samples several (MTP verify), which is why
        // this counts `samples` rather than requests.
        //
        // It used to also count the rows that are not `multi_token`, as a
        // stand-in for "requests", and take the larger. That term is a
        // stand-in that only holds in a DECODE fire, where one row is one
        // request; in a prefill fire every row of a multi-token request
        // carries `multi_token: false` unless the shell states regions, and
        // no shell does for an ordinary step. So a 2048-token prefill said
        // `n_requests = 2048` and sized the readout for every token.
        //
        // What that COST was measured, not argued. `Launch::rows` for the
        // head is this number, and `Rule::Qmv` reads it as its M: on
        // Llama-3.2-1B the lm head ran as a 2048-row matvec against a
        // 128256-wide vocabulary and took 904 ms of a 2184 ms prefill — 41%
        // of the fire, computing 2047 distributions nobody reads, plus a
        // 525 MB logits block to hold them. The answer was never wrong,
        // because sampling takes the row it asked for and the extra rows are
        // simply never looked at, which is why nothing caught it.
        //
        // The floor of one is for a fire that reads nothing out: the
        // epilogue emits no rectangle there (`emit_logits` is
        // `num_sampling > 0`), and a zero would make every `Requests`-shaped
        // value a zero-byte block that a later fire's arena plan would then
        // have to grow.
        let n_requests = rows.iter().filter(|r| r.samples).count().max(1);

        // The values a seam exposes: read off the seam statements, not a
        // per-family table, and now off the statement's OWN value list
        // rather than the operands of whatever op it points at.
        //
        // The probe that did the guessing is kept as the fallback for a
        // record written before seams carried their values, and it was
        // wrong in both directions. It took the neighbouring op's
        // INPUTS, so `attn.qv` -- which names q and v -- pinned q, k and
        // v, costing reuse; and no exposed value that is an OUTPUT was
        // ever pinned at all. That second one is not a cost, it is a
        // wrong answer: the sampler reads the logit softcap's RESULT,
        // which the arena was placing while the driver read `ws.logits`.
        let mut pinned: Vec<ValueId> = Vec::new();
        // Who supplies the read-out's memory, which is the one thing the two
        // backends genuinely disagree about here.
        //
        // A pin means "the BACKEND binds this; the arena does not" -- the CUDA
        // driver hands in a logits buffer and the trace writes through it. The
        // Metal executor has no such buffer: it allocates ONE arena per fire
        // and the read-out lives in it, which is how the reference gate reads
        // logits back at all. Pinning it there would move the distribution to
        // a buffer nobody binds, so the fire would compute it into nothing.
        //
        // So the exit seam PINS on CUDA and PLACES on Metal. It must not be
        // recycled on either -- see `last_use` below, which holds it to the
        // end of the trace.
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
        // A seam names the value at the point it is STATED, and later
        // statements may write over those bytes in place -- the logit
        // softcap accumulates into the logits it was handed. Everything
        // sharing an exposed value's buffer is therefore exposed too, or
        // the arena places the final contents somewhere the reader is
        // not looking.
        {
            let owner = alias_owners(plan);
            let roots: std::collections::BTreeSet<ValueId> =
                pinned.iter().map(|&v| owner[v as usize]).collect();
            // `alias_owners` returns one entry per value, so walking it
            // walks every value.
            for (v, root) in owner.iter().enumerate() {
                if roots.contains(root) {
                    pinned.push(v as ValueId);
                }
            }
        }
        pinned.sort_unstable();
        pinned.dedup();

        // Values that SHARE bytes by construction, and the one of each
        // set that owns the allocation. Two ops mean this: a `Select`
        // output is a window of its operand, and an in-place launcher's
        // output is the operand it accumulates into.
        let owner = alias_owners(plan);

        // Last use, in one op pass — then folded onto the OWNER, which
        // is the correction that makes sharing safe.
        //
        // Read per value id, a shared buffer frees at the last use of
        // whichever member the op happened to name. That is not when the
        // bytes stop being read: a residual stream is a chain of in-place
        // adds, so the first link's id is dead after one op while the
        // bytes stay live for the whole network, and the freed block gets
        // handed to the next value that fits. The window case is the same
        // shape and was previously reasoned away in a comment here ("the
        // window's readers are the source's readers by dataflow") — they
        // are not, because a reader names the WINDOW's id, not the
        // source's.
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
        // The exit outlives the trace. It is the fire's ANSWER -- read after
        // every launch has run -- so a block handed on to a later value would
        // be the read-out overwritten by whatever came next, and the reader
        // would find a plausible tensor that is not logits.
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
            // Free what nobody reads any more. A pinned value never
            // returns to the pool: its bytes are reachable by name.
            live.retain(|&v| {
                if last_use[v as usize] >= i {
                    return true;
                }
                insert_free(&mut free, (offset[v as usize], size[v as usize]));
                false
            });
            // A `Select` allocates nothing: its value IS a window of its
            // operand's bytes, which is the whole of what the op means.
            // It joins the operand's alias set rather than entering
            // `live`, so those bytes return to the pool ONCE, at the last
            // use of the set — see the `last_use` fold above.
            if let OpKind::Select { index } = op.kind {
                let src = op.inputs[0];
                let out = op.outputs[0];
                let want = value_bytes(plan, out, n_tokens, n_requests);
                if offset[src as usize] == Self::NAMED {
                    // A window of a NAMED buffer is still the backend's
                    // to bind; the arena has no address to offset from.
                    offset[out as usize] = Self::NAMED;
                } else {
                    offset[out as usize] = offset[src as usize] + index as usize * want;
                }
                size[out as usize] = want;
                continue;
            }
            // An IN-PLACE op writes over an operand, so its output is
            // that operand's bytes. Giving it an allocation of its own
            // would be a copy
            // the model does not make, and for a text that accumulates
            // into a `select` window it would be worse than wasteful:
            // the window would keep its pre-update value and the streams
            // would silently never see the add.
            //
            // Read from the SAME two tables `alias_owners` reads —
            // the `kernel!` row for a stated symbol, the kind itself for
            // a semantic one. They were not the same for a while: the
            // owner table joined a semantic rope's operand and result
            // while this loop, which only knew about `Launch`, handed
            // the result a block of its own. Liveness then freed one
            // buffer for what placement had made two, and the rotated k
            // was written to an address nothing read.
            {
                let pairs = match &op.kind {
                    OpKind::Launch { kernel, .. } => model_ir::kernels::in_place_pairs(plan, kernel),
                    other => model_ir::kernels::semantic_in_place(other),
                };
                let mut aliased = false;
                for &(o, i) in pairs {
                    // A pair outside this statement's arity is not an
                    // error: one symbol serves a q-only site and a q/k
                    // pair, and the row states the widest form.
                    if let (Some(&src), Some(&out)) =
                        (op.inputs.get(i as usize), op.outputs.get(o as usize))
                    {
                        offset[out as usize] = offset[src as usize];
                        size[out as usize] = value_bytes(plan, out, n_tokens, n_requests);
                        aliased = true;
                    }
                }
                if aliased {
                    // Outputs this kernel does NOT write in place still
                    // need buffers of their own.
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
                    // Reachable by name from outside the trace — the
                    // query a hook observes, the logits the sampler
                    // reads. The backend binds it; the arena does not.
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
        // The epilogue's scratch, sized from the statement it serves.
        // Allocated LAST and never freed: it is live across the three
        // rectangles that make up one statement, and nothing else in
        // the fire runs between them.
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

/// Take `want` bytes from the pool, or bump.
///
/// BEST fit, and SPLIT the remainder back. First-fit-and-keep-the-whole-
/// block was costing 4-15x at the fire shape that sizes the driver's
/// activation block (`arena_soundness.rs` prices it per family): a freed
/// logits-sized block satisfying a one-row norm retired the rest of
/// itself, so the walk bump-allocated almost everything. It read as
/// cheap because the ratio had been measured on an eight-row
/// all-sampled fire, where the logits dominate the arena AND the floor
/// and the loss hides inside both.
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
            // The tail keeps the block's alignment, so a split never
            // hands out an address the bump path would not have.
            let tail = (off + want).div_ceil(256) * 256;
            if tail < off + size_of {
                insert_free(free, (tail, off + size_of - tail));
            }
            off
        }
        None => {
            // 256-byte alignment, and BUMP only: a decode body runs
            // inside a capture, so the same plan must land the same
            // value at the same address on every fire.
            let at = used.div_ceil(256) * 256;
            *used = at + want;
            at
        }
    }
}

/// Return a block to the pool, MERGED with any neighbour it touches.
///
/// The pool is kept sorted by offset so this is one scan. Without it,
/// splitting makes fragmentation worse rather than better: a block cut
/// into pieces to serve small values never becomes whole again, so a
/// later large value bump-allocates past a run of adjacent free bytes
/// that would have held it.
fn insert_free(free: &mut Vec<(usize, usize)>, block: (usize, usize)) {
    let (at, len) = block;
    if len == 0 {
        return;
    }
    let i = free.partition_point(|&(off, _)| off < at);
    free.insert(i, (at, len));
    // Merge forward first, then back, so a block filling a hole between
    // two free neighbours coalesces all three.
    if i + 1 < free.len() && free[i].0 + free[i].1 == free[i + 1].0 {
        let (_, next_len) = free.remove(i + 1);
        free[i].1 += next_len;
    }
    if i > 0 && free[i - 1].0 + free[i - 1].1 == free[i].0 {
        let (_, this_len) = free.remove(i);
        free[i - 1].1 += this_len;
    }
}

/// For each value, the value that OWNS the bytes it lives in.
///
/// Most values own their own. The exceptions are the three constructs
/// whose meaning is that the output does not get memory of its own: a
/// [`OpKind::Select`] output is a window of its operand; a launcher the
/// `kernel!` table marks in-place writes over the operand it
/// accumulates into; and a semantic kind that rewrites its operand says
/// so through [`model_ir::kernels::semantic_in_place`]. All chain — a
/// residual stream is a run of in-place adds — so this is a union-find,
/// and the owner is always the EARLIER value, i.e. the one whose
/// allocation the rest inherit.
///
/// Buffer assignment needs this in two places: the live range of a
/// shared buffer is the union's, not any one member's, and only the
/// owner may return those bytes to the free pool.
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
                .iter()
                .filter_map(|&(o, i)| {
                    Some((*op.inputs.get(i as usize)?, *op.outputs.get(o as usize)?))
                })
                .collect(),
            // The kinds that name no kernel but still write over their
            // operand — see `kernels::semantic_in_place`. Read the same
            // way as the table's, because it is the same fact.
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
                // The earlier value keeps the allocation; SSA numbering
                // makes "earlier" and "smaller id" the same thing, and
                // the ops are walked in order anyway.
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

/// How many elements `shape` describes for a fire of this size.
///
/// One place, because the arena's reading of an extent and the number a
/// kernel is TOLD that extent is must be the same reading — see
/// [`model_ir::trace::OpKind::Launch::param_extents`], which is the channel
/// that lets a statement say "this scalar is that shape" instead of
/// writing a constant beside it.
#[must_use]
pub fn shape_elements(shape: &model_ir::trace::Shape, n_tokens: usize, n_requests: usize) -> usize {
    let mut elements = 1usize;
    for dim in shape.0.iter().copied() {
        elements *= match dim {
            Dim::Tokens => n_tokens,
            Dim::Requests => n_requests,
            Dim::Const(c) => c as usize,
            // The padded route count, which is a function of the fire's
            // tokens and three load-time numbers -- so a residue ledger
            // sizing this value gets the real footprint, not an estimate.
            Dim::MoeAlignedRoutes {
                top_k,
                experts,
                block,
            } => Dim::moe_aligned_rows(n_tokens as u32, top_k, experts, block) as usize,
        };
    }
    elements
}

/// Bytes per element. One place, because two answers to this question is
/// how a row stride and a buffer size disagree.
#[must_use]
pub const fn dtype_bytes(d: DType) -> u32 {
    match d {
        DType::BF16 | DType::F16 => 2,
        DType::F32 | DType::I32 => 4,
    }
}
