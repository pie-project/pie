//! THE HOST ASSIGNS BUFFERS. This is the test that makes that safe.
//!
//! Two allocators existed. `Buffers::assign` here, over the traced plan,
//! and `declared::ValueArena` in the CUDA driver, over the same plan at
//! fire time. Keeping both means they must agree byte-for-byte forever,
//! and they already do not: the driver's copy predates `Select`, the
//! `kernel!` in-place table and `Dim::MoeAlignedRoutes`, so on a text
//! using any of the three it would size or place a value differently —
//! silently, because an allocator that hands back a plausible pointer
//! reports nothing.
//!
//! So the host wins and the driver stops allocating: a rectangle's
//! operand already crosses as `Arg::Arena { at, width }`, which is an
//! address, and `Lowered::arena_bytes` is the block it must fit. That
//! makes this file the load-bearing one — a driver that only adds `at`
//! to a base pointer cannot notice an assignment that overlaps, so the
//! overlap has to be impossible HERE.
//!
//! The check is a write trace. Walk the ops in order; stamp each
//! output's byte range with its value; before an op reads an input,
//! demand that the input's range still carries its own stamp. A value
//! placed over a buffer somebody still reads shows up as a stamp that
//! changed underneath its reader.
//!
//! ALIASING is the wrinkle, and it is intended rather than accidental in
//! exactly two places: a `Select` output IS a window of its operand, and
//! an in-place launcher's output IS the operand it accumulates into.
//! Both are unions, so the trace stamps by the union's ROOT and the
//! intended sharing passes while an accidental one still fails.

use model_compiler::lower::{value_bytes, Buffers, Row};
use model_compiler::trace::{FireClass, ForwardPlan, OpKind, ValueId};

/// A decode-shaped fire: every row samples, so the epilogue's row space
/// is the full row count.
fn plain(n: usize) -> Vec<Row> {
    vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ]
}

/// Union-find over the value ids two ops are allowed to share bytes.
struct Alias(Vec<ValueId>);

impl Alias {
    fn new(n: usize) -> Alias {
        Alias((0..n as ValueId).collect())
    }

    fn root(&mut self, v: ValueId) -> ValueId {
        let mut v = v;
        while self.0[v as usize] != v {
            let up = self.0[v as usize];
            self.0[v as usize] = self.0[up as usize];
            v = self.0[v as usize];
        }
        v
    }

    fn join(&mut self, a: ValueId, b: ValueId) {
        let (ra, rb) = (self.root(a), self.root(b));
        if ra != rb {
            self.0[rb as usize] = ra;
        }
    }
}

/// The two constructs whose whole meaning is that the output shares the
/// input's bytes. Read off the plan, not listed by hand — a third one
/// added to `Buffers::assign` without being added here fails loudly,
/// which is the right way round.
fn aliases(plan: &ForwardPlan) -> Alias {
    let mut alias = Alias::new(plan.values.len());
    for op in &plan.ops {
        match &op.kind {
            OpKind::Select { .. } => {
                if let (Some(&src), Some(&out)) = (op.inputs.first(), op.outputs.first()) {
                    alias.join(src, out);
                }
            }
            OpKind::Launch { kernel, .. } => {
                let Some(idx) = model_compiler::kernels::in_place_operand(plan, kernel) else {
                    continue;
                };
                if let (Some(&src), Some(&out)) =
                    (op.inputs.get(idx as usize), op.outputs.first())
                {
                    alias.join(src, out);
                }
            }
            _ => {}
        }
    }
    alias
}

fn first_clobber(plan: &ForwardPlan, rows: &[Row]) -> Option<String> {
    walk(plan, rows, &Buffers::assign(plan, rows))
}

/// Walks one assignment and returns the first place a reader's bytes had
/// been taken from under it.
///
/// Takes the assignment rather than computing it, so the negative
/// control can hand it a deliberately broken one.
fn walk(plan: &ForwardPlan, rows: &[Row], buffers: &Buffers) -> Option<String> {
    let n_tokens = rows.len();
    let n_requests = rows
        .iter()
        .filter(|r| !r.multi_token)
        .count()
        .max(rows.iter().filter(|r| r.samples).count())
        .max(1);

    let mut alias = aliases(plan);

    // One stamp per arena byte: which value's ROOT owns it right now.
    const FREE: ValueId = ValueId::MAX;
    let mut owner = vec![FREE; buffers.bytes];

    let extent = |v: ValueId| -> Option<(usize, usize)> {
        let at = *buffers.offset.get(v as usize)?;
        if at == Buffers::NAMED {
            return None; // the backend binds it; not the arena's bytes
        }
        Some((at, value_bytes(plan, v, n_tokens, n_requests)))
    };

    for (i, op) in plan.ops.iter().enumerate() {
        for &v in &op.inputs {
            let Some((at, len)) = extent(v) else { continue };
            let want = alias.root(v);
            for b in at..(at + len).min(owner.len()) {
                if owner[b] != want {
                    return Some(format!(
                        "op {i} ({:?}) reads value {v} at [{at}, {}), but byte \
                         {b} now belongs to value {} — the arena placed it \
                         over a buffer this op still reads",
                        op.kind,
                        at + len,
                        owner[b]
                    ));
                }
            }
        }
        for &v in &op.outputs {
            let Some((at, len)) = extent(v) else { continue };
            let root = alias.root(v);
            for b in at..(at + len).min(owner.len()) {
                owner[b] = root;
            }
        }
    }
    None
}

fn families() -> Vec<(&'static str, FireClass, ForwardPlan)> {
    use model::*;
    let mut out: Vec<(&'static str, FireClass, ForwardPlan)> = Vec::new();
    for f in [FireClass::Decode, FireClass::Prefill] {
        out.push((
            "llama_like",
            f,
            families::llama_like::forward::llama_like_cuda(
                &families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b(),
                &families::llama_like::forward::facts::LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                f,
            ),
        ));
    }
    // The DRIVEN families, both classes. These matter most and were the
    // last to be swept: a declared-only family whose assignment overlaps
    // has nothing to corrupt yet, while these three are executing.
    for f in [FireClass::Decode, FireClass::Prefill] {
        out.push((
            "gemma_4",
            f,
            gemma_4::forward::gemma4_cuda(
                &gemma_4::forward::facts::Gemma4Facts::gemma_4_e4b(),
                &gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic(),
                f,
            ),
        ));
        out.push((
            "gpt_oss",
            f,
            gpt_oss::forward::gpt_oss_cuda(
                &gpt_oss::forward::facts::GptOssFacts::gpt_oss_20b(),
                &gpt_oss::forward::facts::GptOssCudaFacts::gpt_oss_20b_synthetic(),
                f,
            ),
        ));
        out.push((
            "qwen3_5",
            f,
            qwen_3_5::forward::qwen3_5_hybrid_cuda(
                &qwen_3_5::forward::facts::Qwen35HybridFacts::qwen3_5_0_8b(),
                &qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                f,
            ),
        ));
    }

    let d = FireClass::Decode;
    out.push((
        "glm5",
        d,
        glm5::forward::glm5_cuda(&glm5::forward::facts::Glm5Facts::glm5_106b_a12b(), d),
    ));
    out.push((
        "kimi_k2",
        d,
        kimi_k2::forward::kimi_cuda(
            &kimi_k2::forward::facts::KimiFacts::kimi_k2(),
            &kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
            d,
        ),
    ));
    out.push((
        "kimi_k3",
        d,
        kimi_k3::forward::kimi_k3_cuda(&kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(), d),
    ));
    out.push((
        "deepseek_v4",
        d,
        deepseek_v4::forward::dsv4_cuda(&deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(), d),
    ));
    out.push((
        "nemotron_h",
        d,
        nemotron_h::forward::nemotron_h_cuda(
            &nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(),
            d,
        ),
    ));
    out.push((
        "gemma3n",
        d,
        gemma3n::forward::gemma3n_cuda(&gemma3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(), d),
    ));
    out.push((
        "gemma_2",
        d,
        gemma_2::forward::gemma2_cuda(&gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(), d),
    ));
    out
}

/// The invariant, over every declared family: nothing the arena hands
/// out lands on bytes a later op still reads.
///
/// Row counts are chosen to move the two extents independently — 1 is
/// the decode fire, 8 is the batched one, and a fire whose rows are
/// sampled separates `Dim::Requests` from `Dim::Tokens`, which is where
/// an epilogue value would be under-sized.
#[test]
fn no_value_lands_on_bytes_a_later_op_still_reads() {
    for (name, class, plan) in families() {
        for n in [1usize, 8] {
            let mut rows = plain(n);
            rows[0].samples = true;
            if let Some(why) = first_clobber(&plan, &rows) {
                panic!("{name} ({class:?}), {n} rows: {why}");
            }
        }
    }
}

/// A value the arena placed must FIT the arena it reports, and the
/// report is what sizes the driver's block.
///
/// Separate from the clobber walk because it fails differently: an
/// out-of-range offset is a driver segfault, not a wrong number, and it
/// would be invisible above (the walk clamps to the block it was given).
#[test]
fn every_placed_value_fits_the_arena_it_reports() {
    for (name, class, plan) in families() {
        let rows = plain(8);
        let buffers = Buffers::assign(&plan, &rows);
        let n_requests = 1usize.max(rows.len());
        // The block the driver's workspace must hold for this family,
        // printed because nothing else states it: `ws.declared_values`
        // is sized by a formula today, and this is the number it has to
        // cover once the host is the one assigning.
        // Placed vs NAMED is the shape of the remaining driver work, not
        // just a statistic. A value the host PLACES is one whose arm has
        // to stop naming a workspace field; a value it leaves NAMED stays
        // exactly where it is, because a seam exposes it and machinery
        // outside the walk reaches it by name. The four executors between
        // them name about twelve buffer roles (`ws.y`, `ws.norm_x`,
        // `ws.q`, …), so the migration is counted in roles, and this says
        // how many of those roles the host is even asking about.
        let named = buffers
            .offset
            .iter()
            .filter(|&&at| at == Buffers::NAMED)
            .count();
        println!(
            "{name:12} {class:?}  arena {:>9} bytes  {} values ({named} named, {} placed)",
            buffers.bytes,
            plan.values.len(),
            plan.values.len() - named
        );
        for v in 0..plan.values.len() {
            let at = buffers.offset[v];
            if at == Buffers::NAMED {
                continue;
            }
            let len = value_bytes(&plan, v as ValueId, rows.len(), n_requests);
            assert!(
                at + len <= buffers.bytes,
                "{name} ({class:?}): value {v} at [{at}, {}) past the \
                 reported arena of {} bytes",
                at + len,
                buffers.bytes
            );
        }
    }
}

/// The sanity check on the check: a deliberately broken assignment must
/// be caught. Without this, a `first_clobber` that silently walked zero
/// ops would read as every family being sound.
#[test]
fn the_walk_catches_an_overlap_it_is_given() {
    let plan = families()
        .into_iter()
        .find(|(n, _, _)| *n == "gemma_2")
        .map(|(_, _, p)| p)
        .expect("gemma_2 declares a decode text");
    let rows = plain(8);
    assert!(
        first_clobber(&plan, &rows).is_none(),
        "the family must be sound before the negative control means anything"
    );

    // Collapse the arena to ONE buffer — every value at offset 0, which
    // is the crudest possible wrong assignment. The same walk must now
    // report a clobber, and if it does not, its silence on the real
    // assignments above means nothing.
    let broken = {
        let mut b = Buffers::assign(&plan, &rows);
        for at in b.offset.iter_mut() {
            if *at != Buffers::NAMED {
                *at = 0;
            }
        }
        b
    };
    assert!(
        walk(&plan, &rows, &broken).is_some(),
        "an arena that puts every value at offset 0 must clobber"
    );
}

/// Which operands a driver can size FROM THE VALUE, and which it cannot.
///
/// The gemma-4 executor's Matmul arm now derives a GEMM's two extents
/// from its operands' value descriptors instead of tracking `cur_hq`,
/// `cur_inter` and friends per layer. That is only the same number when
/// a traced value's trailing dims are constants, so this gates exactly
/// that: every MATMUL operand, every family.
///
/// It is not true in general, and the exception is worth naming rather
/// than asserting away. `launch_transpose_bf16_nld_to_lnd` turns
/// `[N, L, ple_dim]` into `[L, N, ple_dim]`, which puts `Tokens` in a
/// non-leading position — its row width is a runtime number and
/// `Arg::Arena { width }` is 0. A generic driver therefore cannot assume
/// a width is available for every rectangle; the kernels that need one
/// from elsewhere are printed, and they are the ones whose arms will
/// still read something the value alone does not say.
#[test]
fn a_matmul_operand_has_a_row_width_a_driver_can_derive() {
    use model_compiler::lower::{lower, Arg, Fire};
    use std::collections::BTreeSet;

    let mut widthless: BTreeSet<String> = BTreeSet::new();
    for (name, class, plan) in families() {
        for n in [1usize, 8] {
            let rows = plain(n);
            let Ok(out) = lower(&plan, &rows, Fire::default()) else {
                continue; // a text that will not lower has no operands
            };
            for l in &out.launches {
                let kernel = &out.kernels[l.kernel as usize];
                for a in &out.args[l.args.start as usize..l.args.end as usize] {
                    let width = match a {
                        Arg::Arena { width, .. } | Arg::Named { width, .. } => *width,
                        Arg::Weight(_) => continue,
                    };
                    if width > 0 {
                        continue;
                    }
                    widthless.insert(kernel.clone());
                    // The converted arm derives ITS extents this way, so
                    // for a matmul a missing width is a wrong number on
                    // the device rather than a fact about the tree.
                    assert!(
                        !kernel.starts_with("gemm_"),
                        "{name} ({class:?}), {n} rows: the matmul operand of \
                         `{kernel}` has no fixed row width, so the executor \
                         deriving extents from the value descriptor would \
                         get one wrong"
                    );
                }
            }
        }
    }
    println!("kernels with an operand no value descriptor can size:");
    for k in &widthless {
        println!("  {k}");
    }
}

/// How close the assignment is to the LIVENESS BOUND.
///
/// The reported arena is what the driver's block must hold, and it
/// scales with the fire's rows — so a block sized for `max_tokens` is
/// this number times the batch. That only works if the number is near
/// the floor, and the floor is a property of the text rather than of
/// the allocator: at each op, the bytes of every value then live. No
/// assignment can beat that peak; the ratio above it is what the free
/// list is losing to fragmentation.
///
/// Prints rather than gates, because the floor moves with the text and
/// a threshold here would be a number nobody could act on. What it is
/// for is deciding whether `ws.declared_values` can be sized from
/// `arena_bytes` at all.
#[test]
fn how_much_the_arena_costs_over_the_liveness_floor() {
    for (name, class, plan) in families() {
        let rows = plain(8);
        let n_tokens = rows.len();
        let n_requests = rows.len();
        let buffers = Buffers::assign(&plan, &rows);

        // Last use per value, folded onto nothing: the floor does not
        // care who owns which bytes, only how many are readable at once.
        let mut last = vec![0usize; plan.values.len()];
        let mut def = vec![usize::MAX; plan.values.len()];
        for (i, op) in plan.ops.iter().enumerate() {
            for &v in op.inputs.iter().chain(op.outputs.iter()) {
                if let Some(s) = last.get_mut(v as usize) {
                    *s = (*s).max(i);
                }
            }
            for &v in &op.outputs {
                if let Some(s) = def.get_mut(v as usize) {
                    *s = (*s).min(i);
                }
            }
        }
        let mut floor = 0usize;
        for i in 0..plan.ops.len() {
            let mut live = 0usize;
            for v in 0..plan.values.len() {
                if buffers.offset[v] == Buffers::NAMED {
                    continue;
                }
                if def[v] <= i && i <= last[v] {
                    live += value_bytes(&plan, v as ValueId, n_tokens, n_requests);
                }
            }
            floor = floor.max(live);
        }
        let ratio = buffers.bytes as f64 / floor.max(1) as f64;
        println!(
            "{name:12} {class:?}  arena {:>9}  floor {:>9}  x{ratio:.2}",
            buffers.bytes, floor
        );
    }
}

/// WHAT the arena is holding — the largest placed values, by width.
///
/// The floor above is ~1.5 MB per row for gemma-4, which is several
/// times what the hand-written workspace allocates per row, and a block
/// sized from it at `max_tokens` would be the difference between a
/// viable plumbing and an impossible one. So this asks which values it
/// is, rather than reasoning about which it might be.
#[test]
fn which_values_the_arena_is_actually_holding() {
    use std::collections::BTreeMap;

    for (name, class, plan) in families() {
        if class != FireClass::Decode {
            continue;
        }
        if name != "gemma_4" && name != "glm5" && name != "llama_like" {
            continue;
        }
        let rows = plain(8);
        let buffers = Buffers::assign(&plan, &rows);
        // Group by the SHAPE a value has, since the trace is
        // layer-unrolled and 35 layers name 35 of the same role.
        let mut by_shape: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for v in 0..plan.values.len() {
            if buffers.offset[v] == Buffers::NAMED {
                continue;
            }
            let info = &plan.values[v];
            let bytes = value_bytes(&plan, v as ValueId, rows.len(), rows.len());
            let key = format!("{:?} {:?}", info.shape.0, info.dtype);
            let e = by_shape.entry(key).or_insert((0, 0));
            e.0 += 1;
            e.1 += bytes;
        }
        let mut rows_out: Vec<_> = by_shape.into_iter().collect();
        rows_out.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
        println!("\n{name} {class:?} — arena {} bytes", buffers.bytes);
        for (shape, (count, bytes)) in rows_out.into_iter().take(5) {
            println!("  {count:5} x  {bytes:>10} total  {shape}");
        }
    }
}

/// gemma-4 E2B at ONE row, which is the fire the driver refused.
///
/// The driver reported wanting 299 MB of activation block for a
/// single-token decode, and a number that large at N=1 cannot be the
/// text: nothing in gemma-4 is 299 MB wide per row. So either the
/// report is right and the assignment has a defect, or it is wrong --
/// and this is the cheapest place to tell which.
#[test]
fn what_gemma_4_e2b_asks_for_at_one_row() {
    use model::gemma_4::forward::{self, facts};
    use std::collections::BTreeMap;

    for n in [1usize, 8] {
        let plan = forward::gemma4_cuda(
            &facts::Gemma4Facts::gemma_4_e2b(),
            // E2B binds no packed banks: the two-gemm MLP pair and the
            // unfused QKV, which is the branch E4B does not take.
            &facts::Gemma4CudaFacts {
                fused_qkv: false,
                gate_up_fused: false,
                kv_native_bf16: true,
            },
            FireClass::Decode,
        );
        let rows = plain(n);
        let buffers = Buffers::assign(&plan, &rows);
        println!(
            "\ngemma_4 E2B, {n} rows: arena {} bytes over {} values, {} ops",
            buffers.bytes,
            plan.values.len(),
            plan.ops.len()
        );
        let mut by_shape: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for v in 0..plan.values.len() {
            if buffers.offset[v] == Buffers::NAMED {
                continue;
            }
            let info = &plan.values[v];
            let bytes = value_bytes(&plan, v as ValueId, rows.len(), rows.len());
            let e = by_shape
                .entry(format!("{:?} {:?}", info.shape.0, info.dtype))
                .or_insert((0, 0));
            e.0 += 1;
            e.1 += bytes;
        }
        let mut out: Vec<_> = by_shape.into_iter().collect();
        out.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
        for (shape, (count, bytes)) in out.into_iter().take(5) {
            println!("  {count:5} x  {bytes:>12} total  {shape}");
        }
    }
}

/// WHAT IS LIVE at the peak, which is the only thing that can shorten
/// the arena.
///
/// The assignment runs at 1.6x its liveness floor, so a better free list
/// is worth 1.6x and no more; the remaining ~4x over the hand-written
/// workspace is in how long the traced values live. That is a fact about
/// the DECLARATION, and this is the view of it: the op where the most
/// bytes are simultaneously readable, and what is holding them.
///
/// Excludes the values the host declined to place — those are the
/// backend's and are not what the arena is sizing for.
#[test]
fn what_is_live_where_the_arena_peaks() {
    for (name, class, plan) in families() {
        if class != FireClass::Decode {
            continue;
        }
        if name != "gemma_4" && name != "llama_like" {
            continue;
        }
        // A REALISTIC shape, not one row: at one row the logits dwarf
        // everything and the peak lands on the epilogue, which says
        // nothing about the token-scaled cost that actually sizes the
        // block. 64 tokens with 4 sampled puts the peak back in the
        // layers, where it is at 6144.
        let rows: Vec<Row> = (0..64)
            .map(|i| Row {
                samples: i < 4,
                multi_token: i >= 4,
                ..Row::default()
            })
            .collect();
        let (n_tokens, n_requests) = (rows.len(), 4);
        let buffers = Buffers::assign(&plan, &rows);

        let mut last = vec![0usize; plan.values.len()];
        let mut def = vec![usize::MAX; plan.values.len()];
        for (i, op) in plan.ops.iter().enumerate() {
            for &v in op.inputs.iter().chain(op.outputs.iter()) {
                if let Some(s) = last.get_mut(v as usize) {
                    *s = (*s).max(i);
                }
            }
            for &v in &op.outputs {
                if let Some(s) = def.get_mut(v as usize) {
                    *s = (*s).min(i);
                }
            }
        }
        let placed = |v: usize| buffers.offset[v] != Buffers::NAMED;
        let bytes = |v: usize| value_bytes(&plan, v as ValueId, n_tokens, n_requests);

        let mut peak_at = 0usize;
        let mut peak = 0usize;
        for i in 0..plan.ops.len() {
            let live: usize = (0..plan.values.len())
                .filter(|&v| placed(v) && def[v] <= i && i <= last[v])
                .map(bytes)
                .sum();
            if live > peak {
                peak = live;
                peak_at = i;
            }
        }
        println!(
            "\n{name}: peak {peak} bytes live at op {peak_at}/{} ({:?})",
            plan.ops.len(),
            plan.ops[peak_at].kind
        );
        // Who is holding it, longest-lived first: a value defined far
        // before the peak and read far after is the one the text could
        // stop carrying.
        let mut holders: Vec<(usize, usize, usize)> = (0..plan.values.len())
            .filter(|&v| placed(v) && def[v] <= peak_at && peak_at <= last[v])
            .map(|v| (bytes(v), last[v] - def[v], v))
            .collect();
        println!("  holding {} values", holders.len());
        // Grouped by SHAPE: the trace is layer-unrolled, so one role
        // appears once per layer and a per-value list would be 35 copies
        // of the same answer.
        let mut by_shape: std::collections::BTreeMap<String, (usize, usize, usize)> =
            Default::default();
        for (b, span, v) in holders {
            let e = by_shape
                .entry(format!("{:?}", plan.values[v].shape.0))
                .or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += b;
            e.2 = e.2.max(span);
        }
        let mut rows_out: Vec<_> = by_shape.into_iter().collect();
        rows_out.sort_by_key(|(_, (_, b, _))| std::cmp::Reverse(*b));
        println!("  {:>5}  {:>11}  {:>8}  {}", "count", "bytes", "max span", "shape");
        for (shape, (count, b, span)) in rows_out.into_iter().take(8) {
            println!("  {count:>5}  {b:>11}  {span:>8}  {shape}");
        }
    }
}

/// The block a REALISTIC widest fire needs — which is what sizing
/// `ws.declared_values` actually costs.
///
/// The 2.34 MB per row measured through the driver was a harness shape
/// where every row samples, so the `[Requests, vocab]` logits scaled
/// with TOKENS. Production does not: sampled rows are bounded by
/// `max_logit_rows`, far below `max_tokens`, and the peak is otherwise
/// almost entirely those logits. So the honest probe marks the
/// non-sampled rows `multi_token`, which is what makes
/// `Buffers::assign` count requests rather than tokens.
#[test]
fn what_a_widest_fire_actually_costs() {
    for (name, class, plan) in families() {
        if class != FireClass::Decode {
            continue;
        }
        for (tokens, sampled) in [(6144usize, 512usize), (6144, 64)] {
            let rows: Vec<Row> = (0..tokens)
                .map(|i| Row {
                    samples: i < sampled,
                    // Rows a request does not sample are interior rows
                    // of a multi-token request; without this every row
                    // counts as its own request and the logits are
                    // sized by TOKENS.
                    multi_token: i >= sampled,
                    ..Row::default()
                })
                .collect();
            let b = Buffers::assign(&plan, &rows);
            // The FLOOR at this same shape. The earlier ratio was taken
            // at eight all-sampled rows, where the logits dominate both
            // sides and the ratio says little; at the shape that sizes
            // the block it separates "the text holds this much" from
            // "the free list is losing this much".
            let mut last = vec![0usize; plan.values.len()];
            let mut def = vec![usize::MAX; plan.values.len()];
            for (i, op) in plan.ops.iter().enumerate() {
                for &v in op.inputs.iter().chain(op.outputs.iter()) {
                    if let Some(s) = last.get_mut(v as usize) { *s = (*s).max(i); }
                }
                for &v in &op.outputs {
                    if let Some(s) = def.get_mut(v as usize) { *s = (*s).min(i); }
                }
            }
            let sizes: Vec<usize> = (0..plan.values.len())
                .map(|v| if b.offset[v] == Buffers::NAMED { 0 }
                     else { value_bytes(&plan, v as ValueId, tokens, sampled.max(1)) })
                .collect();
            let mut floor = 0usize;
            let mut cur = 0usize;
            let mut ends: Vec<Vec<usize>> = vec![Vec::new(); plan.ops.len() + 1];
            for v in 0..plan.values.len() {
                if sizes[v] == 0 || def[v] == usize::MAX { continue; }
                ends[last[v].min(plan.ops.len())].push(v);
            }
            for i in 0..plan.ops.len() {
                for &v in &plan.ops[i].outputs {
                    if (v as usize) < sizes.len() && def[v as usize] == i {
                        cur += sizes[v as usize];
                    }
                }
                floor = floor.max(cur);
                for &v in &ends[i] { cur -= sizes[v]; }
            }
            println!(
                "{name:12} {tokens}t/{sampled}s -> arena {:>13} ({:.2} GB)  floor {:>13} ({:.2} GB)  x{:.2}",
                b.bytes,
                b.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                floor,
                floor as f64 / (1024.0 * 1024.0 * 1024.0),
                b.bytes as f64 / floor.max(1) as f64
            );
        }
    }
}

/// Values READ over rows their producer never WROTE.
///
/// The leading hypothesis for why gemma-4 goes wrong on host-assigned
/// buffers. A per-role workspace buffer is allocated once and reused
/// across fires, so rows a kernel skips still hold something
/// shape-compatible from last time; a packed arena gives those same rows
/// a different value's bytes. Any statement writing fewer rows than a
/// later one reads is correct under the convention and wrong under the
/// arena — and that is a property of the TEXT, visible here.
///
/// Reports rather than gates: a gap may be legitimate (a consumer that
/// masks the rows it did not get), and the point is to see whether any
/// exist at all before believing the hypothesis.
#[test]
fn which_values_are_read_wider_than_they_are_written() {
    use model_compiler::lower::{lower, Fire};
    use std::collections::BTreeMap;

    for (name, class, plan) in families() {
        if class != FireClass::Decode {
            continue;
        }
        // A fire whose rows are NOT uniform, so peels and live-row
        // windows actually split: half the rows truncated, one sampled.
        let rows: Vec<Row> = (0..16)
            .map(|i| Row {
                samples: i == 0,
                multi_token: i != 0,
                depth_k: if i >= 8 { Some(4) } else { None },
                ..Row::default()
            })
            .collect();
        let Ok(out) = lower(&plan, &rows, Fire::default()) else {
            continue;
        };
        let mut wrote: BTreeMap<ValueId, (u32, u32)> = BTreeMap::new();
        let mut read: BTreeMap<ValueId, (u32, u32)> = BTreeMap::new();
        let span = |m: &mut BTreeMap<ValueId, (u32, u32)>, v: ValueId, r: &std::ops::Range<u32>| {
            let e = m.entry(v).or_insert((u32::MAX, 0));
            e.0 = e.0.min(r.start);
            e.1 = e.1.max(r.end);
        };
        for l in &out.launches {
            let op = &plan.ops[l.op as usize];
            for &v in &op.outputs {
                span(&mut wrote, v, &l.rows);
            }
            for &v in &op.inputs {
                span(&mut read, v, &l.rows);
            }
        }
        let mut gaps = 0usize;
        let mut worst: Option<(ValueId, (u32, u32), (u32, u32))> = None;
        for (&v, &(rlo, rhi)) in &read {
            let Some(&(wlo, whi)) = wrote.get(&v) else {
                continue; // an input nothing in this fire produced
            };
            if rlo < wlo || rhi > whi {
                gaps += 1;
                if worst.is_none() {
                    worst = Some((v, (wlo, whi), (rlo, rhi)));
                }
            }
        }
        // READ THE RESULT CAREFULLY. The epilogue's statements run over
        // `Dim::Requests` while the body runs over `Dim::Tokens`, and
        // this compares row NUMBERS without knowing which space they
        // are in -- so a value on that boundary looks like a gap and is
        // not one. gemma-4, gemma-2 and gemma-3n each report exactly one
        // such value, and it is the last in the plan.
        //
        // llama_like's is the interesting one and is in TOKEN space
        // both sides: written over the full-depth prefix, read over
        // every row. That is what a depth window is for, and whether the
        // consumer masks the truncated rows is the question it raises.
        match worst {
            None => println!("{name:12} no value is read wider than written"),
            Some((v, w, r)) => println!(
                "{name:12} {gaps} read wider than written; first v{v} of {} written {w:?} read {r:?}",
                plan.values.len()
            ),
        }
    }
}
