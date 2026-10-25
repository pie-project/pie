//! What each op takes and yields when ranks split its values. An op without a
//! rule here gets every value whole and stays as it is; it may not read a
//! weight or a cache the ranks split.

use poem_ir::{
    ActFn, Attn, BinaryOp, Collective, Dim, Dtype, Elemwise, Gather, Linear, NormAxis, Operands,
    Operation, RaggedMask, Shard, Ty, ValueId,
};

use super::{Block, Dist, Pass};
use crate::tree::{self, Tree};

/// How an op meets the ranks.
pub(super) enum Rule {
    /// Every value whole.
    Whole,
    /// Split by head or channel with the values it follows.
    Heads(Heads),
    /// `act · w`: a weight split along its `out` axis splits the output, one
    /// split along its `in` axis leaves each rank a partial sum.
    Contract {
        act: &'static str,
        w: &'static str,
        out: u32,
        input: u32,
    },
    /// Linear in `follow` jointly, so partial sums stay one.
    Linear { follow: &'static [&'static str] },
    /// A row lookup in a table the ranks split by row: each rank holds the
    /// rows it has, zeros elsewhere.
    Lookup { table: &'static str },
    /// An attention plan: its head counts are each rank's share.
    Plan { scale: &'static [&'static str] },
    /// A row norm, which needs the whole row unless its weight is split
    /// with it (or it has none): then each rank norms its own share by the
    /// row's sum of squares, its share's summed across the ranks.
    Norm,
    /// Group indices for a grouped matmul, which reads no value: they are
    /// a rank's share of the groups when the row it groups is split.
    Groups,
}

/// An op whose `follow` inputs are split by head or channel, every output
/// with them, and whose `scale` counts are each rank's share. Other
/// activations are whole.
#[derive(Clone, Copy)]
pub(super) struct Heads {
    pub(super) follow: &'static [&'static str],
    pub(super) scale: &'static [&'static str],
    pub(super) cache: Cache,
    /// The kv it appends to a cache or attends to without one, cut as its
    /// kv heads are.
    pub(super) kv: &'static [&'static str],
}

/// How the cache an op reads is held against the values it follows.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Cache {
    /// It reads none the ranks split.
    Whole,
    /// Split exactly when its values are.
    Split,
    /// Split when its values are, or kept whole on every rank while they
    /// split the query heads that share its kv heads.
    SplitOrWhole,
}

impl Cache {
    pub(super) fn allows(self, cut: bool, split: bool) -> bool {
        match self {
            Cache::Whole => !cut,
            Cache::Split => cut == split,
            Cache::SplitOrWhole => split || !cut,
        }
    }
}

impl Heads {
    const fn of(follow: &'static [&'static str], scale: &'static [&'static str]) -> Heads {
        Heads {
            follow,
            scale,
            cache: Cache::Whole,
            kv: &[],
        }
    }

    const fn cache(self, cache: Cache) -> Heads {
        Heads { cache, ..self }
    }

    const fn kv(self, kv: &'static [&'static str]) -> Heads {
        Heads { kv, ..self }
    }
}

/// Unpacks whose widths name, in order, the blocks of the packed share
/// they take apart.
const UNPACK_WIDTHS: [(&str, &[&str]); 2] = [
    ("gather.split_qkv", &["q_width", "kv_width"]),
    ("gather.split_rows", &["width"]),
];

/// The head count of kv a group of ranks may share.
pub(super) const KV_HEADS: &str = "kv_heads";

/// A pair of head counts, q's and then kv's, which a group of ranks may
/// share as `KV_HEADS` it names.
const HEAD_PAIR: &str = "heads";

/// How op `name` splits by head or channel, if it does.
fn heads(name: &str) -> Option<Heads> {
    Some(match name {
        "elemwise.act" | "elemwise.rope_axes" | "gather.rows" | "gather.pack_rows"
        | "gather.unpack_rows" | "linear.gelu_tanh" | "linear.relu2" => Heads::of(&["x"], &[]),
        "elemwise.gate_sigmoid_mul" | "elemwise.rmsnorm_gated" => Heads::of(&["x", "gate"], &[]),
        "elemwise.rmsnorm_gated_by" => Heads::of(&["x", "gate"], &["heads"]),
        // q, and k where the rotation turns one.
        "elemwise.rope" => Heads::of(&["q", "k"], &[]),
        "elemwise.add_bias" => Heads::of(&["out"], &[]),
        "gather.split_qkv" => Heads::of(&["packed"], &["q_width", "kv_width"]),
        // Its heads are q's and kv's, each rank's share of each.
        "gather.split_qkv_padded" => Heads::of(&["packed"], &[HEAD_PAIR]),
        "gather.split_q_gate" => Heads::of(&["packed"], &[]),
        "gather.split_rows" => Heads::of(&["x"], &["width"]),
        "linear.matmul_grouped" => Heads::of(&["x", "routes"], &["groups"]),
        "attn.ragged" => Heads::of(&["q", "k", "v"], &[KV_HEADS]).kv(&["k", "v"]),
        "attn.sink" => Heads::of(&["o", "lse"], &[]),
        "attn.merge_lse" => Heads::of(&["o1", "lse1", "o2", "lse2"], &["heads"]),
        "attn.pool_lse" => Heads::of(&["q"], &["heads"]),
        // k and v, or the one plane a shared pool reads as both.
        "attn.kv_append" => Heads::of(&["k", "v"], &[])
            .cache(Cache::Split)
            .kv(&["k", "v"]),
        "attn.mla_split_q_b" => Heads::of(&["q_b"], &["heads"]),
        "attn.mla_absorb_q" => Heads::of(&["q_nope"], &["heads"]),
        "attn.mla_absorb_out" => Heads::of(&["latent"], &["heads"]),
        "attn.mla_decode" | "attn.mla_prefill" => Heads::of(&["q", "q_pe"], &["heads"]),
        "ssm.causal_conv1d" | "ssm.causal_conv1d_chunked" => {
            Heads::of(&["x"], &[]).cache(Cache::Split)
        }
        "ssm.gdn_gates" => Heads::of(&["ba"], &[]),
        "ssm.short_conv" | "ssm.short_conv_chunked" => Heads::of(&["x"], &[]).cache(Cache::Split),
        // Each head's run of `x` against the one table every head reads.
        "linear.rel_bias" => Heads::of(&["x"], &["heads"]),
        "ssm.mamba2" | "ssm.mamba2_chunked" => {
            Heads::of(&["xbc", "dt", "z"], &["heads", "groups"]).cache(Cache::Split)
        }
        "ssm.gated_delta" | "ssm.gated_delta_chunked" => {
            Heads::of(&["qkv", "z", "gates"], &["k_heads", "v_heads"]).cache(Cache::Split)
        }
        "ssm.kda" | "ssm.kda_chunked" => {
            Heads::of(&["mixed", "f", "b"], &["heads"]).cache(Cache::Split)
        }
        _ => return None,
    })
}

/// [`rule`], a norm over heads as wide as the row it reads (a row norm
/// spelled a head the row wide) read as the row norm it is.
pub(super) fn rule_of(op: &Operation, width: impl Fn(ValueId) -> Option<u64>) -> Rule {
    if let Operation::Elemwise(Elemwise::Rmsnorm {
        x,
        axis: NormAxis::Head(d),
        ..
    }) = op
        && width(*x) == Some(u64::from(*d))
    {
        return Rule::Norm;
    }
    rule(op)
}

pub(super) fn rule(op: &Operation) -> Rule {
    // A norm splits by its axis: a whole row's weight against the row's
    // blocks, a head's like any per-head op, a group's not at all.
    if let Operation::Elemwise(Elemwise::Rmsnorm { axis, .. }) = op {
        return match axis {
            NormAxis::Row => Rule::Norm,
            NormAxis::Head(_) => Rule::Heads(Heads::of(&["x"], &[])),
            NormAxis::Group(_) => Rule::Whole,
        };
    }
    // A gated unit splits with its columns, packed or as two planes; one of
    // routed rows scales each row by its route's weight, which every rank
    // reads whole.
    if let Operation::Linear(Linear::Glu { packed, .. }) = op {
        return match packed {
            Some(_) => Rule::Heads(Heads::of(&["packed"], &["intermediate"])),
            None => Rule::Heads(Heads::of(&["gate", "up"], &["intermediate"])),
        };
    }
    // A scale (the identity times a scalar) is linear in what it scales.
    if let Operation::Elemwise(Elemwise::Act {
        act: ActFn::Identity,
        ..
    }) = op
    {
        return Rule::Linear { follow: &["x"] };
    }
    // A product contracts with its weight's split, its activations quantized
    // or not: one that quantizes them does so on each rank's share.
    if let Operation::Linear(Linear::Matmul { .. }) = op {
        return Rule::Contract {
            act: "act",
            w: "w",
            out: 0,
            input: 1,
        };
    }
    // A plain or concatenated lookup reads the rows each rank holds of a
    // row-split table; a weighted one reads its table whole.
    if let Operation::Gather(Gather::Embed { weights, .. }) = op {
        return if weights.is_none() {
            Rule::Lookup { table: "table" }
        } else {
            Rule::Whole
        };
    }
    // A sum is linear in both operands, so partial sums stay one; a product
    // splits with them by head or channel.
    if let Operation::Elemwise(Elemwise::Binary { op: binary, .. }) = op {
        return match binary {
            BinaryOp::Add => Rule::Linear {
                follow: &["x", "y"],
            },
            BinaryOp::Mul => Rule::Heads(Heads::of(&["x", "y"], &[])),
        };
    }
    // A residual add is linear in its two rows; one under a gate reads its
    // operands whole.
    if let Operation::Elemwise(Elemwise::ResidualAdd { gate, .. }) = op {
        return match gate {
            None => Rule::Linear {
                follow: &["x", "y"],
            },
            Some(_) => Rule::Whole,
        };
    }
    // An expert product contracts with its bank's split, as a product does.
    if let Operation::Linear(Linear::MoeSelect { .. }) = op {
        return Rule::Contract {
            act: "x",
            w: "bank",
            out: 1,
            input: 2,
        };
    }
    // A weighted sum of routed rows is linear in them, and in a shared row
    // it adds.
    if let Operation::Linear(Linear::MoeCombine { shared, .. }) = op {
        return match shared {
            None => Rule::Linear {
                follow: &["routed"],
            },
            Some(_) => Rule::Linear {
                follow: &["routed", "shared"],
            },
        };
    }
    // A paged attention splits with its query heads, its cache split with
    // them or kept whole; a relative bias, a head's run of positions after
    // another, splits with them.
    // A ragged attention under a bucketed relative bias reads its table a
    // row a head, which no share of the last axis cuts: it reads its
    // operands whole.
    if let Operation::Attn(Attn::Ragged {
        mask: RaggedMask::RelativeBias { .. },
        ..
    }) = op
    {
        return Rule::Whole;
    }
    if let Operation::Attn(Attn::Decode { .. }) = op {
        return Rule::Heads(Heads::of(&["q", "bias"], &[]).cache(Cache::SplitOrWhole));
    }
    if let Operation::Attn(Attn::Prefill { .. }) = op {
        return Rule::Heads(Heads::of(&["q", "bias"], &[KV_HEADS]).cache(Cache::SplitOrWhole));
    }
    let name = op.name();
    match name {
        "linear.moe_sigmoid_gate_add" => Rule::Linear {
            follow: &["routed", "shared"],
        },
        "attn.plan_prefill" | "attn.plan_decode" => Rule::Plan {
            scale: &["q_heads", KV_HEADS],
        },
        "attn.mla_plan" => Rule::Plan { scale: &["heads"] },
        "linear.group_routes" => Rule::Groups,
        _ => heads(name).map_or(Rule::Whole, Rule::Heads),
    }
}

/// `op` as one rank runs it, and how the ranks hold each of its outputs.
pub(super) fn apply(pass: &mut Pass, op: &Operation) -> Result<(Operation, Vec<Dist>), String> {
    let mut ins = Vec::new();
    op.inputs(&mut ins);
    let mut at = At {
        tree: tree::of(op),
        name: op.name(),
        ins,
    };
    let mut outs = Vec::new();
    op.outputs(&mut outs);
    let dists = match rule_of(op, |v| pass.width(v)) {
        Rule::Whole => {
            at.whole_except(pass, &[])?;
            at.check_state(pass, false, Cache::Whole)?;
            vec![Dist::Whole; outs.len()]
        }
        Rule::Plan { scale } => {
            at.whole_except(pass, &[])?;
            at.check_state(pass, false, Cache::Whole)?;
            let split_kv = outs.iter().any(|plan| pass.planned_kv_split(*plan));
            at.scale(pass, scale, split_kv)?;
            vec![Dist::Whole; outs.len()]
        }
        Rule::Lookup { table } => {
            at.whole_except(pass, &[table])?;
            let cut = match at.one(table) {
                Some(t) => pass.weight_cut(t)?,
                None => None,
            };
            match cut {
                Some((0, _)) => vec![Dist::Partial; outs.len()],
                Some((axis, _)) => {
                    return Err(format!(
                        "`{}` reads a table split along axis {axis}",
                        at.name
                    ));
                }
                None => vec![Dist::Whole; outs.len()],
            }
        }
        Rule::Norm => {
            at.whole_except(pass, &["x"])?;
            let x = at.one("x").expect("a norm reads a row");
            let cut = match at.one("weight") {
                Some(w) => pass.weight_cut(w)?,
                None => None,
            };
            let weighed = at.one("weight").is_some();
            match (pass.dist(x)?, cut) {
                (Dist::Split(blocks), Some((0, _))) => {
                    let op = shared_norm(pass, op, x, &blocks)?;
                    return Ok((op, vec![Dist::Split(blocks); outs.len()]));
                }
                (Dist::Split(blocks), None) if !weighed => {
                    let op = shared_norm(pass, op, x, &blocks)?;
                    return Ok((op, vec![Dist::Split(blocks); outs.len()]));
                }
                (_, None) => {
                    at.make_whole(pass, x)?;
                    vec![Dist::Whole; outs.len()]
                }
                (held, Some(_)) => {
                    return Err(format!(
                        "`{}` norms a row held {held:?} with a weight the ranks split",
                        at.name
                    ));
                }
            }
        }
        Rule::Groups => {
            let grouped = outs.iter().any(|routes| pass.grouped_row_split(*routes));
            if grouped {
                at.scale(pass, &["groups"], false)?;
                outs.iter()
                    .map(|o| {
                        let width = pass.width(*o).ok_or_else(|| {
                            format!("`{}` yields groups of no fixed width", at.name)
                        })?;
                        Ok(Dist::even(width, pass.world))
                    })
                    .collect::<Result<_, String>>()?
            } else {
                vec![Dist::Whole; outs.len()]
            }
        }
        Rule::Linear { follow } => {
            at.whole_except(pass, follow)?;
            let followed = at.all_of(follow);
            let held: Vec<Dist> = followed
                .iter()
                .map(|v| pass.dist(*v))
                .collect::<Result<_, _>>()?;
            if held.iter().all(|d| *d == held[0]) {
                vec![held[0].clone(); outs.len()]
            } else {
                for v in followed {
                    at.make_whole(pass, v)?;
                }
                vec![Dist::Whole; outs.len()]
            }
        }
        Rule::Contract { act, w, out, input } => {
            at.whole_except(pass, &[act, w])?;
            let a = at.one(act).expect("a contraction reads an activation");
            let weight = at.one(w).expect("a contraction reads a weight");
            match pass.weight_cut(weight)? {
                None => {
                    at.make_whole(pass, a)?;
                    vec![Dist::Whole; outs.len()]
                }
                Some((axis, blocks)) if axis == out => {
                    at.make_whole(pass, a)?;
                    vec![Dist::Split(blocks); outs.len()]
                }
                Some((axis, blocks)) if axis == input => match pass.dist(a)? {
                    held @ Dist::Split(_)
                        if !held.is_even(pass.world)
                            || blocks.iter().any(|b| b.parts != u64::from(pass.world)) =>
                    {
                        return Err(format!(
                            "`{}` contracts heads a group of ranks holds alike, whose \
                             partial sums would count each head once per rank",
                            at.name
                        ));
                    }
                    Dist::Split(_) => vec![Dist::Partial; outs.len()],
                    held => {
                        return Err(format!(
                            "`{}` contracts a weight split along its input with an \
                             activation held {held:?}, made by `{}`",
                            at.name,
                            pass.maker(a)
                        ));
                    }
                },
                Some((axis, _)) => {
                    return Err(format!(
                        "`{}` reads a weight split along axis {axis}",
                        at.name
                    ));
                }
            }
        }
        Rule::Heads(heads) => {
            let follow = heads.follow;
            at.whole_except(pass, follow)?;
            for v in at.all_of(follow) {
                if pass.dist(v)? == Dist::Partial {
                    at.make_whole(pass, v)?;
                }
            }
            let followed = at.all_of(follow);
            let held: Vec<Dist> = followed
                .iter()
                .map(|v| pass.dist(*v))
                .collect::<Result<_, _>>()?;
            let split = held.iter().any(|d| matches!(d, Dist::Split(_)));
            if split && held.contains(&Dist::Whole) {
                return Err(format!("`{}` mixes split and whole values", at.name));
            }
            at.check_state(pass, split, heads.cache)?;
            if split {
                let whole_kv = heads.cache == Cache::SplitOrWhole && !at.reads_split_cache(pass);
                at.check_kv_parts(pass, heads.kv)?;
                let pieces = at.pieces(&held);
                at.scale(pass, heads.scale, !whole_kv)?;
                at.unpack(&pieces);
                outs.iter()
                    .enumerate()
                    .map(|(i, o)| held_like(pass, *o, i, &followed, &held, outs.len()))
                    .collect::<Result<_, String>>()?
            } else {
                vec![Dist::Whole; outs.len()]
            }
        }
    };
    Ok((tree::op(at.tree), dists))
}

/// A row norm over `x`, a row the ranks split in `blocks`, as one rank
/// runs it: its share's sum of squares, summed across the ranks, norms the
/// share (Megatron's norm over a row of heads the ranks split). The ops
/// before it land ahead of the norm; a share a group of ranks holds alike
/// is summed once per rank of the group, which the whole width counts.
fn shared_norm(
    pass: &mut Pass,
    op: &Operation,
    x: ValueId,
    blocks: &[Block],
) -> Result<Operation, String> {
    let Operation::Elemwise(Elemwise::Rmsnorm { weight, eps, y, .. }) = *op else {
        return Err(format!("`{}` is not a row norm", op.name()));
    };
    let world = u64::from(pass.world);
    let parts = blocks.first().map_or(world, |b| b.parts);
    if blocks.iter().any(|b| b.parts != parts) || parts == 0 || !world.is_multiple_of(parts) {
        return Err(format!(
            "`{}` norms a row whose blocks the ranks cut {:?} ways",
            op.name(),
            blocks.iter().map(|b| b.parts).collect::<Vec<_>>()
        ));
    }
    let whole: u64 = blocks.iter().map(|b| b.width).sum::<u64>() * (world / parts);
    let width = u32::try_from(whole).map_err(|_| format!("a row {whole} wide"))?;
    let ty = match &pass.trace.values[x.0 as usize].ty {
        Ty::Tensor { shape, .. } => {
            let mut shape = shape.clone();
            if let Some(last) = shape.last_mut() {
                *last = Dim::Const(1);
            }
            Ty::Tensor {
                shape,
                dtype: Dtype::F32,
            }
        }
        Ty::Struct(_) => return Err(format!("`{}` norms no tensor", op.name())),
    };
    let ss = pass.value(ty.clone(), Dist::Partial);
    let summed = pass.value(ty, Dist::Whole);
    pass.pending
        .push(Operation::Elemwise(Elemwise::SumSquares { x, ss }));
    pass.pending
        .push(Operation::Collective(Collective::AllReduce {
            buf: ss,
            buf_out: summed,
        }));
    Ok(Operation::Elemwise(Elemwise::RmsnormBy {
        x,
        ss: summed,
        weight,
        width,
        eps,
        y,
    }))
}

/// An op being rewritten for one rank, and the values it reads.
struct At {
    tree: Tree,
    name: &'static str,
    ins: Vec<ValueId>,
}

impl At {
    fn all(&self, name: &str) -> Vec<ValueId> {
        tree::field(&self.tree, name)
            .map(tree::ids)
            .unwrap_or_default()
    }

    fn all_of(&self, names: &[&str]) -> Vec<ValueId> {
        names.iter().flat_map(|n| self.all(n)).collect()
    }

    fn one(&self, name: &str) -> Option<ValueId> {
        self.all(name).first().copied()
    }

    /// Every activation outside `keep` made whole.
    fn whole_except(&mut self, pass: &mut Pass, keep: &[&str]) -> Result<(), String> {
        let kept = self.all_of(keep);
        for v in self.ins.clone() {
            if !kept.contains(&v) && !pass.is_weight(v) && pass.cache_split(v).is_none() {
                self.make_whole(pass, v)?;
            }
        }
        Ok(())
    }

    fn make_whole(&mut self, pass: &mut Pass, v: ValueId) -> Result<(), String> {
        let whole = pass.whole(v)?;
        if whole != v {
            let swap = |id: ValueId| if id == v { whole } else { id };
            self.tree = tree::renumbered(std::mem::replace(&mut self.tree, Tree::Unit), &swap);
            for id in &mut self.ins {
                *id = swap(*id);
            }
        }
        Ok(())
    }

    /// An op on whole values reads no weight the ranks split, and a cache it
    /// reads is held as its `cache` allows for how its values are.
    fn check_state(&self, pass: &Pass, split: bool, cache: Cache) -> Result<(), String> {
        for v in &self.ins {
            if !split && pass.weight_cut(*v)?.is_some() {
                return Err(format!("`{}` reads a weight the ranks split", self.name));
            }
            if let Some(cut) = pass.cache_split(*v)
                && !cache.allows(cut, split)
            {
                return Err(format!(
                    "`{}` reads a cache {} while its values are {}",
                    self.name,
                    if cut {
                        "the ranks split"
                    } else {
                        "every rank holds whole"
                    },
                    if split { "split" } else { "whole" },
                ));
            }
        }
        Ok(())
    }

    fn reads_split_cache(&self, pass: &Pass) -> bool {
        self.ins.iter().any(|v| pass.cache_split(*v) == Some(true))
    }

    /// Each count in `names` made a rank's share; the kv head count only when
    /// `kv_split`, and then one head per group of ranks when there are fewer
    /// of them than ranks.
    fn scale(&mut self, pass: &Pass, names: &[&str], kv_split: bool) -> Result<(), String> {
        let world = i128::from(pass.world);
        let op = self.name;
        for (field, value) in tree::fields_mut(&mut self.tree) {
            if !names.contains(field) || *field == KV_HEADS && !kv_split {
                continue;
            }
            // A pair of counts: the query heads cut once per rank, the kv
            // heads as a kv head count is.
            if let Tree::Seq(pair) = value
                && let [Tree::Int(q), Tree::Int(kv)] = &mut pair[..]
            {
                let unsplit = format!("`{op}`.{field} of [{q}, {kv}] does not split {world} ways");
                let heads = u64::try_from(*kv).ok().filter(|h| *h > 0);
                let kv_parts = if kv_split {
                    Shard::parts(heads, u64::from(pass.world))
                        .map(i128::from)
                        .map_err(|_| unsplit.clone())?
                } else {
                    1
                };
                if *q % world != 0 || *kv % kv_parts != 0 {
                    return Err(unsplit);
                }
                *q /= world;
                *kv /= kv_parts;
                continue;
            }
            let Tree::Int(n) = value else {
                return Err(format!("`{op}`.{field} is not a count"));
            };
            let unsplit = format!("`{op}`.{field} of {n} does not split {world} ways");
            let parts = if *field == KV_HEADS {
                let heads = u64::try_from(*n).ok().filter(|h| *h > 0);
                Shard::parts(heads, u64::from(pass.world))
                    .map(i128::from)
                    .map_err(|_| unsplit.clone())?
            } else {
                world
            };
            if *n % parts != 0 || world % parts != 0 {
                return Err(unsplit);
            }
            *n /= parts;
        }
        Ok(())
    }

    /// The widths an unpack names of the blocks of the one packed share it
    /// takes apart, each with a rank's share of it: a block of kv heads a
    /// group of ranks holds alike is cut fewer ways than the ranks.
    fn pieces(&self, held: &[Dist]) -> Vec<(&'static str, i128, i128)> {
        let Some((_, fields)) = UNPACK_WIDTHS.iter().find(|(op, _)| *op == self.name) else {
            return Vec::new();
        };
        let [Dist::Split(blocks)] = held else {
            return Vec::new();
        };
        if blocks.len() < 2 {
            return Vec::new();
        }
        fields
            .iter()
            .zip(blocks)
            .filter_map(|(field, b)| {
                let parts = b.parts.max(1);
                match tree::field(&self.tree, field) {
                    Some(Tree::Int(n)) if *n == i128::from(b.width) => {
                        Some((*field, *n, i128::from(b.width / parts)))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// Each width `pieces` names set to a rank's share of its block.
    fn unpack(&mut self, pieces: &[(&'static str, i128, i128)]) {
        for (field, value) in tree::fields_mut(&mut self.tree) {
            if let Some((_, _, share)) = pieces.iter().find(|(f, _, _)| f == field)
                && let Tree::Int(n) = value
            {
                *n = *share;
            }
        }
    }

    /// The `kv` an op appends to a cache, or attends to without one, is cut
    /// as the cache is, else as its kv head count is.
    fn check_kv_parts(&self, pass: &Pass, kv: &[&str]) -> Result<(), String> {
        let mut want = None;
        for v in &self.ins {
            if let Some(parts) = pass.cache_parts(*v)? {
                want = Some(parts);
            }
        }
        if kv.is_empty() {
            return Ok(());
        }
        let want = match (want, tree::field(&self.tree, KV_HEADS)) {
            (Some(parts), _) => parts,
            (None, Some(Tree::Int(n))) => {
                let n = u64::try_from(*n).map_err(|_| "a negative head count")?;
                Shard::parts(Some(n), u64::from(pass.world))
                    .map_err(|why| format!("`{}`: {why}", self.name))?
            }
            (None, _) => return Ok(()),
        };
        for name in kv {
            for v in self.all(name) {
                if let Dist::Split(blocks) = pass.dist(v)?
                    && blocks.iter().any(|b| b.parts != want)
                {
                    return Err(format!(
                        "`{}` writes or reads kv `{name}` cut {:?} ways where its heads \
                         are cut {want} ways",
                        self.name,
                        blocks.iter().map(|b| b.parts).collect::<Vec<_>>(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// How the ranks hold output `out`, the `index`-th of `count`, of an op
/// whose `followed` inputs they hold as `held`: as the input in its place
/// or the first one as wide, else as the block of a lone packed input it
/// unpacks, else cut once per rank.
fn held_like(
    pass: &Pass,
    out: ValueId,
    index: usize,
    followed: &[ValueId],
    held: &[Dist],
    count: usize,
) -> Result<Dist, String> {
    let Some(width) = pass.width(out) else {
        return Ok(Dist::Whole);
    };
    let same = |i: usize| pass.width(followed[i]) == Some(width);
    if index < followed.len() && same(index) {
        return Ok(held[index].clone());
    }
    if let Some(i) = (0..followed.len()).find(|i| same(*i)) {
        return Ok(held[i].clone());
    }
    if let [Dist::Split(blocks)] = held
        && blocks.len() == count
    {
        // A lone packed input unpacked a block an output, the output cut as
        // its block is however wide the op lays it (padding each head).
        return Ok(Dist::Split(vec![Block {
            width,
            parts: blocks[index].parts,
        }]));
    }
    if held.iter().all(|d| d.is_even(pass.world)) {
        return Ok(Dist::even(width, pass.world));
    }
    Err(format!(
        "an output of `{}` is {width} wide, which no input it follows is, and they \
         are held by groups of ranks alike",
        pass.maker(out),
    ))
}

#[cfg(test)]
mod tests {
    /// Every op name this file keys a rule on names an op: a rule under a
    /// name no op carries never applies, and its op falls to `Whole`.
    #[test]
    fn every_op_a_rule_names_is_an_op() {
        const GROUPS: [&str; 9] = [
            "attn",
            "linear",
            "elemwise",
            "gather",
            "collective",
            "sampling",
            "ssm",
            "fused",
            "conv",
        ];
        let quote = char::from(34);
        let named: Vec<&str> = include_str!("rules.rs")
            .split(quote)
            .skip(1)
            .step_by(2)
            .filter(|lit| {
                lit.split_once('.').is_some_and(|(group, op)| {
                    GROUPS.contains(&group)
                        && !op.is_empty()
                        && op
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                })
            })
            .collect();
        assert!(named.len() > 20, "the rules name {} ops", named.len());
        let stale: Vec<&&str> = named
            .iter()
            .filter(|name| !poem_ir::Operation::names_an_op(name))
            .collect();
        assert!(stale.is_empty(), "named by no op: {stale:?}");
    }
}
