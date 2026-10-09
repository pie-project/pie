//! What each op takes and yields when ranks split its values. An op without a
//! rule here gets every value whole and stays as it is; it may not read a
//! weight or a cache the ranks split.

use poem_ir::{Operands, Operation, Shard, ValueId};

use super::{Dist, Pass};
use crate::tree::{self, Tree};

/// How an op meets the ranks.
enum Rule {
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
    /// with it: then each rank norms its own share.
    Norm,
    /// Group indices for a grouped matmul, which reads no value: they are
    /// a rank's share of the groups when the row it groups is split.
    Groups,
}

/// An op whose `follow` inputs are split by head or channel, every output
/// with them, and whose `scale` counts are each rank's share. Other
/// activations are whole.
#[derive(Clone, Copy)]
struct Heads {
    follow: &'static [&'static str],
    scale: &'static [&'static str],
    cache: Cache,
    /// The kv it appends to a cache or attends to without one, cut as its
    /// kv heads are.
    kv: &'static [&'static str],
}

/// How the cache an op reads is held against the values it follows.
#[derive(Clone, Copy, PartialEq)]
enum Cache {
    /// It reads none the ranks split.
    Whole,
    /// Split exactly when its values are.
    Split,
    /// Split when its values are, or kept whole on every rank while they
    /// split the query heads that share its kv heads.
    SplitOrWhole,
}

impl Cache {
    fn allows(self, cut: bool, split: bool) -> bool {
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

/// The head count of kv a group of ranks may share.
const KV_HEADS: &str = "kv_heads";

/// How op `name` splits by head or channel, if it does.
fn heads(name: &str) -> Option<Heads> {
    Some(match name {
        "elementwise.silu"
        | "elementwise.gelu"
        | "elementwise.tanh"
        | "elementwise.mul_scalar"
        | "elementwise.silu_scaled"
        | "elementwise.scale"
        | "elementwise.rmsnorm_per_head"
        | "elementwise.rmsnorm_per_head_plus_one"
        | "elementwise.rmsnorm_no_scale"
        | "elementwise.rope_axes"
        | "layout.gather_rows"
        | "layout.pack_rows"
        | "layout.unpack_rows"
        | "linear.mlp_gelu_tanh" => Heads::of(&["x"], &[]),
        "elementwise.mul" => Heads::of(&["x", "y"], &[]),
        "elementwise.gate_sigmoid_mul"
        | "elementwise.gate_sigmoid_mul_heads"
        | "elementwise.rmsnorm_gated" => Heads::of(&["x", "gate"], &[]),
        "elementwise.rmsnorm_gated_by" => Heads::of(&["x", "gate"], &["heads"]),
        "elementwise.rope_full"
        | "elementwise.rope_partial"
        | "elementwise.rope_yarn"
        | "elementwise.rope_mrope" => Heads::of(&["q", "k"], &[]),
        "elementwise.rope_partial_q" | "elementwise.rope_partial_last" => Heads::of(&["q"], &[]),
        "elementwise.add_bias" => Heads::of(&["out"], &[]),
        "layout.split_qkv" => Heads::of(&["packed"], &["q_width", "kv_width"]),
        "layout.split_q_gate" => Heads::of(&["packed"], &[]),
        "layout.split_rows" => Heads::of(&["x"], &["width"]),
        "linear.mlp_swiglu"
        | "linear.mlp_swiglu_clamp"
        | "linear.mlp_swiglu_clamp_alpha"
        | "linear.mlp_geglu_tanh_packed"
        | "linear.mlp_situ" => Heads::of(&["packed"], &["intermediate"]),
        "linear.mlp_swiglu_clamp_split" | "linear.mlp_geglu_tanh" => {
            Heads::of(&["gate", "up"], &[])
        }
        "linear.matmul_grouped" => Heads::of(&["x", "routes"], &["groups"]),
        "attention.decode" | "attention.decode_lse" | "attention.decode_selected" => {
            Heads::of(&["q"], &[]).cache(Cache::SplitOrWhole)
        }
        "attention.prefill"
        | "attention.prefill_lse"
        | "attention.masked"
        | "attention.masked_lse"
        | "attention.prefill_selected" => Heads::of(&["q"], &[KV_HEADS]).cache(Cache::SplitOrWhole),
        "attention.ragged" => Heads::of(&["q", "k", "v"], &[KV_HEADS]).kv(&["k", "v"]),
        "attention.sink" => Heads::of(&["o", "lse"], &[]),
        "attention.merge_lse" => Heads::of(&["o1", "lse1", "o2", "lse2"], &["heads"]),
        "attention.pool_lse" | "attention.pool_lse_selected" => Heads::of(&["q"], &["heads"]),
        "attention.kv_append" => Heads::of(&["k", "v"], &[])
            .cache(Cache::Split)
            .kv(&["k", "v"]),
        "attention.kv_append_shared" => Heads::of(&["plane"], &[])
            .cache(Cache::Split)
            .kv(&["plane"]),
        "attention.mla_split_q_b" => Heads::of(&["q_b"], &["heads"]),
        "attention.mla_absorb_q" => Heads::of(&["q_nope"], &["heads"]),
        "attention.mla_absorb_out" => Heads::of(&["latent"], &["heads"]),
        "attention.mla_decode"
        | "attention.mla_prefill"
        | "attention.mla_decode_selected"
        | "attention.mla_prefill_selected" => Heads::of(&["q", "q_pe"], &["heads"]),
        "attention.ssm_causal_conv1d" | "attention.ssm_causal_conv1d_chunked" => {
            Heads::of(&["x"], &[]).cache(Cache::Split)
        }
        "attention.ssm_gdn_prep" => Heads::of(&["ba"], &[]),
        "attention.ssm_gated_delta" | "attention.ssm_gated_delta_chunked" => {
            Heads::of(&["qkv", "z", "gates"], &["k_heads", "v_heads"]).cache(Cache::Split)
        }
        "attention.ssm_kda_step" | "attention.ssm_kda_chunked" => {
            Heads::of(&["mixed", "f", "b"], &["heads"]).cache(Cache::Split)
        }
        _ => return None,
    })
}

fn rule(name: &str) -> Rule {
    match name {
        "linear.matmul" | "linear.lm_head" => Rule::Contract {
            act: "act",
            w: "w",
            out: 0,
            input: 1,
        },
        "linear.moe_matmul_select"
        | "linear.moe_matmul_select_bias"
        | "linear.moe_matmul_select_quant" => Rule::Contract {
            act: "x",
            w: "bank",
            out: 1,
            input: 2,
        },
        "linear.moe_weighted_sum" => Rule::Linear {
            follow: &["routed"],
        },
        "elementwise.residual_add" | "elementwise.add" => Rule::Linear {
            follow: &["x", "y"],
        },
        "linear.moe_sigmoid_gate_add" => Rule::Linear {
            follow: &["routed", "shared"],
        },
        "layout.embed" => Rule::Lookup { table: "table" },
        "attention.plan_prefill" | "attention.plan_decode" => Rule::Plan {
            scale: &["q_heads", KV_HEADS],
        },
        "attention.mla_plan" => Rule::Plan { scale: &["heads"] },
        "elementwise.rmsnorm" | "elementwise.rmsnorm_plus_one" => Rule::Norm,
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
    let dists = match rule(at.name) {
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
            match (pass.dist(x)?, cut) {
                (Dist::Split(blocks), Some((0, _))) => vec![Dist::Split(blocks); outs.len()],
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
                at.scale(pass, heads.scale, !whole_kv)?;
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
        && blocks[index].width == width
    {
        return Ok(Dist::Split(vec![blocks[index]]));
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
