//! How ranks should cut each weight, read off the one-rank trace's dataflow
//! rather than stated by the model: the Megatron pattern, found wherever it
//! holds. A product whose activation every rank holds whole has its weight
//! cut along its output (a column cut), which leaves each rank a share of
//! the output's heads or channels; per-head and elementwise ops carry that
//! share on (`rules.rs` says which); a product that then contracts the share
//! has its weight cut along its input (a row cut), each rank holding a term
//! of the sum, which the shard pass reduces. A weight an op reads beside a
//! share (a bias, a head's sink) is cut as that share is.
//!
//! A row norm with a gain over a share norms it where it is, each rank's
//! sum of squares summed across the ranks, its gain cut as the share is.
//!
//! A column cut is kept only where its share reaches row cuts and nothing
//! that wants it whole: a share that meets an op with no rule, a whole
//! operand or an answer the runtime reads (but the logits, which a
//! vocab-split head gathers) puts its weights back whole, and the trace is
//! read again until nothing changes. Where a share meets the partial sums of
//! another, the product that contracted the other's share is the one put
//! back whole, so a low-rank pair cuts its wide half by its columns. A
//! product whose weight the model states whole reads the share before it
//! gathered instead, where one gather puts the share back in order: the
//! column cut is kept, and the projection after it runs whole on every rank.
//!
//! A packed product's segments are where the ops that read its share take
//! it apart, and a segment's heads are what an op reading it `head_dim`
//! wide, or packing heads it counts, says it holds; a weight an op reads one
//! row a group of (a norm over groups of heads) holds the op's groups.

use std::collections::{BTreeMap, BTreeSet};

use poem_ir::{
    CacheRow, Def, Dim, Operands, Operation, ParamPlacement, Shard, Trace, Ty, ValueId,
};

use super::rules::{self, Rule};
use crate::tree::{self, Tree};

/// The suffixes of a weight's planes beside its leading one: the params a
/// quantized weight stores its factors in, cut as the weight is.
const PLANES: [&str; 4] = [
    poem_ir::SCALES,
    poem_ir::BIASES,
    poem_ir::GLOBAL_SCALES,
    poem_ir::INPUT_SCALES,
];

/// Ops that unpack one packed share into the pieces their outputs are, in
/// order.
const UNPACKS: [&str; 3] = [
    "gather.split_qkv",
    "gather.split_qkv_padded",
    "gather.split_rows",
];

/// Per-head ops affine in what they follow, which a partial sum reaches
/// as it does a sum: reduced first, no pair recut for it.
const AFFINE: [&str; 3] = [
    "elemwise.add_bias",
    "ssm.short_conv",
    "ssm.short_conv_chunked",
];

/// The weights an op reads one row a head or channel of the shares it
/// follows, cut along their rows with them; any other weight a per-head op
/// reads (a head norm's, one row for every head) is whole.
const PER_HEAD: [(&str, &str); 26] = [
    ("linear.matmul_grouped", "w"),
    ("elemwise.add_bias", "bias"),
    ("attn.sink", "sink"),
    ("attn.window_sink", "sink"),
    ("attn.mla_absorb_q", "kv_b"),
    ("attn.mla_absorb_out", "kv_b"),
    ("ssm.gdn_gates", "a_log"),
    ("ssm.gdn_gates", "dt_bias"),
    ("ssm.kda", "a_log"),
    ("ssm.kda", "dt_bias"),
    ("ssm.kda_chunked", "a_log"),
    ("ssm.kda_chunked", "dt_bias"),
    ("ssm.causal_conv1d", "weight"),
    ("ssm.causal_conv1d", "bias"),
    ("ssm.causal_conv1d_chunked", "weight"),
    ("ssm.causal_conv1d_chunked", "bias"),
    ("ssm.short_conv", "weight"),
    ("ssm.short_conv_chunked", "weight"),
    ("ssm.mamba2", "a_log"),
    ("ssm.mamba2", "dt_bias"),
    ("ssm.mamba2", "d"),
    ("ssm.mamba2", "norm"),
    ("ssm.mamba2_chunked", "a_log"),
    ("ssm.mamba2_chunked", "dt_bias"),
    ("ssm.mamba2_chunked", "d"),
    ("ssm.mamba2_chunked", "norm"),
];

/// The weights an op reads one row a group of its shares' heads, and the
/// field that counts its groups: a rank holds whole groups of them.
const GROUPED: [(&str, &str, &str); 2] = [
    ("ssm.mamba2", "norm", "groups"),
    ("ssm.mamba2_chunked", "norm", "groups"),
];

/// Each param's cut, inferred from `trace`'s dataflow whatever the model
/// states.
#[must_use]
pub fn infer(trace: &Trace) -> Vec<Shard> {
    inferred(trace, BTreeSet::new(), 1).0
}

/// `trace` with each weight whose cut the model left to the compiler cut as
/// its dataflow says, a weight it states whole kept whole; and the cut of
/// each such weight over its logical shape, by name, for an import to read
/// it by. A trace that leaves none to the compiler is as it was. A weight
/// whose cut `world` ranks cannot split (a vocabulary of an odd count of
/// rows, a block of scales fewer than the ranks) is whole.
pub fn resolve(trace: &mut Trace, world: u32) -> Vec<(String, Shard)> {
    if !trace.params.iter().any(|p| p.shard == Shard::Inferred) {
        return Vec::new();
    }
    let owner = Reading::of(trace).owner;
    let pinned = (0..trace.params.len())
        .filter(|p| owner[*p].is_none() && trace.params[*p].shard == Shard::Replicated)
        .collect();
    let (shards, lead) = inferred(trace, pinned, world);
    let mut cuts = Vec::new();
    for (p, cut) in shards.into_iter().enumerate() {
        let param = &mut trace.params[p];
        if param.shard != Shard::Inferred {
            continue;
        }
        if owner[p].is_none() {
            cuts.push((param.name.clone(), lead[p].clone()));
        }
        param.shard = cut;
    }
    cuts
}

/// Each param's cut, `pinned` leading params kept whole, and each leading
/// param's cut over its logical shape; a cut `world` ranks cannot split is
/// put back whole.
fn inferred(trace: &Trace, pinned: BTreeSet<usize>, world: u32) -> (Vec<Shard>, Vec<Shard>) {
    let reading = Reading::of(trace);
    let mut whole = pinned.clone();
    loop {
        let mut walk = Walk::new(&reading, &whole, &pinned);
        walk.run();
        let mut rejected = std::mem::take(&mut walk.rejected);
        if rejected.is_subset(&whole) {
            let (shards, lead) = walk.shards();
            rejected.extend(
                (0..shards.len())
                    .filter(|p| !splits(&shards[*p], world))
                    .map(|p| reading.owner[p].unwrap_or(p)),
            );
            if rejected.is_subset(&whole) {
                return (shards, lead);
            }
        }
        whole.extend(rejected);
    }
}

/// Whether `world` ranks split `cut`: each segment into its parts.
fn splits(cut: &Shard, world: u32) -> bool {
    let Shard::Cut {
        segments, heads, ..
    } = cut
    else {
        return true;
    };
    segments.iter().enumerate().all(|(i, segment)| {
        Shard::parts(heads.get(i).copied(), u64::from(world))
            .is_ok_and(|parts| segment.is_multiple_of(parts))
    })
}

/// What a walk reads of the trace that does not change between walks.
struct Reading<'t> {
    trace: &'t Trace,
    /// The param each param is a plane of, if any.
    owner: Vec<Option<usize>>,
    /// Each leading param's logical shape: the shape of the value it is.
    logical: Vec<Option<Vec<u64>>>,
    /// Params a lookup reads that a product also contracts: a tied table.
    tied: BTreeSet<usize>,
    /// Products whose output is the logits.
    logits: BTreeSet<usize>,
}

impl<'t> Reading<'t> {
    fn of(trace: &'t Trace) -> Reading<'t> {
        let by_name: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();
        let owner = trace
            .params
            .iter()
            .map(|p| {
                PLANES.iter().find_map(|suffix| {
                    p.name
                        .strip_suffix(suffix)
                        .and_then(|lead| by_name.get(lead).copied())
                })
            })
            .collect();
        let mut logical = vec![None; trace.params.len()];
        for decl in &trace.values {
            if let (Def::Weight(p), Ty::Tensor { shape, .. }) = (&decl.def, &decl.ty) {
                let dims = shape
                    .iter()
                    .map(|d| match d {
                        Dim::Const(n) => Some(*n),
                        _ => None,
                    })
                    .collect();
                logical[*p as usize] = dims;
            }
        }
        let mut looked = BTreeSet::new();
        let mut contracted = BTreeSet::new();
        let mut logits = BTreeSet::new();
        for node in &trace.nodes {
            let t = tree::of(&node.op);
            match rules::rule(&node.op) {
                Rule::Lookup { table } => looked.extend(param_of(trace, &t, table)),
                Rule::Contract { w, .. } => {
                    let p = param_of(trace, &t, w);
                    contracted.extend(p);
                    if tree::field(&t, "logits") == Some(&Tree::Bool(true)) {
                        logits.extend(p);
                    }
                }
                _ => {}
            }
        }
        Reading {
            trace,
            owner,
            logical,
            tied: looked.intersection(&contracted).copied().collect(),
            logits,
        }
    }
}

fn param_of(trace: &Trace, t: &Tree, field: &str) -> Option<usize> {
    let v = tree::field(t, field).map(tree::ids)?.into_iter().next()?;
    match trace.values[v.0 as usize].def {
        Def::Weight(p) => Some(p as usize),
        _ => None,
    }
}

/// A share's source: the param whose column cut made it, and the run of
/// that param's output axis it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Origin {
    param: usize,
    lo: u64,
    hi: u64,
}

/// How the ranks would hold a value.
#[derive(Clone, Debug, PartialEq)]
enum Held {
    Whole,
    /// Each rank a share of its last axis, from these column cuts.
    Split(BTreeSet<Origin>),
    /// Each rank a term of a sum, contracted from these column cuts' shares.
    Partial(BTreeSet<Origin>),
    /// Downstream of cuts this walk put back: the next walk says.
    Unknown,
}

/// What a walk decides of a param.
#[derive(Clone, Debug, PartialEq)]
enum Cut {
    Whole,
    /// Along its output axis: each rank a share of what it makes.
    Column(u32),
    /// Along its input axis: each rank a term of the sum.
    Row(u32),
    /// Along `axis`, as the shares it is read beside are.
    Beside(u32, BTreeSet<Origin>),
}

struct Walk<'r, 't> {
    reading: &'r Reading<'t>,
    whole: &'r BTreeSet<usize>,
    /// The params the model states whole.
    pinned: &'r BTreeSet<usize>,
    held: Vec<Option<Held>>,
    cuts: Vec<Option<Cut>>,
    /// Where the ops that read a column cut's share take it apart.
    bounds: BTreeMap<usize, BTreeSet<u64>>,
    /// The heads a run of a column cut's output holds.
    heads: BTreeMap<Origin, u64>,
    rejected: BTreeSet<usize>,
    /// Group indices, which follow the row they group.
    groups: BTreeSet<ValueId>,
    /// The groups a weight read one row a group holds.
    grouped: BTreeMap<usize, u64>,
}

impl<'r, 't> Walk<'r, 't> {
    fn new(
        reading: &'r Reading<'t>,
        whole: &'r BTreeSet<usize>,
        pinned: &'r BTreeSet<usize>,
    ) -> Walk<'r, 't> {
        let trace = reading.trace;
        Walk {
            reading,
            whole,
            pinned,
            held: vec![None; trace.values.len()],
            cuts: vec![None; trace.params.len()],
            bounds: BTreeMap::new(),
            heads: BTreeMap::new(),
            rejected: BTreeSet::new(),
            groups: BTreeSet::new(),
            grouped: BTreeMap::new(),
        }
    }

    fn trace(&self) -> &'t Trace {
        self.reading.trace
    }

    fn run(&mut self) {
        let trace = self.trace();
        for node in &trace.nodes {
            self.node(&node.op);
        }
        for seam in &trace.seams {
            if super::output(&seam.seam) {
                for v in &seam.values {
                    self.want_whole(*v);
                }
            }
        }
    }

    /// The leading param a weight value reads, if `v` is one.
    fn param(&self, v: ValueId) -> Option<usize> {
        match self.trace().values[v.0 as usize].def {
            Def::Weight(p) => Some(self.reading.owner[p as usize].unwrap_or(p as usize)),
            _ => None,
        }
    }

    fn is_plane(&self, v: ValueId) -> bool {
        matches!(self.trace().values[v.0 as usize].def,
            Def::Weight(p) if self.reading.owner[p as usize].is_some())
    }

    fn cache(&self, v: ValueId) -> Option<&'t CacheRow> {
        match self.trace().values[v.0 as usize].def {
            Def::Cache(c) => self.trace().caches.get(c as usize),
            _ => None,
        }
    }

    fn held(&mut self, v: ValueId) -> Held {
        match &self.trace().values[v.0 as usize].def {
            Def::Input(_) | Def::Weight(_) | Def::Cache(_) => Held::Whole,
            Def::Op(_) => self.held[v.0 as usize].clone().unwrap_or(Held::Whole),
            Def::Merge(arms) => {
                let helds: Vec<Held> = arms.iter().map(|(arm, _)| self.held(*arm)).collect();
                self.joined(&helds)
            }
        }
    }

    /// Values a linear op sums, or a merge picks between, held as one: the
    /// way they all are, else whole, the shares among them put back.
    fn joined(&mut self, helds: &[Held]) -> Held {
        if helds.contains(&Held::Unknown) {
            Held::Unknown
        } else if helds.iter().all(|h| matches!(h, Held::Split(_))) {
            Held::Split(shares(helds))
        } else if helds.iter().all(|h| matches!(h, Held::Partial(_))) {
            Held::Partial(sums(helds))
        } else if helds.iter().all(|h| *h == Held::Whole) {
            Held::Whole
        } else {
            self.reject(&shares(helds))
        }
    }

    fn width(&self, v: ValueId) -> Option<u64> {
        match &self.trace().values[v.0 as usize].ty {
            Ty::Tensor { shape, .. } => match shape.last() {
                Some(Dim::Const(w)) => Some(*w),
                _ => None,
            },
            Ty::Struct(_) => None,
        }
    }

    /// The column cuts `origins` came from put back whole, and how a value
    /// that read them is held: unknown until the next walk if any were cut,
    /// else whole.
    fn reject(&mut self, origins: &BTreeSet<Origin>) -> Held {
        for o in origins {
            self.rejected.insert(o.param);
        }
        if origins.is_empty() {
            Held::Whole
        } else {
            Held::Unknown
        }
    }

    /// `v` read where an op wants it whole: a share's column cuts are put
    /// back, but the logits, which a vocab-split head gathers.
    fn want_whole(&mut self, v: ValueId) {
        if let Held::Split(origins) = self.held(v)
            && !origins
                .iter()
                .all(|o| self.reading.logits.contains(&o.param))
        {
            self.reject(&origins);
        }
    }

    /// Whether `v`, held as `share`, is one a gather puts back in order:
    /// a share of column cuts' runs each as wide as every segment of them,
    /// so each rank holds one block of it.
    fn gathers(&self, v: ValueId, share: &Held) -> bool {
        let Held::Split(os) = share else {
            return false;
        };
        self.width(v).is_some_and(|w| {
            !os.is_empty()
                && os.iter().all(|o| {
                    let (segments, _) = self.segments(o.param, o.lo, o.hi);
                    segments.iter().all(|s| *s == w)
                })
        })
    }

    /// A value an op reads whole: a share's cuts put back, a weight kept
    /// whole.
    fn read_whole(&mut self, v: ValueId) {
        match self.param(v) {
            Some(p) if !self.is_plane(v) => {
                self.decide(p, Cut::Whole);
            }
            Some(_) => {}
            None => self.want_whole(v),
        }
    }

    /// Param `p` decided `cut`, or put back whole where another read of it
    /// decided otherwise.
    fn decide(&mut self, p: usize, cut: Cut) -> bool {
        if self.whole.contains(&p) && cut != Cut::Whole {
            return false;
        }
        match (&self.cuts[p], &cut) {
            (None, _) => {
                self.cuts[p] = Some(cut);
                true
            }
            (Some(seen), _) if *seen == cut => true,
            (Some(Cut::Beside(axis, seen)), Cut::Beside(a, more)) if a == axis => {
                let joined = seen.union(more).copied().collect();
                self.cuts[p] = Some(Cut::Beside(*axis, joined));
                true
            }
            (Some(_), _) => {
                self.rejected.insert(p);
                false
            }
        }
    }

    fn node(&mut self, op: &Operation) {
        let t = tree::of(op);
        let mut outs = Vec::new();
        op.outputs(&mut outs);
        let mut ins = Vec::new();
        op.inputs(&mut ins);
        let field = |name: &str| -> Vec<ValueId> {
            tree::field(&t, name).map(tree::ids).unwrap_or_default()
        };
        let read = |keep: &[&str]| -> Vec<ValueId> {
            let kept: Vec<ValueId> = keep.iter().flat_map(|n| field(n)).collect();
            ins.iter().copied().filter(|v| !kept.contains(v)).collect()
        };
        // An op reading a value the walk cannot yet say waits for the next.
        if ins.iter().any(|v| self.held(*v) == Held::Unknown) {
            for o in outs {
                self.held[o.0 as usize] = Some(Held::Unknown);
            }
            return;
        }
        let held = match rules::rule(op) {
            Rule::Groups => {
                // Group indices read no value; they follow whatever the
                // grouped product groups.
                self.groups.extend(outs.iter().copied());
                vec![Held::Whole; outs.len()]
            }
            // A row norm with a gain over a share norms it there (q and k
            // normed across their heads), its sum of squares summed across
            // the ranks, its gain cut as the share is. One without a gain
            // (a residual stream's pre-norm) wants its row whole.
            Rule::Norm
                if !field("weight").is_empty()
                    && matches!(
                        field("x").first().map(|x| self.held(*x)),
                        Some(Held::Split(_))
                    ) =>
            {
                let x = field("x")[0];
                let share = self.held(x);
                for w in field("weight") {
                    if let Some(p) = self.param(w)
                        && !self.is_plane(w)
                    {
                        self.beside(p, Some(x), &share, false);
                    }
                }
                vec![share; outs.len()]
            }
            Rule::Whole | Rule::Plan { .. } | Rule::Norm => {
                for v in read(&[]) {
                    self.read_whole(v);
                }
                vec![Held::Whole; outs.len()]
            }
            Rule::Lookup { table } => {
                for v in read(&[table]) {
                    self.read_whole(v);
                }
                // A table the head also contracts is cut by its rows with
                // the head's columns, and one a host holds (read a few rows
                // a token, too large for each rank to hold whole) by its
                // rows alone: each rank looks up the rows it holds, the
                // ranks' rows summed. Any other is whole.
                let tied = field(table).first().and_then(|t| self.param(*t));
                let banded = |p: usize| {
                    self.reading.tied.contains(&p)
                        || self.trace().params[p].placement == ParamPlacement::Host
                };
                match tied {
                    Some(p) if banded(p) && self.decide(p, Cut::Column(0)) => {
                        vec![Held::Partial(BTreeSet::new()); outs.len()]
                    }
                    Some(p) => {
                        self.decide(p, Cut::Whole);
                        vec![Held::Whole; outs.len()]
                    }
                    None => vec![Held::Whole; outs.len()],
                }
            }
            Rule::Linear { follow } => {
                for v in read(follow) {
                    self.read_whole(v);
                }
                let helds: Vec<Held> = follow
                    .iter()
                    .flat_map(|n| field(n))
                    .map(|v| self.held(v))
                    .collect();
                vec![self.joined(&helds); outs.len()]
            }
            Rule::Contract { act, w, out, input } => {
                let a = field(act)[0];
                let p = self
                    .param(field(w)[0])
                    .expect("a product contracts a weight");
                let others = read(&[act, w]);
                for v in &others {
                    if self.param(*v).is_none() {
                        self.read_whole(*v);
                    }
                }
                let share = self.held(a);
                let one = match share.clone() {
                    Held::Split(origins) => {
                        if self.decide(p, Cut::Row(input)) {
                            Held::Partial(origins)
                        } else if self.pinned.contains(&p) && self.gathers(a, &share) {
                            // A weight the model states whole reads its
                            // share gathered.
                            Held::Whole
                        } else {
                            self.reject(&origins)
                        }
                    }
                    _ => {
                        let extent = self.reading.logical[p]
                            .as_ref()
                            .and_then(|s| s.get(out as usize).copied());
                        match extent {
                            Some(hi) if self.decide(p, Cut::Column(out)) => {
                                Held::Split(BTreeSet::from([Origin {
                                    param: p,
                                    lo: 0,
                                    hi,
                                }]))
                            }
                            _ => {
                                self.decide(p, Cut::Whole);
                                Held::Whole
                            }
                        }
                    }
                };
                // A bias the product adds is cut as its output is.
                for v in others {
                    if let Some(q) = self.param(v)
                        && !self.is_plane(v)
                    {
                        self.beside(q, outs.first().copied(), &one, true);
                    }
                }
                vec![one; outs.len()]
            }
            Rule::Heads(heads) => {
                let others = read(heads.follow);
                self.heads(op, &t, heads, &field, &others, &outs)
            }
        };
        for (o, h) in outs.into_iter().zip(held) {
            self.held[o.0 as usize] = Some(h);
        }
    }

    /// Weight `p`, read beside `like` (a value held as `share`): cut along
    /// its axis as wide as that share, its leading one if `rows` lets a
    /// narrower one hold a row a head, else whole.
    fn beside(&mut self, p: usize, like: Option<ValueId>, share: &Held, rows: bool) {
        let Held::Split(origins) = share else {
            self.decide(p, Cut::Whole);
            return;
        };
        let width = like.and_then(|l| self.width(l));
        let shape = self.reading.logical[p].clone().unwrap_or_default();
        let axis = width.and_then(|w| {
            let last = shape.len().checked_sub(1)?;
            if shape[last] == w {
                Some(last)
            } else if shape.first() == Some(&w) || rows && !shape.is_empty() {
                Some(0)
            } else {
                None
            }
        });
        match axis {
            Some(axis) => {
                let cut = Cut::Beside(axis as u32, origins.clone());
                if !self.decide(p, cut) {
                    self.reject(origins);
                }
            }
            None => {
                self.decide(p, Cut::Whole);
            }
        }
    }

    fn heads(
        &mut self,
        op: &Operation,
        t: &Tree,
        heads: rules::Heads,
        field: &dyn Fn(&str) -> Vec<ValueId>,
        others: &[ValueId],
        outs: &[ValueId],
    ) -> Vec<Held> {
        let followed: Vec<(&str, ValueId)> = heads
            .follow
            .iter()
            .flat_map(|n| field(n).into_iter().map(move |v| (*n, v)))
            .filter(|(_, v)| !self.groups.contains(v))
            .collect();
        for v in others {
            if self.param(*v).is_none() && self.cache(*v).is_none() {
                self.want_whole(*v);
            }
        }
        let helds: Vec<Held> = followed.iter().map(|(_, v)| self.held(*v)).collect();
        let split = helds.iter().any(|h| matches!(h, Held::Split(_)));
        let cut = others.iter().filter_map(|v| self.cache(*v)).any(cache_cut);
        let reads_cache = others.iter().any(|v| self.cache(*v).is_some());
        // Partial sums a per-head op reads put back the product whose share
        // was summed, so the next walk cuts the product after it by its
        // columns: a pair is cut where its partial sums reach a sum, not a
        // nonlinearity. A share beside a whole value, or a cache held
        // otherwise, is put back itself.
        let summed = sums(&helds);
        let unlike = helds.iter().any(|h| !matches!(h, Held::Split(_)));
        let refused = split && unlike || reads_cache && !heads.cache.allows(cut, split);
        let held = if !summed.is_empty() && !AFFINE.contains(&op.name()) {
            self.reject(&summed)
        } else if refused {
            self.reject(&shares(&helds))
        } else {
            Held::Whole
        };
        if !split || refused {
            if held == Held::Whole {
                for v in others {
                    if self.param(*v).is_some() {
                        self.read_whole(*v);
                    }
                }
            }
            return vec![held; outs.len()];
        }
        let all = shares(&helds);
        self.count_heads(op, t, &followed, &helds, others);
        // Weights the op reads beside its shares are cut as they are.
        let like = followed.first().map(|(_, v)| *v);
        for v in others {
            if let Some(p) = self.param(*v)
                && !self.is_plane(*v)
            {
                let name = tree::fields(t)
                    .iter()
                    .find(|(_, sub)| tree::ids(sub).contains(v))
                    .map_or("", |(n, _)| *n);
                if PER_HEAD.contains(&(op.name(), name)) {
                    self.beside(p, like, &Held::Split(all.clone()), true);
                    let count = GROUPED
                        .iter()
                        .find(|(o, w, _)| *o == op.name() && *w == name)
                        .and_then(|(_, _, field)| match tree::field(t, field) {
                            Some(Tree::Int(n)) => u64::try_from(*n).ok().filter(|n| *n > 0),
                            _ => None,
                        });
                    if let Some(groups) = count {
                        self.grouped.insert(p, groups);
                    }
                } else {
                    self.decide(p, Cut::Whole);
                }
            }
        }
        // An unpack names the segments of the run it takes apart.
        if UNPACKS.contains(&op.name())
            && let [(_, v)] = followed[..]
            && let Some(o) = self.whole_run(v)
        {
            let widths = padded_qkv(t)
                .map(|pieces| pieces.iter().map(|(w, _)| *w).collect())
                .or_else(|| {
                    outs.iter()
                        .map(|o| self.width(*o))
                        .collect::<Option<Vec<u64>>>()
                });
            if let Some(widths) = widths
                && widths.iter().sum::<u64>() == o.hi - o.lo
            {
                let mut lo = o.lo;
                return widths
                    .iter()
                    .map(|w| {
                        let run = Origin {
                            param: o.param,
                            lo,
                            hi: lo + w,
                        };
                        self.bounds.entry(o.param).or_default().insert(lo);
                        lo += w;
                        Held::Split(BTreeSet::from([run]))
                    })
                    .collect();
            }
        }
        // Each output as the input in its place, or the first as wide, else
        // as all of them.
        (0..outs.len())
            .map(|i| {
                let width = self.width(outs[i]);
                let same = |j: usize| self.width(followed[j].1) == width;
                if i < followed.len() && same(i) {
                    helds[i].clone()
                } else if let Some(j) = (0..followed.len()).find(|j| same(*j)) {
                    helds[j].clone()
                } else {
                    Held::Split(all.clone())
                }
            })
            .collect()
    }

    /// The one run `v` is all of, if it is a share of one column cut's
    /// output as wide as itself.
    fn whole_run(&self, v: ValueId) -> Option<Origin> {
        let Some(Held::Split(os)) = &self.held[v.0 as usize] else {
            return None;
        };
        let [o] = os.iter().copied().collect::<Vec<_>>()[..] else {
            return None;
        };
        (Some(o.hi - o.lo) == self.width(v)).then_some(o)
    }

    /// The heads and segments the op says its shares hold: a share read
    /// `head_dim` wide holds that many heads, and a share the op takes
    /// apart packs the segments it names.
    fn count_heads(
        &mut self,
        op: &Operation,
        t: &Tree,
        followed: &[(&str, ValueId)],
        helds: &[Held],
        others: &[ValueId],
    ) {
        let int = |name: &str| match tree::field(t, name) {
            Some(Tree::Int(n)) => u64::try_from(*n).ok().filter(|n| *n > 0),
            _ => None,
        };
        let head_dim = int("head_dim").or_else(|| {
            others
                .iter()
                .filter_map(|v| self.cache(*v))
                .find_map(|c| match c {
                    CacheRow::Kv { head_dim, .. } => Some(u64::from(*head_dim)).filter(|d| *d > 0),
                    CacheRow::State { .. } => None,
                })
        });
        for ((name, v), h) in followed.iter().zip(helds) {
            if !matches!(h, Held::Split(_)) {
                continue;
            }
            // A share of several runs as wide as itself (the arms of a merge)
            // holds what each of them does.
            for o in self.whole_runs(*v, h) {
                self.count_run(op, t, name, o, head_dim);
            }
        }
    }

    /// The runs `v`, held as `held` (a merge as its arms are), is all of:
    /// each column cut's output it is a share of, if every one is as wide as
    /// `v`.
    fn whole_runs(&self, v: ValueId, held: &Held) -> Vec<Origin> {
        let Held::Split(os) = held else {
            return Vec::new();
        };
        let width = self.width(v);
        if os.iter().all(|o| Some(o.hi - o.lo) == width) {
            os.iter().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// The heads and segments op `op` says run `o` of its `name` input holds.
    fn count_run(
        &mut self,
        op: &Operation,
        t: &Tree,
        name: &str,
        o: Origin,
        head_dim: Option<u64>,
    ) {
        let int = |name: &str| match tree::field(t, name) {
            Some(Tree::Int(n)) => u64::try_from(*n).ok().filter(|n| *n > 0),
            _ => None,
        };
        let width = o.hi - o.lo;
        let pieces: Vec<(u64, Option<u64>)> = match (op.name(), name) {
            ("ssm.gated_delta" | "ssm.gated_delta_chunked", "qkv") => {
                match (int("k_heads"), int("v_heads"), int("k_dim"), int("v_dim")) {
                    (Some(kh), Some(vh), Some(kd), Some(vd)) => {
                        vec![(kh * kd, None), (kh * kd, None), (vh * vd, None)]
                    }
                    _ => Vec::new(),
                }
            }
            ("ssm.kda" | "ssm.kda_chunked", "mixed") => vec![(width / 3, None); 3],
            ("ssm.gdn_gates", "ba") => vec![(width / 2, None); 2],
            ("ssm.mamba2" | "ssm.mamba2_chunked", "xbc") => {
                match (int("heads"), int("groups"), int("state_dim")) {
                    (Some(heads), Some(groups), Some(state)) => {
                        let bc = groups * state;
                        vec![
                            (width.saturating_sub(2 * bc), Some(heads)),
                            (bc, Some(groups)),
                            (bc, Some(groups)),
                        ]
                    }
                    _ => Vec::new(),
                }
            }
            ("ssm.mamba2" | "ssm.mamba2_chunked", "z" | "dt") => {
                vec![(width, int("heads"))]
            }
            ("linear.glu", "packed") => vec![(width / 2, None); 2],
            ("gather.split_qkv_padded", "packed") => padded_qkv(t)
                .map_or_else(Vec::new, |pieces| {
                    pieces.into_iter().map(|(w, h)| (w, Some(h))).collect()
                }),
            _ => match head_dim {
                Some(d) if width.is_multiple_of(d) => vec![(width, Some(width / d))],
                _ => Vec::new(),
            },
        };
        if pieces.iter().map(|(w, _)| w).sum::<u64>() != width {
            return;
        }
        let mut lo = o.lo;
        for (w, heads) in pieces {
            self.bounds.entry(o.param).or_default().insert(lo);
            if let Some(h) = heads {
                self.heads
                    .entry(Origin {
                        param: o.param,
                        lo,
                        hi: lo + w,
                    })
                    .or_insert(h);
            }
            lo += w;
        }
    }

    /// Each param's cut, as the walk left them.
    /// Each param's cut and each leading param's logical one: a weight one
    /// of whose planes cannot be stored cut so is whole.
    fn shards(&self) -> (Vec<Shard>, Vec<Shard>) {
        let trace = self.trace();
        let mut lead: Vec<Shard> = (0..trace.params.len())
            .map(|p| self.logical_shard(p).unwrap_or_default())
            .collect();
        let restated = |lead: &[Shard], p: usize| -> Result<Shard, String> {
            let at = self.reading.owner[p].unwrap_or(p);
            match &self.reading.logical[at] {
                Some(logical) => poem::restate(&lead[at], logical, &trace.params[p].shape),
                None => Ok(Shard::Replicated),
            }
        };
        for p in 0..trace.params.len() {
            if restated(&lead, p).is_err() {
                lead[self.reading.owner[p].unwrap_or(p)] = Shard::Replicated;
            }
        }
        let shards = (0..trace.params.len())
            .map(|p| restated(&lead, p).unwrap_or_default())
            .collect();
        (shards, lead)
    }

    /// The segments of run `lo..hi` of column cut `q`, and the heads each
    /// holds if every one's are known.
    fn segments(&self, q: usize, lo: u64, hi: u64) -> (Vec<u64>, Vec<u64>) {
        let mut cuts: Vec<u64> = self
            .bounds
            .get(&q)
            .map(|b| b.range(lo + 1..hi).copied().collect())
            .unwrap_or_default();
        cuts.insert(0, lo);
        cuts.push(hi);
        let segments: Vec<u64> = cuts.windows(2).map(|w| w[1] - w[0]).collect();
        let heads: Option<Vec<u64>> = cuts
            .windows(2)
            .map(|w| {
                // The narrowest run counted that holds this segment, its
                // heads in proportion.
                self.heads
                    .iter()
                    .filter(|(o, _)| o.param == q && o.lo <= w[0] && w[1] <= o.hi)
                    .min_by_key(|(o, _)| o.hi - o.lo)
                    .and_then(|(o, h)| {
                        let part = h * (w[1] - w[0]);
                        part.is_multiple_of(o.hi - o.lo)
                            .then(|| part / (o.hi - o.lo))
                    })
            })
            .collect();
        (segments, heads.unwrap_or_default())
    }

    /// Param `p`'s cut over its logical shape.
    fn logical_shard(&self, p: usize) -> Option<Shard> {
        let shape = self.reading.logical[p].as_ref()?;
        match self.cuts[p].as_ref()? {
            Cut::Whole => None,
            Cut::Column(axis) => {
                let extent = *shape.get(*axis as usize)?;
                let (segments, heads) = self.segments(p, 0, extent);
                Some(Shard::Cut {
                    axis: *axis,
                    segments,
                    heads,
                })
            }
            Cut::Row(axis) => Some(Shard::Cut {
                axis: *axis,
                segments: vec![*shape.get(*axis as usize)?],
                heads: Vec::new(),
            }),
            Cut::Beside(axis, origins) => {
                let extent = *shape.get(*axis as usize)?;
                let (segments, mut heads) = match origins.iter().collect::<Vec<_>>()[..] {
                    [o] if o.hi - o.lo == extent => self.segments(o.param, o.lo, o.hi),
                    _ => (vec![extent], Vec::new()),
                };
                if let (Some(groups), [_]) = (self.grouped.get(&p), &segments[..]) {
                    heads = vec![*groups];
                }
                Some(Shard::Cut {
                    axis: *axis,
                    segments,
                    heads,
                })
            }
        }
    }
}

/// The runs a padded q, k and v split takes its packed input apart into, as
/// stored (each head its own width, before the op pads it), and the heads
/// of each: its `heads` are q's and kv's, its `widths` a q or k head's and a
/// v head's.
fn padded_qkv(t: &Tree) -> Option<Vec<(u64, u64)>> {
    let pair = |name: &str| match tree::field(t, name) {
        Some(Tree::Seq(items)) => match items[..] {
            [Tree::Int(a), Tree::Int(b)] => Some((u64::try_from(a).ok()?, u64::try_from(b).ok()?)),
            _ => None,
        },
        _ => None,
    };
    let (q_heads, kv_heads) = pair("heads")?;
    let (qk, v) = pair("widths")?;
    Some(vec![
        (q_heads * qk, q_heads),
        (kv_heads * qk, kv_heads),
        (kv_heads * v, kv_heads),
    ])
}

/// The column cuts whose shares `helds` are.
fn shares(helds: &[Held]) -> BTreeSet<Origin> {
    helds
        .iter()
        .flat_map(|h| match h {
            Held::Split(os) => os.iter().copied().collect(),
            _ => Vec::new(),
        })
        .collect()
}

/// The column cuts whose shares `helds` sum.
fn sums(helds: &[Held]) -> BTreeSet<Origin> {
    helds
        .iter()
        .flat_map(|h| match h {
            Held::Partial(os) => os.iter().copied().collect(),
            _ => Vec::new(),
        })
        .collect()
}

fn cache_cut(row: &CacheRow) -> bool {
    matches!(
        row,
        CacheRow::Kv {
            shard: Shard::Cut { .. },
            ..
        } | CacheRow::State {
            shard: Shard::Cut { .. },
            ..
        }
    )
}

#[cfg(test)]
mod tests;
