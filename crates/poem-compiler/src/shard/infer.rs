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
//! A row norm over a share no product reads norms it where it is, each
//! rank's sum of squares summed across the ranks, its gain cut as the
//! share is; a pre-norm, whose row a product reads, wants it whole.
//!
//! A column cut is kept only where its share reaches row cuts and nothing
//! that wants it whole: a share that meets an op with no rule, a whole
//! operand or an answer the runtime reads (but the logits, which a
//! vocab-split head gathers) puts its weights back whole, and the trace is
//! read again until nothing changes. Where a share meets the partial sums of
//! another, the product that contracted the other's share is the one put
//! back whole, so a low-rank pair cuts its wide half by its columns.
//!
//! A chain of cuts (the column cuts whose shares meet, and the row cuts and
//! weights cut with them) is kept only where it saves what its collectives
//! cost: one far lighter than the rows it reduces, which no
//! attention or scan reads (a low-rank pair on a wide row), is put back
//! whole, and the trace read again (`NARROW` says how light).
//!
//! A packed product's segments are where the ops that read its share take
//! it apart, and a segment's heads are what an op reading it `head_dim`
//! wide, or packing heads it counts, says it holds; a weight an op reads one
//! row a group of (a norm over groups of heads) holds the op's groups.

use std::collections::{BTreeMap, BTreeSet};

use poem_ir::{CacheRow, Def, Dim, Operands, Operation, ParamPlacement, Shard, Trace, Ty, ValueId};

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

/// The op families whose work is not a product's: an attention or a scan,
/// whose work grows with the context a sequence holds and whose cache or
/// state a cut divides with its heads. A chain whose share one reads is cut
/// whatever its weights weigh.
const CONTEXT: [&str; 2] = ["attn.", "ssm."];

/// How much lighter than the products its collectives pay for a chain may
/// be before it stays whole. A collective moves a row `w` wide, as a
/// Megatron pair's does: that pair (attention's q and o, an MLP's up and
/// down) holds at least `w * w` a rank's cut divides by `tp`, and its saving
/// is what pays for the collective, its latency and its `w` elements over
/// the link. A chain holding under an eighth of `w * w` for each row it
/// moves (a low-rank pair a few hundred wide on a row of thousands, a
/// sixteenth) saves too little for the same collective, at any count of
/// ranks, and stays whole; a layer's attention and MLP hold several times
/// `w * w`. The cost is read in the model's own widths, not a machine's, so
/// a row and its miniature cut alike, whatever their format or batch.
const NARROW: u64 = 8;

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
    inferred(trace, 1).0
}

/// `trace` with each weight whose cut the model left to the compiler cut as
/// its dataflow says; and the cut of each such weight over its logical
/// shape, by name, for an import to read it by. A trace that leaves none to the compiler is as it was. A weight
/// whose cut `world` ranks cannot split (a vocabulary of an odd count of
/// rows, a block of scales fewer than the ranks) is whole.
pub fn resolve(trace: &mut Trace, world: u32) -> Vec<(String, Shard)> {
    if !trace.params.iter().any(|p| p.shard == Shard::Inferred) {
        return Vec::new();
    }
    let owner = Reading::of(trace).owner;
    let (shards, lead) = inferred(trace, world);
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

/// Each param's cut and each leading param's cut over its logical shape; a
/// cut `world` ranks cannot split is put back whole.
fn inferred(trace: &Trace, world: u32) -> (Vec<Shard>, Vec<Shard>) {
    let reading = Reading::of(trace);
    let mut whole = BTreeSet::new();
    loop {
        let mut walk = Walk::new(&reading, &whole);
        walk.run();
        let mut rejected = std::mem::take(&mut walk.rejected);
        if rejected.is_subset(&whole) {
            let (shards, lead) = walk.shards();
            rejected.extend(
                (0..shards.len())
                    .filter(|p| !splits(&shards[*p], world))
                    .map(|p| reading.owner[p].unwrap_or(p)),
            );
            rejected.extend(walk.cheap());
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
    /// Values a product contracts as its activation.
    contracted: BTreeSet<ValueId>,
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
        let mut contracted_params = BTreeSet::new();
        let mut logits = BTreeSet::new();
        let mut contracted = BTreeSet::new();
        for node in &trace.nodes {
            let t = tree::of(&node.op);
            match rules::rule(&node.op) {
                Rule::Lookup { table } => looked.extend(param_of(trace, &t, table)),
                Rule::Contract { act, w, .. } => {
                    contracted.extend(tree::field(&t, act).map(tree::ids).unwrap_or_default());
                    let p = param_of(trace, &t, w);
                    contracted_params.extend(p);
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
            tied: looked.intersection(&contracted_params).copied().collect(),
            logits,
            contracted,
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
    /// The column cuts each other cut weight is cut with: the shares a row
    /// cut contracts, or a weight is read beside.
    with: BTreeMap<usize, BTreeSet<usize>>,
    /// Each cut weight's elements, once for every op that reads it cut.
    work: Vec<(usize, u64)>,
    /// The values a collective moves, and the column cuts whose partial
    /// sums they are: what an op reduces.
    moved: BTreeMap<ValueId, BTreeSet<usize>>,
    /// Column cuts whose shares an op reads together.
    met: Vec<BTreeSet<usize>>,
    /// Column cuts whose shares an attention or a scan reads.
    context: BTreeSet<usize>,
}

impl<'r, 't> Walk<'r, 't> {
    fn new(reading: &'r Reading<'t>, whole: &'r BTreeSet<usize>) -> Walk<'r, 't> {
        let trace = reading.trace;
        Walk {
            reading,
            whole,
            held: vec![None; trace.values.len()],
            cuts: vec![None; trace.params.len()],
            bounds: BTreeMap::new(),
            heads: BTreeMap::new(),
            rejected: BTreeSet::new(),
            groups: BTreeSet::new(),
            grouped: BTreeMap::new(),
            with: BTreeMap::new(),
            work: Vec::new(),
            moved: BTreeMap::new(),
            met: Vec::new(),
            context: BTreeSet::new(),
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
                    self.reduce(*v);
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
        let rule = rules::rule_of(op, |v| self.width(v));
        // Partial sums any op but a sum reads are reduced first.
        if !matches!(rule, Rule::Linear { .. }) {
            for v in &ins {
                self.reduce(*v);
            }
        }
        let held = match rule {
            Rule::Groups => {
                // Group indices read no value; they follow whatever the
                // grouped product groups.
                self.groups.extend(outs.iter().copied());
                vec![Held::Whole; outs.len()]
            }
            // A row norm over a share norms it there (q and k normed across
            // their heads), its sum of squares summed across the ranks, its
            // gain cut as the share is. One whose row a product reads (a
            // pre-norm) wants its row whole, as the product's activation.
            Rule::Norm
                if !outs.iter().any(|o| self.reading.contracted.contains(o))
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
                let followed: Vec<ValueId> = follow.iter().flat_map(|n| field(n)).collect();
                let helds: Vec<Held> = followed.iter().map(|v| self.held(*v)).collect();
                let held = self.joined(&helds);
                // Partial sums added to what is not are reduced first.
                if !matches!(held, Held::Partial(_)) {
                    for v in followed {
                        self.reduce(v);
                    }
                }
                vec![held; outs.len()]
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
                            self.cut_with(p, &origins);
                            Held::Partial(origins)
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
                                self.cut_with(p, &BTreeSet::new());
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

    /// Weight `p` cut with the column cuts `origins` came from: weighed in
    /// their chain.
    fn cut_with(&mut self, p: usize, origins: &BTreeSet<Origin>) {
        let elements = self.reading.logical[p]
            .as_ref()
            .map_or(0, |shape| shape.iter().product());
        self.work.push((p, elements));
        let from = origins.iter().map(|o| o.param);
        self.with.entry(p).or_default().extend(from);
    }

    /// `v` reduced, if the ranks hold partial sums of it.
    fn reduce(&mut self, v: ValueId) {
        if let Held::Partial(origins) = self.held(v)
            && !origins.is_empty()
        {
            self.moved
                .insert(v, origins.iter().map(|o| o.param).collect());
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
                if self.decide(p, cut) {
                    self.cut_with(p, origins);
                } else {
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
        self.met.push(all.iter().map(|o| o.param).collect());
        if CONTEXT.iter().any(|family| op.name().starts_with(family)) {
            self.context.extend(all.iter().map(|o| o.param));
        }
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

    /// The column cuts of each chain that saves less than its collectives
    /// cost, to be put back whole: a chain is the column cuts whose shares
    /// meet, and the row cuts and weights cut with them; it stays whole when
    /// it holds less than a [`NARROW`]th of the square of each row its
    /// collectives move, and no attention or scan reads its share. A
    /// product whose input arrives cut is weighed in the chain that cut it,
    /// so a light one follows a heavy chain's cut.
    fn cheap(&self) -> BTreeSet<usize> {
        let mut chain: BTreeMap<usize, usize> = BTreeMap::new();
        fn root(chain: &mut BTreeMap<usize, usize>, p: usize) -> usize {
            let up = *chain.entry(p).or_insert(p);
            if up == p {
                return p;
            }
            let top = root(chain, up);
            chain.insert(p, top);
            top
        }
        let mut join = |a: usize, b: usize| {
            let (a, b) = (root(&mut chain, a), root(&mut chain, b));
            chain.insert(a.max(b), a.min(b));
        };
        for (p, from) in &self.with {
            for o in from {
                join(*p, *o);
            }
        }
        for from in self.moved.values().chain(&self.met) {
            for (a, b) in from.iter().zip(from.iter().skip(1)) {
                join(*a, *b);
            }
        }
        let mut work: BTreeMap<usize, u64> = BTreeMap::new();
        for (p, elements) in &self.work {
            *work.entry(root(&mut chain, *p)).or_default() += elements;
        }
        let mut moved: BTreeMap<usize, u64> = BTreeMap::new();
        for (v, from) in &self.moved {
            if let Some(o) = from.first() {
                let w = self.width(*v).unwrap_or(0);
                *moved.entry(root(&mut chain, *o)).or_default() += w * w;
            }
        }
        let read: BTreeSet<usize> = self.context.iter().map(|p| root(&mut chain, *p)).collect();
        let origins: BTreeSet<usize> = self
            .with
            .values()
            .chain(self.moved.values())
            .flatten()
            .copied()
            .collect();
        origins
            .into_iter()
            .filter(|o| {
                let at = root(&mut chain, *o);
                let moved = moved.get(&at).copied().unwrap_or(0);
                !read.contains(&at) && work.get(&at).copied().unwrap_or(0) * NARROW < moved
            })
            .collect()
    }

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
                // Read beside a share as wide as it, the weight holds that
                // share's segments: one run's, or the runs' a concat lays
                // side by side (in the order their weights are stated).
                let whole: u64 = origins.iter().map(|o| o.hi - o.lo).sum();
                let (segments, mut heads) = if whole == extent {
                    let runs: Vec<(Vec<u64>, Vec<u64>)> = origins
                        .iter()
                        .map(|o| self.segments(o.param, o.lo, o.hi))
                        .collect();
                    let headed = runs.iter().all(|(s, h)| h.len() == s.len());
                    (
                        runs.iter().flat_map(|(s, _)| s.iter().copied()).collect(),
                        if headed {
                            runs.iter().flat_map(|(_, h)| h.iter().copied()).collect()
                        } else {
                            Vec::new()
                        },
                    )
                } else {
                    (vec![extent], Vec::new())
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
