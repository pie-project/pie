//! The step compositions a load's step builder can form, stated once at
//! load (CONTRACT-BC 24): the runtime's rules over the engine's facts, so an
//! engine's arm walk records exactly the set the runtime can send it and
//! nothing it never will.
//!
//! - The runtime states its step builder's rules ([`StepRules`], carried by
//!   [`crate::LoadRequest::steps`]).
//! - Over the facts the load states of itself ([`StepFacts`]: its rung
//!   lattice, its fire limits, whether it buffers recurrent rows), one
//!   function, [`StepShapes::of`], computes the finite set, ordered smallest
//!   first across kinds. The engine calls it once its lattice is fixed and
//!   before its arm walk; the runtime calls it again once the load returns,
//!   from the same facts as the load states them, and refuses a load whose
//!   stated set is another by name ([`StatedSteps::agrees`]).
//! - The engine walks the set in its order, recording each shape's bodies
//!   while they fit its measured allowance; a shape whose bodies do not fit
//!   runs eager, stated at load ([`StatedSteps::eager`]), never refused.
//! - The step builder forms no composition outside the set: a member whose
//!   joining would leave it goes in a later step, and a lone member outside
//!   it is refused at admission by name ([`StepShapes::holds`]). Under the
//!   contract check ([`check_on`]) every step formed is checked against the
//!   set, a gap named with the composition and the nearest stated shapes.

use serde::{Deserialize, Serialize};

use crate::chunk::{ChunkFacts, SlotRole};
use crate::fire::{Lane, Readout, RsVerb};

/// The step builder's rules, the runtime's: what kinds of step it forms and
/// how a lane's recurrent rows may come to a step. The engine computes
/// nothing of them; it reads them from the load request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepRules {
    /// The kinds of step the builder forms.
    pub kinds: Vec<StepKind>,
    /// How the builder may hand a step's recurrent rows, on a load with an rs
    /// list: folded, and buffered (a lane's rows written to the rs list and
    /// replayed, a decode lane's under run-ahead or a beam's, beside prompts
    /// or alone). A load without an rs list folds only.
    pub treatments: Vec<Treatment>,
}

impl StepRules {
    /// The runtime's step builder (`runtime::scheduler::batch`): one-row
    /// decode lanes and multi-row prompt lanes each alone or together in a
    /// step, any member's lanes buffered where the load buffers.
    #[must_use]
    pub fn runtime() -> StepRules {
        StepRules {
            kinds: vec![StepKind::Decode, StepKind::Mixed, StepKind::Prefill],
            treatments: vec![Treatment::Folds, Treatment::Buffers],
        }
    }
}

impl Default for StepRules {
    fn default() -> StepRules {
        StepRules::runtime()
    }
}

/// A step's kind, by its lanes: one-row (decode) lanes only, multi-row
/// (prompt, verify) lanes only, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StepKind {
    Decode,
    Mixed,
    Prefill,
}

impl std::fmt::Display for StepKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StepKind::Decode => "decode",
            StepKind::Mixed => "mixed",
            StepKind::Prefill => "prefill",
        })
    }
}

/// How a step's lanes treat their recurrent rows: every lane folding, or at
/// least one lane buffering ([`RsVerb::Buffer`]) its rows, whatever its kind
/// (a decode lane beside a prompt included).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum Treatment {
    #[default]
    Folds,
    Buffers,
}

impl std::fmt::Display for Treatment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Treatment::Folds => "folds",
            Treatment::Buffers => "~rs",
        })
    }
}

/// An inclusive range of counts; `0..=0` where the class is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub const NONE: Span = Span { lo: 0, hi: 0 };

    #[must_use]
    pub fn new(lo: u32, hi: u32) -> Option<Span> {
        (lo <= hi).then_some(Span { lo, hi })
    }

    #[must_use]
    pub fn contains(self, n: u32) -> bool {
        self.lo <= n && n <= self.hi
    }

    /// How far `n` is from the span: 0 inside it.
    #[must_use]
    pub fn distance(self, n: u32) -> u32 {
        if n < self.lo {
            self.lo - n
        } else {
            n.saturating_sub(self.hi)
        }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.lo == self.hi {
            write!(f, "{}", self.lo)
        } else {
            write!(f, "{}-{}", self.lo, self.hi)
        }
    }
}

/// What a step's lanes carry beside their rows, each a class of lane the
/// step builder puts in any step beside any other: an attention mask, an
/// adapter, a guest input fed from a channel, rs snapshots landed mid-lane, a
/// tap, a kv-less float lane. A shape states the attachments its lanes may
/// carry, in any subset and on any of its lanes; a composition carries those
/// its lanes do. An engine whose bodies key on one records the shape with and
/// without it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Attachments {
    pub mask: bool,
    pub adapter: bool,
    pub inputs: bool,
    pub snapshots: bool,
    pub tap: bool,
    pub float: bool,
}

impl Attachments {
    /// The attachments a load's lanes can carry: a mask, a tap and a float
    /// lane whatever it states (the runtime forms them where a program asks;
    /// an engine refuses a lane it cannot serve by name), an adapter where it
    /// serves adapters (`Capabilities::has_lora`), guest inputs where its
    /// trace declares any (`Capabilities::inputs`), rs snapshots where it
    /// lands them (`Capabilities::rs_snapshots`).
    #[must_use]
    pub fn served(adapter: bool, inputs: bool, snapshots: bool) -> Attachments {
        Attachments {
            mask: true,
            adapter,
            inputs,
            snapshots,
            tap: true,
            float: true,
        }
    }

    /// The attachments `lane` carries.
    #[must_use]
    pub fn of(lane: &Lane) -> Attachments {
        Attachments {
            mask: lane.mask.is_some(),
            adapter: lane.adapter.is_some(),
            inputs: !lane.inputs.is_empty(),
            snapshots: !lane.rs_snapshots.is_empty(),
            tap: lane.tap.is_some(),
            float: lane.kv_less,
        }
    }

    #[must_use]
    pub fn union(self, other: Attachments) -> Attachments {
        Attachments {
            mask: self.mask | other.mask,
            adapter: self.adapter | other.adapter,
            inputs: self.inputs | other.inputs,
            snapshots: self.snapshots | other.snapshots,
            tap: self.tap | other.tap,
            float: self.float | other.float,
        }
    }

    /// Every attachment of `self` is one of `stated`.
    #[must_use]
    pub fn within(self, stated: Attachments) -> bool {
        self.union(stated) == stated
    }

    fn names(self) -> Vec<&'static str> {
        [
            (self.mask, "mask"),
            (self.adapter, "adapter"),
            (self.inputs, "inputs"),
            (self.snapshots, "snapshots"),
            (self.tap, "tap"),
            (self.float, "float"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect()
    }
}

impl std::fmt::Display for Attachments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = self.names();
        if names.is_empty() {
            f.write_str("plain")
        } else {
            f.write_str(&names.join("+"))
        }
    }
}

/// One stated shape: at activation rung `rung` (its tokens), a step of
/// `kind` whose lanes treat their recurrent rows by `treatment`, with its
/// rows, decode lanes, prompt lanes and prompt rows, and the rows it reads
/// out each in their span. The spans are each the class's whole reach at
/// the rung (a composition in the shape has every count in its span and its
/// rows the sum of its decode lanes and its prompt rows). A decode lane takes
/// one row; a prompt lane two or more. The readout rows are its sequence
/// lanes' (a kv-less float lane's rows land through the readout landing and
/// are not counted). Its prompt rows split across its prompt lanes in any
/// way with every lane in `lane_rows` (uneven splits and odd lane counts at
/// any rung: the builder joins whole members as they come), and its lanes
/// may carry any of `attachments`. The lane words' classes are no part of a
/// shape: an engine crosses its own classes with each.
///
/// A folded step runs at the smallest rung its rows fit, so a folded shape's
/// rows are those past the rung below up to its own. A buffered step runs at
/// the smallest buffered rung ([`StepFacts::buffered_rungs`]) holding its own
/// and carried rows, padding up to it, so a buffered prompt or mixed shape's
/// rows start at one whatever its rung (its carried rows may be what lifted
/// it); a buffered decode lane carries none, so a buffered decode shape's
/// rows are those past the buffered rung below, as a folded one's.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepShape {
    pub rung: u32,
    pub kind: StepKind,
    pub treatment: Treatment,
    pub rows: Span,
    pub decode_lanes: Span,
    pub prompt_lanes: Span,
    pub prompt_rows: Span,
    /// One prompt lane's rows: any split of `prompt_rows` with each lane in it.
    pub lane_rows: Span,
    pub readout_rows: Span,
    pub attachments: Attachments,
}

impl StepShape {
    /// Whether `step` is a composition of this shape.
    #[must_use]
    pub fn holds(&self, step: &StepComposition) -> bool {
        self.kind == step.kind()
            && self.treatment == step.treatment
            && self.rows.contains(step.rows)
            && self.decode_lanes.contains(step.decode_lanes)
            && self.prompt_lanes.contains(step.prompt_lanes)
            && self.prompt_rows.contains(step.prompt_rows)
            && self.readout_rows.contains(step.readout_rows)
            && (step.prompt_lanes == 0
                || (self.lane_rows.contains(step.lane_rows.lo)
                    && self.lane_rows.contains(step.lane_rows.hi)))
            && step.attachments.within(self.attachments)
            && step.floor() <= self.rung
    }

    /// What of `step` this shape does not hold, each by name with the
    /// step's count and the shape's span: empty where it holds it.
    #[must_use]
    pub fn misses(&self, step: &StepComposition) -> Vec<String> {
        let mut missed = Vec::new();
        if self.kind != step.kind() {
            missed.push(format!("kind {} (the shape's {})", step.kind(), self.kind));
        }
        if self.treatment != step.treatment {
            missed.push(format!(
                "treatment {} (the shape's {})",
                step.treatment, self.treatment
            ));
        }
        for (what, count, span) in [
            ("rows", step.rows, self.rows),
            ("decode lanes", step.decode_lanes, self.decode_lanes),
            ("prompt lanes", step.prompt_lanes, self.prompt_lanes),
            ("prompt rows", step.prompt_rows, self.prompt_rows),
            ("readout rows", step.readout_rows, self.readout_rows),
        ] {
            if !span.contains(count) {
                missed.push(format!("{what} {count} (the shape's {span})"));
            }
        }
        if step.prompt_lanes > 0
            && !(self.lane_rows.contains(step.lane_rows.lo)
                && self.lane_rows.contains(step.lane_rows.hi))
        {
            missed.push(format!(
                "a lane's rows {} (the shape's {})",
                step.lane_rows, self.lane_rows
            ));
        }
        if !step.attachments.within(self.attachments) {
            missed.push(format!(
                "lanes carrying {} (the shape's {})",
                step.attachments, self.attachments
            ));
        }
        if step.floor() > self.rung {
            missed.push(format!(
                "carried rows {} past the rung's {}",
                step.floor(),
                self.rung
            ));
        }
        missed
    }

    /// How far `step` is from this shape, for naming the nearest: kind and
    /// treatment first, then the rows, lanes and readouts each is past by.
    fn distance(&self, step: &StepComposition) -> (u32, u64) {
        let off = u32::from(self.kind != step.kind())
            + u32::from(self.treatment != step.treatment)
            + u32::from(!step.attachments.within(self.attachments));
        let counts = [
            self.rows.distance(step.rows),
            self.decode_lanes.distance(step.decode_lanes),
            self.prompt_lanes.distance(step.prompt_lanes),
            self.prompt_rows.distance(step.prompt_rows),
            self.readout_rows.distance(step.readout_rows),
            step.floor().saturating_sub(self.rung),
        ];
        (off, counts.iter().map(|&n| u64::from(n)).sum())
    }
}

impl std::fmt::Display for StepShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "r{} {} {} rows {} decode {} prompts {} ({} rows, {} a lane) readouts {} lanes \
             carrying {}",
            self.rung,
            self.kind,
            self.treatment,
            self.rows,
            self.decode_lanes,
            self.prompt_lanes,
            self.prompt_rows,
            self.lane_rows,
            self.readout_rows,
            self.attachments
        )
    }
}

/// What a step is, as a stated shape counts it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepComposition {
    /// Its rows: a decode lane one, a prompt lane its own.
    pub rows: u32,
    pub decode_lanes: u32,
    pub prompt_lanes: u32,
    pub prompt_rows: u32,
    /// Its sequence lanes' readout rows.
    pub readout_rows: u32,
    /// Its carried rows as a rung counts them ([`crate::fire::lane_carried`]):
    /// a buffered step's rung holds them.
    pub carried: u32,
    pub treatment: Treatment,
    /// Its prompt lanes' fewest and most rows (`0..=0` with none).
    pub lane_rows: Span,
    /// The attachments its lanes carry, together.
    pub attachments: Attachments,
}

impl Default for Span {
    fn default() -> Span {
        Span::NONE
    }
}

impl StepComposition {
    /// The composition of a step of `lanes`.
    #[must_use]
    pub fn of<'a>(lanes: impl IntoIterator<Item = &'a Lane>) -> StepComposition {
        let mut step = StepComposition::default();
        for lane in lanes {
            step.add(lane);
        }
        step
    }

    /// Adds `lane` to the step.
    pub fn add(&mut self, lane: &Lane) {
        let own = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
        if own <= 1 {
            self.decode_lanes = self.decode_lanes.saturating_add(1);
            self.rows = self.rows.saturating_add(1);
        } else {
            self.prompt_lanes = self.prompt_lanes.saturating_add(1);
            self.prompt_rows = self.prompt_rows.saturating_add(own);
            self.rows = self.rows.saturating_add(own);
            self.lane_rows = if self.prompt_lanes == 1 {
                Span { lo: own, hi: own }
            } else {
                Span {
                    lo: self.lane_rows.lo.min(own),
                    hi: self.lane_rows.hi.max(own),
                }
            };
        }
        self.attachments = self.attachments.union(Attachments::of(lane));
        if !lane.kv_less {
            let read = match &lane.readout {
                Readout::Last => 1,
                Readout::Rows(rows) => u32::try_from(rows.len()).unwrap_or(u32::MAX),
                Readout::None => 0,
            };
            self.readout_rows = self.readout_rows.saturating_add(read);
        }
        if matches!(lane.rs, RsVerb::Buffer { .. }) {
            self.treatment = Treatment::Buffers;
        }
        let replayed = lane.replay_stated.map_or(0, |stated| stated.replayed);
        self.carried = self
            .carried
            .saturating_add(crate::fire::lane_carried(own.max(1), replayed));
    }

    /// Two steps' lanes in one.
    #[must_use]
    pub fn with(self, other: StepComposition) -> StepComposition {
        StepComposition {
            rows: self.rows.saturating_add(other.rows),
            decode_lanes: self.decode_lanes.saturating_add(other.decode_lanes),
            prompt_lanes: self.prompt_lanes.saturating_add(other.prompt_lanes),
            prompt_rows: self.prompt_rows.saturating_add(other.prompt_rows),
            readout_rows: self.readout_rows.saturating_add(other.readout_rows),
            carried: self.carried.saturating_add(other.carried),
            treatment: self.treatment.max(other.treatment),
            lane_rows: match (self.prompt_lanes, other.prompt_lanes) {
                (0, _) => other.lane_rows,
                (_, 0) => self.lane_rows,
                _ => Span {
                    lo: self.lane_rows.lo.min(other.lane_rows.lo),
                    hi: self.lane_rows.hi.max(other.lane_rows.hi),
                },
            },
            attachments: self.attachments.union(other.attachments),
        }
    }

    #[must_use]
    pub fn kind(&self) -> StepKind {
        match (self.decode_lanes > 0, self.prompt_lanes > 0) {
            (_, false) => StepKind::Decode,
            (false, true) => StepKind::Prefill,
            (true, true) => StepKind::Mixed,
        }
    }

    /// The least rung the step can run at: its own rows, or its carried rows
    /// where they are more (a buffered step's).
    #[must_use]
    pub fn floor(&self) -> u32 {
        self.rows.max(self.carried)
    }
}

impl std::fmt::Display for StepComposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} rows {} decode {} prompts {} ({} rows, {} a lane) readouts {} carried {} \
             lanes carrying {}",
            self.kind(),
            self.treatment,
            self.rows,
            self.decode_lanes,
            self.prompt_lanes,
            self.prompt_rows,
            self.lane_rows,
            self.readout_rows,
            self.carried,
            self.attachments
        )
    }
}

/// The load's facts the set is computed from, as the load states them: its
/// activation rungs' tokens ([`ChunkFacts::act_chunks`]; one fire's tokens
/// where it states none), its fire limits, and whether it holds an rs list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepFacts {
    pub rungs: Vec<u32>,
    pub max_lanes: u32,
    pub max_tokens: u32,
    /// The rows one fire reads out at most (0: no bound past its rows):
    /// [`crate::ReadoutFacts::fire_rows`] of the engine's ceiling.
    pub max_readouts: u32,
    pub buffers: bool,
    /// The attachments its lanes can carry ([`Attachments::served`]).
    pub attachments: Attachments,
    /// The mixed steps the engine lays out ([`crate::Capabilities::mixed_lanes`]):
    /// its prompt-lane rungs and the decodes beside each, where it pads a
    /// mixed step's lanes to them. A mixed step past them is not stated, and
    /// the runtime forms it no more (its extra prompts go in the step after).
    #[serde(default)]
    pub mixed: crate::MixedLanes,
}

impl StepFacts {
    /// The facts of a load stating `chunks`, `max_lanes` lanes and
    /// `max_tokens` tokens a fire, and reading out at most `max_readouts`
    /// rows a fire, its lanes carrying `attachments`.
    #[must_use]
    pub fn of(
        chunks: &ChunkFacts,
        max_lanes: u32,
        max_tokens: u32,
        max_readouts: u32,
        attachments: Attachments,
    ) -> StepFacts {
        let mut rungs: Vec<u32> = chunks
            .act_chunks
            .iter()
            .map(|&(tokens, _)| tokens)
            .collect();
        rungs.sort_unstable();
        rungs.dedup();
        StepFacts {
            rungs,
            max_lanes,
            max_tokens,
            max_readouts,
            buffers: chunks
                .shapes
                .iter()
                .any(|shape| shape.role == SlotRole::RsBuffer),
            attachments,
            mixed: crate::MixedLanes::Any,
        }
    }

    /// The facts as a load states them in `caps` (its chunks, fire limits,
    /// and the attachments it serves) and its `readouts` (the rows one fire
    /// reads out, under its ceiling).
    #[must_use]
    pub fn stated(caps: &crate::Capabilities, readouts: &crate::ReadoutFacts) -> StepFacts {
        Self::stated_of(caps, readouts).with_mixed(caps.mixed_lanes.clone())
    }

    /// These facts with the mixed steps the engine lays out (`mixed`).
    #[must_use]
    pub fn with_mixed(self, mixed: crate::MixedLanes) -> StepFacts {
        StepFacts { mixed, ..self }
    }

    fn stated_of(caps: &crate::Capabilities, readouts: &crate::ReadoutFacts) -> StepFacts {
        StepFacts::of(
            &caps.chunks,
            caps.limits.max_lanes,
            caps.limits.max_tokens,
            readouts.fire_rows(caps.limits.max_readouts),
            Attachments::served(
                caps.has_lora,
                !caps.inputs.is_empty(),
                caps.rs_snapshots > 0,
            ),
        )
    }

    /// The rungs a folded step can run at: every stated one up to the first
    /// that holds a whole fire's tokens; a fire's tokens alone where none is.
    #[must_use]
    pub fn lattice(&self) -> Vec<u32> {
        let mut lattice = Vec::new();
        for &rung in &self.rungs {
            lattice.push(rung);
            if rung >= self.max_tokens {
                break;
            }
        }
        if lattice.last().is_none_or(|&top| top < self.max_tokens) {
            lattice.push(self.max_tokens);
        }
        lattice
    }

    /// The folded lattice's widest spacing, `(upper, lower)`: the largest
    /// ratio `R_i / R_(i-1)` of two neighbouring rungs ([`Self::lattice`]),
    /// `(1, 1)` with one rung. A folded step whose rows are just past
    /// `R_(i-1)` runs at `R_i`: this is the most a step's rung exceeds the
    /// rung below it that the load already accepts for its folded steps, the
    /// common ones.
    #[must_use]
    pub fn spacing(&self) -> (u32, u32) {
        let lattice = self.lattice();
        lattice
            .windows(2)
            .map(|pair| (pair[1], pair[0].max(1)))
            .fold((1, 1), |widest, ratio| {
                if u64::from(ratio.0) * u64::from(widest.1)
                    > u64::from(widest.0) * u64::from(ratio.1)
                {
                    ratio
                } else {
                    widest
                }
            })
    }

    /// The rungs a buffered step can run at: the folded lattice thinned, its
    /// top always kept. Every rung a step can run at is a set of bodies (one
    /// per composition, class set and rs body) an engine records or compiles
    /// before ready, and a buffered step (a lane that may retract: a beam, a
    /// draft, an inferlet that does not state its rows final) is rare beside
    /// a folded one, so a buffered rung saves a whole set of bodies at the
    /// price of padding a rare step. The rule, derived from the load's own
    /// lattice and no constant: a buffered step pads up from the rung it
    /// would run at folded (`F`, the smallest folded rung holding it) to its
    /// buffered rung (`B`) by no more than the lattice's widest spacing
    /// ([`Self::spacing`]), `B / F <= R_i / R_(i-1)` at its widest: it pays,
    /// over its folded step, no more than a folded step already pays over
    /// the rung below it. Rows stand for step time here: a step's time grows
    /// no faster than its rows (a weight read every step whatever its rows,
    /// and work per row), so a step at `B` takes at most `B / F` the time of
    /// one at `F`, and the bound in rows bounds the step-time padding too.
    /// Greedy from the smallest folded rung `F`: its buffered rung is the
    /// widest folded rung within `F`'s bound, and the next buffered rung's
    /// `F` the folded rung after it. A lattice of evenly spaced rungs (each
    /// twice the one below) keeps every other rung; one with wider gaps
    /// keeps fewer.
    #[must_use]
    pub fn buffered_rungs(&self) -> Vec<u32> {
        let lattice = self.lattice();
        let (upper, lower) = self.spacing();
        let within = |rung: u32, folded: u32| {
            u64::from(rung) * u64::from(lower) <= u64::from(upper) * u64::from(folded)
        };
        let mut rungs = Vec::new();
        let mut at = 0;
        while at < lattice.len() {
            let folded = lattice[at];
            let mut to = at;
            while to + 1 < lattice.len() && within(lattice[to + 1], folded) {
                to += 1;
            }
            rungs.push(lattice[to]);
            at = to + 1;
        }
        rungs
    }
}

/// The finite set of step compositions a load's step builder forms, ordered
/// smallest first across kinds: by rung, then decode, mixed, prompt, then
/// folded before buffered. What an engine's arm walk takes in its order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepShapes {
    pub shapes: Vec<StepShape>,
}

impl StepShapes {
    /// The set `rules` form over a load of `facts`.
    #[must_use]
    pub fn of(rules: &StepRules, facts: &StepFacts) -> StepShapes {
        let lanes = facts.max_lanes.max(1);
        let reads = |rows: u32| match facts.max_readouts {
            0 => rows,
            cap => rows.min(cap),
        };
        let treatments: Vec<Treatment> = rules
            .treatments
            .iter()
            .copied()
            .filter(|&t| t == Treatment::Folds || facts.buffers)
            .collect();
        let buffered = facts.buffered_rungs();
        let mut shapes = Vec::new();
        let mut below = 0u32;
        let mut buffered_below = 0u32;
        for rung in facts.lattice() {
            let top = rung.min(facts.max_tokens);
            let buffers_here = buffered.contains(&rung);
            for &kind in &rules.kinds {
                for &treatment in &treatments {
                    if treatment == Treatment::Buffers && !buffers_here {
                        continue;
                    }
                    // A buffered decode lane carries no row (`lane_carried`),
                    // so a buffered decode step's rung is its rows' alone.
                    let lo = match (treatment, kind) {
                        (Treatment::Folds, _) => below + 1,
                        (Treatment::Buffers, StepKind::Decode) => buffered_below + 1,
                        (Treatment::Buffers, _) => 1,
                    };
                    let shape = match kind {
                        StepKind::Decode => Span::new(lo, top.min(lanes)).map(|d| StepShape {
                            rung,
                            kind,
                            treatment,
                            rows: d,
                            decode_lanes: d,
                            prompt_lanes: Span::NONE,
                            prompt_rows: Span::NONE,
                            lane_rows: Span::NONE,
                            readout_rows: Span {
                                lo: 0,
                                hi: reads(d.hi),
                            },
                            attachments: facts.attachments,
                        }),
                        StepKind::Prefill => Span::new(lo.max(2), top).and_then(|rows| {
                            Some(StepShape {
                                rung,
                                kind,
                                treatment,
                                rows,
                                decode_lanes: Span::NONE,
                                prompt_lanes: Span::new(1, lanes.min(rows.hi / 2))?,
                                prompt_rows: rows,
                                lane_rows: Span::new(2, rows.hi)?,
                                readout_rows: Span {
                                    lo: 0,
                                    hi: reads(rows.hi),
                                },
                                attachments: facts.attachments,
                            })
                        }),
                        StepKind::Mixed => {
                            // Each prompt-lane rung the engine lays a mixed
                            // step at (its prompts and their rest lane), with
                            // the decodes beside it; any counts within the
                            // lanes where it states none.
                            let bands: Vec<(u32, u32, u32)> = match &facts.mixed {
                                crate::MixedLanes::Any => vec![(1, lanes - 1, lanes - 1)],
                                crate::MixedLanes::Laid(rungs) => {
                                    let mut from = 1;
                                    rungs
                                        .iter()
                                        .map(|r| {
                                            let band = (
                                                from,
                                                r.prompt_lanes.saturating_sub(1),
                                                r.decode_lanes,
                                            );
                                            from = r.prompt_lanes;
                                            band
                                        })
                                        .collect()
                                }
                            };
                            for (p_lo, p_hi, d_hi) in bands {
                                let shape = Span::new(lo.max(3), top).and_then(|rows| {
                                    Some(StepShape {
                                        rung,
                                        kind,
                                        treatment,
                                        rows,
                                        decode_lanes: Span::new(
                                            1,
                                            d_hi.min(lanes - 1).min(rows.hi - 2),
                                        )?,
                                        prompt_lanes: Span::new(
                                            p_lo,
                                            p_hi.min(lanes - 1).min((rows.hi - 1) / 2),
                                        )?,
                                        prompt_rows: Span::new(2, rows.hi - 1)?,
                                        lane_rows: Span::new(2, rows.hi - 1)?,
                                        readout_rows: Span {
                                            lo: 0,
                                            hi: reads(rows.hi),
                                        },
                                        attachments: facts.attachments,
                                    })
                                });
                                shapes.extend(shape);
                            }
                            None
                        }
                    };
                    shapes.extend(shape);
                }
            }
            below = rung;
            if buffers_here {
                buffered_below = rung;
            }
        }
        shapes.sort_by_key(|shape| (shape.rung, shape.kind, shape.treatment));
        StepShapes { shapes }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// The first shape (in the set's order) holding `step`, by its index.
    #[must_use]
    pub fn shape_of(&self, step: &StepComposition) -> Option<usize> {
        self.shapes.iter().position(|shape| shape.holds(step))
    }

    /// Whether the set holds `step`; else the gap, named with the step and
    /// the nearest stated shapes.
    pub fn holds(&self, step: &StepComposition) -> Result<usize, StepGap> {
        self.shape_of(step).ok_or_else(|| StepGap {
            step: *step,
            nearest: self.nearest(step, 3),
        })
    }

    /// The `n` shapes nearest `step`.
    #[must_use]
    pub fn nearest(&self, step: &StepComposition, n: usize) -> Vec<StepShape> {
        let mut by: Vec<&StepShape> = self.shapes.iter().collect();
        by.sort_by_key(|shape| shape.distance(step));
        by.into_iter().take(n).cloned().collect()
    }

    /// The shapes' count by kind and treatment, for a load's line.
    #[must_use]
    pub fn census(&self) -> String {
        let mut parts = Vec::new();
        for kind in [StepKind::Decode, StepKind::Mixed, StepKind::Prefill] {
            for treatment in [Treatment::Folds, Treatment::Buffers] {
                let n = self
                    .shapes
                    .iter()
                    .filter(|shape| shape.kind == kind && shape.treatment == treatment)
                    .count();
                if n > 0 {
                    parts.push(format!("{kind} {treatment} {n}"));
                }
            }
        }
        let rungs = {
            let mut rungs: Vec<u32> = self.shapes.iter().map(|shape| shape.rung).collect();
            rungs.dedup();
            rungs.len()
        };
        format!(
            "{} shapes over {rungs} rungs ({})",
            self.len(),
            parts.join(", ")
        )
    }
}

/// The stated shape a fire's step is of (CONTRACT-BC 24): its index in the
/// load's stated set ([`StatedSteps::shapes`]) and that shape's treatment,
/// set by the runtime as it forms the step from what
/// [`StepShapes::shape_of`] answers for the step's composition. A fire
/// carries its stated shape; an engine keys its recorded body on it rather
/// than computing a key of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FireShape {
    /// The shape's index in the stated set's order.
    pub id: u32,
    /// The shape's treatment: whether any lane buffers its recurrent rows.
    pub treatment: Treatment,
}

impl StepShapes {
    /// The stated shape holding `step`, as a fire carries it; else the gap.
    pub fn fire_shape(&self, step: &StepComposition) -> Result<FireShape, StepGap> {
        let at = self.holds(step)?;
        Ok(FireShape {
            id: u32::try_from(at).unwrap_or(u32::MAX),
            treatment: self.shapes[at].treatment,
        })
    }

    /// Whether `shape`, carried by a fire of `lanes`, is the stated shape
    /// this set holds those lanes in: an engine's cross-check of the
    /// runtime's stamp against its own reading of the step. Else why, by
    /// name.
    pub fn agrees(&self, shape: FireShape, lanes: &[Lane]) -> Result<(), String> {
        let step = StepComposition::of(lanes);
        let Some(stated) = self.shapes.get(shape.id as usize) else {
            return Err(format!(
                "a fire carries stated shape {} of the {} the load stated: {step}",
                shape.id,
                self.len()
            ));
        };
        if stated.treatment != shape.treatment {
            return Err(format!(
                "a fire carries stated shape {} as {} where the set states it {}",
                shape.id, shape.treatment, stated.treatment
            ));
        }
        match self.shape_of(&step) {
            Some(at) if at == shape.id as usize => Ok(()),
            Some(at) => Err(format!(
                "a fire carries stated shape {} [{stated}] where its lanes make shape {at} \
                 [{}]: {step}",
                shape.id, self.shapes[at]
            )),
            None => Err(format!(
                "a fire carries stated shape {} [{stated}] and no stated shape holds its \
                 lanes: {step}",
                shape.id
            )),
        }
    }
}

/// A composition no stated shape holds: a contract bug, named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepGap {
    pub step: StepComposition,
    pub nearest: Vec<StepShape>,
}

impl std::fmt::Display for StepGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a step outside the set the load stated: {}", self.step)?;
        if let Some(nearest) = self.nearest.first() {
            write!(
                f,
                "; past the nearest in {}",
                nearest.misses(&self.step).join(", ")
            )?;
        }
        f.write_str("; nearest stated:")?;
        for shape in &self.nearest {
            write!(f, " [{shape}]")?;
        }
        Ok(())
    }
}

/// The set a load states ([`crate::Capabilities::steps`]): the shapes it
/// computed by [`StepShapes::of`] from the request's rules and its own facts,
/// and those of them it runs eager (indices into `shapes`, ascending): the
/// tail of the set's order its measured allowance did not hold the bodies
/// of. Every other shape's bodies were recorded before the load returned.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatedSteps {
    pub shapes: StepShapes,
    #[serde(default)]
    pub eager: Vec<u32>,
}

impl StatedSteps {
    /// The set a load computes, none of it stated eager yet.
    #[must_use]
    pub fn of(rules: &StepRules, facts: &StepFacts) -> StatedSteps {
        StatedSteps {
            shapes: StepShapes::of(rules, facts),
            eager: Vec::new(),
        }
    }

    /// Whether the engine's stated set is the one the runtime's rules form
    /// over the load's stated facts, and its eager shapes are of it; else
    /// why, by name.
    pub fn agrees(&self, rules: &StepRules, facts: &StepFacts) -> Result<(), String> {
        let ours = StepShapes::of(rules, facts);
        if ours != self.shapes {
            let first = ours
                .shapes
                .iter()
                .zip(&self.shapes.shapes)
                .position(|(a, b)| a != b)
                .unwrap_or(ours.len().min(self.shapes.len()));
            return Err(format!(
                "the engine states {} step shapes and the runtime's rules form {} over the \
                 load's facts {facts:?}; they part at shape {first}: the engine's {:?}, the \
                 runtime's {:?}",
                self.shapes.len(),
                ours.len(),
                self.shapes.shapes.get(first).map(ToString::to_string),
                ours.shapes.get(first).map(ToString::to_string),
            ));
        }
        if let Some(&past) = self
            .eager
            .iter()
            .find(|&&at| at as usize >= self.shapes.len())
        {
            return Err(format!(
                "the engine states shape {past} eager of the {} it states",
                self.shapes.len()
            ));
        }
        Ok(())
    }
}

/// Whether the step builder checks every step it forms against the stated
/// set: debug builds, or `PIE_CHECK_REPLAY` set, the one contract check flag
/// ([`crate::fire::replay_check_on`]); it changes nothing but to name a gap.
#[must_use]
pub fn check_on() -> bool {
    crate::fire::replay_check_on()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(rungs: &[u32], buffers: bool) -> StepFacts {
        StepFacts {
            rungs: rungs.to_vec(),
            max_lanes: 64,
            max_tokens: 2048,
            max_readouts: 0,
            buffers,
            attachments: Attachments::served(true, true, true),
            mixed: crate::MixedLanes::Any,
        }
    }

    /// An engine stating no mixed lane rungs (`MixedLanes::Any`: CUDA,
    /// engine-gpu, Metal, Cerebras) states one mixed shape a rung and
    /// treatment, any prompt and decode counts within its lanes, as before
    /// the rungs; one stating rungs (XLA) a shape a prompt-lane rung, its
    /// decodes no more than the rung lays beside it.
    #[test]
    fn mixed_shapes_follow_the_engines_lane_rungs_and_none_where_it_states_none() {
        let any = facts(&[16, 256, 2048], true);
        let set = StepShapes::of(&StepRules::runtime(), &any);
        let mixed: Vec<&StepShape> = set
            .shapes
            .iter()
            .filter(|s| s.kind == StepKind::Mixed)
            .collect();
        for shape in &mixed {
            let lanes = any.max_lanes;
            assert_eq!(
                shape.prompt_lanes,
                Span {
                    lo: 1,
                    hi: (lanes - 1).min((shape.rows.hi - 1) / 2)
                }
            );
            assert_eq!(
                shape.decode_lanes,
                Span {
                    lo: 1,
                    hi: (lanes - 1).min(shape.rows.hi - 2)
                }
            );
        }
        let per = |set: &StepShapes| {
            let mut keys: Vec<(u32, Treatment)> = set
                .shapes
                .iter()
                .filter(|s| s.kind == StepKind::Mixed)
                .map(|s| (s.rung, s.treatment))
                .collect();
            let n = keys.len();
            keys.dedup();
            (n, keys.len())
        };
        let (n, distinct) = per(&set);
        assert_eq!(n, distinct, "one mixed shape a rung and treatment");
        let rung = |prompt_lanes, decode_lanes| crate::MixedRung {
            prompt_lanes,
            decode_lanes,
        };
        let laid = any.clone().with_mixed(crate::MixedLanes::Laid(vec![
            rung(16, 48),
            rung(32, 32),
            rung(64, 0),
        ]));
        let set = StepShapes::of(&StepRules::runtime(), &laid);
        for shape in set.shapes.iter().filter(|s| s.kind == StepKind::Mixed) {
            let (lo, hi, d) = if shape.prompt_lanes.hi <= 15 {
                (1, 15, 48)
            } else {
                (16, 31, 32)
            };
            assert!(
                shape.prompt_lanes.lo == lo && shape.prompt_lanes.hi <= hi,
                "{shape:?}"
            );
            assert!(shape.decode_lanes.hi <= d, "{shape:?}");
        }
        // a rung laying no decodes beside it states no mixed shape
        assert!(
            set.shapes
                .iter()
                .all(|s| s.kind != StepKind::Mixed || s.prompt_lanes.hi < 32)
        );
    }

    fn lane(rows: usize, buffered: bool) -> Lane {
        Lane {
            tokens: vec![1; rows],
            rs: if buffered {
                RsVerb::Buffer {
                    pages: Vec::new(),
                    base: 0,
                    folded: 0,
                    fold: crate::fire::FoldLen::Host(0),
                    lineage: crate::fire::Lineage::Set,
                }
            } else {
                RsVerb::Fold
            },
            ..Lane::default()
        }
    }

    #[test]
    fn the_set_is_finite_and_ordered_smallest_first_across_kinds() {
        let rungs = [8, 64, 256, 2048];
        let folded = StepShapes::of(&StepRules::runtime(), &facts(&rungs, false));
        // Rung 8: decode, mixed, prompt; 64: the same; 256 and 2048: no
        // decode past the 64 lanes, mixed and prompt.
        assert_eq!(folded.len(), 3 + 3 + 2 + 2, "{}", folded.census());
        let buffered = StepShapes::of(&StepRules::runtime(), &facts(&rungs, true));
        // The widest spacing is 8 (8 to 64, 256 to 2048): buffered rungs 64
        // (8 padded up to 8x) and 2048 (256 up to 8x). At 64 a buffered
        // decode, mixed and prompt; at 2048 mixed and prompt (no decode
        // past the 64 lanes).
        assert_eq!(facts(&rungs, true).buffered_rungs(), vec![64, 2048]);
        assert_eq!(buffered.len(), 10 + 3 + 2, "{}", buffered.census());
        let keys: Vec<_> = buffered
            .shapes
            .iter()
            .map(|s| (s.rung, s.kind, s.treatment))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        assert!(
            buffered
                .shapes
                .iter()
                .any(|s| s.kind == StepKind::Mixed && s.treatment == Treatment::Buffers),
            "decode lanes buffering beside a prompt are stated"
        );
    }

    /// Buffered compositions are stated at fewer rungs than folded ones,
    /// every buffered step pads to a stated buffered rung, and none pads
    /// past its folded rung by more than the lattice's widest spacing.
    #[test]
    fn buffered_steps_pad_to_fewer_rungs_within_the_lattice_s_spacing() {
        for rungs in [
            (0..12).map(|k| 1u32 << k).collect::<Vec<_>>(),
            vec![16, 32, 64, 128, 256, 512, 1024, 2048],
            vec![8, 64, 256, 2048],
            vec![4, 16, 64],
            vec![2048],
        ] {
            let f = facts(&rungs, true);
            let set = StepShapes::of(&StepRules::runtime(), &f);
            let lattice = f.lattice();
            let buffered = f.buffered_rungs();
            let (upper, lower) = f.spacing();
            assert_eq!(buffered.last(), lattice.last(), "the top is kept");
            assert!(buffered.iter().all(|rung| lattice.contains(rung)));
            let count = |t: Treatment| set.shapes.iter().filter(|s| s.treatment == t).count();
            if lattice.len() > 1 {
                assert!(
                    count(Treatment::Buffers) < count(Treatment::Folds),
                    "{rungs:?}: {}",
                    set.census()
                );
            }
            for rows in 1..=2048u32 {
                for prompt in [false, true] {
                    let lanes: Vec<Lane> = if prompt && rows > 1 {
                        vec![lane(rows as usize, true)]
                    } else if rows <= 64 {
                        vec![lane(1, true); rows as usize]
                    } else {
                        continue;
                    };
                    let step = StepComposition::of(&lanes);
                    let at = set.holds(&step).unwrap_or_else(|gap| panic!("{gap}"));
                    let rung = set.shapes[at].rung;
                    assert!(buffered.contains(&rung), "{rows}: r{rung} {buffered:?}");
                    let folded = *lattice.iter().find(|&&r| r >= rows).unwrap();
                    assert!(
                        u64::from(rung) * u64::from(lower) <= u64::from(upper) * u64::from(folded),
                        "{rows} rows: r{rung} over folded r{folded} past {upper}/{lower}"
                    );
                }
            }
        }
    }

    /// A Qwen3.5-like hybrid load (rungs every power of two from 16 to the
    /// fire's 2048 tokens, 64 lanes, an rs list): what it states before and
    /// after buffered rungs were thinned.
    #[test]
    fn a_hybrid_load_states_fewer_buffered_compositions() {
        let f = facts(&[16, 32, 64, 128, 256, 512, 1024, 2048], true);
        let set = StepShapes::of(&StepRules::runtime(), &f);
        let count = |t: Treatment| set.shapes.iter().filter(|s| s.treatment == t).count();
        // Folded: decode, mixed, prompt at 16, 32, 64; mixed, prompt past.
        assert_eq!(count(Treatment::Folds), 3 * 3 + 2 * 5, "{}", set.census());
        // Buffered at 32, 128, 512, 2048 (every rung before: 3 * 3 + 2 * 5):
        // decode, mixed, prompt at 32; decode (33-64 lanes), mixed, prompt at
        // 128; mixed, prompt at 512 and 2048.
        assert_eq!(f.buffered_rungs(), vec![32, 128, 512, 2048]);
        assert_eq!(count(Treatment::Buffers), 3 + 3 + 2 + 2, "{}", set.census());
    }

    #[test]
    fn every_composition_within_the_limits_is_in_the_set() {
        let set = StepShapes::of(&StepRules::runtime(), &facts(&[8, 64, 256, 2048], true));
        for decodes in 0..=8u32 {
            for prompts in 0..=3u32 {
                for rows in [2usize, 5, 60, 500] {
                    for buffered in [false, true] {
                        let mut lanes: Vec<Lane> =
                            (0..decodes).map(|_| lane(1, buffered)).collect();
                        lanes.extend((0..prompts).map(|_| lane(rows, false)));
                        if lanes.is_empty() {
                            continue;
                        }
                        let step = StepComposition::of(&lanes);
                        if step.rows > 2048 {
                            continue;
                        }
                        set.holds(&step).unwrap_or_else(|gap| panic!("{gap}"));
                    }
                }
            }
        }
    }

    /// Uneven prompt splits ([10, 6]), odd lane counts ([3, 3, 3, 3] and a
    /// lone 3) and lanes carrying a mask, an adapter and a guest input at a
    /// small rung are all in the set; a lane carrying an attachment the
    /// load does not serve is not.
    #[test]
    fn uneven_splits_and_attached_lanes_are_in_the_set() {
        let set = StepShapes::of(&StepRules::runtime(), &facts(&[4, 16, 64], false));
        for rows in [
            vec![10usize, 6],
            vec![3, 3, 3, 3],
            vec![3],
            vec![16],
            vec![1, 1],
        ] {
            let mut lanes: Vec<Lane> = rows.iter().map(|&n| lane(n, false)).collect();
            lanes[0].mask = Some(crate::fire::Masking::Extent(crate::fire::Mask::new(
                vec![1],
                1,
            )));
            lanes[0].adapter = Some(0);
            lanes[0].inputs = vec![crate::fire::InputFeed {
                id: 0,
                channel: crate::ChannelId(0),
            }];
            let step = StepComposition::of(&lanes);
            set.holds(&step)
                .unwrap_or_else(|gap| panic!("{rows:?}: {gap}"));
        }
        let plain = StepFacts {
            attachments: Attachments::served(false, false, false),
            ..facts(&[4, 16, 64], false)
        };
        let set = StepShapes::of(&StepRules::runtime(), &plain);
        let mut adapted = lane(4, false);
        adapted.adapter = Some(1);
        let gap = set.holds(&StepComposition::of(&[adapted])).unwrap_err();
        assert!(gap.to_string().contains("carrying adapter"), "{gap}");
    }

    #[test]
    fn a_gap_is_named_with_the_nearest_shapes() {
        let set = StepShapes::of(&StepRules::runtime(), &facts(&[8, 64], false));
        let step = StepComposition::of(&[lane(1, true), lane(4, false)]);
        let gap = set.holds(&step).unwrap_err();
        let said = gap.to_string();
        assert!(said.contains("mixed ~rs rows 5"), "{said}");
        assert!(said.contains("r8 mixed folds"), "{said}");
    }

    #[test]
    fn a_folded_step_runs_at_the_rung_its_rows_fit() {
        let set = StepShapes::of(&StepRules::runtime(), &facts(&[8, 64], false));
        let step = StepComposition::of(&vec![lane(1, false); 9]);
        let at = set.holds(&step).unwrap();
        assert_eq!(set.shapes[at].rung, 64);
        assert_eq!(set.shapes[at].rows, Span { lo: 9, hi: 64 });
    }

    #[test]
    fn a_set_other_than_the_rules_form_is_refused_by_name() {
        let f = facts(&[8, 64], true);
        let mut stated = StatedSteps::of(&StepRules::runtime(), &f);
        assert!(stated.agrees(&StepRules::runtime(), &f).is_ok());
        stated.shapes.shapes.pop();
        let why = stated.agrees(&StepRules::runtime(), &f).unwrap_err();
        assert!(why.contains("part at shape"), "{why}");
    }
}
