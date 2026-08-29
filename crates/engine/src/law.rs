//! The shared vocabulary for what moves per fire — the seat wave B-law fills.
//!
//! Two shells solved "which arguments of a recorded launch move with the
//! composition" with private machinery (CUDA NodeMap/fold, Metal abi/rebind);
//! the language they both speak lands here.
//!
//! # The component language, and why it has exactly three fitted forms
//!
//! (Moved verbatim from `engine_metal::abi`, which is where the language was
//! written down first.)
//!
//! ```text
//! Const  v                                   encoded once, never rewritten
//! Affine v = base + Σ slope[k] · coord[k]     the windowed cut, the extent
//! Ceil   v = mul · ⌈(α·rows + β) / div⌉       the TILING law
//! ```
//!
//! The third one is the Metal wave's addition and the shape of it is not a
//! generalisation for its own sake. Build log 30 derived the table for
//! qwen35-d0.8b and six components of 5579 refused: `sdpa_paged_tiled`'s
//! second grid axis, which `kernels_metal::attn::tiled_grid` writes
//! `rows.div_ceil(SDPA_TILE)`. Two more appear the moment the 127
//! arm-switching slots are fitted rather than skipped — `linear::gemm`'s tile
//! arm dispatches `div_ceil(rows, TILE_M) · TILE_GROUP[1]` row tiles. Both
//! are a ceiling over the WINDOW'S ROWS, scaled, and nothing in the catalog
//! is a ceiling over anything else. So the law reads the window's rows rather
//! than the coordinates, which costs one more law per slot and buys a form a
//! reader can check against the shader source in one glance.
//!
//! **A component matching neither law is still refused by name.** The
//! refusal did not get weaker; it got a second thing to try first.
//!
//! [`Law::Slot`] is the fourth form and it is not fitted at all: it names a
//! descriptor slot the value is READ out of at fire time. It is what a
//! recorder states when the number is not a function of the composition's
//! coordinates but of the fire's own descriptor — the form the alto design
//! (`§3`) reserved for the frame wave, and the reason this enum is four
//! variants rather than the three either shell has produced so far.
//!
//! # The method, and why it is honest rather than clever
//!
//! Walk the same template many times against synthetic descriptors, record
//! each walk, and read the differences:
//!
//! ```text
//! probe    : a base composition, and one LADDER per direction —
//!            base + 1·e_k, base + 2·e_k, ... base + L·e_k
//! check    : a composition no probe visited, held out of every fit
//!
//! for each slot, for each grid axis and each argument:
//!   equal across every sample of the arm   → a CONSTANT, encoded once
//!   moved, and a line fits every sample     → Affine
//!   moved, and a scaled ceiling over the
//!     window's rows fits every sample       → Ceil
//!   moved, and neither                      → Refuse::Unaffine
//! ```
//!
//! **Why a ladder and not two points.** A fit through two points always fits.
//! Build log 30's bug was subtler than that: a grid axis written
//! `rows.div_ceil(32)` is FLAT across every step small enough to stay inside
//! one tile, so a two-point probe called it a constant and would have encoded
//! the wrong grid forever. A ladder that crosses a tile boundary is the
//! smallest thing that can see a staircase at all, and it is also what
//! brackets an arm switch to the row it happens at.
//!
//! # The two recorders, and what each of them supplies
//!
//! The fit is one thing; getting the samples is the shell's. The two shells
//! stand in different places and the vocabulary is what they have in common
//! rather than what they do:
//!
//! ```text
//!                 Metal (engine_metal::abi)     CUDA (engine_cuda::device::map)
//! the sample      a Recording: one walk's       a Walked: one captured graph's
//!                 ICB slots, at a stated        kernel nodes, in the canonical
//!                 point of a probe basis        (depth, symbol, index) order
//! how many        a probe LADDER per direction  TWO — the template capture
//!                 plus a held-out check         and this fire's throwaway
//! what it gets    a fitted Law per component    a Law::Const per component
//!                 (Const / Affine / Ceil)       that moved: an observation,
//!                                               not a function
//! what it does    lowers the table into a       restates the whole node into
//!                 device rebind shader          an instantiated graph exec
//! ```
//!
//! **A two-capture diff is a degenerate fit and the type says so.** CUDA's
//! `diff` observes rather than solves, so every [`Component`] it states
//! carries a [`Law::Const`] of the value the new capture wants. That is not a
//! weaker claim wearing the same word — it is exactly the claim
//! `cudaGraphExecKernelNodeSetParams` needs, which is "this number, now", and
//! writing it in the shared language is what lets one reader compare the two
//! planes' censuses at all. `engine_cuda::record`'s `fit_zeros` is the CUDA
//! plane's one genuine fit — a 4-byte cell that reads the segment's row count
//! in BOTH captures, at two DIFFERENT row counts, is the row count riding in
//! an argument, which is `Affine { base: 0, slope: [1] }` over the one
//! direction that plane can step — and it keeps its own shape because what it
//! produces is a whole zeroed launch statement rather than a component.
//!
//! # What the vocabulary does NOT cover, stated
//!
//! A walk skips a zero-row region's nodes (`crate::fire::walk` rule 1), so a
//! composition with an empty window produces FEWER slots than one without.
//! A recorder therefore probes at compositions that hold every class, and the
//! table it derives is the FULL composition's — which is the point rather
//! than a limitation: design §5's "all compositions live inside it" means the
//! artifact holds every launch and a fire turns the absent ones off. What
//! turns one off is that slot's window-rows law evaluating to zero, and the
//! ICB (Metal) or the enable bit (CUDA) is what acts on it.

pub mod fit;

use std::fmt;

/// One direction the composition can actually be moved along.
///
/// **NOT A COORDINATE OF THE DESCRIPTOR, A DIRECTION IN IT.** The descriptor
/// holds `(rows, lanes)` per class and the two are not always independently
/// reachable — a decode class's word says one token per lane, so every batch
/// that adds a decode row adds a decode lane. A basis of directions the
/// harness can genuinely step along is what a pair of axes would have had to
/// pretend, and the name is what makes the law readable: a slope of 12288 on
/// "a prefill token" means twelve kilobytes of arena per token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Axis {
    /// What one step of this direction does, in words.
    pub name: String,
    /// What one step does to each class's `(rows, lanes)`, for the reader
    /// and for [`Recipe`], which inverts exactly this.
    pub step: Vec<(i32, i32)>,
}

impl Axis {
    /// A direction, named.
    #[must_use]
    pub fn new(name: impl Into<String>, step: Vec<(i32, i32)>) -> Axis {
        Axis {
            name: name.into(),
            step,
        }
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// One direction, read back out of a class table.
///
/// **THE INVERSE OF THE BASIS, AND IT IS WHY THE SHADER NEEDS NO WALK.** The
/// laws are written in the probe basis; a fire carries a class table. This is
/// the one linear functional that turns the second into the first:
///
/// ```text
/// coord[k] = konst[k] + Σ_c ( rows[k][c]·classes[c].rows
///                           + lanes[k][c]·classes[c].lanes )
/// ```
///
/// It is SOLVED, not stated: [`fit::invert`] picks a square subsystem of the
/// step matrix, inverts it exactly over the integers, and then verifies the
/// resulting recipe against every probe's own coordinates. A basis whose
/// inverse is not integral is refused by name rather than rounded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// The value at an empty class table.
    pub konst: i128,
    /// One coefficient per class, over that class's rows.
    pub rows: Vec<i128>,
    /// One coefficient per class, over that class's lanes.
    pub lanes: Vec<i128>,
}

impl Recipe {
    /// This direction's coordinate at a class table.
    #[must_use]
    pub fn at(&self, classes: &[(u32, u32)]) -> i128 {
        let mut sum = self.konst;
        for (c, (rows, lanes)) in classes.iter().enumerate() {
            sum += self.rows.get(c).copied().unwrap_or(0) * i128::from(*rows);
            sum += self.lanes.get(c).copied().unwrap_or(0) * i128::from(*lanes);
        }
        sum
    }
}

/// A descriptor slot a [`Law::Slot`] reads its number out of.
///
/// Opaque on purpose: which table the slot indexes is the frame plane's
/// business (alto design §3), and the law language only has to be able to
/// NAME one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(pub u32);

impl fmt::Display for SlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot[{}]", self.0)
    }
}

/// Where in a recorded launch a law lives.
///
/// The union of the two shells' places, which agree about more than their
/// spellings suggested:
///
/// ```text
/// this          Metal (was abi::At)   CUDA (was map::Component)
/// Entry         — (an ARM switch)     Func
/// Grid(k)       Lane(k)               Grid(k)
/// Block(k)      Group(k)              Block(k)
/// Shared        —                     Smem
/// Arg{at,word}  Arg(at)               Arg{at,word}
/// Shape         —                     Shape
/// ```
///
/// **THE TWO GRIDS ARE NOT THE SAME NUMBER AND THE PLACE STILL IS.** Metal's
/// grid axis is TOTAL THREADS (`MTLSize` of a dispatch) and CUDA's is BLOCKS;
/// what makes them one variant is that both are "grid axis `k` of this
/// launch", which is what a law is written about and what a rebind writes.
/// The units belong to the recorder, and each recorder's laws are only ever
/// evaluated against its own launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum At {
    /// The entrypoint itself — an arm switch, in the census's language.
    Entry,
    /// A grid axis, `0..3`.
    Grid(u8),
    /// A block (Metal: threadgroup) axis, `0..3`.
    Block(u8),
    /// Dynamic shared memory, in bytes.
    Shared,
    /// The `word`-th aligned eight-byte word of argument `at`.
    ///
    /// A scalar or a pointer is one word; a by-value block is as many as it
    /// is wide, and naming the WORD is what makes a moved pointer inside
    /// cutlass's 360-byte `Params` as reportable as a moved scalar of ours.
    /// A recorder that hands every argument over as a value (Metal's
    /// `Encode::fire`) states `word: 0` and means the whole argument.
    Arg {
        /// Which argument, by position in the launch's ABI block.
        at: u16,
        /// Which eight-byte word inside it.
        word: u16,
    },
    /// The argument block's own shape moved — a different count, offset or
    /// width for the same entrypoint. Not a value a rebind can carry: it
    /// means the two recordings disagree about what the launch IS.
    Shape,
}

impl fmt::Display for At {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            At::Entry => f.write_str("entry"),
            At::Grid(axis) => write!(f, "grid.{axis}"),
            At::Block(axis) => write!(f, "block.{axis}"),
            At::Shared => f.write_str("shared"),
            At::Arg { at, word: 0 } => write!(f, "arg[{at}]"),
            At::Arg { at, word } => write!(f, "arg[{at}].w{word}"),
            At::Shape => f.write_str("shape"),
        }
    }
}

/// What one component of one recorded launch is, as a function of the
/// composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Law {
    /// The same number in every composition and every size: encoded once, at
    /// load, and never rewritten.
    ///
    /// It is also what a recorder that OBSERVES rather than SOLVES states —
    /// CUDA's two-capture diff answers "this number, now" and that is a
    /// constant of one fire rather than a constant of the artifact. Which of
    /// the two a `Const` is, is the recorder's to say and not this type's.
    Const(i128),
    /// `base + Σ slope[k] · coord[k]` over the probe basis. The slopes that
    /// are zero are kept, so a law's shape says which directions it reads
    /// without a second table.
    Affine {
        /// The value at the origin of the fit's coordinates.
        base: i128,
        /// One slope per direction, in the recorder's own [`Axis`] order.
        slope: Vec<i128>,
    },
    /// `mul · ⌈(α·rows + β) / div⌉`, where `rows` is the window's own row
    /// count.
    ///
    /// **THE TILING LAW, AND IT READS ROWS RATHER THAN COORDINATES.** Every
    /// instance of it in the catalog is a `div_ceil` an entry writes over the
    /// extent it was handed — `attn::tiled_grid`'s `rows.div_ceil(SDPA_TILE)`
    /// and `linear::gemm::tile_grid`'s `div_ceil(rows, TILE_M) · 2` — so the
    /// numerator is affine in one number and the form says which.
    Ceil {
        /// The scale outside the ceiling: `TILE_GROUP[1]` and its kin.
        mul: i128,
        /// The numerator's slope over the window's rows.
        alpha: i128,
        /// The numerator's offset.
        beta: i128,
        /// The tile.
        div: i128,
    },
    /// The number is not a function of the coordinates at all: it is READ,
    /// per fire, out of the named descriptor slot.
    ///
    /// The one form no fit produces. A recorder states it when the artifact
    /// already carries the number as data — which is Article 5's sanctioned
    /// channel and the reason this is a law rather than a refusal.
    Slot(SlotId),
}

impl Law {
    /// This law at one point of the descriptor's space, with the window's
    /// rows the tiling form divides.
    ///
    /// `None` for [`Law::Slot`], whose value the coordinates do not contain —
    /// use [`Law::at_in`] with the fire's descriptor slots in hand.
    #[must_use]
    pub fn at(&self, coords: &[i128], rows: i128) -> Option<i128> {
        self.at_in(coords, rows, &[])
    }

    /// This law at one point, with the fire's descriptor slots to read a
    /// [`Law::Slot`] out of.
    #[must_use]
    pub fn at_in(&self, coords: &[i128], rows: i128, slots: &[i128]) -> Option<i128> {
        match self {
            Law::Const(v) => Some(*v),
            Law::Affine { base, slope } => Some(
                slope
                    .iter()
                    .zip(coords)
                    .fold(*base, |sum, (b, x)| sum + b * x),
            ),
            Law::Ceil {
                mul,
                alpha,
                beta,
                div,
            } => {
                let numerator = alpha * rows + beta;
                Some(
                    mul * numerator.div_euclid(*div)
                        + mul * i128::from(numerator.rem_euclid(*div) != 0),
                )
            }
            Law::Slot(SlotId(id)) => slots.get(*id as usize).copied(),
        }
    }

    /// Whether the number moves at all.
    #[must_use]
    pub fn varies(&self) -> bool {
        !matches!(self, Law::Const(_))
    }

    /// Which axes this law reads. A tiling law reads whatever the window's
    /// rows read, which is the window-rows law's business and not this one's;
    /// a slot law reads no axis at all.
    #[must_use]
    pub fn reads(&self) -> Vec<usize> {
        match self {
            Law::Const(_) | Law::Ceil { .. } | Law::Slot(_) => Vec::new(),
            Law::Affine { slope, .. } => slope
                .iter()
                .enumerate()
                .filter_map(|(k, b)| (*b != 0).then_some(k))
                .collect(),
        }
    }

    /// How this law names itself in a census.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Law::Const(_) => "const",
            Law::Affine { .. } => "affine",
            Law::Ceil { .. } => "ceil",
            Law::Slot(_) => "slot",
        }
    }
}

impl fmt::Display for Law {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Law::Const(v) => write!(f, "{v}"),
            Law::Affine { base, slope } => {
                write!(f, "{base}")?;
                for (k, b) in slope.iter().enumerate() {
                    if *b != 0 {
                        write!(f, " + {b}·x{k}")?;
                    }
                }
                Ok(())
            }
            Law::Ceil {
                mul,
                alpha,
                beta,
                div,
            } => {
                if *mul == 1 {
                    write!(f, "ceil(({alpha}·rows + {beta}) / {div})")
                } else {
                    write!(f, "{mul}·ceil(({alpha}·rows + {beta}) / {div})")
                }
            }
            Law::Slot(id) => write!(f, "{id}"),
        }
    }
}

/// One component of one recorded launch: which launch, which of its numbers,
/// and what that number is.
///
/// **THE UNIT BOTH PLANES REPORT IN.** Metal fits one per moving argument of
/// one ICB slot's arm; CUDA states one per eight-byte word two captures of
/// one graph node disagree about. The `node` index is the recorder's own
/// canonical order — a slot in Metal's walk order, a node in CUDA's
/// `(depth, symbol, index)` order — and means nothing outside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    /// Which recorded launch, in the recorder's canonical order.
    pub node: u32,
    /// Which of its numbers.
    pub at: At,
    /// What that number is, as a function of the composition.
    pub law: Law,
}

impl Component {
    /// A component, stated.
    #[must_use]
    pub fn new(node: u32, at: At, law: Law) -> Component {
        Component { node, at, law }
    }

    /// This component restated for one fire — the [`Patch`] a rebind writes.
    ///
    /// `None` exactly where [`Law::at`] is: a [`Law::Slot`] needs the fire's
    /// descriptor, which is [`Component::patch_in`]'s argument.
    #[must_use]
    pub fn patch(&self, coords: &[i128], rows: i128) -> Option<Patch> {
        self.patch_in(coords, rows, &[])
    }

    /// This component restated for one fire, with the fire's descriptor slots
    /// in hand.
    #[must_use]
    pub fn patch_in(&self, coords: &[i128], rows: i128, slots: &[i128]) -> Option<Patch> {
        Some(Patch {
            node: self.node,
            at: self.at,
            value: self.law.at_in(coords, rows, slots)?,
        })
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}.{} = {}", self.node, self.at, self.law)
    }
}

/// One component of one launch, restated for one fire: the concrete number.
///
/// **A PATCH IS DERIVED, NEVER DECIDED.** It is what a [`Component`]'s law
/// evaluates to at one composition and nothing else — no handle, no policy,
/// no decision about whether the write is worth making. Which exec, which
/// encoder and whether to write at all stay the shell's, which is why this
/// type carries no device anything and why both planes can own a richer
/// per-launch restatement beside it (`engine_cuda::device::map::Patch` is the
/// whole node, handles included, because `cudaGraphExecKernelNodeSetParams`
/// restates a node in full).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Patch {
    /// Which recorded launch, in the recorder's canonical order.
    pub node: u32,
    /// Which of its numbers.
    pub at: At,
    /// What to write there.
    pub value: i128,
}

impl fmt::Display for Patch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}.{} := {}", self.node, self.at, self.value)
    }
}

/// Why a recorder cannot state a law for a component.
///
/// **A REFUSAL IS A DELIVERABLE.** Both planes' own docs say so in the same
/// words — Metal's "the census is the deliverable even if it is a refusal
/// list", CUDA's "it does not fault, it does not diverge, it returns slightly
/// wrong numbers forever" — so the reasons are named, tallied and printed
/// rather than folded into a fallback nobody counts. This enum is the reason;
/// the sentence that goes with it is [`Refusal::why`], and each shell keeps
/// whatever extra it can say (a symbol, a depth, a class size) in its own
/// refusal type.
///
/// ```text
/// this          Metal (Fault::*)     CUDA (map::Refused::*)
/// Opaque        an argument that     Opaque — the parameter block was never
///               carries a number at  readable, and the driver call restates
///               one sample and none  a node in full, so unreadable is
///               at another           unwritable
/// Ambiguous     an arm switch that   Ambiguous — a same-depth same-symbol
///               interleaves, or is   class the canonical order guessed at,
///               not bracketed to a   whose two captures disagree
///               row
/// Unaffine      Unaffine — no law    —
///               in the language
///               predicts it
/// Unstructured  Unstructured — two   NotSameTopology is the ORDINARY answer
///               probes did not walk  and not this; this is a segment count
///               one template         that moved under an aligned topology
/// Unwritable    —                    a binding, a prebind or a twin
///                                    instantiation the driver refused
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Refuse {
    /// The component could not be READ, so it cannot be written either.
    Opaque,
    /// The two recordings cannot be ALIGNED truthfully: some launch of one is
    /// only a guess about which launch of the other it is.
    ///
    /// Aligning by guess can hand launch A launch B's buffer, and the mistake
    /// COMPUTES — no fault, no divergence, slightly wrong numbers forever.
    Ambiguous,
    /// The component moved and no law in the language predicts it.
    Unaffine,
    /// The two recordings are not one template at all — a different number of
    /// launches, or a different argument arity or kind at one.
    Unstructured,
    /// The restatement itself was refused: the recorder could read the
    /// components and state the laws, and the device would not take them.
    ///
    /// Not a fact about the law language, and named inside it anyway. A tally
    /// of "why is this artifact not rebinding" with nowhere to put a refused
    /// `cudaGraphExecKernelNodeSetParams` is a tally with a silent fallback in
    /// it, which is the one thing both planes' docs refuse. The CUDA fold
    /// plane's mid-list refusals and its twin instantiation are the live
    /// instances.
    Unwritable,
}

impl Refuse {
    /// How this refusal names itself in a tally.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Refuse::Opaque => "opaque",
            Refuse::Ambiguous => "ambiguous",
            Refuse::Unaffine => "unaffine",
            Refuse::Unstructured => "unstructured",
            Refuse::Unwritable => "unwritable",
        }
    }
}

impl fmt::Display for Refuse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind())
    }
}

/// One refusal: the reason, and the sentence that says which component and
/// why.
///
/// The sentence is not decoration. Both planes' refusals are read by an
/// operator who has to find the launch in the MODEL rather than in the graph,
/// so the reason alone was never enough and neither shell ever shipped it
/// alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// Which refusal this is.
    pub reason: Refuse,
    /// What to tell the reader.
    pub why: String,
}

impl Refusal {
    /// A refusal, spelled out.
    #[must_use]
    pub fn new(reason: Refuse, why: impl Into<String>) -> Refusal {
        Refusal {
            reason,
            why: why.into(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.why)
    }
}
