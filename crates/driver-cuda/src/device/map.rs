//! **THE NODE MAP, PROMOTED** (`.wiki/palo/cuda-abi.md` §7, step 3): the walk
//! stops being a census and becomes a coordinate system.
//!
//! [`nodes::walk`](crate::device::nodes::walk) reads a captured graph and
//! hands back what the driver said. This module turns one such reading into
//! the table the fire path rebinds against: per node the live
//! `CUgraphNode_t`, its entrypoint, its `(offset, size)` parameter cells and
//! the bytes the capture froze into them, its grid, block and shared memory —
//! all keyed by an index that TWO captures of the same walk agree on. Step 4
//! keeps one of these beside each exec and applies [`diff`]'s [`Patch`]es
//! with `cudaGraphExecKernelNodeSetParams`.
//!
//! **This module writes nothing.** It reads two graphs and says what moved;
//! the exec it would be written to is `record.rs`'s, and so is the decision
//! to write at all. The split is not fastidiousness: what moved is derivable
//! from the two graphs ALONE and is therefore testable without an exec, a
//! fire, or a checkpoint, while the write needs a policy (which exec, which
//! fire, whether the pass is worth its 68 µs) that no pair of graphs
//! contains.
//!
//! # The fingerprint hashes a multiset, and the reason is the tiebreak
//!
//! `cuGraphGetNodes` returns nodes in an order the driver never specified, so
//! the walk canonicalises: longest-path depth, then symbol, then the
//! enumeration index. The first two keys are facts about the GRAPH. The third
//! is a fact about the ENUMERATION, and the driver has not promised to
//! enumerate the same way twice — which means that for a class of same-depth
//! same-symbol nodes the canonical index is a coin the driver flips. A
//! fingerprint taken over the canonical SEQUENCE would therefore be a
//! fingerprint of that coin: two captures of one walk could disagree about
//! their own identity, and the disagreement would present as a permanent
//! cache miss nobody could explain.
//!
//! So [`Topology`] hashes only material a permutation inside a class cannot
//! move:
//!
//! ```text
//! nodes  the multiset of (depth, kind, symbol)
//! edges  the multiset of ((depth, kind, symbol) -> (depth, kind, symbol))
//! ```
//!
//! and two counts beside it. Both multisets are hashed in the canonical order
//! precisely because that order sorts them: equal multisets give equal
//! sequences of `(depth, symbol, kind)`, which is the alignment [`diff`]
//! needs and is why the fingerprint is worth taking rather than merely
//! comparing lengths.
//!
//! What the fingerprint deliberately does NOT hash: any argument, any grid,
//! any pointer. Those are exactly what a rebind exists to move, and a
//! fingerprint that noticed them would call every fire a new topology — which
//! is today's exact key wearing a hash's costume.
//!
//! # The ambiguity census is a result, not a warning
//!
//! The probe counted 78 same-depth same-symbol nodes on the mixed
//! composition, and that number is the whole risk in this module. Within such
//! a class the canonical index is a guess, so a [`diff`] that aligned two
//! captures by index would be aligning two guesses. Where the guesses cannot
//! matter — every member of the class carries byte-identical arguments in
//! both captures — the alignment is unobservable and the class passes with no
//! patch. Where they can, this module refuses by name
//! ([`Refused::Ambiguous`]): **aligning them by guess can hand node A node
//! B's buffer, and the mistake computes.** It does not fault, it does not
//! diverge, it returns slightly wrong numbers forever, which is the failure
//! mode the whole `palo` graph plane is built to keep structural.
//!
//! Note what is refused along with it: a class whose two captures hold the
//! SAME SET of arguments in a different index order. It looks like a swap and
//! a swap looks repairable, but a permutation that is right about the bytes
//! can still be wrong about the edges — the two nodes have different
//! successors, and only the driver knows which handle it gave to which. A
//! guess that is right by luck is not a mechanism.

use core::ffi::c_void;
use core::fmt;

use crate::device::graph::Graph;
use crate::device::nodes::{self, Node, Param, Walked};
use crate::error::Result;

// ─────────────────────────────────────────────────────────────────────────
// The fingerprint

/// The order-invariant identity of a captured graph's SHAPE.
///
/// Equality is what [`diff`] requires before it aligns anything, and the
/// three fields are compared together on purpose: the hash is 64 bits of
/// FNV-1a over a multiset, and a collision that also matched both counts
/// would still be caught by the per-node `(depth, symbol, kind)` check
/// [`diff`] runs before it trusts an index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Topology {
    /// FNV-1a over the node multiset and the edge multiset.
    ///
    /// FNV rather than [`std::hash::DefaultHasher`] because that one is
    /// SipHash with a per-process seed in the general case and its output is
    /// explicitly not a stable value; this number is meant to be printable in
    /// a log and comparable to the same graph's number in another process.
    pub hash: u64,
    /// How many nodes the graph holds, kernel and otherwise.
    pub nodes: usize,
    /// How many dependency edges the walk could place.
    pub links: usize,
}

impl Topology {
    /// Fingerprint one walked graph.
    #[must_use]
    pub fn of(walked: &Walked) -> Topology {
        let mut hash = FNV_OFFSET;
        // The nodes, in the canonical order — which IS the sorted multiset,
        // because the sort's first two keys are the material being hashed.
        for node in &walked.nodes {
            fold(&mut hash, &node.depth.to_le_bytes());
            fold(&mut hash, &node.kind.to_le_bytes());
            fold(&mut hash, node.symbol.as_bytes());
            fold(&mut hash, b"|");
        }
        // The edges, named by what their endpoints ARE rather than by where
        // the enumeration put them, and sorted so the driver's edge order is
        // as unobservable as its node order.
        let mut edges: Vec<(usize, u32, &str, usize, u32, &str)> = walked
            .links
            .iter()
            .filter_map(|(from, to)| {
                let a = walked.nodes.get(*from)?;
                let b = walked.nodes.get(*to)?;
                Some((
                    a.depth,
                    a.kind,
                    a.symbol.as_str(),
                    b.depth,
                    b.kind,
                    b.symbol.as_str(),
                ))
            })
            .collect();
        edges.sort_unstable();
        for (ad, ak, asym, bd, bk, bsym) in &edges {
            fold(&mut hash, &ad.to_le_bytes());
            fold(&mut hash, &ak.to_le_bytes());
            fold(&mut hash, asym.as_bytes());
            fold(&mut hash, b">");
            fold(&mut hash, &bd.to_le_bytes());
            fold(&mut hash, &bk.to_le_bytes());
            fold(&mut hash, bsym.as_bytes());
            fold(&mut hash, b"|");
        }
        Topology {
            hash,
            nodes: walked.nodes.len(),
            links: edges.len(),
        }
    }
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{:016x} ({} nodes, {} edges)",
            self.hash, self.nodes, self.links
        )
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fold(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The census

/// One class of nodes the canonical order cannot tell apart: same depth, same
/// symbol, and therefore ordered by a tiebreak the driver never promised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ambiguous {
    /// The depth they share.
    pub depth: usize,
    /// The symbol they share.
    pub symbol: String,
    /// Their canonical indices, ascending — a run, since the order sorts by
    /// `(depth, symbol)` first.
    pub at: Vec<usize>,
}

/// What one [`NodeMap`] is, in numbers — the line a probe prints and a load
/// logs.
///
/// Printed, not pinned. The probe's own lesson: pin the rules, print the
/// catalogs — a census that was asserted would fail on the next driver
/// version for reasons that are nobody's bug.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Every node, kernel and otherwise.
    pub nodes: usize,
    /// The kernel nodes — the only ones a rebind can touch.
    pub kernels: usize,
    /// Kernel nodes whose parameter block was read word for word.
    pub readable: usize,
    /// Kernel nodes whose parameter block was not, and which therefore cannot
    /// be patched at all (see [`Refused::Opaque`]).
    pub opaque: usize,
    /// How many `(depth, symbol)` classes hold more than one node.
    pub classes: usize,
    /// How many nodes live in one of those classes.
    pub ambiguous: usize,
    /// The adjacent-pair count `nodes::walk` reports — kept so this census
    /// and the probe's census speak the same number.
    pub pairs: usize,
    /// Components a blanket rebind would rewrite: seven per kernel node plus
    /// one per eight-byte argument word.
    pub components: usize,
}

impl fmt::Display for Census {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} nodes ({} kernel, {} readable, {} opaque), {} components, \
             {} ambiguous in {} classes ({} adjacent pairs)",
            self.nodes,
            self.kernels,
            self.readable,
            self.opaque,
            self.components,
            self.ambiguous,
            self.classes,
            self.pairs,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The map

/// One captured graph, as a rebind coordinate system.
///
/// Built once per capture and held for the life of the exec instantiated from
/// the same graph. **It borrows nothing and owns no handle**: the
/// `CUgraphNode_t` in every [`Node`] belongs to the `cudaGraph_t` the walk
/// read, so the graph has to outlive the map for a [`Patch`] taken from it to
/// be applicable — which is the same rule `record.rs` already keeps for the
/// graph a kept exec was instantiated from.
#[derive(Clone, Debug)]
pub struct NodeMap {
    topology: Topology,
    nodes: Vec<Node>,
    classes: Vec<Ambiguous>,
    /// Per canonical index, which class of [`classes`](NodeMap::classes) it
    /// belongs to — the O(1) question `diff` asks per node.
    class_of: Vec<Option<usize>>,
    census: Census,
}

impl NodeMap {
    /// Walk `graph` and build its map.
    ///
    /// # Errors
    ///
    /// Whatever [`nodes::walk`] refuses: [`Fault::Runtimeless`] for a build
    /// with no runtime, [`Fault::Device`] for an enumeration the driver
    /// refused.
    ///
    /// [`Fault::Runtimeless`]: crate::error::Fault::Runtimeless
    /// [`Fault::Device`]: crate::error::Fault::Device
    pub fn of(graph: &Graph) -> Result<NodeMap> {
        Ok(NodeMap::from_walk(nodes::walk(graph)?))
    }

    /// Build the map from a walk somebody else took.
    ///
    /// Takes the walk by value: the map IS the walk plus the two indices
    /// derived from it, and a version that cloned would put a second copy of
    /// every argument block — ~52 KiB of by-value structs on the mixed
    /// composition — beside the first for no reader.
    #[must_use]
    pub fn from_walk(walked: Walked) -> NodeMap {
        let topology = Topology::of(&walked);
        let Walked {
            nodes, ambiguous, ..
        } = walked;

        // The classes: runs of equal `(depth, symbol)` in the canonical
        // order. They ARE runs — the order sorts by exactly those two keys
        // before it reaches the tiebreak — so one pass finds every class.
        let mut classes: Vec<Ambiguous> = Vec::new();
        let mut class_of: Vec<Option<usize>> = vec![None; nodes.len()];
        let mut at = 0usize;
        while at < nodes.len() {
            let mut end = at + 1;
            while end < nodes.len()
                && nodes[end].depth == nodes[at].depth
                && nodes[end].symbol == nodes[at].symbol
                && nodes[end].kind == nodes[at].kind
            {
                end += 1;
            }
            if end - at > 1 {
                let class = classes.len();
                for held in &mut class_of[at..end] {
                    *held = Some(class);
                }
                classes.push(Ambiguous {
                    depth: nodes[at].depth,
                    symbol: nodes[at].symbol.clone(),
                    at: (at..end).collect(),
                });
            }
            at = end;
        }

        let kernels = nodes.iter().filter(|node| node.kernel()).count();
        let readable = nodes
            .iter()
            .filter(|node| node.kernel() && node.opaque.is_none())
            .count();
        let components = nodes
            .iter()
            .filter(|node| node.kernel())
            .map(|node| 7 + node.params.iter().map(words).sum::<usize>())
            .sum();
        let census = Census {
            nodes: nodes.len(),
            kernels,
            readable,
            opaque: kernels - readable,
            classes: classes.len(),
            ambiguous: class_of.iter().filter(|held| held.is_some()).count(),
            pairs: ambiguous,
            components,
        };

        NodeMap {
            topology,
            nodes,
            classes,
            class_of,
            census,
        }
    }

    /// The shape this map is the map OF — [`diff`]'s precondition.
    #[must_use]
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// Every node, in the canonical order.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The node at a canonical index.
    #[must_use]
    pub fn node(&self, at: usize) -> Option<&Node> {
        self.nodes.get(at)
    }

    /// How many nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Did the graph hold nothing at all?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The ambiguity census, class by class.
    #[must_use]
    pub fn classes(&self) -> &[Ambiguous] {
        &self.classes
    }

    /// Is the node at this canonical index one the order guessed at?
    #[must_use]
    pub fn ambiguous(&self, at: usize) -> bool {
        self.class_of.get(at).copied().flatten().is_some()
    }

    /// The numbers, summarised.
    #[must_use]
    pub fn census(&self) -> Census {
        self.census
    }
}

/// How many eight-byte words a parameter spans — one for a scalar or a
/// pointer, as many as it is wide for a by-value block.
fn words(param: &Param) -> usize {
    param.bytes.len().div_ceil(8).max(1)
}

// ─────────────────────────────────────────────────────────────────────────
// What moved

/// One rewritable component of a kernel node — the granularity a patch names
/// what moved at, and the same vocabulary the descriptor-abi probe fits laws
/// over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Component {
    /// The entrypoint itself: an arm switch, in the census's language.
    Func,
    /// A grid axis, in blocks.
    Grid(usize),
    /// A block axis, in threads.
    Block(usize),
    /// Dynamic shared memory.
    Smem,
    /// The `word`-th aligned eight-byte word of parameter `at`. A scalar or a
    /// pointer is one word; a by-value block is as many as it is wide, and
    /// naming the WORD is what makes a moved pointer inside cutlass's
    /// 360-byte `Params` as reportable as a moved scalar of ours.
    Arg {
        /// Which parameter, by position in the ABI block.
        at: usize,
        /// Which eight-byte word inside it.
        word: usize,
    },
    /// The parameter block's own shape moved — a different count, offset or
    /// width for the same symbol at the same depth. Not a value a rebind can
    /// carry: it means the two captures disagree about what the kernel IS.
    Shape,
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Component::Func => write!(f, "func"),
            Component::Grid(axis) => write!(f, "grid.{axis}"),
            Component::Block(axis) => write!(f, "block.{axis}"),
            Component::Smem => write!(f, "smem"),
            Component::Arg { at, word: 0 } => write!(f, "arg[{at}]"),
            Component::Arg { at, word } => write!(f, "arg[{at}].w{word}"),
            Component::Shape => write!(f, "shape"),
        }
    }
}

/// One node's worth of rebind: everything
/// `cudaGraphExecKernelNodeSetParams` wants, plus the reason it is here.
///
/// **THE HANDLE IS THE OLD MAP'S AND THE VALUES ARE THE NEW WALK'S**, which
/// is the whole point: the exec was instantiated from the graph the map was
/// built from, so that is the node the driver will accept, and the numbers
/// written into it come from the walk that says what this fire wants. A patch
/// is valid only while both graphs are alive — the old one owns
/// [`node`](Patch::node), the new one's module owns [`entry`](Patch::entry).
#[derive(Clone, Debug)]
pub struct Patch {
    /// Which node, in the map's canonical order.
    pub at: usize,
    /// The live `CUgraphNode_t` of the node in the graph the exec came from.
    pub node: *mut c_void,
    /// The `CUfunction` the rebound node should run.
    pub entry: *mut c_void,
    /// Its address, for logs and comparisons.
    pub func: u64,
    /// The grid to launch it at.
    pub grid: [u32; 3],
    /// The block shape.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
    /// The WHOLE parameter block, not only the cells that moved: the driver
    /// call takes all of it, so a patch that carried a delta would make its
    /// consumer reconstruct the rest and get one wrong on the day a component
    /// this module does not compare starts moving.
    pub params: Vec<Param>,
    /// What differed — reported, so a caller can say WHY it rebound.
    pub moved: Vec<Component>,
}

impl Patch {
    /// The parameter block as the ABI lays it out: one buffer, each parameter
    /// at its own `offset`.
    ///
    /// The `CU_LAUNCH_PARAM_BUFFER_POINTER` form of the call takes exactly
    /// this, which is what makes application mechanical — no per-kernel
    /// pointer array to keep alive, one `Vec<u8>` per node.
    #[must_use]
    pub fn block(&self) -> Vec<u8> {
        let len = self
            .params
            .iter()
            .map(|param| param.offset + param.size)
            .max()
            .unwrap_or(0);
        let mut block = vec![0u8; len];
        for param in &self.params {
            let take = param.size.min(param.bytes.len());
            block[param.offset..param.offset + take].copy_from_slice(&param.bytes[..take]);
        }
        block
    }
}

/// Write patches into an instantiated exec: one
/// `cudaGraphExecKernelNodeSetParams` per patch, restating the node in full.
///
/// **THE ONE WRITE THIS MODULE OWNS, AND IT IS STILL POLICY-FREE**: which
/// exec, which patches and whether the pass is worth its microseconds stay
/// `record.rs`'s decisions — this is the mechanical half the module doc
/// promised step 4, kept beside [`Patch`] because the ABI packing
/// ([`Patch::block`]) and the driver call that consumes it are one fact.
///
/// The call takes the node handle of the graph the exec was instantiated
/// from ([`Patch::node`]) and the parameter block is handed over in the
/// `kernelParams` form — one host pointer per argument, each into the packed
/// block — which is the form the probe validated writes through (GRID ✓
/// ARG ✓ FUNC ✓, §1 of the design note). The node's existing params are read
/// first and mutated, so fields this table does not model (the v2 struct's
/// context) keep what instantiation gave them.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime; [`Fault::Device`]
/// naming the first node the driver refused. A refusal mid-list leaves the
/// exec partially written — the caller must treat that exec as unbound (the
/// fold drops it by name rather than launching it).
///
/// [`Fault::Runtimeless`]: crate::error::Fault::Runtimeless
/// [`Fault::Device`]: crate::error::Fault::Device
#[cfg(feature = "_cuda")]
pub fn apply(exec: &crate::device::graph::GraphExec, patches: &[Patch]) -> Result<()> {
    use cudarc::driver::sys as dr;

    let raw: dr::CUgraphExec = exec.raw().cast();
    for patch in patches {
        // The node's current statement, so the fields a patch does not carry
        // stay what they were.
        let mut params: dr::CUDA_KERNEL_NODE_PARAMS = unsafe { core::mem::zeroed() };
        let read = unsafe { dr::cuGraphKernelNodeGetParams_v2(patch.node.cast(), &raw mut params) };
        if read != dr::CUresult::CUDA_SUCCESS {
            return Err(crate::error::Fault::Device {
                call: "cuGraphKernelNodeGetParams_v2",
                code: read as i32,
            });
        }
        params.func = patch.entry.cast();
        params.gridDimX = patch.grid[0];
        params.gridDimY = patch.grid[1];
        params.gridDimZ = patch.grid[2];
        params.blockDimX = patch.block[0];
        params.blockDimY = patch.block[1];
        params.blockDimZ = patch.block[2];
        params.sharedMemBytes = patch.smem;
        // The block, then one pointer per argument into it — alive for the
        // duration of the call, which is all the driver needs: it copies.
        let block = patch.block();
        let mut cells: Vec<*mut c_void> = patch
            .params
            .iter()
            .map(|param| unsafe { block.as_ptr().add(param.offset) as *mut c_void })
            .collect();
        params.kernelParams = cells.as_mut_ptr();
        params.extra = core::ptr::null_mut();
        let wrote =
            unsafe { dr::cuGraphExecKernelNodeSetParams_v2(raw, patch.node.cast(), &raw const params) };
        if wrote != dr::CUresult::CUDA_SUCCESS {
            return Err(crate::error::Fault::Device {
                call: "cuGraphExecKernelNodeSetParams_v2",
                code: wrote as i32,
            });
        }
    }
    Ok(())
}

#[cfg(not(feature = "_cuda"))]
#[allow(unused_variables)]
pub fn apply(exec: &crate::device::graph::GraphExec, patches: &[Patch]) -> Result<()> {
    Err(crate::error::Fault::Runtimeless)
}

/// What two captures of one shape have to say to each other.
#[derive(Clone, Debug)]
pub enum Diff {
    /// The two walks are not the same graph, so no alignment between them
    /// exists.
    ///
    /// **NOT A REFUSAL.** A composition the exec cache has never seen is the
    /// ordinary case — it captures, instantiates and keys a new entry, which
    /// is what today's shell does for every shape. Naming it here keeps the
    /// caller from having to tell "this fire needs a new graph" apart from
    /// "this fire's graph cannot be trusted", which are answers with
    /// different consequences.
    NotSameTopology {
        /// What the map was built from.
        held: Topology,
        /// What the walk brought.
        brought: Topology,
    },
    /// Every node aligned, and here is what a rebind would have to write.
    Aligned {
        /// The nodes that moved, in canonical order.
        patches: Vec<Patch>,
        /// Kernel nodes whose every component was identical.
        unmoved: usize,
        /// Nodes inside an ambiguous class that were identical anyway — the
        /// places the guess could not matter (see the module docs).
        agreed: usize,
    },
}

/// Why a rebind cannot be derived from these two captures.
///
/// A refusal, not a diff: nothing here is repairable by trying harder, and
/// every variant names the symbol and the depth so the operator can find the
/// launch in the model rather than in the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// A class of same-depth same-symbol nodes whose arguments disagree
    /// between the two captures.
    ///
    /// Aligning them by guess can hand node A node B's buffer, and the
    /// mistake COMPUTES — no fault, no divergence, slightly wrong numbers
    /// forever. The fix is upstream: make the two launches distinguishable
    /// (a depth apart, or a symbol apart), or key the exec rather than
    /// rebinding it.
    Ambiguous {
        /// The depth the class shares.
        depth: usize,
        /// The symbol the class shares.
        symbol: String,
        /// The class's first canonical index.
        at: usize,
        /// How many nodes are in it.
        count: usize,
        /// How many of them differ between the two captures.
        differing: usize,
        /// The first component that differed, in canonical order.
        component: Component,
        /// How many OTHER refusals this diff also found — the scale, so a
        /// caller does not learn it one capture at a time.
        more: usize,
    },
    /// A node whose parameter block could not be read, and which moved.
    ///
    /// The driver call restates a node's parameters in full, so a block that
    /// was never readable cannot be written either: patching its grid alone
    /// would mean handing the driver argument cells this module never had.
    /// A node like this that did NOT move is fine and is not reported — the
    /// refusal is against rewriting it, not against its existence.
    Opaque {
        /// Its canonical index.
        at: usize,
        /// Its depth.
        depth: usize,
        /// Its symbol.
        symbol: String,
        /// What `nodes::walk` said when it could not read the block.
        why: &'static str,
        /// The first component that differed anyway.
        component: Component,
        /// How many other refusals this diff found.
        more: usize,
    },
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Ambiguous {
                depth,
                symbol,
                at,
                count,
                differing,
                component,
                more,
            } => write!(
                f,
                "{count} nodes at depth {depth} run `{symbol}` and the canonical order \
                 tells them apart by an enumeration index the driver never promised \
                 twice; {differing} of them differ between the two captures (first at \
                 node {at}, component {component}). Aligning them by guess can hand one \
                 node the other's buffer and the mistake COMPUTES, so no patch is \
                 derived{}",
                plural(*more),
            ),
            Refused::Opaque {
                at,
                depth,
                symbol,
                why,
                component,
                more,
            } => write!(
                f,
                "node {at} at depth {depth} runs `{symbol}`, its {component} moved \
                 between the two captures, and its parameter block was never readable \
                 ({why}); `cudaGraphExecKernelNodeSetParams` restates a node in full, \
                 so a component of it cannot be written on its own{}",
                plural(*more),
            ),
        }
    }
}

impl std::error::Error for Refused {}

fn plural(more: usize) -> String {
    match more {
        0 => String::new(),
        1 => " (and 1 more refusal in this diff)".to_string(),
        n => format!(" (and {n} more refusals in this diff)"),
    }
}

/// Align a fresh walk against a map, and say what a rebind would write.
///
/// The three answers are deliberately three types and not two: a topology
/// that does not match is [`Diff::NotSameTopology`] and is ORDINARY (capture
/// a new graph); a topology that matches produces [`Diff::Aligned`] and a
/// patch list; a topology that matches but cannot be aligned truthfully is
/// [`Refused`] and is an error. Nothing here applies a patch — see the module
/// docs for why the write lives in `record.rs`.
///
/// # Errors
///
/// [`Refused::Ambiguous`] when a same-depth same-symbol class disagrees
/// between the captures; [`Refused::Opaque`] when a node whose parameter
/// block was unreadable moved anyway.
#[allow(clippy::too_many_lines)]
pub fn diff(held: &NodeMap, brought: &Walked) -> core::result::Result<Diff, Refused> {
    let topology = Topology::of(brought);
    let mismatch = || Diff::NotSameTopology {
        held: held.topology,
        brought: topology,
    };
    if topology != held.topology || brought.nodes.len() != held.nodes.len() {
        return Ok(mismatch());
    }

    // Equal multisets give equal `(depth, kind, symbol)` sequences — the sort
    // is over exactly those keys. Checking it anyway is what keeps a 64-bit
    // collision from becoming an alignment: the check costs one comparison a
    // node and it is the only thing standing between a hash and a buffer.
    for (was, now) in held.nodes.iter().zip(&brought.nodes) {
        if was.depth != now.depth || was.kind != now.kind || was.symbol != now.symbol {
            return Ok(mismatch());
        }
    }

    let moved: Vec<Vec<Component>> = held
        .nodes
        .iter()
        .zip(&brought.nodes)
        .map(|(was, now)| if was.kernel() { moved_of(was, now) } else { Vec::new() })
        .collect();

    // Every refusal is collected before one is returned, so the sentence can
    // say how big the problem is rather than how early it was found.
    let mut refusals: Vec<Refused> = Vec::new();
    for class in &held.classes {
        let differing = class.at.iter().filter(|at| !moved[**at].is_empty()).count();
        if differing == 0 {
            continue;
        }
        let first = class
            .at
            .iter()
            .copied()
            .find(|at| !moved[*at].is_empty())
            .unwrap_or(class.at[0]);
        refusals.push(Refused::Ambiguous {
            depth: class.depth,
            symbol: class.symbol.clone(),
            at: first,
            count: class.at.len(),
            differing,
            component: moved[first][0],
            more: 0,
        });
    }
    for (at, node) in held.nodes.iter().enumerate() {
        if held.ambiguous(at) || moved[at].is_empty() {
            continue;
        }
        // A non-kernel node never reaches here: `moved_of` is only asked of
        // kernel nodes, so its list is empty and the line above skipped it.
        if let Some(why) = node.opaque.or(brought.nodes[at].opaque) {
            refusals.push(Refused::Opaque {
                at,
                depth: node.depth,
                symbol: node.symbol.clone(),
                why,
                component: moved[at][0],
                more: 0,
            });
        }
    }
    if !refusals.is_empty() {
        let more = refusals.len() - 1;
        let mut first = refusals.swap_remove(0);
        match &mut first {
            Refused::Ambiguous { more: count, .. } | Refused::Opaque { more: count, .. } => {
                *count = more;
            }
        }
        return Err(first);
    }

    let mut patches = Vec::new();
    let mut unmoved = 0usize;
    let mut agreed = 0usize;
    for (at, node) in held.nodes.iter().enumerate() {
        if !node.kernel() {
            continue;
        }
        if moved[at].is_empty() {
            if held.ambiguous(at) {
                agreed += 1;
            } else {
                unmoved += 1;
            }
            continue;
        }
        let now = &brought.nodes[at];
        patches.push(Patch {
            at,
            node: node.node,
            entry: now.entry,
            func: now.func,
            grid: now.grid,
            block: now.block,
            smem: now.smem,
            params: now.params.clone(),
            moved: moved[at].clone(),
        });
    }

    Ok(Diff::Aligned {
        patches,
        unmoved,
        agreed,
    })
}

/// Every component of `was` that `now` disagrees with.
fn moved_of(was: &Node, now: &Node) -> Vec<Component> {
    let mut moved = Vec::new();
    if was.func != now.func {
        moved.push(Component::Func);
    }
    for axis in 0..3 {
        if was.grid[axis] != now.grid[axis] {
            moved.push(Component::Grid(axis));
        }
    }
    for axis in 0..3 {
        if was.block[axis] != now.block[axis] {
            moved.push(Component::Block(axis));
        }
    }
    if was.smem != now.smem {
        moved.push(Component::Smem);
    }
    if was.params.len() != now.params.len() {
        moved.push(Component::Shape);
        return moved;
    }
    for (at, (old, new)) in was.params.iter().zip(&now.params).enumerate() {
        if old.offset != new.offset || old.size != new.size || old.bytes.len() != new.bytes.len() {
            moved.push(Component::Shape);
            continue;
        }
        for word in 0..words(old) {
            let from = word * 8;
            let to = (from + 8).min(old.bytes.len());
            if from >= old.bytes.len() {
                break;
            }
            if old.bytes[from..to] != new.bytes[from..to] {
                moved.push(Component::Arg { at, word });
            }
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel node, spelled the way a walk would have spelled it.
    fn node(at: usize, depth: usize, symbol: &str, func: u64, args: &[u64]) -> Node {
        Node {
            at,
            depth,
            kind: 0,
            symbol: symbol.to_string(),
            func,
            node: core::ptr::without_provenance_mut(0x1000 + at),
            entry: core::ptr::without_provenance_mut(0x2000 + func as usize),
            grid: [1, 1, 1],
            block: [32, 1, 1],
            smem: 0,
            params: args
                .iter()
                .enumerate()
                .map(|(cell, value)| Param {
                    offset: cell * 8,
                    size: 8,
                    bytes: value.to_le_bytes().to_vec(),
                })
                .collect(),
            opaque: None,
        }
    }

    /// A chain: node `at` depends on node `at - 1`.
    fn chain(nodes: Vec<Node>) -> Walked {
        let links = (1..nodes.len()).map(|at| (at - 1, at)).collect();
        Walked {
            ambiguous: 0,
            edges: nodes.len().saturating_sub(1),
            links,
            nodes,
        }
    }

    fn aligned(diff: Diff) -> (Vec<Patch>, usize, usize) {
        match diff {
            Diff::Aligned {
                patches,
                unmoved,
                agreed,
            } => (patches, unmoved, agreed),
            Diff::NotSameTopology { held, brought } => {
                panic!("expected an alignment, got {held} against {brought}")
            }
        }
    }

    #[test]
    fn two_readings_of_one_graph_fingerprint_the_same() {
        let a = chain(vec![
            node(0, 0, "load", 1, &[0xdead_0000]),
            node(1, 1, "gemm", 2, &[0xdead_0000, 8]),
        ]);
        let b = chain(vec![
            node(0, 0, "load", 1, &[0xbeef_0000]),
            node(1, 1, "gemm", 2, &[0xbeef_0000, 9]),
        ]);
        assert_eq!(
            Topology::of(&a),
            Topology::of(&b),
            "arguments are what a rebind moves; a fingerprint that saw them \
             would call every fire a new shape"
        );
    }

    #[test]
    fn a_permuted_ambiguous_class_does_not_move_the_fingerprint() {
        // Two same-depth same-symbol nodes, enumerated in either order.
        let mut one = chain(vec![
            node(0, 0, "fork", 1, &[0]),
            node(1, 1, "leaf", 2, &[7]),
            node(2, 1, "leaf", 2, &[9]),
        ]);
        one.links = vec![(0, 1), (0, 2)];
        let mut other = chain(vec![
            node(0, 0, "fork", 1, &[0]),
            node(1, 1, "leaf", 2, &[9]),
            node(2, 1, "leaf", 2, &[7]),
        ]);
        other.links = vec![(0, 2), (0, 1)];
        assert_eq!(Topology::of(&one), Topology::of(&other));
    }

    #[test]
    fn one_scalar_that_moved_is_the_only_component_the_patch_names() {
        let held = NodeMap::from_walk(chain(vec![
            node(0, 0, "load", 1, &[0xdead_0000]),
            node(1, 1, "gemm", 2, &[0xdead_0000, 8]),
        ]));
        let brought = chain(vec![
            node(0, 0, "load", 1, &[0xdead_0000]),
            node(1, 1, "gemm", 2, &[0xdead_0000, 9]),
        ]);
        let (patches, unmoved, agreed) = aligned(diff(&held, &brought).expect("no refusal"));
        assert_eq!(patches.len(), 1, "one node moved");
        assert_eq!(patches[0].at, 1);
        assert_eq!(patches[0].moved, vec![Component::Arg { at: 1, word: 0 }]);
        assert_eq!(patches[0].params[1].word(), Some(9), "the NEW value rides");
        assert_eq!((unmoved, agreed), (1, 0));
    }

    #[test]
    fn a_by_value_block_names_the_word_that_moved_and_not_the_block() {
        let mut wide = node(0, 0, "cutlass", 1, &[]);
        wide.params = vec![Param {
            offset: 0,
            size: 24,
            bytes: (0u64..3).flat_map(u64::to_le_bytes).collect(),
        }];
        let held = NodeMap::from_walk(chain(vec![wide.clone()]));
        let mut moved = wide;
        moved.params[0].bytes[16..24].copy_from_slice(&99u64.to_le_bytes());
        let (patches, ..) = aligned(diff(&held, &chain(vec![moved])).expect("no refusal"));
        assert_eq!(patches[0].moved, vec![Component::Arg { at: 0, word: 2 }]);
    }

    #[test]
    fn a_walk_with_a_node_more_is_not_the_same_topology() {
        let held = NodeMap::from_walk(chain(vec![
            node(0, 0, "load", 1, &[1]),
            node(1, 1, "gemm", 2, &[1]),
        ]));
        let brought = chain(vec![
            node(0, 0, "load", 1, &[1]),
            node(1, 1, "gemm", 2, &[1]),
            node(2, 2, "store", 3, &[1]),
        ]);
        assert!(matches!(
            diff(&held, &brought).expect("a different shape is not a refusal"),
            Diff::NotSameTopology { .. }
        ));
    }

    #[test]
    fn a_walk_that_swapped_a_symbol_is_not_the_same_topology() {
        let held = NodeMap::from_walk(chain(vec![
            node(0, 0, "load", 1, &[1]),
            node(1, 1, "cublas", 2, &[1]),
        ]));
        let brought = chain(vec![
            node(0, 0, "load", 1, &[1]),
            node(1, 1, "cutlass", 3, &[1]),
        ]);
        assert!(matches!(
            diff(&held, &brought).expect("an arm switch is not a refusal"),
            Diff::NotSameTopology { .. }
        ));
    }

    #[test]
    fn an_ambiguous_pair_that_agrees_byte_for_byte_needs_no_patch() {
        let mut walk = chain(vec![
            node(0, 0, "fork", 1, &[0]),
            node(1, 1, "leaf", 2, &[7]),
            node(2, 1, "leaf", 2, &[7]),
        ]);
        walk.links = vec![(0, 1), (0, 2)];
        let held = NodeMap::from_walk(walk.clone());
        assert_eq!(held.census().ambiguous, 2, "the pair is counted");
        assert_eq!(held.classes().len(), 1);
        let (patches, unmoved, agreed) = aligned(diff(&held, &walk).expect("no refusal"));
        assert!(patches.is_empty());
        assert_eq!((unmoved, agreed), (1, 2));
    }

    #[test]
    fn an_ambiguous_pair_whose_arguments_differ_refuses_by_name() {
        let mut walk = chain(vec![
            node(0, 0, "fork", 1, &[0]),
            node(1, 1, "leaf", 2, &[7]),
            node(2, 1, "leaf", 2, &[9]),
        ]);
        walk.links = vec![(0, 1), (0, 2)];
        let held = NodeMap::from_walk(walk.clone());
        let mut brought = walk;
        brought.nodes[2].params[0].bytes = 11u64.to_le_bytes().to_vec();

        let refused = diff(&held, &brought).expect_err("the guess is refused");
        let Refused::Ambiguous {
            depth,
            symbol,
            count,
            differing,
            more,
            ..
        } = &refused
        else {
            panic!("the ambiguity is what refused, not {refused}")
        };
        assert_eq!((*depth, symbol.as_str(), *count, *differing), (1, "leaf", 2, 1));
        assert_eq!(*more, 0);
        assert!(
            format!("{refused}").contains("the mistake COMPUTES"),
            "the refusal says why: {refused}"
        );
    }

    #[test]
    fn a_node_whose_block_was_never_read_refuses_to_be_rewritten() {
        let mut blind = node(0, 0, "opaque", 1, &[1]);
        blind.opaque = Some("a kernelParams cell is null");
        let held = NodeMap::from_walk(chain(vec![blind.clone()]));
        let mut brought = blind;
        brought.grid = [4, 1, 1];

        let refused = diff(&held, &chain(vec![brought])).expect_err("unreadable is unwritable");
        assert!(
            matches!(refused, Refused::Opaque { component: Component::Grid(0), .. }),
            "{refused}"
        );
    }

    #[test]
    fn a_patch_packs_its_parameters_where_the_abi_puts_them() {
        let held = NodeMap::from_walk(chain(vec![node(0, 0, "gemm", 1, &[1, 2])]));
        let brought = chain(vec![node(0, 0, "gemm", 1, &[1, 3])]);
        let (patches, ..) = aligned(diff(&held, &brought).expect("no refusal"));
        let block = patches[0].block();
        assert_eq!(block.len(), 16);
        assert_eq!(&block[8..], &3u64.to_le_bytes());
    }
}
