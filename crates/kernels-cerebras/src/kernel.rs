//! Hand-written kernels as modules: what sits outside a kernels-ir `Region` (a mixer, a routed
//! bank, a tiled conv), each under the poem-ir group it implements and registered once in
//! [`KERNELS`]. A module states its CSL sources, its poem-ir ops and, by regime, how it runs.
//! Elementwise ops are not modules: they go through `fuse::chain` into a `Region`.

use crate::compile::emit::stream::HostArray;
use crate::compile::floor::Coord;
use crate::compile::floorplan::stream::KvColumn;
use crate::compile::stream::build::Program;
use crate::compile::stream::place::KvSched;
use crate::compile::stream::unit::{Cols, Joint, Lowering};
use poem_compiler::fuse::chain::Chain;
use poem_ir::Operands as _;

pub(crate) type R<T> = Result<T, String>;

/// A CSL source a module owns: its file name (written under `ops/`) and its text.
pub type Source = (&'static str, &'static str);

/// A hand-written kernel.
pub trait Kernel: Sync {
    /// Its name: `<group>.<kernel>` (`attn.column`).
    fn name(&self) -> &'static str;
    /// The poem-ir ops it implements, by name.
    fn ops(&self) -> &'static [&'static str];
    /// The CSL op modules it owns (`kernels/<family>`), each owned by one kernel.
    fn sources(&self) -> &'static [Source];
    /// How it lowers its ops into a unit's steps (the unit regime), where it runs there.
    fn unit(&self) -> Option<&dyn UnitKernel> {
        None
    }
    /// Its module at a unit's roots, where it runs there (the unit regime): a region step's
    /// interface, its own code.
    fn root(&self) -> Option<&dyn RootKernel> {
        None
    }
    /// Its op in the GEMM regime (a function of `gemm_pe.csl`), where it has one.
    fn gemm(&self) -> Option<&'static GemmForm> {
        None
    }
    /// Its op in the tiled regime (a function of `tiled_pe.csl`), where it has one.
    fn tiled(&self) -> Option<&'static TiledForm> {
        None
    }
    /// Its mixer on a unit's K / V column, where it runs one (the unit regime).
    fn column(&self) -> Option<&dyn Column> {
        None
    }
}

/// The line a GEMM-regime op's function rebinds an input queue along (`check::hazard::lines`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineAlong {
    /// A class's row: every PE of the row reads every other's piece.
    Row,
    /// A column: the op's pieces down a column.
    Column,
}

/// A kernel's op in the GEMM regime: a function of the GEMM PE's one image (`gemm_pe.csl`), and
/// what the hazard checks need of it (H1-H4).
pub struct GemmForm {
    /// Its function in the PE's code.
    pub func: &'static str,
    /// It, as the checks' messages name it ("a GEMM", "an attention").
    pub what: &'static str,
    /// The line it rebinds a queue along, where its function rebinds one.
    pub line: LineAlong,
    /// It runs for the classes of its op's mask alone (else for every class).
    pub masked: bool,
    /// Whether its code (`pe`) ends in a row-wide ordering point: no PE of the row leaves it
    /// before every other has reached its end. Else it leaves the row's bands drifted.
    pub orders_row: fn(pe: &str) -> bool,
    /// It, as the price's ops name it (`gemm`, `attention`).
    pub what_price: &'static str,
    /// Op `i`'s cycles on the busiest PE (past every op's own, `cost::gemm::OP`), its terms
    /// (`cost::gemm::OpCost::terms`) where it states them.
    pub price:
        fn(p: &crate::compile::stream::gemm::GemmProgram, i: usize, terms: &mut [f64; 5]) -> f64,
    /// Why it leaves the row drifted where it does not order it (H2's message).
    pub drift: &'static str,
    /// The same, with ops after it (H4's message).
    pub drift_after: &'static str,
}

/// A kernel's op in the tiled regime: a function of the tiled PE's image (`tiled_pe.csl`), its
/// price and what the hub-loop check needs of it.
pub struct TiledForm {
    /// It, as the price's ops name it (`conv`, `norm`, `up`, `attend`).
    pub what: &'static str,
    /// Op `i`'s cycles on the busiest PE (past every op's own, `cost::tiled::OP`).
    pub price: fn(p: &crate::compile::stream::tiled::TileProgram, i: usize) -> f64,
    /// Its sends towards the hub before its receive off the hub's broadcast, where it has any.
    pub hub: Option<HubSend>,
}

/// A tiled op's sends towards the hub before its receive (`check::hazard::hub_loops`).
pub struct HubSend {
    /// Its function in the PE's code.
    pub func: &'static str,
    /// It, as the check's message names it.
    pub what: &'static str,
    /// The words a PE sends towards the hub before its receive, for op `op` of `p`.
    pub words: fn(
        p: &crate::compile::stream::tiled::TileProgram,
        op: &crate::compile::stream::tiled::TOp,
    ) -> u32,
}

/// The tiled form of the kernel a tiled-regime op is (a conv's, a group norm's, an upsampling's,
/// a voxel attention's); none for the host's ins and outs and a printed region.
#[must_use]
pub fn tiled_form(op: &crate::compile::stream::tiled::TOp) -> Option<&'static TiledForm> {
    use crate::compile::stream::tiled::TOp;
    let k = match op {
        TOp::Conv { .. } => by_op("conv.conv3d"),
        TOp::Norm { .. } => by_op("conv.group_norm"),
        TOp::Up { .. } => by_op("conv.upsample"),
        TOp::Attend { .. } => by_op("attn.voxel"),
        TOp::In(_) | TOp::Out(_) | TOp::Region(_) => None,
    };
    k.and_then(|k| k.tiled())
}

/// A kernel in the unit regime: the steps its ops become in a unit, read node by node.
pub trait UnitKernel: Sync {
    /// Lowers node `at` (one of its ops) into `cx`'s steps; `Ok(false)` where it runs no step for
    /// it. A pattern over several nodes lowers at its first and puts the rest in `cx.skip`.
    fn lower(&self, cx: &mut Lowering, at: u32) -> R<bool>;
}

/// A kernel at a unit's roots: it lowers its node to a region step whose kernels-ir region
/// states only its interface, so the roots place, chunk, feed and price it as any region, and
/// prints the module the roots import (`in<k>` / `out<k>`, `init()`, `row()`) with its own state.
pub trait RootKernel: Sync {
    /// Its module for `place` (its chunk, its roots), over a unit's `rows` rows a step, a decode
    /// step's (its rows lanes, each a state slot and position) where `decode`.
    fn print(
        &self,
        trace: &poem_ir::Trace,
        chain: &Chain,
        place: &crate::compile::emit::region::Place,
        rows: u32,
        decode: bool,
    ) -> R<crate::compile::emit::region::Printed>;
    /// The arrays its module holds that the host writes at a root running chunk `r` of `c`
    /// columns: (the module's array, its words, what fills them).
    fn arrays(&self, trace: &poem_ir::Trace, chain: &Chain, r: u32, c: u32) -> Vec<RootArray>;
    /// Whether its module reads a decode step's rows' (lane slot, state slot, position): the
    /// root's `rowp` (passed in as `rowp`; the root a `rows` region of the state root's).
    fn rows(&self) -> bool;
}

/// An array of a root kernel's module the host writes.
pub struct RootArray {
    /// The module's array (`rg<k>.<name>`).
    pub name: &'static str,
    pub len: u32,
    pub fill: crate::compile::emit::stream::Fill,
}

/// The root kernel a region step's chain is (one node, a kernel's op), if it is one.
#[must_use]
pub fn root_of(trace: &poem_ir::Trace, chain: &Chain) -> Option<&'static dyn RootKernel> {
    let [n] = chain.nodes[..] else { return None };
    by_op(trace.nodes.get(n as usize)?.op.name()).and_then(|k| k.root())
}

/// The ops the unit regime lowers by a kernel of their own where the compiler has no phase for
/// them (`poem_compiler::spatial::layer::stream::lower_with`): a root kernel's, at the roots.
#[must_use]
pub fn claims(op: &poem_ir::Operation) -> Option<poem_compiler::spatial::layer::Claim> {
    by_op(op.name())
        .and_then(|k| k.root())
        .map(|_| poem_compiler::spatial::layer::Claim::Root)
}

/// What sizes a K / V column's PEs ([`Column::pya`]).
#[derive(Clone, Copy, Debug)]
pub struct ColumnAsk {
    /// Q heads a stored head (the column's attention's, else the reading's).
    pub q_per_kv: u32,
    pub head_dim: u32,
    /// The positions it holds: a decode step's cache, else the step's stored rows.
    pub positions: u32,
    /// A decode step's rows (its lanes), 1 in a stream reading.
    pub lanes: u32,
    pub split: crate::compile::stream::build::KvSplit,
}

/// A mixer on a unit's K / V column: its root takes each row's head (`[q | k | v]` and
/// [`Column::head_parts`]), its PEs below hold what the rows leave (keys and values, a state),
/// each row's output goes back. The column's placement and routes are the floorplan's.
pub trait Column: Sync {
    /// Whether it runs the column of the attention (or recurrence) `j`.
    fn claims(&self, j: &Joint) -> bool;
    /// Its kind of region in a decode program's state root.
    fn region(&self, j: &Joint) -> &'static str;
    /// Whether its rows are a cache in pages (a decode step's), not a lane slot's state.
    fn paged(&self) -> bool;
    /// Whether its stored heads may be replicated over columns, a replica its share of the q
    /// heads (the search's choice where a head root's tiles do not fit).
    fn replicates(&self) -> bool;
    /// The PEs a column takes.
    fn pya(&self, ask: &ColumnAsk) -> u32;
    /// Whether its PEs advance `rowp`'s positions at a decode step's end.
    fn advances(&self) -> bool {
        false
    }
    /// What a store sends past a head `h`'s `[q | k | v]`: (value, first column, width).
    fn head_parts(&self, _j: &Joint, _h: u32) -> Vec<(Cols, u32, u32)> {
        Vec::new()
    }
    /// Column `k`'s root file (name, text).
    fn root_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)>;
    /// Column `k`'s PEs' file (name, text).
    fn pe_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)>;
    /// The arrays the host writes into the column's root at `root`.
    fn root_arrays(&self, p: &Program, kc: &KvColumn, ks: &KvSched, root: Coord) -> Vec<HostArray>;
    /// The arrays the host writes into a PE of the column at `c` (past its config words).
    fn pe_arrays(&self, p: &Program, c: Coord) -> Vec<HostArray>;
}

/// The column that runs `j` (the reading's attention where a column has none of its own).
pub fn column(j: Option<&Joint>) -> R<&'static dyn Column> {
    let j = j.copied().unwrap_or(crate::attn::column::UNSTATED);
    KERNELS
        .iter()
        .filter_map(|k| k.column())
        .find(|c| c.claims(&j))
        .ok_or_else(|| "a K / V column no kernel runs".to_string())
}

/// Every kernel, in registry order: the order their sources are listed and digested in.
pub static KERNELS: &[&dyn Kernel] = &[
    &crate::attn::column::ATTENTION,
    &crate::ssm::state::STATE,
    &crate::linear::moe::MOE,
    &crate::linear::gemm::GEMM,
    &crate::conv::norm::GROUP_NORM,
    &crate::conv::conv3d::CONV3D,
    &crate::conv::up::UP,
    &crate::attn::voxel::VOXEL,
    &crate::ssm::short_conv::SHORT_CONV,
];

/// The GEMM form of the kernel a GEMM-regime op is (`linear.matmul`'s GEMM, an attention's);
/// none for the host's ins and outs and a printed region.
#[must_use]
pub fn gemm_form(op: &crate::compile::stream::gemm::GOp) -> Option<&'static GemmForm> {
    use crate::compile::stream::gemm::GOp;
    let k = match op {
        GOp::Gemm { .. } => by_op("linear.matmul"),
        GOp::Attend { .. } => by_op("attn.ragged"),
        _ => None,
    };
    k.and_then(|k| k.gemm())
}

/// The GEMM-regime forms of every kernel that has one.
pub fn gemm_forms() -> impl Iterator<Item = &'static GemmForm> {
    KERNELS.iter().filter_map(|k| k.gemm())
}

/// The kernel that implements the poem-ir op `op`, if one does.
#[must_use]
pub fn by_op(op: &str) -> Option<&'static dyn Kernel> {
    KERNELS.iter().copied().find(|k| k.ops().contains(&op))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names, ops and sources are each a kernel's alone.
    #[test]
    fn a_name_an_op_or_a_source_is_one_kernels() {
        let mut names = std::collections::BTreeSet::new();
        let mut ops = std::collections::BTreeSet::new();
        let mut files = std::collections::BTreeSet::new();
        for k in KERNELS {
            assert!(names.insert(k.name()), "{}", k.name());
            for o in k.ops() {
                assert!(ops.insert(*o), "{o}: two kernels");
            }
            for (f, _) in k.sources() {
                assert!(files.insert(*f), "{f}: two kernels");
            }
        }
        assert_eq!(
            by_op("ssm.gated_delta").map(|k| k.name()),
            Some("ssm.state")
        );
    }
}
