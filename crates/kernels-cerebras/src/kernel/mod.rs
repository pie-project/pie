//! Hand-written kernels as modules (GENERIC.md, "Kernels as modules"): what sits outside a
//! kernels-ir `Region` (a mixer, a routed bank, a tiled conv), each one module under the poem-ir
//! group it implements, as kernels-metal and kernels-cuda lay theirs out, registered once in
//! [`KERNELS`]. A module states its CSL sources, the poem-ir ops it implements and, by regime,
//! how it runs; the compiler asks the registry instead of matching kernels by name. Elementwise
//! ops are not modules: they go through `fuse::chain` into a kernels-ir `Region`
//! (`emit::region`), the path every backend shares.
//!
//! Adding a kernel: a module file here (its sources, ops and the facets it has) and one line in
//! [`KERNELS`].

pub mod attn;
pub mod conv;
pub mod linear;
pub mod ssm;

use crate::emit::stream::HostArray;
use crate::floor::Coord;
use crate::floorplan::stream::KvColumn;
use crate::stream::build::Program;
use crate::stream::place::KvSched;
use crate::stream::unit::{Cols, Joint, Lowering};

type R<T> = Result<T, String>;

/// A CSL source a module owns: its file name under `programs/wafer/ops` and its text.
pub type Source = (&'static str, &'static str);

/// A hand-written kernel.
pub trait Kernel: Sync {
    /// Its name: `<group>.<kernel>` (`attn.column`).
    fn name(&self) -> &'static str;
    /// The poem-ir ops it implements, by name.
    fn ops(&self) -> &'static [&'static str];
    /// The CSL op modules it owns (`programs/wafer/ops`), each owned by one kernel.
    fn sources(&self) -> &'static [Source];
    /// How it lowers its ops into a unit's steps (the unit regime), where it runs there.
    fn unit(&self) -> Option<&dyn UnitKernel> {
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

/// The line a GEMM-regime op's function rebinds an input queue along (`check::hazard::lines`,
/// GENERIC.md "The line windows").
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
    /// Why it leaves the row drifted where it does not order it (H2's message).
    pub drift: &'static str,
    /// The same, with ops after it (H4's message).
    pub drift_after: &'static str,
}

/// A kernel's op in the tiled regime: a function of the tiled PE's image (`tiled_pe.csl`), and
/// what the hub-loop check needs of it.
pub struct TiledForm {
    /// Its function in the PE's code: its sends before its receive off the hub's broadcast.
    pub func: &'static str,
    /// It, as the check's message names it.
    pub what: &'static str,
    /// The words a PE sends towards the hub before its receive, for op `op` of `p`.
    pub words: fn(p: &crate::stream::tiled::TileProgram, op: &crate::stream::tiled::TOp) -> u32,
}

/// A kernel in the unit regime (stream readings, decode through units): the steps its ops
/// become in a unit, read node by node (`stream::unit::lower_through` asks the kernel whose op
/// a node is).
pub trait UnitKernel: Sync {
    /// Lowers node `at` (one of its ops) into `cx`'s steps; `Ok(false)` where it runs no step
    /// for that node as it is (the unit refuses it by name). A pattern over several nodes lowers
    /// at its first and puts the rest in `cx.skip`.
    fn lower(&self, cx: &mut Lowering, at: u32) -> R<bool>;
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
    pub split: crate::stream::build::KvSplit,
}

/// A mixer on a unit's K / V column (P7, P8): its root takes each row's head (`[q | k | v]` and
/// [`Column::head_parts`]) off its unit, its PEs below hold what the rows leave (keys and values,
/// a state), each row's output goes back. The column's placement, chain and colours are the
/// floorplan's; what runs on it is the module's.
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
    let j = j.copied().unwrap_or(attn::column::UNSTATED);
    KERNELS
        .iter()
        .filter_map(|k| k.column())
        .find(|c| c.claims(&j))
        .ok_or_else(|| "a K / V column no kernel runs".to_string())
}

/// Every kernel, in registry order: the order their sources are listed and digested in.
pub static KERNELS: &[&dyn Kernel] = &[
    &attn::column::ATTENTION,
    &ssm::state::STATE,
    &linear::moe::MOE,
    &linear::gemm::GEMM,
    &conv::norm::GROUP_NORM,
    &attn::voxel::VOXEL,
];

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
