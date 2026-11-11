//! The table-driven form's data: a program's op rows and arena
//! ([`Table`]) and the opcodes.

use super::*;

/// Rows a kept-pool program reserves at the table's start for the fire's
/// page moves (three rows a move between PEs), and the words of the
/// scratch page they travel through.
pub const MOVE_ROWS: u32 = 48;
pub const MOVE_SCRATCH: u64 = 1024;

/// Words one arena array holds (a word address of 16 bits): a larger
/// arena is several arrays, an offset naming its array by `offset /
/// ARENA_CHUNK`.
pub const ARENA_CHUNK: u64 = 8192;

/// Words an op row holds beside its length word: the opcode and its
/// operands.
#[must_use]
pub fn op_words(op: &TableOp) -> u32 {
    match op {
        TableOp::Call { args, on, root, .. } => {
            args.len() as u32 + 1 + u32::from(on.is_some()) * 2 + u32::from(*root && on.is_none())
        }
        TableOp::Collective(_) => 4,
        TableOp::Nop => MOVE_ROW_WORDS,
    }
}

/// Words a page-move row holds (`OP_ON pe k_copy dst 0 src 0 words`).
pub const MOVE_ROW_WORDS: u32 = 8;

/// The table-driven form of a program: the kernels it dispatches (opcode =
/// index), the op rows (each its length word, then the opcode and the
/// operands; a zero length ends the table), and the arena every buffer
/// lives in.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub kernels: Vec<Kernel>,
    /// Words the op table holds.
    pub ops_words: u32,
    pub ops: Vec<TableOp>,
    /// Arena slots `(name, offset, words)` per PE: the exports (as laid
    /// out for the PE) and the scratch arrays (zeros).
    pub arena: Vec<(String, u64, u64)>,
    pub arena_words: u64,
    /// The first `keep_chunks` arena arrays hold the kept exports alone:
    /// a resident server uploads them once and never reads them back.
    pub keep_chunks: u32,
    /// Rows the table holds (`ops.len()`, at least one).
    pub rows: u32,
    /// The first `move_rows` rows are no-ops the host fills with the
    /// fire's page moves (kept pools).
    pub move_rows: u32,
    /// Constant tables `(name, words)` the kernels index (rope frequencies,
    /// n-gram primes): kept arena slots the host fills once, in place of
    /// the CSL globals the op table cannot point at.
    pub consts: Vec<(String, Vec<u32>)>,
    /// Phases the table runs (a whole fire's row: its share of the fire).
    pub phases: u32,
}

/// A CSL global array literal `var NAME = [N]T { a, b, … };` of `f32`,
/// `i32`, `u32` or `u64`, as the words it holds (a `u64` low word first,
/// as the PE lays it out).
#[must_use]
pub fn parse_global_table(line: &str) -> Option<(String, Vec<u32>)> {
    let rest = line.trim().strip_prefix("var ")?;
    let (name, rest) = rest.split_once(" = [")?;
    let (count, rest) = rest.split_once(']')?;
    let count: usize = count.trim().parse().ok()?;
    let (elem, rest) = rest.split_once('{')?;
    let elem = elem.trim();
    let body = rest.trim().trim_end_matches(';').trim().strip_suffix('}')?;
    let lits: Vec<&str> = body
        .split(',')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lits.len() != count {
        return None;
    }
    let mut words = Vec::with_capacity(count * 2);
    for l in lits {
        match elem {
            "f32" => words.push(l.parse::<f32>().ok()?.to_bits()),
            "i32" => words.push(l.parse::<i32>().ok()? as u32),
            "u32" => words.push(l.parse::<u32>().ok()?),
            "u64" => {
                let v = l.parse::<u64>().ok()?;
                words.push(v as u32);
                words.push((v >> 32) as u32);
            }
            _ => return None,
        }
    }
    Some((name.trim().to_string(), words))
}

impl Table {
    /// Arena arrays the PE declares: `arena_words` in chunks of
    /// [`ARENA_CHUNK`] (one array a chunk).
    #[must_use]
    pub fn chunks(&self) -> u64 {
        self.arena_words.div_ceil(ARENA_CHUNK).max(1)
    }

    /// Words arena array `c` holds: the chunk's prefix the arena's slots
    /// reach (a kept chunk padded to the boundary declares its kept words
    /// alone, so a model's few kept weights do not cost a whole chunk).
    #[must_use]
    pub fn chunk_words(&self, c: u64) -> u64 {
        let lo = c * ARENA_CHUNK;
        let hi = (lo + ARENA_CHUNK).min(self.arena_words);
        let reach = self
            .arena
            .iter()
            .filter(|(_, off, words)| *off < hi && off + words > lo)
            .map(|(_, off, words)| (off + words).min(hi))
            .max()
            .unwrap_or(lo);
        // A chunk below the arena's end without a slot still carries one
        // word (the arena stays addressable chunk by chunk).
        if c + 1 < self.chunks() && reach == lo {
            return 1;
        }
        reach.saturating_sub(lo).max(1)
    }
}

/// Lays kept slots `(name, words)` from the arena's start, each within one
/// chunk: the slots and the chunks they take.
#[must_use]
pub fn keep_layout(mut kept: Vec<(String, u64)>) -> (Vec<(String, u64, u64)>, u32) {
    kept.sort();
    kept.dedup_by(|a, b| {
        if a.0 == b.0 {
            b.1 = b.1.max(a.1);
            true
        } else {
            false
        }
    });
    let mut at = 0u64;
    let mut placed = Vec::new();
    for (name, words) in kept {
        let chunk = at / ARENA_CHUNK;
        if words <= ARENA_CHUNK && (at + words - 1) / ARENA_CHUNK != chunk {
            at = (chunk + 1) * ARENA_CHUNK;
        }
        placed.push((name, at, words));
        at += words;
    }
    (placed, at.div_ceil(ARENA_CHUNK) as u32)
}

/// One row of the op table.
#[derive(Debug, Clone, PartialEq)]
pub enum TableOp {
    /// Kernel `kernel` with `args` (pointers as arena offsets), on every PE,
    /// the row's root alone (`root`), or one PE alone (`on`).
    Call {
        root: bool,
        on: Option<u32>,
        kernel: u32,
        args: Vec<Arg>,
    },
    /// A collective along the row; the interpreter resumes after it lands.
    Collective(FabricOp),
    /// A row the host may fill (a page move), else skipped.
    Nop,
}

/// Opcodes below the kernels: the collectives and the root-only call.
pub const OP_REDUCE: i32 = -1;
pub const OP_GATHER: i32 = -2;
pub const OP_BROADCAST: i32 = -3;
pub const OP_ROOT: i32 = -4;
pub const OP_NOP: i32 = -5;
pub const OP_ON: i32 = -6;
pub const OP_BROADCAST_FROM: i32 = -7;
pub const OP_HANDOFF: i32 = -8;
