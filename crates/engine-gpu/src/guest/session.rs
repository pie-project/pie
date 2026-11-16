//! What a guest session shares across APIs: a stage's descriptor words and
//! config, and how wide it dispatches.

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
use eta_compiler::codegen::{wgsl, wgsl_analysis};
use eta_exec::{
    ChunkLayout, Extents, Launch, Launches, OpParams, Placed, SELECT_RESERVE, ValueDesc,
    describe, lay_out_chunked, whole_value_temporaries,
};
use eta_ir::op::tags;

pub const DESC_WORDS: usize = 9;

pub const BINDINGS: u32 = 9;

pub const CFG_BINDING: u32 = 5;

/// Where the pool's segments bind, one each: the instance's scratch lies in
/// pool chunks, and a chunk in one segment.
pub const SEGMENT_BINDINGS: [u32; kernels_slang::SEGMENTS] = [4, 6, 7, 8];

/// The top bits of a guest address name its segment
/// (`ptir_runtime.wgsl`'s `SEG_SHIFT`): a segment's words are fewer.
pub const SEGMENT_SHIFT: u32 = 30;

/// Where a pass's spans start in the host staging ring: the scratch's own
/// alignment, which a copy's offsets meet on every API.
pub const RING_ALIGN: u64 = eta_exec::SCRATCH_ALIGN;

pub const WORKGROUP: u32 = eta_compiler::codegen::wgsl::WORKGROUP;

pub const GROUPS_PER_CORE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Cfg {
    pub value_count: u32,
    pub op_count: u32,
    pub lane: u32,
    pub pad: u32,
}

const _: () = assert!(size_of::<Cfg>() == 16);

pub fn desc_words(d: &ValueDesc) -> [u32; DESC_WORDS] {
    let mut out = [0u32; DESC_WORDS];
    out[0] = d.len;
    out[1] = d.rows;
    out[2] = d.last;
    out[3] = d.rank;
    out[4] = d.dtype;
    for (at, &dim) in d.dims.iter().enumerate() {
        out[5 + at] = dim;
    }
    out
}

pub fn words_le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

impl Cfg {
    #[must_use]
    pub fn to_le_bytes(self) -> Vec<u8> {
        words_le(&[self.value_count, self.op_count, self.lane, self.pad])
    }
}

/// How a stage runs here, which its scratch is laid over: the emitted
/// WGSL launches one op after another in the stage's order, each alone in
/// its launch; the host writes a root's bytes before the first and reads
/// a put's after the last.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stepping {
    /// The nodes the WGSL launches ([`wgsl::launched_nodes`]).
    pub launched: Vec<u32>,
    /// The other nodes: boundary ops and reshapes their source covers.
    pub idle: Vec<u32>,
    /// What the host may write before the first launch: an output's, a
    /// channel read's, a take's and a constant's result.
    pub staged: Vec<u32>,
    /// What is read after the last launch: every value a boundary op reads.
    pub kept: Vec<u32>,
    /// The results of the reshapes not launched: their source's bytes
    /// ([`alias_reshapes`]).
    pub hollow: Vec<u32>,
}

impl Stepping {
    #[must_use]
    pub fn of(plan: &LaunchStagePlan) -> Stepping {
        let launched = wgsl::launched_nodes(plan);
        let mut stepping = Stepping::default();
        let mut result_base = 0u32;
        for (node, op) in plan.ops.iter().enumerate() {
            let results = result_base..result_base.saturating_add(u32::from(op.result_count));
            result_base = results.end;
            if matches!(
                op.tag,
                tags::OUTPUT | tags::CHAN_READ | tags::CHAN_TAKE | tags::CONST
            ) {
                stepping.staged.extend(results);
            }
            let node = node as u32;
            if launched.binary_search(&node).is_ok() {
                continue;
            }
            stepping.idle.push(node);
            if wgsl::is_boundary(op.tag) {
                stepping.kept.extend(op.reads());
            }
        }
        stepping.hollow = wgsl_analysis::analyze_stage(plan)
            .aliases
            .into_iter()
            .map(|(result, _)| result)
            .collect();
        stepping.launched = launched;
        stepping
    }

    /// The launches `descriptors` are laid over: below the temporary area
    /// the one-row select's state, in it what a launch that sorts or
    /// reduces a whole value takes.
    #[must_use]
    pub fn launches(&self, descriptors: &[ValueDesc]) -> Launches<'_> {
        Launches {
            each: self.launched.iter().map(Launch::of_node).collect(),
            idle: self.idle.clone(),
            staged: self.staged.clone(),
            kept: self.kept.clone(),
            hollow: self.hollow.clone(),
            reserve: SELECT_RESERVE,
            temporary: whole_value_temporaries(descriptors).unwrap_or(u64::MAX),
        }
    }
}

pub fn alias_reshapes(plan: &LaunchStagePlan, values: &mut [Placed]) {
    for (result, source) in wgsl_analysis::analyze_stage(plan).aliases {
        let Some(&at) = values.get(source as usize) else {
            continue;
        };
        if let Some(slot) = values.get_mut(result as usize) {
            *slot = at;
        }
    }
}

pub fn groups_for(descriptors: &[ValueDesc], cores: u32) -> u32 {
    let widest = descriptors.iter().map(|d| d.len).max().unwrap_or(1).max(1);
    let wanted = widest.div_ceil(WORKGROUP);
    let cap = cores.max(1).saturating_mul(GROUPS_PER_CORE);
    wanted.clamp(1, cap)
}

/// A stage's bind-time launch tables, in the order they lie in the
/// instance's `Instance` slots: descriptors, op params, value places and
/// the config, bound at these bindings.
pub const TABLE_BINDINGS: [u32; 4] = [1, 2, 3, CFG_BINDING];

/// The words of a stage's place table (binding 3) for `values` values:
/// each value's word in its piece, each value's piece, the temporary
/// area's piece and word, then the base of each piece, at most one a value
/// and one for the temporary area ([`bases_at`]).
#[must_use]
pub fn place_words(values: usize) -> usize {
    3 * values + 3
}

/// The byte of a stage's place table its piece bases start at: what a bind
/// and a rebind write.
#[must_use]
pub fn bases_at(values: usize) -> u64 {
    (2 * values as u64 + 2) * 4
}

/// The bytes of a stage's tables ([`TABLE_BINDINGS`] order): what its
/// value and op counts fix, whatever the extents a bind resolves.
#[must_use]
pub fn table_sizes(plan: &LaunchStagePlan) -> [u64; 4] {
    [
        (plan.value_types.len() * DESC_WORDS * 4).max(DESC_WORDS * 4) as u64,
        (plan.ops.len().max(1) * size_of::<OpParams>()) as u64,
        (place_words(plan.value_types.len()) * 4) as u64,
        size_of::<Cfg>() as u64,
    ]
}

/// Every stage's tables, stage by stage, then the instance's fault word,
/// as an instance of `plans` lays them in its slots.
#[must_use]
pub fn program_table_sizes(plans: &[LaunchStagePlan]) -> Vec<u64> {
    plans
        .iter()
        .flat_map(table_sizes)
        .chain([eta_exec::STATUS_BYTES as u64])
        .collect()
}

/// What a stage resolves at a bind: its values' descriptors and places,
/// its scratch in pieces of a chunk, and its tables' bytes ([`table_sizes`]
/// each, [`TABLE_BINDINGS`] order; the place table's bases left for the
/// bind to write, [`bases_at`]).
pub struct StageTables {
    pub descriptors: Vec<ValueDesc>,
    pub places: Vec<Placed>,
    pub scratch: ChunkLayout,
    pub tables: [Vec<u8>; 4],
}

/// Resolves `plan` against `extents`, its scratch in chunks of `chunk`
/// bytes: refused by name where a value is past a chunk.
pub fn stage_tables(
    plan: &LaunchStagePlan,
    extents: &Extents,
    chunk: u64,
) -> Result<StageTables, String> {
    let descriptors = plan
        .value_types
        .iter()
        .map(|value| {
            describe(value, extents).map_err(|why| {
                format!("a guest value's shape does not resolve against this fire: {why:?}")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scratch = lay_out_chunked(
        plan,
        &descriptors,
        &Stepping::of(plan).launches(&descriptors),
        chunk,
    )
    .map_err(|why| format!("a guest pass's scratch does not lie in chunks: {why}"))?;
    let mut places = scratch.values.clone();
    alias_reshapes(plan, &mut places);

    let sizes = table_sizes(plan);
    let sized = |mut bytes: Vec<u8>, at: usize| {
        bytes.resize(usize::try_from(sizes[at]).unwrap_or(usize::MAX), 0);
        bytes
    };
    let descs: Vec<u8> = descriptors
        .iter()
        .flat_map(|d| words_le(&desc_words(d)))
        .collect();
    let mut params = Vec::with_capacity(plan.ops.len() * size_of::<OpParams>());
    let mut result_base = 0u32;
    for op in &plan.ops {
        params.extend_from_slice(&OpParams::of(op, result_base, plan.output_slots).to_le_bytes());
        result_base += u32::from(op.result_count);
    }
    let word = |at: u64| u32::try_from(at / 4).unwrap_or(u32::MAX);
    let place_table: Vec<u32> = places
        .iter()
        .map(|placed| word(placed.offset))
        .chain(places.iter().map(|placed| placed.piece))
        .chain([scratch.temporary.piece, word(scratch.temporary.offset)])
        .collect();
    let config = Cfg {
        value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
        op_count: u32::try_from(plan.ops.len()).unwrap_or(u32::MAX),
        lane: 0,
        pad: 0,
    };
    Ok(StageTables {
        descriptors,
        tables: [
            sized(descs, 0),
            sized(params, 1),
            sized(words_le(&place_table), 2),
            sized(config.to_le_bytes(), 3),
        ],
        places,
        scratch,
    })
}

/// A binding's extents as a stage's shapes resolve against them.
#[must_use]
pub fn extents_of(extents: &engine::BindExtents) -> Extents {
    Extents {
        row_count: extents.row_count,
        token_count: extents.token_count,
        sampled_rows: extents.sampled_rows,
        query_len: extents.query_len,
        key_len: extents.key_len,
    }
}

/// The score rows an instance of a program holds for its capturing lanes
/// (`Shell::score_spec`): each lane a read of which switches capture on, and
/// whether it reads pages too; a key row's bytes and a page row's.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScoreSpec {
    pub lanes: Vec<(u32, bool)>,
    pub key_bytes: u64,
    pub page_bytes: u64,
}

/// Every output `package` reads, by name, and the lane it reads it for.
pub fn reads_of(package: &LaunchPackage) -> impl Iterator<Item = (&str, u32)> {
    package
        .values
        .iter()
        .filter_map(|value| Some((value.output.as_deref()?, value.lane)))
}

/// What an instance holds at its place: its stages' scratch (each stage over
/// the same pieces, so each piece as full as the fullest stage leaves it),
/// then its capturing lanes' score rows, each whole in a piece: a free tail
/// of the first piece with room, else a new piece.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstanceLay {
    /// Each piece's bytes in use.
    pub ends: Vec<u64>,
    /// Each capturing lane's key row and, where it reads pages, page row.
    pub rows: Vec<(u32, Placed, Option<Placed>)>,
}

impl InstanceLay {
    #[must_use]
    pub fn pieces(&self) -> u32 {
        u32::try_from(self.ends.len()).unwrap_or(u32::MAX)
    }

    /// The bytes an instance holds in chunks of `chunk` bytes: the one
    /// piece's where it takes one (a run inside a chunk), else whole
    /// chunks, one a piece.
    #[must_use]
    pub fn held_bytes(&self, chunk: u64) -> u64 {
        match self.ends[..] {
            [] => 0,
            [only] => only,
            _ => u64::from(self.pieces()).saturating_mul(chunk),
        }
    }
}

/// `stages` laid together with `scores`' rows in chunks of `chunk` bytes
/// ([`InstanceLay`]); a row larger than a chunk is refused by name.
pub fn lay_instance(
    stages: &[&ChunkLayout],
    scores: &ScoreSpec,
    chunk: u64,
) -> Result<InstanceLay, String> {
    let mut ends: Vec<u64> = Vec::new();
    for stage in stages {
        for (piece, &end) in stage.ends.iter().enumerate() {
            match ends.get_mut(piece) {
                Some(have) => *have = (*have).max(end),
                None => ends.push(end),
            }
        }
    }
    let align = eta_exec::SCRATCH_ALIGN;
    let mut place = |bytes: u64, what: &str| -> Result<Placed, String> {
        let span = bytes.next_multiple_of(align);
        if span > chunk {
            return Err(format!(
                "{what} takes {bytes} bytes, past a {chunk}-byte chunk, and a score row lies \
                 whole in one"
            ));
        }
        for (piece, end) in ends.iter_mut().enumerate() {
            let at = end.next_multiple_of(align);
            if at + span <= chunk {
                *end = at + span;
                return Ok(Placed {
                    piece: u32::try_from(piece).unwrap_or(u32::MAX),
                    offset: at,
                });
            }
        }
        ends.push(span);
        Ok(Placed {
            piece: u32::try_from(ends.len() - 1).unwrap_or(u32::MAX),
            offset: 0,
        })
    };
    let mut rows = Vec::with_capacity(scores.lanes.len());
    for &(lane, pages) in &scores.lanes {
        let key = place(scores.key_bytes, "a key row")?;
        let page = if pages {
            Some(place(scores.page_bytes, "a page row")?)
        } else {
            None
        };
        rows.push((lane, key, page));
    }
    Ok(InstanceLay { ends, rows })
}

/// What an instance of `plans` holds at `extents` with `scores`' rows, in
/// chunks of `chunk` bytes ([`lay_instance`]).
pub fn instance_lay(
    plans: &[LaunchStagePlan],
    extents: &Extents,
    scores: &ScoreSpec,
    chunk: u64,
) -> Result<InstanceLay, String> {
    let stages = plans
        .iter()
        .map(|plan| stage_tables(plan, extents, chunk).map(|tables| tables.scratch))
        .collect::<Result<Vec<_>, _>>()?;
    lay_instance(&stages.iter().collect::<Vec<_>>(), scores, chunk)
}

/// A capturing lane's score rows at its instance's place: its key row, and
/// its page row where it reads pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreRows {
    pub keys: PiecePlace,
    pub pages: Option<PiecePlace>,
}

/// `lay`'s rows where its pieces lie (`pieces`, [`piece_places`]), by the
/// program lane each is for.
#[must_use]
pub fn score_rows(lay: &InstanceLay, pieces: &[PiecePlace]) -> Vec<(u32, ScoreRows)> {
    let at = |placed: Placed| {
        pieces
            .get(placed.piece as usize)
            .map(|piece| piece.past(placed.offset))
    };
    lay.rows
        .iter()
        .filter_map(|&(lane, keys, pages)| {
            Some((
                lane,
                ScoreRows {
                    keys: at(keys)?,
                    pages: match pages {
                        Some(pages) => Some(at(pages)?),
                        None => None,
                    },
                },
            ))
        })
        .collect()
}

/// A piece of an instance's state as the guest runtime and the kernels
/// address it (or a place inside one): its segment and its byte there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiecePlace {
    pub segment: u32,
    /// Its first byte in the segment.
    pub at: u64,
}

impl PiecePlace {
    /// Byte `offset` of the piece.
    #[must_use]
    pub fn past(self, offset: u64) -> PiecePlace {
        PiecePlace {
            at: self.at + offset,
            ..self
        }
    }

    /// The guest address of its first word: the segment in the top bits.
    #[must_use]
    pub fn address(self) -> u32 {
        (self.segment << SEGMENT_SHIFT) | u32::try_from(self.at / 4).unwrap_or(u32::MAX)
    }
}

/// Where the `pieces` pieces of an instance's scratch lie in the pool, laid
/// at `place` ([`engine::ScratchPlace`]): a run's one piece from its first
/// slot, else each whole chunk's first byte, piece by piece. Refused by
/// name where the place holds fewer pieces than the scratch takes, or a
/// segment's words reach past the address bits.
pub fn piece_places<B: crate::backend::Backend>(
    pools: &crate::store::Pools<B>,
    place: &engine::ScratchPlace,
    pieces: u32,
    held: u64,
) -> crate::error::Result<Vec<PiecePlace>> {
    use crate::store::layout::INSTANCE;
    let refused = |why: String| crate::error::Fault::Program {
        at: "guest::session",
        why,
    };
    let firsts: Vec<u32> = match place {
        engine::ScratchPlace::None if pieces == 0 => Vec::new(),
        engine::ScratchPlace::None => {
            return Err(refused(format!(
                "a guest's scratch takes {pieces} piece(s) and its binding holds none"
            )));
        }
        engine::ScratchPlace::Run { first, slots } => {
            let room = u64::from(*slots) * pools.slot_bytes(INSTANCE)?;
            if pieces != 1 || room < held {
                return Err(refused(format!(
                    "a guest's scratch of {pieces} piece(s), {held} bytes, given a run of                      {slots} instance slot(s), {room} bytes"
                )));
            }
            vec![*first]
        }
        engine::ScratchPlace::Chunks(chunks) => {
            if (chunks.len() as u64) < u64::from(pieces) {
                return Err(refused(format!(
                    "a guest's scratch of {pieces} piece(s) given {} chunk(s)",
                    chunks.len()
                )));
            }
            let per_chunk = pools.slots_per_chunk(INSTANCE)?;
            chunks[..pieces as usize]
                .iter()
                .map(|&chunk| chunk.checked_mul(per_chunk).ok_or_else(|| refused(format!(
                    "chunk {chunk} names no instance slot"
                ))))
                .collect::<crate::error::Result<_>>()?
        }
    };
    firsts
        .into_iter()
        .map(|slot| {
            let (segment, _, at) = pools.slot_segment(INSTANCE, slot)?;
            if at % 4 != 0 || (at + pools.chunk_bytes()) / 4 > 1 << SEGMENT_SHIFT {
                return Err(refused(format!(
                    "a guest piece at byte {at} of segment {segment}, which a guest address \
                     of {SEGMENT_SHIFT} word bits does not reach"
                )));
            }
            Ok(PiecePlace { segment, at })
        })
        .collect()
}
