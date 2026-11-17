//! What a guest session shares across APIs: a stage's descriptor words and
//! config, and how wide it dispatches.

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
use eta_compiler::codegen::{wgsl, wgsl_analysis};
use eta_exec::{
    ChunkLayout, Extents, Launch, Launches, OpParams, Pieces, Placed, SELECT_RESERVE, ValueDesc,
    describe, lay_out_chunked, whole_value_temporaries,
};
use eta_ir::op::tags;

pub const DESC_WORDS: usize = 9;

pub const BINDINGS: u32 = 6;

pub const CFG_BINDING: u32 = 5;

/// The runtime's declarations of the bindings it only reads, the tables
/// read-only storage and the config a uniform (`ptir_runtime.wgsl`), and
/// each as a read_write storage binding: what a backend whose scopes take a
/// buffer one way only binds them as, since they lie in the pool segments
/// the scratch binds read_write (WebGPU).
pub const READ_ONLY_BINDINGS: [(&str, &str); 4] = [
    (
        "@group(0) @binding(1) var<storage, read>       descs",
        "@group(0) @binding(1) var<storage, read_write> descs",
    ),
    (
        "@group(0) @binding(2) var<storage, read>       params",
        "@group(0) @binding(2) var<storage, read_write> params",
    ),
    (
        "@group(0) @binding(3) var<storage, read>       offs",
        "@group(0) @binding(3) var<storage, read_write> offs",
    ),
    (
        "@group(0) @binding(5) var<uniform>             cfg",
        "@group(0) @binding(5) var<storage, read_write> cfg",
    ),
];

/// Where the segment an instance's scratch lies in binds: its pieces lie in
/// pool chunks of one segment (the runtime places them so), and a guest
/// address is a word of it.
pub const SCRATCH_BINDING: u32 = 4;

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
/// the temporary area's guest address, then each value's
/// ([`place_table`]), what a bind and a rebind write whole.
#[must_use]
pub fn place_words(values: usize) -> usize {
    values + 1
}

/// A stage's place table where its pieces lie at `pieces` (each piece's
/// guest address, [`PiecePlace::address`]): the temporary area's address,
/// then each value's, its piece's base and its word there summed at the
/// bind, so a guest access reads one word for its value's place.
#[must_use]
pub fn place_table(places: &[Placed], temporary: Placed, pieces: &[u32]) -> Vec<u8> {
    let address = |placed: &Placed| {
        let base = pieces.get(placed.piece as usize).copied().unwrap_or(0);
        base + u32::try_from(placed.offset / 4).unwrap_or(u32::MAX)
    };
    std::iter::once(&temporary)
        .chain(places)
        .flat_map(|placed| address(placed).to_le_bytes())
        .collect()
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
/// each, [`TABLE_BINDINGS`] order; the place table left for the bind to
/// write where the pieces lie, [`place_table`]).
pub struct StageTables {
    pub descriptors: Vec<ValueDesc>,
    pub places: Vec<Placed>,
    pub scratch: ChunkLayout,
    pub tables: [Vec<u8>; 4],
}

/// Resolves `plan` against `extents`, its scratch in `pieces` (runs of
/// adjacent chunks inside a segment): refused by name where a value is past
/// a segment.
pub fn stage_tables(
    plan: &LaunchStagePlan,
    extents: &Extents,
    pieces: Pieces,
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
        pieces,
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
            sized(Vec::new(), 2),
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
/// of the first piece with room, else a new piece. A piece is a run of
/// `run` adjacent chunks inside one segment ([`eta_exec::Pieces`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstanceLay {
    /// The chunks a piece takes.
    pub run: u32,
    /// A chunk's bytes.
    pub chunk: u64,
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

    /// A piece's bytes: its run of chunks.
    #[must_use]
    pub fn piece_bytes(&self) -> u64 {
        u64::from(self.run.max(1)).saturating_mul(self.chunk)
    }

    /// The bytes an instance holds (`engine::GuestState::per_instance`):
    /// the one piece's where it takes one inside a chunk (a run of slots),
    /// else whole runs of chunks, one a piece.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        match self.ends[..] {
            [] => 0,
            [only] if only <= self.chunk => only,
            _ => u64::from(self.pieces()).saturating_mul(self.piece_bytes()),
        }
    }
}

/// `stages`, laid in pieces of one run, together with `scores`' rows
/// ([`InstanceLay`]); a row past a piece is refused by name.
pub fn lay_instance(
    stages: &[&ChunkLayout],
    scores: &ScoreSpec,
    pieces: Pieces,
) -> Result<InstanceLay, String> {
    let run = stages.iter().map(|stage| stage.run).fold(pieces.run.max(1), u32::max);
    let piece = u64::from(run).saturating_mul(pieces.chunk);
    let mut ends: Vec<u64> = Vec::new();
    for stage in stages {
        if stage.run != run {
            return Err(format!(
                "a stage laid in pieces of {} chunk(s) beside one of {run}: a program lays \
                 every stage in one piece size",
                stage.run
            ));
        }
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
        if span > piece {
            return Err(format!(
                "{what} takes {bytes} bytes, past a {piece}-byte piece ({run} chunk(s)), and a \
                 score row lies whole in one"
            ));
        }
        for (at_piece, end) in ends.iter_mut().enumerate() {
            let at = end.next_multiple_of(align);
            if at + span <= piece {
                *end = at + span;
                return Ok(Placed {
                    piece: u32::try_from(at_piece).unwrap_or(u32::MAX),
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
    Ok(InstanceLay {
        run,
        chunk: pieces.chunk,
        ends,
        rows,
    })
}

/// `plans`' stages at `extents` and an instance's state with `scores`'
/// rows, in `pieces` ([`lay_instance`]): every stage, and every row, in
/// pieces of one run, the least that holds the widest of them whole.
pub fn lay_program(
    plans: &[LaunchStagePlan],
    extents: &Extents,
    scores: &ScoreSpec,
    pieces: Pieces,
) -> Result<(Vec<StageTables>, InstanceLay), String> {
    let first = plans
        .iter()
        .map(|plan| stage_tables(plan, extents, pieces))
        .collect::<Result<Vec<_>, _>>()?;
    let widest_row = scores
        .key_bytes
        .max(scores.page_bytes)
        .next_multiple_of(eta_exec::SCRATCH_ALIGN);
    let row_run = u32::try_from(widest_row.div_ceil(pieces.chunk.max(1))).unwrap_or(u32::MAX);
    let run = first
        .iter()
        .map(|stage| stage.scratch.run)
        .fold(pieces.run.max(1).max(row_run), u32::max);
    let stages = if first.iter().all(|stage| stage.scratch.run == run) {
        first
    } else {
        plans
            .iter()
            .map(|plan| stage_tables(plan, extents, pieces.at_least(run)))
            .collect::<Result<Vec<_>, _>>()?
    };
    let lay = lay_instance(
        &stages.iter().map(|stage| &stage.scratch).collect::<Vec<_>>(),
        scores,
        pieces.at_least(run),
    )?;
    Ok((stages, lay))
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

    /// The guest address of its first word: its word in its segment.
    #[must_use]
    pub fn address(self) -> u32 {
        u32::try_from(self.at / 4).unwrap_or(u32::MAX)
    }
}

/// Where the pieces of an instance's state laid as `lay` lie in the pool,
/// at `place` ([`engine::ScratchPlace`]): a run of slots' one piece from its
/// first slot, else each run of chunks' first byte, piece by piece, all in
/// one segment. Refused by name where the place holds fewer pieces, or runs
/// of another length, than the state takes, its pieces lie in more than one
/// segment, a run passes its segment, or a segment's words pass a guest
/// address.
pub fn piece_places<B: crate::backend::Backend>(
    pools: &crate::store::Pools<B>,
    place: &engine::ScratchPlace,
    lay: &InstanceLay,
) -> crate::error::Result<Vec<PiecePlace>> {
    use crate::store::layout::INSTANCE;
    let refused = |why: String| crate::error::Fault::Program {
        at: "guest::session",
        why,
    };
    let (pieces, held) = (lay.pieces(), lay.held_bytes());
    let firsts: Vec<u32> = match place {
        engine::ScratchPlace::None if pieces == 0 => Vec::new(),
        engine::ScratchPlace::None => {
            return Err(refused(format!(
                "a guest's state takes {pieces} piece(s) and its binding holds none"
            )));
        }
        engine::ScratchPlace::Run { first, slots } => {
            let room = u64::from(*slots) * pools.slot_bytes(INSTANCE)?;
            if pieces != 1 || room < held {
                return Err(refused(format!(
                    "a guest's state of {pieces} piece(s), {held} bytes, given a run of \
                     {slots} instance slot(s), {room} bytes"
                )));
            }
            vec![*first]
        }
        engine::ScratchPlace::Chunks { run, pieces: chunks } => {
            if (chunks.len() as u64) < u64::from(pieces) || *run != lay.run.max(1) {
                return Err(refused(format!(
                    "a guest's state of {pieces} piece(s) of {} chunk(s) given {} piece(s) of \
                     {run}",
                    lay.run,
                    chunks.len()
                )));
            }
            let per_chunk = pools.slots_per_chunk(INSTANCE)?;
            chunks[..pieces as usize]
                .iter()
                .map(|&chunk| {
                    chunk
                        .checked_mul(per_chunk)
                        .ok_or_else(|| refused(format!("chunk {chunk} names no instance slot")))
                })
                .collect::<crate::error::Result<_>>()?
        }
    };
    let span = match place {
        engine::ScratchPlace::Chunks { .. } => lay.piece_bytes(),
        _ => held,
    };
    let places = firsts
        .into_iter()
        .map(|slot| {
            let (segment, buffer, at) = pools.slot_segment(INSTANCE, slot)?;
            let end = at + span;
            if at % 4 != 0 || end > crate::backend::Buffer::bytes(buffer) || end / 4 > 1 << 32 {
                return Err(refused(format!(
                    "a guest piece of {span} bytes at byte {at} of segment {segment}, past the \
                     segment or the words a guest address reaches"
                )));
            }
            Ok(PiecePlace { segment, at })
        })
        .collect::<crate::error::Result<Vec<_>>>()?;
    if places.windows(2).any(|pair| pair[0].segment != pair[1].segment) {
        return Err(refused(format!(
            "a guest's state lies in one segment, which its stages bind, and its place puts \
             its pieces in segments {:?}",
            places.iter().map(|place| place.segment).collect::<Vec<_>>()
        )));
    }
    Ok(places)
}
