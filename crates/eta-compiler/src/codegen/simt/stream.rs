use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;

use eta_ir::Dtype;
use eta_ir::op::tags;

use crate::codegen::launch::{reduction_chunks, reduction_floats};
use crate::codegen::op_view::OpView;
use crate::plan::{CompiledStage, Dimension, Region};

#[derive(Clone, PartialEq, Eq)]
enum RowClass {
    Full(Dimension),
    Scalar,
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Map,
    Broadcast,
    Rng,
    Reduce,
    Argmax,
    Intrinsic,
}

struct Member {
    node: usize,
    role: Role,
}

pub(super) struct Streams<'a> {
    pub stage: &'a CompiledStage,
    pub region: &'a Region,
    pub ops: &'a [OpView],
    pub bases: &'a [u32],
    pub kinds: &'a [u8],
    pub direct_intrinsic: &'a [u16],
    pub skipped: &'a [u8],
    pub readers: Vec<Vec<usize>>,
    pub producers: Vec<u32>,
    pub direct_topk: &'a [Option<super::fused::TopKDirect>],
    pub order: Vec<usize>,
    /// Whether each row runs across several blocks of its lane
    /// ([`super::fused::splits_rows`]): each block streams its stretch of
    /// the row, its reductions' partials meeting in the row's work before
    /// every block folds them alike.
    pub split: bool,
    /// The meetings the kernel's reductions have emitted so far.
    pub meets: core::cell::Cell<u32>,
}

/// The narrowest row a split stream's reduction folds: below it a row's
/// chunks fit one block well and the meeting costs more than it saves.
pub(super) const SPLIT_ROW_MIN: u64 = 32_768;

impl<'a> Streams<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stage: &'a CompiledStage,
        region: &'a Region,
        ops: &'a [OpView],
        bases: &'a [u32],
        kinds: &'a [u8],
        direct_intrinsic: &'a [u16],
        skipped: &'a [u8],
        direct_topk: &'a [Option<super::fused::TopKDirect>],
    ) -> Self {
        let mut readers = vec![Vec::new(); stage.normalized.value_types.len()];
        let mut producers = vec![u32::MAX; stage.normalized.value_types.len()];
        for (node, op) in ops.iter().enumerate() {
            for arg in op.reads() {
                if let Some(list) = readers.get_mut(arg as usize) {
                    list.push(node);
                }
            }
            for result in 0..op.results {
                if let Some(slot) = producers.get_mut((bases[node] + result) as usize) {
                    *slot = node as u32;
                }
            }
        }
        let mut streams = Self {
            stage,
            region,
            ops,
            bases,
            kinds,
            direct_intrinsic,
            skipped,
            readers,
            producers,
            direct_topk,
            order: Vec::new(),
            split: false,
            meets: core::cell::Cell::new(0),
        };
        streams.order = streams.emission_order();
        streams
    }

    fn rematerializable(&self, value: u32) -> bool {
        let root_readers = self.readers.get(value as usize).map_or(0, Vec::len);
        self.remat_depth(value, 0, root_readers)
    }

    fn remat_depth(&self, value: u32, depth: u32, root_readers: usize) -> bool {
        if depth > 4 {
            return false;
        }
        match self.class(value) {
            RowClass::Scalar => return true,
            RowClass::Full(_) => {}
            // An output read in place through its views of one row.
            RowClass::Other if self.output_source(value) == Some(value) => {
                if self.output_row(value).is_none() {
                    return false;
                }
            }
            RowClass::Other => return false,
        }
        if self.reshaped_intrinsic(value).is_some() {
            return true;
        }
        let Some(&node) = self.producers.get(value as usize) else {
            return false;
        };
        if node == u32::MAX || self.skipped[node as usize] != 0 {
            return false;
        }
        let node = node as usize;
        let inside = self.region.nodes.iter().any(|n| n.index() == node);
        // A row below the root that is stored anyway is loaded, not
        // recomputed.
        if depth > 0
            && (!inside
                || self.region.outputs.contains(&value)
                || self.role(node, &mut None).is_none())
        {
            return true;
        }
        // An output is read in place from any region of the stage.
        if !inside && self.ops[node].tag != tags::OUTPUT {
            return false;
        }
        let op = &self.ops[node];
        if op.results != 1 {
            return false;
        }
        if op.tag == eta_ir::op::tags::OUTPUT {
            return matches!(self.role(node, &mut None), Some(Role::Intrinsic));
        }
        match self.role(node, &mut None) {
            Some(Role::Rng) => root_readers <= 1 && self.readers[value as usize].len() <= 1,
            Some(Role::Map | Role::Broadcast) => {
                (depth == 0 || self.readers[value as usize].len() <= 1)
                    && op
                        .reads()
                        .all(|a| self.remat_depth(a, depth + 1, root_readers))
            }
            _ => false,
        }
    }

    fn reshaped_intrinsic(&self, value: u32) -> Option<usize> {
        if !matches!(self.class(value), RowClass::Full(_))
            || self.dtype(value) != Dtype::F32
            || self.region.outputs.contains(&value)
            || self.region.sinks.iter().any(|sink| sink.value == value)
        {
            return None;
        }
        let inside = |v: u32| -> Option<usize> {
            let node = *self.producers.get(v as usize)?;
            (node != u32::MAX
                && self.skipped[node as usize] == 0
                && self.region.nodes.iter().any(|n| n.index() == node as usize))
            .then_some(node as usize)
        };
        let reshape = &self.ops[inside(value)?];
        if reshape.tag != tags::RESHAPE || reshape.args.len() != 1 {
            return None;
        }
        let source = reshape.args[0];
        if !crate::codegen::alias::covers(&self.stage.normalized.value_types, source, value) {
            return None;
        }
        let output = inside(source)?;
        (self.ops[output].tag == tags::OUTPUT
            && self.ops[output].results == 1
            && self.dtype(source) == Dtype::F32)
            .then_some(output)
    }

    pub(super) fn intrinsic_unread(&self, value: u32) -> bool {
        if self.region.outputs.contains(&value)
            || self.region.sinks.iter().any(|sink| sink.value == value)
        {
            return false;
        }
        let Some(readers) = self.readers.get(value as usize) else {
            return false;
        };
        !readers.is_empty()
            && readers
                .iter()
                .all(|&reshape| self.reshape_read_off_intrinsic(reshape))
    }

    fn reshape_read_off_intrinsic(&self, node: usize) -> bool {
        let reshaped = self.bases[node];
        self.ops[node].tag == tags::RESHAPE
            && self.reshaped_intrinsic(reshaped).is_some()
            && self.readers[reshaped as usize]
                .iter()
                .all(|&reader| self.stream_reader(reader))
    }

    fn stream_reader(&self, node: usize) -> bool {
        self.region.nodes.iter().any(|n| n.index() == node)
            && self.skipped[node] == 0
            && self.role(node, &mut None).is_some()
    }

    /// Whether `node` is an argmax that scans the model's row where it lies
    /// (the fused emitter's direct or Gumbel scan), its operand unread.
    fn scans_model_row(&self, node: usize) -> bool {
        matches!(
            self.ops[node].tag,
            tags::REDUCE_ARGMAX | tags::GUMBEL_ARGMAX
        ) && self.direct_intrinsic[node] != u16::MAX
    }

    fn topk_reads_direct(&self, node: usize, value: u32) -> bool {
        self.direct_topk.get(node).copied().flatten().is_some()
            && self.ops[node].args.first() == Some(&value)
    }

    /// The model output a value is a view of ([`output_source`]): a stream
    /// reads such a value's row in place, in the output's own storage.
    fn output_source(&self, value: u32) -> Option<u32> {
        output_source(self.stage, self.ops, &self.producers, value)
    }

    /// The row width an output is read at in place: its own, where it is a
    /// row, or that of the views of it this region reads, each a whole row
    /// of the output's width (so the output is that one row).
    fn output_row(&self, out: u32) -> Option<Dimension> {
        if let RowClass::Full(dim) = self.class(out) {
            return Some(dim);
        }
        let ty = &self.stage.normalized.value_types[out as usize];
        let last = *ty.dims.last()?;
        // Every reader a view, and every view a stream reads (past views of
        // views) a whole row of the output's width.
        let mut pending = vec![out];
        let mut any = false;
        while let Some(at) = pending.pop() {
            let readers = self.readers.get(at as usize)?;
            for &reader in readers {
                let view = self.bases[reader];
                let is_view = self.ops[reader].tag == tags::RESHAPE
                    && self.output_source(view) == Some(out)
                    && crate::codegen::alias::covers(&self.stage.normalized.value_types, at, view);
                if is_view {
                    pending.push(view);
                } else if at == out
                    || self.class(at) != RowClass::Full(last)
                    || !matches!(last, Dimension::Static(_))
                {
                    return None;
                } else {
                    any = true;
                }
            }
        }
        any.then_some(last)
    }

    /// The nodes past `value`'s views that read it: a view of an output is
    /// read where its readers read it.
    fn readers_through_views(&self, value: u32) -> Vec<usize> {
        let mut out = Vec::new();
        let mut pending = vec![value];
        while let Some(value) = pending.pop() {
            for &reader in self
                .readers
                .get(value as usize)
                .map_or(&[][..], Vec::as_slice)
            {
                let view = self.bases[reader];
                if self.ops[reader].tag == tags::RESHAPE
                    && self.output_source(view).is_some()
                    && crate::codegen::alias::covers(
                        &self.stage.normalized.value_types,
                        value,
                        view,
                    )
                {
                    pending.push(view);
                } else {
                    out.push(reader);
                }
            }
        }
        out
    }

    fn dtype(&self, value: u32) -> Dtype {
        self.stage.normalized.value_types[value as usize].dtype
    }

    fn class(&self, value: u32) -> RowClass {
        let ty = &self.stage.normalized.value_types[value as usize];
        match self.kinds.get(value as usize).copied().unwrap_or(0) {
            1 => ty
                .dims
                .last()
                .map(|d| RowClass::Full(*d))
                .unwrap_or(RowClass::Other),
            2 => RowClass::Scalar,
            _ => {
                if ty.dims.iter().all(|d| *d == Dimension::Static(1)) {
                    RowClass::Scalar
                } else {
                    RowClass::Other
                }
            }
        }
    }

    fn role(&self, node: usize, width: &mut Option<Dimension>) -> Option<Role> {
        if self.skipped[node] != 0 {
            return None;
        }
        let op = &self.ops[node];
        let out = self.bases[node];
        if op.results != 1 || out as usize >= self.stage.normalized.value_types.len() {
            return None;
        }
        let out_dtype = self.dtype(out);
        if !matches!(
            out_dtype,
            Dtype::F32 | Dtype::I32 | Dtype::U32 | Dtype::Bool | Dtype::Bf16
        ) {
            return None;
        }
        let full = |value: u32, width: &mut Option<Dimension>| -> bool {
            match self.class(value) {
                RowClass::Full(dim) => match width {
                    Some(w) => *w == dim,
                    None => {
                        *width = Some(dim);
                        true
                    }
                },
                _ => false,
            }
        };
        let indexed = |value: u32, width: &mut Option<Dimension>| -> bool {
            matches!(self.class(value), RowClass::Scalar) || full(value, width)
        };
        let scalar_dtypes = |values: &[u32]| {
            values.iter().all(|&v| {
                matches!(
                    self.dtype(v),
                    Dtype::F32 | Dtype::I32 | Dtype::U32 | Dtype::Bool | Dtype::Bf16
                )
            })
        };
        let tag = op.tag;
        match tag {
            tags::EXP
            | tags::LOG
            | tags::RECIP
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
            | tags::CAST
            | tags::NOT => {
                if op.args.len() != 1 || !scalar_dtypes(&op.args) {
                    return None;
                }
                if tag == tags::NOT && self.dtype(op.args[0]) != Dtype::Bool {
                    return None;
                }
                let mut w = *width;
                if !full(out, &mut w) || !indexed(op.args[0], &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::NEG | tags::ABS | tags::SIGN => {
                if op.args.len() != 1
                    || self.dtype(op.args[0]) == Dtype::Bool
                    || !scalar_dtypes(&op.args)
                {
                    return None;
                }
                let mut w = *width;
                if !full(out, &mut w) || !indexed(op.args[0], &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::MAX_ELEM
            | tags::MIN_ELEM
            | tags::REM
            | tags::GT
            | tags::GE
            | tags::EQ
            | tags::NE
            | tags::LT
            | tags::LE
            | tags::AND
            | tags::OR => {
                if op.args.len() != 2 || !scalar_dtypes(&op.args) {
                    return None;
                }
                let logic = matches!(tag, tags::AND | tags::OR);
                let left = self.dtype(op.args[0]);
                if logic && (left != Dtype::Bool || self.dtype(op.args[1]) != Dtype::Bool) {
                    return None;
                }
                if !logic && left == Dtype::Bool {
                    return None;
                }
                let mut w = *width;
                if !full(out, &mut w)
                    || !indexed(op.args[0], &mut w)
                    || !indexed(op.args[1], &mut w)
                {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::SELECT => {
                if op.args.len() != 3
                    || !scalar_dtypes(&op.args)
                    || self.dtype(op.args[0]) != Dtype::Bool
                {
                    return None;
                }
                let mut w = *width;
                if !full(out, &mut w) || !op.args.iter().all(|&a| indexed(a, &mut w)) {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::BROADCAST => {
                if op.args.len() != 1
                    || !scalar_dtypes(&op.args)
                    || self.class(op.args[0]) != RowClass::Scalar
                {
                    return None;
                }
                let mut w = *width;
                if !full(out, &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Broadcast)
            }
            tags::RNG | tags::RNG_KEYED => {
                if out_dtype != Dtype::F32 {
                    return None;
                }
                if tag == tags::RNG_KEYED
                    && (op.args.len() != 1 || self.dtype(op.args[0]) != Dtype::U32)
                {
                    return None;
                }
                let mut w = *width;
                if !full(out, &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Rng)
            }
            tags::OUTPUT => {
                if out_dtype != Dtype::F32 {
                    return None;
                }
                let mut w = *width;
                let row = self.output_row(out)?;
                match w {
                    Some(dim) if dim != row => return None,
                    _ => w = Some(row),
                }
                *width = w;
                Some(Role::Intrinsic)
            }
            tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX => {
                if op.args.len() != 1 || !eta_ir::types::is_float(self.dtype(op.args[0])) {
                    return None;
                }
                if tag == tags::REDUCE_ARGMAX && self.direct_intrinsic[node] != u16::MAX {
                    return None;
                }
                if self.class(out) != RowClass::Scalar {
                    return None;
                }
                let mut w = *width;
                if !full(op.args[0], &mut w) {
                    return None;
                }
                *width = w;
                Some(if tag == tags::REDUCE_ARGMAX {
                    Role::Argmax
                } else {
                    Role::Reduce
                })
            }
            tags::GUMBEL_ARGMAX => {
                // logits * scale + gumbel(state, i), -inf where keep is
                // false: an argmax whose lanes the stream makes in f32. The
                // fused draw takes one that reads the model's row itself.
                if self.direct_intrinsic[node] != u16::MAX
                    || op.args.len() != 4
                    || !eta_ir::types::is_float(self.dtype(op.args[0]))
                    || self.dtype(op.args[1]) != Dtype::Bool
                    || !eta_ir::types::is_float(self.dtype(op.args[2]))
                    || self.dtype(op.args[3]) != Dtype::U32
                    || self.class(op.args[2]) != RowClass::Scalar
                    || self.class(out) != RowClass::Scalar
                {
                    return None;
                }
                let mut w = *width;
                if !full(op.args[0], &mut w) || !indexed(op.args[1], &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Argmax)
            }
            tags::PIVOT_THRESHOLD => {
                if op.pred_tag != eta_ir::wire::predicate_tags::PROB_GE
                    || op.args.len() != 1
                    || out_dtype != Dtype::Bool
                    || !scalar_dtypes(&[op.args[0], op.pred_payload])
                {
                    return None;
                }
                let mut w = *width;
                if !full(out, &mut w)
                    || !full(op.args[0], &mut w)
                    || !indexed(op.pred_payload, &mut w)
                {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            _ => None,
        }
    }

    /// The ops the stream at `at` covers, and whether its rows may split
    /// across blocks: its members map, broadcast, read the model's row or
    /// reduce a static row at least [`SPLIT_ROW_MIN`] wide, which a block's
    /// stretch computes alone or folds after the meeting.
    pub(super) fn split_safe(&self, at: usize) -> Option<(usize, bool)> {
        let members = self.plan(at)?;
        let safe = members.iter().all(|member| match member.role {
            Role::Map | Role::Broadcast => true,
            // An output whole rows of its region read, or one read in place
            // through views a stream reads as whole rows of its width: each
            // block reads its stretch of the model's row.
            Role::Intrinsic => {
                let out = self.bases[member.node];
                matches!(self.class(out), RowClass::Full(_)) || self.output_row(out).is_some()
            }
            // A draw's lanes run on from its block's row whole.
            Role::Argmax if self.ops[member.node].tag == tags::GUMBEL_ARGMAX => false,
            Role::Reduce | Role::Argmax => self.ops[member.node].args.first().is_some_and(|&arg| {
                matches!(
                    self.class(arg),
                    RowClass::Full(Dimension::Static(width)) if u64::from(width) >= SPLIT_ROW_MIN
                )
            }),
            Role::Rng => false,
        });
        Some((members.len(), safe))
    }

    /// The ops the stream at `at` covers, and whether it reduces or argmaxes
    /// (in a split row, a phase boundary after it).
    pub(super) fn meets_after(&self, at: usize) -> Option<(usize, bool)> {
        let members = self.plan(at)?;
        let meets = members
            .iter()
            .any(|member| matches!(member.role, Role::Reduce | Role::Argmax));
        Some((members.len(), meets))
    }

    fn plan(&self, at: usize) -> Option<Vec<Member>> {
        let mut width: Option<Dimension> = None;
        let mut members: Vec<Member> = Vec::new();
        let mut reduced: Vec<u32> = Vec::new();
        for &node in &self.order[at..] {
            let op = &self.ops[node];
            if op.reads().any(|a| reduced.contains(&a)) {
                break;
            }
            let Some(role) = self.role(node, &mut width) else {
                break;
            };
            if matches!(role, Role::Reduce | Role::Argmax) {
                reduced.push(self.bases[node]);
            }
            members.push(Member { node, role });
        }
        // An output read in place alone holds no row of its own: where it is
        // read past its views' readers, it is copied as it was.
        let in_place_alone = members.len() == 1
            && members[0].role == Role::Intrinsic
            && !matches!(self.class(self.bases[members[0].node]), RowClass::Full(_));
        if members.is_empty()
            || (in_place_alone && self.escapes(self.bases[members[0].node], &members))
        {
            None
        } else {
            Some(members)
        }
    }

    fn escapes(&self, value: u32, members: &[Member]) -> bool {
        // An output read in place is read through its views; a view a later
        // region or a sink reads holds its bytes, so the output does.
        let in_place = self.ops[self.producers[value as usize] as usize].tag == tags::OUTPUT;
        let readers = if in_place {
            let mut views = vec![value];
            let mut at = 0;
            while at < views.len() {
                for &reader in self
                    .readers
                    .get(views[at] as usize)
                    .map_or(&[][..], Vec::as_slice)
                {
                    let view = self.bases[reader];
                    if self.ops[reader].tag == tags::RESHAPE
                        && self.output_source(view).is_some()
                        && crate::codegen::alias::covers(
                            &self.stage.normalized.value_types,
                            views[at],
                            view,
                        )
                    {
                        views.push(view);
                    }
                }
                at += 1;
            }
            if views
                .iter()
                .any(|&view| self.region.sinks.iter().any(|s| s.value == view))
            {
                return true;
            }
            // Read past this region, the row is read in place there too, or
            // the bytes are kept for whoever reads them.
            let beyond = views.iter().any(|view| self.region.outputs.contains(view))
                || self
                    .readers_through_views(value)
                    .iter()
                    .any(|&node| !self.region.nodes.iter().any(|n| n.index() == node));
            if beyond && !read_in_place_beyond(self.stage, self.region, value) {
                return true;
            }
            self.readers_through_views(value)
                .into_iter()
                .filter(|&node| self.region.nodes.iter().any(|n| n.index() == node))
                .collect()
        } else {
            self.readers
                .get(value as usize)
                .cloned()
                .unwrap_or_default()
        };
        let outside: Vec<usize> = readers
            .into_iter()
            .filter(|node| !members.iter().any(|m| m.node == *node))
            .collect();
        let remat = self.rematerializable(value);
        let served = |node: &usize| {
            (remat && self.stream_reader(*node))
                || self.topk_reads_direct(*node, value)
                || (in_place && self.scans_model_row(*node))
                || self.reshape_read_off_intrinsic(*node)
        };
        if outside.iter().any(|node| !served(node)) {
            return true;
        }
        self.region.outputs.contains(&value) && outside.is_empty()
    }

    fn emission_order(&self) -> Vec<usize> {
        use eta_ir::op::tags;
        let ops = self.ops;
        let bases = self.bases;
        let region = self.region;
        let nodes: Vec<usize> = region.nodes.iter().map(|n| n.index()).collect();
        let count = nodes.len();
        let producer = |value: u32| -> Option<usize> {
            nodes.iter().position(|&n| {
                let base = bases[n];
                value >= base && value < base + ops[n].results
            })
        };
        let reduction = |tag: u8| {
            matches!(
                tag,
                tags::REDUCE_SUM
                    | tags::REDUCE_MAX
                    | tags::REDUCE_MIN
                    | tags::REDUCE_ARGMAX
                    | tags::GUMBEL_ARGMAX
            )
        };
        let channel = |tag: u8| matches!(tag, tags::CHAN_TAKE | tags::CHAN_READ | tags::CHAN_PUT);
        let kind = |tag: u8| -> u32 {
            match tag {
                tags::CHAN_TAKE | tags::CHAN_READ => 0,
                tags::CONST | tags::RESHAPE => 1,
                tags::GATHER_ROW => 3,
                tags::CHAN_PUT => 4,
                _ => 2,
            }
        };
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut depth = vec![0u32; count];
        let mut last_channel: Option<usize> = None;
        for i in 0..count {
            let op = &ops[nodes[i]];
            for arg in op.reads() {
                if let Some(p) = producer(arg) {
                    deps[i].push(p);
                    depth[i] = depth[i].max(depth[p] + u32::from(reduction(ops[nodes[p]].tag)));
                }
            }
            if channel(op.tag) {
                if let Some(previous) = last_channel {
                    deps[i].push(previous);
                }
                last_channel = Some(i);
            }
        }
        for i in (0..count).rev() {
            let node = nodes[i];
            let value = bases[node];
            if !matches!(
                self.role(node, &mut None),
                Some(Role::Map | Role::Broadcast | Role::Rng)
            ) {
                continue;
            }
            if region.outputs.contains(&value) || region.sinks.iter().any(|s| s.value == value) {
                continue;
            }
            let min_consumer = (0..count)
                .filter(|&c| deps[c].contains(&i))
                .map(|c| depth[c])
                .min();
            if let Some(d) = min_consumer {
                depth[i] = depth[i].max(d);
            }
        }
        let mut placed = vec![false; count];
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let next = (0..count)
                .filter(|&i| !placed[i] && deps[i].iter().all(|&d| placed[d]))
                .min_by_key(|&i| (depth[i], kind(ops[nodes[i]].tag), i));
            let Some(i) = next else {
                return nodes;
            };
            placed[i] = true;
            out.push(nodes[i]);
        }
        out
    }
}

/// Whether every reader of model output `value` (through its views) past
/// `region` reads the row in place: a node of a generated region whose
/// stream reads it through the output, or an argmax that scans the model's
/// row itself. Only then does `region` leave the row unwritten.
pub(super) fn read_in_place_beyond(stage: &CompiledStage, region: &Region, value: u32) -> bool {
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = crate::codegen::op_view::result_bases(&ops);
    let mut producers = vec![u32::MAX; stage.normalized.value_types.len()];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); stage.normalized.value_types.len()];
    for (node, op) in ops.iter().enumerate() {
        for result in 0..op.results {
            if let Some(slot) = producers.get_mut((bases[node] + result) as usize) {
                *slot = node as u32;
            }
        }
        for arg in op.reads() {
            if let Some(list) = readers.get_mut(arg as usize) {
                list.push(node);
            }
        }
    }
    let Some(source) = output_source(stage, &ops, &producers, value) else {
        return false;
    };
    // The readers past the views, wherever they are.
    let mut pending = vec![source];
    let mut outside = Vec::new();
    while let Some(at) = pending.pop() {
        for &reader in &readers[at as usize] {
            let view = bases[reader];
            if ops[reader].tag == tags::RESHAPE
                && output_source(stage, &ops, &producers, view).is_some()
                && crate::codegen::alias::covers(&stage.normalized.value_types, at, view)
            {
                pending.push(view);
            } else if !region.nodes.iter().any(|n| n.index() == reader) {
                outside.push(reader);
            }
        }
    }
    outside.iter().all(|&reader| {
        let Some(home) = stage
            .fused
            .regions
            .iter()
            .find(|r| r.nodes.iter().any(|n| n.index() == reader))
        else {
            return false;
        };
        // The lane sampler's bf16 draw reads the model's row where it lies.
        if let Some(values) = super::sampler::sampler_values(stage, home) {
            return values
                .draw
                .and_then(|draw| draw.output)
                .is_some_and(|(node, _)| bases[node as usize] == source);
        }
        if home.kind != crate::plan::RegionKind::Generated
            || super::validate::validate_generated_region(stage, home).is_err()
        {
            return false;
        }
        let Some(kinds) = super::fused::stream_rows(stage, home) else {
            return false;
        };
        let direct = super::fused::analyze_direct_argmax(stage, home, &bases);
        let direct_topk = super::fused::analyze_direct_topk(stage);
        let claimed = direct.claimed();
        let streams = Streams::new(
            stage,
            home,
            &ops,
            &bases,
            &kinds,
            &claimed,
            &direct.skipped,
            &direct_topk,
        );
        streams.scans_model_row(reader)
            || (streams.stream_reader(reader) && streams.rematerializable(source))
    })
}

/// The model output `value` is a view of: `value` itself where an output
/// produces it, or the output a chain of reshapes covering it (the same
/// elements in the same order) reads, in any region of the stage. The row is the
/// model's for the whole stage, so any region may read it where it lies.
pub(super) fn output_source(
    stage: &CompiledStage,
    ops: &[OpView],
    producers: &[u32],
    value: u32,
) -> Option<u32> {
    let node = *producers.get(value as usize)?;
    if node == u32::MAX {
        return None;
    }
    let op = &ops[node as usize];
    match op.tag {
        tags::OUTPUT => Some(value),
        // A reshape covering its source, itself an output or a view of one
        // (a one-row fire's squeezed `[v]` reshaped to `[1, v]`, then again).
        tags::RESHAPE if !op.args.is_empty() => {
            let source = op.args[0];
            if !crate::codegen::alias::covers(&stage.normalized.value_types, source, value) {
                return None;
            }
            output_source(stage, ops, producers, source)
        }
        _ => None,
    }
}

/// The dtype a stream computes and holds a value in: a `bf16` value is an
/// f32 in a register, the stored value exactly (each `bf16` result is
/// rounded once where it is computed, [`typed_expr`]).
fn compute(dtype: Dtype) -> Dtype {
    if dtype == Dtype::Bf16 {
        Dtype::F32
    } else {
        dtype
    }
}

fn c_type(dtype: Dtype) -> &'static str {
    match compute(dtype) {
        Dtype::F32 => "float",
        Dtype::I32 => "int",
        Dtype::U32 => "m1_u32",
        _ => "bool",
    }
}

fn letter(dtype: Dtype) -> char {
    match compute(dtype) {
        Dtype::F32 => 'f',
        Dtype::I32 => 'i',
        Dtype::U32 => 'u',
        _ => 'b',
    }
}

fn code(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::F32 => 0,
        Dtype::Bf16 => u32::from(eta_ir::types::WIRE_BF16),
        Dtype::I32 => 1,
        Dtype::U32 => 2,
        _ => 3,
    }
}

fn convert(expr: &str, from: Dtype, to: Dtype) -> String {
    let (from, to) = (compute(from), compute(to));
    if from == to {
        return expr.into();
    }
    match (to, from) {
        (Dtype::F32, Dtype::I32) | (Dtype::F32, Dtype::U32) => format!("float({expr})"),
        (Dtype::F32, _) => format!("({expr} ? 1.0f : 0.0f)"),
        (Dtype::I32, Dtype::F32) => format!("m1_float_to_i32({expr})"),
        (Dtype::I32, Dtype::U32) => format!("m1_bits_i32({expr})"),
        (Dtype::I32, _) => format!("({expr} ? 1 : 0)"),
        (Dtype::U32, Dtype::F32) => format!("m1_float_to_u32({expr})"),
        (Dtype::U32, Dtype::I32) => format!("(m1_u32)({expr})"),
        (Dtype::U32, _) => format!("({expr} ? 1u : 0u)"),
        (_, Dtype::F32) => format!("({expr} != 0.0f)"),
        (_, Dtype::I32) => format!("({expr} != 0)"),
        (_, _) => format!("({expr} != 0u)"),
    }
}

#[derive(Default)]
struct Bodies {
    s: String,
    v: String,
    pre: String,
    post: String,
    /// Each reduction's fold of the chunks the pass just read, after a
    /// scalar and after a vector iteration.
    fold_s: String,
    fold_v: String,
    ptrs: Vec<String>,
    conds: Vec<String>,
}

impl Bodies {
    fn both(&mut self, line: &str) {
        self.s.push_str(line);
        self.s.push('\n');
        self.v.push_str(line);
        self.v.push('\n');
    }

    fn touch(&mut self, ptr: &str) {
        if !self.ptrs.iter().any(|p| p == ptr) {
            self.ptrs.push(ptr.into());
        }
    }

    fn load(&mut self, name: &str, want: Dtype, ptr: &str, from: u32) {
        let l = letter(want);
        let _ = writeln!(
            self.s,
            "      const {} {name} = m1_load_{l}({ptr}, i, {from}u);",
            c_type(want)
        );
        let _ = writeln!(
            self.pre,
            "        const {} q_{name} = m1_load4_{l}({ptr}, i4 >> 2u, {from}u);",
            vec_type(want)
        );
        let _ = writeln!(
            self.v,
            "      const {} {name} = m1_pick4_{l}(q_{name}, j);",
            c_type(want)
        );
        self.touch(ptr);
    }

    /// A stored value: a `bf16` one an element's two bytes (`m1_store_h`)
    /// a scalar iteration, a vector iteration's four elements eight bytes,
    /// the two words they fill, by the thread that computed them.
    fn store(&mut self, dtype: Dtype, ptr: &str, reg: &str) {
        let l = letter(dtype);
        let st = if dtype == Dtype::Bf16 { 'h' } else { l };
        let _ = writeln!(
            self.s,
            "      if (stream_live) m1_store_{st}({ptr}, i, {reg});"
        );
        let _ = writeln!(self.pre, "        {} w_{reg};", vec_type(dtype));
        let _ = writeln!(self.v, "      m1_set4_{l}(w_{reg}, j, {reg});");
        let _ = writeln!(
            self.post,
            "        if (stream_live) m1_store4_{st}({ptr}, i4 >> 2u, w_{reg});"
        );
        self.touch(ptr);
    }
}

fn vec_type(dtype: Dtype) -> &'static str {
    match compute(dtype) {
        Dtype::F32 => "float4",
        Dtype::I32 => "int4",
        Dtype::U32 => "uint4",
        _ => "uchar4",
    }
}

struct Pass<'p> {
    registers: Vec<(u32, Dtype)>,
    hoisted: Vec<(u32, Dtype)>,
    loaded: Vec<(u32, Dtype)>,
    intrinsics: Vec<usize>,
    rngs: Vec<usize>,
    prologue: String,
    body: Bodies,
    pointer: &'p mut dyn FnMut(u32) -> String,
}

fn read(streams: &Streams<'_>, value: u32, want: Dtype, pass: &mut Pass<'_>) -> String {
    let want = compute(want);
    // A view of an output is the output's row, read in place.
    let value = match streams.output_source(value) {
        Some(source) if streams.rematerializable(source) => source,
        _ => value,
    };
    if let Some((_, from)) = pass.registers.iter().find(|(v, _)| *v == value) {
        return convert(&format!("r{value}"), *from, want);
    }
    let from = streams.dtype(value);
    let l = letter(want);
    match streams.class(value) {
        RowClass::Scalar => {
            let name = format!("s{value}_{l}");
            if !pass.hoisted.contains(&(value, want)) {
                pass.hoisted.push((value, want));
                let ptr = (pass.pointer)(value);
                let _ = writeln!(
                    pass.prologue,
                    "    const {} {name} = m1_load_{l}({ptr}, 0u, {}u);",
                    c_type(want),
                    code(from)
                );
            }
            name
        }
        _ => {
            if streams.rematerializable(value) {
                let reg = rematerialize(streams, value, pass);
                return convert(&reg, from, want);
            }
            let name = format!("l{value}_{l}");
            if !pass.loaded.contains(&(value, want)) {
                pass.loaded.push((value, want));
                let ptr = (pass.pointer)(value);
                pass.body.load(&name, want, &ptr, code(from));
            }
            name
        }
    }
}

fn rematerialize(streams: &Streams<'_>, value: u32, pass: &mut Pass<'_>) -> String {
    if let Some(output) = streams.reshaped_intrinsic(value) {
        intrinsic_read(streams, output, value, pass);
        return format!("r{value}");
    }
    let node = streams.producers[value as usize] as usize;
    let op = &streams.ops[node];
    let out = streams.bases[node];
    if op.tag == tags::OUTPUT {
        intrinsic_read(streams, node, out, pass);
    } else if matches!(op.tag, tags::RNG | tags::RNG_KEYED) {
        rng_read(streams, node, out, pass);
    } else {
        let out_dtype = streams.dtype(out);
        let expr = typed_expr(streams, node, out_dtype, pass);
        pass.body.both(&format!(
            "      const {} r{out} = {expr};",
            c_type(out_dtype)
        ));
        pass.registers.push((out, out_dtype));
    }
    format!("r{out}")
}

fn rng_read(streams: &Streams<'_>, node: usize, out: u32, pass: &mut Pass<'_>) {
    let op = &streams.ops[node];
    let tag = op.tag;
    if !pass.rngs.contains(&node) {
        pass.rngs.push(node);
        let _ = writeln!(pass.prologue, "    M1OpParams p{node} = params[{node}u];");
        let _ = writeln!(pass.prologue, "    p{node}.rng_seed = 0u;");
        let _ = writeln!(
            pass.prologue,
            "    p{node}.imm3 = lane_row * descriptors[p{node}.o0].len;"
        );
        if tag == tags::RNG {
            let _ = writeln!(
                pass.prologue,
                "    const m1_u64 seed{node} = ptir_rng_seed_eff_stream((m1_u32)p{node}.rng_seed, p{node}.imm);"
            );
        } else {
            let state = op.args[0];
            let sp = (pass.pointer)(state);
            let _ = writeln!(
                pass.prologue,
                "    const m1_u64 seed{node} = ptir_rng_state_seed({sp}, descriptors[{state}u]);"
            );
        }
    }
    if op.kind == eta_ir::types::RngKind::Normal as u8 {
        pass.body.both(&format!(
            "      const float r{out} = ptir_rng_hash_normal(seed{node}, i + p{node}.imm3);"
        ));
    } else if op.kind == eta_ir::types::RngKind::Uniform as u8 {
        pass.body.both(&format!(
            "      const float r{out} = ptir_rng_hash_uniform(seed{node}, i + p{node}.imm3);"
        ));
    } else {
        pass.body.both(&format!(
            "      const float u{node} = ptir_rng_hash_uniform(seed{node}, i + p{node}.imm3);"
        ));
        pass.body.both(&format!(
            "      const float r{out} = -logf(-logf(u{node}));"
        ));
    }
    pass.registers.push((out, Dtype::F32));
}

fn intrinsic_read(streams: &Streams<'_>, node: usize, out: u32, pass: &mut Pass<'_>) {
    if !pass.intrinsics.contains(&node) {
        pass.intrinsics.push(node);
        let slots = super::fused::output_pitch(streams.stage);
        let p = &mut pass.prologue;
        let _ = writeln!(p, "    M1OpParams p{node} = params[{node}u];");
        let _ = writeln!(p, "    p{node}.rng_seed = 0u;");
        let _ = writeln!(
            p,
            "    const m1_u32 intrinsic_index{node} = dispatch_lane * {slots}u + p{node}.intr;"
        );
        let _ = writeln!(
            p,
            "    p{node}.intrinsic_dtype = intrinsic_modes[intrinsic_index{node}];"
        );
        let _ = writeln!(
            p,
            "    p{node}.imm = intrinsic_widths[intrinsic_index{node}];"
        );
        let _ = writeln!(
            p,
            "    p{node}.intrinsic_row_stride = intrinsic_strides[intrinsic_index{node}];"
        );
        let _ = writeln!(
            p,
            "    p{node}.intrinsic_row_offset = intrinsic_offsets[intrinsic_index{node}] + {};",
            if streams.split {
                "ptir_row"
            } else {
                "lane_row"
            }
        );
        let _ = writeln!(
            p,
            "    p{node}.intrinsic_rows = intrinsic_rows[intrinsic_index{node}];"
        );
        let _ = writeln!(
            p,
            "    const m1_u8* ibase{node} = reinterpret_cast<const m1_u8*>(intrinsic_bases[intrinsic_index{node}]);"
        );
        let _ = writeln!(
            p,
            "    const M1ValueDesc idesc{node} = descriptors[{out}u];"
        );
        let _ = writeln!(
            p,
            "    const m1_u32 iwidth{node} = idesc{node}.last == 0u ? p{node}.imm : idesc{node}.last;"
        );
        let _ = writeln!(
            p,
            "    const m1_u32 istride{node} = p{node}.intrinsic_row_stride == 0u ? iwidth{node} : p{node}.intrinsic_row_stride;"
        );
        let _ = writeln!(
            p,
            "    const m1_u64 ifirst{node} = (m1_u64)p{node}.intrinsic_row_offset + (m1_u64)p{node}.imm2;"
        );
        pass.body.conds.push(format!(
            "(iwidth{node} & 3u) == 0u && stream_width <= iwidth{node} && m1_intrinsic_row_vectorable(ibase{node}, ifirst{node}, istride{node}, p{node}.intrinsic_dtype, p{node}.intrinsic_rows)"
        ));
    }
    let _ = writeln!(
        pass.body.s,
        "      const float r{out} = m1_intrinsic_row_load(ibase{node}, ifirst{node} + i / iwidth{node}, i % iwidth{node}, istride{node}, p{node}.intrinsic_dtype, p{node}.intrinsic_rows);"
    );
    let _ = writeln!(
        pass.body.pre,
        "        const float4 q_r{out} = m1_intrinsic_row_load4(ibase{node}, ifirst{node}, i4, istride{node}, p{node}.intrinsic_dtype, p{node}.intrinsic_rows);"
    );
    let _ = writeln!(
        pass.body.v,
        "      const float r{out} = m1_pick4_f(q_r{out}, j);"
    );
    pass.registers.push((out, Dtype::F32));
}

/// Lane `i` of a gumbel_argmax `node`, as a register: `logits[i] *
/// scale + gumbel(state, i)`, `-inf` where keep is false, in the f32
/// operations of `reduce_argmax(select(keep, logits * scale, -inf) +
/// rng_keyed(state))` (eta_ir::types), the noise lane running on from this
/// block's row as a keyed draw's does.
fn gumbel_lane(streams: &Streams<'_>, node: usize, pass: &mut Pass<'_>) -> String {
    let op = &streams.ops[node];
    let (logits, state) = (op.args[0], op.args[3]);
    let sp = (pass.pointer)(state);
    let _ = writeln!(
        pass.prologue,
        "    const m1_u64 seed{node} = ptir_rng_state_seed({sp}, descriptors[{state}u]);"
    );
    let _ = writeln!(
        pass.prologue,
        "    const m1_u32 noise{node} = lane_row * descriptors[{logits}u].len;"
    );
    let x = read(streams, logits, Dtype::F32, pass);
    let keep = read(streams, op.args[1], Dtype::Bool, pass);
    let divisor = read(streams, op.args[2], Dtype::F32, pass);
    pass.body.both(&format!(
        "      const float u{node} = ptir_rng_hash_uniform(seed{node}, i + noise{node});"
    ));
    pass.body.both(&format!(
        "      const float n{node} = -logf(-logf(u{node}));"
    ));
    pass.body.both(&format!(
        "      const float g{node} = {keep} ? {x} * {divisor} + n{node} : m1_neg_inf();"
    ));
    format!("g{node}")
}

/// `node`'s value as its register holds it: a `bf16` result rounded once,
/// as its store rounds it, so a reader in the stream sees the stored value.
fn typed_expr(streams: &Streams<'_>, node: usize, out_dtype: Dtype, pass: &mut Pass<'_>) -> String {
    let expr = map_expr(streams, node, compute(out_dtype), pass);
    if out_dtype == Dtype::Bf16 {
        format!("m1_round_bf16({expr})")
    } else {
        expr
    }
}

fn map_expr(streams: &Streams<'_>, node: usize, out_dtype: Dtype, pass: &mut Pass<'_>) -> String {
    let op = &streams.ops[node];
    let tag = op.tag;
    if tag == tags::BROADCAST {
        read(streams, op.args[0], out_dtype, pass)
    } else if matches!(
        tag,
        tags::EXP | tags::LOG | tags::RECIP | tags::SIN | tags::COS | tags::SQRT | tags::RSQRT
    ) {
        let x = read(streams, op.args[0], Dtype::F32, pass);
        match tag {
            tags::EXP => format!("expf({x})"),
            tags::LOG => format!("logf({x})"),
            tags::SIN => format!("sinf({x})"),
            tags::COS => format!("cosf({x})"),
            tags::SQRT => format!("sqrtf({x})"),
            tags::RSQRT => format!("(1.0f / sqrtf({x}))"),
            _ => format!("(1.0f / {x})"),
        }
    } else if tag == tags::CAST {
        read(streams, op.args[0], out_dtype, pass)
    } else if tag == tags::NOT {
        let x = read(streams, op.args[0], Dtype::Bool, pass);
        format!("(!{x})")
    } else if matches!(tag, tags::NEG | tags::ABS | tags::SIGN) {
        let path = compute(streams.dtype(op.args[0]));
        let x = read(streams, op.args[0], path, pass);
        match (tag, path) {
            (tags::NEG, Dtype::F32) => format!("(-{x})"),
            (tags::ABS, Dtype::F32) => format!("fabsf({x})"),
            (tags::SIGN, Dtype::F32) => {
                format!("({x} > 0.0f ? 1.0f : ({x} < 0.0f ? -1.0f : 0.0f))")
            }
            (tags::NEG, Dtype::I32) => format!("(int)(0u - (m1_u32){x})"),
            (tags::ABS, Dtype::I32) => {
                format!("((m1_u32){x} == 0x80000000u ? {x} : ({x} < 0 ? -{x} : {x}))")
            }
            (tags::SIGN, Dtype::I32) => format!("({x} > 0 ? 1 : ({x} < 0 ? -1 : 0))"),
            (tags::NEG, _) => format!("(0u - {x})"),
            (tags::ABS, _) => x,
            (_, _) => format!("({x} != 0u ? 1u : 0u)"),
        }
    } else if tag == tags::SELECT {
        let c = read(streams, op.args[0], Dtype::Bool, pass);
        let a = read(streams, op.args[1], out_dtype, pass);
        let b = read(streams, op.args[2], out_dtype, pass);
        format!("({c} ? {a} : {b})")
    } else if matches!(tag, tags::AND | tags::OR) {
        let a = read(streams, op.args[0], Dtype::Bool, pass);
        let b = read(streams, op.args[1], Dtype::Bool, pass);
        if tag == tags::AND {
            format!("({a} && {b})")
        } else {
            format!("({a} || {b})")
        }
    } else if tag == tags::PIVOT_THRESHOLD {
        let a = read(streams, op.args[0], Dtype::F32, pass);
        let b = read(streams, op.pred_payload, Dtype::F32, pass);
        format!("({a} >= {b})")
    } else {
        // Float operands compute in f32 whichever float dtype each is.
        let path = if eta_ir::types::is_float(streams.dtype(op.args[1])) {
            Dtype::F32
        } else {
            compute(streams.dtype(op.args[0]))
        };
        let a = read(streams, op.args[0], path, pass);
        let b = read(streams, op.args[1], path, pass);
        match (tag, path) {
            (tags::GT, _) => format!("({a} > {b})"),
            (tags::GE, _) => format!("({a} >= {b})"),
            (tags::EQ, _) => format!("({a} == {b})"),
            (tags::NE, _) => format!("({a} != {b})"),
            (tags::LT, _) => format!("({a} < {b})"),
            (tags::LE, _) => format!("({a} <= {b})"),
            (tags::ADD, Dtype::F32) => format!("({a} + {b})"),
            (tags::SUB, Dtype::F32) => format!("({a} - {b})"),
            (tags::MUL, Dtype::F32) => format!("({a} * {b})"),
            (tags::DIV, Dtype::F32) => format!("({a} / {b})"),
            (tags::MAX_ELEM, Dtype::F32) => format!("m1_element_max({a}, {b})"),
            (tags::MIN_ELEM, Dtype::F32) => format!("m1_element_min({a}, {b})"),
            (_, Dtype::F32) => format!("fmodf({a}, {b})"),
            (tags::ADD, Dtype::I32) => format!("(int)((m1_u32){a} + (m1_u32){b})"),
            (tags::SUB, Dtype::I32) => format!("(int)((m1_u32){a} - (m1_u32){b})"),
            (tags::MUL, Dtype::I32) => format!("(int)((m1_u32){a} * (m1_u32){b})"),
            (tags::DIV, Dtype::I32) => format!("m1_i32_div({a}, {b})"),
            (tags::MAX_ELEM, Dtype::I32) => format!("({a} > {b} ? {a} : {b})"),
            (tags::MIN_ELEM, Dtype::I32) => format!("({a} < {b} ? {a} : {b})"),
            (_, Dtype::I32) => format!("m1_i32_rem({a}, {b})"),
            (tags::ADD, _) => format!("({a} + {b})"),
            (tags::SUB, _) => format!("({a} - {b})"),
            (tags::MUL, _) => format!("({a} * {b})"),
            (tags::DIV, _) => format!("({b} == 0u ? 0u : {a} / {b})"),
            (tags::MAX_ELEM, _) => format!("({a} > {b} ? {a} : {b})"),
            (tags::MIN_ELEM, _) => format!("({a} < {b} ? {a} : {b})"),
            (_, _) => format!("({b} == 0u ? 0u : {a} % {b})"),
        }
    }
}

pub(super) fn emit_stream(
    source: &mut String,
    streams: &Streams<'_>,
    at: usize,
    pointer: &mut dyn FnMut(u32) -> String,
    tail: &str,
) -> Option<usize> {
    let members = streams.plan(at)?;
    let ops = streams.ops;
    let bases = streams.bases;

    let witness = members
        .iter()
        .find_map(|m| {
            let op = &ops[m.node];
            let out = bases[m.node];
            if matches!(streams.class(out), RowClass::Full(_)) {
                Some(out)
            } else {
                op.args
                    .iter()
                    .copied()
                    .find(|&a| matches!(streams.class(a), RowClass::Full(_)))
            }
        })
        .or_else(|| {
            // An output read in place, one row of its views' width.
            members
                .iter()
                .find(|m| m.role == Role::Intrinsic)
                .map(|m| bases[m.node])
        })
        .expect("a stream has a full value");
    // A row a stream reads is a static width: a reduction's slots are laid
    // out from it.
    let stream_width = match streams.class(witness) {
        RowClass::Full(Dimension::Static(width)) => Some(width),
        _ => match streams.output_row(witness) {
            Some(Dimension::Static(width)) => Some(width),
            _ => None,
        },
    };

    let mut s = String::new();
    let _ = writeln!(s, "  {{ // stream of {} op(s)", members.len());
    s.push_str("    const m1_u32 stream_lane = threadIdx.x & 31u;\n");
    s.push_str("    const m1_u32 stream_warp = threadIdx.x >> 5u;\n");
    s.push_str("    const m1_u32 stream_warps = blockDim.x >> 5u;\n");
    // Only a reduction keeps anything in the temporary area.
    let reduces = members.iter().any(|m| m.role == Role::Reduce);
    if streams.split && reduces {
        // The row's reductions' first levels lie in its work, every block's
        // partials at their chunks; each block folds them in its own shared
        // memory, so a block keeps nothing of its own in the temporary area
        // and the row's area is the row's, however many blocks it splits
        // across.
        s.push_str("    float* stream_work = reinterpret_cast<float*>(ptir_row_work);\n");
    } else if reduces {
        s.push_str("    float* stream_work = reinterpret_cast<float*>(temporary);\n");
    }
    let _ = writeln!(
        s,
        "    const m1_u32 stream_width = descriptors[{witness}u].len;"
    );
    if streams.split {
        s.push_str(
            "    const m1_u32 stream_lo = ptir_stretch(stream_width, ptir_part, ptir_split);\n",
        );
        s.push_str(
            "    const m1_u32 stream_hi = ptir_stretch(stream_width, ptir_part + 1u, ptir_split);\n",
        );
    }

    let mut needed = vec![false; members.len()];
    for idx in (0..members.len()).rev() {
        let node = members[idx].node;
        let out = bases[node];
        if needed[idx]
            || matches!(members[idx].role, Role::Reduce | Role::Argmax)
            || streams.escapes(out, &members)
        {
            needed[idx] = true;
            for arg in ops[node].reads() {
                if let Some(p) = members.iter().position(|m| bases[m.node] == arg) {
                    needed[p] = true;
                }
            }
        }
    }
    if !needed.iter().any(|&b| b) {
        return Some(members.len());
    }

    let mut pass = Pass {
        registers: Vec::new(),
        hoisted: Vec::new(),
        loaded: Vec::new(),
        intrinsics: Vec::new(),
        rngs: Vec::new(),
        prologue: String::new(),
        body: Bodies::default(),
        pointer,
    };
    let mut epilogue = String::new();
    // A split row's work after its blocks' partials are all in: run in the
    // region's next phase (a launch of its own, so every block's partials
    // are written before any block folds them; no block waits on another).
    let mut post = String::new();
    let mut float_reductions = 0u32;
    let mut argmaxes = 0u32;

    for (idx, member) in members.iter().enumerate() {
        if !needed[idx] {
            continue;
        }
        let node = member.node;
        let op = &ops[node];
        let out = bases[node];
        let out_dtype = streams.dtype(out);
        let tag = op.tag;
        match member.role {
            Role::Map | Role::Broadcast => {
                let expr = typed_expr(streams, node, out_dtype, &mut pass);
                pass.body.both(&format!(
                    "      const {} r{out} = {expr};",
                    c_type(out_dtype)
                ));
                pass.registers.push((out, out_dtype));
                if streams.escapes(out, &members) {
                    let ptr = (pass.pointer)(out);
                    pass.body.store(out_dtype, &ptr, &format!("r{out}"));
                }
            }
            Role::Intrinsic => {
                intrinsic_read(streams, node, out, &mut pass);
                if streams.escapes(out, &members) {
                    let ptr = (pass.pointer)(out);
                    pass.body.store(Dtype::F32, &ptr, &format!("r{out}"));
                }
            }
            Role::Rng => {
                rng_read(streams, node, out, &mut pass);
                if streams.escapes(out, &members) {
                    let ptr = (pass.pointer)(out);
                    pass.body.store(Dtype::F32, &ptr, &format!("r{out}"));
                }
            }
            Role::Reduce => {
                // The interpreter's `canonical_reduce`: each run of 32
                // consecutive elements folds by the warp tree into one value
                // at the reduction's slot, the levels above fold after the
                // pass. A scalar iteration holds a run in a warp's lanes; a
                // vector one in eight lanes' quads.
                let x = read(streams, op.args[0], Dtype::F32, &mut pass);
                let identity = format!("m1_reduce_identity({tag}u)");
                let combine = |a: &str, b: &str| format!("m1_reduce_combine({tag}u, {a}, {b})");
                let width = u64::from(stream_width.expect("a reducing stream's row is static"));
                let chunks = reduction_chunks(width);
                // A split row's partials alternate between the slot's halves
                // by phase: the next phase folds this phase's half while its
                // own streams write the other, so no block overwrites a
                // partial another block has yet to fold. (Its fold's levels
                // above the first are in shared memory, so the half is free.)
                let bank = if streams.split {
                    u64::from(streams.meets.get() % 2) * chunks
                } else {
                    0
                };
                let slot = u64::from(float_reductions) * reduction_floats(width) + bank;
                float_reductions += 1;
                // As in the reference, a level folds only while more than one
                // value is left: a one-element row is its own reduction.
                let first = if width > 1 {
                    format!("m1_reduce_chunk({tag}u, stream_live ? {x} : {identity})")
                } else {
                    x.clone()
                };
                let _ = writeln!(pass.body.s, "      const float red{node} = {first};");
                let _ = writeln!(
                    pass.body.fold_s,
                    "      if (stream_lane == 0u) {};",
                    work_put(
                        streams.split,
                        &format!("{slot}u + (stream_base >> 5u)"),
                        &format!("red{node}")
                    )
                );
                let _ = writeln!(pass.body.pre, "        float red{node}[4];");
                let _ = writeln!(pass.body.v, "      red{node}[j] = {x};");
                let f = &mut pass.body.fold_v;
                let _ = writeln!(f, "        {{");
                let _ = writeln!(
                    f,
                    "          float r0 = stream_live ? red{node}[0] : {identity};"
                );
                let _ = writeln!(
                    f,
                    "          float r1 = stream_live ? red{node}[1] : {identity};"
                );
                let _ = writeln!(
                    f,
                    "          float r2 = stream_live ? red{node}[2] : {identity};"
                );
                let _ = writeln!(
                    f,
                    "          float r3 = stream_live ? red{node}[3] : {identity};"
                );
                // Element `4t + j` of a run sits in its `t`-th lane's quad:
                // offsets 16, 8 and 4 are 4, 2 and 1 lanes away, 2 and 1
                // within the quad.
                let _ = writeln!(
                    f,
                    "          for (m1_u32 apart = 4u; apart > 0u; apart >>= 1u) {{"
                );
                let _ = writeln!(f, "            const float o0 = m1_chunk_down(r0, apart);");
                let _ = writeln!(f, "            const float o1 = m1_chunk_down(r1, apart);");
                let _ = writeln!(f, "            const float o2 = m1_chunk_down(r2, apart);");
                let _ = writeln!(f, "            const float o3 = m1_chunk_down(r3, apart);");
                let _ = writeln!(f, "            if ((stream_lane & 7u) < apart) {{");
                let _ = writeln!(f, "              r0 = {};", combine("r0", "o0"));
                let _ = writeln!(f, "              r1 = {};", combine("r1", "o1"));
                let _ = writeln!(f, "              r2 = {};", combine("r2", "o2"));
                let _ = writeln!(f, "              r3 = {};", combine("r3", "o3"));
                let _ = writeln!(f, "            }}");
                let _ = writeln!(f, "          }}");
                let _ = writeln!(f, "          r0 = {};", combine("r0", "r2"));
                let _ = writeln!(f, "          r1 = {};", combine("r1", "r3"));
                let _ = writeln!(f, "          r0 = {};", combine("r0", "r1"));
                let _ = writeln!(
                    f,
                    "          if ((stream_lane & 7u) == 0u && stream_live) {};",
                    work_put(streams.split, &format!("{slot}u + (i4 >> 5u)"), "r0")
                );
                let _ = writeln!(f, "        }}");
                let ptr = (pass.pointer)(out);
                let _ = writeln!(epilogue, "    __syncthreads();");
                if streams.split {
                    // In the next phase every block's partials are in: each
                    // folds them all in the reference's order, so every block
                    // holds the one-block answer. The levels above the first:
                    // a value a run of 32 of the level below, folded by
                    // `ptir_canonical_reduce` in place.
                    let upper = chunks.div_ceil(32);
                    let _ = writeln!(post, "    {{");
                    let _ = writeln!(
                        post,
                        "      __shared__ float stream_fold{node}[{}u];",
                        upper + upper.div_ceil(32)
                    );
                    let _ = writeln!(
                        post,
                        "      const float total = ptir_canonical_reduce_met({tag}u, stream_work + {slot}u, {chunks}u, stream_fold{node}, stream_fold{node} + {upper}u);"
                    );
                    let _ = writeln!(
                        post,
                        "      if (threadIdx.x == 0u) m1_store_f({ptr}, 0u, total);"
                    );
                    let _ = writeln!(post, "    }}");
                    let _ = writeln!(post, "    __syncthreads();");
                } else {
                    let _ = writeln!(epilogue, "    {{");
                    let _ = writeln!(
                        epilogue,
                        "      const float total = ptir_canonical_reduce({tag}u, stream_work + {slot}u, {chunks}u, stream_work + {}u, stream_work + {slot}u);",
                        slot + chunks
                    );
                    let _ = writeln!(
                        epilogue,
                        "      if (threadIdx.x == 0u) m1_store_f({ptr}, 0u, total);"
                    );
                    let _ = writeln!(epilogue, "    }}");
                }
            }
            Role::Argmax => {
                let x = if tag == tags::GUMBEL_ARGMAX {
                    gumbel_lane(streams, node, &mut pass)
                } else {
                    read(streams, op.args[0], Dtype::F32, &mut pass)
                };
                argmaxes += 1;
                let _ = writeln!(
                    pass.prologue,
                    "    M1ArgmaxCandidate cand{node}{{m1_neg_inf(), 0u, 0u, 0u}};"
                );
                pass.body.both(&format!(
                    "      if (stream_live) cand{node} = m1_argmax_combine(cand{node}, M1ArgmaxCandidate{{{x}, i, m1_isnan({x}) ? 0u : 1u, 0u}});"
                ));
                let _ = writeln!(
                    epilogue,
                    "    cand{node} = m1_argmax_warp_reduce(cand{node}, stream_lane);"
                );
                let _ = writeln!(
                    epilogue,
                    "    if (stream_lane == 0u) stream_candidates[stream_warp] = cand{node};"
                );
                let _ = writeln!(epilogue, "    __syncthreads();");
                let _ = writeln!(epilogue, "    if (stream_warp == 0u) {{");
                let _ = writeln!(
                    epilogue,
                    "      cand{node} = stream_lane < stream_warps ? stream_candidates[stream_lane] : M1ArgmaxCandidate{{m1_neg_inf(), 0u, 0u, 0u}};"
                );
                let _ = writeln!(
                    epilogue,
                    "      cand{node} = m1_argmax_warp_reduce(cand{node}, stream_lane);"
                );
                let ptr = (pass.pointer)(out);
                if streams.split {
                    // The candidates' bank alternates by phase, as a
                    // reduction's partials do.
                    let bank = streams.meets.get() % 2;
                    // Each part's candidate to the lane arena; in the next
                    // phase every block takes the row's best of them, the one
                    // a block's scan picks (the order is total). A row on
                    // one block (its launch did not split it: more rows than
                    // the arena's lines hold) has its answer here, and its
                    // phases run one after another all the same.
                    let _ = writeln!(
                        epilogue,
                        "      if (stream_lane == 0u && ptir_split == 1u) m1_store_i({ptr}, 0u, (int)cand{node}.index);"
                    );
                    let _ = writeln!(
                        epilogue,
                        "      else if (stream_lane == 0u) ptir_put_candidate(ptir_meet_word, {bank}u, ptir_row, ptir_part, cand{node});"
                    );
                    let _ = writeln!(epilogue, "    }}");
                    let _ = writeln!(post, "    if (stream_warp == 0u && ptir_split != 1u) {{");
                    let _ = writeln!(
                        post,
                        "      cand{node} = stream_lane < ptir_split ? ptir_get_candidate(ptir_meet_word, {bank}u, ptir_row, stream_lane) : M1ArgmaxCandidate{{m1_neg_inf(), 0u, 0u, 0u}};"
                    );
                    let _ = writeln!(
                        post,
                        "      cand{node} = m1_argmax_warp_reduce(cand{node}, stream_lane);"
                    );
                    let _ = writeln!(
                        post,
                        "      if (stream_lane == 0u) m1_store_i({ptr}, 0u, (int)cand{node}.index);"
                    );
                    let _ = writeln!(post, "    }}");
                    let _ = writeln!(post, "    __syncthreads();");
                } else {
                    let _ = writeln!(
                        epilogue,
                        "      if (stream_lane == 0u) m1_store_i({ptr}, 0u, (int)cand{node}.index);"
                    );
                    let _ = writeln!(epilogue, "    }}");
                }
            }
        }
        if matches!(member.role, Role::Reduce | Role::Argmax) {
            epilogue.push_str("    __syncthreads();\n");
        }
    }

    if argmaxes > 0 {
        s.push_str("    __shared__ M1ArgmaxCandidate stream_candidates[32];\n");
    }
    let Pass { prologue, body, .. } = pass;
    s.push_str(&prologue);
    // A split row's stream runs in its region's current phase; its work after
    // the blocks' partials are in, in the next.
    let phase = streams.meets.get();
    if streams.split {
        let _ = writeln!(s, "    if (ptir_phase == {phase}u) {{");
    }
    let mut conds: Vec<String> = vec!["(stream_width & 3u) == 0u".into()];
    conds.extend(body.ptrs.iter().map(|p| format!("m1_aligned16({p})")));
    conds.extend(body.conds.iter().cloned());
    let _ = writeln!(s, "    const bool stream_vec = {};", conds.join(" && "));
    // Whole warps iterate together, so that a reduction's chunk folds by
    // shuffles: a lane past the row reads the row's last element and keeps
    // what it computes to itself.
    // A split row's block walks its stretch, from `stream_lo` to `stream_hi`.
    let (lo, hi) = if streams.split {
        ("stream_lo + ", "stream_hi")
    } else {
        ("", "stream_width")
    };
    s.push_str("    if (stream_vec) {\n");
    let _ = writeln!(
        s,
        "      for (m1_u32 stream_base = {lo}stream_warp * 128u; stream_base < {hi}; stream_base += blockDim.x * 4u) {{"
    );
    let _ = writeln!(
        s,
        "        const bool stream_live = stream_base + stream_lane * 4u < {hi};"
    );
    s.push_str(
        "        const m1_u32 i4 = stream_live ? stream_base + stream_lane * 4u : stream_width - 4u;\n",
    );
    s.push_str(&body.pre);
    s.push_str("#pragma unroll\n");
    s.push_str("        for (m1_u32 j = 0u; j < 4u; ++j) {\n");
    s.push_str("      const m1_u32 i = i4 + j;\n");
    s.push_str(&body.v);
    s.push_str("        }\n");
    s.push_str(&body.post);
    s.push_str(&body.fold_v);
    s.push_str("      }\n");
    s.push_str("    } else {\n");
    let _ = writeln!(
        s,
        "    for (m1_u32 stream_base = {lo}stream_warp * 32u; stream_base < {hi}; stream_base += blockDim.x) {{"
    );
    let _ = writeln!(
        s,
        "      const bool stream_live = stream_base + stream_lane < {hi};"
    );
    s.push_str(
        "      const m1_u32 i = stream_live ? stream_base + stream_lane : stream_width - 1u;\n",
    );
    s.push_str(&body.s);
    s.push_str(&body.fold_s);
    s.push_str("    }\n");
    s.push_str("    }\n");
    s.push_str(&epilogue);
    if streams.split {
        s.push_str("    }\n");
        if !post.is_empty() {
            streams.meets.set(phase + 1);
            let _ = writeln!(s, "    if (ptir_phase == {}u) {{", phase + 1);
            s.push_str(&post);
            s.push_str("    }\n");
        }
    }
    s.push_str(tail);
    s.push_str("  }\n");
    source.push_str(&s);
    Some(members.len())
}

/// The order a generated region's ops run in: the streams' when its rows run
/// in parallel, else the region's own. Its scratch is laid out over this
/// order, so a value's slot is free only once the ops that run before it are
/// done with it.
#[must_use]
pub fn emitted_order(stage: &CompiledStage, region: &Region) -> Vec<u32> {
    let own = || region.nodes.iter().map(|node| node.get()).collect();
    if region.kind != crate::plan::RegionKind::Generated
        || super::validate::validate_generated_region(stage, region).is_err()
    {
        return own();
    }
    let Some(kinds) = super::fused::stream_rows(stage, region) else {
        return own();
    };
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = crate::codegen::op_view::result_bases(&ops);
    let direct = super::fused::analyze_direct_argmax(stage, region, &bases);
    let direct_topk = super::fused::analyze_direct_topk(stage);
    let claimed = direct.claimed();
    let streams = Streams::new(
        stage,
        region,
        &ops,
        &bases,
        &kinds,
        &claimed,
        &direct.skipped,
        &direct_topk,
    );
    streams.order.into_iter().map(|node| node as u32).collect()
}

/// The results of a generated `region`'s nodes whose bytes its kernel
/// names nowhere, streams aside: what a node it leaves out
/// ([`super::fused::left_out`]) would have written, and a reshape it reads
/// as its source ([`super::fused::is_view`]). One a sink or another region
/// reads keeps its bytes, and so does a left-out node's result that a node
/// the kernel does run reads where it lies.
fn unnamed_values(
    stage: &CompiledStage,
    region: &Region,
    ops: &[OpView],
    bases: &[u32],
    direct: &super::fused::ArgmaxScan,
) -> Vec<u32> {
    let within = |node: usize| region.nodes.iter().any(|n| n.index() == node);
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); stage.normalized.value_types.len()];
    for (node, op) in ops.iter().enumerate() {
        for arg in op.reads() {
            if let Some(list) = readers.get_mut(arg as usize) {
                list.push(node);
            }
        }
    }
    let view = |node: usize| super::fused::is_view(stage, region, &ops[node], bases[node]);
    let left_out = |node: usize| super::fused::left_out(&direct.skipped, ops, node);
    // Whether a node the kernel runs reads `value`'s own bytes: a view of
    // it is read as it is.
    let read_where_it_lies = |value: u32| -> bool {
        let mut pending = vec![value];
        while let Some(value) = pending.pop() {
            for &reader in readers.get(value as usize).map_or(&[][..], Vec::as_slice) {
                if !within(reader) {
                    return true;
                }
                if left_out(reader) || super::fused::scans_model_row(direct, ops, reader) {
                    continue;
                }
                if !view(reader) {
                    return true;
                }
                pending.push(bases[reader]);
            }
        }
        false
    };
    let read_past = crate::codegen::alias::escaping_values(region);
    let mut unnamed = Vec::new();
    for node in region.nodes.iter().map(|n| n.index()) {
        let base = bases[node];
        if view(node) {
            if !read_past.contains(&base)
                || super::fused::output_view_read_in_place(stage, region, base)
            {
                unnamed.push(base);
            }
        } else if left_out(node) {
            unnamed.extend(
                (base..base + ops[node].results)
                    .filter(|value| !read_past.contains(value) && !read_where_it_lies(*value)),
            );
        }
    }
    unnamed
}

/// The values of `region` its kernel neither reads nor writes in scratch,
/// which so take no bytes there: what a generated region streams through
/// registers, the results of the nodes it leaves out and the reshapes it
/// reads as their sources, and what a sampler region's kernel leaves
/// untouched.
#[must_use]
pub fn spent_values(stage: &CompiledStage, region: &Region) -> Vec<u32> {
    if region.kind != crate::plan::RegionKind::Generated {
        return super::sampler::untouched_values(stage, region);
    }
    if super::validate::validate_generated_region(stage, region).is_err() {
        return Vec::new();
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = crate::codegen::op_view::result_bases(&ops);
    let direct = super::fused::analyze_direct_argmax(stage, region, &bases);
    let mut spent = unnamed_values(stage, region, &ops, &bases, &direct);
    let Some(kinds) = super::fused::stream_rows(stage, region) else {
        return spent;
    };
    let direct_topk = super::fused::analyze_direct_topk(stage);
    let claimed = direct.claimed();
    let streams = Streams::new(
        stage,
        region,
        &ops,
        &bases,
        &kinds,
        &claimed,
        &direct.skipped,
        &direct_topk,
    );
    let mut at = 0usize;
    while at < streams.order.len() {
        let node = streams.order[at];
        at += 1;
        if streams.skipped[node] != 0 {
            continue;
        }
        let op = &ops[node];
        let base = bases[node];
        if matches!(op.tag, tags::SCATTER_ADD | tags::SCATTER_SET)
            && super::fused::can_elide_scatter_base(stage, region, &ops, &bases, node, base)
            && !region.sinks.iter().any(|s| s.value == base)
        {
            spent.push(base);
            continue;
        }
        let Some(members) = streams.plan(at - 1) else {
            continue;
        };
        for member in &members {
            if matches!(
                member.role,
                Role::Map | Role::Broadcast | Role::Rng | Role::Intrinsic
            ) {
                let value = bases[member.node];
                if !streams.escapes(value, &members) {
                    spent.push(value);
                }
            }
        }
        at += members.len() - 1;
    }
    spent
}

/// A reduction's chunk partial stored at `at` of the row's work. Where the
/// row is split across blocks the others read it after the meeting
/// (`ptir_canonical_reduce_met`): stored at the device's coherence point
/// (`M1_LANE_PUT32`, agent scope), where every block's reads look, not in
/// the writing block's cache. On a device whose last-level caches are not
/// coherent with each other (MI300X's eight XCDs) a plain store waits there
/// unseen.
fn work_put(split: bool, at: &str, value: &str) -> String {
    if split {
        format!(
            "M1_LANE_PUT32(reinterpret_cast<m1_u32*>(stream_work + ({at})), __float_as_uint({value}))"
        )
    } else {
        format!("stream_work[{at}] = {value}")
    }
}
