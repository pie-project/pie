//! The table-driven renderings of a program: a fused group as one
//! interpreter program (`render_table`), and a whole fire as one program
//! over a rectangle of `pes × rows` PEs, each row holding one fused
//! group's kernels and arena (`render_whole`).
//!
//! A row's binary holds its own kernels alone: the one `pe.csl` takes a
//! comptime `row` parameter that gates the dispatch cases, and the
//! compiler drops the kernels a row never names. Buffers a row produces
//! and a later row reads go down the columns as `collectives_2d`
//! broadcasts on the y dimension (`OP_HANDOFF`, rooted at the producing
//! row, every row taking part; a row that does not read the buffer lands
//! it in `handoff_scratch`).

use super::*;

/// One row of a table program, built before any text is emitted.
pub(super) struct RowBuild {
    /// The phases' rendered statements (what the row mentions).
    body: String,
    arena: Vec<(String, u64, u64)>,
    arena_words: u64,
    keep_chunks: u32,
    consts: Vec<(String, Vec<u32>)>,
    kernels: Vec<String>,
    ops: Vec<TableOp>,
    /// `ops_words` and `rows` once padded to a unified capacity.
    capacity: Option<(u32, u32)>,
    move_rows: u32,
    collectives: bool,
    used: Vec<Export>,
    lanes: Vec<LanePlan>,
    /// Phases in the row.
    phases: usize,
    /// Under a unified shape: the arena arrays are declared at the shape's
    /// size, not at the slots' reach, so every class's text is one.
    fixed: bool,
}

impl RowBuild {
    /// Words arena array `c` declares.
    fn declared(&self, table: &Table, c: u64) -> u64 {
        // A kept chunk reaches as far as the kept layout (the same on every
        // row and class); a work chunk under a unified shape reaches the
        // shape's size.
        if self.fixed && c >= u64::from(table.keep_chunks) {
            table.arena_words.saturating_sub(c * ARENA_CHUNK).clamp(1, ARENA_CHUNK)
        } else {
            table.chunk_words(c)
        }
    }
}

impl RowBuild {
    fn table(&self) -> Table {
        let own = self.ops.iter().map(|op| 1 + op_words(op)).sum::<u32>() + 1;
        let (ops_words, rows) = self
            .capacity
            .unwrap_or((own, self.ops.len().max(1) as u32));
        Table {
            kernels: self.kernels.clone(),
            ops_words,
            ops: self.ops.clone(),
            arena: self.arena.clone(),
            arena_words: self.arena_words,
            keep_chunks: self.keep_chunks,
            rows,
            move_rows: self.move_rows,
            consts: self.consts.clone(),
            phases: self.phases as u32,
        }
    }
}

/// Kept slots `(name, offset, words)` and the chunks they take.
type KeptLayout = (Vec<(String, u64, u64)>, u32);

/// A buffer one row hands down to later rows.
struct Handoff {
    root: usize,
    name: String,
    count: u64,
}

fn lines_of(text: &str) -> u64 {
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//"))
        .count() as u64
}

fn texts_of(group: &[&Phase]) -> Vec<String> {
    group
        .iter()
        .map(|p| {
            let mut t = String::new();
            p.func.render(&mut t);
            if let Some(r) = &p.reduce {
                t.push_str(&r.mentions());
            }
            t
        })
        .collect()
}

impl Program {
    /// A fire as one program: its phases in fused groups, one group a row
    /// of the rectangle, every row's PEs the fire's `pes` across. Refused
    /// (`None`) when a phase runs on the host, the phases disagree on their
    /// PEs or rectangle, a view is named, two lane plans lay one buffer out
    /// differently, a buffer is sharded two ways, a statement is not a call
    /// the table carries, or a row does not fit a PE.
    pub(super) fn render_whole(&self, exports: &[Export]) -> Option<Rendered> {
        let all_phases: Vec<&Phase> = self.phases.iter().collect();
        if all_phases.is_empty() || all_phases.iter().any(|p| p.host.is_some()) {
            trace_fusion("whole fire refused: a host phase");
            return None;
        }
        let pes = all_phases[0].pes;
        if all_phases.iter().any(|p| p.pes != pes) {
            trace_fusion("whole fire refused: phases on different PE counts");
            return None;
        }
        let w = pes.max(1);
        if all_phases
            .iter()
            .filter_map(|p| p.reduce.as_ref())
            .any(|r| r.rect != (w, 1))
        {
            trace_fusion("whole fire refused: a reduce off the row rectangle");
            return None;
        }
        let texts = texts_of(&all_phases);
        let all: String = texts.concat();
        if self.views.iter().any(|v| mentions(&all, &v.name)) {
            trace_fusion("whole fire refused: a view");
            return None;
        }
        // A buffer two lane plans lay out must be laid out alike (strided).
        let strided = |l: &LanePlan| l.pages.as_ref().is_some_and(|p| p.strided);
        let owned = |l: &LanePlan| -> Vec<String> {
            let mut names: Vec<String> = vec![l.header.clone(), l.slots.clone()];
            names.extend(l.banks.iter().map(|(n, ..)| n.clone()));
            names.extend(l.row_outputs.iter().map(|(n, ..)| n.clone()));
            names.extend(l.cols.iter().map(|c| c.name.clone()));
            if let Some(p) = &l.pages {
                names.extend(p.rewritten.iter().cloned());
                match &p.source {
                    PageSource::Csr { indptr, indices } => {
                        names.push(indptr.clone());
                        names.push(indices.clone());
                    }
                    PageSource::Rows { table } => names.push(table.clone()),
                }
            }
            names.retain(|n| !n.is_empty());
            names
        };
        let plans: Vec<&LanePlan> = all_phases.iter().filter_map(|p| p.lane.as_ref()).collect();
        for (i, a) in plans.iter().enumerate() {
            for b in &plans[i + 1..] {
                let shared = owned(a).iter().any(|n| owned(b).contains(n));
                if shared && !(strided(a) && strided(b) && a.pes == b.pes) {
                    trace_fusion("whole fire refused: two lane plans over one buffer");
                    return None;
                }
            }
        }
        // A buffer's one shard across the fire.
        let shard_of = |name: &str| -> Option<Shard> {
            let mut shard: Option<Shard> = None;
            for p in &all_phases {
                for (n, s) in &p.shards {
                    if n == name {
                        match shard {
                            Some(t) if t != *s => return None,
                            _ => shard = Some(*s),
                        }
                    }
                }
            }
            Some(shard.unwrap_or(Shard::Whole))
        };
        let mut used_all = Vec::new();
        for e in exports {
            if !mentions(&all, &e.name) {
                continue;
            }
            let Some(shard) = shard_of(&e.name) else {
                trace_fusion(&format!("whole fire refused: {} sharded two ways", e.name));
                return None;
            };
            used_all.push(Export {
                shard,
                ..e.clone()
            });
        }

        // The kept slots (weights, pools, constant tables): one layout over
        // the whole rectangle, the unified shape's when there is one.
        let unified = self.unified.as_ref().filter(|u| !u.rows.is_empty());
        let consts: Vec<(String, Vec<u32>)> = self
            .globals
            .iter()
            .filter_map(|g| parse_global_table(g))
            .filter(|(n, _)| mentions(&all, n))
            .collect();
        let kept = self.kept_of(&used_all, &consts, unified)?;
        let kept_words: u64 = kept.0.iter().map(|(_, _, w)| *w).sum();
        let pools: Vec<&Export> = used_all.iter().filter(|e| e.keep && e.role == Symbol::InOut).collect();
        // Every row carries the kept slots, and a row holding a pool its
        // move rows and scratch.
        let extra = kept_words
            + if pools.is_empty() {
                0
            } else {
                MOVE_SCRATCH + u64::from(MOVE_ROWS) * u64::from(1 + MOVE_ROW_WORDS)
            };
        // The rows: the fused groups, in phase order. A unified shape
        // fixes the partition.
        let mut groups: Vec<Vec<&Phase>> = match self.unified.as_ref().and_then(|u| u.row_phases.clone()) {
            Some(sizes) => {
                if sizes.iter().sum::<usize>() != all_phases.len() {
                    trace_fusion("whole fire refused: the unified partition does not match the phases");
                    return None;
                }
                let mut at = 0;
                sizes
                    .iter()
                    .map(|n| {
                        let g = all_phases[at..at + n].to_vec();
                        at += n;
                        g
                    })
                    .collect()
            }
            None => {
                let mut groups: Vec<Vec<&Phase>> = self.fused_phases_with(extra).into_iter().map(|g| g.to_vec()).collect();
                // A kept pool is never handed off, so the phases naming one
                // sit on one row: a boundary between phases that share a
                // pool moves back past them.
                let names = |p: &Phase| {
                    let mut t = String::new();
                    p.func.render(&mut t);
                    t
                };
                let mut g = 0;
                while g + 1 < groups.len() {
                    let shared = |a: &Phase, next: &[&Phase]| {
                        let ta = names(a);
                        pools
                            .iter()
                            .any(|e| mentions(&ta, &e.name) && next.iter().any(|b| mentions(&names(b), &e.name)))
                    };
                    while let Some(last) = groups[g].last().copied()
                        && shared(last, &groups[g + 1])
                    {
                        groups[g].pop();
                        groups[g + 1].insert(0, last);
                    }
                    if groups[g].is_empty() {
                        groups.remove(g);
                    } else {
                        g += 1;
                    }
                }
                groups
            }
        };
        groups.retain(|g| !g.is_empty());
        let h = groups.len() as u32;
        // The rows' shapes under a unified fire (none while only the
        // partition is fixed).
        let shapes: Vec<Option<&RowShape>> = match unified {
            Some(u) => {
                if u.rows.len() != groups.len() {
                    trace_fusion("whole fire refused: the unified shape has other rows");
                    return None;
                }
                u.rows.iter().map(Some).collect()
            }
            None => vec![None; groups.len()],
        };
        let group_texts: Vec<String> = groups.iter().map(|g| texts_of(g).concat()).collect();
        let on_row = |name: &str, g: usize| mentions(&group_texts[g], name);

        // The handoffs: a buffer the host does not upload to every reader
        // (anything but an input) that a later row reads too.
        let mut handoffs: Vec<Handoff> = Vec::new();
        if h > 1 {
            let scratch_words = |name: &str| -> Option<u64> {
                self.scratch
                    .iter()
                    .find(|(n, ..)| n == name)
                    .map(|(_, len, elem)| if *elem == "u64" { len * 2 } else { *len }.max(1))
            };
            for g in 0..groups.len() - 1 {
                for e in &used_all {
                    if e.role == Symbol::Input || e.keep || !on_row(&e.name, g) {
                        continue;
                    }
                    if (g + 1..groups.len()).any(|later| on_row(&e.name, later)) {
                        handoffs.push(Handoff {
                            root: g,
                            name: e.name.clone(),
                            count: e.local().max(1),
                        });
                    }
                }
                for (name, ..) in &self.scratch {
                    if !on_row(name, g) {
                        continue;
                    }
                    if (g + 1..groups.len()).any(|later| on_row(name, later)) {
                        handoffs.push(Handoff {
                            root: g,
                            name: name.clone(),
                            count: scratch_words(name)?,
                        });
                    }
                }
            }
        }
        let scratch_words = handoffs.iter().map(|h| h.count).max().unwrap_or(0);
        // A kept in-out (a pool) is never handed off: it must sit on one row.
        if h > 1 {
            for e in &used_all {
                if e.keep && e.role == Symbol::InOut && (0..groups.len()).filter(|g| on_row(&e.name, *g)).count() > 1 {
                    trace_fusion(&format!("whole fire refused: kept pool {} on several rows", e.name));
                    return None;
                }
            }
        }

        let mut rows: Vec<RowBuild> = Vec::new();
        for (g, group) in groups.iter().enumerate() {
            let used: Vec<Export> = used_all.iter().filter(|e| on_row(&e.name, g)).cloned().collect();
            let reduces: Vec<&FabricReduce> = group.iter().filter_map(|p| p.reduce.as_ref()).collect();
            // What this row hands off or receives lives across its phases.
            let pinned: Vec<String> = handoffs
                .iter()
                .filter(|hd| on_row(&hd.name, g))
                .map(|hd| hd.name.clone())
                .collect();
            let extra: Vec<(String, u64)> = if handoffs.iter().any(|hd| !on_row(&hd.name, g)) {
                vec![("handoff_scratch".to_string(), scratch_words)]
            } else {
                Vec::new()
            };
            let Some(mut row) = self.build_row(group, &used, &reduces, true, shapes[g], &kept, &consts, &pinned, &extra) else {
                trace_fusion(&format!("whole fire refused: row {g} does not render as a table"));
                return None;
            };
            // The handoffs around this row's ops: earlier rows' first (this
            // row receives), its own and later rows' after.
            let op_of = |hd: &Handoff| {
                let buf = if on_row(&hd.name, g) { hd.name.clone() } else { "handoff_scratch".to_string() };
                TableOp::Collective(FabricOp::Handoff {
                    root: hd.root as u32,
                    buf,
                    count: hd.count,
                })
            };
            let mut ops: Vec<TableOp> = handoffs.iter().filter(|hd| hd.root < g).map(op_of).collect();
            ops.append(&mut row.ops);
            ops.extend(handoffs.iter().filter(|hd| hd.root >= g).map(op_of));
            row.ops = ops;
            row.collectives |= !handoffs.is_empty();
            rows.push(row);
        }
        let rendered = self.emit_table(&rows, (w, h), !handoffs.is_empty())?;

        // Every row must fit a PE: its own kernels' code beside the arena
        // and table arrays, declared at the largest row's size.
        let tables: Vec<Table> = rows.iter().map(RowBuild::table).collect();
        let chunks = tables.iter().map(Table::chunks).max().unwrap_or(1);
        let data = (0..chunks)
            .map(|c| tables.iter().map(|t| t.chunk_words(c)).max().unwrap_or(1))
            .sum::<u64>()
            + u64::from(tables.iter().map(|t| t.ops_words).max().unwrap_or(1));
        let fixed = lines_of(crate::library::prelude()) + 60;
        let forced = std::env::var("PIE_CEREBRAS_WHOLE_FIRE").is_ok_and(|v| v == "force");
        for (g, row) in rows.iter().enumerate() {
            let helpers: u64 = self.helpers_for(&row.body).iter().map(|h| lines_of(h)).sum();
            let code = TABLE_CODE_WORDS_PER_LINE * (helpers + fixed + row.kernels.len() as u64) + TABLE_FIXED_WORDS;
            if std::env::var_os("PIE_CEREBRAS_TRACE_FUSION").is_some() {
                eprintln!(
                    "fusion: whole fire row {g} of {h}: {} phases, {} kernels, {data} data words ({} ops), {code} code words (budget {FUSED_PE_WORDS})",
                    row.phases,
                    row.kernels.len(),
                    tables[g].ops_words
                );
                if std::env::var_os("PIE_CEREBRAS_TRACE_ARENA").is_some() {
                    for (n, o, w) in &tables[g].arena {
                        eprintln!("fusion:   row {g} arena {o:>6} +{w:<6} {n}");
                    }
                }
            }
            if data + code > FUSED_PE_WORDS && !forced {
                trace_fusion(&format!("whole fire refused: row {g} over the budget"));
                return None;
            }
        }
        Some(rendered)
    }

    /// A fused group as one table-driven program on its own rectangle.
    pub(super) fn render_table(&self, group: &[&Phase], used: &[Export], reduces: &[&FabricReduce], whole: bool) -> Option<Rendered> {
        let unified = if whole { self.unified.as_ref().filter(|u| !u.rows.is_empty()) } else { None };
        let shape = unified.and_then(|u| u.rows.first());
        let mut body = String::new();
        for p in group {
            p.func.render(&mut body);
        }
        let consts: Vec<(String, Vec<u32>)> = self
            .globals
            .iter()
            .filter_map(|g| parse_global_table(g))
            .filter(|(n, _)| mentions(&body, n))
            .collect();
        let kept = self.kept_of(used, &consts, unified)?;
        let row = self.build_row(group, used, reduces, whole, shape, &kept, &consts, &[], &[])?;
        let pes = group[0].pes.max(1);
        let rect = reduces.first().map_or_else(|| rect_of(pes), |r| r.rect);
        self.emit_table(&[row], rect, false)
    }

    /// The kept layout a program follows: the unified shape's (every kept
    /// export and constant must have a slot there), else its own kept
    /// exports and constant tables laid from the arena's start.
    fn kept_of(
        &self,
        used: &[Export],
        consts: &[(String, Vec<u32>)],
        unified: Option<&Unified>,
    ) -> Option<KeptLayout> {
        let need: Vec<(String, u64)> = used
            .iter()
            .filter(|e| e.keep)
            .map(|e| (e.name.clone(), e.local().max(1)))
            .chain(consts.iter().map(|(n, w)| (n.clone(), w.len().max(1) as u64)))
            .collect();
        match unified {
            Some(u) => {
                for (name, words) in &need {
                    let Some((_, _, have)) = u.keep.iter().find(|(n, ..)| n == name) else {
                        trace_fusion(&format!("table refused: kept {name} off the unified layout"));
                        return None;
                    };
                    if have < words {
                        trace_fusion(&format!("table refused: kept {name} over its unified slot"));
                        return None;
                    }
                }
                Some((u.keep.clone(), u.keep_chunks))
            }
            None => Some(keep_layout(need)),
        }
    }

    /// The arena and the op table of one row (a fused group): the kept
    /// slots first (`kept`: the program's layout, the same on every row),
    /// then the buffers (by liveness when `liveness`, else in order), the
    /// calls as rows of the table. `pinned` buffers live across every
    /// phase; `extra` scratch slots are placed beside the program's own.
    #[allow(clippy::too_many_arguments)]
    fn build_row(
        &self,
        group: &[&Phase],
        used: &[Export],
        reduces: &[&FabricReduce],
        liveness: bool,
        unified: Option<&RowShape>,
        kept: &KeptLayout,
        consts: &[(String, Vec<u32>)],
        pinned: &[String],
        extra: &[(String, u64)],
    ) -> Option<RowBuild> {
        let texts = texts_of(group);
        let mut body = String::new();
        for p in group {
            p.func.render(&mut body);
        }
        for r in reduces {
            for l in r.mentions().lines() {
                let _ = writeln!(body, "// {l}");
            }
        }
        let mut arena: Vec<(String, u64, u64)> = Vec::new();
        let place = |name: &str, words: u64, at: &mut u64, arena: &mut Vec<(String, u64, u64)>| {
            // An array stays within its chunk.
            let chunk = *at / ARENA_CHUNK;
            if words <= ARENA_CHUNK && (*at + words - 1) / ARENA_CHUNK != chunk {
                *at = (chunk + 1) * ARENA_CHUNK;
            }
            arena.push((name.to_string(), *at, words));
            *at += words;
        };
        // The kept slots first, in arrays of their own (padded to the
        // chunk boundary), every one of the layout whether this row names
        // it or not, so every row and class declares one kept region and a
        // server leaves it where it is.
        let (keep_slots, keep_chunks) = kept;
        let keep_chunks = *keep_chunks;
        arena.extend(keep_slots.iter().cloned());
        let mut at = u64::from(keep_chunks) * ARENA_CHUNK;
        // The constant tables (the fire's, so every row's packed arena
        // fills every kept constant slot).
        let consts: Vec<(String, Vec<u32>)> = consts.to_vec();
        place("k_dummy", 2, &mut at, &mut arena);
        // Page moves ride the rows that hold a kept pool (a kept in-out),
        // under a unified shape or not, so the shapes a dry shell unites
        // already hold the move rows and scratch.
        let has_pool = used.iter().any(|e| e.keep && e.role == Symbol::InOut);
        let moving = liveness && keep_pools() && has_pool;
        if moving {
            place("move_scratch", MOVE_SCRATCH, &mut at, &mut arena);
        }
        // Without liveness the extra scratch is a slot like any other.
        if !liveness {
            for (name, words) in extra {
                place(name, *words, &mut at, &mut arena);
            }
        }
        let scratch: Vec<(&String, u64)> = self
            .scratch
            .iter()
            .filter(|(name, ..)| mentions(&body, name))
            .map(|(name, len, elem)| (name, if *elem == "u64" { len * 2 } else { *len }.max(1)))
            .collect();
        if liveness {
            // By liveness: a slot frees once its last phase has run and a
            // later buffer may take it. An input lives from the start, an
            // output to the end, an in-out (and a handed-off buffer)
            // throughout.
            let n = group.len();
            let span = |name: &str, role: Option<Symbol>| -> (usize, usize) {
                if pinned.iter().any(|p| p == name) {
                    return (0, n - 1);
                }
                let first = texts.iter().position(|t| mentions(t, name)).unwrap_or(0);
                let last = texts.iter().rposition(|t| mentions(t, name)).unwrap_or(n - 1);
                match role {
                    Some(Symbol::Input) => (0, last),
                    Some(Symbol::Output) => (first, n - 1),
                    Some(Symbol::InOut) => (0, n - 1),
                    None => (first, last),
                }
            };
            let mut items: Vec<(String, u64, usize, usize)> = used
                .iter()
                .filter(|e| !e.keep)
                .map(|e| {
                    let (a, b) = span(&e.name, Some(e.role));
                    (e.name.clone(), e.local().max(1), a, b)
                })
                .chain(scratch.iter().map(|(name, words)| {
                    let (a, b) = span(name, None);
                    (name.to_string(), *words, a, b)
                }))
                .collect();
            items.sort_by_key(|(_, _, a, b)| (*a, *b));
            // The extra scratch (a handoff landing) is used only outside
            // the row's phases (before its first, after its last), so it
            // may overlap the buffers that live strictly inside them: the
            // edge items (live at the first or the last phase) are laid
            // first, the scratch at the high-water mark, the inner items
            // over it.
            let (edge, inner): (Vec<_>, Vec<_>) = items.into_iter().partition(|(_, _, a, b)| *a == 0 || *b + 1 == n);
            let mut free: Vec<(u64, u64)> = vec![(at, u64::MAX)];
            let mut live: Vec<(usize, u64, u64)> = Vec::new();
            let mut high = at;
            let mut scratch_end = at;
            let first_fit = |items: Vec<(String, u64, usize, usize)>,
                                 free: &mut Vec<(u64, u64)>,
                                 live: &mut Vec<(usize, u64, u64)>,
                                 high: &mut u64,
                                 arena: &mut Vec<(String, u64, u64)>|
             -> Option<()> {
            for (name, words, a, b) in items {
                // Free what ended before this one starts.
                live.retain(|(end, off, len)| {
                    if *end < a {
                        free.push((*off, *off + *len));
                        false
                    } else {
                        true
                    }
                });
                free.sort_unstable();
                let mut merged: Vec<(u64, u64)> = Vec::new();
                for (lo, hi) in free.drain(..) {
                    match merged.last_mut() {
                        Some(last) if last.1 >= lo => last.1 = last.1.max(hi),
                        _ => merged.push((lo, hi)),
                    }
                }
                *free = merged;
                // First fit, within one array.
                let mut chosen = None;
                for (i, (lo, hi)) in free.iter().enumerate() {
                    let mut start = *lo;
                    if words <= ARENA_CHUNK && (start + words - 1) / ARENA_CHUNK != start / ARENA_CHUNK {
                        start = (start / ARENA_CHUNK + 1) * ARENA_CHUNK;
                    }
                    if start + words <= *hi {
                        chosen = Some((i, start));
                        break;
                    }
                }
                let (i, start) = chosen?;
                let (lo, hi) = free.remove(i);
                if lo < start {
                    free.push((lo, start));
                }
                if start + words < hi {
                    free.push((start + words, hi));
                }
                arena.push((name, start, words));
                live.push((b, start, words));
                *high = (*high).max(start + words);
            }
            Some(())
            };
            first_fit(edge, &mut free, &mut live, &mut high, &mut arena)?;
            if !extra.is_empty() {
                let mut scratch_at = high;
                for (name, words) in extra {
                    place(name, *words, &mut scratch_at, &mut arena);
                }
                scratch_end = scratch_at;
                free = vec![(high, u64::MAX)];
                live.clear();
            }
            first_fit(inner, &mut free, &mut live, &mut high, &mut arena)?;
            at = high.max(scratch_end);
        } else {
            for e in used.iter().filter(|e| !e.keep) {
                place(&e.name, e.local().max(1), &mut at, &mut arena);
            }
            for (name, words) in &scratch {
                place(name, *words, &mut at, &mut arena);
            }
        }
        let mut arena_words = at;
        if let Some(u) = unified {
            if at > u.arena_words {
                trace_fusion(&format!("table refused: the arena ({at} words) over the unified size ({})", u.arena_words));
                return None;
            }
            arena_words = u.arena_words;
        }
        let slot = |name: &str| arena.iter().find(|(n, ..)| n == name).map(|(_, o, _)| *o);
        // Every argument the table carries must resolve on the host.
        let carried = |a: &Arg| match a {
            Arg::Ptr(b) => slot(&b.name).is_some(),
            Arg::Scratch(name, _) => slot(name).is_some(),
            Arg::Expr(e) => {
                e.parse::<i64>().is_ok()
                    || e.starts_with("@ptrcast(")
                        && e.trim_end_matches(')').rsplit('&').next().is_some_and(|n| slot(n).is_some() || n.starts_with("k_dummy"))
                    || e.contains('[')
            }
            _ => true,
        };
        // The kernels: the model's list under a unified shape (a kernel off
        // it fails the whole-fire form), else this group's in order of use.
        let fixed = unified.is_some();
        let mut kernels: Vec<String> = unified.map(|u| u.kernels.clone()).unwrap_or_default();
        let kernel_id = |name: &str, kernels: &mut Vec<String>| -> Option<u32> {
            match kernels.iter().position(|k| k == name) {
                Some(i) => Some(i as u32),
                None if fixed => None,
                None => {
                    kernels.push(name.to_string());
                    Some((kernels.len() - 1) as u32)
                }
            }
        };
        let mut ops: Vec<TableOp> = Vec::new();
        let move_rows = if moving { MOVE_ROWS } else { 0 };
        for _ in 0..move_rows {
            ops.push(TableOp::Nop);
        }
        if moving && !fixed && !kernels.iter().any(|k| k == "k_copy") {
            kernels.push("k_copy".to_string());
        }
        let push_call = |g: &Guarded, ops: &mut Vec<TableOp>, kernels: &mut Vec<String>| -> Option<()> {
            if !g.call.args.iter().all(carried) {
                trace_fusion(&format!("table refused: call {} not carried", g.call.kernel));
                return None;
            }
            let Some(kernel) = kernel_id(&g.call.kernel, kernels) else {
                trace_fusion(&format!("table refused: kernel {} off the unified list", g.call.kernel));
                return None;
            };
            ops.push(TableOp::Call {
                root: g.root,
                on: None,
                kernel,
                args: g.call.args.clone(),
            });
            Some(())
        };
        for p in group {
            let calls: Vec<&Call> = p.func.body.calls().collect();
            if calls.len() != p.func.body.len() {
                trace_fusion("table refused: a statement that is not a call");
                return None;
            }
            for c in calls {
                push_call(&Guarded { root: false, call: c.clone() }, &mut ops, &mut kernels)?;
            }
            if let Some(r) = &p.reduce {
                for step in &r.steps {
                    for g in &step.calls {
                        push_call(g, &mut ops, &mut kernels)?;
                    }
                    ops.push(TableOp::Collective(step.op.clone()));
                }
                for g in &r.finish {
                    push_call(g, &mut ops, &mut kernels)?;
                }
                for (name, n) in &r.round {
                    let b = used.iter().find(|e| e.name == *name)?;
                    let buf = Buf {
                        name: b.name.clone(),
                        rows: b.rows,
                        width: b.width,
                        elem: b.elem,
                    };
                    push_call(
                        &Guarded::all("k_round_bf16", vec![Arg::Ptr(buf), Arg::Int(*n as i64)]),
                        &mut ops,
                        &mut kernels,
                    )?;
                }
            }
        }
        // The rows' words, and one zero word to end the table.
        let own_words = ops.iter().map(|op| 1 + op_words(op)).sum::<u32>() + 1;
        let own_rows = ops.len().max(1) as u32;
        let capacity = match unified {
            Some(u) => {
                if own_words > u.ops_words || own_rows > u.rows {
                    trace_fusion("table refused: the op table over the unified capacity");
                    return None;
                }
                Some((u.ops_words, u.rows))
            }
            None => None,
        };
        Some(RowBuild {
            body,
            arena,
            arena_words,
            keep_chunks,
            consts,
            kernels,
            ops,
            capacity,
            move_rows,
            // The collectives module: for the reduces, and for the page moves.
            collectives: !reduces.is_empty() || moving,
            used: used.to_vec(),
            lanes: group.iter().filter_map(|p| p.lane.clone()).collect(),
            phases: group.len(),
            fixed: unified.is_some(),
        })
    }

    /// The program text of one or several rows: one `pe.csl` whose
    /// comptime `row` parameter picks the row's dispatch, the arena and
    /// table arrays at the largest row's size, and the layout over `rect`.
    fn emit_table(&self, rows: &[RowBuild], rect: (u32, u32), handoffs: bool) -> Option<Rendered> {
        let (w, h) = rect;
        let tables: Vec<Table> = rows.iter().map(RowBuild::table).collect();
        let collectives = rows.iter().any(|r| r.collectives) || handoffs;
        let split = rows.len() > 1;
        let mut all_body = String::new();
        for r in rows {
            all_body.push_str(&r.body);
        }
        // Every row's kernels' parameter types, for the dispatch.
        let mut sigs: Vec<Vec<Vec<String>>> = Vec::new();
        for r in rows {
            let s: Option<Vec<Vec<String>>> = r.kernels.iter().map(|k| crate::library::signature(k)).collect();
            let Some(s) = s else {
                trace_fusion(&format!("table refused: a kernel without a signature among {:?}", r.kernels));
                return None;
            };
            sigs.push(s);
        }

        let mut pe = String::new();
        pe.push_str("param memcpy_params;\nparam pe_id: u16;\n");
        if split {
            pe.push_str("param row: u16;\n");
        }
        if collectives {
            pe.push_str("param c2d_params;\n");
        }
        pe.push_str("const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n");
        pe.push_str("const math = @import_module(\"<math>\");\n");
        if collectives {
            pe.push_str(
                "const mpi_x = @import_module(\"<collectives_2d/pe>\", .{ .dim_params = c2d_params.x, .queues = [2]u16{2, 4}, .dest_dsr_ids = [1]u16{1}, .src0_dsr_ids = [1]u16{1}, .src1_dsr_ids = [1]u16{1} });\n",
            );
            if handoffs {
                pe.push_str(
                    "const mpi_y = @import_module(\"<collectives_2d/pe>\", .{ .dim_params = c2d_params.y, .queues = [2]u16{3, 5}, .dest_dsr_ids = [1]u16{2}, .src0_dsr_ids = [1]u16{2}, .src1_dsr_ids = [1]u16{2} });\n",
                );
            }
            pe.push_str("const step_id: local_task_id = @get_local_task_id(15);\n");
        }
        pe.push('\n');
        pe.push_str(crate::library::prelude());
        pe.push('\n');
        // The arena and table arrays, at the largest row's size.
        let chunks = tables.iter().map(Table::chunks).max().unwrap_or(1);
        for c in 0..chunks {
            let words = rows.iter().zip(&tables).map(|(r, t)| r.declared(t, c)).max().unwrap_or(1);
            let _ = writeln!(pe, "var arena{c}: [{words}]u32;");
            let _ = writeln!(pe, "var arena{c}_ptr: [*]u32 = &arena{c};");
        }
        let ops_words = tables.iter().map(|t| t.ops_words).max().unwrap_or(1);
        let _ = writeln!(pe, "var ops: [{ops_words}]i32;");
        pe.push_str("var ops_ptr: [*]i32 = &ops;\n");
        // Every kept slot is a symbol of its own (the same slot on every
        // row): the host uploads it once per server and never reads it.
        let keep_chunks = tables.first().map_or(0, |t| t.keep_chunks);
        let kept: Vec<(String, u64, u64)> = tables
            .first()
            .map(|t| {
                t.arena
                    .iter()
                    .filter(|(_, off, _)| *off < u64::from(keep_chunks) * ARENA_CHUNK)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        for i in 0..kept.len() {
            let _ = writeln!(pe, "var kp{i}: [*]u32 = &arena0;");
        }
        // Pointed into the arena at `init` (a comptime pointer cast is not
        // supported); the host launches `init` before any copy.
        pe.push_str("fn init() void {\n");
        for (i, (_, off, _)) in kept.iter().enumerate() {
            let c = off / ARENA_CHUNK;
            let _ = writeln!(pe, "  kp{i} = @ptrcast([*]u32, &arena{c}[{}]);", off - c * ARENA_CHUNK);
        }
        pe.push_str("  sys_mod.unblock_cmd_stream();\n}\n\n");
        let is_const = |n: &str| rows.iter().any(|r| r.consts.iter().any(|(c, _)| c == n));
        for g in &self.globals {
            if global_name(g).is_some_and(|n| mentions(&all_body, n) && !is_const(n)) {
                pe.push_str(g);
                pe.push('\n');
            }
        }
        pe.push('\n');
        for h in self.helpers_for(&all_body) {
            pe.push_str(h);
            pe.push('\n');
        }
        // Arena pointers by offset, one accessor an element type.
        for (name, elem) in [("pf", "f32"), ("pi", "i32"), ("pu", "u32"), ("pq", "u64")] {
            let _ = writeln!(pe, "fn {name}(o: i32) [*]{elem} {{");
            for c in 0..chunks {
                let lo = c * ARENA_CHUNK;
                let cond = if c + 1 == chunks { String::new() } else { format!("if (o < {}) ", lo + ARENA_CHUNK) };
                let _ = writeln!(
                    pe,
                    "  {cond}{{ var p: [*]{elem} = @ptrcast([*]{elem}, &arena{c}[o - {lo}]); return p; }}"
                );
            }
            pe.push_str("}\n");
        }
        // The dispatch: opcode to kernel, arguments by their declared types,
        // the row's own cases alone (the opcode switch wants a u16; an i32
        // switch does not compile).
        pe.push_str("fn run_call(k: u16, a: [*]i32) void {\n");
        for (g, (r, sig)) in rows.iter().zip(&sigs).enumerate() {
            if split {
                let _ = writeln!(pe, "  if (row == {g}) {{");
            }
            pe.push_str("  switch (k) {\n");
            for (i, (k, sig)) in r.kernels.iter().zip(sig).enumerate() {
                let args: Vec<String> = sig
                    .iter()
                    .enumerate()
                    .map(|(j, t)| match t.as_str() {
                        "[*]f32" => format!("pf(a[{j}])"),
                        "[*]i32" => format!("pi(a[{j}])"),
                        "[*]u32" => format!("pu(a[{j}])"),
                        "[*]u64" => format!("pq(a[{j}])"),
                        "i32" => format!("a[{j}]"),
                        "u32" => format!("@bitcast(u32, a[{j}])"),
                        "f32" => format!("@bitcast(f32, a[{j}])"),
                        "bool" => format!("(a[{j}] != 0)"),
                        other => format!("@as({other}, a[{j}])"),
                    })
                    .collect();
                let _ = writeln!(pe, "    {i} => {{ {k}({}); }},", args.join(", "));
            }
            pe.push_str("    else => {},\n  }\n");
            if split {
                pe.push_str("  }\n");
            }
        }
        pe.push_str("}\n");
        pe.push_str("var px: i32 = 0;\nvar pc: i32 = 0;\n");
        let _ = writeln!(pe, "fn step_fn() void {{\n  while (pc < {ops_words}) {{\n    var len: i32 = ops[pc];\n    if (len == 0) {{ break; }}\n    var base: i32 = pc + 1;\n    var k: i32 = ops[base];\n    pc = base + len;");
        if collectives {
            let _ = writeln!(pe, "    if (k == {OP_REDUCE}) {{ mpi_x.reduce_fadds(0, pf(ops[base + 1]), pf(ops[base + 2]), @as(u16, ops[base + 3]), step_id); return; }}");
            let _ = writeln!(pe, "    if (k == {OP_GATHER}) {{ mpi_x.gather(0, pu(ops[base + 1]), pu(ops[base + 2]), @as(u16, ops[base + 3]), step_id); return; }}");
            let _ = writeln!(pe, "    if (k == {OP_BROADCAST}) {{ mpi_x.broadcast(0, pu(ops[base + 1]), @as(u16, ops[base + 2]), step_id); return; }}");
            let _ = writeln!(pe, "    if (k == {OP_BROADCAST_FROM}) {{ mpi_x.broadcast(@as(u16, ops[base + 1]), pu(ops[base + 2]), @as(u16, ops[base + 3]), step_id); return; }}");
        }
        if handoffs {
            let _ = writeln!(pe, "    if (k == {OP_HANDOFF}) {{ mpi_y.broadcast(@as(u16, ops[base + 1]), pu(ops[base + 2]), @as(u16, ops[base + 3]), step_id); return; }}");
        }
        let _ = writeln!(pe, "    if (k == {OP_NOP}) {{ continue; }}");
        let _ = writeln!(pe, "    if (k == {OP_ON}) {{ if (px == ops[base + 1]) {{ run_call(@as(u16, ops[base + 2]), @ptrcast([*]i32, &ops[base + 3])); }} continue; }}");
        let _ = writeln!(pe, "    if (k == {OP_ROOT}) {{ if (px == 0) {{ run_call(@as(u16, ops[base + 1]), @ptrcast([*]i32, &ops[base + 2])); }} continue; }}");
        pe.push_str("    run_call(@as(u16, k), @ptrcast([*]i32, &ops[base + 1]));\n  }\n  sys_mod.unblock_cmd_stream();\n}\n");
        pe.push_str("task step_task() void { step_fn(); }\n");
        pe.push_str("fn run() void {\n");
        if collectives {
            pe.push_str("  mpi_x.init();\n  px = @as(i32, mpi_x.pe_id);\n");
            if handoffs {
                pe.push_str("  mpi_y.init();\n");
            }
        } else {
            pe.push_str("  px = @as(i32, pe_id);\n");
        }
        pe.push_str("  pc = 0;\n  step_fn();\n}\n");
        pe.push_str("comptime {\n");
        if collectives {
            pe.push_str("  @bind_local_task(step_task, step_id);\n");
        }
        for c in 0..chunks {
            let _ = writeln!(pe, "  @export_symbol(arena{c}_ptr, \"arena{c}\");");
        }
        for (i, (name, ..)) in kept.iter().enumerate() {
            let _ = writeln!(pe, "  @export_symbol(kp{i}, \"{name}\");");
        }
        pe.push_str("  @export_symbol(ops_ptr, \"ops\");\n  @export_symbol(init);\n  @export_symbol(run);\n}\n");

        let mut layout = String::new();
        let _ = writeln!(
            layout,
            "const memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {w}, .height = {h} }});"
        );
        if collectives {
            layout.push_str("const c2d = @import_module(\"<collectives_2d/params>\");\n");
        }
        let _ = writeln!(layout, "\nlayout {{\n  @set_rectangle({w}, {h});");
        let _ = writeln!(layout, "  for (@range(i16, {h})) |y| {{\n    for (@range(i16, {w})) |x| {{");
        let row_param = if split { ", .row = @as(u16, y)" } else { "" };
        if collectives {
            layout.push_str(
                "      const params = c2d.get_params(@as(u16, x), @as(u16, y), .{ .x_colors = .{ @get_color(0), @get_color(1) }, .x_entrypoints = .{ @get_local_task_id(10), @get_local_task_id(11) }, .y_colors = .{ @get_color(4), @get_color(5) }, .y_entrypoints = .{ @get_local_task_id(12), @get_local_task_id(13) } });\n",
            );
            let _ = writeln!(layout, "      @set_tile_code(x, y, \"pe.csl\", .{{ .memcpy_params = memcpy.get_params(x), .pe_id = @as(u16, y * {w} + x){row_param}, .c2d_params = params }});");
        } else {
            let _ = writeln!(layout, "      @set_tile_code(x, y, \"pe.csl\", .{{ .memcpy_params = memcpy.get_params(x), .pe_id = @as(u16, y * {w} + x){row_param} }});");
        }
        layout.push_str("    }\n  }\n");
        for c in 0..chunks {
            let _ = writeln!(layout, "  @export_name(\"arena{c}\", [*]u32, true);");
        }
        for (name, ..) in &kept {
            let _ = writeln!(layout, "  @export_name(\"{name}\", [*]u32, true);");
        }
        layout.push_str("  @export_name(\"ops\", [*]i32, true);\n  @export_name(\"init\", fn()void);\n  @export_name(\"run\", fn()void);\n}\n");

        // The exports: every row's, once each (the first row's shard; the
        // whole fire checked they agree).
        let mut exports: Vec<Export> = Vec::new();
        for r in rows {
            for e in &r.used {
                if !exports.iter().any(|x| x.name == e.name) {
                    exports.push(e.clone());
                }
            }
        }
        let views = self.views.iter().filter(|v| exports.iter().any(|e| e.name == v.name)).cloned().collect();
        let (table, tables) = if split { (None, tables) } else { (tables.into_iter().next(), Vec::new()) };
        Some(Rendered {
            layout,
            pe,
            manifest: Manifest {
                exports,
                entry: "run".into(),
                rect: (w, h),
                views,
                lanes: rows.iter().flat_map(|r| r.lanes.iter().cloned()).collect(),
                host: None,
                table,
                tables,
            },
        })
    }
}
