//! A program rendered as functions: its phases fused into groups that fit
//! a PE together (`fused_phases`), each group one program whose single RPC
//! entry, `run`, calls the phases in order (a reducing group as a step
//! machine over its collectives), with only the buffers, globals and
//! helpers its phases name.

use super::*;

/// The CSL of a fused group's step machine: the phases' functions run
/// in order, each phase's steps after its function, every step handing
/// the next to its collective's callback; the last state rounds and
/// finishes the last reduce and ends the run.
fn render_chain(chain: &[(&str, Option<&FabricReduce>)], pe: &mut String) {
    // States: statements, then one collective; the trailing statements
    // (functions after the last collective, roundings, finishes) end
    // the run.
    let mut states: Vec<(Vec<String>, &FabricOp)> = Vec::new();
    let mut pending: Vec<String> = Vec::new();
    for (func, reduce) in chain {
        pending.push(format!("{func}();"));
        if let Some(r) = reduce {
            for step in &r.steps {
                let mut lines = std::mem::take(&mut pending);
                lines.extend(step.calls.iter().map(Guarded::render));
                states.push((lines, &step.op));
            }
            // The finish lays the merged words out, then they round.
            pending.extend(r.finish.iter().map(Guarded::render));
            for (name, n) in &r.round {
                pending.push(format!("k_round_bf16(@ptrcast([*]f32, &{name}), {n});"));
            }
        }
    }
    pe.push_str("var px: i32 = 0;\nvar st: u16 = 0;\nfn step_fn() void {\n  switch (st) {\n");
    for (i, (lines, op)) in states.iter().enumerate() {
        let _ = writeln!(pe, "    {i} => {{");
        for l in lines {
            let _ = writeln!(pe, "      {l}");
        }
        let _ = writeln!(pe, "      st = {};", i + 1);
        match op {
            FabricOp::Reduce { send, recv, count } => {
                let _ = writeln!(
                    pe,
                    "      mpi_x.reduce_fadds(0, @ptrcast([*]f32, &{send}), @ptrcast([*]f32, &{recv}), {count}, step_id);"
                );
            }
            FabricOp::Gather { send, recv, count } => {
                let _ = writeln!(
                    pe,
                    "      mpi_x.gather(0, @ptrcast([*]u32, &{send}), @ptrcast([*]u32, &{recv}), {count}, step_id);"
                );
            }
            FabricOp::Broadcast { buf, count } => {
                let _ = writeln!(
                    pe,
                    "      mpi_x.broadcast(0, @ptrcast([*]u32, &{buf}), {count}, step_id);"
                );
            }
            FabricOp::BroadcastFrom { root, buf, count } => {
                let _ = writeln!(
                    pe,
                    "      mpi_x.broadcast({root}, @ptrcast([*]u32, &{buf}), {count}, step_id);"
                );
            }
            // A handoff belongs to a whole fire's rows, never to a
            // function program's step machine.
            FabricOp::Handoff { .. } => {}
        }
        pe.push_str("    },\n");
    }
    let _ = writeln!(pe, "    {} => {{", states.len());
    for l in &pending {
        let _ = writeln!(pe, "      {l}");
    }
    pe.push_str("      sys_mod.unblock_cmd_stream();\n    },\n    else => {},\n  }\n}\n");
    pe.push_str("task step_task() void { step_fn(); }\n");
    pe.push_str("fn run() void {\n  mpi_x.init();\n  px = @as(i32, mpi_x.pe_id);\n  st = 0;\n  step_fn();\n}\n");
}

/// Every name the steps and the finish mention (for export selection).
pub(super) fn reduce_mentions(r: &FabricReduce) -> String {
    let mut text = String::new();
    for step in &r.steps {
        for g in &step.calls {
            text.push_str(&g.render());
            text.push('\n');
        }
        match &step.op {
            FabricOp::Reduce { send, recv, .. } | FabricOp::Gather { send, recv, .. } => {
                let _ = writeln!(text, "{send} {recv}");
            }
            FabricOp::Broadcast { buf, .. }
            | FabricOp::BroadcastFrom { buf, .. }
            | FabricOp::Handoff { buf, .. } => {
                let _ = writeln!(text, "{buf}");
            }
        }
    }
    for g in &r.finish {
        text.push_str(&g.render());
        text.push('\n');
    }
    for (name, _) in &r.round {
        text.push_str(name);
        text.push('\n');
    }
    text
}

impl Program {
    /// The helpers `body` needs, in declaration order: those it names, and
    /// those the chosen helpers name in turn (a kernel's own helpers).
    pub(super) fn helpers_for(&self, body: &str) -> Vec<&str> {
        let wanted = |text: &str, name: &str, h: &str| {
            mentions(text, name)
                || h.lines()
                    .any(|l| l.starts_with("fn ") && mentions(text, fn_name(l)))
        };
        let all: Vec<(&str, &str)> = self
            .helpers
            .iter()
            .map(|(n, h)| (n.as_str(), h.as_str()))
            .collect();
        let mut chosen: Vec<bool> = all.iter().map(|(n, h)| wanted(body, n, h)).collect();
        loop {
            let mut grew = false;
            for i in 0..all.len() {
                if chosen[i] {
                    continue;
                }
                let (n, h) = all[i];
                let by_chosen = all
                    .iter()
                    .zip(&chosen)
                    .any(|((_, src), c)| *c && wanted(src, n, h));
                if by_chosen {
                    chosen[i] = true;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        all.iter()
            .zip(&chosen)
            .filter(|(_, c)| **c)
            .map(|((_, h), _)| *h)
            .collect()
    }

    /// The phases in run order, adjacent fabric phases fused into one
    /// program when they run on the same PEs, hold every shared buffer
    /// the same way, carry no lane plan, and fit a PE together: a fused
    /// group runs its functions in sequence, so a buffer one writes and
    /// the next reads stays on the PE instead of crossing the host twice
    /// (the `fuse` switch at 0 keeps every phase apart).
    fn fused_phases(&self) -> Vec<Vec<&Phase>> {
        self.fused_phases_with(0)
    }

    /// The fused groups when every group also carries `extra` data words
    /// (a whole fire's kept slots, the same on every row).
    pub(crate) fn fused_phases_with(&self, extra: u64) -> Vec<Vec<&Phase>> {
        let fuse = crate::switches::get("fuse").is_none_or(|v| v != "0" && v != "false");
        let exports = self.exports();
        let mut groups: Vec<Vec<&Phase>> = Vec::new();
        let words_of = |group: &[&Phase]| -> u64 {
            let mut body = String::new();
            for p in group {
                p.func.render(&mut body);
            }
            let shard_of = |name: &str| -> Shard {
                group
                    .iter()
                    .flat_map(|p| p.shards.iter())
                    .find(|(n, _)| n == name)
                    .map_or(Shard::Whole, |(_, s)| *s)
            };
            let buffers: u64 = exports
                .iter()
                .filter(|e| mentions(&body, &e.name))
                .map(|e| {
                    Export {
                        shard: shard_of(&e.name),
                        ..e.clone()
                    }
                    .local()
                })
                .sum();
            let scratch: u64 = self
                .scratch
                .iter()
                .filter(|(name, _, _)| mentions(&body, name))
                .map(|(_, len, _)| *len)
                .sum();
            buffers + scratch
        };
        // A view is its own array on the PE, cut from its root before the
        // program runs: a phase naming a view cannot share a program with
        // one naming the view's root or another view of it, or one would
        // read what the other wrote through a stale copy.
        let text_of = |p: &Phase| {
            let mut t = String::new();
            p.func.render(&mut t);
            t
        };
        let roots_of = |text: &str| -> Vec<(String, String)> {
            self.views
                .iter()
                .filter(|v| mentions(text, &v.name))
                .map(|v| (v.name.clone(), v.root.clone()))
                .collect()
        };
        let views_clash = |a: &str, b: &str| {
            let clash = |x: &str, y: &str| {
                roots_of(x).iter().any(|(view, root)| {
                    mentions(y, root)
                        || self
                            .views
                            .iter()
                            .any(|w| w.root == *root && w.name != *view && mentions(y, &w.name))
                })
            };
            clash(a, b) || clash(b, a)
        };
        // The code a fused group links beside its data: the kernels it
        // names, the prelude, its own functions and the export plumbing,
        // at `CODE_WORDS_PER_LINE` (the data budget alone assumes one
        // phase's worth of code).
        let code_of = |group: &[&Phase]| -> u64 {
            let mut body = String::new();
            for p in group {
                p.func.render(&mut body);
            }
            let lines = |t: &str| {
                t.lines()
                    .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//"))
                    .count() as u64
            };
            let helpers: u64 = self.helpers_for(&body).iter().map(|h| lines(h)).sum();
            let used = exports.iter().filter(|e| mentions(&body, &e.name)).count() as u64;
            // The step machine: about eight lines a collective, and the
            // collectives module's own code and buffers once.
            let steps: u64 = group
                .iter()
                .filter_map(|p| p.reduce.as_ref())
                .map(|r| r.steps.len() as u64 + r.finish.len() as u64 + r.round.len() as u64)
                .sum();
            // A table-driven group also holds its op table: a row (about
            // twelve words) per call and per step.
            let calls: u64 = group
                .iter()
                .map(|p| p.func.body.calls().count() as u64)
                .sum();
            let source = lines(kernels_cerebras::library::prelude())
                + helpers
                + lines(&body)
                + 2 * used
                + 2 * group.len() as u64
                + 8 * steps
                + 30;
            if table_mode() {
                // A table program's operands are runtime values and its
                // dispatch unpacks them: measured over linked programs,
                // about `TABLE_CODE_WORDS_PER_LINE` a line plus a fixed
                // part (collectives included), the op table beside.
                return TABLE_CODE_WORDS_PER_LINE * source
                    + TABLE_FIXED_WORDS
                    + 12 * (calls + steps);
            }
            let module = if steps > 0 { COLLECTIVES_WORDS } else { 0 };
            CODE_WORDS_PER_LINE * source + module
        };
        // Reducing phases bring their own rectangle (one group of partials
        // a row); every reduce of a program must agree on it, and PE `p`
        // sits at the same place under any rectangle of the same size.
        let rects_agree = |g: &[&Phase], p: &Phase| {
            let rects: Vec<(u32, u32)> = g
                .iter()
                .chain(std::iter::once(&p))
                .filter_map(|q| q.reduce.as_ref().map(|r| r.rect))
                .collect();
            rects.windows(2).all(|w| w[0] == w[1])
        };
        // Lane phases bring their own plans: a buffer one plan lays out
        // must be no other plan's.
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
        // Two strided plans (the resident placement) lay every shared
        // buffer out the same way (pages by `page % pes`, lane lists
        // whole), so they may share.
        let strided = |l: &LanePlan| l.pages.as_ref().is_some_and(|p| p.strided);
        let lanes_apart = |g: &[&Phase], p: &Phase| {
            let Some(mine) = &p.lane else {
                return true;
            };
            let names = owned(mine);
            g.iter().filter_map(|q| q.lane.as_ref()).all(|other| {
                (strided(mine) && strided(other) && mine.pes == other.pes)
                    || owned(other).iter().all(|n| !names.contains(n))
            })
        };
        for phase in &self.phases {
            let text = text_of(phase);
            let joins = fuse
                && phase.host.is_none()
                && groups.last().is_some_and(|g| {
                    let head = g[0];
                    let group_text: String = g.iter().map(|p| text_of(p)).collect();
                    head.host.is_none()
                        && head.pes == phase.pes
                        && !views_clash(&text, &group_text)
                        && rects_agree(g, phase)
                        && lanes_apart(g, phase)
                        && phase.shards.iter().all(|(n, s)| {
                            g.iter()
                                .flat_map(|p| p.shards.iter())
                                .all(|(m, t)| m != n || t == s)
                        })
                });
            if joins {
                let last = groups.last_mut().expect("a group");
                last.push(phase);
                let (words, code) = (words_of(last), code_of(last));
                tracing::trace!(
                    phases = last.len(),
                    words,
                    code,
                    budget = FUSED_PE_WORDS,
                    "cerebras fusion: a group grows"
                );
                if words + code + extra > FUSED_PE_WORDS {
                    last.pop();
                    groups.push(vec![phase]);
                }
            } else {
                groups.push(vec![phase]);
            }
        }
        groups
    }

    /// Renders every phase as its own single-PE memcpy program: each one
    /// exports only the buffers its statements name, all as in-out (the host
    /// carries every buffer between phases). Globals, helpers and scratch
    /// arrays follow the phases that name them.
    pub fn render_phases(&self) -> Vec<Rendered> {
        self.render_phases_whole().0
    }

    /// `render_phases`, and whether the fire rendered as one whole-fire
    /// program (a fire that did not falls back to its fused groups; under
    /// kept pools that fallback would run on a pool of its own, so the
    /// caller refuses the fire instead).
    pub fn render_phases_whole(&self) -> (Vec<Rendered>, bool) {
        let exports = self.exports();
        if whole_fire()
            && let Some(rendered) = self.render_whole(&exports)
        {
            return (vec![rendered], true);
        }
        (self.render_groups(&exports), false)
    }

    fn render_groups(&self, exports: &[Export]) -> Vec<Rendered> {
        let exports = exports.to_vec();
        self.fused_phases()
            .into_iter()
            .map(|group| {
                let phase = group[0];
                let f = &phase.func;
                if let Some(host) = &phase.host {
                    let names = host.op.names();
                    let used: Vec<Export> = exports
                        .iter()
                        .filter(|e| names.contains(&e.name.as_str()))
                        .map(|e| Export {
                            role: Symbol::InOut,
                            ..e.clone()
                        })
                        .collect();
                    let views = self.views.iter().filter(|v| used.iter().any(|e| e.name == v.name)).cloned().collect();
                    return Rendered {
                        layout: String::new(),
                        pe: String::new(),
                        manifest: Manifest {
                            exports: used,
                            entry: "host".into(),
                            rect: (0, 0),
                            views,
                            lanes: Vec::new(),
                            host: Some(host.clone()),
                            table: None,
                            tables: Vec::new(),
                        },
                    };
                }
                // A fused group: every phase's function, run in order.
                let mut body = String::new();
                for p in &group {
                    p.func.render(&mut body);
                }
                // The reduces' epilogues name their partials and outputs:
                // they count as mentioned for export, scratch and helper
                // selection.
                let reduces: Vec<&FabricReduce> = group.iter().filter_map(|p| p.reduce.as_ref()).collect();
                let reduce = reduces.first().copied();
                for r in &reduces {
                    for l in reduce_mentions(r).lines() {
                        let _ = writeln!(body, "// {l}");
                    }
                }
                let shard_of = |name: &str| -> Shard {
                    group
                        .iter()
                        .flat_map(|p| p.shards.iter())
                        .find(|(n, _)| n == name)
                        .map_or(Shard::Whole, |(_, s)| *s)
                };
                // Every buffer travels both ways (the host holds the state)
                // but a packed one: a packed input (a weight) is never read
                // back, a packed in-out (a pool) comes back to be unpacked.
                let used: Vec<Export> = exports
                    .iter()
                    .filter(|e| mentions(&body, &e.name))
                    .map(|e| Export {
                        shard: shard_of(&e.name),
                        role: if e.packed { e.role } else { Symbol::InOut },
                        ..e.clone()
                    })
                    .collect();
                if table_mode()
                    && let Some(rendered) = self.render_table(&group, &used, &reduces, false)
                {
                    if let Some(t) = &rendered.manifest.table {
                        tracing::debug!(
                            phases = group.len(),
                            arena = t.arena_words,
                            ops = u64::from(t.ops_words),
                            lines = rendered
                                .pe
                                .lines()
                                .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//"))
                                .count(),
                            "cerebras fusion: a table program"
                        );
                        for (name, offset, words) in &t.arena {
                            tracing::trace!(offset, words, name, "cerebras fusion: arena");
                        }
                    }
                    return rendered;
                }
                let mut pe = String::new();
                pe.push_str("param memcpy_params;\n");
                // The PE's index in the rectangle (`p` of the shards).
                pe.push_str("param pe_id: u16;\n");
                if reduce.is_some() {
                    pe.push_str("param c2d_params;\n");
                }
                pe.push_str("const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n");
                pe.push_str("const math = @import_module(\"<math>\");\n");
                if reduce.is_some() {
                    pe.push_str(
                        "const mpi_x = @import_module(\"<collectives_2d/pe>\", .{ .dim_params = c2d_params.x, .queues = [2]u16{2, 4}, .dest_dsr_ids = [1]u16{1}, .src0_dsr_ids = [1]u16{1}, .src1_dsr_ids = [1]u16{1} });\n",
                    );
                    pe.push_str("const step_id: local_task_id = @get_local_task_id(15);\n");
                }
                pe.push('\n');
                pe.push_str(kernels_cerebras::library::prelude());
                pe.push('\n');
                for e in &used {
                    let _ = writeln!(pe, "var {}: [{}]{};", e.name, e.local(), e.elem);
                    let _ = writeln!(pe, "var {0}_ptr: [*]{1} = &{0};", e.name, e.elem);
                }
                for (name, len, elem) in &self.scratch {
                    if mentions(&body, name) {
                        let _ = writeln!(pe, "var {name} = @zeros([{len}]{elem});");
                    }
                }
                pe.push('\n');
                for g in &self.globals {
                    if global_name(g).is_some_and(|n| mentions(&body, n)) {
                        pe.push_str(g);
                        pe.push('\n');
                    }
                }
                pe.push('\n');
                for h in self.helpers_for(&body) {
                    pe.push_str(h);
                    pe.push('\n');
                }
                pe.push_str(&body);
                pe.push('\n');
                match reduce {
                    // The functions run with their collectives between them
                    // (a step machine), the last callback ending the run.
                    Some(_) => {
                        let chain: Vec<(&str, Option<&FabricReduce>)> = group
                            .iter()
                            .map(|p| (p.func.name.as_str(), p.reduce.as_ref()))
                            .collect();
                        render_chain(&chain, &mut pe);
                    }
                    None => {
                        pe.push_str("fn run() void {\n");
                        for p in &group {
                            let _ = writeln!(pe, "  {}();", p.func.name);
                        }
                        pe.push_str("  sys_mod.unblock_cmd_stream();\n}\n");
                    }
                }
                let _ = f;
                pe.push_str("comptime {\n");
                if reduce.is_some() {
                    pe.push_str("  @bind_local_task(step_task, step_id);\n");
                }
                for e in &used {
                    let _ = writeln!(pe, "  @export_symbol({0}_ptr, \"{0}\");", e.name);
                }
                pe.push_str("  @export_symbol(run);\n}\n");

                let pes = phase.pes.max(1);
                // PE `p` sits at `(p % w, p / w)` of a rectangle as tall as
                // fits: a row-major memcpy hands block `p` to that PE, and
                // the host copies stream one channel a row. A reducing phase
                // takes its own rectangle, one group of partials a row.
                let (w, h) = reduce.map_or_else(|| rect_of(pes), |r| r.rect);
                let mut layout = String::new();
                let _ = writeln!(
                    layout,
                    "const memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {w}, .height = {h} }});"
                );
                if reduce.is_some() {
                    layout.push_str("const c2d = @import_module(\"<collectives_2d/params>\");\n");
                }
                let _ = writeln!(layout, "\nlayout {{\n  @set_rectangle({w}, {h});");
                let _ = writeln!(
                    layout,
                    "  for (@range(i16, {h})) |y| {{\n    for (@range(i16, {w})) |x| {{"
                );
                if reduce.is_some() {
                    layout.push_str(
                        "      const params = c2d.get_params(@as(u16, x), @as(u16, y), .{ .x_colors = .{ @get_color(0), @get_color(1) }, .x_entrypoints = .{ @get_local_task_id(10), @get_local_task_id(11) }, .y_colors = .{ @get_color(4), @get_color(5) }, .y_entrypoints = .{ @get_local_task_id(12), @get_local_task_id(13) } });\n",
                    );
                    let _ = writeln!(layout, "      @set_tile_code(x, y, \"pe.csl\", .{{ .memcpy_params = memcpy.get_params(x), .pe_id = @as(u16, y * {w} + x), .c2d_params = params }});");
                } else {
                    let _ = writeln!(layout, "      @set_tile_code(x, y, \"pe.csl\", .{{ .memcpy_params = memcpy.get_params(x), .pe_id = @as(u16, y * {w} + x) }});");
                }
                layout.push_str("    }\n  }\n");
                for e in &used {
                    let _ = writeln!(layout, "  @export_name(\"{}\", [*]{}, true);", e.name, e.elem);
                }
                layout.push_str("  @export_name(\"run\", fn()void);\n}\n");
                let views = self.views.iter().filter(|v| used.iter().any(|e| e.name == v.name)).cloned().collect();
                Rendered {
                    layout,
                    pe,
                    manifest: Manifest {
                        exports: used,
                        entry: "run".into(),
                        rect: (w, h),
                        views,
                        lanes: group.iter().filter_map(|p| p.lane.clone()).collect(),
                        host: None,
                        table: None,
                        tables: Vec::new(),
                    },
                }
            })
            .collect()
    }

    /// The table-driven rendering of a fused group: `None` when a phase
    /// carries a statement that is not a kernel call, or an argument the
    /// table cannot carry (a pointer to a global), so the group renders as
    /// functions instead.
    /// A whole fire as one table-driven program: every phase on the same
    /// PEs (the resident placement), no host phase, no view, each buffer
    /// laid out one way throughout; `None` when the fire does not fit that
    /// shape, and the phases render as groups.
    /// Renders a single-PE memcpy program.
    pub fn render(&self) -> Rendered {
        let exports = self.exports();
        let entry = "run";

        let mut layout = String::new();
        layout.push_str("const memcpy = @import_module(\"<memcpy/get_params>\", .{ .width = 1, .height = 1 });\n\n");
        layout.push_str("layout {\n  @set_rectangle(1, 1);\n");
        layout.push_str(
            "  @set_tile_code(0, 0, \"pe.csl\", .{ .memcpy_params = memcpy.get_params(0), .pe_id = @as(u16, 0) });\n",
        );
        for e in &exports {
            let _ = writeln!(
                layout,
                "  @export_name(\"{}\", [*]{}, true);",
                e.name, e.elem
            );
        }
        let _ = writeln!(layout, "  @export_name(\"{entry}\", fn()void);");
        layout.push_str("}\n");

        let mut pe = String::new();
        pe.push_str("param memcpy_params;\nparam pe_id: u16;\n");
        pe.push_str("const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n");
        pe.push_str("const math = @import_module(\"<math>\");\n\n");
        pe.push_str(kernels_cerebras::library::prelude());
        pe.push('\n');
        for e in &exports {
            let n = e.rows as u64 * e.width as u64;
            match e.role {
                Symbol::Input | Symbol::InOut => {
                    let _ = writeln!(pe, "var {}: [{n}]{};", e.name, e.elem);
                }
                Symbol::Output => {
                    let _ = writeln!(pe, "var {} = @zeros([{n}]{});", e.name, e.elem);
                }
            }
            let _ = writeln!(pe, "var {0}_ptr: [*]{1} = &{0};", e.name, e.elem);
        }
        for (name, len, elem) in &self.scratch {
            let _ = writeln!(pe, "var {name} = @zeros([{len}]{elem});");
        }
        pe.push('\n');
        for g in &self.globals {
            pe.push_str(g);
            pe.push('\n');
        }
        pe.push('\n');
        for h in self.helpers.values() {
            pe.push_str(h);
            pe.push('\n');
        }
        for phase in &self.phases {
            phase.func.render(&mut pe);
            pe.push('\n');
        }
        let _ = writeln!(pe, "fn {entry}() void {{");
        for line in &self.prologue {
            let _ = writeln!(pe, "  {line}");
        }
        for phase in &self.phases {
            let _ = writeln!(pe, "  {}();", phase.func.name);
        }
        pe.push_str("  sys_mod.unblock_cmd_stream();\n}\n\n");
        pe.push_str("comptime {\n");
        for e in &exports {
            let _ = writeln!(pe, "  @export_symbol({0}_ptr, \"{0}\");", e.name);
        }
        let _ = writeln!(pe, "  @export_symbol({entry});");
        pe.push_str("}\n");

        Rendered {
            layout,
            pe,
            manifest: Manifest {
                exports,
                entry: entry.into(),
                rect: (1, 1),
                views: self.views.clone(),
                lanes: Vec::new(),
                host: None,
                table: None,
                tables: Vec::new(),
            },
        }
    }
}
