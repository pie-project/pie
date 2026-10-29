//! The wafer programs' PE sources, generated from the kernels in `csl/`.
//!
//! serve2d and serve2d_pipe do not keep their own copies of the decoder layer,
//! the head or the VM. Each per-PE program is a shared kernel (`csl/kernels`,
//! `csl/vm`) patched at checked anchors, with the design's `*_extra.csl`
//! appended and dummy exports for every name another program exports (memcpy
//! needs every PE to export every name). This is a port of the designs'
//! `sync.py` generators, and gives byte-identical output; the tests pin it.
//! A missing anchor is an error, not a silently different program.

type R<T> = std::result::Result<T, String>;

/// One source file of a program: its file name in the build directory and its text.
pub type File = (&'static str, String);

mod src {
    pub const QWEN2D: &str = include_str!("csl/kernels/qwen2d.csl");
    pub const QWEN2D_PAGED: &str = include_str!("csl/kernels/qwen2d_paged.csl");
    pub const QWEN2D_PRING: &str = include_str!("csl/kernels/qwen2d_pring.csl");
    pub const HEAD2D_V9: &str = include_str!("csl/kernels/head2d_v9.csl");
    pub const QWEN2D_PAGED8B: &str = include_str!("csl/kernels/qwen2d_paged8b.csl");
    pub const HEAD2D_V10: &str = include_str!("csl/kernels/head2d_v10.csl");
    pub const VM_HM: &str = include_str!("csl/vm/guest4hm_vm.csl");
    pub const VM_HS: &str = include_str!("csl/vm/guest4hs_vm.csl");

    pub mod s2d {
        pub const LAYOUT: &str = include_str!("csl/serve2d/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve2d/paths.csl");
        pub const RT: &str = include_str!("csl/serve2d/rt.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve2d/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve2d/head_extra.csl");
        pub const VM_EXTRA: &str = include_str!("csl/serve2d/vm_extra.csl");
        pub const STUB_BASE: &str = include_str!("csl/serve2d/stub_base.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve2d/filler_base.csl");
    }

    pub mod s8b {
        pub const LAYOUT: &str = include_str!("csl/serve8b/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve8b/paths.csl");
        pub const RT: &str = include_str!("csl/serve8b/rt.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve8b/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve8b/head_extra.csl");
        pub const EMBED_BASE: &str = include_str!("csl/serve8b/embed_base.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve8b/filler_base.csl");
    }

    pub mod pipe {
        pub const LAYOUT: &str = include_str!("csl/serve2d_pipe/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve2d_pipe/paths.csl");
        pub const RT: &str = include_str!("csl/serve2d_pipe/rt.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve2d_pipe/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve2d_pipe/head_extra.csl");
        pub const VM_EXTRA: &str = include_str!("csl/serve2d_pipe/vm_extra.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve2d_pipe/filler_base.csl");
    }
}

/// The program designs PIE builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Design {
    /// Grids of qwen2d / qwen2d_paged layers chained on one head (serve2d).
    Serve2d,
    /// The paged layer as a pipeline ring with the head as its stage 0 (serve2d_pipe).
    Serve2dPipe,
    /// Qwen3-8B: qwen2d_paged8b layer grids in strips, head2d v10 and the
    /// embedding strip (serve8b).
    Serve8b,
}

impl Design {
    pub fn name(self) -> &'static str {
        match self {
            Design::Serve2d => "serve2d",
            Design::Serve2dPipe => "serve2d_pipe",
            Design::Serve8b => "serve8b",
        }
    }

    pub fn parse(name: &str) -> Option<Design> {
        match name {
            "serve2d" => Some(Design::Serve2d),
            "serve2d_pipe" => Some(Design::Serve2dPipe),
            "serve8b" => Some(Design::Serve8b),
            _ => None,
        }
    }

    /// Every file of the program's build directory, `layout.csl` first.
    pub fn files(self) -> R<Vec<File>> {
        match self {
            Design::Serve2d => serve2d(),
            Design::Serve2dPipe => serve2d_pipe(),
            Design::Serve8b => serve8b(),
        }
    }
}

// ---------------------------------------------------------------- text helpers

/// `old` replaced by `new`; `old` must occur exactly once.
fn patch(text: &str, old: &str, new: &str, what: &str) -> R<String> {
    let n = text.matches(old).count();
    if n != 1 {
        return Err(format!("gen: anchor for '{what}' found {n} times"));
    }
    Ok(text.replacen(old, new, 1))
}

/// The names a source exports (`@export_symbol(ptr, "name")`).
fn exports(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("@export_symbol(") {
        rest = &rest[i + "@export_symbol(".len()..];
        let ident_len = rest.bytes().take_while(|b| b.is_ascii_alphanumeric() || *b == b'_').count();
        if ident_len == 0 {
            continue;
        }
        let after = &rest[ident_len..];
        if let Some(q) = after.strip_prefix(", \"") {
            let name_len = q.bytes().take_while(|b| b.is_ascii_alphanumeric() || *b == b'_').count();
            if name_len > 0 && q[name_len..].starts_with("\")") {
                out.push(q[..name_len].to_string());
            }
        }
    }
    out
}

/// An insertion-ordered name -> type map (Python dict semantics: updating a
/// key keeps its place).
#[derive(Clone, Default)]
struct Names(Vec<(&'static str, &'static str)>);

impl Names {
    fn of(items: &[(&'static str, &'static str)]) -> Names {
        let mut n = Names::default();
        n.update(items);
        n
    }

    fn update(&mut self, items: &[(&'static str, &'static str)]) {
        for &(k, v) in items {
            match self.0.iter_mut().find(|(kk, _)| *kk == k) {
                Some(e) => e.1 = v,
                None => self.0.push((k, v)),
            }
        }
    }

    fn with(&self, items: &[(&'static str, &'static str)]) -> Names {
        let mut n = self.clone();
        n.update(items);
        n
    }
}

/// The start of the block `\n(//[^\n]*\n)*<head>`: the newline before the
/// comment lines directly above `head` (or before `head` itself).
fn comment_block_start(s: &str, head_at: usize) -> Option<usize> {
    if head_at == 0 || s.as_bytes()[head_at - 1] != b'\n' {
        return None;
    }
    let mut start = head_at - 1;
    loop {
        // The line ending at `start` (exclusive).
        let line_start = s[..start].rfind('\n').map(|i| i + 1)?;
        if s[line_start..start].starts_with("//") {
            start = line_start - 1;
        } else {
            return Some(start);
        }
    }
}

/// The range of `\n(//[^\n]*\n)*<head>.*?\n}\n`, for the first `head` (at a
/// line start) that has one.
fn fn_block(s: &str, head: &str) -> Option<(usize, usize)> {
    let mut from = 0;
    while let Some(i) = s[from..].find(head) {
        let at = from + i;
        if let Some(start) = comment_block_start(s, at) {
            let end = at + s[at..].find("\n}\n")? + 3;
            return Some((start, end));
        }
        from = at + 1;
    }
    None
}

/// `re.sub(r"\n(\s*)<call>", r"\n\1<call>\n\1<extra>", s)` and the match count.
fn after_each_call(s: &str, call: &str, extra: &str) -> (String, usize) {
    let mut out = String::with_capacity(s.len() + 256);
    let mut last = 0;
    let mut n = 0;
    let mut from = 0;
    while let Some(i) = s[from..].find(call) {
        let at = from + i;
        // The maximal whitespace run before `call`; the match starts at its first newline.
        let ws_start = s[..at]
            .char_indices()
            .rev()
            .take_while(|(_, c)| c.is_whitespace())
            .last()
            .map(|(j, _)| j)
            .unwrap_or(at);
        match s[ws_start..at].find('\n') {
            Some(k) if ws_start >= last => {
                let nl = ws_start + k;
                let group = &s[nl + 1..at];
                out.push_str(&s[last..nl]);
                out.push('\n');
                out.push_str(group);
                out.push_str(call);
                out.push('\n');
                out.push_str(group);
                out.push_str(extra);
                last = at + call.len();
                n += 1;
            }
            _ => {}
        }
        from = at + call.len();
    }
    out.push_str(&s[last..]);
    (out, n)
}

// ---------------------------------------------------------------- serve2d

const COPY_FNS: &str = "fn copy_page(src: u32, dst: u32) void { sys_mod.unblock_cmd_stream(); }\n\
fn copy_move(src: u32, dst: u32) void { sys_mod.unblock_cmd_stream(); }\n\
comptime {\n  @export_symbol(copy_page);\n  @export_symbol(copy_move);\n}\n";

const S2D_GRID_NAMES: &[(&str, &str)] = &[
    ("x", "f32"), ("gamma1", "f32"), ("gamma2", "f32"), ("wqkv", "WS"), ("qkn_w", "f32"),
    ("inv_freq", "f32"), ("kc", "KS"), ("vc", "KS"), ("wo", "WS"), ("wgu", "WS"), ("wd", "WS"),
    ("prof", "u32"), ("fallbacks", "u32"), ("stamps", "u32"),
    ("tab", "u32"), ("wqa", "f32"), ("wqv", "f32"), ("wv", "f32"), ("rot_c", "f32"), ("rot_s", "f32"),
    ("dims", "u16"),
];
const S2D_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"), ("et", "u32"), ("l2", "f32"), ("force", "u32"), ("lmax", "f32"), ("gmode", "u32"),
    ("vm_tok", "f32"), ("vin", "f32"), ("xin", "f32"), ("gam", "f32"), ("y", "f32"), ("emb", "f32"),
    ("tok", "f32"), ("prof", "u32"), ("in_tokens", "u32"), ("out_tokens", "u32"), ("yscale", "f32"),
    ("hlogits", "f32"),
];
const S2D_VM_NAMES: &[(&str, &str)] = &[
    ("gwide", "f32"), ("gpool", "u32"), ("gops", "u32"), ("gout", "u32"), ("gprof", "u32"),
    ("gostep", "u32"), ("gring", "u32"), ("gcycles", "u32"), ("gvalid", "u32"), ("gtok_home", "u32"),
];

/// serve2d's dummies: one 1-word array a type, exported under each missing name.
fn s2d_dummies(names: &Names, own: &[String]) -> String {
    let mut lines = Vec::new();
    let mut types: Vec<(&str, String)> = Vec::new();
    for &(n, ty) in &names.0 {
        if own.iter().any(|o| o == n) {
            continue;
        }
        let k = match types.iter().find(|(t, _)| *t == ty) {
            Some((_, k)) => k.clone(),
            None => {
                let k = format!("s2d_dum{}", types.len());
                types.push((ty, k.clone()));
                k
            }
        };
        lines.push(format!("  @export_symbol({k}_ptr, \"{n}\");"));
    }
    let decls: Vec<String> = types
        .iter()
        .map(|(ty, k)| format!("var {k} = @zeros([1]{ty});\nvar {k}_ptr: [*]{ty} = &{k};"))
        .collect();
    format!(
        "// serve2d: the other programs' exported names.\n{}\ncomptime {{\n{}\n}}\n",
        decls.join("\n"),
        lines.join("\n")
    )
}

const S2D_GRID_PARAMS: &str = "// serve2d: the grids' arrangement, the head's feeder row, the head mode, the\n\
// head's VM column (count, place in a block), bisecting.\n\
param GW: u16;\nparam GH: u16;\nparam NL: u16;\nparam F: u16;\nparam HEADMODE: u16;\n\
param S2D_VMC: u16;\nparam S2D_VMAT: u16;\nparam S2D_DBG: u16;\n";

fn s2d_variant(kernel: &str) -> String {
    match kernel {
        "qwen2d" => format!(
            "const S2D_Q2C = c_gb;             // queue 2's own color (the segment broadcast)\n\
const S2D_XN: u16 = Hc;           // x's words a PE\n\
// qwen2d has no paged pool: the copy launches only return.\n{COPY_FNS}"
        ),
        _ => "const PX: u16 = P;                // the square grid's sides\n\
const PY: u16 = P;\n\
// Queue 2: free in qwen2d_paged (its page copies ride on c_bc); serve2d's.\n\
const q_gb_in = @get_input_queue(2);\n\
const q_gb_out = @get_output_queue(2);\n\
const S2D_Q2C = c_cp;             // its resting color (c_cp is unused there)\n\
comptime {\n  @initialize_queue(q_gb_in, .{ .color = c_cp });\n  @initialize_queue(q_gb_out, .{ .color = c_cp });\n}\n\
const S2D_XN: u16 = B * Hc;       // x's words a PE: B lanes of Hc, lane-major\n\
const WS = f32;                   // its weights and KV (the exports' types)\n\
const KS = f32;\n"
            .to_string(),
    }
}

fn s2d_grid(kernel: &str, src: &str) -> R<String> {
    let mut s = patch(src, "param c_r1_0: color;\n", &format!("{S2D_GRID_PARAMS}param c_r1_0: color;\n"), "params")?;
    s = patch(&s, "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n", "  s2d_place();\n", "init coordinates")?;
    s = patch(
        &s,
        "  route_gbcast();\n  axis_y();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "  route_gbcast();\n  axis_y();\n  s2d_routes();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "init routes",
    )?;
    // compute(steps[, start_pos]) is replaced; its per-pass prologue is kept.
    let (start, end, body) = {
        let mut found = None;
        let mut from = 0;
        while let Some(i) = s[from..].find("fn compute(steps: u32") {
            let at = from + i;
            let rest = &s[at + "fn compute(steps: u32".len()..];
            let hdr = [") void {\n", ", start_pos: u32) void {\n"].into_iter().find(|h| rest.starts_with(h));
            if let (Some(h), Some(st)) = (hdr, comment_block_start(&s, at)) {
                let body_at = at + "fn compute(steps: u32".len() + h.len();
                let close = s[body_at..].find("\n}\n").ok_or_else(|| format!("gen: {kernel}: compute() unterminated"))?;
                found = Some((st, body_at + close + 3, s[body_at..body_at + close].to_string()));
                break;
            }
            from = at + 1;
        }
        found.ok_or_else(|| format!("gen: {kernel}/pe.csl changed: compute() not found"))?
    };
    let open = "  while (i < steps) : (i += 1) {\n";
    let close = "\n    timing = i > 0;\n";
    let lo = body.find(open).ok_or_else(|| format!("gen: {kernel}: compute()'s per-pass prologue not found"))? + open.len();
    let hi = lo + body[lo..].find(close).ok_or_else(|| format!("gen: {kernel}: compute()'s per-pass prologue not found"))?;
    let prologue: Vec<&str> = body[lo..hi]
        .split('\n')
        .map(|l| l.strip_prefix("  ").unwrap_or(l))
        .collect();
    s = format!("{}\n{}", &s[..start], &s[end..]);
    let pass_fn = format!(
        "// The kernel's per-pass prologue (positions, RoPE), lifted by sync.py.\n\
fn s2d_pass_start(i: u32, start_pos: u32) void {{\n{}\n}}\n",
        prologue.join("\n")
    );
    let mut own = exports(&s);
    own.push("stamps".into());
    let all = Names::of(S2D_GRID_NAMES).with(S2D_HEAD_NAMES).with(S2D_VM_NAMES);
    let extra = format!(
        "{}\n{}\n{}\n{}",
        s2d_variant(kernel),
        pass_fn,
        src::s2d::GRID_EXTRA,
        s2d_dummies(&all, &own)
    );
    s = patch(
        &s,
        "\ncomptime {\n  @initialize_queue(q_r1_0_in",
        &format!("\n{extra}\ncomptime {{\n  @initialize_queue(q_r1_0_in"),
        "comptime block",
    )?;
    Ok(format!(
        "// GENERATED by serve2d/sync.py from {kernel}/pe.csl + serve2d/grid_extra.csl.\n\
// Do not edit: change grid_extra.csl or sync.py and rerun.\n{s}"
    ))
}

/// head2d's init: `extra` before each of its two unblocks.
fn head_init(s: &str, extra: &str) -> R<String> {
    let at = s.find("\nfn init() void {").ok_or("gen: head2d/pe.csl changed: init() not found")?;
    let end = at + s[at..].find("\n}\n").ok_or("gen: head2d init() unterminated")? + 3;
    let body = &s[at..end];
    if body.matches("sys_mod.unblock_cmd_stream();").count() != 2 {
        return Err("gen: head2d/pe.csl changed: init's unblocks".into());
    }
    let body = body.replace("sys_mod.unblock_cmd_stream();", &format!("{extra}\n  sys_mod.unblock_cmd_stream();"));
    Ok(format!("{}{}{}", &s[..at], body, &s[end..]))
}

/// head2d's own `compute` (and the comment block above it) removed.
fn drop_head_compute(s: &str) -> R<String> {
    let c = s.find("\n// `steps` head steps").ok_or("gen: head2d/pe.csl changed: compute() not found")?;
    let f = c + s[c..].find("\nfn compute(steps: u32) void {").ok_or("gen: head2d/pe.csl changed: compute() not found")?;
    let end = f + s[f..].find("\n}\n").ok_or("gen: head2d compute() unterminated")? + 3;
    Ok(format!("{}\n{}", &s[..c], &s[end..]))
}

const VM_COLUMNS_OLD: &str = "  vm = VMC >= 1 and cc == R2X + 1;\n  col = if (VMC >= 1 and cc > R2X) cc - 1 else cc;\n";
const VM_COLUMNS_NEW: &str = concat!(
    "  vm = VMC >= 1 and cc > R2X and cc <= R2X + VMC;\n",
    "  col = if (VMC >= 1 and cc > R2X + VMC) cc - VMC else (if (vm) R2X else cc);\n",
);
const HEAD_XY_OLD: &str = "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n  ry = ";
const HEAD_XY_NEW: &str = "  px = layout_mod.get_x_coord() - GW * GPX;\n  py = layout_mod.get_y_coord();\n  ry = ";

fn s2d_head() -> R<String> {
    let mut s = patch(
        src::HEAD2D_V9,
        "param WF32: u16;\n",
        "param WF32: u16;\n// serve2d: the grids beside the head (x0 = GW * GPX, y0 = 0), the tokens' capacity,\n\
// the grids' element types (their exports).\nparam GPX: u16;\nparam GPY: u16;\nparam GW: u16;\n\
param GH: u16;\nparam CAP: u16;\nparam SB: u16;\nparam HLB: u16;\nparam WS: type;\nparam KS: type;\n\
// 2 (bisecting): no embedding chunk is sent on.\nparam S2D_DBG: u16;\n",
        "params",
    )?;
    s = s.replace("VMC == 1", "VMC >= 1");
    s = patch(&s, VM_COLUMNS_OLD, VM_COLUMNS_NEW, "VM columns")?;
    s = patch(&s, HEAD_XY_OLD, HEAD_XY_NEW, "init coordinates")?;
    if !s.contains("if (NB == 1) { return; }") {
        s = patch(
            &s,
            "fn row_pairs() void {\n  const r = blk % Gb;\n",
            "fn row_pairs() void {\n  if (NB == 1) { return; }\n  const r = blk % Gb;\n",
            "row_pairs",
        )?;
    }
    s = head_init(&s, "s2d_head_init();")?;
    s = patch(&s, "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  mark(0);\n",
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  s2d_after_x();\n  mark(0);\n", "region x arrival")?;
    s = patch(&s, "  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "  s2d_feeder_in();\n  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n", "feeder input")?;
    s = patch(&s, "      ib[0] = if (ok) best[1] else -1.0;\n",
        "      ib[0] = s2d_decide(if (ok) best[1] else -1.0);\n", "final root decision")?;
    s = patch(&s, "      tok[1] = ib[0];\n", "      ib[0] = s2d_decide(ib[0]);\n      tok[1] = ib[0];\n",
        "final root decision (gmode)")?;
    let (t, n) = after_each_call(&s, "emb_copy();", "s2d_emb_out();");
    if n != 2 {
        return Err(format!("gen: head2d/pe.csl changed: emb_copy() called {n} times"));
    }
    s = t;
    s = patch(
        &s,
        "  if (CERTIFY and w < 0.0) { const w2 = feeder_round(true); ib[0] = w2; }\n  mark(2);\n}\n",
        "  if (CERTIFY and w < 0.0) { const w2 = feeder_round(true); ib[0] = w2; }\n  mark(2);\n  s2d_feeder_emb();\n}\n",
        "feeder relay",
    )?;
    s = drop_head_compute(&s)?;
    let mut own = exports(&s);
    own.extend(["in_tokens", "out_tokens", "yscale", "hlogits"].map(String::from));
    let extra = format!(
        "{}\n{}{}",
        src::s2d::HEAD_EXTRA,
        s2d_dummies(&Names::of(S2D_GRID_NAMES).with(S2D_VM_NAMES), &own),
        COPY_FNS
    );
    s = format!("{}\n\n{}", s.trim_end_matches('\n'), extra);
    Ok(format!(
        "// GENERATED by serve2d/sync.py from head2d_v9/pe.csl (Fork 7) + serve2d/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s2d_plain(base: &str, base_name: &str, what: &str) -> String {
    let all = Names::of(S2D_GRID_NAMES).with(S2D_HEAD_NAMES).with(S2D_VM_NAMES);
    format!(
        "// GENERATED by serve2d/sync.py from {base_name} ({what}).\n{base}\n{}{COPY_FNS}",
        s2d_dummies(&all, &[])
    )
}

const ONE_COLUMN_K: &str =
    "      color_config.reset_routes(col_of(C_K), .{ .rx = RAMP, .tx = [3]direction{ WEST, NORTH, SOUTH } });\n";

fn one_column_result(s: &str, what: &str) -> R<String> {
    let k2 = ONE_COLUMN_K.replace("[3]direction{ WEST, NORTH, SOUTH }", "[2]direction{ NORTH, SOUTH }");
    patch(
        s,
        ONE_COLUMN_K,
        &format!("      if (hub_first) {{\n  {k2}      }} else {{\n  {ONE_COLUMN_K}      }}\n"),
        what,
    )
}

fn s2d_vm() -> R<String> {
    let mut s = patch(
        src::VM_HM,
        "param ring_w: u16;\n",
        "param ring_w: u16;\n// serve2d: the grids west of the head (the staircases cross this column).\n\
param GPX: u16;\nparam GPY: u16;\nparam GW: u16;\nparam GH: u16;\n\
// The grids' weight and KV element types (their exports' dummies).\nparam WS: type;\nparam KS: type;\n",
        "params",
    )?;
    if !s.contains("param vm_prof: u16;") {
        s = patch(&s, "param ring_w: u16;\n", "param ring_w: u16;\nparam vm_prof: u16;     // unused here\n", "vm_prof")?;
    }
    s = patch(&s, "  vx = layout_mod.get_x_coord();\n", "  vx = layout_mod.get_x_coord() - GW * GPX;\n", "head-relative x")?;
    s = patch(
        &s,
        "  timestamp.enable_tsc();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "  s2d_vm_paths();\n  timestamp.enable_tsc();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "init routes",
    )?;
    s = one_column_result(&s, "one-block result route")?;
    let (start, end) = fn_block(&s, "fn compute(steps: u32) void {").ok_or("gen: vm.csl changed: compute() not found")?;
    s = format!("{}\n{}{}", &s[..start], src::s2d::VM_EXTRA, &s[end..]);
    let own = exports(&s);
    let all = Names::of(S2D_GRID_NAMES).with(S2D_HEAD_NAMES);
    s = format!("{}\n\n{}{}", s.trim_end_matches('\n'), s2d_dummies(&all, &own), COPY_FNS);
    Ok(format!(
        "// GENERATED by serve2d/sync.py from guest4hm_vm.csl + serve2d/vm_extra.csl.\n\
// Do not edit: change vm_extra.csl or sync.py and rerun.\n{s}"
    ))
}

/// serve2d's build directory.
pub fn serve2d() -> R<Vec<File>> {
    Ok(vec![
        ("layout.csl", src::s2d::LAYOUT.to_string()),
        ("paths.csl", src::s2d::PATHS.to_string()),
        ("rt.csl", src::s2d::RT.to_string()),
        ("grid_pe.csl", s2d_grid("qwen2d", src::QWEN2D)?),
        ("grid_paged_pe.csl", s2d_grid("qwen2d_paged", src::QWEN2D_PAGED)?),
        ("head_pe.csl", s2d_head()?),
        ("head_stub.csl", s2d_plain(src::s2d::STUB_BASE, "stub_base.csl", "the stub head")),
        ("filler.csl", s2d_plain(src::s2d::FILLER_BASE, "filler_base.csl", "the filler")),
        ("vm_pe.csl", s2d_vm()?),
    ])
}

// ---------------------------------------------------------------- serve2d_pipe

const PIPE_GRID_NAMES: &[(&str, &str)] = &[
    ("x", "f32"), ("gamma1", "f32"), ("gamma2", "f32"), ("wqa", "f32"), ("wqv", "f32"), ("wv", "f32"),
    ("qkn_w", "f32"), ("rot_c", "f32"), ("rot_s", "f32"), ("kc", "f32"), ("vc", "f32"), ("wo", "f32"),
    ("wgu", "f32"), ("wd", "f32"), ("prof", "u32"), ("dims", "u16"), ("fallbacks", "u32"),
];
const PIPE_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"), ("et", "u32"), ("l2", "f32"), ("force", "u32"), ("lmax", "f32"), ("gmode", "u32"),
    ("vm_tok", "f32"), ("vin", "f32"), ("xin", "f32"), ("gam", "f32"), ("y", "f32"), ("emb", "f32"),
    ("tok", "f32"), ("prof", "u32"), ("in_tokens", "u32"), ("out_tokens", "u32"), ("yscale", "f32"),
    ("hlogits", "f32"), ("mtab", "f32"), ("spec_ctl", "u32"), ("spec_hist", "u32"),
    ("spec_out", "u32"), ("spec_acc", "u32"), ("smp", "u32"), ("vmlane", "u32"),
];
const PIPE_VM_NAMES: &[(&str, &str)] = &[
    ("gwide", "f32"), ("gpool", "u32"), ("gops", "u32"), ("gout", "u32"), ("gprof", "u32"),
    ("gostep", "u32"), ("gring", "u32"), ("gcycles", "u32"), ("gvalid", "u32"), ("gtok_home", "u32"),
    ("gbank", "u32"), ("vm_ctl", "u32"), ("gsin", "u32"), ("gsdesc", "u32"), ("glane", "u32"), ("gbdesc", "u32"),
];

/// serve2d_pipe's dummies: a 1-word array per missing name.
fn pipe_dummies(names: &Names, own: &[String]) -> String {
    let mut lines = Vec::new();
    let mut decls = Vec::new();
    for &(n, ty) in &names.0 {
        if own.iter().any(|o| o == n) {
            continue;
        }
        decls.push(format!("var s2d_d_{n} = @zeros([1]{ty});\nvar s2d_d_{n}_ptr: [*]{ty} = &s2d_d_{n};"));
        lines.push(format!("  @export_symbol(s2d_d_{n}_ptr, \"{n}\");"));
    }
    format!(
        "// serve2d_pipe: the other programs' exported names.\n{}\ncomptime {{\n{}\n}}\n",
        decls.join("\n"),
        lines.join("\n")
    )
}

fn pipe_all() -> Names {
    Names::of(PIPE_GRID_NAMES).with(PIPE_HEAD_NAMES).with(PIPE_VM_NAMES)
}

fn pipe_grid() -> R<String> {
    let mut s = patch(
        src::QWEN2D_PRING,
        "param c_r1_0: color;\n",
        "// serve2d_pipe: the grids' arrangement, the head's feeder row and VM column.\n\
param GW: u16;\nparam GH: u16;\nparam NL: u16;\nparam F: u16;\n\
param S2D_VMC: u16;\nparam S2D_VMAT: u16;\nparam c_r1_0: color;\n",
        "params",
    )?;
    s = patch(&s, "  lx = layout_mod.get_x_coord() % P;\n", "  s2d_place();\n", "init place")?;
    s = patch(&s, "  px = lx;\n  py = layout_mod.get_y_coord();\n", "", "init coordinates")?;
    s = patch(&s, "  route_handoff();\n", "  s2d_routes();\n", "init routes")?;
    s = patch(&s, "  if (stage == K - 1) {\n    b = 0;", "  if (L == NL - 1) {\n    b = 0;", "last stage advances")?;
    s = patch(&s, "  if (more and lx == P - 1) {\n    @fmovs(fout(q_h_out, NIO), io_dsd);\n  }\n", "  s2d_send();\n", "send")?;
    s = patch(
        &s,
        "task recvd_task() void {\n  @fmovs(io_dsd, in_dsd);",
        "task recvd_task() void {\n  s2d_relay();\n  @fmovs(io_dsd, in_dsd);",
        "relay",
    )?;
    s = patch(
        &s,
        "fn compute(steps: u32) void {\n  timestamp.enable_tsc();\n  steps_total = steps;\n  pass_i = 0;\n  @activate(STEP);\n}\n",
        "fn compute(laps: u32, nmb: u32) void {\n  timestamp.enable_tsc();\n  steps_total = laps * nmb;\n  pass_i = 0;\n  s2d_start();\n}\n",
        "compute",
    )?;
    let own = exports(&s);
    let extra = format!("{}\n{}", src::pipe::GRID_EXTRA, pipe_dummies(&pipe_all(), &own));
    s = patch(
        &s,
        "\ncomptime {\n  @initialize_queue(q_h_in",
        &format!("\n{extra}\ncomptime {{\n  @initialize_queue(q_h_in"),
        "comptime block",
    )?;
    Ok(format!(
        "// GENERATED by serve2d_pipe/sync.py from qwen2d_pring/pe.csl + serve2d_pipe/grid_extra.csl.\n\
// Do not edit: change grid_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn pipe_head() -> R<String> {
    let mut s = patch(
        src::HEAD2D_V9,
        "param WF32: u16;\n",
        "param WF32: u16;\n// serve2d_pipe: the grids beside the head (x0 = GW * GPX, y0 = 0), the tokens'\n\
// capacity, lanes a micro-batch, micro-batches at most, the grids' E.\n\
param GPX: u16;\nparam GPY: u16;\nparam GW: u16;\n\
param GH: u16;\nparam CAP: u16;\nparam SB: u16;\nparam HB: u16;\nparam HLB: u16;\nparam MMAX: u16;\nparam GE: u16;\n\
param S2D_DBG: u16;\n// The PE's role, for memory: a region row, the root column, the final root.\n\
param S2D_REG: u16;\nparam S2D_ROOTC: u16;\nparam S2D_FINAL: u16;\n\
// Speculative decode (head_extra.csl): on/off, the grids' page size, the history,\n\
// lookup table and output capacities; the fused naive sampler.\nparam SMP: u16;\nparam SPEC: u16;\nparam GPS: u16;\nparam SPEC_H: u16;\nparam SPEC_T: u16;\nparam SPEC_O: u16;\n",
        "params",
    )?;
    s = s.replace("VMC == 1", "VMC >= 1");
    s = patch(&s, VM_COLUMNS_OLD, VM_COLUMNS_NEW, "VM columns")?;
    s = patch(&s, HEAD_XY_OLD, HEAD_XY_NEW, "init coordinates")?;
    s = patch(
        &s,
        "fn row_pairs() void {\n  const r = blk % Gb;\n",
        "fn row_pairs() void {\n  if (NB == 1) { return; }\n  const r = blk % Gb;\n",
        "row_pairs",
    )?;
    s = head_init(&s, "s2d_head_init();")?;
    s = patch(
        &s,
        "var wt32: [if (WF32 == 1) NW + 1 else 1]f32 align(16) = @zeros([if (WF32 == 1) NW + 1 else 1]f32);",
        "var wt32: [if (WF32 == 1 and S2D_REG == 1) NW + 1 else 1]f32 align(16) = \
@zeros([if (WF32 == 1 and S2D_REG == 1) NW + 1 else 1]f32);",
        "wt32 by role",
    )?;
    for (decl, n) in [
        ("var sc: [VP]f32 align(16) = @zeros([VP]f32);", "VP"),
        ("var negiv: [VP]f32 align(16) = @zeros([VP]f32);", "VP"),
        ("var score: [Vr]f32 align(16) = @zeros([Vr]f32);", "Vr"),
    ] {
        let name = decl.split_whitespace().nth(1).unwrap_or("").trim_end_matches(':');
        let z = format!("if (S2D_ROOTC == 1) {n} else 1");
        s = patch(&s, decl, &format!("var {name}: [{z}]f32 align(16) = @zeros([{z}]f32);"), &format!("{name} by role"))?;
    }
    s = patch(
        &s,
        "  for (@range(u16, Vr)) |i| { negiv[i] = -@as(f32, i); }\n",
        "  if (S2D_ROOTC == 1) { for (@range(u16, Vr)) |i| { negiv[i] = -@as(f32, i); } }\n",
        "negiv init",
    )?;
    s = drop_head_compute(&s)?;
    let mut own = exports(&s);
    own.extend(
        [
            "in_tokens", "out_tokens", "yscale", "hlogits", "mtab", "spec_ctl", "spec_hist", "spec_out",
            "spec_acc", "smp", "vmlane",
        ]
        .map(String::from),
    );
    let extra = format!(
        "{}\n{}{}",
        src::pipe::HEAD_EXTRA,
        pipe_dummies(&Names::of(PIPE_GRID_NAMES).with(PIPE_VM_NAMES), &own),
        COPY_FNS
    );
    s = format!("{}\n\n{}", s.trim_end_matches('\n'), extra);
    Ok(format!(
        "// GENERATED by serve2d_pipe/sync.py from head2d_v9/pe.csl + serve2d_pipe/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn pipe_filler() -> String {
    format!(
        "// GENERATED by serve2d_pipe/sync.py from filler_base.csl.\n{}\n{}{COPY_FNS}",
        src::pipe::FILLER_BASE,
        pipe_dummies(&pipe_all(), &[])
    )
}

fn pipe_vm() -> R<String> {
    let mut s = patch(
        src::VM_HS,
        "param vm_lanes: u16;",
        "param vm_lanes: u16;\n// serve2d_pipe: the grids west of the head, lanes a micro-batch.\n\
param GPX: u16;\nparam GPY: u16;\nparam GW: u16;\nparam GH: u16;\nparam S2D_SB: u16;",
        "params",
    )?;
    s = patch(&s, "  vx = layout_mod.get_x_coord();\n", "  vx = layout_mod.get_x_coord() - GW * GPX;\n", "head-relative x")?;
    s = patch(
        &s,
        "  timestamp.enable_tsc();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "  s2d_vm_paths();\n  timestamp.enable_tsc();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "init routes",
    )?;
    s = one_column_result(&s, "one-column result route")?;
    s = patch(&s, "fn compute(steps: u32) void {", "fn s2d_vm_run(steps: u32) void {", "compute")?;
    s = s.replace("  @export_symbol(compute);", "");
    let own = exports(&s);
    s = format!(
        "{}\n\n{}\ncomptime {{\n  @export_symbol(compute);\n}}\n{}{}",
        s.trim_end_matches('\n'),
        src::pipe::VM_EXTRA,
        pipe_dummies(&pipe_all(), &own),
        COPY_FNS
    );
    Ok(format!(
        "// GENERATED by serve2d_pipe/sync.py from guest4hs_vm.csl + serve2d_pipe/vm_extra.csl.\n\
// Do not edit: change vm_extra.csl or sync.py and rerun.\n{s}"
    ))
}

/// serve2d_pipe's build directory.
pub fn serve2d_pipe() -> R<Vec<File>> {
    Ok(vec![
        ("layout.csl", src::pipe::LAYOUT.to_string()),
        ("paths.csl", src::pipe::PATHS.to_string()),
        ("rt.csl", src::pipe::RT.to_string()),
        ("grid_pe.csl", pipe_grid()?),
        ("head_pe.csl", pipe_head()?),
        ("filler.csl", pipe_filler()),
        ("vm_pe.csl", pipe_vm()?),
    ])
}

// ---------------------------------------------------------------- serve8b

const S8B_GRID_NAMES: &[(&str, &str)] = &[
    ("x", "f32"), ("tab", "u32"), ("gamma1", "f32"), ("gamma2", "f32"), ("wqa", "WS"), ("wqv", "WS"),
    ("wv", "WS"), ("qkn_w", "f32"), ("rot_c", "f32"), ("rot_s", "f32"), ("kc", "KS"), ("vc", "KS"),
    ("wo", "WS"), ("wgu", "WS"), ("wd", "WS"), ("prof", "u32"), ("dims", "u16"), ("rtab", "u16"),
];
const S8B_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"), ("et", "u32"), ("l2", "f32"), ("force", "u32"), ("lmax", "f32"), ("gmode", "u32"),
    ("vm_tok", "f32"), ("vin", "f32"), ("xin", "f32"), ("gam", "f32"), ("y", "f32"), ("emb", "f32"),
    ("tok", "f32"), ("prof", "u32"), ("in_tokens", "u32"), ("out_tokens", "u32"), ("yscale", "f32"),
    ("hlogits", "f32"),
];
const S8B_EMBED_NAMES: &[(&str, &str)] = &[("etab", "u32")];

fn s8b_dummies(own: &[String]) -> String {
    let all = Names::of(S8B_GRID_NAMES).with(S8B_HEAD_NAMES).with(S8B_EMBED_NAMES);
    pipe_dummies(&all, own).replacen("// serve2d_pipe:", "// serve8b:", 1)
}

const S8B_GRID_VARIANT: &str = "// serve8b: queue 2 (free in qwen2d_paged8b) carries the handoffs; it rests on\n\
// C_Q2, which nothing routes to a ramp.\n\
const C_Q2: color = @get_color(16);\n\
const q_gb_in = @get_input_queue(2);\n\
const q_gb_out = @get_output_queue(2);\n\
comptime {\n  @initialize_queue(q_gb_in, .{ .color = C_Q2 });\n  @initialize_queue(q_gb_out, .{ .color = C_Q2 });\n}\n\
const S2D_XN: u16 = B * Hc;       // x's words a PE: B lanes of Hc, lane-major\n";

fn s8b_grid() -> R<String> {
    let tag = "qwen2d_paged8b/pe.csl";
    let mut s = patch(
        src::QWEN2D_PAGED8B,
        "param c_r1_0: color;\n",
        "// serve8b: 1 times each pass (prof[0]).\nparam S2D_TIME: u16;\nparam c_r1_0: color;\n",
        "params",
    )?;
    s = patch(&s, "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n", "  s2d_place();\n", "init coordinates")?;
    s = patch(
        &s,
        "  route_gbcast();\n  axis_y();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "  route_gbcast();\n  axis_y();\n  s2d_routes();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "init routes",
    )?;
    let (start, end) = fn_block(&s, "fn compute(steps: u32) void {\n").ok_or_else(|| format!("gen: {tag} changed: compute() not found"))?;
    let body_at = start + s[start..].find("fn compute(steps: u32) void {\n").unwrap_or(0) + "fn compute(steps: u32) void {\n".len();
    let body = &s[body_at..end - 3];
    let open = "  while (i < steps) : (i += 1) {\n";
    let close = "\n    timing = i > 0;\n    pass_start();\n    timestamp.get_timestamp(&t_mark);\n    layer();\n    pass_end();\n  }\n";
    let lo = body.find(open).ok_or_else(|| format!("gen: {tag} changed: compute()'s pass loop not as expected"))? + open.len();
    let hi = lo + body[lo..].find(close).ok_or_else(|| format!("gen: {tag} changed: compute()'s pass loop not as expected"))?;
    let prologue: Vec<&str> = body[lo..hi].split('\n').map(|l| l.strip_prefix("  ").unwrap_or(l)).collect();
    let prologue = prologue.join("\n");
    s = format!("{}\n{}", &s[..start], &s[end..]);
    let pass_fn = format!(
        "// The kernel's per-pass prologue (lanes advance), lifted by sync.py.\n\
fn s2d_pass_start(i: u32, start_pos: u32) void {{\n{prologue}\n}}\n"
    );
    let mut own = exports(&s);
    own.extend(exports(src::s8b::GRID_EXTRA));
    let extra = format!("{S8B_GRID_VARIANT}\n{pass_fn}\n{}\n{}", src::s8b::GRID_EXTRA, s8b_dummies(&own));
    s = patch(
        &s,
        "\ncomptime {\n  @initialize_queue(q_r1_0_in",
        &format!("\n{extra}\ncomptime {{\n  @initialize_queue(q_r1_0_in"),
        "comptime block",
    )?;
    Ok(format!(
        "// GENERATED by serve8b/sync.py from {tag} + serve8b/grid_extra.csl.\n\
// Do not edit: change grid_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s8b_head() -> R<String> {
    let tag = "head2d_v10/pe.csl";
    let mut s = patch(
        src::HEAD2D_V10,
        "param WF32: u16;\n",
        concat!(
            "param WF32: u16;\n// serve8b: the grids beside the head (x0 = GW * GPX, y0 = 0; GHc dims a grid\n",
            "// row), the tokens' capacity, the lanes, the copy kernel, the grids' element\n",
            "// types (their exports).\nparam GPX: u16;\nparam GPY: u16;\nparam GHc: u16;\nparam GW: u16;\n",
            "param GH: u16;\nparam GS: u16;\nparam NL: u16;\nparam CAP: u16;\nparam SB: u16;\nparam HLB: u16;\n",
            "param KVCP: u16;\nparam WS: type;\nparam KS: type;\n"
        ),
        "params",
    )?;
    s = patch(&s, HEAD_XY_OLD, HEAD_XY_NEW, "init coordinates")?;
    if !s.contains("if (NB == 1) { return; }") {
        s = patch(
            &s,
            "fn row_pairs() void {\n  const r = blk % Gb;\n",
            "fn row_pairs() void {\n  if (NB == 1) { return; }\n  const r = blk % Gb;\n",
            "row_pairs",
        )?;
    }
    s = patch(
        &s,
        "    if (px == W - 1) { color_config.reset_routes(c_ib, .{ .rx = WEST, .tx = RAMP }); }\n",
        "    if (px == W - 1) { color_config.reset_routes(c_ib, .{ .rx = WEST, .tx = [2]direction{ EAST, RAMP } }); }\n",
        "winner multicast east end",
    )?;
    s = head_init(&s, "s2d_head_init();")?;
    s = patch(&s, "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  mark(0);\n",
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  s2d_after_x();\n  mark(0);\n", "region x arrival")?;
    s = patch(&s, "  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "  s2d_feeder_in();\n  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n", "feeder input")?;
    s = patch(&s, "      ib[0] = if (ok) best[1] else -1.0;\n",
        "      ib[0] = s2d_decide(if (ok) best[1] else -1.0);\n", "final root decision")?;
    s = patch(&s, "      tok[1] = ib[0];\n", "      ib[0] = s2d_decide(ib[0]);\n      tok[1] = ib[0];\n",
        "final root decision (gmode)")?;
    let (t, n) = after_each_call(&s, "emb_copy();", "s2d_emb_out();");
    if n != 2 {
        return Err(format!("gen: {tag} changed: emb_copy() called {n} times"));
    }
    s = drop_head_compute(&t)?;
    let mut own = exports(&s);
    own.extend(["in_tokens", "out_tokens", "yscale", "hlogits"].map(String::from));
    s = format!("{}\n\n{}\n{}", s.trim_end_matches('\n'), src::s8b::HEAD_EXTRA, s8b_dummies(&own));
    Ok(format!(
        "// GENERATED by serve8b/sync.py from {tag} (Fork 7) + serve8b/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s8b_plain(base: &str, base_name: &str, what: &str) -> String {
    format!("// GENERATED by serve8b/sync.py from {base_name} ({what}).\n{base}\n{}", s8b_dummies(&exports(base)))
}

/// serve8b's build directory.
pub fn serve8b() -> R<Vec<File>> {
    Ok(vec![
        ("layout.csl", src::s8b::LAYOUT.to_string()),
        ("paths.csl", src::s8b::PATHS.to_string()),
        ("rt.csl", src::s8b::RT.to_string()),
        ("grid_pe.csl", s8b_grid()?),
        ("head_pe.csl", s8b_head()?),
        ("embed_pe.csl", s8b_plain(src::s8b::EMBED_BASE, "embed_base.csl", "the embedding strip")),
        ("filler.csl", s8b_plain(src::s8b::FILLER_BASE, "filler_base.csl", "the filler")),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated files' blake3, from the designs' sync.py outputs at the
    /// time of the port. A change to a kernel or an extra changes these: rerun
    /// sync.py, diff (`PIE_CEREBRAS_PAYLOAD` below), then update.
    const PINNED: &[(&str, &str, &str)] = &[
        ("serve2d", "grid_pe.csl", "5d23b7aeda4cf25889e7643c5feede43a9d599c80631d6c13cbfab563bfad774"),
        ("serve2d", "grid_paged_pe.csl", "c65fa9876ccb870e2312d4fc736d110be030d7ff121af4c52f0e3f28b32a1be9"),
        ("serve2d", "head_pe.csl", "18a10d1ef3b3b4f028eadb6f4f094b0893f28bee97f076c2c3d707f424116fa4"),
        ("serve2d", "head_stub.csl", "e2e5c74bdc834d404dd312a2862bd94def494d2d654b11317628dce43e005bdd"),
        ("serve2d", "filler.csl", "d4e6a8c03eb2184f4afb5a516bfe38307b7472f99c36b8c223d235266def48d3"),
        ("serve2d", "vm_pe.csl", "43b3a8f0ebe4def64ef4c3137d557008c45f9493b6f7c11d691374a7b442f6be"),
        ("serve2d_pipe", "grid_pe.csl", "b53a58e0819723e17fcb900aa3b3a9c46b3084d10e78245e88fbcc81e788afc9"),
        ("serve2d_pipe", "head_pe.csl", "4db3bc7f18fd9ef32f420f8ad4881cbb3477d74bcf71045ff5eb1559e5c26ee5"),
        ("serve2d_pipe", "filler.csl", "79788748a6a34253d18ba364f05033129db41342f7bc4c86687436fcfaab0e61"),
        ("serve2d_pipe", "vm_pe.csl", "4753a16de7cc6036f7890df4f8e286b8934b5edc1d7baf094b77dc636851b723"),
        ("serve8b", "grid_pe.csl", "9d6e67f1ed0c88c1c852248296fb7be7bfc0ac7ecc24a8456e1d45ec2dab0f76"),
        ("serve8b", "head_pe.csl", "b08dbbaef37c00d3b0a6f91888766eda3ec77cfddb868b1780f7b59c85f27256"),
        ("serve8b", "embed_pe.csl", "0c17e992957bce93e17360b590ad8148641f6e02f42c87d29006370b6fc44a18"),
        ("serve8b", "filler.csl", "ba91cf083cc4235ea8da24076a75668549432601fb80e0e3cac8ee545fb8f5db"),
    ];

    #[test]
    fn generators_match_their_pins() {
        if std::env::var_os("PIE_CEREBRAS_PRINT_PINS").is_some() {
            for d in [Design::Serve2d, Design::Serve2dPipe, Design::Serve8b] {
                for (name, text) in d.files().unwrap() {
                    println!("PIN (\"{}\", \"{name}\", \"{}\"),", d.name(), blake3::hash(text.as_bytes()).to_hex());
                }
            }
        }
        for (design, file, want) in PINNED {
            let d = Design::parse(design).unwrap();
            let files = d.files().unwrap();
            let got = files.iter().find(|(n, _)| n == file).unwrap();
            assert_eq!(blake3::hash(got.1.as_bytes()).to_hex().as_str(), *want, "{design}/{file}");
        }
    }

    /// With `PIE_CEREBRAS_PAYLOAD` set to the bench payload directory, every
    /// generated file must equal its sync.py output there byte for byte.
    #[test]
    fn generators_match_payload() {
        let Some(root) = std::env::var_os("PIE_CEREBRAS_PAYLOAD") else { return };
        let root = std::path::PathBuf::from(root);
        for d in [Design::Serve2d, Design::Serve2dPipe, Design::Serve8b] {
            for (name, text) in d.files().unwrap() {
                let want = std::fs::read_to_string(root.join(d.name()).join(name)).unwrap();
                if want != text {
                    let line = want.lines().zip(text.lines()).position(|(a, b)| a != b);
                    let show = |t: &str| line.map(|l| t.lines().skip(l.saturating_sub(2)).take(4).collect::<Vec<_>>().join("\n"));
                    panic!("{}/{name} differs at line {line:?}\n--- payload:\n{:?}\n--- generated:\n{:?}",
                        d.name(), show(&want), show(&text));
                }
            }
        }
    }

    #[test]
    fn exports_and_emb_copy() {
        let s = "a\n  @export_symbol(p_ptr, \"x\");\n  @export_symbol(init);\n";
        assert_eq!(exports(s), vec!["x".to_string()]);
        let (t, n) = after_each_call("x\n    emb_copy();\ny", "emb_copy();", "s2d_emb_out();");
        assert_eq!(n, 1);
        assert_eq!(t, "x\n    emb_copy();\n    s2d_emb_out();\ny");
    }
}
