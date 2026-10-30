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
    pub const HEAD2D_V10: &str = include_str!("csl/kernels/head2d_v10.csl");
    pub const QWEN2D_PAGED8B_HR: &str = include_str!("csl/kernels/qwen2d_paged8b_hr.csl");
    pub const QWEN2D_REGION: &str = include_str!("csl/kernels/qwen2d_region.csl");
    /// Fork 11's p8fast softmax levers S1-S3 as (old, new) text pairs on
    /// qwen2d_paged8b's or qwen2d_region's pe.csl (softmax_s123.py's REPS).
    pub const SOFTMAX_S123: &str = include_str!("csl/kernels/softmax_s123.json");
    pub const VM_HM: &str = include_str!("csl/vm/guest4hm_vm.csl");
    pub const VM_H2S: &str = include_str!("csl/vm/guest4h2s_vm.csl");

    pub mod s2d {
        pub const LAYOUT: &str = include_str!("csl/serve2d/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve2d/paths.csl");
        pub const RT: &str = include_str!("csl/serve2d/rt.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve2d/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve2d/head_extra.csl");
        pub const VM_EXTRA: &str = include_str!("csl/serve2d/vm_extra.csl");
        pub const STUB_BASE: &str = include_str!("csl/serve2d/stub_base.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve2d/filler_base.csl");
        pub const KVSTAGE: &str = include_str!("csl/serve2d/kvstage.csl");
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

    pub mod s8br {
        pub const LAYOUT: &str = include_str!("csl/serve8b_region/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve8b_region/paths.csl");
        pub const RT: &str = include_str!("csl/serve8b_region/rt.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve8b_region/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve8b_region/head_extra.csl");
        pub const EMBED_BASE: &str = include_str!("csl/serve8b_region/embed_base.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve8b_region/filler_base.csl");
        pub const VM_BASE: &str = include_str!("csl/serve8b_region/vm_base.csl");
        pub const VM_STUB_BASE: &str = include_str!("csl/serve8b_region/vm_stub_base.csl");
    }

    pub mod pipe {
        pub const LAYOUT: &str = include_str!("csl/serve2d_pipe/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve2d_pipe/paths.csl");
        pub const RT: &str = include_str!("csl/serve2d_pipe/rt.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve2d_pipe/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve2d_pipe/head_extra.csl");
        pub const VM_EXTRA: &str = include_str!("csl/serve2d_pipe/vm_extra.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve2d_pipe/filler_base.csl");
        pub const KVSTAGE: &str = include_str!("csl/serve2d_pipe/kvstage.csl");
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
    /// Qwen3-8B as layer units: qwen2d_paged8b grids and qwen2d_region
    /// regions of several layers, chained by host tables (serve8b_region).
    Serve8bRegion,
}

impl Design {
    pub fn name(self) -> &'static str {
        match self {
            Design::Serve2d => "serve2d",
            Design::Serve2dPipe => "serve2d_pipe",
            Design::Serve8b => "serve8b",
            Design::Serve8bRegion => "serve8b_region",
        }
    }

    pub fn parse(name: &str) -> Option<Design> {
        match name {
            "serve2d" => Some(Design::Serve2d),
            "serve2d_pipe" => Some(Design::Serve2dPipe),
            "serve8b" => Some(Design::Serve8b),
            "serve8b_region" => Some(Design::Serve8bRegion),
            _ => None,
        }
    }

    /// The eta VM image the design's sources carry (enabled by its params).
    pub fn vm_image(self) -> Option<&'static str> {
        match self {
            Design::Serve2d => Some("guest4hm"),
            Design::Serve2dPipe => Some("guest4h2s"),
            Design::Serve8b => None,
            // Fork 3's guest4hz as the block east of the strip (with VMX).
            Design::Serve8bRegion => Some("guest4hz"),
        }
    }

    /// How a KV-staging build's slots hold their elements (`kvst_pack`):
    /// f32 words on serve2d*, bf16 one a word on serve8b_region, bf16 pairs
    /// in serve8b's aliased scratch; and whether they alias it.
    pub fn kv_staging(self) -> (&'static str, bool) {
        match self {
            Design::Serve2d | Design::Serve2dPipe => ("f32", false),
            Design::Serve8b => ("bf16x2", true),
            Design::Serve8bRegion => ("bf16", false),
        }
    }

    /// Every file of the program's build directory, `layout.csl` first.
    pub fn files(self) -> R<Vec<File>> {
        match self {
            Design::Serve2d => serve2d(),
            Design::Serve2dPipe => serve2d_pipe(),
            Design::Serve8b => serve8b(),
            Design::Serve8bRegion => serve8b_region(),
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

/// The names a layout exports (`@export_name("n", ...)`), in order.
#[cfg(test)]
fn layout_exports(layout: &str) -> Vec<String> {
    layout
        .split("@export_name(")
        .skip(1)
        .filter_map(|rest| rest.trim_start().strip_prefix('"')?.split_once('"').map(|(n, _)| n.to_string()))
        .collect()
}

// ---------------------------------------------------------------- serve2d

const COPY_FNS: &str = "fn copy_page(src: u32, dst: u32) void { sys_mod.unblock_cmd_stream(); }\n\
fn copy_move(src: u32, dst: u32) void { sys_mod.unblock_cmd_stream(); }\n\
comptime {\n  @export_symbol(copy_page);\n  @export_symbol(copy_move);\n}\n";

/// KV page staging's launches, no-ops where there is no paged pool.
const STAGE_FNS: &str = "fn page_stage(page: u32, slot: u32) void { sys_mod.unblock_cmd_stream(); }\n\
fn page_unstage(page: u32, slot: u32) void { sys_mod.unblock_cmd_stream(); }\n\
comptime {\n  @export_symbol(page_stage);\n  @export_symbol(page_unstage);\n}\n";

const S2D_GRID_NAMES: &[(&str, &str)] = &[
    ("x", "f32"), ("gamma1", "f32"), ("gamma2", "f32"), ("wqkv", "WS"), ("qkn_w", "f32"),
    ("inv_freq", "f32"), ("kc", "KS"), ("vc", "KS"), ("wo", "WS"), ("wgu", "WS"), ("wd", "WS"),
    ("prof", "u32"), ("fallbacks", "u32"), ("stamps", "u32"),
    ("tab", "u32"), ("wqa", "f32"), ("wqv", "f32"), ("wv", "f32"), ("rot_c", "f32"), ("rot_s", "f32"),
    ("dims", "u16"), ("kstage", "f32"), ("vstage", "f32"),
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
param S2D_VMC: u16;\nparam S2D_VMAT: u16;\nparam S2D_DBG: u16;\n\
// KV page staging slots a PE (kvstage.csl, paged grids; 0: none).\nparam KVST: u16;\n";

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
    // The paged grid holds the staging slots; qwen2d's launches only return.
    let stage = if kernel == "qwen2d_paged" {
        own.extend(["kstage", "vstage"].map(String::from));
        src::s2d::KVSTAGE
    } else {
        STAGE_FNS
    };
    let all = Names::of(S2D_GRID_NAMES).with(S2D_HEAD_NAMES).with(S2D_VM_NAMES);
    let extra = format!(
        "{}\n{}\n{}\n{stage}\n{}",
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
        "{}\n{}{COPY_FNS}{STAGE_FNS}",
        src::s2d::HEAD_EXTRA,
        s2d_dummies(&Names::of(S2D_GRID_NAMES).with(S2D_VM_NAMES), &own),
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
        "// GENERATED by serve2d/sync.py from {base_name} ({what}).\n{base}\n{}{COPY_FNS}{STAGE_FNS}",
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
    s = format!("{}\n\n{}{COPY_FNS}{STAGE_FNS}", s.trim_end_matches('\n'), s2d_dummies(&all, &own));
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
    ("kstage", "f32"), ("vstage", "f32"),
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
param S2D_VMC: u16;\nparam S2D_VMAT: u16;\n\
// KV page staging slots a PE (kvstage.csl; 0: none).\nparam KVST: u16;\nparam c_r1_0: color;\n",
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
    let mut own = exports(&s);
    own.extend(["kstage", "vstage"].map(String::from));
    let extra = format!("{}\n{}\n{}", src::pipe::GRID_EXTRA, src::pipe::KVSTAGE, pipe_dummies(&pipe_all(), &own));
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
        "{}\n{}{COPY_FNS}{STAGE_FNS}",
        src::pipe::HEAD_EXTRA,
        pipe_dummies(&Names::of(PIPE_GRID_NAMES).with(PIPE_VM_NAMES), &own),
    );
    s = format!("{}\n\n{}", s.trim_end_matches('\n'), extra);
    Ok(format!(
        "// GENERATED by serve2d_pipe/sync.py from head2d_v9/pe.csl + serve2d_pipe/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn pipe_filler() -> String {
    format!(
        "// GENERATED by serve2d_pipe/sync.py from filler_base.csl.\n{}\n{}{COPY_FNS}{STAGE_FNS}",
        src::pipe::FILLER_BASE,
        pipe_dummies(&pipe_all(), &[])
    )
}

fn pipe_vm() -> R<String> {
    let mut s = patch(
        src::VM_H2S,
        "param vm_lanes: u16;",
        "param vm_lanes: u16;\n// serve2d_pipe: the grids west of the head, lanes a micro-batch.\n\
param GPX: u16;\nparam GPY: u16;\nparam GW: u16;\nparam GH: u16;\nparam S2D_SB: u16;",
        "params",
    )?;
    s = patch(
        &s,
        "  vx = if (OUTSIDE) layout_mod.get_x_coord() - vm_x0 + R2X + 1 else layout_mod.get_x_coord();\n",
        "  vx = if (OUTSIDE) layout_mod.get_x_coord() - vm_x0 + R2X + 1 else layout_mod.get_x_coord() - GW * GPX;\n",
        "head-relative x",
    )?;
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
        "{}\n\n{}\ncomptime {{\n  @export_symbol(compute);\n}}\n{}{COPY_FNS}{STAGE_FNS}",
        s.trim_end_matches('\n'),
        src::pipe::VM_EXTRA,
        pipe_dummies(&pipe_all(), &own),
    );
    Ok(format!(
        "// GENERATED by serve2d_pipe/sync.py from guest4h2s_vm.csl + serve2d_pipe/vm_extra.csl.\n\
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

/// softmax_s123.py's `apply`: Fork 11's p8fast softmax levers (S1: the
/// max-shift and rescale as one 2-D op, exp over act-sized chunks; S2: the
/// max and sum folds as a halving tree; S3: the exp sums ride the values
/// all-reduce) on an 8B unit kernel's text, each anchor found once.
fn softmax_s123(text: &str, tag: &str) -> R<String> {
    let reps: serde_json::Value =
        serde_json::from_str(src::SOFTMAX_S123).map_err(|e| format!("softmax_s123.json: {e}"))?;
    let mut s = text.to_string();
    for r in reps.as_array().ok_or("softmax_s123.json: not a list")? {
        let (old, new) = (r["old"].as_str().unwrap_or(""), r["new"].as_str().unwrap_or(""));
        let n = s.matches(old).count();
        if old.is_empty() || n != 1 {
            return Err(format!("softmax_s123: {tag} changed: anchor found {n} times"));
        }
        s = s.replacen(old, new, 1);
    }
    Ok(s)
}

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
/// KV page staging (the grids' own, under KVST > 0): f32 words aliasing the
/// grid's `s` / `act`, two bf16 cells a word.
const S8B_KVST_NAMES: &[(&str, &str)] = &[("kstage", "f32"), ("vstage", "f32")];

/// serve8b's dummies: a 1-word array per missing name (the staging slots
/// under `KVST > 0`), and no-op staging launches on programs without slots.
fn s8b_dummies(own: &[String]) -> String {
    let all = Names::of(S8B_GRID_NAMES).with(S8B_KVST_NAMES).with(S8B_HEAD_NAMES).with(S8B_EMBED_NAMES);
    let mut lines = Vec::new();
    let mut decls = Vec::new();
    for &(n, ty) in &all.0 {
        if own.iter().any(|o| o == n) {
            continue;
        }
        decls.push(format!("var s2d_d_{n} = @zeros([1]{ty});\nvar s2d_d_{n}_ptr: [*]{ty} = &s2d_d_{n};"));
        if S8B_KVST_NAMES.iter().any(|k| k.0 == n) {
            lines.push(format!("  if (KVST > 0) {{ @export_symbol(s2d_d_{n}_ptr, \"{n}\"); }}"));
        } else {
            lines.push(format!("  @export_symbol(s2d_d_{n}_ptr, \"{n}\");"));
        }
    }
    if !own.iter().any(|o| o == "kstage") {
        // The staging launches (KVST > 0) exist on every program: no-ops off the grids.
        decls.push(
            "fn page_stage(page: u32, slot: u32) void { sys_mod.unblock_cmd_stream(); }\n\
fn page_unstage(page: u32, slot: u32) void { sys_mod.unblock_cmd_stream(); }"
                .to_string(),
        );
        lines.push("  if (KVST > 0) { @export_symbol(page_stage); @export_symbol(page_unstage); }".to_string());
    }
    format!(
        "// serve8b: the other programs' exported names.\nparam KVST: u16;   // KV staging slots (0: none)\n{}\ncomptime {{\n{}\n}}\n",
        decls.join("\n"),
        lines.join("\n")
    )
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
        &softmax_s123(src::QWEN2D_PAGED8B_HR, tag)?,
        "param c_r1_0: color;\n",
        "// serve8b: 1 times each pass (prof[0]).\nparam S2D_TIME: u16;\nparam c_r1_0: color;\n",
        "params",
    )?;
    s = patch(&s, "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n", "  s2d_place();\n", "init coordinates")?;
    // paged8b's init calls route_gbcast() directly, or (since its HROUTE
    // param) only when the host does not write the routes.
    let gb = if s.contains("  route_gbcast();\n  axis_y();\n  sys_mod.unblock_cmd_stream();\n}\n") {
        "  route_gbcast();\n"
    } else {
        "  if (HROUTE == 0) { route_gbcast(); }\n"
    };
    s = patch(
        &s,
        &format!("{gb}  axis_y();\n  sys_mod.unblock_cmd_stream();\n}}\n"),
        &format!("{gb}  axis_y();\n  s2d_routes();\n  sys_mod.unblock_cmd_stream();\n}}\n"),
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

// ---------------------------------------------------------------- serve8b_region

/// The grid kernels' HROUTE exports (their own under `HROUTE` 1).
const S8BR_HROUTE_NAMES: &[(&str, &str)] = &[("rcfg", "u32"), ("rbits", "u32")];

/// The head's names on serve8b_region: serve8b's plus the external VM's
/// per-lane control (`vmctl`), in sync.py's order.
const S8BR_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"), ("et", "u32"), ("l2", "f32"), ("force", "u32"), ("lmax", "f32"), ("gmode", "u32"),
    ("vm_tok", "f32"), ("vin", "f32"), ("xin", "f32"), ("gam", "f32"), ("y", "f32"), ("emb", "f32"),
    ("tok", "f32"), ("prof", "u32"), ("in_tokens", "u32"), ("out_tokens", "u32"), ("yscale", "f32"),
    ("vmctl", "u32"), ("hlogits", "f32"),
];
/// Fork 3's eta VM's exports (vm_base.csl, the block east of the strip).
const S8BR_VM_NAMES: &[(&str, &str)] = &[
    ("gwide", "f32"), ("gpool", "u32"), ("gops", "u32"), ("gout", "u32"), ("gprof", "u32"), ("gostep", "u32"),
    ("gring", "u32"), ("vm_ctl", "u32"), ("gbank", "u32"), ("gcycles", "u32"), ("gvalid", "u32"),
    ("gtok_home", "u32"), ("gsin", "u32"), ("glane", "u32"), ("gbdesc", "u32"), ("gsdesc", "u32"), ("gdbg", "u32"),
];
/// KV page staging (the unit kernels' own, under KVST > 0): bf16 slots.
const S8BR_KVST_NAMES: &[(&str, &str)] = &[("kstage", "KS"), ("vstage", "KS")];

/// serve8b_region's dummies: a 1-word array per missing name (the HROUTE
/// and staging ones conditional), and no-op staging launches off the units.
fn s8br_dummies(own: &[String]) -> String {
    let all = Names::of(S8B_GRID_NAMES)
        .with(S8BR_KVST_NAMES)
        .with(S8BR_HEAD_NAMES)
        .with(S8B_EMBED_NAMES)
        .with(S8BR_VM_NAMES)
        .with(S8BR_HROUTE_NAMES);
    let mut lines = Vec::new();
    let mut decls = Vec::new();
    for &(n, ty) in &all.0 {
        if own.iter().any(|o| o == n) {
            continue;
        }
        decls.push(format!("var s2d_d_{n} = @zeros([1]{ty});\nvar s2d_d_{n}_ptr: [*]{ty} = &s2d_d_{n};"));
        if S8BR_HROUTE_NAMES.iter().any(|h| h.0 == n) {
            lines.push(format!("  if (HROUTE == 1) {{ @export_symbol(s2d_d_{n}_ptr, \"{n}\"); }}"));
        } else if S8BR_KVST_NAMES.iter().any(|h| h.0 == n) {
            lines.push(format!("  if (KVST > 0) {{ @export_symbol(s2d_d_{n}_ptr, \"{n}\"); }}"));
        } else {
            lines.push(format!("  @export_symbol(s2d_d_{n}_ptr, \"{n}\");"));
        }
    }
    if !own.iter().any(|o| o == "kstage") {
        // The staging launches (KVST > 0) exist on every program: no-ops off the units.
        decls.push(
            "fn page_stage(page: u32, slot: u32) void { sys_mod.unblock_cmd_stream(); }\n\
fn page_unstage(page: u32, slot: u32) void { sys_mod.unblock_cmd_stream(); }"
                .to_string(),
        );
        lines.push("  if (KVST > 0) { @export_symbol(page_stage); @export_symbol(page_unstage); }".to_string());
    }
    format!(
        "// serve8b: the other programs' exported names.\nparam KVST: u16;   // KV staging slots (0: none)\n{}\ncomptime {{\n{}\n}}\n",
        decls.join("\n"),
        lines.join("\n")
    )
}

/// A unit kernel (qwen2d_paged8b or qwen2d_region) with serve8b_region's
/// grid_extra: the host's placement and handoffs, its pass prologue and its
/// layer(s) lifted out of `compute`.
fn s8br_unit(kernel: &str, src: &str) -> R<String> {
    let tag = format!("{kernel}/pe.csl");
    let mut s = patch(
        &softmax_s123(src, &tag)?,
        "param c_r1_0: color;\n",
        concat!(
            "// serve8b: 1 times each pass (prof[0]); the route table's route capacity.\n",
            "param S2D_TIME: u16;\nparam S2D_RTR: u16;\nparam S2D_NOLAYER: u16;\nparam c_r1_0: color;\n"
        ),
        "params",
    )?;
    s = patch(&s, "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n", "  s2d_place();\n", "init coordinates")?;
    s = patch(
        &s,
        "  axis_y();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "  axis_y();\n  s2d_routes();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "init routes",
    )?;
    let head = "fn compute(steps: u32) void {\n";
    let (start, end) = fn_block(&s, head).ok_or_else(|| format!("gen: {tag} changed: compute() not found"))?;
    let body_at = start + s[start..].find(head).unwrap_or(0) + head.len();
    let body = &s[body_at..end - 3];
    let open = "  while (i < steps) : (i += 1) {\n";
    let mid = "\n    timing = i > 0;\n    pass_start();\n    timestamp.get_timestamp(&t_mark);\n";
    let close = "\n    pass_end();\n  }\n";
    let bad = || format!("gen: {tag} changed: compute()'s pass loop not as expected");
    let lo = body.find(open).ok_or_else(bad)? + open.len();
    let m = lo + body[lo..].find(mid).ok_or_else(bad)?;
    let hi = m + mid.len() + body[m + mid.len()..].find(close).ok_or_else(bad)?;
    let dedent = |t: &str| t.split('\n').map(|l| l.strip_prefix("  ").unwrap_or(l)).collect::<Vec<_>>().join("\n");
    let pass_fn = format!(
        "// The kernel's per-pass prologue (lanes advance) and its layer(s), lifted by sync.py.\n\
fn s2d_pass_start(i: u32, start_pos: u32) void {{\n{}\n}}\nfn s2d_layers() void {{\n{}\n}}\n",
        dedent(&body[lo..m]),
        dedent(&body[m + mid.len()..hi])
    );
    s = format!("{}\n{}", &s[..start], &s[end..]);
    let mut own = exports(&s);
    own.extend(exports(src::s8br::GRID_EXTRA));
    let nlk = if kernel == "qwen2d_region" { "NL" } else { "1" };
    let extra = format!(
        "{S8B_GRID_VARIANT}const S2D_KL: u16 = {nlk};       // layers a unit (KV staging)\n\n{pass_fn}\n{}\n{}",
        src::s8br::GRID_EXTRA,
        s8br_dummies(&own)
    );
    // `\ncomptime {\n(  if (NL > 1)...\n)?  @initialize_queue(q_r1_0_in`
    let mut at = None;
    let mut from = 0;
    while let Some(i) = s[from..].find("\ncomptime {\n") {
        let c = from + i;
        let mut rest = &s[c + "\ncomptime {\n".len()..];
        if rest.starts_with("  if (NL > 1)") {
            rest = rest.find('\n').map_or("", |k| &rest[k + 1..]);
        }
        if rest.starts_with("  @initialize_queue(q_r1_0_in") {
            at = Some(c);
            break;
        }
        from = c + 1;
    }
    let at = at.ok_or_else(|| format!("gen: {tag} changed: the queues' comptime block not found"))?;
    s = format!("{}\n{extra}{}", &s[..at], &s[at..]);
    Ok(format!(
        "// GENERATED by serve8b/sync.py from {tag} + serve8b/grid_extra.csl.\n\
// Do not edit: change grid_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s8br_head() -> R<String> {
    let tag = "head2d_v10/pe.csl";
    let mut s = patch(
        src::HEAD2D_V10,
        "param WF32: u16;\n",
        concat!(
            "param WF32: u16;\n// serve8b: the end grids (GPX x GPY, GHc dims a row; the last at XL, YL), the head's first column XH0,\n",
            "// row), the tokens' capacity, the lanes, the copy kernel, the grids' element\n",
            "// types (their exports).\nparam GPX: u16;\nparam GPY: u16;\nparam GHc: u16;\nparam XH0: u16;\n",
            "param XL: u16;\nparam YL: u16;\nparam CAP: u16;\nparam SB: u16;\nparam HLB: u16;\n",
            "param KVCP: u16;\nparam HROUTE: u16;\n// 1: the VM is the block east of the strip (VM5.md), not a head column.\n",
            "param S2D_VMX: u16;\nparam WS: type;\nparam KS: type;\n"
        ),
        "params",
    )?;
    s = patch(
        &s,
        HEAD_XY_OLD,
        "  px = layout_mod.get_x_coord() - XH0;\n  py = layout_mod.get_y_coord();\n  ry = ",
        "init coordinates",
    )?;
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
    // An external VM block (S2D_VMX, VM5.md): the VM's branches run with no
    // VM column in the head (VMC 0): the root column sends its logits row
    // east on c_vl, the final root takes the VM's token from the east on c_vt.
    let n = s.matches("VMC == 1 and gmode[0] == 1").count();
    if n != 2 {
        return Err(format!("gen: {tag} changed: the VM's gmode branches found {n} times"));
    }
    s = s.replace("VMC == 1 and gmode[0] == 1", "(VMC == 1 or S2D_VMX == 1) and gmode[0] == 1");
    s = patch(
        &s,
        "  if (VMC == 1) {\n    if (root_col) {",
        "  if (VMC == 1 or S2D_VMX == 1) {\n    if (root_col) {",
        "init's VM routes",
    )?;
    s = patch(
        &s,
        "      y[Vr] = yscale;\n      @fmovs(fout(q_x_out, Vr + 1), yv_dsd);\n",
        concat!(
            "      y[Vr] = yscale;\n      if (S2D_VMX == 1) {\n        // The external VM's row: Vr logits, 1/rms, the lane step's ctl word.\n",
            "        y[Vr + 1] = s2d_ctl();\n        @fmovs(fout(q_x_out, Vr + 2), yx_dsd);\n",
            "      } else {\n        @fmovs(fout(q_x_out, Vr + 1), yv_dsd);\n      }\n"
        ),
        "the VM's logits row",
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
    own.extend(["in_tokens", "out_tokens", "yscale", "hlogits", "vmctl"].map(String::from));
    s = format!("{}\n\n{}\n{}", s.trim_end_matches('\n'), src::s8br::HEAD_EXTRA, s8br_dummies(&own));
    Ok(format!(
        "// GENERATED by serve8b/sync.py from {tag} (Fork 7) + serve8b/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s8br_plain(base: &str, base_name: &str, what: &str) -> String {
    format!("// GENERATED by serve8b/sync.py from {base_name} ({what}).\n{base}\n{}", s8br_dummies(&exports(base)))
}

/// vm_pe.csl: Fork 3's VM (vm_base.csl) for the block east of the strip; its
/// compute takes the grids' (steps, start_pos) and runs a row each of the
/// head's steps + 1 iterations x S2D_SB lanes.
fn s8br_vm() -> R<String> {
    let mut s = patch(
        src::s8br::VM_BASE,
        "param memcpy_params;\n",
        concat!(
            "param memcpy_params;\n// serve8b_region: the head's lanes a step, the copy kernel, routes by host,\n",
            "// the grids' element types (the dummies' exports).\nparam S2D_SB: u16;\nparam KVCP: u16;\n",
            "param HROUTE: u16;\nparam WS: type;\nparam KS: type;\n"
        ),
        "params",
    )?;
    s = patch(
        &s,
        "fn compute(steps: u32) void {\n",
        concat!(
            "fn compute(steps_: u32, start_pos: u32) void {\n",
            "  // serve8b_region: a head row each lane step of steps + 1 iterations.\n",
            "  const steps = (steps_ + 1) * @as(u32, S2D_SB);\n"
        ),
        "compute",
    )?;
    let own = exports(&s);
    Ok(format!(
        "// GENERATED by serve8b_region/sync.py from vm_base.csl (Fork 3's eta VM).\n{s}\n\
fn copy_kv(src: u32, dst: u32, phase: u32) void {{ sys_mod.unblock_cmd_stream(); }}\n\
comptime {{\n  if (KVCP == 1) {{ @export_symbol(copy_kv); }}\n}}\n{}",
        s8br_dummies(&own)
    ))
}

/// serve8b_region's build directory.
pub fn serve8b_region() -> R<Vec<File>> {
    Ok(vec![
        ("layout.csl", src::s8br::LAYOUT.to_string()),
        ("paths.csl", src::s8br::PATHS.to_string()),
        ("rt.csl", src::s8br::RT.to_string()),
        ("grid_pe.csl", s8br_unit("qwen2d_paged8b", src::QWEN2D_PAGED8B_HR)?),
        ("region_pe.csl", s8br_unit("qwen2d_region", src::QWEN2D_REGION)?),
        ("head_pe.csl", s8br_head()?),
        ("embed_pe.csl", s8br_plain(src::s8br::EMBED_BASE, "embed_base.csl", "the embedding strip")),
        ("filler.csl", s8br_plain(src::s8br::FILLER_BASE, "filler_base.csl", "the filler")),
        ("vm_stub.csl", s8br_plain(src::s8br::VM_STUB_BASE, "vm_stub_base.csl", "a stand-in for the VM block")),
        ("vm_pe.csl", s8br_vm()?),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated files' blake3, from the designs' sync.py outputs at the
    /// time of the port. A change to a kernel or an extra changes these: rerun
    /// sync.py, diff (`PIE_CEREBRAS_PAYLOAD` below), then update.
    const PINNED: &[(&str, &str, &str)] = &[
        ("serve2d", "grid_pe.csl", "b591b05358b5bdde516e3f65e5eb361402c4f0eef41f74521b52b86e89d604ec"),
        ("serve2d", "grid_paged_pe.csl", "e9b85b9f807ec3ddf12c600006159fed22b42ea26f7cbe9fcd0ed75805608352"),
        ("serve2d", "head_pe.csl", "bdc1e087f3066c21efcf2ae9d75acb489e80d3a9d12630220f623faa86095ffc"),
        ("serve2d", "head_stub.csl", "572f7e7c6f8065a29970ea1785b55023d77d313daf0b2f6051254be096235a1f"),
        ("serve2d", "filler.csl", "9f435cfd600f91915db70d1bd947c202a90a21fe3f72decd974173f977f09211"),
        ("serve2d", "vm_pe.csl", "ac1e9110a0d4e1253d48df48477ddcaf53aad7cf007969b6077eec53f2a88c42"),
        ("serve2d_pipe", "grid_pe.csl", "a5a5663acb00e4f4fa46ea5066d6eb69c317bb665c45ee73d1871ce423ff9ff1"),
        ("serve2d_pipe", "head_pe.csl", "fb24f5dbfd907520336bd07311d9b9ca8554909e6c93b6f6c428985e657525a0"),
        ("serve2d_pipe", "filler.csl", "6ee6aeea07fcdaf3a46c72ab03dcab7bf2c1b92b4f2c12bb29feab9e43b57bb5"),
        ("serve2d_pipe", "vm_pe.csl", "20008b5ed4887f1f8d6dff7fecea16984ed503271c0fb037b58fee8bdfcf1984"),
        ("serve8b", "grid_pe.csl", "04c7a6d99f99b15cd845ef9c3d0ef20aac4ef23d2cae04a2af0315278860b56f"),
        ("serve8b", "head_pe.csl", "c9d5b14ad454fd62eae6c592fd2c2e21b484c59b29c160a1c363e6307bfd054d"),
        ("serve8b", "embed_pe.csl", "991c1ce046039415a39f9afb0c6e2e8673a91ff3e7057637abd6477fc99f39e9"),
        ("serve8b", "filler.csl", "bf9e83043319a16c8bac8d11f9d552c22249cfe16dbfb1f8c4a23316e2dbe843"),
        ("serve8b_region", "grid_pe.csl", "cbb8fe8351a810a725cd7a4db2213930247469684fa52e21c2a473ac1c5c8e87"),
        ("serve8b_region", "region_pe.csl", "7bc64cd4c7280db5a1564a9594c14156bf238a75f7c5274dbb30def72f3b94e3"),
        ("serve8b_region", "head_pe.csl", "ec8349abdb11aabd454cd6113a60f9da5614f374519ab14ea162684c4148fcfe"),
        ("serve8b_region", "embed_pe.csl", "8e2769143d5e799f25483d53fb9b8aeae108b1f3e4a95ebecc65a9d967cdf65b"),
        ("serve8b_region", "filler.csl", "0f8106389be8053db738c6bb2189987e0a5716a31372b537f33dcd5aaa54fff5"),
        ("serve8b_region", "vm_pe.csl", "31fa54184438f17f0666961f60c3f46b4e34ec73b95c4f0dbf3b287ab5549f54"),
        ("serve8b_region", "vm_stub.csl", "de0cae945badccacc9b9583cf74c1e47bb14fd82b8287d1b47a0defca4243dbb"),
    ];

    #[test]
    fn generators_match_their_pins() {
        if std::env::var_os("PIE_CEREBRAS_PRINT_PINS").is_some() {
            for d in [Design::Serve2d, Design::Serve2dPipe, Design::Serve8b, Design::Serve8bRegion] {
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
    /// generated file must equal its sync.py output there byte for byte
    /// (`PIE_CEREBRAS_PAYLOAD_ONLY`: just that design).
    #[test]
    fn generators_match_payload() {
        let Some(root) = std::env::var_os("PIE_CEREBRAS_PAYLOAD") else { return };
        let root = std::path::PathBuf::from(root);
        let only = std::env::var("PIE_CEREBRAS_PAYLOAD_ONLY").ok();
        for d in [Design::Serve2d, Design::Serve2dPipe, Design::Serve8b, Design::Serve8bRegion] {
            if only.as_deref().is_some_and(|o| o != d.name()) {
                continue;
            }
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

    /// Every data `@export_name` of a layout before its first function one
    /// (`init`, `compute`, the copy and staging launches): memcpy's symbol
    /// lookup then stays data; a data name declared after a function turns it
    /// into code (~0.8 KB a PE, as measured on serve8b_region). The programs'
    /// own export order does not matter.
    #[test]
    fn layouts_declare_data_before_functions() {
        let late = |layout: &str| -> Vec<String> {
            let mut seen_fn = false;
            let mut late = Vec::new();
            for rest in layout.split("@export_name(").skip(1) {
                let Some((name, after)) = rest.trim_start().strip_prefix('"').and_then(|r| r.split_once('"')) else {
                    continue;
                };
                if after.trim_start_matches([',', ' ']).starts_with("fn(") {
                    seen_fn = true;
                } else if seen_fn {
                    late.push(name.to_string());
                }
            }
            late
        };
        for d in [Design::Serve2d, Design::Serve2dPipe, Design::Serve8b, Design::Serve8bRegion] {
            let files = d.files().unwrap();
            let layout = &files.iter().find(|f| f.0 == "layout.csl").unwrap().1;
            assert!(!layout_exports(layout).is_empty(), "{}", d.name());
            assert_eq!(late(layout), Vec::<String>::new(), "{}: data exports after a function", d.name());
        }
        assert_eq!(late("@export_name(\"init\", fn()void);\n@export_name(\"x\", [*]f32, true);"), ["x"]);
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
