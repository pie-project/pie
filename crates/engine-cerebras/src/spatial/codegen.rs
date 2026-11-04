//! The wafer programs' PE sources, generated from the kernels in `csl/`.
//!
//! serve2d and serve2d_pipe do not keep their own copies of the decoder layer,
//! the head or the VM. Each per-PE program is a shared kernel (`csl/kernels`,
//! `csl/vm`) patched at checked anchors, with the design's `*_extra.csl`
//! appended and dummy exports for every name another program exports (memcpy
//! needs every PE to export every name). The output is byte-identical to the
//! programs validated on CS-3 (the generated files' headers still name the
//! generator scripts they were first made with); the tests pin every file's
//! hash. A missing anchor is an error, not a silently different program.

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
    /// The fast softmax levers S1-S3 as (old, new) text pairs on
    /// qwen2d_paged8b's or qwen2d_region's pe.csl.
    pub const SOFTMAX_S123: &str = include_str!("csl/kernels/softmax_s123.json");
    /// Their S1 and S3 on the 0.6B layer, qwen2d_paged's or qwen2d_pring's
    /// pe.csl.
    pub const SOFTMAX_S13_06B: &str = include_str!("csl/kernels/softmax_s13_06b.json");
    pub const VM_HM: &str = include_str!("csl/vm/guest4hm_vm.csl");
    pub const VM_HZ: &str = include_str!("csl/vm/guest4hz_vm.csl");
    /// Every design's runtime route and queue-color helpers.
    pub const RT: &str = include_str!("csl/rt.csl");
    /// Qwen3.5's layer units and serve_q35's layout.
    pub const QWEN35_GDN: &str = include_str!("csl/kernels/qwen35_gdn.csl");
    pub const QWEN35_ATTN: &str = include_str!("csl/kernels/qwen35_attn.csl");
    pub const Q35_LAYOUT: &str = include_str!("csl/serve_q35/layout.csl");
    /// Gemma 4's layer unit and serve_g4's own files.
    pub const GEMMA4_LAYER: &str = include_str!("csl/kernels/gemma4_layer.csl");
    pub mod g4 {
        pub const LAYOUT: &str = include_str!("csl/serve_g4/layout.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve_g4/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve_g4/head_extra.csl");
        pub const PLE_BASE: &str = include_str!("csl/serve_g4/ple_base.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve_g4/filler_base.csl");
        pub const PATHS: &str = include_str!("csl/serve_g4/paths.csl");
    }

    pub mod s2d {
        pub const LAYOUT: &str = include_str!("csl/serve2d/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve2d/paths.csl");
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
        pub const GRID_EXTRA: &str = include_str!("csl/serve8b/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve8b/head_extra.csl");
        pub const EMBED_BASE: &str = include_str!("csl/serve8b/embed_base.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve8b/filler_base.csl");
    }

    pub mod s8br {
        pub const LAYOUT: &str = include_str!("csl/serve8b_region/layout.csl");
        pub const PATHS: &str = include_str!("csl/serve8b_region/paths.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve8b_region/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve8b_region/head_extra.csl");
        pub const EMBED_BASE: &str = include_str!("csl/serve8b_region/embed_base.csl");
        pub const FILLER_BASE: &str = include_str!("csl/serve8b_region/filler_base.csl");
        pub const VM_BASE: &str = include_str!("csl/serve8b_region/vm_base.csl");
        pub const VM_STUB_BASE: &str = include_str!("csl/serve8b_region/vm_stub_base.csl");
    }

    /// serve8b_region_pipe: serve8b_region's own files where the ring's
    /// micro-batches change them (the rest are serve8b_region's).
    pub mod s8brp {
        pub const LAYOUT: &str = include_str!("csl/serve8b_region_pipe/layout.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve8b_region_pipe/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve8b_region_pipe/head_extra.csl");
        pub const EMBED_BASE: &str = include_str!("csl/serve8b_region_pipe/embed_base.csl");
    }

    /// serve2d_pipe: its own files where the ring changes them (the paths,
    /// the KV staging and the filler are serve2d's).
    pub mod pipe {
        pub use super::s2d::{FILLER_BASE, KVSTAGE, PATHS};
        pub const LAYOUT: &str = include_str!("csl/serve2d_pipe/layout.csl");
        pub const GRID_EXTRA: &str = include_str!("csl/serve2d_pipe/grid_extra.csl");
        pub const HEAD_EXTRA: &str = include_str!("csl/serve2d_pipe/head_extra.csl");
        pub const VM_EXTRA: &str = include_str!("csl/serve2d_pipe/vm_extra.csl");
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
    /// serve8b_region's units as a pipeline ring: micro-batches round the
    /// units and the head each lap (serve8b_region_pipe).
    Serve8bRegionPipe,
    /// Qwen3.5 dense: serve2d's program around Gated DeltaNet and
    /// gated-attention grids, layer by layer (serve_q35).
    ServeQ35,
}

impl Design {
    pub fn name(self) -> &'static str {
        match self {
            Design::Serve2d => "serve2d",
            Design::Serve2dPipe => "serve2d_pipe",
            Design::Serve8b => "serve8b",
            Design::Serve8bRegion => "serve8b_region",
            Design::Serve8bRegionPipe => "serve8b_region_pipe",
            Design::ServeQ35 => "serve_q35",
        }
    }

    pub fn parse(name: &str) -> Option<Design> {
        match name {
            "serve2d" => Some(Design::Serve2d),
            "serve2d_pipe" => Some(Design::Serve2dPipe),
            "serve8b" => Some(Design::Serve8b),
            "serve8b_region" => Some(Design::Serve8bRegion),
            "serve8b_region_pipe" => Some(Design::Serve8bRegionPipe),
            "serve_q35" => Some(Design::ServeQ35),
            _ => None,
        }
    }

    /// The eta VM image the design's sources carry (enabled by its params).
    pub fn vm_image(self) -> Option<&'static str> {
        match self {
            Design::Serve2d | Design::ServeQ35 => Some("guest4hm"),
            // The guest4hz image: one or two resident sets (VMB).
            Design::Serve2dPipe => Some("guest4hz"),
            Design::Serve8b => None,
            // guest4hz as the block east of the strip (with VMX).
            Design::Serve8bRegion | Design::Serve8bRegionPipe => Some("guest4hz"),
        }
    }

    /// How a KV-staging build's slots hold their elements (`kvst_pack`):
    /// f32 words on serve2d*, bf16 one a word on serve8b_region, bf16 pairs
    /// in serve8b's aliased scratch; and whether they alias it.
    pub fn kv_staging(self) -> (&'static str, bool) {
        match self {
            Design::Serve2d | Design::Serve2dPipe | Design::ServeQ35 => ("f32", false),
            Design::Serve8b => ("bf16x2", true),
            Design::Serve8bRegion | Design::Serve8bRegionPipe => ("bf16", false),
        }
    }

    /// Every file of the program's build directory, `layout.csl` first.
    pub fn files(self) -> R<Vec<File>> {
        self.files_with(Variant::default())
    }

    /// [`Design::files`] for a model `variant`: the default's files are
    /// [`Design::files`]' byte for byte; another variant patches the layer
    /// units (serve8b_region and serve8b_region_pipe only).
    pub fn files_with(self, variant: Variant) -> R<Vec<File>> {
        if variant != Variant::default()
            && !matches!(self, Design::Serve8bRegion | Design::Serve8bRegionPipe)
        {
            return Err(format!(
                "{}: no model variant ({variant:?}); serve8b_region's units take one",
                self.name()
            ));
        }
        match self {
            Design::Serve2d => serve2d(),
            Design::Serve2dPipe => serve2d_pipe(),
            Design::Serve8b => serve8b(),
            Design::Serve8bRegion => serve8b_region(false, variant),
            Design::Serve8bRegionPipe => serve8b_region(true, variant),
            Design::ServeQ35 => serve_q35(),
        }
    }
}

/// What a model changes in the paged8b layer units, as text patches on the
/// kernels (the default patches nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Variant {
    /// No q/k RMSNorm (Llama): the per-head sums of squares, their segment
    /// all-reduce and the two scales are cut; the norm weights (`qkn_w`)
    /// still scale q and k (the host writes 1/sqrt(head_dim) and 1).
    pub identity_qk_norm: bool,
    /// The RMSNorm epsilon's f32 bits, when not the kernels' literal 1e-6.
    pub eps_bits: Option<u32>,
}

impl Variant {
    /// The variant's patches on a paged8b unit kernel (`tag` names it).
    fn apply(self, text: &str, tag: &str) -> R<String> {
        let mut s = text.to_string();
        if self.identity_qk_norm {
            let start = "  const sqa = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{QH} -> s[@as(i16, QH) + i] });\n";
            let end = "  @fmuls(half_v, half_v, scl_t());\n";
            let bad = || format!("gen: {tag} changed: the q/k RMSNorm block not found once");
            if s.matches(start).count() != 1 || s.matches(end).count() != 1 {
                return Err(bad());
            }
            let a = s.find(start).ok_or_else(bad)?;
            let b = a + s[a..].find(end).ok_or_else(bad)? + end.len();
            let block = &s[a..b];
            if !block.contains("allreduce_seg(ssq3_dsd, NH * B);\n")
                || !block.contains("  axis_xx();\n")
            {
                return Err(bad());
            }
            // The axis switch stays right after the QKV all-reduce (the next
            // collective, the scores', is along X); `k` is the layer's.
            s = format!(
                "{}  // (identity q/k norm: no per-head RMSNorm)\n  axis_xx();\n  var k: u16 = 0;\n{}",
                &s[..a],
                &s[b..]
            );
        }
        if let Some(bits) = self.eps_bits {
            let eps = f32::from_bits(bits);
            if !(eps.is_finite() && eps > 0.0) {
                return Err(format!("gen: RMSNorm eps {eps} is not a positive number"));
            }
            let mut lit = format!("{eps:e}");
            if !lit.contains('.') {
                lit = lit.replacen('e', ".0e", 1);
            }
            s = patch(
                &s,
                "const EPS: f32 = 1.0e-6;\n",
                &format!("const EPS: f32 = {lit};\n"),
                &format!("{tag}'s EPS"),
            )?;
        }
        Ok(s)
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
        let ident_len = rest
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        if ident_len == 0 {
            continue;
        }
        let after = &rest[ident_len..];
        if let Some(q) = after.strip_prefix(", \"") {
            let name_len = q
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
                .count();
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
pub(crate) fn layout_exports(layout: &str) -> Vec<String> {
    layout
        .split("@export_name(")
        .skip(1)
        .filter_map(|rest| {
            rest.trim_start()
                .strip_prefix('"')?
                .split_once('"')
                .map(|(n, _)| n.to_string())
        })
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
    ("x", "f32"),
    ("gamma1", "f32"),
    ("gamma2", "f32"),
    ("wqkv", "WS"),
    ("qkn_w", "f32"),
    ("inv_freq", "f32"),
    ("kc", "KS"),
    ("vc", "KS"),
    ("wo", "WS"),
    ("wgu", "WS"),
    ("wd", "WS"),
    ("prof", "u32"),
    ("fallbacks", "u32"),
    ("stamps", "u32"),
    ("tab", "u32"),
    ("wqa", "f32"),
    ("wqv", "f32"),
    ("wv", "f32"),
    ("rot_c", "f32"),
    ("rot_s", "f32"),
    ("dims", "u16"),
    ("kstage", "f32"),
    ("vstage", "f32"),
];
const S2D_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"),
    ("et", "u32"),
    ("l2", "f32"),
    ("force", "u32"),
    ("lmax", "f32"),
    ("gmode", "u32"),
    ("vm_tok", "f32"),
    ("vin", "f32"),
    ("xin", "f32"),
    ("gam", "f32"),
    ("y", "f32"),
    ("emb", "f32"),
    ("tok", "f32"),
    ("prof", "u32"),
    ("in_tokens", "u32"),
    ("out_tokens", "u32"),
    ("yscale", "f32"),
    ("hlogits", "f32"),
];
const S2D_VM_NAMES: &[(&str, &str)] = &[
    ("gwide", "f32"),
    ("gpool", "u32"),
    ("gops", "u32"),
    ("gout", "u32"),
    ("gprof", "u32"),
    ("gostep", "u32"),
    ("gring", "u32"),
    ("gcycles", "u32"),
    ("gvalid", "u32"),
    ("gtok_home", "u32"),
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
    // The softmax levers S1 + S3 on the paged layer (exact on hardware);
    // the kernel sources stay as their owners keep them.
    let src = if kernel == "qwen2d_paged" {
        softmax_s13_06b(src, "qwen2d_paged/pe.csl")?
    } else {
        src.to_string()
    };
    let all = Names::of(S2D_GRID_NAMES)
        .with(S2D_HEAD_NAMES)
        .with(S2D_VM_NAMES);
    grid_gen(&GridGen {
        kernel,
        src: &src,
        params: S2D_GRID_PARAMS,
        variant: &s2d_variant(kernel),
        extra: src::s2d::GRID_EXTRA,
        names: &all,
        header: &format!(
            "// GENERATED by serve2d/sync.py from {kernel}/pe.csl + serve2d/grid_extra.csl.\n\
// Do not edit: change grid_extra.csl or sync.py and rerun.\n"
        ),
    })
}

/// A layer kernel made a serve2d-style grid program: `params` before its
/// own, its init placing and routing the grid (`s2d_place`, `s2d_routes`),
/// its compute replaced by the extra's (the per-pass prologue lifted into
/// `s2d_pass_start`), `variant` + the prologue + `extra` + the KV staging +
/// dummies for `names` before its queue block.
struct GridGen<'a> {
    kernel: &'a str,
    src: &'a str,
    params: &'a str,
    variant: &'a str,
    extra: &'a str,
    names: &'a Names,
    header: &'a str,
}

fn grid_gen(g: &GridGen) -> R<String> {
    let kernel = g.kernel;
    let mut s = patch(
        g.src,
        "param c_r1_0: color;\n",
        &format!("{}param c_r1_0: color;\n", g.params),
        "params",
    )?;
    s = patch(
        &s,
        "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n",
        "  s2d_place();\n",
        "init coordinates",
    )?;
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
            let hdr = [") void {\n", ", start_pos: u32) void {\n"]
                .into_iter()
                .find(|h| rest.starts_with(h));
            if let (Some(h), Some(st)) = (hdr, comment_block_start(&s, at)) {
                let body_at = at + "fn compute(steps: u32".len() + h.len();
                let close = s[body_at..]
                    .find("\n}\n")
                    .ok_or_else(|| format!("gen: {kernel}: compute() unterminated"))?;
                found = Some((
                    st,
                    body_at + close + 3,
                    s[body_at..body_at + close].to_string(),
                ));
                break;
            }
            from = at + 1;
        }
        found.ok_or_else(|| format!("gen: {kernel}/pe.csl changed: compute() not found"))?
    };
    let open = "  while (i < steps) : (i += 1) {\n";
    let close = "\n    timing = i > 0;\n";
    let lo = body
        .find(open)
        .ok_or_else(|| format!("gen: {kernel}: compute()'s per-pass prologue not found"))?
        + open.len();
    let hi = lo
        + body[lo..]
            .find(close)
            .ok_or_else(|| format!("gen: {kernel}: compute()'s per-pass prologue not found"))?;
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
    // The paged grids hold the staging slots; qwen2d's launches only return.
    let stage = if kernel == "qwen2d_paged" || kernel == "qwen35_attn" {
        own.extend(["kstage", "vstage"].map(String::from));
        src::s2d::KVSTAGE
    } else {
        STAGE_FNS
    };
    let extra = format!(
        "{}\n{}\n{}\n{stage}\n{}",
        g.variant,
        pass_fn,
        g.extra,
        s2d_dummies(g.names, &own)
    );
    s = patch(
        &s,
        "\ncomptime {\n  @initialize_queue(q_r1_0_in",
        &format!("\n{extra}\ncomptime {{\n  @initialize_queue(q_r1_0_in"),
        "comptime block",
    )?;
    Ok(format!("{}{s}", g.header))
}

/// head2d's init: `extra` before each of its two unblocks.
fn head_init(s: &str, extra: &str) -> R<String> {
    let at = s
        .find("\nfn init() void {")
        .ok_or("gen: head2d/pe.csl changed: init() not found")?;
    let end = at
        + s[at..]
            .find("\n}\n")
            .ok_or("gen: head2d init() unterminated")?
        + 3;
    let body = &s[at..end];
    if body.matches("sys_mod.unblock_cmd_stream();").count() != 2 {
        return Err("gen: head2d/pe.csl changed: init's unblocks".into());
    }
    let body = body.replace(
        "sys_mod.unblock_cmd_stream();",
        &format!("{extra}\n  sys_mod.unblock_cmd_stream();"),
    );
    Ok(format!("{}{}{}", &s[..at], body, &s[end..]))
}

/// head2d's own `compute` (and the comment block above it) removed.
fn drop_head_compute(s: &str) -> R<String> {
    let c = s
        .find("\n// `steps` head steps")
        .ok_or("gen: head2d/pe.csl changed: compute() not found")?;
    let f = c + s[c..]
        .find("\nfn compute(steps: u32) void {")
        .ok_or("gen: head2d/pe.csl changed: compute() not found")?;
    let end = f
        + s[f..]
            .find("\n}\n")
            .ok_or("gen: head2d compute() unterminated")?
        + 3;
    Ok(format!("{}\n{}", &s[..c], &s[end..]))
}

const VM_COLUMNS_OLD: &str =
    "  vm = VMC >= 1 and cc == R2X + 1;\n  col = if (VMC >= 1 and cc > R2X) cc - 1 else cc;\n";
const VM_COLUMNS_NEW: &str = concat!(
    "  vm = VMC >= 1 and cc > R2X and cc <= R2X + VMC;\n",
    "  col = if (VMC >= 1 and cc > R2X + VMC) cc - VMC else (if (vm) R2X else cc);\n",
);
const HEAD_XY_OLD: &str =
    "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n  ry = ";
const HEAD_XY_NEW: &str =
    "  px = layout_mod.get_x_coord() - GW * GPX;\n  py = layout_mod.get_y_coord();\n  ry = ";

fn s2d_head() -> R<String> {
    s2d_head_with(&HeadGen {
        params: "",
        init: "s2d_head_init();",
        patches: &[],
        extra: src::s2d::HEAD_EXTRA.to_string(),
        names: &Names::of(S2D_GRID_NAMES).with(S2D_VM_NAMES),
        header: "// GENERATED by serve2d/sync.py from head2d_v9/pe.csl (Fork 7) + serve2d/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n",
    })
}

/// serve2d's head2d program and what a design adds: `params` after
/// serve2d's, `init` before init's unblocks, `patches` (old, new, what) on
/// head2d, `extra` (serve2d's head extra, or a design's version of it),
/// dummies for `names`.
struct HeadGen<'a> {
    params: &'a str,
    init: &'a str,
    patches: &'a [(&'a str, &'a str, &'a str)],
    extra: String,
    names: &'a Names,
    header: &'a str,
}

fn s2d_head_with(h: &HeadGen) -> R<String> {
    let mut s = patch(
        src::HEAD2D_V9,
        "param WF32: u16;\n",
        &format!(
            "param WF32: u16;\n// serve2d: the grids beside the head (x0 = GW * GPX, y0 = 0), the tokens' capacity,\n\
// the grids' element types (their exports).\nparam GPX: u16;\nparam GPY: u16;\nparam GW: u16;\n\
param GH: u16;\nparam CAP: u16;\nparam SB: u16;\nparam HLB: u16;\nparam WS: type;\nparam KS: type;\n\
// 2 (bisecting): no embedding chunk is sent on.\nparam S2D_DBG: u16;\n{}",
            h.params
        ),
        "params",
    )?;
    for (old, new, what) in h.patches {
        s = patch(&s, old, new, what)?;
    }
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
    s = head_init(&s, h.init)?;
    s = patch(
        &s,
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  mark(0);\n",
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  s2d_after_x();\n  mark(0);\n",
        "region x arrival",
    )?;
    s = patch(
        &s,
        "  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "  s2d_feeder_in();\n  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "feeder input",
    )?;
    s = patch(
        &s,
        "      ib[0] = if (ok) best[1] else -1.0;\n",
        "      ib[0] = s2d_decide(if (ok) best[1] else -1.0);\n",
        "final root decision",
    )?;
    s = patch(
        &s,
        "      tok[1] = ib[0];\n",
        "      ib[0] = s2d_decide(ib[0]);\n      tok[1] = ib[0];\n",
        "final root decision (gmode)",
    )?;
    let (t, n) = after_each_call(&s, "emb_copy();", "s2d_emb_out();");
    if n != 2 {
        return Err(format!(
            "gen: head2d/pe.csl changed: emb_copy() called {n} times"
        ));
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
        h.extra,
        s2d_dummies(h.names, &own),
    );
    s = format!("{}\n\n{}", s.trim_end_matches('\n'), extra);
    Ok(format!("{}{s}", h.header))
}

fn s2d_plain(base: &str, base_name: &str, what: &str) -> String {
    let all = Names::of(S2D_GRID_NAMES)
        .with(S2D_HEAD_NAMES)
        .with(S2D_VM_NAMES);
    format!(
        "// GENERATED by serve2d/sync.py from {base_name} ({what}).\n{base}\n{}{COPY_FNS}{STAGE_FNS}",
        s2d_dummies(&all, &[])
    )
}

const ONE_COLUMN_K: &str = "      color_config.reset_routes(col_of(C_K), .{ .rx = RAMP, .tx = [3]direction{ WEST, NORTH, SOUTH } });\n";

fn one_column_result(s: &str, what: &str) -> R<String> {
    let k2 = ONE_COLUMN_K.replace(
        "[3]direction{ WEST, NORTH, SOUTH }",
        "[2]direction{ NORTH, SOUTH }",
    );
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
        s = patch(
            &s,
            "param ring_w: u16;\n",
            "param ring_w: u16;\nparam vm_prof: u16;     // unused here\n",
            "vm_prof",
        )?;
    }
    s = patch(
        &s,
        "  vx = layout_mod.get_x_coord();\n",
        "  vx = layout_mod.get_x_coord() - GW * GPX;\n",
        "head-relative x",
    )?;
    s = patch(
        &s,
        "  timestamp.enable_tsc();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "  s2d_vm_paths();\n  timestamp.enable_tsc();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "init routes",
    )?;
    s = one_column_result(&s, "one-block result route")?;
    let (start, end) = fn_block(&s, "fn compute(steps: u32) void {")
        .ok_or("gen: vm.csl changed: compute() not found")?;
    s = format!("{}\n{}{}", &s[..start], src::s2d::VM_EXTRA, &s[end..]);
    let own = exports(&s);
    let all = Names::of(S2D_GRID_NAMES).with(S2D_HEAD_NAMES);
    s = format!(
        "{}\n\n{}{COPY_FNS}{STAGE_FNS}",
        s.trim_end_matches('\n'),
        s2d_dummies(&all, &own)
    );
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
        ("rt.csl", src::RT.to_string()),
        ("grid_pe.csl", s2d_grid("qwen2d", src::QWEN2D)?),
        (
            "grid_paged_pe.csl",
            s2d_grid("qwen2d_paged", src::QWEN2D_PAGED)?,
        ),
        ("head_pe.csl", s2d_head()?),
        (
            "head_stub.csl",
            s2d_plain(src::s2d::STUB_BASE, "stub_base.csl", "the stub head"),
        ),
        (
            "filler.csl",
            s2d_plain(src::s2d::FILLER_BASE, "filler_base.csl", "the filler"),
        ),
        ("vm_pe.csl", s2d_vm()?),
    ])
}

// ---------------------------------------------------------------- serve_q35

/// The names only serve_q35's units export: the Gated DeltaNet unit's
/// tables and state, the attention unit's gates.
const Q35_NAMES: &[(&str, &str)] = &[
    ("wp", "f32"),
    ("cw", "f32"),
    ("cst", "f32"),
    ("gab", "f32"),
    ("nw", "f32"),
    ("st", "f32"),
    ("wg", "f32"),
];

/// A slot reset that only returns (every PE but the GDN unit's).
const RESET_FN: &str = "fn reset(b: u32) void { sys_mod.unblock_cmd_stream(); }\ncomptime {\n  @export_symbol(reset);\n}\n";

/// serve2d's program `text` made serve_q35's: a dummy for each serve_q35
/// name it does not export, and `fns`.
fn q35_more(text: String, fns: &str) -> String {
    let own = exports(&text);
    let mut decls = Vec::new();
    let mut lines = Vec::new();
    for &(n, ty) in Q35_NAMES {
        if own.iter().any(|o| o == n) {
            continue;
        }
        if decls.is_empty() {
            decls.push(format!(
                "var q35_dum = @zeros([1]{ty});\nvar q35_dum_ptr: [*]{ty} = &q35_dum;"
            ));
        }
        lines.push(format!("  @export_symbol(q35_dum_ptr, \"{n}\");"));
    }
    let dummies = if lines.is_empty() {
        String::new()
    } else {
        format!(
            "// serve_q35: the units' exported names.\n{}\ncomptime {{\n{}\n}}\n",
            decls.join("\n"),
            lines.join("\n")
        )
    };
    format!(
        "// serve_q35: serve2d's generated program with the qwen35 units' names.\n{}\n{dummies}{fns}",
        text.trim_end_matches('\n')
    )
}

/// serve_q35's build directory: serve2d's handoffs, head, filler and VM
/// around Qwen3.5's two units.
pub fn serve_q35() -> R<Vec<File>> {
    // The GDN unit has no pool: its copy and staging launches only return.
    let gdn = q35_more(s2d_grid("qwen35_gdn", src::QWEN35_GDN)?, COPY_FNS);
    let attn = q35_more(s2d_grid("qwen35_attn", src::QWEN35_ATTN)?, RESET_FN);
    Ok(vec![
        ("layout.csl", src::Q35_LAYOUT.to_string()),
        ("paths.csl", src::s2d::PATHS.to_string()),
        ("rt.csl", src::RT.to_string()),
        ("gdn_pe.csl", gdn),
        ("attn_pe.csl", attn),
        ("head_pe.csl", q35_more(s2d_head()?, RESET_FN)),
        (
            "head_stub.csl",
            q35_more(
                s2d_plain(src::s2d::STUB_BASE, "stub_base.csl", "the stub head"),
                RESET_FN,
            ),
        ),
        (
            "filler.csl",
            q35_more(
                s2d_plain(src::s2d::FILLER_BASE, "filler_base.csl", "the filler"),
                RESET_FN,
            ),
        ),
        ("vm_pe.csl", q35_more(s2d_vm()?, RESET_FN)),
    ])
}

// ---------------------------------------------------------------- serve_g4

/// Every name serve_g4's layout exports (data), with its type in the
/// programs (`WS`: u16 weights).
const G4_NAMES: &[(&str, &str)] = &[
    ("x", "f32"),
    ("tab", "u32"),
    ("gamma1", "f32"),
    ("gamma2", "f32"),
    ("gamma3", "f32"),
    ("gamma4", "f32"),
    ("gamma5", "f32"),
    ("scal", "f32"),
    ("kvi", "f32"),
    ("kvo", "f32"),
    ("x0", "f32"),
    ("prow", "f32"),
    ("wqa", "WS"),
    ("wqv", "WS"),
    ("wv", "WS"),
    ("wpm", "WS"),
    ("wpg", "WS"),
    ("wpp", "WS"),
    ("pnw", "f32"),
    ("qkn_w", "f32"),
    ("rot_c", "f32"),
    ("rot_s", "f32"),
    ("kc", "KS"),
    ("vc", "KS"),
    ("wo", "WS"),
    ("wgu", "WS"),
    ("wd", "WS"),
    ("prof", "u32"),
    ("dims", "u16"),
    ("fallbacks", "u32"),
    ("stamps", "u32"),
    ("kstage", "f32"),
    ("vstage", "f32"),
    ("wt", "u32"),
    ("et", "u32"),
    ("l2", "f32"),
    ("force", "u32"),
    ("lmax", "f32"),
    ("gmode", "u32"),
    ("vm_tok", "f32"),
    ("vin", "f32"),
    ("xin", "f32"),
    ("gam", "f32"),
    ("y", "f32"),
    ("emb", "f32"),
    ("tok", "f32"),
    ("in_tokens", "u32"),
    ("out_tokens", "u32"),
    ("yscale", "f32"),
    ("hlogits", "f32"),
    ("ptab", "u32"),
];

const G4_GRID_PARAMS: &str = "// serve_g4: the tail's kv a row of the handoff (KVW a lane) and this layer's\n\
// slot in it (KO), the PLE row's u32 words a lane (NWW), f32 bits of sqrt(hidden)\n\
// and sqrt(PLE) / sqrt(2), the staircases crossing the grid (1 E, 2 S).\nparam S2D_PATHS: u16;\nparam KVW: u16;\nparam KO: u16;\nparam NWW: u16;\nparam ESB: u32;\nparam PSB: u32;\n";

const G4_VARIANT: &str = "const PX: u16 = P;                // the square grid's sides\n\
const PY: u16 = P;\n\
// Queue 2: free in gemma4_layer; serve2d's handoffs and the PLE row borrow it.\n\
const q_gb_in = @get_input_queue(2);\n\
const q_gb_out = @get_output_queue(2);\n\
const S2D_Q2C = c_cp;             // its resting color (c_cp is unused there)\n\
comptime {\n  @initialize_queue(q_gb_in, .{ .color = c_cp });\n  @initialize_queue(q_gb_out, .{ .color = c_cp });\n}\n\
const S2D_XN: u16 = B * Hc;       // x's words a PE: B lanes of Hc, lane-major\n\
const WS = u16;                   // the exports' types\n\
const KS = f32;\n";

/// gemma4_layer with serve_g4's hooks: the tail's kv onto its columns after
/// the QKV all-reduce, a source's onto the rows after the layer; a tail
/// layer (OWN 0) holds no v projection (the host writes no `wv` there).
fn g4_kernel() -> R<String> {
    let s = patch(
        src::GEMMA4_LAYER,
        "var wv: [Hc * Dc + 1]WST align(16) = @zeros([Hc * Dc + 1]WST);",
        "const WVN: u16 = if (OWN == 1) Hc * Dc + 1 else 2;\nvar wv: [WVN]WST align(16) = @zeros([WVN]WST);",
        "wv",
    )?;
    let s = patch(
        &s,
        "      mvparts(at(v_dsd, b * Dc), wv_dsd, Hc);\n",
        "      if (OWN == 1) { mvparts(at(v_dsd, b * Dc), wv_dsd, Hc); }\n",
        "v projection",
    )?;
    let s = patch(
        &s,
        "  allreduce(false, qkv_all, QKV_N);\n",
        "  allreduce(false, qkv_all, QKV_N);\n  g4_kv_in();\n",
        "QKV all-reduce",
    )?;
    patch(
        &s,
        "  if (PC > 0) { ple(); }\n  @fmuls(x_all, x_all, scal[0]);\n",
        "  if (PC > 0) { ple(); }\n  @fmuls(x_all, x_all, scal[0]);\n  g4_kv_out();\n",
        "layer end",
    )
}

/// serve_g4's build directory: serve2d's handoffs (with x0 and the tail's
/// kv), staircases and f32 head2d around gemma4_layer grids, the PLE region.
pub fn serve_g4() -> R<Vec<File>> {
    let names = Names::of(G4_NAMES);
    let grid = grid_gen(&GridGen {
        kernel: "gemma4_layer",
        src: &g4_kernel()?,
        params: &format!("{S2D_GRID_PARAMS}{G4_GRID_PARAMS}"),
        variant: G4_VARIANT,
        extra: src::g4::GRID_EXTRA,
        names: &names,
        header: "// GENERATED by codegen::serve_g4 from gemma4_layer.csl + serve_g4/grid_extra.csl.\n",
    })?;
    let head = s2d_head_with(&HeadGen {
        params: "// serve_g4: the layers (the last one's S staircase), the PLE region east of the block.\n\
param G4_NL: u16;\nparam G4_EAST: u16;\n",
        init: "s2d_head_init();\n  g4_head_init();",
        patches: &[(
            "    if (px == W - 1) { color_config.reset_routes(c_ib, .{ .rx = WEST, .tx = RAMP }); }\n",
            "    if (px == W - 1 and G4_EAST == 0) { color_config.reset_routes(c_ib, .{ .rx = WEST, .tx = RAMP }); }\n",
            "winner east end",
        )],
        extra: format!(
            "{}\n{}",
            patch(
                src::s2d::HEAD_EXTRA,
                ".NL = GW * GH, .XH = GW * GPX,",
                ".NL = G4_NL, .XH = GW * GPX,",
                "head paths NL",
            )?,
            src::g4::HEAD_EXTRA
        ),
        names: &names,
        header: "// GENERATED by codegen::serve_g4 from head2d_v9.csl + serve2d/head_extra.csl + serve_g4/head_extra.csl.\n",
    })?;
    let plain = |base: &str, what: &str| {
        let own = exports(base);
        format!(
            "// GENERATED by codegen::serve_g4 ({what}).\n{base}\n{}{COPY_FNS}{STAGE_FNS}",
            s2d_dummies(&names, &own)
        )
    };
    Ok(vec![
        ("layout.csl", src::g4::LAYOUT.to_string()),
        ("paths.csl", src::g4::PATHS.to_string()),
        ("rt.csl", src::RT.to_string()),
        ("grid_pe.csl", grid),
        ("head_pe.csl", head),
        ("ple_pe.csl", plain(src::g4::PLE_BASE, "the PLE region")),
        ("filler.csl", plain(src::g4::FILLER_BASE, "the filler")),
    ])
}

// ---------------------------------------------------------------- serve2d_pipe

const PIPE_GRID_NAMES: &[(&str, &str)] = &[
    ("x", "f32"),
    ("gamma1", "f32"),
    ("gamma2", "f32"),
    ("wqa", "f32"),
    ("wqv", "f32"),
    ("wv", "f32"),
    ("qkn_w", "f32"),
    ("rot_c", "f32"),
    ("rot_s", "f32"),
    ("kc", "f32"),
    ("vc", "f32"),
    ("wo", "f32"),
    ("wgu", "f32"),
    ("wd", "f32"),
    ("prof", "u32"),
    ("dims", "u16"),
    ("fallbacks", "u32"),
    ("kstage", "f32"),
    ("vstage", "f32"),
];
const PIPE_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"),
    ("et", "u32"),
    ("l2", "f32"),
    ("force", "u32"),
    ("lmax", "f32"),
    ("gmode", "u32"),
    ("vm_tok", "f32"),
    ("vin", "f32"),
    ("xin", "f32"),
    ("gam", "f32"),
    ("y", "f32"),
    ("emb", "f32"),
    ("tok", "f32"),
    ("prof", "u32"),
    ("in_tokens", "u32"),
    ("out_tokens", "u32"),
    ("yscale", "f32"),
    ("hlogits", "f32"),
    ("mtab", "f32"),
    ("spec_ctl", "u32"),
    ("spec_hist", "u32"),
    ("spec_out", "u32"),
    ("spec_acc", "u32"),
    ("smp", "u32"),
    ("vmlane", "u32"),
];
const PIPE_VM_NAMES: &[(&str, &str)] = &[
    ("gwide", "f32"),
    ("gpool", "u32"),
    ("gops", "u32"),
    ("gout", "u32"),
    ("gprof", "u32"),
    ("gostep", "u32"),
    ("gring", "u32"),
    ("gcycles", "u32"),
    ("gvalid", "u32"),
    ("gtok_home", "u32"),
    ("gbank", "u32"),
    ("vm_ctl", "u32"),
    ("gsin", "u32"),
    ("gsdesc", "u32"),
    ("glane", "u32"),
    ("gbdesc", "u32"),
    ("gdbg", "u32"),
];

/// serve2d_pipe's dummies: one shared 1-word dummy a type, every missing
/// name exported off it (a PE pays the export entries, not a word and a
/// pointer a name).
fn pipe_dummies(names: &Names, own: &[String]) -> String {
    let mut lines = Vec::new();
    let mut types: Vec<&str> = Vec::new();
    for &(n, ty) in &names.0 {
        if own.iter().any(|o| o == n) {
            continue;
        }
        if !types.contains(&ty) {
            types.push(ty);
        }
        lines.push(format!("  @export_symbol(s2d_d_{ty}_ptr, \"{n}\");"));
    }
    let decls: Vec<String> = types
        .iter()
        .map(|ty| {
            format!("var s2d_d_{ty} = @zeros([1]{ty});\nvar s2d_d_{ty}_ptr: [*]{ty} = &s2d_d_{ty};")
        })
        .collect();
    format!(
        "// serve2d_pipe: the other programs' exported names.\n{}\ncomptime {{\n{}\n}}\n",
        decls.join("\n"),
        lines.join("\n")
    )
}

fn pipe_all() -> Names {
    Names::of(PIPE_GRID_NAMES)
        .with(PIPE_HEAD_NAMES)
        .with(PIPE_VM_NAMES)
}

fn pipe_grid() -> R<String> {
    // The softmax levers S1 + S3 (exact on hardware).
    let src = softmax_s13_06b(src::QWEN2D_PRING, "qwen2d_pring/pe.csl")?;
    let mut s = patch(
        &src,
        "param c_r1_0: color;\n",
        "// serve2d_pipe: the grids' arrangement, the head's feeder row and VM column.\n\
param GW: u16;\nparam GH: u16;\nparam NL: u16;\nparam F: u16;\n\
param S2D_VMC: u16;\nparam S2D_VMAT: u16;\n\
// KV page staging slots a PE (kvstage.csl; 0: none).\nparam KVST: u16;\nparam c_r1_0: color;\n",
        "params",
    )?;
    s = patch(
        &s,
        "  lx = layout_mod.get_x_coord() % P;\n",
        "  s2d_place();\n",
        "init place",
    )?;
    s = patch(
        &s,
        "  px = lx;\n  py = layout_mod.get_y_coord();\n",
        "",
        "init coordinates",
    )?;
    s = patch(
        &s,
        "  route_handoff();\n",
        "  s2d_routes();\n",
        "init routes",
    )?;
    s = patch(
        &s,
        "  if (stage == K - 1) {\n    b = 0;",
        "  if (L == NL - 1) {\n    b = 0;",
        "last stage advances",
    )?;
    s = patch(
        &s,
        "  if (more and lx == P - 1) {\n    @fmovs(fout(q_h_out, NIO), io_dsd);\n  }\n",
        "  s2d_send();\n",
        "send",
    )?;
    s = patch(
        &s,
        "task recvd_task() void {\n  @fmovs(io_dsd, in_dsd);",
        "task recvd_task() void {\n  s2d_relay();\n  @fmovs(io_dsd, in_dsd);",
        "relay",
    )?;
    s = patch(
        &s,
        "fn compute(steps: u32) void {\n  timestamp.enable_tsc();\n  steps_total = steps;\n  pass_i = 0;\n  @activate(STEP);\n}\n",
        "fn compute(laps: u32, nmb: u32) void {\n  timestamp.enable_tsc();\n  // nmb's high half: the head's launch flags (DEFER).\n  steps_total = laps * (nmb & 0xffff);\n  pass_i = 0;\n  s2d_start();\n}\n",
        "compute",
    )?;
    let mut own = exports(&s);
    own.extend(["kstage", "vstage"].map(String::from));
    let extra = format!(
        "{}\n{}\n{}",
        src::pipe::GRID_EXTRA,
        src::pipe::KVSTAGE,
        pipe_dummies(&pipe_all(), &own)
    );
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
        let name = decl
            .split_whitespace()
            .nth(1)
            .unwrap_or("")
            .trim_end_matches(':');
        let z = format!("if (S2D_ROOTC == 1) {n} else 1");
        s = patch(
            &s,
            decl,
            &format!("var {name}: [{z}]f32 align(16) = @zeros([{z}]f32);"),
            &format!("{name} by role"),
        )?;
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
            "in_tokens",
            "out_tokens",
            "yscale",
            "hlogits",
            "mtab",
            "spec_ctl",
            "spec_hist",
            "spec_out",
            "spec_acc",
            "smp",
            "vmlane",
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
        src::VM_HZ,
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
    s = patch(
        &s,
        "fn compute(steps_arg: u32) void {",
        "fn s2d_vm_run(steps_arg: u32) void {",
        "compute",
    )?;
    s = s.replace("  @export_symbol(compute);", "");
    let own = exports(&s);
    s = format!(
        "{}\n\n{}\ncomptime {{\n  @export_symbol(compute);\n}}\n{}{COPY_FNS}{STAGE_FNS}",
        s.trim_end_matches('\n'),
        src::pipe::VM_EXTRA,
        pipe_dummies(&pipe_all(), &own),
    );
    Ok(format!(
        "// GENERATED by serve2d_pipe/sync.py from guest4hz_vm.csl + serve2d_pipe/vm_extra.csl.\n\
// Do not edit: change vm_extra.csl or sync.py and rerun.\n{s}"
    ))
}

/// serve2d_pipe's build directory.
pub fn serve2d_pipe() -> R<Vec<File>> {
    Ok(vec![
        ("layout.csl", src::pipe::LAYOUT.to_string()),
        ("paths.csl", src::pipe::PATHS.to_string()),
        ("rt.csl", src::RT.to_string()),
        ("grid_pe.csl", pipe_grid()?),
        ("head_pe.csl", pipe_head()?),
        ("filler.csl", pipe_filler()),
        ("vm_pe.csl", pipe_vm()?),
    ])
}

// ---------------------------------------------------------------- serve8b

/// The fast softmax levers (S1: the max-shift and rescale as one 2-D op, exp
/// over act-sized chunks; S2: the max and sum folds as a halving tree; S3:
/// the exp sums ride the values all-reduce) on an 8B unit kernel's text,
/// each anchor found once.
fn softmax_s123(text: &str, tag: &str) -> R<String> {
    apply_reps("softmax_s123", src::SOFTMAX_S123, text, tag)
}

/// The softmax levers S1 and S3 on the 0.6B paged layer
/// (qwen2d_paged / qwen2d_pring; S2 is already there).
fn softmax_s13_06b(text: &str, tag: &str) -> R<String> {
    apply_reps("softmax_s13_06b", src::SOFTMAX_S13_06B, text, tag)
}

/// A JSON list of {old, new} text pairs applied in order, each anchor found once.
fn apply_reps(name: &str, json: &str, text: &str, tag: &str) -> R<String> {
    let reps: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("{name}.json: {e}"))?;
    let mut s = text.to_string();
    for r in reps.as_array().ok_or(format!("{name}.json: not a list"))? {
        let (old, new) = (
            r["old"].as_str().unwrap_or(""),
            r["new"].as_str().unwrap_or(""),
        );
        let n = s.matches(old).count();
        if old.is_empty() || n != 1 {
            return Err(format!("{name}: {tag} changed: anchor found {n} times"));
        }
        s = s.replacen(old, new, 1);
    }
    Ok(s)
}

const S8B_GRID_NAMES: &[(&str, &str)] = &[
    ("x", "f32"),
    ("tab", "u32"),
    ("gamma1", "f32"),
    ("gamma2", "f32"),
    ("wqa", "WS"),
    ("wqv", "WS"),
    ("wv", "WS"),
    ("qkn_w", "f32"),
    ("rot_c", "f32"),
    ("rot_s", "f32"),
    ("kc", "KS"),
    ("vc", "KS"),
    ("wo", "WS"),
    ("wgu", "WS"),
    ("wd", "WS"),
    ("prof", "u32"),
    ("dims", "u16"),
    ("rtab", "u16"),
];
const S8B_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"),
    ("et", "u32"),
    ("l2", "f32"),
    ("force", "u32"),
    ("lmax", "f32"),
    ("gmode", "u32"),
    ("vm_tok", "f32"),
    ("vin", "f32"),
    ("xin", "f32"),
    ("gam", "f32"),
    ("y", "f32"),
    ("emb", "f32"),
    ("tok", "f32"),
    ("prof", "u32"),
    ("in_tokens", "u32"),
    ("out_tokens", "u32"),
    ("yscale", "f32"),
    ("hlogits", "f32"),
];
const S8B_EMBED_NAMES: &[(&str, &str)] = &[("etab", "u32")];
/// KV page staging (the grids' own, under KVST > 0): f32 words aliasing the
/// grid's `s` / `act`, two bf16 cells a word.
const S8B_KVST_NAMES: &[(&str, &str)] = &[("kstage", "f32"), ("vstage", "f32")];

/// serve8b's dummies: a 1-word array per missing name (the staging slots
/// under `KVST > 0`), and no-op staging launches on programs without slots.
fn s8b_dummies(own: &[String]) -> String {
    let all = Names::of(S8B_GRID_NAMES)
        .with(S8B_KVST_NAMES)
        .with(S8B_HEAD_NAMES)
        .with(S8B_EMBED_NAMES);
    let mut lines = Vec::new();
    let mut decls = Vec::new();
    for &(n, ty) in &all.0 {
        if own.iter().any(|o| o == n) {
            continue;
        }
        decls.push(format!(
            "var s2d_d_{n} = @zeros([1]{ty});\nvar s2d_d_{n}_ptr: [*]{ty} = &s2d_d_{n};"
        ));
        if S8B_KVST_NAMES.iter().any(|k| k.0 == n) {
            lines.push(format!(
                "  if (KVST > 0) {{ @export_symbol(s2d_d_{n}_ptr, \"{n}\"); }}"
            ));
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
        lines.push(
            "  if (KVST > 0) { @export_symbol(page_stage); @export_symbol(page_unstage); }"
                .to_string(),
        );
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
    s = patch(
        &s,
        "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n",
        "  s2d_place();\n",
        "init coordinates",
    )?;
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
    let (start, end) = fn_block(&s, "fn compute(steps: u32) void {\n")
        .ok_or_else(|| format!("gen: {tag} changed: compute() not found"))?;
    let body_at = start
        + s[start..]
            .find("fn compute(steps: u32) void {\n")
            .unwrap_or(0)
        + "fn compute(steps: u32) void {\n".len();
    let body = &s[body_at..end - 3];
    let open = "  while (i < steps) : (i += 1) {\n";
    let close = "\n    timing = i > 0;\n    pass_start();\n    timestamp.get_timestamp(&t_mark);\n    layer();\n    pass_end();\n  }\n";
    let lo = body
        .find(open)
        .ok_or_else(|| format!("gen: {tag} changed: compute()'s pass loop not as expected"))?
        + open.len();
    let hi = lo
        + body[lo..]
            .find(close)
            .ok_or_else(|| format!("gen: {tag} changed: compute()'s pass loop not as expected"))?;
    let prologue: Vec<&str> = body[lo..hi]
        .split('\n')
        .map(|l| l.strip_prefix("  ").unwrap_or(l))
        .collect();
    let prologue = prologue.join("\n");
    s = format!("{}\n{}", &s[..start], &s[end..]);
    let pass_fn = format!(
        "// The kernel's per-pass prologue (lanes advance), lifted by sync.py.\n\
fn s2d_pass_start(i: u32, start_pos: u32) void {{\n{prologue}\n}}\n"
    );
    let mut own = exports(&s);
    own.extend(exports(src::s8b::GRID_EXTRA));
    let extra = format!(
        "{S8B_GRID_VARIANT}\n{pass_fn}\n{}\n{}",
        src::s8b::GRID_EXTRA,
        s8b_dummies(&own)
    );
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
    s = patch(
        &s,
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  mark(0);\n",
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  s2d_after_x();\n  mark(0);\n",
        "region x arrival",
    )?;
    s = patch(
        &s,
        "  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "  s2d_feeder_in();\n  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "feeder input",
    )?;
    s = patch(
        &s,
        "      ib[0] = if (ok) best[1] else -1.0;\n",
        "      ib[0] = s2d_decide(if (ok) best[1] else -1.0);\n",
        "final root decision",
    )?;
    s = patch(
        &s,
        "      tok[1] = ib[0];\n",
        "      ib[0] = s2d_decide(ib[0]);\n      tok[1] = ib[0];\n",
        "final root decision (gmode)",
    )?;
    let (t, n) = after_each_call(&s, "emb_copy();", "s2d_emb_out();");
    if n != 2 {
        return Err(format!("gen: {tag} changed: emb_copy() called {n} times"));
    }
    s = drop_head_compute(&t)?;
    let mut own = exports(&s);
    own.extend(["in_tokens", "out_tokens", "yscale", "hlogits"].map(String::from));
    s = format!(
        "{}\n\n{}\n{}",
        s.trim_end_matches('\n'),
        src::s8b::HEAD_EXTRA,
        s8b_dummies(&own)
    );
    Ok(format!(
        "// GENERATED by serve8b/sync.py from {tag} (Fork 7) + serve8b/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s8b_plain(base: &str, base_name: &str, what: &str) -> String {
    format!(
        "// GENERATED by serve8b/sync.py from {base_name} ({what}).\n{base}\n{}",
        s8b_dummies(&exports(base))
    )
}

/// serve8b's build directory.
pub fn serve8b() -> R<Vec<File>> {
    Ok(vec![
        ("layout.csl", src::s8b::LAYOUT.to_string()),
        ("paths.csl", src::s8b::PATHS.to_string()),
        ("rt.csl", src::RT.to_string()),
        ("grid_pe.csl", s8b_grid()?),
        ("head_pe.csl", s8b_head()?),
        (
            "embed_pe.csl",
            s8b_plain(
                src::s8b::EMBED_BASE,
                "embed_base.csl",
                "the embedding strip",
            ),
        ),
        (
            "filler.csl",
            s8b_plain(src::s8b::FILLER_BASE, "filler_base.csl", "the filler"),
        ),
    ])
}

// ---------------------------------------------------------------- serve8b_region

/// The grid kernels' HROUTE exports (their own under `HROUTE` 1).
const S8BR_HROUTE_NAMES: &[(&str, &str)] = &[("rcfg", "u32"), ("rbits", "u32")];

/// The head's names on serve8b_region: serve8b's plus the external VM's
/// per-lane control (`vmctl`), in the generated program's order.
const S8BR_HEAD_NAMES: &[(&str, &str)] = &[
    ("wt", "u32"),
    ("et", "u32"),
    ("l2", "f32"),
    ("force", "u32"),
    ("lmax", "f32"),
    ("gmode", "u32"),
    ("vm_tok", "f32"),
    ("vin", "f32"),
    ("xin", "f32"),
    ("gam", "f32"),
    ("y", "f32"),
    ("emb", "f32"),
    ("tok", "f32"),
    ("prof", "u32"),
    ("in_tokens", "u32"),
    ("out_tokens", "u32"),
    ("yscale", "f32"),
    ("vmctl", "u32"),
    ("hlogits", "f32"),
];
/// The eta VM's exports (vm_base.csl, the block east of the strip).
const S8BR_VM_NAMES: &[(&str, &str)] = &[
    ("gwide", "f32"),
    ("gpool", "u32"),
    ("gops", "u32"),
    ("gout", "u32"),
    ("gprof", "u32"),
    ("gostep", "u32"),
    ("gring", "u32"),
    ("vm_ctl", "u32"),
    ("gbank", "u32"),
    ("gcycles", "u32"),
    ("gvalid", "u32"),
    ("gtok_home", "u32"),
    ("gsin", "u32"),
    ("glane", "u32"),
    ("gbdesc", "u32"),
    ("gsdesc", "u32"),
    ("gdbg", "u32"),
];
/// KV page staging (the unit kernels' own, under KVST > 0): bf16 slots.
const S8BR_KVST_NAMES: &[(&str, &str)] = &[("kstage", "KS"), ("vstage", "KS")];

/// serve8b_region_pipe's names past serve8b_region's: the units' micro-batch
/// tables, and the head's micro-batch check (before `hlogits`).
const S8BRP_GRID_NAMES: &[(&str, &str)] = &[("mtab", "u32")];
const S8BRP_HEAD_NAMES: &[(&str, &str)] = &[("perr", "u32")];

/// serve8b_region's dummies: a 1-word array per missing name (the HROUTE
/// and staging ones conditional), and no-op staging launches off the units.
fn s8br_dummies(own: &[String], pipe: bool) -> String {
    let (grid, head): (&[_], &[_]) = if pipe {
        (S8BRP_GRID_NAMES, S8BRP_HEAD_NAMES)
    } else {
        (&[], &[])
    };
    let (hlogits, rest) = S8BR_HEAD_NAMES.split_last().unwrap();
    let all = Names::of(S8B_GRID_NAMES)
        .with(grid)
        .with(S8BR_KVST_NAMES)
        .with(rest)
        .with(head)
        .with(&[*hlogits])
        .with(S8B_EMBED_NAMES)
        .with(S8BR_VM_NAMES)
        .with(S8BR_HROUTE_NAMES);
    let mut lines = Vec::new();
    let mut decls = Vec::new();
    for &(n, ty) in &all.0 {
        if own.iter().any(|o| o == n) {
            continue;
        }
        decls.push(format!(
            "var s2d_d_{n} = @zeros([1]{ty});\nvar s2d_d_{n}_ptr: [*]{ty} = &s2d_d_{n};"
        ));
        if S8BR_HROUTE_NAMES.iter().any(|h| h.0 == n) {
            lines.push(format!(
                "  if (HROUTE == 1) {{ @export_symbol(s2d_d_{n}_ptr, \"{n}\"); }}"
            ));
        } else if S8BR_KVST_NAMES.iter().any(|h| h.0 == n) {
            lines.push(format!(
                "  if (KVST > 0) {{ @export_symbol(s2d_d_{n}_ptr, \"{n}\"); }}"
            ));
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
        lines.push(
            "  if (KVST > 0) { @export_symbol(page_stage); @export_symbol(page_unstage); }"
                .to_string(),
        );
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
fn s8br_unit(kernel: &str, src: &str, pipe: bool, variant: Variant) -> R<String> {
    let tag = format!("{kernel}/pe.csl");
    let mut s = patch(
        &variant.apply(&softmax_s123(src, &tag)?, &tag)?,
        "param c_r1_0: color;\n",
        &format!(
            "// serve8b: 1 times each pass (prof[0]); the route table's route capacity.\n\
param S2D_TIME: u16;\nparam S2D_RTR: u16;\nparam S2D_NOLAYER: u16;\n{}param c_r1_0: color;\n",
            if pipe {
                "param S2D_PIPE: u16 = 0;\nparam S2D_MMAX: u16 = 1;   // serve8b_region_pipe\n"
            } else {
                ""
            }
        ),
        "params",
    )?;
    s = patch(
        &s,
        "  px = layout_mod.get_x_coord();\n  py = layout_mod.get_y_coord();\n",
        "  s2d_place();\n",
        "init coordinates",
    )?;
    s = patch(
        &s,
        "  axis_y();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "  axis_y();\n  s2d_routes();\n  sys_mod.unblock_cmd_stream();\n}\n",
        "init routes",
    )?;
    let head = "fn compute(steps: u32) void {\n";
    let (start, end) =
        fn_block(&s, head).ok_or_else(|| format!("gen: {tag} changed: compute() not found"))?;
    let body_at = start + s[start..].find(head).unwrap_or(0) + head.len();
    let body = &s[body_at..end - 3];
    let open = "  while (i < steps) : (i += 1) {\n";
    let mid = "\n    timing = i > 0;\n    pass_start();\n    timestamp.get_timestamp(&t_mark);\n";
    let close = "\n    pass_end();\n  }\n";
    let bad = || format!("gen: {tag} changed: compute()'s pass loop not as expected");
    let lo = body.find(open).ok_or_else(bad)? + open.len();
    let m = lo + body[lo..].find(mid).ok_or_else(bad)?;
    let hi = m + mid.len() + body[m + mid.len()..].find(close).ok_or_else(bad)?;
    let dedent = |t: &str| {
        t.split('\n')
            .map(|l| l.strip_prefix("  ").unwrap_or(l))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let pass_fn = format!(
        "// The kernel's per-pass prologue (lanes advance) and its layer(s), lifted by sync.py.\n\
fn s2d_pass_start(i: u32, start_pos: u32) void {{\n{}\n}}\nfn s2d_layers() void {{\n{}\n}}\n",
        dedent(&body[lo..m]),
        dedent(&body[m + mid.len()..hi])
    );
    s = format!("{}\n{}", &s[..start], &s[end..]);
    let grid_extra = if pipe {
        src::s8brp::GRID_EXTRA
    } else {
        src::s8br::GRID_EXTRA
    };
    let mut own = exports(&s);
    own.extend(exports(grid_extra));
    let nlk = if kernel == "qwen2d_region" { "NL" } else { "1" };
    let extra = format!(
        "{S8B_GRID_VARIANT}const S2D_KL: u16 = {nlk};       // layers a unit (KV staging)\n\n{pass_fn}\n{grid_extra}\n{}",
        s8br_dummies(&own, pipe)
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
    let at =
        at.ok_or_else(|| format!("gen: {tag} changed: the queues' comptime block not found"))?;
    s = format!("{}\n{extra}{}", &s[..at], &s[at..]);
    Ok(format!(
        "// GENERATED by serve8b/sync.py from {tag} + serve8b/grid_extra.csl.\n\
// Do not edit: change grid_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s8br_head(pipe: bool) -> R<String> {
    let tag = "head2d_v10/pe.csl";
    let mut s = patch(
        src::HEAD2D_V10,
        "param WF32: u16;\n",
        &format!(
            "param WF32: u16;\n// serve8b: the end grids (GPX x GPY, GHc dims a row; the last at XL, YL), the head's first column XH0,\n\
// row), the tokens' capacity, the lanes, the copy kernel, the grids' element\n\
// types (their exports).\nparam GPX: u16;\nparam GPY: u16;\nparam GHc: u16;\nparam XH0: u16;\n\
param XL: u16;\nparam YL: u16;\nparam CAP: u16;\nparam SB: u16;\nparam HLB: u16;\n\
param KVCP: u16;\nparam HROUTE: u16;\n// 1: the VM is the block east of the strip (VM5.md), not a head column.\n\
param S2D_VMX: u16;\n{}param WS: type;\nparam KS: type;\n",
            if pipe {
                "param S2D_PIPE: u16 = 0;\nparam S2D_MMAX: u16 = 1;\nparam S2D_FINAL: u16 = 1;\n"
            } else {
                ""
            }
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
    // An external VM block (S2D_VMX): the VM's branches run with no
    // VM column in the head (VMC 0): the root column sends its logits row
    // east on c_vl, the final root takes the VM's token from the east on c_vt.
    let n = s.matches("VMC == 1 and gmode[0] == 1").count();
    if n != 2 {
        return Err(format!(
            "gen: {tag} changed: the VM's gmode branches found {n} times"
        ));
    }
    s = s.replace(
        "VMC == 1 and gmode[0] == 1",
        "(VMC == 1 or S2D_VMX == 1) and gmode[0] == 1",
    );
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
    s = patch(
        &s,
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  mark(0);\n",
        "  @fmovs(xs_dsd, fin(q_x_in, Hc));\n  s2d_after_x();\n  mark(0);\n",
        "region x arrival",
    )?;
    s = patch(
        &s,
        "  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "  s2d_feeder_in();\n  @fmuls(xs_dsd, xin_dsd, gam_dsd);\n",
        "feeder input",
    )?;
    s = patch(
        &s,
        "      ib[0] = if (ok) best[1] else -1.0;\n",
        "      ib[0] = s2d_decide(if (ok) best[1] else -1.0);\n",
        "final root decision",
    )?;
    s = patch(
        &s,
        "      tok[1] = ib[0];\n",
        "      ib[0] = s2d_decide(ib[0]);\n      tok[1] = ib[0];\n",
        "final root decision (gmode)",
    )?;
    let (t, n) = after_each_call(&s, "emb_copy();", "s2d_emb_out();");
    if n != 2 {
        return Err(format!("gen: {tag} changed: emb_copy() called {n} times"));
    }
    s = drop_head_compute(&t)?;
    let mut own = exports(&s);
    own.extend(["in_tokens", "out_tokens", "yscale", "hlogits", "vmctl"].map(String::from));
    let head_extra = if pipe {
        own.push("perr".into());
        src::s8brp::HEAD_EXTRA
    } else {
        src::s8br::HEAD_EXTRA
    };
    s = format!(
        "{}\n\n{}\n{}",
        s.trim_end_matches('\n'),
        head_extra,
        s8br_dummies(&own, pipe)
    );
    Ok(format!(
        "// GENERATED by serve8b/sync.py from {tag} (Fork 7) + serve8b/head_extra.csl.\n\
// Do not edit: change head_extra.csl or sync.py and rerun.\n{s}"
    ))
}

fn s8br_plain(base: &str, base_name: &str, what: &str, pipe: bool) -> String {
    format!(
        "// GENERATED by serve8b/sync.py from {base_name} ({what}).\n{base}\n{}",
        s8br_dummies(&exports(base), pipe)
    )
}

/// vm_pe.csl: the eta VM (vm_base.csl) for the block east of the strip; its
/// compute takes the grids' (steps, start_pos) and runs a row each of the
/// head's steps + 1 iterations x S2D_SB lanes.
fn s8br_vm(pipe: bool) -> R<String> {
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
        s8br_dummies(&own, pipe)
    ))
}

/// serve8b_region's build directory, or serve8b_region_pipe's (`pipe`).
pub fn serve8b_region(pipe: bool, variant: Variant) -> R<Vec<File>> {
    let (layout, embed) = if pipe {
        (src::s8brp::LAYOUT, src::s8brp::EMBED_BASE)
    } else {
        (src::s8br::LAYOUT, src::s8br::EMBED_BASE)
    };
    Ok(vec![
        ("layout.csl", layout.to_string()),
        ("paths.csl", src::s8br::PATHS.to_string()),
        ("rt.csl", src::RT.to_string()),
        (
            "grid_pe.csl",
            s8br_unit("qwen2d_paged8b", src::QWEN2D_PAGED8B_HR, pipe, variant)?,
        ),
        (
            "region_pe.csl",
            s8br_unit("qwen2d_region", src::QWEN2D_REGION, pipe, variant)?,
        ),
        ("head_pe.csl", s8br_head(pipe)?),
        (
            "embed_pe.csl",
            s8br_plain(embed, "embed_base.csl", "the embedding strip", pipe),
        ),
        (
            "filler.csl",
            s8br_plain(
                src::s8br::FILLER_BASE,
                "filler_base.csl",
                "the filler",
                pipe,
            ),
        ),
        (
            "vm_stub.csl",
            s8br_plain(
                src::s8br::VM_STUB_BASE,
                "vm_stub_base.csl",
                "a stand-in for the VM block",
                pipe,
            ),
        ),
        ("vm_pe.csl", s8br_vm(pipe)?),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Q35_OUT=<dir>`: serve_q35's build directory written there (to
    /// compile it with cslc by hand).
    #[test]
    #[ignore]
    fn dump_serve_q35() {
        let dir = std::path::PathBuf::from(std::env::var("Q35_OUT").expect("Q35_OUT"));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in serve_q35().expect("serve_q35 generates") {
            std::fs::write(dir.join(name), text).unwrap();
        }
    }

    /// `G4_OUT=<dir>`: serve_g4's build directory written there.
    #[test]
    #[ignore]
    fn dump_serve_g4() {
        let dir = std::path::PathBuf::from(std::env::var("G4_OUT").expect("G4_OUT"));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in serve_g4().expect("serve_g4 generates") {
            std::fs::write(dir.join(name), text).unwrap();
        }
    }

    /// Every serve_g4 program exports each data name of its layout.
    #[test]
    fn serve_g4_exports_every_name() {
        let files = serve_g4().unwrap();
        let layout = &files.iter().find(|f| f.0 == "layout.csl").unwrap().1;
        let want: Vec<String> = layout_exports(layout)
            .into_iter()
            .filter(|n| !layout.contains(&format!("\"{n}\", fn(")))
            .collect();
        for (name, text) in &files {
            if !name.ends_with("_pe.csl") && *name != "filler.csl" {
                continue;
            }
            let own = exports(text);
            let missing: Vec<&String> = want.iter().filter(|n| !own.contains(n)).collect();
            assert!(missing.is_empty(), "{name}: {missing:?}");
            let extra: Vec<&String> = own.iter().filter(|n| !want.contains(n)).collect();
            assert!(
                extra.is_empty(),
                "{name} exports {extra:?} the layout lacks"
            );
        }
    }

    /// The generated files' blake3. These are the programs validated on CS-3
    /// hardware: a change to a kernel, an extra or a generator moves a hash,
    /// and a moved hash is a different wafer program, to be validated again
    /// before it is repinned (the failure prints the new hash).
    const PINNED: &[(&str, &str, &str)] = &[
        (
            "serve2d",
            "grid_pe.csl",
            "b591b05358b5bdde516e3f65e5eb361402c4f0eef41f74521b52b86e89d604ec",
        ),
        (
            "serve2d",
            "grid_paged_pe.csl",
            "d813715d243315a8cbddbd65a223c87d5911ac99bb9447de51e64383e6970ed0",
        ),
        (
            "serve2d",
            "head_pe.csl",
            "bdc1e087f3066c21efcf2ae9d75acb489e80d3a9d12630220f623faa86095ffc",
        ),
        (
            "serve2d",
            "head_stub.csl",
            "572f7e7c6f8065a29970ea1785b55023d77d313daf0b2f6051254be096235a1f",
        ),
        (
            "serve2d",
            "filler.csl",
            "9f435cfd600f91915db70d1bd947c202a90a21fe3f72decd974173f977f09211",
        ),
        (
            "serve2d",
            "vm_pe.csl",
            "ac1e9110a0d4e1253d48df48477ddcaf53aad7cf007969b6077eec53f2a88c42",
        ),
        (
            "serve2d_pipe",
            "grid_pe.csl",
            "01cad0a29fb7d9d9a2780160bde68ca5e9510dac712f4be376ddedbdc4caeb8c",
        ),
        (
            "serve2d_pipe",
            "head_pe.csl",
            "3e5ec4d008e92b055b42c1d315fe7aa328ebf4cd6f8a3acc801062644dc0449e",
        ),
        (
            "serve2d_pipe",
            "filler.csl",
            "9d5a34b90889efbb80c0f5bff82fc6e0723f50143729cf57d4edf05156e38f71",
        ),
        (
            "serve2d_pipe",
            "vm_pe.csl",
            "e4a4b11b46a9638b7d4727b028b6b61dc2dd3a1f7c013d1ce44df45a510ff95d",
        ),
        (
            "serve8b",
            "grid_pe.csl",
            "24449b97365e202aacec8b35408490458f94990652201508e796d7410b5bbcd0",
        ),
        (
            "serve8b",
            "head_pe.csl",
            "c9d5b14ad454fd62eae6c592fd2c2e21b484c59b29c160a1c363e6307bfd054d",
        ),
        (
            "serve8b",
            "embed_pe.csl",
            "991c1ce046039415a39f9afb0c6e2e8673a91ff3e7057637abd6477fc99f39e9",
        ),
        (
            "serve8b",
            "filler.csl",
            "bf9e83043319a16c8bac8d11f9d552c22249cfe16dbfb1f8c4a23316e2dbe843",
        ),
        (
            "serve8b_region",
            "grid_pe.csl",
            "8bf4075a5b95e1d79ba68a4cd8f2b6194573ff69298976f5c920443750eaaad3",
        ),
        (
            "serve8b_region",
            "region_pe.csl",
            "c59b6089733cc1241a766f26e896bcc99fbdb2bba345d8b26388a233314211c8",
        ),
        (
            "serve8b_region",
            "head_pe.csl",
            "ec8349abdb11aabd454cd6113a60f9da5614f374519ab14ea162684c4148fcfe",
        ),
        (
            "serve8b_region",
            "embed_pe.csl",
            "8e2769143d5e799f25483d53fb9b8aeae108b1f3e4a95ebecc65a9d967cdf65b",
        ),
        (
            "serve8b_region",
            "filler.csl",
            "0f8106389be8053db738c6bb2189987e0a5716a31372b537f33dcd5aaa54fff5",
        ),
        (
            "serve8b_region",
            "vm_pe.csl",
            "31fa54184438f17f0666961f60c3f46b4e34ec73b95c4f0dbf3b287ab5549f54",
        ),
        (
            "serve8b_region",
            "vm_stub.csl",
            "de0cae945badccacc9b9583cf74c1e47bb14fd82b8287d1b47a0defca4243dbb",
        ),
        (
            "serve8b_region_pipe",
            "grid_pe.csl",
            "f18ef599882f6218fa6883287fa26729bb14b6f56fcb38fa6e2d118a0599ff48",
        ),
        (
            "serve8b_region_pipe",
            "region_pe.csl",
            "dfd2dc07a0a5601ea0906d306950a33856947d33c0fb270588023e4b0becb3b8",
        ),
        (
            "serve8b_region_pipe",
            "head_pe.csl",
            "99548cc08c74fd71dbf90bf6bfbe249b30bafcba4e1011887355b8b71e4ac116",
        ),
        (
            "serve8b_region_pipe",
            "embed_pe.csl",
            "8ce07d934b7a506849e9a949a9fef782def232f0224b7bd25bb607e66665e273",
        ),
        (
            "serve8b_region_pipe",
            "filler.csl",
            "98033013e29a7ca11a09f2242d253c79a69867c6e7d115d55e85eb6dc69181a2",
        ),
        (
            "serve8b_region_pipe",
            "vm_stub.csl",
            "30699f5aa05a06b41bc77f1e0344239ce82506104ee1424d57c6127be7b1c067",
        ),
        (
            "serve8b_region_pipe",
            "vm_pe.csl",
            "d098decfc668200e89eb43c17d3e450b993129f26a65a2f052425a7536abe125",
        ),
    ];

    /// serve_q35's files (not yet a `Design`): pinned as the others, the
    /// programs run on the fabric simulator (payload serve_q35, `--mode sim`).
    const Q35_PINNED: &[(&str, &str)] = &[
        (
            "layout.csl",
            "c9990fb36f57d8c9cc8342d93c8d454db4d6831d6e3cb854a333429851952552",
        ),
        (
            "gdn_pe.csl",
            "86299e87db24dce6dbb3414a89b36a90c176282765ca13262f222e41b4da39f5",
        ),
        (
            "attn_pe.csl",
            "ead473de9a09fa71d79f2b0da7c8ec898da425472668765b7008ace2a0622568",
        ),
        (
            "head_pe.csl",
            "435dc5db1596f4a82050335eebc6f1194436d2f7b639e2584ec0087f3aa94071",
        ),
        (
            "head_stub.csl",
            "5e53795b6e8d4dae4817e75b70ad33c1dec8d930cad152672fb4954dd768bd07",
        ),
        (
            "filler.csl",
            "8ead8263ba862ef2a3fc0adf294a006c8ac434f03b55c8f8dc15b6efc0a97c0d",
        ),
        (
            "vm_pe.csl",
            "b3e6bd93cb3ac384a190b909eaf3c5f72a010761bee407890e49ce7574575642",
        ),
    ];

    #[test]
    fn serve_q35_matches_its_pins() {
        let files = serve_q35().unwrap();
        let mut moved = Vec::new();
        for (file, want) in Q35_PINNED {
            let got = files.iter().find(|(n, _)| n == file).unwrap();
            let have = blake3::hash(got.1.as_bytes()).to_hex().to_string();
            if have != *want {
                moved.push(format!("{file}: {have}"));
            }
        }
        assert!(moved.is_empty(), "moved: {moved:?}");
    }

    #[test]
    fn generators_match_their_pins() {
        for (design, file, want) in PINNED {
            let d = Design::parse(design).unwrap();
            let files = d.files().unwrap();
            let got = files.iter().find(|(n, _)| n == file).unwrap();
            assert_eq!(
                blake3::hash(got.1.as_bytes()).to_hex().as_str(),
                *want,
                "{design}/{file}"
            );
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
                let Some((name, after)) = rest
                    .trim_start()
                    .strip_prefix('"')
                    .and_then(|r| r.split_once('"'))
                else {
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
        for d in [
            Design::Serve2d,
            Design::Serve2dPipe,
            Design::Serve8b,
            Design::Serve8bRegion,
            Design::Serve8bRegionPipe,
        ] {
            let files = d.files().unwrap();
            let layout = &files.iter().find(|f| f.0 == "layout.csl").unwrap().1;
            assert!(!layout_exports(layout).is_empty(), "{}", d.name());
            assert_eq!(
                late(layout),
                Vec::<String>::new(),
                "{}: data exports after a function",
                d.name()
            );
        }
        assert_eq!(
            late("@export_name(\"init\", fn()void);\n@export_name(\"x\", [*]f32, true);"),
            ["x"]
        );
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
