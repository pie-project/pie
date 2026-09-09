use crate::codegen::error::{EmitError, RegionForm};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::{intrinsic_tags, tags};
use eta_ir::wire::predicate_tags;

use crate::plan::{CompiledStage, Dimension, Region, SymbolicType};

use super::fused::{
    METAL_M3_REGION_THREADS, emit_logits_argmax, emit_logits_gather, emit_mtp_drafts,
    emit_pixels_gather, emit_score_gather,
};
use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{grouped_intrinsics_bindable, library_region_valid, used_channel_slots};
use crate::codegen::alias::AliasTable;
use crate::codegen::fault::M3_THREADS_EXCEEDED;
use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::slots::Slots;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StepKind {
    Wide = 0,
    Single = 1,
    Reduce = 2,
    Argmax = 3,
    Partial = 4,
}

#[must_use]
pub const fn streamed_step(value: u32, kind: StepKind) -> u32 {
    (value << 8) | kind as u32
}

#[must_use]
pub const fn step_value(step: u32) -> u32 {
    step >> 8
}

#[must_use]
pub const fn step_kind(step: u32) -> Option<StepKind> {
    match step & 0xFF {
        0 => Some(StepKind::Wide),
        1 => Some(StepKind::Single),
        2 => Some(StepKind::Reduce),
        3 => Some(StepKind::Argmax),
        4 => Some(StepKind::Partial),
        _ => None,
    }
}

#[must_use]
pub fn reduce_levels(last: u32) -> u32 {
    let mut count = last;
    let mut levels = 0u32;
    while count > 1 {
        count = count.div_ceil(32);
        levels += 1;
    }
    levels.max(1)
}

#[must_use]
pub fn reduce_dispatch_levels(last: u32) -> Vec<u32> {
    (0..reduce_levels(last)).step_by(2).collect()
}

#[must_use]
pub fn op_step_kind(tag: u8, pred_tag: u8, argmax_over_f32: bool) -> StepKind {
    match tag {
        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN => StepKind::Reduce,
        tags::REDUCE_ARGMAX => {
            if argmax_over_f32 {
                StepKind::Argmax
            } else {
                StepKind::Single
            }
        }
        tags::CUMSUM
        | tags::CUMPROD
        | tags::SORT_DESC
        | tags::TOP_K
        | tags::MATMUL
        | tags::SCATTER_ADD
        | tags::SCATTER_SET => StepKind::Single,
        tags::PIVOT_THRESHOLD => {
            if pred_tag == predicate_tags::PROB_GE {
                StepKind::Wide
            } else {
                StepKind::Single
            }
        }
        _ => StepKind::Wide,
    }
}

fn wire_dtype(value_types: &[SymbolicType], value: u32) -> Option<u8> {
    value_types
        .get(value as usize)
        .and_then(|ty| eta_ir::types::to_wire(ty.dtype))
}

fn length_key(value_types: &[SymbolicType], value: u32) -> Vec<Dimension> {
    value_types
        .get(value as usize)
        .map(|ty| {
            ty.dims
                .iter()
                .filter(|dim| **dim != Dimension::Static(1))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn msl_type(dtype: u8) -> &'static str {
    match dtype {
        0 => "float",
        1 => "int",
        2 => "uint",
        _ => "bool",
    }
}

fn typed_load(want: u8, ptr: &str, index: &str, own: u8) -> String {
    let load = match want {
        0 => "m1_load_f",
        1 => "m1_load_i",
        2 => "m1_load_u",
        _ => "m1_load_b",
    };
    format!("{load}({ptr}, {index}, {own}u)")
}

fn typed_from_bits(want: u8, bits: &str, own: u8) -> String {
    let as_own = match own {
        0 => format!("as_type<float>({bits})"),
        1 => format!("int({bits})"),
        2 => format!("({bits})"),
        _ => format!("(({bits}) != 0u)"),
    };
    if want == own {
        return as_own;
    }
    match (want, own) {
        (0, 3) => format!("({as_own} ? 1.0f : 0.0f)"),
        (1, 3) => format!("({as_own} ? 1 : 0)"),
        (2, 3) => format!("({as_own} ? 1u : 0u)"),
        (3, _) => format!("({as_own} != 0)"),
        (0, _) => format!("float({as_own})"),
        (1, _) => format!("int({as_own})"),
        _ => format!("uint({as_own})"),
    }
}

fn bits_of(ty: u8, value: &str) -> String {
    match ty {
        0 => format!("as_type<uint>({value})"),
        1 => format!("uint({value})"),
        2 => value.to_string(),
        _ => format!("({value} ? 1u : 0u)"),
    }
}

struct DirectCode {
    pre: String,
    decl: String,
    compute: String,
    store: String,
}

#[allow(clippy::too_many_lines)]
fn direct_wide(
    op: &OpView,
    node: usize,
    base: u32,
    slots: &Slots,
    value_types: &[SymbolicType],
    alias: &AliasTable,
    regs: &BTreeMap<u32, String>,
) -> Option<DirectCode> {
    let arg = |k: usize| op.args.get(k).map(|&a| alias.resolve(a));
    let dt = |value: Option<u32>| value.and_then(|v| wire_dtype(value_types, v));
    let d0 = dt(arg(0));
    let d1 = dt(arg(1));
    let d2 = dt(arg(2));
    let dout = wire_dtype(value_types, base)?;
    let pa = |k: usize| format!("pa{k}_{node}");
    let po0 = format!("po0_{node}");
    let mut pre = String::new();
    for (k, slot) in [&slots.a0, &slots.a1, &slots.a2].into_iter().enumerate() {
        if k == 0 || k < op.args.len() {
            let _ = writeln!(pre, "    const device uchar* {} = {slot};", pa(k));
        }
    }
    let _ = writeln!(pre, "    device uchar* {po0} = {};", slots.o0);
    let n = format!("m4_n_{node}");
    let _ = writeln!(pre, "    const uint {n} = descriptors[{base}].len;");
    let stride = |k: usize, pre: &mut String| -> String {
        let name = format!("s{k}_{node}");
        if let Some(value) = arg(k)
            && !regs.contains_key(&value)
        {
            let _ = writeln!(
                pre,
                "    const uint {name} = descriptors[{value}].len == 1u ? 0u : 1u;"
            );
        }
        name
    };
    let load = |k: usize, want: u8, own: u8, stride_name: &str| -> String {
        match arg(k).and_then(|value| regs.get(&value)) {
            Some(bits) => typed_from_bits(want, bits, own),
            None => typed_load(want, &pa(k), &format!("i * {stride_name}"), own),
        }
    };
    let v = format!("v_{node}");
    let decl = format!("      {} {v} = {}(0);\n", msl_type(dout), msl_type(dout));
    let store_fn = match dout {
        0 => "m1_store_f",
        1 => "m1_store_i",
        2 => "m1_store_u",
        _ => "m1_store_b",
    };
    let store = format!("{store_fn}({po0}, i, {v});");
    let mut compute = String::new();
    match op.tag {
        tags::EXP | tags::LOG | tags::RECIP | tags::SIN | tags::COS | tags::SQRT | tags::RSQRT => {
            let d0 = d0?;
            let s0 = stride(0, &mut pre);
            let x = load(0, 0, d0, &s0);
            let expr = match op.tag {
                tags::EXP => format!("precise::exp({x})"),
                tags::LOG => format!("precise::log({x})"),
                tags::SIN => format!("precise::sin({x})"),
                tags::COS => format!("precise::cos({x})"),
                tags::SQRT => format!("sqrt({x})"),
                tags::RSQRT => format!("1.0f / sqrt({x})"),
                _ => format!("1.0f / {x}"),
            };
            let _ = writeln!(compute, "{v} = {expr};");
        }
        tags::NEG | tags::ABS | tags::SIGN => {
            let d0 = d0?;
            if d0 == 3 {
                return None;
            }
            let s0 = stride(0, &mut pre);
            let x = load(0, d0, d0, &s0);
            let expr = match (d0, op.tag) {
                (0, tags::NEG) => "-x",
                (0, tags::ABS) => "abs(x)",
                (0, _) => "(x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f))",
                (1, tags::NEG) => "int(0u - uint(x))",
                (1, tags::ABS) => "(x == INT_MIN ? x : abs(x))",
                (1, _) => "(x > 0 ? 1 : (x < 0 ? -1 : 0))",
                (_, tags::NEG) => "(0u - x)",
                (_, tags::SIGN) => "(x != 0 ? 1u : 0u)",
                (_, _) => "x",
            };
            let _ = writeln!(
                compute,
                "{{ const {} x = {x}; {v} = {expr}; }}",
                msl_type(d0)
            );
        }
        tags::CAST => {
            let d0 = d0?;
            let s0 = stride(0, &mut pre);
            let _ = writeln!(compute, "{v} = {};", load(0, dout, d0, &s0));
        }
        tags::ADD
        | tags::SUB
        | tags::MUL
        | tags::DIV
        | tags::MAX_ELEM
        | tags::MIN_ELEM
        | tags::REM => {
            let (d0, d1) = (d0?, d1?);
            if d0 == 3 {
                return None;
            }
            let s0 = stride(0, &mut pre);
            let s1 = stride(1, &mut pre);
            let x = load(0, d0, d0, &s0);
            let y = load(1, d0, d1, &s1);
            let expr = match (d0, op.tag) {
                (0, tags::ADD) => "x + y",
                (0, tags::SUB) => "x - y",
                (0, tags::MUL) => "x * y",
                (0, tags::DIV) => "x / y",
                (0, tags::MAX_ELEM) => "m1_element_max(x, y)",
                (0, tags::MIN_ELEM) => "m1_element_min(x, y)",
                (0, _) => "fmod(x, y)",
                (1, tags::ADD) => "int(uint(x) + uint(y))",
                (1, tags::SUB) => "int(uint(x) - uint(y))",
                (1, tags::MUL) => "int(uint(x) * uint(y))",
                (1, tags::DIV) => "(y == 0 ? 0 : x / y)",
                (1, tags::MAX_ELEM) => "max(x, y)",
                (1, tags::MIN_ELEM) => "min(x, y)",
                (1, _) => "(y == 0 ? 0 : x % y)",
                (_, tags::ADD) => "x + y",
                (_, tags::SUB) => "x - y",
                (_, tags::MUL) => "x * y",
                (_, tags::DIV) => "(y == 0 ? 0 : x / y)",
                (_, tags::MAX_ELEM) => "max(x, y)",
                (_, tags::MIN_ELEM) => "min(x, y)",
                (_, _) => "(y == 0 ? 0 : x % y)",
            };
            let t = msl_type(d0);
            let _ = writeln!(
                compute,
                "{{ const {t} x = {x}; const {t} y = {y}; {v} = {expr}; }}"
            );
        }
        tags::GT | tags::GE | tags::EQ | tags::NE | tags::LT | tags::LE => {
            let (d0, d1) = (d0?, d1?);
            let s0 = stride(0, &mut pre);
            let s1 = stride(1, &mut pre);
            let ty = if d0 == 3 { 2 } else { d0 };
            let x = load(0, ty, d0, &s0);
            let y = load(1, ty, d1, &s1);
            let cmp = match op.tag {
                tags::GT => "x > y",
                tags::GE => "x >= y",
                tags::EQ => "x == y",
                tags::NE => "x != y",
                tags::LT => "x < y",
                _ => "x <= y",
            };
            let t = msl_type(ty);
            let _ = writeln!(
                compute,
                "{{ const {t} x = {x}; const {t} y = {y}; {v} = {cmp}; }}"
            );
        }
        tags::AND | tags::OR => {
            let (d0, d1) = (d0?, d1?);
            let s0 = stride(0, &mut pre);
            let s1 = stride(1, &mut pre);
            let x = load(0, 3, d0, &s0);
            let y = load(1, 3, d1, &s1);
            let _ = writeln!(
                compute,
                "{{ const bool x = {x}; const bool y = {y}; {v} = {}; }}",
                if op.tag == tags::AND {
                    "x && y"
                } else {
                    "x || y"
                }
            );
        }
        tags::NOT => {
            let d0 = d0?;
            let s0 = stride(0, &mut pre);
            let _ = writeln!(compute, "{v} = !{};", load(0, 3, d0, &s0));
        }
        tags::SELECT => {
            let (d0, d1, d2) = (d0?, d1?, d2?);
            let s0 = stride(0, &mut pre);
            let s1 = stride(1, &mut pre);
            let s2 = stride(2, &mut pre);
            let c = load(0, 3, d0, &s0);
            let x = load(1, dout, d1, &s1);
            let y = load(2, dout, d2, &s2);
            let _ = writeln!(compute, "{v} = ({c}) ? ({x}) : ({y});");
        }
        tags::CONST => {
            let bits = op.lit_bits;
            let literal = match op.lit_dtype {
                0 => format!("as_type<float>({bits}u)"),
                1 => format!("int({bits}u)"),
                2 => format!("{bits}u"),
                _ => (if bits != 0 { "true" } else { "false" }).to_string(),
            };
            let _ = writeln!(
                compute,
                "{v} = {};",
                typed_from_bits(dout, &bits_of(op.lit_dtype, &literal), op.lit_dtype)
            );
        }
        tags::IOTA => {
            let _ = writeln!(compute, "{v} = {};", typed_from_bits(dout, "i", 2));
        }
        tags::BROADCAST => {
            let d0 = d0?;
            let src = arg(0)?;
            if let Some(bits) = regs.get(&src) {
                let _ = writeln!(compute, "{v} = {};", typed_from_bits(dout, bits, d0));
            } else {
                let mode = format!("bc_{node}");
                let _ = writeln!(
                    pre,
                    "    const uint {mode} = descriptors[{src}].len == 1u ? 0u : 2u;"
                );
                let _ = writeln!(
                    pre,
                    "    const M1ValueDesc bd0_{node} = descriptors[{src}];"
                );
                let _ = writeln!(
                    pre,
                    "    const M1ValueDesc bo0_{node} = descriptors[{base}];"
                );
                let _ = writeln!(compute, "uint x = {mode} == 0u ? 0u : i;");
                let _ = writeln!(
                    compute,
                    "if ({mode} == 2u) x = m4_broadcast_index(bd0_{node}, bo0_{node}, i);"
                );
                let _ = writeln!(compute, "{v} = {};", typed_load(dout, &pa(0), "x", d0));
            }
        }
        tags::GATHER => {
            let (d0, d1) = (d0?, d1?);
            let src = arg(0)?;
            if value_types
                .get(src as usize)
                .is_none_or(|ty| ty.dims.len() > 1)
            {
                return None;
            }
            let s1 = stride(1, &mut pre);
            let _ = writeln!(pre, "    const uint n0_{node} = descriptors[{src}].len;");
            let index = load(1, 1, d1, &s1);
            let _ = writeln!(
                compute,
                "{{ const int idx = {index}; {v} = (idx >= 0 && uint(idx) < n0_{node}) ? {} : {}(0); }}",
                typed_load(dout, &pa(0), "uint(idx)", d0),
                msl_type(dout)
            );
        }
        tags::RESHAPE | tags::CHAN_TAKE | tags::CHAN_READ => {
            if dout == 3 && op.tag != tags::RESHAPE {
                return None;
            }
            let own = if op.tag == tags::RESHAPE { d0? } else { dout };
            match arg(0).and_then(|value| regs.get(&value)) {
                Some(bits) => {
                    let _ = writeln!(compute, "{v} = {};", typed_from_bits(dout, bits, own));
                }
                None => {
                    let _ = writeln!(compute, "{v} = {};", typed_load(dout, &pa(0), "i", own));
                }
            }
        }
        _ => return None,
    }
    Some(DirectCode {
        pre,
        decl,
        compute,
        store,
    })
}

pub const STEP_STRUCT: &str =
    "struct M4Step {\n  uint index;\n  uint level;\n  uint groups;\n  uint reserved;\n};\n";

const BROADCAST_INDEX: &str = r"
inline uint m4_broadcast_index(const M1ValueDesc bd0, const M1ValueDesc bo0, uint i) {
  uint rem = i, source_index = 0;
  uint source_stride[4] = {1, 1, 1, 1};
  for (int dim = int(bo0.rank) - 2; dim >= 0; --dim)
    source_stride[dim] = source_stride[dim + 1] * (uint(dim + 1) < bd0.rank ? bd0.dims[dim + 1] : 1u);
  for (uint dim = 0; dim < bo0.rank; ++dim) {
    uint stride = 1;
    for (uint next = dim + 1; next < bo0.rank; ++next) stride *= bo0.dims[next];
    const uint coordinate = rem / max(stride, 1u);
    rem %= max(stride, 1u);
    const uint source_dim = dim < bd0.rank ? bd0.dims[dim] : 1u;
    if (source_dim != 1) source_index += coordinate * source_stride[dim];
  }
  return source_index;
}
";

const MAX_SPLIT_REDUCTIONS: usize = 64;

const SELECT_BYTES: usize = 16384;

const SELECT_ROUNDS: usize = 3;

enum Plan {
    Direct {
        base: u32,
        len: Vec<Dimension>,
        scalar: bool,
        reads: Vec<u32>,
        cross: Vec<u32>,
    },
    Gather {
        text: String,
        writes: u32,
        single_row: bool,
    },
    Generic {
        text: String,
        reads: Vec<u32>,
        writes: u32,
        len: Vec<Dimension>,
        cross: bool,
    },
    Put {
        text: String,
        value: u32,
        len: Vec<Dimension>,
    },
    Single(String),
    Select {
        node: u32,
        mode: u32,
        a0: String,
        a1: String,
        o0: String,
        input: u32,
    },
    Scatter {
        copy: String,
        rmw: String,
        base_len: Vec<Dimension>,
        result: u32,
    },
    Reduce {
        tag: u8,
        dtype: u8,
        input: u32,
        result: u32,
        len: Vec<Dimension>,
        split: bool,
        two_level: String,
    },
    Argmax {
        text: String,
        input: u32,
    },
}

struct PendingFinal {
    tag: u8,
    dtype: u8,
    input: u32,
    result: u32,
    plane: usize,
    input_ptr: String,
    result_ptr: String,
}

struct Open {
    value: u32,
    kind: StepKind,
    len: Option<Vec<Dimension>>,
    passes: String,
    writes: BTreeSet<u32>,
    prologue_writes: BTreeSet<u32>,
    loop_members: Vec<(u32, u32)>,
    loop_len: Option<Vec<Dimension>>,
}

struct Scheduler<'a> {
    ops: &'a [OpView],
    value_types: &'a [SymbolicType],
    alias: &'a AliasTable,
    slots: BTreeMap<u32, Slots>,
    regs: BTreeMap<u32, String>,
    cases: String,
    steps: Vec<u32>,
    pending_finals: Vec<PendingFinal>,
    pending_scalars: Vec<(u32, u32)>,
    open: Option<Open>,
}

impl Scheduler<'_> {
    fn direct(&self, node: u32, base: u32) -> DirectCode {
        direct_wide(
            &self.ops[node as usize],
            node as usize,
            base,
            &self.slots[&node],
            self.value_types,
            self.alias,
            &self.regs,
        )
        .expect("planned as direct")
    }

    fn close_loop(&mut self) {
        let Some(o) = self.open.as_mut() else { return };
        if o.loop_members.is_empty() {
            return;
        }
        let members = core::mem::take(&mut o.loop_members);
        o.loop_len = None;
        let first = members[0].0;
        let mut pre = String::new();
        let mut body = String::new();
        for &(node, base) in &members {
            let code = self.direct(node, base);
            pre.push_str(&code.pre);
            body.push_str(&code.decl);
            let _ = writeln!(body, "      if (i < m4_n_{node}) {{");
            for line in code.compute.lines() {
                let _ = writeln!(body, "        {line}");
            }
            let _ = writeln!(body, "        {}", code.store);
            body.push_str("      }\n");
        }
        let o = self.open.as_mut().expect("open");
        o.passes.push_str(&pre);
        let _ = writeln!(
            o.passes,
            "    for (uint i = m4_gtid; i < m4_n_{first}; i += m4_gthreads) {{"
        );
        o.passes.push_str(&body);
        o.passes.push_str("    }\n");
    }

    fn flush_open(&mut self) {
        self.close_loop();
        if let Some(o) = self.open.take() {
            let _ = writeln!(self.cases, "  case {}u: {{", self.steps.len());
            self.cases.push_str(&o.passes);
            self.cases.push_str("    return;\n  }\n");
            self.steps.push(streamed_step(o.value, o.kind));
        }
    }

    fn prologue(&mut self) -> (String, BTreeSet<u32>) {
        self.regs.clear();
        let mut text = String::new();
        let mut writes = BTreeSet::new();
        for f in core::mem::take(&mut self.pending_finals) {
            let r = format!("r_{}", f.result);
            let _ = writeln!(
                text,
                "    const uint {r} = m4_reduce_final({}u, {}u, {}, descriptors[{}].len, reinterpret_cast<const device uint*>(m4_planes + {}u * m4_plane_bytes), status, m3_threads, m4_simd_lane, m4_simd_id, m3_tgbuf);",
                f.tag, f.dtype, f.input_ptr, f.input, f.plane
            );
            let _ = writeln!(
                text,
                "    if (m4_gtid == 0u) reinterpret_cast<device uint*>({})[0] = {r};",
                f.result_ptr
            );
            self.regs.insert(f.result, r);
            writes.insert(f.result);
        }
        for (node, base) in core::mem::take(&mut self.pending_scalars) {
            text.push_str(&self.scalar_inline(node, base));
            writes.insert(base);
        }
        (text, writes)
    }

    fn scalar_inline(&mut self, node: u32, base: u32) -> String {
        let code = self.direct(node, base);
        let dout = wire_dtype(self.value_types, base).unwrap_or(0);
        let mut text = String::new();
        text.push_str(&code.pre);
        text.push_str(&code.decl.replace("      ", "    "));
        text.push_str("    { const uint i = 0u;\n");
        for line in code.compute.lines() {
            let _ = writeln!(text, "      {line}");
        }
        let _ = writeln!(text, "      if (m4_gtid == 0u) {} }}", code.store);
        let r = format!("r_{base}");
        let _ = writeln!(
            text,
            "    const uint {r} = {};",
            bits_of(dout, &format!("v_{node}"))
        );
        self.regs.insert(base, r);
        text
    }

    fn begin(&mut self, value: u32, kind: StepKind, len: Option<Vec<Dimension>>) {
        let (passes, prologue_writes) = self.prologue();
        self.open = Some(Open {
            value,
            kind,
            len,
            passes,
            writes: BTreeSet::new(),
            prologue_writes,
            loop_members: Vec::new(),
            loop_len: None,
        });
    }

    fn flush_pending(&mut self) {
        self.flush_open();
        if self.pending_finals.is_empty() && self.pending_scalars.is_empty() {
            return;
        }
        let value = self
            .pending_finals
            .first()
            .map(|f| f.result)
            .or_else(|| self.pending_scalars.first().map(|s| s.1))
            .unwrap_or(0);
        let (text, _) = self.prologue();
        let _ = writeln!(self.cases, "  case {}u: {{", self.steps.len());
        self.cases.push_str("    if (m4_group.x != 0) return;\n");
        self.cases.push_str(&text);
        self.cases.push_str("    return;\n  }\n");
        self.steps.push(streamed_step(value, StepKind::Single));
    }

    fn reads_pending(&self, reads: &[u32]) -> bool {
        reads.iter().any(|v| {
            self.pending_finals.iter().any(|f| f.result == *v)
                || self.pending_scalars.iter().any(|s| s.1 == *v)
        })
    }

    fn ensure_open(
        &mut self,
        value: u32,
        kind: StepKind,
        len: &[Dimension],
        fits: impl Fn(&Open) -> bool,
    ) {
        if self.open.as_ref().is_some_and(|o| !fits(o)) {
            self.flush_open();
        }
        if self.open.is_none() {
            self.begin(value, kind, Some(len.to_vec()));
        }
        let o = self.open.as_mut().expect("opened");
        if o.len.is_none() {
            o.len = Some(len.to_vec());
        }
    }
}

pub(super) fn kernel_head(function_name: &str, channel_count: usize, extra: &str) -> String {
    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    source.push_str(grouped_preamble());
    source.push_str(STEP_STRUCT);
    source.push_str(BROADCAST_INDEX);
    let _ = writeln!(source, "kernel void {function_name}(");
    source.push_str("    const device uchar* lane_bytes [[buffer(0)]],\n");
    source.push_str("    const device M1ValueDesc* all_descriptors [[buffer(1)]],\n");
    source.push_str("    const device M1OpParams* params [[buffer(2)]],\n");
    source.push_str("    const device uint* offsets [[buffer(3)]],\n");
    source.push_str("    device uchar* all_scratch [[buffer(4)]],\n");
    source.push_str("    const device M3GroupLayout* layout [[buffer(5)]],\n");
    source.push_str("    const device uint* channel_bindings [[buffer(6)]],\n");
    source.push_str("    device uchar* pending_flags [[buffer(7)]],\n");
    source.push_str("    const device uint* lane_indices [[buffer(8)]],\n");
    source.push_str("    const device M3RowMeta* all_row_meta [[buffer(9)]],\n");
    source.push_str("    const device uint* row_indices [[buffer(10)]],\n");
    source.push_str("    constant M4Step& step [[buffer(11)]],\n");
    source.push_str("    uint2 m4_group [[threadgroup_position_in_grid]],\n");
    source.push_str("    uint2 m4_groups [[threadgroups_per_grid]],\n");
    source.push_str("    uint2 m4_tid [[thread_position_in_threadgroup]],\n");
    source.push_str("    uint2 m4_threads [[threads_per_threadgroup]],\n");
    source.push_str("    uint m4_simd_lane [[thread_index_in_simdgroup]],\n");
    source.push_str("    uint m4_simd_id [[simdgroup_index_in_threadgroup]]) {\n");
    source.push_str("  const uint m3_tid = m4_tid.x;\n");
    source.push_str("  const uint m3_threads = m4_threads.x;\n");
    let _ = writeln!(
        source,
        "  threadgroup M1ArgmaxCandidate m3_tgbuf[{METAL_M3_REGION_THREADS}];"
    );
    source.push_str(extra);
    source.push_str("  const uint dispatch_lane = m4_group.y;\n");
    source.push_str("  if (dispatch_lane >= layout->lane_count) return;\n");
    source.push_str("  const uint lane_index = lane_indices[dispatch_lane];\n");
    source.push_str(
        "  const device M3LaneHeader* header = \
         reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n",
    );
    source.push_str(
        "  const device M3LaneRecord* lanes = \
         reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n",
    );
    source.push_str(
        "  const device M3LaneChannelSlot* slots = \
         reinterpret_cast<const device M3LaneChannelSlot*>(lane_bytes + \
         sizeof(M3LaneHeader) + header->lane_count * sizeof(M3LaneRecord));\n",
    );
    source.push_str("  const M3LaneRecord lane = lanes[lane_index];\n");
    source.push_str("  const M3RowMeta row_meta = all_row_meta[lane_index];\n");
    source.push_str(
        "  device M1Status* status = \
         reinterpret_cast<device M1Status*>(lane.commit_slot);\n",
    );
    source.push_str("  if (status->state != 1) return;\n");
    let _ = writeln!(
        source,
        "  if (m3_threads > {METAL_M3_REGION_THREADS}u) {{ \
         if (m3_tid == 0 && m4_group.x == 0) m1_fault(status, {M3_THREADS_EXCEEDED:#X}u); return; }}"
    );
    source.push_str(
        "  const device M1ValueDesc* descriptors = all_descriptors + \
         dispatch_lane * layout->value_count;\n",
    );
    source.push_str(
        "  const device M1OpParams* lane_params = params + \
         dispatch_lane * layout->reserved2;\n",
    );
    source.push_str(
        "  device uchar* scratch = all_scratch + dispatch_lane * layout->scratch_stride;\n",
    );
    source.push_str("  device uchar* temporary = scratch + layout->temporary_offset;\n");
    source.push_str(
        "  const device bfloat* logits = \
         reinterpret_cast<const device bfloat*>(lane.logits_base);\n",
    );
    for channel in 0..channel_count {
        let _ = writeln!(
            source,
            "  const uint dense_{channel} = channel_bindings[dispatch_lane * layout->reserved0 + {channel}];"
        );
        let _ = writeln!(
            source,
            "  const M3LaneChannelSlot channel_{channel} = slots[lane.channel_slot_offset + dense_{channel}];"
        );
        let _ = writeln!(
            source,
            "  const uint pending_index_{channel} = lane.channel_slot_offset + dense_{channel};"
        );
        let _ = writeln!(
            source,
            "  const device uchar* current_{channel} = reinterpret_cast<const device uchar*>(\
             pending_flags[pending_index_{channel}] != 0 ? channel_{channel}.pending_cell : \
             channel_{channel}.committed_cell);"
        );
        let _ = writeln!(
            source,
            "  device uchar* pending_{channel} = reinterpret_cast<device uchar*>(\
             channel_{channel}.pending_cell);"
        );
    }
    source.push_str("  const uint m4_gtid = m4_group.x * m3_threads + m3_tid;\n");
    source.push_str("  const uint m4_gthreads = m4_groups.x * m3_threads;\n");

    source
}

#[allow(clippy::too_many_lines)]
pub fn emit_streamed_region(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<(String, Vec<u32>), EmitError> {
    if !library_region_valid(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedFused));
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    grouped_intrinsics_bindable(&ops, region)?;
    let bases = result_bases(&ops);
    let channel_count = used_channel_slots(&ops);
    let value_types = &stage.normalized.value_types;

    let has_select = ops.iter().zip(0u32..).any(|(op, node)| {
        region.nodes.iter().any(|n| n.index() as u32 == node)
            && op.tag == tags::PIVOT_THRESHOLD
            && matches!(
                op.pred_tag,
                predicate_tags::RANK_LE | predicate_tags::CUMMASS_LE
            )
    });
    let extra = if has_select {
        "  threadgroup atomic_uint m4_sel_tg_hist[256];\n  threadgroup uint m4_sel_key[1024];\n  threadgroup uint m4_sel_idx[1024];\n  threadgroup uint m4_sel_scan[1024];\n"
    } else {
        ""
    };
    let mut source = kernel_head(function_name, channel_count, extra);

    let escapes = crate::codegen::alias::escaping_values(region);
    let covers =
        |source: u32, result: u32| crate::codegen::alias::covers(value_types, source, result);
    let is_view_reshape = |node: usize| -> bool {
        let Some(op) = ops.get(node) else {
            return false;
        };
        op.tag == tags::RESHAPE
            && op.results == 1
            && op.args.len() == 1
            && !escapes.contains(&bases[node])
            && covers(op.args[0], bases[node])
    };
    let mut alias = AliasTable::new();
    for &node in &region.nodes {
        let node = node.index();
        if is_view_reshape(node) {
            let arg = ops[node].args[0];
            alias.elide(bases[node], arg);
        }
    }
    let mut consumers: BTreeMap<u32, usize> = BTreeMap::new();
    for &node in &region.nodes {
        if is_view_reshape(node.index()) {
            continue;
        }
        if let Some(op) = ops.get(node.index()) {
            for &arg in &op.args {
                *consumers.entry(alias.resolve(arg)).or_insert(0) += 1;
            }
        }
    }
    let mut fused_argmax: BTreeMap<usize, usize> = BTreeMap::new();
    let mut elided_gather: BTreeSet<usize> = BTreeSet::new();
    for &node in &region.nodes {
        let node = node.index();
        let Some(op) = ops.get(node) else { continue };
        if op.tag != tags::REDUCE_ARGMAX || op.args.len() != 1 {
            continue;
        }
        let source_value = alias.resolve(op.args[0]);
        if consumers.get(&source_value).copied().unwrap_or(0) != 1
            || escapes.contains(&source_value)
        {
            continue;
        }
        let producer = region.nodes.iter().map(|n| n.index()).find(|&n| {
            bases[n] == source_value
                && ops.get(n).is_some_and(|p| {
                    p.tag == tags::INTRINSIC_VAL
                        && (p.intr == intrinsic_tags::LOGITS
                            || p.intr == intrinsic_tags::VELOCITY
                            || p.intr == intrinsic_tags::HIDDEN
                            || p.intr == intrinsic_tags::MTP_LOGITS)
                })
        });
        if let Some(producer) = producer {
            fused_argmax.insert(node, producer);
            elided_gather.insert(producer);
        }
    }
    let value_ptr = |value: u32| format!("scratch + offsets[{value}]");
    let argmax_over_f32 = |op: &OpView| -> bool {
        op.args
            .first()
            .and_then(|&arg| value_types.get(alias.resolve(arg) as usize))
            .is_some_and(|ty| ty.dtype == eta_ir::Dtype::F32)
    };
    let key_of = |value: u32| length_key(value_types, value);

    let mut planned: Vec<(u32, Plan)> = Vec::new();
    let mut planned_slots: BTreeMap<u32, Slots> = BTreeMap::new();
    for &node in &region.nodes {
        let node = node.index();
        let Some(op) = ops.get(node) else {
            return Err(EmitError::RegionNodeOutOfRange(RegionForm::GroupedFused));
        };
        let base = bases[node];
        if elided_gather.contains(&node) || is_view_reshape(node) {
            continue;
        }
        let node_u32 = u32::try_from(node)
            .map_err(|_| EmitError::RegionNodeOutOfRange(RegionForm::GroupedFused))?;
        if let Some(&producer) = fused_argmax.get(&node) {
            let slots = Slots::of(op, base, |value| value_ptr(alias.resolve(value)));
            let mut body = String::new();
            emit_logits_argmax(
                &mut body,
                bases[producer],
                ops[producer].intr == intrinsic_tags::MTP_LOGITS,
                &slots.o0,
            );
            planned.push((node_u32, Plan::Single(body)));
            continue;
        }
        let mut slots = Slots::of(op, base, |value| value_ptr(alias.resolve(value)));
        let mut reads: Vec<u32> = op.args.iter().map(|&a| alias.resolve(a)).collect();
        if op.tag == tags::PIVOT_THRESHOLD {
            reads.push(alias.resolve(op.pred_payload));
        }
        if op.tag == tags::INTRINSIC_VAL {
            let single_row = key_of(base).len() <= 1;
            let mut body = String::new();
            match op.intr {
                intrinsic_tags::MTP_DRAFTS => {
                    emit_mtp_drafts(&mut body, base, &slots.o0, "m4_gtid", "m4_gthreads");
                }
                intrinsic_tags::ATTN_SCORE => {
                    emit_score_gather(&mut body, base, &slots.o0, "m4_gtid", "m4_gthreads");
                }
                intrinsic_tags::PIXELS => {
                    emit_pixels_gather(&mut body, base, &slots.o0, "m4_gtid", "m4_gthreads");
                }
                intrinsic_tags::LOGITS
                | intrinsic_tags::VELOCITY
                | intrinsic_tags::HIDDEN
                | intrinsic_tags::MTP_LOGITS => {
                    emit_logits_gather(
                        &mut body,
                        base,
                        op.intr == intrinsic_tags::MTP_LOGITS,
                        &slots.o0,
                        "m4_gtid",
                        "m4_gthreads",
                    );
                }
                _ => {
                    slots.a0 = "reinterpret_cast<const device uchar*>(logits)".to_string();
                    let text = format!(
                        "    ptir_m1_execute_part({}u, status, descriptors, lane_params + {node}, {}, {}, {}, {}, {}, temporary, m4_gtid, m4_gthreads);\n",
                        op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
                    );
                    planned.push((
                        node_u32,
                        Plan::Generic {
                            text,
                            reads: Vec::new(),
                            writes: base,
                            len: key_of(base),
                            cross: true,
                        },
                    ));
                    continue;
                }
            }
            planned.push((
                node_u32,
                Plan::Gather {
                    text: body,
                    writes: base,
                    single_row,
                },
            ));
            continue;
        }
        if op.tag == tags::CHAN_TAKE || op.tag == tags::CHAN_READ {
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == tags::CHAN_PUT {
            slots.o0 = format!("pending_{}", op.chan);
        }
        let kind = op_step_kind(op.tag, op.pred_tag, argmax_over_f32(op));
        let generic_wide = format!(
            "    ptir_m1_execute_part({}u, status, descriptors, lane_params + {node}, {}, {}, {}, {}, {}, temporary, m4_gtid, m4_gthreads);\n",
            op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
        );
        let plan = match kind {
            StepKind::Single
                if op.tag == tags::PIVOT_THRESHOLD
                    && matches!(
                        op.pred_tag,
                        predicate_tags::RANK_LE | predicate_tags::CUMMASS_LE
                    ) =>
            {
                Plan::Select {
                    node: node_u32,
                    mode: u32::from(op.pred_tag == predicate_tags::CUMMASS_LE),
                    a0: slots.a0.clone(),
                    a1: slots.a1.clone(),
                    o0: slots.o0.clone(),
                    input: reads.first().copied().unwrap_or(base),
                }
            }
            StepKind::Single if matches!(op.tag, tags::SCATTER_ADD | tags::SCATTER_SET) => {
                Plan::Scatter {
                    copy: format!(
                        "    m1_copy_typed_range({}, {}, descriptors[lane_params[{node}].a0].len, descriptors[lane_params[{node}].a0].dtype, m4_gtid, m4_gthreads);\n",
                        slots.a0, slots.o0
                    ),
                    rmw: format!(
                        "    if (m4_group.x != 0) return;\n    if (m3_tid == 0) m1_scatter_rmw({}u, {}, {}, {}, descriptors[lane_params[{node}].a0], descriptors[lane_params[{node}].a1], descriptors[lane_params[{node}].a2]);\n",
                        op.tag, slots.a1, slots.a2, slots.o0
                    ),
                    base_len: reads.first().map_or_else(Vec::new, |&v| key_of(v)),
                    result: base,
                }
            }
            StepKind::Wide if op.tag == tags::CHAN_PUT => Plan::Put {
                text: format!(
                    "{generic_wide}    if (m4_gtid == 0) pending_flags[pending_index_{}] = 1;\n",
                    op.chan
                ),
                value: reads.first().copied().unwrap_or(base),
                len: reads.first().map_or_else(Vec::new, |&v| key_of(v)),
            },
            StepKind::Wide => {
                let probe = BTreeMap::new();
                if direct_wide(op, node, base, &slots, value_types, &alias, &probe).is_some() {
                    let len = key_of(base);
                    planned_slots.insert(node_u32, slots);
                    Plan::Direct {
                        base,
                        scalar: len.is_empty(),
                        len,
                        cross: if matches!(op.tag, tags::BROADCAST | tags::GATHER) {
                            reads.first().copied().into_iter().collect()
                        } else {
                            Vec::new()
                        },
                        reads,
                    }
                } else {
                    let same_index = (tags::EXP..=tags::SELECT).contains(&op.tag)
                        || matches!(
                            op.tag,
                            tags::RNG
                                | tags::RNG_KEYED
                                | tags::RESHAPE
                                | tags::CHAN_TAKE
                                | tags::CHAN_READ
                                | tags::BROADCAST
                        );
                    Plan::Generic {
                        text: generic_wide,
                        reads,
                        writes: base,
                        len: key_of(base),
                        cross: !same_index || op.tag == tags::BROADCAST,
                    }
                }
            }
            StepKind::Single => {
                let mut body = String::from("    if (m4_group.x != 0) return;\n");
                let _ = writeln!(
                    body,
                    "    ptir_m1_execute_mt({}u, status, descriptors, lane_params + {node}, {}, {}, {}, {}, {}, temporary, m3_tid, m3_threads, m3_tgbuf);",
                    op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
                );
                if op.tag == tags::CHAN_PUT {
                    let _ = writeln!(
                        body,
                        "    if (m3_tid == 0) pending_flags[pending_index_{}] = 1;",
                        op.chan
                    );
                }
                Plan::Single(body)
            }
            StepKind::Reduce => {
                let input = reads.first().copied().unwrap_or(base);
                let dtype = wire_dtype(value_types, input).unwrap_or(3);
                let len = key_of(op.args.first().copied().unwrap_or(input));
                let two_level = format!(
                    "    m4_reduce_two_levels({}u, {}, {}, temporary, descriptors[lane_params[{node}].a0], step.level, m4_group.x, m3_tid, m3_threads, m3_tgbuf, m4_simd_lane, m4_simd_id);\n",
                    op.tag, slots.a0, slots.o0
                );
                planned_slots.insert(node_u32, slots);
                Plan::Reduce {
                    tag: op.tag,
                    dtype,
                    input,
                    result: base,
                    split: len.len() <= 1 && dtype != 3,
                    len,
                    two_level,
                }
            }
            StepKind::Argmax => Plan::Argmax {
                text: format!(
                    "    if (step.level == 0u) m4_argmax_partial({}, temporary, descriptors[lane_params[{node}].a0], m4_group.x, m4_groups.x, m3_tid, m3_threads, m3_tgbuf);\n    else {{ if (m4_group.x != 0) return; m4_argmax_final(temporary, {}, descriptors[lane_params[{node}].a0], step.groups, m3_tid, m3_threads, m3_tgbuf); }}\n",
                    slots.a0, slots.o0
                ),
                input: reads.first().copied().unwrap_or(base),
            },
            StepKind::Partial => unreachable!("op_step_kind never answers Partial"),
        };
        planned.push((node_u32, plan));
    }

    let select_count = planned
        .iter()
        .filter(|(_, plan)| matches!(plan, Plan::Select { .. }))
        .count();
    let select_bytes = select_count * SELECT_BYTES;
    let split_count = planned
        .iter()
        .filter(|(_, plan)| matches!(plan, Plan::Reduce { split: true, .. }))
        .count()
        .min(MAX_SPLIT_REDUCTIONS);
    if split_count > 0 {
        let _ = writeln!(
            source,
            "  device uchar* m4_planes = temporary + {select_bytes}u;\n  const uint m4_plane_bytes = ((layout->scratch_stride - layout->temporary_offset - {select_bytes}u) / {split_count}u) & ~15u;"
        );
    }

    let mut sched = Scheduler {
        ops: &ops,
        value_types,
        alias: &alias,
        slots: planned_slots,
        regs: BTreeMap::new(),
        cases: String::new(),
        steps: Vec::new(),
        pending_finals: Vec::new(),
        pending_scalars: Vec::new(),
        open: None,
    };
    let mut planes_used = 0usize;
    let mut select_used = 0usize;
    for (node, plan) in &planned {
        let node = *node;
        match plan {
            Plan::Direct {
                base,
                scalar: true,
                reads,
                cross,
                ..
            } => {
                let wrote = sched
                    .open
                    .as_ref()
                    .is_some_and(|o| reads.iter().any(|v| o.writes.contains(v)));
                let _ = cross;
                if sched.open.is_some() && !wrote {
                    sched.close_loop();
                    let text = sched.scalar_inline(node, *base);
                    let o = sched.open.as_mut().expect("open");
                    o.passes.push_str(&text);
                    o.prologue_writes.insert(*base);
                } else {
                    if wrote {
                        sched.flush_open();
                    }
                    sched.pending_scalars.push((node, *base));
                }
            }
            Plan::Direct {
                base,
                len,
                reads,
                cross,
                ..
            } => {
                let regs = sched.regs.clone();
                sched.ensure_open(*base, StepKind::Wide, len, |o| {
                    o.len.as_ref().is_none_or(|l| l == len)
                        && !cross.iter().any(|v| o.writes.contains(v))
                        && !reads
                            .iter()
                            .any(|v| o.prologue_writes.contains(v) && !regs.contains_key(v))
                });
                if sched
                    .open
                    .as_ref()
                    .is_some_and(|o| o.loop_len.as_ref().is_some_and(|l| l != len))
                {
                    sched.close_loop();
                }
                let o = sched.open.as_mut().expect("opened");
                if o.loop_len.is_none() {
                    o.loop_len = Some(len.clone());
                }
                o.loop_members.push((node, *base));
                o.writes.insert(*base);
            }
            Plan::Gather {
                text,
                writes,
                single_row,
            } => {
                let len = key_of(*writes);
                sched.ensure_open(*writes, StepKind::Wide, &len, |o| {
                    o.len.as_ref().is_none_or(|l| *l == len)
                });
                sched.close_loop();
                let o = sched.open.as_mut().expect("opened");
                o.passes.push_str(text);
                if *single_row {
                    o.writes.insert(*writes);
                } else {
                    sched.flush_open();
                }
            }
            Plan::Generic {
                text,
                reads,
                writes,
                len,
                cross,
            } => {
                if sched.open.is_none() && sched.reads_pending(reads) {
                    sched.flush_pending();
                }
                sched.ensure_open(*writes, StepKind::Wide, len, |o| {
                    o.len.as_ref().is_none_or(|l| l == len)
                        && !(*cross && reads.iter().any(|v| o.writes.contains(v)))
                        && !reads.iter().any(|v| o.prologue_writes.contains(v))
                });
                sched.close_loop();
                let o = sched.open.as_mut().expect("opened");
                o.passes.push_str(text);
                o.writes.insert(*writes);
            }
            Plan::Put { text, value, len } => {
                sched.ensure_open(*value, StepKind::Wide, len, |o| {
                    o.len.as_ref().is_none_or(|l| l == len || len.is_empty())
                });
                sched.close_loop();
                let o = sched.open.as_mut().expect("opened");
                o.passes.push_str(text);
            }
            Plan::Reduce {
                tag,
                dtype,
                input,
                result,
                len,
                split,
                two_level,
            } => {
                if *split && planes_used < split_count {
                    if sched.open.is_none() && sched.reads_pending(&[*input]) {
                        sched.flush_pending();
                    }
                    sched.ensure_open(*input, StepKind::Partial, len, |o| {
                        o.len.as_ref().is_none_or(|l| l == len)
                            && !o.prologue_writes.contains(input)
                    });
                    sched.close_loop();
                    let plane = planes_used;
                    planes_used += 1;
                    let (input_ptr, result_ptr) = {
                        let slots = &sched.slots[&node];
                        (slots.a0.clone(), slots.o0.clone())
                    };
                    let o = sched.open.as_mut().expect("opened");
                    let _ = writeln!(
                        o.passes,
                        "    m4_reduce_partial({tag}u, {dtype}u, {input_ptr}, reinterpret_cast<device uint*>(m4_planes + {plane}u * m4_plane_bytes), descriptors[{input}].len, m4_gtid, m4_gthreads, m4_simd_lane);"
                    );
                    sched.flush_open();
                    sched.pending_finals.push(PendingFinal {
                        tag: *tag,
                        dtype: *dtype,
                        input: *input,
                        result: *result,
                        plane,
                        input_ptr,
                        result_ptr,
                    });
                } else {
                    sched.flush_pending();
                    let _ = writeln!(sched.cases, "  case {}u: {{", sched.steps.len());
                    sched.cases.push_str(two_level);
                    sched.cases.push_str("    return;\n  }\n");
                    sched.steps.push(streamed_step(*input, StepKind::Reduce));
                }
            }
            Plan::Select {
                node,
                mode,
                a0,
                a1,
                o0,
                input,
            } => {
                sched.flush_pending();
                let q = select_used;
                select_used += 1;
                let base = format!("temporary + {}u", q * SELECT_BYTES);
                let d0 = format!("descriptors[lane_params[{node}].a0]");
                let d1 = format!("descriptors[lane_params[{node}].a1]");
                let mut case = |text: String, kind: StepKind| {
                    let _ = writeln!(sched.cases, "  case {}u: {{", sched.steps.len());
                    sched.cases.push_str(&text);
                    sched.cases.push_str("    return;\n  }\n");
                    sched.steps.push(streamed_step(*input, kind));
                };
                case(
                    format!(
                        "    m4_sel_init({base}, {o0}, {d0}, {d1}, {a1}, {mode}u, m4_gtid, m4_gthreads);\n"
                    ),
                    StepKind::Wide,
                );
                for _ in 0..SELECT_ROUNDS {
                    for pass in 0..4u32 {
                        case(
                            format!(
                                "    m4_sel_hist_pass({base}, {a0}, {d0}, {pass}u, m4_gtid, m4_gthreads, m3_tid, m3_threads, m4_sel_tg_hist);\n"
                            ),
                            StepKind::Wide,
                        );
                        case(
                            format!(
                                "    if (m4_group.x != 0) return;\n    m4_sel_pick({base}, {pass}u, m3_tid, m3_threads);\n"
                            ),
                            StepKind::Single,
                        );
                    }
                    case(
                        format!("    m4_sel_compact({base}, {a0}, {d0}, m4_gtid, m4_gthreads);\n"),
                        StepKind::Wide,
                    );
                    case(
                        format!(
                            "    if (m4_group.x != 0) return;\n    m4_sel_finish({base}, {a0}, {a1}, {o0}, {d0}, {d1}, {mode}u, m3_tid, m3_threads, m4_sel_key, m4_sel_idx, m4_sel_scan);\n"
                        ),
                        StepKind::Single,
                    );
                }
                case(
                    format!(
                        "    if (m4_group.x != 0) return;\n    m4_sel_fallback({base}, {a0}, {a1}, {o0}, {d0}, {d1}, {mode}u, m3_tid, m3_threads, m3_tgbuf);\n"
                    ),
                    StepKind::Single,
                );
            }
            Plan::Scatter {
                copy,
                rmw,
                base_len,
                result,
            } => {
                sched.ensure_open(*result, StepKind::Wide, base_len, |o| {
                    o.len.as_ref().is_none_or(|l| l == base_len)
                });
                sched.close_loop();
                let o = sched.open.as_mut().expect("opened");
                o.passes.push_str(copy);
                o.writes.insert(*result);
                sched.flush_open();
                let _ = writeln!(sched.cases, "  case {}u: {{", sched.steps.len());
                sched.cases.push_str(rmw);
                sched.cases.push_str("    return;\n  }\n");
                sched.steps.push(streamed_step(*result, StepKind::Single));
            }
            Plan::Single(text) => {
                sched.flush_pending();
                let _ = writeln!(sched.cases, "  case {}u: {{", sched.steps.len());
                sched.cases.push_str(text);
                sched.cases.push_str("    return;\n  }\n");
                sched.steps.push(streamed_step(0, StepKind::Single));
            }
            Plan::Argmax { text, input } => {
                sched.flush_pending();
                let _ = writeln!(sched.cases, "  case {}u: {{", sched.steps.len());
                sched.cases.push_str(text);
                sched.cases.push_str("    return;\n  }\n");
                sched.steps.push(streamed_step(*input, StepKind::Argmax));
            }
        }
    }
    sched.flush_pending();

    source.push_str("  switch (step.index) {\n");
    source.push_str(&sched.cases);
    source.push_str("  default: return;\n  }\n}\n");
    Ok((source, sched.steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_every_case() {
        the_tree_has_the_levels_the_runtime_walks();
        a_step_round_trips();
        the_walks_with_state_stay_on_one_threadgroup();
        a_register_reads_as_a_memory_load_would();
    }

    fn the_tree_has_the_levels_the_runtime_walks() {
        assert_eq!(reduce_levels(0), 1);
        assert_eq!(reduce_levels(1), 1);
        assert_eq!(reduce_levels(32), 1);
        assert_eq!(reduce_levels(33), 2);
        assert_eq!(reduce_levels(1024), 2);
        assert_eq!(reduce_levels(1025), 3);
        assert_eq!(reduce_levels(248_320), 4);
    }

    fn a_step_round_trips() {
        let step = streamed_step(41, StepKind::Reduce);
        assert_eq!(step_value(step), 41);
        assert_eq!(step_kind(step), Some(StepKind::Reduce));
        assert_eq!(
            step_kind(streamed_step(3, StepKind::Partial)),
            Some(StepKind::Partial)
        );
        assert_eq!(step_kind(0xFF), None);
    }

    fn the_walks_with_state_stay_on_one_threadgroup() {
        assert_eq!(op_step_kind(tags::EXP, 0, false), StepKind::Wide);
        assert_eq!(op_step_kind(tags::CUMSUM, 0, false), StepKind::Single);
        assert_eq!(
            op_step_kind(tags::PIVOT_THRESHOLD, predicate_tags::CUMMASS_LE, false),
            StepKind::Single
        );
        assert_eq!(
            op_step_kind(tags::PIVOT_THRESHOLD, predicate_tags::PROB_GE, false),
            StepKind::Wide
        );
        assert_eq!(op_step_kind(tags::REDUCE_SUM, 0, false), StepKind::Reduce);
        assert_eq!(op_step_kind(tags::REDUCE_ARGMAX, 0, true), StepKind::Argmax);
        assert_eq!(
            op_step_kind(tags::REDUCE_ARGMAX, 0, false),
            StepKind::Single
        );
    }

    fn a_register_reads_as_a_memory_load_would() {
        assert_eq!(typed_from_bits(0, "r", 0), "as_type<float>(r)");
        assert_eq!(typed_from_bits(0, "r", 3), "(((r) != 0u) ? 1.0f : 0.0f)");
        assert_eq!(typed_from_bits(0, "r", 1), "float(int(r))");
    }
}
