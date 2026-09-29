//! A typed StableHLO text builder.
//!
//! Every op is printed in MLIR's generic form (`"stablehlo.add"(%a, %b) :
//! (...) -> ...`), which is stable across StableHLO versions and needs no
//! per-op pretty syntax. Float constants are printed as hex bit patterns so a
//! value round-trips exactly. A [`Func`] tracks the type of every SSA value,
//! so shape errors surface here, as a [`Malformed`], and not as an MLIR
//! diagnostic from inside the plugin.

use std::fmt::{self, Write as _};

/// Element types a StableHLO tensor holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    Pred,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F8E4m3fn,
    F8E5m2,
    F8E8m0fnu,
    F4E2m1fn,
    /// Unsigned 4-bit integer (`ui4`): a quantized code stored one per
    /// element (the device packs two per byte).
    U4,
}

impl Elem {
    #[must_use]
    pub const fn mlir(self) -> &'static str {
        match self {
            Self::Pred => "i1",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "ui8",
            Self::U16 => "ui16",
            Self::U32 => "ui32",
            Self::U64 => "ui64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F8E4m3fn => "f8E4M3FN",
            Self::F8E5m2 => "f8E5M2",
            Self::F8E8m0fnu => "f8E8M0FNU",
            Self::F4E2m1fn => "f4E2M1FN",
            Self::U4 => "ui4",
        }
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Pred => 1,
            Self::F4E2m1fn | Self::U4 => 4,
            Self::I8 | Self::U8 | Self::F8E4m3fn | Self::F8E5m2 | Self::F8E8m0fnu => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 => 64,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::Bf16
                | Self::F32
                | Self::F8E4m3fn
                | Self::F8E5m2
                | Self::F8E8m0fnu
                | Self::F4E2m1fn
        )
    }

    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::U4 | Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    #[must_use]
    pub const fn is_int(self) -> bool {
        !self.is_float() && !matches!(self, Self::Pred)
    }
}

/// A ranked tensor type; `dims` empty is a scalar.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ty {
    pub dims: Vec<i64>,
    pub elem: Elem,
}

impl Ty {
    #[must_use]
    pub fn new(elem: Elem, dims: &[i64]) -> Self {
        Self {
            dims: dims.to_vec(),
            elem,
        }
    }

    #[must_use]
    pub fn scalar(elem: Elem) -> Self {
        Self::new(elem, &[])
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[must_use]
    pub fn elements(&self) -> i64 {
        self.dims.iter().product()
    }

    #[must_use]
    pub fn with_elem(&self, elem: Elem) -> Self {
        Self {
            dims: self.dims.clone(),
            elem,
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tensor<")?;
        for d in &self.dims {
            write!(f, "{d}x")?;
        }
        write!(f, "{}>", self.elem.mlir())
    }
}

/// An SSA value inside one [`Func`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Val(u32);

impl Val {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A builder-side refusal: the op was asked for with operands it cannot take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Malformed {
    pub op: &'static str,
    pub detail: String,
}

impl fmt::Display for Malformed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stablehlo.{}: {}", self.op, self.detail)
    }
}

impl std::error::Error for Malformed {}

pub type Built<T> = Result<T, Malformed>;

fn malformed<T>(op: &'static str, detail: impl Into<String>) -> Built<T> {
    Err(Malformed {
        op,
        detail: detail.into(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    const fn mlir(self) -> &'static str {
        match self {
            Self::Eq => "EQ",
            Self::Ne => "NE",
            Self::Lt => "LT",
            Self::Le => "LE",
            Self::Gt => "GT",
            Self::Ge => "GE",
        }
    }
}

/// Which reduction a region computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Sum,
    Max,
    Min,
    Prod,
    And,
    Or,
}

/// Gather dimension numbers, spelled as StableHLO spells them.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GatherDims {
    pub offset_dims: Vec<i64>,
    pub collapsed_slice_dims: Vec<i64>,
    pub operand_batching_dims: Vec<i64>,
    pub start_indices_batching_dims: Vec<i64>,
    pub start_index_map: Vec<i64>,
    pub index_vector_dim: i64,
}

/// Scatter dimension numbers.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ScatterDims {
    pub update_window_dims: Vec<i64>,
    pub inserted_window_dims: Vec<i64>,
    pub input_batching_dims: Vec<i64>,
    pub scatter_indices_batching_dims: Vec<i64>,
    pub scatter_dims_to_operand_dims: Vec<i64>,
    pub index_vector_dim: i64,
}

/// How a scatter combines an update with what it lands on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combine {
    Set,
    Add,
    Max,
}

/// One function under construction. Values are numbered across the whole
/// function (regions included), so any value may be named anywhere below its
/// definition.
#[derive(Clone, Debug)]
pub struct Func {
    name: String,
    names: Vec<String>,
    tys: Vec<Ty>,
    params: Vec<(Val, Option<u32>)>,
    body: String,
    depth: usize,
    next: u32,
    /// The source location every op emitted now carries (`loc("…")`), which
    /// XLA keeps as the op's name in profiles.
    scope: Option<String>,
}

fn list<T: fmt::Display>(xs: &[T]) -> String {
    let mut s = String::new();
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{x}");
    }
    s
}

fn i64s(xs: &[i64]) -> String {
    list(xs)
}

fn array(xs: &[i64]) -> String {
    if xs.is_empty() {
        "array<i64>".into()
    } else {
        format!("array<i64: {}>", i64s(xs))
    }
}

/// The bit pattern of `x` in `elem`, rounded to nearest-even for narrow
/// floats, as MLIR's hex float literal expects.
fn float_bits(elem: Elem, x: f64) -> String {
    match elem {
        Elem::F32 => format!("0x{:08X}", (x as f32).to_bits()),
        Elem::Bf16 => format!("0x{:04X}", bf16_bits(x as f32)),
        Elem::F16 => format!("0x{:04X}", f16_bits(x as f32)),
        _ => format!("{x:e}"),
    }
}

/// f32 → bf16 bits, round to nearest even; NaN stays NaN.
#[must_use]
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x40;
    }
    let lsb = (b >> 16) & 1;
    ((b + 0x7FFF + lsb) >> 16) as u16
}

/// f32 → f16 bits, round to nearest even.
#[must_use]
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1FFF;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

impl Func {
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            names: Vec::new(),
            tys: Vec::new(),
            params: Vec::new(),
            body: String::new(),
            depth: 1,
            next: 0,
            scope: None,
        }
    }

    #[must_use]
    pub fn ty(&self, v: Val) -> &Ty {
        &self.tys[v.0 as usize]
    }

    #[must_use]
    pub fn dims(&self, v: Val) -> &[i64] {
        &self.tys[v.0 as usize].dims
    }

    #[must_use]
    pub fn elem(&self, v: Val) -> Elem {
        self.tys[v.0 as usize].elem
    }

    #[must_use]
    pub fn params(&self) -> usize {
        self.params.len()
    }

    fn fresh(&mut self, name: String, ty: Ty) -> Val {
        let v = Val(self.names.len() as u32);
        self.names.push(name);
        self.tys.push(ty);
        v
    }

    fn name(&self, v: Val) -> &str {
        &self.names[v.0 as usize]
    }

    fn names_of(&self, vs: &[Val]) -> String {
        let mut s = String::new();
        for (i, v) in vs.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(self.name(*v));
        }
        s
    }

    fn tys_of(&self, vs: &[Val]) -> String {
        let mut s = String::new();
        for (i, v) in vs.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let _ = write!(s, "{}", self.ty(*v));
        }
        s
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.body.push_str("  ");
        }
        self.body.push_str(text);
        // An op's line ends in its type; block headers and region openers
        // (`^bb0(...):`, `... ({`, `}, {`) carry no location.
        if let Some(scope) = &self.scope
            && !text.ends_with('{')
            && !text.ends_with(':')
            && !text.starts_with('^')
        {
            self.body.push_str(" loc(\"");
            self.body.push_str(scope);
            self.body.push_str("\")");
        }
        self.body.push('\n');
    }

    /// Names every op emitted from now on (`None` stops naming them).
    pub fn set_scope(&mut self, scope: Option<&str>) {
        self.scope = scope.map(|s| s.replace('"', "'"));
    }

    /// Declares the next function parameter. `alias` names the result index
    /// this parameter's buffer is donated to.
    pub fn param(&mut self, ty: Ty, alias: Option<u32>) -> Val {
        let name = format!("%arg{}", self.params.len());
        let v = self.fresh(name, ty);
        self.params.push((v, alias));
        v
    }

    /// Donates parameter `param` to result `result`: the executable may
    /// write the result into the parameter's buffer.
    pub fn alias(&mut self, param: usize, result: u32) {
        if let Some(p) = self.params.get_mut(param) {
            p.1 = Some(result);
        }
    }

    /// Emits `op` with generic syntax; one result.
    fn op(&mut self, op: &str, operands: &[Val], attrs: &str, out: Ty) -> Val {
        let id = self.next;
        self.next += 1;
        let name = format!("%{id}");
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{attrs}}}")
        };
        let text = format!(
            "{name} = \"stablehlo.{op}\"({}){attrs} : ({}) -> {out}",
            self.names_of(operands),
            self.tys_of(operands)
        );
        self.line(&text);
        self.fresh(name, out)
    }

    /// Emits an op of any dialect (`"chlo.ragged_dot"`) with generic
    /// syntax; several results.
    pub fn op_named(&mut self, name: &str, operands: &[Val], attrs: &str, outs: Vec<Ty>) -> Vec<Val> {
        let id = self.next;
        self.next += 1;
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{attrs}}}")
        };
        let single = outs.len() == 1;
        let head = if single {
            format!("%{id}")
        } else {
            format!("%{id}:{}", outs.len())
        };
        let results = if single {
            format!("{}", outs[0])
        } else {
            format!("({})", list(&outs))
        };
        let text = format!(
            "{head} = \"{name}\"({}){attrs} : ({}) -> {results}",
            self.names_of(operands),
            self.tys_of(operands)
        );
        self.line(&text);
        outs.into_iter()
            .enumerate()
            .map(|(i, ty)| {
                let n = if single {
                    format!("%{id}")
                } else {
                    format!("%{id}#{i}")
                };
                self.fresh(n, ty)
            })
            .collect()
    }

    /// Emits `op` with generic syntax; several results.
    pub fn op_n(&mut self, op: &str, operands: &[Val], attrs: &str, outs: Vec<Ty>) -> Vec<Val> {
        if outs.len() == 1 {
            let out = outs.into_iter().next().unwrap_or_else(|| Ty::scalar(Elem::F32));
            return vec![self.op(op, operands, attrs, out)];
        }
        let id = self.next;
        self.next += 1;
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{attrs}}}")
        };
        let text = format!(
            "%{id}:{} = \"stablehlo.{op}\"({}){attrs} : ({}) -> ({})",
            outs.len(),
            self.names_of(operands),
            self.tys_of(operands),
            list(&outs)
        );
        self.line(&text);
        outs.into_iter()
            .enumerate()
            .map(|(i, ty)| self.fresh(format!("%{id}#{i}"), ty))
            .collect()
    }

    /// Emits an op carrying one region. `region` receives the block arguments
    /// and returns what the region yields.
    pub fn op_region(
        &mut self,
        op: &str,
        operands: &[Val],
        block: &[Ty],
        region: impl FnOnce(&mut Self, &[Val]) -> Built<Vec<Val>>,
        attrs: &str,
        outs: Vec<Ty>,
    ) -> Built<Vec<Val>> {
        let id = self.next;
        self.next += 1;
        let head = if outs.len() == 1 {
            format!("%{id}")
        } else {
            format!("%{id}:{}", outs.len())
        };
        let text = format!("{head} = \"stablehlo.{op}\"({}) ({{", self.names_of(operands));
        self.line(&text);
        let args = self.block(block, region)?;
        let _ = args;
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{attrs}}}")
        };
        let text = format!(
            "}}){attrs} : ({}) -> ({})",
            self.tys_of(operands),
            list(&outs)
        );
        self.line(&text);
        let single = outs.len() == 1;
        Ok(outs
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                let name = if single {
                    format!("%{id}")
                } else {
                    format!("%{id}#{i}")
                };
                self.fresh(name, ty)
            })
            .collect())
    }

    fn block(
        &mut self,
        block: &[Ty],
        region: impl FnOnce(&mut Self, &[Val]) -> Built<Vec<Val>>,
    ) -> Built<Vec<Val>> {
        let mut args = Vec::with_capacity(block.len());
        let mut head = String::from("^bb0(");
        for (i, ty) in block.iter().enumerate() {
            let id = self.next;
            self.next += 1;
            let name = format!("%b{id}");
            if i > 0 {
                head.push_str(", ");
            }
            let _ = write!(head, "{name}: {ty}");
            args.push(self.fresh(name, ty.clone()));
        }
        head.push_str("):");
        self.depth += 1;
        self.line(&head);
        self.depth += 1;
        let yields = region(self, &args)?;
        let text = format!(
            "\"stablehlo.return\"({}) : ({}) -> ()",
            self.names_of(&yields),
            self.tys_of(&yields)
        );
        self.line(&text);
        self.depth -= 2;
        Ok(args)
    }

    /// Prints the finished function; `results` are what it returns.
    #[must_use]
    pub fn render(&self, results: &[Val]) -> String {
        let mut s = String::new();
        let _ = write!(s, "func.func public @{}(", self.name);
        for (i, (v, alias)) in self.params.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let _ = write!(s, "{}: {}", self.name(*v), self.ty(*v));
            if let Some(out) = alias {
                let _ = write!(s, " {{tf.aliasing_output = {out} : i32}}");
            }
        }
        let _ = writeln!(s, ") -> ({}) {{", self.tys_of(results));
        s.push_str(&self.body);
        let _ = writeln!(
            s,
            "  \"func.return\"({}) : ({}) -> ()",
            self.names_of(results),
            self.tys_of(results)
        );
        s.push_str("}\n");
        s
    }

    /// Wraps the function in a module named `name`.
    #[must_use]
    pub fn module(&self, name: &str, results: &[Val]) -> String {
        format!("module @{name} {{\n{}}}\n", self.render(results))
    }

    // ---------------------------------------------------------------- consts

    /// A float splat of `x` over `dims`.
    pub fn const_f(&mut self, elem: Elem, x: f64, dims: &[i64]) -> Val {
        let lit = if elem.is_float() {
            float_bits(elem, x)
        } else {
            format!("{}", x as i64)
        };
        self.splat_lit(elem, &lit, dims)
    }

    /// A scalar constant broadcast over `dims`: a splat of any size stays one
    /// scalar in the executable (a `dense<..>` constant of the full shape is
    /// materialized as a literal of every element when XLA imports it).
    fn splat_lit(&mut self, elem: Elem, lit: &str, dims: &[i64]) -> Val {
        let scalar = Ty::scalar(elem);
        let c = self.op("constant", &[], &format!("value = dense<{lit}> : {scalar}"), scalar);
        if dims.is_empty() {
            return c;
        }
        let ty = Ty::new(elem, dims);
        self.op("broadcast_in_dim", &[c], "broadcast_dimensions = array<i64>", ty)
    }

    /// An integer (or bool) splat of `x` over `dims`.
    pub fn const_i(&mut self, elem: Elem, x: i64, dims: &[i64]) -> Val {
        let lit = match elem {
            Elem::Pred => (if x != 0 { "true" } else { "false" }).to_string(),
            e if e.is_float() => float_bits(e, x as f64),
            _ => format!("{x}"),
        };
        self.splat_lit(elem, &lit, dims)
    }

    /// A dense integer array constant.
    pub fn const_ints(&mut self, elem: Elem, xs: &[i64], dims: &[i64]) -> Built<Val> {
        let ty = Ty::new(elem, dims);
        if ty.elements() != xs.len() as i64 {
            return malformed(
                "constant",
                format!("{} values for {ty}", xs.len()),
            );
        }
        let body = if dims.is_empty() {
            format!("{}", xs.first().copied().unwrap_or(0))
        } else {
            format!("[{}]", i64s(xs))
        };
        if dims.len() > 1 {
            // Flat literal, then reshape: dense<[..]> must be nested per rank.
            let flat = self.const_ints(elem, xs, &[xs.len() as i64])?;
            return self.reshape(flat, dims);
        }
        Ok(self.op("constant", &[], &format!("value = dense<{body}> : {ty}"), ty))
    }

    /// A dense float array constant (1-d, reshaped as asked).
    pub fn const_floats(&mut self, elem: Elem, xs: &[f64], dims: &[i64]) -> Built<Val> {
        let n = xs.len() as i64;
        if Ty::new(elem, dims).elements() != n {
            return malformed("constant", format!("{n} values for {dims:?}"));
        }
        let lits: Vec<String> = xs.iter().map(|&x| float_bits(elem, x)).collect();
        let flat_ty = Ty::new(elem, &[n]);
        let flat = self.op(
            "constant",
            &[],
            &format!("value = dense<[{}]> : {flat_ty}", lits.join(", ")),
            flat_ty.clone(),
        );
        if dims == [n] {
            Ok(flat)
        } else {
            self.reshape(flat, dims)
        }
    }

    pub fn iota(&mut self, elem: Elem, dims: &[i64], dim: i64) -> Val {
        let ty = Ty::new(elem, dims);
        self.op("iota", &[], &format!("iota_dimension = {dim} : i64"), ty)
    }

    // ------------------------------------------------------------ elementwise

    fn same(&self, op: &'static str, a: Val, b: Val) -> Built<Ty> {
        let (ta, tb) = (self.ty(a), self.ty(b));
        if ta != tb {
            return malformed(op, format!("{ta} against {tb}"));
        }
        Ok(ta.clone())
    }

    pub fn binary(&mut self, op: &'static str, a: Val, b: Val) -> Built<Val> {
        let ty = self.same(op, a, b)?;
        Ok(self.op(op, &[a, b], "", ty))
    }

    pub fn unary(&mut self, op: &'static str, x: Val) -> Val {
        let ty = self.ty(x).clone();
        self.op(op, &[x], "", ty)
    }

    pub fn add(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("add", a, b)
    }
    pub fn sub(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("subtract", a, b)
    }
    pub fn mul(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("multiply", a, b)
    }
    pub fn div(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("divide", a, b)
    }
    pub fn max(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("maximum", a, b)
    }
    pub fn min(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("minimum", a, b)
    }
    pub fn rem(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("remainder", a, b)
    }
    pub fn pow(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("power", a, b)
    }
    pub fn and(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("and", a, b)
    }
    pub fn or(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("or", a, b)
    }
    pub fn xor(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("xor", a, b)
    }
    pub fn shl(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("shift_left", a, b)
    }
    pub fn shr(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("shift_right_logical", a, b)
    }
    pub fn sar(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("shift_right_arithmetic", a, b)
    }
    pub fn atan2(&mut self, a: Val, b: Val) -> Built<Val> {
        self.binary("atan2", a, b)
    }

    pub fn neg(&mut self, x: Val) -> Val {
        self.unary("negate", x)
    }
    pub fn exp(&mut self, x: Val) -> Val {
        self.unary("exponential", x)
    }
    pub fn expm1(&mut self, x: Val) -> Val {
        self.unary("exponential_minus_one", x)
    }
    pub fn log(&mut self, x: Val) -> Val {
        self.unary("log", x)
    }
    pub fn log1p(&mut self, x: Val) -> Val {
        self.unary("log_plus_one", x)
    }
    pub fn sqrt(&mut self, x: Val) -> Val {
        self.unary("sqrt", x)
    }
    pub fn rsqrt(&mut self, x: Val) -> Val {
        self.unary("rsqrt", x)
    }
    pub fn tanh(&mut self, x: Val) -> Val {
        self.unary("tanh", x)
    }
    pub fn sigmoid(&mut self, x: Val) -> Val {
        self.unary("logistic", x)
    }
    pub fn abs(&mut self, x: Val) -> Val {
        self.unary("abs", x)
    }
    pub fn sign(&mut self, x: Val) -> Val {
        self.unary("sign", x)
    }
    pub fn floor(&mut self, x: Val) -> Val {
        self.unary("floor", x)
    }
    pub fn ceil(&mut self, x: Val) -> Val {
        self.unary("ceil", x)
    }
    pub fn round_even(&mut self, x: Val) -> Val {
        self.unary("round_nearest_even", x)
    }
    pub fn sin(&mut self, x: Val) -> Val {
        self.unary("sine", x)
    }
    pub fn cos(&mut self, x: Val) -> Val {
        self.unary("cosine", x)
    }
    pub fn not(&mut self, x: Val) -> Val {
        self.unary("not", x)
    }
    pub fn popcnt(&mut self, x: Val) -> Val {
        self.unary("popcnt", x)
    }
    pub fn is_finite(&mut self, x: Val) -> Val {
        let ty = self.ty(x).with_elem(Elem::Pred);
        self.op("is_finite", &[x], "", ty)
    }

    pub fn compare(&mut self, dir: Cmp, a: Val, b: Val) -> Built<Val> {
        let ty = self.same("compare", a, b)?;
        let kind = if ty.elem.is_float() {
            "FLOAT"
        } else if ty.elem.is_unsigned() || ty.elem == Elem::Pred {
            "UNSIGNED"
        } else {
            "SIGNED"
        };
        Ok(self.op(
            "compare",
            &[a, b],
            &format!(
                "comparison_direction = #stablehlo<comparison_direction {}>, compare_type = #stablehlo<comparison_type {kind}>",
                dir.mlir()
            ),
            ty.with_elem(Elem::Pred),
        ))
    }

    pub fn select(&mut self, pred: Val, a: Val, b: Val) -> Built<Val> {
        let ty = self.same("select", a, b)?;
        if self.ty(pred).elem != Elem::Pred
            || (self.ty(pred).rank() != 0 && self.ty(pred).dims != ty.dims)
        {
            return malformed("select", format!("predicate {} for {ty}", self.ty(pred)));
        }
        Ok(self.op("select", &[pred, a, b], "", ty))
    }

    /// `clamp(lo, x, hi)`; `lo`/`hi` may be scalars.
    pub fn clamp(&mut self, lo: Val, x: Val, hi: Val) -> Built<Val> {
        let ty = self.ty(x).clone();
        for b in [lo, hi] {
            let tb = self.ty(b);
            if tb.elem != ty.elem || (tb.rank() != 0 && tb.dims != ty.dims) {
                return malformed("clamp", format!("bound {tb} for {ty}"));
            }
        }
        Ok(self.op("clamp", &[lo, x, hi], "", ty))
    }

    /// Value conversion (int↔float, widen, narrow); a no-op when already `elem`.
    pub fn convert(&mut self, x: Val, elem: Elem) -> Val {
        if self.elem(x) == elem {
            return x;
        }
        let ty = self.ty(x).with_elem(elem);
        self.op("convert", &[x], "", ty)
    }

    /// Reinterprets bits. Same width keeps the shape; a narrower target grows
    /// a trailing axis, a wider one consumes it.
    pub fn bitcast(&mut self, x: Val, elem: Elem) -> Built<Val> {
        let from = self.ty(x).clone();
        let (a, b) = (from.elem.bits(), elem.bits());
        let dims = if a == b {
            from.dims.clone()
        } else if a > b {
            let mut d = from.dims.clone();
            d.push(i64::from(a / b));
            d
        } else {
            let k = i64::from(b / a);
            match from.dims.split_last() {
                Some((&last, rest)) if last == k => rest.to_vec(),
                _ => {
                    return malformed(
                        "bitcast_convert",
                        format!("{from} does not end in {k} lanes of {}", elem.mlir()),
                    );
                }
            }
        };
        Ok(self.op("bitcast_convert", &[x], "", Ty::new(elem, &dims)))
    }

    // ------------------------------------------------------------------ shape

    pub fn reshape(&mut self, x: Val, dims: &[i64]) -> Built<Val> {
        let from = self.ty(x).clone();
        let to = Ty::new(from.elem, dims);
        if from.elements() != to.elements() {
            return malformed("reshape", format!("{from} to {to}"));
        }
        if from.dims == to.dims {
            return Ok(x);
        }
        Ok(self.op("reshape", &[x], "", to))
    }

    pub fn transpose(&mut self, x: Val, perm: &[i64]) -> Built<Val> {
        let from = self.ty(x).clone();
        if perm.len() != from.rank() {
            return malformed("transpose", format!("perm {perm:?} for {from}"));
        }
        if perm.iter().enumerate().all(|(i, &p)| i as i64 == p) {
            return Ok(x);
        }
        let dims: Vec<i64> = perm.iter().map(|&p| from.dims[p as usize]).collect();
        Ok(self.op(
            "transpose",
            &[x],
            &format!("permutation = {}", array(perm)),
            Ty::new(from.elem, &dims),
        ))
    }

    /// `broadcast_in_dim`: operand axis `i` lands on output axis `map[i]`.
    pub fn broadcast(&mut self, x: Val, dims: &[i64], map: &[i64]) -> Built<Val> {
        let from = self.ty(x).clone();
        if map.len() != from.rank() {
            return malformed("broadcast_in_dim", format!("map {map:?} for {from}"));
        }
        for (i, &m) in map.iter().enumerate() {
            let d = from.dims[i];
            if m as usize >= dims.len() || (d != 1 && d != dims[m as usize]) {
                return malformed(
                    "broadcast_in_dim",
                    format!("{from} axis {i} onto {dims:?} axis {m}"),
                );
            }
        }
        let to = Ty::new(from.elem, dims);
        if from == to {
            return Ok(x);
        }
        Ok(self.op(
            "broadcast_in_dim",
            &[x],
            &format!("broadcast_dimensions = {}", array(map)),
            to,
        ))
    }

    /// Broadcasts a scalar over `dims`.
    pub fn splat(&mut self, x: Val, dims: &[i64]) -> Built<Val> {
        self.broadcast(x, dims, &[])
    }

    /// Broadcasts `x` to `dims` by matching its axes to the trailing ones
    /// (numpy rule), for a same-rank or lower-rank `x`.
    pub fn broadcast_to(&mut self, x: Val, dims: &[i64]) -> Built<Val> {
        let r = self.ty(x).rank();
        if r > dims.len() {
            return malformed("broadcast_in_dim", format!("{} onto {dims:?}", self.ty(x)));
        }
        let off = (dims.len() - r) as i64;
        let map: Vec<i64> = (0..r as i64).map(|i| i + off).collect();
        self.broadcast(x, dims, &map)
    }

    pub fn slice(&mut self, x: Val, start: &[i64], limit: &[i64], strides: &[i64]) -> Built<Val> {
        let from = self.ty(x).clone();
        if start.len() != from.rank() || limit.len() != from.rank() || strides.len() != from.rank()
        {
            return malformed("slice", format!("{start:?}..{limit:?} of {from}"));
        }
        let mut dims = Vec::with_capacity(from.rank());
        for i in 0..from.rank() {
            if start[i] < 0 || limit[i] > from.dims[i] || start[i] > limit[i] || strides[i] < 1 {
                return malformed("slice", format!("{start:?}..{limit:?} of {from}"));
            }
            dims.push((limit[i] - start[i]).div_euclid(strides[i])
                + i64::from((limit[i] - start[i]).rem_euclid(strides[i]) != 0));
        }
        if dims == from.dims {
            return Ok(x);
        }
        Ok(self.op(
            "slice",
            &[x],
            &format!(
                "start_indices = {}, limit_indices = {}, strides = {}",
                array(start),
                array(limit),
                array(strides)
            ),
            Ty::new(from.elem, &dims),
        ))
    }

    /// Unit-stride slice of axis `axis` to `[lo, hi)`.
    pub fn slice_axis(&mut self, x: Val, axis: usize, lo: i64, hi: i64) -> Built<Val> {
        let d = self.dims(x).to_vec();
        let mut start = vec![0; d.len()];
        let mut limit = d.clone();
        start[axis] = lo;
        limit[axis] = hi;
        self.slice(x, &start, &limit, &vec![1; d.len()])
    }

    /// `dynamic_slice`: `starts` are i32 scalars, clamped into range by XLA.
    pub fn dynamic_slice(&mut self, x: Val, starts: &[Val], sizes: &[i64]) -> Built<Val> {
        let from = self.ty(x).clone();
        if starts.len() != from.rank() || sizes.len() != from.rank() {
            return malformed("dynamic_slice", format!("{sizes:?} of {from}"));
        }
        let mut operands = vec![x];
        operands.extend_from_slice(starts);
        Ok(self.op(
            "dynamic_slice",
            &operands,
            &format!("slice_sizes = {}", array(sizes)),
            Ty::new(from.elem, sizes),
        ))
    }

    pub fn dynamic_update_slice(&mut self, x: Val, update: Val, starts: &[Val]) -> Built<Val> {
        let from = self.ty(x).clone();
        let up = self.ty(update).clone();
        if starts.len() != from.rank() || up.rank() != from.rank() || up.elem != from.elem {
            return malformed("dynamic_update_slice", format!("{up} into {from}"));
        }
        let mut operands = vec![x, update];
        operands.extend_from_slice(starts);
        Ok(self.op("dynamic_update_slice", &operands, "", from))
    }

    pub fn concat(&mut self, xs: &[Val], axis: i64) -> Built<Val> {
        let Some(&first) = xs.first() else {
            return malformed("concatenate", "nothing to join");
        };
        if xs.len() == 1 {
            return Ok(first);
        }
        let mut dims = self.dims(first).to_vec();
        let elem = self.elem(first);
        dims[axis as usize] = 0;
        for &x in xs {
            let t = self.ty(x);
            if t.elem != elem || t.rank() != dims.len() {
                return malformed("concatenate", format!("{t} joined with {elem:?}"));
            }
            dims[axis as usize] += t.dims[axis as usize];
        }
        Ok(self.op(
            "concatenate",
            xs,
            &format!("dimension = {axis} : i64"),
            Ty::new(elem, &dims),
        ))
    }

    pub fn pad(&mut self, x: Val, value: Val, lo: &[i64], hi: &[i64], interior: &[i64]) -> Built<Val> {
        let from = self.ty(x).clone();
        let r = from.rank();
        if lo.len() != r || hi.len() != r || interior.len() != r {
            return malformed("pad", format!("{lo:?}/{hi:?} for {from}"));
        }
        let dims: Vec<i64> = (0..r)
            .map(|i| lo[i] + hi[i] + from.dims[i] + (from.dims[i] - 1).max(0) * interior[i])
            .collect();
        Ok(self.op(
            "pad",
            &[x, value],
            &format!(
                "edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}",
                array(lo),
                array(hi),
                array(interior)
            ),
            Ty::new(from.elem, &dims),
        ))
    }

    /// Reverses the listed axes.
    pub fn reverse(&mut self, x: Val, axes: &[i64]) -> Val {
        let ty = self.ty(x).clone();
        self.op("reverse", &[x], &format!("dimensions = {}", array(axes)), ty)
    }

    // ----------------------------------------------------------------- linear

    /// `dot_general` with explicit batch and contracting axes; the result
    /// holds batch axes, then lhs free axes, then rhs free axes, in `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn dot_general(
        &mut self,
        lhs: Val,
        rhs: Val,
        lhs_batch: &[i64],
        rhs_batch: &[i64],
        lhs_contract: &[i64],
        rhs_contract: &[i64],
        out: Elem,
    ) -> Built<Val> {
        let highest = self.elem(lhs) == Elem::F32 || self.elem(rhs) == Elem::F32;
        self.dot_general_at(lhs, rhs, lhs_batch, rhs_batch, lhs_contract, rhs_contract, out, highest)
    }

    /// `dot_general` with the precision stated: `highest` asks the MXU for
    /// full f32 (six bf16 passes on TPU); off, an f32 operand is rounded to
    /// bf16 once. `dot_general` picks `highest` whenever an operand is f32.
    #[allow(clippy::too_many_arguments)]
    pub fn dot_general_at(
        &mut self,
        lhs: Val,
        rhs: Val,
        lhs_batch: &[i64],
        rhs_batch: &[i64],
        lhs_contract: &[i64],
        rhs_contract: &[i64],
        out: Elem,
        highest: bool,
    ) -> Built<Val> {
        let (l, r) = (self.ty(lhs).clone(), self.ty(rhs).clone());
        if lhs_batch.len() != rhs_batch.len() || lhs_contract.len() != rhs_contract.len() {
            return malformed("dot_general", format!("{l} · {r}: axis lists differ"));
        }
        for (a, b) in lhs_batch.iter().zip(rhs_batch).chain(lhs_contract.iter().zip(rhs_contract)) {
            if l.dims[*a as usize] != r.dims[*b as usize] {
                return malformed("dot_general", format!("{l} · {r}: axis {a} against {b}"));
            }
        }
        let mut dims: Vec<i64> = lhs_batch.iter().map(|&a| l.dims[a as usize]).collect();
        for (i, &d) in l.dims.iter().enumerate() {
            let i = i as i64;
            if !lhs_batch.contains(&i) && !lhs_contract.contains(&i) {
                dims.push(d);
            }
        }
        for (i, &d) in r.dims.iter().enumerate() {
            let i = i as i64;
            if !rhs_batch.contains(&i) && !rhs_contract.contains(&i) {
                dims.push(d);
            }
        }
        Ok(self.op(
            "dot_general",
            &[lhs, rhs],
            &(format!(
                "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [{}], rhs_batching_dimensions = [{}], lhs_contracting_dimensions = [{}], rhs_contracting_dimensions = [{}]>",
                i64s(lhs_batch),
                i64s(rhs_batch),
                i64s(lhs_contract),
                i64s(rhs_contract)
            ) + if highest {
                ", precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]"
            } else {
                ""
            }),
            Ty::new(out, &dims),
        ))
    }

    /// `[m, k] · [n, k]ᵀ → [m, n]`, the row-major weight form every linear
    /// takes.
    pub fn matmul_nt(&mut self, x: Val, w: Val, out: Elem) -> Built<Val> {
        self.dot_general(x, w, &[], &[], &[1], &[1], out)
    }

    // ------------------------------------------------------------- reductions

    fn fold_init(&mut self, fold: Fold, elem: Elem) -> Val {
        match fold {
            Fold::Sum | Fold::Or => self.const_i(elem, 0, &[]),
            Fold::Prod => self.const_i(elem, 1, &[]),
            Fold::And => self.const_i(elem, -1, &[]),
            Fold::Max => {
                if elem.is_float() {
                    self.const_f(elem, f64::NEG_INFINITY, &[])
                } else {
                    self.const_i(elem, int_min(elem), &[])
                }
            }
            Fold::Min => {
                if elem.is_float() {
                    self.const_f(elem, f64::INFINITY, &[])
                } else {
                    self.const_i(elem, int_max(elem), &[])
                }
            }
        }
    }

    fn fold_op(&mut self, fold: Fold, a: Val, b: Val) -> Built<Val> {
        match fold {
            Fold::Sum => self.add(a, b),
            Fold::Max => self.max(a, b),
            Fold::Min => self.min(a, b),
            Fold::Prod => self.mul(a, b),
            Fold::And => self.and(a, b),
            Fold::Or => self.or(a, b),
        }
    }

    /// Reduces `axes` of `x` with `fold`.
    pub fn reduce(&mut self, x: Val, axes: &[i64], fold: Fold) -> Built<Val> {
        let ty = self.ty(x).clone();
        let init = self.fold_init(fold, ty.elem);
        let dims: Vec<i64> = ty
            .dims
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(&(*i as i64)))
            .map(|(_, &d)| d)
            .collect();
        let s = Ty::scalar(ty.elem);
        let out = self.op_region(
            "reduce",
            &[x, init],
            &[s.clone(), s],
            |f, b| Ok(vec![f.fold_op(fold, b[0], b[1])?]),
            &format!("dimensions = {}", array(axes)),
            vec![Ty::new(ty.elem, &dims)],
        )?;
        Ok(out[0])
    }

    /// A variadic reduce with a caller-built region.
    pub fn reduce_with(
        &mut self,
        xs: &[Val],
        inits: &[Val],
        axes: &[i64],
        region: impl FnOnce(&mut Self, &[Val], &[Val]) -> Built<Vec<Val>>,
    ) -> Built<Vec<Val>> {
        let n = xs.len();
        let mut operands = xs.to_vec();
        operands.extend_from_slice(inits);
        let mut block = Vec::with_capacity(2 * n);
        let mut outs = Vec::with_capacity(n);
        for i in 0..2 * n {
            block.push(Ty::scalar(self.elem(operands[i % n])));
        }
        for &x in xs {
            let ty = self.ty(x).clone();
            let dims: Vec<i64> = ty
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !axes.contains(&(*i as i64)))
                .map(|(_, &d)| d)
                .collect();
            outs.push(Ty::new(ty.elem, &dims));
        }
        self.op_region(
            "reduce",
            &operands,
            &block,
            |f, b| region(f, &b[..n], &b[n..]),
            &format!("dimensions = {}", array(axes)),
            outs,
        )
    }

    /// First index of the maximum along `axis` (ties → lowest index), with
    /// the maximum itself.
    pub fn argmax(&mut self, x: Val, axis: i64, index: Elem) -> Built<(Val, Val)> {
        let ty = self.ty(x).clone();
        let iota = self.iota(index, &ty.dims, axis);
        let lo = if ty.elem.is_float() {
            self.const_f(ty.elem, f64::NEG_INFINITY, &[])
        } else {
            self.const_i(ty.elem, int_min(ty.elem), &[])
        };
        let zero = self.const_i(index, 0, &[]);
        let out = self.reduce_with(&[x, iota], &[lo, zero], &[axis], |f, a, b| {
            let (av, ai, bv, bi) = (a[0], a[1], b[0], b[1]);
            let gt = f.compare(Cmp::Gt, av, bv)?;
            let eq = f.compare(Cmp::Eq, av, bv)?;
            let lt_i = f.compare(Cmp::Lt, ai, bi)?;
            let tie = f.and(eq, lt_i)?;
            // NaN wins, as a NaN is never less than anything.
            let a_nan = f.compare(Cmp::Ne, av, av)?;
            let take_a = f.or(gt, tie)?;
            let take_a = f.or(take_a, a_nan)?;
            let v = f.select(take_a, av, bv)?;
            let i = f.select(take_a, ai, bi)?;
            Ok(vec![v, i])
        })?;
        Ok((out[1], out[0]))
    }

    /// `reduce_window` with a caller-chosen fold; used for prefix scans.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_window(
        &mut self,
        x: Val,
        fold: Fold,
        window: &[i64],
        strides: &[i64],
        pad_lo: &[i64],
        pad_hi: &[i64],
    ) -> Built<Val> {
        let ty = self.ty(x).clone();
        let init = self.fold_init(fold, ty.elem);
        let dims: Vec<i64> = (0..ty.rank())
            .map(|i| (ty.dims[i] + pad_lo[i] + pad_hi[i] - window[i]) / strides[i] + 1)
            .collect();
        let padding: Vec<String> = (0..ty.rank())
            .map(|i| format!("[{}, {}]", pad_lo[i], pad_hi[i]))
            .collect();
        let s = Ty::scalar(ty.elem);
        let out = self.op_region(
            "reduce_window",
            &[x, init],
            &[s.clone(), s],
            |f, b| Ok(vec![f.fold_op(fold, b[0], b[1])?]),
            &format!(
                "window_dimensions = {}, window_strides = {}, padding = dense<[{}]> : tensor<{}x2xi64>",
                array(window),
                array(strides),
                padding.join(", "),
                ty.rank()
            ),
            vec![Ty::new(ty.elem, &dims)],
        )?;
        Ok(out[0])
    }

    /// Inclusive prefix fold along `axis`.
    pub fn scan(&mut self, x: Val, axis: usize, fold: Fold) -> Built<Val> {
        let d = self.dims(x).to_vec();
        let r = d.len();
        let mut window = vec![1; r];
        window[axis] = d[axis];
        let mut lo = vec![0; r];
        lo[axis] = d[axis] - 1;
        self.reduce_window(x, fold, &window, &vec![1; r], &lo, &vec![0; r])
    }

    // ----------------------------------------------------------- gather/scatter

    pub fn gather(
        &mut self,
        operand: Val,
        indices: Val,
        dims: &GatherDims,
        slice_sizes: &[i64],
    ) -> Built<Val> {
        let op = self.ty(operand).clone();
        let idx = self.ty(indices).clone();
        // Result shape: batch dims of indices (all but index_vector_dim) with
        // offset dims inserted at `offset_dims`.
        let mut batch: Vec<i64> = idx.dims.clone();
        if (dims.index_vector_dim as usize) < batch.len() {
            batch.remove(dims.index_vector_dim as usize);
        }
        let offsets: Vec<i64> = slice_sizes
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let i = *i as i64;
                !dims.collapsed_slice_dims.contains(&i) && !dims.operand_batching_dims.contains(&i)
            })
            .map(|(_, &s)| s)
            .collect();
        if offsets.len() != dims.offset_dims.len() {
            return malformed(
                "gather",
                format!("{} offset dims for {} slice axes", dims.offset_dims.len(), offsets.len()),
            );
        }
        let rank = batch.len() + offsets.len();
        let mut out = Vec::with_capacity(rank);
        let (mut bi, mut oi) = (0, 0);
        for i in 0..rank as i64 {
            if dims.offset_dims.contains(&i) {
                out.push(offsets[oi]);
                oi += 1;
            } else {
                out.push(batch[bi]);
                bi += 1;
            }
        }
        let attrs = format!(
            "dimension_numbers = #stablehlo.gather<offset_dims = [{}], collapsed_slice_dims = [{}], operand_batching_dims = [{}], start_indices_batching_dims = [{}], start_index_map = [{}], index_vector_dim = {}>, slice_sizes = {}, indices_are_sorted = false",
            i64s(&dims.offset_dims),
            i64s(&dims.collapsed_slice_dims),
            i64s(&dims.operand_batching_dims),
            i64s(&dims.start_indices_batching_dims),
            i64s(&dims.start_index_map),
            dims.index_vector_dim,
            array(slice_sizes)
        );
        Ok(self.op("gather", &[operand, indices], &attrs, Ty::new(op.elem, &out)))
    }

    /// Rows of a rank-2 `table` picked by the i32 vector `ids`: `[n, width]`.
    pub fn take_rows(&mut self, table: Val, ids: Val) -> Built<Val> {
        let width = self.dims(table)[1];
        let n = self.dims(ids)[0];
        let ids = self.reshape(ids, &[n, 1])?;
        self.gather(
            table,
            ids,
            &GatherDims {
                offset_dims: vec![1],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                ..GatherDims::default()
            },
            &[1, width],
        )
    }

    pub fn scatter(
        &mut self,
        operand: Val,
        indices: Val,
        updates: Val,
        dims: &ScatterDims,
        combine: Combine,
    ) -> Built<Val> {
        self.scatter_hinted(operand, indices, updates, dims, combine, false)
    }

    /// `scatter` telling XLA the indices are pairwise distinct (out-of-range
    /// ones included), which lets it update in parallel instead of in order.
    /// Only sound when they are: duplicate indices under the hint are
    /// undefined.
    pub fn scatter_hinted(
        &mut self,
        operand: Val,
        indices: Val,
        updates: Val,
        dims: &ScatterDims,
        combine: Combine,
        unique: bool,
    ) -> Built<Val> {
        let ty = self.ty(operand).clone();
        let s = Ty::scalar(ty.elem);
        let attrs = format!(
            "scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [{}], inserted_window_dims = [{}], input_batching_dims = [{}], scatter_indices_batching_dims = [{}], scatter_dims_to_operand_dims = [{}], index_vector_dim = {}>, indices_are_sorted = false, unique_indices = false",
            i64s(&dims.update_window_dims),
            i64s(&dims.inserted_window_dims),
            i64s(&dims.input_batching_dims),
            i64s(&dims.scatter_indices_batching_dims),
            i64s(&dims.scatter_dims_to_operand_dims),
            dims.index_vector_dim
        )
        .replace(
            "unique_indices = false",
            if unique { "unique_indices = true" } else { "unique_indices = false" },
        );
        let out = self.op_region(
            "scatter",
            &[operand, indices, updates],
            &[s.clone(), s],
            |f, b| {
                Ok(vec![match combine {
                    Combine::Set => b[1],
                    Combine::Add => f.add(b[0], b[1])?,
                    Combine::Max => f.max(b[0], b[1])?,
                }])
            },
            &attrs,
            vec![ty],
        )?;
        Ok(out[0])
    }

    /// Writes `rows[i]` over `table[ids[i]]`; out-of-range ids are dropped.
    pub fn put_rows(&mut self, table: Val, ids: Val, rows: Val, combine: Combine) -> Built<Val> {
        self.put_rows_hinted(table, ids, rows, combine, false)
    }

    /// `put_rows` with the distinct-indices hint (see `scatter_hinted`).
    pub fn put_rows_hinted(
        &mut self,
        table: Val,
        ids: Val,
        rows: Val,
        combine: Combine,
        unique: bool,
    ) -> Built<Val> {
        let n = self.dims(ids)[0];
        let ids = self.reshape(ids, &[n, 1])?;
        self.scatter_hinted(
            table,
            ids,
            rows,
            &ScatterDims {
                update_window_dims: vec![1],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..ScatterDims::default()
            },
            combine,
            unique,
        )
    }

    // ------------------------------------------------------------------- sort

    /// Sorts `xs` together along `axis`, ordered by `less(lhs, rhs)` over the
    /// per-operand scalars.
    pub fn sort(
        &mut self,
        xs: &[Val],
        axis: i64,
        stable: bool,
        less: impl FnOnce(&mut Self, &[Val], &[Val]) -> Built<Val>,
    ) -> Built<Vec<Val>> {
        let n = xs.len();
        let mut block = Vec::with_capacity(2 * n);
        for &x in xs {
            let s = Ty::scalar(self.elem(x));
            block.push(s.clone());
            block.push(s);
        }
        let outs: Vec<Ty> = xs.iter().map(|&x| self.ty(x).clone()).collect();
        self.op_region(
            "sort",
            xs,
            &block,
            |f, b| {
                let lhs: Vec<Val> = (0..n).map(|i| b[2 * i]).collect();
                let rhs: Vec<Val> = (0..n).map(|i| b[2 * i + 1]).collect();
                Ok(vec![less(f, &lhs, &rhs)?])
            },
            &format!("dimension = {axis} : i64, is_stable = {stable}"),
            outs,
        )
    }

    // ---------------------------------------------------------------- control

    /// `while` over loop-carried `inits`: `cond` yields an i1 scalar, `body`
    /// the next carried values.
    pub fn while_loop(
        &mut self,
        inits: &[Val],
        cond: impl FnOnce(&mut Self, &[Val]) -> Built<Val>,
        body: impl FnOnce(&mut Self, &[Val]) -> Built<Vec<Val>>,
    ) -> Built<Vec<Val>> {
        let tys: Vec<Ty> = inits.iter().map(|&v| self.ty(v).clone()).collect();
        let id = self.next;
        self.next += 1;
        let head = if tys.len() == 1 {
            format!("%{id}")
        } else {
            format!("%{id}:{}", tys.len())
        };
        let text = format!("{head} = \"stablehlo.while\"({}) ({{", self.names_of(inits));
        self.line(&text);
        self.block(&tys, |f, a| Ok(vec![cond(f, a)?]))?;
        self.line("}, {");
        self.block(&tys, body)?;
        let text = format!("}}) : ({}) -> ({})", self.tys_of(inits), list(&tys));
        self.line(&text);
        let single = tys.len() == 1;
        Ok(tys
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                let name = if single {
                    format!("%{id}")
                } else {
                    format!("%{id}#{i}")
                };
                self.fresh(name, ty)
            })
            .collect())
    }

    /// A counted loop `for i in 0..n` carrying `inits`; `body` receives the
    /// i32 counter and the carried values.
    pub fn for_loop(
        &mut self,
        n: i64,
        inits: &[Val],
        body: impl FnOnce(&mut Self, Val, &[Val]) -> Built<Vec<Val>>,
    ) -> Built<Vec<Val>> {
        let zero = self.const_i(Elem::I32, 0, &[]);
        let mut carried = vec![zero];
        carried.extend_from_slice(inits);
        let out = self.while_loop(
            &carried,
            |f, a| {
                let bound = f.const_i(Elem::I32, n, &[]);
                f.compare(Cmp::Lt, a[0], bound)
            },
            |f, a| {
                let one = f.const_i(Elem::I32, 1, &[]);
                let next = f.add(a[0], one)?;
                let mut rest = body(f, a[0], &a[1..])?;
                rest.insert(0, next);
                Ok(rest)
            },
        )?;
        Ok(out[1..].to_vec())
    }

    /// `case` on an i32 index over branches that each yield `outs`.
    pub fn case(
        &mut self,
        index: Val,
        outs: &[Ty],
        branches: Vec<Box<dyn FnOnce(&mut Self) -> Built<Vec<Val>> + '_>>,
    ) -> Built<Vec<Val>> {
        let id = self.next;
        self.next += 1;
        let head = if outs.len() == 1 {
            format!("%{id}")
        } else {
            format!("%{id}:{}", outs.len())
        };
        let text = format!("{head} = \"stablehlo.case\"({}) ({{", self.name(index));
        self.line(&text);
        let count = branches.len();
        for (i, branch) in branches.into_iter().enumerate() {
            self.depth += 1;
            self.line("^bb0:");
            self.depth += 1;
            let ys = branch(self)?;
            let text = format!(
                "\"stablehlo.return\"({}) : ({}) -> ()",
                self.names_of(&ys),
                self.tys_of(&ys)
            );
            self.line(&text);
            self.depth -= 2;
            if i + 1 < count {
                self.line("}, {");
            }
        }
        let text = format!("}}) : ({}) -> ({})", self.ty(index), list(outs));
        self.line(&text);
        let single = outs.len() == 1;
        Ok(outs
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let name = if single {
                    format!("%{id}")
                } else {
                    format!("%{id}#{i}")
                };
                self.fresh(name, ty.clone())
            })
            .collect())
    }

    // ------------------------------------------------------------- composites

    /// A float scalar of `x` in `elem`, broadcast to `v`'s shape.
    pub fn like_f(&mut self, v: Val, x: f64) -> Val {
        let ty = self.ty(v).clone();
        self.const_f(ty.elem, x, &ty.dims)
    }

    /// An int scalar of `x` in `v`'s element type, broadcast to `v`'s shape.
    pub fn like_i(&mut self, v: Val, x: i64) -> Val {
        let ty = self.ty(v).clone();
        self.const_i(ty.elem, x, &ty.dims)
    }

    /// `x * s` for a float scalar `s`.
    pub fn scale(&mut self, x: Val, s: f64) -> Built<Val> {
        let c = self.like_f(x, s);
        self.mul(x, c)
    }

    /// `x + s` for a float scalar `s`.
    pub fn offset(&mut self, x: Val, s: f64) -> Built<Val> {
        let c = self.like_f(x, s);
        self.add(x, c)
    }

    /// `x * silu`-style `x * sigmoid(x)`.
    pub fn silu(&mut self, x: Val) -> Built<Val> {
        let s = self.sigmoid(x);
        self.mul(x, s)
    }

    /// tanh-approximated gelu.
    pub fn gelu_tanh(&mut self, x: Val) -> Built<Val> {
        // 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))
        let x2 = self.mul(x, x)?;
        let x3 = self.mul(x2, x)?;
        let k = self.scale(x3, 0.044715)?;
        let inner = self.add(x, k)?;
        let inner = self.scale(inner, (2.0 / std::f64::consts::PI).sqrt())?;
        let t = self.tanh(inner);
        let t = self.offset(t, 1.0)?;
        let hx = self.scale(x, 0.5)?;
        self.mul(hx, t)
    }

    /// Exact (erf) gelu via the Abramowitz–Stegun 7.1.26 erf; f32 accurate
    /// to ~1e-7, well past bf16.
    pub fn gelu_erf(&mut self, x: Val) -> Built<Val> {
        let z = self.scale(x, std::f64::consts::FRAC_1_SQRT_2)?;
        let e = self.erf(z)?;
        let e = self.offset(e, 1.0)?;
        let hx = self.scale(x, 0.5)?;
        self.mul(hx, e)
    }

    /// erf(x), rational approximation (max abs err 1.5e-7).
    pub fn erf(&mut self, x: Val) -> Built<Val> {
        let sign = self.sign(x);
        let a = self.abs(x);
        let p = self.scale(a, 0.327_591_1)?;
        let p = self.offset(p, 1.0)?;
        let one = self.like_f(x, 1.0);
        let t = self.div(one, p)?;
        let coeffs = [1.061_405_429, -1.453_152_027, 1.421_413_741, -0.284_496_736, 0.254_829_592];
        let mut poly = self.like_f(x, coeffs[0]);
        for &c in &coeffs[1..] {
            poly = self.mul(poly, t)?;
            poly = self.offset(poly, c)?;
        }
        poly = self.mul(poly, t)?;
        let a2 = self.mul(a, a)?;
        let na2 = self.neg(a2);
        let ex = self.exp(na2);
        let y = self.mul(poly, ex)?;
        let one = self.like_f(x, 1.0);
        let y = self.sub(one, y)?;
        self.mul(sign, y)
    }

    /// softplus(x) = log(1 + e^x), stable for large |x|.
    pub fn softplus(&mut self, x: Val) -> Built<Val> {
        let zero = self.like_f(x, 0.0);
        let m = self.max(x, zero)?;
        let a = self.abs(x);
        let na = self.neg(a);
        let e = self.exp(na);
        let l = self.log1p(e);
        self.add(m, l)
    }

    /// Softmax along `axis`, in `x`'s element type.
    pub fn softmax(&mut self, x: Val, axis: i64) -> Built<Val> {
        let dims = self.dims(x).to_vec();
        let keep: Vec<i64> = (0..dims.len() as i64).filter(|&i| i != axis).collect();
        let m = self.reduce(x, &[axis], Fold::Max)?;
        let m = self.broadcast(m, &dims, &keep)?;
        let z = self.sub(x, m)?;
        let e = self.exp(z);
        let s = self.reduce(e, &[axis], Fold::Sum)?;
        let s = self.broadcast(s, &dims, &keep)?;
        self.div(e, s)
    }
}

const fn int_min(elem: Elem) -> i64 {
    match elem {
        Elem::I8 => i8::MIN as i64,
        Elem::I16 => i16::MIN as i64,
        Elem::I32 => i32::MIN as i64,
        Elem::I64 => i64::MIN,
        _ => 0,
    }
}

const fn int_max(elem: Elem) -> i64 {
    match elem {
        Elem::I8 => i8::MAX as i64,
        Elem::I16 => i16::MAX as i64,
        Elem::I32 => i32::MAX as i64,
        Elem::I64 => i64::MAX,
        Elem::U4 => 15,
        Elem::U8 => u8::MAX as i64,
        Elem::U16 => u16::MAX as i64,
        Elem::U32 => u32::MAX as i64,
        Elem::Pred => 1,
        _ => i64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_function_prints_its_signature_body_and_return() {
        let mut f = Func::new("main");
        let a = f.param(Ty::new(Elem::F32, &[2, 3]), None);
        let b = f.param(Ty::new(Elem::F32, &[2, 3]), Some(0));
        let c = f.add(a, b).unwrap();
        let s = f.reduce(c, &[1], Fold::Sum).unwrap();
        let text = f.module("m", &[c, s]);
        assert!(text.contains("func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}) -> (tensor<2x3xf32>, tensor<2xf32>)"));
        assert!(text.contains("\"stablehlo.add\"(%arg0, %arg1)"));
        assert!(text.contains("\"stablehlo.reduce\""));
    }

    #[test]
    fn a_mismatched_binary_is_refused_here() {
        let mut f = Func::new("main");
        let a = f.param(Ty::new(Elem::F32, &[2]), None);
        let b = f.param(Ty::new(Elem::Bf16, &[2]), None);
        assert!(f.add(a, b).is_err());
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3F80);
        assert_eq!(bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f16_bits(1.0), 0x3C00);
        assert_eq!(f16_bits(65504.0), 0x7BFF);
    }
}
