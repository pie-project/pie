//! A small text builder for CSL source.
//!
//! CSL is emitted as lines with explicit indentation; the builder only keeps
//! nesting honest and gives names to the few constructs kernels use:
//! functions, `for (@range(..))` loops, and DSD declarations.

use std::fmt::Write;

/// One argument of a kernel call: what the table-driven program needs to
/// know about it (where a pointer points, what a number is), rendered to
/// CSL text for the phase programs.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// A many-item pointer to an exported buffer.
    Ptr(crate::program::Buf),
    /// A pointer to a scratch array (`name`, element type).
    Scratch(String, &'static str),
    Int(i64),
    Bool(bool),
    Float(f32),
    /// The PE's index in its rectangle times a constant (a block base).
    PeTimes(i64),
    /// Word `index` of an exported i32 buffer (a lane header word).
    Word(String, u32),
    /// A pointer to the prelude's one-element dummy of this element type
    /// (an absent optional array).
    Dummy(&'static str),
    /// Raw CSL (a header word, a comptime expression).
    Expr(String),
}

impl Arg {
    /// The comma-separated expressions of an assembled argument list
    /// (header ranges), each its own raw argument.
    pub fn exprs(list: &str) -> Vec<Arg> {
        list.split(", ").map(|e| Arg::Expr(e.to_string())).collect()
    }

    pub fn render(&self) -> String {
        match self {
            Arg::Ptr(b) => b.ptr(),
            Arg::Scratch(name, elem) => format!("@ptrcast([*]{elem}, &{name})"),
            Arg::Int(v) => v.to_string(),
            Arg::Bool(v) => v.to_string(),
            Arg::Float(v) => f32_lit(*v),
            Arg::PeTimes(n) => format!("@as(i32, pe_id) * {n}"),
            Arg::Word(name, i) => format!("{name}[{i}]"),
            Arg::Dummy(elem) => {
                let dummy = if *elem == "f32" { "k_dummy_f32" } else { "k_dummy_u32" };
                format!("@ptrcast([*]{elem}, &{dummy})")
            }
            Arg::Expr(e) => e.clone(),
        }
    }
}

/// A kernel call, kept structured beside its rendered line.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub kernel: String,
    pub args: Vec<Arg>,
}

impl Call {
    pub fn render(&self) -> String {
        let args: Vec<String> = self.args.iter().map(Arg::render).collect();
        format!("{}({});", self.kernel, args.join(", "))
    }
}

/// A block of statements at one indentation depth.
#[derive(Debug, Default, Clone)]
pub struct Block {
    lines: Vec<String>,
    depth: usize,
    /// The kernel calls among the lines, by line index.
    calls: Vec<(usize, Call)>,
}

impl Block {
    pub fn new(depth: usize) -> Self {
        Block {
            lines: Vec::new(),
            depth,
            calls: Vec::new(),
        }
    }

    /// One statement; the trailing `;` is the caller's.
    pub fn line(&mut self, s: impl AsRef<str>) -> &mut Self {
        let mut l = "  ".repeat(self.depth);
        l.push_str(s.as_ref());
        self.lines.push(l);
        self
    }

    /// A kernel call: rendered as a line and kept structured.
    pub fn call(&mut self, call: Call) -> &mut Self {
        let text = call.render();
        self.calls.push((self.lines.len(), call));
        self.line(text)
    }

    /// The kernel calls in this block, in order.
    pub fn calls(&self) -> impl Iterator<Item = &Call> {
        self.calls.iter().map(|(_, c)| c)
    }

    /// A nested block between `open` and `close`, e.g. a loop.
    pub fn nest(&mut self, open: &str, close: &str, body: impl FnOnce(&mut Block)) -> &mut Self {
        self.line(open);
        let mut inner = Block::new(self.depth + 1);
        body(&mut inner);
        self.append(inner);
        self.line(close);
        self
    }

    /// `for (@range(i16, n)) |var| { .. }`.
    pub fn for_range(
        &mut self,
        var: &str,
        n: impl std::fmt::Display,
        body: impl FnOnce(&mut Block),
    ) -> &mut Self {
        self.nest(&format!("for (@range(i16, {n})) |{var}| {{"), "}", body)
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Statements in the block.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Appends `other`'s lines (and calls) after this block's.
    pub fn append(&mut self, other: Block) {
        let base = self.lines.len();
        self.calls
            .extend(other.calls.into_iter().map(|(i, c)| (base + i, c)));
        self.lines.extend(other.lines);
    }

    /// Rewrites every occurrence of `from` in the block's lines to `to`.
    pub fn replace(&mut self, from: &str, to: &str) {
        for l in &mut self.lines {
            if l.contains(from) {
                *l = l.replace(from, to);
            }
        }
    }

    pub fn render(&self, out: &mut String) {
        for l in &self.lines {
            out.push_str(l);
            out.push('\n');
        }
    }
}

/// A CSL `fn`.
#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    /// `name: type` pairs.
    pub params: Vec<(String, String)>,
    pub ret: String,
    pub body: Block,
}

impl Func {
    pub fn new(name: impl Into<String>) -> Self {
        Func {
            name: name.into(),
            params: Vec::new(),
            ret: "void".into(),
            body: Block::new(1),
        }
    }

    pub fn param(mut self, name: &str, ty: &str) -> Self {
        self.params.push((name.into(), ty.into()));
        self
    }

    pub fn render(&self, out: &mut String) {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|(n, t)| format!("{n}: {t}"))
            .collect();
        let _ = writeln!(
            out,
            "fn {}({}) {} {{",
            self.name,
            params.join(", "),
            self.ret
        );
        self.body.render(out);
        out.push_str("}\n");
    }
}

/// A `mem1d_dsd` over an f32 array: `tensor_access = |i|{extent} -> arr[base + i * stride]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dsd1 {
    pub array: String,
    pub extent: String,
    pub base: String,
    pub stride: String,
}

impl Dsd1 {
    /// `arr[i]` for `i < extent`.
    pub fn contiguous(array: impl Into<String>, extent: impl std::fmt::Display) -> Self {
        Dsd1 {
            array: array.into(),
            extent: extent.to_string(),
            base: "0".into(),
            stride: "1".into(),
        }
    }

    pub fn strided(
        array: impl Into<String>,
        extent: impl std::fmt::Display,
        base: impl std::fmt::Display,
        stride: impl std::fmt::Display,
    ) -> Self {
        Dsd1 {
            array: array.into(),
            extent: extent.to_string(),
            base: base.to_string(),
            stride: stride.to_string(),
        }
    }

    /// The `@get_dsd(..)` expression.
    pub fn expr(&self) -> String {
        let index = match (self.base.as_str(), self.stride.as_str()) {
            ("0", "1") => "i".to_string(),
            ("0", s) => format!("i * {s}"),
            (b, "1") => format!("{b} + i"),
            (b, s) => format!("{b} + i * {s}"),
        };
        format!(
            "@get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{}}} -> {}[{index}] }})",
            self.extent, self.array
        )
    }
}

/// Formats an `f32` as a CSL literal that parses back to the same value.
pub fn f32_lit(v: f32) -> String {
    if v.is_nan() {
        "(0.0 / 0.0)".into()
    } else if v.is_infinite() {
        if v > 0.0 {
            "(1.0 / 0.0)".into()
        } else {
            "(-1.0 / 0.0)".into()
        }
    } else if v == v.trunc() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v:e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_function_renders_its_loops_indented() {
        let mut f = Func::new("phase_0").param("n", "u32");
        f.body.line("var acc: f32 = 0.0;");
        f.body.for_range("i", "N", |b| {
            b.line("acc = acc + x[i];");
        });
        let mut out = String::new();
        f.render(&mut out);
        assert_eq!(
            out,
            "fn phase_0(n: u32) void {\n  var acc: f32 = 0.0;\n  for (@range(i16, N)) |i| {\n    acc = acc + x[i];\n  }\n}\n"
        );
    }

    #[test]
    fn a_dsd_names_its_access_pattern() {
        assert_eq!(
            Dsd1::contiguous("y", 4).expr(),
            "@get_dsd(mem1d_dsd, .{ .tensor_access = |i|{4} -> y[i] })"
        );
        assert_eq!(
            Dsd1::strided("A", "M", "r * N", "1").expr(),
            "@get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[r * N + i] })"
        );
        assert_eq!(
            Dsd1::strided("A", "M", 0, "N").expr(),
            "@get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i * N] })"
        );
    }

    #[test]
    fn float_literals_round_trip() {
        for v in [0.0f32, 1.0, -2.5, 1e-6, 1.234_567_9, 1e30, -7.25e-3] {
            let text = f32_lit(v);
            let back: f32 = text.trim_matches(|c| c == '(' || c == ')').parse().unwrap();
            assert_eq!(back, v, "{text}");
        }
    }
}
