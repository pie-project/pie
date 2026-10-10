//! The MIL text the Neural Engine programs are written in: literals, tensor
//! types, and the `tensor_buffer` prologue every procedure opens with.

use super::Surface;

#[must_use]
pub fn fp16(value: f64) -> String {
    format!("fp16({})", hex_float(value))
}

#[must_use]
pub fn hex_float(value: f64) -> String {
    if value == 0.0 {
        return "0x0p+0".into();
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 1 { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = bits & ((1u64 << 52) - 1);
    let mut digits = format!("{mantissa:013x}");
    while digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() {
        format!("{sign}0x1p{exponent:+}")
    } else {
        format!("{sign}0x1.{digits}p{exponent:+}")
    }
}

#[must_use]
pub fn dims(rows: u64, width: u64) -> String {
    format!("[1, 1, {rows}, {width}]")
}

#[must_use]
pub fn tensor(kind: &str, rows: u64, width: u64) -> String {
    format!("tensor<{kind}, {}>", dims(rows, width))
}

/// `fp16` of `value`, rounded to nearest even.
#[must_use]
pub fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if value == 0.0 || exponent <= 0 {
        return sign;
    }
    let mantissa = bits & 0x7f_ffff;
    let rounded = (mantissa + 0x1000) >> 13;
    let (exponent, mantissa) = if rounded == 0x400 {
        (exponent + 1, 0)
    } else {
        (exponent, rounded)
    };
    sign | ((exponent as u16) << 10) | mantissa as u16
}

/// One input of a procedure: its name, the surface behind it in each weight
/// set, and its tensor shape. A `width` of zero means the row count the
/// procedure is specialized for.
pub struct Input<'a> {
    pub name: String,
    pub surfaces: [&'a Surface; 2],
    pub rows: u32,
    pub width: u32,
}

/// A procedure's body as it is written: one line per statement.
pub struct Body {
    parameters: Vec<String>,
    text: String,
}

impl Body {
    /// Opens the procedure: every input declared as a `tensor_buffer`
    /// parameter and read into a tensor named `{input}_t`.
    #[must_use]
    pub fn open(inputs: &[Input<'_>], rows: u32) -> Body {
        let mut body = Body {
            parameters: Vec::new(),
            text: String::new(),
        };
        for input in inputs {
            let width = if input.width == 0 { rows } else { input.width };
            body.parameters.push(format!(
                "{} {}",
                input.surfaces[0].buffer_type(input.rows, width),
                input.name
            ));
            body.line(format!(
                "{} {}_t = tensor_buffer_to_tensor<ios17>(input = {})",
                tensor(
                    input.surfaces[0].mil_type(),
                    u64::from(input.rows),
                    u64::from(width)
                ),
                input.name,
                input.name
            ));
        }
        body
    }

    pub fn line(&mut self, text: String) {
        self.text.push_str("        ");
        self.text.push_str(&text);
        self.text.push_str(";\n");
    }

    /// `name = expression`, an fp16 tensor of `height x width`.
    pub fn f16(&mut self, name: &str, height: u64, width: u64, expression: String) {
        self.line(format!(
            "{} {name} = {expression}",
            tensor("fp16", height, width)
        ));
    }

    /// Dequantizes `{weights}_t` by `1/128` and multiplies it by `values`:
    /// `name`, an fp16 `height x rows` tensor.
    pub fn matmul(
        &mut self,
        name: &str,
        weights: &str,
        values: &str,
        height: u64,
        width: u64,
        rows: u64,
    ) {
        self.f16(
            &format!("{weights}_d"),
            height,
            width,
            format!(
                "dequantize(input = {weights}_t, scale = {})",
                fp16(1.0 / super::ffn::INT8_UNIT)
            ),
        );
        self.line(format!(
            "{} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {weights}_d, y = {values})",
            tensor("fp16", height, rows)
        ));
    }

    /// Adds `{prefix}0 .. {prefix}{terms-1}` and names the total.
    pub fn sum(&mut self, prefix: &str, terms: usize, height: u64, rows: u64) -> String {
        let mut total = format!("{prefix}0");
        for term in 1..terms {
            let next = format!("{prefix}_sum{term}");
            self.f16(
                &next,
                height,
                rows,
                format!("add(x = {total}, y = {prefix}{term})"),
            );
            total = next;
        }
        total
    }

    /// Closes the procedure `name`, returning `y`.
    #[must_use]
    pub fn close(self, name: &str) -> String {
        format!(
            "    func {name}<ios18>({}) {{\n{}    }} -> (y);\n",
            self.parameters.join(", "),
            self.text
        )
    }
}

/// The row counts every program has a procedure for.
#[must_use]
pub fn function_rows() -> Vec<u32> {
    (super::ffn::MIN_ROWS..=super::ffn::MAX_ROWS)
        .step_by(super::ffn::STEP as usize)
        .collect()
}

/// The widest hidden-axis segment the Neural Engine takes at once.
pub const SEGMENT_MAX: u32 = 2560;

/// How a contraction of `k` splits into segments the program multiplies
/// one at a time: the widest divisor of `k` up to [`SEGMENT_MAX`] that is
/// whole input blocks, if `k` has one and is not past what a program holds.
#[must_use]
pub fn segment_of(k: u32) -> Option<u32> {
    if k == 0 || k > 8192 {
        return None;
    }
    (1..=SEGMENT_MAX / super::ffn::INPUT_BLOCK)
        .rev()
        .map(|blocks| blocks * super::ffn::INPUT_BLOCK)
        .find(|segment| k.is_multiple_of(*segment))
}
