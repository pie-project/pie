//! The MIL text of the Neural Engine's share of the MLP: for its channels,
//! gate and up over the int8 activations, swiglu, a block Hadamard, int8
//! requantization per token, and down, written as an fp16 partial.

use super::memory::Input;
use super::shape::{
    INT8_PEAK, INT8_UNIT, INTERMEDIATE_BLOCK, INTERMEDIATE_FLOOR, MAX_ROWS, MIN_ROWS, SEGMENT,
    STEP, Shape,
};
use crate::ane::{CONSTANT_OFFSET, Surface, constant_blob};

fn fp16(value: f64) -> String {
    format!("fp16({})", hex_float(value))
}

fn hex_float(value: f64) -> String {
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

fn dims(rows: u64, width: u64) -> String {
    format!("[1, 1, {rows}, {width}]")
}

fn tensor(kind: &str, rows: u64, width: u64) -> String {
    format!("tensor<{kind}, {}>", dims(rows, width))
}

#[must_use]
pub fn function_name(rows: u32) -> String {
    format!("ffn{rows}")
}

/// The row counts the program has a procedure for.
#[must_use]
pub fn function_rows() -> Vec<u32> {
    (MIN_ROWS..=MAX_ROWS).step_by(STEP as usize).collect()
}

/// The procedure specialized for `rows` rows.
pub(super) fn function(shape: &Shape, inputs: &[Input<'_>], output: &Surface, rows: u32) -> String {
    let unit = fp16(1.0 / INT8_UNIT);
    let channels = u64::from(shape.ane);
    let r = u64::from(rows);
    let mut parameters = Vec::new();
    let mut body = String::new();
    let mut line = |text: String| {
        body.push_str("        ");
        body.push_str(&text);
        body.push_str(";\n");
    };
    for input in inputs {
        let width = if input.width == 0 { rows } else { input.width };
        parameters.push(format!(
            "{} {}",
            input.surfaces[0].buffer_type(input.rows, width),
            input.name
        ));
        line(format!(
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
    let f16 = |name: &str, height: u64, width: u64, expression: String| {
        format!("{} {name} = {expression}", tensor("fp16", height, width))
    };
    let matmul = |line: &mut dyn FnMut(String),
                  name: &str,
                  weights: &str,
                  values: &str,
                  height: u64,
                  width: u64| {
        line(f16(
            &format!("{weights}_d"),
            height,
            width,
            format!("dequantize(input = {weights}_t, scale = {unit})"),
        ));
        line(format!(
            "{} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {weights}_d, y = {values})",
            tensor("fp16", height, r)
        ));
    };
    let sum = |line: &mut dyn FnMut(String), prefix: &str, terms: usize, height: u64| -> String {
        let mut total = format!("{prefix}0");
        for term in 1..terms {
            let next = format!("{prefix}_sum{term}");
            line(f16(
                &next,
                height,
                r,
                format!("add(x = {total}, y = {prefix}{term})"),
            ));
            total = next;
        }
        total
    };
    let segments = shape.segments() as usize;
    for k in 0..segments {
        line(f16(
            &format!("x{k}_d"),
            u64::from(SEGMENT),
            r,
            format!("dequantize(input = x{k}_t, scale = {unit})"),
        ));
    }
    for p in ["g", "u"] {
        for k in 0..segments {
            matmul(
                &mut line,
                &format!("{p}m{k}"),
                &format!("w{p}{k}"),
                &format!("x{k}_d"),
                channels,
                u64::from(SEGMENT),
            );
        }
        let total = sum(&mut line, &format!("{p}m"), segments, channels);
        line(f16(
            &format!("{p}s"),
            channels,
            r,
            format!("mul(x = {total}, y = s{p}_t)"),
        ));
    }
    let c = channels;
    // silu(g) = g * sigmoid(g) = 0.5 * g * (1 + tanh(g / 2)).
    line(f16("gt", c, r, "mul(x = gs, y = tx_t)".into()));
    line(f16("gh", c, r, format!("mul(x = gt, y = {})", fp16(0.5))));
    line(f16("th", c, r, "tanh(x = gh)".into()));
    line(f16("tp", c, r, format!("add(x = th, y = {})", fp16(1.0))));
    line(f16("silu", c, r, "mul(x = gh, y = tp)".into()));
    line(f16("h", c, r, "mul(x = silu, y = us)".into()));
    // The block Hadamard is a grouped 1x1 convolution over the channel axis
    // with the rotation as its weight.
    line(format!(
        "tensor<fp16, [1, {c}, 1, {r}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {c}, 1, {r}]))"
    ));
    line(format!(
        "tensor<fp16, [{c}, {b}, 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [{c}, {b}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64({CONSTANT_OFFSET})))]",
        b = INTERMEDIATE_BLOCK
    ));
    line(format!(
        "tensor<fp16, [1, {c}, 1, {r}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32({g}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)",
        g = c / u64::from(INTERMEDIATE_BLOCK)
    ));
    line(f16(
        "hr",
        c,
        r,
        format!(
            "reshape(x = hr4, shape = tensor<int32, [4]>({}))",
            dims(c, r)
        ),
    ));
    line(f16("habs", c, r, "abs(x = hr)".into()));
    line(f16(
        "peak",
        1,
        r,
        "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))".into(),
    ));
    line(f16(
        "floor",
        1,
        r,
        format!("maximum(x = peak, y = {})", fp16(INTERMEDIATE_FLOOR)),
    ));
    line(f16(
        "inverse",
        1,
        r,
        format!("real_div(x = {}, y = floor)", fp16(INT8_PEAK)),
    ));
    line(f16("hs", c, r, "mul(x = hr, y = inverse)".into()));
    line(format!(
        "{} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))",
        tensor("int8", c, r)
    ));
    line(f16(
        "hd",
        c,
        r,
        format!("dequantize(input = hq, scale = {unit})"),
    ));
    line(f16(
        "hscale",
        1,
        r,
        format!("mul(x = floor, y = {})", fp16(INT8_UNIT / INT8_PEAK)),
    ));
    let hidden = u64::from(shape.hidden);
    let mut begin = 0u32;
    for (i, &width) in shape.down.iter().enumerate() {
        let slice = format!("hd{i}");
        line(f16(
            &slice,
            u64::from(width),
            r,
            format!(
                "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>({}))",
                dims(u64::from(width), r)
            ),
        ));
        matmul(
            &mut line,
            &format!("dm{i}"),
            &format!("wd{i}"),
            &slice,
            hidden,
            u64::from(width),
        );
        begin += width;
    }
    let total = sum(&mut line, "dm", shape.down.len(), hidden);
    line(f16("ds", hidden, r, format!("mul(x = {total}, y = sd_t)")));
    line(f16(
        "yt",
        hidden + 1,
        r,
        "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))".into(),
    ));
    line(format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
        output.buffer_type(shape.hidden + 1, rows),
        output.strides(shape.hidden + 1)
    ));
    format!(
        "    func {}<ios18>({}) {{\n{body}    }} -> (y);\n",
        function_name(rows),
        parameters.join(", ")
    )
}

/// The signs of the randomized Hadamard, one per channel of a block; the GPU
/// side rotates with the same ones.
#[must_use]
pub fn signs() -> Vec<f32> {
    let mut state = 0x2026_0930_u64;
    (0..INTERMEDIATE_BLOCK)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 1 { -1.0 } else { 1.0 }
        })
        .collect()
}

fn half(value: f32) -> u16 {
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

/// The rotation's weights as the program's `weights.bin`: for each channel,
/// its block's signed Hadamard row, normalized.
pub(super) fn rotation_blob(channels: u32, signs: &[f32]) -> Vec<u8> {
    let block = INTERMEDIATE_BLOCK as usize;
    let norm = 1.0 / (block as f32).sqrt();
    let mut values = vec![0u16; channels as usize * block];
    for channel in 0..channels as usize {
        let output = channel % block;
        for input in 0..block {
            let hadamard = if (output & input).count_ones() & 1 == 1 {
                -1.0
            } else {
                1.0
            };
            values[channel * block + input] = half(signs[output] * hadamard * norm);
        }
    }
    constant_blob(&values)
}
