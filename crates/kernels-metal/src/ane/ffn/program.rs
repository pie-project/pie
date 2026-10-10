//! The MIL text of the Neural Engine's share of the MLP: for its channels,
//! gate and up over the int8 activations, swiglu, a block Hadamard, int8
//! requantization per token, and down, written as an fp16 partial.

use super::shape::{INT8_PEAK, INT8_UNIT, INTERMEDIATE_BLOCK, INTERMEDIATE_FLOOR, Shape};
use crate::ane::mil::{Body, Input, dims, fp16, half, tensor};
use crate::ane::{CONSTANT_OFFSET, Surface, constant_blob};

#[must_use]
pub fn function_name(rows: u32) -> String {
    format!("ffn{rows}")
}

pub use crate::ane::mil::function_rows;

/// The procedure specialized for `rows` rows.
pub(super) fn function(shape: &Shape, inputs: &[Input<'_>], output: &Surface, rows: u32) -> String {
    let unit = fp16(1.0 / INT8_UNIT);
    let channels = u64::from(shape.ane);
    let r = u64::from(rows);
    let segment = u64::from(shape.segment);
    let mut b = Body::open(inputs, rows);
    let segments = shape.segments() as usize;
    for k in 0..segments {
        b.f16(
            &format!("x{k}_d"),
            segment,
            r,
            format!("dequantize(input = x{k}_t, scale = {unit})"),
        );
    }
    for p in ["g", "u"] {
        for k in 0..segments {
            b.matmul(
                &format!("{p}m{k}"),
                &format!("w{p}{k}"),
                &format!("x{k}_d"),
                channels,
                segment,
                r,
            );
        }
        let total = b.sum(&format!("{p}m"), segments, channels, r);
        b.f16(
            &format!("{p}s"),
            channels,
            r,
            format!("mul(x = {total}, y = s{p}_t)"),
        );
    }
    let c = channels;
    // silu(g) = g * sigmoid(g) = 0.5 * g * (1 + tanh(g / 2)).
    b.f16("gt", c, r, "mul(x = gs, y = tx_t)".into());
    b.f16("gh", c, r, format!("mul(x = gt, y = {})", fp16(0.5)));
    b.f16("th", c, r, "tanh(x = gh)".into());
    b.f16("tp", c, r, format!("add(x = th, y = {})", fp16(1.0)));
    b.f16("silu", c, r, "mul(x = gh, y = tp)".into());
    b.f16("h", c, r, "mul(x = silu, y = us)".into());
    // The block Hadamard is a grouped 1x1 convolution over the channel axis
    // with the rotation as its weight.
    b.line(format!(
        "tensor<fp16, [1, {c}, 1, {r}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {c}, 1, {r}]))"
    ));
    b.line(format!(
        "tensor<fp16, [{c}, {blk}, 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [{c}, {blk}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64({CONSTANT_OFFSET})))]",
        blk = INTERMEDIATE_BLOCK
    ));
    b.line(format!(
        "tensor<fp16, [1, {c}, 1, {r}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32({g}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)",
        g = c / u64::from(INTERMEDIATE_BLOCK)
    ));
    b.f16(
        "hr",
        c,
        r,
        format!(
            "reshape(x = hr4, shape = tensor<int32, [4]>({}))",
            dims(c, r)
        ),
    );
    b.f16("habs", c, r, "abs(x = hr)".into());
    b.f16(
        "peak",
        1,
        r,
        "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))".into(),
    );
    b.f16(
        "floor",
        1,
        r,
        format!("maximum(x = peak, y = {})", fp16(INTERMEDIATE_FLOOR)),
    );
    b.f16(
        "inverse",
        1,
        r,
        format!("real_div(x = {}, y = floor)", fp16(INT8_PEAK)),
    );
    b.f16("hs", c, r, "mul(x = hr, y = inverse)".into());
    b.line(format!(
        "{} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))",
        tensor("int8", c, r)
    ));
    b.f16(
        "hd",
        c,
        r,
        format!("dequantize(input = hq, scale = {unit})"),
    );
    b.f16(
        "hscale",
        1,
        r,
        format!("mul(x = floor, y = {})", fp16(INT8_UNIT / INT8_PEAK)),
    );
    let hidden = u64::from(shape.hidden);
    let mut begin = 0u32;
    for (i, &width) in shape.down.iter().enumerate() {
        let slice = format!("hd{i}");
        b.f16(
            &slice,
            u64::from(width),
            r,
            format!(
                "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>({}))",
                dims(u64::from(width), r)
            ),
        );
        b.matmul(
            &format!("dm{i}"),
            &format!("wd{i}"),
            &slice,
            hidden,
            u64::from(width),
            r,
        );
        begin += width;
    }
    let total = b.sum("dm", shape.down.len(), hidden, r);
    b.f16("ds", hidden, r, format!("mul(x = {total}, y = sd_t)"));
    b.f16(
        "yt",
        hidden + 1,
        r,
        "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))".into(),
    );
    b.line(format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
        output.buffer_type(shape.hidden + 1, rows),
        output.strides(shape.hidden + 1)
    ));
    b.close(&function_name(rows))
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
