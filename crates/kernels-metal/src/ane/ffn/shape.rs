//! How an MLP's intermediate channels divide between the two engines, and
//! the sizes the Neural Engine program is written for.

/// The widest hidden-axis segment; a shape takes the widest divisor of its
/// hidden size up to this (see [`crate::ane::mil::segment_of`]).
pub const SEGMENT: u32 = crate::ane::mil::SEGMENT_MAX;
/// The GPU rotates activations in blocks this wide before quantizing them.
pub const INPUT_BLOCK: u32 = 128;
/// The program rotates the swiglu intermediate in blocks this wide.
pub const INTERMEDIATE_BLOCK: u32 = 512;
/// The Neural Engine takes intermediate channels in units this wide.
pub const UNIT: u32 = 512;
/// The program has one procedure per row count from `MIN_ROWS` to
/// `MAX_ROWS` in steps of `STEP`; fewer rows than `MIN_ROWS` stay on the GPU.
pub const MIN_ROWS: u32 = 512;
pub const MAX_ROWS: u32 = 2048;
pub const STEP: u32 = 128;
pub const INT8_PEAK: f64 = 127.0;
pub const INT8_UNIT: f64 = 128.0;
/// The smallest per-token peak the intermediate requantizes against.
pub(super) const INTERMEDIATE_FLOOR: f64 = 1.0 / 512.0;

#[derive(Clone, Debug)]
pub struct Shape {
    pub hidden: u32,
    /// How wide a segment of the hidden axis the program multiplies at once.
    pub segment: u32,
    pub intermediate: u32,
    /// The intermediate channels the GPU keeps: the leading ones.
    pub gpu: u32,
    /// The intermediate channels the Neural Engine takes: the trailing ones.
    pub ane: u32,
    /// The widths the down projection's input splits into, at most one
    /// segment each.
    pub down: Vec<u32>,
}

impl Shape {
    /// `units` of [`UNIT`] channels go to the Neural Engine. The hidden size
    /// must split into segments, and both engines must be left something.
    pub fn new(hidden: u32, intermediate: u32, units: u32) -> Result<Shape, String> {
        let Some(segment) = crate::ane::mil::segment_of(hidden) else {
            return Err(format!(
                "hidden size {hidden} does not split into {INPUT_BLOCK}-block segments of at most {SEGMENT}"
            ));
        };
        if !intermediate.is_multiple_of(UNIT) || units == 0 || units >= intermediate / UNIT {
            return Err(format!(
                "{units} units leave the GPU or the Neural Engine nothing of {intermediate}"
            ));
        }
        let ane = units * UNIT;
        let down = (0..ane)
            .step_by(SEGMENT as usize)
            .map(|begin| SEGMENT.min(ane - begin))
            .collect();
        Ok(Shape {
            hidden,
            segment,
            intermediate,
            gpu: intermediate - ane,
            ane,
            down,
        })
    }

    #[must_use]
    pub fn segments(&self) -> u32 {
        self.hidden / self.segment
    }
}
