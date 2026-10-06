pub const SEGMENT: u32 = 2560;
pub const INPUT_BLOCK: u32 = 128;
pub const INTERMEDIATE_BLOCK: u32 = 512;
pub const UNIT: u32 = 512;
pub const MIN_ROWS: u32 = 512;
pub const MAX_ROWS: u32 = 2048;
pub const STEP: u32 = 128;
pub const INT8_PEAK: f64 = 127.0;
pub const INT8_UNIT: f64 = 128.0;
pub(super) const INTERMEDIATE_FLOOR: f64 = 1.0 / 512.0;

#[derive(Clone, Debug)]
pub struct Shape {
    pub hidden: u32,
    pub intermediate: u32,
    pub gpu: u32,
    pub ane: u32,
    pub down: Vec<u32>,
}

impl Shape {
    pub fn new(hidden: u32, intermediate: u32, units: u32) -> Result<Shape, String> {
        if !hidden.is_multiple_of(SEGMENT) || hidden / INPUT_BLOCK / 8 > 8 {
            return Err(format!(
                "hidden size {hidden} is not whole {SEGMENT}-channel segments"
            ));
        }
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
            intermediate,
            gpu: intermediate - ane,
            ane,
            down,
        })
    }

    #[must_use]
    pub fn segments(&self) -> u32 {
        self.hidden / SEGMENT
    }
}
