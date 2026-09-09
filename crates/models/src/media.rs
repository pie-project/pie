use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Grid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl Grid {
    #[must_use]
    pub const fn still(h: u32, w: u32) -> Grid {
        Grid { t: 1, h, w }
    }

    #[must_use]
    pub const fn cells(&self) -> u32 {
        self.t * self.h * self.w
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Budget {
    #[default]
    Still,
    VideoFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delimiters {
    pub prefix: &'static str,
    pub placeholder: &'static str,
    pub suffix: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8 {
    pub h: u32,
    pub w: u32,
    pub data: Vec<u8>,
}

impl Rgb8 {
    pub fn new(h: u32, w: u32, data: Vec<u8>) -> Result<Rgb8> {
        if w == 0 || h == 0 {
            return Err(Fault::Empty(format!(
                "a frame of {h} x {w} pixels occupies no rows"
            )));
        }
        let owed = h as usize * w as usize * 3;
        if data.len() != owed {
            return Err(Fault::Decode(format!(
                "a {h} x {w} RGB frame is {owed} bytes and {} arrived",
                data.len()
            )));
        }
        Ok(Rgb8 { h, w, data })
    }
}

pub type Resample = fn(&Rgb8, u32, u32) -> Rgb8;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EncodedSpan {
    pub token_count: u32,
    pub position_span: u32,
    pub grid: Grid,
    pub patch_grid: Grid,
    pub uses_mrope: bool,
    pub payload: Vec<f32>,
    pub rows: u32,
    pub positions: Vec<u32>,
    pub embed_rows: Vec<i32>,
    pub embed_weights: Vec<f32>,
    pub prefix: Vec<u32>,
    pub placeholder: u32,
    pub suffix: Vec<u32>,
}

impl EncodedSpan {
    #[must_use]
    pub fn tokens(&self) -> Vec<u32> {
        let mut out =
            Vec::with_capacity(self.prefix.len() + self.token_count as usize + self.suffix.len());
        out.extend_from_slice(&self.prefix);
        out.extend(std::iter::repeat_n(
            self.placeholder,
            self.token_count as usize,
        ));
        out.extend_from_slice(&self.suffix);
        out
    }

    pub fn spell_with(&mut self, prefix: Vec<u32>, placeholder: u32, suffix: Vec<u32>) {
        self.prefix = prefix;
        self.placeholder = placeholder;
        self.suffix = suffix;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    NoVisionFrontEnd { model: String, arch: String },
    NoAudioFrontEnd { model: String, arch: String },
    Decode(String),
    Empty(String),
}

impl Fault {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Fault::NoVisionFrontEnd { .. } => "NoVisionFrontEnd",
            Fault::NoAudioFrontEnd { .. } => "NoAudioFrontEnd",
            Fault::Decode(_) => "Decode",
            Fault::Empty(_) => "Empty",
        }
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::NoVisionFrontEnd { model, arch } => write!(
                f,
                "NoVisionFrontEnd: model '{model}' (arch '{arch}') has no vision front-end"
            ),
            Fault::NoAudioFrontEnd { model, arch } => write!(
                f,
                "NoAudioFrontEnd: model '{model}' (arch '{arch}') has no audio front-end"
            ),
            Fault::Decode(why) => write!(f, "Decode: {why}"),
            Fault::Empty(why) => write!(f, "Empty: {why}"),
        }
    }
}

impl std::error::Error for Fault {}

pub type Result<T> = std::result::Result<T, Fault>;

pub trait VisionFrontEnd: Send + Sync {
    fn arch(&self) -> &'static str;

    fn delimiters(&self) -> Delimiters;

    fn encode(&self, src: &Rgb8, budget: Budget, resample: Resample) -> Result<EncodedSpan>;
}

pub trait AudioFrontEnd: Send + Sync {
    fn arch(&self) -> &'static str;

    fn delimiters(&self) -> Delimiters;

    fn encode_audio(&self, bytes: &[u8]) -> Result<EncodedSpan>;
}

#[must_use]
pub fn vision_front_end(arch: &str) -> Option<Box<dyn VisionFrontEnd>> {
    match arch {
        crate::qwen_3::media::ARCH => Some(Box::new(crate::qwen_3::media::Qwen35Vision::new())),
        crate::qwen_4::ARCH => Some(Box::new(crate::qwen_3::media::Qwen35Vision::new())),
        crate::gemma_4::media::ARCH => Some(Box::new(crate::gemma_4::media::Gemma4Vision::new())),
        crate::glm_5_next::media::ARCH => {
            Some(Box::new(crate::glm_5_next::media::Glm5Vision::new()))
        }
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StubFrontEnd {
    pub arch: &'static str,
    pub token_count: u32,
    pub delimiters: Delimiters,
}

impl StubFrontEnd {
    #[must_use]
    pub const fn new(arch: &'static str, token_count: u32) -> StubFrontEnd {
        StubFrontEnd {
            arch,
            token_count,
            delimiters: Delimiters {
                prefix: "<|vision_start|>",
                placeholder: "<|image_pad|>",
                suffix: "<|vision_end|>",
            },
        }
    }
}

impl VisionFrontEnd for StubFrontEnd {
    fn arch(&self) -> &'static str {
        self.arch
    }

    fn delimiters(&self) -> Delimiters {
        self.delimiters
    }

    fn encode(&self, src: &Rgb8, budget: Budget, _resample: Resample) -> Result<EncodedSpan> {
        let rows = match budget {
            Budget::Still => self.token_count,
            Budget::VideoFrame => self.token_count.div_ceil(2),
        };
        if rows == 0 {
            return Err(Fault::Empty("stub front-end: zero-row span".into()));
        }
        Ok(EncodedSpan {
            token_count: rows,
            position_span: rows,
            grid: Grid::still(1, rows),
            patch_grid: Grid::still(1, rows),
            uses_mrope: false,
            payload: src.data.iter().map(|&b| f32::from(b)).collect(),
            rows,
            positions: Vec::new(),
            embed_rows: Vec::new(),
            embed_weights: Vec::new(),
            prefix: Vec::new(),
            placeholder: 0,
            suffix: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixels(bytes: &[u8]) -> Rgb8 {
        let mut data = bytes.to_vec();
        data.resize(3, 0);
        Rgb8::new(1, 1, data).expect("one pixel")
    }

    fn spelled(rows: u32, pad: u32) -> EncodedSpan {
        let mut span = StubFrontEnd::new("stub", rows)
            .encode(&pixels(b"abc"), Budget::Still, |src, _, _| src.clone())
            .expect("stub encodes");
        span.spell_with(vec![7], pad, vec![8]);
        span
    }

    #[test]
    fn media_every_case() {
        tokens_are_prefix_then_the_run_then_suffix();
        degenerate_pixels_are_refused_by_name();
        the_two_vision_archs_spell_their_runs_differently();
    }

    fn tokens_are_prefix_then_the_run_then_suffix() {
        let span = spelled(4, 99);
        assert_eq!(span.tokens(), vec![7, 99, 99, 99, 99, 8]);
        assert_eq!(
            span.tokens().len(),
            span.prefix.len() + span.token_count as usize + span.suffix.len()
        );
    }

    fn degenerate_pixels_are_refused_by_name() {
        assert_eq!(
            Rgb8::new(0, 4, Vec::new()).expect_err("zero side").name(),
            "Empty"
        );
        assert_eq!(
            Rgb8::new(2, 2, vec![0; 5])
                .expect_err("wrong length")
                .name(),
            "Decode"
        );
    }

    fn the_two_vision_archs_spell_their_runs_differently() {
        let qwen = vision_front_end("qwen3_5")
            .expect("qwen has a tower")
            .delimiters();
        let gemma = vision_front_end("gemma4")
            .expect("gemma has a tower")
            .delimiters();
        assert_eq!(qwen.placeholder, "<|image_pad|>");
        assert_eq!(qwen.prefix, "<|vision_start|>");
        assert_eq!(gemma.placeholder, "<|image|>");
        assert_ne!(qwen.placeholder, gemma.placeholder);
        assert!(vision_front_end("deepseek_v4").is_none());
    }
}
