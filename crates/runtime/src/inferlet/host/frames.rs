use crate::codec::{mp4, still, wav, y4m};
use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::inferlet::host::pie::inferlet::frames::{AudioFormat, ImageFormat};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

pub enum FrameStore {
    Host(Vec<u8>),
    #[allow(dead_code)]
    Device(DevicePlane),
}

#[allow(dead_code)]
pub struct DevicePlane {
    _private: (),
}

pub struct Frames {
    pub store: FrameStore,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub fps: f32,
}

pub struct Pcm {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub channels: u32,
}

impl std::fmt::Debug for Frames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let store = match &self.store {
            FrameStore::Host(v) => format!("host {} B", v.len()),
            FrameStore::Device(_) => "device".to_string(),
        };
        write!(
            f,
            "Frames({}x{} x{} @{} fps, {store})",
            self.width, self.height, self.count, self.fps
        )
    }
}

impl std::fmt::Debug for Pcm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pcm({} samples, {} Hz x{})",
            self.samples.len(),
            self.rate,
            self.channels
        )
    }
}

#[cfg(feature = "cuda")]
fn nvenc_device() -> usize {
    std::env::var("PIE_NVENC_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

impl Frames {
    pub fn rgb8(&self) -> Result<&[u8], String> {
        match &self.store {
            FrameStore::Host(v) => Ok(v),
            FrameStore::Device(_) => Err("frames: this handle's pixels are on the device and the \
                     device path is not built yet"
                .to_string()),
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let (rgb, width, height) = still::decode(bytes)?;
        Ok(Frames {
            store: FrameStore::Host(rgb),
            width,
            height,
            count: 1,
            fps: 0.0,
        })
    }

    pub fn to_pixels(&self) -> Result<Vec<f32>, String> {
        Ok(self
            .rgb8()?
            .iter()
            .map(|b| f32::from(*b) / 255.0 * 2.0 - 1.0)
            .collect())
    }

    pub fn from_rgb8(
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 || count == 0 {
            return Err(format!(
                "frames.from-rgb8: a handle needs a non-zero extent, got \
                 {width}x{height} x{count}"
            ));
        }
        let want = (width as u64) * (height as u64) * (count as u64) * 3;
        if bytes.len() as u64 != want {
            return Err(format!(
                "frames.from-rgb8: {} bytes for {count} frames of {width}x{height} RGB8 \
                 (expected {want})",
                bytes.len()
            ));
        }
        Ok(Self {
            store: FrameStore::Host(bytes),
            width,
            height,
            count,
            fps,
        })
    }

    pub fn from_pixels(
        values: &[f32],
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 || count == 0 {
            return Err(format!(
                "frames.from-channel: a handle needs a non-zero extent, got \
                 {width}x{height} x{count}"
            ));
        }
        let want = (width as u64) * (height as u64) * (count as u64) * 3;
        if values.len() as u64 != want {
            return Err(format!(
                "frames.from-channel: the cell holds {} f32 and {count} frame(s) of \
                 {width}x{height} RGB is {want}; a `pixels` seam lands one row per \
                 output voxel, three wide",
                values.len()
            ));
        }
        let bytes = values
            .iter()
            .map(|v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok(Self {
            store: FrameStore::Host(bytes),
            width,
            height,
            count,
            fps,
        })
    }

    pub fn encode(&self, format: ImageFormat) -> Result<Vec<u8>, String> {
        let rgb = self.rgb8()?;
        let still_only = |what: &str| {
            format!(
                "{what} is a still format and this handle has {} frames; \
                 use y4m or mp4-h264",
                self.count
            )
        };
        match format {
            ImageFormat::Png => {
                if self.count != 1 {
                    return Err(still_only("png"));
                }
                still::png(rgb, self.width, self.height)
            }
            ImageFormat::Jpeg => {
                if self.count != 1 {
                    return Err(still_only("jpeg"));
                }
                still::jpeg(rgb, self.width, self.height)
            }
            ImageFormat::Webp => {
                if self.count != 1 {
                    return Err(still_only("webp"));
                }
                still::webp(rgb, self.width, self.height)
            }
            ImageFormat::RawRgb8 => Ok(rgb.to_vec()),
            ImageFormat::Y4m => y4m::encode(rgb, self.width, self.height, self.count, self.fps),
            ImageFormat::Mp4H264 => self.mp4_h264(rgb),
        }
    }

    #[cfg(feature = "cuda")]
    fn mp4_h264(&self, rgb: &[u8]) -> Result<Vec<u8>, String> {
        let pictures = crate::codec::nvenc::encode_h264(
            rgb,
            self.width,
            self.height,
            self.count,
            self.fps,
            nvenc_device(),
        )?;
        mp4::mux_annexb(self.width, self.height, self.fps, &pictures)
    }

    #[cfg(not(feature = "cuda"))]
    fn mp4_h264(&self, _rgb: &[u8]) -> Result<Vec<u8>, String> {
        let _ = &mp4::mux_annexb;
        Err(
            "mp4-h264 needs the NVIDIA H.264 encoder (NVENC), and this runtime \
             was built without the CUDA shell. Encode `y4m` for an \
             uncompressed clip, or rebuild with `--features cuda`."
                .to_string(),
        )
    }
}

impl Pcm {
    pub fn from_f32(samples: Vec<f32>, rate: u32, channels: u32) -> Result<Self, String> {
        if rate == 0 || channels == 0 {
            return Err(format!(
                "pcm.from-f32: a buffer needs a rate and a channel count, got \
                 {rate} Hz x{channels}"
            ));
        }
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(format!(
                "pcm.from-f32: {} samples do not divide into {channels} channels",
                samples.len()
            ));
        }
        Ok(Self {
            samples,
            rate,
            channels,
        })
    }

    pub fn encode(&self, format: AudioFormat) -> Vec<u8> {
        match format {
            AudioFormat::Wav => wav::encode(&self.samples, self.rate, self.channels),
            AudioFormat::RawF32 => wav::raw_f32(&self.samples),
        }
    }
}

pub fn image_extension(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
        ImageFormat::Webp => "webp",
        ImageFormat::RawRgb8 => "rgb",
        ImageFormat::Y4m => "y4m",
        ImageFormat::Mp4H264 => "mp4",
    }
}

pub fn audio_extension(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::RawF32 => "f32",
    }
}

impl pie::inferlet::frames::Host for ProcessCtx {}

impl pie::inferlet::frames::HostFrames for ProcessCtx {
    async fn from_rgb8(
        &mut self,
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Result<Resource<Frames>, String>> {
        match Frames::from_rgb8(bytes, width, height, count, fps) {
            Ok(f) => Ok(Ok(self.ctx().table.push(f)?)),
            Err(e) => Ok(Err(e)),
        }
    }

    async fn from_channel(
        &mut self,
        ch: Resource<super::forward::Channel>,
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Result<Resource<Frames>, String>> {
        let cell = match super::forward::materialize_channel_blocking(
            self,
            ch,
            super::forward::ChannelReadMode::Take,
        )
        .await?
        {
            Ok(bytes) => bytes,
            Err(why) => return Ok(Err(why)),
        };
        if !cell.len().is_multiple_of(4) {
            return Ok(Err(format!(
                "frames.from-channel: the cell is {} bytes, which is not whole f32 rows; \
                 a `pixels` seam's channel is f32",
                cell.len()
            )));
        }
        let values: Vec<f32> = cell
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        match Frames::from_pixels(&values, width, height, count, fps) {
            Ok(f) => Ok(Ok(self.ctx().table.push(f)?)),
            Err(e) => Ok(Err(e)),
        }
    }

    async fn decode(&mut self, bytes: Vec<u8>) -> Result<Result<Resource<Frames>, String>> {
        match Frames::decode(&bytes) {
            Ok(f) => Ok(Ok(self.ctx().table.push(f)?)),
            Err(why) => Ok(Err(why)),
        }
    }

    async fn to_channel(
        &mut self,
        this: Resource<Frames>,
        ch: Resource<super::forward::Channel>,
    ) -> Result<Result<(), String>> {
        let values = match self.ctx().table.get(&this)?.to_pixels() {
            Ok(values) => values,
            Err(why) => return Ok(Err(why)),
        };
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let cell = self.ctx().table.get(&ch)?.cell.clone();
        let result = cell
            .lock()
            .unwrap()
            .put_ref(&bytes)
            .map_err(|error| format!("frames.to-channel: {error}"));
        Ok(result)
    }

    async fn width(&mut self, this: Resource<Frames>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.width)
    }

    async fn height(&mut self, this: Resource<Frames>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.height)
    }

    async fn count(&mut self, this: Resource<Frames>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.count)
    }

    async fn fps(&mut self, this: Resource<Frames>) -> Result<f32> {
        Ok(self.ctx().table.get(&this)?.fps)
    }

    async fn encode(
        &mut self,
        this: Resource<Frames>,
        format: ImageFormat,
    ) -> Result<Result<Vec<u8>, String>> {
        Ok(self.ctx().table.get(&this)?.encode(format))
    }

    async fn drop(&mut self, this: Resource<Frames>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::frames::HostPcm for ProcessCtx {
    async fn from_f32(
        &mut self,
        samples: Vec<f32>,
        rate: u32,
        channels: u32,
    ) -> Result<Result<Resource<Pcm>, String>> {
        match Pcm::from_f32(samples, rate, channels) {
            Ok(p) => Ok(Ok(self.ctx().table.push(p)?)),
            Err(e) => Ok(Err(e)),
        }
    }

    async fn rate(&mut self, this: Resource<Pcm>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.rate)
    }

    async fn channels(&mut self, this: Resource<Pcm>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.channels)
    }

    async fn encode(&mut self, this: Resource<Pcm>, format: AudioFormat) -> Result<Vec<u8>> {
        Ok(self.ctx().table.get(&this)?.encode(format))
    }

    async fn drop(&mut self, this: Resource<Pcm>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
