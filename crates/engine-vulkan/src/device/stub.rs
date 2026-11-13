use crate::api::DeviceBoot;
use engine_gpu::error::{Fault, Result};

#[must_use]
pub fn reservations() -> u64 {
    0
}

#[must_use]
pub fn present() -> bool {
    false
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Enabled {
    pub shader_int16: bool,
    pub shader_int64: bool,
    pub shader_float16: bool,
    pub storage_16bit: bool,
    pub storage_8bit: bool,
    pub memory_model: bool,
    pub subgroup_size_control: bool,
    pub memory_budget: bool,
}

#[derive(Debug)]
pub struct Context {
    never: std::convert::Infallible,
}

impl Context {
    pub fn bind(_boot: &DeviceBoot, _share: f64) -> Result<Context> {
        Err(Fault::Deviceless)
    }

    #[must_use]
    pub fn name(&self) -> &str {
        match self.never {}
    }

    #[must_use]
    pub fn working_set(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn left(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        match self.never {}
    }

    #[must_use]
    pub fn subgroup_size(&self) -> u32 {
        match self.never {}
    }

    #[must_use]
    pub fn lanes(&self) -> u32 {
        match self.never {}
    }

    #[must_use]
    pub fn limits(&self) -> crate::tuning::Limits {
        match self.never {}
    }

    #[must_use]
    pub fn api_version(&self) -> u32 {
        match self.never {}
    }

    #[must_use]
    pub fn device_index(&self) -> u32 {
        match self.never {}
    }

    #[must_use]
    pub fn pipeline_cache_path(&self) -> Option<&std::path::Path> {
        match self.never {}
    }

    #[must_use]
    pub fn enabled(&self) -> Enabled {
        match self.never {}
    }

    #[must_use]
    pub fn chip(&self) -> crate::chip::Device {
        match self.never {}
    }

    #[must_use]
    pub fn used(&self) -> u64 {
        match self.never {}
    }

    pub fn bind_thread(&self) -> Result<()> {
        match self.never {}
    }

    pub fn frame(&self) -> Result<Frame> {
        match self.never {}
    }

    pub fn frame_kept(&self) -> Result<Frame> {
        match self.never {}
    }
}

pub struct Frame {
    never: std::convert::Infallible,
}

impl Frame {
    pub fn flush_settled(&self) -> Result<()> {
        self.flush()
    }

    pub fn flush(&self) -> Result<()> {
        Err(Fault::Deviceless)
    }

    #[must_use]
    pub fn dispatches(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn cuts(&self) -> usize {
        match self.never {}
    }

    pub fn commit(self) -> Result<()> {
        match self.never {}
    }

    pub fn commit_timed(self) -> Result<f64> {
        match self.never {}
    }

    pub fn commit_async(self, _on_done: Option<engine_gpu::backend::OnDone>) -> Result<Pending> {
        match self.never {}
    }

    pub fn copy(
        &mut self,
        _source: &Buffer,
        _source_at: u64,
        _into: &Buffer,
        _into_at: u64,
        _len: u64,
    ) -> Result<()> {
        match self.never {}
    }

    pub fn copy_regions(
        &mut self,
        _source: &Buffer,
        _into: &Buffer,
        _regions: &[(u64, u64, u64)],
    ) -> Result<()> {
        match self.never {}
    }

    pub fn fill_zero(&mut self, _into: &Buffer, _at: u64, _len: u64) -> Result<()> {
        match self.never {}
    }

    pub fn into_kept(self) -> Result<Kept> {
        match self.never {}
    }
}

#[derive(Debug)]
pub struct Kept {
    never: std::convert::Infallible,
}

impl Kept {
    #[must_use]
    pub fn cuts(&self) -> usize {
        match self.never {}
    }

    pub fn replay_cut(&self, _at: usize) -> Result<()> {
        match self.never {}
    }

    pub fn replay(
        &self,
        _on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
        match self.never {}
    }
}

#[derive(Debug)]
pub struct Pending {
    never: std::convert::Infallible,
}

impl Pending {
    #[must_use]
    pub fn landed(&self) -> bool {
        match self.never {}
    }

    pub fn wait(&self) -> Result<()> {
        match self.never {}
    }
}

pub use engine_gpu::Memory;

#[derive(Clone, Debug)]
pub struct Buffer {
    never: std::convert::Infallible,
}

impl Buffer {
    pub fn zeroed(_device: &Context, _bytes: u64) -> Result<Buffer> {
        Err(Fault::Deviceless)
    }

    pub fn host(_device: &Context, _bytes: u64) -> Result<Buffer> {
        Err(Fault::Deviceless)
    }

    pub fn with(_device: &Context, _bytes: u64, _kind: Memory) -> Result<Buffer> {
        Err(Fault::Deviceless)
    }

    #[must_use]
    pub fn is_mapped(&self) -> bool {
        match self.never {}
    }

    #[must_use]
    pub fn is_host(&self) -> bool {
        match self.never {}
    }

    pub(crate) fn mapped_ptr(&self) -> Option<usize> {
        match self.never {}
    }

    pub(crate) fn write_shared(&self, _offset: u64, _bytes: &[u8]) -> Result<()> {
        match self.never {}
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self.never {}
    }

    pub fn span(&self, _offset: u64, _len: u64) -> Result<()> {
        match self.never {}
    }

    pub fn write(&mut self, _offset: u64, _bytes: &[u8]) -> Result<()> {
        match self.never {}
    }

    pub fn zero_span(&mut self, _offset: u64, _len: u64) -> Result<()> {
        match self.never {}
    }

    pub fn read(&self, _offset: u64, _into: &mut [u8]) -> Result<()> {
        match self.never {}
    }

    pub fn write_from_file(
        &mut self,
        _file: &std::fs::File,
        _jobs: &[(u64, u64, u64)],
        _threads: usize,
    ) -> Result<()> {
        match self.never {}
    }

    pub fn file_writer(&mut self, _jobs: &[(u64, u64, u64)]) -> Result<FileWriter> {
        match self.never {}
    }
}

impl engine_gpu::Buffer<super::Api> for Buffer {
    fn with(device: &Context, bytes: u64, kind: Memory) -> Result<Buffer> {
        Buffer::with(device, bytes, kind)
    }

    fn bytes(&self) -> u64 {
        match self.never {}
    }

    fn slab_size(&self) -> u64 {
        match self.never {}
    }

    fn slab_id(&self) -> u64 {
        match self.never {}
    }

    fn is_host(&self) -> bool {
        match self.never {}
    }

    fn mapped_ptr(&self) -> Option<usize> {
        match self.never {}
    }

    fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        Buffer::write(self, offset, bytes)
    }

    fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        Buffer::write_shared(self, offset, bytes)
    }

    fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        Buffer::zero_span(self, offset, len)
    }

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        Buffer::read(self, offset, into)
    }

    fn write_from_file(
        &mut self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        Buffer::write_from_file(self, file, jobs, threads)
    }
}

pub struct FileWriter {
    never: std::convert::Infallible,
}

impl FileWriter {
    pub fn pread(
        &self,
        _file: &std::fs::File,
        _jobs: &[(u64, u64, u64)],
        _threads: usize,
    ) -> Result<()> {
        match self.never {}
    }
}

pub struct Pipeline {
    pub bindings: u32,
    pub used: Vec<bool>,
    pub push_bytes: u32,
    pub local: [u32; 3],
}

#[derive(Default, Debug)]
pub struct Pipelines {
    compiles: std::cell::Cell<u64>,
}

impl Pipelines {
    #[must_use]
    pub fn new() -> Pipelines {
        Pipelines::default()
    }

    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.compiles.get()
    }

    pub fn warm(&self, _device: &Context, _fire: kernels_slang::Fire) -> Result<()> {
        Err(Fault::Deviceless)
    }

    pub fn get(
        &self,
        _device: &Context,
        _fire: kernels_slang::Fire,
    ) -> Result<std::sync::Arc<Pipeline>> {
        Err(Fault::Deviceless)
    }
}
