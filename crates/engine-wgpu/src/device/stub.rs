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
    pub subgroups: bool,
    pub f16: bool,
    pub timestamps: bool,
}

#[derive(Debug)]
pub struct Context {
    never: std::convert::Infallible,
}

pub struct Handed {
    never: std::convert::Infallible,
}

impl Context {
    pub fn bind(_boot: &DeviceBoot, _allowance: engine::Allowance) -> Result<Context> {
        Err(Fault::Deviceless)
    }

    pub fn adopt(
        _boot: &DeviceBoot,
        _allowance: engine::Allowance,
        handed: Handed,
    ) -> Result<Context> {
        match handed.never {}
    }

    #[must_use]
    pub fn name(&self) -> &str {
        match self.never {}
    }

    #[must_use]
    pub fn backend(&self) -> &'static str {
        match self.never {}
    }

    #[must_use]
    pub fn capacity(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn outside(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn measures_outside(&self) -> bool {
        match self.never {}
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        match self.never {}
    }

    #[must_use]
    pub fn table_align(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn subgroup_size(&self) -> u32 {
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
    pub fn enabled(&self) -> Enabled {
        match self.never {}
    }

    #[must_use]
    pub fn chip(&self) -> crate::chip::Device {
        match self.never {}
    }

    #[must_use]
    pub fn storage_buffers(&self) -> u32 {
        match self.never {}
    }

    #[must_use]
    pub fn left(&self) -> u64 {
        match self.never {}
    }

    #[must_use]
    pub fn books(&self) -> &engine_gpu::Books {
        match self.never {}
    }

    pub fn bind_thread(&self) -> Result<()> {
        match self.never {}
    }

    pub fn frame(&self) -> Result<Frame> {
        match self.never {}
    }
}

pub struct Frame {
    never: std::convert::Infallible,
}

impl Frame {
    pub(crate) fn update(&self, _into: &Buffer, _at: u64, _bytes: &[u8]) -> Result<()> {
        self.flush()
    }

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

    pub fn commit(self) -> Result<()> {
        match self.never {}
    }

    pub fn commit_timed(self) -> Result<f64> {
        match self.never {}
    }

    pub fn commit_async(self, _on_done: Option<OnDone>) -> Result<Pending> {
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

    pub fn fill_zero(&mut self, _into: &Buffer, _at: u64, _len: u64) -> Result<()> {
        match self.never {}
    }
}

pub use engine_gpu::backend::OnDone;

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
    pub fn zeroed(_device: &Context, _bytes: u64, _owner: engine_gpu::Owner) -> Result<Buffer> {
        Err(Fault::Deviceless)
    }

    pub fn host(_device: &Context, _bytes: u64, _owner: engine_gpu::Owner) -> Result<Buffer> {
        Err(Fault::Deviceless)
    }

    pub fn with(
        _device: &Context,
        _bytes: u64,
        _kind: Memory,
        _owner: engine_gpu::Owner,
    ) -> Result<Buffer> {
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

    pub(crate) fn write_shared(&self, _offset: u64, _bytes: &[u8]) -> Result<()> {
        match self.never {}
    }

    pub fn zero_span(&mut self, _offset: u64, _len: u64) -> Result<()> {
        match self.never {}
    }

    pub fn read(&self, _offset: u64, _into: &mut [u8]) -> Result<()> {
        match self.never {}
    }

    pub fn read_async(
        &self,
        _offset: u64,
        _len: u64,
        _on_read: impl FnOnce(Result<Vec<u8>>) + Send + 'static,
    ) {
        match self.never {}
    }

    pub(crate) fn queue_verdict(&self) -> Result<()> {
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
    fn with(
        device: &Context,
        bytes: u64,
        kind: Memory,
        owner: engine_gpu::Owner,
    ) -> Result<Buffer> {
        Buffer::with(device, bytes, kind, owner)
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
    pub read_only: Vec<bool>,
    pub uniform: Option<u32>,
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

    pub fn get(
        &self,
        _device: &Context,
        _file: &'static str,
        _key: &'static str,
    ) -> Result<std::sync::Arc<Pipeline>> {
        Err(Fault::Deviceless)
    }
}

#[must_use]
pub fn bind_traffic() -> (u64, u64) {
    (0, 0)
}
