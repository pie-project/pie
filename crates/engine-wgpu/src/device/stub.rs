use std::cell::{Ref, RefCell};

use crate::api::DeviceBoot;
use crate::error::{Fault, Result};

pub const NIL: u32 = u32::MAX;

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
    pub mappable_primary: bool,
    pub pipeline_cache: bool,
}

#[derive(Debug)]
pub struct Context {
    never: std::convert::Infallible,
}

impl Context {
    pub fn bind(_boot: &DeviceBoot) -> Result<Context> {
        Err(Fault::Deviceless)
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
    pub fn working_set(&self) -> u64 {
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
    pub fn tiers(&self) -> Vec<kernels_wgpu::Capability> {
        Vec::new()
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
    pub fn pipeline_cache_path(&self) -> Option<&std::path::Path> {
        match self.never {}
    }

    #[must_use]
    pub fn enabled(&self) -> Enabled {
        match self.never {}
    }

    #[must_use]
    pub fn info(&self) -> kernels_wgpu::DeviceInfo {
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
}

pub struct Frame {
    never: std::convert::Infallible,
}

impl Frame {
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

    pub fn commit_async(
        self,
        _on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    Device,
    Host,
    Staging,
}

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

#[derive(Clone, Debug)]
pub struct Binding {
    offset: u64,
}

impl Binding {
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    #[must_use]
    pub fn remaining(&self) -> u64 {
        0
    }

    #[must_use]
    pub fn slab_id(&self) -> u64 {
        0
    }
}

#[derive(Default, Debug)]
pub struct Handles {
    rows: RefCell<Vec<Binding>>,
    sealed: std::cell::Cell<usize>,
}

impl Handles {
    #[must_use]
    pub fn new() -> Handles {
        Handles::default()
    }

    pub fn bind(&self, buffer: &Buffer, _offset: u64, _len: u64) -> Result<u32> {
        match buffer.never {}
    }

    pub fn read(&self, handle: u32, _len: u64) -> Result<Vec<u8>> {
        Err(Fault::Unbound {
            what: format!("handle {handle}, which no row answers"),
        })
    }

    pub fn cut(&self, handle: u32, _skip: u64, _len: u64) -> Result<u32> {
        Err(Fault::Unbound {
            what: format!("handle {handle}, which this load minted no row for"),
        })
    }

    #[must_use]
    pub fn get(&self, handle: u32) -> Option<Ref<'_, Binding>> {
        if handle == NIL {
            return None;
        }
        let rows = self.rows.borrow();
        if handle as usize >= rows.len() {
            return None;
        }
        Some(Ref::map(rows, |rows| &rows[handle as usize]))
    }

    pub fn seal(&self) {
        if self.sealed.get() == 0 {
            self.sealed.set(self.rows.borrow().len());
        }
    }

    pub fn rewind(&self) {
        self.rows.borrow_mut().truncate(self.sealed.get());
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.borrow().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.borrow().is_empty()
    }

    #[must_use]
    pub fn sealed(&self) -> usize {
        self.sealed.get()
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

    pub fn warm(&self, _device: &Context, _fire: kernels_wgpu::Fire) -> Result<()> {
        Err(Fault::Deviceless)
    }

    pub fn get(
        &self,
        _device: &Context,
        _fire: kernels_wgpu::Fire,
    ) -> Result<std::sync::Arc<Pipeline>> {
        Err(Fault::Deviceless)
    }

    pub fn persist(&self) -> Result<()> {
        Ok(())
    }
}

#[must_use]
pub fn bind_traffic() -> (u64, u64) {
    (0, 0)
}
