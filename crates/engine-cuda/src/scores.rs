//! A capturing lane's attention score rows (`attn.scores`, its key row of
//! the load's context, and `attn.scores.pages`, that row summed a kv page,
//! right after it): state of the instance reading them, not the fire's. The
//! rows lie in the reader's `Instance` slots, after its own item
//! (`program::Plane::held`), so the pool holds them where the runtime holds
//! the instance: a bind, a park and an evacuation move them with it, and
//! nothing is backed for them at load or at bind. Each fire stages two
//! tables of one address a fire lane, in the fire's order, the key row's and
//! the page row's (0 for a lane that captures nothing, which no kernel
//! touches), and the score kernels write through them; the reader reads its
//! own rows.

use kernels_cuda::attn::score::CaptureRows;
use poem_ir::ClassSet;

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::exports::Exports;

/// The names a trace's captured scores are read out by: the key row and
/// the page row.
pub const SCORES: &str = "attn.scores";
pub const SCORE_PAGES: &str = "attn.scores.pages";

/// Whether `name` is a capturing lane's row.
#[must_use]
pub fn captured(name: &str) -> bool {
    name == SCORES || name == SCORE_PAGES
}

/// The load's capture rows: their widths, the classes that compute them
/// (a lane of one captures), and the fire's two row tables.
#[derive(Debug)]
pub struct CaptureLayout {
    context: u32,
    page_size: u32,
    classes: ClassSet,
    /// The key rows' table, then the page rows': a run of the activation's
    /// span laid with the staged inputs (`Shell::staged_runs`).
    tables: Buffer,
    lanes: u32,
    /// Whether the tables name a row: a fire with none stages them only to
    /// clear what an earlier one named.
    named: bool,
}

impl CaptureLayout {
    /// The capture rows of a load whose trace reads scores out, tables of
    /// `lanes` lanes; `None` where it reads none.
    pub(crate) fn of(
        exports: &Exports,
        context: u32,
        page_size: u32,
        lanes: u32,
    ) -> Option<CaptureLayout> {
        let mut classes = ClassSet::default();
        for out in exports
            .outputs
            .iter()
            .filter(|out| captured(&out.decl.name))
        {
            for class in out.classes.iter() {
                classes.insert(class);
            }
        }
        (!classes.is_empty()).then(|| CaptureLayout {
            context,
            page_size: page_size.max(1),
            classes,
            tables: Buffer::lent(2 * lanes as usize * size_of::<u64>()),
            lanes,
            named: false,
        })
    }

    /// A capturing lane's rows, its key row and its page row together
    /// (`engine::load::armed::capture_row_bytes`).
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        engine::load::armed::capture_row_bytes(self.context, self.page_size)
    }

    /// Where a lane's page row starts past its key row.
    #[must_use]
    pub fn key_bytes(&self) -> u64 {
        u64::from(self.context) * size_of::<f32>() as u64
    }

    /// The floats of the key row and of the page row.
    #[must_use]
    pub fn widths(&self) -> (u32, u32) {
        (self.context, self.context.div_ceil(self.page_size))
    }

    /// Whether a lane of `class` captures.
    #[must_use]
    pub fn captures(&self, class: usize) -> bool {
        self.classes.contains(class)
    }

    /// The tables' bytes, a run of the activation's span.
    #[must_use]
    pub fn table_bytes(&self) -> usize {
        self.tables.bytes()
    }

    /// The tables lie at `ptr` from here on.
    pub fn lay(&mut self, ptr: u64) {
        self.tables.lay(ptr);
    }

    /// What a fire's score kernels read.
    #[must_use]
    pub fn seat(&self) -> CaptureRows {
        CaptureRows {
            keys: self.tables.ptr(),
            pages: self.tables.ptr() + u64::from(self.lanes) * 8,
            lanes: self.lanes,
            context: self.context,
            page_size: self.page_size,
        }
    }

    /// Stages the fire's tables on `stream`: `keys_at[lane]`, the key row of
    /// the fire's lane `lane` (in the fire's order), 0 where it captures
    /// nothing; its page row right after it.
    pub fn stage(&mut self, stream: *mut core::ffi::c_void, keys_at: &[u64]) -> Result<()> {
        if keys_at.len() > self.lanes as usize {
            return Err(Fault::Ceiling {
                what: "lanes of the score row tables",
                need: keys_at.len() as u64,
                have: u64::from(self.lanes),
            });
        }
        let named = keys_at.iter().any(|&row| row != 0);
        if !named && !self.named {
            return Ok(());
        }
        self.named = named;
        let key_bytes = self.key_bytes();
        let mut bytes = vec![0u8; self.tables.bytes()];
        let pages = self.lanes as usize * 8;
        for (at, &row) in keys_at.iter().enumerate() {
            let page = if row == 0 { 0 } else { row + key_bytes };
            bytes[at * 8..at * 8 + 8].copy_from_slice(&row.to_le_bytes());
            bytes[pages + at * 8..pages + at * 8 + 8].copy_from_slice(&page.to_le_bytes());
        }
        self.tables.stage(stream, 0, &bytes)
    }
}
