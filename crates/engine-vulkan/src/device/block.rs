//! The fire block: the memory a frame's dispatches read what a fire changes
//! from. Each dispatch's scalars lie in it as a uniform at an offset of
//! their own, and its group counts as an indirect dispatch's three words, so
//! a recording's commands name places in the block, never the values in
//! them. The host keeps an image of what a walk laid; a replay writes the
//! image of its own fire's walk, laid out alike, before it submits.

use std::sync::{Arc, Mutex};

use engine_gpu::error::{Fault, Result};

use super::alloc::{Buffer, Memory, Slab};
use super::ctx::Core;

/// The first chunk's bytes; each next chunk doubles the last, so a walk of
/// any length lays its block in a handful.
const FIRST_CHUNK: u64 = 64 << 10;

/// What a walk laid in a fire block, chunk by chunk, through each chunk's
/// last byte laid: what a replay for a fire of that walk's shape writes.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Image(Vec<Vec<u8>>);

impl Image {
    /// The bytes each chunk holds, in order: the block's layout.
    #[must_use]
    pub fn layout(&self) -> Vec<usize> {
        self.0.iter().map(Vec::len).collect()
    }
}

#[derive(Default)]
pub(crate) struct FireBlock {
    chunks: Vec<Buffer>,

    image: Image,

    /// The image a refill wrote last: one written again changes nothing.
    written: Mutex<Option<Arc<Image>>>,
}

impl FireBlock {
    /// Lays `bytes` at the next offset in the block a multiple of `align`,
    /// in a new chunk where the last has no room, and answers the chunk's
    /// slab, its place among the chunks, and the offset.
    pub(crate) fn lay(
        &mut self,
        core: &Arc<Core>,
        bytes: &[u8],
        align: u64,
    ) -> Result<(Slab, usize, u64)> {
        let len = bytes.len() as u64;
        let room = match (self.chunks.last(), self.image.0.last()) {
            (Some(chunk), Some(used)) => {
                (used.len() as u64).next_multiple_of(align) + len <= chunk.bytes()
            }
            _ => false,
        };
        if !room {
            let size = self
                .chunks
                .last()
                .map_or(FIRST_CHUNK, |last| last.bytes() * 2)
                .max(len.next_multiple_of(align));
            self.chunks.push(Buffer::on(
                core,
                size,
                Memory::Host,
                engine_gpu::books::owners::UNIFORMS,
            )?);
            self.image.0.push(Vec::new());
        }
        let chunk = self.chunks.last().expect("a chunk with room was just made");
        let used = self.image.0.last_mut().expect("a chunk has an image");
        let at = (used.len() as u64).next_multiple_of(align);
        used.resize(at as usize, 0);
        used.extend_from_slice(bytes);
        chunk.write_shared(at, bytes)?;
        Ok((Arc::clone(chunk.slab()), self.chunks.len() - 1, at))
    }

    pub(crate) fn image(&self) -> &Image {
        &self.image
    }

    /// Writes `image`, laid by a walk whose recording is this block's, into
    /// the block. The caller holds the block idle: no submission reads it.
    pub(crate) fn refill(&self, image: &Arc<Image>) -> Result<()> {
        let mut written = self.written.lock().expect("fire block lock poisoned");
        if written
            .as_ref()
            .is_some_and(|held| Arc::ptr_eq(held, image))
        {
            return Ok(());
        }
        if image.layout() != self.image.layout() {
            return Err(Fault::Unbound {
                what: format!(
                    "a fire block image laid as {:?} into a block laid as {:?}",
                    image.layout(),
                    self.image.layout()
                ),
            });
        }
        for (chunk, bytes) in self.chunks.iter().zip(&image.0) {
            chunk.write_shared(0, bytes)?;
        }
        *written = Some(Arc::clone(image));
        Ok(())
    }
}
