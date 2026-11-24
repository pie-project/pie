//! The fire block: the memory a frame's dispatches read what a fire changes
//! from. Each fired kernel's scalars lie in it as a uniform at an offset of
//! their own, and each dispatch's group counts as an indirect dispatch's
//! three words, so a recording's commands name places in the block, never
//! the values in them.
//!
//! A serving fire's block is its arm's ([`ArmBlock`], laid at load): device
//! memory the host never writes. The host lays what a walk puts there in an
//! [`Image`]; a submission of the fire copies the image from the arm's
//! staging slice into the block at its head, ordered by the queue behind the
//! arm's last fire, which had landed before the arm was taken again. A frame
//! with no arm (a test's, a tune's) lays its block in chunks of its own that
//! the host writes as it walks.

use std::sync::{Arc, Mutex};

use engine_gpu::error::{Fault, Result};

use super::alloc::{Buffer, Memory, Slab};
use super::ctx::{Context, Core};

/// An arm-less frame's first chunk's bytes; each next chunk doubles the
/// last, so a walk of any length lays its block in a handful.
const FIRST_CHUNK: u64 = 64 << 10;

/// What a walk laid in a fire block, chunk by chunk, through each chunk's
/// last byte laid: what a fire of that walk's shape writes there.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Image {
    chunks: Vec<Vec<u8>>,
    /// The bytes' hash, taken once the walk is done ([`Image::sealed`]).
    hash: u64,
}

impl Image {
    /// Whether `other` holds the same bytes in the same chunks.
    #[must_use]
    pub fn same(&self, other: &Image) -> bool {
        self.chunks == other.chunks
    }

    /// The bytes each chunk holds, in order: the block's layout.
    #[must_use]
    pub fn layout(&self) -> Vec<usize> {
        self.chunks.iter().map(Vec::len).collect()
    }

    /// The image with its hash taken, as a fire keeps it.
    #[must_use]
    pub fn sealed(&self) -> Image {
        Image {
            chunks: self.chunks.clone(),
            hash: 0,
        }
        .seal()
    }

    /// [`Self::sealed`], taking the image rather than copying it.
    #[must_use]
    pub fn seal(mut self) -> Image {
        // Kept as long as its shape's fill: no room past its bytes.
        for chunk in &mut self.chunks {
            chunk.shrink_to_fit();
        }
        self.hash = words_hash(&self.chunks);
        self
    }

    fn bytes(&self) -> u64 {
        self.chunks.first().map_or(0, |chunk| chunk.len() as u64)
    }
}

/// An arm's fire block: device memory every dispatch of the arm's fires
/// reads, and the host-visible slice a fire's image reaches it through.
/// One fire of an arm is in flight at most, so the two are the arm's alone
/// until it lands.
pub struct ArmBlock {
    pub(crate) block: Buffer,
    pub(crate) staging: Buffer,
    /// The bytes of the sealed image the staging slice holds (and, every
    /// write being followed by a head that copies the slice, the block),
    /// with their hash: a fire's image that matches both needs no head.
    staged: Mutex<Option<(u64, Vec<u8>)>>,
    /// The heads recordings of the arm share ([`super::Frame`]'s head), by
    /// the bytes each copies (a power of two, at most the block's): one
    /// command buffer for every recording whose image fits, not one each.
    pub(crate) heads: Mutex<std::collections::HashMap<u64, ash::vk::CommandBuffer>>,
}

impl ArmBlock {
    pub fn new(device: &Context, bytes: u64) -> Result<Arc<ArmBlock>> {
        Ok(Arc::new(ArmBlock {
            block: Buffer::with(
                device,
                bytes,
                Memory::Device,
                engine_gpu::books::owners::UNIFORMS,
            )?,
            staging: Buffer::with(
                device,
                bytes,
                Memory::Staging,
                engine_gpu::books::owners::UNIFORMS,
            )?,
            heads: Mutex::new(std::collections::HashMap::new()),
            staged: Mutex::new(None),
        }))
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.block.bytes()
    }

    /// Writes `image` into the staging slice, unless it holds it already,
    /// and answers whether it wrote it. Every write is followed by a head
    /// that copies the slice into the block, so an image the slice already
    /// held is the block's too, and its fire needs no head. The caller holds
    /// the arm: its last fire has landed.
    pub(crate) fn stage(&self, image: &Image) -> Result<bool> {
        let mut staged = self.staged.lock().expect("staged lock poisoned");
        let bytes = image.chunks.first().map_or(&[][..], Vec::as_slice);
        // The hash over the whole image, then the bytes themselves: a few
        // kilobytes, against a head's copy and two barriers.
        if image.hash != 0
            && staged
                .as_ref()
                .is_some_and(|(hash, held)| *hash == image.hash && held.as_slice() == bytes)
        {
            return Ok(false);
        }
        // Forgotten first: a write that fails leaves no image held.
        *staged = None;
        if !bytes.is_empty() {
            self.staging.write_shared(0, bytes)?;
        }
        if image.hash != 0 {
            *staged = Some((image.hash, bytes.to_vec()));
        }
        Ok(true)
    }

    /// The slice's image may not have reached the block (its head was not
    /// submitted): the next fire writes and copies it again.
    pub(crate) fn forget_staged(&self) {
        *self.staged.lock().expect("staged lock poisoned") = None;
    }
}

enum Place {
    /// Chunks the frame owns, the host writing each as it is laid.
    Own(Vec<Buffer>),
    /// The arm's block, the host laying the image alone.
    Arm(Arc<ArmBlock>),
}

/// A hash of `chunks`' bytes, eight at a time: what tells a staged image
/// from another before its bytes are compared (a walk lays hundreds of KiB
/// of it, sealed once a fill).
fn words_hash(chunks: &[Vec<u8>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ chunks.len() as u64;
    let mut mix = |word: u64| {
        hash = (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(23);
    };
    for chunk in chunks {
        mix(chunk.len() as u64);
        let (words, tail) = chunk.as_chunks::<8>();
        for word in words {
            mix(u64::from_le_bytes(*word));
        }
        let mut last = [0u8; 8];
        last[..tail.len()].copy_from_slice(tail);
        mix(u64::from_le_bytes(last));
    }
    // 0 is no hash at all ([`ArmBlock::stage`]).
    hash | 1
}

pub(crate) struct FireBlock {
    place: Place,
    image: Image,
}

impl Default for FireBlock {
    fn default() -> FireBlock {
        FireBlock::own()
    }
}

impl FireBlock {
    pub(crate) fn own() -> FireBlock {
        FireBlock {
            place: Place::Own(Vec::new()),
            image: Image::default(),
        }
    }

    pub(crate) fn of_arm(arm: Arc<ArmBlock>) -> FireBlock {
        FireBlock {
            place: Place::Arm(arm),
            image: Image::default(),
        }
    }

    pub(crate) fn arm(&self) -> Option<&Arc<ArmBlock>> {
        match &self.place {
            Place::Arm(arm) => Some(arm),
            Place::Own(_) => None,
        }
    }

    /// Lays `bytes` at the next offset in the block a multiple of `align`
    /// and answers the slab it lies in, its place among the chunks, and the
    /// offset.
    pub(crate) fn lay(
        &mut self,
        core: &Arc<Core>,
        bytes: &[u8],
        align: u64,
    ) -> Result<(Slab, usize, u64)> {
        let len = bytes.len() as u64;
        match &mut self.place {
            Place::Arm(arm) => {
                if self.image.chunks.is_empty() {
                    self.image.chunks.push(Vec::new());
                }
                let used = &mut self.image.chunks[0];
                let at = (used.len() as u64).next_multiple_of(align);
                if at + len > arm.bytes() {
                    return Err(Fault::Ceiling {
                        what: "bytes of an arm's fire block (laid at load for the load's widest fire)",
                        need: at + len,
                        have: arm.bytes(),
                    });
                }
                used.resize(at as usize, 0);
                used.extend_from_slice(bytes);
                Ok((Arc::clone(arm.block.slab()), 0, at))
            }
            Place::Own(chunks) => {
                let room = match (chunks.last(), self.image.chunks.last()) {
                    (Some(chunk), Some(used)) => {
                        (used.len() as u64).next_multiple_of(align) + len <= chunk.bytes()
                    }
                    _ => false,
                };
                if !room {
                    let size = chunks
                        .last()
                        .map_or(FIRST_CHUNK, |last| last.bytes() * 2)
                        .max(len.next_multiple_of(align));
                    chunks.push(Buffer::on(
                        core,
                        size,
                        Memory::Host,
                        engine_gpu::books::owners::UNIFORMS,
                    )?);
                    self.image.chunks.push(Vec::new());
                }
                let chunk = chunks.last().expect("a chunk with room was just made");
                let used = self
                    .image
                    .chunks
                    .last_mut()
                    .expect("a chunk has an image");
                let at = (used.len() as u64).next_multiple_of(align);
                used.resize(at as usize, 0);
                used.extend_from_slice(bytes);
                chunk.write_shared(at, bytes)?;
                Ok((Arc::clone(chunk.slab()), chunks.len() - 1, at))
            }
        }
    }

    /// The image laid so far, taken from the block.
    pub(crate) fn take_image(&mut self) -> Image {
        std::mem::take(&mut self.image)
    }

    pub(crate) fn image(&self) -> &Image {
        &self.image
    }

    /// The bytes laid in an arm's block: what its head copies.
    pub(crate) fn arm_bytes(&self) -> u64 {
        match self.place {
            Place::Arm(_) => self.image.bytes(),
            Place::Own(_) => 0,
        }
    }
}
