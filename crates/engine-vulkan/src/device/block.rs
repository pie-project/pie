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

/// What a walk laid in a fire block: what a fire of that walk's shape
/// writes there. An arm-less frame's lies chunk by chunk, through each
/// chunk's last byte laid. An arm's is sparse: each lay's bytes through its
/// last non-zero one (a table laid at its full width is mostly zeros past
/// its rows), where in the block each run lies, and the block's extent;
/// what lies between the runs is zero.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Image {
    chunks: Vec<Vec<u8>>,
    /// An arm's runs' bytes, one after another.
    bytes: Vec<u8>,
    /// Each run's place in the block and length, in order, none touching.
    runs: Vec<(u32, u32)>,
    /// The block's bytes through the last laid.
    extent: u64,
    /// The bytes' hash, taken once the walk is done ([`Image::sealed`]).
    hash: u64,
}

impl Image {
    /// Whether `other` holds the same bytes in the same places.
    #[must_use]
    pub fn same(&self, other: &Image) -> bool {
        self.chunks == other.chunks
            && self.runs == other.runs
            && self.bytes == other.bytes
            && self.extent == other.extent
    }

    /// The bytes the image keeps, each chunk's or the runs'.
    #[must_use]
    pub fn layout(&self) -> Vec<usize> {
        let mut kept: Vec<usize> = self.chunks.iter().map(Vec::len).collect();
        kept.push(self.bytes.len());
        kept
    }

    /// The image with its hash taken, as a fire keeps it.
    #[must_use]
    pub fn sealed(&self) -> Image {
        self.clone().seal()
    }

    /// [`Self::sealed`], taking the image rather than copying it.
    #[must_use]
    pub fn seal(mut self) -> Image {
        // Kept as long as its shape's fill: no room past its bytes.
        for chunk in &mut self.chunks {
            chunk.shrink_to_fit();
        }
        self.bytes.shrink_to_fit();
        self.runs.shrink_to_fit();
        let mut hash = words_hash(&self.chunks) ^ words_hash(std::slice::from_ref(&self.bytes));
        for &(at, len) in &self.runs {
            hash =
                (hash ^ (u64::from(at) << 32 | u64::from(len))).wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.hash = (hash ^ self.extent) | 1;
        self
    }

    fn bytes(&self) -> u64 {
        if self.chunks.is_empty() {
            self.extent
        } else {
            self.chunks.first().map_or(0, |chunk| chunk.len() as u64)
        }
    }

    /// Each run's place and bytes, in order.
    fn runs(&self) -> impl Iterator<Item = (u64, &[u8])> {
        let mut from = 0usize;
        self.runs.iter().map(move |&(at, len)| {
            let run = &self.bytes[from..from + len as usize];
            from += len as usize;
            (u64::from(at), run)
        })
    }

    /// Lays `bytes` at `at` of an arm's block: through their last non-zero
    /// byte, joined to the run before where it ends at `at`.
    fn lay_sparse(&mut self, at: u64, bytes: &[u8]) {
        self.extent = self.extent.max(at + bytes.len() as u64);
        let kept = bytes
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |last| last + 1);
        if kept == 0 {
            return;
        }
        let place = u32::try_from(at).expect("an arm's block lies inside u32 bytes");
        match self.runs.last_mut() {
            Some((start, len)) if u64::from(*start) + u64::from(*len) == at => *len += kept as u32,
            _ => self.runs.push((place, kept as u32)),
        }
        self.bytes.extend_from_slice(&bytes[..kept]);
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
    staged: Mutex<Option<Image>>,
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
        if image.hash != 0
            && staged
                .as_ref()
                .is_some_and(|held| held.hash == image.hash && held.same(image))
        {
            return Ok(false);
        }
        // Forgotten first: a write that fails leaves the slice unknown, and
        // the next stage writes it whole.
        let before = staged.take();
        // Each run, then zeros where the image the slice held had bytes the
        // new one does not cover (between its runs the block is zero).
        for (at, run) in image.runs() {
            self.staging.write_shared(at, run)?;
        }
        let zeros = |from: u64, to: u64| -> Result<()> {
            const ZEROS: [u8; 4096] = [0; 4096];
            let mut at = from;
            while at < to {
                let len = (to - at).min(ZEROS.len() as u64);
                self.staging.write_shared(at, &ZEROS[..len as usize])?;
                at += len;
            }
            Ok(())
        };
        let stale = match &before {
            Some(held) => stale(&held.runs, &image.runs),
            // Nothing known of what the slice holds: zeros between the runs
            // through the image's extent.
            None => gaps(&image.runs, image.extent),
        };
        for (from, to) in stale {
            zeros(from, to)?;
        }
        if image.hash != 0 {
            *staged = Some(Image {
                chunks: Vec::new(),
                bytes: image.bytes.clone(),
                runs: image.runs.clone(),
                extent: image.extent,
                hash: image.hash,
            });
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

/// What of `held`'s runs `new`'s do not cover, each `(from, to)` in order:
/// the bytes a slice that held `held` must zero to hold `new`.
fn stale(held: &[(u32, u32)], new: &[(u32, u32)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut next = new.iter().peekable();
    for &(at, len) in held {
        let (mut from, to) = (u64::from(at), u64::from(at) + u64::from(len));
        while from < to {
            while next
                .peek()
                .is_some_and(|&&(start, len)| u64::from(start) + u64::from(len) <= from)
            {
                next.next();
            }
            match next.peek() {
                Some(&&(start, len)) if u64::from(start) <= from => {
                    from = u64::from(start) + u64::from(len);
                }
                Some(&&(start, _)) => {
                    let end = u64::from(start).min(to);
                    out.push((from, end));
                    from = end;
                }
                None => {
                    out.push((from, to));
                    from = to;
                }
            }
        }
    }
    out
}

/// The bytes through `extent` no run of `runs` covers, in order.
fn gaps(runs: &[(u32, u32)], extent: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut from = 0u64;
    for &(at, len) in runs {
        if u64::from(at) > from {
            out.push((from, u64::from(at)));
        }
        from = u64::from(at) + u64::from(len);
    }
    if extent > from {
        out.push((from, extent));
    }
    out
}

/// A hash of `chunks`' bytes, eight at a time: what tells a staged image
/// from another before its bytes are compared (a walk lays hundreds of KiB
/// of it, sealed once a fill).
fn words_hash(chunks: &[Vec<u8>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ chunks.len() as u64;
    let mut mix = |word: u64| {
        hash = (hash ^ word)
            .wrapping_mul(0x0000_0100_0000_01b3)
            .rotate_left(23);
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
    /// Past an arm's block, where a walk run once lays the rest of what it
    /// lays ([`FireBlock::lay`]): chunks the frame owns, each with the bytes
    /// laid in it.
    past: Vec<(Buffer, u64)>,
}

/// Fires whose walk laid past their arm's block.
static PAST_THE_ARMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
            past: Vec::new(),
        }
    }

    pub(crate) fn of_arm(arm: Arc<ArmBlock>) -> FireBlock {
        FireBlock {
            place: Place::Arm(arm),
            image: Image::default(),
            past: Vec::new(),
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
    /// offset. Past an arm's block (laid at load for the naming walk's
    /// widest fire) a walk run once (`once`) lays in chunks of its frame's
    /// own, warned of by name and counted; a kept or a fill walk is refused.
    pub(crate) fn lay(
        &mut self,
        core: &Arc<Core>,
        bytes: &[u8],
        align: u64,
        kept: usize,
        once: bool,
    ) -> Result<(Slab, usize, u64)> {
        let len = bytes.len() as u64;
        match &mut self.place {
            Place::Arm(arm) => {
                let at = self.image.extent.next_multiple_of(align);
                if (at + len > arm.bytes() || !self.past.is_empty()) && once {
                    return past(&mut self.past, core, arm.bytes(), bytes, align);
                }
                if at + len > arm.bytes() {
                    return Err(Fault::Ceiling {
                        what: "bytes of an arm's fire block (laid at load for the load's widest fire)",
                        need: at + len,
                        have: arm.bytes(),
                    });
                }
                self.image.lay_sparse(at, &bytes[..kept.min(bytes.len())]);
                self.image.extent = self.image.extent.max(at + len);
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
                let used = self.image.chunks.last_mut().expect("a chunk has an image");
                let at = (used.len() as u64).next_multiple_of(align);
                used.resize(at as usize, 0);
                used.extend_from_slice(bytes);
                chunk.write_shared(at, bytes)?;
                Ok((Arc::clone(chunk.slab()), chunks.len() - 1, at))
            }
        }
    }

    /// Whether `buffer` is the arm's block, which outlives the frame.
    pub(crate) fn is_arms(&self, buffer: ash::vk::Buffer) -> bool {
        self.arm()
            .is_some_and(|arm| arm.block.slab().buffer == buffer)
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

/// Lays `bytes` past an arm's block of `arm` bytes, in `chunks`.
fn past(
    chunks: &mut Vec<(Buffer, u64)>,
    core: &Arc<Core>,
    arm: u64,
    bytes: &[u8],
    align: u64,
) -> Result<(Slab, usize, u64)> {
    let len = bytes.len() as u64;
    let fits = chunks
        .last()
        .is_some_and(|(chunk, used)| used.next_multiple_of(align) + len <= chunk.bytes());
    if !fits {
        let size = chunks
            .last()
            .map_or(FIRST_CHUNK, |(last, _)| last.bytes() * 2)
            .max(len.next_multiple_of(align));
        if chunks.is_empty() {
            let fires = PAST_THE_ARMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            tracing::warn!(
                arm,
                fires,
                "a fire that walked eagerly laid past its arm's fire block (laid at load for the \
                 naming walk's widest fire): the rest lies in a reservation of the fire's own"
            );
        }
        let chunk = Buffer::on(
            core,
            size,
            Memory::Host,
            engine_gpu::books::owners::UNIFORMS,
        )?;
        chunks.push((chunk, 0));
    }
    let at = chunks.len();
    let (chunk, used) = chunks.last_mut().expect("a chunk with room was just made");
    let from = used.next_multiple_of(align);
    chunk.write_shared(from, bytes)?;
    *used = from + len;
    Ok((Arc::clone(chunk.slab()), at, from))
}

#[cfg(test)]
mod tests {
    use super::{Image, gaps, stale};

    #[test]
    fn a_lay_keeps_its_bytes_through_the_last_non_zero_and_joins_its_neighbour() {
        let mut image = Image::default();
        image.lay_sparse(0, &[1, 2, 0, 0]);
        image.lay_sparse(4, &[3, 0]);
        image.lay_sparse(64, &[0, 0, 0]);
        image.lay_sparse(128, &[0, 5, 0]);
        assert_eq!(image.runs, vec![(0, 2), (4, 1), (128, 2)]);
        assert_eq!(image.bytes, vec![1, 2, 3, 0, 5]);
        assert_eq!(image.extent, 131);
        let mut joined = Image::default();
        joined.lay_sparse(0, &[1, 1]);
        joined.lay_sparse(2, &[2]);
        assert_eq!(joined.runs, vec![(0, 3)]);
    }

    #[test]
    fn the_slice_zeros_what_the_held_runs_had_and_the_new_ones_miss() {
        assert_eq!(stale(&[(0, 10)], &[(2, 3)]), vec![(0, 2), (5, 10)]);
        assert_eq!(stale(&[(0, 4), (8, 4)], &[(0, 4), (8, 2)]), vec![(10, 12)]);
        assert_eq!(stale(&[(4, 4)], &[]), vec![(4, 8)]);
        assert_eq!(stale(&[(4, 4)], &[(0, 16)]), Vec::<(u64, u64)>::new());
        assert_eq!(
            stale(&[(0, 16)], &[(2, 2), (6, 2)]),
            vec![(0, 2), (4, 6), (8, 16)]
        );
        assert_eq!(gaps(&[(2, 2), (8, 4)], 16), vec![(0, 2), (4, 8), (12, 16)]);
        assert_eq!(gaps(&[], 5), vec![(0, 5)]);
    }
}
