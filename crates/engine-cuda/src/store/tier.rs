//! The host and disk tiers beside the device pool: their slots, the copies
//! between them and the device, and the taps and exports of kv pages.

#[allow(clippy::wildcard_imports)]
use super::*;

/// `slots` of `shelf`, one after another, each its planes in order.
pub(super) fn read_shelf(shelf: &Shelf, slots: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &slot in slots {
        for (plane, &bytes) in shelf.planes.iter().enumerate() {
            let (pinned, at) = shelf.place(slot, plane).expect("backed by the caller");
            out.extend_from_slice(pinned.view(at, bytes).expect("inside its chunk"));
        }
    }
    out
}

/// Writes `bytes`, as [`read_shelf`] lays them out, into `slots` of `shelf`.
pub(super) fn write_shelf(shelf: &Shelf, slots: &[u32], bytes: &[u8], what: &'static str) -> Result<()> {
    let slot_bytes: u64 = shelf.planes.iter().sum();
    let need = slots.len() as u64 * slot_bytes;
    if bytes.len() as u64 != need {
        return Err(Fault::Ceiling {
            what,
            need,
            have: bytes.len() as u64,
        });
    }
    let mut from = 0;
    for &slot in slots {
        for (plane, &len) in shelf.planes.iter().enumerate() {
            let (pinned, at) = shelf.place(slot, plane).expect("backed by the caller");
            let len = usize::try_from(len).unwrap_or(0);
            let at = usize::try_from(at).unwrap_or(usize::MAX);
            assert!(
                pinned.write(at, &bytes[from..from + len]),
                "inside its chunk"
            );
            from += len;
        }
    }
    Ok(())
}

/// A slot-file write failing for want of room on its file system is the
/// tier below full ([`Fault::TierFull`]): nothing on the device changed.
pub(super) fn slot_file(error: std::io::Error) -> Fault {
    if error.kind() == std::io::ErrorKind::StorageFull {
        return Fault::TierFull(error.to_string());
    }
    Fault::Integrity {
        at: "the disk tier's slot file",
        why: error.to_string(),
    }
}

/// A tier a slot copy reads or writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,
    Host,
    Disk,
}

/// A move of kv pages between the device and their host slots, on the
/// swap stream, or a tap's read-out: its device pages, the host slots it
/// reads or writes, and the event it lands at.
#[derive(Debug)]
pub(super) struct Swap {
    pub(super) landed: crate::device::graph::Event,
    pub(super) paged: Vec<u32>,
    pub(super) host: Vec<u32>,
    /// It holds slots a fire names by no kv run (a window list's, a
    /// record's, the rs list's): every fire waits for it.
    pub(super) wide: bool,
}

/// The disk tier below host memory: a slot file of disk
/// chunks, each a pool chunk on the direct I/O boundary
/// ([`Pools::disk_chunk_bytes`]) laid out as the device's, any state
/// shape's slots in any chunk; and the pinned stage a slot's cover (the
/// aligned blocks it lies in) moves through.
#[derive(Debug)]
pub(super) struct DiskTier {
    pub(super) file: std::sync::Arc<engine::disk::SlotFile>,
    /// Held across a slot cover's read-modify-write, here and on the host
    /// tier's landing thread ([`host::DiskMove`]).
    pub(super) io: std::sync::Arc<std::sync::Mutex<()>>,
    pub(super) stage: DiskStage,
}

/// A tell handed to the next host<->disk move ([`Pools::defer_disk_tell`]).
#[derive(Default)]
pub(super) struct DiskTell(pub(super) Option<crate::serve::landing::Told>);

impl std::fmt::Debug for DiskTell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "DiskTell(pending)" } else { "DiskTell(none)" })
    }
}

/// The pinned stage of the disk tier.
#[derive(Debug)]
pub(super) struct DiskStage {
    pub(super) pinned: crate::device::Pinned,
    /// The bytes it holds: the widest slot's cover.
    pub(super) room: usize,
    /// An h2d out of it may still be in flight on the stream.
    pub(super) landing: bool,
}

impl DiskStage {
    /// The stage's first `span` bytes, a slot's cover, for the host to
    /// read or write. The caller has waited out every copy on a stream
    /// that touches the stage (`landing`).
    fn staged(&mut self, span: usize) -> Result<&mut [u8]> {
        if span > self.room {
            return Err(Fault::Ceiling {
                what: "bytes of the disk tier's stage",
                need: span as u64,
                have: self.room as u64,
            });
        }
        // SAFETY: the stage is this load's pinned allocation of `room`
        // bytes, page aligned; no copy on a stream reads or writes it while
        // the host does (the caller waits out `landing` first).
        Ok(unsafe { std::slice::from_raw_parts_mut(self.pinned.host(), span) })
    }
}

/// Rank `rank`'s slot file of the disk tier.
#[must_use]
pub fn disk_file(rank: u32) -> String {
    format!("r{rank}.state")
}

/// The aligned span of the slot file the bytes `[at, at + len)` lie in:
/// its first byte, its length, and the byte `at` falls on within it.
pub(super) fn cover(at: u64, len: u64) -> (u64, usize, usize) {
    let align = engine::disk::ALIGN as u64;
    let start = at / align * align;
    let end = (at + len).next_multiple_of(align);
    (
        start,
        usize::try_from(end - start).unwrap_or(usize::MAX),
        usize::try_from(at - start).unwrap_or(usize::MAX),
    )
}

/// The bytes a cover of a slot of `slot_bytes` takes at most.
pub(super) fn cover_room(slot_bytes: u64) -> usize {
    let align = engine::disk::ALIGN as u64;
    usize::try_from(slot_bytes.next_multiple_of(align) + align).unwrap_or(usize::MAX)
}

/// Fills `span` (the cover from `start`) outside the slot's `[lead, lead +
/// len)` from the file, so writing the whole span leaves the slots beside
/// it as they are; `edge` is one aligned block of scratch. Run right before
/// the write, after any earlier write of the same blocks.
pub(super) fn keep_edges(
    file: &engine::disk::SlotFile,
    start: u64,
    span: &mut [u8],
    lead: usize,
    len: usize,
    edge: &mut engine::disk::Aligned,
) -> Result<()> {
    let align = engine::disk::ALIGN;
    let block = edge.as_mut_slice();
    if lead > 0 {
        file.read_span(start, block).map_err(slot_file)?;
        span[..lead].copy_from_slice(&block[..lead]);
    }
    let end = lead + len;
    if end < span.len() {
        let last = span.len() - align;
        file.read_span(start + last as u64, block)
            .map_err(slot_file)?;
        span[end..].copy_from_slice(&block[end - last..]);
    }
    Ok(())
}
/// The pinned host kv slots a page's bytes are read out to and written in
/// from by plane ([`Pools::read_host_kv`], a tap's read-out). Each paged
/// plane keeps its host pages side by side, as its device arena does, so a
/// run of pages moves in one copy. What a runtime parks moves through the
/// host tier instead ([`Pools::copy_tier_slots`]).
#[derive(Debug)]
pub(super) struct HostKv {
    pub(super) paged: Shelf,
}

/// The bytes a host shelf pins at a time.
pub(super) const SHELF_CHUNK: u64 = 256 << 20;

/// One kind of host slot, pinned a chunk of slots at a time as a copy first
/// touches it. A chunk lays its slots out plane by plane, so a run of slots
/// within one chunk is one span a plane.
#[derive(Debug, Default)]
pub(super) struct Shelf {
    pub(super) planes: Vec<u64>,
    pub(super) slots: u32,
    pub(super) per_chunk: u32,
    pub(super) chunks: Vec<Option<crate::device::Pinned>>,
}

impl Shelf {
    fn new(planes: Vec<u64>, slots: u32) -> Shelf {
        let slot: u64 = planes.iter().sum();
        let per_chunk = (SHELF_CHUNK / slot.max(1)).clamp(1, u64::from(slots.max(1)));
        let per_chunk = u32::try_from(per_chunk).unwrap_or(u32::MAX);
        Shelf {
            planes,
            slots,
            per_chunk,
            chunks: (0..slots.div_ceil(per_chunk)).map(|_| None).collect(),
        }
    }

    /// Pins the chunks that hold `slots`.
    fn back(&mut self, slots: impl IntoIterator<Item = u32>, what: &'static str) -> Result<()> {
        let slot_bytes: u64 = self.planes.iter().sum();
        for slot in slots {
            if slot >= self.slots {
                return Err(Fault::Ceiling {
                    what,
                    need: u64::from(slot) + 1,
                    have: u64::from(self.slots),
                });
            }
            let chunk = slot / self.per_chunk;
            if self.chunks[chunk as usize].is_none() {
                let held = (self.slots - chunk * self.per_chunk).min(self.per_chunk);
                let bytes = usize::try_from(u64::from(held) * slot_bytes).unwrap_or(usize::MAX);
                let pinned = huge_tier(bytes)
                    .map_or_else(|| crate::device::Pinned::mapped_uninit(bytes), Ok)?;
                self.chunks[chunk as usize] = Some(pinned);
            }
        }
        Ok(())
    }

    /// The pinned chunk holding `slot`, and where its `plane` starts in it;
    /// `None` until [`Self::back`] pins it.
    fn place(&self, slot: u32, plane: usize) -> Option<(&crate::device::Pinned, u64)> {
        let chunk = slot / self.per_chunk;
        let pinned = self.chunks.get(chunk as usize)?.as_ref()?;
        let held = (self.slots - chunk * self.per_chunk).min(self.per_chunk);
        let before: u64 = self.planes[..plane].iter().sum();
        Some((
            pinned,
            before * u64::from(held) + u64::from(slot % self.per_chunk) * self.planes[plane],
        ))
    }

    fn at(&self, slot: u32, plane: usize) -> Result<u64> {
        self.place(slot, plane)
            .map(|(pinned, offset)| pinned.host() as u64 + offset)
            .ok_or(Fault::Ceiling {
                what: "pinned host slots",
                need: u64::from(slot) + 1,
                have: 0,
            })
    }
}

/// `bytes` pinned on reserved huge pages, where the host's pool holds them.
#[cfg(target_os = "linux")]
pub(super) fn huge_tier(bytes: usize) -> Option<crate::device::Pinned> {
    crate::device::Pinning::huge(bytes)?.lock().ok()
}

#[cfg(not(target_os = "linux"))]
pub(super) fn huge_tier(_bytes: usize) -> Option<crate::device::Pinned> {
    None
}

/// A group a tap reads a host slot's planes in: its layer, its planes (of
/// the paged planes, in a slot's order), and their spans `(offset, bytes)`
/// of a slot.
#[derive(Debug, Clone, Default)]
pub struct TapGroup {
    pub layer: u32,
    pub planes: Vec<usize>,
    pub spans: Vec<(u64, u64)>,
}

/// A tap's pages of a group: the copies `(dst, src, bytes)` out to the host
/// tier, and where each page's planes land there, page after page,
/// `(address, bytes)`.
#[derive(Debug, Clone, Default)]
pub struct TapSpans {
    pub copies: Vec<(u64, u64, usize)>,
    pub lands: Vec<(u64, usize)>,
}

/// What [`Pools::export_kv`] hands out: the planes' regions and the shape
/// their pages have.
#[derive(Debug, Clone)]
pub struct ExportedKv {
    pub regions: Vec<engine::KvRegion>,
    pub layers: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
    pub page_size: u32,
}

impl Pools {
    /// Seats the host tier of `budget` bytes: as many whole chunks of the
    /// state pool's size, in pageable memory the tier commits as moves
    /// first write it, behind the staging `ring` the load pinned before it
    /// measured the device ([`staging_ring_for`]); a tier of no chunks lets
    /// it go.
    pub fn seat_tier(&mut self, budget: u64, ring: Option<Ring>) -> Result<()> {
        // No more than the machine has free, known now: a tier stated larger
        // would fail mid-serve, at the first move past what the host holds.
        let free = crate::weights::available_memory();
        let budget = if budget > free {
            tracing::warn!(
                "the host tier's budget of {} MiB is more than this \
                 machine has free ({} MiB); the tier is seated with {} MiB",
                budget >> 20,
                free >> 20,
                free >> 20
            );
            free
        } else {
            budget
        };
        let chunk = self.chunks.layout().chunk_bytes;
        let chunks = u32::try_from(budget.checked_div(chunk).unwrap_or(0)).unwrap_or(u32::MAX);
        self.tier = match (chunks, ring) {
            (0, _) | (_, None) => None,
            (chunks, Some(ring)) => Some(host::HostTier::new(chunk, chunks, ring)?),
        };
        Ok(())
    }

    /// The host tier's chunks.
    #[must_use]
    pub fn tier_chunks(&self) -> u32 {
        self.tier.as_ref().map_or(0, host::HostTier::chunks)
    }

    /// The disk tier's chunks.
    #[must_use]
    pub fn disk_chunks(&self) -> u32 {
        self.disk.as_ref().map_or(0, |disk| disk.file.slots())
    }

    /// The pool's chunk layout: its chunk bytes and shapes.
    #[must_use]
    pub fn chunk_layout(&self) -> &crate::store::layout::Layout {
        self.chunks.layout()
    }

    /// The bytes one disk chunk takes: a pool chunk, on the slot file's
    /// direct I/O boundary, laid out as the device's.
    #[must_use]
    pub fn disk_chunk_bytes(&self) -> u64 {
        self.chunks
            .layout()
            .chunk_bytes
            .next_multiple_of(engine::disk::ALIGN as u64)
    }

    /// How the disk tier divides a disk chunk: each state shape's id, its
    /// slots a chunk and the bytes of each, packed as on the device.
    #[must_use]
    pub fn disk_division(&self) -> Vec<(engine::chunk::ShapeId, u32, u64)> {
        self.chunks
            .layout()
            .state_shapes()
            .map(|shape| (shape.id, shape.per_chunk, shape.slot_bytes))
            .collect()
    }

    /// The byte of the slot file disk ref `slot` of `kind` starts at: its
    /// place in its disk chunk, as on the device.
    pub(super) fn disk_place(&self, kind: Kind, slot: u32) -> Result<u64> {
        let shape = self.chunks.layout().shape(kind).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!("a disk slot of a {} in a load that lays none", kind.name()),
            )
        })?;
        let (chunk, offset) = shape.tier_place(slot);
        let have = self.disk_chunks();
        if chunk >= have {
            return Err(Fault::Ceiling {
                what: "chunks of the disk tier",
                need: u64::from(chunk) + 1,
                have: u64::from(have),
            });
        }
        Ok(u64::from(chunk) * self.disk_chunk_bytes() + offset)
    }

    /// The load's shape `shape` names.
    pub(super) fn kind_of(shape: engine::chunk::ShapeId) -> Result<Kind> {
        Ok(match shape {
            layout::LIST => Kind::List,
            window if layout::window_of(window).is_some() => {
                Kind::Window(layout::window_of(window).unwrap_or(0) as u8)
            }
            layout::FIXED => Kind::Fixed,
            layout::RING => Kind::Ring,
            layout::RS => Kind::Rs,
            layout::INSTANCE => Kind::Instance,
            layout::CHANNEL => Kind::Channel,
            other => {
                return Err(Fault::program(
                    "store::chunks",
                    format!("a slot copy names shape {other}, which this load does not state"),
                ));
            }
        })
    }

    /// Where host ref `slot` of `kind` lies in the host tier, from its
    /// start: its chunk, and its place among the chunk's slots.
    pub(super) fn tier_offset(&self, kind: Kind, slot: u32) -> Result<u64> {
        let shape = self.chunks.layout().shape(kind).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!("a host slot of a {} in a load that lays none", kind.name()),
            )
        })?;
        let (chunk, offset) = shape.tier_place(slot);
        let have = self.tier_chunks();
        if chunk >= have {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: u64::from(chunk) + 1,
                have: u64::from(have),
            });
        }
        Ok(u64::from(chunk) * self.chunks.layout().chunk_bytes + offset)
    }

    /// Where each of `refs`, host tier slots of `shape`, lies: its chunk of
    /// the tier (backed) and its byte offset in that chunk; and a slot's
    /// bytes.
    pub(super) fn tier_places(
        &mut self,
        shape: engine::chunk::ShapeId,
        refs: &[u32],
    ) -> Result<(u64, Vec<(u32, u64)>)> {
        let kind = Self::kind_of(shape)?;
        let facts = self.chunks.layout().shape(kind).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!("a host slot of a {} in a load that lays none", kind.name()),
            )
        })?;
        let bytes = facts.slot_bytes;
        let places: Vec<(u32, u64)> = refs.iter().map(|&slot| facts.tier_place(slot)).collect();
        let Some(tier) = self.tier.as_mut() else {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: 1,
                have: 0,
            });
        };
        if let Some(&(chunk, _)) = places.iter().find(|&&(chunk, _)| chunk >= tier.chunks()) {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: u64::from(chunk) + 1,
                have: u64::from(tier.chunks()),
            });
        }
        Ok((bytes, places))
    }

    /// See `Engine::read_tier_slots`: the bytes of host tier slots, shape
    /// by shape, each laid out as its device slot. The caller has settled
    /// every copy into them.
    pub fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for (shape, refs) in slots {
            let (bytes, places) = self.tier_places(*shape, refs)?;
            let tier = self.tier.as_ref().expect("backed above");
            let chunk_bytes = tier.chunk_bytes();
            for (chunk, offset) in places {
                let len = usize::try_from(bytes).unwrap_or(usize::MAX);
                out.extend_from_slice(tier.slice(u64::from(chunk) * chunk_bytes + offset, len)?);
            }
        }
        Ok(out)
    }

    /// See `Engine::write_tier_slots`: `bytes`, as [`Self::read_tier_slots`]
    /// lays them out, into host tier slots. The caller has settled every
    /// copy that touches them.
    pub fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> Result<()> {
        let mut spans = Vec::new();
        for (shape, refs) in slots {
            let (len, places) = self.tier_places(*shape, refs)?;
            spans.extend(places.into_iter().map(|place| (place, len)));
        }
        let need: u64 = spans.iter().map(|&(_, len)| len).sum();
        if bytes.len() as u64 != need {
            return Err(Fault::Ceiling {
                what: "bytes of host tier slots",
                need,
                have: bytes.len() as u64,
            });
        }
        let tier = self.tier.as_ref().expect("backed above");
        let chunk_bytes = tier.chunk_bytes();
        let mut from = 0usize;
        for ((chunk, offset), len) in spans {
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            tier.slice(u64::from(chunk) * chunk_bytes + offset, len)?
                .copy_from_slice(&bytes[from..from + len]);
            from += len;
        }
        Ok(())
    }

    /// Waits out every move between the device and the host tier still
    /// landing: what a byte read or write of host tier slots stands behind.
    pub fn settle_all_swaps(&mut self) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            swap.landed.settle()?;
        }
        Ok(())
    }

    /// The device address of slot ref `slot` of `shape`, its chunk seated
    /// for the shape: where a copy to or from a tier reads or writes it.
    pub fn slot_span(&mut self, shape: engine::chunk::ShapeId, slot: u32) -> Result<u64> {
        let kind = Self::kind_of(shape)?;
        let bytes = self.chunks.layout().shape(kind).map_or(0, |shape| shape.slot_bytes);
        self.seat_pages(kind, [u64::from(slot)])?;
        self.chunks.span(kind, slot, 0, bytes)
    }

    /// Copies whole slots of `shape`, `(src, dst)`, from tier `from` to tier
    /// `to`, on `stream` behind every fire enqueued on it
    /// and ahead of every one after: a device ref is the runtime's (through
    /// the chunk table), a host ref a place in the host tier, a disk ref a
    /// place in the disk tier, of any state shape. A slot is laid the same
    /// on every tier, so each moves as one span. What goes to disk is
    /// written once the stream has it; what comes from disk is read first.
    ///
    /// Between the device and the host tier the copy rides `swap` where one
    /// is given: behind what `stream` has enqueued (the fires that wrote
    /// the slots), beside the fires after, of which one that names a slot it
    /// holds waits for it ([`Pools::fence_swaps`]).
    pub fn copy_tier_slots(
        &mut self,
        stream: *mut core::ffi::c_void,
        swap: Option<*mut core::ffi::c_void>,
        shape: engine::chunk::ShapeId,
        from: Tier,
        to: Tier,
        pairs: &[(u32, u32)],
    ) -> Result<()> {
        if pairs.is_empty() || from == to {
            return Ok(());
        }
        let kind = Self::kind_of(shape)?;
        let bytes = self
            .chunks
            .layout()
            .shape(kind)
            .map(|shape| shape.slot_bytes)
            .ok_or_else(|| {
                Fault::program(
                    "store::chunks",
                    format!("a slot copy of a {} in a load that lays none", kind.name()),
                )
            })?;
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        let device: Vec<u64> = pairs
            .iter()
            .filter_map(|&(src, dst)| match (from, to) {
                (Tier::Device, _) => Some(u64::from(src)),
                (_, Tier::Device) => Some(u64::from(dst)),
                _ => None,
            })
            .collect();
        self.seat_pages(kind, device.iter().copied())?;
        match (from, to) {
            (Tier::Device, Tier::Host) | (Tier::Host, Tier::Device) => {
                let to_host = to == Tier::Host;
                let mut slots = Vec::with_capacity(pairs.len());
                for &(src, dst) in pairs {
                    let (on_device, on_host) = if to_host { (src, dst) } else { (dst, src) };
                    let on_host = self.tier_offset(kind, on_host)?;
                    let on_device = self.chunks.span(kind, on_device, 0, bytes)?;
                    slots.push(if to_host {
                        (on_device, on_host, len)
                    } else {
                        (on_host, on_device, len)
                    });
                }
                // Behind the fires that wrote the slots, and behind every
                // move before it (the swap stream's own order); without a
                // swap stream, on `stream` behind every move still landing.
                let lane = match swap {
                    Some(swap) => {
                        let written = crate::device::graph::Event::new()?;
                        written.record(stream)?;
                        written.wait(swap)?;
                        swap
                    }
                    None => {
                        self.fence_all_swaps(stream)?;
                        stream
                    }
                };
                let tier = self.tier.as_mut().ok_or(Fault::Ceiling {
                    what: "chunks of the host tier",
                    need: 1,
                    have: 0,
                })?;
                let landed = if to_host {
                    tier.to_host(lane, &slots)?
                } else {
                    tier.to_device(lane, &slots)?
                };
                if let (Some(landed), Some(_)) = (landed, swap) {
                    self.swaps.push(Swap {
                        landed,
                        paged: if kind == Kind::List {
                            device.iter().map(|&slot| slot as u32).collect()
                        } else {
                            Vec::new()
                        },
                        host: Vec::new(),
                        wide: kind != Kind::List,
                    });
                }
                Ok(())
            }
            (Tier::Device, Tier::Disk) | (Tier::Disk, Tier::Device) => {
                self.fence_all_swaps(stream)?;
                let to_disk = to == Tier::Disk;
                let mut slots = Vec::with_capacity(pairs.len());
                for &(src, dst) in pairs {
                    let (on_device, on_disk) = if to_disk { (src, dst) } else { (dst, src) };
                    slots.push((
                        self.chunks.span(kind, on_device, 0, bytes)?,
                        self.disk_place(kind, on_disk)?,
                    ));
                }
                self.through_disk(stream, to_disk, &slots, bytes)
            }
            (Tier::Host, Tier::Disk) | (Tier::Disk, Tier::Host) => {
                // On the host tier's landing thread, behind every move it
                // holds (a move into or out of these host slots lands before
                // it): the engine thread only enqueues. Told later where a
                // tell was handed in ([`Pools::defer_disk_tell`]); waited out
                // here otherwise, as a copy that answers its sources read.
                let to_disk = to == Tier::Disk;
                let len = usize::try_from(bytes).unwrap_or(usize::MAX);
                let mut slots = Vec::with_capacity(pairs.len());
                for &(src, dst) in pairs {
                    let (on_host, on_disk) = if to_disk { (src, dst) } else { (dst, src) };
                    slots.push((
                        self.tier_offset(kind, on_host)?,
                        self.disk_place(kind, on_disk)?,
                        len,
                    ));
                }
                let told = self.disk_told.0.take();
                let (Some(spill), Some(tier)) = (self.disk.as_ref(), self.tier.as_mut()) else {
                    if let Some(told) = told {
                        told(engine::Landing::Landed);
                    }
                    return Ok(());
                };
                let waited = told.is_none();
                let id = tier.disk(host::DiskMove {
                    to_disk,
                    file: std::sync::Arc::clone(&spill.file),
                    io: std::sync::Arc::clone(&spill.io),
                    slots,
                    told,
                })?;
                if waited {
                    tier.wait_job(id)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Moves whole slots of `bytes` between the device and the disk tier,
    /// `(device address, disk place)` each, through the stage one slot at a
    /// time: a slot's cover read whole, and written keeping its neighbours'
    /// bytes.
    pub(super) fn through_disk(
        &mut self,
        stream: *mut core::ffi::c_void,
        to_disk: bool,
        slots: &[(u64, u64)],
        bytes: u64,
    ) -> Result<()> {
        // Behind every move between the host tier and these bytes of the
        // file, done on the tier's landing thread.
        if let Some(tier) = self.tier.as_ref() {
            tier.wait_disk(slots.iter().map(|&(_, disk)| (disk, bytes)))?;
        }
        let Some(spill) = self.disk.as_mut() else {
            return Ok(());
        };
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        let mut edge = engine::disk::Aligned::zeroed(engine::disk::ALIGN);
        let io = std::sync::Arc::clone(&spill.io);
        for &(device, disk) in slots {
            let _io = io.lock().expect("the slot file's I/O");
            if spill.stage.landing {
                crate::device::ctx::sync(stream)?;
                spill.stage.landing = false;
            }
            let (start, span, lead) = cover(disk, bytes);
            let at = spill.stage.pinned.host() as u64 + lead as u64;
            let staged = spill.stage.staged(span)?;
            if to_disk {
                crate::device::copy_any(stream, at, device, len)?;
                crate::device::ctx::sync(stream)?;
                keep_edges(&spill.file, start, staged, lead, len, &mut edge)?;
                spill.file.write_span(start, staged).map_err(slot_file)?;
            } else {
                spill.file.read_span(start, staged).map_err(slot_file)?;
                crate::device::copy_any(stream, device, at, len)?;
                spill.stage.landing = true;
            }
        }
        Ok(())
    }

    /// Seats `pages` host kv pages, each pinned a chunk at a time as a copy
    /// first touches it ([`Shelf`]), on reserved huge pages where the host
    /// keeps a pool that holds them. Pinning at once would hold boot for as
    /// long as the driver takes to pin it all, and when a process dies the
    /// driver unpins its memory under a lock every process on the host
    /// waits on for any allocation, about 0.17 s a GiB of 4 KiB pages and a
    /// fifth of that of huge ones.
    pub fn seat_host(&mut self, pages: u32) {
        let planes = self.paged_planes().into_iter().map(|(_, b)| b).collect();
        self.host = (pages > 0).then(|| HostKv {
            paged: Shelf::new(planes, pages),
        });
    }

    #[must_use]
    pub fn host_pages(&self) -> u32 {
        self.host.as_ref().map_or(0, |host| host.paged.slots)
    }

    /// Opens `dir`'s slot file for this rank with `budget` bytes of disk
    /// chunks ([`Self::disk_chunk_bytes`]), any state shape's slots in any
    /// chunk, laid out as on the device; a budget short of one chunk seats
    /// none. `keep` names the load when the file is kept for the next run
    /// ([`engine::disk::SlotFile::open_kept`]): with the chunk and its
    /// division, the layout its slots are read back by.
    pub fn seat_disk(
        &mut self,
        dir: &std::path::Path,
        rank: u32,
        budget: u64,
        keep: Option<&str>,
    ) -> Result<()> {
        let name = disk_file(rank);
        let chunk = self.disk_chunk_bytes();
        let widest = self
            .chunks
            .layout()
            .state_shapes()
            .map(|shape| shape.slot_bytes)
            .max()
            .unwrap_or(0);
        let file = match keep {
            None => engine::disk::SlotFile::open(dir, &name, chunk, budget),
            Some(load) => {
                let layout = format!("cuda {load} chunk {chunk} division {:?}", self.disk_division());
                engine::disk::SlotFile::open_kept(dir, &name, chunk, budget, &layout)
            }
        }
        .map_err(slot_file)?;
        let room = cover_room(widest);
        self.disk = file
            .map(|file| {
                crate::device::Pinned::mapped_uninit(room).map(|pinned| DiskTier {
                    file: std::sync::Arc::new(file),
                    io: std::sync::Arc::default(),
                    stage: DiskStage {
                        pinned,
                        room,
                        landing: false,
                    },
                })
            })
            .transpose()?;
        Ok(())
    }

    pub fn unseat_disk(&mut self) {
        self.disk = None;
    }

    /// The slot file is kept, and holds what the last run left.
    #[must_use]
    pub fn disk_kept(&self) -> bool {
        self.disk.as_ref().is_some_and(|disk| disk.file.kept())
    }

    /// Copies whole kv pages between the device pool and the host pages,
    /// `device[i]` with `host[i]`: on `swap`, behind what `stream` has
    /// enqueued (the fires that wrote them), so the fires after run beside
    /// it; one that holds its pages waits for it ([`Pools::fence_swaps`]).
    pub fn swap_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        swap: *mut core::ffi::c_void,
        to_host: bool,
        device: &[u32],
        host: &[u32],
    ) -> Result<()> {
        let paged: Vec<(u32, u32)> = device.iter().copied().zip(host.iter().copied()).collect();
        self.ensure_kv_pages(device)?;
        let Some(tier) = self.host.as_mut() else {
            return match host.first() {
                None => Ok(()),
                Some(_) => Err(Fault::Ceiling {
                    what: "host kv pages",
                    need: 1,
                    have: 0,
                }),
            };
        };
        tier.paged.back(host.iter().copied(), "host kv pages")?;
        let shelf = &self.host.as_ref().expect("seated above").paged;
        let mut spans: Vec<(u64, u64, usize)> = Vec::new();
        // A page's planes lie apart in its slot, so each is a span.
        for (at, (plane, page_bytes)) in self.paged_planes().into_iter().enumerate() {
            for &(d, h) in &paged {
                let on_device = self.plane_span(plane, d, page_bytes)?;
                let on_host = shelf.at(h, at)?;
                let (dst, src) = if to_host {
                    (on_host, on_device)
                } else {
                    (on_device, on_host)
                };
                spans.push((dst, src, usize::try_from(page_bytes).unwrap_or(0)));
            }
        }
        if spans.is_empty() {
            return Ok(());
        }
        let written = crate::device::graph::Event::new()?;
        written.record(stream)?;
        written.wait(swap)?;
        crate::device::copy_batch(swap, to_host, &spans)?;
        let landed = crate::device::graph::Event::new()?;
        landed.record(swap)?;
        self.swaps.push(Swap {
            landed,
            paged: device.to_vec(),
            host: host.to_vec(),
            wide: false,
        });
        Ok(())
    }

    /// `paged` device pages are read, and `host` slots written, on `stream`
    /// by what it has enqueued: a fire that names one of the pages waits for
    /// it, as for a move, and so does the host before it touches a slot.
    pub fn hold_until(
        &mut self,
        stream: *mut core::ffi::c_void,
        paged: Vec<u32>,
        host: Vec<u32>,
    ) -> Result<()> {
        let landed = crate::device::graph::Event::new()?;
        landed.record(stream)?;
        self.swaps.push(Swap {
            landed,
            paged,
            host,
            wide: false,
        });
        Ok(())
    }

    /// Waits, on the host, for the moves and read-outs still landing that
    /// read or write any of the paged host `slots`: the host reads or
    /// writes them next, and nothing else on the device touches them.
    pub fn settle_host_slots(&mut self, slots: &[u32]) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            if swap.host.iter().any(|slot| slots.contains(slot)) {
                swap.landed.settle()?;
            }
        }
        Ok(())
    }

    /// Orders `stream` behind the moves still landing that hold any of the
    /// device pages a fire names: its paged runs `(first, count)`.
    pub fn fence_swaps(
        &mut self,
        stream: *mut core::ffi::c_void,
        paged: &[(u64, u64)],
    ) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            let holds = swap.paged.iter().any(|&page| {
                let page = u64::from(page);
                paged
                    .iter()
                    .any(|&(first, count)| first <= page && page < first + count)
            });
            if holds || swap.wide {
                swap.landed.wait(stream)?;
            }
        }
        Ok(())
    }

    /// Where the next move between the host tier and the slot file is told
    /// how it ended, on the tier's landing thread, instead of waited out by
    /// the caller. Returns any tell no such move took since.
    pub fn defer_disk_tell(
        &mut self,
        told: Option<crate::serve::landing::Told>,
    ) -> Option<crate::serve::landing::Told> {
        std::mem::replace(&mut self.disk_told.0, told)
    }

    /// Waits, on the host, for every move still landing.
    pub fn settle_swaps(&mut self) -> Result<()> {
        for swap in self.swaps.drain(..) {
            swap.landed.settle()?;
        }
        Ok(())
    }

    /// Orders `stream` behind every move still landing: for work that reads
    /// or writes pages without naming them.
    pub fn fence_all_swaps(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            swap.landed.wait(stream)?;
        }
        Ok(())
    }

    pub(super) fn prune_swaps(&mut self) -> Result<()> {
        let mut kept = Vec::with_capacity(self.swaps.len());
        for swap in self.swaps.drain(..) {
            if !swap.landed.done()? {
                kept.push(swap);
            }
        }
        self.swaps = kept;
        Ok(())
    }

    /// How a host kv slot lays out its pages, rank by rank: each paged plane
    /// `[token][head][dim]`, a row's keys then its values. `None` with the
    /// compressor planes some models page beside their kv.
    #[must_use]
    pub fn host_kv_layout(&self, ranks: u32) -> Option<state::PageLayout> {
        if !self.pooled.is_empty() {
            return None;
        }
        let mut planes = Vec::new();
        for shape in &self.shapes {
            let Shape::Kv {
                windowed: false,
                dtype,
                keys_width,
                values_width,
                head_stride,
                ..
            } = *shape
            else {
                continue;
            };
            for width in [keys_width, values_width] {
                if head_stride == 0 || width % head_stride != 0 {
                    return None;
                }
                planes.push(state::Plane {
                    heads: u32::try_from(width / head_stride).ok()? * ranks,
                    head_dim: u32::try_from(head_stride).ok()?,
                    elem_bytes: elem_size(dtype),
                });
            }
        }
        Some(state::PageLayout {
            ranks,
            page_tokens: self.paging.page_size,
            planes,
        })
    }

    /// The cache row of each paged plane, in a host slot's order.
    pub(super) fn paged_plane_rows(&self) -> Vec<usize> {
        let mut rows = Vec::new();
        for (row, shape) in self.shapes.iter().enumerate() {
            if let Shape::Kv {
                windowed: false,
                values_plane,
                ..
            } = *shape
            {
                rows.extend(std::iter::repeat_n(row, values_plane + 1));
            }
        }
        rows
    }

    /// How a tap reads a host slot's planes out: a group a layer, each the
    /// planes of the kv rows `row_layer` puts in it, as `(planes, spans of a
    /// slot)`, in the layers' order. `None` where a tap cannot read a layer
    /// whole: windowed rows (named by their own pages), compressor slabs, or
    /// a kv row no layer writes.
    #[must_use]
    pub fn tap_groups(&self, row_layer: &[Option<u32>]) -> Option<Vec<TapGroup>> {
        let windowed = self
            .shapes
            .iter()
            .any(|shape| matches!(shape, Shape::Kv { windowed: true, .. }));
        if windowed || !self.pooled.is_empty() {
            return None;
        }
        let mut groups: std::collections::BTreeMap<u32, TapGroup> =
            std::collections::BTreeMap::new();
        let mut offset = 0u64;
        for (plane, (row, (_, bytes))) in self
            .paged_plane_rows()
            .into_iter()
            .zip(self.paged_planes())
            .enumerate()
        {
            let layer = (*row_layer.get(row)?)?;
            let group = groups.entry(layer).or_insert_with(|| TapGroup {
                layer,
                ..TapGroup::default()
            });
            group.planes.push(plane);
            group.spans.push((offset, bytes));
            offset += bytes;
        }
        Some(groups.into_values().collect())
    }

    /// For planes `planes`, device pages `device[i]` to host slots
    /// `host[i]`.
    pub fn tap_spans(
        &mut self,
        planes: &[usize],
        device: &[u32],
        host: &[u32],
    ) -> Result<TapSpans> {
        self.ensure_kv_pages(device)?;
        let Some(tier) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what: "host kv pages",
                need: 1,
                have: 0,
            });
        };
        tier.paged.back(host.iter().copied(), "host kv pages")?;
        let shelf = &self.host.as_ref().expect("seated above").paged;
        let all = self.paged_planes();
        let mut copies = Vec::with_capacity(planes.len() * device.len());
        let mut lands = Vec::with_capacity(planes.len() * device.len());
        for (&d, &h) in device.iter().zip(host) {
            for &plane in planes {
                let (at, bytes) = all[plane];
                let on_host = shelf.at(h, plane)?;
                let on_device = self.plane_span(at, d, bytes)?;
                let len = usize::try_from(bytes).unwrap_or(0);
                copies.push((on_host, on_device, len));
                lands.push((on_host, len));
            }
        }
        Ok(TapSpans { copies, lands })
    }

    /// Host kv slots `slots`, one after another, each its planes in order.
    /// The caller has synchronized every copy into them.
    pub fn read_host_kv(&mut self, slots: &[u32]) -> Result<Vec<u8>> {
        Ok(read_shelf(self.host_shelf(slots)?, slots))
    }

    /// Writes `bytes`, as [`Self::read_host_kv`] lays them out, into host kv
    /// slots `slots`.
    pub fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        write_shelf(
            self.host_shelf(slots)?,
            slots,
            bytes,
            "bytes of host kv slots",
        )
    }

    /// The host kv shelf, with the chunks holding `slots` pinned.
    pub(super) fn host_shelf(&mut self, slots: &[u32]) -> Result<&Shelf> {
        let what = "host kv pages";
        let Some(host) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what,
                need: 1,
                have: 0,
            });
        };
        host.paged.back(slots.iter().copied(), what)?;
        Ok(&host.paged)
    }
}
