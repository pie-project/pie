//! The device channels' epilogue against a host model of the rings: the
//! pull (tickets checked against host words, host-written cells pulled),
//! the commit bump, the publish back to the host mirrors and the settle of
//! the words, at 1, 7, 128 and 512 lanes of four tickets each, refusals
//! included.

#![cfg(feature = "rocm")]

mod common;

use common::Gpu;
use kernels_rocm::hip::{self, rt};
use kernels_rocm::runtime::channel::{
    self, BumpLane, MAX_RING, PublishLane, PullLane, SettleLane, Ticket,
};

/// Host memory the device maps, as the engine's endpoints hold their words
/// and mirrors.
struct Pinned {
    host: *mut u8,
    device: u64,
    bytes: usize,
}

impl Pinned {
    fn new(bytes: usize) -> Self {
        let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
        let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: a fresh mapped allocation and its device alias.
        unsafe {
            assert_eq!(
                rt::hipHostMalloc(
                    &raw mut host,
                    bytes,
                    rt::hipHostMallocMapped | rt::hipHostMallocPortable
                ),
                0
            );
            assert_eq!(rt::hipHostGetDevicePointer(&raw mut device, host, 0), 0);
            core::ptr::write_bytes(host.cast::<u8>(), 0, bytes);
        }
        Self {
            host: host.cast(),
            device: device as u64,
            bytes,
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the allocation is `bytes` long and this test's alone.
        unsafe { core::slice::from_raw_parts(self.host, self.bytes) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, borrowed mutably through `self`.
        unsafe { core::slice::from_raw_parts_mut(self.host, self.bytes) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by `hipHostMalloc` above, freed once.
        unsafe { rt::hipHostFree(self.host.cast()) };
    }
}

/// One ticket's channel: its ring capacity, cell bytes, flags.
#[derive(Clone, Copy)]
struct Shape {
    flags: u32,
    cap1: u32,
    native: u32,
}

/// The Qwen3.8-Flash-Next decode's ten tickets a lane (host-written
/// inputs, four advancing their head, a device-only consume, the sampled
/// token out), and two bit-packed bool rows, one each way.
const SHAPES: [Shape; 12] = [
    shape(INPUT | Ticket::ADVANCE_HEAD, 92),
    shape(INPUT, 8),
    shape(INPUT | Ticket::ADVANCE_HEAD, 92),
    shape(INPUT | Ticket::ADVANCE_HEAD, 92),
    shape(INPUT | Ticket::ADVANCE_HEAD, 92),
    shape(INPUT, 4),
    shape(INPUT, 24),
    shape(INPUT, 8),
    shape(
        Ticket::CONSUME | Ticket::REQUIRE_INPUT | Ticket::ADVANCE_HEAD,
        8,
    ),
    shape(OUTPUT, 4),
    shape(OUTPUT | Ticket::PACKED_BOOL, 12),
    shape(
        Ticket::CONSUME | Ticket::HOST_WRITER | Ticket::PACKED_BOOL | Ticket::ADVANCE_HEAD,
        40,
    ),
];

const INPUT: u32 = Ticket::CONSUME | Ticket::HOST_WRITER | Ticket::REQUIRE_INPUT;
const OUTPUT: u32 = Ticket::PUBLISH | Ticket::HOST_READER | Ticket::ADVANCE_TAIL;

const fn shape(flags: u32, native: u32) -> Shape {
    Shape {
        flags,
        cap1: 2 + native % 3,
        native,
    }
}

/// Each ticket's words and mirror on a page of its own, as each endpoint
/// allocates them.
const PAGE: usize = 4096;

fn wire(shape: Shape) -> u32 {
    if shape.flags & Ticket::PACKED_BOOL != 0 {
        shape.native.div_ceil(8)
    } else {
        shape.native
    }
}

/// Every lane's rings, words, mirrors and staged tables.
struct Rig {
    lanes: usize,
    words: Pinned,
    mirror: Pinned,
    commits: Pinned,
    cells: hip::Buffer,
    registry: hip::Buffer,
    tables: hip::Buffer,
    tickets: Vec<Ticket>,
    pull: Vec<PullLane>,
    bump: Vec<BumpLane>,
    taken: Vec<u32>,
    put: Vec<u32>,
    at: [usize; 7],
    /// The registry's initial head, tail and full bytes.
    head0: Vec<u32>,
    tail0: Vec<u32>,
    full0: Vec<u8>,
    cells0: Vec<u8>,
}

const SLOTS: usize = SHAPES.len();

fn registry_bytes() -> usize {
    3 * SLOTS * 4 + SLOTS * MAX_RING as usize
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed >> 33
}

impl Rig {
    /// `lanes` lanes, lane i refused when `refuse(i)` (its consume ticket
    /// expects a head the ring is not at).
    fn new(lanes: usize, refuse: impl Fn(usize) -> bool) -> Self {
        let mut seed = 0x5eed_u64 + lanes as u64;
        let mut words = Pinned::new(lanes * SLOTS * PAGE);
        let mut mirror = Pinned::new(lanes * SLOTS * PAGE);
        let commits = Pinned::new(lanes * PAGE);
        let cell_pitch = 128 * MAX_RING as usize;
        let mut cells0 = vec![0u8; lanes * SLOTS * cell_pitch];
        for b in &mut cells0 {
            *b = lcg(&mut seed) as u8;
        }
        let cells = hip::Buffer::from_slice(&cells0).unwrap();
        let mut reg = vec![0u8; lanes * registry_bytes()];
        let (mut head0, mut tail0, mut full0) = (Vec::new(), Vec::new(), Vec::new());
        for lane in 0..lanes {
            let base = lane * registry_bytes();
            for (s, shape) in SHAPES.iter().enumerate() {
                let head = (lcg(&mut seed) % u64::from(shape.cap1)) as u32;
                let tail = (lcg(&mut seed) % u64::from(shape.cap1)) as u32;
                reg[base + s * 4..base + s * 4 + 4].copy_from_slice(&head.to_le_bytes());
                reg[base + (SLOTS + s) * 4..base + (SLOTS + s) * 4 + 4]
                    .copy_from_slice(&tail.to_le_bytes());
                reg[base + (2 * SLOTS + s) * 4..base + (2 * SLOTS + s) * 4 + 4]
                    .copy_from_slice(&shape.cap1.to_le_bytes());
                head0.push(head);
                tail0.push(tail);
            }
            for r in 0..SLOTS * MAX_RING as usize {
                let b = (lcg(&mut seed) & 1) as u8;
                reg[base + 3 * SLOTS * 4 + r] = b;
                full0.push(b);
            }
        }
        let registry = hip::Buffer::from_slice(&reg).unwrap();

        // The host side: words a sequence each, mirrors filled.
        for b in mirror.bytes_mut().iter_mut() {
            *b = lcg(&mut seed) as u8;
        }
        let (mut tickets, mut pull, mut bump, mut publish, mut settle) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut taken, mut put) = (Vec::new(), Vec::new());
        let mut spans = Vec::new();
        for lane in 0..lanes {
            let offset = tickets.len() as u32;
            let (taken_at, put_at) = (taken.len(), put.len());
            for (s, shape) in SHAPES.iter().enumerate() {
                let word_at = (lane * SLOTS + s) * PAGE;
                let head = 10 + lcg(&mut seed) % 50;
                let consume = shape.flags & Ticket::CONSUME != 0;
                let tail = if consume {
                    head + 1
                } else {
                    head + lcg(&mut seed) % 2
                };
                let w = &mut words.bytes_mut()[word_at..word_at + 16];
                w[..8].copy_from_slice(&head.to_le_bytes());
                w[8..].copy_from_slice(&tail.to_le_bytes());
                let expected_head = if consume && refuse(lane) {
                    head + 1
                } else {
                    head
                };
                tickets.push(Ticket {
                    slot: s as u32,
                    flags: shape.flags,
                    expected_head: if consume {
                        expected_head
                    } else {
                        channel::NO_TICKET
                    },
                    expected_tail: if consume { channel::NO_TICKET } else { tail },
                    words: words.device + word_at as u64,
                    mirror: mirror.device + ((lane * SLOTS + s) * PAGE) as u64,
                    cells: cells.device_ptr() + ((lane * SLOTS + s) * cell_pitch) as u64,
                    cap1: shape.cap1,
                    wire_bytes: wire(*shape),
                    native_bytes: shape.native,
                });
                if consume {
                    taken.push(s as u32);
                } else {
                    put.push(s as u32);
                }
            }
            let commit = commits.device + (lane * PAGE) as u64;
            let reg = registry.device_ptr() + (lane * registry_bytes()) as u64;
            let rings = channel::Rings::new(
                reg + 3 * SLOTS as u64 * 4,
                reg,
                reg + SLOTS as u64 * 4,
                reg + 2 * SLOTS as u64 * 4,
                SLOTS as u32,
            );
            pull.push(PullLane {
                full: rings.full,
                pass_commit: commit,
                ticket_offset: offset,
                ticket_count: SLOTS as u32,
                initial_commit: 1,
                diagnose: 0,
            });
            spans.push((taken_at, taken.len() - taken_at, put_at, put.len() - put_at));
            bump.push(rings.bump_lane(0, 0, 0, 0, commit));
            publish.push(PublishLane {
                commit,
                ticket_offset: offset,
                ticket_count: SLOTS as u32,
            });
            settle.push(SettleLane {
                commit,
                ticket_offset: offset,
                ticket_count: SLOTS as u32,
            });
        }
        let mut at = [0usize; 7];
        let mut size = 0usize;
        let sizes = [
            tickets.len() * size_of::<Ticket>(),
            taken.len() * 4,
            put.len() * 4,
            pull.len() * size_of::<PullLane>(),
            bump.len() * size_of::<BumpLane>(),
            publish.len() * size_of::<PublishLane>(),
            settle.len() * size_of::<SettleLane>(),
        ];
        for (i, s) in sizes.iter().enumerate() {
            at[i] = size;
            size += s.div_ceil(8) * 8;
        }
        let tables = hip::Buffer::alloc(size.max(8)).unwrap();
        for (lane, &(ta, tl, pa, pl)) in spans.iter().enumerate() {
            bump[lane].taken = tables.device_ptr() + (at[1] + ta * 4) as u64;
            bump[lane].taken_count = tl as u32;
            bump[lane].put = tables.device_ptr() + (at[2] + pa * 4) as u64;
            bump[lane].put_count = pl as u32;
        }
        let mut image = vec![0u8; size.max(8)];
        let mut place =
            |i: usize, bytes: &[u8]| image[at[i]..at[i] + bytes.len()].copy_from_slice(bytes);
        place(0, as_bytes(&tickets));
        place(1, as_bytes(&taken));
        place(2, as_bytes(&put));
        place(3, as_bytes(&pull));
        place(4, as_bytes(&bump));
        place(5, as_bytes(&publish));
        place(6, as_bytes(&settle));
        // SAFETY: a device allocation of `image.len()` bytes and a host slice.
        unsafe {
            assert_eq!(
                rt::hipMemcpy(
                    tables.device_ptr() as *mut _,
                    image.as_ptr().cast(),
                    image.len(),
                    rt::hipMemcpyHostToDevice
                ),
                0
            );
        }
        Self {
            lanes,
            words,
            mirror,
            commits,
            cells,
            registry,
            tables,
            tickets,
            pull,
            bump,
            taken,
            put,
            at,
            head0,
            tail0,
            full0,
            cells0,
        }
    }

    fn table(&self, i: usize) -> u64 {
        self.tables.device_ptr() + self.at[i] as u64
    }

    fn pull_validate(&self, gpu: &Gpu) {
        channel::pull_validate(&gpu.ctx(), self.table(0), self.table(3), self.lanes as u32)
            .unwrap();
    }

    fn land(&self, gpu: &Gpu) {
        let ctx = gpu.ctx();
        let n = self.lanes as u32;
        channel::commit_bump(&ctx, self.table(4), n).unwrap();
        channel::scatter_publish(&ctx, self.table(0), self.table(5), n).unwrap();
        channel::settle(&ctx, self.table(0), self.table(6), n).unwrap();
    }
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data views of the staged tables.
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// The epilogue on the host, from the rig's initial state: the words, the
/// mirrors, the commit words, the registry and the device cells it leaves.
struct Model {
    words: Vec<u8>,
    mirror: Vec<u8>,
    commits: Vec<u32>,
    head: Vec<u32>,
    tail: Vec<u32>,
    full: Vec<u8>,
    cells: Vec<u8>,
}

fn model(rig: &Rig, words: &[u8], mirror: &[u8]) -> Model {
    let mut m = Model {
        words: words.to_vec(),
        mirror: mirror.to_vec(),
        commits: vec![0; rig.lanes],
        head: rig.head0.clone(),
        tail: rig.tail0.clone(),
        full: rig.full0.clone(),
        cells: rig.cells0.clone(),
    };
    let cells_base = rig.cells.device_ptr();
    let mirror_base = rig.mirror.device;
    let words_base = rig.words.device;
    let fullb = SLOTS * MAX_RING as usize;
    for lane in 0..rig.lanes {
        let p = rig.pull[lane];
        let tickets =
            &rig.tickets[p.ticket_offset as usize..(p.ticket_offset + p.ticket_count) as usize];
        // Pull.
        let mut commit = p.initial_commit;
        for t in tickets {
            let wa = (t.words - words_base) as usize;
            let (head, tail) = (u64_at(words, wa), u64_at(words, wa + 8));
            let mut ok = true;
            if t.flags & Ticket::CONSUME != 0 {
                ok = head == t.expected_head;
            }
            if t.flags & Ticket::REQUIRE_INPUT != 0 {
                ok = ok && tail > head;
            }
            if t.flags & Ticket::PUBLISH != 0 {
                let same = u64::from(t.flags & Ticket::CONSUME != 0);
                ok = ok && tail == t.expected_tail && tail - head < u64::from(t.cap1 - 1) + same;
            }
            if !ok {
                commit = 0;
                continue;
            }
            if t.flags & Ticket::HOST_WRITER == 0 || t.flags & Ticket::CONSUME == 0 {
                continue;
            }
            let ring = (t.expected_head % u64::from(t.cap1)) as usize;
            let src = (t.mirror - mirror_base) as usize + ring * t.wire_bytes as usize;
            let dst = (t.cells - cells_base) as usize + ring * t.native_bytes as usize;
            for i in 0..t.native_bytes as usize {
                m.cells[dst + i] = if t.flags & Ticket::PACKED_BOOL != 0 {
                    (mirror[src + i / 8] >> (i % 8)) & 1
                } else {
                    mirror[src + i]
                };
            }
            m.full[lane * fullb + t.slot as usize * MAX_RING as usize + ring] = 1;
        }
        m.commits[lane] = commit;
        if commit == 0 {
            continue;
        }
        // Bump.
        let b = rig.bump[lane];
        let span = |ptr: u64, at: usize, n: u32, list: &[u32]| -> Vec<u32> {
            let first = ((ptr - rig.table(at)) / 4) as usize;
            list[first..first + n as usize].to_vec()
        };
        for s in span(b.put, 2, b.put_count, &rig.put) {
            let k = lane * SLOTS + s as usize;
            m.full[lane * fullb + s as usize * MAX_RING as usize + m.tail[k] as usize] = 1;
            m.tail[k] = (m.tail[k] + 1) % SHAPES[s as usize].cap1;
        }
        for s in span(b.taken, 1, b.taken_count, &rig.taken) {
            let k = lane * SLOTS + s as usize;
            m.full[lane * fullb + s as usize * MAX_RING as usize + m.head[k] as usize] = 0;
            m.head[k] = (m.head[k] + 1) % SHAPES[s as usize].cap1;
        }
        // Publish, from the cells the pull left.
        for t in tickets {
            let out = Ticket::PUBLISH | Ticket::HOST_READER;
            if t.flags & out != out || t.flags & Ticket::SHADOW != 0 {
                continue;
            }
            let ring = (t.expected_tail % u64::from(t.cap1)) as usize;
            let src = (t.cells - cells_base) as usize + ring * t.native_bytes as usize;
            let dst = (t.mirror - mirror_base) as usize + ring * t.wire_bytes as usize;
            for i in 0..t.wire_bytes as usize {
                m.mirror[dst + i] = if t.flags & Ticket::PACKED_BOOL != 0 {
                    let mut packed = 0u8;
                    for bit in 0..8 {
                        if i * 8 + bit < t.native_bytes as usize && m.cells[src + i * 8 + bit] != 0
                        {
                            packed |= 1 << bit;
                        }
                    }
                    packed
                } else {
                    m.cells[src + i]
                };
            }
        }
        // Settle.
        for t in tickets {
            let wa = (t.words - words_base) as usize;
            if t.flags & Ticket::ADVANCE_HEAD != 0 {
                m.words[wa..wa + 8].copy_from_slice(&(t.expected_head + 1).to_le_bytes());
            }
            if t.flags & Ticket::ADVANCE_TAIL != 0 {
                m.words[wa + 8..wa + 16].copy_from_slice(&(t.expected_tail + 1).to_le_bytes());
            }
        }
    }
    m
}

fn check(lanes: usize) {
    let Some(gpu) = Gpu::open() else { return };
    let rig = Rig::new(lanes, |lane| lane % 5 == 3);
    let (words, mirror) = (rig.words.bytes().to_vec(), rig.mirror.bytes().to_vec());
    let want = model(&rig, &words, &mirror);
    rig.pull_validate(&gpu);
    rig.land(&gpu);
    gpu.sync();
    let commits: Vec<u32> = (0..lanes)
        .map(|l| {
            u32::from_le_bytes(
                rig.commits.bytes()[l * PAGE..l * PAGE + 4]
                    .try_into()
                    .unwrap(),
            )
        })
        .collect();
    assert_eq!(commits, want.commits, "{lanes} lanes: the commit words");
    assert_eq!(
        rig.words.bytes(),
        &want.words[..],
        "{lanes} lanes: the settled words"
    );
    assert_eq!(
        rig.mirror.bytes(),
        &want.mirror[..],
        "{lanes} lanes: the published mirrors"
    );
    assert_eq!(
        rig.cells.read::<u8>().unwrap(),
        want.cells,
        "{lanes} lanes: the pulled cells"
    );
    let reg = rig.registry.read::<u8>().unwrap();
    for lane in 0..lanes {
        let base = lane * registry_bytes();
        let word =
            |i: usize| u32::from_le_bytes(reg[base + i * 4..base + i * 4 + 4].try_into().unwrap());
        for s in 0..SLOTS {
            assert_eq!(
                word(s),
                want.head[lane * SLOTS + s],
                "lane {lane} slot {s}: head"
            );
            assert_eq!(
                word(SLOTS + s),
                want.tail[lane * SLOTS + s],
                "lane {lane} slot {s}: tail"
            );
        }
        let full = &reg[base + 3 * SLOTS * 4..base + registry_bytes()];
        let fullb = SLOTS * MAX_RING as usize;
        assert_eq!(
            full,
            &want.full[lane * fullb..(lane + 1) * fullb],
            "lane {lane}: full bytes"
        );
    }
}

#[test]
fn the_epilogue_matches_the_host_rings() {
    for lanes in [1, 7, 128, 512] {
        check(lanes);
    }
}
