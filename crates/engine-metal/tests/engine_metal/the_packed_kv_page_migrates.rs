#![cfg(target_vendor = "apple")]

//! C2d belt — a packed `KvU4` page MIGRATES correctly through `store::copy_kv`.
//!
//! `copy_kv` (page defrag / fork) sizes a packed page's cell stride from the
//! head_dim (`head_dim/2 + 2` bytes per head, NOT `width * element`), so a
//! token-run blit moves the codes and their inline scales together. This
//! drives the REAL store: reserve a packed `KvU4` chunk pool, quantize K/V into
//! one list slot with the production `attention.kv_append`, migrate that slot's
//! tokens into another with `Pools::copy_kv`, and assert the destination slot
//! is byte-for-byte the source — its key run and its value run. A stride
//! computed as `width * element` (512 B/head at head_dim 256, not 130) would
//! blit the wrong ranges and the two slots would differ.
//!
//! We run it at head_dim 256 AND 128 — the two blocks the packed codec ships.

use engine::transfer::KvRun;
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use engine_metal::run::CachePool;
use engine_metal::store::kv::{Facts, Paging, SpaceFacts};
use engine_metal::store::layout::{LIST, Layout};
use engine_metal::store::{Pools, Rows, Seats, SpaceSeat};
use kernels_metal::Tensor;
use poem_compiler::chunk::{ChunkProfile, ChunkRecipe, RowId, SlotLayout, TokenRun};
use poem_ir::{CacheRow, Dtype, Platform, Trace};

const HEADS: usize = 2; // kv_heads
const PAGE_SIZE: u32 = 8;
const PAGES: u64 = 4;

/// Packed bytes for one head at codec block `block`: `block/2` nibble bytes plus
/// one inline fp16 scale.
fn packed_bytes(block: usize) -> usize {
    block / 2 + 2
}

// --- deterministic outlier-heavy KV data (same mixer as the C2a/C2b tests) ----
fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E5E_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit01(at: u64) -> f64 {
    (f64::from(noise(at)) + 1.0) / (f64::from(u32::MAX) + 2.0)
}

fn gaussian(at: u64) -> f32 {
    let u1 = unit01(at);
    let u2 = unit01(at ^ 0x9999_5A5A_1357_2468);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// A `rows x width` plane of small Gaussians with a few big spikes per `block` —
/// the outlier shape a per-block absmax quantizer must survive.
fn outlier_plane(rows: usize, width: usize, block: usize, salt: u64) -> Vec<f32> {
    const SPIKES: usize = 3;
    let mut v: Vec<f32> = (0..(rows * width) as u64)
        .map(|i| gaussian(i ^ salt))
        .collect();
    for blk in 0..(rows * width) / block {
        let base = blk * block;
        for k in 0..SPIKES {
            let key =
                (blk as u64).wrapping_mul(0x100_0193) ^ (k as u64).wrapping_mul(0x9E37) ^ salt;
            let pos = (noise(key) as usize) % block;
            let mag = 20.0 + 20.0 * (unit01(key ^ 0xBEEF) as f32);
            let sign = if noise(key ^ 0xF00D) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            v[base + pos] = sign * mag;
        }
    }
    v
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&f| f32_to_bf16_bits(f).to_le_bytes())
        .collect()
}

// --- a small Metal rig (owns the buffers its handles index) ------------------
struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    keep: Vec<Buffer>,
}

impl Rig {
    fn open() -> Option<Self> {
        Some(Self {
            device: Context::bind().ok()?,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
            keep: Vec::new(),
        })
    }

    fn bf16(&mut self, data: &[f32]) -> u32 {
        let bytes = bf16_bytes(data);
        let mut b = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        b.write(0, &bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.push(b);
        h
    }

    fn u32s(&mut self, data: &[u32]) -> Tensor {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut b = Buffer::zeroed(&self.device, bytes.len().max(4) as u64).expect("a table");
        b.write(0, &bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.push(b);
        Tensor::new(h, data.len() as u32, 1, Dtype::U32)
    }
}

/// A minimal `Seats` whose page tables cover `PAGES` pages. `copy_kv` never
/// reads it, and the append path ignores the pool's own page tables — so
/// dummy-but-valid tensors suffice for `Pools::table` to hand back the pool.
fn seats(rig: &mut Rig) -> Seats {
    let page_indices = rig.u32s(&(0..PAGES as u32).collect::<Vec<_>>());
    let page_indptr = rig.u32s(&[0u32, PAGES as u32]);
    let last_page_lens = rig.u32s(&[PAGE_SIZE]);
    let row_valid = rig.u32s(&[1u32]);
    let slot_ids = rig.u32s(&[0u32]);
    let slot_of_row = rig.u32s(&[0u32]);
    Seats {
        lanes: 1,
        rows: 1,
        pages: PAGES as u32,
        spaces: vec![SpaceSeat {
            page_indptr,
            page_indices,
            last_page_lens,
            row_valid,
        }],
        slot_ids,
        slot_of_row,
        lane_bands: Vec::new(),
        row_bands: Vec::new(),
    }
}

#[test]
fn the_packed_kv_page_migrates() {
    eprintln!("=== C2d belt: a packed KvU4 page migrates byte-exact through copy_kv ===");
    let Some(mut rig) = Rig::open() else {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    };
    for &head_dim in &[256usize, 128] {
        migrate_case(&mut rig, head_dim);
    }
    eprintln!("=== C2d belt done: packed pages migrate at head_dim 128 & 256 ===");
}

fn migrate_case(rig: &mut Rig, head_dim: usize) {
    let block = head_dim;
    let per = packed_bytes(block);
    let width = (HEADS * head_dim) as u64;
    let paging = Paging::of(PAGE_SIZE, PAGE_SIZE, 1, PAGES).expect("paging");

    // One packed KV cache row, a key half and a value half, each `width` wide,
    // carrying the head_dim so the store sizes it per head.
    let trace = Trace {
        facts: Default::default(),
        inputs: Vec::new(),
        name: String::from("kv_migrate_probe"),
        platform: Platform::Metal,
        params: Vec::new(),
        caches: vec![CacheRow::Kv {
            name: String::from("kv"),
            planes: vec![width, width],
            dtype: Dtype::KvU4,
            space: 0,
            window: None,
            head_dim: head_dim as u32,
            live: None,
            shard: Default::default(),
            rows: None,
        }],
        values: Vec::new(),
        nodes: Vec::new(),
        outputs: Vec::new(),
        drafter: None,
    };
    // Consumers state the seat (kv_heads of head_dim) so `reserve` sizes packed.
    let facts = Facts {
        rows: vec![Some(SpaceFacts {
            head_dim: head_dim as u32,
            kv_heads: HEADS as u32,
            q_heads: HEADS as u32,
            window: None,
        })],
        plans: Vec::new(),
    };

    // A pool of two chunks of four list slots: the null chunk, and chunk 1
    // whose first two slots the migration names.
    // The compiler's recipe sizes no packed row (KvU4 has no element), so
    // the test states the list slot a recipe would: one row, key run then
    // value run.
    let run = (PAGE_SIZE as usize * HEADS * per) as u64;
    let slot = (2 * run).next_multiple_of(256);
    let recipe = ChunkRecipe {
        chunk_bytes: 4 * slot,
        slots: vec![SlotLayout::List {
            tokens: PAGE_SIZE,
            bytes: slot,
            rows: vec![TokenRun {
                row: RowId::Cache(0),
                layer: None,
                offset: 0,
                per_token: 2 * (HEADS * per) as u64,
            }],
        }],
        atoms: Default::default(),
        shards: Default::default(),
        short_runs: Vec::new(),
        cow_sets: 1,
        waste: Default::default(),
        profile: ChunkProfile::ON_DEVICE_METAL,
        weights: Vec::new(),
        rs: Default::default(),
    };
    let rows = Rows::plan(&trace, &recipe, &facts, None).expect("plan packed KV");
    let layout = Layout::new(
        4 * slot,
        2,
        engine_metal::store::pool::segment_chunks(4 * slot),
        PAGE_SIZE,
        [slot, 0],
        1,
        0,
        0,
    )
    .expect("a layout");
    let mut pools = Pools::reserve(&rig.device, rows, layout, paging).expect("reserve packed KV");
    pools.commit_segments(1).expect("map the pool");
    let (src, dst) = (4u32, 5u32);
    let seats = seats(rig);
    let table = pools
        .table(&rig.handles, &seats, false)
        .expect("bind plane handles");
    let CachePool::Kv(pool_planes) = table.pools[0] else {
        panic!("the one cache row is a KV pool");
    };

    // Seed PAGE_SIZE tokens of bf16 K/V and quantize them into PHYSICAL page 0.
    let tokens = PAGE_SIZE as usize;
    let k_data = outlier_plane(tokens, HEADS * head_dim, block, 0xC2D0 ^ head_dim as u64);
    let v_data = outlier_plane(tokens, HEADS * head_dim, block, 0xC2DF ^ head_dim as u64);
    let hk = rig.bf16(&k_data);
    let hv = rig.bf16(&v_data);
    let kt = Tensor::new(hk, tokens as u32, width as u32, Dtype::Bf16);
    let vt = Tensor::new(hv, tokens as u32, width as u32, Dtype::Bf16);
    // token i -> the source slot, in-page offset i (the identity table: a
    // slot ref is its physical page).
    let wpt = rig.u32s(&vec![src; tokens]);
    let wot = rig.u32s(&(0..tokens as u32).collect::<Vec<_>>());

    {
        let frame = rig.device.frame().expect("a frame");
        let sink = Sink::new(&rig.device, &frame, &rig.pipelines, &rig.handles);
        kernels_metal::attn::kv_append(&sink, kt, vt, &pool_planes, wpt, wot)
            .expect("kv_append launch");
        frame.commit().expect("commit append");
    }

    // A packed run: PAGE_SIZE cells x HEADS heads x per, a key run and a
    // value run in a slot.
    let page_bytes = tokens * HEADS * per;
    let read_slot = |pools: &Pools, page: u32| -> Vec<u8> {
        pools.read_slot_bytes(LIST, page).expect("read a slot")
    };
    let before = read_slot(&pools, src);
    assert_eq!(before.len() as u64, slot);
    assert!(
        slot >= 2 * page_bytes as u64,
        "head_dim {head_dim}: a slot holds a packed key run and value run"
    );
    assert!(
        before.iter().any(|&b| b != 0),
        "head_dim {head_dim}: the source slot should hold the quantized codes"
    );
    assert!(
        read_slot(&pools, dst).iter().all(|&b| b == 0),
        "head_dim {head_dim}: the destination slot should start zeroed"
    );

    let moves = [KvRun {
        src_page: src,
        src_token: 0,
        dst_page: dst,
        dst_token: 0,
        tokens: PAGE_SIZE,
    }];
    {
        let mut frame = rig.device.frame().expect("a frame");
        pools.copy_kv(&mut frame, &moves).expect("copy_kv");
        frame.commit().expect("commit migrate");
    }

    // The destination is byte-for-byte the source — codes AND the inline
    // fp16 scales moved together, at the packed stride.
    assert_eq!(
        read_slot(&pools, dst),
        before,
        "head_dim {head_dim}: the migrated slot differs from the source — packed stride is wrong"
    );
    assert_eq!(
        read_slot(&pools, src),
        before,
        "head_dim {head_dim}: the source slot was disturbed by the migration"
    );

    eprintln!(
        "[migrate] head_dim={head_dim:>3} | page = {page_bytes} B/plane ({HEADS} heads x {per} B) | \
         slot {src} -> slot {dst} byte-exact for K and V"
    );
}
