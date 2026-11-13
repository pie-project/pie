//! A packed pool's pages laid out again as a state chunk lays them: each
//! page one struct of several layers' planes (layer-major), the structs a
//! stride apart with bytes of garbage between, their order shuffled. A
//! kernel handed the spread pool (its planes at the middle layer's offsets,
//! its page tables renamed) lands what it lands over the packed one, byte
//! for byte; handed the first layer's offsets instead, it does not.

use super::{Gpu, Lcg};
use kernels_cuda::tensor::{KvPool, Tensor};

/// Layers a page's struct holds; the pool under test is layer 1's.
const LAYERS: u64 = 3;

/// Garbage bytes past each page's struct.
const GAP: u64 = 256;

/// Each plane's page rounded up to this in the struct.
const ALIGN: u64 = 256;

/// A pool's planes: each distinct plane's pointer and the bytes a page of it
/// holds.
fn planes_of(pool: &KvPool) -> Vec<(u64, u64)> {
    let mut planes: Vec<(u64, u64)> = Vec::new();
    for (plane, stride) in [pool.keys, pool.values].into_iter().zip(pool.page_stride) {
        if plane.ptr != 0 && stride != 0 && !planes.iter().any(|&(at, _)| at == plane.ptr) {
            planes.push((plane.ptr, stride));
        }
    }
    planes
}

/// Pages a packed plane holds.
fn pages_of(pool: &KvPool) -> usize {
    let bytes = u64::from(pool.keys.rows)
        * (u64::from(pool.keys.width) * pool.keys.dtype.bits()).div_ceil(8);
    usize::try_from(bytes / pool.page_stride[0].max(1)).unwrap_or(0)
}

/// One pool's pages spread through one buffer.
struct Spread {
    base: u64,
    image: Vec<u8>,
    stride: u64,
    /// Each plane's offset in a page's struct, by layer.
    offsets: Vec<[u64; LAYERS as usize]>,
    planes: Vec<(u64, u64)>,
    /// The packed pool's page `i` is the spread one's `map[i]`.
    map: Vec<u32>,
}

impl Spread {
    fn of(gpu: &mut Gpu, pool: &KvPool, lcg: &mut Lcg) -> Self {
        let planes = planes_of(pool);
        let pages = pages_of(pool);
        let mut offsets = vec![[0; LAYERS as usize]; planes.len()];
        let mut at = 0;
        for layer in 0..LAYERS as usize {
            for (plane, &(_, bytes)) in offsets.iter_mut().zip(&planes) {
                plane[layer] = at;
                at += bytes.next_multiple_of(ALIGN);
            }
        }
        let stride = at + GAP;
        // Twice the pages and a few, the packed ones dealt among them.
        let spread = pages * 2 + 3;
        let mut order: Vec<u32> = (0..spread as u32).collect();
        for i in (1..order.len()).rev() {
            let j = (lcg.unit().abs() * (i + 1) as f32) as usize % (i + 1);
            order.swap(i, j);
        }
        let map = order[..pages].to_vec();
        let total = usize::try_from(stride).unwrap_or(0) * spread;
        // Garbage: a xorshift stream, eight bytes a step.
        let mut x = u64::from(lcg.unit().to_bits()) | 1;
        let mut image: Vec<u8> = Vec::with_capacity(total + 8);
        while image.len() < total {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            image.extend_from_slice(&x.to_le_bytes());
        }
        image.truncate(total);
        for (p, &(ptr, bytes)) in planes.iter().enumerate() {
            let packed: Vec<u8> = gpu.down(ptr, bytes as usize * pages);
            for (page, &to) in map.iter().enumerate() {
                let dst = (u64::from(to) * stride + offsets[p][1]) as usize;
                let src = page * bytes as usize;
                image[dst..dst + bytes as usize]
                    .copy_from_slice(&packed[src..src + bytes as usize]);
            }
        }
        let base = gpu.up(&image);
        Self {
            base,
            image,
            stride,
            offsets,
            planes,
            map,
        }
    }

    /// `pool` over the spread pages, its planes at `layer`'s offsets.
    fn pool(&self, gpu: &mut Gpu, pool: &KvPool, layer: usize) -> KvPool {
        let at = |plane: Tensor| -> Tensor {
            match self.planes.iter().position(|&(ptr, _)| ptr == plane.ptr) {
                Some(p) if plane.ptr != 0 => Tensor {
                    ptr: self.base + self.offsets[p][layer],
                    ..plane
                },
                _ => plane,
            }
        };
        KvPool {
            keys: at(pool.keys),
            values: at(pool.values),
            page_indices: self.remap(gpu, pool.page_indices),
            page_stride: [self.stride; 2],
            ..*pool
        }
    }

    /// A table of the packed pool's page ids (i32 or u32) renamed.
    fn remap(&self, gpu: &mut Gpu, table: Tensor) -> Tensor {
        if table.ptr == 0 {
            return table;
        }
        let ids: Vec<u32> = gpu.down(table.ptr, table.elements() as usize);
        let renamed: Vec<u32> = ids
            .iter()
            .map(|&id| self.map.get(id as usize).copied().unwrap_or(id))
            .collect();
        Tensor {
            ptr: gpu.up(&renamed),
            ..table
        }
    }

    /// The spread pool's planes read back packed, and every byte outside
    /// them checked as it was.
    fn gathered(&self, gpu: &Gpu, what: &str) -> Vec<Vec<u8>> {
        let now: Vec<u8> = gpu.down(self.base, self.image.len());
        let mut kept = self.image.clone();
        let mut planes = Vec::new();
        for (p, &(_, bytes)) in self.planes.iter().enumerate() {
            let mut packed = vec![0u8; bytes as usize * self.map.len()];
            for (page, &to) in self.map.iter().enumerate() {
                let src = (u64::from(to) * self.stride + self.offsets[p][1]) as usize;
                let range = src..src + bytes as usize;
                packed[page * bytes as usize..(page + 1) * bytes as usize]
                    .copy_from_slice(&now[range.clone()]);
                kept[range.clone()].copy_from_slice(&now[range]);
            }
            planes.push(packed);
        }
        assert!(
            now == kept,
            "{what}: a write landed outside the spread pool's own rows"
        );
        planes
    }
}

/// A plane's packed bytes.
fn packed(gpu: &Gpu, pool: &KvPool) -> Vec<Vec<u8>> {
    let pages = pages_of(pool);
    planes_of(pool)
        .into_iter()
        .map(|(ptr, bytes)| gpu.down(ptr, bytes as usize * pages))
        .collect()
}

/// A fire over the pools and their write tables.
type Fire<'a> = dyn FnMut(&mut Gpu, &[KvPool], &[Vec<Tensor>]) + 'a;

/// Where a fire lands: `(at, bytes)`, set to a fill before each fire.
pub type Out = (u64, usize);

/// Runs `fire` over `pools` packed, then over their pages spread (each a
/// `Spread`, its planes at the middle layer's offsets and every page table
/// in `tables[i]` renamed), and checks the two land the same bytes in
/// `outs` and in the pools' planes; then over the spread pages at the first
/// layer's offsets, and checks that does not. `tables[i]` are the extra
/// page-id tables pool `i`'s pages are named by (a write's pages); `fire`
/// gets the pools and those tables.
pub fn agrees(
    gpu: &mut Gpu,
    what: &str,
    pools: &[KvPool],
    tables: &[Vec<Tensor>],
    outs: &[Out],
    mut fire: impl FnMut(&mut Gpu, &[KvPool], &[Vec<Tensor>]),
) {
    for pool in pools {
        assert!(pool.packed(), "{what}: the pool under test starts packed");
    }
    let mut lcg = Lcg::seeded(0x5_7a7e_c4ac);
    // Both spreads off the pools as they are, before any fire writes them.
    let good: Vec<Spread> = pools.iter().map(|p| Spread::of(gpu, p, &mut lcg)).collect();
    let bad: Vec<Spread> = pools.iter().map(|p| Spread::of(gpu, p, &mut lcg)).collect();
    let before: Vec<Vec<Vec<u8>>> = pools.iter().map(|p| packed(gpu, p)).collect();
    let run = |gpu: &mut Gpu, pools: &[KvPool], tables: &[Vec<Tensor>], fire: &mut Fire<'_>| {
        for &(at, bytes) in outs {
            gpu.fill(at, 0x5a, bytes);
        }
        fire(gpu, pools, tables);
        gpu.sync();
        outs.iter()
            .map(|&(at, bytes)| gpu.down::<u8>(at, bytes))
            .collect::<Vec<_>>()
    };
    let landed = run(gpu, pools, tables, &mut fire);
    let planes: Vec<Vec<Vec<u8>>> = pools.iter().map(|p| packed(gpu, p)).collect();
    let wrote = planes != before;
    assert!(
        wrote || landed.iter().any(|o| !o.is_empty()),
        "{what}: the fire lands nothing to compare"
    );

    let spread = |gpu: &mut Gpu, spreads: &[Spread], layer: usize| {
        let pools: Vec<KvPool> = spreads
            .iter()
            .zip(pools)
            .map(|(s, p)| s.pool(gpu, p, layer))
            .collect();
        let tables: Vec<Vec<Tensor>> = spreads
            .iter()
            .zip(tables.iter().chain(core::iter::repeat(&Vec::new())))
            .map(|(s, t)| t.iter().map(|&t| s.remap(gpu, t)).collect())
            .collect();
        (pools, tables)
    };
    let (pools_s, tables_s) = spread(gpu, &good, 1);
    let landed_s = run(gpu, &pools_s, &tables_s, &mut fire);
    for (i, (a, b)) in landed.iter().zip(&landed_s).enumerate() {
        assert!(
            a == b,
            "{what}: output {i} over the spread pages is not the packed pool's, byte for byte"
        );
    }
    for (s, (p, want)) in good.iter().zip(planes.iter().enumerate()) {
        assert!(
            &s.gathered(gpu, what) == want,
            "{what}: pool {p}'s planes written over the spread pages are not the packed pool's"
        );
    }

    let (pools_w, tables_w) = spread(gpu, &bad, 0);
    let landed_w = run(gpu, &pools_w, &tables_w, &mut fire);
    let planes_w: Vec<Vec<Vec<u8>>> = bad
        .iter()
        .map(|s| {
            let now: Vec<u8> = gpu.down(s.base, s.image.len());
            let mut out = Vec::new();
            for (p, &(_, bytes)) in s.planes.iter().enumerate() {
                let mut packed = vec![0u8; bytes as usize * s.map.len()];
                for (page, &to) in s.map.iter().enumerate() {
                    let src = (u64::from(to) * s.stride + s.offsets[p][1]) as usize;
                    packed[page * bytes as usize..(page + 1) * bytes as usize]
                        .copy_from_slice(&now[src..src + bytes as usize]);
                }
                out.push(packed);
            }
            out
        })
        .collect();
    assert!(
        landed_w != landed || (wrote && planes_w != planes),
        "{what}: a wrong plane offset landed the packed pool's bytes too, so the check checks nothing"
    );
}
