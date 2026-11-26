use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::Frame;
use engine_gpu::guest::widen::{Cfg, TABLE_SOURCE, WG, config_bytes, word};

use engine_gpu::error::{Fault, Result};

/// A row of a pass's outputs to widen: `lanes` bf16 values at byte `src_at`
/// of `src` into f32 at byte `dst_at` of `dst`.
pub struct Row<'b> {
    pub src: &'b Buffer,
    pub src_at: u64,
    pub dst: &'b Buffer,
    pub dst_at: u64,
    pub lanes: u32,
}

/// The readout widen, all of it made at load: its pipeline, a table of
/// row configs split into one region per thread that runs guest passes,
/// and a bind group for every pair of a fire's output seat and a pool
/// segment a stage's heap lies in.
pub struct Widen {
    pipeline: wgpu::ComputePipeline,
    table: Buffer,
    /// Bytes of one region, and the rows it holds.
    region: u64,
    entries: u64,
    regions: u32,
    /// The regions handed to threads so far.
    taken: AtomicU32,
    align: u64,
    groups: HashMap<(usize, usize), wgpu::BindGroup>,
}

impl std::fmt::Debug for Widen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Widen")
    }
}

thread_local! {
    /// The table region this thread writes its widens' rows to, taken on its
    /// first widen.
    static REGION: Cell<Option<u32>> = const { Cell::new(None) };
}

fn key(buffer: &Buffer) -> usize {
    Arc::as_ptr(buffer.slab()) as usize
}

impl Widen {
    /// The bytes of the table a widen takes, with `regions` regions of
    /// `entries` rows each and the one beside them.
    pub fn table_bytes(device: &Context, regions: u32, entries: u64) -> u64 {
        Self::region_bytes(device, entries) * (u64::from(regions) + 1)
    }

    /// A table region's bytes: the rows of a frame's widens (every pass it
    /// carries, its lanes' together at most `entries`), each group of rows
    /// from an aligned start, and a group as few as one row.
    fn region_bytes(device: &Context, entries: u64) -> u64 {
        let align = u64::from(device.core().limits.min_storage_buffer_offset_alignment).max(16);
        entries * (size_of::<Cfg>() as u64).next_multiple_of(align)
    }

    /// The widen over `seats` (each arm's readout and pixel seats) into
    /// `segments` (the pool's), with `regions` table regions of `entries`
    /// rows each in `table` (the load's shelf reserved it: at least
    /// [`Widen::table_bytes`], refused by name where it is less).
    pub fn new(
        device: &Context,
        table: Buffer,
        seats: &[&Buffer],
        segments: &[&Buffer],
        regions: u32,
        entries: u64,
    ) -> Result<Widen> {
        let core = device.core().clone();
        let align = u64::from(core.limits.min_storage_buffer_offset_alignment).max(16);
        let region = Self::region_bytes(device, entries);
        let need = Self::table_bytes(device, regions, entries);
        if table.bytes() < need {
            return Err(Fault::Ceiling {
                what: "bytes of the readout widen's table",
                need,
                have: table.bytes(),
            });
        }

        let storage = |at: u32, read_only: bool, dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding: at,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let binds = core
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pie widen"),
                entries: &[
                    storage(0, true, false),
                    storage(1, false, false),
                    storage(2, true, true),
                ],
            });
        let layout = core
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pie widen"),
                bind_group_layouts: &[Some(&binds)],
                immediate_size: 0,
            });
        core.take_error("create_pipeline_layout")?;

        let pipeline = crate::guest::validated(&core, "create_compute_pipeline", || {
            let module = core
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("pie widen"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TABLE_SOURCE)),
                });
            core.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("pie widen"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        })
        .map_err(|error| Fault::Shader {
            file: "widen.wgsl",
            entrypoint: "main",
            why: format!("the readout widen was refused: {error}"),
        })?;

        let mut groups = HashMap::with_capacity(seats.len() * segments.len());
        for src in seats {
            for dst in segments {
                let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pie widen"),
                    layout: &binds,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: whole(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: whole(dst),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &table.slab().buffer,
                                offset: 0,
                                size: std::num::NonZeroU64::new(region),
                            }),
                        },
                    ],
                });
                core.take_error("create_bind_group")?;
                groups.insert((key(src), key(dst)), group);
            }
        }

        Ok(Widen {
            pipeline,
            table,
            region,
            entries,
            regions,
            taken: AtomicU32::new(0),
            align,
            groups,
        })
    }

    /// This thread's table region, taken on its first widen.
    fn region(&self) -> Result<u32> {
        if let Some(region) = REGION.get() {
            return Ok(region);
        }
        let region = self.taken.fetch_add(1, Ordering::Relaxed);
        if region >= self.regions {
            return Err(Fault::Ceiling {
                what: "threads widening a pass's outputs (the widen's table regions)",
                need: u64::from(region) + 1,
                have: u64::from(self.regions),
            });
        }
        REGION.set(Some(region));
        Ok(region)
    }

    /// Widens `rows` on `frame`: one dispatch for the rows of each pair of
    /// source and destination, their configs in this thread's region.
    pub fn encode(&self, frame: &Frame, rows: &[Row<'_>]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if rows.len() as u64 > self.entries {
            return Err(Fault::Ceiling {
                what: "rows one guest stage widens (a fire's lanes times its planes)",
                need: rows.len() as u64,
                have: self.entries,
            });
        }
        let base = u64::from(self.region()?) * self.region;
        // Past the rows the frame's earlier widens wrote: each widen's rows
        // stay its own until the frame runs.
        let mut need = 0u64;
        {
            let mut seen: Vec<(usize, usize)> = Vec::new();
            for row in rows {
                let pair = (key(row.src), key(row.dst));
                if !seen.contains(&pair) {
                    seen.push(pair);
                }
            }
            for pair in &seen {
                let members = rows
                    .iter()
                    .filter(|row| (key(row.src), key(row.dst)) == *pair)
                    .count() as u64;
                need += (members * size_of::<Cfg>() as u64).next_multiple_of(self.align);
            }
        }
        let from = frame.widen_rows(need);
        if from + need > self.region {
            return Err(Fault::Ceiling {
                what: "bytes of one frame's widen rows (a widen table region)",
                need: from + need,
                have: self.region,
            });
        }
        let mut order: Vec<(usize, usize)> = Vec::new();
        let mut grouped: HashMap<(usize, usize), Vec<&Row<'_>>> = HashMap::new();
        for row in rows {
            if !row.src_at.is_multiple_of(4) || !row.dst_at.is_multiple_of(4) {
                return Err(Fault::Program {
                    at: "guest::widen",
                    why: format!(
                        "a widen reads and writes whole words, and was offered byte offsets \
                         {} and {}",
                        row.src_at, row.dst_at
                    ),
                });
            }
            let pair = (key(row.src), key(row.dst));
            grouped
                .entry(pair)
                .or_insert_with(|| {
                    order.push(pair);
                    Vec::new()
                })
                .push(row);
        }
        let mut cursor = from;
        for pair in order {
            let group = self.groups.get(&pair).ok_or_else(|| Fault::Program {
                at: "guest::widen",
                why: "a widen reads a seat or writes a segment the load made it no bind group for"
                    .to_owned(),
            })?;
            let members = &grouped[&pair];
            let mut bytes = Vec::with_capacity(members.len() * size_of::<Cfg>());
            let mut widest = 0u32;
            for row in members {
                bytes.extend_from_slice(&config_bytes(Cfg {
                    src_word: word(row.src_at)?,
                    dst_word: word(row.dst_at)?,
                    lanes: row.lanes,
                    pad: 0,
                }));
                widest = widest.max(row.lanes);
            }
            let at = base + cursor;
            self.table.write_shared(at, &bytes)?;
            cursor = (cursor + bytes.len() as u64).next_multiple_of(self.align);
            let offset = u32::try_from(at).map_err(|_| Fault::Ceiling {
                what: "bytes into the widen's table (a dynamic offset is a u32)",
                need: at,
                have: u64::from(u32::MAX),
            })?;
            frame.dispatch_at(
                "widen",
                &self.pipeline,
                group,
                &[offset],
                [
                    widest.div_ceil(WG).max(1),
                    u32::try_from(members.len()).unwrap_or(u32::MAX),
                    1,
                ],
            )?;
        }
        Ok(())
    }
}

fn whole(buffer: &Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &buffer.slab().buffer,
        offset: 0,
        size: std::num::NonZeroU64::new(buffer.bytes()),
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::{Row, Widen};
    use crate::device::alloc::Memory;
    use crate::device::{Buffer, Context, present};

    fn device() -> Option<Context> {
        if !present() {
            return None;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        Some(Context::bind(&crate::api::DeviceBoot::default(), allowance).unwrap())
    }

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn rows_from_two_seats_widen_in_one_call() {
        let Some(device) = device() else { return };
        let owner = engine_gpu::books::owners::TEST;
        let mut a = Buffer::zeroed(&device, 256, owner).unwrap();
        let mut b = Buffer::zeroed(&device, 256, owner).unwrap();
        let mut dst = Buffer::zeroed(&device, 256, owner).unwrap();
        let first = [1.0f32, -2.5, 3.25];
        let second = [0.5f32, 7.0];
        let third = [-1.0f32, 4.0, 8.0, 16.0];
        a.write(0, &bf16(&first)).unwrap();
        a.write(64, &bf16(&second)).unwrap();
        b.write(0, &bf16(&third)).unwrap();
        let table = Buffer::with(
            &device,
            Widen::table_bytes(&device, 1, 3),
            Memory::Device,
            owner,
        )
        .unwrap();
        let widen = Widen::new(&device, table, &[&a, &b], &[&dst], 1, 3).unwrap();
        let frame = device.frame().unwrap();
        let rows = [
            Row {
                src: &a,
                src_at: 0,
                dst: &dst,
                dst_at: 0,
                lanes: 3,
            },
            Row {
                src: &b,
                src_at: 0,
                dst: &dst,
                dst_at: 64,
                lanes: 4,
            },
            Row {
                src: &a,
                src_at: 64,
                dst: &dst,
                dst_at: 128,
                lanes: 2,
            },
        ];
        widen.encode(&frame, &rows).unwrap();
        frame.commit().unwrap();
        let mut got = [0u8; 256];
        dst.read(0, &mut got).unwrap();
        let f = |at: usize, n: usize| -> Vec<f32> {
            got[at..at + 4 * n]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect()
        };
        assert_eq!(f(0, 3), first);
        assert_eq!(f(64, 4), third);
        assert_eq!(f(128, 2), second);
        let _ = &mut dst;
    }

    #[test]
    fn rows_past_a_region_are_refused() {
        let Some(device) = device() else { return };
        let owner = engine_gpu::books::owners::TEST;
        let src = Buffer::zeroed(&device, 64, owner).unwrap();
        let dst = Buffer::zeroed(&device, 64, owner).unwrap();
        let table = Buffer::with(
            &device,
            Widen::table_bytes(&device, 1, 1),
            Memory::Device,
            owner,
        )
        .unwrap();
        let widen = Widen::new(&device, table, &[&src], &[&dst], 1, 1).unwrap();
        let frame = device.frame().unwrap();
        let row = |dst_at| Row {
            src: &src,
            src_at: 0,
            dst: &dst,
            dst_at,
            lanes: 1,
        };
        assert!(widen.encode(&frame, &[row(0), row(4)]).is_err());
    }
}
