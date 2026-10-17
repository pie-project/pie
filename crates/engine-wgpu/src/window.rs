use kernels_wgpu::Tensor;
use poem_compiler::CompiledModel;
use poem_exec::fire::{MaskSpan, WindowTable, fallback};
use poem_exec::store::check::{self, rebase};
use poem_exec::store::kv::Geometry;
use poem_ir::{Dtype, Trace};

use crate::device::Handles;
use crate::device::handles::NIL;
use crate::error::{Fault, Result};

#[derive(Debug, Clone)]
pub struct Window {
    pub span: MaskSpan,

    pub indptr_host: Vec<i32>,

    pub indptr: Tensor,

    pub gathered: Option<Gathered>,

    pub pass: u32,
    pub passes: u32,

    pub patch: MaskSpan,
}

#[derive(Debug, Clone)]
pub struct Gathered {
    pub runs: Vec<MaskSpan>,

    pub rows_host: Vec<i32>,

    pub rows: Tensor,

    pub positions_host: Vec<i32>,

    pub positions: Tensor,

    pub request_of_token_host: Vec<i32>,

    pub request_of_token: Tensor,

    pub spaces: Vec<GatheredSpace>,
}

#[derive(Debug, Clone)]
pub struct GatheredSpace {
    pub page_indptr_host: Vec<i32>,

    pub page_indices_host: Vec<i32>,

    pub last_page_lens_host: Vec<i32>,

    pub kv_len_host: Vec<i32>,

    pub page_indptr: Tensor,

    pub page_indices: Tensor,

    pub last_page_lens: Tensor,

    pub kv_len: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct Copies<'a> {
    pub bucket: u32,

    pub enabled: bool,

    pub spaces: &'a [Geometry],

    pub positions: &'a [i32],

    pub request_of_token: &'a [i32],
}

impl Copies<'_> {
    #[must_use]
    pub fn off() -> Copies<'static> {
        Copies {
            bucket: 0,
            enabled: false,
            spaces: &[],
            positions: &[],
            request_of_token: &[],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,

    runs: Vec<u32>,

    of_region: Vec<(u32, u32)>,
}

fn seat(windows: &mut Vec<Window>, window: Window) -> u32 {
    let same = |held: &Window| {
        held.span == window.span
            && held.pass == window.pass
            && held.passes == window.passes
            && held.gathered.as_ref().map(|g| &g.runs) == window.gathered.as_ref().map(|g| &g.runs)
    };
    let index = match windows.iter().position(same) {
        Some(index) => index,
        None => {
            windows.push(window);
            windows.len() - 1
        }
    };
    index as u32
}

fn gather_of(runs: &[MaskSpan], indptr_host: &[i32], copies: Copies<'_>) -> Window {
    let mut rows_host: Vec<i32> = Vec::new();
    let mut lanes: Vec<usize> = Vec::new();
    let mut bounds: Vec<i32> = vec![0];
    for run in runs {
        for row in run.row_offset..run.row_offset + run.rows {
            rows_host.push(row as i32);
        }
        for lane in run.lane_offset..run.lane_offset + run.lanes {
            let lane = lane as usize;

            let width = indptr_host
                .get(lane + 1)
                .zip(indptr_host.get(lane))
                .map_or(0, |(end, start)| end - start);
            bounds.push(bounds.last().copied().unwrap_or(0) + width);
            lanes.push(lane);
        }
    }

    let relay = |table: &[i32]| -> Vec<i32> {
        rows_host
            .iter()
            .map(|&row| table.get(row as usize).copied().unwrap_or(0))
            .collect()
    };
    let positions_host = relay(copies.positions);
    let request_of_token_host = relay(copies.request_of_token);

    let spaces = copies
        .spaces
        .iter()
        .map(|space| {
            let mut page_indptr_host: Vec<i32> = vec![0];
            let mut page_indices_host: Vec<i32> = Vec::new();
            let mut last_page_lens_host: Vec<i32> = Vec::new();
            let mut kv_len_host: Vec<i32> = Vec::new();
            for &lane in &lanes {
                let start = space.indptr.get(lane).copied().unwrap_or(0).max(0) as usize;
                let end = space.indptr.get(lane + 1).copied().unwrap_or(0).max(0) as usize;
                let pages = space.indices.get(start..end).unwrap_or(&[]);
                page_indices_host.extend_from_slice(pages);
                page_indptr_host.push(page_indices_host.len() as i32);
                last_page_lens_host.push(space.last_page_len.get(lane).copied().unwrap_or(0));
                kv_len_host.push(space.kv_len.get(lane).copied().unwrap_or(0));
            }
            GatheredSpace {
                page_indptr_host,
                page_indices_host,
                last_page_lens_host,
                kv_len_host,
                page_indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
                page_indices: Tensor::new(NIL, 0, 1, Dtype::I32),
                last_page_lens: Tensor::new(NIL, 0, 1, Dtype::I32),
                kv_len: Tensor::new(NIL, 0, 1, Dtype::I32),
            }
        })
        .collect();

    Window {
        span: MaskSpan {
            row_offset: 0,
            rows: rows_host.len() as u32,
            lane_offset: 0,
            lanes: lanes.len() as u32,
        },
        indptr_host: bounds,
        indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
        gathered: Some(Gathered {
            runs: runs.to_vec(),
            rows: Tensor::new(NIL, 0, 1, Dtype::I32),
            rows_host,
            positions: Tensor::new(NIL, 0, 1, Dtype::I32),
            positions_host,
            request_of_token: Tensor::new(NIL, 0, 1, Dtype::I32),
            request_of_token_host,
            spaces,
        }),

        patch: MaskSpan::default(),
        pass: 0,
        passes: 1,
    }
}

impl Windows {
    #[allow(clippy::too_many_arguments)]
    pub fn of(
        trace: &Trace,
        compiled: &CompiledModel,
        classes: &WindowTable,
        patches: &WindowTable,
        indptr_host: &[i32],
        copies: Copies<'_>,
        run_caps: &[u32],
        run_passes: &[u32],
    ) -> Result<Windows> {
        let mut windows: Vec<Window> = Vec::new();
        let mut runs: Vec<u32> = Vec::with_capacity(compiled.template().len());
        let mut of_region: Vec<(u32, u32)> = Vec::with_capacity(compiled.template().len());
        let mut spans: Vec<MaskSpan> = Vec::new();

        for (at, region) in compiled.template().iter().enumerate() {
            let axis = compiled.axis_of(at);
            match axis {
                poem_ir::RowAxis::Tokens => classes.spans_into(&region.mask, &mut spans),
                poem_ir::RowAxis::Patches => patches.spans_into(&region.mask, &mut spans),
                poem_ir::RowAxis::Voxels => spans.clear(),
            }

            let patch = match patches.span(&region.mask) {
                Ok(span) => span.unwrap_or_default(),
                Err(runs) => {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs,
                        promised: None,
                    });
                }
            };

            if spans.len() > 1 {
                let bound = fallback::bound(compiled, axis, &region.mask);
                if fallback::promised(compiled, axis, region) || spans.len() > bound as usize {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs: spans.len(),
                        promised: fallback::promised(compiled, axis, region).then_some(bound),
                    });
                }
            }

            if spans.is_empty() {
                spans.push(MaskSpan::default());
            }

            if spans.len() > 1
                && copies.enabled
                && fallback::copies(compiled, axis, &region.mask, copies.bucket)
                && fallback::copyable_mask(trace, compiled, &region.mask)
            {
                let mut gathered = gather_of(&spans, indptr_host, copies);
                gathered.patch = patch;
                of_region.push((runs.len() as u32, 1));
                runs.push(seat(&mut windows, gathered));
                continue;
            }

            let cap = run_caps.get(at).copied().unwrap_or(0);
            let max_passes = run_passes.get(at).copied().unwrap_or(0);

            let (capped, passes) = if cap > 0 && max_passes > 1 {
                (
                    false,
                    poem_exec::fire::pass_spans(&mut spans, cap, max_passes),
                )
            } else {
                let capped = cap > 0 && spans.iter().any(|span| span.rows > cap);
                if capped {
                    poem_exec::fire::chunk_spans(&mut spans, cap);
                }
                (capped, 1)
            };
            of_region.push((runs.len() as u32, spans.len() as u32));
            for (i, &span) in spans.iter().enumerate() {
                let window = Window {
                    span,
                    indptr_host: match axis {
                        poem_ir::RowAxis::Tokens if capped => vec![0, span.rows as i32],
                        poem_ir::RowAxis::Tokens => rebase(indptr_host, span)?,
                        poem_ir::RowAxis::Patches | poem_ir::RowAxis::Voxels => Vec::new(),
                    },
                    indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
                    gathered: None,
                    patch,
                    pass: (i as u32) % passes,
                    passes,
                };
                runs.push(seat(&mut windows, window));
            }
        }

        Ok(Windows {
            windows,
            runs,
            of_region,
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        for window in &self.windows {
            out.extend_from_slice(&window.indptr_host);
            let Some(gathered) = &window.gathered else {
                continue;
            };
            out.extend_from_slice(&gathered.rows_host);
            out.extend_from_slice(&gathered.positions_host);
            out.extend_from_slice(&gathered.request_of_token_host);
            for space in &gathered.spaces {
                out.extend_from_slice(&space.page_indptr_host);
                out.extend_from_slice(&space.page_indices_host);
                out.extend_from_slice(&space.last_page_lens_host);
                out.extend_from_slice(&space.kv_len_host);
            }
        }
        out
    }

    pub fn bind(&mut self, handles: &Handles, packed: u32) -> Result<()> {
        let mut at = 0u64;
        let mut take = |vector: &[i32]| -> Result<Tensor> {
            let rows = vector.len() as u32;
            let bytes = u64::from(rows) * 4;
            let cut = handles.cut(packed, at, bytes)?;
            at += bytes;
            Ok(Tensor::new(cut, rows, 1, Dtype::I32))
        };
        for window in &mut self.windows {
            window.indptr = take(&window.indptr_host)?;
            let Some(gathered) = &mut window.gathered else {
                continue;
            };
            gathered.rows = take(&gathered.rows_host)?;
            gathered.positions = take(&gathered.positions_host)?;
            gathered.request_of_token = take(&gathered.request_of_token_host)?;
            for space in &mut gathered.spaces {
                space.page_indptr = take(&space.page_indptr_host)?;
                space.page_indices = take(&space.page_indices_host)?;
                space.last_page_lens = take(&space.last_page_lens_host)?;
                space.kv_len = take(&space.kv_len_host)?;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn runs(&self, region: u32) -> u32 {
        self.of_region.get(region as usize).map_or(0, |held| held.1)
    }

    #[must_use]
    pub fn launches(&self) -> u32 {
        self.of_region.iter().map(|&(_, runs)| runs.max(1)).sum()
    }

    #[must_use]
    pub fn copied(&self) -> u32 {
        self.of_region
            .iter()
            .filter(|&&(start, _)| {
                self.runs
                    .get(start as usize)
                    .and_then(|&index| self.windows.get(index as usize))
                    .is_some_and(|window| window.gathered.is_some())
            })
            .count() as u32
    }

    #[must_use]
    pub fn max_runs(&self) -> u32 {
        self.of_region
            .iter()
            .map(|&(_, runs)| runs)
            .max()
            .unwrap_or(1)
            .max(1)
    }

    #[must_use]
    pub fn at(&self, region: u32, run: u32) -> &Window {
        self.of_region
            .get(region as usize)
            .filter(|&&(_, runs)| run < runs)
            .and_then(|&(start, _)| self.runs.get((start + run) as usize))
            .and_then(|index| self.windows.get(*index as usize))
            .unwrap_or_else(|| {
                panic!(
                    "region {region} has no run {run}; this fire cut it into {} \
                     over a template of {}",
                    self.runs(region),
                    self.of_region.len()
                )
            })
    }
}

pub fn no_schedule_straddles_its_readers(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
    Ok(check::no_schedule_straddles_its_readers(trace, compiled)?)
}
