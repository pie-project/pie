use std::cell::Cell;

use kernels_xla::Tensor;
use poem_compiler::{CompiledModel, Lowering, Region};
use poem_exec::fire::{EventId, MaskSpan, Sink, WindowTable, fallback};
use poem_exec::store::check::{self, rebase};
use poem_ir::{Dtype, Trace};

use crate::inputs::Inputs;
use crate::trace::Handles;

/// A handle not yet bound to an input.
const NIL: u32 = u32::MAX;
use crate::error::{Fault, Result};

#[derive(Debug, Clone)]
pub struct Window {
    pub span: MaskSpan,

    pub indptr_host: Vec<i32>,

    pub indptr: Tensor,

    pub pass: u32,
    pub passes: u32,

    pub patch: MaskSpan,

    /// This region's span of the voxel axis (clips as lanes, voxels as
    /// rows): the one window a spatial launch covers.
    pub voxel: MaskSpan,
}

#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,

    runs: Vec<u32>,

    of_region: Vec<(u32, u32)>,
}

fn seat(windows: &mut Vec<Window>, window: Window) -> u32 {
    let same = |held: &Window| {
        held.span == window.span && held.pass == window.pass && held.passes == window.passes
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

impl Windows {
    pub fn of(
        compiled: &CompiledModel,
        classes: &WindowTable,
        patches: &WindowTable,
        voxels: &WindowTable,
        indptr_host: &[i32],
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
                poem_ir::RowAxis::Voxels => voxels.spans_into(&region.mask, &mut spans),
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

            let voxel = match voxels.span(&region.mask) {
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
                    patch,
                    voxel,
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

    /// Every distinct window of the fire.
    #[must_use]
    pub fn all(&self) -> &[Window] {
        &self.windows
    }

    /// What of the windows a traced program depends on when no window's
    /// lane offsets reach the text (no recurrent-state map): the spans and
    /// passes, not each lane's row offsets.
    #[must_use]
    pub fn shape(&self) -> String {
        let spans: Vec<_> = self
            .windows
            .iter()
            .map(|w| (w.span, w.pass, w.passes, w.patch, w.voxel))
            .collect();
        format!("{spans:?}|{:?}|{:?}", self.runs, self.of_region)
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
        }
        out
    }

    /// Lands every window's integer tables as fire inputs.
    pub fn bind(&mut self, handles: &Handles, inputs: &mut Inputs) -> Result<()> {
        let mut take = |vector: &[i32]| -> Tensor {
            let t = inputs.i32s(handles, vector, 1);
            Tensor::new(t.buf, vector.len() as u32, 1, Dtype::I32)
        };
        for window in &mut self.windows {
            window.indptr = take(&window.indptr_host);
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

#[derive(Debug, Default)]
pub struct At {
    pub region: Cell<u32>,

    pub run: Cell<u32>,

    pub tail: Cell<bool>,
}

impl At {
    #[must_use]
    pub fn new() -> At {
        At::default()
    }
}

#[derive(Debug)]
pub struct Cursor<'a> {
    at: u32,
    place: &'a At,
}

impl<'a> Cursor<'a> {
    #[must_use]
    pub fn new(place: &'a At) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        Cursor { at: 0, place }
    }

    #[allow(clippy::unnecessary_wraps, reason = "the seam: see the item doc")]
    pub fn settle(self) -> Result<()> {
        Ok(())
    }
}

impl Sink for Cursor<'_> {
    fn region_begin(&mut self, _region: &Region) {
        self.place.region.set(self.at);
        self.place.run.set(0);
        self.at += 1;
    }
    fn region_end(&mut self, _region: &Region) {}

    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
        self.place.tail.set(false);
    }
    fn tail(&mut self, in_tail: bool) {
        self.place.tail.set(in_tail);
    }
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}

    fn fork(&mut self, _event: EventId) {}

    fn join(&mut self, _event: EventId) {}
}
