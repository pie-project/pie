//! A beam group's head program (`new-memory/designs/BEAM-PAGE-LISTS.md`
//! §5–6), run in stream order before the fire whose beams it resolves: the
//! group's step over its state in its `Instance` run and the parents its
//! last epilogue wrote (both read from the pool, never from the host), the
//! forks' tail copies over every kv plane, and each beam's resolved page
//! entries, write page and fault word landed beside its state for the fire
//! and the settle to read. One program a width rung, laid at load.

use super::{Fault, Pools, Result};
use crate::device::Device;
use crate::trace::Handles;
use crate::trace::Tracer;
use kernels_xla::beam::{At, Layout};
use kernels_xla::hlo::{Cmp, Combine, Elem};

/// Words a pool row holds.
const ROW_WORDS: i64 = 128;

/// A group's head, this fire: where its state and its parents lie, the
/// fire's numbers (§3), its live beams, and its pool's pages (grains, from
/// the set's first pool page; at most the layout's columns).
#[derive(Clone, Debug)]
pub struct Head {
    pub state_row: u32,
    pub parents_row: u32,
    pub folded: u32,
    pub slot: u32,
    pub live: u32,
    pub pool: Vec<u32>,
}

/// Rows of `words` i32 words, whole pool rows.
fn rows_of(words: i64) -> i64 {
    (words + ROW_WORDS - 1) / ROW_WORDS
}

/// Where a group's results lie past its state: each beam's resolved entries
/// (`[lanes, cap]` grains) and write page (`[lanes]` grain) for the fire
/// next in stream order, then its faults (`[frames + 1, lanes]`): one set a
/// fire in flight, fire `t`'s at `t mod (frames + 1)`, since the next fire's
/// head may run before this fire's settle reads them (§7).
#[must_use]
pub fn results_words(layout: Layout) -> i64 {
    layout.lanes * layout.cap + layout.lanes + fault_sets(layout) * layout.lanes
}

/// The fault sets a group keeps: one a fire in flight and the one landing.
#[must_use]
pub fn fault_sets(layout: Layout) -> i64 {
    layout.frames.max(1) + 1
}

/// Where fire `t`'s faults lie among a group's results (words).
#[must_use]
pub fn faults_at(layout: Layout, fire: u64) -> i64 {
    let set = (fire % fault_sets(layout) as u64) as i64;
    layout.lanes * layout.cap + layout.lanes + set * layout.lanes
}

/// The rows a group's `Instance` run holds: its state's, then its results'.
#[must_use]
pub fn run_rows(layout: Layout) -> i64 {
    rows_of(layout.words()) + rows_of(results_words(layout))
}

impl Pools {
    /// Lays (compiles) the head programs of width rungs `layouts`: traced in
    /// turn, compiled side by side.
    pub fn lay_beam_heads(
        &mut self,
        device: &Device,
        handles: &Handles,
        layouts: &[Layout],
    ) -> Result<()> {
        let blank = Head {
            state_row: 0,
            parents_row: 0,
            folded: 0,
            slot: 0,
            live: 0,
            pool: Vec::new(),
        };
        let mut texts = Vec::with_capacity(layouts.len());
        for &layout in layouts {
            let (tracer, _) = self.beam_head_tracer(handles, layout, &blank)?;
            texts.push(tracer.finish(&[]));
        }
        let compiled: Vec<Result<std::sync::Arc<crate::device::Program>>> =
            std::thread::scope(|scope| {
                let jobs: Vec<_> = texts
                    .iter()
                    .map(|(text, sig)| scope.spawn(move || device.program(text, sig.clone())))
                    .collect();
                jobs.into_iter()
                    .map(|job| {
                        job.join().unwrap_or_else(|_| {
                            Err(Fault::Unbound {
                                what: "a beam head's compile panicked".to_string(),
                            })
                        })
                    })
                    .collect()
            });
        for (&layout, program) in layouts.iter().zip(compiled) {
            let program = program?;
            self.beam_heads.retain(|(laid, _)| *laid != layout);
            self.beam_heads.push((layout, program));
        }
        Ok(())
    }

    /// Runs group `head`'s step on width rung `layout` (laid at load),
    /// before its fire: its inputs staged as the laid program reads them.
    pub fn run_beam_head(
        &mut self,
        device: &Device,
        handles: &Handles,
        layout: Layout,
        head: &Head,
    ) -> Result<()> {
        let program = self
            .beam_heads
            .iter()
            .find(|(laid, _)| *laid == layout)
            .map(|(_, program)| std::sync::Arc::clone(program))
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "a beam head of {} lanes, which the load did not lay",
                    layout.lanes
                ),
            })?;
        let (inputs, ..) = head_inputs(handles, layout, head)?;
        crate::exec::run_program(device, &program, None, self, &inputs, false).map(|_| ())
    }

    fn beam_head_tracer<'h>(
        &self,
        handles: &'h Handles,
        layout: Layout,
        head: &Head,
    ) -> Result<(Tracer<'h>, crate::inputs::Inputs)> {
        let Some(kv) = self.kv_shape() else {
            return Err(Fault::Unbound {
                what: "a beam group's pages on a load that keeps no kv".to_string(),
            });
        };
        let cap = layout.cap;
        if head.pool.len() as i64 > cap {
            return Err(Fault::Unbound {
                what: format!(
                    "a beam group's pool of {} pages, past the {cap} its state is laid for",
                    head.pool.len()
                ),
            });
        }
        let (inputs, params_t, pool_t) = head_inputs(handles, layout, head)?;
        // A page of the kv shape is one slot: every plane of it, whole pool
        // rows from its grain's row on (`Pools::rows_at`).
        let shape = self.shape(kv)?;
        let grain_rows = i64::try_from(self.layout.grain(shape) / super::ROW_BYTES).unwrap_or(0);
        let slot_rows = i64::try_from(shape.slot_bytes / super::ROW_BYTES).unwrap_or(0);
        if grain_rows == 0 || slot_rows == 0 {
            return Err(Fault::Unbound {
                what: "a kv page narrower than a pool row, which a beam copy moves whole"
                    .to_string(),
            });
        }
        let root = self.root();
        let tracer = Tracer::new(handles).with_stage(self.stage_at(inputs.rows()));
        let (b_n, words) = (layout.lanes, layout.words());
        let (state_rows, result_rows) = (rows_of(words), rows_of(results_words(layout)));
        {
            use crate::emit::Emit as _;
            tracer.emit(&mut |cx| {
                let whole = cx.read(root)?;
                let p = cx.read(params_t)?;
                let p = cx.reshape(p, &[6])?;
                let scalar = |cx: &mut crate::emit::Cx<'_>, k: i64| {
                    let v = cx.slice_axis(p, 0, k, k + 1)?;
                    cx.reshape(v, &[])
                };
                let state_row = scalar(cx, 0)?;
                let parents_row = scalar(cx, 1)?;
                let at = At {
                    folded: scalar(cx, 2)?,
                    reserve: scalar(cx, 3)?,
                    slot: scalar(cx, 4)?,
                    live: scalar(cx, 5)?,
                };
                let translation = cx.read(pool_t)?;
                let translation = cx.reshape(translation, &[cap])?;
                // the state's rows and the parents cell's row
                let rows_at = |cx: &mut crate::emit::Cx<'_>, first, n: i64| {
                    let io = cx.iota(Elem::I32, &[n], 0);
                    let first = cx.splat(first, &[n])?;
                    cx.add(io, first)
                };
                let s_rows = rows_at(cx, state_row, state_rows)?;
                let state = cx.take_rows(whole, s_rows)?;
                let state = cx.bitcast(state, Elem::I32)?;
                let state = cx.reshape(state, &[state_rows * ROW_WORDS])?;
                let state = cx.slice_axis(state, 0, 0, words)?;
                let p_rows = rows_at(cx, parents_row, 1)?;
                let parents = cx.take_rows(whole, p_rows)?;
                let parents = cx.bitcast(parents, Elem::I32)?;
                let parents = cx.reshape(parents, &[ROW_WORDS])?;
                let parents = cx.slice_axis(parents, 0, 0, b_n)?;
                let step = kernels_xla::beam::step(cx, layout, state, parents, at)?;
                let entries = kernels_xla::beam::resolve(cx, step.lists, translation)?;
                // each beam's write page: its entry at the fire's column
                let col = cx.iota(Elem::I32, &[b_n, cap], 1);
                let c = cx.splat(step.column, &[b_n, cap])?;
                let at_c = cx.compare(Cmp::Eq, col, c)?;
                let none = cx.const_i(Elem::I32, 0, &[b_n, cap]);
                let w = cx.select(at_c, entries, none)?;
                let write = cx.reduce(w, &[1], kernels_xla::hlo::Fold::Sum)?;

                // The forks' tail copies (§6.4), before anything of the fire reads
                // or appends: a forking beam's page copied whole, every plane at
                // once (the slots past `tokens` it brings are written before any
                // row reads them: a beam reads its own length, and its next
                // append lands at `o`), one block a forking beam.
                let src = cx.slice_axis(step.copies, 1, 0, 1)?;
                let src = cx.reshape(src, &[b_n])?;
                let dst = cx.slice_axis(step.copies, 1, 1, 2)?;
                let dst = cx.reshape(dst, &[b_n])?;
                let tok = cx.slice_axis(step.copies, 1, 2, 3)?;
                let tok = cx.reshape(tok, &[b_n])?;
                let src_g = kernels_xla::beam::resolve(cx, src, translation)?;
                let dst_g = kernels_xla::beam::resolve(cx, dst, translation)?;
                let zero_b = cx.const_i(Elem::I32, 0, &[b_n]);
                let some = cx.compare(Cmp::Gt, tok, zero_b)?;
                let clean = cx.compare(Cmp::Eq, step.faults, zero_b)?;
                let moves = cx.and(some, clean)?;
                let grain_b = cx.const_i(Elem::I32, grain_rows, &[b_n]);
                let src_row = cx.mul(src_g, grain_b)?;
                let dst_row = cx.mul(dst_g, grain_b)?;
                let moves_i = cx.convert(moves, Elem::I32);
                let n_moves = cx.reduce(moves_i, &[0], kernels_xla::hlo::Fold::Sum)?;
                // each forking beam's place among them
                let rank = {
                    let me = cx.broadcast(moves_i, &[b_n, b_n], &[1])?;
                    let ei = cx.iota(Elem::I32, &[b_n, b_n], 1);
                    let bi = cx.iota(Elem::I32, &[b_n, b_n], 0);
                    let before = cx.compare(Cmp::Lt, ei, bi)?;
                    let z = cx.const_i(Elem::I32, 0, &[b_n, b_n]);
                    let me = cx.select(before, me, z)?;
                    cx.reduce(me, &[1], kernels_xla::hlo::Fold::Sum)?
                };
                let whole = cx.read(root)?;
                let zero = cx.const_i(Elem::I32, 0, &[]);
                let out = cx.while_loop(
                    &[zero, whole],
                    |cx, a| cx.compare(Cmp::Lt, a[0], n_moves),
                    |cx, a| {
                        let k = cx.splat(a[0], &[b_n])?;
                        let at = cx.compare(Cmp::Eq, rank, k)?;
                        let pick = cx.and(at, moves)?;
                        let z = cx.const_i(Elem::I32, 0, &[b_n]);
                        let s_row = cx.select(pick, src_row, z)?;
                        let s_row = cx.reduce(s_row, &[0], kernels_xla::hlo::Fold::Sum)?;
                        let d_row = cx.select(pick, dst_row, z)?;
                        let d_row = cx.reduce(d_row, &[0], kernels_xla::hlo::Fold::Sum)?;
                        let lane0 = cx.const_i(Elem::I32, 0, &[]);
                        let block =
                            cx.dynamic_slice(a[1], &[s_row, lane0], &[slot_rows, ROW_WORDS])?;
                        let next = cx.dynamic_update_slice(a[1], block, &[d_row, lane0])?;
                        let one = cx.const_i(Elem::I32, 1, &[]);
                        Ok(vec![cx.add(a[0], one)?, next])
                    },
                )?;
                cx.write(root, out[1])?;

                // the state after, and the results beside it
                let whole = cx.read(root)?;
                let land = |cx: &mut crate::emit::Cx<'_>, whole, v, first, rows: i64| {
                    let n = cx.dims(v)[0];
                    let pad = cx.const_i(Elem::I32, 0, &[]);
                    let v = cx.pad(v, pad, &[0], &[rows * ROW_WORDS - n], &[0])?;
                    let v = cx.reshape(v, &[rows, ROW_WORDS])?;
                    let v = cx.bitcast(v, Elem::U32)?;
                    let at = rows_at(cx, first, rows)?;
                    cx.put_rows(whole, at, v, Combine::Set)
                };
                let whole = land(cx, whole, step.state, state_row, state_rows)?;
                let rows = cx.const_i(Elem::I32, state_rows, &[]);
                let out_row = cx.add(state_row, rows)?;
                // the fault sets of fires still in flight stay as they are
                let r_rows = rows_at(cx, out_row, result_rows)?;
                let kept = cx.take_rows(whole, r_rows)?;
                let kept = cx.bitcast(kept, Elem::I32)?;
                let kept = cx.reshape(kept, &[result_rows * ROW_WORDS])?;
                let sets = fault_sets(layout) * b_n;
                let first = b_n * cap + b_n;
                let fault_words = cx.slice_axis(kept, 0, first, first + sets)?;
                let fired = cx.slice_axis(state, 0, 0, 1)?;
                let fired = cx.reshape(fired, &[])?;
                let n_sets = cx.const_i(Elem::I32, fault_sets(layout), &[]);
                let set = cx.rem(fired, n_sets)?;
                let width = cx.const_i(Elem::I32, b_n, &[]);
                let set_at = cx.mul(set, width)?;
                let fault_words = cx.dynamic_update_slice(fault_words, step.faults, &[set_at])?;
                let entries = cx.reshape(entries, &[b_n * cap])?;
                let results = cx.concat(&[entries, write, fault_words], 0)?;
                let whole = land(cx, whole, results, out_row, result_rows)?;
                cx.write(root, whole)
            })?;
        }
        Ok((tracer, inputs))
    }
}

/// A head's inputs, in the order its program reads them: the fire's
/// numbers, then its pool's pages padded to the layout's columns.
fn head_inputs(
    handles: &Handles,
    layout: Layout,
    head: &Head,
) -> Result<(
    crate::inputs::Inputs,
    crate::handle::Tensor,
    crate::handle::Tensor,
)> {
    let narrow = |v: u32| {
        i32::try_from(v).map_err(|_| Fault::Unbound {
            what: format!("a beam head number {v}, past an i32"),
        })
    };
    let mut inputs = crate::inputs::Inputs::new();
    let params = [
        head.state_row,
        head.parents_row,
        head.folded,
        head.pool.len() as u32,
        head.slot,
        head.live,
    ]
    .iter()
    .map(|&v| narrow(v))
    .collect::<Result<Vec<i32>>>()?;
    let params_t = inputs.i32s(handles, &params, 1);
    let mut pool = head
        .pool
        .iter()
        .map(|&g| narrow(g))
        .collect::<Result<Vec<i32>>>()?;
    pool.resize(layout.cap as usize, 0);
    let pool_t = inputs.i32s(handles, &pool, 1);
    Ok((inputs, params_t, pool_t))
}
