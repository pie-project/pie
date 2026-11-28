//! Binding a traced program to buffers and running it.

use crate::device::Device;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::pjrt::{Arg, Buffer};
use crate::store::Pools;
use crate::trace::{Signature, Source, Tracer};
use crate::weights::Weights;

/// Compiles (or finds) the program `tracer` built and runs it with pools
/// donated where it writes them; returns its extra outputs.
pub fn run(
    device: &Device,
    tracer: Tracer<'_>,
    extra: &[kernels_xla::hlo::Val],
    weights: Option<&Weights>,
    pools: &mut Pools,
    inputs: &Inputs,
    wait: bool,
) -> Result<Vec<Buffer>> {
    let t0 = std::time::Instant::now();
    let (text, sig) = tracer.finish(extra);
    let t1 = std::time::Instant::now();
    // The program is cached by its text: one traced over other roots of the
    // same shapes (another host bank, say) binds this trace's roots.
    let program = device.program(&text, sig.clone())?;
    let t2 = std::time::Instant::now();
    let out = bind_and_run(device, &sig, &program, weights, pools, inputs, &[], wait);
    if crate::serve::timing() {
        eprintln!(
            "  xla run: finish {:.2}ms ({} KiB) lookup {:.2}ms bind+run {:.2}ms",
            (t1 - t0).as_secs_f64() * 1e3,
            text.len() >> 10,
            (t2 - t1).as_secs_f64() * 1e3,
            t2.elapsed().as_secs_f64() * 1e3
        );
    }
    out
}

/// Compiles `tracer`'s program (or finds it) without running it.
pub fn compile(
    device: &Device,
    tracer: Tracer<'_>,
    extra: &[kernels_xla::hlo::Val],
) -> Result<std::sync::Arc<crate::device::Program>> {
    let (text, sig) = tracer.finish(extra);
    device.program(&text, sig)
}

/// Runs an already compiled program.
pub fn run_program(
    device: &Device,
    program: &crate::device::Program,
    weights: Option<&Weights>,
    pools: &mut Pools,
    inputs: &Inputs,
    wait: bool,
) -> Result<Vec<Buffer>> {
    bind_and_run(
        device,
        &program.sig,
        program,
        weights,
        pools,
        inputs,
        &[],
        wait,
    )
}

/// A readout program: [`run_program`] with its entries (`Source::Entry`),
/// the planes its fire's forward returned.
pub fn run_readout(
    device: &Device,
    program: &crate::device::Program,
    weights: Option<&Weights>,
    pools: &mut Pools,
    inputs: &Inputs,
    entries: &[Buffer],
) -> Result<Vec<Buffer>> {
    bind_and_run(
        device,
        &program.sig,
        program,
        weights,
        pools,
        inputs,
        entries,
        false,
    )
}

/// A maintenance program: pools and `inputs` in, nothing out.
pub fn maintain(
    device: &Device,
    tracer: Tracer<'_>,
    pools: &mut Pools,
    inputs: &Inputs,
) -> Result<()> {
    run(device, tracer, &[], None, pools, inputs, false).map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn bind_and_run(
    device: &Device,
    sig: &Signature,
    program: &crate::device::Program,
    weights: Option<&Weights>,
    pools: &mut Pools,
    inputs: &Inputs,
    entries: &[Buffer],
    wait: bool,
) -> Result<Vec<Buffer>> {
    if device.is_dry() {
        return Ok(Vec::new());
    }
    // Bulk data, written into the bulk ring's next slot.
    let mut bulk: Vec<(u32, usize)> = Vec::new();
    for source in &sig.params {
        if let Source::Bulk { input } = *source {
            let ring = pools.bulk().ok_or_else(|| Fault::Unbound {
                what: "the bulk ring, which this load did not lay".into(),
            })?;
            bulk.push((input, ring.write_bytes(inputs.bulk_bytes(input))?));
        }
    }
    // The fire's inputs, written into the staging ring's next slot.
    let slot = if sig.params.contains(&Source::Stage) {
        let ring = pools.ring().ok_or_else(|| Fault::Unbound {
            what: "the staging ring, which this load did not lay".into(),
        })?;
        Some(ring.write(inputs.image())?)
    } else {
        None
    };
    let donated: Vec<usize> = sig
        .results
        .iter()
        .filter_map(|r| r.map(|(_, param)| param))
        .collect();
    // A readout ring's next plane, still read, refuses the fire before it
    // takes the pool or anything else.
    for (at, source) in sig.params.iter().enumerate() {
        if let (true, Source::Readout { index }) = (donated.contains(&at), source) {
            pools.readout_ready(*index)?;
        }
    }
    let mut taken: Vec<(usize, Buffer)> = Vec::new();
    for (at, source) in sig.params.iter().enumerate() {
        match (donated.contains(&at), source) {
            (true, Source::Pool { row, plane }) => taken.push((at, pools.take(*row, *plane)?)),
            (true, Source::Readout { index }) => {
                taken.push((at, pools.readout_take(device, *index)?));
            }
            _ => {}
        }
    }
    let mut args: Vec<Arg<'_>> = Vec::with_capacity(sig.params.len());
    let mut taken = taken.into_iter().peekable();
    for (at, source) in sig.params.iter().enumerate() {
        if taken.peek().is_some_and(|(t, _)| *t == at) {
            let (_, buffer) = taken.next().expect("peeked");
            args.push(Arg::Donate(buffer));
            continue;
        }
        let buffer = match *source {
            Source::Weight { param, plane, .. } => {
                weights.and_then(|w| w.buffer_plane(param, plane))
            }
            Source::Entry { index } => entries.get(usize::from(index)),
            Source::Pool { row, plane } => pools.get(row, plane),
            Source::HostBank { param } => pools.host_bank(param),
            Source::Stage => slot.and_then(|at| pools.ring_buffer(at)),
            Source::Bulk { input } => bulk
                .iter()
                .find(|(i, _)| *i == input)
                .and_then(|&(_, at)| pools.bulk_buffer(at)),
            Source::Readout { .. }
            | Source::Temp
            | Source::Staged { .. }
            | Source::Streamed { .. } => None,
        }
        .ok_or_else(|| Fault::Unbound {
            what: format!("parameter {at} ({source:?}), which nothing on this device holds"),
        })?;
        args.push(Arg::Keep(buffer));
    }
    let (outs, done) = device.run_fenced(program, args, wait)?;
    // The slots it read are free again once this run has completed on the
    // device: its own event, whatever it outputs.
    let done = done.map(crate::store::stage::Done::of);
    if let Some(at) = slot
        && let Some(ring) = pools.ring()
    {
        ring.read_by(at, done.clone());
    }
    for &(_, at) in &bulk {
        if let Some(ring) = pools.bulk() {
            ring.read_by(at, done.clone());
        }
    }
    let mut extra = Vec::new();
    for (out, result) in outs.into_iter().zip(&sig.results) {
        match result {
            Some((Source::Pool { row, plane }, _)) => pools.put(*row, *plane, out),
            Some((Source::Readout { index }, _)) => pools.readout_put(*index, out),
            Some((other, _)) => {
                return Err(Fault::Unbound {
                    what: format!("a result aliasing {other:?}, which is not a pool"),
                });
            }
            None => extra.push(out),
        }
    }
    Ok(extra)
}
