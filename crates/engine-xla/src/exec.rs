//! Binding a traced program to buffers and running it.

use crate::device::Device;
use crate::inputs::Inputs;
use crate::error::{Fault, Result};
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
    let program = device.program(&text, sig)?;
    let t2 = std::time::Instant::now();
    let out = run_program(device, &program, weights, pools, inputs, wait);
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
pub fn compile(device: &Device, tracer: Tracer<'_>, extra: &[kernels_xla::hlo::Val]) -> Result<std::sync::Arc<crate::device::Program>> {
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
    bind_and_run(device, &program.sig, program, weights, pools, inputs, wait)
}

/// A maintenance program: pools only, nothing out.
pub fn maintain(device: &Device, tracer: Tracer<'_>, pools: &mut Pools) -> Result<()> {
    run(device, tracer, &[], None, pools, &Inputs::new(), true).map(|_| ())
}

fn bind_and_run(
    device: &Device,
    sig: &Signature,
    program: &crate::device::Program,
    weights: Option<&Weights>,
    pools: &mut Pools,
    inputs: &Inputs,
    wait: bool,
) -> Result<Vec<Buffer>> {
    if device.is_dry() {
        return Ok(Vec::new());
    }
    let mut uploaded: Vec<(u32, Buffer)> = Vec::new();
    let mut pack: Option<Buffer> = None;
    for source in &sig.params {
        match *source {
            Source::Input { input } => uploaded.push((input, inputs.upload(device, input)?)),
            Source::Pack if pack.is_none() => pack = Some(inputs.upload_pack(device)?),
            _ => {}
        }
    }
    let donated: Vec<usize> = sig
        .results
        .iter()
        .filter_map(|r| r.map(|(_, param)| param))
        .collect();
    let mut taken: Vec<(usize, Buffer)> = Vec::new();
    for (at, source) in sig.params.iter().enumerate() {
        if let (true, Source::Pool { row, plane }) = (donated.contains(&at), source) {
            taken.push((at, pools.take(*row, *plane)?));
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
            Source::Weight { param, .. } => weights.and_then(|w| w.buffer(param)),
            Source::Pool { row, plane } => pools.get(row, plane),
            Source::Input { input } => uploaded
                .iter()
                .find(|(i, _)| *i == input)
                .map(|(_, b)| b),
            Source::Pack => pack.as_ref(),
            Source::Temp | Source::Packed { .. } => None,
        }
        .ok_or_else(|| Fault::Unbound {
            what: format!("parameter {at} ({source:?}), which nothing on this device holds"),
        })?;
        args.push(Arg::Keep(buffer));
    }
    let outs = device.run(program, args, wait)?;
    let mut extra = Vec::new();
    for (out, result) in outs.into_iter().zip(&sig.results) {
        match result {
            Some((Source::Pool { row, plane }, _)) => pools.put(*row, *plane, out),
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
