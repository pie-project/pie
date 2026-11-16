//! The standard samplers inferlets register (greedy, temperature
//! Gumbel-max, top-p, top-k with top-p, top logprobs, a sorted row), traced
//! against the served model, so a load can build their guest code before
//! ready (`engine::Engine::prewarm_programs`): an engine that compiles a
//! program's regions keys them by their text, which a sampler's constants
//! (its temperature, its cut, its seed) do not change, so a request that
//! registers one of these shapes builds nothing.

use eta_dsl::Traced;
use eta_dsl::prelude::*;
use eta_ir::registry::{OutputDecl, Port};

/// The decode epilogue's ports and its token channel, as an inferlet's
/// `ForwardPass` binds them.
fn builder<'a>(
    vocab: u32,
    page_size: u32,
    outputs: &[OutputDecl],
    tok_in: &'a Channel,
    indptr: &'a Channel,
    fold: &'a Channel,
) -> Builder<'a> {
    let mut b = Builder::new(vocab, page_size);
    b.outputs(vec![outputs.to_vec()]);
    b.bind_port(Port::EmbedTokens, tok_in);
    b.bind_port(Port::EmbedIndptr, indptr);
    b.bind_port(Port::Fold, fold);
    b
}

/// One standard sampler's decode epilogue, traced.
fn traced(
    vocab: u32,
    page_size: u32,
    outputs: &[OutputDecl],
    body: impl Fn(&Channel, &Channel, &Channel) + Clone,
) -> Option<Traced> {
    let tok_in = Channel::from([0i32]);
    let indptr = Channel::from([0u32, 1]);
    let fold = Channel::from([u32::MAX]);
    let rng = Channel::from([0u32, 0]);
    let out = Channel::new([1], dtype::i32).capacity(8);
    let mut b = builder(vocab, page_size, outputs, &tok_in, &indptr, &fold);
    let (t, r, o) = (&tok_in, &rng, &out);
    b.stage(Stage::Epilogue, move || body(t, r, o));
    out.note_host_take();
    b.build().ok()
}

/// The standard samplers' traces at the served model's `vocab`.
#[must_use]
pub fn traces(vocab: u32, page_size: u32, outputs: &[OutputDecl]) -> Vec<(&'static str, Traced)> {
    let v = vocab;
    let mut all: Vec<(&'static str, Option<Traced>)> = Vec::new();
    all.push((
        "greedy",
        traced(v, page_size, outputs, |tok_in, _, out| {
            let t = reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]);
            tok_in.put(&t);
            out.put(&t);
        }),
    ));
    all.push((
        "top-p",
        traced(v, page_size, outputs, move |tok_in, rng, out| {
            let scaled = intrinsics::output(0, "logits") / 0.7f32;
            let probs = softmax(&scaled);
            let keep = pivot_threshold(probs, cummass_le(0.9f32));
            let neg_inf = broadcast(f32::NEG_INFINITY, [v]);
            let masked = select(&keep, &scaled, &neg_inf);
            let r = rng.take();
            let g = gumbel(&r, [v]);
            rng.put(&r + iota(2));
            let t = reshape(reduce_argmax(masked + g), [1]);
            tok_in.put(&t);
            out.put(&t);
        }),
    ));
    all.push((
        "top-k + top-p",
        traced(v, page_size, outputs, move |tok_in, rng, out| {
            let r = rng.take();
            let logits = reshape(intrinsics::output(0, "logits"), [v]);
            let scaled = &logits / 0.8f32;
            let probs = softmax(&scaled);
            let ranked = pivot_threshold(&probs, rank_le(40u32));
            let nucleus = pivot_threshold(&probs, cummass_le(0.9f32));
            let keep = and(&ranked, &nucleus);
            let t = reshape(gumbel_max(mask_apply(&scaled, keep), &r), [1]);
            rng.put(&r + iota(2));
            tok_in.put(&t);
            out.put(&t);
        }),
    ));
    all.push((
        "gumbel-max",
        traced(v, page_size, outputs, |tok_in, rng, out| {
            let r = rng.take();
            let scaled = intrinsics::output(0, "logits") / 0.7f32;
            let t = reshape(gumbel_max(scaled, &r), [1]);
            rng.put(&r + iota(2));
            tok_in.put(&t);
            out.put(&t);
        }),
    ));
    all.push(("top logprobs", top_logprobs(v, page_size, outputs)));
    all.into_iter()
        .filter_map(|(name, traced)| Some((name, traced?)))
        .collect()
}

/// A greedy step that lands its top logprobs.
fn top_logprobs(vocab: u32, page_size: u32, outputs: &[OutputDecl]) -> Option<Traced> {
    let tok_in = Channel::from([0i32]);
    let indptr = Channel::from([0u32, 1]);
    let fold = Channel::from([u32::MAX]);
    let out = Channel::new([1], dtype::i32).capacity(8);
    let values_out = Channel::new([5], dtype::f32).capacity(8);
    let ids_out = Channel::new([5], dtype::i32).capacity(8);
    let mut b = builder(vocab, page_size, outputs, &tok_in, &indptr, &fold);
    let (t_in, o, values, ids) = (&tok_in, &out, &values_out, &ids_out);
    b.stage(Stage::Epilogue, move || {
        let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
        let t = reshape(reduce_argmax(&logits), [1]);
        let (top, at) = top_k(log_softmax(&logits), 5);
        values.put(&top);
        ids.put(cast(&at, dtype::i32));
        t_in.put(&t);
        o.put(&t);
    });
    values_out.note_host_take();
    ids_out.note_host_take();
    out.note_host_take();
    b.build().ok()
}

/// The standard samplers as the served model registers them: traced at its
/// vocabulary and default lane's outputs, registered with the runtime as a
/// guest's program is, and with what `backend`'s host codegen emits.
#[must_use]
pub fn registrations(backend: Option<&str>) -> Vec<::engine::ProgramRegistration> {
    let model = crate::model::model();
    let Ok(lane) = model.lane(&[]) else {
        return Vec::new();
    };
    let outputs: Vec<OutputDecl> = lane
        .outputs
        .iter()
        .map(crate::inferlet::host::forward::eta_output)
        .collect();
    let Some(vocab) = outputs
        .iter()
        .find(|output| output.name == "logits")
        .map(|output| output.width)
    else {
        return Vec::new();
    };
    traces(vocab, model.kv_page_size(), &outputs)
        .into_iter()
        .filter_map(|(name, traced)| {
            let program = super::program::register(traced.encode(), vec![outputs.clone()])
                .map_err(|error| tracing::warn!(sampler = name, %error, "not prewarmed"))
                .ok()?;
            let registration = ::engine::ProgramRegistration {
                program_hash: program.hash,
                launch: program.launch().clone(),
                reference_ptir: program.bytes.clone(),
                ..Default::default()
            };
            Some(super::program::with_host_codegen(&registration, backend).into_owned())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_standard_sampler_traces() {
        let outputs = vec![OutputDecl {
            name: "logits".into(),
            axis: eta_ir::registry::Axis::Lane,
            width: 151_936,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        }];
        let names: Vec<&str> = traces(151_936, 16, &outputs)
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            ["greedy", "top-p", "top-k + top-p", "gumbel-max", "top logprobs"]
        );
    }
}
