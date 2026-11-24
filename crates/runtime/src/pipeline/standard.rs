//! The SDK's standard samplers as the served model registers them, so a
//! load can build their guest code before ready
//! (`engine::Engine::prewarm_programs`). The stage bodies are the SDK's own
//! ([`eta_dsl::sampler`], which the built-in inferlets' generation loop
//! runs as `inferlet::sampler`), traced for each of
//! [`eta_dsl::sampler::standard`]'s shapes over the channels and ports that
//! loop binds: the prompt's last fire and the decode fire. An engine keys a
//! program's regions by their text, which a request's knob values do not
//! change, so a request of one of these shapes builds nothing.
//!
//! The limit: the channels and ports a pass binds here mirror the built-in
//! inferlets' generation loop (compat's `generate`), not taken from it.
//! compat builds in its own wasm32-wasip3 workspace and the SDK's `Channel`
//! and `ForwardPass` call WIT imports, so the runtime cannot trace compat
//! itself; a change to that loop's channels or ports (not to the sampler's
//! stage bodies) is not caught by this module's test, and its programs
//! then build at registration, counted in `Tally::BuiltAtRegister`.

use eta_dsl::Traced;
use eta_dsl::prelude::*;
use eta_dsl::sampler::{self, Knobs, Shape, Slots};
use eta_ir::registry::OutputDecl;

/// The channels a pass's knobs ride in, as the built-in inferlets seed them
/// (read in the stage, never consumed): the temperature as its reciprocal
/// (`sampler::inverse_temperature`).
struct KnobChannels {
    temperature: Option<Channel>,
    top_p: Option<Channel>,
    top_k: Option<Channel>,
    min_p: Option<Channel>,
    repetition: Option<Channel>,
    frequency: Option<Channel>,
    presence: Option<Channel>,
    logprobs: Option<u32>,
}

/// A request's values for a shape's knobs, its seed and its prompt's
/// length: what the prewarm traces at, and any request's in their place.
#[derive(Clone, Copy)]
struct Request {
    temperature: f32,
    knob: f32,
    top_k: u32,
    seed: u32,
    prompt: usize,
}

const PREWARM: Request = Request {
    temperature: 1.0,
    knob: 0.9,
    top_k: 40,
    seed: 0,
    prompt: 4,
};

impl KnobChannels {
    fn new(shape: &Shape, request: &Request) -> Self {
        let f = |on: bool| on.then(|| Channel::from([request.knob]));
        Self {
            temperature: shape
                .temperature
                .then(|| Channel::from([sampler::inverse_temperature(request.temperature)])),
            top_p: f(shape.top_p),
            top_k: shape.top_k.then(|| Channel::from([request.top_k])),
            min_p: f(shape.min_p),
            repetition: f(shape.repetition),
            frequency: f(shape.frequency),
            presence: f(shape.presence),
            logprobs: shape.logprobs,
        }
    }

    fn read(&self) -> Knobs {
        let t = |c: &Option<Channel>| c.as_ref().map(|c| reshape(c.read(), []));
        Knobs {
            temperature: t(&self.temperature),
            top_p: t(&self.top_p),
            top_k: t(&self.top_k),
            min_p: t(&self.min_p),
            repetition: t(&self.repetition),
            frequency: t(&self.frequency),
            presence: t(&self.presence),
            logprobs: self.logprobs,
        }
    }
}

fn penalized(shape: &Shape) -> bool {
    shape.repetition || shape.frequency || shape.presence
}

fn logprob_channels(logprobs: Option<u32>, capacity: u32) -> Option<(Channel, Option<Channel>)> {
    logprobs.map(|top| {
        let values = Channel::new([1 + top], dtype::f32).capacity(capacity);
        values.note_host_take();
        let ids = (top > 0).then(|| {
            let ids = Channel::new([top], dtype::i32).capacity(capacity);
            ids.note_host_take();
            ids
        });
        (values, ids)
    })
}

fn histogram(vocab: u32, penalized: bool) -> Option<(Channel, Channel)> {
    penalized.then(|| {
        (
            Channel::from(vec![0.0f32; vocab as usize]),
            Channel::from(vec![0.0f32; vocab as usize]),
        )
    })
}

/// The prompt's last fire and the decode fire of one shape, as the built-in
/// inferlets' unguided generation loop builds them.
fn passes(
    vocab: u32,
    page_size: u32,
    outputs: &[OutputDecl],
    capacity: u32,
    shape: &Shape,
    request: &Request,
) -> Result<[Traced; 2], String> {
    let penalized = penalized(shape);
    // Unguided and landing no logprobs, the prompt's token is carried into
    // the decode pass's channels on the device.
    let carried = shape.logprobs.is_none();
    let tok_in = if carried {
        Channel::new([1], dtype::i32).queued()
    } else {
        Channel::from([0i32])
    };
    let carried_hist = (carried && penalized).then(|| {
        (
            Channel::new([vocab], dtype::f32).queued(),
            Channel::new([vocab], dtype::f32).queued(),
        )
    });

    let prompt = Channel::from(vec![0i32; request.prompt]);
    let prompt_indptr = Channel::from([0u32, request.prompt as u32]);
    let fold = Channel::from(vec![u32::MAX]);
    let mut prefill = Builder::new(vocab, page_size);
    prefill.outputs(vec![outputs.to_vec()]);
    prefill.bind_port(Port::EmbedTokens, &prompt);
    prefill.bind_port(Port::EmbedIndptr, &prompt_indptr);
    prefill.bind_port(Port::Fold, &fold);
    let g0 = Channel::new([1], dtype::i32);
    g0.note_host_take();
    let slots = Slots {
        rng: Channel::from([request.seed, 0u32]),
        hist: histogram(vocab, penalized),
        mask: None,
        logprobs: logprob_channels(shape.logprobs, 1),
    };
    let knobs = KnobChannels::new(shape, request);
    let carried_on = carried.then(|| (tok_in.clone(), carried_hist.clone()));
    prefill.stage(Stage::Epilogue, move || {
        sampler::prefill(vocab, &knobs.read(), &slots, &g0, carried_on.as_ref());
    });
    let prefill = prefill.build().map_err(|error| error.to_string())?;

    let lane = Channel::from([0u32, 1]);
    let fold = Channel::from(vec![u32::MAX]);
    let mut decode = Builder::new(vocab, page_size);
    decode.outputs(vec![outputs.to_vec()]);
    decode.bind_port(Port::EmbedTokens, &tok_in);
    decode.bind_port(Port::EmbedIndptr, &lane);
    decode.bind_port(Port::Fold, &fold);
    let out = Channel::new([1], dtype::i32).capacity(capacity);
    out.note_host_take();
    let slots = Slots {
        rng: Channel::from([request.seed ^ 0x9e37, 1u32]),
        hist: carried_hist.or_else(|| histogram(vocab, penalized)),
        mask: None,
        logprobs: logprob_channels(shape.logprobs, capacity),
    };
    let knobs = KnobChannels::new(shape, request);
    let (fed, landed) = (tok_in.clone(), out.clone());
    decode.stage(Stage::Epilogue, move || {
        sampler::decode(vocab, &knobs.read(), &slots, &fed, &landed);
    });
    let decode = decode.build().map_err(|error| error.to_string())?;
    Ok([prefill, decode])
}

/// The standard shapes' programs at the served model's `vocab`, by name.
///
/// # Errors
///
/// The first shape whose program does not trace, by name.
pub fn traces(
    vocab: u32,
    page_size: u32,
    outputs: &[OutputDecl],
    capacity: u32,
) -> Result<Vec<(String, Traced)>, String> {
    let mut all = Vec::new();
    for (name, shape) in &sampler::standard() {
        let [prefill, decode] = passes(vocab, page_size, outputs, capacity, shape, &PREWARM)
            .map_err(|error| format!("the standard sampler `{name}`: {error}"))?;
        all.push((format!("{name} (prompt)"), prefill));
        all.push((format!("{name} (decode)"), decode));
    }
    all.push(("the built-in `prefill`".to_owned(), builtin_prefill(vocab, page_size, outputs)?));
    Ok(all)
}

/// The built-in `prefill` program's one pass (`builtins/inferlets/prefill`):
/// its prompt embedded, every row folded, and its end awaited through the
/// argmax of the last row's logits. Its channels are made in that program's
/// order; its prompt's length is data ([`super::program::register`]), so
/// this is the program a prefill of any length registers.
fn builtin_prefill(vocab: u32, page_size: u32, outputs: &[OutputDecl]) -> Result<Traced, String> {
    let toks = Channel::from(vec![0i32; PREWARM.prompt]);
    let indptr = Channel::from([0u32, PREWARM.prompt as u32]);
    let done = Channel::new([1], dtype::i32);
    done.note_host_take();
    let fold = Channel::from(vec![u32::MAX]);
    let mut pass = Builder::new(vocab, page_size);
    pass.outputs(vec![outputs.to_vec()]);
    pass.bind_port(Port::EmbedTokens, &toks);
    pass.bind_port(Port::EmbedIndptr, &indptr);
    pass.bind_port(Port::Fold, &fold);
    pass.stage(Stage::Epilogue, move || {
        done.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
    });
    pass.build()
        .map_err(|error| format!("the built-in `prefill`: {error}"))
}

/// The standard samplers as the served model registers them: traced at its
/// vocabulary and default lane's outputs, registered with the runtime as a
/// guest's program is, and with what `backend`'s host codegen emits.
///
/// # Errors
///
/// A sampler that does not trace or register, by name: the load fails
/// rather than build it after ready. `rows` is the most rows one fire binds,
/// which a request's prompt channel is declared at
/// ([`super::program::register`]): the prompt pass registered here is then
/// the one a request of any prompt length registers.
pub fn registrations(
    backend: Option<&str>,
    rows: u32,
) -> anyhow::Result<Vec<::engine::ProgramRegistration>> {
    let model = crate::model::model();
    let lane = model
        .lane(&[])
        .map_err(|error| anyhow::anyhow!("the standard samplers' lane: {error}"))?;
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
        // A model that lands no logits samples nothing.
        return Ok(Vec::new());
    };
    let capacity = u32::try_from(crate::scheduler::channel_capacity().max(2)).unwrap_or(u32::MAX);
    traces(vocab, model.kv_page_size(), &outputs, capacity)
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .map(|(name, traced)| {
            let registration = registration(traced, &outputs, rows)
                .map_err(|error| anyhow::anyhow!("the standard sampler `{name}`: {error}"))?;
            Ok(super::program::with_host_codegen(&registration, backend).into_owned())
        })
        .collect()
}

fn registration(
    traced: Traced,
    outputs: &[OutputDecl],
    rows: u32,
) -> anyhow::Result<::engine::ProgramRegistration> {
    let program =
        super::program::register(traced.encode(), vec![outputs.to_vec()], None, Some(rows))?;
    Ok(::engine::ProgramRegistration {
        program_hash: program.hash,
        launch: program.launch().clone(),
        reference_ptir: program.bytes.clone(),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_ir::registry::{Axis, EngineCaps};

    const VOCAB: u32 = 151_936;

    fn outputs() -> Vec<OutputDecl> {
        vec![OutputDecl {
            name: "logits".into(),
            axis: Axis::Lane,
            width: VOCAB,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        }]
    }

    /// Every standard sampler, prompt and decode at Qwen3's vocabulary,
    /// planned with the forms Metal states (`Backend::library_forms`), forms
    /// only library regions Metal's emitter serves: the boot registers them
    /// all, and a region Metal declines fails the load (e.g. the
    /// temperature-scaled draw formed as a `NucleusSample` region). The
    /// control: planned with every
    /// form, a region Metal declines is formed, so the check bites.
    #[test]
    fn metal_plans_every_standard_sampler_in_forms_it_serves() {
        use eta_compiler::codegen::metal::validate::library_region_valid;
        use eta_compiler::codegen::program::Backend;
        use eta_compiler::plan::{LibraryForms, RegionKind, compile_bound_with};
        let traced = traces(VOCAB, 16, &outputs(), 8).expect("the standard samplers trace");
        let declined = |forms: LibraryForms| -> Vec<String> {
            let mut declined = Vec::new();
            for (name, traced) in &traced {
                let bound =
                    eta_ir::validate::bind(traced.container().clone(), vec![outputs()], EngineCaps::default())
                        .expect("the sampler binds");
                for (at, stage) in compile_bound_with(&bound, forms).iter().enumerate() {
                    for region in &stage.fused.regions {
                        if matches!(region.kind, RegionKind::Library(_))
                            && !library_region_valid(stage, region)
                        {
                            declined.push(format!("{name}: stage {at} {:?}", region.kind));
                        }
                    }
                }
            }
            declined
        };
        let metal = declined(Backend::Metal.library_forms());
        assert!(metal.is_empty(), "Metal declines regions planned in its forms: {metal:?}");
        assert!(
            !declined(LibraryForms::ALL).is_empty(),
            "planned with every form, a standard sampler forms a region Metal declines"
        );
    }

    /// The regions `backend`'s emitter gives a program, each by its text
    /// under one symbol: what engine-cuda and engine-metal key a region's
    /// module by. For `wgsl`, the step calls its stages dispatch, each by its
    /// module's text: the pipelines engine-vulkan and engine-wgpu run it on.
    fn regions(traced: &Traced, backend: &str) -> std::collections::BTreeSet<String> {
        use eta_compiler::codegen::wgsl;
        let mut registry =
            super::super::program::Registry::new(std::num::NonZeroUsize::new(1).unwrap());
        let program = registry
            .register(
                traced.encode(),
                vec![outputs()],
                EngineCaps::default(),
                eta_ir::validate::Budget {
                    room_bytes: u64::MAX,
                    submit: None,
                },
                None,
                None,
            )
            .expect("registers");
        if backend == "wgsl" {
            let calls = wgsl::step_calls();
            return program
                .launch()
                .plans
                .iter()
                .flat_map(|plan| {
                    wgsl::emit_launch_steps("ptir_step", plan)
                        .expect("lowers to steps")
                        .steps
                })
                .map(|step| wgsl::step_module("ptir_step", &calls[step.call as usize]))
                .collect();
        }
        program
            .emitted(backend)
            .expect("emits")
            .kernels
            .iter()
            .filter(|kernel| !kernel.source.is_empty())
            .map(|kernel| kernel.source.replace(&kernel.entry_name, "ptir_region"))
            .collect()
    }

    /// A standard prompt pass and the built-in `prefill` register, at the
    /// row bound, the one program the prewarm registered, whatever the
    /// prompt's length: the length is data, not the program's identity.
    #[test]
    fn a_prompt_of_any_length_is_the_prewarmed_program() {
        const ROWS: u32 = 1024;
        let mut registry =
            super::super::program::Registry::new(std::num::NonZeroUsize::new(64).unwrap());
        let mut hash = |traced: &Traced| {
            registry
                .register(
                    traced.encode(),
                    vec![outputs()],
                    EngineCaps::default(),
                    eta_ir::validate::Budget {
                        room_bytes: u64::MAX,
                        submit: None,
                    },
                    None,
                    Some(ROWS),
                )
                .expect("registers")
                .hash
        };
        let prewarmed = traces(VOCAB, 16, &outputs(), 8).expect("traces");
        let (_, shape) = &sampler::standard()[0];
        for prompt in [1, 37, 300, 1500] {
            let request = Request { prompt, ..PREWARM };
            let [asked, _] = passes(VOCAB, 16, &outputs(), 8, shape, &request).expect("traces");
            assert_eq!(hash(&asked), hash(&prewarmed[0].1), "a {prompt}-token prompt");
        }
        let builtin = &prewarmed.last().expect("the built-in `prefill`").1;
        assert_eq!(hash(builtin), hash(builtin));
        assert_eq!(registry.counts().0, 2, "one program each: the prompt pass and `prefill`");
    }

    /// Every standard shape's prompt and decode programs, traced through
    /// the SDK's own stage bodies (`eta_dsl::sampler`, which the built-in
    /// inferlets run), register the regions the prewarm built, whatever a
    /// request's knob values, temperature, seed and prompt length: a change
    /// to the SDK's sampler that a request's values reach, or to the shapes
    /// the prewarm covers, fails here.
    #[test]
    fn a_standard_request_registers_the_regions_the_prewarm_built() {
        let prewarmed = traces(VOCAB, 16, &outputs(), 8).expect("traces");
        let standard = sampler::standard();
        assert_eq!(prewarmed.len(), 2 * standard.len() + 1, "and the built-in `prefill`");
        let requests = [
            Request {
                temperature: 0.7,
                knob: 0.9,
                top_k: 7,
                seed: 1234,
                prompt: 37,
            },
            Request {
                temperature: 1.3,
                knob: 0.5,
                top_k: 20,
                seed: 7,
                prompt: 280,
            },
        ];
        for ((name, shape), built) in standard.iter().zip(prewarmed.chunks(2)) {
            for request in &requests {
                let asked = passes(VOCAB, 16, &outputs(), 8, shape, request).expect("traces");
                for (pass, (built, asked)) in
                    ["prompt", "decode"].iter().zip(built.iter().zip(&asked))
                {
                    for backend in ["cuda", "metal", "wgsl"] {
                        let (built, asked) = (regions(&built.1, backend), regions(asked, backend));
                        assert!(
                            !built.is_empty(),
                            "{name} ({pass}, {backend}) emits no region"
                        );
                        assert_eq!(
                            built, asked,
                            "{name} ({pass}, {backend}): a request at temperature {} builds a \
                             region the prewarm did not",
                            request.temperature
                        );
                    }
                }
            }
        }
    }
}
