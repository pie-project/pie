//! A request's sampler knob values are data: the standard sampler's
//! programs (`eta_dsl::sampler`, traced over the channels the built-in
//! inferlets' generation loop binds) are the same program whatever a
//! request's top-p, top-k, min-p, penalties, temperature or seed, by every
//! key an engine caches its compiled code by: the program's hash, its
//! launch package (engine-xla keys its form and stage cache by the
//! package's `Debug` text), the CUDA, HIP and Metal regions' text and the
//! WGSL steps' modules. A value that reaches any of them builds after
//! ready, once for each new value a request brings.

use std::collections::BTreeSet;

use eta_compiler::codegen::launch::{self, LaunchPackage};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::codegen::wgsl;
use eta_dsl::Traced;
use eta_dsl::prelude::*;
use eta_dsl::sampler::{self, Knobs, Shape, Slots};
use eta_ir::registry::{Axis, EngineCaps, OutputDecl};
use eta_ir::validate::{Budget, bind_within};

const VOCAB: u32 = 4096;

fn outputs() -> Vec<OutputDecl> {
    vec![OutputDecl {
        name: "logits".into(),
        axis: Axis::Lane,
        width: VOCAB,
        dtype: eta_dsl::Dtype::F32,
        layers: 0,
        readout: true,
    }]
}

/// A request's values for every knob a shape may have, and its seed.
#[derive(Clone, Copy, Debug)]
struct Request {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    repetition: f32,
    frequency: f32,
    presence: f32,
    seed: u32,
}

const BASE: Request = Request {
    temperature: 1.0,
    top_p: 0.9,
    top_k: 40,
    min_p: 0.05,
    repetition: 1.1,
    frequency: 0.2,
    presence: 0.3,
    seed: 0,
};

/// The channels a pass's knobs ride in, as the built-in inferlets seed
/// them (read in the stage, never consumed): the temperature as its
/// reciprocal.
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

impl KnobChannels {
    fn new(shape: &Shape, r: &Request) -> Self {
        let f = |on: bool, v: f32| on.then(|| Channel::from([v]));
        Self {
            temperature: f(
                shape.temperature,
                sampler::inverse_temperature(r.temperature),
            ),
            top_p: f(shape.top_p, r.top_p),
            top_k: shape.top_k.then(|| Channel::from([r.top_k])),
            min_p: f(shape.min_p, r.min_p),
            repetition: f(shape.repetition, r.repetition),
            frequency: f(shape.frequency, r.frequency),
            presence: f(shape.presence, r.presence),
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

fn histogram(penalized: bool) -> Option<(Channel, Channel)> {
    penalized.then(|| {
        (
            Channel::from(vec![0.0f32; VOCAB as usize]),
            Channel::from(vec![0.0f32; VOCAB as usize]),
        )
    })
}

fn logprob_channels(logprobs: Option<u32>) -> Option<(Channel, Option<Channel>)> {
    logprobs.map(|top| {
        let values = Channel::new([1 + top], dtype::f32).capacity(4);
        values.note_host_take();
        let ids = (top > 0).then(|| {
            let ids = Channel::new([top], dtype::i32).capacity(4);
            ids.note_host_take();
            ids
        });
        (values, ids)
    })
}

/// The prompt's last fire and the decode fire of `shape` at `r`, as the
/// runtime's prewarm (`runtime::pipeline::standard`) traces them.
fn passes(shape: &Shape, r: &Request) -> [Traced; 2] {
    let penalized = penalized(shape);
    let carried = shape.logprobs.is_none();
    let tok_in = if carried {
        Channel::new([1], dtype::i32).queued()
    } else {
        Channel::from([0i32])
    };
    let carried_hist = (carried && penalized).then(|| {
        (
            Channel::new([VOCAB], dtype::f32).queued(),
            Channel::new([VOCAB], dtype::f32).queued(),
        )
    });

    let prompt = Channel::from(vec![0i32; 4]);
    let prompt_indptr = Channel::from([0u32, 4]);
    let fold = Channel::from(vec![u32::MAX]);
    let mut prefill = Builder::new(VOCAB, 16);
    prefill.outputs(vec![outputs()]);
    prefill.bind_port(Port::EmbedTokens, &prompt);
    prefill.bind_port(Port::EmbedIndptr, &prompt_indptr);
    prefill.bind_port(Port::Fold, &fold);
    let g0 = Channel::new([1], dtype::i32);
    g0.note_host_take();
    let slots = Slots {
        rng: Channel::from([r.seed, 0u32]),
        hist: histogram(penalized),
        mask: None,
        logprobs: logprob_channels(shape.logprobs),
    };
    let knobs = KnobChannels::new(shape, r);
    let carried_on = carried.then(|| (tok_in.clone(), carried_hist.clone()));
    prefill.stage(Stage::Epilogue, move || {
        sampler::prefill(VOCAB, &knobs.read(), &slots, &g0, carried_on.as_ref());
    });
    let prefill = prefill.build().expect("the prompt's fire traces");

    let lane = Channel::from([0u32, 1]);
    let fold = Channel::from(vec![u32::MAX]);
    let mut decode = Builder::new(VOCAB, 16);
    decode.outputs(vec![outputs()]);
    decode.bind_port(Port::EmbedTokens, &tok_in);
    decode.bind_port(Port::EmbedIndptr, &lane);
    decode.bind_port(Port::Fold, &fold);
    let out = Channel::new([1], dtype::i32).capacity(4);
    out.note_host_take();
    let slots = Slots {
        rng: Channel::from([r.seed ^ 0x9e37, 1u32]),
        hist: carried_hist.or_else(|| histogram(penalized)),
        mask: None,
        logprobs: logprob_channels(shape.logprobs),
    };
    let knobs = KnobChannels::new(shape, r);
    let (fed, landed) = (tok_in.clone(), out.clone());
    decode.stage(Stage::Epilogue, move || {
        sampler::decode(VOCAB, &knobs.read(), &slots, &fed, &landed);
    });
    let decode = decode.build().expect("the decode fire traces");
    [prefill, decode]
}

/// What an engine keys a program's compiled code by.
#[derive(PartialEq)]
struct Keys {
    /// The registry's program hash (`eta_ir::container_hash`).
    hash: u64,
    /// The launch package's `Debug` text: engine-xla's form and stage key.
    package: String,
    /// Each backend's region texts, the entry renamed to one symbol: what
    /// engine-cuda, engine-rocm and engine-metal key a module by.
    regions: Vec<(&'static str, BTreeSet<String>)>,
    /// The WGSL step modules the plans dispatch: engine-vulkan's and
    /// engine-wgpu's pipelines.
    wgsl: BTreeSet<String>,
}

impl Keys {
    fn of(traced: &Traced) -> Self {
        let bytes = traced.encode();
        let container = eta_ir::container::decode(&bytes).expect("decodes");
        let bound = bind_within(
            container,
            vec![outputs()],
            EngineCaps::default(),
            Budget {
                room_bytes: u64::MAX,
                submit: None,
            },
        )
        .expect("binds");
        let stages = eta_compiler::plan::compile_bound(&bound);
        let package: LaunchPackage = launch::build(&bound, &stages);
        let regions = Backend::ALL
            .iter()
            .map(|&backend| {
                let texts = emit_program(backend, &stages, &bound)
                    .iter()
                    .filter(|kernel| !kernel.source.is_empty())
                    .map(|kernel| kernel.source.replace(&kernel.entry_name, "ptir_region"))
                    .collect();
                (backend.name(), texts)
            })
            .collect();
        let calls = wgsl::step_calls();
        let wgsl = package
            .plans
            .iter()
            .flat_map(|plan| {
                wgsl::emit_launch_steps("ptir_step", plan)
                    .expect("lowers to steps")
                    .steps
            })
            .map(|step| wgsl::step_module("ptir_step", &calls[step.call as usize]))
            .collect();
        Self {
            hash: eta_ir::container_hash(&bytes),
            package: format!("{package:?}"),
            regions,
            wgsl,
        }
    }

    /// The keys `self` and `other` differ by, by name.
    fn differ(&self, other: &Self) -> Vec<&'static str> {
        let mut differ = Vec::new();
        if self.hash != other.hash {
            differ.push("program hash");
        }
        if self.package != other.package {
            differ.push("launch package (XLA)");
        }
        for ((name, a), (_, b)) in self.regions.iter().zip(&other.regions) {
            if a != b {
                differ.push(name);
            }
        }
        if self.wgsl != other.wgsl {
            differ.push("wgsl");
        }
        differ
    }
}

/// Every knob the sampler has, and the standard shapes.
fn shapes() -> Vec<(String, Shape)> {
    let mut shapes = sampler::standard();
    shapes.push((
        "every knob".into(),
        Shape {
            temperature: true,
            top_p: true,
            top_k: true,
            min_p: true,
            repetition: true,
            frequency: true,
            presence: true,
            logprobs: Some(5),
        },
    ));
    shapes
}

/// Each knob moved alone from [`BASE`], to a value no other request in
/// the test holds.
fn requests() -> Vec<(&'static str, Request)> {
    vec![
        (
            "top_p",
            Request {
                top_p: 0.77,
                ..BASE
            },
        ),
        ("top_k", Request { top_k: 7, ..BASE }),
        (
            "min_p",
            Request {
                min_p: 0.13,
                ..BASE
            },
        ),
        (
            "repetition",
            Request {
                repetition: 1.37,
                ..BASE
            },
        ),
        (
            "frequency",
            Request {
                frequency: -0.6,
                ..BASE
            },
        ),
        (
            "presence",
            Request {
                presence: 1.7,
                ..BASE
            },
        ),
        (
            "temperature",
            Request {
                temperature: 0.7,
                ..BASE
            },
        ),
        ("seed", Request { seed: 1234, ..BASE }),
    ]
}

/// Whatever a request's knob values, temperature and seed, each standard
/// shape's prompt and decode programs and a shape with every knob are the
/// same programs by every key an engine compiles by.
#[test]
fn a_sampler_knob_is_data() {
    let mut failures = Vec::new();
    for (name, shape) in shapes() {
        let base = passes(&shape, &BASE).map(|traced| Keys::of(&traced));
        for (knob, request) in requests() {
            let asked = passes(&shape, &request);
            for (pass, (base, asked)) in ["prompt", "decode"].iter().zip(base.iter().zip(&asked)) {
                let differ = base.differ(&Keys::of(asked));
                if !differ.is_empty() {
                    failures.push(format!("{name} ({pass}): {knob} changes {differ:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "a request's knob value reaches the program's identity:\n{}",
        failures.join("\n")
    );
}
