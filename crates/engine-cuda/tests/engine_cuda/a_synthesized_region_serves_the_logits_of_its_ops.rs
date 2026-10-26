#![cfg(feature = "cuda")]

//! A model served with `compiler.synth_regions` on (CUDA's default), each
//! chain of elementwise and row ops one generated kernel, answers the
//! logits it answers with the flag off, where the chain's ops run as their
//! hand kernels: a prefill and the decode steps after it, armed and
//! captured. Qwen3's residual norms; Gemma 4's sandwich norms with their
//! layer scale, its scaled per-layer embedding and its q's norm and split
//! turn (a 512-wide global head's pair-packed); dots3's k row's norm and
//! turn. CUDA forms no hand fused op for any of them.
//!
//! Close, not bit for bit. A chain rounds every value it defines where the
//! unfused ops store it, its stream among them (read on by the norm as
//! stored, even where nothing past the chain reads it), and sums a norm in
//! its own order (`kernels_cuda`'s `elemwise::synth` tests hold it to the
//! kernels-ir interpreter within each op's tolerance). Both runs decode one
//! sequence, the reference's greedy picks, and each step's logits stay
//! within a share of their spread or of the ops' own carry ([`close`]).

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const PAGE: u32 = 16;

/// Each chain's ops as their own kernels, and each matmul's epilogue.
const OPS: &str = "[compiler]\nsynth_regions = false\nsynth_anchors = false\n";

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().expect("one at a time lock poisoned")
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fnv(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A bf16 (f32 where the plan states one) checkpoint of `trace`'s own
/// planes, seeded by name: norms near one, the rest small.
fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let wide = match param.dtype {
            Dtype::Bf16 => false,
            Dtype::F32 => true,
            other => panic!("`{}` is {other:?}, neither bf16 nor f32", param.name),
        };
        let count: usize = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let mut seed = fnv(&param.name);
        let mut bytes = Vec::with_capacity(count * if wide { 4 } else { 2 });
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            if wide {
                bytes.extend_from_slice(&value.to_le_bytes());
            } else {
                bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        writer
            .add(
                param.name.as_str(),
                param.shape.clone(),
                if wide {
                    ztensor::Leaf::F32
                } else {
                    ztensor::Leaf::BF16
                },
                &bytes,
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// A catalog model's trace and a seeded checkpoint of its planes.
struct Fixture {
    model: &'static str,
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    _dir: Scratch,
}

fn fixture(model: &'static str) -> Fixture {
    fixture_of(
        model,
        poem_compiler::catalog::trace_of(model, Dtype::Bf16, Dtype::Bf16, Platform::Cuda),
    )
}

/// [`fixture`] of the trace packed as a deployment's is (`pack::pack_as`):
/// q, k and v one product, a gate and an up one packed bank, so a norm
/// before them is one product's operand alone and folds into its anchor.
fn packed_fixture(model: &'static str) -> Fixture {
    let trace = poem_compiler::catalog::trace_of(model, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let shape = poem_compiler::pack::shape(&trace);
    fixture_of(model, poem_compiler::pack::pack_as(trace, &shape))
}

fn fixture_of(model: &'static str, trace: Trace) -> Fixture {
    let dir = std::env::temp_dir().join(format!("pie-synth-{model}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    let container = dir.join("model.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes is read by the plan's own names");
    drop(source);
    Fixture {
        model,
        trace,
        contract,
        container,
        _dir: Scratch(dir),
    }
}

/// The model loaded for one lane of `tokens` a fire, graphs on, under the
/// profile `overrides` states.
fn load(fixture: &Fixture, overrides: &str, tokens: u32) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, tokens),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: 2 * tokens,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::profile(overrides)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads under {overrides:?}: {why}", fixture.model))
}

fn argmax(logits: &[f32]) -> u32 {
    let (at, _) = logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(at, best), (i, &v)| {
            if v > best { (i, v) } else { (at, best) }
        });
    u32::try_from(at).expect("a vocab index")
}

/// `tokens` fired as slot 0's next step: its logits.
fn step(shell: &mut Shell, trace: &Trace, tokens: &[u32]) -> Vec<f32> {
    let word = trace.facts.word(&Request::new(
        u32::try_from(tokens.len()).expect("a count"),
        false,
    ));
    let mut rows = shell
        .fire(&[Lane {
            slot: 0,
            word,
            tokens,
        }])
        .expect("the step fires");
    rows.swap_remove(0)
}

fn prompt(tokens: u32) -> Vec<u32> {
    (0..tokens).map(|t| (t * 7 + 11) % 1000).collect()
}

/// `got` is `want` bit for bit, step by step.
fn bit_for_bit(got: &[Vec<f32>], want: &[Vec<f32>], against: &str) {
    assert_eq!(got.len(), want.len());
    for (step, (got, want)) in got.iter().zip(want).enumerate() {
        assert_eq!(got.len(), want.len(), "step {step}: the logits' width");
        let apart: Vec<usize> = (0..got.len())
            .filter(|&at| got[at].to_bits() != want[at].to_bits())
            .collect();
        assert!(
            apart.is_empty(),
            "step {step}: {} of {} logits are not {against}'s, the first at {}: {} and {}",
            apart.len(),
            got.len(),
            apart[0],
            got[apart[0]],
            want[apart[0]]
        );
    }
}

/// How far apart two runs' logits are at each step: the largest
/// deviation of a logit, and the reference's spread (its largest logit
/// less its least).
fn deviations(got: &[Vec<f32>], want: &[Vec<f32>]) -> Vec<(f32, f32)> {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(got, want)| {
            assert_eq!(got.len(), want.len(), "the logits' width");
            let (lo, hi) = want
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            let deviation = got
                .iter()
                .zip(want)
                .fold(0f32, |most, (g, w)| most.max((g - w).abs()));
            (deviation, hi - lo)
        })
        .collect()
}

/// A generated kernel sums its rows in its own order, so its logits are
/// the unfused ops' up to how the model carries a last-bit difference: a
/// deep random-weight miniature, or a routed one picking an expert apart,
/// carries one far. That carry is measured on the ops themselves, served
/// again in another order of their own (the norms' register forms,
/// `kernel.elemwise.rows_vec`). Each step's logits then stay within
/// [`SPREAD`] of the reference's spread, or twice the ops' own carry where
/// that is wider, and the top token is the reference's unless the
/// reference's gap to it is within the deviation (a near tie either run
/// may break).
fn close(got: &[Vec<f32>], want: &[Vec<f32>], carry: &[Vec<f32>], against: &str) {
    let carried = deviations(carry, want);
    for (step, (((deviation, spread), (own, _)), (got, want))) in deviations(got, want)
        .into_iter()
        .zip(carried)
        .zip(got.iter().zip(want))
        .enumerate()
    {
        let bound = (SPREAD * spread).max(2.0 * own);
        eprintln!(
            "{against}, step {step}: logits at most {deviation} apart over a spread of {spread}; \
             the ops in another order {own} apart"
        );
        assert!(
            deviation <= bound,
            "step {step}: a logit {deviation} from {against}'s (spread {spread}, the ops' own \
             carry {own})"
        );
        let (top, picked) = (argmax(want), argmax(got));
        let gap = want[top as usize] - want[picked as usize];
        assert!(
            top == picked || gap <= deviation,
            "step {step}: token {picked} tops the logits, {against} token {top} by {gap}"
        );
    }
}

/// A step's logits within this share of their spread of the reference's.
const SPREAD: f32 = 2e-2;

/// The norms' register forms: the unfused ops summing in another order.
const NUDGED: &str = "[kernel]\nelemwise.rows_vec = true\n";

const TOKENS: u32 = 256;
const DECODES: usize = 4;

/// The models served: Qwen3's residual norms, Gemma 4's sandwich norms.
const MODELS: [&str; 5] = [
    "qwen3-micro-text",
    "gemma4-e4b-mini-l1",
    "dots3-note-mini",
    "gemma4-e4b-mini-l30",
    "gemma4-e4b-mini-l36",
];

/// `fixture` served under `overrides`: a prefill's logits, then each greedy
/// decode step's after it; `generates` whether the load generates a kernel.
fn served(fixture: &Fixture, overrides: &str, generates: bool) -> Vec<Vec<f32>> {
    served_along(fixture, overrides, generates, None)
}

/// [`served`], each decode step fed the token `along`'s step before it
/// picks where given (a reference's, so two kernels decode one sequence).
fn served_along(
    fixture: &Fixture,
    overrides: &str,
    generates: bool,
    along: Option<&[Vec<f32>]>,
) -> Vec<Vec<f32>> {
    served_from(fixture, overrides, generates, along, 37)
}

/// [`served_along`] from a prompt of `prompted` tokens.
fn served_from(
    fixture: &Fixture,
    overrides: &str,
    generates: bool,
    along: Option<&[Vec<f32>]>,
    prompted: u32,
) -> Vec<Vec<f32>> {
    let model = fixture.model;
    let mut shell = load(fixture, overrides, TOKENS);
    assert_eq!(
        !shell.regions().is_empty(),
        generates,
        "{model} under {overrides:?}: the load generates a kernel exactly where a chain folds"
    );
    shell.open(0).expect("slot 0 opens");
    let mut steps = vec![step(&mut shell, &fixture.trace, &prompt(prompted))];
    for at in 0..DECODES {
        let next = argmax(along.map_or_else(|| steps.last().expect("a step"), |along| &along[at]));
        steps.push(step(&mut shell, &fixture.trace, &[next]));
    }
    steps
}

#[test]
fn a_synthesized_region_serves_the_logits_of_its_ops() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    for model in MODELS {
        let fixture = fixture(model);
        let ops = served(&fixture, OPS, false);
        let carry = served_along(&fixture, &format!("{OPS}{NUDGED}"), false, Some(&ops));
        let synth = served_along(&fixture, "", true, Some(&ops));
        close(&synth, &ops, &carry, &format!("{model}'s unfused ops"));
    }
}

/// Each matmul and the pointwise ops over its product served as one
/// anchored region (`compiler.synth_anchors`, the hand rules
/// `fuse.matmul_bias`, `fuse.matmul_softcap` and `fuse.matmul_glu` gone
/// from CUDA's list), its product read by the epilogue as it is summed,
/// against the products and their ops as their own kernels, the row chains
/// off in both: GLM-4.5's and GLM-4.7's biased projections (and the
/// residual adds after their outputs), Gemma 4's GEGLU and soft-capped
/// logits, and Qwen3's attention prologue (its packed qkv product split, q's
/// and k's heads normed and turned, each piece landing its own plane: an
/// epilogue of one head's span), its SwiGLU and the residual adds after its
/// output and down projections. Each model packed as its deployment is, so
/// an RMS norm before a product's epilogue (a packed gate and up's, q, k
/// and v's where nothing else reads the normed row) folds into it
/// (`compiler.fold_norms`): its gain folded into the bank at load, each
/// row's scale the contraction's own. A prefill of a few rows (the contraction skeletons, as the tuner
/// picks) and one past every skeleton's (cuBLASLt's bias, the unfused leg),
/// then the decode steps (the gemv), held as a routed model's logits are
/// ([`close_past_routing`]).
#[test]
fn an_anchored_region_serves_the_logits_of_its_ops() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    const ROWS_OFF: &str = "[compiler]\nsynth_regions = false\n";
    for (model, prompts, routed) in ANCHORED {
        let fixture = packed_fixture(model);
        for &prompted in prompts {
            let ops = served_from(&fixture, OPS, false, None, prompted);
            let carry = served_from(
                &fixture,
                &format!("{OPS}{REORDERED}"),
                false,
                Some(&ops),
                prompted,
            );
            let anchored = served_from(&fixture, ROWS_OFF, true, Some(&ops), prompted);
            let against = format!("{model}'s unfused products ({prompted} rows)");
            if routed {
                close_past_routing(&anchored, &ops, &carry, &against);
            } else {
                close(&anchored, &ops, &carry, &against);
            }
        }
    }
}

/// [`close`] for a routed model, whose experts a last-bit difference may
/// pick apart: the top token is the reference's at every step (or its
/// runner-up within twice what the ops in another order move a logit, a tie
/// the ops break themselves), and the logits stay within [`close`]'s bound at
/// most steps. A step answers for one row (the prompt's last, or the token
/// decoded), and a last-bit difference may route that row to other experts
/// than the reference's: a step past the bound with the reference's top
/// token is let through with a note, a minority of them.
fn close_past_routing(got: &[Vec<f32>], want: &[Vec<f32>], carry: &[Vec<f32>], against: &str) {
    let carried = deviations(carry, want);
    let mut routed_apart = 0usize;
    for (step, (((deviation, spread), (own, _)), (got, want))) in deviations(got, want)
        .into_iter()
        .zip(carried)
        .zip(got.iter().zip(want))
        .enumerate()
    {
        let bound = (SPREAD * spread).max(2.0 * own);
        let (top, picked) = (argmax(want), argmax(got));
        eprintln!(
            "{against}, step {step}: logits at most {deviation} apart over a spread of {spread}; \
             the ops in another order {own} apart; top-1 {}",
            if top == picked { "agrees" } else { "differs" }
        );
        // A tie the ops break themselves is no disagreement: the
        // reference's own top two within what the bound grants each logit
        // past the spread's share, twice what the ops in another order move
        // one (each of the two may move so, the one up and the other down).
        let gap = want[top as usize] - want[picked as usize];
        assert!(
            top == picked || gap <= 2.0 * own,
            "step {step}: token {picked} tops the logits, {against} token {top} by {gap} (the \
             ops' own carry {own})"
        );
        if deviation > bound {
            eprintln!(
                "note: {against}, step {step}: its row {deviation} apart, past {bound}, the top \
                 token the reference's: it routes to other experts than the reference's"
            );
            routed_apart += 1;
        }
    }
    assert!(
        routed_apart * 2 < got.len(),
        "{routed_apart} of {} steps past the bound: more than a routing tie apart",
        got.len()
    );
}

/// The unfused ops summing in another order, their products too: an
/// anchored region's product is a contraction skeleton's or cuBLASLt's own
/// algorithm's, each its own order, so its carry is measured on the ops'
/// products in another algorithm's (one cuBLASLt heuristic, a smaller
/// workspace).
const REORDERED: &str =
    "[kernel]\nelemwise.rows_vec = true\ngemm.heuristics = 1\ngemm.workspace_mb = 4\n";

/// The models served by [`an_anchored_region_serves_the_logits_of_its_ops`],
/// each from a few rows (the contraction skeletons) and past every skeleton's,
/// and whether it routes its rows to experts.
const ANCHORED: [(&str, &[u32], bool); 4] = [
    ("glm45-air-mini", &[37, 100], true),
    ("glm47-mini", &[37, 100], true),
    ("gemma4-e4b-mini-l1", &[37, 100], false),
    ("qwen3-micro-text", &[37, 100], false),
];

/// Two models loaded and served at once in one process, a thread each, as a
/// server holding two models serves them: each answers, bit for bit, what
/// it answers served alone. Their generated kernels are keyed apart (body,
/// width, the planes they read) and each load arms and captures its own
/// graphs while the other's fire. A load's synchronous copies run on its
/// thread's own stream, not the legacy one, which waits on every blocking
/// stream of the context, one the other thread is capturing into among
/// them, and so would invalidate that capture
/// (`cudaErrorStreamCaptureInvalidated`, 901; the copy itself
/// `cudaErrorStreamCaptureImplicit`, 906).
#[test]
fn two_models_served_at_once_answer_what_each_answers_alone() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixtures: Vec<Fixture> = MODELS[..2].iter().copied().map(fixture).collect();
    let alone: Vec<Vec<Vec<f32>>> = fixtures.iter().map(|f| served(f, "", true)).collect();
    // Each loaded and served again and again, so one's loads (its weights'
    // uploads) land beside the other's arming (its captures) and its fires.
    const ROUNDS: usize = 4;
    let together: Vec<Vec<Vec<Vec<f32>>>> = std::thread::scope(|scope| {
        let serving: Vec<_> = fixtures
            .iter()
            .map(|f| scope.spawn(move || (0..ROUNDS).map(|_| served(f, "", true)).collect()))
            .collect();
        serving
            .into_iter()
            .map(|thread| thread.join().expect("the model serves"))
            .collect()
    });
    for ((fixture, alone), rounds) in fixtures.iter().zip(&alone).zip(&together) {
        for (round, together) in rounds.iter().enumerate() {
            bit_for_bit(
                together,
                alone,
                &format!("{} served alone (round {round})", fixture.model),
            );
        }
    }
}
