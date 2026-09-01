//! **A CONDITIONAL GRAPH NODE, RECORDED AND REPLAYED BOTH WAYS** — the
//! mechanism gate for `engine_cuda::device::conditional` and
//! `kernels_cuda::graph`.
//!
//! The claim under test is the one `Fault::Unlowered` used to say could not be
//! made: that this shell can put a decision INSIDE a graph. Everything else
//! about a conditional is policy — which region gets one, and whether it pays
//! — and none of that is testable until the node exists.
//!
//! ```text
//! capture                       replay, live = 1     replay, live = 0
//! ------------------------      ------------------   ------------------
//! mint a handle on the graph    the setter stores 1  the setter stores 0
//! setter kernel (device-side)   the body runs        the body is skipped
//! IF node at the frontier       out += 1             out unchanged
//! body captured on its own      after += 1           after += 1
//!   stream: out += 1
//! after the bracket: after += 1
//! ```
//!
//! **THE PREDICATE IS A POINTER AND NOT AN IMMEDIATE**, which is the whole
//! point. One exec, four launches, two answers: an immediate would have been
//! frozen into the setter's node parameters at capture and the second pair of
//! launches would say what the first pair said. What moves between them is one
//! device byte the host rewrote — the same way a fire's row counts move — and
//! the graph reads it for itself.
//!
//! `after` is the control. It stands behind the bracket on the parent stream
//! and increments on every launch under both arms, which is what says the
//! conditional SKIPPED its body rather than that the exec stopped: a graph
//! that fell over would leave `after` unmoved too.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!   --test conditional_nodes -- --nocapture --test-threads=1
//! ```
//!
//! # Gating
//!
//! Behind `_cuda`, and skipped at run time when the machine has no device —
//! `graph_replay.rs`'s rule, not `#[ignore]`.
#![cfg(feature = "_cuda")]

use core::ffi::c_void;
use std::sync::{Mutex, MutexGuard, PoisonError};

use cudarc::driver::sys as dr;
use cudarc::runtime::sys as rt;

use engine_cuda::device::conditional::Kind;
use engine_cuda::device::{Buffer, Graph, conditional};
use kernels_cuda::Ctx;
use kernels_cuda::graph::{Arm, set_conditional, set_conditional_byte, set_switch};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The body, and the control behind it. One symbol, so that what distinguishes
/// the two nodes is where they stand and not what they are.
const SOURCE: &str = r#"
extern "C" __global__ void bump(float* out, int at) {
    out[at] += 1.0f;
}
"#;

/// A bound device, two streams, the bump kernel, and somewhere to count.
struct Rig {
    main: *mut c_void,
    body: *mut c_void,
    bump: dr::CUfunction,
    /// One `f32` counter per slot. `[0]` is the `IF` body's and `[1]` the
    /// control's; a `SWITCH` gate uses one per arm and puts its control last.
    counts: Buffer,
    /// One byte: the predicate the setter kernel reads.
    live: Buffer,
    ctx: Ctx,
}

impl Rig {
    fn open(what: &str) -> Option<Rig> {
        if !engine_cuda::device::present() {
            eprintln!("skipping {what}: no CUDA device on this machine");
            return None;
        }
        // SAFETY: live out-parameters, and this thread is the one that will
        // capture. `cudaSetDevice` makes the primary context current for the
        // driver calls this test and `device::conditional` both make.
        unsafe {
            assert_eq!(rt::cudaSetDevice(0), rt::cudaError::cudaSuccess);
            let mut main: rt::cudaStream_t = core::ptr::null_mut();
            let mut body: rt::cudaStream_t = core::ptr::null_mut();
            assert_eq!(
                rt::cudaStreamCreate(&raw mut main),
                rt::cudaError::cudaSuccess
            );
            assert_eq!(
                rt::cudaStreamCreate(&raw mut body),
                rt::cudaError::cudaSuccess
            );

            let ptx = cudarc::nvrtc::compile_ptx(SOURCE).expect("the body kernel compiles");
            let image = std::ffi::CString::new(ptx.to_src()).expect("ptx holds no NUL");
            let mut module: dr::CUmodule = core::ptr::null_mut();
            assert_eq!(
                dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()),
                dr::CUresult::CUDA_SUCCESS
            );
            let name = std::ffi::CString::new("bump").expect("a literal holds no NUL");
            let mut bump: dr::CUfunction = core::ptr::null_mut();
            assert_eq!(
                dr::cuModuleGetFunction(&raw mut bump, module, name.as_ptr()),
                dr::CUresult::CUDA_SUCCESS
            );

            let main: *mut c_void = main.cast();
            Some(Rig {
                main,
                body: body.cast(),
                bump,
                counts: Buffer::zeroed(32).expect("eight floats"),
                live: Buffer::zeroed(1).expect("one byte"),
                ctx: Ctx::on(main),
            })
        }
    }

    /// Enqueue `counts[at] += 1` on `stream`.
    fn launch(&self, stream: *mut c_void, at: i32) {
        let mut ptr = self.counts.ptr();
        let mut at = at;
        let mut args: [*mut c_void; 2] = [
            core::ptr::from_mut(&mut ptr).cast(),
            core::ptr::from_mut(&mut at).cast(),
        ];
        // SAFETY: `args` names two live locals for the duration of the call,
        // which is all `cuLaunchKernel` needs — it copies the argument values
        // before returning, capture or no capture.
        let code = unsafe {
            dr::cuLaunchKernel(
                self.bump,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                stream.cast(),
                args.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "the body launch enqueues");
    }

    /// Write the predicate byte and wait for it to land.
    fn set_live(&mut self, live: bool) {
        self.live
            .write(0, &[u8::from(live)])
            .expect("one byte lands");
    }

    /// `(body count, control count)` — the two the `IF` gates use.
    fn read(&self) -> (f32, f32) {
        let all = self.counters();
        (all[0], all[1])
    }

    /// Every counter, in slot order.
    fn counters(&self) -> [f32; 8] {
        let mut bytes = [0u8; 32];
        self.counts.read(0, &mut bytes).expect("eight floats come back");
        let mut out = [0.0f32; 8];
        for (at, slot) in out.iter_mut().enumerate() {
            *slot = f32::from_le_bytes([
                bytes[at * 4],
                bytes[at * 4 + 1],
                bytes[at * 4 + 2],
                bytes[at * 4 + 3],
            ]);
        }
        out
    }

    /// Zero every counter, so a launch's effect is read directly rather than
    /// as a difference.
    fn clear(&mut self) {
        self.counts.write(0, &[0u8; 32]).expect("the counters clear");
    }

    fn settle(&self) {
        // SAFETY: the stream is this rig's and live.
        let code = unsafe { rt::cudaStreamSynchronize(self.main.cast()) };
        assert_eq!(code, rt::cudaError::cudaSuccess, "the stream settles");
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: both streams are this rig's, and nothing is in flight —
        // every test synchronizes before it ends.
        unsafe {
            rt::cudaStreamDestroy(self.main.cast());
            rt::cudaStreamDestroy(self.body.cast());
        }
    }
}

/// **THE GATE.** One exec, two predicates, two answers.
#[test]
fn a_recorded_if_node_takes_its_body_on_the_arm_a_device_byte_names() {
    let _serial = serialized();
    let Some(mut rig) = Rig::open("the conditional-node gate") else {
        return;
    };

    // **THE WARM PASS, AND IT IS NOT OPTIONAL.** The setter's unit is
    // compiled and its module loaded on first launch, which is host work; host
    // work inside a capture is what `Graph::capture`'s thread-local mode
    // exists to refuse. `Arm::Warm` returns before it reaches the handle, so
    // this launch loads the module and stores nothing.
    set_conditional_byte(&rig.ctx, 0, rig.live.ptr(), false, Arm::Warm)
        .expect("the setter compiles, loads and warms");
    rig.settle();

    let graph = Graph::capture(rig.main, || {
        let handle = conditional::handle(rig.main, Kind::If)?;
        // The predicate: a device-side store into the handle, from a byte the
        // host may rewrite between replays.
        set_conditional_byte(&rig.ctx, handle, rig.live.ptr(), false, Arm::Set)
            .expect("the setter enqueues into the capture");
        let cond = conditional::open(rig.main, handle, Kind::If)?;
        conditional::begin_body(rig.body, cond.body(0).expect("an IF has one body"))?;
        rig.launch(rig.body, 0);
        conditional::end_body(rig.body)?;
        // Behind the bracket, on the parent: the control.
        rig.launch(rig.main, 1);
        Ok(())
    })
    .expect("a capture holding a conditional node ends cleanly");

    let exec = graph.instantiate(rig.main).expect("it instantiates");

    // Arm one: the byte says take it.
    rig.set_live(true);
    for _ in 0..2 {
        exec.launch(rig.main).expect("the taken arm launches");
    }
    rig.settle();
    let (body, control) = rig.read();
    assert!(
        (body - 2.0).abs() < 1e-6 && (control - 2.0).abs() < 1e-6,
        "two launches with the predicate set should run the body twice: \
         body {body}, control {control}",
    );

    // Arm two: the SAME exec, the same node parameters, one byte different.
    rig.set_live(false);
    for _ in 0..2 {
        exec.launch(rig.main).expect("the skipped arm launches");
    }
    rig.settle();
    let (body, control) = rig.read();
    assert!(
        (body - 2.0).abs() < 1e-6,
        "the body ran under a predicate of 0: body {body} after two more launches",
    );
    assert!(
        (control - 4.0).abs() < 1e-6,
        "the control did not run: the exec stopped rather than skipping the \
         body — control {control} after four launches",
    );

    eprintln!("conditional node: body {body}, control {control} over four launches");
}

/// **A GRAPH HOLDING A CONDITIONAL NODE CAN STILL BE COUNTED** — the query
/// gate, and the one that would have caught the empty-script fault the day
/// this module landed.
///
/// `record::Graphs::fire_body` drops a captured stretch that recorded no
/// nodes: a cut holding only PREPARE regions captures an empty graph, and an
/// exec of nothing costs a driver call to do nothing. The count it drops on
/// used to come from `cudaGraphGetNodes` — the RUNTIME spelling — under an
/// `else { 0 }`, while this module records `CU_GRAPH_NODE_TYPE_CONDITIONAL`
/// through the DRIVER one, for the ABI reason its header states. The runtime
/// query cannot represent that node type and refuses; the refusal read as
/// "recorded nothing"; every stretch of every body holding a conditional was
/// dropped; and the key was seated with a SCRIPT OF ZERO STEPS. A fire of it
/// then launched nothing, counted a hit, and settled a frame whose readout
/// rectangle still held the last fire that actually ran.
///
/// So the claim is not "the graph is non-empty" — it is that **the count is an
/// ANSWER and not a refusal**, which is what `Option` now spells at the type.
/// `None` here is the whole fault, and it is why this asserts the variant
/// before it asserts the number.
#[test]
fn a_captured_conditional_graph_answers_its_own_node_count() {
    let _serial = serialized();
    let Some(mut rig) = Rig::open("the conditional-node count gate") else {
        return;
    };

    // The warm pass, for the reason the gate above states: host work inside a
    // capture is what the thread-local mode refuses.
    set_conditional_byte(&rig.ctx, 0, rig.live.ptr(), false, Arm::Warm)
        .expect("the setter compiles, loads and warms");
    rig.settle();

    let graph = Graph::capture(rig.main, || {
        let handle = conditional::handle(rig.main, Kind::If)?;
        set_conditional_byte(&rig.ctx, handle, rig.live.ptr(), false, Arm::Set)
            .expect("the setter enqueues into the capture");
        let cond = conditional::open(rig.main, handle, Kind::If)?;
        conditional::begin_body(rig.body, cond.body(0).expect("an IF has one body"))?;
        rig.launch(rig.body, 0);
        conditional::end_body(rig.body)?;
        rig.launch(rig.main, 1);
        Ok(())
    })
    .expect("a capture holding a conditional node ends cleanly");

    let counted = graph.nodes();
    assert!(
        counted.is_some(),
        "the driver would not count a graph holding a conditional node. That answer \
         is indistinguishable from an empty graph at every caller that reads it as a \
         number, and `record::Graphs::fire_body` DROPS a stretch that recorded \
         nothing — which is how a body came to be armed with a script that launches \
         nothing at all",
    );
    let counted = counted.unwrap_or(0);
    assert!(
        counted > 0,
        "this capture holds a conditional node, its body's launch and the control \
         behind it, and the count came back {counted}",
    );
    // The edge count is the same question of the same graph, and it is asked
    // the same way for the same reason.
    assert!(
        graph.edges().is_some(),
        "the node count answered and the edge count did not, on one graph",
    );
    eprintln!("a conditional-bearing capture counts {counted} nodes");
}

/// **THE FORM THE FIRE PATH ACTUALLY USES**, and the arithmetic in it.
///
/// `Cursor::cond_begin` does not hand the setter a byte — it hands it a
/// window's rebased row CSR and the window's lane count, and the setter takes
/// the body iff `indptr[lanes] != 0`. That is design §5's "the kernel reads
/// the count" and it is the whole reason the predicate is the artifact's own
/// semantics rather than a second opinion about them, so it is worth one gate
/// of its own with a CSR in it.
///
/// Two windows, four lanes each, staged as real `i32` vectors:
///
/// ```text
/// live  [0, 3, 7, 12, 20]   indptr[4] = 20   ->  the body runs
/// empty [0, 0,  0,  0,  0]  indptr[4] =  0   ->  it does not
/// ```
///
/// The empty one is not a contrivance: it is exactly what
/// `model_exec::store::check::rebase` writes for a window whose class has no
/// rows in this fire, which is the case a conditional exists for.
#[test]
fn the_setter_reads_a_row_count_out_of_a_window_s_own_boundary_vector() {
    let _serial = serialized();
    let Some(mut rig) = Rig::open("the row-count predicate") else {
        return;
    };

    /// Four lanes, so the count is the fifth entry.
    const LANES: u32 = 4;
    let csr = |bounds: [i32; 5]| -> Vec<u8> {
        bounds.iter().flat_map(|v| v.to_le_bytes()).collect()
    };

    let mut boundaries = Buffer::zeroed(40).expect("two five-entry vectors");
    boundaries
        .write(0, &csr([0, 3, 7, 12, 20]))
        .expect("the live window's bounds land");
    boundaries
        .write(20, &csr([0, 0, 0, 0, 0]))
        .expect("the empty window's bounds land");

    set_conditional(&rig.ctx, 0, 0, 0, false, Arm::Warm).expect("the setter warms");
    rig.settle();

    // Two graphs, one per window, so that the two answers are two captures of
    // the same code rather than one capture asked twice.
    for (at, expected, what) in [(0u64, 1.0f32, "a window with rows"), (20, 0.0, "an empty one")] {
        let indptr = boundaries.at(at).expect("the vector is in the buffer");
        let graph = Graph::capture(rig.main, || {
            let handle = conditional::handle(rig.main, Kind::If)?;
            set_conditional(&rig.ctx, handle, indptr, LANES, false, Arm::Set)
                .expect("the setter enqueues into the capture");
            let cond = conditional::open(rig.main, handle, Kind::If)?;
            conditional::begin_body(rig.body, cond.body(0).expect("an IF has one body"))?;
            rig.launch(rig.body, 0);
            conditional::end_body(rig.body)?;
            rig.launch(rig.main, 1);
            Ok(())
        })
        .expect("the capture ends cleanly");

        let before = rig.read();
        graph
            .instantiate(rig.main)
            .expect("it instantiates")
            .launch(rig.main)
            .expect("it launches");
        rig.settle();
        let after = rig.read();

        assert!(
            (after.0 - before.0 - expected).abs() < 1e-6,
            "{what}: the body ran {} times and should have run {expected}",
            after.0 - before.0,
        );
        assert!(
            (after.1 - before.1 - 1.0).abs() < 1e-6,
            "{what}: the control did not run, so the exec stopped rather than \
             deciding",
        );
    }
}

/// **THE `SWITCH` FORM: ONE HANDLE, THREE BODIES, AND AN INDEX.**
///
/// An `IF` node asks "does this run"; a `SWITCH` asks "which of these does",
/// and the difference is not a second mechanism — it is the same node with
/// `size: arms` and a handle holding an index instead of a bool. So this is
/// the same gate as the one above with the answers spread out:
///
/// ```text
/// live arm    counters after one launch      what it says
/// --------    ---------------------------    ---------------------------
/// 0           [1, 0, 0, control 1]           the index picked arm 0
/// 1           [0, 1, 0, control 1]           the same exec, a different arm
/// 2           [0, 0, 1, control 1]
/// none        [0, 0, 0, control 1]           the quiescent default runs none
/// ```
///
/// **THE LAST ROW IS THE ONE THAT NEEDED A DECISION.** A `SWITCH` handle whose
/// default were `0` would take arm 0 on a fire where no arm has rows, because
/// "nothing stored" and "stored zero" are the same state. `Kind::quiescent`
/// mints it at `arms` — past the last body — so a group with nothing to do
/// does nothing, and the setters stay pure stores that only ever fire for a
/// live arm.
///
/// The predicate is `arms` launches of `set_switch`, each reading its OWN
/// arm's boundary vector, which is how the fire path drives it: a group's arms
/// are separate regions with separate windows and there is no vector holding
/// all their counts.
#[test]
fn a_switch_node_runs_the_arm_its_own_window_claims_and_none_when_none_claims() {
    let _serial = serialized();
    let Some(mut rig) = Rig::open("the switch-node gate") else {
        return;
    };

    /// Three arms, so there is a middle one — a two-arm switch and an `IF`
    /// with an else would be the same shape, and the index would be a bool
    /// wearing a wider type.
    const ARMS: u32 = 3;
    /// Four lanes per window, so the row count is the fifth entry.
    const LANES: u32 = 4;
    /// Where the control counter lives, behind every arm.
    const CONTROL: i32 = 3;

    let kind = Kind::Switch { arms: ARMS };
    assert_eq!(
        kind.quiescent(),
        ARMS,
        "a switch's quiescent value has to be past its last body, or an empty \
         group takes arm 0",
    );

    // One five-entry CSR per arm, `20` bytes apart.
    let mut boundaries = Buffer::zeroed(20 * ARMS as usize).expect("three windows");
    let claim = |rig: &mut Rig, boundaries: &mut Buffer, live: Option<u32>| {
        let _ = rig;
        for arm in 0..ARMS {
            // `[0, 3, 7, 12, 20]` is a window with rows; all-zero is what
            // `rebase` writes for a class this fire has none of.
            let bounds: [i32; 5] = if live == Some(arm) {
                [0, 3, 7, 12, 20]
            } else {
                [0, 0, 0, 0, 0]
            };
            let bytes: Vec<u8> = bounds.iter().flat_map(|v| v.to_le_bytes()).collect();
            boundaries
                .write(u64::from(arm) * 20, &bytes)
                .expect("an arm's bounds land");
        }
    };

    set_switch(&rig.ctx, 0, 0, 0, 0, Arm::Warm).expect("the switch setter warms");
    rig.settle();

    // **ONE CAPTURE, AND EVERY ANSWER BELOW COMES OUT OF IT.** The node's
    // parameters are frozen at instantiation; what moves between the launches
    // is three device words, which is the whole claim.
    claim(&mut rig, &mut boundaries, Some(0));
    let graph = Graph::capture(rig.main, || {
        let handle = conditional::handle(rig.main, kind)?;
        for arm in 0..ARMS {
            let indptr = boundaries
                .at(u64::from(arm) * 20)
                .expect("the arm's vector is in the buffer");
            set_switch(&rig.ctx, handle, arm, indptr, LANES, Arm::Set)
                .expect("an arm's setter enqueues into the capture");
        }
        let cond = conditional::open(rig.main, handle, kind)?;
        assert_eq!(cond.arms, ARMS, "the driver minted a body per arm");
        for arm in 0..ARMS {
            let body = cond.body(arm).expect("every arm has a body graph");
            conditional::begin_body(rig.body, body)?;
            rig.launch(rig.body, arm as i32);
            conditional::end_body(rig.body)?;
        }
        rig.launch(rig.main, CONTROL);
        Ok(())
    })
    .expect("a capture holding a three-armed switch ends cleanly");
    let exec = graph.instantiate(rig.main).expect("it instantiates");

    for live in [Some(0), Some(1), Some(2), None] {
        claim(&mut rig, &mut boundaries, live);
        rig.clear();
        exec.launch(rig.main).expect("the switch launches");
        rig.settle();
        let counters = rig.counters();

        for arm in 0..ARMS {
            let want = f32::from(u8::from(live == Some(arm)));
            assert!(
                (counters[arm as usize] - want).abs() < 1e-6,
                "with arm {live:?} live, arm {arm}'s body ran {} times and should \
                 have run {want}: {counters:?}",
                counters[arm as usize],
            );
        }
        assert!(
            (counters[CONTROL as usize] - 1.0).abs() < 1e-6,
            "the control behind the switch did not run, so the exec stopped \
             rather than choosing: {counters:?}",
        );
    }

    eprintln!("switch node: {ARMS} arms, four launches of one exec, each answered by a device word");
}
