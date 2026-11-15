//! A registration is its program's ready, and the lane does not wait on
//! it: the lane starts the build and goes back to its queue, the build's
//! end reaches it as one item on that queue, and only then is the
//! registering process answered: with the id once what an instance of the
//! program holds is learnt, or with why not, nothing of the program kept.
//! A build that ends untold (dropped by its engine, or pending when the
//! lane ends) answers its process by name.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use ::engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole, SlotShape,
};
use ::engine::{ProgramBuilt, ProgramFacts, ProgramId, ProgramRegistration};

use super::{BUILD_ABANDONED, BUILD_UNTOLD, Lane, LaneCommit, LaunchReplyRx};
use crate::scheduler::stats::SchedulerStats;
use crate::scheduler::worker::QueuedItem;
use crate::store::roles::{Roles, RolesHandle};
use crate::store::shared::SharedPool;

/// The roles the lanes of these tests find for their engines
/// ([`super::roles_of`]): the registry's model 0 is whichever test
/// registered first.
static ROLES: LazyLock<Mutex<HashMap<usize, RolesHandle>>> = LazyLock::new(Mutex::default);

pub(super) fn roles_of(engine_idx: usize) -> Option<RolesHandle> {
    ROLES.lock().unwrap().get(&engine_idx).cloned()
}

const KIB: u64 = 1 << 10;
const CHUNK: u64 = 1 << 20;

/// A pool that states an `Instance` shape beside a kv list, so the lane
/// learns each program's facts.
fn roles(engine_idx: usize) -> RolesHandle {
    let shape = |id, kind, slot_bytes: u64, role| SlotShape {
        id: ShapeId(id),
        kind,
        slot_bytes,
        slots_per_chunk: u32::try_from(CHUNK / slot_bytes).unwrap(),
        sets_per_working_set: 1,
        rows: Vec::new(),
        pinned: None,
        role,
    };
    let list = SlotKind::List {
        tokens: 16,
        rate: 1,
        window: None,
    };
    let facts = ChunkFacts {
        chunk_bytes: CHUNK,
        total_chunks: 32,
        segment_chunks: 32,
        segments: SegmentFacts {
            committed: 1,
            max: 1,
            growth: Growth::Fixed,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![
            shape(0, list, 64 * KIB, SlotRole::State),
            shape(
                1,
                SlotKind::Fixed { pieces: 1 },
                4 * KIB,
                SlotRole::Instance,
            ),
        ],
        act_chunks: vec![(16, 1), (256, 4)],
        ..ChunkFacts::default()
    };
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let roles = Arc::new(Mutex::new(
        Roles::new(&device, None).expect("the pool states an instance shape"),
    ));
    ROLES.lock().unwrap().insert(engine_idx, roles.clone());
    roles
}

const PROGRAM: ProgramId = ProgramId(41);
const REFUSAL: &str = "stage 3 has no kernel for this device";

/// How the engine ends a registration's build.
enum Ends {
    /// Tells `built` inside the call, before it returns the id.
    Inside(::engine::Result<()>),
    /// Returns the id and keeps `built`, for another thread to tell.
    Held,
    /// Returns the id and drops `built` untold.
    Dropped,
    /// Refuses the registration: the call returns `Err`, `built` untold.
    Refused,
}

type Gate = crate::rt::channel::Sender<crate::rt::channel::Sender<()>>;

#[derive(Default)]
struct Seen {
    /// The `built` of each registration the engine holds, oldest first.
    held: Mutex<Vec<ProgramBuilt>>,
    facts_asked: AtomicUsize,
    fires: AtomicUsize,
}

struct Builder {
    /// How each registration ends, in the order they come.
    ends: VecDeque<Ends>,
    seen: Arc<Seen>,
    /// `program_facts` hands the test a sender and answers when it is
    /// sent on or dropped.
    gate: Option<Gate>,
}

impl ::engine::Engine for Builder {
    fn kind(&self) -> &'static str {
        "builder"
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        self.seen.fires.fetch_add(1, Ordering::SeqCst);
        Ok(::engine::FrameTicket {
            id: ::engine::FrameId(0),
            steps: Vec::new(),
        })
    }

    fn register_program(
        &mut self,
        _registration: &ProgramRegistration,
        built: ProgramBuilt,
    ) -> ::engine::Result<ProgramId> {
        match self.ends.pop_front().expect("a registration the test set") {
            Ends::Inside(result) => built(result),
            Ends::Held => self.seen.held.lock().unwrap().push(built),
            Ends::Dropped => drop(built),
            Ends::Refused => return Err(::engine::Error::Program(REFUSAL.to_string())),
        }
        Ok(PROGRAM)
    }

    fn program_facts(&self, program: ProgramId) -> ::engine::Result<ProgramFacts> {
        assert_eq!(program, PROGRAM);
        self.seen.facts_asked.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            let (go, wait) = crate::rt::channel::bounded(1);
            gate.send(go).unwrap();
            let _ = wait.recv();
        }
        Ok(ProgramFacts::default())
    }
}

type Answer = tokio::sync::oneshot::Receiver<anyhow::Result<ProgramId>>;

fn spawn<const N: usize>(
    engine_idx: usize,
    ends: [Ends; N],
    gate: Option<Gate>,
) -> (Lane, Arc<Seen>, RolesHandle) {
    let roles = roles(engine_idx);
    let seen = Arc::new(Seen::default());
    let engine = Builder {
        ends: ends.into(),
        seen: seen.clone(),
        gate,
    };
    let lane = Lane::spawn(
        engine_idx,
        Some(Box::new(engine)),
        Arc::new(SchedulerStats::default()),
    );
    (lane, seen, roles)
}

fn within<T>(what: &str, wait: impl Future<Output = T>) -> T {
    crate::rt::block_on(async { crate::rt::time::timeout(Duration::from_secs(10), wait).await })
        .unwrap_or_else(|_| panic!("timed out: {what}"))
}

/// Queues a registration and waits for the lane to be done with the item:
/// the lane is back at its queue, whatever the build is doing.
fn register(lane: &Lane) -> Answer {
    let (response, answer) = tokio::sync::oneshot::channel();
    let done = lane.control(QueuedItem::RegisterProgram {
        program: Box::new(ProgramRegistration::default()),
        response,
    });
    let commit = within("the lane takes the registration", done).expect("the lane answers");
    assert!(matches!(commit, LaneCommit::None));
    answer
}

fn answered(answer: Answer) -> anyhow::Result<ProgramId> {
    within("the registration is answered", answer).expect("answered, not dropped")
}

fn pending(answer: &mut Answer) -> bool {
    matches!(
        answer.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    )
}

fn fire(lane: &Lane) -> LaunchReplyRx {
    lane.fire(
        crate::engine::FrameFire {
            steps: vec![crate::engine::StepFire {
                submission: ::engine::Step {
                    lanes: vec![::engine::Lane::default()],
                    attachments: Vec::new(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                },
                terminal_cells: Vec::new(),
                completions: Vec::new(),
                instances: Vec::new(),
            }],
        },
        false,
    )
}

fn fires(lane: &Lane) {
    within("a fire is answered", fire(lane))
        .expect("a fire is answered")
        .expect("the lane fires");
}

fn knows(roles: &RolesHandle) -> bool {
    roles.lock().unwrap().knows(PROGRAM)
}

/// The `built`s the engine holds, taken from it.
fn held(seen: &Seen) -> Vec<ProgramBuilt> {
    std::mem::take(&mut *seen.held.lock().unwrap())
}

/// `built` told inside the call, before the lane has the id: the process
/// gets the id, and not before the program's facts were asked.
#[test]
fn a_build_told_inside_the_call_answers_once_the_program_is_learnt() {
    let (gate, asked) = crate::rt::channel::unbounded();
    let (mut lane, seen, roles) = spawn(9301, [Ends::Inside(Ok(()))], Some(gate));
    let mut answer = register(&lane);
    let go = asked
        .recv_timeout(Duration::from_secs(10))
        .expect("the program's facts are asked");
    assert!(
        pending(&mut answer),
        "no answer before the facts are learnt"
    );
    assert!(!knows(&roles));
    go.send(()).unwrap();
    assert_eq!(answered(answer).expect("registered"), PROGRAM);
    assert!(knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 1);
    lane.shutdown();
}

/// `built` held by the engine and told from another thread: the process
/// waits until then, the lane fires meanwhile, and nothing in the runtime
/// has the program until the tell.
#[test]
fn a_build_told_later_answers_then_and_the_lane_fires_meanwhile() {
    let (mut lane, seen, roles) = spawn(9302, [Ends::Held], None);
    let mut answer = register(&lane);
    fires(&lane);
    fires(&lane);
    assert_eq!(seen.fires.load(Ordering::SeqCst), 2);
    assert!(pending(&mut answer), "the build has not ended");
    assert!(!knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 0);
    let built = held(&seen).pop().expect("the engine holds it");
    std::thread::spawn(move || built(Ok(())))
        .join()
        .expect("told from another thread");
    assert_eq!(answered(answer).expect("registered"), PROGRAM);
    assert!(knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 1);
    fires(&lane);
    lane.shutdown();
}

/// `built` told `Err`: the process gets that error, and the runtime keeps
/// nothing of the program.
#[test]
fn a_build_told_err_answers_it_and_keeps_nothing() {
    let failed = Err(::engine::Error::Program(REFUSAL.to_string()));
    let (mut lane, seen, roles) = spawn(9303, [Ends::Inside(failed)], None);
    let error = answered(register(&lane)).expect_err("not built");
    assert!(format!("{error:#}").contains(REFUSAL), "{error:#}");
    fires(&lane);
    assert!(!knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 0);
    lane.shutdown();
}

/// The call returns `Err`, its `built` dropped untold: the process gets
/// the call's error, the drop means nothing, and the runtime keeps nothing
/// of the program.
#[test]
fn a_refused_registration_answers_it_and_keeps_nothing() {
    let (mut lane, seen, roles) = spawn(9304, [Ends::Refused], None);
    let error = answered(register(&lane)).expect_err("refused");
    assert!(format!("{error:#}").contains(REFUSAL), "{error:#}");
    assert!(!format!("{error:#}").contains(BUILD_UNTOLD), "{error:#}");
    // What the dropped `built` posted is passed over.
    fires(&lane);
    assert!(!knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 0);
    lane.shutdown();
}

/// `built` dropped untold by an engine that returned an id: the process is
/// answered by name instead of waiting for ever.
#[test]
fn a_build_dropped_untold_answers_by_name() {
    let (mut lane, seen, roles) = spawn(9305, [Ends::Dropped], None);
    let error = answered(register(&lane)).expect_err("never built");
    assert!(format!("{error:#}").contains(BUILD_UNTOLD), "{error:#}");
    assert!(!knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 0);
    lane.shutdown();
}

/// The lane ends with a build pending: its process is answered by name,
/// and a tell that comes after has nobody to reach.
#[test]
fn a_lane_that_ends_with_a_build_pending_answers_by_name() {
    let (mut lane, seen, roles) = spawn(9306, [Ends::Held], None);
    let mut answer = register(&lane);
    assert!(pending(&mut answer));
    let (engine, _channels) = lane.shutdown();
    let error = answered(answer).expect_err("the lane ended");
    assert!(format!("{error:#}").contains(BUILD_ABANDONED), "{error:#}");
    let built = held(&seen).pop().expect("the engine holds it");
    built(Ok(()));
    drop(engine);
    assert!(!knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 0);
}

/// A registration met while its program's one build still runs is given
/// the same id and its own `built`: each process waits for that build, and
/// both are answered when it ends, the program's facts asked once.
#[test]
fn registrations_of_a_program_still_building_are_answered_by_its_one_build() {
    let (mut lane, seen, roles) = spawn(9307, [Ends::Held, Ends::Held], None);
    let mut first = register(&lane);
    let mut second = register(&lane);
    fires(&lane);
    assert!(pending(&mut first), "the build has not ended");
    assert!(pending(&mut second), "the build has not ended");
    assert!(!knows(&roles));
    let built = held(&seen);
    assert_eq!(built.len(), 2);
    std::thread::spawn(move || built.into_iter().for_each(|built| built(Ok(()))))
        .join()
        .expect("the one build ends");
    assert_eq!(answered(first).expect("registered"), PROGRAM);
    assert_eq!(answered(second).expect("registered"), PROGRAM);
    assert!(knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 1);
    lane.shutdown();
}

/// A build that failed leaves nothing in the runtime, so the same program
/// registers again and is answered by its new build.
#[test]
fn a_program_whose_build_failed_registers_again() {
    let failed = Err(::engine::Error::Program(REFUSAL.to_string()));
    let ends = [Ends::Inside(failed), Ends::Refused, Ends::Inside(Ok(()))];
    let (mut lane, seen, roles) = spawn(9308, ends, None);
    let error = answered(register(&lane)).expect_err("not built");
    assert!(format!("{error:#}").contains(REFUSAL), "{error:#}");
    let error = answered(register(&lane)).expect_err("refused");
    assert!(format!("{error:#}").contains(REFUSAL), "{error:#}");
    assert!(!knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 0);
    assert_eq!(answered(register(&lane)).expect("registered"), PROGRAM);
    assert!(knows(&roles));
    assert_eq!(seen.facts_asked.load(Ordering::SeqCst), 1);
    lane.shutdown();
}
