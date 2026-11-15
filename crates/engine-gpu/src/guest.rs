//! Guest passes on the device: each stage of a launch package emitted as
//! WGSL, lowered to what the backend's pipelines take, and the forms a
//! program admits. The session that dispatches a stage over the API's
//! bindings stays in each backend crate; what it shares is here.

use engine::ProgramBuilt;
use engine::live::{Live, Unbuilt};
use engine::program::{ProgramId, ProgramRegistration};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
use eta_compiler::codegen::wgsl;
use eta_exec::ExecPlan;

use crate::program::Refusal;

pub mod held;
pub mod run;
pub mod session;
pub mod widen;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    Ops {
        stage: usize,

        missing: Vec<(u8, &'static str)>,
    },

    Emitting {
        stage: usize,

        why: String,
    },

    Lowering {
        stage: usize,

        why: String,
    },
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ops { stage, missing } => {
                let named = missing
                    .iter()
                    .map(|(tag, name)| format!("`{name}` ({tag:#04x})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "stage {stage} runs {named}, which this backend emits no WGSL for, and a \
                     guest pass runs on the device whole or not at all"
                )
            }
            Self::Emitting { stage, why } => {
                write!(f, "stage {stage} has no emitted WGSL: {why}")
            }
            Self::Lowering { stage, why } => {
                write!(f, "stage {stage}'s emitted WGSL did not lower: {why}")
            }
        }
    }
}

pub const ENTRY: &str = "main";

/// How a backend lowers a stage's emitted WGSL to what its pipelines take.
pub trait Lower {
    type Code;

    fn lower(stage: usize, source: String) -> Result<Self::Code, Refused>;
}

fn every_op_emits(stage: usize, plan: &LaunchStagePlan) -> Result<(), Refused> {
    let mut missing: Vec<(u8, &'static str)> = Vec::new();
    for op in &plan.ops {
        if wgsl::is_boundary(op.tag) || wgsl::emits(op.tag) {
            continue;
        }
        if !missing.iter().any(|&(tag, _)| tag == op.tag) {
            missing.push((op.tag, eta_ir::op::spec(op.tag).map_or("?", |row| row.name)));
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Refused::Ops { stage, missing })
    }
}

pub fn step_source(stage: usize, plan: &LaunchStagePlan) -> Result<wgsl::Stepwise, Refused> {
    every_op_emits(stage, plan)?;
    wgsl::emit_launch_steps(ENTRY, plan).map_err(|why| Refused::Emitting {
        stage,
        why: why.to_string(),
    })
}

pub struct Compiled<L: Lower> {
    stages: Vec<Lowered<L::Code>>,
}

pub struct Lowered<C> {
    pub code: C,

    pub entries: Vec<String>,

    pub strides_the_grid: bool,
}

impl<L: Lower> std::fmt::Debug for Compiled<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiled")
            .field("stages", &self.stages.len())
            .field(
                "dispatches",
                &self.stages.iter().map(|s| s.entries.len()).sum::<usize>(),
            )
            .finish()
    }
}

impl<L: Lower> Compiled<L> {
    pub fn of(package: &LaunchPackage) -> Result<Compiled<L>, Refused> {
        let mut stages = Vec::with_capacity(package.plans.len());
        for (at, plan) in package.plans.iter().enumerate() {
            let stepwise = step_source(at, plan)?;
            stages.push(Lowered {
                code: L::lower(at, stepwise.source)?,
                entries: stepwise.steps.into_iter().map(|step| step.entry).collect(),
                strides_the_grid: true,
            });
        }
        Ok(Compiled { stages })
    }

    #[must_use]
    pub fn stage(&self, at: usize) -> Option<&Lowered<L::Code>> {
        self.stages.get(at)
    }

    #[must_use]
    pub fn all(&self) -> &[Lowered<L::Code>] {
        &self.stages
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

/// A registered program: its plan, its stages lowered, and what its fires
/// dispatch through, which is there once its build has ended well.
pub struct Program<L: Lower, T> {
    pub hash: u64,
    pub plan: Arc<ExecPlan>,
    pub compiled: Arc<Compiled<L>>,
    build: Arc<Build<T>>,
}

/// What a program's entry and its build share: what was built, written
/// once and read by a bind with no lock, and the registrations to tell
/// when the build ends.
struct Build<T> {
    built: OnceLock<Arc<T>>,
    told: Mutex<Told>,
}

enum Told {
    /// The build runs: the registrations it tells when it ends.
    Building(Vec<ProgramBuilt>),
    /// The build ended, with this; a failed one keeps nothing.
    Ended(Result<(), String>),
}

impl<T> Build<T> {
    fn told(&self) -> std::sync::MutexGuard<'_, Told> {
        self.told.lock().expect("a build's registrations poisoned")
    }

    /// The build's end: what it built is the program's from here on, or it
    /// has none, and every registration waiting on it is told, once. A
    /// build that has ended ends no further.
    fn end(&self, built: Result<T, String>) {
        let mut told = self.told();
        let Told::Building(waiting) = &mut *told else {
            return;
        };
        let waiting = std::mem::take(waiting);
        let result = built.map(|built| {
            if self.built.set(Arc::new(built)).is_err() {
                unreachable!("a program's build ends once");
            }
        });
        *told = Told::Ended(result.clone());
        drop(told);
        for tell in waiting {
            tell(result.clone().map_err(engine::Error::Program));
        }
    }

    /// One more registration of the program: told when the build ends, or
    /// at once with how it ended.
    fn wait(&self, built: ProgramBuilt) {
        let ended = match &mut *self.told() {
            Told::Building(waiting) => {
                waiting.push(built);
                return;
            }
            Told::Ended(result) => result.clone(),
        };
        built(ended.map_err(engine::Error::Program));
    }

    fn failed(&self) -> bool {
        matches!(*self.told(), Told::Ended(Err(_)))
    }
}

/// A program's build, as a value: the lowered stages to build from and
/// where what is built goes. It holds nothing of the plane, so it runs
/// and ends on any thread; however it ends, the registrations waiting on
/// it are told exactly once.
pub struct Job<L: Lower, T> {
    hash: u64,
    compiled: Arc<Compiled<L>>,
    build: Arc<Build<T>>,
}

impl<L: Lower, T> Job<L, T> {
    /// Builds what the program's stages dispatch through and ends the
    /// build with it: the one place a program becomes one a bind finds
    /// built.
    pub fn run(self, build: impl FnOnce(&Compiled<L>) -> Result<T, String>) {
        let built = build(&self.compiled)
            .map_err(|why| format!("program 0x{:016x} does not build: {why}", self.hash));
        self.build.end(built);
    }

    /// The build never started, and the registration that would have
    /// started it is refused by its own call: nobody is told.
    pub fn abandon(self) {
        *self.build.told() = Told::Ended(Err(format!(
            "program 0x{:016x} was not built: its build never started",
            self.hash
        )));
    }
}

impl<L: Lower, T> Drop for Job<L, T> {
    /// A build dropped before it ended (its thread unwound) still tells.
    fn drop(&mut self) {
        self.build.end(Err(format!(
            "program 0x{:016x} was not built: its build ended without a result",
            self.hash
        )));
    }
}

/// What a registration found.
pub enum Adopted<L: Lower, T> {
    /// A program of this hash is registered: nothing to build. The
    /// registration is told with that program's build.
    Known(ProgramId),
    /// A program entered, lowered, with nothing built: the caller hands
    /// `job` to what builds, or abandons it and forgets `id`.
    New { id: ProgramId, job: Job<L, T> },
}

/// What a bind takes of a built program.
pub struct Ready<L: Lower, T> {
    pub plan: Arc<ExecPlan>,
    pub compiled: Arc<Compiled<L>>,
    pub built: Arc<T>,
}

/// The programs a plane has registered, by id and by hash, each with what
/// was built for it at its registration. A bind takes a program only built;
/// nothing here builds.
pub struct Registry<L: Lower, T> {
    programs: BTreeMap<ProgramId, Program<L, T>>,
    hashes: HashMap<u64, ProgramId>,
    next: u64,
    /// Builds started.
    builds: u64,
    /// Where a bind or a fire that met something unbuilt is counted.
    live: Live,
}

impl<L: Lower, T> Default for Registry<L, T> {
    fn default() -> Registry<L, T> {
        Registry {
            programs: BTreeMap::new(),
            hashes: HashMap::new(),
            next: 0,
            builds: 0,
            live: Live::armed(),
        }
    }
}

impl<L: Lower, T> Registry<L, T> {
    /// Counts into the engine's record from here on.
    pub fn counts_into(&mut self, live: &Live) {
        self.live = live.clone();
    }

    /// Adopts and lowers `registration`, on the caller's thread. A program
    /// that does not interpret, or whose stages do not lower, is refused
    /// and never entered, and so is a registered hash under another launch
    /// package: `built` is dropped untold. A hash already registered names
    /// the program it was registered with and starts no build: `built` is
    /// told when that program's build ends, at once where it has. A
    /// program whose build failed is forgotten first, so its hash registers
    /// anew.
    pub fn adopt(
        &mut self,
        registration: &ProgramRegistration,
        built: ProgramBuilt,
    ) -> Result<Adopted<L, T>, Refusal> {
        let failed: Vec<ProgramId> = (self.programs.iter())
            .filter(|(_, program)| program.build.failed())
            .map(|(&id, _)| id)
            .collect();
        for id in failed {
            self.forget(id);
        }
        let hash = registration.program_hash;
        if let Some(&id) = self.hashes.get(&hash) {
            let known = &self.programs[&id];
            if known.plan.package != registration.launch {
                return Err(Refusal::Program(format!(
                    "program 0x{hash:016x} is registered ({id}) with another launch package"
                )));
            }
            known.build.wait(built);
            return Ok(Adopted::Known(id));
        }
        let plan = eta_exec::adopt_launch_package(registration.launch.clone())
            .map_err(|error| Refusal::Program(error.to_string()))?;
        if !plan.executable {
            return Err(Refusal::Program(format!(
                "program 0x{hash:016x} does not interpret: {}",
                plan.reject_reason.as_deref().unwrap_or("no reason given")
            )));
        }
        let compiled = Arc::new(Compiled::of(&plan.package).map_err(|why| {
            Refusal::Program(format!(
                "program 0x{hash:016x} does not lower to a shader: {why}"
            ))
        })?);
        self.next += 1;
        let id = ProgramId(self.next);
        let build = Arc::new(Build {
            built: OnceLock::new(),
            told: Mutex::new(Told::Building(vec![built])),
        });
        self.programs.insert(
            id,
            Program {
                hash,
                plan: Arc::new(plan),
                compiled: Arc::clone(&compiled),
                build: Arc::clone(&build),
            },
        );
        self.hashes.insert(hash, id);
        self.builds += 1;
        Ok(Adopted::New {
            id,
            job: Job {
                hash,
                compiled,
                build,
            },
        })
    }

    /// Drops a program that has no build, so its hash registers anew.
    pub fn forget(&mut self, id: ProgramId) {
        if let Some(program) = self.programs.remove(&id) {
            self.hashes.remove(&program.hash);
        }
    }

    pub fn get(&self, id: ProgramId) -> Result<&Program<L, T>, Refusal> {
        self.programs.get(&id).ok_or(Refusal::Closed {
            what: "program",
            id: id.0,
        })
    }

    #[must_use]
    pub fn package(&self, id: ProgramId) -> Option<&LaunchPackage> {
        self.programs.get(&id).map(|program| &program.plan.package)
    }

    /// What was built for `id`, where its build has ended well.
    #[must_use]
    pub fn built(&self, id: ProgramId) -> Option<&Arc<T>> {
        self.programs.get(&id)?.build.built.get()
    }

    /// The program a bind takes. One with nothing built (its build still
    /// runs, or failed) is refused by name and counted: a bind builds
    /// nothing and waits on no build.
    pub fn ready(&self, id: ProgramId) -> Result<Ready<L, T>, Refusal> {
        let program = self.get(id)?;
        match program.build.built.get() {
            Some(built) => Ok(Ready {
                plan: Arc::clone(&program.plan),
                compiled: Arc::clone(&program.compiled),
                built: Arc::clone(built),
            }),
            None => Err(self.refuse_unbuilt(format!(
                "program {id} (0x{:016x}) is not built: its registration has not told that \
                 what its stages dispatch through is built, and a bind builds nothing",
                program.hash
            ))),
        }
    }

    /// The refusal of a bind or a fire that met something of a program
    /// unbuilt, counted ([`engine::live::Tally::Unbuilt`]).
    pub fn refuse_unbuilt(&self, what: String) -> Refusal {
        self.live.unbuilt(Unbuilt::Program);
        Refusal::Unbuilt(what)
    }

    /// Builds started.
    #[must_use]
    pub fn builds(&self) -> u64 {
        self.builds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::live::Tally;
    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{EngineCaps, Stage};
    use eta_ir::types::{Dtype, Shape};
    use std::sync::mpsc::{Receiver, channel};

    /// Lowers a stage to the length of its source: no device.
    struct Text;

    impl Lower for Text {
        type Code = usize;

        fn lower(_stage: usize, source: String) -> Result<usize, Refused> {
            Ok(source.len())
        }
    }

    struct Refuses;

    impl Lower for Refuses {
        type Code = ();

        fn lower(stage: usize, _source: String) -> Result<(), Refused> {
            Err(Refused::Lowering {
                stage,
                why: "this backend lowers nothing".to_owned(),
            })
        }
    }

    type Programs = Registry<Text, Vec<usize>>;

    /// A program of one stage: it takes its own seeded cell of `wide`
    /// ids, puts it doubled where the host reads, and keeps the doubled.
    fn registration(hash: u64, wide: u32) -> ProgramRegistration {
        let decl = |host_role: HostRole, capacity: u32, seeded: bool| ChannelDecl {
            shape: Shape::matrix(1, wide),
            dtype: ChanDType::Concrete(Dtype::I32),
            capacity,
            host_role,
            seeded,
        };
        let container = TraceContainer {
            names: Vec::new(),
            channels: vec![
                decl(HostRole::None, 1, true),
                decl(HostRole::Reader, 2, false),
            ],
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Add(0, 0),
                    Op::ChanPut { chan: 1, value: 1 },
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
            externs: Vec::new(),
            inputs: Vec::new(),
        };
        let bound = eta_ir::validate::bind(container, vec![Vec::new()], EngineCaps::default())
            .expect("the program binds");
        let stages = eta_compiler::plan::compile_bound(&bound);
        ProgramRegistration {
            program_hash: hash,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        }
    }

    /// What a backend's build makes of a lowered program, here the steps
    /// each stage dispatches.
    fn steps(compiled: &Compiled<Text>) -> Result<Vec<usize>, String> {
        Ok((compiled.all().iter())
            .map(|stage| stage.entries.len())
            .collect())
    }

    /// A registration's `built`, and where what it is told arrives: an
    /// `Err` to receive once it is dropped untold.
    fn told() -> (ProgramBuilt, Receiver<Result<(), String>>) {
        let (tell, told) = channel();
        let built: ProgramBuilt = Box::new(move |built| {
            // A test that reads nothing of it has dropped the other end.
            let _ = tell.send(built.map_err(|why| why.to_string()));
        });
        (built, told)
    }

    fn entered(registry: &mut Programs, hash: u64) -> (ProgramId, Job<Text, Vec<usize>>) {
        let (built, _) = told();
        match registry
            .adopt(&registration(hash, 4), built)
            .expect("the program registers")
        {
            Adopted::New { id, job } => (id, job),
            Adopted::Known(id) => panic!("program {id} was not registered before"),
        }
    }

    #[test]
    fn a_registration_is_told_once_every_stage_is_built_before_any_bind() {
        let mut registry = Programs::default();
        let registration = registration(0xb17, 4);
        let (built, told) = told();
        let Adopted::New { id, job } = registry
            .adopt(&registration, built)
            .expect("the program registers")
        else {
            panic!("a first registration enters the program");
        };
        assert!(told.try_recv().is_err(), "nothing is told while it builds");
        assert!(registry.built(id).is_none());
        // The build ends on whichever thread ran it.
        std::thread::spawn(move || job.run(steps))
            .join()
            .expect("the build ends");
        assert_eq!(told.recv(), Ok(Ok(())), "told once, built");
        assert!(told.recv().is_err(), "and told no more");
        let built = registry.built(id).expect("the registration built it");
        assert_eq!(built.len(), registration.launch.plans.len());
        assert!(
            built.iter().all(|&steps| steps > 0),
            "every stage dispatches through something built: {built:?}"
        );
        let ready = registry.ready(id).expect("a bind takes it built");
        assert_eq!(*ready.built, **built);
        assert_eq!(registry.live.tallied(Tally::Unbuilt), 0);
    }

    #[test]
    fn a_second_registration_of_a_program_builds_nothing() {
        let mut registry = Programs::default();
        let (first, job) = entered(&mut registry, 0xb17);
        assert_eq!(registry.builds(), 1);
        // While the one build runs: the same id, told when it ends.
        let (built, waiting) = told();
        match registry
            .adopt(&registration(0xb17, 4), built)
            .expect("the program registers again")
        {
            Adopted::Known(id) => assert_eq!(id, first),
            Adopted::New { .. } => panic!("a hash whose build runs starts no second build"),
        }
        assert!(waiting.try_recv().is_err(), "told when the build ends");
        job.run(steps);
        assert_eq!(waiting.recv(), Ok(Ok(())));
        // Built: the same id, told at once.
        let (built, at_once) = told();
        assert!(matches!(
            registry.adopt(&registration(0xb17, 4), built),
            Ok(Adopted::Known(id)) if id == first
        ));
        assert_eq!(at_once.try_recv(), Ok(Ok(())));
        assert_eq!(registry.builds(), 1);
        // The hash names the program it was registered with and no other:
        // refused by the call, and told nothing.
        let (built, untold) = told();
        let Err(why) = registry.adopt(&registration(0xb17, 8), built) else {
            panic!("another program under a registered hash is refused");
        };
        assert!(why.to_string().contains("another launch package"), "{why}");
        assert!(untold.recv().is_err(), "dropped untold");
        assert_eq!(registry.builds(), 1);
        // Another hash is another program, with its own build.
        assert_ne!(entered(&mut registry, 0xb18).0, first);
        assert_eq!(registry.builds(), 2);
    }

    #[test]
    fn a_bind_that_meets_an_unbuilt_program_is_refused_by_name_and_counted() {
        let mut registry = Programs::default();
        let live = Live::armed();
        registry.counts_into(&live);
        let (id, job) = entered(&mut registry, 0xb17);
        let Err(refusal) = registry.ready(id) else {
            panic!("a bind takes no program whose build has not ended");
        };
        assert!(matches!(refusal, Refusal::Unbuilt(_)), "{refusal}");
        assert!(refusal.to_string().contains("is not built"), "{refusal}");
        assert_eq!(live.tallied(Tally::Unbuilt), 1);
        assert!(live.unbuilt_at(Unbuilt::Program));
        assert!(!live.unbuilt_at(Unbuilt::Kernel));
        job.run(steps);
        assert!(registry.ready(id).is_ok());
        assert_eq!(live.tallied(Tally::Unbuilt), 1);
        assert_eq!(registry.builds(), 1);
    }

    #[test]
    fn a_failed_build_tells_every_registration_and_keeps_nothing() {
        let mut registry = Programs::default();
        let (built, first) = told();
        let Adopted::New { id, job } = registry
            .adopt(&registration(0xb17, 4), built)
            .expect("the program registers")
        else {
            panic!("a first registration enters the program");
        };
        let (built, second) = told();
        assert!(matches!(
            registry.adopt(&registration(0xb17, 4), built),
            Ok(Adopted::Known(known)) if known == id
        ));
        job.run(|_| Err("the device refused a pipeline".to_owned()));
        for told in [first, second] {
            let why = told.recv().expect("told").expect_err("told that it failed");
            assert!(why.contains("does not build"), "{why}");
            assert!(why.contains("the device refused a pipeline"), "{why}");
            assert!(told.recv().is_err(), "told once");
        }
        // Until the engine's next registration its id names nothing built.
        assert!(matches!(registry.ready(id), Err(Refusal::Unbuilt(_))));
        // Its hash registers anew, under another id, and the old one is gone.
        let (again, job) = entered(&mut registry, 0xb17);
        assert_ne!(again, id);
        assert!(matches!(
            registry.ready(id),
            Err(Refusal::Closed {
                what: "program",
                ..
            })
        ));
        job.run(steps);
        assert!(registry.ready(again).is_ok());
        assert_eq!(registry.builds(), 2);
    }

    #[test]
    fn a_build_that_ends_without_a_result_still_tells() {
        let mut registry = Programs::default();
        let (built, told) = told();
        let Adopted::New { job, .. } = registry
            .adopt(&registration(0xb17, 4), built)
            .expect("the program registers")
        else {
            panic!("a first registration enters the program");
        };
        let unwound = std::thread::spawn(move || {
            job.run(|_| panic!("the build's thread unwinds"));
        })
        .join();
        assert!(unwound.is_err());
        let why = told.recv().expect("told").expect_err("told it has none");
        assert!(why.contains("ended without a result"), "{why}");
        assert!(told.recv().is_err(), "told once");
    }

    #[test]
    fn a_build_that_never_started_tells_nobody() {
        let mut registry = Programs::default();
        let (built, told) = told();
        let Adopted::New { id, job } = registry
            .adopt(&registration(0xb17, 4), built)
            .expect("the program registers")
        else {
            panic!("a first registration enters the program");
        };
        // What the caller does where it has nothing to build on: its own
        // call refuses.
        job.abandon();
        registry.forget(id);
        assert!(told.recv().is_err(), "dropped untold");
        assert!(matches!(registry.ready(id), Err(Refusal::Closed { .. })));
        assert_eq!(registry.live.tallied(Tally::Unbuilt), 0);
    }

    #[test]
    fn a_program_that_does_not_lower_is_refused_and_not_registered() {
        let mut registry: Registry<Refuses, ()> = Registry::default();
        let (built, told) = told();
        let Err(why) = registry.adopt(&registration(0xb17, 4), built) else {
            panic!("a program with no shader form does not register");
        };
        assert!(why.to_string().contains("does not lower"), "{why}");
        assert!(told.recv().is_err(), "dropped untold");
        assert_eq!(registry.builds(), 0);
        assert!(registry.package(ProgramId(1)).is_none());
    }
}
