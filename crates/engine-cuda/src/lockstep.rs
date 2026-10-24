//! Expert-parallel ranks in one process fire in lockstep: dispatch and
//! combine meet every rank, so every rank fires every step, a rank with no
//! frame of its own an idle step at the step's bucket. Each rank keeps its
//! own scheduler and submits on its own; [`Rendezvous`] turns those
//! submissions into one sequence of steps.
//!
//! A step forms when its first frame arrives. It waits for each rank that
//! has announced a next frame (`Engine::expect_fire`) and has not yet
//! brought it, up to a deadline; a rank that announced nothing is idle and
//! is not waited for. Without the wait two busy ranks a moment apart would
//! split every step in two, each idling through the other's. The first
//! arrival leads: it is handed every arrived frame and the step's bucket
//! (the largest asked, which Dispatch's token count must agree on), fires
//! the step on every rank, and posts each joined rank its result. Each rank
//! fires on a thread of its own, held for the group's life ([`Firing`]): a
//! rank's fire may wait on its device, which waits on every other rank's
//! launches.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::api::{ContractFor, Cuda, DeviceBoot, PendingStep};
use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::DeviceFacts;
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireTicket, FrameSubmission, FrameTicket, Step};
use engine::load::{LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, KvHandle, StateCopy};

/// What a rank brings to a step: its frame and the token bucket it asks.
pub struct Arrival<F> {
    pub frame: F,
    pub bucket: u32,
}

/// The step a leader fires: every rank's frame (`None`: the rank idles) and
/// the bucket every rank fires at.
pub struct Formed<F> {
    pub step: u64,
    pub bucket: u32,
    pub frames: Vec<Option<F>>,
}

/// A rank's part in a step it brought a frame to.
pub enum Part<F, R> {
    /// Fire `Formed` on every rank, then [`Rendezvous::finish`] it.
    Lead(Formed<F>),
    /// The leader fired it; this is this rank's result.
    Joined(R),
}

struct State<F, R> {
    /// The step arrivals join; it forms on its first.
    open: u64,
    /// A step whose leader has not closed it yet: later arrivals open the next.
    forming: bool,
    /// Steps fired and finished: a step closes only once the one before it
    /// has, so two steps never fire at once.
    fired: u64,
    announced: Vec<bool>,
    arrived: Vec<Option<Arrival<F>>>,
    /// Results of fired steps, by rank, until the rank takes its own.
    results: Vec<Option<(u64, R)>>,
    poisoned: Option<String>,
}

pub struct Rendezvous<F, R> {
    state: Mutex<State<F, R>>,
    changed: Condvar,
    wait: Duration,
}

impl<F, R> Rendezvous<F, R> {
    #[must_use]
    pub fn new(ranks: usize, wait: Duration) -> Self {
        Self {
            state: Mutex::new(State {
                open: 0,
                forming: false,
                fired: 0,
                announced: vec![false; ranks],
                arrived: (0..ranks).map(|_| None).collect(),
                results: (0..ranks).map(|_| None).collect(),
                poisoned: None,
            }),
            changed: Condvar::new(),
            wait,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State<F, R>> {
        self.state.lock().expect("lockstep state lock poisoned")
    }

    /// `rank` will bring a frame soon: a forming step waits for it.
    pub fn announce(&self, rank: usize) {
        self.lock().announced[rank] = true;
        self.changed.notify_all();
    }

    /// Brings `rank`'s frame to the open step. The first rank to arrive
    /// leads it; the others wait for the leader's result.
    pub fn arrive(&self, rank: usize, arrival: Arrival<F>) -> Result<Part<F, R>, String> {
        let mut s = self.lock();
        // A rank leads or waits for its result before it submits again.
        debug_assert!(
            s.arrived[rank].is_none(),
            "rank {rank} arrived twice in a step"
        );
        if let Some(why) = &s.poisoned {
            return Err(why.clone());
        }
        s.announced[rank] = false;
        s.arrived[rank] = Some(arrival);
        let step = s.open;
        self.changed.notify_all();
        if s.forming {
            return self.joined(s, rank, step);
        }
        s.forming = true;
        let deadline = Instant::now() + self.wait;
        loop {
            if let Some(why) = &s.poisoned {
                return Err(why.clone());
            }
            let waiting = s
                .announced
                .iter()
                .zip(&s.arrived)
                .any(|(&announced, arrived)| announced && arrived.is_none());
            let now = Instant::now();
            let previous = s.fired == step;
            if previous && (!waiting || now >= deadline) {
                break;
            }
            // The step before still firing: keep gathering until it is done.
            let until = if previous {
                deadline - now
            } else {
                Duration::from_millis(50)
            };
            s = self
                .changed
                .wait_timeout(s, until)
                .expect("lockstep state lock poisoned")
                .0;
        }
        let frames: Vec<Option<Arrival<F>>> = s.arrived.iter_mut().map(Option::take).collect();
        s.open += 1;
        s.forming = false;
        self.changed.notify_all();
        let bucket = frames.iter().flatten().map(|a| a.bucket).max().unwrap_or(0);
        Ok(Part::Lead(Formed {
            step,
            bucket,
            frames: frames.into_iter().map(|a| a.map(|a| a.frame)).collect(),
        }))
    }

    fn joined(
        &self,
        mut s: MutexGuard<'_, State<F, R>>,
        rank: usize,
        step: u64,
    ) -> Result<Part<F, R>, String> {
        loop {
            if let Some(why) = &s.poisoned {
                return Err(why.clone());
            }
            if s.results[rank].as_ref().is_some_and(|(at, _)| *at == step) {
                let (_, result) = s.results[rank].take().expect("checked above");
                return Ok(Part::Joined(result));
            }
            s = self.changed.wait(s).expect("lockstep state lock poisoned");
        }
    }

    /// The leader of `step` posts each joined rank's result (`None` for the
    /// leader's own and for idle ranks).
    pub fn finish(&self, step: u64, results: Vec<Option<R>>) {
        let mut s = self.lock();
        s.fired = step + 1;
        for (slot, result) in s.results.iter_mut().zip(results) {
            if let Some(result) = result {
                *slot = Some((step, result));
            }
        }
        self.changed.notify_all();
    }

    /// The group stops: every waiting and later arrival answers `why`.
    pub fn poison(&self, why: &str) {
        let mut s = self.lock();
        if s.poisoned.is_none() {
            s.poisoned = Some(why.to_owned());
        }
        self.changed.notify_all();
    }
}

/// How long a forming step waits for an announced rank's frame.
const GATHER: Duration = Duration::from_millis(2);

type Fired = EngineResult<(FireTicket, PendingStep)>;

/// What every rank of one expert-parallel group shares.
struct Group {
    ranks: Vec<Arc<Mutex<Cuda>>>,
    ordinals: Vec<i32>,
    steps: Rendezvous<(Step, engine::StepDone), Fired>,
    /// Each rank's shape as it reaches arming, and the group's wires once
    /// the last has: every rank's load meets the others there.
    wiring: Mutex<Wiring>,
    /// Ranks loaded, and whether every one has; a rank's first frame waits
    /// for the group.
    wired: Mutex<(usize, Option<Result<(), String>>)>,
    wired_cv: Condvar,
    firing: Firing,
    rebalancer: Option<Mutex<crate::ep::rebalance::Rebalancer>>,
}

/// What a rank's thread fires: its frame's step, or an idle step.
enum Job {
    Step(Step, engine::StepDone, u32),
    Idle(u32),
}

/// One thread a rank, bound to its device once, firing what the step's
/// leader hands it. Only one step fires at a time, so the leader alone
/// reads the answers.
struct Firing {
    jobs: Vec<mpsc::Sender<Job>>,
    answers: Mutex<mpsc::Receiver<(usize, Option<Fired>)>>,
    threads: Vec<JoinHandle<()>>,
}

impl Firing {
    fn start(ranks: &[Arc<Mutex<Cuda>>], ordinals: &[i32]) -> Result<Firing, String> {
        let (answer, answers) = mpsc::channel();
        let mut jobs = Vec::with_capacity(ranks.len());
        let mut threads = Vec::with_capacity(ranks.len());
        for (rank, (cuda, &ordinal)) in ranks.iter().zip(ordinals).enumerate() {
            let (job, inbox) = mpsc::channel();
            let (cuda, answer) = (Arc::clone(cuda), answer.clone());
            let thread = std::thread::Builder::new()
                .name(format!("ep-rank-{rank}"))
                .spawn(move || serve(rank, ordinal, &cuda, &inbox, &answer))
                .map_err(|why| format!("rank {rank}'s firing thread: {why}"))?;
            jobs.push(job);
            threads.push(thread);
        }
        Ok(Firing {
            jobs,
            answers: Mutex::new(answers),
            threads,
        })
    }
}

impl Drop for Firing {
    fn drop(&mut self) {
        self.jobs.clear();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// A rank's firing thread: each job it is handed, answered in turn, until
/// the group is gone.
fn serve(
    rank: usize,
    ordinal: i32,
    cuda: &Mutex<Cuda>,
    inbox: &mpsc::Receiver<Job>,
    answer: &mpsc::Sender<(usize, Option<Fired>)>,
) {
    let bound = crate::device::ctx::bind_thread(ordinal).map_err(|f| f.to_string());
    for job in inbox {
        // A panic answers as a refusal: the leader waits on every rank.
        let fired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bound.clone().map_err(Error::Device)?;
            let mut cuda = cuda.lock().expect("rank lock poisoned");
            cuda.bind_thread()?;
            match &job {
                Job::Step(step, at, bucket) => cuda.fire_step_at(step, *at, *bucket).map(Some),
                Job::Idle(bucket) => cuda.fire_idle(*bucket).map(|()| None),
            }
        }))
        .unwrap_or_else(|_| Err(Error::Device(format!("rank {rank}'s fire panicked"))));
        let fired = match fired {
            Ok(Some(done)) => Some(Ok(done)),
            Ok(None) => None,
            Err(error) => {
                eprintln!("engine-cuda: expert-parallel rank {rank} refused: {error}");
                Some(Err(error))
            }
        };
        if answer.send((rank, fired)).is_err() {
            return;
        }
    }
}

struct Wiring {
    shapes: Vec<Option<crate::ep::Shape>>,
    wires: Option<Result<Vec<crate::ep::Wire>, String>>,
}

/// How long a rank's load waits at arming for the group's other ranks: the
/// worker loads every rank of a group at once.
const MEET: Duration = Duration::from_secs(3600);

impl Group {
    fn poison(&self, why: &str) {
        self.steps.poison(why);
        let mut wired = self.wired.lock().expect("lockstep wiring lock poisoned");
        if wired.1.is_none() {
            wired.1 = Some(Err(why.to_owned()));
        }
        self.wired_cv.notify_all();
    }

    /// A rank's load at arming: it states what it dispatches and waits for
    /// the others; the last to come maps every rank's EP buffers into the
    /// others. Each rank then lays its layers' placements on its own device.
    fn meet(&self, rank: usize, shell: &mut crate::serve::Shell) -> Result<(), String> {
        let shape = shell
            .ep_shape()
            .ok_or("the loaded trace dispatches nothing: it is not an expert-parallel rank's")?;
        let mut wiring = self.wiring.lock().expect("lockstep wiring lock poisoned");
        wiring.shapes[rank] = Some(shape.clone());
        if wiring.wires.is_none() && wiring.shapes.iter().all(Option::is_some) {
            let same = wiring
                .shapes
                .iter()
                .all(|other| other.as_ref() == Some(&shape));
            wiring.wires = Some(if same {
                let wired =
                    crate::ep::wire(&self.ordinals, shape.widest).map_err(|f| f.to_string());
                crate::device::ctx::bind_thread(self.ordinals[rank]).map_err(|f| f.to_string())?;
                wired
            } else {
                Err("the group's ranks loaded traces that dispatch differently".into())
            });
            self.wired_cv.notify_all();
        }
        let deadline = Instant::now() + MEET;
        while wiring.wires.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let missing: Vec<usize> = (0..self.ranks.len())
                    .filter(|&r| wiring.shapes[r].is_none())
                    .collect();
                return Err(format!(
                    "rank(s) {missing:?} of the expert-parallel group never reached arming \
                     within {}s; every rank of a group loads at once",
                    MEET.as_secs()
                ));
            }
            wiring = self
                .wired_cv
                .wait_timeout(wiring, left)
                .expect("lockstep wiring lock poisoned")
                .0;
        }
        let wires = wiring.wires.clone().expect("checked above")?;
        drop(wiring);
        let ep =
            crate::ep::EpRank::new(wires, rank as u32, &shape.layers).map_err(|f| f.to_string())?;
        shell.join_ep(ep)
    }

    /// A rank has loaded; the group serves once every rank has.
    fn loaded(&self) {
        let mut wired = self.wired.lock().expect("lockstep wiring lock poisoned");
        wired.0 += 1;
        if wired.0 == self.ranks.len() && wired.1.is_none() {
            wired.1 = Some(Ok(()));
        }
        self.wired_cv.notify_all();
    }

    fn ready(&self) -> EngineResult<()> {
        let mut wired = self.wired.lock().expect("lockstep wiring lock poisoned");
        loop {
            match &wired.1 {
                Some(Ok(())) => return Ok(()),
                Some(Err(why)) => {
                    return Err(Error::Device(format!(
                        "this expert-parallel group is stopped: {why}"
                    )));
                }
                None => {
                    wired = self
                        .wired_cv
                        .wait(wired)
                        .expect("lockstep wiring lock poisoned");
                }
            }
        }
    }

    /// Fires `formed` on every rank at once, each on its own thread, the
    /// group rebalanced first when it is due.
    fn fire(&self, formed: Formed<(Step, engine::StepDone)>) -> Vec<Option<Fired>> {
        if let Some(why) = self.rebalance().err() {
            self.poison(&why);
            let refused = |_| Some(Err(Error::Device(why.clone())));
            return (0..self.ranks.len()).map(refused).collect();
        }
        let bucket = formed.bucket;
        let answers = self
            .firing
            .answers
            .lock()
            .expect("lockstep answers lock poisoned");
        let mut results: Vec<Option<Fired>> = (0..self.ranks.len()).map(|_| None).collect();
        let mut owed = vec![false; self.ranks.len()];
        for (rank, frame) in formed.frames.into_iter().enumerate() {
            let job = match frame {
                Some((step, at)) => Job::Step(step, at, bucket),
                None => Job::Idle(bucket),
            };
            owed[rank] = self.firing.jobs[rank].send(job).is_ok();
            if !owed[rank] {
                results[rank] = Some(Err(Error::Device(format!(
                    "rank {rank}'s firing thread is gone"
                ))));
            }
        }
        while owed.iter().any(|&o| o) {
            let Ok((rank, fired)) = answers.recv() else {
                for (rank, _) in owed.iter().enumerate().filter(|(_, o)| **o) {
                    results[rank] =
                        Some(Err(Error::Device(format!("rank {rank}'s fire panicked"))));
                }
                break;
            };
            owed[rank] = false;
            results[rank] = fired;
        }
        results
    }
}

impl Group {
    /// Rebalances the group's experts when it is due, before the step's
    /// launches: every rank's work finished first, the leader's thread
    /// running the copies on rank 0's device. A failure leaves the planes
    /// and the tables apart, so the group stops.
    fn rebalance(&self) -> Result<(), String> {
        let Some(rebalancer) = &self.rebalancer else {
            return Ok(());
        };
        let mut rebalancer = rebalancer.lock().expect("lockstep rebalance lock poisoned");
        if !rebalancer.due() {
            return Ok(());
        }
        let mut cudas: Vec<_> = self
            .ranks
            .iter()
            .map(|cuda| cuda.lock().expect("rank lock poisoned"))
            .collect();
        let mut views = Vec::with_capacity(cudas.len());
        for (cuda, &ordinal) in cudas.iter_mut().zip(&self.ordinals) {
            crate::device::ctx::bind_thread(ordinal).map_err(|f| f.to_string())?;
            let shell = cuda.shell_mut().ok_or("a rank has no model loaded")?;
            shell.quiesce().map_err(|f| f.to_string())?;
            views.push(
                shell
                    .rebalance_view()
                    .ok_or("a rank is in no expert-parallel group")?,
            );
        }
        crate::device::ctx::bind_thread(self.ordinals[0]).map_err(|f| f.to_string())?;
        let stream = views[0].1;
        let mut ranks: Vec<&mut crate::ep::EpRank> = views.into_iter().map(|(ep, _)| ep).collect();
        rebalancer
            .run(&mut ranks, stream)
            .map(|_| ())
            .map_err(|f| format!("the expert rebalance failed: {f}"))
    }
}

/// One rank of an expert-parallel group in this process, an engine of its
/// own: its scheduler submits to it alone, and its steps fire in lockstep
/// with the group's.
pub struct Rank {
    rank: usize,
    group: Arc<Group>,
    facts: Option<DeviceFacts>,
}

/// Opens an expert-parallel group of one rank per boot, in rank order.
pub fn open_lockstep(
    boots: Vec<DeviceBoot>,
    contract_for: ContractFor,
) -> Result<Vec<Rank>, String> {
    let size = boots.len();
    if !(2..=kernels_cuda::collective::EP_MAX_RANKS as usize).contains(&size) {
        return Err(format!(
            "an expert-parallel group is 2 to 8 ranks, not {size}"
        ));
    }
    let ordinals: Vec<i32> = boots.iter().map(|boot| boot.ordinal).collect();
    let knobs = &boots[0].knobs;
    let rebalance = (knobs.expert_rebalance_steps.filter(|&every| every > 0)).map(|every| {
        crate::ep::rebalance::Rebalance {
            every,
            window: knobs.expert_load_window.unwrap_or(every).max(every),
        }
    });
    let mut ranks = Vec::with_capacity(size);
    for (rank, boot) in boots.into_iter().enumerate() {
        let mut cuda = crate::boot::open(boot, contract_for)?;
        cuda.set_expert_rank(rank as u32, size as u32);
        ranks.push(Arc::new(Mutex::new(cuda)));
    }
    let firing = Firing::start(&ranks, &ordinals)?;
    let rebalancer = rebalance.map(|knob| Mutex::new(crate::ep::rebalance::Rebalancer::new(knob)));
    let group = Arc::new(Group {
        ranks,
        ordinals,
        steps: Rendezvous::new(size, GATHER),
        wiring: Mutex::new(Wiring {
            shapes: vec![None; size],
            wires: None,
        }),
        wired: Mutex::new((0, None)),
        wired_cv: Condvar::new(),
        firing,
        rebalancer,
    });
    // Each rank's load meets the group at arming. The hook holds the group
    // weakly: the group holds the ranks that hold the hooks.
    for (rank, cuda) in group.ranks.iter().enumerate() {
        let weak = Arc::downgrade(&group);
        cuda.lock()
            .expect("rank lock poisoned")
            .before_arming(Box::new(move |shell| {
                let group = weak.upgrade().ok_or("the expert-parallel group is gone")?;
                group.meet(rank, shell).inspect_err(|why| group.poison(why))
            }));
    }
    Ok((0..size)
        .map(|rank| Rank {
            rank,
            group: Arc::clone(&group),
            facts: None,
        })
        .collect())
}

impl Rank {
    /// Clears seat `slot` for a new sequence (`Cuda::open`).
    pub fn open(&self, slot: u32) -> EngineResult<()> {
        self.with(|cuda| cuda.open(slot))
    }

    /// The slot copies the group's rebalances made since load.
    #[must_use]
    pub fn moved_experts(&self) -> u64 {
        self.group.rebalancer.as_ref().map_or(0, |r| {
            r.lock().expect("lockstep rebalance lock poisoned").moved()
        })
    }

    fn cuda(&self) -> EngineResult<std::sync::MutexGuard<'_, Cuda>> {
        crate::device::ctx::bind_thread(self.group.ordinals[self.rank])
            .map_err(|f| Error::Device(f.to_string()))?;
        let cuda = self.group.ranks[self.rank]
            .lock()
            .map_err(|_| Error::Device("this rank's engine panicked".into()))?;
        Ok(cuda)
    }

    fn with<R>(&self, verb: impl FnOnce(&mut Cuda) -> EngineResult<R>) -> EngineResult<R> {
        let mut cuda = self.cuda()?;
        cuda.bind_thread()?;
        verb(&mut cuda)
    }

    /// One step of this rank's frame through the group: led (fired on every
    /// rank) or joined (fired by the step's leader).
    fn step(&self, step: &Step, at: engine::StepDone, bucket: u32) -> Fired {
        let arrival = Arrival {
            frame: (step.clone(), at),
            bucket,
        };
        match self.group.steps.arrive(self.rank, arrival) {
            Err(why) => Err(Error::Device(format!(
                "this expert-parallel group is stopped: {why}"
            ))),
            Ok(Part::Joined(fired)) => fired,
            Ok(Part::Lead(formed)) => {
                let index = formed.step;
                let mut results = self.group.fire(formed);
                if let Some(Some(Err(error))) = results.iter().find(|r| matches!(r, Some(Err(_)))) {
                    self.group.poison(&error.to_string());
                }
                let own = results[self.rank].take();
                self.group.steps.finish(index, results);
                own.unwrap_or_else(|| {
                    Err(Error::Device("the leader's own step did not fire".into()))
                })
            }
        }
    }
}

impl Engine for Rank {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.facts.as_ref()
    }

    fn export_kv_handle(&self) -> Option<KvHandle> {
        self.group.ranks[self.rank]
            .try_lock()
            .ok()?
            .export_kv_handle()
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        self.with(Engine::bind_thread)
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        let loaded = self.with(|cuda| cuda.load(request));
        match &loaded {
            Ok(_) => {
                let facts = self.cuda()?.device_facts().cloned();
                self.facts = facts;
                self.group.loaded();
            }
            Err(error) => self
                .group
                .poison(&format!("rank {} did not load: {error}", self.rank)),
        }
        loaded
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.with(|cuda| cuda.register_program(registration))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        self.with(|cuda| cuda.register_channel(registration))
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        self.with(|cuda| cuda.bind_instance(binding))
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.with(|cuda| cuda.close_instance(id))
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.with(|cuda| cuda.close_channel(id))
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.with(|cuda| cuda.publish_channel(instance, channel, cell))
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.with(|cuda| cuda.take_channel(instance, channel))
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        self.with(|cuda| cuda.register_adapter(registration))
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        self.group.ready()?;
        let (id, buckets) = self.with(|cuda| cuda.lockstep_frame(frame))?;
        let mut tickets = Vec::with_capacity(frame.steps.len());
        let mut settled = Vec::with_capacity(frame.steps.len());
        for (index, (step, bucket)) in frame.steps.iter().zip(buckets).enumerate() {
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            let (ticket, pending) = self.step(step, at, bucket)?;
            tickets.push(ticket);
            settled.push(pending);
        }
        self.with(|cuda| {
            cuda.adopt(id, settled);
            Ok(())
        })?;
        Ok(FrameTicket { id, steps: tickets })
    }

    fn settles_asynchronously(&self) -> bool {
        true
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        if let Ok(mut cuda) = self.cuda() {
            cuda.on_complete(sink);
        }
    }

    fn tap_groups(&self) -> Option<engine::TapGroups> {
        self.group.ranks[self.rank].try_lock().ok()?.tap_groups()
    }

    fn on_tapped(&mut self, sink: engine::TapSink) {
        if let Ok(mut cuda) = self.cuda() {
            cuda.on_tapped(sink);
        }
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        self.with(|cuda| cuda.settle_frame(ticket))
    }

    fn publish_landed(&mut self) -> EngineResult<()> {
        self.with(Engine::publish_landed)
    }

    fn expect_fire(&mut self, _submission: &Step) {
        self.group.steps.announce(self.rank);
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        self.with(|cuda| cuda.copy_kv(copy))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> EngineResult<()> {
        self.with(|cuda| cuda.copy_state(copy))
    }

    fn host_kv_layout(&self) -> Option<state::PageLayout> {
        self.group.ranks[self.rank]
            .try_lock()
            .ok()?
            .host_kv_layout()
    }

    fn read_host_kv(&mut self, slots: &[u32]) -> EngineResult<Vec<u8>> {
        self.with(|cuda| cuda.read_host_kv(slots))
    }

    fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> EngineResult<()> {
        self.with(|cuda| cuda.write_host_kv(slots, bytes))
    }

    fn disconnect(&self, message: &str) {
        self.group.poison(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn lead_or_join(r: &Rendezvous<u32, u32>, rank: usize, frame: u32, bucket: u32) -> u32 {
        match r
            .arrive(rank, Arrival { frame, bucket })
            .expect("not poisoned")
        {
            Part::Lead(formed) => {
                // The stand-in fire: each rank's result is its frame, plus
                // the bucket so a joined rank sees the step's.
                let results = formed
                    .frames
                    .iter()
                    .enumerate()
                    .map(|(at, f)| f.filter(|_| at != rank).map(|f| f * 1000 + formed.bucket))
                    .collect();
                r.finish(formed.step, results);
                frame * 1000 + formed.bucket
            }
            Part::Joined(result) => result,
        }
    }

    /// A rank alone leads at once: nobody announced, nobody is waited for,
    /// and the others idle in its step.
    #[test]
    fn a_lone_frame_fires_without_waiting_for_idle_ranks() {
        let r = Rendezvous::<u32, u32>::new(4, Duration::from_secs(30));
        let began = Instant::now();
        match r
            .arrive(
                2,
                Arrival {
                    frame: 7,
                    bucket: 64,
                },
            )
            .unwrap()
        {
            Part::Lead(formed) => {
                assert_eq!(formed.bucket, 64);
                assert_eq!(formed.frames, vec![None, None, Some(7), None]);
            }
            Part::Joined(_) => panic!("the first arrival leads"),
        }
        assert!(
            began.elapsed() < Duration::from_secs(5),
            "it waited on idle ranks"
        );
    }

    /// Announced ranks are waited for, so two busy ranks share one step at
    /// the larger bucket instead of idling through each other's.
    #[test]
    fn announced_ranks_share_a_step_at_the_largest_bucket() {
        let r = Arc::new(Rendezvous::<u32, u32>::new(2, Duration::from_secs(30)));
        r.announce(1);
        let late = {
            let r = Arc::clone(&r);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(50));
                lead_or_join(&r, 1, 2, 256)
            })
        };
        let first = lead_or_join(&r, 0, 1, 32);
        assert_eq!(first, 1256, "the leader fired at the joined rank's bucket");
        assert_eq!(
            late.join().unwrap(),
            2256,
            "the joined rank's frame fired in the same step"
        );
    }

    /// An announced rank that never comes holds the step only to its
    /// deadline.
    #[test]
    fn a_rank_that_announces_and_never_comes_holds_the_step_to_its_deadline() {
        let r = Rendezvous::<u32, u32>::new(2, Duration::from_millis(100));
        r.announce(1);
        let began = Instant::now();
        let Part::Lead(formed) = r
            .arrive(
                0,
                Arrival {
                    frame: 1,
                    bucket: 8,
                },
            )
            .unwrap()
        else {
            panic!("the first arrival leads");
        };
        assert!(began.elapsed() >= Duration::from_millis(100));
        assert_eq!(formed.frames, vec![Some(1), None]);
    }

    /// Steps keep their order under many ranks submitting many frames: each
    /// rank sees every one of its frames fire exactly once.
    #[test]
    fn every_frame_fires_once_across_many_steps() {
        let ranks = 4;
        let r = Arc::new(Rendezvous::<u32, u32>::new(ranks, Duration::from_millis(5)));
        let threads: Vec<_> = (0..ranks)
            .map(|rank| {
                let r = Arc::clone(&r);
                thread::spawn(move || {
                    (0..200u32)
                        .map(|i| {
                            r.announce(rank);
                            let frame = (rank as u32) * 1_000 + i;
                            lead_or_join(&r, rank, frame, 16) / 1000
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for (rank, t) in threads.into_iter().enumerate() {
            let got = t.join().unwrap();
            let want: Vec<u32> = (0..200).map(|i| (rank as u32) * 1_000 + i).collect();
            assert_eq!(got, want, "rank {rank}'s frames");
        }
    }

    /// A step forms while the one before it fires, but closes only after:
    /// no two leaders fire at once.
    #[test]
    fn no_two_steps_fire_at_once() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let ranks = 4;
        let r = Arc::new(Rendezvous::<u32, u32>::new(ranks, Duration::from_millis(1)));
        let firing = Arc::new(AtomicBool::new(false));
        let threads: Vec<_> = (0..ranks)
            .map(|rank| {
                let (r, firing) = (Arc::clone(&r), Arc::clone(&firing));
                thread::spawn(move || {
                    for i in 0..100u32 {
                        match r
                            .arrive(
                                rank,
                                Arrival {
                                    frame: i,
                                    bucket: 8,
                                },
                            )
                            .unwrap()
                        {
                            Part::Lead(formed) => {
                                assert!(
                                    !firing.swap(true, Ordering::SeqCst),
                                    "two steps fired at once"
                                );
                                thread::sleep(Duration::from_micros(200));
                                firing.store(false, Ordering::SeqCst);
                                let results = formed.frames.iter().map(|f| f.map(|_| 0)).collect();
                                r.finish(formed.step, results);
                            }
                            Part::Joined(_) => {}
                        }
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
    }

    #[test]
    fn a_poisoned_group_answers_every_waiting_rank() {
        let r = Arc::new(Rendezvous::<u32, u32>::new(2, Duration::from_secs(30)));
        r.announce(1);
        let waiting = {
            let r = Arc::clone(&r);
            thread::spawn(move || {
                r.arrive(
                    0,
                    Arrival {
                        frame: 1,
                        bucket: 8,
                    },
                )
                .err()
            })
        };
        thread::sleep(Duration::from_millis(50));
        r.poison("rank 1 refused");
        assert_eq!(waiting.join().unwrap().as_deref(), Some("rank 1 refused"));
        assert!(
            r.arrive(
                1,
                Arrival {
                    frame: 2,
                    bucket: 8
                }
            )
            .is_err()
        );
    }
}
