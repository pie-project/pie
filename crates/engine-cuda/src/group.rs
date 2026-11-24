use std::sync::{Arc, Mutex};
use std::time::Duration;

use engine::{Channels, Completion, Engine, Peers, Pool, Programs, Tiers, WeightCache};
use engine::adapter::AdapterRegistration;
use engine::caps::DeviceFacts;
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FrameSubmission, FrameTicket, Step};
use engine::load::{LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, KvHandle};
use eta_ir::container::HostRole;

use crate::api::{ContractFor, Cuda, DeviceBoot, World};
use crate::comm::{Comm, Id};

const INIT_WAIT: Duration = Duration::from_secs(120);
const LOAD_WAIT: Duration = Duration::from_secs(3600);
const VERB_WAIT: Duration = Duration::from_secs(600);
const GRACE: Duration = Duration::from_secs(30);

pub struct Group {
    ranks: Vec<Arc<Mutex<Cuda>>>,
    ordinals: Vec<i32>,
    poison: Poison,
    facts: Option<DeviceFacts>,
    /// Each copy still landing on some rank, by token: how many ranks have
    /// yet to tell it landed ([`Tiers::copy_landing`]).
    landing: Arc<Mutex<std::collections::HashMap<u64, usize>>>,
}

/// The first refusal poisons the group: every later verb answers it without
/// touching a rank, and the communicators are aborted so a rank the refusing
/// rank left inside a collective comes back.
struct Poison {
    why: Mutex<Option<String>>,
    aborts: Vec<Arc<dyn Fn() + Send + Sync>>,
}

impl Poison {
    fn over(comms: &[Arc<Comm>]) -> Poison {
        Poison {
            why: Mutex::new(None),
            aborts: comms
                .iter()
                .map(|comm| {
                    let comm = Arc::clone(comm);
                    Arc::new(move || comm.abort()) as Arc<dyn Fn() + Send + Sync>
                })
                .collect(),
        }
    }

    fn why(&self) -> Option<String> {
        self.why.lock().map(|held| held.clone()).unwrap_or(None)
    }

    fn set(&self, why: &str) {
        if let Ok(mut held) = self.why.lock() {
            if held.is_some() {
                return;
            }
            *held = Some(why.to_string());
        }
        // `ncclCommAbort` raises NCCL's abort flag first — which is what frees
        // a rank spinning in a collective — and then tears the communicator
        // down, and that teardown waits for every CUDA graph that captured a
        // collective on it to die. The armed bodies hold those graphs for the
        // life of the shell, so an abort run on the lane thread never comes
        // back and the runtime never hears the refusal: the frame stays
        // pending until the client gives up (#649). The cause recorded above
        // is what refuses the next verb; the abort runs apart from the lane.
        for (rank, abort) in self.aborts.iter().enumerate() {
            let abort = Arc::clone(abort);
            let _ = std::thread::Builder::new()
                .name(format!("tp-abort-{rank}"))
                .spawn(move || abort());
        }
    }
}

pub fn open_group(boots: Vec<DeviceBoot>, contract_for: ContractFor) -> Result<Group, String> {
    let size = boots.len();
    if size < 2 {
        return Err(format!(
            "a tensor-parallel group is two or more ranks; {size} boot(s) were given \
             (one device opens through `open`)"
        ));
    }
    let size32 = u32::try_from(size).map_err(|_| "more ranks than a u32 counts".to_string())?;
    let id = Id::new(boots[0].knobs.nccl_transport).map_err(|fault| fault.to_string())?;
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<Comm, String>)>();
    for (rank, boot) in boots.iter().enumerate() {
        let tx = tx.clone();
        let id = id.clone();
        let ordinal = boot.ordinal;
        std::thread::Builder::new()
            .name(format!("nccl-init-{rank}"))
            .spawn(move || {
                let opened = crate::device::ctx::bind_thread(ordinal)
                    .map_err(|fault| fault.to_string())
                    .and_then(|()| {
                        Comm::open(&id, rank as u32, size32).map_err(|fault| fault.to_string())
                    });
                let _ = tx.send((rank, opened));
            })
            .map_err(|why| format!("rank {rank}: no thread to open its communicator on: {why}"))?;
    }
    drop(tx);
    let mut comms: Vec<Option<Comm>> = (0..size).map(|_| None).collect();
    let deadline = std::time::Instant::now() + INIT_WAIT;
    let mut refusal: Option<String> = None;
    for _ in 0..size {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match rx.recv_timeout(left) {
            Ok((rank, Ok(comm))) => comms[rank] = Some(comm),
            Ok((rank, Err(why))) => {
                refusal.get_or_insert(format!("rank {rank} (cuda:{}): {why}", boots[rank].ordinal));
            }
            Err(_) => {
                let missing: Vec<String> = comms
                    .iter()
                    .enumerate()
                    .filter(|(_, comm)| comm.is_none())
                    .map(|(rank, _)| format!("rank {rank} (cuda:{})", boots[rank].ordinal))
                    .collect();
                refusal.get_or_insert(format!(
                    "{} did not join the NCCL group within {}s; the group is {size} \
                     rank(s) over devices {:?} and every rank must open at once",
                    missing.join(", "),
                    INIT_WAIT.as_secs(),
                    boots.iter().map(|boot| boot.ordinal).collect::<Vec<_>>()
                ));
                break;
            }
        }
    }
    if let Some(why) = refusal {
        for comm in comms.iter().flatten() {
            comm.abort();
        }
        return Err(why);
    }
    let devices: Vec<i32> = boots.iter().map(|boot| boot.ordinal).collect();
    if let Some(peers) = crate::comm::open_peers(&devices) {
        for (comm, peers) in comms.iter_mut().flatten().zip(peers) {
            comm.set_peers(peers);
        }
    }
    let mut ranks = Vec::with_capacity(size);
    let mut ordinals = Vec::with_capacity(size);
    let mut held = Vec::with_capacity(size);
    for (rank, (mut boot, comm)) in boots.into_iter().zip(comms).enumerate() {
        let comm = Arc::new(comm.expect("every rank answered above"));
        boot.world = World {
            rank: rank as u32,
            size: size32,
        };
        boot.comm = Some(Arc::clone(&comm));
        ordinals.push(boot.ordinal);
        held.push(comm);
        ranks.push(Arc::new(Mutex::new(crate::boot::open(boot, contract_for)?)));
    }
    let facts = ranks[0]
        .lock()
        .ok()
        .and_then(|rank| rank.device_facts().cloned());
    Ok(Group {
        ranks,
        ordinals,
        poison: Poison::over(&held),
        facts,
        landing: Arc::default(),
    })
}

impl Group {
    #[must_use]
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    fn poison(&self) -> Option<String> {
        self.poison.why()
    }

    fn poison_with(&self, why: &str) {
        self.poison.set(why);
    }

    fn each_within<R, F>(&mut self, wait: Duration, verb: F) -> EngineResult<Vec<R>>
    where
        R: Send + 'static,
        F: Fn(&mut Cuda) -> EngineResult<R> + Send + Sync + 'static,
    {
        if let Some(why) = self.poison() {
            return Err(Error::Device(format!(
                "this tensor-parallel group is poisoned: {why}"
            )));
        }
        let verb = Arc::new(verb);
        let (tx, rx) = std::sync::mpsc::channel::<(usize, EngineResult<R>)>();
        for (rank, (shell, &ordinal)) in self.ranks.iter().zip(&self.ordinals).enumerate() {
            let shell = Arc::clone(shell);
            let verb = Arc::clone(&verb);
            let tx = tx.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("tp-rank-{rank}"))
                .spawn(move || {
                    let answer = (|| {
                        crate::device::ctx::bind_thread(ordinal)
                            .map_err(|fault| Error::Device(fault.to_string()))?;
                        let mut shell = shell.try_lock().map_err(|_| {
                            Error::Device(format!(
                                "rank {rank} is still held by a verb that never came back"
                            ))
                        })?;
                        shell.bind_thread()?;
                        verb(&mut shell)
                    })();
                    let _ = tx.send((rank, answer));
                });
            if let Err(why) = spawned {
                self.poison_with(&format!("rank {rank}: no thread to run it on: {why}"));
                return Err(Error::Device(format!(
                    "rank {rank}: no thread to run the verb on: {why}"
                )));
            }
        }
        drop(tx);
        let size = self.ranks.len();
        let mut answers: Vec<Option<R>> = (0..size).map(|_| None).collect();
        let mut refused: Option<Error> = None;
        let deadline = std::time::Instant::now() + wait;
        let mut heard = 0usize;
        while heard < size {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            match rx.recv_timeout(left) {
                Ok((rank, Ok(value))) => {
                    answers[rank] = Some(value);
                    heard += 1;
                }
                Ok((rank, Err(error))) => {
                    heard += 1;
                    tracing::warn!("tensor-parallel rank {rank} refused: {error}");
                    if refused.is_none() {
                        // Every rank refusing the same lanes ran nothing:
                        // the group stays whole (`engine::Error::Lanes`).
                        let lanes = match &error {
                            Error::Lanes(lanes) => Some(lanes.clone()),
                            _ => None,
                        };
                        let mut agreed = lanes.is_some() && answers.iter().all(Option::is_none);
                        let first = format!("rank {rank} refused: {error}");
                        refused = Some(error);
                        let grace = std::time::Instant::now() + GRACE;
                        while heard < size {
                            let left = grace.saturating_duration_since(std::time::Instant::now());
                            match rx.recv_timeout(left) {
                                Ok((rank, Err(error))) => {
                                    heard += 1;
                                    agreed &= matches!(
                                        (&error, &lanes),
                                        (Error::Lanes(these), Some(first)) if these == first
                                    );
                                    tracing::warn!(
                                        "tensor-parallel rank {rank} refused: {error}"
                                    );
                                }
                                Ok(_) => {
                                    heard += 1;
                                    agreed = false;
                                }
                                Err(_) => break,
                            }
                        }
                        if !(agreed && heard == size) {
                            self.poison_with(&first);
                        }
                        break;
                    }
                }
                Err(_) => {
                    let missing: Vec<usize> = answers
                        .iter()
                        .enumerate()
                        .filter(|(_, answer)| answer.is_none())
                        .map(|(rank, _)| rank)
                        .collect();
                    let why = format!(
                        "rank(s) {missing:?} of the tensor-parallel group answered nothing \
                         within {}s; the group is aborted and poisoned",
                        wait.as_secs()
                    );
                    tracing::warn!("{why}");
                    self.poison_with(&why);
                    refused = Some(Error::Device(why));
                    break;
                }
            }
        }
        if let Some(error) = refused {
            return Err(error);
        }
        Ok(answers
            .into_iter()
            .map(|answer| answer.expect("every rank answered"))
            .collect())
    }

    fn each<R, F>(&mut self, verb: F) -> EngineResult<Vec<R>>
    where
        R: Send + 'static,
        F: Fn(&mut Cuda) -> EngineResult<R> + Send + Sync + 'static,
    {
        self.each_within(VERB_WAIT, verb)
    }

    fn on<R, F>(&mut self, rank: usize, verb: F) -> EngineResult<R>
    where
        F: FnOnce(&mut Cuda) -> EngineResult<R>,
    {
        if let Some(why) = self.poison() {
            return Err(Error::Device(format!(
                "this tensor-parallel group is poisoned: {why}"
            )));
        }
        crate::device::ctx::bind_thread(self.ordinals[rank])
            .map_err(|fault| Error::Device(fault.to_string()))?;
        let mut shell = self.ranks[rank].try_lock().map_err(|_| {
            Error::Device(format!(
                "rank {rank} is still held by a verb that never came back"
            ))
        })?;
        shell.bind_thread()?;
        verb(&mut shell).inspect_err(|error| {
            tracing::warn!("tensor-parallel rank {rank} refused: {error}");
        })
    }

    fn lead<R, F>(&mut self, verb: F) -> EngineResult<R>
    where
        R: Send + 'static,
        F: Fn(&mut Cuda) -> EngineResult<R> + Send + Sync + 'static,
    {
        let mut answers = self.each(verb)?;
        Ok(answers.swap_remove(0))
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        let ranks = std::mem::take(&mut self.ranks);
        let ordinals = std::mem::take(&mut self.ordinals);
        if self.poison().is_some() {
            for rank in ranks {
                std::mem::forget(rank);
            }
            return;
        }
        std::thread::scope(|scope| {
            for (rank, ordinal) in ranks.into_iter().zip(ordinals) {
                scope.spawn(move || {
                    let _ = crate::device::ctx::bind_thread(ordinal);
                    drop(rank);
                });
            }
        });
    }
}

impl Engine for Group {
    /// A group keys no host cover, and `measure = false` is refused for one by name (worker config).
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    /// Not tracked for a tensor-parallel group: its ranks measure at every boot (`measure =
    /// false` is refused for one), said so by name.
    fn measured_at_load(&self) -> Vec<String> {
        vec!["cuda (a tensor-parallel group): what it measured is not tracked".to_string()]
    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.facts.as_ref()
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        Ok(())
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        let mut answers = self.each_within(LOAD_WAIT, move |rank| rank.load(request.clone()))?;
        // The shared pool's tiers: each rank pins and opens
        // its own, and the group moves what every rank holds.
        let least = |of: fn(&engine::chunk::ChunkFacts) -> u32| {
            answers
                .iter()
                .map(|loaded| of(&loaded.caps.chunks))
                .min()
                .unwrap_or(0)
        };
        let (host_chunks, disk_chunks) = (least(|c| c.host_chunks), least(|c| c.disk_chunks));
        let every = |of: fn(&engine::chunk::ChunkFacts) -> bool| {
            answers.iter().all(|loaded| of(&loaded.caps.chunks))
        };
        let (persistent, kept) = (every(|c| c.disk_persistent), every(|c| c.disk_kept));
        for loaded in &mut answers {
            loaded.caps.chunks.host_chunks = host_chunks;
            loaded.caps.chunks.disk_chunks = disk_chunks;
            loaded.caps.chunks.disk_persistent = persistent;
            loaded.caps.chunks.disk_kept = kept;
        }
        // a rank's facts (codegen backend, pools) exist only once it has loaded;
        // the snapshot taken at open was empty, and a group that keeps it says
        // "no codegen backend" to the host, which then emits no kernels and
        // every program registration is refused ("stage 0 region 0 is a
        // generated region and the host emitted nothing for it")
        self.facts = self.ranks[0]
            .try_lock()
            .ok()
            .and_then(|rank| rank.device_facts().cloned());
        Ok(answers.swap_remove(0))
    }

    fn readout_facts(&self) -> EngineResult<engine::ReadoutFacts> {
        Ok(crate::api::readout_facts())
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        let frame = frame.clone();
        let tickets = self.each(move |rank| rank.submit(&frame))?;
        // Ranks fire one frame: a frame one rank stopped short where another
        // did not has its ranks apart.
        if tickets
            .iter()
            .any(|ticket| ticket.steps.len() != tickets[0].steps.len() || ticket.refused != tickets[0].refused)
        {
            let why = "the tensor-parallel ranks stopped one frame at different steps";
            self.poison_with(why);
            return Err(Error::Device(why.to_string()));
        }
        Ok(tickets.into_iter().next().expect("a group has a rank"))
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let template = ticket.clone();
        let mut settled = self.each(move |rank| {
            let mut own = template.clone();
            rank.settle_frame(&mut own)?;
            Ok(own)
        })?;
        *ticket = settled.swap_remove(0);
        Ok(())
    }

    fn completion(&mut self) -> Option<&mut dyn Completion> {
        Some(self)
    }

    fn programs(&mut self) -> Option<&mut dyn Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn Programs> {
        Some(self)
    }

    fn channels(&mut self) -> Option<&mut dyn Channels> {
        Some(self)
    }

    fn channels_ref(&self) -> Option<&dyn Channels> {
        Some(self)
    }

    fn pool(&mut self) -> Option<&mut dyn Pool> {
        Some(self)
    }

    fn weights(&mut self) -> Option<&mut dyn WeightCache> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn Tiers> {
        Some(self)
    }

    fn peers(&mut self) -> Option<&mut dyn Peers> {
        Some(self)
    }
}

impl Completion for Group {
    fn asynchronous(&self) -> bool {
        self.ranks[0]
            .try_lock()
            .map(|rank| rank.asynchronous())
            .unwrap_or(true)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        let silent: engine::CompletionSink = Arc::new(|_, _| {});
        for (rank, shell) in self.ranks.iter().enumerate() {
            if let Ok(mut shell) = shell.try_lock() {
                shell.on_complete(if rank == 0 {
                    sink.clone()
                } else {
                    silent.clone()
                });
            }
        }
    }

    fn publish_landed(&mut self) -> EngineResult<()> {
        self.each(|rank| rank.publish_landed())?;
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        for rank in &self.ranks {
            if let Ok(mut rank) = rank.try_lock() {
                rank.expect_fire(submission);
            }
        }
    }
}

impl Programs for Group {
    fn register(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        let registration = registration.clone();
        self.lead(move |rank| rank.register(&registration))
    }

    /// Each rank plans its own build, and the compile builds every rank's
    /// in turn; each rank loads its own modules at the finish.
    fn prepare(
        &mut self,
        registration: &ProgramRegistration,
    ) -> EngineResult<engine::ProgramCompile> {
        let mut compiles = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            compiles.push(self.on(rank, |cuda| cuda.prepare(registration))?);
        }
        Ok(Box::new(move || {
            let built: Vec<engine::Prepared> =
                compiles.into_iter().map(|compile| compile()).collect();
            engine::Prepared(Box::new(built))
        }))
    }

    fn finish(&mut self, prepared: engine::Prepared) -> EngineResult<ProgramId> {
        let Ok(built) = prepared.0.downcast::<Vec<engine::Prepared>>() else {
            return Err(Error::Program(
                "a prepared program this group did not prepare".to_string(),
            ));
        };
        let mut ids = Vec::with_capacity(built.len());
        for (rank, prepared) in built.into_iter().enumerate() {
            ids.push(self.on(rank, |cuda| cuda.finish(prepared))?);
        }
        Ok(ids.swap_remove(0))
    }

    fn close_program(&mut self, program: ProgramId) -> EngineResult<()> {
        for rank in 0..self.ranks.len() {
            self.on(rank, |cuda| cuda.close_program(program))?;
        }
        Ok(())
    }

    fn facts(&self, program: ProgramId) -> EngineResult<engine::program::ProgramFacts> {
        match self.ranks.first().and_then(|rank| rank.try_lock().ok()) {
            Some(rank) => rank.facts(program),
            None => Err(Error::Program(
                "a rank is busy: its program's facts are asked between frames".to_string(),
            )),
        }
    }

    fn guest_state(
        &self,
        program: ProgramId,
        extents: &engine::program::BindExtents,
        class: engine::ports::GeometryClass,
    ) -> EngineResult<engine::GuestState> {
        match self.ranks.first().and_then(|rank| rank.try_lock().ok()) {
            Some(rank) => rank.guest_state(program, extents, class),
            None => Err(Error::Program(
                "a rank is busy: a program's guest state is asked between frames".to_string(),
            )),
        }
    }

    fn rebind(&mut self, rebind: engine::program::Rebind) -> EngineResult<()> {
        let rebind = rebind.clone();
        self.lead(move |rank| rank.rebind(rebind.clone()))
    }

    fn bind(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let binding = binding.clone();
        self.lead(move |rank| rank.bind(&binding))
    }

    fn close(&mut self, id: InstanceId) -> EngineResult<u64> {
        self.lead(move |rank| Programs::close(&mut *rank, id))
    }

    fn adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        let registration = registration.clone();
        self.lead(move |rank| rank.adapter(&registration))
    }
}

impl Channels for Group {
    /// As each rank: rank 0's answer, which every rank shares.
    fn holds(&self, registration: &ChannelRegistration) -> bool {
        let _ = registration;
        true
    }

    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
    ) -> EngineResult<Option<engine::program::RingFacts>> {
        match self.ranks.first().and_then(|rank| rank.try_lock().ok()) {
            Some(rank) => rank.ring_facts(registration),
            None => Err(Error::Program(
                "a rank is busy: a ring's facts are asked between frames".to_string(),
            )),
        }
    }

    fn open(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        let registered = self.on(0, |rank| Channels::open(&mut *rank, registration))?;
        if registration.host_role == HostRole::None {
            for rank in 1..self.ranks.len() {
                self.on(rank, |shell| {
                    Channels::open(&mut *shell, registration).map(|_| ())
                })?;
            }
        } else {
            let endpoint = self
                .on(0, |rank| Ok(rank.endpoint(registration.id)))?
                .expect("rank 0 just registered this channel");
            let id = registration.id;
            for rank in 1..self.ranks.len() {
                let endpoint = endpoint.clone();
                self.on(rank, move |shell| {
                    shell.adopt_channel(id, endpoint).map(|_| ())
                })?;
            }
        }
        Ok(registered)
    }

    fn close(&mut self, id: ChannelId) -> EngineResult<()> {
        self.lead(move |rank| Channels::close(&mut *rank, id))
    }

    fn publish(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.on(0, |rank| rank.publish(instance, channel, cell))
    }

    fn take(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.on(0, |rank| rank.take(instance, channel))
    }
}

impl Pool for Group {
    /// Each rank moves its band of every slot to and from its own tiers,
    /// whole slots laid alike.
    fn commit(&mut self, segments: u32) -> EngineResult<()> {
        self.lead(move |rank| rank.commit(segments))
    }

    fn release(&mut self, segments: u32) -> EngineResult<()> {
        self.lead(move |rank| rank.release(segments))
    }

    /// Every rank moves the same chunks and rebases its guests' rings.
    fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> EngineResult<()> {
        let evacuation = evacuation.clone();
        self.lead(move |rank| rank.evacuate(&evacuation))
    }

    /// Every rank's table, which every evacuation moves alike: one where
    /// they agree, refused by name where a rank's differs.
    fn chunk_table(&self) -> EngineResult<engine::chunk::ChunkTable> {
        let mut tables = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let Ok(rank) = rank.try_lock() else {
                return Err(Error::Program(
                    "a rank is busy: its chunk table is read between frames".to_string(),
                ));
            };
            tables.push(rank.chunk_table()?);
        }
        let first = tables
            .first()
            .cloned()
            .ok_or_else(|| Error::Program("an expert-parallel group of no ranks".to_string()))?;
        if let Some(at) = tables.iter().position(|table| *table != first) {
            return Err(Error::Program(format!(
                "rank {at}'s chunk table differs from rank 0's: an evacuation moved them apart"
            )));
        }
        Ok(first)
    }
}

impl Tiers for Group {
    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        let copy = copy.clone();
        self.lead(move |rank| rank.copy_kv(&copy))
    }

    fn copy(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        let copy = copy.clone();
        self.lead(move |rank| rank.copy(&copy))
    }

    /// Every rank's moves landed.
    fn settle(&mut self) -> EngineResult<()> {
        self.each(|rank| rank.settle())?;
        Ok(())
    }

    /// Landed once every rank's copy has: the ranks that answer it landed
    /// inside the call count down here, the others from their own sinks
    /// ([`Tiers::on_landed`]), and whichever brings the count to nothing
    /// tells it.
    fn copy_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        token: u64,
    ) -> EngineResult<bool> {
        self.landing
            .lock()
            .unwrap()
            .insert(token, self.ranks.len());
        let copy = copy.clone();
        let answers = match self.each(move |rank| rank.copy_landing(&copy, token)) {
            Ok(answers) => answers,
            Err(error) => {
                self.landing.lock().unwrap().remove(&token);
                return Err(error);
            }
        };
        let landed = answers.iter().filter(|landed| **landed).count();
        let mut left = self.landing.lock().unwrap();
        // Gone: every rank still landing told it, and their last told it on.
        let Some(count) = left.get_mut(&token) else {
            return Ok(false);
        };
        *count -= landed;
        if *count == 0 {
            left.remove(&token);
            return Ok(true);
        }
        Ok(false)
    }

    fn on_landed(&mut self, sink: engine::LandingSink) {
        let left = Arc::clone(&self.landing);
        // A rank's copy that failed fails the group's: the first rank's
        // fault is told once the last rank has told.
        let failed: Arc<Mutex<std::collections::HashMap<u64, String>>> = Arc::default();
        let each: engine::LandingSink = Arc::new(move |token, landing| {
            if let engine::Landing::Failed(why) = landing {
                failed.lock().unwrap().entry(token).or_insert(why);
            }
            let last = {
                let mut left = left.lock().unwrap();
                match left.get_mut(&token) {
                    Some(count) if *count > 1 => {
                        *count -= 1;
                        false
                    }
                    Some(_) => {
                        left.remove(&token);
                        true
                    }
                    None => false,
                }
            };
            if last {
                match failed.lock().unwrap().remove(&token) {
                    Some(why) => sink(token, engine::Landing::Failed(why)),
                    None => sink(token, engine::Landing::Landed),
                }
            }
        });
        for shell in &self.ranks {
            if let Ok(mut shell) = shell.try_lock() {
                shell.on_landed(each.clone());
            }
        }
    }

    /// Rank after rank: each holds its own share of every slot.
    fn read(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> EngineResult<Vec<u8>> {
        let mut out = Vec::new();
        for rank in 0..self.ranks.len() {
            out.extend(self.on(rank, |shell| Tiers::read(&mut *shell, slots))?);
        }
        Ok(out)
    }

    fn write(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> EngineResult<()> {
        let ranks = self.ranks.len();
        if ranks == 0 || !bytes.len().is_multiple_of(ranks) {
            return Err(Error::Device(format!(
                "{} bytes of host tier slots do not split across {ranks} ranks",
                bytes.len()
            )));
        }
        let share = bytes.len() / ranks;
        for (rank, bytes) in bytes.chunks(share).enumerate() {
            self.on(rank, |shell| Tiers::write(&mut *shell, slots, bytes))?;
        }
        Ok(())
    }
}

impl Peers for Group {
    fn export(&self) -> Option<KvHandle> {
        None
    }

    fn layout(&self) -> Option<state::PageLayout> {
        let ranks = u32::try_from(self.ranks.len()).ok()?;
        let rank = self.ranks.first()?.try_lock().ok()?;
        let one = rank.layout()?;
        Some(state::PageLayout {
            ranks,
            page_tokens: one.page_tokens,
            planes: one
                .planes
                .into_iter()
                .map(|plane| state::Plane {
                    heads: plane.heads * ranks,
                    ..plane
                })
                .collect(),
        })
    }

    /// Rank after rank: each holds its own heads of every slot.
    fn read(&mut self, slots: &[u32]) -> EngineResult<Vec<u8>> {
        let mut out = Vec::new();
        for rank in 0..self.ranks.len() {
            out.extend(self.on(rank, |shell| Peers::read(&mut *shell, slots))?);
        }
        Ok(out)
    }

    fn write(&mut self, slots: &[u32], bytes: &[u8]) -> EngineResult<()> {
        let ranks = self.ranks.len();
        if ranks == 0 || !bytes.len().is_multiple_of(ranks) {
            return Err(Error::Device(format!(
                "{} bytes of host kv do not split across {ranks} ranks",
                bytes.len()
            )));
        }
        let share = bytes.len() / ranks;
        for (rank, bytes) in bytes.chunks(share).enumerate() {
            self.on(rank, |shell| Peers::write(&mut *shell, slots, bytes))?;
        }
        Ok(())
    }
}

impl WeightCache for Group {
    fn set_residency(&mut self, residency: &engine::chunk::WeightResidency) -> EngineResult<u64> {
        let _ = residency;
        Err(Error::unsupported(self.kind(), "weight_residency"))
    }

    fn uses(&mut self) -> EngineResult<engine::chunk::WeightUses> {
        Err(Error::unsupported(self.kind(), "weight_uses"))
    }

    fn reload(&mut self, reload: &engine::Reload) -> EngineResult<engine::Reloaded> {
        let reload = reload.clone();
        let ranks = self.each(move |rank| rank.reload(&reload))?;
        Ok(ranks
            .iter()
            .fold(engine::Reloaded::default(), |sum, rank| engine::Reloaded {
                planes: sum.planes.max(rank.planes),
                bytes: sum.bytes + rank.bytes,
            }))
    }
}


#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Instant;

    use super::*;

    /// A rank's refusal must reach the runtime even when aborting the
    /// communicators never returns (a shell whose armed bodies hold the graphs
    /// NCCL's teardown waits for): the cause is recorded at once, the next
    /// verb reads it, and the abort is asked of each communicator exactly once.
    #[test]
    fn a_poison_whose_abort_never_returns_still_answers() {
        let aborts = Arc::new(AtomicUsize::new(0));
        let (release, parked) = mpsc::channel::<()>();
        let parked = Mutex::new(parked);
        let poison = Poison {
            why: Mutex::new(None),
            aborts: vec![Arc::new({
                let aborts = Arc::clone(&aborts);
                move || {
                    aborts.fetch_add(1, Ordering::SeqCst);
                    if let Ok(parked) = parked.lock() {
                        let _ = parked.recv();
                    }
                }
            })],
        };
        assert_eq!(poison.why(), None);

        let began = Instant::now();
        poison.set("rank 0 refused: load failed: this plan names a verb the shell does not bind");
        poison.set("rank 1 refused: later, and not the cause");
        assert!(
            began.elapsed() < Duration::from_secs(5),
            "poisoning waited on the abort"
        );
        assert_eq!(
            poison.why().as_deref(),
            Some("rank 0 refused: load failed: this plan names a verb the shell does not bind")
        );

        let deadline = Instant::now() + Duration::from_secs(5);
        while aborts.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(
            aborts.load(Ordering::SeqCst),
            1,
            "the abort was not asked once"
        );
        drop(release);
    }
}
