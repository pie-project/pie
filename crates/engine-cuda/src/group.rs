use std::sync::{Arc, Mutex};
use std::time::Duration;

use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::DeviceFacts;
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::MediaEncode;
use engine::fire::{FrameSubmission, FrameTicket, Step};
use engine::load::{LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, KvHandle, StateCopy};
use eta_ir::container::HostRole;

use crate::api::{ClassifyFor, ContractFor, Cuda, DeviceBoot, World};
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

pub fn open_group(
    boots: Vec<DeviceBoot>,
    contract_for: ContractFor,
    classify_for: ClassifyFor,
) -> Result<Group, String> {
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
        ranks.push(Arc::new(Mutex::new(crate::boot::open(
            boot,
            contract_for,
            classify_for,
        )?)));
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
                    eprintln!("engine-cuda: tensor-parallel rank {rank} refused: {error}");
                    if refused.is_none() {
                        self.poison_with(&format!("rank {rank} refused: {error}"));
                        refused = Some(error);
                        let grace = std::time::Instant::now() + GRACE;
                        while heard < size {
                            let left = grace.saturating_duration_since(std::time::Instant::now());
                            match rx.recv_timeout(left) {
                                Ok((rank, Err(error))) => {
                                    heard += 1;
                                    eprintln!(
                                        "engine-cuda: tensor-parallel rank {rank} refused: {error}"
                                    );
                                }
                                Ok(_) => heard += 1,
                                Err(_) => break,
                            }
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
                    eprintln!("engine-cuda: {why}");
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
            eprintln!("engine-cuda: tensor-parallel rank {rank} refused: {error}");
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
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.facts.as_ref()
    }

    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        Ok(())
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        let mut answers = self.each_within(LOAD_WAIT, move |rank| rank.load(request.clone()))?;
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

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        let registration = registration.clone();
        self.lead(move |rank| rank.register_program(&registration))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        let registered = self.on(0, |rank| rank.register_channel(registration))?;
        if registration.host_role == HostRole::None {
            for rank in 1..self.ranks.len() {
                self.on(rank, |shell| {
                    shell.register_channel(registration).map(|_| ())
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

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let binding = binding.clone();
        self.lead(move |rank| rank.bind_instance(&binding))
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.lead(move |rank| rank.close_instance(id))
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.lead(move |rank| rank.close_channel(id))
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.on(0, |rank| rank.publish_channel(instance, channel, cell))
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.on(0, |rank| rank.take_channel(instance, channel))
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        let registration = registration.clone();
        self.lead(move |rank| rank.register_adapter(&registration))
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        let frame = frame.clone();
        self.each(move |rank| rank.submit(&frame))
            .map(|mut tickets| tickets.swap_remove(0))
    }

    fn settles_asynchronously(&self) -> bool {
        self.ranks[0]
            .try_lock()
            .map(|rank| rank.settles_asynchronously())
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

    fn expect_fire(&mut self, submission: &Step) {
        for rank in &self.ranks {
            if let Ok(mut rank) = rank.try_lock() {
                rank.expect_fire(submission);
            }
        }
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        let copy = copy.clone();
        self.lead(move |rank| rank.copy_kv(&copy))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> EngineResult<()> {
        let copy = copy.clone();
        self.lead(move |rank| rank.copy_state(&copy))
    }

    fn encode(&mut self, plan: &mut MediaEncode) -> EngineResult<()> {
        self.on(0, |rank| rank.encode(plan))
    }

    fn disconnect(&self, message: &str) {
        for rank in &self.ranks {
            if let Ok(rank) = rank.try_lock() {
                rank.disconnect(message);
            }
        }
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
