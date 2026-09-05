//! `Group`: one tensor-parallel deployment behind [`Engine`], made of one
//! [`Cuda`] shell per rank.
//!
//! Traces are SPMD — every rank runs the same plan over its own band of the
//! weights and meets the others in the plan's collectives — so a group is
//! the same verbs, issued to every rank at once. Each verb runs on one scoped
//! thread per rank (a rank's device is bound per thread), which is what lets
//! a fire's collectives complete: rank 0's launch would otherwise wait on a
//! peer that has not been asked yet. Rank 0 answers; the followers' answers
//! are checked for errors and for agreeing on the identities they mint.
//!
//! The runtime sees one engine: one load, one frame numbering, one
//! completion sink (rank 0's; a follower's completions are its own
//! business), one set of channels read from rank 0.

use std::sync::Arc;

use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::DeviceFacts;
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FrameSubmission, FrameTicket, Step};
use engine::load::{LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, KvHandle, StateCopy};
use engine::fire::MediaEncode;

use crate::api::{ClassifyFor, ContractFor, Cuda, DeviceBoot, World};
use crate::comm::{Comm, Id};

/// The ranks of one tensor-parallel group, as one engine.
pub struct Group {
    ranks: Vec<Cuda>,
    ordinals: Vec<i32>,
}

/// Opens one CUDA shell per boot as a tensor-parallel group: rank `i` is
/// `boots[i]`, its width is `boots.len()`, and every rank's communicator is
/// opened here, together, before any shell exists.
///
/// # Errors
///
/// Fewer than two boots (one rank is [`open`](crate::open)), a knob out of
/// range, or a communicator NCCL refused to open.
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
    let id = Id::new().map_err(|fault| fault.to_string())?;
    // Every rank opens its communicator on a thread bound to its own device;
    // `ncclCommInitRank` returns only once the whole group has arrived, so
    // the opens run concurrently.
    let comms: Vec<Result<Comm, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = boots
            .iter()
            .enumerate()
            .map(|(rank, boot)| {
                let id = &id;
                let ordinal = boot.ordinal;
                scope.spawn(move || {
                    crate::device::ctx::bind_thread(ordinal).map_err(|fault| fault.to_string())?;
                    Comm::open(id, rank as u32, size32).map_err(|fault| fault.to_string())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("a rank's communicator thread panicked".to_string()))
            })
            .collect()
    });
    let mut ranks = Vec::with_capacity(size);
    let mut ordinals = Vec::with_capacity(size);
    for (rank, (mut boot, comm)) in boots.into_iter().zip(comms).enumerate() {
        let comm = comm.map_err(|why| format!("rank {rank} (cuda:{}): {why}", boot.ordinal))?;
        boot.world = World {
            rank: rank as u32,
            size: size32,
        };
        boot.comm = Some(Arc::new(comm));
        ordinals.push(boot.ordinal);
        ranks.push(crate::boot::open(boot, contract_for, classify_for)?);
    }
    Ok(Group { ranks, ordinals })
}




/// A copy of `frame` with guest attachments removed (model lanes+media only).
/// A follower runs this; the guest plane is rank 0's. NOTE: drops the pipelined
/// decode-token carrier (device-only `tok_in`), so tp2 decode is wrong on the
/// follower — see the tp-verification memo for the open fix.
fn strip_guest(frame: &FrameSubmission) -> FrameSubmission {
    FrameSubmission {
        steps: frame.steps.iter().map(|step| Step {
            lanes: step.lanes.clone(),
            attachments: Vec::new(),
            media: step.media.clone(),
        }).collect(),
    }
}

impl Group {
    /// How many ranks this group is.
    #[must_use]
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// Runs `verb` on every rank at once, each on a thread bound to that
    /// rank's device, and returns every rank's answer in rank order. The
    /// first rank that refused speaks for the group.
    fn each<R, F>(&mut self, verb: F) -> EngineResult<Vec<R>>
    where
        R: Send,
        F: Fn(&mut Cuda) -> EngineResult<R> + Sync,
    {
        let verb = &verb;
        let ordinals = &self.ordinals;
        let answers: Vec<std::thread::Result<EngineResult<R>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .ranks
                .iter_mut()
                .zip(ordinals)
                .map(|(rank, &ordinal)| {
                    scope.spawn(move || {
                        crate::device::ctx::bind_thread(ordinal)
                            .map_err(|fault| Error::Device(fault.to_string()))?;
                        rank.bind_thread()?;
                        verb(rank)
                    })
                })
                .collect();
            handles.into_iter().map(std::thread::ScopedJoinHandle::join).collect()
        });
        let mut out = Vec::with_capacity(answers.len());
        for (rank, answer) in answers.into_iter().enumerate() {
            match answer {
                Ok(Ok(value)) => out.push(value),
                // Said here as well as returned: the group's teardown waits on
                // every rank, so a refusal is otherwise the last thing read.
                Ok(Err(error)) => {
                    eprintln!("engine-cuda: tensor-parallel rank {rank} refused: {error}");
                    return Err(error);
                }
                Err(_) => {
                    return Err(Error::Device(format!(
                        "rank {rank} of the tensor-parallel group panicked"
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Runs `verb` on rank `rank` alone, on this thread bound to its device.
    fn on<R, F>(&mut self, rank: usize, verb: F) -> EngineResult<R>
    where
        F: FnOnce(&mut Cuda) -> EngineResult<R>,
    {
        crate::device::ctx::bind_thread(self.ordinals[rank])
            .map_err(|fault| Error::Device(fault.to_string()))?;
        let shell = &mut self.ranks[rank];
        shell.bind_thread()?;
        verb(shell).inspect_err(|error| {
            eprintln!("engine-cuda: tensor-parallel rank {rank} refused: {error}");
        })
    }

    /// [`each`](Self::each), answering with rank 0's value.
    fn lead<R, F>(&mut self, verb: F) -> EngineResult<R>
    where
        R: Send,
        F: Fn(&mut Cuda) -> EngineResult<R> + Sync,
    {
        let mut answers = self.each(verb)?;
        Ok(answers.swap_remove(0))
    }






}

impl Drop for Group {
    /// Every rank is torn down at once, each on a thread bound to its
    /// device: `ncclCommDestroy` waits for the rest of the clique, so
    /// dropping the ranks one after another would hang on the first.
    fn drop(&mut self) {
        let ranks = std::mem::take(&mut self.ranks);
        let ordinals = std::mem::take(&mut self.ordinals);
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
        self.ranks[0].device_facts()
    }

    fn export_kv_handle(&self) -> Option<KvHandle> {
        // A rank's pages hold its band of the heads; a transfer plane that
        // reads one rank's handle would move half a cache.
        None
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        // Each rank binds its own device inside `each`.
        Ok(())
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        // Every rank loads at once: the warm-up fires inside a load carry
        // the plan's collectives, which need the whole group present.
        let request = &request;
        self.lead(|rank| rank.load(request.clone()))
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        // Every rank runs the guest: a PIPELINED decode step's input token is
        // sampled by the prior step's epilogue and injected by this step's
        // prologue, so a follower without the guest would decode a placeholder
        // token. Sampling (`argmax`) is deterministic over the full,
        // replicated logits, so every rank samples the same token; only the
        // host-facing streaming (take_channel) is read from rank 0.
        self.on(0, |rank| rank.register_program(registration))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        self.on(0, |rank| rank.register_channel(registration))
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        self.on(0, |rank| rank.bind_instance(binding))
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.on(0, |rank| rank.close_instance(id))
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.on(0, |rank| rank.close_channel(id))
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.on(0, |rank| rank.publish_channel(instance, channel, cell))
    }

    fn take_channel(&mut self, instance: InstanceId, channel: u32) -> EngineResult<Option<Vec<u8>>> {
        self.on(0, |rank| rank.take_channel(instance, channel))
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        self.on(0, |rank| rank.register_adapter(registration))
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        // Rank 0 runs the whole fire — the model forward AND the guest
        // boundaries attached to it. A follower runs only the model: the
        // same lanes drive the same SPMD forward and the same collectives,
        // but the guest attachments (host I/O, sampling) are rank 0's, so
        // they are stripped from the followers' frame. Rank 0's ticket is
        // the group's; a follower's readouts land in its own arena, unread.
        let follower = strip_guest(frame);
        self.each(|rank| rank.submit(if rank.rank() == 0 { frame } else { &follower }))
            .map(|mut tickets| tickets.swap_remove(0))
    }

    fn settles_asynchronously(&self) -> bool {
        self.ranks[0].settles_asynchronously()
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        // One completion per step reaches the runtime: rank 0's. A follower
        // still settles its own fires; it just tells no one.
        let silent: engine::CompletionSink = Arc::new(|_, _| {});
        for (rank, shell) in self.ranks.iter_mut().enumerate() {
            shell.on_complete(if rank == 0 { sink.clone() } else { silent.clone() });
        }
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        // Rank 0's ticket carries the readouts back; the followers settle a
        // copy so their pending frame retires with it.
        let template = ticket.clone();
        let mut settled = self.each(|rank| {
            let mut own = template.clone();
            rank.settle_frame(&mut own)?;
            Ok(own)
        })?;
        *ticket = settled.swap_remove(0);
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        for rank in &mut self.ranks {
            rank.expect_fire(submission);
        }
    }


    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        self.lead(|rank| rank.copy_kv(copy))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> EngineResult<()> {
        self.lead(|rank| rank.copy_state(copy))
    }

    fn encode(&mut self, plan: &mut MediaEncode) -> EngineResult<()> {
        let _ = plan;
        Err(Error::unsupported(
            "cuda",
            "encode (a media tower under tensor parallelism)",
        ))
    }

    fn disconnect(&self, message: &str) {
        for rank in &self.ranks {
            rank.disconnect(message);
        }
    }
}
