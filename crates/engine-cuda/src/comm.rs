use core::ffi::c_void;
use std::fmt;

use crate::error::{Fault, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transport {
    #[default]
    Shm,
    Peer,
    Nccl,
}

impl std::str::FromStr for Transport {
    type Err = String;

    fn from_str(word: &str) -> std::result::Result<Transport, String> {
        match word {
            "shm" => Ok(Transport::Shm),
            "peer" => Ok(Transport::Peer),
            "nccl" => Ok(Transport::Nccl),
            other => Err(format!(
                "`{other}` does not name an NCCL transport; the spellings are \
                 `shm` (P2P off, the default), `peer` (P2P on), and `nccl` \
                 (whatever NCCL's own environment says)"
            )),
        }
    }
}

#[derive(Clone)]
pub struct Id(pub [u8; 128]);

impl Id {
    pub fn new(transport: Transport) -> Result<Id> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            transport_defaults(transport);
            let mut id = nccl::ncclUniqueId { internal: [0; 128] };
            // SAFETY: a live out-parameter of the exact type NCCL writes.
            let code = unsafe { nccl::ncclGetUniqueId(&raw mut id) };
            answered("ncclGetUniqueId", code)?;
            Ok(Id(id.internal.map(|byte| byte as u8)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = transport;
            Err(Fault::Runtimeless)
        }
    }
}

pub struct Comm {
    raw: *mut c_void,
    rank: u32,
    size: u32,
    peers: Option<kernels_cuda::collective::Peers>,
}

// SAFETY: NCCL communicators are used from any thread as long as calls on one
// communicator are not concurrent, which the group guarantees by driving each
// rank from one thread at a time.
unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl Comm {
    pub fn open(id: &Id, rank: u32, size: u32) -> Result<Comm> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            let unique = nccl::ncclUniqueId {
                internal: id.0.map(|byte| byte as core::ffi::c_char),
            };
            let mut raw: nccl::ncclComm_t = core::ptr::null_mut();
            // SAFETY: a live out-parameter; `unique` is by value, as the
            // binding declares it.
            let code = unsafe {
                nccl::ncclCommInitRank(
                    &raw mut raw,
                    size as core::ffi::c_int,
                    unique,
                    rank as core::ffi::c_int,
                )
            };
            answered("ncclCommInitRank", code)?;
            Ok(Comm {
                raw: raw.cast(),
                rank,
                size,
                peers: None,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (id, rank, size);
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn abort(&self) {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            if self.raw.is_null() {
                return;
            }
            // SAFETY: a live communicator this group opened; NCCL allows an
            // abort from any thread while other threads are inside calls on
            // the same communicator — that is what it is for.
            let _ = unsafe { nccl::ncclCommAbort(self.raw.cast()) };
        }
    }

    #[must_use]
    pub fn peers(&self) -> Option<kernels_cuda::collective::Peers> {
        self.peers
    }

    pub fn set_peers(&mut self, peers: kernels_cuda::collective::Peers) {
        self.peers = Some(peers);
    }

    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    #[must_use]
    pub fn size(&self) -> u32 {
        self.size
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        self.raw = core::ptr::null_mut();
    }
}

impl fmt::Debug for Comm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Comm")
            .field("rank", &self.rank)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Comm {
    fn eq(&self, other: &Comm) -> bool {
        self.raw == other.raw
    }
}

/// Bytes of each rank's peer stage; a longer message goes in stage-sized
/// chunks.
const STAGE_BYTES: u64 = 16 << 20;

/// Maps every rank's stage into every other rank when each pair of devices
/// reaches the other: the group's ranks share this process, so a peer's
/// buffer is a plain device pointer once peer access is on. `None` when some
/// pair does not, and the collectives stay on NCCL.
#[must_use]
pub fn open_peers(ordinals: &[i32]) -> Option<Vec<kernels_cuda::collective::Peers>> {
    #[cfg(feature = "cuda")]
    {
        match wire_peers(ordinals) {
            Ok(peers) => peers,
            Err(fault) => {
                eprintln!("engine-cuda: peer collectives are off, NCCL carries them: {fault}");
                None
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinals;
        None
    }
}

/// Replays per size and path; the median decides, so one descheduled thread
/// or cold first replay does not.
const SAMPLES: usize = 9;

/// Sets how long a message the peer path carries. Its kernels read the peers'
/// stages over whatever link joins the devices, and some PCIe pairs deliver
/// those reads far slower than NCCL's own transport past short messages
/// (#741), so every rank times both paths on the same message sizes, for the
/// all-reduce and the all-gather apart, and each carries its sizes up to the
/// reach that costs least. `None` when that is none for both.
#[must_use]
pub fn calibrate(
    ordinals: &[i32],
    comms: &[&Comm],
    peers: Vec<kernels_cuda::collective::Peers>,
) -> Option<Vec<kernels_cuda::collective::Peers>> {
    let sizes: Vec<u64> = (14..=STAGE_BYTES.ilog2())
        .map(|shift| 1u64 << shift)
        .collect();
    let barrier = std::sync::Barrier::new(ordinals.len());
    let broken = std::sync::atomic::AtomicBool::new(false);
    // Every rank starts each replay together, and all stop at the same one
    // once any has failed, so none is left waiting on a partner.
    let meet = |failed: bool| {
        if failed {
            broken.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        barrier.wait();
        broken.load(std::sync::atomic::Ordering::Relaxed)
    };
    let timed: Vec<Result<Samples>> = std::thread::scope(|scope| {
        let racers: Vec<_> = ordinals
            .iter()
            .zip(comms)
            .zip(&peers)
            .map(|((&ordinal, comm), &peers)| {
                let (sizes, meet) = (&sizes, &meet);
                scope.spawn(move || race(ordinal, comm, peers, sizes, meet))
            })
            .collect();
        racers
            .into_iter()
            .map(|racer| racer.join().unwrap_or(Err(Fault::Runtimeless)))
            .collect()
    });
    let mut ranks = Vec::with_capacity(timed.len());
    for rank in timed {
        match rank {
            Ok(rank) => ranks.push(rank),
            Err(fault) => {
                eprintln!("engine-cuda: peer collectives are off, NCCL carries them: {fault}");
                return None;
            }
        }
    }
    let costs = costs(&ranks);
    let [reduce, gather] = [0, 1].map(|op| {
        let pairs: Vec<[f64; 2]> = costs
            .iter()
            .map(|cost| [cost[2 * op], cost[2 * op + 1]])
            .collect();
        let reach = reach(&sizes, &pairs);
        let carried = match reach {
            None => "no messages".to_owned(),
            Some(u64::MAX) => "messages of every size".to_owned(),
            Some(bytes) => format!("messages up to {bytes} bytes"),
        };
        eprintln!(
            "engine-cuda: peer {} carry {carried}, NCCL the rest (peer/NCCL us from 16 KiB \
             up: {:?})",
            ["all-reduces", "all-gathers"][op],
            pairs
                .iter()
                .map(|[peer, nccl]| format!("{peer:.1}/{nccl:.1}"))
                .collect::<Vec<_>>()
        );
        reach.unwrap_or(0)
    });
    (reduce > 0 || gather > 0).then(|| {
        peers
            .into_iter()
            .map(|peers| kernels_cuda::collective::Peers {
                reach: reduce,
                gather_reach: gather,
                ..peers
            })
            .collect()
    })
}

/// The paths timed at each size: the all-reduce on the peers and on NCCL,
/// then the all-gather the same two ways.
const PATHS: usize = 4;

/// One rank's microseconds per collective on each path at each size, for each
/// replay.
type Samples = [Vec<[f64; PATHS]>; SAMPLES];

/// Each size's cost on each path: the median over replays of the slowest
/// rank's time, since a collective ends when its last rank does.
fn costs(ranks: &[Samples]) -> Vec<[f64; PATHS]> {
    let sizes = ranks.first().map_or(0, |rank| rank[0].len());
    (0..sizes)
        .map(|size| {
            std::array::from_fn(|path| {
                let mut slowest: [f64; SAMPLES] = std::array::from_fn(|replay| {
                    ranks
                        .iter()
                        .map(|rank| rank[replay][size][path])
                        .fold(0.0, f64::max)
                });
                slowest.sort_by(f64::total_cmp);
                slowest[SAMPLES / 2]
            })
        })
        .collect()
}

/// The reach that costs least over one message of every timed size: a near
/// tie at one size then cannot hand NCCL the sizes the peer path wins past it.
fn reach(sizes: &[u64], costs: &[[f64; 2]]) -> Option<u64> {
    let total = |n: usize| {
        costs[..n].iter().map(|[peer, _]| peer).sum::<f64>()
            + costs[n..].iter().map(|[_, nccl]| nccl).sum::<f64>()
    };
    match (0..=costs.len()).min_by(|&a, &b| total(a).total_cmp(&total(b)))? {
        0 => None,
        n if n == sizes.len() => Some(u64::MAX),
        n => Some(sizes[n - 1]),
    }
}

/// One rank's samples at each size. Every rank captures the same graphs and
/// replays them in the same order, so each collective finds its partners.
fn race(
    ordinal: i32,
    comm: &Comm,
    peers: kernels_cuda::collective::Peers,
    sizes: &[u64],
    meet: &(dyn Fn(bool) -> bool + Sync),
) -> Result<Samples> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        use crate::device::ctx::check;
        use crate::device::graph::{Event, Graph, GraphExec};

        // Collectives in one captured graph, replayed as one sample: the
        // forward's bodies are graphs, so replay time, not launch time, is
        // what a path costs.
        const ROUNDS: u32 = 16;

        // Gathers come in rows, as the logits do, each gathered row `ROW`
        // bytes.
        const ROW: u64 = 16 << 10;

        let world = u64::from(peers.world);
        let partner = || Fault::program("comm::calibrate", "a partner rank failed");
        let fire = |paths: &[kernels_cuda::Ctx; 2],
                    bufs: &[crate::device::Buffer; 2],
                    path: usize,
                    bytes: u64|
         -> Result<()> {
            let ctx = &paths[path % 2];
            let bf16 = |at: u64, rows: u64, width: u64| {
                kernels_cuda::Tensor::new(at, rows as u32, width as u32, model_ir::Dtype::Bf16)
            };
            let fired = if path < 2 {
                kernels_cuda::collective::all_reduce(ctx, &mut bf16(bufs[0].ptr(), 1, bytes / 2))
            } else {
                let (rows, width) = (bytes.div_ceil(ROW), ROW.min(bytes) / 2);
                kernels_cuda::collective::all_gather(
                    ctx,
                    bf16(bufs[0].ptr(), rows, width / world),
                    &mut bf16(bufs[1].ptr(), rows, width),
                )
            };
            Ok(fired.map_err(crate::error::kernel)?)
        };

        let mut stream: *mut c_void = core::ptr::null_mut();
        // Every size runs once eagerly, and every rank drains, before any
        // capture: binding the thread, building the kernels and NCCL's
        // connections are not things a capture may do.
        let warmed = (|| -> Result<_> {
            crate::device::ctx::bind_thread(ordinal)?;
            // SAFETY: a live out-parameter.
            check("cudaStreamCreate", unsafe {
                rt::cudaStreamCreateWithFlags((&raw mut stream).cast(), rt::cudaStreamNonBlocking)
            })?;
            let bufs = [
                crate::device::Buffer::zeroed(STAGE_BYTES as usize)?,
                crate::device::Buffer::zeroed(STAGE_BYTES as usize)?,
            ];
            // SAFETY: the stream is live until the end of this function, and
            // so are the communicator and the stages the group holds.
            let paths = unsafe {
                let nccl = kernels_cuda::Ctx::on(stream).with_comm(comm.raw());
                [
                    nccl.with_peers(peers),
                    kernels_cuda::Ctx::on(stream).with_comm(comm.raw()),
                ]
            };
            for &bytes in sizes {
                for path in 0..PATHS {
                    fire(&paths, &bufs, path, bytes)?;
                }
            }
            // SAFETY: the stream created above.
            check("cudaStreamSynchronize", unsafe {
                rt::cudaStreamSynchronize(stream.cast())
            })?;
            Ok((bufs, paths))
        })();
        let armed = match warmed {
            Err(fault) => {
                meet(true);
                Err(fault)
            }
            Ok(_) if meet(false) => Err(partner()),
            Ok((bufs, paths)) => (|| -> Result<_> {
                let mut graphs: Vec<Vec<GraphExec>> = Vec::with_capacity(sizes.len());
                for &bytes in sizes {
                    let captured = (0..PATHS).map(|path| {
                        Graph::capture(stream, || {
                            (0..ROUNDS).try_for_each(|_| fire(&paths, &bufs, path, bytes))
                        })?
                        .instantiate(stream)
                    });
                    graphs.push(captured.collect::<Result<_>>()?);
                }
                Ok((bufs, graphs, Event::timing()?, Event::timing()?))
            })(),
        };
        let timed = match armed {
            Err(fault) => {
                meet(true);
                Err(fault)
            }
            Ok((_bufs, graphs, start, end)) => {
                let mut out: Samples = std::array::from_fn(|_| vec![[0.0; PATHS]; sizes.len()]);
                let (mut fault, mut halted) = (None, false);
                'replays: for replay in &mut out {
                    for (size, pair) in graphs.iter().enumerate() {
                        for (path, graph) in pair.iter().enumerate() {
                            if meet(fault.is_some()) {
                                halted = true;
                                break 'replays;
                            }
                            let took = (|| -> Result<f64> {
                                start.record(stream)?;
                                graph.launch(stream)?;
                                end.record(stream)?;
                                end.settle()?;
                                Ok(f64::from(start.elapsed_ms(&end)?) * 1e3 / f64::from(ROUNDS))
                            })();
                            match took {
                                Ok(us) => replay[size][path] = us,
                                Err(failed) => fault = Some(failed),
                            }
                        }
                    }
                }
                match (fault, halted) {
                    (Some(fault), _) => Err(fault),
                    (None, true) => Err(partner()),
                    (None, false) => Ok(out),
                }
            }
        };
        if !stream.is_null() {
            // SAFETY: the stream created above, idle once `timed` is in.
            unsafe { rt::cudaStreamDestroy(stream.cast()) };
        }
        timed
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ordinal, comm, peers, sizes, meet);
        Err(Fault::Runtimeless)
    }
}

#[cfg(feature = "cuda")]
fn wire_peers(ordinals: &[i32]) -> Result<Option<Vec<kernels_cuda::collective::Peers>>> {
    use cudarc::runtime::sys as rt;

    use crate::device::ctx::check;

    let off = |why: String| Fault::program("comm::open_peers", why);
    let world = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    if !matches!(world, 2 | 4 | 8) {
        return Err(off(format!("no peer kernel for a group of {world}")));
    }
    for &a in ordinals {
        let mut vmm = 0;
        // SAFETY: a live out-parameter; a driver device is its ordinal.
        let asked = unsafe {
            cudarc::driver::sys::cuDeviceGetAttribute(
                &raw mut vmm,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                a,
            )
        };
        if asked != cudarc::driver::sys::CUresult::CUDA_SUCCESS || vmm == 0 {
            return Err(off(format!("cuda:{a} does not map memory for peers")));
        }
        for &b in ordinals.iter().filter(|&&b| b != a) {
            let mut reach = 0;
            let mut native = 0;
            // SAFETY: live out-parameters and two ordinals the runtime counts.
            unsafe {
                check(
                    "cudaDeviceCanAccessPeer",
                    rt::cudaDeviceCanAccessPeer(&raw mut reach, a, b),
                )?;
                check(
                    "cudaDeviceGetP2PAttribute",
                    rt::cudaDeviceGetP2PAttribute(
                        &raw mut native,
                        rt::cudaDeviceP2PAttr::cudaDevP2PAttrAccessSupported,
                        a,
                        b,
                    ),
                )?;
            }
            if reach == 0 || native == 0 {
                return Err(off(format!("cuda:{a} cannot access cuda:{b}")));
            }
        }
    }
    let table = 2 * size_of::<[u64; 8]>();
    let mut stages = [0u64; 8];
    let mut signals = [0u64; 8];
    let mut tables = Vec::with_capacity(ordinals.len());
    for (rank, &a) in ordinals.iter().enumerate() {
        // SAFETY: each call takes live out-parameters or pointers this loop
        // just allocated on device `a`.
        unsafe {
            check("cudaSetDevice", rt::cudaSetDevice(a))?;
            let signal_bytes = kernels_cuda::collective::SIGNAL_BYTES as usize + table;
            let stage =
                crate::device::elastic::shared(a, ordinals, STAGE_BYTES + signal_bytes as u64)?;
            let signal = stage + STAGE_BYTES;
            check(
                "cudaMemset",
                rt::cudaMemset(signal as *mut c_void, 0, signal_bytes),
            )?;
            stages[rank] = stage;
            signals[rank] = signal;
            tables.push(signal + kernels_cuda::collective::SIGNAL_BYTES);
        }
    }
    let mut bytes = Vec::with_capacity(table);
    for word in stages.iter().chain(&signals) {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut peers = Vec::with_capacity(ordinals.len());
    for (rank, &a) in ordinals.iter().enumerate() {
        // SAFETY: the table region is `table` bytes past this rank's signal,
        // and the stage is at least a word.
        unsafe {
            check("cudaSetDevice", rt::cudaSetDevice(a))?;
            check(
                "cudaMemcpy",
                rt::cudaMemcpy(
                    tables[rank] as *mut c_void,
                    bytes.as_ptr().cast(),
                    table,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                ),
            )?;
            check(
                "cudaMemset",
                rt::cudaMemset(stages[rank] as *mut c_void, 0x40 + rank as i32, 4),
            )?;
        }
        peers.push(kernels_cuda::collective::Peers {
            rank: rank as u32,
            world,
            stage: stages[rank],
            stage_bytes: STAGE_BYTES,
            signal: signals[rank],
            stages: tables[rank],
            signals: tables[rank] + size_of::<[u64; 8]>() as u64,
            reach: u64::MAX,
            gather_reach: u64::MAX,
        });
    }
    // What the driver advertises is not always what the link carries (#757).
    // Every rank posts to and reads from every peer with the stores and loads
    // the collectives use, but in a kernel that never waits: the first round
    // posts, and the second, after every device has drained, finds what
    // arrived.
    let mut statuses = Vec::with_capacity(ordinals.len());
    for (&a, _) in ordinals.iter().zip(&peers) {
        crate::device::ctx::bind_thread(a)?;
        statuses.push(crate::device::Buffer::zeroed(4)?);
    }
    for _ in 0..2 {
        for ((&a, peers), status) in ordinals.iter().zip(&peers).zip(&statuses) {
            crate::device::ctx::bind_thread(a)?;
            // SAFETY: the device's legacy stream; the stages and tables are live.
            let ctx = unsafe { kernels_cuda::Ctx::on(core::ptr::null_mut()) };
            peers
                .answer(&ctx, status.ptr())
                .map_err(crate::error::kernel)?;
        }
        for &a in ordinals {
            crate::device::ctx::bind_thread(a)?;
            // SAFETY: no arguments; waits for the rounds' kernels.
            check("cudaDeviceSynchronize", unsafe {
                rt::cudaDeviceSynchronize()
            })?;
        }
    }
    for (&a, status) in ordinals.iter().zip(&statuses) {
        crate::device::ctx::bind_thread(a)?;
        let mut word = [0u8; 4];
        status.read(0, &mut word)?;
        match u32::from_le_bytes(word) {
            0 => {}
            1 => {
                return Err(off(format!(
                    "cuda:{a} heard nothing from a peer through the mapping"
                )));
            }
            _ => {
                return Err(off(format!(
                    "cuda:{a} read the wrong word from a peer's stage"
                )));
            }
        }
    }
    Ok(Some(peers))
}

#[cfg(feature = "cuda")]
fn transport_defaults(transport: Transport) {
    let disable_p2p = match transport {
        Transport::Shm => "1",
        Transport::Peer => "0",
        Transport::Nccl => return,
    };
    // SAFETY: called before any communicator exists, from the group opener,
    // on the thread that starts the rank threads.
    unsafe { std::env::set_var("NCCL_P2P_DISABLE", disable_p2p) };
}

#[cfg(feature = "cuda")]
fn answered(call: &'static str, code: cudarc::nccl::sys::ncclResult_t) -> Result<()> {
    if code == cudarc::nccl::sys::ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn one_slow_replay_or_near_tie_does_not_set_the_reach() {
        let sizes: Vec<u64> = (14..=24).map(|shift| 1u64 << shift).collect();
        // A PRO 6000 x2 boot's costs, which the first-loss rule sent to NCCL
        // whole on the 16 KiB near tie.
        let costs = [
            [11.4, 11.2],
            [60.0, 13.0],
            [12.0, 14.0],
            [48.0, 19.0],
            [19.0, 56.0],
            [26.0, 66.0],
            [40.0, 60.0],
            [88.0, 103.0],
            [134.0, 163.0],
            [247.0, 284.0],
            [490.0, 545.0],
        ];
        assert_eq!(super::reach(&sizes, &costs), Some(u64::MAX));
        let mut rank: super::Samples = std::array::from_fn(|_| vec![[5.0, 9.0, 6.0, 4.0]]);
        rank[3][0][0] = 60.0;
        assert_eq!(
            super::costs(&[rank.clone(), rank]),
            vec![[5.0, 9.0, 6.0, 4.0]]
        );
    }
}
