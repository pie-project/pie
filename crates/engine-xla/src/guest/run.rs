//! Running lowered eta stages on the device.
//!
//! [`OnDevice`] is an `eta_exec::StageRunner`: `eta_exec::step_with` drives
//! one instance's pass and hands each stage here, which packs the stage's
//! roots into one upload, runs the stage's executable (the outputs read
//! straight out of the fire's device planes) and unpacks the one download
//! of what the stage puts.
//!
//! [`step_group`] is the same pass for several instances of one program at
//! once: `step_with`'s control flow (readiness, the per-pass overlay, the
//! commit) per instance, and every stage one executable over the lanes.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, InterpInstance, StagePlan, StageRunner, StepOutcome, Value};
use eta_ir::Dtype;
use eta_ir::registry::Stage;
use eta_ir::validate::Direction;

use super::lower::{self, Batch, Feed, Layout, Read, Slot};
use crate::device::{Admission, Device, Program, available_memory, compile_need, hold_bytes};
use crate::pjrt::{Arg, Buffer};
use crate::readout::{Kept, Seat};

/// Where a pass reads output `name` of its lane `lane`: plane `plane` of
/// the fire's, at `seat`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bound {
    pub name: String,
    pub lane: u32,
    pub plane: usize,
    pub seat: Seat,
}

/// Where an instance attached at the fire's lanes `lane..lane + lanes`
/// reads every output `plan` reads, when each binds on the device. Its
/// seats are its own lanes' rows: a program reading a lane past the ones
/// it is attached over is refused by name, and no seat runs into the
/// next attachment's lanes.
pub fn bound(
    plan: &ExecPlan,
    kept: Option<&Kept>,
    lane: u32,
    lanes: u32,
) -> Result<Option<Vec<Bound>>, String> {
    let outputs = plan.outputs();
    if let Some((name, k)) = outputs.iter().find(|(_, k)| *k >= lanes) {
        return Err(format!(
            "its program reads output `{name}` of its lane {k}, and it is attached over \
             {lanes} lane(s)"
        ));
    }
    let end = (lane + lanes) as usize;
    let bound: Option<Vec<Bound>> = (outputs.into_iter())
        .map(|(name, k)| {
            let (plane, seat) = kept?.bound(lane as usize + k as usize, end, name)?;
            Some(Bound {
                name: name.to_string(),
                lane: k,
                plane,
                seat,
            })
        })
        .collect();
    let Some(bound) = bound else {
        return Ok(None);
    };
    let binds = plan
        .package
        .values
        .iter()
        .all(|v| v.source != ValueOrigin::Output || device_binds(v, kept, &bound));
    Ok(binds.then_some(bound))
}

/// A lowered stage, compiled; `program` is `None` for a stage that puts
/// nothing (there is nothing to run).
pub struct Staged {
    pub program: Option<Arc<Program>>,
    pub layout: Layout,
    /// The device's cache key of `program`'s text: what an eviction drops
    /// there too, so the compiled form is freed and not only unlisted.
    pub text_key: Option<[u8; 32]>,
    /// The host memory the compiled program holds ([`hold_bytes`]) and what
    /// compiling it took ([`compile_need`]): what the stage cache charges
    /// and yields against the host's room.
    pub held: u64,
    pub need: u64,
}

/// Compiled stages by (program content, stage, batch shape): a pass that
/// was seen skips the lowering and the compile both.
/// A compiled stage's key: program content, stage, batch shape, carried cells.
type StageKey = ([u8; 32], usize, Batch, Vec<u32>);

#[derive(Default)]
pub struct Stages {
    map: HashMap<StageKey, Arc<Staged>>,
    /// When each program's stages last ran or were installed, by program
    /// content: the order an eviction takes them in ([`Stages::yield_room`]).
    last_used: HashMap<[u8; 32], u64>,
    clock: u64,
    /// Compiled forms an eviction dropped from the map, whose device cache
    /// entries the engine drops next ([`Stages::take_evicted`]).
    evicted: Vec<[u8; 32]>,
    /// Stage executions and lanes they carried, for the tally.
    pub runs: u64,
    pub lanes: u64,
    /// The most lanes a fire seats: a stage first seen compiles for every
    /// lane rung up to it, side by side.
    pub top: u32,
    /// The readout rows of the fire shapes the warm ladder traced
    /// (`Shell::readouts`).
    pub readouts: Vec<u32>,
    /// The pool's rows (u32 `[pool_rows, 128]`), which a stage that carries
    /// cells takes, and the sink's first row, where a padding lane's cells
    /// go.
    pub pool_rows: i64,
    pub sink_row: u32,
    /// Every plane a fire of the load can return, as a stage reads it
    /// (`Shell::guest_planes`): output, rows, width, layers, readout flag,
    /// dtype. A registration's outputs are checked against these, and each
    /// stage compiled for them.
    pub planes: Vec<(String, u32, u32, u32, bool, Dtype, bool)>,
    /// Stages a fire compiled (`Stages::get` past `compile_all`): a defect.
    pub fire_compiles: u64,
}

/// A rung's stage as a compile thread leaves it.
type Built = std::sync::Mutex<Option<Result<Arc<Staged>, String>>>;

/// A stage shape a pass can take: the stage, its batch, its carried
/// channels.
pub type Want = (usize, Batch, Vec<u32>);

/// The stages `wanted` names, compiled side by side on any thread (a
/// registration's, off the engine's): in `wanted`'s order. The first
/// refusal refuses them all.
pub fn compile_wanted(
    device: &Device,
    package: &LaunchPackage,
    wanted: &[Want],
    pool_rows: i64,
) -> Result<Vec<Arc<Staged>>, String> {
    // Compiles admitted into the host memory free now (held programs
    // already take theirs), as the warm ladder's are.
    let admission = Admission::new(available_memory().unwrap_or(u64::MAX));
    let started = std::time::Instant::now();
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, wanted.len().clamp(1, 8));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let built: Vec<Built> = wanted.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((at, batch, c)) = wanted.get(i) else {
                        break;
                    };
                    let staged =
                        stage(device, package, *at, batch.clone(), c, pool_rows, Some(&admission));
                    if let Ok(mut slot) = built[i].lock() {
                        *slot = Some(staged);
                    }
                }
            });
        }
    });
    let staged = (built.into_iter())
        .map(|built| {
            built
                .into_inner()
                .map_err(|_| "a stage compile panicked".to_string())?
                .ok_or("a stage was never compiled")?
        })
        .collect::<Result<Vec<_>, String>>()?;
    tracing::info!(
        programs = wanted.len(),
        seconds = started.elapsed().as_secs_f32(),
        need_mib = guest_compile_need() >> 20,
        "guest stages compiled"
    );
    Ok(staged)
}

/// The process's ledger of guest stage compiles, every registration's:
/// the host memory the running ones may take (each at the learned need),
/// how many run, and the need learned, the largest peak a stage compile
/// running alone grew the process by. A guest stage's compile takes host
/// memory its text does not tell (v6e: 0.1-1 GB a 45 KB sampler stage), so
/// the need is measured, not read off the text; concurrent registrations
/// each sizing their own room from the free memory OOM-killed a server at
/// its cgroup cap.
struct GuestLedger {
    state: std::sync::Mutex<(u64, u32, u64)>,
    ended: std::sync::Condvar,
}

static GUEST_LEDGER: std::sync::OnceLock<GuestLedger> = std::sync::OnceLock::new();

/// A guest stage compile the ledger let in. Calibrating (none learned
/// yet): it runs alone and its peak growth becomes the need.
struct GuestAdmitted {
    need: u64,
    calibrating: Option<u64>,
}

fn guest_ledger() -> &'static GuestLedger {
    GUEST_LEDGER.get_or_init(|| GuestLedger {
        state: std::sync::Mutex::new((0, 0, 0)),
        ended: std::sync::Condvar::new(),
    })
}

/// The process's peak RSS (`VmHWM`), bytes.
fn peak_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib << 10)
}

/// Waits until a guest stage compile fits; refuses by name one that does
/// not fit even alone.
fn guest_admit() -> Result<GuestAdmitted, String> {
    let ledger = guest_ledger();
    let mut state = ledger
        .state
        .lock()
        .map_err(|_| "the guest compile ledger poisoned")?;
    loop {
        let (running, count, learned) = *state;
        if learned == 0 {
            if count > 0 {
                state = ledger
                    .ended
                    .wait(state)
                    .map_err(|_| "the guest compile ledger poisoned")?;
                continue;
            }
            // the first compile runs alone; its peak growth is the need
            // (the peak reset to the resident set first)
            let _ = std::fs::write("/proc/self/clear_refs", "5");
            *state = (running, 1, learned);
            return Ok(GuestAdmitted {
                need: 0,
                calibrating: peak_rss(),
            });
        }
        let room = available_memory().unwrap_or(u64::MAX);
        if count == 0 && learned > room {
            return Err(format!(
                "a guest stage compile takes ~{} MiB of host memory and {} MiB is free under the \
                 host's and its cgroups' caps: the program is refused, not compiled",
                learned >> 20,
                room >> 20
            ));
        }
        if count == 0 || running + learned <= room {
            *state = (running + learned, count + 1, learned);
            return Ok(GuestAdmitted {
                need: learned,
                calibrating: None,
            });
        }
        state = ledger
            .ended
            .wait(state)
            .map_err(|_| "the guest compile ledger poisoned")?;
    }
}

impl Drop for GuestAdmitted {
    fn drop(&mut self) {
        let ledger = guest_ledger();
        if let Ok(mut state) = ledger.state.lock() {
            let (running, count, learned) = *state;
            let learned = match (self.calibrating, peak_rss()) {
                (Some(before), Some(after)) => learned.max(after.saturating_sub(before).max(1)),
                _ => learned,
            };
            *state = (
                running.saturating_sub(self.need),
                count.saturating_sub(1),
                learned,
            );
        }
        ledger.ended.notify_all();
    }
}

/// The need the guest ledger has learned, bytes (0: none yet).
pub fn guest_compile_need() -> u64 {
    guest_ledger().state.lock().map_or(0, |s| s.2)
}

/// Stage `at` of `package` lowered for `batch` and compiled.
fn stage(
    device: &Device,
    package: &LaunchPackage,
    at: usize,
    batch: Batch,
    carried: &[u32],
    pool_rows: i64,
    registering: Option<&Admission>,
) -> Result<Arc<Staged>, String> {
    let lowered =
        lower::lower(package, at, batch, carried, pool_rows).map_err(|why| why.to_string())?;
    if !lowered.text.is_empty()
        && let Some(path) = std::env::var_os("PIE_XLA_DUMP_GUEST")
    {
        let _ = std::fs::write(path, &lowered.text);
    }
    let (text_key, held, need) = if lowered.layout.runs() {
        (
            Some(*blake3::hash(lowered.text.as_bytes()).as_bytes()),
            hold_bytes(&lowered.text),
            compile_need(&lowered.text),
        )
    } else {
        (None, 0, 0)
    };
    let program = if !lowered.layout.runs() {
        None
    } else {
        let sig = crate::trace::Signature {
            params: Vec::new(),
            results: Vec::new(),
        };
        let program = match registering {
            Some(admission) => {
                let _admitted = admission.admit(&lowered.text);
                // A compile (not a held or cached program) is admitted on
                // the process's guest ledger: one learned need a stage,
                // against the memory the host and its cgroups leave.
                let _guest = if device.has_without_compile(&lowered.text) {
                    None
                } else {
                    Some(guest_admit()?)
                };
                device.prepare(&lowered.text, sig)
            }
            None => device.program(&lowered.text, sig),
        };
        Some(program.map_err(|e| format!("stage {at} did not compile: {e}"))?)
    };
    Ok(Arc::new(Staged {
        program,
        layout: lowered.layout,
        text_key,
        held,
        need,
    }))
}

impl Stages {
    /// Stage `at` of `package` for `batch`, as registration compiled it
    /// (`Stages::compile_all`). A shape registration did not reach
    /// compiles here, inside the fire, counted (`fire_compiles`) and, where
    /// the load sealed its ladder, a miss (WARM_STRICT refuses it).
    pub fn get(
        &mut self,
        device: &Device,
        package: &LaunchPackage,
        key: [u8; 32],
        at: usize,
        batch: Batch,
        carried: &[u32],
    ) -> Result<Arc<Staged>, String> {
        let slot = (key, at, batch.clone(), carried.to_vec());
        self.clock += 1;
        self.last_used.insert(key, self.clock);
        if let Some(hit) = self.map.get(&slot) {
            return Ok(Arc::clone(hit));
        }
        // Registration compiled every shape a fire of the load takes: one
        // compiled here waits a fire on it.
        self.fire_compiles += 1;
        tracing::warn!(
            stage = at,
            lanes = batch.lanes,
            compiles = self.fire_compiles,
            "a guest stage compiled inside a fire, past its program's registration"
        );
        let staged = stage(device, package, at, batch, carried, self.pool_rows, None)?;
        self.map.insert(slot, Arc::clone(&staged));
        self.yield_room(key, available_memory());
        Ok(staged)
    }

    /// Compiles every stage of `package` for every shape a pass can take
    /// (`Stages::wanted`), in place: a registration whole, on the calling
    /// thread. The first refusal refuses the program.
    pub fn compile_all(
        &mut self,
        device: &Device,
        package: &LaunchPackage,
        key: [u8; 32],
        carried: &[u32],
    ) -> Result<(), String> {
        let wanted = self.wanted(package, key, carried)?;
        let staged = compile_wanted(device, package, &wanted, self.pool_rows)?;
        self.install(key, wanted, staged);
        Ok(())
    }

    /// Every shape a pass of `package` can take that no stage compiled yet
    /// serves: each lane rung up to the widest fire (`top`), each
    /// combination of the planes its outputs can be read off (the load's,
    /// `planes`), with the `carried` channels in the pool; and one lane
    /// with none carried (an instance that steps alone). A stage reading an
    /// output no fire returns refuses the program.
    pub fn wanted(
        &self,
        package: &LaunchPackage,
        key: [u8; 32],
        carried: &[u32],
    ) -> Result<Vec<Want>, String> {
        let top = padded(self.top as usize);
        let rungs: Vec<u32> =
            std::iter::successors(Some(1u32), |&r| (r < top).then(|| r.saturating_mul(2)))
                .collect();
        let mut wanted: Vec<(usize, Batch, Vec<u32>)> = Vec::new();
        for at in 0..package.stages.len() {
            let mut combos: Vec<Vec<Read>> = vec![Vec::new()];
            for (name, lane) in lower::outputs(package, at) {
                // Every output a stage reads is a plane a fire of this load
                // returns on the device, of the dtype it reads: else the
                // program is refused here, at its registration, and a pass
                // never finds an output unbound.
                let wants: Vec<Dtype> = (package.values.iter())
                    .filter(|v| {
                        v.source == ValueOrigin::Output && v.output.as_deref() == Some(&name)
                    })
                    .map(|v| v.dtype)
                    .collect();
                if !self
                    .planes
                    .iter()
                    .any(|p| p.0 == name && wants.iter().all(|&d| d == p.5))
                {
                    return Err(format!(
                        "stage {at} reads output `{name}` ({wants:?}), which no fire of this load \
                         returns on the device as that"
                    ));
                }
                let options: Vec<Read> = (self.planes.iter())
                    .filter(|p| p.0 == name)
                    .map(|p| Read {
                        name: name.clone(),
                        lane,
                        rows: p.1,
                        readout: p.4,
                        layers: p.3,
                        width: p.2,
                        dtype: p.5,
                        normal: p.6,
                    })
                    .collect();
                combos = combos
                    .into_iter()
                    .flat_map(|c| {
                        options.iter().map(move |o| {
                            let mut c = c.clone();
                            c.push(o.clone());
                            c
                        })
                    })
                    .collect();
            }
            for reads in &combos {
                // A fire of the load's top lanes carries at most `top /
                // rows` guest lanes reading `rows` rows each, padded to its
                // rung: no batch past that is compiled (a beam group of 17
                // read 64 × 17 rows, a stage of more temp than the device).
                let rows = lower::lane_rows(package, at, reads);
                let carried_most = padded((top as usize / rows.max(1) as usize).max(1));
                for &lanes in rungs.iter().filter(|&&lanes| lanes <= carried_most) {
                    wanted.push((at, Batch { lanes, reads: reads.clone() }, carried.to_vec()));
                }
                wanted.push((at, Batch { lanes: 1, reads: reads.clone() }, Vec::new()));
            }
        }
        let mut seen = std::collections::HashSet::new();
        wanted.retain(|(at, batch, c)| {
            let slot = (key, *at, batch.clone(), c.clone());
            !self.map.contains_key(&slot) && seen.insert(slot)
        });
        Ok(wanted)
    }

    /// The pool's rows, which a stage that carries cells takes.
    #[must_use]
    pub fn pool_rows(&self) -> i64 {
        self.pool_rows
    }

    /// Stages `compile_wanted` made for `wanted`, held from now on.
    pub fn install(&mut self, key: [u8; 32], wanted: Vec<Want>, staged: Vec<Arc<Staged>>) {
        let count = wanted.len();
        for ((at, batch, c), staged) in wanted.into_iter().zip(staged) {
            self.map.insert((key, at, batch, c), staged);
        }
        self.clock += 1;
        self.last_used.insert(key, self.clock);
        tracing::info!(programs = count, "guest stages compiled at registration");
        self.yield_room(key, available_memory());
    }

    /// The stage cache's eviction rule, tied to the host-memory ledger its
    /// compiles are admitted against ([`Admission`] over `available`, the
    /// host's `MemAvailable`): it holds compiled stages only while the host
    /// keeps room for the largest compile it has run, so a registration's
    /// compile is never starved by the stages of programs registered
    /// before it. Past that, the stages of the program run least recently
    /// (never `keep`'s, just installed or run) go first, each freeing the
    /// host bytes it held ([`Staged::held`]), until the room is back or
    /// nothing else is held. A program evicted and run again compiles in
    /// its fire, counted (`fire_compiles`). `None` (a host with no reading)
    /// admits every compile, as [`Admission`] does, and evicts nothing.
    /// Returns the bytes freed.
    pub fn yield_room(&mut self, keep: [u8; 32], available: Option<u64>) -> u64 {
        let Some(available) = available else {
            return 0;
        };
        let need = self.map.values().map(|staged| staged.need).max().unwrap_or(0);
        let mut freed = 0u64;
        while available.saturating_add(freed) < need {
            let Some(victim) = self
                .last_used
                .iter()
                .filter(|(key, _)| **key != keep)
                .min_by_key(|&(_, &at)| at)
                .map(|(key, _)| *key)
            else {
                break;
            };
            self.last_used.remove(&victim);
            let slots: Vec<StageKey> = self
                .map
                .keys()
                .filter(|slot| slot.0 == victim)
                .cloned()
                .collect();
            for slot in slots {
                if let Some(staged) = self.map.remove(&slot) {
                    freed = freed.saturating_add(staged.held);
                    self.evicted.extend(staged.text_key);
                }
            }
            tracing::info!(
                freed,
                available,
                need,
                "guest stages of the program run least recently evicted: the host's \
                 room for the next compile"
            );
        }
        freed
    }

    /// The device cache keys of the compiled forms evicted since the last
    /// call: the engine drops them from the device's cache
    /// ([`Device::forget_key`]) so their memory is freed.
    pub fn take_evicted(&mut self) -> Vec<[u8; 32]> {
        std::mem::take(&mut self.evicted)
    }

    /// What a pass of the program `key` names takes on the device beyond
    /// the load's own (`Engine::fire_facts`), from its stages as
    /// registration compiled them: at each lane rung, the largest temp past
    /// `load_temp` (the runtime's one temp reservation is the largest loaded
    /// program's) and the outputs its stages return (not the pool they
    /// alias). `fixed + lanes · a_lane` covers every rung, up to `top`.
    /// `carried` names the group variant (the program's carried channels).
    #[must_use]
    pub fn fire_facts(
        &self,
        key: [u8; 32],
        carried: &[u32],
        load_temp: u64,
    ) -> engine::program::FireFacts {
        let mut by_lanes: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
        // A group's passes (the carried variants): what a fire holds.
        for ((k, _, batch, c), staged) in &self.map {
            if *k != key || c.as_slice() != carried {
                continue;
            }
            let Some(program) = &staged.program else {
                continue;
            };
            let m = program.memory;
            let entry = by_lanes.entry(batch.lanes).or_default();
            entry.0 = entry.0.max(m.temp.saturating_sub(load_temp));
            entry.1 += m.output.saturating_sub(m.alias);
        }
        let bytes = |lanes: u32| by_lanes.get(&lanes).map_or(0, |&(t, o)| t + o);
        let one = bytes(1);
        let a_lane = by_lanes
            .keys()
            .filter(|&&r| r > 1)
            .map(|&r| bytes(r).saturating_sub(one).div_ceil(u64::from(r - 1)))
            .max()
            .unwrap_or(0);
        engine::program::FireFacts {
            a_lane,
            fixed: one.saturating_sub(a_lane),
            max_lanes: by_lanes.keys().copied().max().unwrap_or(0),
            laid: engine::program::LaidLanes::PowerOfTwo,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// The batch shape stage `at` runs at for `lanes` lanes against `kept`,
/// its outputs read where `bound` says.
#[must_use]
pub fn batch_of(
    package: &LaunchPackage,
    at: usize,
    lanes: u32,
    kept: Option<&Kept>,
    bound: &[Bound],
) -> Batch {
    let reads = kept.map_or_else(Vec::new, |kept| {
        (lower::outputs(package, at).into_iter())
            .filter_map(|(name, lane)| {
                let b = bound.iter().find(|b| b.name == name && b.lane == lane)?;
                let plane = &kept.planes[b.plane];
                Some(Read {
                    name,
                    lane,
                    rows: plane.rows,
                    readout: plane.readout,
                    layers: plane.layers,
                    width: plane.width,
                    dtype: plane.dtype,
                    normal: plane.normal,
                })
            })
            .collect()
    });
    Batch { lanes, reads }
}

/// Lanes padded to a power of two, so a group's size changes the
/// executable only at powers of two.
#[must_use]
pub fn padded(lanes: usize) -> u32 {
    u32::try_from(lanes.max(1).next_power_of_two()).unwrap_or(u32::MAX)
}

/// One lane's feed for a stage: its values (by package-global id; only
/// the stage's host-fed roots are read) and where its outputs sit.
pub struct Feeding<'a> {
    pub vals: &'a [Value],
    pub bound: &'a [Bound],
    /// The pool row each carried channel's cell starts at, for this lane
    /// (its `Channel` slots).
    pub cells: &'a [(u32, u32)],
}

fn encode(value: &Value, dtype: Dtype, n: usize, out: &mut Vec<u32>) -> Result<(), String> {
    if value.len() != n || value.dtype() != dtype {
        return Err(format!(
            "a root holds {} {:?} lane(s) where the stage reads {n} {dtype:?}",
            value.len(),
            value.dtype()
        ));
    }
    match value {
        Value::F32(v) => out.extend(v.iter().map(|x| x.to_bits())),
        Value::I32(v) => out.extend(v.iter().map(|&x| x as u32)),
        Value::U32(v) => out.extend_from_slice(v),
        Value::Bool(v) => {
            let base = out.len();
            out.resize(base + n.div_ceil(32), 0);
            for (j, &b) in v.iter().enumerate() {
                if b {
                    out[base + j / 32] |= 1 << (j % 32);
                }
            }
        }
        // two to a word, element `2w` in the low half (`eta_ir::types`)
        Value::Bf16(v) => out.extend(
            v.chunks(2)
                .map(|pair| u32::from(pair[0]) | pair.get(1).map_or(0, |&h| u32::from(h) << 16)),
        ),
    }
    Ok(())
}

fn decode(words: &[u32], dtype: Dtype, n: usize) -> Value {
    match dtype {
        Dtype::F32 => Value::F32(words[..n].iter().map(|&w| f32::from_bits(w)).collect()),
        Dtype::I32 => Value::I32(words[..n].iter().map(|&w| w as i32).collect()),
        Dtype::Bf16 => Value::Bf16(
            (0..n)
                .map(|j| (words[j / 2] >> (16 * (j % 2))) as u16)
                .collect(),
        ),
        Dtype::Bool => Value::Bool(
            (0..n)
                .map(|j| (words[j / 32] >> (j % 32)) & 1 != 0)
                .collect(),
        ),
        _ => Value::U32(words[..n].to_vec()),
    }
}

/// What one stage run answers: per lane, the values it puts (host side).
pub type Ran = Vec<Vec<(u32, Value)>>;

fn upload_words(
    device: &Device,
    words: &[u32],
    rows: usize,
    width: usize,
) -> Result<Buffer, String> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    device
        .upload(Dtype::U32, rows as u32, width as u32, &bytes)
        .map_err(|e| e.to_string())
}

/// A stage launched on the device and not read back: its packed output
/// (`u32 [batch, out_words]`, `None` when it puts nothing to the host) and
/// the carried cells it returned.
pub struct Launched {
    pub out: Option<Arc<Buffer>>,
    pub batch: usize,
    pub lanes: usize,
    /// Host words packed and uploaded, for the timing line.
    pub packed: usize,
}

/// Packs `lanes`' roots (padded to the staged batch by repeating lane 0),
/// uploads them and enqueues the stage; nothing waits for the device. A
/// stage that carries cells reads and writes them in `pool` at each lane's
/// rows (a padding lane's at `sink_row`), the pool donated and put back.
pub fn launch_stage(
    device: &Device,
    staged: &Staged,
    kept: Option<&Kept>,
    lanes: &[Feeding<'_>],
    pool: &mut Option<Buffer>,
    sink_row: u32,
) -> Result<Launched, String> {
    let batch = padded(lanes.len()) as usize;
    let Some(program) = &staged.program else {
        return Ok(Launched {
            out: None,
            batch,
            lanes: lanes.len(),
            packed: 0,
        });
    };
    let layout = &staged.layout;
    let mut words: Vec<u32> = Vec::with_capacity(batch * layout.in_words);
    for b in 0..batch {
        let lane = &lanes[if b < lanes.len() { b } else { 0 }];
        for slot in &layout.inputs {
            match slot.feed {
                Feed::Value(dtype, n) => {
                    encode(&lane.vals[slot.value as usize], dtype, n, &mut words)?;
                }
                Feed::Row { read, rows } => {
                    let read = &layout.reads[read];
                    let b = (lane.bound.iter())
                        .find(|b| b.name == read.name && b.lane == read.lane)
                        .ok_or_else(|| {
                            format!(
                                "this pass reads output `{}` and its lane binds none",
                                read.name
                            )
                        })?;
                    if rows > b.seat.count {
                        return Err(format!(
                            "this pass reads {rows} rows of output `{}` and its lane holds {}",
                            read.name, b.seat.count
                        ));
                    }
                    words.push(b.seat.first);
                }
            }
        }
        for (chan, _) in &layout.carried {
            let row = if b < lanes.len() {
                (lane.cells.iter())
                    .find(|(c, _)| c == chan)
                    .map(|&(_, row)| row)
                    .ok_or_else(|| format!("this lane holds no slots for carried channel {chan}"))?
            } else {
                sink_row
            };
            words.push(row);
        }
    }
    let pack: Option<Buffer> = if layout.in_words > 0 {
        Some(upload_words(device, &words, batch, layout.in_words)?)
    } else {
        None
    };
    let mut args: Vec<Arg<'_>> = Vec::with_capacity(3);
    if let Some(pack) = &pack {
        args.push(Arg::Keep(pack));
    }
    for read in &layout.reads {
        let kept = kept.ok_or("this stage reads an output and the fire kept none")?;
        let plane = (lanes[0].bound.iter())
            .find(|b| b.name == read.name && b.lane == read.lane)
            .ok_or("this stage reads an output its lanes bind none of")?
            .plane;
        args.push(Arg::Keep(&kept.planes[plane].buffer));
    }
    if layout.pooled() {
        let whole = pool.take().ok_or("the pool, which another program holds")?;
        args.push(Arg::Donate(whole));
    }
    let outs = device
        .run(program, args, crate::serve::timing())
        .map_err(|e| e.to_string())?;
    let mut outs = outs.into_iter();
    let out = if layout.out_words > 0 {
        Some(Arc::new(outs.next().ok_or("the stage returned nothing")?))
    } else {
        None
    };
    if layout.pooled() {
        *pool = Some(outs.next().ok_or("the stage returned no pool")?);
    }
    Ok(Launched {
        out,
        batch,
        lanes: lanes.len(),
        packed: words.len(),
    })
}

/// The output words of a stage's packed download.
#[must_use]
pub fn words_of(raw: &[u8]) -> Vec<u32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Lane `row`'s value in `slot` of a stage's output words.
pub fn cell_of(all: &[u32], out_words: usize, row: usize, slot: &Slot) -> Result<Value, String> {
    let Feed::Value(dtype, n) = slot.feed else {
        return Err("a stage output slot holds no value".to_owned());
    };
    let from = row * out_words + slot.at;
    let words = all
        .get(from..from + slot.words)
        .ok_or_else(|| format!("output row {row} is past the {} words returned", all.len()))?;
    Ok(decode(words, dtype, n))
}

/// Every lane's put values out of a stage's download.
pub fn unpack(
    layout: &Layout,
    raw: &[u8],
    lanes: usize,
    batch: usize,
) -> Result<Vec<Vec<(u32, Value)>>, String> {
    let all = words_of(raw);
    if all.len() != batch * layout.out_words {
        return Err(format!(
            "the stage returned {} words for {batch} lane(s) of {}",
            all.len(),
            layout.out_words
        ));
    }
    (0..lanes)
        .map(|b| {
            layout
                .outputs
                .iter()
                .map(|slot| Ok((slot.value, cell_of(&all, layout.out_words, b, slot)?)))
                .collect()
        })
        .collect()
}

/// Runs one staged stage over `lanes` and reads its outputs back
/// (`launch_stage`, then the download).
pub fn run_stage(
    device: &Device,
    staged: &Staged,
    kept: Option<&Kept>,
    lanes: &[Feeding<'_>],
    pool: &mut Option<Buffer>,
    sink_row: u32,
) -> Result<Ran, String> {
    let t0 = std::time::Instant::now();
    let launched = launch_stage(device, staged, kept, lanes, pool, sink_row)?;
    let t1 = std::time::Instant::now();
    let raw = match &launched.out {
        Some(out) => out.download().map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    if crate::serve::timing() && staged.program.is_some() {
        eprintln!(
            "  xla guest stage: {} lane(s) in {}: pack+upload+run {:.2}ms ({} KiB) download {:.2}ms ({} KiB)",
            lanes.len(),
            launched.batch,
            (t1 - t0).as_secs_f64() * 1e3,
            launched.packed >> 8,
            t1.elapsed().as_secs_f64() * 1e3,
            raw.len() >> 10
        );
    }
    if staged.program.is_none() {
        return Ok(vec![Vec::new(); lanes.len()]);
    }
    unpack(&staged.layout, &raw, lanes.len(), launched.batch)
}

/// Whether the device binds the output `root` reads where `bound` says: a
/// plane of `kept` landed as the dtype the program reads.
fn device_binds(root: &LaunchValue, kept: Option<&Kept>, bound: &[Bound]) -> bool {
    let (Some(kept), Some(name)) = (kept, root.output.as_deref()) else {
        return false;
    };
    (bound.iter())
        .find(|b| b.name == name && b.lane == root.lane)
        .is_some_and(|b| kept.planes[b.plane].dtype == root.dtype)
}

/// One instance's stages on the device, as `step_with`'s runner.
pub struct OnDevice<'a> {
    device: &'a Device,
    stages: &'a mut Stages,
    key: [u8; 32],
    kept: Option<&'a Kept>,
    bound: Vec<Bound>,
    ran: usize,
}

impl<'a> OnDevice<'a> {
    pub fn new(
        device: &'a Device,
        stages: &'a mut Stages,
        key: [u8; 32],
        kept: Option<&'a Kept>,
        bound: Vec<Bound>,
    ) -> OnDevice<'a> {
        OnDevice {
            device,
            stages,
            key,
            kept,
            bound,
            ran: 0,
        }
    }

    /// Stages this runner executed.
    #[must_use]
    pub fn ran(&self) -> usize {
        self.ran
    }
}

impl StageRunner for OnDevice<'_> {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        _roots: &[u32],
        _wanted: &[u32],
        vals: &mut [Value],
    ) -> eta_exec::Result<()> {
        let at = sp.stage_index;
        let fail = |message: String| eta_exec::Error { message };
        let batch = batch_of(&plan.package, at, 1, self.kept, &self.bound);
        let staged = self
            .stages
            .get(self.device, &plan.package, self.key, at, batch, &[])
            .map_err(fail)?;
        // No carried channels on this path: the stage never takes the pool.
        let outs = run_stage(
            self.device,
            &staged,
            self.kept,
            &[Feeding {
                vals,
                bound: &self.bound,
                cells: &[],
            }],
            &mut None,
            0,
        )
        .map_err(fail)?;
        self.stages.runs += 1;
        self.stages.lanes += 1;
        for (id, value) in outs.into_iter().next().unwrap_or_default() {
            vals[id as usize] = value;
        }
        self.ran += 1;
        Ok(())
    }

    fn binds(&self, root: &LaunchValue) -> bool {
        device_binds(root, self.kept, &self.bound)
    }
}

/// One instance of a group: every member reads its outputs off the same
/// planes.
pub struct Member<'a> {
    pub inst: &'a mut InterpInstance,
    pub bound: Vec<Bound>,
    /// Each carried channel's cell: the pool row its `Channel` slots start
    /// at. The cells there are the instance's (`write_cells` lands the
    /// rings' before a first pass on the device).
    pub cells: Vec<(u32, u32)>,
}

/// A cell a pass puts: known on the host, or lane `row`'s `slot` of the
/// last stage's output, still on its way back from the device.
#[derive(Clone)]
enum Cell {
    Known(Value),
    Out { row: usize, slot: Slot },
}

struct Overlay {
    pending: BTreeMap<u32, Cell>,
    taken: Vec<bool>,
    put: Vec<bool>,
}

impl Overlay {
    fn new(channels: usize) -> Overlay {
        Overlay {
            pending: BTreeMap::new(),
            taken: vec![false; channels],
            put: vec![false; channels],
        }
    }

    fn resolve(&self, inst: &InterpInstance, chan: u32) -> Value {
        if let Some(Cell::Known(v)) = self.pending.get(&chan) {
            return v.clone();
        }
        inst.channels[chan as usize].current()
    }

    fn take(&mut self, inst: &InterpInstance, chan: u32) -> Value {
        self.taken[chan as usize] = true;
        self.resolve(inst, chan)
    }
}

/// A put still on the device: lane `row`'s `slot` of the pass's last stage
/// output, to land in `ring` at sequence `seq`. The ring's tail is stored
/// only once the cell is written, so no reader (the runtime reads a host
/// ring's mirror directly) sees it early.
pub struct DeferredPut {
    pub ring: Arc<eta_exec::ChannelState>,
    pub seq: u64,
    pub row: usize,
    pub slot: Slot,
}

/// What a deferred group pass left on the device: its last stage's packed
/// output and the puts that read it, per member (`members[i]` is the
/// member index of `puts[i]`).
pub struct Deferred {
    pub out: Arc<Buffer>,
    pub out_words: usize,
    pub batch: usize,
    pub puts: Vec<DeferredPut>,
    pub members: Vec<usize>,
}

fn const_root(root: &eta_compiler::codegen::launch::LaunchValue) -> Value {
    match root.dtype {
        Dtype::I32 => Value::I32(vec![root.literal_bits as i32]),
        Dtype::U32 => Value::U32(vec![root.literal_bits]),
        Dtype::Bool => Value::Bool(vec![root.literal_bits != 0]),
        _ => Value::F32(vec![f32::from_bits(root.literal_bits)]),
    }
}

/// A carried channel's cell: its dtype, elements and u32 words.
fn cell_shape(plan: &ExecPlan, chan: u32) -> Result<(Dtype, usize, usize), String> {
    let decl = plan
        .package
        .channels
        .get(chan as usize)
        .ok_or_else(|| format!("carried channel {chan} is past the program's"))?;
    let dtype = eta_exec::concrete_dtype(decl.dtype);
    let n = decl
        .shape
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    Ok((dtype, n, lower::words(dtype, n)))
}

/// Reads an instance's carried cells out of `pool` at `cells` (channel,
/// first row) into the front of its rings: where the host runs its next
/// pass, or the cells leave the device.
pub fn read_cells(
    plan: &ExecPlan,
    pool: &Buffer,
    cells: &[(u32, u32)],
    inst: &InterpInstance,
) -> Result<(), String> {
    for &(chan, row) in cells {
        let (dtype, n, w) = cell_shape(plan, chan)?;
        let raw = pool
            .read_raw(i64::from(row) * 512, (w * 4) as i64)
            .map_err(|e| e.to_string())?;
        if let Some(ring) = inst.channels.get(chan as usize) {
            ring.encode_sequence(ring.head(), &decode(&words_of(&raw), dtype, n));
        }
    }
    Ok(())
}

/// Rows `write_cells` lands a program run: one program, warmed at load.
pub const CELL_ROWS: usize = 16;

/// Lands each instance's carried cells, the front of its rings, in `pool`
/// at its rows (`cells[i]` for `insts[i]`): before a first pass on the
/// device, or after the host ran one. One program over `CELL_ROWS` rows a
/// run, padded with the sink row.
pub fn write_cells(
    device: &Device,
    plan: &ExecPlan,
    pool: &mut Option<Buffer>,
    pool_rows: i64,
    sink_row: u32,
    writes: &[(&InterpInstance, &[(u32, u32)])],
) -> Result<(), String> {
    let mut rows: Vec<(u32, Vec<u32>)> = Vec::new();
    for (inst, cells) in writes {
        for &(chan, first) in *cells {
            let (dtype, n, w) = cell_shape(plan, chan)?;
            let ring = inst
                .channels
                .get(chan as usize)
                .ok_or_else(|| format!("carried channel {chan} is past the instance's"))?;
            let mut words = Vec::with_capacity(w);
            encode(&ring.current(), dtype, n, &mut words)?;
            words.resize(lower::cell_rows(w) * 128, 0);
            for (k, row) in words.chunks(128).enumerate() {
                rows.push((first + k as u32, row.to_vec()));
            }
        }
    }
    let program = cells_program(device, pool_rows)?;
    for batch in rows.chunks(CELL_ROWS) {
        let mut at: Vec<i32> = batch.iter().map(|(r, _)| *r as i32).collect();
        let mut data: Vec<u32> = batch.iter().flat_map(|(_, w)| w.iter().copied()).collect();
        at.resize(CELL_ROWS, sink_row as i32);
        data.resize(CELL_ROWS * 128, 0);
        let at_bytes: Vec<u8> = at.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let at = device
            .upload(dtype::Dtype::I32, CELL_ROWS as u32, 1, &at_bytes)
            .map_err(|e| e.to_string())?;
        let data = device
            .upload(dtype::Dtype::U32, CELL_ROWS as u32, 128, &data_bytes)
            .map_err(|e| e.to_string())?;
        let whole = pool.take().ok_or("the pool, which another program holds")?;
        let mut outs = device
            .run(&program, vec![Arg::Donate(whole), Arg::Keep(&at), Arg::Keep(&data)], false)
            .map_err(|e| e.to_string())?;
        *pool = Some(outs.pop().ok_or("the cells program returned no pool")?);
    }
    Ok(())
}

/// The program `write_cells` runs: `pool` with `data` (u32 `[CELL_ROWS,
/// 128]`) landed at rows `at` (i32 `[CELL_ROWS, 1]`).
pub fn cells_program(device: &Device, pool_rows: i64) -> Result<Arc<Program>, String> {
    use kernels_xla::hlo::{Combine, Elem, Func, Ty};
    let mut f = Func::new("main");
    let pool = f.param(Ty::new(Elem::U32, &[pool_rows.max(1), 128]), Some(0));
    f.pin_row_major(pool);
    let at = f.param(Ty::new(Elem::I32, &[CELL_ROWS as i64, 1]), None);
    let data = f.param(Ty::new(Elem::U32, &[CELL_ROWS as i64, 128]), None);
    let built = (|| -> Result<_, kernels_xla::Error> {
        let at = f.reshape(at, &[CELL_ROWS as i64])?;
        let out = f.put_rows(pool, at, data, Combine::Set)?;
        Ok(f.module("eta_cells", &[out]))
    })()
    .map_err(|e| e.to_string())?;
    let sig = crate::trace::Signature {
        params: Vec::new(),
        results: Vec::new(),
    };
    device.program(&built, sig).map_err(|e| e.to_string())
}

/// `eta_exec::step_with` for every member at once: the same readiness
/// check, overlay and commit per instance, and each stage one executable
/// over the members. Members must all run `plan` and every output it reads
/// must bind on the device (`bound`).
///
/// The `carried` channels' cells stay on the device, in each member's
/// `Channel` slots of `pool` (`Member::cells`): a pass reads and writes
/// them there.
#[allow(clippy::too_many_arguments)]
pub fn step_group(
    device: &Device,
    stages: &mut Stages,
    key: [u8; 32],
    plan: &ExecPlan,
    kept: Option<&Kept>,
    members: &mut [Member<'_>],
    carried: &[u32],
    pool: &mut Option<Buffer>,
    at: Stage,
) -> Vec<StepOutcome> {
    let (outcomes, deferred) = step_group_with(
        device, stages, key, plan, kept, members, carried, pool, false, at,
    );
    debug_assert!(deferred.is_none(), "a synchronous pass defers nothing");
    outcomes
}

/// `step_group` whose last stage is left running: its output comes back
/// later (`Deferred`), and the puts that read it are committed without
/// their cells, whose ring tails wait for them. Every other part of the
/// commit (heads, host-known puts) is done here.
#[allow(clippy::too_many_arguments)]
pub fn step_group_deferred(
    device: &Device,
    stages: &mut Stages,
    key: [u8; 32],
    plan: &ExecPlan,
    kept: Option<&Kept>,
    members: &mut [Member<'_>],
    carried: &[u32],
    pool: &mut Option<Buffer>,
    at: Stage,
) -> (Vec<StepOutcome>, Option<Deferred>) {
    step_group_with(
        device, stages, key, plan, kept, members, carried, pool, true, at,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn step_group_with(
    device: &Device,
    stages: &mut Stages,
    key: [u8; 32],
    plan: &ExecPlan,
    kept: Option<&Kept>,
    members: &mut [Member<'_>],
    carried: &[u32],
    pool: &mut Option<Buffer>,
    defer: bool,
    at: Stage,
) -> (Vec<StepOutcome>, Option<Deferred>) {
    let mut outcome: Vec<Option<StepOutcome>> = vec![None; members.len()];
    for (m, member) in members.iter().enumerate() {
        if member.inst.poisoned {
            outcome[m] = Some(StepOutcome::Faulted("instance is poisoned".to_string()));
            continue;
        }
        for (channel, ring) in member.inst.channels.iter().enumerate() {
            let readiness = plan.readiness(channel as u32, at);
            let ready = match readiness {
                Some(Direction::NeedsFull) => !ring.is_empty(),
                Some(Direction::NeedsEmpty) => !ring.is_full(),
                None => true,
            };
            if !ready {
                outcome[m] = Some(StepOutcome::Blocked(channel as u32));
                break;
            }
        }
    }
    let mut overlays: Vec<Overlay> = members
        .iter()
        .map(|member| Overlay::new(member.inst.channels.len()))
        .collect();
    let mut vals: Vec<Vec<Value>> = members
        .iter()
        .map(|_| vec![Value::F32(Vec::new()); plan.package.values.len()])
        .collect();

    let kinds = [Stage::Prologue, Stage::Epilogue];
    // The stage that runs last: the one whose output may stay in flight.
    let last = kinds
        .iter()
        .flat_map(|kind| {
            plan.stages.iter().filter(move |sp| {
                plan.package.stages[sp.stage_index].stage == *kind
                    && plan.runs_at(sp.stage_index, at)
            })
        })
        .last()
        .map(|sp| sp.stage_index);
    let mut launched_out: Option<(Arc<Buffer>, usize, usize)> = None;
    for (k, kind) in kinds.iter().enumerate() {
        if k == 1 && at != Stage::Prologue {
            for (m, member) in members.iter().enumerate() {
                if outcome[m].is_some() {
                    continue;
                }
                for port in &plan.package.ports {
                    if !port.is_const && port.port.consumes() {
                        let _ = overlays[m].take(member.inst, port.channel);
                    }
                }
            }
        }
        for sp in &plan.stages {
            if plan.package.stages[sp.stage_index].stage != *kind
                || !plan.runs_at(sp.stage_index, at)
            {
                continue;
            }
            let live: Vec<usize> = (0..members.len())
                .filter(|&m| outcome[m].is_none())
                .collect();
            if live.is_empty() {
                break;
            }
            for &m in &live {
                if let Err(why) =
                    bind_roots(plan, sp, members[m].inst, &mut overlays[m], &mut vals[m])
                {
                    outcome[m] = Some(StepOutcome::Faulted(why));
                }
            }
            let live: Vec<usize> = live.into_iter().filter(|&m| outcome[m].is_none()).collect();
            if live.is_empty() {
                continue;
            }
            let at = sp.stage_index;
            let batch = batch_of(
                &plan.package,
                at,
                padded(live.len()),
                kept,
                &members[live[0]].bound,
            );
            let staged = stages.get(device, &plan.package, key, at, batch, carried);
            // The puts that stay in the pool only (carried cells): the host
            // sees every other one, a fed channel's included.
            let pool_only: Vec<u32> = staged.as_ref().map_or_else(
                |_| Vec::new(),
                |s| s.layout.carried_out.iter().map(|(c, _)| *c).collect(),
            );
            let leave = defer
                && last == Some(at)
                && staged
                    .as_ref()
                    .is_ok_and(|s| s.program.is_some() && s.layout.out_words > 0);
            let ran = staged.and_then(|staged| {
                let feeds: Vec<Feeding<'_>> = live
                    .iter()
                    .map(|&m| Feeding {
                        vals: &vals[m],
                        bound: &members[m].bound,
                        cells: &members[m].cells,
                    })
                    .collect();
                if leave {
                    let launched =
                        launch_stage(device, &staged, kept, &feeds, pool, stages.sink_row)?;
                    let out = launched.out.ok_or("a deferred stage returned no output")?;
                    launched_out = Some((out, staged.layout.out_words, launched.batch));
                    Ok((None, Some(staged)))
                } else {
                    run_stage(device, &staged, kept, &feeds, pool, stages.sink_row)
                        .map(|outs| (Some(outs), None))
                }
            });
            match ran {
                Ok((outs, left)) => {
                    stages.runs += 1;
                    stages.lanes += live.len() as u64;
                    let stage = &plan.package.stages[at];
                    match outs {
                        Some(outs) => {
                            for (&m, out) in live.iter().zip(outs) {
                                for (id, value) in out {
                                    vals[m][id as usize] = value;
                                }
                                for put in &stage.puts {
                                    if !pool_only.contains(&put.channel) {
                                        overlays[m].pending.insert(
                                            put.channel,
                                            Cell::Known(vals[m][put.value as usize].clone()),
                                        );
                                    }
                                    overlays[m].put[put.channel as usize] = true;
                                }
                            }
                        }
                        None => {
                            let layout = &left.expect("a deferred stage is kept").layout;
                            for (row, &m) in live.iter().enumerate() {
                                for put in &stage.puts {
                                    if !pool_only.contains(&put.channel) {
                                        let cell = match layout
                                            .outputs
                                            .iter()
                                            .find(|slot| slot.value == put.value)
                                        {
                                            Some(slot) => Cell::Out { row, slot: *slot },
                                            None => {
                                                Cell::Known(vals[m][put.value as usize].clone())
                                            }
                                        };
                                        overlays[m].pending.insert(put.channel, cell);
                                    }
                                    overlays[m].put[put.channel as usize] = true;
                                }
                            }
                        }
                    }
                }
                Err(why) => {
                    for &m in &live {
                        outcome[m] = Some(StepOutcome::Faulted(why.clone()));
                    }
                }
            }
        }
    }

    let mut puts: Vec<DeferredPut> = Vec::new();
    let mut put_members: Vec<usize> = Vec::new();
    let outcomes: Vec<StepOutcome> = members
        .iter_mut()
        .zip(outcome)
        .zip(&overlays)
        .enumerate()
        .map(|(m, ((member, outcome), overlay))| match outcome {
            Some(StepOutcome::Faulted(why)) => {
                member.inst.poisoned = true;
                StepOutcome::Faulted(why)
            }
            Some(other) => other,
            None => {
                let (outcome, left) = commit(member.inst, overlay);
                for (chan, seq, row, slot) in left {
                    puts.push(DeferredPut {
                        ring: Arc::clone(&member.inst.channels[chan]),
                        seq,
                        row,
                        slot,
                    });
                    put_members.push(m);
                }
                outcome
            }
        })
        .collect();
    let deferred = match launched_out {
        Some((out, out_words, batch)) if !puts.is_empty() => Some(Deferred {
            out,
            out_words,
            batch,
            puts,
            members: put_members,
        }),
        _ => None,
    };
    // The carried cells are in the members' slots: a member that faulted
    // after running is poisoned, so no cell of its is read again.
    (outcomes, deferred)
}

fn bind_roots(
    plan: &ExecPlan,
    sp: &StagePlan,
    inst: &InterpInstance,
    overlay: &mut Overlay,
    vals: &mut [Value],
) -> Result<(), String> {
    for &id in &sp.value_ids {
        if sp.op_by_result.contains_key(&id) {
            continue;
        }
        let root = &plan.package.values[id as usize];
        let cell = match root.source {
            ValueOrigin::Const => const_root(root),
            ValueOrigin::ChannelTake => overlay.take(inst, root.channel),
            ValueOrigin::ChannelRead => overlay.resolve(inst, root.channel),
            ValueOrigin::Output => Value::F32(Vec::new()),
            ValueOrigin::OpResult => {
                return Err(
                    "unresolved value root (output/host input) reached execution".to_owned(),
                );
            }
        };
        vals[id as usize] = cell;
    }
    Ok(())
}

/// Commits a pass's overlay into the instance's rings. A put whose cell is
/// still on the device (`Cell::Out`) is left out of its ring's tail and
/// answered as (channel, sequence, row, slot) for whoever lands it.
#[allow(clippy::type_complexity)]
fn commit(
    inst: &mut InterpInstance,
    overlay: &Overlay,
) -> (StepOutcome, Vec<(usize, u64, usize, Slot)>) {
    let n = inst.channels.len();
    let mut old_tails = vec![0u64; n];
    let mut new_heads = vec![0u64; n];
    let mut new_tails = vec![0u64; n];

    for (ci, ring) in inst.channels.iter().enumerate() {
        let head = ring.head();
        let tail = ring.tail();
        if tail < head {
            inst.poisoned = true;
            return (
                StepOutcome::Faulted(format!("channel {ci}: tail precedes head at commit")),
                Vec::new(),
            );
        }
        let mut next_head = head;
        let mut next_tail = tail;
        let mut used = tail - head;
        if overlay.taken[ci] && used != 0 {
            next_head += 1;
            used -= 1;
        }
        if overlay.put[ci] {
            if used >= ring.capacity() as u64 {
                inst.poisoned = true;
                return (
                    StepOutcome::Faulted(format!(
                        "channel {ci}: put overflows capacity {} at commit",
                        ring.capacity()
                    )),
                    Vec::new(),
                );
            }
            next_tail += 1;
        }
        old_tails[ci] = tail;
        new_heads[ci] = next_head;
        new_tails[ci] = next_tail;
    }

    let mut left = Vec::new();
    for (ci, ring) in inst.channels.iter().enumerate() {
        // A carried cell has no host value: it stays on the device.
        if !overlay.put[ci] {
            continue;
        }
        match overlay.pending.get(&(ci as u32)) {
            Some(Cell::Known(value)) => ring.encode_sequence(old_tails[ci], value),
            Some(Cell::Out { row, slot }) => {
                left.push((ci, old_tails[ci], *row, *slot));
                new_tails[ci] = old_tails[ci];
            }
            None => {}
        }
    }
    for (ci, ring) in inst.channels.iter().enumerate() {
        if new_heads[ci] != ring.head() {
            ring.store_head(new_heads[ci]);
        }
        if new_tails[ci] != ring.tail() {
            ring.store_tail(new_tails[ci]);
        }
    }
    (StepOutcome::Committed, left)
}

#[cfg(test)]
mod tests {
    use super::{Batch, Layout, Staged, Stages};
    use std::sync::Arc;

    fn staged(held: u64, need: u64, text: u8) -> Arc<Staged> {
        Arc::new(Staged {
            program: None,
            layout: Layout::default(),
            text_key: Some([text; 32]),
            held,
            need,
        })
    }

    fn one(lanes: u32) -> Vec<super::Want> {
        vec![(0, Batch { lanes, reads: Vec::new() }, Vec::new())]
    }

    /// The stage cache holds what it compiled while the host keeps room for
    /// its largest compile; short of it, the program run least recently
    /// goes first (never the one just installed), its compiled forms handed
    /// to the device cache to drop, until the room is back.
    #[test]
    fn the_stage_cache_yields_the_least_recent_program_to_the_hosts_room() {
        let mut stages = Stages::default();
        let (a, b, c) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        stages.install(a, one(1), vec![staged(100, 1000, 1)]);
        stages.install(b, one(1), vec![staged(300, 1000, 2)]);
        stages.install(c, one(1), vec![staged(500, 1000, 3)]);
        stages.take_evicted();
        assert_eq!(stages.len(), 3);

        // Room for the largest compile: nothing goes.
        assert_eq!(stages.yield_room(c, Some(1000)), 0);
        assert_eq!(stages.len(), 3);

        // 700 short of it... a (100) then b (300) go, c stays.
        assert_eq!(stages.yield_room(c, Some(600)), 400);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages.take_evicted(), vec![[1u8; 32], [2u8; 32]]);

        // Only the kept program is left: it stays however short the room.
        assert_eq!(stages.yield_room(c, Some(0)), 0);
        assert_eq!(stages.len(), 1);

        // No reading of the host: nothing is evicted.
        assert_eq!(stages.yield_room([9; 32], None), 0);
        assert_eq!(stages.len(), 1);
    }
}
