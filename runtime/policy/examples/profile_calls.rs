use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use pie_plex::v0_5::{ContractVersion, Manifest, Operation, PolicyLimits};
use pie_policy::{
    AttachedPolicy, AttachmentRegistry, Invocation, InvocationFailureKind, PolicyEngine,
    PolicyEngineConfig, RejectingQueryHandler, StateSnapshot,
};
use serde_json::json;
use wasmtime::component::{Component, HasSelf, Linker};
use wasmtime::{Config, Engine, PoolingAllocationConfig, Store, StoreLimits, StoreLimitsBuilder};

mod bindings {
    wasmtime::component::bindgen!({
        path: "../../interface/plex/wit",
        world: "plex-policy",
        anyhow: true,
    });
}

use bindings::exports::pie::plex::policy::Invocation as WitInvocation;
use bindings::pie::plex::host;
use bindings::{PlexPolicy, PlexPolicyPre};

const MEMORY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
enum Allocation {
    OnDemand,
    Pooling,
}

#[derive(Clone, Copy)]
enum RawMode {
    Fresh,
    Reuse,
}

struct RawHost {
    limits: StoreLimits,
}

impl host::Host for RawHost {
    fn query(&mut self, method: String, _args_json: String) -> Result<String, String> {
        Err(format!("unsupported query {method}"))
    }

    fn action(&mut self, method: String, _args_json: String) -> Result<u64, String> {
        Err(format!("unsupported action {method}"))
    }
}

struct RawSetup {
    engine: Engine,
    pre: PlexPolicyPre<RawHost>,
    input: WitInvocation,
}

struct PlexSetup {
    snapshot: pie_policy::AttachmentSnapshot,
    context: pie_plex::Document,
    state: StateSnapshot,
    query: Arc<RejectingQueryHandler>,
    actions: Arc<BTreeSet<String>>,
}

#[derive(Default)]
struct WorkerStats {
    samples_ns: Vec<u64>,
    store_ns: u128,
    instantiate_ns: u128,
    call_ns: u128,
    teardown_ns: u128,
    successes: u64,
    saturated: u64,
}

struct Sample {
    total_ns: u64,
    store_ns: u64,
    instantiate_ns: u64,
    call_ns: u64,
    teardown_ns: u64,
}

struct BenchResult {
    mode: String,
    threads: usize,
    permit_cap: Option<u32>,
    elapsed: Duration,
    stats: WorkerStats,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let component_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("tests/policies/target/components/plex_paper_helium.component.wasm")
    });
    let total_operations = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(20_000usize);
    let thread_counts = args
        .next()
        .map(|value| parse_thread_counts(&value))
        .transpose()?
        .unwrap_or_else(default_thread_counts);
    if args.next().is_some() {
        return Err(
            "usage: profile_calls [component.wasm] [total-operations] [thread-counts]".into(),
        );
    }
    if total_operations == 0 || thread_counts.is_empty() {
        return Err("total operations and thread counts must be non-zero".into());
    }

    let component = Arc::new(std::fs::read(&component_path)?);
    let unconstrained_cap =
        u32::try_from(thread_counts.iter().copied().max().unwrap_or(1).max(128))?;
    let raw_on_demand = Arc::new(RawSetup::new(
        &component,
        Allocation::OnDemand,
        unconstrained_cap,
    )?);
    let raw_pooling = Arc::new(RawSetup::new(
        &component,
        Allocation::Pooling,
        unconstrained_cap,
    )?);
    let plex = Arc::new(PlexSetup::new(&component, unconstrained_cap)?);

    eprintln!(
        "component={} bytes={} total_operations={} cpus={}",
        component_path.display(),
        component.len(),
        total_operations,
        thread::available_parallelism()?.get()
    );
    eprintln!(
        "raw-worker-reuse-no-reset is an upper-bound experiment: it preserves guest memory and \
         globals across calls and is not a production-safe PLEX mode"
    );
    print_header();
    for threads in thread_counts.iter().copied() {
        print_result(run_raw(
            "raw-fresh-on-demand",
            raw_on_demand.clone(),
            RawMode::Fresh,
            threads,
            total_operations,
        )?);
        print_result(run_raw(
            "raw-fresh-pooling",
            raw_pooling.clone(),
            RawMode::Fresh,
            threads,
            total_operations,
        )?);
        print_result(run_raw(
            "raw-worker-reuse-no-reset",
            raw_on_demand.clone(),
            RawMode::Reuse,
            threads,
            total_operations,
        )?);
        print_result(run_plex(
            format!("plex-invoke-cap-{unconstrained_cap}"),
            plex.clone(),
            threads,
            total_operations,
            unconstrained_cap,
        )?);
    }

    let contention_threads = thread_counts.iter().copied().max().unwrap_or(1);
    for cap in [1, 4, 8, 16, 32, 64, 128]
        .into_iter()
        .filter(|cap| *cap < contention_threads as u32)
    {
        let setup = Arc::new(PlexSetup::new(&component, cap)?);
        print_result(run_plex(
            format!("plex-invoke-cap-{cap}"),
            setup,
            contention_threads,
            total_operations,
            cap,
        )?);
    }
    Ok(())
}

impl RawSetup {
    fn new(
        component_bytes: &[u8],
        allocation: Allocation,
        pool_capacity: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let mut config = Config::new();
        config.epoch_interruption(true);
        config.wasm_threads(false);
        if matches!(allocation, Allocation::Pooling) {
            let aggregate = pool_capacity
                .checked_mul(4)
                .ok_or("raw profiler pool capacity overflow")?;
            let mut pooling = PoolingAllocationConfig::new();
            pooling
                .total_component_instances(pool_capacity)
                .max_core_instances_per_component(4)
                .total_core_instances(aggregate)
                .max_memories_per_component(1)
                .total_memories(pool_capacity)
                .max_tables_per_component(4)
                .total_tables(aggregate)
                .table_elements(1024)
                .max_memory_size(MEMORY_BYTES)
                .max_unused_warm_slots(pool_capacity);
            config.memory_reservation(MEMORY_BYTES as u64);
            config.allocation_strategy(pooling);
        }
        let engine = Engine::new(&config)?;
        let component = Component::new(&engine, component_bytes)?;
        let mut linker = Linker::<RawHost>::new(&engine);
        PlexPolicy::add_to_linker::<RawHost, HasSelf<RawHost>>(&mut linker, |host| host)?;
        let pre = PlexPolicyPre::new(linker.instantiate_pre(&component)?)?;
        let (context, state) = invocation_documents();
        Ok(Self {
            engine,
            pre,
            input: WitInvocation {
                context_json: serde_json::to_string(&context)?,
                state_json: serde_json::to_string(&state.document())?,
            },
        })
    }

    fn store(&self) -> Result<Store<RawHost>, Box<dyn Error + Send + Sync>> {
        let limits = StoreLimitsBuilder::new()
            .memory_size(MEMORY_BYTES)
            .table_elements(1024)
            .instances(4)
            .tables(4)
            .memories(1)
            .build();
        let mut store = Store::new(&self.engine, RawHost { limits });
        store.limiter(|host| &mut host.limits);
        reset_store(&mut store)?;
        Ok(store)
    }
}

impl PlexSetup {
    fn new(component_bytes: &[u8], permit_cap: u32) -> Result<Self, Box<dyn Error>> {
        let engine = PolicyEngine::new(PolicyEngineConfig {
            max_concurrent_invocations: permit_cap,
            epoch_tick: None,
            ..PolicyEngineConfig::default()
        })
        .map_err(|error| format!("create PolicyEngine(cap={permit_cap}): {error}"))?;
        let manifest = Manifest {
            contract: ContractVersion::V0_5,
            package_name: format!("profile-{permit_cap}"),
            package_version: "0.5.0".into(),
            operations: BTreeSet::from([Operation::Schedule]),
            limits: PolicyLimits {
                memory_bytes: MEMORY_BYTES as u64,
                fuel: 1,
                deadline_ms: 100,
                input_bytes: 1 << 20,
                output_bytes: 1 << 20,
            },
        };
        let policy = AttachedPolicy::compile(engine.clone(), component_bytes, manifest)
            .map_err(|error| format!("compile policy(cap={permit_cap}): {error}"))?;
        let registry = AttachmentRegistry::new(engine);
        registry
            .attach_prepared(policy)
            .map_err(|error| format!("attach policy(cap={permit_cap}): {error}"))?;
        let snapshot = registry
            .snapshot()
            .map_err(|error| format!("snapshot registry(cap={permit_cap}): {error}"))?;
        let (context, state) = invocation_documents();
        Ok(Self {
            snapshot,
            context,
            state,
            query: Arc::new(RejectingQueryHandler),
            actions: Arc::new(BTreeSet::new()),
        })
    }

    fn invoke(&self) -> Result<bool, String> {
        match self.snapshot.invoke(
            Operation::Schedule,
            self.context.clone(),
            self.state.clone(),
            self.query.clone(),
            self.actions.clone(),
        ) {
            Invocation::Success(result) => {
                black_box(result);
                Ok(true)
            }
            Invocation::FallbackRequired(failure)
                if failure.kind == InvocationFailureKind::HostSaturated =>
            {
                Ok(false)
            }
            Invocation::FallbackRequired(failure) => {
                Err(format!("{:?}: {}", failure.kind, failure.message))
            }
            Invocation::Unavailable => Err("schedule operation is unavailable".into()),
        }
    }
}

fn invocation_documents() -> (pie_plex::Document, StateSnapshot) {
    let context = json!({
        "cause": "service-step",
        "runnable": [{
            "request_id": "request",
            "max_token_budget": 32,
            "facts": {
                "ready": true,
                "dependency_depth": 3,
                "prefix_reuse_tokens": 256,
                "earliest_start": 0,
                "profiled_token_cost": 32
            }
        }],
        "capacity": {
            "max_selected": 1,
            "max_total_tokens": 32,
            "max_token_budget": 32
        },
        "context": {
            "capabilities": {
                "token_budget": false
            }
        }
    });
    let request = json!({
        "facts": {
            "logical_request_id": "request",
            "generation_id": 0
        },
        "fields": {},
        "scratch": {}
    });
    let state = StateSnapshot::from_parts(
        json!({}),
        BTreeMap::from([("request".into(), request)]),
        0,
        BTreeMap::from([("request".into(), 0)]),
    )
    .expect("benchmark state is valid");
    (context, state)
}

fn reset_store(store: &mut Store<RawHost>) -> Result<(), wasmtime::Error> {
    store.set_epoch_deadline(u64::MAX);
    store.epoch_deadline_trap();
    Ok(())
}

fn raw_fresh(setup: &RawSetup) -> Result<Sample, String> {
    let total_start = Instant::now();
    let store_start = total_start;
    let mut store = setup.store().map_err(|error| error.to_string())?;
    let store_end = Instant::now();
    let policy = setup
        .pre
        .instantiate(&mut store)
        .map_err(|error| error.to_string())?;
    let instantiate_end = Instant::now();
    let output = policy
        .pie_plex_policy()
        .call_schedule(&mut store, &setup.input)
        .map_err(|error| error.to_string())?
        .map_err(|error| format!("policy rejected invocation: {error}"))?;
    black_box(&output);
    let call_end = Instant::now();
    drop(output);
    drop(policy);
    drop(store);
    let teardown_end = Instant::now();
    Ok(Sample {
        total_ns: nanos(total_start.elapsed()),
        store_ns: nanos(store_end.duration_since(store_start)),
        instantiate_ns: nanos(instantiate_end.duration_since(store_end)),
        call_ns: nanos(call_end.duration_since(instantiate_end)),
        teardown_ns: nanos(teardown_end.duration_since(call_end)),
    })
}

fn raw_reuse(
    setup: &RawSetup,
    store: &mut Store<RawHost>,
    policy: &PlexPolicy,
) -> Result<Sample, String> {
    let start = Instant::now();
    reset_store(store).map_err(|error| error.to_string())?;
    let output = policy
        .pie_plex_policy()
        .call_schedule(store, &setup.input)
        .map_err(|error| error.to_string())?
        .map_err(|error| format!("policy rejected invocation: {error}"))?;
    black_box(&output);
    drop(output);
    let elapsed = nanos(start.elapsed());
    Ok(Sample {
        total_ns: elapsed,
        store_ns: 0,
        instantiate_ns: 0,
        call_ns: elapsed,
        teardown_ns: 0,
    })
}

fn run_raw(
    mode_name: &str,
    setup: Arc<RawSetup>,
    mode: RawMode,
    threads: usize,
    total_operations: usize,
) -> Result<BenchResult, Box<dyn Error>> {
    let ready = Arc::new(Barrier::new(threads + 1));
    let go = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);
    for thread_index in 0..threads {
        let count = operations_for_thread(total_operations, threads, thread_index);
        let setup = setup.clone();
        let ready = ready.clone();
        let go = go.clone();
        handles.push(thread::spawn(move || -> Result<WorkerStats, String> {
            let mut reused = match mode {
                RawMode::Fresh => None,
                RawMode::Reuse => {
                    let mut store = setup.store().map_err(|error| error.to_string())?;
                    let policy = setup
                        .pre
                        .instantiate(&mut store)
                        .map_err(|error| error.to_string())?;
                    Some((store, policy))
                }
            };
            for _ in 0..count.min(32) {
                match &mut reused {
                    Some((store, policy)) => {
                        raw_reuse(&setup, store, policy)?;
                    }
                    None => {
                        raw_fresh(&setup)?;
                    }
                }
            }
            ready.wait();
            go.wait();
            let mut stats = WorkerStats {
                samples_ns: Vec::with_capacity(count),
                ..WorkerStats::default()
            };
            for _ in 0..count {
                let sample = match &mut reused {
                    Some((store, policy)) => raw_reuse(&setup, store, policy)?,
                    None => raw_fresh(&setup)?,
                };
                stats.record(sample);
            }
            Ok(stats)
        }));
    }
    ready.wait();
    let started = Instant::now();
    go.wait();
    let mut stats = WorkerStats::default();
    for handle in handles {
        stats.merge(
            handle
                .join()
                .map_err(|_| "raw benchmark thread panicked")??,
        );
    }
    Ok(BenchResult {
        mode: mode_name.into(),
        threads,
        permit_cap: None,
        elapsed: started.elapsed(),
        stats,
    })
}

fn run_plex(
    mode: String,
    setup: Arc<PlexSetup>,
    threads: usize,
    total_operations: usize,
    permit_cap: u32,
) -> Result<BenchResult, Box<dyn Error>> {
    for warmup in 0..32 {
        let success = setup
            .invoke()
            .map_err(|error| format!("PLEX warmup {warmup} failed: {error}"))?;
        if !success {
            return Err("PLEX warmup unexpectedly saturated".into());
        }
    }
    let ready = Arc::new(Barrier::new(threads + 1));
    let go = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);
    for thread_index in 0..threads {
        let count = operations_for_thread(total_operations, threads, thread_index);
        let setup = setup.clone();
        let ready = ready.clone();
        let go = go.clone();
        handles.push(thread::spawn(move || -> Result<WorkerStats, String> {
            ready.wait();
            go.wait();
            let mut stats = WorkerStats {
                samples_ns: Vec::with_capacity(count),
                ..WorkerStats::default()
            };
            for _ in 0..count {
                let started = Instant::now();
                let success = setup.invoke()?;
                stats.samples_ns.push(nanos(started.elapsed()));
                if success {
                    stats.successes += 1;
                } else {
                    stats.saturated += 1;
                }
            }
            Ok(stats)
        }));
    }
    ready.wait();
    let started = Instant::now();
    go.wait();
    let mut stats = WorkerStats::default();
    for handle in handles {
        stats.merge(
            handle
                .join()
                .map_err(|_| "PLEX benchmark thread panicked")??,
        );
    }
    Ok(BenchResult {
        mode,
        threads,
        permit_cap: Some(permit_cap),
        elapsed: started.elapsed(),
        stats,
    })
}

impl WorkerStats {
    fn record(&mut self, sample: Sample) {
        self.samples_ns.push(sample.total_ns);
        self.store_ns += u128::from(sample.store_ns);
        self.instantiate_ns += u128::from(sample.instantiate_ns);
        self.call_ns += u128::from(sample.call_ns);
        self.teardown_ns += u128::from(sample.teardown_ns);
        self.successes += 1;
    }

    fn merge(&mut self, mut other: Self) {
        self.samples_ns.append(&mut other.samples_ns);
        self.store_ns += other.store_ns;
        self.instantiate_ns += other.instantiate_ns;
        self.call_ns += other.call_ns;
        self.teardown_ns += other.teardown_ns;
        self.successes += other.successes;
        self.saturated += other.saturated;
    }
}

fn print_header() {
    println!(
        "mode,threads,permit_cap,attempts,success_pct,throughput_success_ops_s,\
         mean_us,p50_us,p95_us,p99_us,store_mean_us,instantiate_mean_us,\
         call_mean_us,teardown_mean_us,instantiate_pct"
    );
}

fn print_result(mut result: BenchResult) {
    result.stats.samples_ns.sort_unstable();
    let attempts = result.stats.samples_ns.len() as u64;
    let phase_count = result.stats.successes.max(1) as f64;
    let phase_total = result.stats.store_ns
        + result.stats.instantiate_ns
        + result.stats.call_ns
        + result.stats.teardown_ns;
    let instantiate_pct = if phase_total == 0 {
        f64::NAN
    } else {
        result.stats.instantiate_ns as f64 * 100.0 / phase_total as f64
    };
    println!(
        "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        result.mode,
        result.threads,
        result
            .permit_cap
            .map_or_else(|| "none".into(), |cap| cap.to_string()),
        attempts,
        percent(result.stats.successes, attempts),
        result.stats.successes as f64 / result.elapsed.as_secs_f64(),
        mean(&result.stats.samples_ns) / 1_000.0,
        percentile(&result.stats.samples_ns, 0.50) / 1_000.0,
        percentile(&result.stats.samples_ns, 0.95) / 1_000.0,
        percentile(&result.stats.samples_ns, 0.99) / 1_000.0,
        result.stats.store_ns as f64 / phase_count / 1_000.0,
        result.stats.instantiate_ns as f64 / phase_count / 1_000.0,
        result.stats.call_ns as f64 / phase_count / 1_000.0,
        result.stats.teardown_ns as f64 / phase_count / 1_000.0,
        instantiate_pct,
    );
}

fn parse_thread_counts(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut counts = value
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    counts.sort_unstable();
    counts.dedup();
    if counts.contains(&0) {
        return Err("thread counts must be non-zero".into());
    }
    Ok(counts)
}

fn default_thread_counts() -> Vec<usize> {
    let cpus = thread::available_parallelism().map_or(1, |value| value.get());
    [1, 2, 4, 8, 16, 32, 64]
        .into_iter()
        .filter(|threads| *threads <= cpus.saturating_mul(2))
        .collect()
}

fn operations_for_thread(total: usize, threads: usize, thread_index: usize) -> usize {
    total / threads + usize::from(thread_index < total % threads)
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn mean(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.iter().map(|sample| *sample as f64).sum::<f64>() / samples.len() as f64
}

fn percentile(samples: &[u64], quantile: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let index = ((samples.len() - 1) as f64 * quantile).round() as usize;
    samples[index] as f64
}

fn percent(part: u64, total: u64) -> f64 {
    if total == 0 {
        return f64::NAN;
    }
    part as f64 * 100.0 / total as f64
}
