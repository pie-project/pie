use std::sync::atomic::AtomicU64;

#[derive(Debug, Default)]
pub struct FireProbes {
    pub inter_fire_us: AtomicU64,

    pub post_dispatch_to_fire_us: AtomicU64,

    pub recv_block_wait_us: AtomicU64,

    pub last_fire_spawn_micros: AtomicU64,

    pub last_dispatch_end_micros: AtomicU64,

    pub accumulate: AccumulateProbes,
    pub pre_dispatch: PreDispatchProbes,
    pub execute: ExecuteProbes,
    pub post_dispatch: PostDispatchProbes,
    pub quorum: QuorumProbes,
}

#[derive(Debug, Default)]
pub struct QuorumProbes {
    pub inter_batch_bubble_us: AtomicU64,

    pub quorum_latency_us: AtomicU64,

    pub escape_fires: AtomicU64,

    pub submit_ahead_fires: AtomicU64,

    pub straggler_fires: AtomicU64,
    pub straggler_demotions: AtomicU64,

    pub readiness_miss: AtomicU64,

    pub wave_active_sum: AtomicU64,
    pub wave_missing_sum: AtomicU64,
    pub wave_fires: AtomicU64,

    pub seal_events: AtomicU64,
    pub seal_while_executing: AtomicU64,

    pub dispatch_blocked_holds: AtomicU64,

    pub device_idle_us: AtomicU64,
    pub device_idle_gaps: AtomicU64,

    pub idle_break_control: AtomicU64,
    pub idle_break_depth: AtomicU64,

    pub idle_park_control_us: AtomicU64,
    pub idle_park_other_us: AtomicU64,

    pub accept_us: AtomicU64,
    pub accept_calls: AtomicU64,

    pub turnaround_sum_us: AtomicU64,
    pub turnaround_max_us: AtomicU64,
    pub turnaround_n: AtomicU64,

    pub lane_launch_us: AtomicU64,
    pub lane_launch_n: AtomicU64,
    pub lane_prefill_us: AtomicU64,
    pub lane_prefill_n: AtomicU64,
    pub lane_control_us: AtomicU64,
    pub lane_control_n: AtomicU64,
    pub lane_control_max_us: AtomicU64,
}

#[derive(Debug, Default)]
pub struct AccumulateProbes {
    pub accum_loop_us: AtomicU64,
}

#[derive(Debug, Default)]
pub struct PreDispatchProbes {
    pub fire_prepare_us: AtomicU64,
}

#[derive(Debug, Default)]
pub struct ExecuteProbes {
    pub total_us: AtomicU64,

    pub batch_build_us: AtomicU64,

    pub engine_fire_us: AtomicU64,
}

#[derive(Debug, Default)]
pub struct PostDispatchProbes {
    pub context_tick_us: AtomicU64,
    pub stats_update_us: AtomicU64,
}

#[cfg(feature = "profile-fire")]
#[macro_export]
macro_rules! probe_fire {
    ($target:expr, $body:expr) => {{
        let __probe_start = $crate::rt::Instant::now();
        let __probe_result = $body;
        $target.fetch_add(
            __probe_start.elapsed().as_micros() as u64,
            ::std::sync::atomic::Ordering::Relaxed,
        );
        __probe_result
    }};
}

#[cfg(not(feature = "profile-fire"))]
#[macro_export]
macro_rules! probe_fire {
    ($target:expr, $body:expr) => {{
        let _ = &$target;
        $body
    }};
}

#[cfg(feature = "profile-fire")]
#[macro_export]
macro_rules! probe_fire_record {
    ($target:expr, $duration:expr) => {{
        $target.fetch_add(
            $duration.as_micros() as u64,
            ::std::sync::atomic::Ordering::Relaxed,
        );
    }};
}

#[cfg(not(feature = "profile-fire"))]
#[macro_export]
macro_rules! probe_fire_record {
    ($target:expr, $duration:expr) => {{
        let _ = (&$target, &$duration);
    }};
}

#[derive(Debug, Default)]
pub struct HostSubmitProbes {
    pub submits: AtomicU64,
    pub total_us: AtomicU64,
    pub drain_settled_us: AtomicU64,
    pub geometry_us: AtomicU64,
    pub kv_prepare_us: AtomicU64,
    pub scheduler_submit_us: AtomicU64,
    pub shadow_advance_us: AtomicU64,
    pub validate_frame_us: AtomicU64,
    pub validate_frame_calls: AtomicU64,
}

pub fn host_submit() -> &'static HostSubmitProbes {
    static PROBES: std::sync::OnceLock<HostSubmitProbes> = std::sync::OnceLock::new();
    PROBES.get_or_init(HostSubmitProbes::default)
}

#[derive(Clone, Copy, Debug)]
pub struct ProbeClock {
    #[cfg(feature = "profile-fire")]
    began: crate::rt::Instant,
}

impl ProbeClock {
    #[must_use]
    pub fn start() -> Self {
        Self {
            #[cfg(feature = "profile-fire")]
            began: crate::rt::Instant::now(),
        }
    }

    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        #[cfg(feature = "profile-fire")]
        {
            self.began.elapsed()
        }
        #[cfg(not(feature = "profile-fire"))]
        {
            std::time::Duration::ZERO
        }
    }
}

#[cfg(feature = "profile-fire")]
#[macro_export]
macro_rules! probe_fire_count {
    ($target:expr) => {{
        $target.fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    }};
}

#[cfg(not(feature = "profile-fire"))]
#[macro_export]
macro_rules! probe_fire_count {
    ($target:expr) => {{
        let _ = &$target;
    }};
}
