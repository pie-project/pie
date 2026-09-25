use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    pub apple_family: u32,

    pub gpu_core_count: u32,
}

impl DeviceInfo {
    #[must_use]
    pub fn of_name(name: &str) -> Self {
        let family = if name.contains("M1") {
            7
        } else if name.contains("M2") {
            8
        } else if name.contains("M3") || name.contains("M4") {
            9
        } else {
            0
        };
        Self {
            apple_family: family,
            gpu_core_count: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceTuning {
    pub qmm_min_batch: u32,

    pub qmm_min_batch_moe: u32,

    pub qmm_min_batch_emulated: u32,

    pub qmm_bn_crossover_tg: u32,

    pub moe_tile_mid_per: u32,

    pub moe_tile_wide_per: u32,

    pub qmm_wide_range: bool,

    pub fp16_qmm: bool,

    pub sdpa_tile_min_rows_per_request: u32,

    pub sdpa_mma: bool,

    pub gdn_scan_lanes: u32,

    pub gdn_scan_rows: u32,

    pub moe_batch_min_per_expert: u32,

    pub moe_batch_min_pairs: u32,

    pub qmv_rows_max: u32,

    pub qmv_rows_packs: u32,

    pub stream_rows_per_cut: u32,
}

impl Default for DeviceTuning {
    fn default() -> Self {
        Self {
            qmm_min_batch: 5,
            qmm_min_batch_moe: 8,
            qmm_min_batch_emulated: 12,
            qmm_bn_crossover_tg: 160,
            moe_tile_mid_per: 32,
            moe_tile_wide_per: 1 << 24,
            qmm_wide_range: false,
            fp16_qmm: true,
            sdpa_tile_min_rows_per_request: 32,
            sdpa_mma: true,
            gdn_scan_lanes: 32,
            gdn_scan_rows: 4,
            moe_batch_min_per_expert: 2,
            moe_batch_min_pairs: 0,
            qmv_rows_max: 8,
            qmv_rows_packs: 1,
            stream_rows_per_cut: 0,
        }
    }
}

impl DeviceTuning {
    #[must_use]
    pub fn of(info: DeviceInfo) -> Self {
        let mut t = Self::default();
        match info.apple_family {
            9 => {
                t.qmm_bn_crossover_tg = 96;
            }
            8 => {
                t.qmm_min_batch = 8;
                t.qmm_min_batch_moe = 12;
            }
            _ => {}
        }
        t
    }

    #[must_use]
    pub fn with(mut self, over: &Overrides) -> Self {
        macro_rules! lay {
            ($($field:ident),+ $(,)?) => {
                $(if let Some(v) = over.$field { self.$field = v; })+
            };
        }
        lay!(
            qmm_min_batch,
            qmm_min_batch_moe,
            qmm_min_batch_emulated,
            qmm_bn_crossover_tg,
            moe_tile_mid_per,
            moe_tile_wide_per,
            qmm_wide_range,
            fp16_qmm,
            sdpa_tile_min_rows_per_request,
            sdpa_mma,
            gdn_scan_lanes,
            gdn_scan_rows,
            moe_batch_min_per_expert,
            moe_batch_min_pairs,
            qmv_rows_max,
            qmv_rows_packs,
            stream_rows_per_cut,
        );
        self
    }

    #[must_use]
    pub const fn qmm_min_batch(&self, routed: bool, fp16_gemm: bool) -> u32 {
        if !fp16_gemm {
            return self.qmm_min_batch_emulated;
        }
        if routed {
            self.qmm_min_batch_moe
        } else {
            self.qmm_min_batch
        }
    }

    #[must_use]
    pub const fn fp16_gemm_format(&self, bits: u32, group: u32) -> bool {
        !self.qmm_wide_range && self.fp16_qmm && bits == 4 && group == 64
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overrides {
    pub qmm_min_batch: Option<u32>,
    pub qmm_min_batch_moe: Option<u32>,
    pub qmm_min_batch_emulated: Option<u32>,
    pub qmm_bn_crossover_tg: Option<u32>,
    pub moe_tile_mid_per: Option<u32>,
    pub moe_tile_wide_per: Option<u32>,
    pub qmm_wide_range: Option<bool>,
    pub fp16_qmm: Option<bool>,
    pub sdpa_tile_min_rows_per_request: Option<u32>,
    pub sdpa_mma: Option<bool>,
    pub gdn_scan_lanes: Option<u32>,
    pub gdn_scan_rows: Option<u32>,
    pub moe_batch_min_per_expert: Option<u32>,
    pub moe_batch_min_pairs: Option<u32>,
    pub qmv_rows_max: Option<u32>,
    pub qmv_rows_packs: Option<u32>,
    pub stream_rows_per_cut: Option<u32>,
}

static DEVICE: OnceLock<DeviceInfo> = OnceLock::new();
static OVERRIDES: OnceLock<Overrides> = OnceLock::new();
static RESOLVED: OnceLock<DeviceTuning> = OnceLock::new();

pub fn describe(info: DeviceInfo) -> bool {
    DEVICE.set(info).is_ok()
}

pub fn override_with(over: Overrides) -> bool {
    OVERRIDES.set(over).is_ok()
}

#[must_use]
pub fn current() -> DeviceTuning {
    *RESOLVED.get_or_init(|| {
        let info = DEVICE.get().copied().unwrap_or_default();
        let over = OVERRIDES.get().copied().unwrap_or_default();
        DeviceTuning::of(info).with(&over)
    })
}
