//! The hipBLASLt matmul the dense projections take: bf16 operands, f32
//! accumulation, an optional bf16 bias folded into the epilogue. A problem's
//! descriptors and the heuristic's first algorithm are built once per
//! (device, shape, bias) and kept for the process: the planning is host
//! work, so a captured step replays only the launch.
#![allow(unsafe_code)]

use core::ffi::{c_int, c_void};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::sys::{self, Lt, LtDesc, LtHeuristic, LtLayout, LtStatus, Stream};

/// A failed hipBLASLt call: the entry it was, and what it answered.
#[derive(Clone, Debug)]
pub struct Refused {
    pub call: &'static str,
    pub status: LtStatus,
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.status == ABSENT {
            write!(f, "`{}` has no hipBLASLt to call: the library did not load", self.call)
        } else {
            write!(f, "`{}` answered {}", self.call, self.status)
        }
    }
}

/// The status a call is refused with where hipBLASLt did not load.
const ABSENT: LtStatus = -1;

/// hipBLASLt, which every dense product above a decode's few rows calls: a
/// box without it is refused by name, at the load ([`open`]) and at a call.
fn library(call: &'static str) -> Result<&'static sys::Hipblaslt, Refused> {
    sys::hipblaslt().map_err(|_| Refused {
        call,
        status: ABSENT,
    })
}

fn answered(call: &'static str, status: LtStatus) -> Result<(), Refused> {
    if status == sys::LT_SUCCESS {
        Ok(())
    } else {
        Err(Refused { call, status })
    }
}

/// The workspace a plan may ask for. On gfx950 (MI350) hipBLASLt's
/// heuristic still offers its best algorithm when that one needs more than
/// the cap, reporting a workspace of 0, and the launch then answers 6: a
/// bf16 prefill projection (M=1024 N=2560 K=10240) wants 40-63 MiB there,
/// so a 32 MiB cap refused every gemma-4 and Qwen3.6 prefill on the MI350X.
pub const WORKSPACE: usize = 128 << 20;

/// Row-major `y[m, n] = x[m, k] · w[n, k]ᵀ (+ bias[n])`; with `groups` > 1,
/// that for each group g of a row: `y[m, g·n..][n] = x[m, g·k..][k] ·
/// w[g][n, k]ᵀ`, x a [m, groups·k] and y a [m, groups·n] plane, w the
/// groups' [n, k] blocks one after another (one strided-batched launch).
/// `f32_out` lands y at f32 rather than bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub bias: bool,
    pub groups: u32,
    pub f32_out: bool,
}

struct Plan {
    desc: LtDesc,
    a: LtLayout,
    b: LtLayout,
    d: LtLayout,
    algo: LtHeuristic,
}

// SAFETY: a plan's descriptors are host objects hipBLASLt never mutates once
// built; they are read by any thread that launches the plan.
unsafe impl Send for Plan {}

struct Device {
    handle: Lt,
    plans: HashMap<Shape, Option<Plan>>,
}

// SAFETY: the handle is used under the registry's lock or read-only by
// `hipblasLtMatmul`, which hipBLASLt documents as safe across threads.
unsafe impl Send for Device {}

/// The device bytes hipBLASLt took on its own: its library's code on the
/// first handle, and what each handle keeps (about 26 MiB on gfx942). No
/// call hands that memory to the caller, so it is measured as the device's
/// free bytes before and after each create.
static HELD: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// The device bytes hipBLASLt took on its own since the process began.
#[must_use]
pub fn held_bytes() -> u64 {
    HELD.load(core::sync::atomic::Ordering::Relaxed)
}

/// Runs `create` and adds what the device has fewer free after it to
/// [`held_bytes`].
pub(crate) fn measured<T>(create: impl FnOnce() -> T) -> T {
    let free = || {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals the call writes.
        let _ = unsafe { crate::hip::rt::hipMemGetInfo(&raw mut free, &raw mut total) };
        free as u64
    };
    let before = free();
    let made = create();
    HELD.fetch_add(
        before.saturating_sub(free()),
        core::sync::atomic::Ordering::Relaxed,
    );
    made
}

fn registry() -> &'static Mutex<HashMap<c_int, Device>> {
    static DEVICES: OnceLock<Mutex<HashMap<c_int, Device>>> = OnceLock::new();
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The current device's handle and plans, its handle created the first
/// time.
fn device_of<'a>(
    lt: &sys::Hipblaslt,
    devices: &'a mut HashMap<c_int, Device>,
) -> Result<&'a mut Device, Refused> {
    Ok(match devices.entry(current_device()) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            let mut handle: Lt = core::ptr::null_mut();
            answered(
                "hipblasLtCreate",
                // SAFETY: writes the handle.
                measured(|| unsafe { (lt.hipblasLtCreate)(&raw mut handle) }),
            )?;
            e.insert(Device {
                handle,
                plans: HashMap::new(),
            })
        }
    })
}

/// Creates the current device's hipBLASLt handle now (its device memory,
/// about 26 MiB, taken at create), so no product's first call takes it
/// after the load sized its pools. Refused where hipBLASLt does not load.
pub fn open() -> Result<(), Refused> {
    let lt = library("hipblasLtCreate")?;
    let mut devices = registry().lock().expect("lock poisoned");
    device_of(lt, &mut devices).map(|_| ())
}

fn current_device() -> c_int {
    let mut device: c_int = 0;
    // SAFETY: writes one int.
    unsafe { super::rt::hipGetDevice(&raw mut device) };
    device
}

impl Plan {
    /// Column-major, as hipBLASLt reads it: `yᵀ[n, m] = op_T(W)[n, k] · xᵀ[k, m]`
    /// with W as a k × n matrix (ld k) and x as k × m (ld k).
    fn build(lt: &sys::Hipblaslt, handle: Lt, s: Shape) -> Result<Option<Self>, Refused> {
        let (m, n, k) = (u64::from(s.m), u64::from(s.n), u64::from(s.k));
        let mut desc: LtDesc = core::ptr::null_mut();
        let mut a: LtLayout = core::ptr::null_mut();
        let mut b: LtLayout = core::ptr::null_mut();
        let mut d: LtLayout = core::ptr::null_mut();
        let mut pref: sys::LtPreference = core::ptr::null_mut();
        // SAFETY: each call writes the out-pointer it is handed; attribute
        // values point at locals sized as passed.
        unsafe {
            answered(
                "hipblasLtMatmulDescCreate",
                (lt.hipblasLtMatmulDescCreate)(&raw mut desc, sys::LT_COMPUTE_F32, sys::LT_R_32F),
            )?;
            let set = |attr: c_int, v: &c_int| {
                (lt.hipblasLtMatmulDescSetAttribute)(desc, attr, core::ptr::from_ref(v).cast(), 4)
            };
            answered(
                "hipblasLtMatmulDescSetAttribute",
                set(sys::LT_DESC_TRANSA, &sys::LT_OP_T),
            )?;
            answered(
                "hipblasLtMatmulDescSetAttribute",
                set(sys::LT_DESC_TRANSB, &sys::LT_OP_N),
            )?;
            if s.bias {
                answered(
                    "hipblasLtMatmulDescSetAttribute",
                    set(sys::LT_DESC_EPILOGUE, &sys::LT_EPILOGUE_BIAS),
                )?;
                answered(
                    "hipblasLtMatmulDescSetAttribute",
                    set(sys::LT_DESC_BIAS_DATA_TYPE, &sys::LT_R_16BF),
                )?;
            }
            let g = u64::from(s.groups.max(1));
            answered(
                "hipblasLtMatrixLayoutCreate",
                (lt.hipblasLtMatrixLayoutCreate)(&raw mut a, sys::LT_R_16BF, k, n, k as i64),
            )?;
            answered(
                "hipblasLtMatrixLayoutCreate",
                (lt.hipblasLtMatrixLayoutCreate)(&raw mut b, sys::LT_R_16BF, k, m, (g * k) as i64),
            )?;
            answered(
                "hipblasLtMatrixLayoutCreate",
                (lt.hipblasLtMatrixLayoutCreate)(
                    &raw mut d,
                    if s.f32_out {
                        sys::LT_R_32F
                    } else {
                        sys::LT_R_16BF
                    },
                    n,
                    m,
                    (g * n) as i64,
                ),
            )?;
            if g > 1 {
                // A group's weight block follows the last; its activation and
                // result columns sit k and n along their rows.
                let count = g as i32;
                for (layout, stride) in [(a, n * k), (b, k), (d, n)] {
                    let stride = stride as i64;
                    answered(
                        "hipblasLtMatrixLayoutSetAttribute",
                        (lt.hipblasLtMatrixLayoutSetAttribute)(
                            layout,
                            sys::LT_LAYOUT_BATCH_COUNT,
                            core::ptr::from_ref(&count).cast(),
                            4,
                        ),
                    )?;
                    answered(
                        "hipblasLtMatrixLayoutSetAttribute",
                        (lt.hipblasLtMatrixLayoutSetAttribute)(
                            layout,
                            sys::LT_LAYOUT_STRIDED_BATCH_OFFSET,
                            core::ptr::from_ref(&stride).cast(),
                            8,
                        ),
                    )?;
                }
            }
            answered(
                "hipblasLtMatmulPreferenceCreate",
                (lt.hipblasLtMatmulPreferenceCreate)(&raw mut pref),
            )?;
            let ws: u64 = WORKSPACE as u64;
            answered(
                "hipblasLtMatmulPreferenceSetAttribute",
                (lt.hipblasLtMatmulPreferenceSetAttribute)(
                    pref,
                    sys::LT_PREF_MAX_WORKSPACE_BYTES,
                    core::ptr::from_ref(&ws).cast(),
                    8,
                ),
            )?;
            let mut algo = LtHeuristic::default();
            let mut returned: c_int = 0;
            let status = (lt.hipblasLtMatmulAlgoGetHeuristic)(
                handle,
                desc,
                a,
                b,
                d,
                d,
                pref,
                1,
                &raw mut algo,
                &raw mut returned,
            );
            (lt.hipblasLtMatmulPreferenceDestroy)(pref);
            if status != sys::LT_SUCCESS || returned < 1 || algo.state != sys::LT_SUCCESS {
                (lt.hipblasLtMatmulDescDestroy)(desc);
                for layout in [a, b, d] {
                    (lt.hipblasLtMatrixLayoutDestroy)(layout);
                }
                // A heuristic that fails is a fault; one that answers no
                // algorithm says the shape is not hipBLASLt's.
                answered("hipblasLtMatmulAlgoGetHeuristic", status)?;
                return Ok(None);
            }
            Ok(Some(Self {
                desc,
                a,
                b,
                d,
                algo,
            }))
        }
    }
}

/// The waves the heuristic's first algorithm for `s` fills on the current
/// device: its workgroups (the tiles of the macro tile hipBLASLt's logic
/// states for that algorithm's solution, times its split of K) over those
/// the device holds at once (its compute units times the workgroups of that
/// kernel one holds, by HIP's occupancy calculator; [`super::lt_kernel`]).
/// hipBLASLt 1.4 states no `wavesCount` itself.
///
/// A solution that leaves its split of K to the launch (`globalSplitU` -1)
/// states no split, and hipBLASLt answers none: its split from a wave is
/// taken as one by inference, not stated. Of 24 launches traced with
/// rocprofv3 on gfx942, 23 launched exactly their tiles (256 to 1728) and
/// the one under a wave (96 tiles) three slices each. Below a wave the
/// waves are therefore not stated: `Ok(None)`, as where hipBLASLt offers no
/// algorithm. `Err` where the solution could
/// not be read, and why ([`super::lt_kernel::pricing`] states it).
pub fn waves(s: Shape) -> Result<Option<f64>, String> {
    const CALL: &str = "hipblasLtMatmulAlgoGetHeuristic";
    let lt = library(CALL).map_err(|e| e.to_string())?;
    let ordinal = current_device();
    let mut algo = {
        let mut devices = registry().lock().expect("lock poisoned");
        let dev = device_of(lt, &mut devices).map_err(|e| e.to_string())?;
        let handle = dev.handle;
        let plan = match dev.plans.get(&s) {
            Some(plan) => plan,
            None => {
                let plan = Plan::build(lt, handle, s).map_err(|e| e.to_string())?;
                crate::tally::built();
                dev.plans.entry(s).or_insert(plan)
            }
        };
        match plan {
            Some(plan) => plan.algo.algo,
            None => return Ok(None),
        }
    };
    let super::lt_kernel::Stated { solution, resident } =
        super::lt_kernel::stated(ordinal, &mut algo)?;
    let [rows, cols] = solution.tile;
    let tiles =
        u64::from(s.n.div_ceil(rows)) * u64::from(s.m.div_ceil(cols)) * u64::from(s.groups.max(1));
    let cus = super::multiprocessors(ordinal.unsigned_abs()).map_err(|e| e.to_string())?;
    let held = u64::from(cus) * u64::from(resident);
    let slices = match solution.split {
        Some(split) => u64::from(split),
        None if tiles >= held => 1,
        None => return Ok(None),
    };
    Ok(Some((tiles * slices) as f64 / held as f64))
}

/// Enqueues `y = x · wᵀ (+ bias)` on `stream`. `Ok(false)` when hipBLASLt
/// offers no algorithm for the shape: the shape's route is then the
/// caller's other one (rocBLAS, the routed select), every fire of it alike.
/// The plan, and with it the route, is the shape's, made where the load's
/// size-only walk first reaches it (every rung of the lattice, those later
/// stated eager too), so no serving fire chooses one. A hipBLASLt that did
/// not load is refused.
///
/// # Safety
/// `x`, `w`, `y` (and `bias` when the shape has one) must be live device
/// ranges of [m, k], [n, k], [m, n] (and [n]) bf16, and `workspace` one of
/// [`WORKSPACE`] bytes no other stream uses until this launch completes.
pub unsafe fn matmul(
    s: Shape,
    x: u64,
    w: u64,
    bias: u64,
    y: u64,
    workspace: u64,
    stream: Stream,
) -> Result<bool, Refused> {
    const CALL: &str = "hipblasLtMatmul";
    let lt = library(CALL)?;
    let mut devices = registry().lock().expect("lock poisoned");
    let dev = device_of(lt, &mut devices)?;
    let handle = dev.handle;
    let plan = match dev.plans.get(&s) {
        Some(plan) => plan,
        None => {
            let plan = Plan::build(lt, handle, s)?;
            crate::tally::built();
            dev.plans.entry(s).or_insert(plan)
        }
    };
    let Some(plan) = plan else {
        return Ok(false);
    };
    if s.bias {
        let p = bias as usize as *const c_void;
        // SAFETY: the descriptor is this shape's own; the pointer value is
        // read at the call below, under the registry's lock.
        answered("hipblasLtMatmulDescSetAttribute", unsafe {
            (lt.hipblasLtMatmulDescSetAttribute)(
                plan.desc,
                sys::LT_DESC_BIAS_POINTER,
                core::ptr::from_ref(&p).cast(),
                8,
            )
        })?;
    }
    // A size-only walk (`crate::sizing`) plans the product and runs none.
    if crate::sizing::active() {
        crate::sizing::counted();
        return Ok(true);
    }
    let (alpha, beta) = (1f32, 0f32);
    // SAFETY: the caller's ranges match the plan's shape; alpha and beta are
    // host f32 as the f32 scale type reads them.
    answered(CALL, unsafe {
        (lt.hipblasLtMatmul)(
            handle,
            plan.desc,
            core::ptr::from_ref(&alpha).cast(),
            w as usize as *const c_void,
            plan.a,
            x as usize as *const c_void,
            plan.b,
            core::ptr::from_ref(&beta).cast(),
            y as usize as *const c_void,
            plan.d,
            y as usize as *mut c_void,
            plan.d,
            plan.algo.algo.as_ptr().cast(),
            workspace as usize as *mut c_void,
            WORKSPACE,
            stream,
        )
    })?;
    Ok(true)
}
