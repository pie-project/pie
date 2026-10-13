//! The hipBLASLt matmul the dense projections take: bf16 operands, f32
//! accumulation, an optional bf16 bias folded into the epilogue. A problem's
//! descriptors and the heuristic's first algorithm are built once per
//! (device, shape, bias) and kept for the process: the planning is host
//! work, so a captured step replays only the launch.

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
        write!(f, "`{}` answered {}", self.call, self.status)
    }
}

fn answered(call: &'static str, status: LtStatus) -> Result<(), Refused> {
    if status == sys::LT_SUCCESS {
        Ok(())
    } else {
        Err(Refused { call, status })
    }
}

/// The workspace a plan may ask for; the heuristic only offers algorithms
/// that fit it.
pub const WORKSPACE: usize = 32 << 20;

/// Row-major `y[m, n] = x[m, k] · w[n, k]ᵀ (+ bias[n])`; with `groups` > 1,
/// that for each group g of a row: `y[m, g·n..][n] = x[m, g·k..][k] ·
/// w[g][n, k]ᵀ`, x a [m, groups·k] and y a [m, groups·n] plane, w the
/// groups' [n, k] blocks one after another (one strided-batched launch).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub bias: bool,
    pub groups: u32,
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

fn registry() -> &'static Mutex<HashMap<c_int, Device>> {
    static DEVICES: OnceLock<Mutex<HashMap<c_int, Device>>> = OnceLock::new();
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
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
                (lt.hipblasLtMatrixLayoutCreate)(&raw mut d, sys::LT_R_16BF, n, m, (g * n) as i64),
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

/// Enqueues `y = x · wᵀ (+ bias)` on `stream`. `Ok(false)` when hipBLASLt is
/// absent or offers no algorithm for the shape, so the caller takes another
/// tactic.
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
    let Ok(lt) = sys::hipblaslt() else {
        return Ok(false);
    };
    let device = current_device();
    let mut devices = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dev = match devices.entry(device) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            let mut handle: Lt = core::ptr::null_mut();
            // SAFETY: writes the handle.
            answered("hipblasLtCreate", unsafe {
                (lt.hipblasLtCreate)(&raw mut handle)
            })?;
            e.insert(Device {
                handle,
                plans: HashMap::new(),
            })
        }
    };
    let handle = dev.handle;
    let plan = match dev.plans.get(&s) {
        Some(plan) => plan,
        None => {
            let plan = Plan::build(lt, handle, s)?;
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

/// Every algorithm the heuristic offers for `s` (up to `count`), each timed
/// over `reps` launches on `stream`: (index, mean µs, workspace bytes). The
/// launches cycle through the copies in `w`, so copies past the last-level
/// cache time a cold weight, as a decode step streams it.
///
/// # Safety
/// As [`matmul`], for every copy in `w`; the stream must not be capturing.
#[allow(clippy::too_many_arguments)]
pub unsafe fn survey(
    s: Shape,
    x: u64,
    w: &[u64],
    y: u64,
    workspace: u64,
    stream: Stream,
    count: usize,
    reps: u32,
) -> Result<Vec<(usize, f64, usize)>, Refused> {
    let Ok(lt) = sys::hipblaslt() else {
        return Ok(Vec::new());
    };
    if w.is_empty() {
        return Ok(Vec::new());
    }
    let mut handle: Lt = core::ptr::null_mut();
    // SAFETY: writes the handle.
    answered("hipblasLtCreate", unsafe {
        (lt.hipblasLtCreate)(&raw mut handle)
    })?;
    let Some(plan) = Plan::build(lt, handle, s)? else {
        return Ok(Vec::new());
    };
    let mut pref: sys::LtPreference = core::ptr::null_mut();
    let mut algos = vec![LtHeuristic::default(); count];
    let mut returned: c_int = 0;
    // SAFETY: the plan's descriptors are live; the out-array holds `count`.
    unsafe {
        answered(
            "hipblasLtMatmulPreferenceCreate",
            (lt.hipblasLtMatmulPreferenceCreate)(&raw mut pref),
        )?;
        let ws: u64 = WORKSPACE as u64;
        (lt.hipblasLtMatmulPreferenceSetAttribute)(
            pref,
            sys::LT_PREF_MAX_WORKSPACE_BYTES,
            core::ptr::from_ref(&ws).cast(),
            8,
        );
        answered(
            "hipblasLtMatmulAlgoGetHeuristic",
            (lt.hipblasLtMatmulAlgoGetHeuristic)(
                handle,
                plan.desc,
                plan.a,
                plan.b,
                plan.d,
                plan.d,
                pref,
                count as c_int,
                algos.as_mut_ptr(),
                &raw mut returned,
            ),
        )?;
    }
    let (alpha, beta) = (1f32, 0f32);
    let mut out = Vec::new();
    for (index, algo) in algos.iter().take(returned.max(0) as usize).enumerate() {
        if algo.state != sys::LT_SUCCESS {
            continue;
        }
        let launch = |w: u64| unsafe {
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
                algo.algo.as_ptr().cast(),
                workspace as usize as *mut c_void,
                WORKSPACE,
                stream,
            )
        };
        if w.iter().any(|&w| launch(w) != sys::LT_SUCCESS) {
            continue;
        }
        let mut ms = f32::NAN;
        // SAFETY: two fresh events on a live stream, destroyed here.
        unsafe {
            let (mut start, mut stop): (super::rt::Event, super::rt::Event) =
                (core::ptr::null_mut(), core::ptr::null_mut());
            super::rt::hipEventCreate(&raw mut start);
            super::rt::hipEventCreate(&raw mut stop);
            super::rt::hipEventRecord(start, stream.cast());
            for r in 0..reps as usize {
                launch(w[r % w.len()]);
            }
            super::rt::hipEventRecord(stop, stream.cast());
            super::rt::hipEventSynchronize(stop);
            super::rt::hipEventElapsedTime(&raw mut ms, start, stop);
            super::rt::hipEventDestroy(start);
            super::rt::hipEventDestroy(stop);
        }
        out.push((index, f64::from(ms) * 1e3 / f64::from(reps), algo.workspace));
    }
    Ok(out)
}
