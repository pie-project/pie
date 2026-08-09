//! Which template argument forms can a row's `elem` actually spell?
//!
//! `DeviceKernel::instantiation` builds the name expression as
//! `::pie_cuda_driver::kernels::{template_path}<::pie_cuda_driver::kernels::{elem}>`
//! — one qualification, glued to the FRONT of the whole `elem` string. That
//! detail decides what a row can name, and three agents have now guessed at it
//! in three different directions. This asks the compiler.
//!
//! # The twelfth case is a different question, and it is the one that decides
//! a fix
//!
//! Cases 1-11 all ask *"can slot N be spelled like this"*, which presumes
//! there ARE slots. Three independent agents then hit the case with NO slots:
//! `attn::pack_dense_mask`, `attn::pack_structured_mask`, `attn::write_mla`,
//! `quant::wna16_gate_up_decode` and `quant::wna16_down_decode` are plain
//! `__global__`s, and every one of them was refused with the same sentence —
//! *"`instantiation()` always emits `path<...>`, so a plain `__global__`
//! cannot be named at all"*. That sentence is a claim about a Rust `format!`
//! and it is TRUE. The claim underneath it — that a plain `__global__` is
//! therefore unnameable — is a claim about NVRTC, and nobody had asked NVRTC.
//!
//! So case 12 asks it: `nvrtcAddNameExpression` on a bare qualified path with
//! no `<>` at all. And because an accepted expression that lowers to a name
//! the DRIVER will not resolve is worth nothing, [`resolution`] takes the
//! measurement the whole way — compile, `nvrtcGetCUBIN`, `cuModuleLoadData`,
//! `cuModuleGetFunction` — for a plain kernel and for a template one as the
//! control. A lowered name is a string; a `CUfunction` is a kernel.
fn main() {
    let src = SRC;
    // Each case is what `instantiation()` would produce for that `elem`.
    for (what, expr) in CASES {
        println!("{:<46} {}", what, compile(src, expr));
    }
    println!();
    resolution();
}

/// The probe's device text: one `__global__` per argument form under test.
const SRC: &str = r#"
namespace pie_cuda_driver { namespace kernels {
namespace device {
struct bf16 { unsigned short raw; };
struct false_type { static constexpr bool value = false; };
struct true_type { static constexpr bool value = true; };
constexpr int kBlock256 = 256;
}
namespace probe {
template <class T, int BLOCK = 256>
__global__ void scaled(T* out, int n) { if ((int)threadIdx.x < n) out[0] = out[0]; }
template <class T, bool FLAG>
__global__ void flagged(T* out, int n) { if (FLAG && (int)threadIdx.x < n) out[0] = out[0]; }
template <int BLOCK>
__global__ void sized(int* out, int n) { if ((int)threadIdx.x < n) out[0] = BLOCK; }
template <class T, int BLOCK>
__global__ void sizedT(T* out, int n) { if ((int)threadIdx.x < n) out[0] = out[0]; }
template <bool A, bool B>
__global__ void flags(int* out) { if (A || B) out[0] = 1; }
template <bool HND>
__global__ void oneflag(int* out) { out[0] = HND ? 1 : 0; }
// No template parameters at all -- the shape five refusals across three
// agents said could not be named. Two of them, because the second question
// is whether OVERLOAD ambiguity is what actually bites: a bare path names a
// FUNCTION rather than an instantiation, and a function can be overloaded.
__global__ void plain(int* out, int n) { if ((int)threadIdx.x < n) out[0] = 7; }
__global__ void plain_second(int* out) { out[0] = 9; }
}}}
"#;
/// Each case is what `instantiation()` would produce for that `elem`.
const CASES: &[(&str, &str)] = &[
    ("elem = \"device::bf16\"",
     "::pie_cuda_driver::kernels::probe::scaled<::pie_cuda_driver::kernels::device::bf16>"),
    ("elem = \"device::bf16, 256\"  (the form in use)",
     "::pie_cuda_driver::kernels::probe::scaled<::pie_cuda_driver::kernels::device::bf16, 256>"),
    ("elem = \"device::bf16, true\"",
     "::pie_cuda_driver::kernels::probe::flagged<::pie_cuda_driver::kernels::device::bf16, true>"),
    ("a bare leading non-type  elem = \"256\"",
     "::pie_cuda_driver::kernels::probe::sized<::pie_cuda_driver::kernels::256>"),
    ("slot 1 via a constexpr    elem = \"device::kBlock256\"",
     "::pie_cuda_driver::kernels::probe::sized<::pie_cuda_driver::kernels::device::kBlock256>"),
    ("slot 1 via a bool member  elem = \"device::false_type::value, false\"",
     "::pie_cuda_driver::kernels::probe::flags<::pie_cuda_driver::kernels::device::false_type::value, false>"),
    ("slot 2, named UNqualified elem = \"device::bf16, device::kBlock256\"",
     "::pie_cuda_driver::kernels::probe::sizedT<::pie_cuda_driver::kernels::device::bf16, device::kBlock256>"),
    ("slot 2, named FULLY qual  (the fix)",
     "::pie_cuda_driver::kernels::probe::sizedT<::pie_cuda_driver::kernels::device::bf16, ::pie_cuda_driver::kernels::device::kBlock256>"),
    // The shape the nine `template <bool>` rows need: ONE parameter, and
    // it is the flag, so the flag lands in slot 1 — the prefixed one.
    ("a bare leading bool       elem = \"true\"",
     "::pie_cuda_driver::kernels::probe::oneflag<::pie_cuda_driver::kernels::true>"),
    ("the #hnd arm              elem = \"device::true_type::value\"",
     "::pie_cuda_driver::kernels::probe::oneflag<::pie_cuda_driver::kernels::device::true_type::value>"),
    ("the #nhd arm              elem = \"device::false_type::value\"",
     "::pie_cuda_driver::kernels::probe::oneflag<::pie_cuda_driver::kernels::device::false_type::value>"),
    // Case 12: NO argument list at all. Five rows across three agents were
    // refused on the belief that this cannot be written.
    ("NO argument list          (a plain __global__)",
     "::pie_cuda_driver::kernels::probe::plain"),
    // And the two ways to get it wrong, so the refusal that stays is the
    // right one: an empty list is not "no list", and a plain kernel does not
    // grow one.
    ("an EMPTY argument list    `path<>`",
     "::pie_cuda_driver::kernels::probe::plain<>"),
    ("a plain kernel WITH a list",
     "::pie_cuda_driver::kernels::probe::plain<::pie_cuda_driver::kernels::device::bf16>"),
    // The mirror mistake: a TEMPLATE stated as if it were plain. This is the
    // one `DeviceKernel::PLAIN` exists to keep unwritable, and NVRTC refusing
    // it is what makes `tests/units.rs` the check rather than a convention.
    ("a TEMPLATE with no list",
     "::pie_cuda_driver::kernels::probe::oneflag"),
    // A second plain kernel, to check the answer is not an accident of there
    // being exactly one.
    ("a second plain __global__",
     "::pie_cuda_driver::kernels::probe::plain_second"),
];

/// The whole way to a `CUfunction`: does a lowered name the compiler hands
/// back for a BARE path actually resolve on the device?
///
/// `nvrtcAddNameExpression` accepting a string proves nothing on its own —
/// the expression could lower to a name `cuModuleGetFunction` has never heard
/// of, which is exactly the failure `runtime::module` puts at load rather
/// than at first fire. The control is the template kernel beside it: if the
/// plain one resolves and the template one does too, the bare path is not a
/// weaker kind of name, it is the same kind.
fn resolution() {
    use cudarc::driver::sys as dr;
    use cudarc::nvrtc::sys as nv;
    use std::ffi::{CStr, CString};

    let plain = "::pie_cuda_driver::kernels::probe::plain";
    let template = "::pie_cuda_driver::kernels::probe::oneflag<::pie_cuda_driver::kernels::device::true_type::value>";

    let s = CString::new(SRC).unwrap();
    let n = CString::new("probe.cu").unwrap();
    let mut p: nv::nvrtcProgram = std::ptr::null_mut();
    unsafe {
        nv::nvrtcCreateProgram(&raw mut p, s.as_ptr(), n.as_ptr(), 0, std::ptr::null(), std::ptr::null())
    };
    let exprs: Vec<CString> = [plain, template].iter().map(|e| CString::new(*e).unwrap()).collect();
    for e in &exprs {
        unsafe { nv::nvrtcAddNameExpression(p, e.as_ptr()) };
    }
    let a = CString::new("--gpu-architecture=sm_89").unwrap();
    let c = CString::new("-std=c++17").unwrap();
    let opts = [a.as_ptr(), c.as_ptr()];
    if unsafe { nv::nvrtcCompileProgram(p, 2, opts.as_ptr()) } != nv::nvrtcResult::NVRTC_SUCCESS {
        println!("resolution: the probe source did not compile");
        return;
    }
    let mut lowered = Vec::new();
    for e in &exprs {
        let mut low: *const i8 = std::ptr::null();
        let r = unsafe { nv::nvrtcGetLoweredName(p, e.as_ptr(), &raw mut low) };
        if r != nv::nvrtcResult::NVRTC_SUCCESS || low.is_null() {
            println!("resolution: {} has no lowered name", e.to_string_lossy());
            return;
        }
        lowered.push(unsafe { CStr::from_ptr(low) }.to_string_lossy().into_owned());
    }
    let mut size = 0usize;
    unsafe { nv::nvrtcGetCUBINSize(p, &raw mut size) };
    let mut cubin = vec![0u8; size];
    unsafe { nv::nvrtcGetCUBIN(p, cubin.as_mut_ptr().cast()) };

    unsafe { dr::cuInit(0) };
    let mut ctx: dr::CUcontext = std::ptr::null_mut();
    if unsafe { dr::cuDevicePrimaryCtxRetain(&raw mut ctx, 0) } != dr::CUresult::CUDA_SUCCESS {
        println!("resolution: no device, so the driver half is unmeasured");
        return;
    }
    unsafe { dr::cuCtxSetCurrent(ctx) };
    let mut module: dr::CUmodule = std::ptr::null_mut();
    if unsafe { dr::cuModuleLoadData(&raw mut module, cubin.as_ptr().cast()) }
        != dr::CUresult::CUDA_SUCCESS
    {
        println!("resolution: the cubin did not load");
        return;
    }
    println!("cuModuleGetFunction against the names NVRTC lowered to:");
    for (expr, mangled) in [plain, template].iter().zip(&lowered) {
        let name = CString::new(mangled.as_str()).unwrap();
        let mut f: dr::CUfunction = std::ptr::null_mut();
        let code = unsafe { dr::cuModuleGetFunction(&raw mut f, module, name.as_ptr()) };
        let verdict = if code == dr::CUresult::CUDA_SUCCESS && !f.is_null() {
            "RESOLVED"
        } else {
            "NOT FOUND"
        };
        println!("  {verdict:<10} {mangled}\n             from {expr}");
    }
    unsafe { dr::cuModuleUnload(module) };
    unsafe { dr::cuDevicePrimaryCtxRelease_v2(0) };
}

fn compile(src: &str, expr: &str) -> String {
    use cudarc::nvrtc::sys as nv;
    use std::ffi::{CStr, CString};
    let s = CString::new(src).unwrap();
    let n = CString::new("probe.cu").unwrap();
    let mut p: nv::nvrtcProgram = std::ptr::null_mut();
    unsafe {
        nv::nvrtcCreateProgram(&raw mut p, s.as_ptr(), n.as_ptr(), 0, std::ptr::null(), std::ptr::null())
    };
    let e = CString::new(expr).unwrap();
    unsafe { nv::nvrtcAddNameExpression(p, e.as_ptr()) };
    let a = CString::new("--gpu-architecture=sm_89").unwrap();
    let c = CString::new("-std=c++17").unwrap();
    let opts = [a.as_ptr(), c.as_ptr()];
    let code = unsafe { nv::nvrtcCompileProgram(p, 2, opts.as_ptr()) };
    if code != nv::nvrtcResult::NVRTC_SUCCESS {
        let mut sz = 0;
        unsafe { nv::nvrtcGetProgramLogSize(p, &raw mut sz) };
        let mut log = vec![0u8; sz.max(1)];
        unsafe { nv::nvrtcGetProgramLog(p, log.as_mut_ptr().cast()) };
        let log = String::from_utf8_lossy(&log);
        let first = log.lines().find(|l| l.contains("error")).unwrap_or("(no diagnosis)");
        return format!("REFUSED  {}", first.trim());
    }
    let mut low: *const i8 = std::ptr::null();
    let r = unsafe { nv::nvrtcGetLoweredName(p, e.as_ptr(), &raw mut low) };
    if r != nv::nvrtcResult::NVRTC_SUCCESS || low.is_null() {
        return "compiled, but NO lowered name".into();
    }
    format!("OK       {}", unsafe { CStr::from_ptr(low) }.to_string_lossy())
}
