//! The compiler's shared-memory number, against the host's.
//!
//! Three kernel families size their shared memory from a C++ `sizeof` that the
//! host must know BEFORE the launch, because it is what
//! `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` is given and what
//! `cuLaunchKernelEx` is passed. Upstream reads it back out of the cubin;
//! under this crate the host derives it in Rust instead — twice as a literal
//! (XQA, MLA) and once as a re-derivation of the C++ layout arithmetic (FA2
//! prefill).
//!
//! A host number that is too small is a kernel that runs off the end of its
//! shared memory. A host number that is too large is, for MLA, worse than
//! that: `arm_for` picks the widest arm the device's budget admits, so an
//! overstated size silently selects a NARROWER arm and the attention runs
//! correctly and slowly, forever.
//!
//! So each family exports the compiler's own answer as a `__device__`
//! variable, and this file reads it back and compares. The three echo symbols
//! were written when the sources were ported and have had no reader since:
//! `pie_xqa_smem_size` (`attention_xqa_mha.cuh:269`),
//! `smem_bytes_paged<KTraits>` (`fa2.cuh:239`) and `smem_bytes_mla<KTraits>`
//! (`attention_mla_fa2.cuh:211`). This is that reader.
//!
//! # The question `attention_xqa_mha.cuh` could not answer
//!
//! That header records the re-export coming out of NVRTC as a plain `.global`
//! with no `.visible`, and says:
//!
//! > Whether `cuModuleGetGlobal` resolves a non-`.visible` `.global` could not
//! > be answered here: it needs a CUDA context, and the brief this file was
//! > written under forbids creating one.
//!
//! It does. That is measured here rather than asserted: if it ever stops
//! being true, [`echo`] returns `None` for every name and the assertions below
//! say so instead of passing vacuously.

#![cfg(feature = "_cuda")]

use std::collections::BTreeMap;
use std::ffi::CString;

use cudarc::driver::sys as dr;
use kernels_cuda::jit::{Root, Toolchain, cache};
use kernels_cuda::jit::nvrtc;

/// Compile `root` asking for `wanted`, then read each name back as a `u32`
/// device global.
///
/// The name expressions must be handed to the COMPILE, not just looked up
/// afterwards: `nvrtcAddNameExpression` is what instantiates a variable
/// template and what makes its lowered name knowable. Two of the three echoes
/// are templates, so without this they would not exist in the cubin at all.
///
/// `None` for a name that resolved to no global, which is how a failure to
/// read one is distinguished from a wrong value.
fn echo(root: &Root, wanted: &[String]) -> BTreeMap<String, Option<u32>> {
    let arch = cache::arch().expect("a device");
    let job = nvrtc::Job {
        name: root.name,
        source: root.text.to_owned(),
        arch,
        options: root.options,
        headers: root.header_set(),
        floor: Toolchain::ANY,
        wanted,
        device_link: root.needs_device_runtime(),
    };
    let built = nvrtc::compile_text(&job)
        .unwrap_or_else(|why| panic!("`{}` will not compile: {why}", root.name));
    assert_eq!(built.lowered.len(), wanted.len(), "one lowered name per name expression");

    let mut module: dr::CUmodule = std::ptr::null_mut();
    // SAFETY: `cubin` is the image NVRTC just produced and outlives the call.
    let code = unsafe { dr::cuModuleLoadData(&raw mut module, built.cubin.as_ptr().cast()) };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "`{}` will not load", root.name);

    let mut out = BTreeMap::new();
    for (asked, lowered) in wanted.iter().zip(&built.lowered) {
        let name = CString::new(lowered.as_str()).expect("a lowered name has no NUL");
        let (mut ptr, mut size) = (0 as dr::CUdeviceptr, 0usize);
        // SAFETY: `module` is loaded and both out-parameters are live.
        let code = unsafe {
            dr::cuModuleGetGlobal_v2(&raw mut ptr, &raw mut size, module, name.as_ptr())
        };
        if code != dr::CUresult::CUDA_SUCCESS || size != 4 {
            out.insert(asked.clone(), None);
            continue;
        }
        let mut value = 0u32;
        // SAFETY: the global is four bytes, which `size` just confirmed.
        let code = unsafe { dr::cuMemcpyDtoH_v2(std::ptr::from_mut(&mut value).cast(), ptr, 4) };
        out.insert(asked.clone(), (code == dr::CUresult::CUDA_SUCCESS).then_some(value));
    }
    out
}

/// A context, or a stated reason there is none.
fn ready(what: &str) -> bool {
    if cache::arch().is_none() {
        eprintln!("SKIP {what}: no CUDA device is current");
        return false;
    }
    match cache::bind_context() {
        Ok(()) => true,
        Err(why) => {
            eprintln!("SKIP {what}: no usable context ({why})");
            false
        }
    }
}

/// XQA's `sizeof(SharedMem)`, in all five lattice members.
///
/// One host constant covers five compiles, which is a claim in itself:
/// `attention_xqa_mha.cuh` records that the size "depends on neither the head
/// group nor the page size", measured across the lattice. If a `-D` set ever
/// makes that false, one member disagrees and this names which.
#[test]
fn xqa_shared_memory_is_what_the_host_states() {
    if !ready("xqa_shared_memory_is_what_the_host_states") {
        return;
    }
    use kernels_cuda::attn::xqa;

    // `&`, even though the re-export is `extern "C"` and its lowered name is
    // its source name: `nvrtcAddNameExpression` wants a variable's ADDRESS.
    // Handing it the bare name is "expression must have a constant value".
    let name = "&pie_xqa_smem_size".to_owned();
    let mut wrong = Vec::new();
    for root in &xqa::ROOTS {
        match echo(root, std::slice::from_ref(&name))[&name] {
            Some(bytes) if bytes == xqa::XQA_SMEM_BYTES => {}
            answer => wrong.push((root.name, answer)),
        }
    }
    assert!(
        wrong.is_empty(),
        "XQA_SMEM_BYTES is {}, and {} lattice member(s) disagree: {wrong:?}",
        xqa::XQA_SMEM_BYTES,
        wrong.len()
    );
}

/// MLA's three `DISPATCH_SMEM_CONFIG` arms, against their three literals.
///
/// The narrowest check to write and the most consequential to have. The Rust
/// side is three numbers copied out of upstream's own comparisons; nothing
/// re-derives them, so nothing would notice upstream changing `SharedStorage`.
/// The failure that produces is not a crash — `arm_for` compares the stale
/// literal against the device budget and picks an arm, and the launch is sized
/// by a `sizeof` that no longer matches it.
#[test]
fn mla_arm_sizes_are_what_the_host_states() {
    if !ready("mla_arm_sizes_are_what_the_host_states") {
        return;
    }
    use kernels_cuda::attn::mla_fa2;

    let wanted: Vec<String> = mla_fa2::SMEM_ECHO.iter().map(|&e| e.to_owned()).collect();
    let read = echo(&mla_fa2::ROOT, &wanted);

    let mut wrong = Vec::new();
    for (arm, asked) in mla_fa2::ARMS.iter().zip(&wanted) {
        match read[asked] {
            Some(bytes) if bytes == arm.smem => {}
            answer => wrong.push((arm.cta_tile_kv, arm.smem, answer)),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of three MLA arms disagree with `sizeof(KTraits::SharedStorage)` \
         — (cta_tile_kv, host says, device says): {wrong:?}",
        wrong.len()
    );
}

/// FA2 prefill's re-derived layout, against the compiler's `sizeof`.
///
/// The one echo that checks arithmetic rather than a literal.
/// `PrefillGeometry::derive` reimplements `BatchPrefillWithPagedKVCacheKernel`'s
/// `constexpr` prologue in Rust — the staged buffers, the tail, the tile
/// counts — so that a launch can be sized without compiling first. Thirty-six
/// lattice points, ten arms each; every arm's traits pack is a different
/// `SharedStoragePaged`, and the host derivation claims one number covers all
/// ten of a point's arms.
#[test]
fn fa2_prefill_derivation_matches_the_compiler() {
    if !ready("fa2_prefill_derivation_matches_the_compiler") {
        return;
    }
    use kernels_cuda::attn::fa2::geometry::{Device, KvWidth, PrefillGeometry};
    use kernels_cuda::attn::fa2;

    // The traits pack is already spelled inside each arm's template-id, so the
    // echo is built from the arm rather than restated.
    //
    // `BatchPrefillWithPagedKVCacheKernel<KTraits, Params>` takes two
    // arguments and `smem_bytes_paged<KTraits>` takes one, so this is the
    // first of the two — up to the comma at depth zero, since `KTraits` is
    // itself a `PagedTraits<..>` full of commas. The trailing space keeps
    // `>>` out of the name expression.
    fn echo_for(arm: &str) -> String {
        let open = arm.find('<').expect("an arm names a traits pack");
        let (mut depth, mut end) = (0usize, arm.len() - 1);
        for (at, c) in arm[open..].char_indices() {
            match c {
                '<' => depth += 1,
                '>' => depth -= 1,
                ',' if depth == 1 => {
                    end = open + at;
                    break;
                }
                _ => {}
            }
        }
        format!("{}<{} >", PrefillGeometry::ECHO_TEMPLATE, &arm[open + 1..end])
    }

    // A lattice point is reached by a DEVICE, not asked for: `x::fa2::prefill`
    // derives a geometry and then looks up `prefill_root(hd, q,
    // geometry.num_mma_kv)`, so the tile count is chosen by the shared-memory
    // budget and never by the caller. The lattice carries a point per tile
    // count because a smaller part picks a smaller one — which means checking
    // every point takes a budget for each, not one part's.
    //
    // The budget is swept rather than spelled as a list of real GPUs: what
    // selects a point is the number, and a sweep says "some part with this
    // much shared memory selects it" without claiming that part exists.
    /// The smallest budget, in KiB per SM, whose derivation lands on `point`.
    fn budget_reaching(point: &fa2::PrefillRoot) -> Option<(u32, PrefillGeometry)> {
        (16..=2048).find_map(|kib| {
            let dev = Device {
                cc_major: 8,
                max_smem_per_sm: kib * 1024,
                max_smem_per_block_optin: kib * 1024,
            };
            PrefillGeometry::derive(point.head_dim, point.cta_tile_q, KvWidth::BF16, false, dev)
                .ok()
                .filter(|g| g.num_mma_kv == point.num_mma_kv)
                .map(|g| (kib, g))
        })
    }

    /// `cudaDevAttrMaxSharedMemoryPerMultiprocessor` on the largest part that
    /// exists, in KiB: Hopper's 228. A point needing more than this is
    /// compiled for no hardware, which is a fact about today and so is
    /// reported rather than asserted.
    const LARGEST_REAL_PART: u32 = 228;

    /// The lattice point no shared-memory budget can select, and why.
    ///
    /// `max_mma_kv_reg` is `8 / num_mma_q`, and `num_mma_q` is 2 for every
    /// `cta_tile_q > 64`, so a `q128` point is capped at `NUM_MMA_KV = 4` by
    /// REGISTERS — shared memory never enters it. Every other `q128` family in
    /// the lattice stops at `kv4` accordingly; this one entry does not, which
    /// is what makes it look like a transcription and not a decision.
    const UNSELECTABLE: &[&str] = &["hd64 q128 kv8"];

    let (mut wrong, mut checked, mut unreachable) = (Vec::new(), 0usize, Vec::new());
    let mut beyond_real = Vec::new();
    for point in &fa2::PREFILL {
        let at = format!("hd{} q{} kv{}", point.head_dim, point.cta_tile_q, point.num_mma_kv);
        let Some((kib, host)) = budget_reaching(point) else {
            unreachable.push(at);
            continue;
        };
        if kib > LARGEST_REAL_PART {
            beyond_real.push(format!("{at} (needs {kib} KiB/SM)"));
        }

        let wanted: Vec<String> = point.arms.iter().map(|&a| echo_for(a)).collect();
        let read = echo(&point.root, &wanted);
        for (nth, asked) in wanted.iter().enumerate() {
            checked += 1;
            match read[asked] {
                Some(bytes) if bytes == host.smem_bytes => {}
                answer => wrong.push(format!(
                    "{at} arm{nth}: host derives {}, compiler says {answer:?}",
                    host.smem_bytes
                )),
            }
        }
    }

    eprintln!("{checked} prefill arms checked over {} points", fa2::PREFILL.len());
    if !beyond_real.is_empty() {
        eprintln!(
            "{} point(s) are selected only above {LARGEST_REAL_PART} KiB/SM, so no \
             part that exists reaches them today: {beyond_real:?}",
            beyond_real.len()
        );
    }
    assert_eq!(
        unreachable, UNSELECTABLE,
        "the set of lattice points no budget selects has changed — see \
         `UNSELECTABLE` for the one that is expected and why"
    );
    assert!(
        wrong.is_empty(),
        "{} of {checked} prefill arms disagree with `sizeof(SharedStoragePaged)`:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
