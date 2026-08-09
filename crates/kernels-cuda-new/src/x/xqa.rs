use crate::by_value;
use crate::device::DeviceKernel;
use crate::families::attn::{XQA_COMMON_OPTIONS, XQA_LATTICE, XQA_ROOT};
use crate::unit::Unit;
use crate::x::abi::MaybeConst;
use crate::{bind, contract};
use core::ffi::c_void;
use core::ptr::NonNull;
use kernels::{Cap, KernelSig, Operand, Prepare, Ty};

/// A device address held as an opaque word.
pub use crate::x::abi::DevicePtr;

/// `KVCacheList<true>` — XQA's paged KV cache descriptor, passed by value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct KvCacheList {
    /// `kCacheVLLM` — `GMemCacheHead*`, the K pages.
    pub k_cache: DevicePtr,
    /// `vCacheVLLM` — `GMemCacheHead*`, the V pages.
    pub v_cache: DevicePtr,
    /// `kvCachePageList` — `KVCachePageIndex const*`, shape
    pub page_list: DevicePtr,
    /// `seqLenList` — `SeqLenDataType const*`, shape `[batchSize][beamWidth]`.
    pub seq_len_list: DevicePtr,
    /// `maxNbPagesPerSeq` — the third stride of `page_list`, in pages.
    pub max_pages_per_seq: u32,
}

by_value! {
    KvCacheList as "KVCacheList<true>",
    tag = KvCacheLayerView,
    probe = "nvrtc-probes/xqa_kvcachelist.py",
    size = 40, align = 8,
    {
        k_cache           @ 0  as "kCacheVLLM",
        v_cache           @ 8  as "vCacheVLLM",
        page_list         @ 16 as "kvCachePageList",
        seq_len_list      @ 24 as "seqLenList",
        max_pages_per_seq @ 32 as "maxNbPagesPerSeq",
    }
}

impl KvCacheList {
    /// The descriptor for one paged KV cache.
    #[must_use]
    pub const fn paged(
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        page_list: DevicePtr,
        seq_len_list: DevicePtr,
        max_pages_per_seq: u32,
    ) -> Self {
        Self { k_cache, v_cache, page_list, seq_len_list, max_pages_per_seq }
    }
}

/// The by-value aggregates the XQA units pass, for the typecheck TU.
pub static LAYOUTS: &[crate::x::Layout] = &[<KvCacheList as crate::x::ByValue>::LAYOUT];

/// `IOHead` and `OutputHead` — XQA's per-head vector, as a pointee.
pub enum XqaIoHead {}

impl crate::x::Abi for *const XqaIoHead {
    const CPP: &'static str = "const IOHead*";
    const TY: Ty = Ty::Buf;
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::Ptr(*self as *mut c_void)
    }
}

impl crate::x::Abi for *mut XqaIoHead {
    const CPP: &'static str = "OutputHead*";
    const TY: Ty = Ty::BufMut;
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::Ptr(self.cast::<c_void>())
    }
}

/// `kernel_mha`'s parameter list, written ONCE and expanded three ways.
macro_rules! kernel_mha_declaration {
    ($($pname:ident : $pty:ty),* $(,)?) => {
        /// Each parameter's C++ type, for the typecheck TU.
        pub static PARAMS: &[&str] = &[$(<$pty as crate::x::Abi>::CPP),*];

        /// The same list as operand rows.
        static OPERANDS: &[Operand] = &[$(Operand {
            name: stringify!($pname),
            ty: <$pty as crate::x::Abi>::TY,
            nullable: <$pty as crate::x::Abi>::NULLABLE,
            source: ::kernels::Source::Unbound,
        }),*];

        /// The typed launcher, one for all five members.
        #[cfg(feature = "_cuda")]
        pub mod raw {
            #[allow(unused_imports)]
            use super::*;
            use crate::x::launch::Launch;

            /// Launch one member of the XQA lattice.
            ///
            /// # Safety
            ///
            /// Every pointer must address live device memory of the extent
            /// this kernel will read or write, `cache_list`'s five fields
            /// must be device addresses of a paged cache laid out as
            /// `KVCacheList<true>` describes, and `stream` must be live
            /// across the launch.
            ///
            /// `launch.smem` must be `XQA_SMEM_BYTES` (79 488), which is
            /// over the 48 KiB default and therefore drives
            /// `runtime::module::raise_dynamic_smem_cap`. A smaller value
            /// launches a kernel whose `SharedMem` does not fit.
            #[allow(clippy::too_many_arguments, unused_unsafe)]
            pub unsafe fn kernel_mha(
                symbol: &'static str,
                launch: Launch,
                $($pname: $pty,)*
                stream: *mut c_void,
            ) {
                unsafe {
                    crate::x::fire::fire(
                        symbol,
                        launch,
                        &[$(<$pty as crate::x::Abi>::arg(&$pname)),*],
                        stream,
                    );
                }
            }
        }
    };
}

kernel_mha_declaration! {
    nb_k_heads: u32,
    q_scale: f32,
    q_scale_ptr: MaybeConst<f32>,
    output: *mut XqaIoHead,
    q: *const XqaIoHead,
    attention_sinks: MaybeConst<f32>,
    cache_list: KvCacheList,
    batch_size: u32,
    kv_cache_scale: f32,
    kv_scale_ptr: MaybeConst<f32>,
    kv_stride_page: u32,
    kv_stride_token: u32,
    kv_stride_head: u32,
    semaphores: Option<NonNull<u32>>,
    scratch: Option<NonNull<c_void>>,
}

/// The file every row cites, for a `KernelSig`'s `file`.
const FILE: &str = "attn/attention_xqa_mha.cuh";

/// One member's `-D` set: the thirteen shared with every other, then the
const fn options_of(member: usize) -> [&'static str; 17] {
    let extra = XQA_LATTICE[member].options;
    let mut out = [""; 17];
    let mut i = 0;
    while i < XQA_COMMON_OPTIONS.len() {
        out[i] = XQA_COMMON_OPTIONS[i];
        i += 1;
    }
    let mut j = 0;
    while j < extra.len() {
        out[XQA_COMMON_OPTIONS.len() + j] = extra[j];
        j += 1;
    }
    out
}

const _: () = assert!(
    XQA_COMMON_OPTIONS.len() == 13,
    "XQA_COMMON_OPTIONS changed width; `options_of`'s array must change with it",
);
const _: () = {
    let mut i = 0;
    while i < XQA_LATTICE.len() {
        assert!(
            XQA_LATTICE[i].options.len() == 4,
            "an XQA lattice member states a number of `-D`s `options_of` was not sized for",
        );
        i += 1;
    }
};

const OPTIONS_GQA2_P32: [&str; 17] = options_of(0);
const OPTIONS_GQA2_P16: [&str; 17] = options_of(1);
const OPTIONS_GQA4_P32: [&str; 17] = options_of(2);
const OPTIONS_GQA5_P32: [&str; 17] = options_of(3);
const OPTIONS_GQA8_P32: [&str; 17] = options_of(4);

/// One row's `KernelSig`, which is the same for all five but for the symbol.
const fn row_sig(symbol: &'static str) -> KernelSig {
    KernelSig {
        name: symbol,
        symbol,
        file: Some(FILE),
        operands: OPERANDS,
        ..crate::x::contract::SIG_BASE
    }
}

/// The five device entry points, by the name the `-Dkernel_mha=…` rename
static ROW_SIGS: [KernelSig; 5] = [
    row_sig(XQA_LATTICE[0].entry),
    row_sig(XQA_LATTICE[1].entry),
    row_sig(XQA_LATTICE[2].entry),
    row_sig(XQA_LATTICE[3].entry),
    row_sig(XQA_LATTICE[4].entry),
];

/// Each member's one row.
static ROWS: [[DeviceKernel; 1]; 5] = [
    [DeviceKernel {
        sig: &ROW_SIGS[0],
        template_path: "::kernel_mha_xqa_gqa2_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[1],
        template_path: "::kernel_mha_xqa_gqa2_bf16_p16_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[2],
        template_path: "::kernel_mha_xqa_gqa4_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[3],
        template_path: "::kernel_mha_xqa_gqa5_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
    [DeviceKernel {
        sig: &ROW_SIGS[4],
        template_path: "::kernel_mha_xqa_gqa8_bf16_p32_h128",
        elem: DeviceKernel::PLAIN,
    }],
];

/// `attn/attention_xqa_mha_gqa2_p32` — head group 2 at a 32-token page.
pub const XQA_GQA2_P32: Unit = Unit {
    name: XQA_LATTICE[0].unit,
    root: XQA_ROOT,
    rows: &ROWS[0],
    options: &OPTIONS_GQA2_P32,
};

/// `attn/attention_xqa_mha_gqa2_p16` — the same head group at a 16-token
pub const XQA_GQA2_P16: Unit = Unit {
    name: XQA_LATTICE[1].unit,
    root: XQA_ROOT,
    rows: &ROWS[1],
    options: &OPTIONS_GQA2_P16,
};

/// `attn/attention_xqa_mha_gqa4_p32` — head group 4, Qwen3-4B and Qwen3-8B
pub const XQA_GQA4_P32: Unit = Unit {
    name: XQA_LATTICE[2].unit,
    root: XQA_ROOT,
    rows: &ROWS[2],
    options: &OPTIONS_GQA4_P32,
};

/// `attn/attention_xqa_mha_gqa5_p32` — head group 5, the ratio
pub const XQA_GQA5_P32: Unit = Unit {
    name: XQA_LATTICE[3].unit,
    root: XQA_ROOT,
    rows: &ROWS[3],
    options: &OPTIONS_GQA5_P32,
};

/// `attn/attention_xqa_mha_gqa8_p32` — head group 8, Qwen3-32B and
pub const XQA_GQA8_P32: Unit = Unit {
    name: XQA_LATTICE[4].unit,
    root: XQA_ROOT,
    rows: &ROWS[4],
    options: &OPTIONS_GQA8_P32,
};

/// The five units this family compiles.
pub static UNITS: &[Unit] =
    &[XQA_GQA2_P32, XQA_GQA2_P16, XQA_GQA4_P32, XQA_GQA5_P32, XQA_GQA8_P32];

contract! {
    /// XQA's paged decode, the whole fire.
    XQA_DECODE_BF16_PREPARED = "attn::attention_xqa_decode_bf16_prepared" as xqa_decode {
        whole: true,
        needs: Prepare::FireWide,
        lacks: &[Cap::Scores],
    }
}

bind! {
    XQA_DECODE_BF16_PREPARED => { none: "xqa decode's host program is \
        `driver-cuda/src/fire/xqa.rs::plan_decode`, a driver-side program in \
        the crate above this one: `Cx` states neither the attention \
        workspace nor `sm_scale`, and the semaphore bank's zeroing is a \
        device call no bind body may make" },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five enrolled units and the lattice describe the same five.
    #[test]
    fn every_row_names_the_lattice_entry_it_compiles() {
        assert_eq!(UNITS.len(), 5, "the sixth member is deliberately not enrolled");
        for (i, unit) in UNITS.iter().enumerate() {
            assert_eq!(unit.name, XQA_LATTICE[i].unit);
            assert_eq!(unit.rows.len(), 1, "one `__global__` per option set");
            assert_eq!(
                unit.rows[0].sig.symbol, XQA_LATTICE[i].entry,
                "the row's symbol is not the name `-Dkernel_mha=` gives it"
            );
            assert_eq!(
                unit.rows[0].template_path,
                format!("::{}", XQA_LATTICE[i].entry),
                "`template_path` must be the entry at GLOBAL scope: `CUBIN_EXPORT` is \
                 `extern \"C\"` under `GENERATE_CUBIN`, and a path without the leading \
                 `::` is qualified into `::pie_cuda_driver::kernels::`"
            );
            assert_eq!(unit.options.len(), 17, "thirteen shared and four its own");
            assert_eq!(&unit.options[..13], XQA_COMMON_OPTIONS);
            assert_eq!(&unit.options[13..], XQA_LATTICE[i].options);
        }
    }

    /// The Hopper member is absent from every list, not merely from `UNITS`.
    #[test]
    fn the_sm90_member_is_enrolled_nowhere() {
        let sm90 = XQA_LATTICE[5].entry;
        assert!(
            UNITS.iter().all(|u| u.rows.iter().all(|r| r.sig.symbol != sm90)),
            "the sm90 member is enrolled; `UNITS`'s doc says why it must not be"
        );
        assert!(
            crate::unit::unit_of(sm90).is_none(),
            "some unit hosts the sm90 entry -- `XqaMember::pick` would then launch a \
             kernel that was never measured to compile"
        );
    }

    /// Fifteen parameters, and the three spellings agree on the count.
    #[test]
    fn the_parameter_list_is_one_list() {
        assert_eq!(PARAMS.len(), 15, "`mha.cuh:2783` under the lattice's own defines");
        assert_eq!(PARAMS.len(), OPERANDS.len());
        assert_eq!(OPERANDS[6].name, "cache_list", "the by-value aggregate is the seventh");
        assert_eq!(
            PARAMS[6],
            <KvCacheList as crate::x::ByValue>::LAYOUT.cpp,
            "the parameter's spelling and the asserted layout's must be one string"
        );
    }
}
