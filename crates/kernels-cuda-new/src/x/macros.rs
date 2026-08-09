//! §1, §2.1 — the three declarations, as `macro_rules!`.
//!
//! `unit!` declares the device text and its instantiations. `contract!`
//! declares what a trace may say. `bind!` declares what happens when a trace
//! says it. Nothing else is declared, and everything else is a `fn`.
//!
//! # Why `macro_rules!` and not a proc macro
//!
//! §5 step 2 asks for thin `macro_rules!` first and a proc macro only if the
//! grammar demands it. The grammar did not demand it. Three things were
//! close:
//!
//! 1. **Substituting a template's type argument into a parameter list.**
//!    `standard_table<class P>` is declared once and instantiated at `i32`;
//!    the row's operand types and its C++ typecheck line both need
//!    `*const P` with `P` replaced. `macro_rules!` cannot substitute inside
//!    a type — but it can emit `type P = i32;` into a block and let the
//!    *compiler* substitute. That is what the row blocks below do, and it is
//!    strictly better than a proc macro doing it textually, because a name
//!    that does not resolve is a compile error at the declaration.
//! 2. **Indexing a parallel array.** `DeviceKernel::sig` is a
//!    `&'static KernelSig` and the old table wrote `&ROPE_SIGS[3]`. Counting
//!    repetitions in `macro_rules!` needs a TT muncher. It is not needed:
//!    a `const SIG` item inside the row's own block is promoted to
//!    `'static` when referenced, so each row carries its own signature and
//!    no index can be off by one.
//! 3. **Generating an identifier from a symbol string.** Nothing needs it.
//!    The one place that looked like it did — a per-row module for the
//!    typecheck list — is a block instead.
//!
//! What a proc macro WOULD buy is parsing the `.cuh` and generating the
//! declaration from it. §6.1 refuses that, and the refusal is part of the
//! design: the declaration and the `__global__` are two sides of one FFI
//! contract, and the typecheck translation unit is the oracle that holds
//! them together. "Nothing is written twice" means "nothing is written twice
//! **unchecked**".

/// The device text of one unit, its instantiations, and the typed stubs that
/// launch them.
///
/// ```ignore
/// unit! {
///     /// What this unit's text is.
///     unit ROPE = "rope/rope",
///         text = include_str!("../../csrc/src/rope/rope.cuh"),
///         file = "rope/rope.cuh";
///
///     /// The `__global__`, in its own words.
///     fn rotate_partial = "rope::device::rotate_partial" <T> (
///         q: *mut T, k: *mut T, positions: *const i32,
///         position_delta: i32, num_q_heads: i32, num_kv_heads: i32,
///         head_dim: i32, rotary_dim: i32, theta: f32,
///     ) {
///         "rope::rope_partial_bf16" => [T = bf16] "device::bf16",
///         "rope::rope_partial_f16"  => [T = f16]  "device::f16",
///     }
/// }
/// ```
///
/// It generates:
///
/// * `pub const $UNIT: Unit` and `pub static UNITS: &[Unit]` — what the
///   NVRTC cache compiles, unchanged from what the family table produced.
/// * `static ROWS: &[DeviceKernel]` — one per instantiation, each carrying
///   its own `KernelSig` whose operand types come from [`Abi::TY`].
///   `LaunchRule::Unstated` on every one: the geometry is a `fn`'s.
/// * `pub static PARAMS: &[&[&str]]` — the same rows' C++ parameter types,
///   from [`Abi::CPP`], for the typecheck translation unit. Parallel to
///   `ROWS` and asserted so by `x::abi::typecheck_tu`.
/// * `pub mod raw` — one `unsafe fn` per declared `__global__`, taking the
///   symbol, a [`Launch`](crate::x::launch::Launch), the parameters in the
///   device's order, and a stream. Typed: a `*mut bf16` where the kernel
///   wants `*mut f16` will not compile.
///
/// The NVRTC name-expression list is not generated: it is
/// `UNITS[..].rows[..].instantiation()`, which is what `cache::module` and
/// `tests/units.rs` already walk, and generating a second copy of it would
/// be the one thing this design refuses.
///
/// [`Abi::TY`]: crate::x::Abi::TY
/// [`Abi::CPP`]: crate::x::Abi::CPP
#[macro_export]
macro_rules! unit {
    (
        $(#[$umeta:meta])*
        unit $unit:ident = $uname:literal, text = $utext:expr, file = $ufile:literal;
        $(
            $(#[$fmeta:meta])*
            fn $fname:ident = $path:literal $(<$($g:ident),+ $(,)?>)? (
                $($pname:ident : $pty:ty),* $(,)?
            ) $(where $($wty:ty),+ $(,)?)? {
                $(
                    $(#[$rmeta:meta])*
                    $symbol:literal => $([$($bg:ident = $bty:ty),+ $(,)?])? $elem:expr
                ),* $(,)?
            }
        )*
    ) => {
        $(#[$umeta])*
        pub const $unit: $crate::unit::Unit = $crate::unit::Unit {
            name: $uname,
            root: $utext,
            rows: ROWS,
            options: &[],
        };

        /// The units this family compiles.
        pub static UNITS: &[$crate::unit::Unit] = &[$unit];

        /// One row per declared instantiation.
        ///
        /// Each carries its own `KernelSig`: the `__global__`'s parameter
        /// list, typed by `Abi`, with every `Source::Unbound`. **A device
        /// row has no sources in fn-world** — a `fn` binds its own
        /// arguments — and `LaunchRule::Unstated` for the same reason.
        /// What survives is exactly what NVRTC and `Args::bind` read.
        static ROWS: &[$crate::device::DeviceKernel] = &[$($(
            $(#[$rmeta])*
            {
                $($(type $bg = $bty;)+)?
                const SIG: ::kernels::KernelSig = ::kernels::KernelSig {
                    // The symbol in both columns. `name` is `emit_c_shim`'s
                    // `pie_k_` stem and no device row is shimmed, so the
                    // only honest answer is the symbol itself.
                    name: $symbol,
                    symbol: $symbol,
                    file: Some($ufile),
                    operands: &[$(::kernels::Operand {
                        name: stringify!($pname),
                        ty: <$pty as $crate::x::Abi>::TY,
                        nullable: <$pty as $crate::x::Abi>::NULLABLE,
                        source: ::kernels::Source::Unbound,
                    }),*],
                    ..$crate::x::contract::SIG_BASE
                };
                $crate::device::DeviceKernel {
                    sig: &SIG,
                    template_path: $path,
                    elem: $elem,
                }
            }
        ),*),*];

        /// Each row's C++ parameter types, parallel to `ROWS`.
        ///
        /// From [`Abi::CPP`](crate::x::Abi::CPP), one string per parameter,
        /// with any template type argument substituted by the compiler
        /// rather than by this macro. This is the input to the typecheck
        /// translation unit §6.1 keeps.
        pub static PARAMS: &[&[&str]] = &[$($(
            {
                $($(type $bg = $bty;)+)?
                &[$(<$pty as $crate::x::Abi>::CPP),*] as &[&str]
            }
        ),*),*];

        /// Typed launchers, one per declared `__global__`.
        ///
        /// `symbol` picks the instantiation; everything else is the device
        /// text's own parameter list, in its own order, in Rust types.
        #[cfg(feature = "_cuda")]
        pub mod raw {
            #[allow(unused_imports)]
            use super::*;
            $(
                $(#[$fmeta])*
                ///
                /// # Safety
                ///
                /// Every pointer must address live device memory of the
                /// extent this kernel will read or write, and `stream` must
                /// be live across the launch.
                #[allow(clippy::too_many_arguments, unused_unsafe)]
                pub unsafe fn $fname $(<$($g),+>)? (
                    symbol: &'static str,
                    launch: $crate::x::launch::Launch,
                    $($pname: $pty,)*
                    stream: *mut ::core::ffi::c_void,
                )
                $(where $($wty: $crate::x::Abi,)+)?
                {
                    unsafe {
                        $crate::x::fire::fire(
                            symbol,
                            launch,
                            &[$(<$pty as $crate::x::Abi>::arg($pname)),*],
                            stream,
                        );
                    }
                }
            )*
        }
    };
}

/// What a trace may say about this family's symbols.
///
/// ```ignore
/// contract! {
///     /// The plain rotation.
///     ROPE_BF16 = "rope::rope_bf16" as rope {
///         in_place: &[(0, 0), (1, 1)],
///     }
/// }
/// ```
///
/// It generates one `pub const $NAME: Contract` per declaration, plus
/// `pub static CONTRACTS: &[Contract]` and `pub static SIGS: &[KernelSig]`
/// — the row view, derived by [`Contract::sig`], that `model-compiler`
/// reads and that keeps `check_plan` able to refuse a symbol nothing
/// declares.
///
/// Any field of [`Contract`] may be stated; unstated fields come from
/// [`Contract::DEFAULT`], so a declaration says only what is unusual about
/// it and a reader's eye goes to exactly that.
///
/// [`Contract`]: crate::x::Contract
/// [`Contract::sig`]: crate::x::Contract::sig
/// [`Contract::DEFAULT`]: crate::x::Contract::DEFAULT
#[macro_export]
macro_rules! contract {
    (
        $(
            $(#[$meta:meta])*
            $name:ident = $symbol:literal as $dsl:ident $({
                $($field:ident : $value:expr),* $(,)?
            })?
        )*
    ) => {
        $(
            $(#[$meta])*
            pub const $name: $crate::x::Contract = $crate::x::Contract {
                name: stringify!($dsl),
                symbol: $symbol,
                $($($field: $value,)*)?
                ..$crate::x::Contract::DEFAULT
            };
        )*

        /// Every contract this family declares.
        pub static CONTRACTS: &[$crate::x::Contract] = &[$($name),*];

        /// The same, as the rows `model-compiler` reads.
        ///
        /// Derived, never written: see [`Contract::sig`](crate::x::Contract::sig)
        /// for why these state no `operands` and what that costs a row.
        pub static SIGS: &[::kernels::KernelSig] = &[$($name.sig()),*];
    };
}

/// What happens when a trace says it.
///
/// ```ignore
/// bind! {
///     ROPE_BF16 => { cx, stream => {
///         let head_dim = cx.head_dim()?;
///         unsafe { rope_bf16(/* ... */) }.ok()
///     }},
///     ROPE_YARN_BF16 => { none: "no statement carries llama-3's \
///         low/high frequency factors" },
/// }
/// ```
///
/// It generates `pub static ENTRIES: &[Entry]`, one per declaration, in the
/// order written. A `none:` arm is a symbol that is declared, callable as a
/// `fn`, and **not trace-fired** — §1's ladder allows exactly that, and the
/// reason becomes the sentence a load-time refusal prints.
///
/// **Each arm's body is one token tree**, which is why the braces around
/// `cx, stream => { .. }` and around `none: ".."` are part of the grammar
/// rather than decoration: `macro_rules!` needs a single `tt` to dispatch
/// on before it can tell a bind from a refusal. **No attributes are
/// accepted on an arm**, because the arms become elements of an array
/// expression and Rust has no attributes there — so a reason is written as
/// an ordinary `//` comment beside the arm and as the `none:` string, which
/// is the copy a load-time refusal prints.
///
/// The body is a `fn` item rather than a closure so that the two parameters
/// are typed by this macro and a bind body can never accidentally capture.
/// A bind body has no `&mut` in scope, no device API and no allocator: the
/// whole of what it can do is read [`Cx`](crate::x::Cx) and call a host
/// program.
#[macro_export]
macro_rules! bind {
    (
        $(
            $name:ident => $body:tt
        ),* $(,)?
    ) => {
        /// Every symbol this family declares, with what fires it.
        pub static ENTRIES: &[$crate::x::Entry] = &[$(
            $crate::bind!(@entry $name $body)
        ),*];
    };
    (@entry $name:ident { $cx:ident, $stream:ident => $body:block }) => {
        $crate::x::Entry {
            contract: &$name,
            bind: Some({
                fn bound(
                    $cx: &$crate::x::Cx<'_>,
                    $stream: *mut ::core::ffi::c_void,
                ) -> ::core::result::Result<(), $crate::x::Refusal> {
                    $body
                }
                bound
            }),
            unbound: None,
        }
    };
    (@entry $name:ident { none: $why:expr }) => {
        $crate::x::Entry { contract: &$name, bind: None, unbound: Some($why) }
    };
}
