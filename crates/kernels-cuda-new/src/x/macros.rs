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
//!    `&'static KernelSig` and the old table wrote `&ROPE_SIGS[3]`. No
//!    counting is needed: a `const SIG` item inside the row's own block is
//!    promoted to `'static` when referenced, so each row carries its own
//!    signature and no index can be off by one. A TT muncher IS needed, for
//!    a different reason: a row's operand list repeats over the `fn`'s
//!    parameters, and `macro_rules!` refuses a params-depth metavariable
//!    transcribed under the rows repetition ("meta-variable repeats N
//!    times"). The `@rows`/`@params` accumulators below take the rows one
//!    at a time, so inside each step the parameters are the only repetition
//!    in sight.
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
///         "rope::rope_partial_bf16" => where [T = bf16] "device::bf16",
///         "rope::rope_partial_f16"  => where [T = f16]  "device::f16",
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
/// # A by-value aggregate needs nothing here
///
/// A `__global__` that takes a struct whole — XQA's
/// `KVCacheList<true> const cacheList` — is declared exactly like a scalar:
///
/// ```ignore
/// fn kernel_mha = "kernel_mha" (
///     ...
///     cache_list: crate::x::xqa::KvCacheList,
///     ...
/// ) { ... }
/// ```
///
/// because [`by_value!`](crate::by_value) makes the mirror an ordinary
/// [`Abi`](crate::x::Abi) impl. `PARAMS` gets its [`Abi::CPP`], `ROWS` gets
/// its [`Abi::TY`], `raw::` takes it by value and `arg` answers
/// `ArgValue::Bytes`. That the macro needed no new grammar for the case §3.2
/// called out is the evidence that "an open set of impls, not a closed enum"
/// was the right shape.
///
/// The one thing that is NOT derived is the layout assertion list: a unit
/// with by-value parameters passes its `ByValue::LAYOUT`s to
/// `x::abi::typecheck_tu` alongside `PARAMS`. `PARAMS` is `&[&[&str]]` of
/// spellings and a spelling cannot be turned back into a type, so the
/// family names them. See `x::xqa::LAYOUTS`.
///
/// # ONE UNIT PER INVOCATION, and what a multi-unit family does
///
/// Every item above is at module scope, so two invocations in one file
/// collide on `UNITS`, `ROWS`, `PARAMS` and `raw`. §5.1 predicted this would
/// be where the macros needed real work; the sweep answered it without a
/// grammar change, and the answer is better than a grammar change would have
/// been.
///
/// A family with several roots wraps each invocation in its own inline
/// module and writes the family-level list by hand:
///
/// ```ignore
/// pub mod envelope { unit! { unit ENVELOPE = "layout/envelope", ... } }
/// pub mod embed    { unit! { unit EMBED    = "layout/embed",    ... } }
///
/// pub static UNITS: &[Unit] = &[envelope::ENVELOPE, embed::EMBED];
/// ```
///
/// `layout` is five units and reads well this way — `raw::` picks up a
/// natural qualifier (`envelope::raw::merge_written`) that a flat file would
/// have had to spell into every stub name. The hand-written family `UNITS`
/// is four words per root and is the only line that is written twice, which
/// is under the bar §0 sets.
///
/// **An `n`-unit grammar would have to answer a question this does not.**
/// `ROWS` is per-unit and `PARAMS` is asserted parallel to it; a single
/// invocation producing several `Unit`s would need to say which `ROWS` each
/// one names, and the parallel-array assertion that makes `PARAMS`
/// trustworthy would become an index the macro maintains rather than a
/// property of the expansion. Reach for it only if a root count arrives that
/// makes the module wrapper genuinely unreadable — FA2's 56 units on one
/// root is the case to judge it on, and that shape is 56 *instantiations*
/// under one `unit!`, which this grammar already expresses.
///
/// # A CROSS-FAMILY LAUNCH NEEDS NOTHING, and the reason is worth reading
///
/// `norm::rmsnorm_bf16_with_fp16`'s second launch is `quant::bf16_to_fp16`;
/// `gemm::act_x_wt_bias_bf16`'s is `norm::add_bias_bf16`. Both were reported
/// as a hole in the floor — *"`rmsnorm::raw::` cannot spell a `quant::`
/// symbol"* — with the untyped `x::fire::fire` and a hand-built
/// `&[ArgValue]` as the workaround, and the alternative being a duplicate
/// `quant` declaration, which would write the kernel twice.
///
/// **Neither is necessary. Call the other family's stub.**
///
/// ```ignore
/// // in x/norm.rs, inside the rmsnorm host program:
/// unsafe {
///     crate::x::quant::cast::raw::bf16_to_fp16(
///         "quant::bf16_to_fp16", launch, src, dst, n, stream,
///     );
/// }
/// ```
///
/// The premise is true and does not matter: `rmsnorm::raw::` indeed cannot
/// spell a `quant::` symbol, and nothing asks it to. A `raw::` stub is
/// **not bound to the unit it was declared beside**. Read the expansion —
/// it takes `symbol`, `launch`, its typed parameters and `stream`, and calls
/// `x::fire::fire(symbol, ..)`, which resolves `unit::unit_of(symbol)`
/// globally. `$UNIT` appears nowhere in a stub's body. The module path is
/// Rust namespacing and only that, so `pub mod raw` in one family file is
/// reachable from every other with full typing, the real `Abi::CPP`
/// spellings behind it, and no second declaration anywhere.
///
/// So the answer to *"a cross-family launch is not exotic — it is what
/// `Composed` is"* is: correct, and it was already ordinary. Reach for
/// `x::fire::fire` by hand only for a symbol **no `unit!` declares**, which
/// after the sweep is nothing.
///
/// ## The one consequence, which is real
///
/// A cross-family call makes the callee's unit a dependency of the caller's
/// host program, and nothing in the type system says so — `symbol` is a
/// `&'static str`. If the callee's unit is absent or its symbol misspelled,
/// `unit_of` returns `None` and `fire` panics naming the symbol, which is
/// the deliberate behaviour for a broken JIT and is the right failure, but
/// it happens at the fire and not at the load. Put the callee's family in a
/// `// fires: quant::bf16_to_fp16` note beside the call, so a reader
/// deleting a `quant` unit finds the caller by grep.
///
/// # No `units!`, and this one is a refusal
///
/// A multi-root family hand-writes `pub static UNITS: &[Unit] = &[..]` and
/// that is *"the one place where adding a root is two edits instead of
/// one"*. It stays two edits.
///
/// The aggregate is the family's MANIFEST — the one place a reader learns
/// what device text this family compiles — and it has real reading
/// consumers: `cache::module` walks it and `tests/units.rs` walks it.
/// §0's rule is *data only for what has a reading consumer*, and this list
/// has two. A macro that synthesised it would make the reader chase an
/// expansion to answer "what does `norm` compile", which is the question the
/// file should answer at a glance. Four words per root, once, is the price
/// of that, and it is the right price.
///
/// The only thing that would genuinely make it one edit is a build-time
/// derivation from the directory — and §6.1 has already refused that class
/// once, for the declaration itself, on the ground that an FFI-adjacent fact
/// which is derived is a fact nothing checks.
///
/// [`Abi::TY`]: crate::x::Abi::TY
/// [`Abi::CPP`]: crate::x::Abi::CPP
#[macro_export]
macro_rules! unit {
    (
        $(#[$umeta:meta])*
        unit $unit:ident = $uname:literal, text = $utext:expr, file = $ufile:literal
             $(, options = $uopts:expr)? ;
        $(
            $(#[$fmeta:meta])*
            fn $fname:ident = $path:literal $(<$($g:ident),+ $(,)?>)? (
                $($pname:ident : $pty:ty),* $(,)?
            ) $(, cooperative = $coop:literal)? $(where $($wty:ty),+ $(,)?)? {
                $(
                    $(#[$rmeta:meta])*
                    // `where` before the binding group, because `[` opens an
                    // array expression too and the matcher could not tell the
                    // group from `$elem`. `where` cannot begin an expression,
                    // so one keyword settles it.
                    $symbol:literal => $(where [$($bg:ident = $bty:ty),+ $(,)?])? $elem:expr
                ),* $(,)?
            }
        )*
    ) => {
        $(#[$umeta])*
        pub const $unit: $crate::unit::Unit = $crate::unit::Unit {
            name: $uname,
            root: $utext,
            rows: ROWS,
            // EMPTY UNLESS THE ROOT SAYS OTHERWISE, and until `mla_fa2` no
            // root did. It hard-coded `&[]` on the reasoning that a
            // compile-option list is a property of the recipe rather than of
            // a unit — true for eleven families and false for the twelfth:
            // `attn/attention_mla_fa2.cuh` needs
            // `--device-as-default-execution-space` and produces **sixteen
            // errors without it** (one in `shim/type_traits`, seven in
            // `cascade.cuh`, eight in `prefill.cuh`).
            //
            // So the option is not a recipe preference, it is a fact about
            // that root's text, and it belongs where the text is named. The
            // default stays `&[]` because a unit that states nothing should
            // compile with what every other unit compiles with — an option
            // that had to be repeated per unit would be a recipe wearing a
            // declaration.
            options: $crate::unit_options!($($uopts)?),
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
        //
        // Through the `@rows` accumulator and not a nested repetition: a
        // row's operand list repeats over `$pname`, which is bound at
        // params depth, and the transcriber refuses it under the rows
        // repetition. See the module header, point 2.
        static ROWS: &[$crate::device::DeviceKernel] = &$crate::unit!(@rows [] $(
            {
                path = $path; file = $ufile;
                params = [$(($pname: $pty))*];
                $(row = [$(#[$rmeta])* $symbol => $(where [$(($bg = $bty))+])? $elem];)*
            }
        )*);

        /// Each row's C++ parameter types, parallel to `ROWS`.
        ///
        /// From [`Abi::CPP`](crate::x::Abi::CPP), one string per parameter,
        /// with any template type argument substituted by the compiler
        /// rather than by this macro. This is the input to the typecheck
        /// translation unit §6.1 keeps.
        pub static PARAMS: &[&[&str]] = &$crate::unit!(@params [] $(
            {
                path = $path; file = $ufile;
                params = [$(($pname: $pty))*];
                $(row = [$(#[$rmeta])* $symbol => $(where [$(($bg = $bty))+])? $elem];)*
            }
        )*);

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
                        $crate::x::fire::fire_ex(
                            symbol,
                            launch,
                            // FALSE UNLESS THE `fn` LINE SAYS OTHERWISE.
                            // A cooperative launch is a property of the
                            // KERNEL — `mla.cuh:1061`'s two stages are
                            // separated by a `this_grid().sync()` and every
                            // other kernel in the tree syncs no further than
                            // its block — so it is declared beside the
                            // template path rather than passed at the call
                            // site, where a caller could forget it and get a
                            // wrong answer instead of a compile error.
                            $crate::unit_cooperative!($($coop)?),
                            // `arg` borrows. For a scalar or a pointer that
                            // is a copy either way; for a by-value aggregate
                            // the borrow is of THIS binding, which lives
                            // across the call, and `Args::bind` copies the
                            // bytes out before `fire` returns.
                            &[$(<$pty as $crate::x::Abi>::arg(&$pname)),*],
                            stream,
                        );
                    }
                }
            )*
        }
    };

    // -----------------------------------------------------------------------
    // The `@rows` / `@params` accumulators.
    //
    // Each takes the fn groups one ROW at a time, carrying the fn's
    // parameter list beside the row, so a step never transcribes a
    // params-depth metavariable under the rows repetition — which the
    // transcriber refuses ("meta-variable `symbol` repeats 1 time, but
    // `pname` repeats 4 times"). The accumulator collects finished array
    // elements; the terminal arm emits them as one array literal, which is
    // what the `&` at the invocation site borrows.
    // -----------------------------------------------------------------------

    (@rows [$($acc:tt)*]) => { [$($acc)*] };
    (@rows [$($acc:tt)*]
        {
            path = $path:literal; file = $ufile:literal;
            params = [$(($pname:ident : $pty:ty))*];
            row = [$(#[$rmeta:meta])* $symbol:literal => $(where [$(($bg:ident = $bty:ty))+])? $elem:expr];
            $($rows:tt)*
        }
        $($rest:tt)*
    ) => {
        $crate::unit!(@rows
            [
                $($acc)*
                $(#[$rmeta])*
                {
                    $($(type $bg = $bty;)+)?
                    const SIG: ::kernels::KernelSig = ::kernels::KernelSig {
                        // The symbol in both columns. `name` is
                        // `emit_c_shim`'s `pie_k_` stem and no device row is
                        // shimmed, so the only honest answer is the symbol
                        // itself.
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
                },
            ]
            { path = $path; file = $ufile; params = [$(($pname: $pty))*]; $($rows)* }
            $($rest)*
        )
    };
    (@rows [$($acc:tt)*]
        { path = $path:literal; file = $ufile:literal; params = [$($p:tt)*]; }
        $($rest:tt)*
    ) => {
        $crate::unit!(@rows [$($acc)*] $($rest)*)
    };

    (@params [$($acc:tt)*]) => { [$($acc)*] };
    (@params [$($acc:tt)*]
        {
            path = $path:literal; file = $ufile:literal;
            params = [$(($pname:ident : $pty:ty))*];
            row = [$(#[$rmeta:meta])* $symbol:literal => $(where [$(($bg:ident = $bty:ty))+])? $elem:expr];
            $($rows:tt)*
        }
        $($rest:tt)*
    ) => {
        $crate::unit!(@params
            [
                $($acc)*
                {
                    $($(type $bg = $bty;)+)?
                    &[$(<$pty as $crate::x::Abi>::CPP),*] as &[&str]
                },
            ]
            { path = $path; file = $ufile; params = [$(($pname: $pty))*]; $($rows)* }
            $($rest)*
        )
    };
    (@params [$($acc:tt)*]
        { path = $path:literal; file = $ufile:literal; params = [$($p:tt)*]; }
        $($rest:tt)*
    ) => {
        $crate::unit!(@params [$($acc)*] $($rest)*)
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

/// `&[]` or the caller's list — [`unit!`]'s optional `options =` clause.
///
/// A `macro_rules!` fragment cannot be conditionally substituted inside a
/// struct literal field, because `$(...)?` around the field would make the
/// field itself optional and `Unit` has no `..Default`. So the choice is
/// pushed into a helper with two arms, which is the shape the language
/// actually offers.
#[macro_export]
#[doc(hidden)]
macro_rules! unit_options {
    () => {
        &[]
    };
    ($opts:expr) => {
        $opts
    };
}

/// `false` or the caller's literal — [`unit!`]'s optional `cooperative =`
/// clause, for [`unit_options!`]'s reason.
#[macro_export]
#[doc(hidden)]
macro_rules! unit_cooperative {
    () => {
        false
    };
    ($coop:literal) => {
        $coop
    };
}
