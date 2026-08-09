/// The device text of one unit, its instantiations, and the typed stubs that
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
            options: $crate::unit_options!($($uopts)?),
        };

        /// The units this family compiles.
        pub static UNITS: &[$crate::unit::Unit] = &[$unit];

        /// One row per declared instantiation.
        static ROWS: &[$crate::device::DeviceKernel] = &$crate::unit!(@rows [] $(
            {
                path = $path; file = $ufile;
                params = [$(($pname: $pty))*];
                $(row = [$(#[$rmeta])* $symbol => $(where [$(($bg = $bty))+])? $elem];)*
            }
        )*);

        /// Each row's C++ parameter types, parallel to `ROWS`.
        pub static PARAMS: &[&[&str]] = &$crate::unit!(@params [] $(
            {
                path = $path; file = $ufile;
                params = [$(($pname: $pty))*];
                $(row = [$(#[$rmeta])* $symbol => $(where [$(($bg = $bty))+])? $elem];)*
            }
        )*);

        /// Typed launchers, one per declared `__global__`.
        #[cfg(feature = "_cuda")]
        pub mod raw {
            #[allow(unused_imports)]
            use super::*;
            $(
                $(#[$fmeta])*
                /// # Safety
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
                            $crate::unit_cooperative!($($coop)?),
                            &[$(<$pty as $crate::x::Abi>::arg(&$pname)),*],
                            stream,
                        );
                    }
                }
            )*
        }
    };

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
        pub static SIGS: &[::kernels::KernelSig] = &[$($name.sig()),*];
    };
}

/// What happens when a trace says it.
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
