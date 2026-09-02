//! One declaration per tagged enum: a `#[repr(u8)]` whose discriminants are
//! frozen wire tags. `declare_tagged_enum!` derives `ALL`, `from_u8`, and
//! `name` from a single variant list, so adding a variant is a one-line edit
//! that cannot leave one of the three out of sync.

/// Declare a `#[repr(u8)]` enum with frozen wire tags.
///
/// Exported, so it must not name a Cargo feature. A conditional derive
/// writes it as a leading attribute, which the macro forwards.
///
/// ```ignore
/// declare_tagged_enum! {
///     /// What this enum is.
///     pub enum Colour {
///         /// Per-variant docs are preserved.
///         Red = 0, "red";
///         Green = 1, "green";
///     }
/// }
/// ```
#[macro_export]
macro_rules! declare_tagged_enum {
    (
        $(#[$enum_meta:meta])*
        $vis:vis enum $enum_name:ident {
            $($(#[$variant_meta:meta])* $variant:ident = $tag:literal, $spelling:literal;)*
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(u8)]
        $vis enum $enum_name {
            $($(#[$variant_meta])* $variant = $tag,)*
        }

        impl $enum_name {
            /// Every variant, in wire-tag order.
            ///
            /// Anything that enumerates this type derives from here rather than
            /// re-listing the variants.
            pub const ALL: &'static [$enum_name] = &[$($enum_name::$variant,)*];

            /// The variant with this wire tag, if any.
            pub fn from_u8(tag: u8) -> Option<Self> {
                Some(match tag {
                    $($tag => $enum_name::$variant,)*
                    _ => return None,
                })
            }

            /// The snake-case spelling used for generated C identifiers.
            pub fn name(self) -> &'static str {
                match self {
                    $($enum_name::$variant => $spelling,)*
                }
            }
        }
    };
}

