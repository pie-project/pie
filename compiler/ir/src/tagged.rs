//! One declaration per tagged enum.
//!
//! Several enums in this workspace are the same shape: a `#[repr(u8)]` whose
//! discriminants are frozen wire tags, plus three things that have to agree
//! with the variant list and with each other —
//!
//! * `ALL`, which the generated C header and every "for each variant" loop walk;
//! * `from_u8`, which turns a tag back into a variant;
//! * `name`, the snake-case spelling the C identifier and the debug output use.
//!
//! Written out by hand, each of those is a separate opportunity to add a
//! variant and update two of the three. That is not hypothetical: `Stage::ALL`
//! sat six lines from a `for stage in [Stage::Prologue, ..]` in
//! `pie-dsl`'s builder that re-listed the same four stages, so a fifth stage
//! would have compiled, built a container, and simply never been traced.
//!
//! [`declare_tagged_enum!`] takes the list once and derives all three, the same
//! way [`declare_ops!`](crate::declare_ops) does for the op table. Adding a
//! variant is then a one-line edit that cannot be half-done.

/// Declare a `#[repr(u8)]` enum with frozen wire tags.
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

#[cfg(test)]
mod tests {
    use crate::registry::{Port, Stage};

    /// The three derived items agree, for every enum declared this way.
    ///
    /// This cannot fail by drift — one list generates all three — so it is a
    /// check on the *declarations*: a duplicated or out-of-order tag.
    fn check<T: 'static + Copy + PartialEq + core::fmt::Debug>(
        all: &[T],
        tag_of: fn(T) -> u8,
        from_u8: fn(u8) -> Option<T>,
        name: fn(T) -> &'static str,
    ) {
        for (index, &variant) in all.iter().enumerate() {
            assert_eq!(
                tag_of(variant) as usize,
                index,
                "{:?}: ALL must be dense and in tag order",
                variant
            );
            assert_eq!(from_u8(tag_of(variant)), Some(variant));
            assert!(!name(variant).is_empty());
        }
        assert!(
            from_u8(all.len() as u8).is_none(),
            "from_u8 accepts a tag past the end of ALL"
        );
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(name(*a), name(*b), "duplicate spelling");
            }
        }
    }

    #[test]
    fn stage_and_port_tags_are_dense_and_unique() {
        check(Stage::ALL, |s| s as u8, Stage::from_u8, Stage::name);
        check(Port::ALL, |p| p as u8, Port::from_u8, Port::name);
    }
}
