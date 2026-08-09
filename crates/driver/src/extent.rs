//! What a normalized value's shape actually is, once the lane's numbers are in.
//!
//! A launch plan describes a value's type once, at compile time, for every fire
//! that will ever use it. It therefore cannot write down most of the extents:
//! how many KV entries a sequence has, how many pages back them, how many rows
//! this batch member contributes are all facts of the fire, not of the program.
//! So the plan writes a *role* in the axis's place -- `kv_len`, `page_count`,
//! `row_count`, `token_count`, `sampled_rows`, `query_len`, `key_len` -- and the
//! runtime substitutes the seven numbers it has when the fire is dispatched.
//! [`Extents`] is that set of seven, and [`describe`] is the substitution.
//!
//! The result, [`ValueDesc`], is not a host convenience: it is uploaded to the
//! GPU verbatim, one per value, and the kernels index it to find where their
//! rows begin. That is why it is `#[repr(C)]`, why its size is asserted, and why
//! it carries `rows` and `last` alongside `len` even though both are derivable
//! -- a kernel that recomputed them per thread would be doing a division that
//! the host already did once.
//!
//! # Why this refuses where the C++ returned one
//!
//! The C++ `symbolic_extent` ended in `default: return 1`. An extent role the
//! runtime does not recognise -- a plan built by a newer compiler, a byte
//! corrupted in transit, a role added on one side of the ABI and not the other
//! -- became a length of one. That is the quietest possible wrong answer: no
//! diagnostic, a descriptor that looks entirely reasonable, a buffer sized for
//! one element where the kernel will address thousands, and a fault (or worse,
//! silently wrong logits) somewhere far away. The same is true of the rank cap:
//! `min(dims.size(), 4)` dropped trailing axes, and dropping an axis drops a
//! *factor* from the element count, so the scratch allocation derived from it is
//! too small by exactly that factor. Both are refusals here.
//!
//! # Why the byte counts are `u64`
//!
//! The C++ `value_bytes` computed `descriptor.len * 4` where `len` is a
//! `uint32_t`. The multiplication happens in 32 bits and only then widens to
//! `size_t`, so any value with 2^30 or more elements wrapped: a 2^30-element f32
//! value reported *zero* bytes, was clamped to the four-byte floor, and the
//! kernel then wrote four gibibytes into a four-byte allocation. `wire_value_bytes`
//! had the same shape in `(len + 7) / 8`. Widening before multiplying costs
//! nothing and is the difference between a large allocation and a heap
//! corruption, so both functions here answer in [`u64`].

use driver_api::local::PIE_EXTENT_STATIC;
use driver_api::plan::LaunchPlanValue;
use tensor_ir::DType;
use tensor_ir::types::MAX_RANK;

use super::value::concrete_dtype;

/// A runtime extent a plan can name in place of a literal dimension.
///
/// The discriminants are the `PTIR_EXTENT_*` wire bytes and must stay in that
/// order: [`Role::from_wire`] is a match on them, and the plan encoder on the
/// other side of the ABI writes the same numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// How many KV entries the sequence holds.
    KvLen = 0,
    /// How many pages back those entries.
    PageCount = 1,
    /// How many rows this fire contributes.
    RowCount = 2,
    /// How many tokens the member submitted.
    TokenCount = 3,
    /// How many rows were selected for sampling.
    SampledRows = 4,
    /// The query length of an attention stage.
    QueryLen = 5,
    /// The key length of an attention stage.
    KeyLen = 6,
}

impl Role {
    /// Every role, in wire order.
    pub const ALL: [Role; 7] = [
        Role::KvLen,
        Role::PageCount,
        Role::RowCount,
        Role::TokenCount,
        Role::SampledRows,
        Role::QueryLen,
        Role::KeyLen,
    ];

    /// The role a wire byte names, or `None` if it names none.
    ///
    /// `None` is the case the C++ turned into an extent of one. It is kept as a
    /// distinct answer so [`describe`] can refuse rather than invent.
    #[must_use]
    pub fn from_wire(byte: u8) -> Option<Role> {
        Role::ALL.get(usize::from(byte)).copied()
    }
}

/// The seven runtime extents a fire supplies.
///
/// Every field defaults to one, not zero. A zero would make the product of any
/// shape naming that role zero, and a zero-length value is refused downstream --
/// so an extent the caller forgot to set would surface as a shape error rather
/// than as the "this axis is not in play" that it actually means. One is the
/// multiplicative identity and is what an absent axis contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extents {
    /// How many KV entries the sequence holds.
    pub kv_len: u32,
    /// How many pages back those entries.
    pub page_count: u32,
    /// How many rows this fire contributes.
    pub row_count: u32,
    /// How many tokens the member submitted.
    pub token_count: u32,
    /// How many rows were selected for sampling.
    pub sampled_rows: u32,
    /// The query length of an attention stage.
    pub query_len: u32,
    /// The key length of an attention stage.
    pub key_len: u32,
}

impl Default for Extents {
    fn default() -> Self {
        Extents {
            kv_len: 1,
            page_count: 1,
            row_count: 1,
            token_count: 1,
            sampled_rows: 1,
            query_len: 1,
            key_len: 1,
        }
    }
}

impl Extents {
    /// The extent a role stands for.
    #[must_use]
    pub fn get(&self, role: Role) -> u32 {
        match role {
            Role::KvLen => self.kv_len,
            Role::PageCount => self.page_count,
            Role::RowCount => self.row_count,
            Role::TokenCount => self.token_count,
            Role::SampledRows => self.sampled_rows,
            Role::QueryLen => self.query_len,
            Role::KeyLen => self.key_len,
        }
    }
}

/// One value's resolved shape, in the layout the kernels read.
///
/// `len` is the element count, `rows` the product of every axis but the last,
/// and `last` the trailing axis. The three are redundant by construction -- a
/// kernel could divide -- and are all carried because the host can compute them
/// once per value instead of once per thread.
///
/// `dims` is fixed at [`MAX_RANK`] and the unused tail stays one, so a kernel
/// that strides over all four axes regardless of `rank` gets the right answer
/// without branching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ValueDesc {
    /// Total element count: the product of `dims[..rank]`.
    pub len: u32,
    /// The product of `dims[..rank - 1]`, or one below rank two.
    pub rows: u32,
    /// The trailing axis, or one for a scalar.
    pub last: u32,
    /// How many axes are in play.
    pub rank: u32,
    /// The `PTIR_DT_*` dtype byte, widened to the device's word.
    pub dtype: u32,
    /// Per-axis extents; entries at or past `rank` are one.
    pub dims: [u32; MAX_RANK],
}

/// The device struct is uploaded as bytes, so its size is part of the ABI: a
/// field added here without the matching change in the Metal source would shift
/// every subsequent value's descriptor by four bytes and mis-address every row.
const _: () = assert!(size_of::<ValueDesc>() == 36);

impl Default for ValueDesc {
    fn default() -> Self {
        ValueDesc {
            len: 1,
            rows: 1,
            last: 1,
            rank: 0,
            dtype: 0,
            dims: [1; MAX_RANK],
        }
    }
}

impl ValueDesc {
    /// How many bytes this value occupies in device scratch.
    ///
    /// A bool is one byte per lane on the device -- addressable, because the
    /// elementwise kernels index it -- where on the wire it is one *bit*. Every
    /// other dtype is four bytes per lane. The four-byte floor matters only for
    /// bools of fewer than four lanes; every other case already clears it.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        let len = u64::from(self.len);
        let bytes = if concrete_dtype(self.dtype_byte()) == DType::Bool {
            len
        } else {
            len * 4
        };
        bytes.max(4)
    }

    /// How many bytes this value occupies on the wire.
    ///
    /// Defers to [`wire_cell_bytes`](super::value::wire_cell_bytes), which is
    /// the codec's own answer, so the two cannot drift. The C++ had a separate
    /// `wire_value_bytes` that restated the same rule and clamped it to one
    /// byte; the clamp was dead, because a shape's product is at least one and a
    /// one-lane bool already occupies a byte.
    #[must_use]
    pub fn wire_bytes(&self) -> u64 {
        super::value::wire_cell_bytes(concrete_dtype(self.dtype_byte()), self.len as usize) as u64
    }

    /// The dtype byte, narrowed back from the device word.
    ///
    /// The truncation is safe by construction: [`describe`] only ever widens a
    /// `u8` into this field.
    #[must_use]
    fn dtype_byte(&self) -> u8 {
        self.dtype as u8
    }
}

/// Why a value's shape could not be resolved.
///
/// Each variant is a case the C++ answered with a plausible-looking descriptor
/// instead of a diagnostic. Naming them costs one match arm at the call site and
/// buys a message that says which value and which axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unresolvable {
    /// The type declared more axes than the ABI carries.
    ///
    /// The C++ took the first four and dropped the rest, which drops a factor
    /// from the element count and under-sizes every allocation derived from it.
    Rank {
        /// How many axes the type declared.
        rank: usize,
        /// How many the descriptor holds.
        limit: usize,
    },
    /// The extent-kind table and the literal table disagreed on the rank.
    ///
    /// The wire form keeps them as two parallel arrays, so nothing structural
    /// forces them to the same length; a producer that filled one and not the
    /// other would otherwise be read as a shape of whichever it indexed.
    Mismatch {
        /// How many extent kinds were given.
        extents: usize,
        /// How many literal dims were given.
        dims: usize,
    },
    /// An axis named a runtime extent this driver does not know.
    UnknownRole {
        /// The axis.
        axis: usize,
        /// The unrecognised wire byte.
        role: u8,
    },
    /// An axis resolved to zero.
    ///
    /// A zero-length value is not a value the kernels can address, and it is far
    /// more often an unset extent than a deliberate empty tensor.
    ZeroExtent {
        /// The axis.
        axis: usize,
    },
    /// The element count did not fit in the descriptor's `u32`.
    Overflow {
        /// The axis at which the running product would have wrapped.
        axis: usize,
    },
}

/// Resolve a plan's value type against a fire's extents.
///
/// # Errors
///
/// [`Unresolvable`], naming the axis at fault. Every arm is a case the value
/// cannot be described at all, not one where a fallback would do.
pub fn describe(value: &LaunchPlanValue, extents: &Extents) -> Result<ValueDesc, Unresolvable> {
    if value.extents.len() != value.dims.len() {
        return Err(Unresolvable::Mismatch {
            extents: value.extents.len(),
            dims: value.dims.len(),
        });
    }
    let rank = value.extents.len();
    if rank > MAX_RANK {
        return Err(Unresolvable::Rank {
            rank,
            limit: MAX_RANK,
        });
    }

    let mut descriptor = ValueDesc {
        rank: rank as u32,
        dtype: u32::from(value.dtype),
        ..ValueDesc::default()
    };

    // The product is checked *before* each multiply rather than after, so it
    // never actually wraps -- a post-hoc check on a wrapped `u32` cannot tell a
    // legal small product from an overflowed one.
    let mut len: u32 = 1;
    for (axis, (&kind, &literal)) in value.extents.iter().zip(value.dims.iter()).enumerate() {
        let dim = if kind == PIE_EXTENT_STATIC {
            literal
        } else {
            let role =
                Role::from_wire(kind).ok_or(Unresolvable::UnknownRole { axis, role: kind })?;
            extents.get(role)
        };
        if dim == 0 {
            return Err(Unresolvable::ZeroExtent { axis });
        }
        len = len
            .checked_mul(dim)
            .ok_or(Unresolvable::Overflow { axis })?;
        descriptor.dims[axis] = dim;
    }

    descriptor.len = len;
    // `rows` divides `len` exactly and so cannot overflow on its own: it is the
    // same product with one factor left out. The C++ re-checked it anyway, and
    // then guarded `len / rows` against a zero `rows` that its own zero-extent
    // refusal had already made impossible.
    descriptor.rows = descriptor.dims[..rank.saturating_sub(1)].iter().product();
    descriptor.last = if rank == 0 {
        1
    } else {
        descriptor.dims[rank - 1]
    };
    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire byte a role encodes is the role's discriminant, which is what
    /// makes `from_wire` an index rather than a match. Reordering `ALL` would
    /// silently remap every plan the compiler has ever emitted.
    #[test]
    fn roles_are_listed_in_wire_order_with_no_gaps() {
        for (wire, role) in Role::ALL.iter().enumerate() {
            assert_eq!(*role as usize, wire);
            assert_eq!(Role::from_wire(wire as u8), Some(*role));
        }
        assert_eq!(Role::from_wire(Role::ALL.len() as u8), None);
        assert_eq!(Role::from_wire(PIE_EXTENT_STATIC), None);
    }

    fn value(dtype: u8, axes: &[(u8, u32)]) -> LaunchPlanValue {
        LaunchPlanValue {
            dtype,
            extents: axes.iter().map(|&(kind, _)| kind).collect(),
            dims: axes.iter().map(|&(_, dim)| dim).collect(),
        }
    }

    /// The substitution itself, and the three derived numbers.
    #[test]
    fn a_symbolic_axis_takes_the_fires_extent_and_a_static_one_keeps_its_literal() {
        let extents = Extents {
            kv_len: 7,
            ..Extents::default()
        };
        let desc = describe(
            &value(0, &[(Role::KvLen as u8, 999), (PIE_EXTENT_STATIC, 5)]),
            &extents,
        )
        .expect("a rank-two shape of legal extents resolves");
        assert_eq!(
            desc.dims,
            [7, 5, 1, 1],
            "the literal under a symbolic axis is ignored"
        );
        assert_eq!(desc.len, 35);
        assert_eq!(desc.rows, 7);
        assert_eq!(desc.last, 5);
        assert_eq!(desc.rank, 2);
    }

    /// `last` is the trailing axis, which is also `len / rows`. The C++ computed
    /// it by that division; this pins that the direct form agrees, so the
    /// descriptor a kernel reads is unchanged by the simplification.
    #[test]
    fn last_equals_len_over_rows_for_every_rank() {
        let extents = Extents::default();
        for axes in [
            vec![],
            vec![(PIE_EXTENT_STATIC, 6)],
            vec![(PIE_EXTENT_STATIC, 2), (PIE_EXTENT_STATIC, 6)],
            vec![
                (PIE_EXTENT_STATIC, 2),
                (PIE_EXTENT_STATIC, 3),
                (PIE_EXTENT_STATIC, 5),
                (PIE_EXTENT_STATIC, 7),
            ],
        ] {
            let desc = describe(&value(0, &axes), &extents).expect("legal shape");
            assert_eq!(desc.last, desc.len / desc.rows, "rank {}", desc.rank);
        }
    }

    /// A shape with no axes is a scalar of one lane, not of zero.
    #[test]
    fn a_rank_zero_value_is_one_lane() {
        let desc = describe(&value(0, &[]), &Extents::default()).expect("a scalar resolves");
        assert_eq!((desc.len, desc.rows, desc.last, desc.rank), (1, 1, 1, 0));
        assert_eq!(desc.dims, [1; MAX_RANK]);
    }

    /// The case the C++ answered with `1`.
    #[test]
    fn an_unknown_extent_role_is_refused_rather_than_treated_as_one() {
        let unknown = Role::ALL.len() as u8;
        assert_eq!(
            describe(&value(0, &[(unknown, 0)]), &Extents::default()),
            Err(Unresolvable::UnknownRole {
                axis: 0,
                role: unknown
            }),
            "a role this driver does not know must not silently become a \
             one-element axis: the buffer sized from it would be short by \
             whatever the extent really was"
        );
    }

    /// The case the C++ answered by dropping axes.
    #[test]
    fn a_rank_past_the_descriptor_is_refused_rather_than_truncated() {
        let axes = vec![(PIE_EXTENT_STATIC, 2); MAX_RANK + 1];
        assert_eq!(
            describe(&value(0, &axes), &Extents::default()),
            Err(Unresolvable::Rank {
                rank: MAX_RANK + 1,
                limit: MAX_RANK
            }),
            "truncating to four axes drops a factor from the element count, so \
             every allocation derived from it is too small by that factor"
        );
    }

    /// Two parallel wire arrays can disagree; nothing structural stops them.
    #[test]
    fn a_kind_table_and_a_dim_table_of_different_lengths_are_refused() {
        let mut broken = value(0, &[(PIE_EXTENT_STATIC, 2), (PIE_EXTENT_STATIC, 3)]);
        broken.dims.pop();
        assert_eq!(
            describe(&broken, &Extents::default()),
            Err(Unresolvable::Mismatch {
                extents: 2,
                dims: 1
            })
        );
    }

    /// A zero extent is refused, which is what makes `rows` non-zero and the
    /// C++'s division guard dead.
    #[test]
    fn a_zero_extent_is_refused() {
        assert_eq!(
            describe(
                &value(0, &[(PIE_EXTENT_STATIC, 4), (PIE_EXTENT_STATIC, 0)]),
                &Extents::default()
            ),
            Err(Unresolvable::ZeroExtent { axis: 1 })
        );
    }

    /// The product is bounded before it is taken, so a shape that would not fit
    /// is named rather than wrapped into a plausible small one.
    #[test]
    fn an_element_count_past_u32_is_refused_at_the_axis_that_overflows_it() {
        assert_eq!(
            describe(
                &value(
                    0,
                    &[(PIE_EXTENT_STATIC, 1 << 16), (PIE_EXTENT_STATIC, 1 << 16)]
                ),
                &Extents::default()
            ),
            Err(Unresolvable::Overflow { axis: 1 })
        );
    }

    /// Bool is a byte per lane on the device and a bit per lane on the wire.
    /// This asymmetry is the whole reason the two byte counts are separate
    /// functions rather than one.
    #[test]
    fn bool_is_a_byte_on_the_device_and_a_bit_on_the_wire() {
        let desc = describe(
            &value(DType::Bool as u8, &[(PIE_EXTENT_STATIC, 64)]),
            &Extents::default(),
        )
        .expect("legal shape");
        assert_eq!(desc.device_bytes(), 64);
        assert_eq!(desc.wire_bytes(), 8);
    }

    /// The floor is live for exactly one shape of value: a bool short enough
    /// that its addressable form is under a word.
    #[test]
    fn a_short_bool_is_still_rounded_up_to_a_word_on_the_device() {
        let desc = describe(
            &value(DType::Bool as u8, &[(PIE_EXTENT_STATIC, 3)]),
            &Extents::default(),
        )
        .expect("legal shape");
        assert_eq!(desc.device_bytes(), 4);
        assert_eq!(desc.wire_bytes(), 1);
    }

    /// The C++ multiplied a `uint32_t` count by four in 32 bits before widening,
    /// so a value of 2^30 f32 lanes reported zero bytes, was clamped to the
    /// four-byte floor, and the kernel then wrote four gibibytes into it.
    #[test]
    fn a_huge_value_reports_its_real_byte_count_instead_of_wrapping_to_zero() {
        let desc = describe(
            &value(DType::F32 as u8, &[(PIE_EXTENT_STATIC, 1 << 30)]),
            &Extents::default(),
        )
        .expect("2^30 lanes fits a u32 count");
        assert_eq!(
            desc.device_bytes(),
            4 << 30,
            "widening before multiplying is the difference between a large \
             allocation and a heap corruption"
        );
    }

    /// Every extent defaults to the multiplicative identity, so an axis whose
    /// role the fire did not set contributes nothing rather than annihilating
    /// the shape.
    #[test]
    fn every_default_extent_is_one() {
        let extents = Extents::default();
        for role in Role::ALL {
            assert_eq!(extents.get(role), 1, "{role:?}");
        }
    }
}
