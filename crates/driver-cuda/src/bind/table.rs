//! The shape wgpu and metal already have, on CUDA: [`Facts`], [`Handles`],
//! [`operands`], [`dispatch`]. One operand order for all three backends,
//! stated at [`BoundLaunch::args`]: inputs, outputs, weights.
//!
//! The arms below are `#[cfg(test)]` fixtures, a second opinion the derived
//! column is diffed against.

use core::ffi::c_void;

use kernels::Derived;
use kernels::keys;
use kernels::Lit;
use kernels::Refusal;
use kernels::{Kind, Source};
use kernels_cuda::ArgValue;
use kernels_cuda::attn::Rows;

use super::cx::Cx;
use super::{BoundArg, BoundLaunch, LaunchSpec};

// ── Facts ─────────────────────────────────────────────────────────────────

/// `let x = f.thing()?` where the refusal is the one [`Cx`] mints — one
/// string, in `bind/cx.rs`, so the two paths cannot name an absence twice.
macro_rules! forward {
    ($(#[$m:meta])* $name:ident -> $ty:ty) => {
        $(#[$m])*
        /// # Errors
        #[must_use = "a fact is read to be bound"]
        pub fn $name(self) -> Result<$ty, Refusal> {
            self.cx.$name()
        }
    };
}

/// Where a launch's facts are read from, not what was found.
///
/// A cursor: `cx`, plus the two answers that cannot refuse. Every other fact
/// is a query at the ask, forwarding to `bind/cx.rs` and refusing in its own
/// words. `cublas` must never appear here — `Cx` is query-only.
#[derive(Clone, Copy, Debug)]
pub struct Facts<'a> {
    /// Where every fact below is read from. `pub` for [`operand`]'s match.
    pub cx: &'a Cx<'a>,
    /// Which rows this launch covers, and how many the whole fire has.
    pub rows: Rows,
    /// Which layer this statement belongs to. Always answerable.
    pub layer: usize,
}

impl<'a> Facts<'a> {
    /// Look at one fire. Infallible, and it resolves nothing.
    #[must_use]
    pub fn at(cx: &'a Cx<'a>) -> Self {
        Self { cx, rows: cx.rows(), layer: cx.layer() }
    }

    forward!(
        token_ids -> *const i32
    );
    forward!(
        positions -> *const i32
    );
    forward!(
        /// The rows a sampling gather collects.
        sampling_indices -> *const i32
    );
    forward!(
        vocab -> i32
    );
    forward!(
        /// The per-layer-embedding width.
        ple_dim -> i32
    );
    forward!(
        head_dim -> i32
    );
    forward!(
        /// How many of each head's elements rotate.
        rotary_width -> i32
    );
    forward!(
        /// How far left of a row attention may see; `-1` is unbounded.
        window_left -> i32
    );
    forward!(
        /// The rotary base for THIS STATEMENT'S LAYER.
        theta -> f32
    );
    forward!(
        /// The FIRE's rope base, which is not [`Facts::theta`]: gemma-4 splits
        /// theta by layer kind, so the two differ on one fire.
        rope_theta -> f32
    );
    forward!(
        rms_eps -> f32
    );
    forward!(
        /// The softmax scale this fire was PLANNED with. Refuses rather than
        /// answering `1.0`, which is both the fallback and a real family's scale.
        sm_scale -> f32
    );
    forward!(
        /// How many experts one token visits. Refuses rather than answering zero.
        experts_per_token -> i32
    );
    forward!(
        moe_norm_topk -> bool
    );
    forward!(
        moe_routed_scaling -> f32
    );
    forward!(
        glu_alpha -> f32
    );
    forward!(
        /// [`Facts::glu_alpha`]'s pair, and the half that actually varies.
        glu_limit -> f32
    );

    // ── Four that spell a different name: `kv_layer_heads` is the CACHE's ──

    /// How many query heads.
    #[must_use = "a fact is read to be bound"]
    pub fn q_heads(self) -> Result<i32, Refusal> {
        self.cx.num_q_heads()
    }

    /// How many key/value heads. NOT [`Facts::kv_layer_heads`].
    #[must_use = "a fact is read to be bound"]
    pub fn kv_heads(self) -> Result<i32, Refusal> {
        self.cx.num_kv_heads()
    }

    /// The statement's per-head width, or `None` for the plain kind.
    #[must_use]
    pub fn per_head_dim(self) -> Option<i32> {
        self.cx.per_head_dim()
    }

    /// Whether the rotation pairs adjacent elements rather than halves.
    #[must_use]
    pub fn rope_interleaved(self) -> bool {
        self.cx.rope_interleaved()
    }

    // ── Four the `Copy` bound put out of reach: `cx.kv_layer()` fields ──

    /// Token rows per KV page. NOT [`Facts::head_dim`]: equal in no deployment.
    #[must_use = "a fact is read to be bound"]
    pub fn page_size(self) -> Result<i32, Refusal> {
        self.cx.kv_layer().map(|l| l.page_size)
    }

    /// The CACHE's elements per head, NOT [`Facts::head_dim`] — that is the
    /// width attention is COMPUTED at. The two agree wherever the rotary width
    /// is the full head, so a swap reads a neighbour's row rather than faults.
    #[must_use = "a fact is read to be bound"]
    pub fn kv_head_dim(self) -> Result<i32, Refusal> {
        self.cx.kv_layer().map(|l| l.head_dim)
    }

    /// The CACHE's key/value head count. NOT [`Facts::kv_heads`].
    #[must_use = "a fact is read to be bound"]
    pub fn kv_layer_heads(self) -> Result<i32, Refusal> {
        self.cx.kv_layer().map(|l| l.num_kv_heads)
    }

    /// Whether this layer's pages are laid out head-major.
    #[must_use = "a fact is read to be bound"]
    pub fn kv_hnd(self) -> Result<bool, Refusal> {
        self.cx.kv_layer().map(|l| l.hnd)
    }

    /// How many REQUESTS, NOT how many ROWS: equal during a one-token decode.
    #[must_use = "a fact is read to be bound"]
    pub fn requests(self) -> Result<i32, Refusal> {
        self.cx.plan().map(|p| p.requests)
    }

    /// The `i`th named weight's address. Refuses `Absent`, being indexed.
    #[must_use = "a fact is read to be bound"]
    pub fn weight_named(self, i: usize) -> Result<*mut c_void, Refusal> {
        self.cx.weight_named(i)
    }
}

// ── Handles ───────────────────────────────────────────────────────────────

/// The launch's operands, handed out as the values a body binds.
///
/// On CUDA an operand IS an address, so this mints [`ArgValue::Ptr`]. Each
/// call appends to [`Handles::taken`], so the order a body asks in is the
/// order `kernels_cuda::call` binds positionally against the `fn`'s
/// parameters.
pub struct Handles<'a> {
    /// The statement's inputs, in stated order.
    ins: &'a [BoundArg],
    /// Its outputs.
    outs: &'a [BoundArg],
    /// The weights it names positionally.
    weights: &'a [BoundArg],
    /// What the body has asked for so far, in the order it asked.
    taken: Vec<*mut c_void>,
}

impl<'a> Handles<'a> {
    /// The operands of one launch, ready to be asked for.
    ///
    /// A WIDTH CANNOT DISCRIMINATE — `width == 0` is both a weight and a value
    /// stating no row width — so the counts split the run, clamped not trusted.
    #[must_use]
    pub fn over(args: &'a [BoundArg], inputs: usize, results: usize) -> Self {
        let inputs = inputs.min(args.len());
        let (ins, rest) = args.split_at(inputs);
        let results = results.min(rest.len());
        let (outs, weights) = rest.split_at(results);
        Self { ins, outs, weights, taken: Vec::new() }
    }

    /// The `n`th INPUT, as the address the kernel receives.
    pub fn input(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.ins.get(n).ok_or(Refusal::Absent { what: "an input operand" })?;
        Ok(self.take(at))
    }

    /// The `n`th OUTPUT.
    pub fn output(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.outs.get(n).ok_or(Refusal::Absent { what: "an output operand" })?;
        Ok(self.take(at))
    }

    /// The `n`th positional WEIGHT.
    pub fn weight(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.weights.get(n).ok_or(Refusal::Absent { what: "a weight" })?;
        Ok(self.take(at))
    }

    /// The `n`th input's row width in elements. A width occupies no argument.
    pub fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        width(self.ins.get(n), "an input's width")
    }

    /// The `n`th output's row width. See [`Self::in_width`].
    pub fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        width(self.outs.get(n), "an output's width")
    }

    /// The addresses the body asked for, in the order it asked.
    #[must_use]
    pub fn taken(&self) -> &[*mut c_void] {
        &self.taken
    }

    fn take(&mut self, at: BoundArg) -> ArgValue {
        self.taken.push(at.ptr);
        ArgValue::Ptr(at.ptr)
    }
}

/// One operand's row width, or the refusal that names it. Absence is the live
/// half: `unwrap_or(0)` at a call site turns "there is no second result" into
/// "the second result has no pitch".
fn width(at: Option<&BoundArg>, what: &'static str) -> Result<i32, Refusal> {
    let w = at.ok_or(Refusal::Absent { what })?.width;
    i32::try_from(w).ok().filter(|w| *w > 0).ok_or(Refusal::Absent { what })
}

// ── The arms ──────────────────────────────────────────────────────────────

/// One kernel's operand resolution, as the table path states it. `#[cfg(test)]`
/// with the six bodies below: a second opinion written by hand from the
/// kernels' docs, which `every_table_arm_agrees_with_its_own_column` consults.
#[cfg(test)]
pub type TableArm = fn(&mut Handles<'_>, Facts) -> Result<Vec<ArgValue>, Refusal>;

/// `sample::lm_head_gemv_argmax_int8`. Head and scale are NAMED weights.
#[cfg(test)]
pub fn lm_head_gemv_argmax_int8(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let hidden_states = o.input(0)?;
    let lm_head_weight = ArgValue::Ptr(f.weight_named(0)?);
    let scale_inv = ArgValue::Ptr(f.weight_named(1)?);
    let token_ids = o.output(0)?;
    let hidden = o.in_width(0)?;
    Ok(vec![
        as_region(hidden_states, f.rows.count, hidden),
        lm_head_weight,
        scale_inv,
        // `vocab` is ASKED for now, so it is not the statement's to bind.
        as_region(token_ids, f.rows.count, o.out_width(0).unwrap_or(0)),
    ])
}

/// `layout::split_bf16_rows`. `left_dim`/`right_dim` are the OUTPUTS' widths.
#[cfg(test)]
pub fn split_bf16_rows(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let src = o.input(0)?;
    let left = o.output(0)?;
    let right = o.output(1)?;
    let left_dim = o.out_width(0)?;
    let right_dim = o.out_width(1)?;
    Ok(vec![
        as_region(src, f.rows.count, o.in_width(0)?),
        as_region(left, f.rows.count, left_dim),
        as_region(right, f.rows.count, right_dim),
    ])
}

/// An operand and its shape, assembled independently of [`operand`].
#[cfg(test)]
fn as_region(v: ArgValue, rows: i32, width: i32) -> ArgValue {
    match v {
        ArgValue::Ptr(ptr) => ArgValue::Region { ptr, rows, width },
        other => other,
    }
}

/// `layout::split_qwen_gdn_ba_bf16`. One list for either instantiation.
#[cfg(test)]
pub fn split_qwen_gdn_ba_bf16(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let ba = o.input(0)?;
    let b_out = o.output(0)?;
    let a_out = o.output(1)?;
    let v_h = o.out_width(0)?;
    Ok(vec![
        as_region(ba, f.rows.count, o.in_width(0).unwrap_or(0)),
        as_region(b_out, f.rows.count, v_h),
        as_region(a_out, f.rows.count, o.out_width(1).unwrap_or(0)),
    ])
}

/// `layout::embed_bf16`, as the STATEMENT sources it.
///
/// The token ids and the vocabulary left this list when they became asks: a
/// column carries what the statement placed, and a fact the body asks for
/// reaches the kernel through `Answering` instead. Comparing the column
/// against a hand list that still held them compared two different questions.
#[cfg(test)]
pub fn embed_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let weight = ArgValue::Ptr(f.weight_named(0)?);
    let y = o.output(0)?;
    let hidden = o.out_width(0)?;
    Ok(vec![weight, as_region(y, f.rows.count, hidden)])
}

/// `layout::gather_bf16_rows`. `row_indices` derives as `In(1)`, `*const i32`.
#[cfg(test)]
pub fn gather_bf16_rows(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let src = o.input(0)?;
    let dst = o.output(0)?;
    let width = o.out_width(0)?;
    // `row_indices` is ASKED for (`keys::SamplingIndices`), so it is not the
    // statement's to bind.
    Ok(vec![
        as_region(src, f.rows.count, o.in_width(0).unwrap_or(0)),
        as_region(dst, f.rows.count, width),
    ])
}

/// `layout::transpose_bf16_nld_to_lnd`. Places operands, computes nothing.
#[cfg(test)]
pub fn transpose_bf16_nld_to_lnd(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let src = o.input(0)?;
    let dst = o.output(0)?;
    let width = o.in_width(0)?;
    // `ple_dim` is ASKED for now, so it is not the statement's to bind.
    Ok(vec![
        as_region(src, f.rows.count, width),
        as_region(dst, f.rows.count, o.out_width(0).unwrap_or(0)),
    ])
}

// ── The arm nobody writes ─────────────────────────────────────────────────

/// Bind one launcher's whole operand list from the column `#[routine]` derived.
///
/// It takes the COLUMN and not a symbol: a lookup by name would re-introduce
/// the registry the derivation deletes. `&[]` refuses rather than binding
/// nothing.
pub fn operands(
    o: &mut Handles<'_>,
    f: Facts,
    derived: &[Derived],
    sources: &[Option<Source>],
    args: &[kernels::Ty],
) -> Result<Vec<ArgValue>, Refusal> {
    if derived.is_empty() {
        return Err(Refusal::Unstated { what: "an operand column: the row states no `derived`" });
    }
    // THE SAME COLUMN THE THREE SHADER PLANES BIND FROM. It used to be a
    // second one: `#[routine]` read the SYNTAX and filled `Derived::source`,
    // while `KernelFn::SOURCES` read the TYPES, and CUDA bound from the first.
    // A dump of the two found rows where they disagreed. `derived` carries
    // what a type cannot -- the parameter's name, and whether a null may land
    // there -- and the source comes from here.
    derived
        .iter()
        .zip(sources.iter().copied().chain(core::iter::repeat(None)))
        .enumerate()
        .map(|(i, (d, source))| {
            // A NULLABLE PARAMETER WHOSE SOURCE IS ABSENT TAKES THE NULL, and
            // `Derived::nullable` comes from the parameter's own type. Narrowed
            // to `Absent` deliberately: any other refusal a null would bury.
            let v = match operand(o, f, d, source) {
                Err(Refusal::Absent { .. }) if d.nullable => {
                    ArgValue::Ptr(core::ptr::null_mut())
                }
                other => other?,
            };
            Ok(as_declared(args.get(i).copied(), v))
        })
        .collect()
}

/// The same value, in the width the launcher's own signature asks for.
///
/// A [`Source`] says WHICH fact, never how wide. A negative `I32` is not made
/// unsigned, nor a `U32` past `i32::MAX` signed: `call()` refuses with the
/// parameter's real name rather than wrapping.
fn as_declared(t: Option<kernels::Ty>, v: ArgValue) -> ArgValue {
    use kernels::Ty;
    let Some(t) = t else { return v };
    match (t, v) {
        (Ty::Usize, ArgValue::I32(n)) if n >= 0 => ArgValue::Usize(n as usize),
        (Ty::Usize, ArgValue::U32(n)) => ArgValue::Usize(n as usize),
        (Ty::I64, ArgValue::I32(n)) => ArgValue::I64(i64::from(n)),
        (Ty::I64, ArgValue::U32(n)) => ArgValue::I64(i64::from(n)),
        (Ty::U32, ArgValue::I32(n)) if n >= 0 => ArgValue::U32(n as u32),
        (Ty::I32, ArgValue::U32(n)) if n <= i32::MAX as u32 => ArgValue::I32(n as i32),
        _ => v,
    }
}





/// One half of a [`Source::Times`] or a [`Source::Over`], as a number.
///
/// The arithmetic sources compose — a factor may itself be a chain or another
/// product — so this is [`operand`] again with the answer narrowed. A POINTER
/// is not a factor: it refuses by the parameter's name rather than being cast
/// to one, because an address multiplied by a width is a grid nothing checks.
fn count(o: &mut Handles<'_>, f: Facts<'_>, d: &Derived, source: Source) -> Result<i32, Refusal> {
    let too_wide = || Refusal::Unstated { what: "a factor that fits an `i32`" };
    match operand(o, f, d, Some(source))? {
        ArgValue::I32(n) => Ok(n),
        ArgValue::U32(n) => i32::try_from(n).map_err(|_| too_wide()),
        ArgValue::Usize(n) => i32::try_from(n).map_err(|_| too_wide()),
        ArgValue::I64(n) => i32::try_from(n).map_err(|_| too_wide()),
        _ => Err(Refusal::Unstated { what: d.name }),
    }
}

/// One operand, from where its [`Source`] says it comes.
///
/// Handles are the statement's own, minted in the order asked. Facts go
/// through the accessors the hand arms use and refuse in the same words. The
/// aggregates refuse by design — they cannot be `Copy` and lifetime-free — so
/// a symbol needing one keeps its hand arm. The binder computes nothing.
fn operand(
    o: &mut Handles<'_>,
    f: Facts<'_>,
    d: &Derived,
    source: Option<Source>,
) -> Result<ArgValue, Refusal> {
    let ptr = |p: *const i32| ArgValue::Ptr(p.cast_mut().cast::<c_void>());
    /// The same, for a fact whose declared pointee is not `i32`.
    fn anyptr<T>(p: *const T) -> ArgValue {
        ArgValue::Ptr(p.cast_mut().cast::<c_void>())
    }
    // Total, per the YaRN keys' declarations: absence is `NONE`'s numbers.
    fn yarn_or_none(f: Facts<'_>) -> kernels_cuda::rope::Yarn {
        f.cx.yarn().unwrap_or(kernels_cuda::rope::Yarn::NONE)
    }
    // A statement places a region — address, row count, pitch — and the
    // SIGNATURE decides how much to keep. A missing width is zero, not a refusal.
    let region = |v: ArgValue, width: i32| match v {
        ArgValue::Ptr(p) => ArgValue::Region { ptr: p, rows: f.rows.count, width },
        other => other,
    };
    match source {
        // ONE ADDRESS IN TWO SLOTS, resolved as the INPUT: that is the address
        // the statement placed, and the allocator has already given the result
        // the same offset off the same `Source::Alias`.
        Some(Source::Alias(n, _)) => {
            let width = o.in_width(n as usize).unwrap_or(0);
            o.input(n as usize).map(|v| region(v, width))
        }
        // The statement's own three.
        Some(Source::Slot(Kind::In, n)) => {
            let width = o.in_width(n as usize).unwrap_or(0);
            o.input(n as usize).map(|v| region(v, width))
        }
        Some(Source::Slot(Kind::Out, n)) => {
            let width = o.out_width(n as usize).unwrap_or(0);
            o.output(n as usize).map(|v| region(v, width))
        }
        // NOT A REGION: a weight's shape is the MODEL's, not the statement's.
        Some(Source::Slot(Kind::Weight, n)) => o.weight(n as usize),
        Some(Source::Named(<keys::NamedWeight as keys::Fact>::KEY)) => f.weight_named(0).map(ArgValue::Ptr),
        // The second bank: `sample::lm_head_gemv_argmax_int8` names two weights.
        Some(Source::Named(<keys::NamedWeight2 as keys::Fact>::KEY)) => f.weight_named(1).map(ArgValue::Ptr),

        // THE WEIGHT CHAIN: the named bank first, the positional one after.
        //
        // Both halves are arms directly above, and for the five marks that was
        // the whole of it — `Weight<N, _>` derived the named one and
        // `Bank<N, _>` the positional one, so each reached its own arm and no
        // chain existed to resolve. The four marks resolve ONE mark to BOTH:
        // `Const<Tensor<E>>` derives `Or(Named("weight"), Slot(Kind::Weight,
        // 0))` for every weight a routine takes. Without this arm that chain
        // fell to the catch-all and EVERY weight refused "nothing states
        // weight" — `layout::embed_bf16`, the first launch of every fire, so
        // no model reached its second kernel.
        //
        // `kernels::bind` carries the same arm for the shader planes and says
        // the same thing; this is the CUDA half of it.
        //
        // Order matters, and so does what each half COSTS. `Facts` accessors
        // read and mint nothing, so a discarded first half shifts no handle;
        // `o.weight(n)` is the one that numbers into `taken`, and it is
        // reached only on the fallback.
        Some(Source::Or(&Source::Named(key), fallback)) => {
            match operand(o, f, d, Some(Source::Named(key))) {
                Ok(v) => Ok(v),
                Err(_) => operand(o, f, d, Some(*fallback)),
            }
        }
        // THE SCALAR CHAIN: the statement's own scalar first, a fact or a
        // literal after. ZERO IS ABSENT — a grid axis of zero launches
        // nothing, so a statement carrying one is stating no preference.
        //
        // Safe for the reason the weight chain is, and the reason is the other
        // way round: `param` reads the wire run and mints nothing at all, so
        // it is the FALLBACK here that may number a handle, and it runs once.
        Some(Source::Or(&Source::Slot(Kind::Param, n), fallback)) => {
            match f.cx.param(n as usize).ok().filter(|v| *v > 0) {
                Some(stated) => Ok(ArgValue::U32(stated)),
                None => operand(o, f, d, Some(*fallback)),
            }
        }
        // THE SLOT CHAIN: a statement's own operand first, a fact after.
        //
        // `arms/fa2.rs`'s `o_or` written as a source. That function read
        // `cx.arg_out(0)` and fell back on the guard's arena slot when the
        // text declined to declare a result, and it was called from six arms;
        // `Or(Slot(Out, 0), Named(AttnOOut))` is the same chain said once, at
        // the parameter, where the routine's own signature can carry it.
        //
        // Safe for the reason the two chains above are, and the reason must be
        // checked rather than assumed: every `Handles` accessor refuses BEFORE
        // it mutates `taken`, so a discarded first half numbers no handle and
        // the fallback's is the only one that lands. An accessor that minted
        // first would shift every handle after it.
        //
        // AFTER the `Param` arm and not before: that one reads the wire run
        // and treats a stated zero as absent, which this must not do -- slot
        // zero is a real operand.
        Some(Source::Or(&Source::Slot(kind, n), fallback)) => {
            match operand(o, f, d, Some(Source::Slot(kind, n))) {
                Ok(v) => Ok(v),
                Err(_) => operand(o, f, d, Some(*fallback)),
            }
        }
        // A CHAIN WHOSE FIRST HALF IS ITSELF COMPOSITE, which the three arms
        // above cannot spell as patterns. Recursion answers it and the same
        // minting rule holds, because it is the accessors' and not this
        // function's.
        Some(Source::Or(first, fallback)) => match operand(o, f, d, Some(*first)) {
            Ok(v) => Ok(v),
            Err(_) => operand(o, f, d, Some(*fallback)),
        },

        // ARITHMETIC ON WHAT IS KNOWN. Both halves are themselves sources,
        // because a factor may be a chain: `rms_strided_head_row` wants a row's
        // width over one head's length, and a statement may carry the head
        // length where the fire answers when it does not.
        //
        // `kernels::bind` carries these two for the shader planes and has
        // since before this file had either. Their absence here was not a
        // refusal by design -- the catch-all below said `Unstated` with the
        // parameter's name, which reads as *"nothing states it"* about a
        // parameter whose source is fully stated and merely unresolved.
        Some(Source::Times(a, b)) => {
            let (x, y) = (count(o, f, d, *a)?, count(o, f, d, *b)?);
            Ok(ArgValue::I32(x.saturating_mul(y)))
        }
        Some(Source::Over(a, b)) => {
            let divisor = count(o, f, d, *b)?;
            if divisor == 0 {
                return Err(Refusal::Empty { what: "a divisor" });
            }
            Ok(ArgValue::I32(count(o, f, d, *a)? / divisor))
        }

        // Extents the statement carries. Not arithmetic -- a width is read.
        Some(Source::Slot(Kind::InWidth, n)) => o.in_width(n as usize).map(ArgValue::I32),
        Some(Source::Slot(Kind::OutWidth, n)) => o.out_width(n as usize).map(ArgValue::I32),
        Some(Source::Named(<keys::Rows as keys::Fact>::KEY)) => Ok(ArgValue::I32(f.rows.count)),

        // A REGION'S OWN SIZE: only result `n` knows, so nothing can disagree.
        Some(Source::Slot(Kind::OutElements, n)) => o
            .out_width(n as usize)
            .map(|w| ArgValue::I32(f.rows.count.saturating_mul(w))),

        // The fire's, through the accessors the hand arms use.
        Some(Source::Named(<keys::TokenIds as keys::Fact>::KEY)) => f.token_ids().map(ptr),
        Some(Source::Named(<keys::Positions as keys::Fact>::KEY)) => f.positions().map(ptr),
        Some(Source::Named(<keys::SamplingIndices as keys::Fact>::KEY)) => f.sampling_indices().map(ptr),
        Some(Source::Named(<keys::KvPageSize as keys::Fact>::KEY)) => f.page_size().map(ArgValue::I32),

        // THE THREE FIELDS OF A LAYER VIEW THAT ARE NUMBERS; dtype, scheme and
        // page strides stay refused. `KvHeadDim` is NOT `HeadDim` — see above.
        Some(Source::Named(<keys::KvHeadDim as keys::Fact>::KEY)) => f.kv_head_dim().map(ArgValue::I32),
        Some(Source::Named(<keys::KvNumHeads as keys::Fact>::KEY)) => f.kv_layer_heads().map(ArgValue::I32),
        Some(Source::Named(<keys::KvHndLayout as keys::Fact>::KEY)) => {
            f.kv_hnd().map(ArgValue::Bool)
        }

        // Zero is the PLAIN KIND, not absence: `if per_head_dim == 0 { width }`.
        Some(Source::Named(<keys::PerHeadDim as keys::Fact>::KEY)) => Ok(ArgValue::I32(f.per_head_dim().unwrap_or(0))),
        Some(Source::Named(<keys::Vocab as keys::Fact>::KEY)) => f.vocab().map(ArgValue::I32),
        Some(Source::Named(<keys::PleDim as keys::Fact>::KEY)) => f.ple_dim().map(ArgValue::I32),
        Some(Source::Named(<keys::RmsEps as keys::Fact>::KEY)) => f.rms_eps().map(ArgValue::F32),
        Some(Source::Named(<keys::Theta as keys::Fact>::KEY)) => f.theta().map(ArgValue::F32),
        // NOT `Theta`: this is the fire's field, and gemma-4 makes them differ.
        Some(Source::Named(<keys::RopeTheta as keys::Fact>::KEY)) => f.rope_theta().map(ArgValue::F32),

        // TOTAL ON `DispatchCtx`, SO THE FALLBACK IS THE FACT AND NOT A GUESS:
        // `false` and `1.0` are what a router that says nothing means.
        Some(Source::Named(<keys::MoeNormTopk as keys::Fact>::KEY)) => {
            Ok(ArgValue::Bool(f.moe_norm_topk().unwrap_or(false)))
        }
        Some(Source::Named(<keys::MoeRoutedScaling as keys::Fact>::KEY)) => {
            Ok(ArgValue::F32(f.moe_routed_scaling().unwrap_or(1.0)))
        }
        // The same, both from `MlpGate::SiluClamped`; the numbers are gpt-oss's.
        Some(Source::Named(<keys::GluAlpha as keys::Fact>::KEY)) => {
            Ok(ArgValue::F32(f.glu_alpha().unwrap_or(1.702)))
        }
        Some(Source::Named(<keys::GluLimit as keys::Fact>::KEY)) => {
            Ok(ArgValue::F32(f.glu_limit().unwrap_or(7.0)))
        }
        // AND THE FOURTH REFUSES: a dense fire gets `Unstated`, not a zero.
        Some(Source::Named(<keys::ExpertsPerToken as keys::Fact>::KEY)) => f.experts_per_token().map(ArgValue::I32),
        // `Option` for the inverse reason: 1.0 is gemma-4's real scale.
        Some(Source::Named(<keys::SmScale as keys::Fact>::KEY)) => f.sm_scale().map(ArgValue::F32),
        Some(Source::Named(<keys::RotaryWidth as keys::Fact>::KEY)) => f.rotary_width().map(ArgValue::I32),
        Some(Source::Named(<keys::WindowLeft as keys::Fact>::KEY)) => f.window_left().map(ArgValue::I32),
        Some(Source::Named(<keys::HeadDim as keys::Fact>::KEY)) => f.head_dim().map(ArgValue::I32),
        Some(Source::Named(<keys::NumQHeads as keys::Fact>::KEY)) => f.q_heads().map(ArgValue::I32),
        Some(Source::Named(<keys::NumKvHeads as keys::Fact>::KEY)) => f.kv_heads().map(ArgValue::I32),
        Some(Source::Named(<keys::RequestCount as keys::Fact>::KEY)) => f.requests().map(ArgValue::I32),

        // §2 of `keys.rs`. A pointee is documentation; the key buys the name.
        Some(Source::Named(<keys::KvKeys as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| anyptr(l.k_pages.cast_const()))
        }
        Some(Source::Named(<keys::KvValues as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| anyptr(l.v_pages.cast_const()))
        }
        Some(Source::Named(<keys::KvHasEnvelopes as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::Bool(l.has_envelopes))
        }

        // THE PLAN'S ARRAYS: `AttnCtx` carries a `Vec` and arrives borrowed.
        Some(Source::Named(<keys::KvPageIndices as keys::Fact>::KEY)) => {
            f.cx.plan().map(|p| anyptr(p.kv_page_indices))
        }
        Some(Source::Named(<keys::KvPageIndptr as keys::Fact>::KEY)) => {
            f.cx.plan().map(|p| anyptr(p.kv_page_indptr))
        }
        Some(Source::Named(<keys::KvLastPageLens as keys::Fact>::KEY)) => {
            f.cx.plan().map(|p| anyptr(p.kv_last_page_lens))
        }

        // ── The linear-attention shape, and the plan's last two. `Cx::gdn` and
        // `Cx::plan` answered all along; a `keys::` type was the missing half.
        Some(Source::Named(<keys::GdnKHeads as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I32(g.k_h))
        }
        Some(Source::Named(<keys::GdnVHeads as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I32(g.v_h))
        }
        Some(Source::Named(<keys::GdnKDim as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I32(g.k_d))
        }
        Some(Source::Named(<keys::GdnVDim as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I32(g.v_d))
        }
        Some(Source::Named(<keys::GdnConvDim as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I32(g.conv_dim))
        }
        Some(Source::Named(<keys::GdnConvK as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I32(g.conv_k))
        }
        Some(Source::Named(<keys::GdnNumGroups as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I32(g.n_groups))
        }
        // THE TWO STRIDES STAY 64-BIT: the product overflows `i32` at scale.
        Some(Source::Named(<keys::GdnConvStride as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I64(g.conv_stride_elems))
        }
        Some(Source::Named(<keys::GdnStateStride as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::I64(g.state_stride_elems))
        }
        Some(Source::Named(<keys::GdnSlotIds as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| anyptr(g.slot_ids_d))
        }
        Some(Source::Named(<keys::GdnWriteState as keys::Fact>::KEY)) => {
            f.cx.gdn().map(|g| ArgValue::Bool(g.write_state))
        }
        // THE TWO `arms/norm.rs` ASKED FOR BY NAME.
        Some(Source::Named(<keys::AltupActive as keys::Fact>::KEY)) => f
            .cx
            .altup_active()
            .map(ArgValue::I32)
            .ok_or(Refusal::Unstated { what: "which altup stream is active" }),
        Some(Source::Named(<keys::LayerScale as keys::Fact>::KEY)) => f
            .cx
            .named_scale("layer_scale")
            .map(ArgValue::F32)
            .ok_or(Refusal::Unstated { what: "the layer's residual scale" }),

        // THE PREFILL CARVE, OFF ITS OWN QUERY: a key taking a "which carve"
        // argument would let a prefill launcher bind the decode storage.
        Some(Source::Named(<keys::AttnPrefillWorkspaceFloat as keys::Fact>::KEY)) => {
            f.cx.attn_prefill_workspace().map(|w| ArgValue::Ptr(w.float_buffer))
        }
        Some(Source::Named(<keys::AttnPrefillWorkspaceInt as keys::Fact>::KEY)) => {
            f.cx.attn_prefill_workspace().map(|w| ArgValue::Ptr(w.int_buffer))
        }
        Some(Source::Named(<keys::AttnPrefillWorkspaceFloatBytes as keys::Fact>::KEY)) => {
            f.cx.attn_prefill_workspace().map(|w| ArgValue::Usize(w.float_bytes))
        }
        Some(Source::Named(<keys::AttnPrefillWorkspaceIntBytes as keys::Fact>::KEY)) => {
            f.cx.attn_prefill_workspace().map(|w| ArgValue::Usize(w.int_bytes))
        }

        // THE ATTENTION WORKSPACE; byte counts keyed beside their buffers.
        Some(Source::Named(<keys::AttnWorkspaceFloat as keys::Fact>::KEY)) => {
            f.cx.attn_workspace().map(|w| ArgValue::Ptr(w.float_buffer))
        }
        Some(Source::Named(<keys::AttnWorkspaceInt as keys::Fact>::KEY)) => {
            f.cx.attn_workspace().map(|w| ArgValue::Ptr(w.int_buffer))
        }
        Some(Source::Named(<keys::AttnWorkspaceFloatBytes as keys::Fact>::KEY)) => {
            f.cx.attn_workspace().map(|w| ArgValue::Usize(w.float_bytes))
        }
        Some(Source::Named(<keys::AttnWorkspaceIntBytes as keys::Fact>::KEY)) => {
            f.cx.attn_workspace().map(|w| ArgValue::Usize(w.int_bytes))
        }

        // THE TWO STATE SLABS, two keys: an index makes a misspelling writable.
        Some(Source::Named(<keys::GdnConvSlab as keys::Fact>::KEY)) => {
            f.cx.slab(kernels_cuda::ssm::Slab::Conv).map(ArgValue::Ptr)
        }
        Some(Source::Named(<keys::GdnRecurrentSlab as keys::Fact>::KEY)) => {
            f.cx.slab(kernels_cuda::ssm::Slab::Recurrent).map(ArgValue::Ptr)
        }
        Some(Source::Named(<keys::QoIndptr as keys::Fact>::KEY)) => {
            f.cx.plan().map(|p| anyptr(p.qo_indptr))
        }
        Some(Source::Named(<keys::RowValid as keys::Fact>::KEY)) => {
            f.cx.plan().map(|p| anyptr(p.row_valid))
        }
        Some(Source::Named(<keys::KvEnvMin as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| anyptr(l.k_env_min))
        }
        Some(Source::Named(<keys::KvEnvMax as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| anyptr(l.k_env_max))
        }
        Some(Source::Named(<keys::KvKeyScales as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::Ptr(l.k_scales))
        }
        Some(Source::Named(<keys::KvValueScales as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::Ptr(l.v_scales))
        }
        Some(Source::Named(<keys::KvBlockSize as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::I32(l.block_size))
        }
        Some(Source::Named(<keys::KvSchemeByte as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::I32(l.scheme as i32))
        }
        Some(Source::Named(<keys::KvStorageDtype as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::I32(l.storage_dtype as i32))
        }
        Some(Source::Named(<keys::KvBf16Keys as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::Ptr(l.k_bf16_pages))
        }
        Some(Source::Named(<keys::KvBf16Values as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::Ptr(l.v_bf16_pages))
        }
        Some(Source::Named(<keys::KvPagesInBatch as keys::Fact>::KEY)) => {
            f.cx.num_pages_in_batch().map(ArgValue::I32)
        }
        Some(Source::Named(<keys::KvMaxPagesPerRequest as keys::Fact>::KEY)) => {
            f.cx.max_pages_per_request().map(ArgValue::I32)
        }
        Some(Source::Named(<keys::PeelWindow as keys::Fact>::KEY)) => {
            f.cx.peel_window().map(|p| ArgValue::Ptr(p.as_ptr().cast()))
        }
        Some(Source::Named(<keys::RopeInterleaved as keys::Fact>::KEY)) => {
            Ok(ArgValue::Bool(f.cx.rope_interleaved()))
        }
        Some(Source::Named(<keys::FirstToken as keys::Fact>::KEY)) => {
            f.cx.first_token().map(ArgValue::I32)
        }
        Some(Source::Named(<keys::YarnFactor as keys::Fact>::KEY)) => {
            Ok(ArgValue::F32(yarn_or_none(f).factor))
        }
        Some(Source::Named(<keys::YarnBetaFast as keys::Fact>::KEY)) => {
            Ok(ArgValue::F32(yarn_or_none(f).beta_fast))
        }
        Some(Source::Named(<keys::YarnBetaSlow as keys::Fact>::KEY)) => {
            Ok(ArgValue::F32(yarn_or_none(f).beta_slow))
        }
        Some(Source::Named(<keys::YarnAttentionFactor as keys::Fact>::KEY)) => {
            Ok(ArgValue::F32(yarn_or_none(f).attention_factor))
        }
        Some(Source::Named(<keys::YarnOriginalMaxPosition as keys::Fact>::KEY)) => {
            Ok(ArgValue::I32(yarn_or_none(f).original_max_position))
        }
        Some(Source::Named(<keys::AttnLseOut as keys::Fact>::KEY)) => {
            f.cx.lse_out().map(|p| ArgValue::Ptr(p.cast()))
        }
        Some(Source::Named(<keys::AttnLogitsSoftCap as keys::Fact>::KEY)) => {
            f.cx.logits_soft_cap().map(ArgValue::F32)
        }
        Some(Source::Named(<keys::FinalLogitSoftcap as keys::Fact>::KEY)) => {
            f.cx.final_logit_softcap().map(ArgValue::F32)
        }
        Some(Source::Named(<keys::KvNativeBf16 as keys::Fact>::KEY)) => {
            f.cx.kv_layer().map(|l| ArgValue::Bool(l.is_native_bf16))
        }
        Some(Source::Named(<keys::RowsTotal as keys::Fact>::KEY)) => {
            Ok(ArgValue::I32(i32::try_from(f.cx.rows().total).unwrap_or(0)))
        }
        Some(Source::Named(<keys::KvWritePage as keys::Fact>::KEY)) => {
            f.cx.w_page_d().map(|p| anyptr(p))
        }
        Some(Source::Named(<keys::KvWriteOffset as keys::Fact>::KEY)) => {
            f.cx.w_off_d().map(|p| anyptr(p))
        }
        Some(Source::Named(<keys::KvWritePageOrNull as keys::Fact>::KEY)) => {
            Ok(anyptr(f.cx.w_page_d().unwrap_or(core::ptr::null())))
        }
        Some(Source::Named(<keys::KvWriteOffsetOrNull as keys::Fact>::KEY)) => {
            Ok(anyptr(f.cx.w_off_d().unwrap_or(core::ptr::null())))
        }

        // THE SUFFIXED WEIGHTS, refused here because an unindexed absence is
        // `Unstated`. NOT `Fire::weight_bias` (`weight_named(1)`, the conv's
        // stated second name) — a different fact that shared the accessor.
        Some(Source::Named(<keys::WeightBias as keys::Fact>::KEY)) => f
            .cx
            .weight_suffixed("_bias")
            .map(ArgValue::Ptr)
            .ok_or(Refusal::Unstated { what: "the statement's bias weight" }),
        Some(Source::Named(<keys::WeightScales as keys::Fact>::KEY)) => f
            .cx
            .weight_suffixed("_scales")
            .map(ArgValue::Ptr)
            .ok_or(Refusal::Unstated { what: "the statement's scales weight" }),
        // THE ARRAY, NOT THE BANK. `arms/quant.rs` filled these two by hand
        // and said why: the kernel's first act is `packed_ptrs[expert]`, and
        // the weight chain answers the bank's own base for both of its halves.
        Some(Source::Named(<keys::WeightExpertPtrs as keys::Fact>::KEY)) => f
            .cx
            .weight_suffixed("_ptrs")
            .map(ArgValue::Ptr)
            .ok_or(Refusal::Unstated {
                what: "a per-expert pointer array for this bank; \
                       `serve::load::build_moe_expert_ptrs` builds one per plane at load, \
                       so its absence means the bank's byte count did not divide by the \
                       row's `n_experts`",
            }),
        Some(Source::Named(<keys::WeightExpertScalePtrs as keys::Fact>::KEY)) => f
            .cx
            .weight_suffixed("_scales_ptrs")
            .map(ArgValue::Ptr)
            .ok_or(Refusal::Unstated {
                what: "a per-expert scale pointer array for this bank; see `_ptrs`",
            }),
        Some(Source::Named(<keys::WeightUpBias as keys::Fact>::KEY)) => f
            .cx
            .weight_suffixed("_up_bias")
            .map(ArgValue::Ptr)
            .ok_or(Refusal::Unstated { what: "the statement's up-bias weight" }),
        Some(Source::Named(<keys::WeightGateBias as keys::Fact>::KEY)) => f
            .cx
            .weight_suffixed("_gate_bias")
            .map(ArgValue::Ptr)
            .ok_or(Refusal::Unstated { what: "the statement's gate-bias weight" }),
        // Two arms: a param is a byte run, so `f32` is a different channel.
        Some(Source::Slot(Kind::Param, n)) => f.cx.param(n as usize).map(ArgValue::U32),
        Some(Source::Slot(Kind::ParamF32, n)) => f.cx.param_f32(n as usize).map(ArgValue::F32),

        // THE FA2 DECODE PLAN, LEAF BY LEAF. See [`fa2_decode_leaves`].
        //
        // WHAT A LAUNCHER THAT PLANS ITS OWN FIRE READS.
        //
        // Not leaves: a leaf is read OFF a plan and these are read to BUILD
        // one, so they come straight off the attention context rather than
        // through `fa2_prefill_leaves` -- which would refuse, the plan it
        // reads not being valid until the launcher has filled it.
        Some(Source::Named(<keys::Fa2PrefillPlanCache as keys::Fact>::KEY)) => {
            let a = f.cx.attn_ctx()?;
            let layer = u32::try_from(f.cx.layer()).unwrap_or(0);
            let raised = super::attn_plan(a);
            if raised.is_null() {
                return Err(Refusal::Unstated { what: "a prefill plan cache to fill" });
            }
            Ok(ArgValue::Ptr(raised))
        }
        Some(Source::Named(<keys::QoIndptrHost as keys::Fact>::KEY)) => {
            let a = f.cx.attn_ctx()?;
            if a.qo_indptr_h.is_null() {
                return Err(Refusal::Absent { what: "the host QO indptr the planner walks" });
            }
            Ok(anyptr(a.qo_indptr_h))
        }
        Some(Source::Named(<keys::KvPageIndptrHost as keys::Fact>::KEY)) => {
            let a = f.cx.attn_ctx()?;
            if a.kv_page_indptr_h.is_null() {
                return Err(Refusal::Absent { what: "the host KV page indptr the planner walks" });
            }
            Ok(anyptr(a.kv_page_indptr_h))
        }
        Some(Source::Named(<keys::FireRequests as keys::Fact>::KEY)) => {
            f.requests().map(ArgValue::I32)
        }
        // THE TWO RAGGED SINKS, and NULL is an answer for all four: a capture
        // that captures nothing and a mask that masks nothing are legitimate,
        // and the launcher's own test is what decides. An `Absent` refusal
        // here would refuse the ordinary case.
        Some(Source::Named(<keys::AttnScoreOut as keys::Fact>::KEY)) => {
            f.cx.attn_ctx().map(|a| anyptr(a.score_out.cast_const()))
        }
        Some(Source::Named(<keys::AttnScoreIndptr as keys::Fact>::KEY)) => {
            f.cx.attn_ctx().map(|a| ptr(a.score_indptr_d))
        }
        Some(Source::Named(<keys::AttnMask as keys::Fact>::KEY)) => {
            f.cx.attn_ctx().map(|a| anyptr(a.mask_d))
        }
        Some(Source::Named(<keys::AttnMaskIndptr as keys::Fact>::KEY)) => {
            f.cx.attn_ctx().map(|a| ptr(a.mask_indptr_d))
        }
        Some(Source::Named(<keys::AttnScoreWindow as keys::Fact>::KEY)) => {
            f.cx.attn_ctx().map(|a| ArgValue::U32(a.score_window))
        }

        // A LITERAL NEEDS NEITHER `Handles` NOR `Facts`, and is not a default:
        // each is a number every call site passes the same.
        Some(Source::Lit(l)) => Ok(match l {
            Lit::Null => ArgValue::Ptr(core::ptr::null_mut()),
            Lit::Bool(b) => ArgValue::Bool(b),
            Lit::F32(x) => ArgValue::F32(x),
            Lit::I32(n) => ArgValue::I32(n),
        }),
        // Every accessor refuses BEFORE it mutates `taken`; `.or_else` needs it.

        // Everything else refuses; the refusal is the design, not a to-do.
        _ => Err(Refusal::Unstated { what: d.name }),
    }
}

// ── Resolution and dispatch ───────────────────────────────────────────────

/// Run one launch through the table path, if this backend has crossed it.
/// `None` means the row states no derived column.
///
/// # Errors
///
/// Whatever the arm or the routine refuses.
///
/// # Safety
///
/// `stream` must be live across the launch, and every address the arm binds
/// must address live device memory large enough for its argument.
pub unsafe fn dispatch(
    bound: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    cx: &Cx<'_>,
    stream: *mut c_void,
) -> Option<Result<(), Refusal>> {
    let mut handles = Handles::over(&bound.args, spec.n_in, spec.n_out);
    // THE ROUTED SYMBOL, NOT THE STATED ONE.
    //
    // A trace may state a name that stands for a CHOICE rather than a
    // routine: `attn::write_kv_to_pages` is declared by an `untraced!` row so
    // `check_plan` can measure a text against it, and `Boot::route` resolves
    // it to `_bf16` or `_quantised` from the KV dtype the boot settled. That
    // resolution already happened -- `spec.route` holds the row it landed on
    // -- and looking the COLUMN up by `bound.kernel` threw it away, found the
    // declaration-only row, and refused with "nothing states an operand
    // column" at the sixth launch of every fire.
    //
    // The routed row's own `symbol` is the concrete one, so the column comes
    // from the routine that is about to run rather than from the name the
    // text happened to spell.
    let symbol = match spec.route {
        crate::bind::route::Route::Bound(routed) => routed,
        _ => bound.kernel,
    };
    // `None`, NOT a refusal, for a row with no column: `None` means "take the
    // path you take today", `Some(Err(_))` means "this fire cannot run".
    let row = kernels_cuda::routine(symbol)?;
    if row.derived.is_empty() {
        return None;
    }
    // THE PRECONDITION, BEFORE THE OPERANDS. `arms/fa2.rs`'s `no_join_extras`
    // stood at the top of eight arms and was the reason six of them could not
    // become columns: a `Source` FILLS a slot and this requires two to be
    // empty, so no vocabulary of sources could ever spell it.
    //
    // It runs first for the reason the arms ran it first. A join changes the
    // ARITHMETIC and not the operands, so every address below binds correctly
    // and the launch computes the wrong thing -- the one failure a refusal
    // cannot catch after the fact.
    if row.no_join && (!spec.aux.is_empty() || spec.per_head_dim.is_some()) {
        return Some(Err(Refusal::Unstated {
            what: "a dispatch without an aux value or a per-head reading",
        }));
    }
    let args = operands(&mut handles, Facts::at(cx), row.derived, row.sources, row.args);
    let args = match args {
        Ok(args) => args,
        Err(why) => return Some(Err(why)),
    };
    // THE WHOLE BOUND LIST, on request. A wrong ADDRESS is invisible: the
    // launch succeeds and the numbers come out wrong, which is the one failure
    // a refusal-driven bring-up cannot see. `PIE_TRACE_BINDS=1` prints it.
    if tracing("PIE_TRACE_BINDS") {
        eprintln!("[bind] {symbol} n_in={} n_out={} -> {args:?}", spec.n_in, spec.n_out);
    }
    // AND WHAT THE LAUNCH LEFT IN ITS RESULTS. A binding can be right in every
    // address and still run on the wrong NUMBERS; `PIE_TRACE_VALUES=1` reads
    // the first few bf16 of every region back after the launch, which is the
    // only way to see where a forward pass stops being arithmetic.
    //
    // Deliberately expensive: it synchronises the stream. A diagnostic that
    // changed the schedule it is measuring would be worse than none.
    let peek = tracing("PIE_TRACE_VALUES");
    // WHAT THE BODY MAY STILL ASK FOR. `Env` left the parameter list, so a
    // fact only the fire can answer is no longer bound into `args` above --
    // the body asks, and this lends it the same `Handles` and `Facts` the
    // column was just bound from.
    let answering = Answering {
        handles: core::cell::RefCell::new(handles),
        facts: Facts::at(cx),
    };
    // SAFETY: this function's own contract, forwarded unchanged. Every pointer
    // the column bound came from `bound.args` or a `Facts` field the dispatch
    // site resolved; anything from nowhere at all is a refusal, not a null.
    let fired = unsafe {
        kernels_cuda::call_answering(
            symbol,
            &args,
            stream,
            cx.cublas(),
            Some(&answering),
        )
    };
    if peek && fired.is_ok() {
        peek_results(symbol, &args, spec.n_in, stream);
    }
    Some(fired)
}

/// Read the head of every REGION this launch bound back off the device.
///
/// A diagnostic, and the only one that can answer "where do the numbers stop
/// being right". Synchronises the stream and copies sixteen bytes per region;
/// reached only when `PIE_TRACE_VALUES` is set.
fn peek_results(symbol: &str, args: &[ArgValue], n_in: usize, stream: *mut c_void) {
    // SAFETY: `stream` is the fire's own and live; the sync is what makes the
    // reads below observe the launch that just ran.
    unsafe {
        cudarc::runtime::sys::cudaStreamSynchronize(stream.cast());
    }
    let mut shown = 0usize;
    for (i, a) in args.iter().enumerate() {
        let ArgValue::Region { ptr, rows, width } = *a else {
            continue;
        };
        if ptr.is_null() || rows <= 0 || width <= 0 {
            continue;
        }
        let mut host = [0u16; 8];
        // SAFETY: the region's own claim -- `rows * width` live bf16 at `ptr`.
        let rc = unsafe {
            cudarc::runtime::sys::cudaMemcpy(
                host.as_mut_ptr().cast(),
                ptr.cast_const().cast(),
                core::mem::size_of_val(&host),
                cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        if rc != cudarc::runtime::sys::cudaError::cudaSuccess {
            continue;
        }
        let f: Vec<f32> = host.iter().map(|b| bf16_to_f32(*b)).collect();
        let role = if i < n_in { "in " } else { "out" };
        eprintln!("[val] {symbol} {role}{i} rows={rows} w={width} {f:.4?}");
        shown += 1;
        if shown >= 6 {
            break;
        }
    }
}

/// Whether a diagnostic env var is set, read ONCE.
///
/// `var_os` walks the environment, and these sit in `dispatch`, which runs per
/// launch per fire -- a probe that cost a scan per launch would change the
/// schedule it exists to describe.
fn tracing(var: &'static str) -> bool {
    use std::collections::BTreeMap;
    use std::sync::{OnceLock, RwLock};
    static SEEN: OnceLock<RwLock<BTreeMap<&'static str, bool>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| RwLock::new(BTreeMap::new()));
    if let Some(&hit) = seen.read().ok().and_then(|m| m.get(var).copied()).as_ref() {
        return hit;
    }
    let hit = std::env::var_os(var).is_some();
    if let Ok(mut m) = seen.write() {
        m.insert(var, hit);
    }
    hit
}

/// A bf16 bit pattern as the float it stands for.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// THIS FIRE'S ANSWERS, for a body that asks.
///
/// The answering side of the `Env` -> `ask` move, and it answers with exactly
/// what the column answered with: [`operand`], over the same [`Handles`] and
/// the same [`Facts`]. Nothing new resolves — what changed is only that the
/// question can now be asked from inside a body instead of having to be a
/// parameter for the column to carry it.
///
/// `RefCell` because answering MINTS -- a staged fact takes a handle -- while
/// a body holds only a `&self`. The column's own borrow has ended by the time
/// the body runs, so the two never overlap.
pub(crate) struct Answering<'a> {
    handles: core::cell::RefCell<Handles<'a>>,
    facts: Facts<'a>,
}

impl<'a> Answering<'a> {
    /// THE FIRE'S ANSWERS WITH NO OPERANDS BEHIND THEM.
    ///
    /// For a hand-written arm: it places its own pointers and asks only for
    /// facts, so the handle side is empty and a body that reaches for a SLOT
    /// on one of these is refused -- which is the honest answer, there being
    /// no statement here to have placed one.
    #[must_use]
    pub(crate) fn over_facts(cx: &'a Cx<'a>) -> Self {
        Self {
            handles: core::cell::RefCell::new(Handles::over(&[], 0, 0)),
            facts: Facts::at(cx),
        }
    }
}

impl kernels::routine::Answers<kernels_cuda::jit::Cuda> for Answering<'_> {
    fn resolve(&self, ty: kernels::Ty, source: Source) -> Result<ArgValue, Refusal> {
        // A NAME THE REFUSAL CAN USE. `Derived` carries the parameter's own
        // identifier, and an asked fact has no parameter -- so it says so
        // rather than borrowing a neighbour's name.
        const ASKED: Derived = Derived { name: "a fact the body asked for", nullable: false };
        let v = operand(&mut self.handles.borrow_mut(), self.facts, &ASKED, Some(source))?;
        let v = as_declared(Some(ty), v);
        // WHAT A BODY ACTUALLY GOT, on request. An ask resolves silently and a
        // wrong ANSWER is invisible where a missing one is a refusal — so the
        // one thing that cannot be read off a failing fire is the number the
        // kernel ran on. `PIE_TRACE_ASKS=1` prints it.
        if tracing("PIE_TRACE_ASKS") {
            eprintln!("[ask] {source:?} -> {v:?}");
        }
        Ok(v)
    }
}

/// The arm a crossed symbol names instead of a hand-written one.
///
/// A family crosses by pointing its `ARMS` row here and deleting the
/// hand-written one; `route` still answers `Route::Bound`, so `unfireable`
/// still accounts for the symbol at load. NOT `arm: None`, which is
/// a `#[routine(driver)]` row and means the opposite.
///
/// # Errors
///
/// Whatever the column refuses.
pub fn derived_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let fire = cx.fire();
    // SAFETY: forwarded from `Bound::call`'s contract — `stream` is the fire's,
    // live across the launch, and `bound.args` came from the dispatch site.
    unsafe { dispatch(fire.bound, fire.spec, cx, stream) }
        .unwrap_or(Err(Refusal::Unstated { what: "an operand column, on a row that named `derived_arm`" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::facts::Fire;
    use super::super::{AttnCtx, DispatchCtx};

    // RUN THESE WITH `--features cuda-12`, OR THEY DO NOT EXIST: `pub mod bind`
    // is behind `#[cfg(feature = "_cuda")]`, so a bare check compiles nothing.

    /// A fake device address, distinct per `n`. Never dereferenced.
    fn at(n: usize) -> *mut c_void {
        (0x1000 + n * 0x100) as *mut c_void
    }

    // ── The fire fixtures ─────────────────────────────────────────────────
    // `Facts` is a CURSOR, so a fixture is a fire and goes through the same
    // accessors production does. `Probe` OWNS the parts `Fire` borrows.

    /// What a `Fire` borrows, owned so a test can hand out a [`Cx`].
    struct Probe {
        ctx: DispatchCtx,
        attn: Option<AttnCtx>,
        bound: BoundLaunch<'static>,
        spec: LaunchSpec,
        rows: i32,
        w_named: *const c_void,
        w_named2: *const c_void,
    }

    impl Probe {
        /// The fire these parts describe.
        fn fire(&self) -> Fire<'_> {
            Fire {
                bound: &self.bound,
                spec: &self.spec,
                ctx: &self.ctx,
                attn: self.attn.as_ref(),
                gdn: None,
                rows: self.rows,
                w_named: self.w_named,
                w_named2: self.w_named2,
                w_suffixed: &[],
            }
        }

        /// A fire that states NOTHING: every scalar the value its accessor reads
        /// as an absence, and `attn` is `None`.
        fn silent() -> Self {
            Self {
                ctx: dispatch_ctx(),
                attn: None,
                bound: BoundLaunch { kernel: "", rows: 0..0, layers: 0..1, args: Vec::new() },
                spec: LaunchSpec::default(),
                rows: 0,
                w_named: core::ptr::null(),
                w_named2: core::ptr::null(),
            }
        }

        /// A fire with every optional fact stated.
        ///
        /// Three constants are deliberately wrong-looking — `kv_head_dim` vs
        /// `head_dim`, `rope_theta` vs `theta`, `glu_*` vs the gpt-oss defaults
        /// — so a column reading a neighbour cannot agree by coincidence.
        fn stating_everything() -> Self {
            let mut ctx = dispatch_ctx();
            ctx.token_ids = at(5);
            ctx.positions = at(6);
            ctx.sampling_indices = at(7).cast::<i32>();
            ctx.vocab = 128_256;
            ctx.ple_dim = 256;
            ctx.head_dim = 128;
            ctx.num_q_heads = 32;
            ctx.num_kv_heads = 8;
            ctx.eps = 1e-5;
            // `theta` reads the PER-LAYER table first, so stating both differs.
            ctx.rope_theta_by_layer = vec![10_000.0];
            ctx.rope_theta = 500_000.0;
            ctx.rotary_by_layer = vec![64];
            ctx.moe_norm_topk = true;
            ctx.moe_routed_scaling = 2.5;
            // `6` because nothing else here is: 6 × 7 rows = 42, a unique product.
            ctx.experts_per_token = 6;
            ctx.glu_alpha = 1.25;
            ctx.glu_limit = 3.5;
            ctx.rows_total = 7;

            let mut attn = attn_ctx();
            attn.num_requests = 3;
            attn.sm_scale = 0.125;
            attn.layers = vec![kv_layer_view()];

            let mut spec = LaunchSpec::default();
            spec.per_head_dim = Some(128);

            Self {
                ctx,
                attn: Some(attn),
                bound: BoundLaunch { kernel: "", rows: 0..7, layers: 0..1, args: Vec::new() },
                spec,
                rows: 7,
                w_named: at(8),
                w_named2: at(9),
            }
        }
    }

    /// One KV layer, stated. `head_dim` is DELIBERATELY not the attention's:
    /// the two agree on most real checkpoints, so the fixture must disagree.
    fn kv_layer_view() -> crate::bind::abi::KvCacheLayerView {
        crate::bind::abi::KvCacheLayerView {
            layer: 0,
            source_layer: 0,
            num_pages: 32,
            page_size: 16,
            num_kv_heads: 4,
            head_dim: 80,
            scheme: crate::bind::abi::KvCacheScheme::Native,
            storage_dtype: crate::dtype::DType::Bf16,
            block_size: 0,
            k_pages: at(20),
            v_pages: at(21),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: core::ptr::null_mut(),
            v_bf16_pages: core::ptr::null_mut(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: true,
            native_bf16: true,
        }
    }

    /// An empty attention workspace.
    fn workspace() -> crate::bind::abi::AttentionWorkspaceView {
        crate::bind::abi::AttentionWorkspaceView {
            float_buffer: core::ptr::null_mut(),
            float_bytes: 0,
            int_buffer: core::ptr::null_mut(),
            int_bytes: 0,
            page_locked_int: core::ptr::null_mut(),
        }
    }

    /// A `DispatchCtx` stating nothing: every field reads as an absence.
    fn dispatch_ctx() -> DispatchCtx {
        DispatchCtx {
            stream: core::ptr::null_mut(),
            cublas: core::ptr::null_mut(),
            eps: 0.0,
            rope_theta: 0.0,
            rope_theta_by_layer: Vec::new(),
            rotary_by_layer: Vec::new(),
            head_dim: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            vocab: 0,
            gate_second: false,
            rope_interleaved: false,
            token_ids: core::ptr::null_mut(),
            positions: core::ptr::null_mut(),
            final_logit_softcap: 0.0,
            ple_dim: 0,
            moe_norm_topk: false,
            moe_routed_scaling: 0.0,
            yarn: [0.0; 4],
            yarn_original_max: 0,
            glu_limit: 0.0,
            glu_alpha: 0.0,
            situ_beta: 0.0,
            situ_linear_beta: 0.0,
            wna16_group_size: 0,
            experts_per_token: 0,
            altup_streams: 0,
            altup_active: 0,
            altup_std_mult_by_layer: Vec::new(),
            peel_window: core::ptr::null(),
            rows_total: 0,
            moe_ptrs: std::cell::Cell::new(None),
            sampling_indices: core::ptr::null(),
            sampled_rows: 0,
            scales: std::collections::BTreeMap::new(),
            lora: None,
        }
    }

    /// An `AttnCtx` stating nothing: every field reads as an absence.
    fn attn_ctx() -> AttnCtx {
        AttnCtx {
            decode_plan: core::ptr::null_mut(),
            decode_plan_full: core::ptr::null_mut(),
            prefill_plan: core::ptr::null_mut(),
            workspace: workspace(),
            prefill_workspace: workspace(),
            layers: Vec::new(),
            kv_page_indices_d: core::ptr::null(),
            kv_page_indptr_d: core::ptr::null(),
            kv_last_page_lens_d: core::ptr::null(),
            qo_indptr_d: core::ptr::null(),
            qo_indptr_h: core::ptr::null(),
            kv_page_indptr_h: core::ptr::null(),
            num_requests: 0,
            num_pages_in_batch: 0,
            max_pages_per_request: 0,
            first_token: 0,
            w_page_d: core::ptr::null(),
            w_off_d: core::ptr::null(),
            row_valid_d: core::ptr::null(),
            q_out: core::ptr::null_mut(),
            score_out: core::ptr::null_mut(),
            folded_out: core::ptr::null_mut(),
            score_indptr_d: core::ptr::null(),
            mask_d: core::ptr::null(),
            mask_indptr_d: core::ptr::null(),
            o_out: core::ptr::null_mut(),
            lse_out_d: core::ptr::null_mut(),
            score_window: 0,
            window_left: 0,
            window_left_by_layer: Vec::new(),
            logits_soft_cap: 0.0,
            sm_scale: 0.0,
        }
    }

    /// The same binding, with the SHAPE dropped. `operand()` mints a region
    /// where a hand arm answers one fact per call, so the two differ in
    /// representation only: a wrong operand is still a different address.
    fn addressed(v: &[ArgValue]) -> Vec<ArgValue> {
        v.iter()
            .map(|a| match *a {
                ArgValue::Region { ptr, .. } => ArgValue::Ptr(ptr),
                other => other,
            })
            .collect()
    }

    fn decl(symbol: &str) -> &'static [kernels::Ty] {
        kernels_cuda::routine(symbol).map_or(&[][..], |r| r.args)
    }

    /// The row's SOURCE column, which `operands` takes beside the names.
    ///
    /// One column split in two: `Derived` carries the name and the
    /// nullability, `Source` where the value comes from. `#[routine]` reads
    /// the first off the syntax and the second off the types, which is what
    /// keeps them from drifting -- see `operands`.
    fn srcs(symbol: &str) -> &'static [Option<Source>] {
        kernels_cuda::routine(symbol).map_or(&[][..], |r| r.sources)
    }

    fn operand(n: usize, width: u32) -> BoundArg {
        BoundArg { ptr: at(n), width }
    }

    /// The crossed symbols are counted, and the count moves on purpose. A
    /// symbol is crossed when [`route`](super::route::route) sends it to the
    /// derived column; crossing one changes how a real fire is planned, so the
    /// number is asserted.
    ///
    /// IT ASKS `route` AND NOT A ROW, which is the whole of what changed here.
    /// Crossing used to mean *"a `Bound` row names `derived_arm`"* and there
    /// were 129 such rows; the routine's own column says the same thing, so
    /// the rows are gone and this asks the question they answered.
    #[test]
    fn the_crossed_symbols_are_counted_and_the_count_moves_on_purpose() {
        use super::super::route::{Route, route};
        let crossed = |symbol: &str| matches!(route(symbol), Route::Bound(_));

        // One per crossed family, so a family losing its crossing fails by
        // name. ALL of them now, including `fa2` — see the negative block
        // below for what is left out and why.
        assert!(crossed("rope::rope_bf16"));
        assert!(crossed("layout::embed_bf16"));
        assert!(crossed("sample::lm_head_gemv_argmax_int8"));
        assert!(crossed("norm::add_bias_bf16"));
        assert!(crossed("mlp::swiglu_bf16"));
        assert!(crossed("moe::topk_softmax_bf16"));
        assert!(crossed("quant::bf16_to_fp16"));
        assert!(crossed("ssm::nemotron_mamba_split_bf16"));
        assert!(crossed("attn::split_qkv_bf16"));
        // XQA is a second family under `attn`'s namespace, so it needs its own
        // line: `attn::split_qkv_bf16` above would keep passing if this one
        // lost its crossing.
        assert!(crossed("attn::attention_xqa_decode_bf16_prepared"));
        assert!(crossed("gemm::act_x_wt_bf16_out_fp32"));
        assert!(crossed("dist::all_reduce_bf16"));
        // AND FA2's EIGHT, which the negative block used to hold. Their arms
        // are gone: the upload and the widening are the launchers' own calls,
        // `no_join` is a row fact, and `o_or`/`lse_slab` are a slot chain and
        // an asked key. One line for the pair that had the most left to move.
        assert!(crossed("attn::dispatch_attention_flashinfer_decode"));
        assert!(crossed("attn::attention_flashinfer_prefill"));

        // A `layout` routine no text states. DECLARED and not crossed, which
        // is the case this line is for: `#[routine(internal)]` says a trace
        // may not name it, so `route` answers `Unknown` rather than binding a
        // body other routines call.
        assert!(!crossed("layout::copy_if_valid_slot"));
        // AND THE LAST TWO HAND-WRITTEN ARMS IN THIS DRIVER, which crossed
        // when what they knew became a key. `Const<Tensor<u8>>` derived the
        // weight chain for `packed_ptrs` and both halves of that chain answer
        // the BANK -- while the kernel's first act is `packed_ptrs[expert]`.
        // `keys::WeightExpertPtrs` says the array, and its `PerExpert` wrapper
        // (a driver deciding `keys::WeightScales` meant the scale ARRAY) went
        // with it.
        assert!(crossed("quant::mxfp4_moe_gate_up_decode_bf16"));
        assert!(crossed("quant::mxfp4_moe_down_decode_bf16"));
        // A DRIVER OP, declared so `check_plan` can measure a text against it
        // and fired by `bind::dispatch`'s own match.
        assert!(!crossed("gemm::lora_qkv_correction"));
        // AND THE SYMBOL THAT WAS BLOCKED ON ONE MISSING ANSWER: its second
        // output is `Out<1, f16>` now rather than an optional argument.
        assert!(crossed("norm::rmsnorm_bf16_with_fp16"));
        // And a name no backend has.
        assert!(!crossed("not_a_kernel"));

        // The count. It is derived from the same registry `route` reads, so it
        // cannot drift from a row list any more -- there is no row list. What
        // it still catches is a signature change: a parameter losing its
        // source takes its symbol out of this count and into `Route::Unbound`,
        // which is a real change to what a model can fire.
        //
        // 129 -> 140, and the eleven are accounted for. FA2's eight crossed
        // when their arms went. The three capture/custom launchers were
        // `#[routine(untraced)]` -- two sink pointers, a mask pair and an
        // `#[unbound]` window between them -- and the planless pair carried an
        // `#[unbound] plan`; those five are inside the eight. The other three
        // are `attn::write_kv_explicit_bf16_devwin`, `gemm::grouped_act_x_wt_bf16`
        // and `moe::gather_moe_aligned_inputs_bf16`, which a text states and
        // which never had a row at all: `Route::Rows` covered for them, and it
        // is gone.
        //
        // 140 -> 143 with the table itself: `quant`'s two MXFP4 arms crossed
        // when `keys::WeightExpertPtrs` said what they knew, and
        // `mlp::swiglu_clamp_bf16` crossed because its refusal was stale --
        // the prose said *"the join's foreign operands"* and the pair form
        // (`deepseek_v4.rs`'s `swiglu_clamp_pair`) states gate and up as two
        // ordinary operands, which is what its column reads.
        let n = kernels_cuda::sigs().iter().filter(|s| crossed(s.symbol)).count();
        assert_eq!(
            n, 143,
            "{n} declared symbols are crossed onto the derived column. \
             Crossing one changes how a real fire is planned, so the count \
             moves only on purpose."
        );
    }

    /// How many slots are still reached by COUNTING rather than stated.
    ///
    // TWO CENSUSES STOOD HERE AND BOTH MEASURED A STATE THAT NO LONGER
    // EXISTS.
    //
    // `the_counted_slots_are_counted_and_the_count_falls_on_purpose` split
    // every `In`/`Out` slot into ones a signature STATED and ones the macro
    // COUNTED to, and pinned the second at zero. `the_facts_that_came_from_a_
    // name_...` pinned the parameter-NAME table at zero beside it.
    //
    // `kernels::routine::resolve` ended the first: a mark carries a `Claim`
    // and no number, so every slot is derived from position and there is no
    // second way for one to arrive. The name table went with `Derived::source`
    // -- the column these read is `name` and `nullable` now, because a type
    // answers the rest.


    /// Parameters whose fact came from their name rather than their type. Zero,
    /// and `fact_of` is deleted, so this is a one-way door. `Env<keys::…>` is
    /// the conversion; a bare `Env` clears this census by hiding the parameter.
    const NAMED_FACTS: usize = 0;

    /// A body asks in its own order, and that order is what gets bound.
    /// `lm_head_gemv_argmax_int8` asks an INPUT, two addresses the statement
    /// does not carry, then an OUTPUT — five values, two operands touched.
    #[test]
    fn handles_are_minted_in_the_order_the_body_asks() {
        // One input, one output, no positional weights.
        let args = [operand(0, 4096), operand(1, 1)];
        let mut o = Handles::over(&args, 1, 1);
        let mut p = Probe::silent();
        p.rows = 7;
        p.ctx.rows_total = 7;
        p.ctx.vocab = 128_256;
        p.w_named = at(8);
        p.w_named2 = at(9);
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);

        let bound = lm_head_gemv_argmax_int8(&mut o, f).expect("the four operands");
        assert_eq!(
            bound,
            vec![
                ArgValue::Region { ptr: at(0), rows: 7, width: 4096 },
                ArgValue::Ptr(at(8)),
                ArgValue::Ptr(at(9)),
                // `vocab` is asked for, so it is not in the column.
                ArgValue::Region { ptr: at(1), rows: 7, width: 1 },
            ]
        );
        assert_eq!(
            o.taken(),
            &[at(0), at(1)][..],
            "the body touched an operand it did not ask for, or skipped one \
             it did"
        );
    }

    /// The same claim where the two ADDRESSES cross, which no crossed body does
    /// yet. A fixture, so the reorder is not left for the first family to need.
    #[test]
    fn a_body_that_crosses_its_operands_gets_them_crossed() {
        fn crossing(o: &mut Handles<'_>, _: Facts) -> Result<Vec<ArgValue>, Refusal> {
            let out = o.output(0)?;
            let second = o.input(1)?;
            let w = o.weight(0)?;
            let first = o.input(0)?;
            Ok(vec![out, second, w, first])
        }

        // Two inputs, one output, one weight — the order the binder resolves.
        let args = [operand(0, 8), operand(1, 8), operand(2, 8), operand(3, 0)];
        let mut o = Handles::over(&args, 2, 1);
        let p = Probe::silent();
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let bound = crossing(&mut o, Facts::at(&cx)).expect("four operands");

        // THE CLAIM. The statement is `in0, in1, out0, weight0`; the body asked
        // `out0, in1, weight0, in0`.
        assert_eq!(
            bound,
            vec![
                ArgValue::Ptr(at(2)),
                ArgValue::Ptr(at(1)),
                ArgValue::Ptr(at(3)),
                ArgValue::Ptr(at(0)),
            ]
        );
        assert_eq!(o.taken(), &[at(2), at(1), at(3), at(0)][..]);
    }

    /// A statement short of an operand is refused, not indexed past: given one
    /// output, `split_bf16_rows` must not bind the weight run's first entry.
    #[test]
    fn a_statement_the_arm_cannot_fill_is_refused() {
        let p = Probe::silent();
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let args = [operand(0, 64), operand(1, 32)];
        let mut o = Handles::over(&args, 1, 1);
        assert_eq!(
            split_bf16_rows(&mut o, Facts::at(&cx)),
            Err(Refusal::Absent { what: "an output operand" })
        );

        // And a fact the fire does not state refuses by NAME, in `Cx`'s words.
        // `embed_bf16`'s remaining fact is its named WEIGHT: the token ids and
        // the vocabulary are asks now, so the column no longer reaches for
        // them and this probe reaches the next one that is still the
        // statement's.
        let args = [operand(0, 64), operand(1, 64)];
        let mut o = Handles::over(&args, 1, 1);
        assert_eq!(
            embed_bf16(&mut o, Facts::at(&cx)),
            Err(Refusal::Absent { what: "a named weight" })
        );
    }

    /// A count the launch does not have is clamped, not panicked on: `n_in` and
    /// `n_out` come off the op and the args off the lowering.
    #[test]
    fn counts_wider_than_the_launch_refuse_rather_than_panic() {
        let args = [operand(0, 16)];
        let mut o = Handles::over(&args, 4, 4);
        assert_eq!(o.input(0), Ok(ArgValue::Ptr(at(0))));
        assert_eq!(o.output(0), Err(Refusal::Absent { what: "an output operand" }));
        assert_eq!(o.weight(0), Err(Refusal::Absent { what: "a weight" }));
    }

    /// The generated list and the hand-written one are the same list — the diff
    /// the derived column exists for, run rather than read.
    #[test]
    fn the_derived_column_binds_what_the_hand_arm_binds() {

        // `embed_bf16`: one input, one output, two addresses not in the statement.
        let args = [operand(0, 4096), operand(1, 4096)];
        let mut p = Probe::silent();
        p.rows = 7;
        p.ctx.rows_total = 7;
        p.ctx.token_ids = at(5);
        p.ctx.vocab = 128_256;
        p.w_named = at(8);
        p.w_named2 = at(9);
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);

        let hand = embed_bf16(&mut Handles::over(&args, 0, 1), f).expect("the hand arm binds");
        let made = operands(
            &mut Handles::over(&args, 0, 1),
            f,
            <kernels_cuda::layout::embed_bf16 as ::kernels::Derivation>::DERIVED,
            srcs("layout::embed_bf16"),
            decl("layout::embed_bf16"),
        );
        assert_eq!(
            made.as_deref().map(addressed),
            Ok(addressed(&hand)),
            "`embed_bf16`'s derived column and its hand arm bind different lists"
        );

        // `gather_bf16_rows`, whose indices needed `#[source(...)]`.
        let args = [operand(0, 4096), operand(1, 4096)];
        let mut p = Probe::silent();
        p.rows = 3;
        p.ctx.rows_total = 7;
        p.ctx.sampling_indices = at(6).cast::<i32>();
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);
        let hand =
            gather_bf16_rows(&mut Handles::over(&args, 1, 1), f).expect("the hand arm binds");
        let made = operands(
            &mut Handles::over(&args, 1, 1),
            f,
            <kernels_cuda::layout::gather_bf16_rows as ::kernels::Derivation>::DERIVED,
            srcs("layout::gather_bf16_rows"),
            decl("layout::gather_bf16_rows"),
        );
        assert_eq!(made.as_deref().map(addressed), Ok(addressed(&hand)));
    }

    /// The probe every "what does this RESOLVE" test shares. A cursor borrows
    /// its fire, so the fire has to outlive the call.
    fn stated_probe() -> Probe {
        Probe::stating_everything()
    }

    /// EVERY table arm binds what its own signature says. The fixture is
    /// over-supplied: the question is whether the two AGREE.
    #[test]
    fn every_table_arm_agrees_with_its_own_column() {

        let args: Vec<BoundArg> = (0..24).map(|n| operand(n, 4096)).collect();
        let p = stated_probe();
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);

        // Each row is `(what, the table arm, its column, n_in, n_out)`.
        type Arm = fn(&mut Handles<'_>, Facts<'_>) -> Result<Vec<ArgValue>, Refusal>;
        let cases: &[(&str, Arm, &[Derived], usize, usize)] = &[
            ("layout::split_bf16_rows", split_bf16_rows, <kernels_cuda::layout::split_bf16_rows as ::kernels::Derivation>::DERIVED, 1, 2),
            (
                "layout::split_qwen_gdn_ba_bf16",
                split_qwen_gdn_ba_bf16,
                <kernels_cuda::layout::split_qwen_gdn_ba as ::kernels::Derivation>::DERIVED,
                1,
                2,
            ),
            (
                "layout::transpose_bf16_nld_to_lnd",
                transpose_bf16_nld_to_lnd,
                <kernels_cuda::layout::transpose_bf16_nld_to_lnd as ::kernels::Derivation>::DERIVED,
                1,
                1,
            ),
            (
                "sample::lm_head_gemv_argmax_int8",
                lm_head_gemv_argmax_int8,
                <kernels_cuda::sample::lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED,
                1,
                1,
            ),
        ];

        for &(what, arm, column, n_in, n_out) in cases {
            let hand = arm(&mut Handles::over(&args, n_in, n_out), f);
            let made = operands(
                &mut Handles::over(&args, n_in, n_out),
                f,
                column,
                srcs(what),
                decl(what),
            )
            .map(|v| addressed(&v));
            assert_eq!(
                made.as_ref().map(Vec::as_slice).map_err(|e| format!("{e:?}")),
                hand.as_ref().map(|h| addressed(h)).as_ref()
                    .map(Vec::as_slice).map_err(|e| format!("{e:?}")),
                "`{what}`'s derived column and its table arm disagree"
            );
        }
    }

    /// Whether `call()` would accept this value for a parameter of this type.
    /// Six types are checked and the rest pass, `jit/abi.rs`'s asymmetry:
    /// `scalar_abi!` admits one `ArgValue`, `ptr_abi!` admits every pointer.
    fn abi_admits(t: Option<kernels::Ty>, v: &ArgValue) -> bool {
        use kernels::Ty;
        match t {
            Some(Ty::I32) => matches!(v, ArgValue::I32(_)),
            Some(Ty::U32) => matches!(v, ArgValue::U32(_)),
            Some(Ty::F32) => matches!(v, ArgValue::F32(_)),
            Some(Ty::Bool) => matches!(v, ArgValue::Bool(_)),
            Some(Ty::I64) => matches!(v, ArgValue::I64(_)),
            Some(Ty::Usize) => matches!(v, ArgValue::Usize(_)),
            _ => true,
        }
    }

    #[test]
    fn the_derived_column_reaches_most_of_the_surface() {
        // A ROW WITH NO COLUMN IS TWO DIFFERENT FACTS: `untraced!` has no
        // signature to derive from; a `routine!` row without one is work to do.
        let mut untraced = Vec::new();
        let mut uncolumned = Vec::new();
        let mut reachable = Vec::new();
        let mut blocked: Vec<(&str, &Derived, Option<Source>)> = Vec::new();
        // Resolved by `operand`, refused by the ABI. See the loop below.
        let mut mistyped: Vec<(&str, &Derived, Option<Source>)> = Vec::new();

        // Thirty-six operands, every `Facts` field filled. THE PROBE MUST BE
        // WIDER THAN THE WIDEST SIGNATURE: 12/12 of 24 left the weight run empty.
        let args: Vec<BoundArg> = (0..36).map(|n| operand(n, 4096)).collect();
        let probe = stated_probe();
        let fire = probe.fire();
        let cx = Cx::new(&fire);
        let stated = Facts::at(&cx);

        for sig in kernels_cuda::sigs() {
            let Some(row) = kernels_cuda::routine(sig.symbol) else { continue };
            if row.derived.is_empty() {
                if row.args.is_empty() { &mut untraced } else { &mut uncolumned }.push(sig.symbol);
                continue;
            }
            // PROBE WITH EVERYTHING STATED, BECAUSE A DEFAULT CANNOT TELL THE
            // TWO REFUSALS APART. `super::`, this module having its own `operand`.
            let mut why = None;
            let mut wrong_kind = None;
            for (i, d) in row.derived.iter().enumerate() {
                let src = row.sources.get(i).copied().flatten();
                match super::operand(&mut Handles::over(&args, 12, 12), stated, d, src) {
                    Err(_) => {
                        why = Some((d, src));
                        break;
                    }
                    // RESOLVING IS NOT BINDING: `call()` refuses a wrong kind.
                    Ok(v) => {
                        let t = sig.args.get(i).copied();
                        if !abi_admits(t, &super::as_declared(t, v)) {
                            wrong_kind = Some((d, src));
                            break;
                        }
                    }
                }
            }
            match (why, wrong_kind) {
                (Some(d), _) => blocked.push((sig.symbol, d.0, d.1)),
                (None, Some(d)) => mistyped.push((sig.symbol, d.0, d.1)),
                (None, None) => reachable.push(sig.symbol),
            }
        }

        for symbol in &uncolumned {
            println!("  NO COLUMN {symbol}");
        }
        println!(
            "derived column: {} reachable, {} blocked, {} mistyped, \
             {} uncolumned, {} driver-bound",
            reachable.len(),
            blocked.len(),
            mistyped.len(),
            uncolumned.len(),
            untraced.len()
        );
        for (symbol, d, src) in &mistyped {
            println!(
                "  mistyped {symbol}  -- `{}` resolves {:?}, wrong ABI kind",
                d.name, src
            );
        }
        for (symbol, d, src) in &blocked {
            println!("  blocked  {symbol}  -- `{}` is {src:?}", d.name);
        }

        // EVERY `routine!` ROW HAS A COLUMN. `attn::qkv_decode_fused_dispatch`
        // is excluded BY NAME: no `Bound` row and a signature of bare pointers,
        // so a column would claim a binding nobody stated.
        const NO_COLUMN_ON_PURPOSE: &[&str] = &["attn::qkv_decode_fused_dispatch"];
        let missing: Vec<_> =
            uncolumned.iter().filter(|s| !NO_COLUMN_ON_PURPOSE.contains(s)).collect();
        assert!(
            missing.is_empty(),
            "these `routine!` rows carry an empty column: {missing:?}. Every \
             row built from a `fn` signature reads one off `Derivation`; only \
             `untraced!` rows, which have no signature to derive from, \
             and a row spelled `uncolumned` do not."
        );

        // THE CLAIM: A BLOCKED PARAMETER IS `None` OR AT THE BOUNDARY. `None`
        // says `#[routine]` had no name for a scalar; anything else is a source
        // this binder should answer. The allowance is what `Facts` cannot carry.
        for (symbol, d, src) in &blocked {
            assert!(
                matches!(
                    src,
                    // Only what `Facts` cannot hold — the rest are answered.
                    None | Some(
                        Source::Slot(Kind::Aux | Kind::Param | Kind::ParamF32, _)
                            | Source::Named(<keys::KvKeys as keys::Fact>::KEY)
                            | Source::Named(<keys::KvValues as keys::Fact>::KEY)
                            | Source::Named(<keys::KvPageIndices as keys::Fact>::KEY)
                            | Source::Named(<keys::KvPageIndptr as keys::Fact>::KEY)
                            | Source::Named(<keys::KvHeadStride as keys::Fact>::KEY)
                            | Source::Named(<keys::KvSeqStride as keys::Fact>::KEY)
                    )
                ),
                "`{symbol}`'s `{}` derived {:?}, which is neither `None` \
                 nor one of the fire-scoped facts `Facts`' own doc says it \
                 cannot carry -- so this binder should answer it and does \
                 not. Fix `operand`, not the launcher",
                d.name,
                src
            );
        }

        // AND THE SECOND BUCKET IS EMPTY, BECAUSE `as_declared` EMPTIES IT: the
        // repair is a new `as_declared` case or a wrong declaration.
        assert!(
            mistyped.is_empty(),
            "{mistyped:?} resolve their whole column and would still be \
             refused by `call()` on the kind of a scalar. Widen \
             `as_declared`, or fix the declaration"
        );
    }

    #[test]
    fn a_row_with_no_column_refuses_before_the_launch() {
        let p = Probe::silent();
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let args = [operand(0, 16)];
        assert_eq!(
            operands(&mut Handles::over(&args, 1, 0), Facts::at(&cx), &[], &[], &[]),
            Err(Refusal::Unstated { what: "an operand column: the row states no `derived`" })
        );
    }
}
