//! The shape wgpu and metal already have, on CUDA: [`Facts`], [`Handles`],
//! [`operands`], [`dispatch`]. One operand order for all three backends,
//! stated at [`BoundLaunch::args`]: inputs, outputs, weights.
//!
//! The arms below are `#[cfg(test)]` fixtures, a second opinion the derived
//! column is diffed against.

use core::ffi::c_void;

use kernels::Derived;
use kernels::Refusal;
use kernels::{Kind, Source};
use kernels_cuda::ArgValue;
use kernels_cuda::attn::Rows;

use super::cx::Cx;
use super::{BoundArg, BoundLaunch, LaunchSpec};

// ── Facts ─────────────────────────────────────────────────────────────────

/// Where a launch's facts are read from, not what was found.
///
/// A cursor: `cx`, plus the two answers that cannot refuse. This carried
/// ~40 forwarded fact queries while bind bodies could ask; the swept
/// signatures carry those numbers as marks, so what remains is what a COLUMN
/// still reads — the rows, the layer, the named weights, the params run.
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
        Self {
            cx,
            rows: cx.rows(),
            layer: cx.layer(),
        }
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
        Self {
            ins,
            outs,
            weights,
            taken: Vec::new(),
        }
    }

    /// The `n`th INPUT, as the address the kernel receives.
    pub fn input(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.ins.get(n).ok_or(Refusal::Absent {
            what: "an input operand",
        })?;
        Ok(self.take(at))
    }

    /// The `n`th OUTPUT.
    pub fn output(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.outs.get(n).ok_or(Refusal::Absent {
            what: "an output operand",
        })?;
        Ok(self.take(at))
    }

    /// The `n`th positional WEIGHT.
    pub fn weight(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self
            .weights
            .get(n)
            .ok_or(Refusal::Absent { what: "a weight" })?;
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

    /// The `n`th input's OWN row count, zero where the operand has none
    /// apart from the launch's rectangle. Nonzero only for a NAMED stream
    /// the driver stages whole — `bind` reads it off `Lowered::arg_rows` —
    /// which is what lets a CSR operand answer its boundary count.
    pub fn in_rows(&self, n: usize) -> i32 {
        own_rows(self.ins.get(n))
    }

    /// The `n`th output's OWN row count. See [`Self::in_rows`].
    pub fn out_rows(&self, n: usize) -> i32 {
        own_rows(self.outs.get(n))
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
    i32::try_from(w)
        .ok()
        .filter(|w| *w > 0)
        .ok_or(Refusal::Absent { what })
}

/// One operand's OWN row count, zero where it has none — no refusal, because
/// zero is this crate's word for "no extent" and the reader's fallback is the
/// launch's rectangle.
fn own_rows(at: Option<&BoundArg>) -> i32 {
    at.map_or(0, |a| i32::try_from(a.rows).unwrap_or(0))
}

// ── The arms ──────────────────────────────────────────────────────────────

/// One kernel's operand resolution, as the table path states it. `#[cfg(test)]`
/// with the six bodies below: a second opinion written by hand from the
/// kernels' docs, which `every_table_arm_agrees_with_its_own_column` consults.
#[cfg(test)]
pub type TableArm = fn(&mut Handles<'_>, Facts) -> Result<Vec<ArgValue>, Refusal>;

/// `sample::lm_head_gemv_argmax_int8`. Head and scale are POSITIONAL weights.
///
/// They were NAMED, read through `Facts::weight_named`, because a semantic op
/// carried its weights as strings on `LaunchSpec` and only a launch put them
/// in the operand list. There is one shape of statement now, `Source::Named`
/// is retired, and a `Const<Tensor<..>>` mark claims the weight run by
/// position -- so the two reads become `o.weight(0)` and `o.weight(1)`.
///
/// `vocab` came back for the same reason it left `embed_bf16`'s list and
/// returned: a vocabulary is a load-time constant, so it is a `Const<i32>`
/// mark on the params run rather than a fact the driver lends.
#[cfg(test)]
pub fn lm_head_gemv_argmax_int8(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let hidden_states = o.input(0)?;
    let lm_head_weight = o.weight(0)?;
    let scale_inv = o.weight(1)?;
    let token_ids = o.output(0)?;
    let hidden = o.in_width(0)?;
    let vocab = f.cx.param(0)?;
    Ok(vec![
        as_region(hidden_states, f.rows.count, hidden),
        lm_head_weight,
        scale_inv,
        as_region(token_ids, f.rows.count, o.out_width(0).unwrap_or(0)),
        ArgValue::I32(i32::try_from(vocab).unwrap_or(i32::MAX)),
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
pub fn split_qwen_gdn_ba_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
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
/// The token ids and the vocabulary came BACK to this list, having left it
/// once. They left when they were facts the driver's `DispatchCtx` lent and a
/// column carried only what the statement placed; the no-ask migration made
/// both of them marks -- `token_ids: In<Tensor<i32>>` is the statement's first
/// input, `vocab: Const<i32>` its first param -- so all four are the
/// statement's again and the arm binds four.
///
/// This arm reading two while the column read four is exactly the drift the
/// diff below exists to catch, and it did not catch it, because the test that
/// runs the diff had not compiled since the migration.
#[cfg(test)]
pub fn embed_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let weight = ArgValue::Ptr(f.weight_named(0)?);
    let y = o.output(0)?;
    let hidden = o.out_width(0)?;
    let token_ids = o.input(0)?;
    let vocab = f.cx.param(0)?;
    Ok(vec![
        weight,
        as_region(y, f.rows.count, hidden),
        token_ids,
        ArgValue::I32(i32::try_from(vocab).unwrap_or(i32::MAX)),
    ])
}

/// `layout::gather_bf16_rows`. `sampling_indices` derives as `In(1)`.
///
/// It used to be ASKED for -- `keys::SamplingIndices`, a fact the driver lent
/// -- and so was not the statement's to bind, which is why this list was two
/// long. The no-ask migration made it the statement's SECOND INPUT, and
/// `TraceBuilder::lm_head` mints the runtime tensor that fills it. So the list
/// is three, and a fixture for this arm places two inputs.
#[cfg(test)]
pub fn gather_bf16_rows(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let src = o.input(0)?;
    let dst = o.output(0)?;
    let width = o.out_width(0)?;
    let sampling_indices = o.input(1)?;
    Ok(vec![
        as_region(src, f.rows.count, o.in_width(0).unwrap_or(0)),
        as_region(dst, f.rows.count, width),
        sampling_indices,
    ])
}

/// `layout::transpose_bf16_nld_to_lnd`. Places operands, computes nothing.
#[cfg(test)]
pub fn transpose_bf16_nld_to_lnd(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let src = o.input(0)?;
    let dst = o.output(0)?;
    let width = o.in_width(0)?;
    // `dim` WAS `ple_dim`, ASKED for, and so not the statement's to bind. It
    // is a `Const<i32>` mark now -- a per-deployment constant belongs in the
    // statement -- so it rides the params run and the list is three.
    let dim = f.cx.param(0)?;
    Ok(vec![
        as_region(src, f.rows.count, width),
        as_region(dst, f.rows.count, o.out_width(0).unwrap_or(0)),
        ArgValue::I32(i32::try_from(dim).unwrap_or(i32::MAX)),
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
        return Err(Refusal::Unstated {
            what: "an operand column: the row states no `derived`",
        });
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
                Err(Refusal::Absent { .. }) if d.nullable => ArgValue::Ptr(core::ptr::null_mut()),
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
        // THE PARAM CHANNEL IS UNTYPED BITS, and this is the only place an
        // `ArgValue::U32` is ever minted (`operand`'s `Kind::Param` arm; every
        // other channel arrives typed). A statement writes its scalar into
        // that run as `n as u32` and the SIGNATURE says how to read it back,
        // so a routine declaring `Const<i32>` is declaring that the bits are
        // an i32 -- reinterpreting them is the whole contract, not a wrap.
        //
        // The range guard that stood here (`n <= i32::MAX`) read the run as a
        // genuine unsigned count and refused anything above the signed
        // ceiling. Every NEGATIVE i32 a statement carries is above it, and the
        // commonest one in this tree is `window_left = -1` -- "this layer
        // slides over nothing". So the decode dispatch refused, by argument
        // NUMBER rather than name, on every deployment without a sliding
        // window: `argument 3 is I32 and arrived otherwise`.
        (Ty::I32, ArgValue::U32(n)) => ArgValue::I32(n.cast_signed()),
        // A RAISE HAS NO SHAPE, so the region `operand` mints for an `In` slot
        // comes back off again. `In<Struct<Fa2Decode>>` is a mark like any
        // other and resolves to `Source::Slot(Kind::In, n)`, and the arm for
        // that slot wraps every pointer in `ArgValue::Region` because a
        // statement's `In` is normally a rectangle. This one is not: a raise
        // is ONE OBJECT with one lifetime, `raises.rs` binds it as a typed
        // pointer on purpose, and `raise_abi!`'s `unpack` takes `Ptr` alone.
        //
        // Without this the decode dispatch refused every fire it reached with
        // *argument 1 is Raised and arrived otherwise* -- the plan arriving as
        // a region of zero width, which is the right address wearing the wrong
        // shape. The signature is what knows the difference, which is why the
        // answer is here rather than in the slot arm: `Ty::Raised` is
        // `kernels-cuda`'s own declaration for the parameter.
        (Ty::Raised, ArgValue::Region { ptr, .. }) => ArgValue::Ptr(ptr),
        // A FLAG IS ONE BYTE ON THE WIRE AND ONE WORD IN THE RUN. The param
        // channel is untyped bits and the generated wrapper writes a `bool` in
        // as `u32::from(write_state)`, so the signature is again the only
        // thing that knows what came back -- exactly as for `Ty::I32` above,
        // and for the same reason: `Const<bool>` declares that the bits are a
        // flag. `jit::abi`'s `scalar_abi!(bool, ..)` unpacks `ArgValue::Bool`
        // ALONE, so without this every routine in the tree that states one
        // refused, by argument NUMBER rather than name.
        //
        // That is sixteen routines and not a corner: every ssm recurrence and
        // conv (`write_state`), every rope (`interleaved`), and the moe top-k
        // pair (`renormalize`/`normalize`). `real_hybrid` is what said so --
        // `ssm::causal_conv1d_prefill_batched: argument 7 is Bool and arrived
        // otherwise` -- and it could only say it once the state slabs reached
        // the raise channel, which is the fix directly before this one. The
        // rope and moe arms are hand-written in this file and pass their own
        // `ArgValue::Bool`, which is why the shipped gates never saw it.
        (Ty::Bool, ArgValue::U32(n)) => ArgValue::Bool(n != 0),
        (Ty::Bool, ArgValue::I32(n)) => ArgValue::Bool(n != 0),
        _ => v,
    }
}

/// One operand, from where its [`Source`] says it comes.
///
/// Handles are the statement's own, minted in the order asked. The arm set is
/// what a DERIVED COLUMN can spell now, and nothing more: the marks resolve
/// to slots (`In`/`Out`/`InOut`/`Const`), the weight run to the two named
/// chains, and the params run to its two readings. Every keyed fact this
/// match once answered — the KV geometry, the workspaces, the state slabs,
/// the streams — reaches a routine as a runtime OPERAND instead, resolved by
/// the fire's resolver over `bind::views` before this binder ever runs. The
/// catch-all refusal is the design, not a to-do.
fn operand(
    o: &mut Handles<'_>,
    f: Facts<'_>,
    d: &Derived,
    source: Option<Source>,
) -> Result<ArgValue, Refusal> {
    // A statement places a region — address, row count, pitch — and the
    // SIGNATURE decides how much to keep. A missing width is zero, not a
    // refusal. The rows are the operand's OWN where it has them (a NAMED
    // stream staged whole — a CSR carries its boundary count), the launch's
    // rectangle everywhere else.
    let region = |v: ArgValue, own: i32, width: i32| match v {
        ArgValue::Ptr(p) => ArgValue::Region {
            ptr: p,
            rows: if own > 0 { own } else { f.rows.count },
            width,
        },
        other => other,
    };
    match source {
        // ONE ADDRESS IN TWO SLOTS, resolved as the INPUT: that is the address
        // the statement placed, and the allocator has already given the result
        // the same offset off the same `Source::Alias`.
        Some(Source::Alias(n, _)) => {
            let width = o.in_width(n as usize).unwrap_or(0);
            let own = o.in_rows(n as usize);
            o.input(n as usize).map(|v| region(v, own, width))
        }
        // The statement's own three.
        Some(Source::Slot(Kind::In, n)) => {
            let width = o.in_width(n as usize).unwrap_or(0);
            let own = o.in_rows(n as usize);
            o.input(n as usize).map(|v| region(v, own, width))
        }
        Some(Source::Slot(Kind::Out, n)) => {
            let width = o.out_width(n as usize).unwrap_or(0);
            let own = o.out_rows(n as usize);
            o.output(n as usize).map(|v| region(v, own, width))
        }
        // NOT A REGION: a weight's shape is the MODEL's, not the statement's.
        //
        // TWO PLACES A WEIGHT CAN COME FROM, and a column has one spelling for
        // both. A trace `OpKind::Launch` lists its weights BY NAME and the
        // walk pushes each as an `Arg::Weight` onto the launch's own run, so
        // `Handles::weight` finds it past the inputs and outputs. A SEMANTIC
        // op -- `OpKind::LmHead`, `OpKind::Embed`, `OpKind::Matmul` -- has no
        // such list: the walk emits its kernel from the op kind, and the
        // weight reaches the dispatch through `LaunchSpec::weight`, resolved
        // once into the fire's `w_named` before the `Cx` exists.
        //
        // The chain is the whole of the difference. Without it every semantic
        // op's kernel refused with "the fire does not carry a weight" -- and
        // the loudest was the epilogue, where `gemm::act_x_w` is the lm-head
        // projection and the run ended one launch short of its logits on
        // every fire.
        Some(Source::Slot(Kind::Weight, n)) => o
            .weight(n as usize)
            .or_else(|_| f.weight_named(n as usize).map(ArgValue::Ptr)),
        // Two arms: a param is a byte run, so `f32` is a different channel.
        Some(Source::Slot(Kind::Param, n)) => f.cx.param(n as usize).map(ArgValue::U32),
        Some(Source::Slot(Kind::ParamF32, n)) => f.cx.param_f32(n as usize).map(ArgValue::F32),

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
    // `None`, NOT a refusal, for a row with no column AND no ask list: `None`
    // means "take the path you take today", `Some(Err(_))` means "this fire
    // cannot run". A row that asks for its whole input is NOT one of these --
    // see `route::route`, where the same pair of conditions decides whether
    // the driver's hand-written match is even the right place to look.
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
    if row.no_join && !spec.aux.is_empty() {
        return Some(Err(Refusal::Unstated {
            what: "a dispatch without an aux value",
        }));
    }
    // AN EMPTY COLUMN IS AN EMPTY LIST HERE, and only here. `operands` refuses
    // `&[]` on purpose -- for a caller that meant to pass a column and passed
    // nothing, refusing is the whole value of the guard -- but a row that
    // reached this line with no column and a non-empty ask list has already
    // been read for exactly that shape twice, in `route::route` and above, and
    // means it. The body takes no arguments and asks for everything.
    let args = if row.derived.is_empty() {
        Vec::new()
    } else {
        match operands(
            &mut handles,
            Facts::at(cx),
            row.derived,
            row.sources,
            row.args,
        ) {
            Ok(args) => args,
            Err(why) => return Some(Err(why)),
        }
    };
    // THE WHOLE BOUND LIST, on request. A wrong ADDRESS is invisible: the
    // launch succeeds and the numbers come out wrong, which is the one failure
    // a refusal-driven bring-up cannot see. `PIE_TRACE_BINDS=1` prints it.
    if tracing("PIE_TRACE_BINDS") {
        eprintln!(
            "[bind] {symbol} n_in={} n_out={} -> {args:?}",
            spec.n_in, spec.n_out
        );
    }
    // AND WHAT THE LAUNCH LEFT IN ITS RESULTS. A binding can be right in every
    // address and still run on the wrong NUMBERS; `PIE_TRACE_VALUES=1` reads
    // the first few bf16 of every region back after the launch, which is the
    // only way to see where a forward pass stops being arithmetic.
    //
    // Deliberately expensive: it synchronises the stream. A diagnostic that
    // changed the schedule it is measuring would be worse than none.
    let peek = tracing("PIE_TRACE_VALUES");
    // The `Answering` handoff stood here: the same `Handles` and `Facts` the
    // column was bound from, lent to the body for `ctx.ask`. Zero routines on
    // this plane ask now -- the signature carries all data -- so the column
    // above IS the whole binding and the body gets no environment.
    drop(handles);
    // SAFETY: this function's own contract, forwarded unchanged. Every pointer
    // the column bound came from `bound.args` or a `Facts` field the dispatch
    // site resolved; anything from nowhere at all is a refusal, not a null.
    let fired = unsafe { kernels_cuda::call_with_cublas(symbol, &args, stream, cx.cublas()) };
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
    unsafe { dispatch(fire.bound, fire.spec, cx, stream) }.unwrap_or(Err(Refusal::Unstated {
        what: "an operand column, on a row that named `derived_arm`",
    }))
}

#[cfg(test)]
mod tests {
    use super::super::facts::Fire;
    use super::super::{AttnCtx, DispatchCtx};
    use super::*;

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
            }
        }

        /// A fire that states NOTHING: every scalar the value its accessor reads
        /// as an absence, and `attn` is `None`.
        fn silent() -> Self {
            Self {
                ctx: dispatch_ctx(),
                attn: None,
                bound: BoundLaunch {
                    kernel: "",
                    rows: 0..0,
                    layers: 0..1,
                    args: Vec::new(),
                },
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

            // A per-op head width USED TO BE STATED HERE, as
            // `LaunchSpec::per_head_dim`, so the fixture could disagree with
            // the attention's. RETIRED with the field: an op's scalars live in
            // `LaunchSpec::params` now, and `kv_layer_view` still disagrees
            // with `attn_ctx`, which is the disagreement the tests below read.
            let spec = LaunchSpec::default();

            Self {
                ctx,
                attn: Some(attn),
                bound: BoundLaunch {
                    kernel: "",
                    rows: 0..7,
                    layers: 0..1,
                    args: Vec::new(),
                },
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
        BoundArg {
            ptr: at(n),
            width,
            rows: 0,
        }
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
        assert!(crossed("norm::add_bias"));
        assert!(crossed("mlp::swiglu"));
        assert!(crossed("moe::topk_softmax"));
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
        // AND THE THREE WITH THE MOST TO LOSE FROM THE `#[unbound]` SWEEP,
        // one per shape of parameter it retired: a YaRN scheme's two scalar
        // factors, a compressed-cache program's whole geometry, and an
        // AltUp correction's coefficient bank.
        assert!(crossed("rope::rope_yarn_bf16"));
        assert!(crossed("attn::attention_compressed_paged_bf16"));
        assert!(crossed("norm::altup_correct"));
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
        //
        // 143 -> 165 WITH THE END OF `#[unbound]`. The attribute is a
        // parameter saying *"nothing states me"* -- the sentence without a
        // fake key -- and `route` cannot bind a symbol that holds one, so
        // every routine carrying one sat in `Route::Unbound`. There is not a
        // single `#[unbound]` left in `kernels-cuda/src`: the marks gave each
        // of those parameters a key, and twenty-two symbols crossed in one
        // move. By family, and none was lost:
        //
        // `attn` nine -- `attention_compressed_paged_bf16`, the two
        // `dsa_index_*` ropes, the four `dsv4_*` compressed-cache
        // programs and the two `mtp_*` hidden shifts.
        // `norm` three -- `altup_correct_bf16` and the two `hc_*_postprocess`.
        // `rope` three -- `qk_rmsnorm_mrope_bf16`, `rope_write_kv_bf16` and
        // `rope_yarn_bf16`, whose `low_freq_factor`/`high_freq_factor` were
        // the last thing in this driver that had to name llama-3 out loud.
        // `mlp` two (`situ_bf16` and its chunked form), `quant` two (the
        // WNA16 decode pair), `ssm` two (the two nemotron MoE pointer
        // builders) and `moe` one (`transpose_expert_scales_u8`).
        //
        // WHAT CROSSING DOES NOT SAY is that a fire of one of these
        // succeeds. `route` binds on the parameter having a SOURCE; whether
        // this driver answers that source is a different question and
        // `kernels/tests/every_plane_is_answered` is where it is asked. It
        // currently names fifty §M keys `kernels-cuda` asks for and
        // `driver-cuda` answers nowhere, and several of them are on the
        // symbols listed above. The two counts moving in opposite directions
        // is the migration being half done, not either census being wrong.
        //
        // 165 -> 167 WHEN AN EMPTY COLUMN STOPPED MEANING AN EMPTY INPUT.
        // `route` sent every column-less row to `Route::Driver` on the
        // reading that there was nothing to bind, and for
        // `attn::write_kv_to_pages` -- a declaration standing for a CHOICE --
        // that is still true. It is not true of a routine written
        // `fn f(ctx: &Ctx<'_>)` that reads every address and every number
        // through `ctx.ask`: its input is complete and its column is empty
        // because asking is how it takes it. Two rows are that shape,
        // `attn::dequant_kv_cache_layer_to_bf16_active` and
        // `gemm::grouped_act_x_wt_bf16`, and the first refused the ninth
        // launch of every fire with "a driver op with no arm" -- a hand-
        // written match being asked for an arm whose entire purpose was to
        // not have to exist.
        let n = kernels_cuda::sigs()
            .iter()
            .filter(|s| crossed(s.symbol))
            .count();
        assert_eq!(
            n, 168,
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

    // AND THE SECOND CENSUS'S CONSTANT OUTLIVED ITS TEST. `NAMED_FACTS = 0`
    // stood right here, documented as a one-way door, with nothing left that
    // read it: a ratchet is a number plus an assertion, and the assertion had
    // already been removed by the paragraph above. A number alone ratchets
    // nothing and reads as though something still checks it.

    /// A body asks in its own order, and that order is what gets bound.
    /// `lm_head_gemv_argmax_int8` asks an INPUT, two WEIGHTS, then an OUTPUT,
    /// and reads a param — five values, four operands touched.
    ///
    /// It used to be *"two addresses the statement does not carry"*: the head
    /// and the scale were NAMED weights off `Fire::w_named`, and the operand
    /// run held only the two the statement placed. Both are positional now, so
    /// the fixture places four and `taken()` records all four.
    #[test]
    fn handles_are_minted_in_the_order_the_body_asks() {
        // One input, one output, TWO positional weights.
        let args = [
            operand(0, 4096),
            operand(1, 1),
            operand(8, 0),
            operand(9, 0),
        ];
        let mut o = Handles::over(&args, 1, 1);
        let mut p = Probe::silent();
        p.rows = 7;
        p.ctx.rows_total = 7;
        p.ctx.vocab = 128_256;
        p.spec.params = vec![128_256];
        p.w_named = at(8);
        p.w_named2 = at(9);
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);

        let bound = lm_head_gemv_argmax_int8(&mut o, f).expect("the five values");
        assert_eq!(
            bound,
            vec![
                ArgValue::Region {
                    ptr: at(0),
                    rows: 7,
                    width: 4096
                },
                ArgValue::Ptr(at(8)),
                ArgValue::Ptr(at(9)),
                ArgValue::Region {
                    ptr: at(1),
                    rows: 7,
                    width: 1
                },
                ArgValue::I32(128_256),
            ]
        );
        assert_eq!(
            o.taken(),
            &[at(0), at(8), at(9), at(1)][..],
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
            Err(Refusal::Absent {
                what: "an output operand"
            })
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
            Err(Refusal::Absent {
                what: "a named weight"
            })
        );
    }

    /// A count the launch does not have is clamped, not panicked on: `n_in` and
    /// `n_out` come off the op and the args off the lowering.
    #[test]
    fn counts_wider_than_the_launch_refuse_rather_than_panic() {
        let args = [operand(0, 16)];
        let mut o = Handles::over(&args, 4, 4);
        assert_eq!(o.input(0), Ok(ArgValue::Ptr(at(0))));
        assert_eq!(
            o.output(0),
            Err(Refusal::Absent {
                what: "an output operand"
            })
        );
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
        // `vocab` RIDES THE PARAMS RUN NOW, not `DispatchCtx`: it is a
        // load-time constant, so the statement carries it and the mark reads
        // slot 0. `p.ctx.vocab` is left stated on purpose, at the same value,
        // so this stays a fixture for a real fire rather than a proof that the
        // old place happens to be empty.
        p.spec.params = vec![128_256];
        p.w_named = at(8);
        p.w_named2 = at(9);
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);

        let hand = embed_bf16(&mut Handles::over(&args, 1, 1), f).expect("the hand arm binds");
        let made = operands(
            &mut Handles::over(&args, 1, 1),
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

        // `gather_bf16_rows`, whose indices are the statement's SECOND INPUT
        // now rather than a fact the driver lends, so the fixture places two.
        let args = [operand(0, 4096), operand(6, 0), operand(1, 4096)];
        let mut p = Probe::silent();
        p.rows = 3;
        p.ctx.rows_total = 7;
        p.ctx.sampling_indices = at(6).cast::<i32>();
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);
        let hand =
            gather_bf16_rows(&mut Handles::over(&args, 2, 1), f).expect("the hand arm binds");
        let made = operands(
            &mut Handles::over(&args, 2, 1),
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
        let mut p = stated_probe();
        // THE PARAMS RUN, which `stating_everything` does not fill because it
        // states the driver's facts and a param is the STATEMENT's. One entry
        // is enough here: `transpose_bf16_nld_to_lnd`'s `dim` is the only mark
        // in this batch that reads it, and it reads slot 0.
        p.spec.params = vec![256];
        let fire = p.fire();
        let cx = Cx::new(&fire);
        let f = Facts::at(&cx);

        // Each row is `(what, the table arm, its column, n_in, n_out)`.
        type Arm = fn(&mut Handles<'_>, Facts<'_>) -> Result<Vec<ArgValue>, Refusal>;
        let cases: &[(&str, Arm, &[Derived], usize, usize)] = &[
            (
                "layout::split_bf16_rows",
                split_bf16_rows,
                <kernels_cuda::layout::split_bf16_rows as ::kernels::Derivation>::DERIVED,
                1,
                2,
            ),
            (
                "layout::split_qwen_gdn_ba",
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
                made.as_ref()
                    .map(Vec::as_slice)
                    .map_err(|e| format!("{e:?}")),
                hand.as_ref()
                    .map(|h| addressed(h))
                    .as_ref()
                    .map(Vec::as_slice)
                    .map_err(|e| format!("{e:?}")),
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
            let Some(row) = kernels_cuda::routine(sig.symbol) else {
                continue;
            };
            if row.derived.is_empty() {
                if row.args.is_empty() {
                    &mut untraced
                } else {
                    &mut uncolumned
                }
                .push(sig.symbol);
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
        let missing: Vec<_> = uncolumned
            .iter()
            .filter(|s| !NO_COLUMN_ON_PURPOSE.contains(s))
            .collect();
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
                    //
                    // SIX `Source::Named(<keys::Kv… as keys::Fact>::KEY)`
                    // ALLOWANCES STOOD HERE: the KV keys, values, page indices,
                    // page indptr, head stride and seq stride. RETIRED with the
                    // ask gate that named them -- there is no `Source::Named`
                    // and no `keys` module any more, so a fire-scoped fact
                    // cannot be spelled as one, and anything still blocked has
                    // to be a slot.
                    None | Some(Source::Slot(Kind::Aux | Kind::Param | Kind::ParamF32, _))
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
            operands(
                &mut Handles::over(&args, 1, 0),
                Facts::at(&cx),
                &[],
                &[],
                &[]
            ),
            Err(Refusal::Unstated {
                what: "an operand column: the row states no `derived`"
            })
        );
    }
}
