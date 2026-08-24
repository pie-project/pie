use crate::routine::Ctx;

/// The `Ssm` family: NONE of seven points lands, and the cause is one shape
/// disagreement rather than seven missing shaders.
///
/// # This plane's GDN is fused where the declaration is cut
///
/// `kernels::points::Ssm` cuts qwen3.5's gated-delta mixer into three
/// statements, each declaring what it reads and nothing else:
///
/// | point | operands |
/// | --- | --- |
/// | `ssm.causal_conv1d` | the packed `qkv` row, ONE conv weight, the state |
/// | `ssm.gdn_prep` | the packed `[b \| a]` projection, `dt_bias`, `a_log` |
/// | `ssm.gated_delta{,_chunked}` | the POST-CONV `qkv`, `z`, the packed `[g_log \| beta]`, the state |
///
/// `ssm/gdn_core.wgsl` and `ssm/gdn_prep.wgsl` do not cut there. `gdn_core`
/// is ONE launch that convolves, l2-normalises, cooks the gates and runs the
/// recurrence; `gdn_prep` is the first three of those and `gdn_core_recurrent`
/// the last. Both halves take `conv_w`, `conv_b` and the conv state, and both
/// take the WHOLE post-projection row with `q_off`/`k_off`/`v_off` to cut it
/// by — nine scalars where the points state four.
///
/// So there is no arrangement of these shaders that answers a point:
///
/// * `ssm.causal_conv1d` has no shader at all. The convolution exists here
///   only fused into a launch that also runs the recurrence, and a point that
///   fired `gdn_core` would be running the mixer twice.
/// * `ssm.gdn_prep` declares three slots and NOT ONE OF THEM is the packed
///   row `gdn_prep_bfloat16` reads. `gdn_prep_slotted` claimed that point by
///   name under the routine layer and the claim was a LIE against the
///   current declaration — the same lie `qwen_gdn_post_conv_prep_bf16` told
///   on cuda, which W10 removed by writing a shader that takes the
///   declaration's slots. Nothing asserts it now; the family's default
///   refuses by name instead.
/// * `ssm.gated_delta{,_chunked}` declare a post-conv `qkv` and read the
///   gates packed. `gdn_core_recurrent*` read THREE compact f32 planes
///   (`pre_q`, `pre_k`, `pre_gate`) that only `gdn_prep` produces, and take
///   the conv weights again on top. A body could fire the pair — prep then
///   scan — but it would have to invent `pre_q`/`pre_k`/`pre_gate`, which is
///   scratch this plane cannot allocate (`Ctx` is `dyn Encode`; there is no
///   `Ctx::scratch` behind it), and it would still have no conv weight,
///   because this point declares none.
/// * `ssm.kda_step` and `ssm.kda_chunked` are kimi's KDA rule and no `.wgsl`
///   in this tree mentions it.
///
/// # SEAM — what closes it, and the choice is real
///
/// The prefill scan's launch geometry is the shader's and stays there:
/// `ssm/gdn_prep.wgsl` compiles nine `(PIE_LANES, PIE_VROWS)` variants and
/// its own `main` states how a 32-lane workgroup divides into `32 / LANES`
/// value groups of `VROWS` rows each, so a body that picks the pair fitting
/// `Dk`/`Dv` reads the division off the file it fires.
///
///  1. **Shaders that take the declaration's slots.** A `qwen_gdn_ba_gates`
///     twin (packed `[b | a]` in, packed `[g_log | beta]` out, `v_heads` read
///     off half the operand's width) and a scan that reads the packed row and
///     the packed gates. This is what W10 did on cuda and its GATE is worth
///     restating: the packed cut has to be the KERNEL's, because the two
///     halves are `2 * v_heads` apart and a host that offsets by `v_heads` is
///     right at exactly one token.
///  2. **Tier-2.** Make the fused core an inherent method on this plane and
///     have the text gate on `inputs.wgpu()` with a tier-1 else. Cheap, and
///     it costs the text its plane-agnosticism for qwen3.5.
///  3. **Scratch on `Encode`.** Would let a body fire prep-then-scan, and is
///     the wrong shape for the same reason cuda's `Ctx::scratch` is a named
///     device slab and not an arena: the three planes are alive only between
///     the two launches, so they are the BODY's and not the plan's.
///
/// (1) is the one that matches the declaration and the one every other plane
/// has converged on. It is also the largest: three shaders, and the
/// rounding-trajectory law from W2 applies — a prefill tail must round the
/// way the decode step rounds, or the second decoded token diverges.
#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {}
