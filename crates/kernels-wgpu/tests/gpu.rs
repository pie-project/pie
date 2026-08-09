//! What a module has to survive before a device is even asked for, and then
//! what it RETURNS when one is.
//!
//! `naga` is a Rust WGSL front end and it ships inside `wgpu`, so the thing
//! that turns a `.wgsl` into a pipeline can be called from a `#[test]` on a
//! machine with no adapter at all. That is this backend's one structural
//! advantage over its siblings -- `kernels-vulkan` needs `glslc` to learn the
//! same thing -- and it is worth spending: `source::tests::
//! every_declared_variant_expands` proves the directives resolve, and this
//! proves the WGSL they resolve to is WGSL.
//!
//! # The half of this file that needs a device
//!
//! Everything above `// The device half` is structural and runs anywhere.
//! Everything below it dispatches a real entrypoint against a real adapter and
//! compares what comes back with a scalar reference. A parse and a validation
//! prove a module is one a device would ACCEPT; they say nothing whatsoever
//! about the number it produces, and `.wiki/new-driver/wgpu.md` §7 is the
//! record of 52 green tests that proved exactly that much and no more.
//!
//! The five rules the device half is written to are `kernels-vulkan`'s, and
//! each one is a defect somebody already paid for:
//!
//! 1. **The reference is computed from the same bf16 the device was given.**
//!    [`pack`] returns both the bytes and the widened values, so a reference
//!    built from the original `f32` is not spellable here. Comparing against
//!    the `f32` folds the input's own rounding into the tolerance and quietly
//!    widens it by a bf16 ulp per operand.
//! 2. **The tolerance scales by the ROW's own largest magnitude**, not by
//!    `max(|want|, 1.0)`. That floor of one turned a 2% claim into a flat
//!    absolute 0.02, which is 7% of an attention value and 16% of a router
//!    weight. See [`agrees`].
//! 3. **Every size is ragged.** 13 rows of 460, a 461-wide bias, 11 keys, 7
//!    experts, a page size of 3. `.wiki/new-driver/vulkan.md` §12 is the whole
//!    lesson: three pointwise tests there ran at n = 512 against a 256-wide
//!    workgroup and two GEMV tests at 16 rows against a kernel covering 8, so
//!    the last partial group never existed and `div_ceil` and plain division
//!    were the same expression. `every_grid_here_is_ragged` holds every extent
//!    this file dispatches against its workgroup and asserts the division is
//!    inexact, so swapping [`over`]'s `div_ceil` for `/` fails.
//! 4. **Every check is confirmed to FAIL when its reference is perturbed.**
//!    [`agrees`] returns an `Err` rather than panicking so that each test can
//!    run its own control on its own device answer through
//!    [`refuses_a_perturbed_reference`]. A test that has never failed has not
//!    been shown to test anything.
//! 5. **Every layout comes from the TABLE.** [`run`] builds its bind group
//!    layout from `kernels_wgpu::bindings()` and refuses a buffer list that is
//!    not `storage_count()` long; [`Block`] writes uniform fields BY NAME at
//!    the offsets `uniform_layout()` gives and refuses to finish with a field
//!    unwritten. Reading a layout off the shader would test the BODY and not
//!    the ABI.
//!
//! ## Two things that are this backend's alone
//!
//! **bf16 crosses as `array<u32>`, two values per word, low half first.** So a
//! bf16 pointwise kernel's x-extent is in WORDS, which is half the element
//! count, and every grid below that looks half the size it should be is that.
//! A host that copies Metal's or Vulkan's element extents overshoots by 2x --
//! harmless, because every pointwise body guards on `arrayLength(&out_)` --
//! but a host that then halves the wrong quantity undershoots, and an
//! undershot grid writes nothing, reads back as the zeros the buffer was born
//! with, and completes successfully.
//!
//! **`dispatch_workgroups` counts WORKGROUPS**, where Metal's
//! `dispatchThreads` counts threads. Every grid here is therefore divided by
//! the module's own `@workgroup_size`, rounded UP, and the rounding is what
//! `every_grid_here_is_ragged` pins.
//!
//! ## Why none of these is `#[ignore]`
//!
//! An ignored test is one nobody runs, including on the machine that has the
//! hardware. There is no `native` feature here to gate them with either --
//! this crate has no build product -- so a test that needs an adapter and
//! finds none prints why and returns. A build box stays green; a machine with
//! a GPU is asked the real question.
//!
//! # What this file found, and what it did not
//!
//! `a_bias_is_broadcast_down_every_row_at_an_odd_width` was red on every
//! adapter it ran on before `kernels/norm/add_bias.wgsl` was rewritten to own
//! a WORD rather than a column pair. It is green now, and swapping the two
//! halves at the pack makes it red again, so the repair is load-bearing.
//!
//! What it did NOT find, recorded because the distinction matters for how the
//! rest of this file should be read: removing that body's single-writer guard
//! does not fail it. `biased` derives each half's column from its own element
//! index, so both writers of a straddling word compute identical bytes. The
//! guard buys that the write is not a RACE — a memory-model property, which no
//! comparison against a reference can see. A test says what a device returned;
//! it does not say the return was well-defined.
//!
//! # How much of the table this file answers for
//!
//! `every_stated_row_is_dispatched_or_named` is the assertion, and [`COVERAGE`]
//! is the list it holds. All 44 STATED rows are dispatched; the other 56 carry
//! axes and a name and no operands, so no layout can be derived from them and
//! this harness cannot bind one. The count is pinned in both directions — the
//! list must be exactly the table's stated set, every name in it must still
//! appear in this file, and [`run`] refuses to dispatch a row the list does
//! not claim — so the number shrinking is a failure rather than a silence.

#![allow(clippy::print_stdout)]

/// Every entrypoint of the `norm` and `mlp` families, through `naga`'s front
/// end.
///
/// An expander that resolves the wrong `//#if` arm produces a file that is
/// still a file: an activation swapped for its neighbour, a binding declared in
/// a variant that does not bind it, a `const` a dead arm was supposed to
/// provide. Most of those land as a parse or resolve error here, which is a
/// test failure with a line number rather than a pipeline that fails to build
/// at the first fire -- or worse, one that builds and computes a gelu where the
/// text asked for a silu.
#[test]
fn norm_and_mlp_modules_parse() {
    let entrypoints = [
        "gated_rms_bfloat16",
        "gated_rms_strided_bfloat16",
        "geglu_tanh_bfloat16",
        "geglu_tanh_strided_bfloat16",
        "gptoss_swiglu_bfloat16",
        "layer_scalar_mul_bfloat16",
        "residual_add_bfloat16",
        "residual_add_strided_bfloat16",
        "rms_residual_bfloat16",
        "rms_residual_scaled_bfloat16",
        "rms_single_row_bfloat16",
        "rms_strided_head_row_bfloat16",
        "rms_strided_row_bfloat16",
        "silu_mul_bfloat16",
        "silu_mul_strided_bfloat16",
        "vnorm_single_row_bfloat16",
    ];

    for entrypoint in entrypoints {
        let source =
            kernels_wgpu::entrypoint_source(entrypoint, kernels_wgpu::Capability::Baseline)
                .unwrap_or_else(|why| panic!("`{entrypoint}` has no source: {why}"));

        // `parse_str` is the front end's whole first half: it lexes, it
        // resolves every identifier and it infers every type, so an undeclared
        // binding and a call with the wrong arity both fail here. The second
        // half is `naga::valid::Validator`, which is a different question and a
        // sharper one -- a pointer parameter in the `storage` address space
        // parses cleanly and then fails validation, which is how this tree
        // briefly held 478 modules that could never have become a pipeline.
        // That half is `examples/validate_all` and the crate-wide validation
        // test, over every family at once; repeating it per family here would
        // be two places to keep in step for one answer.
        if let Err(why) = naga::front::wgsl::parse_str(&source) {
            panic!(
                "`{entrypoint}` is not WGSL:\n{}",
                why.emit_to_string(&source)
            );
        }
    }
}

/// Every entrypoint of the `attn` family, through `naga`'s front end.
///
/// This family is where a bad port is quietest. The bodies are guarded by
/// `//#if PIE_HEAD_DIM > 128` and friends, the paged arms bind eleven storage
/// buffers whose numbers were derived from the row and not transcribed, and
/// `sdpa_paged_mma` is a name inherited from Metal's table rather than a
/// hardware claim -- three separate ways to emit a file that is still a file.
/// A resolve error here costs a line number; the same mistake found at the
/// first dispatch costs a wrong tensor.
#[test]
fn attn_modules_parse() {
    let entrypoints = [
        "gate_bfloat16",
        "kv_append_bfloat16",
        "kv_append_paged_bfloat16",
        "logit_softcap_bfloat16",
        "q_gate_split_bfloat16",
        "sdpa_paged_decode_bfloat16_d_128",
        "sdpa_paged_decode_bfloat16_d_128_p32",
        "sdpa_paged_decode_bfloat16_d_256",
        "sdpa_paged_decode_bfloat16_d_512",
        "sdpa_paged_decode_bfloat16_d_64",
        "sdpa_paged_decode_bfloat16_d_64_p32",
        "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
        "sdpa_paged_decode_sink_bfloat16_d_64",
        "sdpa_paged_mma_bfloat16_d_64",
        "sdpa_paged_mma_sink_bfloat16_d_64",
        "sdpa_paged_tiled_bfloat16_d_128",
        "sdpa_paged_tiled_bfloat16_d_256",
        "sdpa_paged_tiled_bfloat16_d_512",
        "sdpa_paged_tiled_bfloat16_d_64",
        "sdpa_paged_tiled_sink_bfloat16_d_64",
        "sdpa_paged_tiled_strided_bfloat16_d_256",
        "sdpa_vector_decode_bfloat16_d_128",
        "sdpa_vector_decode_bfloat16_d_256",
        "sdpa_vector_decode_bfloat16_d_64",
        "sdpa_vector_decode_sink_bfloat16_d_64",
        "sdpa_vector_decode_swa_bfloat16_d_256",
        "sdpa_vector_decode_swa_bfloat16_d_512",
        "split_qkv_bf16",
    ];

    // `common/bf16.inc.wgsl` declares `pie_load_bf16`/`pie_store_bf16` taking
    // `ptr<storage, array<u32>, read>`. That is WGSL's
    // `unrestricted_pointer_parameters` extension, and naga 30 lists it
    // `Unimplemented`: `naga::valid::Validator` rejects the DECLARATION, not
    // just a call, so the mere presence of the include would sink the second
    // half of this test for every module in the tree. No attn body CALLS
    // either -- each restates the half-index addressing against its own
    // binding and delegates the conversion -- so dropping the two declarations
    // costs this file nothing and buys real validation. Delete this helper the
    // day the fragment stops taking pointers.
    fn without_pointer_helpers(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut depth = 0usize;
        for line in src.lines() {
            if depth == 0
                && (line.starts_with("fn pie_load_bf16(") || line.starts_with("fn pie_store_bf16("))
            {
                depth = 1;
                continue;
            }
            if depth > 0 {
                depth += line.matches('{').count();
                depth -= line.matches('}').count();
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    for entrypoint in entrypoints {
        let source =
            kernels_wgpu::entrypoint_source(entrypoint, kernels_wgpu::Capability::Baseline)
                .unwrap_or_else(|why| panic!("`{entrypoint}` has no source: {why}"));

        // Half one: lex, resolve, infer. An undeclared binding, a call with the
        // wrong arity and a `const` that only a dead `//#if` arm defined all
        // fail right here.
        if let Err(why) = naga::front::wgsl::parse_str(&source) {
            panic!(
                "`{entrypoint}` is not WGSL:\n{}",
                why.emit_to_string(&source)
            );
        }

        // Half two: the analysis that catches the mistakes this family is
        // actually prone to -- a `workgroupBarrier()` reached under
        // non-uniform control flow, which on a device is a HANG and not a wrong
        // number, and a `var<workgroup>` array indexed out of its declared
        // bound.
        let trimmed = without_pointer_helpers(&source);
        let module = match naga::front::wgsl::parse_str(&trimmed) {
            Ok(module) => module,
            Err(why) => panic!(
                "`{entrypoint}` stopped being WGSL once the pointer helpers were dropped:\n{}",
                why.emit_to_string(&trimmed)
            ),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        if let Err(why) = validator.validate(&module) {
            panic!(
                "`{entrypoint}` does not validate:\n{}",
                why.emit_to_string(&trimmed)
            );
        }
    }
}

/// Every entrypoint of the `layout`, `rope`, `sample` and `ptir` families,
/// through `naga`'s front end.
///
/// These four families are the ones whose bodies are addressing arithmetic
/// almost end to end, and addressing arithmetic type-checks whatever it
/// computes. What this test can still catch is the layer under it: the
/// `embed_gather` arms are a four-axis product of `//#if`s over `PIE_BITS`,
/// `PIE_GROUP`, `PIE_MB` and `PIE_SCALED` where a mis-nested arm leaves
/// `PIE_CODES_PER_WORD` undefined; `neox` has seven variants that disagree
/// about whether `inv_freq` is BOUND and therefore about which uniform field
/// is which; `row_gather` declares no uniform block at all, so a stray
/// `params.` there is a resolve error rather than a silently misread word.
#[test]
fn layout_rope_sample_ptir_modules_parse() {
    let entrypoints = [
        // layout: the 24-way `{,_scaled}` x `{,_mb}` x gs x b product, plus
        // the two unparameterised rows.
        "embed_gather_4bit_bfloat16_gs_32_b_4",
        "embed_gather_4bit_bfloat16_gs_32_b_8",
        "embed_gather_4bit_bfloat16_gs_64_b_4",
        "embed_gather_4bit_bfloat16_gs_64_b_8",
        "embed_gather_4bit_bfloat16_gs_128_b_4",
        "embed_gather_4bit_bfloat16_gs_128_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
        "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
        "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
        "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
        "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
        "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
        "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
        "row_gather_bfloat16",
        "ple_combine_bfloat16",
        // rope
        "neox_decode_bfloat16",
        "neox_freqs_decode_bfloat16",
        "neox_freqs_mb_bfloat16",
        "neox_mb_bfloat16",
        "neox_prop_decode_bfloat16",
        "neox_prop_mb_bfloat16",
        "neox_strided_bfloat16",
        // sample
        "argmax_logits_bfloat16",
        // ptir
        "copy_logits_bf16",
    ];

    // Restated here rather than shared with the tests above, because this file
    // is written by several hands at once and a fixture is the one thing that
    // makes two `#[test]`s fail together. See `attn_modules_parse` for the full
    // argument: `common/bf16.inc.wgsl` declares `pie_load_bf16`/
    // `pie_store_bf16` taking `ptr<storage, array<u32>, read>`, naga 30 lists
    // `unrestricted_pointer_parameters` `Unimplemented`, and the validator
    // rejects the DECLARATION whether or not anything calls it. `embed_gather`
    // is the only body here that includes the fragment and it calls neither --
    // it unpacks against its own binding -- so dropping the two declarations
    // costs nothing and lets the second half of this test run.
    fn without_pointer_helpers(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut depth = 0usize;
        for line in src.lines() {
            if depth == 0
                && (line.starts_with("fn pie_load_bf16(") || line.starts_with("fn pie_store_bf16("))
            {
                depth = 1;
                continue;
            }
            if depth > 0 {
                depth += line.matches('{').count();
                depth -= line.matches('}').count();
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    for entrypoint in entrypoints {
        let source =
            kernels_wgpu::entrypoint_source(entrypoint, kernels_wgpu::Capability::Baseline)
                .unwrap_or_else(|why| panic!("`{entrypoint}` has no source: {why}"));

        // Half one, over the source AS SERVED -- the exact text
        // `create_shader_module` would be handed, pointer helpers and all.
        if let Err(why) = naga::front::wgsl::parse_str(&source) {
            panic!(
                "`{entrypoint}` is not WGSL:\n{}",
                why.emit_to_string(&source)
            );
        }

        // Half two: uniformity and bounds. `argmax_logits_bfloat16` is the row
        // this half exists for -- its `workgroupBarrier()`es sit inside the
        // reduction loop with the guard on the STORE and not on the barrier,
        // and getting that backwards is a device hang rather than a wrong
        // token.
        let trimmed = without_pointer_helpers(&source);
        let module = match naga::front::wgsl::parse_str(&trimmed) {
            Ok(module) => module,
            Err(why) => panic!(
                "`{entrypoint}` stopped being WGSL once the pointer helpers were dropped:\n{}",
                why.emit_to_string(&trimmed)
            ),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        if let Err(why) = validator.validate(&module) {
            panic!(
                "`{entrypoint}` does not validate:\n{}",
                why.emit_to_string(&trimmed)
            );
        }
    }
}

/// Every entrypoint of the `quant` family, through `naga`'s front end.
///
/// 303 modules from three files, and the list is not typed out: it is asked of
/// `source::declared()` and filtered to `quant/`, because a hand-kept list of
/// 282 `qmm_t` variants would be a second copy of the `// pie:instantiate`
/// block that could disagree with it. The count is asserted instead, so a
/// directive that stops expanding fails here rather than quietly shrinking the
/// set this test covers.
///
/// What it is looking for is not arithmetic. Every one of these modules is the
/// same body under a different `//#if`, and the ways a variant breaks are:
/// `PIE_BM`/`PIE_BN` reaching an array size from a dead arm, a binding declared
/// by a flag its variant does not set, `PIE_CODES_PER_WORD` coming from an
/// include a `//#if` skipped. All three are resolve errors here and a wrong
/// tensor at the first dispatch.
#[test]
fn quant_modules_parse() {
    // `common/bf16.inc.wgsl` declares `pie_load_bf16`/`pie_store_bf16` taking
    // `ptr<storage, array<u32>, read>`, which naga 30 rejects at VALIDATION
    // (the `unrestricted_pointer_parameters` extension is `Unimplemented`) --
    // and it rejects the declaration, not the call, so their mere presence
    // would sink the second half of this test for every module in the tree. No
    // quant body calls either: each restates the half-index split against its
    // own binding and delegates only the conversion, for exactly this reason.
    // Delete this once the fragment stops taking pointers.
    fn without_pointer_helpers(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut depth = 0usize;
        for line in src.lines() {
            if depth == 0
                && (line.starts_with("fn pie_load_bf16(") || line.starts_with("fn pie_store_bf16("))
            {
                depth = 1;
                continue;
            }
            if depth > 0 {
                depth += line.matches('{').count();
                depth -= line.matches('}').count();
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    let quant: Vec<String> = kernels_wgpu::source::declared()
        .into_iter()
        .filter(|(path, variant)| {
            path.starts_with("quant/") && variant.tier == kernels_wgpu::Capability::Baseline
        })
        .map(|(_, variant)| variant.entrypoint)
        .collect();
    assert_eq!(
        quant.len(),
        303,
        "the quant tree declares 282 + 18 + 3 baseline variants; it now declares {}",
        quant.len()
    );

    for entrypoint in quant {
        let source =
            kernels_wgpu::entrypoint_source(&entrypoint, kernels_wgpu::Capability::Baseline)
                .unwrap_or_else(|why| panic!("`{entrypoint}` has no source: {why}"));

        // Half one: lex, resolve, infer.
        if let Err(why) = naga::front::wgsl::parse_str(&source) {
            panic!(
                "`{entrypoint}` is not WGSL:\n{}",
                why.emit_to_string(&source)
            );
        }

        // Half two: the analysis this family actually needs. `qmm_t`'s GEMM
        // arm has two `workgroupBarrier()` calls inside its K loop, and a
        // barrier under non-uniform control flow is a HANG on a device rather
        // than a wrong number -- naga's uniformity analysis is the only thing
        // short of hardware that says so.
        let trimmed = without_pointer_helpers(&source);
        let module = match naga::front::wgsl::parse_str(&trimmed) {
            Ok(module) => module,
            Err(why) => panic!(
                "`{entrypoint}` stopped being WGSL once the pointer helpers were dropped:\n{}",
                why.emit_to_string(&trimmed)
            ),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        if let Err(why) = validator.validate(&module) {
            panic!(
                "`{entrypoint}` does not validate:\n{}",
                why.emit_to_string(&trimmed)
            );
        }
    }
}

/// Every entrypoint of the `moe` and `ssm` families, through `naga`'s front
/// end and then its validator.
///
/// 98 modules from five files, and the list is asked of `source::declared()`
/// rather than typed out: 72 of them are `qmm_t_routed`'s tile grid, and a
/// hand-kept copy of that block is a second list that can disagree with the
/// first. The two counts are asserted so a directive that stops expanding
/// fails here instead of quietly shrinking the set this test covers.
///
/// What the two halves actually prove, because it is less than it looks:
///
/// * The FRONT END is where the atomic rules land. Every bf16 store in these
///   two families is an `atomicAnd`/`atomicOr` pair -- two invocations own the
///   two halves of one `u32` and WGSL has no sub-word atomic -- and both
///   "atomic operation on a pointer to a non-atomic" and "atomic variables
///   cannot be accessed directly" are `parse_str` errors, not validation ones.
///   So is every arity, type and undeclared-binding mistake a bad `//#if` arm
///   makes.
/// * The VALIDATOR adds the module-level rules the parser does not see: two
///   `var`s on one `@group`/`@binding` pair, a uniform member at an offset its
///   type may not sit at, an entry point whose resources do not resolve.
///
/// What NEITHER half checks is control-flow uniformity around
/// `workgroupBarrier()`. naga 30 accepts a barrier inside `if (lid.x < 3u)`
/// and a barrier after `if (lid.x < 3u) { return; }`, both measured against
/// this same `Validator` configuration -- WGSL demotes uniformity to a
/// diagnostic and naga does not raise it. `qmv_routed`, `gdn_core` and
/// `gdn_prep` guard their STORES rather than returning early for a reason that
/// therefore stands on review and on the device, not on this test: a barrier
/// some invocations never reach is a HANG, and nothing here would say so.
#[test]
fn moe_and_ssm_modules_parse() {
    // `common/bf16.inc.wgsl` declares `pie_load_bf16`/`pie_store_bf16` taking
    // `ptr<storage, array<u32>, ..>`, which naga 30 rejects at VALIDATION --
    // `unrestricted_pointer_parameters` is `Unimplemented` (gfx-rs/wgpu#5158)
    // -- and it rejects the DECLARATION, not the call, so their presence alone
    // would sink the validation half for every module that includes the
    // fragment. No body in these two families calls either: each restates the
    // half-index split against its own binding and delegates only the
    // conversion, for exactly this reason. A local copy rather than a shared
    // helper, so that deleting this once the fragment stops taking pointers is
    // one test's edit.
    fn without_pointer_helpers(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut depth = 0usize;
        for line in src.lines() {
            if depth == 0
                && (line.starts_with("fn pie_load_bf16(") || line.starts_with("fn pie_store_bf16("))
            {
                depth = 1;
                continue;
            }
            if depth > 0 {
                depth += line.matches('{').count();
                depth -= line.matches('}').count();
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    let declared = kernels_wgpu::source::declared();
    let family = |prefix: &str| -> Vec<String> {
        declared
            .iter()
            .filter(|(path, variant)| {
                path.starts_with(prefix) && variant.tier == kernels_wgpu::Capability::Baseline
            })
            .map(|(_, variant)| variant.entrypoint.clone())
            .collect()
    };

    let moe = family("moe/");
    let ssm = family("ssm/");
    // 7 routing + 3 routed qmv + 72 routed qmm tiles; 2 fused gdn cores + 14
    // split prep/recurrent/scan variants.
    assert_eq!(
        moe.len(),
        82,
        "moe declares {} entrypoints, not 82",
        moe.len()
    );
    assert_eq!(
        ssm.len(),
        16,
        "ssm declares {} entrypoints, not 16",
        ssm.len()
    );

    for entrypoint in moe.into_iter().chain(ssm) {
        let source =
            kernels_wgpu::entrypoint_source(&entrypoint, kernels_wgpu::Capability::Baseline)
                .unwrap_or_else(|why| panic!("`{entrypoint}` has no source: {why}"));

        if let Err(why) = naga::front::wgsl::parse_str(&source) {
            panic!(
                "`{entrypoint}` is not WGSL:\n{}",
                why.emit_to_string(&source)
            );
        }

        let trimmed = without_pointer_helpers(&source);
        let module = match naga::front::wgsl::parse_str(&trimmed) {
            Ok(module) => module,
            Err(why) => panic!(
                "`{entrypoint}` stopped being WGSL once the pointer helpers were dropped:\n{}",
                why.emit_to_string(&trimmed)
            ),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        if let Err(why) = validator.validate(&module) {
            panic!(
                "`{entrypoint}` does not validate:\n{}",
                why.emit_to_string(&trimmed)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The check that supersedes the five above, and the session that produced it.
// ---------------------------------------------------------------------------

/// Every declared variant, at every tier, through `naga`'s FULL front end —
/// parse and then validate.
///
/// The five family tests above call `parse_str` and stop. That was enough to
/// catch an expander that resolved the wrong `//#if` arm, and it was not enough
/// to catch anything else: a parse proves the text is WGSL, and `naga`'s
/// VALIDATOR is what proves it is a WGSL a device would accept.
///
/// The gap was not hypothetical. `common/bf16.inc.wgsl` was first written with
///
/// ```wgsl
/// fn pie_load_bf16(words: ptr<storage, array<u32>, read>, i: u32) -> f32
/// ```
///
/// which parses cleanly and is not legal WGSL: core WGSL allows a pointer
/// parameter only in the `function`, `private` and `workgroup` address spaces,
/// and `ptr<storage, ...>` needs the `unrestricted_pointer_parameters` language
/// extension that `naga` refuses and no WebGPU implementation owes anyone. Four
/// of the five family tests were green. **478 of the 480 modules failed here**,
/// and every one of them would have failed at `create_shader_module` on the
/// first fire.
///
/// So this runs over the whole tree, and it is the check to add to rather than
/// the ones above.
///
/// ## Why `Capabilities::all()`
///
/// The point of this test is the LANGUAGE, not the device: a body that needs a
/// capability should fail where the capability is asked for, which is
/// `Capability::requires()` and the tier that names it. Validating with an
/// empty capability set would fold two different failures — "this is not WGSL"
/// and "this adapter cannot run it" — into one message.
///
/// The second question is checked separately and it matters just as much: see
/// `no_baseline_module_needs_a_capability` below, which is the one that would
/// have caught `unpack2x16float` reaching a Baseline entrypoint.
#[test]
fn every_module_parses_and_validates() {
    let mut checked = 0usize;
    let mut broken = Vec::new();

    for (file, variant) in kernels_wgpu::source::declared() {
        let name = format!("{}@{}", variant.entrypoint, variant.tier.tag());
        let source = match kernels_wgpu::entrypoint_source(&variant.entrypoint, variant.tier) {
            Ok(source) => source,
            Err(why) => {
                broken.push(format!("kernels/{file}: `{name}` has no source: {why}"));
                continue;
            }
        };

        let module = match naga::front::wgsl::parse_str(&source) {
            Ok(module) => module,
            Err(why) => {
                broken.push(format!(
                    "kernels/{file}: `{name}` is not WGSL:\n{}",
                    why.emit_to_string(&source)
                ));
                continue;
            }
        };

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        match validator.validate(&module) {
            Ok(_) => checked += 1,
            Err(why) => broken.push(format!(
                "kernels/{file}: `{name}` does not validate:\n{}",
                why.emit_to_string(&source)
            )),
        }
    }

    assert!(
        broken.is_empty(),
        "{} of {} modules do not survive naga:\n\n{}",
        broken.len(),
        broken.len() + checked,
        broken.join("\n\n"),
    );
    assert!(
        checked >= 481,
        "only {checked} modules were checked; the table declares 481 \
         entrypoints and a check that silently reads nothing is a check that \
         passes",
    );
}

/// No BASELINE module needs a capability, which is what Baseline MEANS.
///
/// `Capability::Baseline::requires()` answers `&[]`, and every fallback in the
/// crate lands on Baseline. A baseline module that needed an optional feature
/// would be an entrypoint that resolves on the author's adapter and on no
/// other — and it would resolve silently, because the failure arrives at
/// `create_shader_module` during a model load rather than anywhere a person is
/// looking.
///
/// This is `kernels-vulkan`'s "the baseline tier required an optional feature"
/// finding, ported. There, seven modules carried `uint64_t` strides, so declared
/// `Int64`, so needed `shaderInt64` — for an overflow that could not occur. Here
/// it was `unpack2x16float`, which is spelled like a core builtin, needs no
/// `enable`, and is gated by `naga` behind `SHADER_FLOAT16_IN_FLOAT32` — which
/// `wgpu` grants only from the `SHADER_F16_IN_F32` DOWNLEVEL flag, absent on
/// adapters that are otherwise fine. The 41 `_fp16_precast` entrypoints now
/// convert with integer arithmetic instead.
///
/// The check is mechanical: validate with the EMPTY capability set. Anything a
/// baseline module needs beyond core shows up as a validation error naming the
/// capability, which is a far better message than a device would give.
#[test]
fn no_baseline_module_needs_a_capability() {
    let mut broken = Vec::new();
    let mut checked = 0usize;

    for (file, variant) in kernels_wgpu::source::declared() {
        if variant.tier != kernels_wgpu::Capability::Baseline {
            continue;
        }
        let Ok(source) = kernels_wgpu::entrypoint_source(&variant.entrypoint, variant.tier) else {
            // `every_module_parses_and_validates` owns this failure; reporting
            // it twice would make one defect look like two.
            continue;
        };
        let Ok(module) = naga::front::wgsl::parse_str(&source) else {
            continue;
        };

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        match validator.validate(&module) {
            Ok(_) => checked += 1,
            Err(why) => broken.push(format!(
                "kernels/{file}:{}: `{}` is BASELINE and needs more than core \
                 WebGPU:\n{}",
                variant.line,
                variant.entrypoint,
                why.emit_to_string(&source),
            )),
        }
    }

    assert!(
        broken.is_empty(),
        "{} baseline modules ask for a capability. Either the body must be \
         rewritten in core WGSL, or the variant belongs behind a tier that \
         NAMES what it needs in `Capability::requires()`:\n\n{}",
        broken.len(),
        broken.join("\n\n"),
    );
    assert_eq!(
        checked, 481,
        "every entrypoint has a baseline variant, so {checked} is not 481 and \
         either a variant is missing or this check stopped reading",
    );
}

// ---------------------------------------------------------------------------
// The device half: what these modules actually RETURN.
// ---------------------------------------------------------------------------

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

/// Rows most dispatches here fire over. Not a multiple of anything.
const ROWS: u32 = 13;

/// Elements in one row. 460 is not a multiple of `norm/rms.wgsl`'s
/// 1024-element chunk, and its 230 words are not a multiple of that file's
/// 256-lane store loop or of `residual_add`'s 256-wide workgroup.
const WIDTH: u32 = 460;

/// What an untouched byte holds.
///
/// `0x4780` is bf16 for `65536.0`, so a word of two of them is a value nothing
/// below can produce: every input here is in `[-2, 2)` and the widest thing
/// done to a pair of them is a dot product over a few hundred terms.
///
/// **Zero cannot be used**: it is what a fresh `wgpu` buffer already holds, so
/// a slot that was never written and one written with a zero are the same
/// bytes, and a dispatch that ran nothing would pass a check written against
/// zero. Nor can `-1.0`, which [`spread`] can produce.
const SENTINEL: u32 = 0x4780_4780;

/// How long to wait for a device that has stopped answering.
const WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// An adapter, opened once for the whole binary.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    features: wgpu::Features,
}

/// One device at a time, for the whole suite.
///
/// **Not a style choice — measured next door.** With `cargo test`'s default
/// parallelism a file like this one opens ten `wgpu::Device`s at once, each of
/// which is a `VkDevice` with driver-owned helper threads behind it, and
/// roughly one run in three then wedges with no progress on the NVIDIA
/// proprietary driver. `driver-wgpu/tests/device.rs` records the whole
/// finding. So there is ONE device, built under this lock, and a test holds
/// the lock for as long as it is using it.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The device, or the printed reason there is none. Built at most once.
static OPENED: OnceLock<Result<Gpu, String>> = OnceLock::new();

/// A device nothing else is using, or `None` after printing why.
///
/// `PIE_WGPU_FALLBACK=1` asks for the SOFTWARE adapter instead, which is how
/// this suite is run a second time against a completely different
/// implementation of the same WGSL on the same machine. `WGPU_BACKEND` and
/// `WGPU_POWER_PREF` pick between several hardware adapters. None of the three
/// is a deployment knob; they exist because "it agrees on the card it was
/// written on" is the weakest form of agreement there is.
fn adapter() -> Option<(&'static Gpu, MutexGuard<'static, ()>)> {
    // The lock is taken BEFORE the device is built, so two test threads
    // arriving together do not each construct one and throw one away.
    let held = ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner);
    match OPENED.get_or_init(open) {
        Ok(gpu) => Some((gpu, held)),
        Err(why) => {
            println!("SKIP: {why}");
            None
        }
    }
}

/// Open one adapter and its device, asking for the ADAPTER's limits.
///
/// Not `Limits::downlevel_defaults()`, and this is the one line in the harness
/// that is a claim rather than plumbing: WebGPU's guaranteed
/// `maxStorageBuffersPerShaderStage` is
/// [`kernels_wgpu::DOWNLEVEL_STORAGE_BUFFERS`] = 8, and
/// `sdpa_paged_decode` binds ELEVEN. A harness that requested the guaranteed
/// floor out of caution would fail to create exactly the attention pipelines,
/// on hardware that would have run them, with a message about a limit rather
/// than about attention.
fn open() -> Result<Gpu, String> {
    let instance = wgpu::Instance::new(
        // `with_env` so `WGPU_BACKEND=vulkan` selects one.
        wgpu::InstanceDescriptor::new_without_display_handle().with_env(),
    );
    let fallback = std::env::var("PIE_WGPU_FALLBACK").is_ok();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference:
            wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance),
        force_fallback_adapter: fallback,
        compatible_surface: None,
        // Deliberately off: bucketing rounds the reported limits DOWN, and the
        // whole point of asking is the real `max_storage_buffers_per_shader_stage`.
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no adapter answered: {e}"))?;

    let info = adapter.get_info();
    let features = adapter.features();
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kernels-wgpu gpu harness"),
        // NOTHING. Every entrypoint dispatched here is BASELINE, and
        // `no_baseline_module_needs_a_capability` above is the static half of
        // the same claim: a baseline module that quietly needed `SHADER_F16`
        // would fail to create a pipeline here rather than run.
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| format!("{} would not open a device: {e}", info.name))?;

    Ok(Gpu {
        device,
        queue,
        info,
        limits,
        features,
    })
}

/// Round an `f32` to a bf16 bit pattern, round to nearest even.
///
/// The host copy of `pie_f32_to_bf16` in `kernels/common/bf16.inc.wgsl`,
/// including its NaN branch — the rounding add can carry a NaN's mantissa to
/// zero and turn it into an infinity. Written out rather than truncated
/// because truncating is a real accuracy loss over a long accumulation and
/// because a reference that rounds differently from the shader would spend the
/// whole tolerance on the rounding.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// Widen a bf16 bit pattern. Exact: bf16 IS the top half of an f32.
///
/// By SHIFT and not by cast: `v as f32` turns `0x3f80` into 16256.0 where it
/// means 1.0, and both are finite floats no assertion downstream would object
/// to.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// One `f32` through bf16 and back — what the device will hold after a store.
fn rounded(x: f32) -> f32 {
    from_bf16(to_bf16(x))
}

/// A run of `f32` as the packed bf16 words a shader reads, plus what the
/// shader will actually SEE.
///
/// Two answers from one call on purpose, and it is rule 1 made structural:
/// every reference in this file is computed from the second return value, so
/// building one from the original `f32` would take an extra variable that
/// nothing hands out.
fn pack(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let seen: Vec<f32> = values.iter().copied().map(rounded).collect();
    let mut bytes = Vec::with_capacity(values.len().next_multiple_of(2) * 2);
    for v in values {
        bytes.extend_from_slice(&to_bf16(*v).to_le_bytes());
    }
    // A whole number of words, since every shader here addresses `array<u32>`.
    if bytes.len() % 4 != 0 {
        bytes.extend_from_slice(&[0, 0]);
    }
    (bytes, seen)
}

/// Bytes back as the first `n` bf16 values they hold.
fn unpack(bytes: &[u8], n: usize) -> Vec<f32> {
    assert!(
        bytes.len() >= n * 2,
        "{} bytes back is not {n} bf16 values",
        bytes.len()
    );
    bytes
        .chunks_exact(2)
        .take(n)
        .map(|c| from_bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Bytes back as `i32`.
fn unpack_i32(bytes: &[u8], n: usize) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .take(n)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A spread of values that is not symmetric, not sorted and not near zero.
///
/// Seeded by a plain LCG so the same shape produces the same numbers on every
/// machine, which is what makes a disagreement between two adapters a finding
/// rather than a coincidence. The range is `[-2, 2)`, which keeps every value
/// and every partial sum well clear of the denormal range — a flush-to-zero
/// adapter and a conforming one must not be allowed to disagree about an
/// answer for a reason that is not the kernel's.
fn spread(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 4_194_304.0 - 2.0
        })
        .collect()
}

/// A run of small positive values, for a scale plane that must not flip signs.
fn positives(n: usize, seed: u32) -> Vec<f32> {
    spread(n, seed)
        .iter()
        .map(|v| v.abs() * 0.25 + 0.05)
        .collect()
}

/// Workgroups for `extent` lanes at `workgroup` lanes each.
///
/// `div_ceil` and not `/`, and the difference is the whole of
/// `.wiki/new-driver/vulkan.md` §12: an UNDERSHOT grid runs nothing over the
/// tail, the gap reads back as whatever the buffer was born holding, and the
/// dispatch completes successfully. `every_grid_here_is_ragged` asserts that
/// every extent this file uses makes the two expressions differ, so replacing
/// this body with `extent / workgroup` fails a dozen tests rather than none.
fn over(extent: u32, workgroup: u32) -> u32 {
    extent.div_ceil(workgroup)
}

/// Compare one row against its reference, on TWO claims at once.
///
/// `Err` rather than a panic so that [`refuses_a_perturbed_reference`] can
/// assert this FAILS on the same device answer — a check that has never failed
/// has not been shown to check anything.
///
/// # The first claim: no element is far out
///
/// Scaled by `max|want|` over the ROW, and the budget is two bf16 ulps of it.
/// A bf16 has an eight-bit significand, so the quantum at magnitude `M` is
/// `M/128`; two of them is what a rounded output can cost. NOT
/// `max(|want|, 1.0)`: `kernels-vulkan` found that floor of one turned a 2%
/// claim into a flat absolute 0.02, which is 7% of an attention value and 16%
/// of a router weight.
///
/// # The second claim, which is the one with teeth
///
/// The first claim ALONE is far too weak, and that was measured rather than
/// reasoned about: a reference that divided by `axis - 1` instead of `axis` —
/// a 0.1% error, and exactly the kind of off-by-one a port introduces — passes
/// it. It has to, because 0.1% is well inside a bf16 half-ulp.
///
/// What can see it is the COUNT. Both sides round to bf16 the same way and
/// their pre-rounding values differ only by reduction order and by WGSL's
/// couple-of-ulp allowance on `inverseSqrt` and `exp`, so an element lands on
/// a different bf16 value only if it sat within about `1e-5` of a rounding
/// boundary — roughly one element in a few thousand. A systematic shift moves
/// EVERY element and flips something like one in eight. So: at most one
/// element in fifty may differ by more than the rounding noise.
fn agrees(got: &[f32], want: &[f32], what: &str) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} values back and {} expected",
            got.len(),
            want.len()
        ));
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let budget = (scale / 128.0).max(f32::MIN_POSITIVE);
    let noise = (scale * 1e-5).max(f32::MIN_POSITIVE);
    let mut moved = 0usize;
    for (at, (g, w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() || (g - w).abs() > budget {
            return Err(format!(
                "{what}: element {at} is {g} and should be {w}; the row's \
                 largest magnitude is {scale} and the budget is {budget}"
            ));
        }
        if (g - w).abs() > noise {
            moved += 1;
        }
    }
    if moved * 50 > got.len() {
        return Err(format!(
            "{what}: {moved} of {} elements landed on a different bf16 value \
             than the reference. Each one is inside the per-element budget, \
             which is why this count exists: two computations that agree \
             differ only where an element sat within a rounding boundary of \
             the other, and that is about one in a few thousand rather than \
             one in {}",
            got.len(),
            got.len() / moved.max(1)
        ));
    }
    Ok(())
}

/// The same comparison against a reference that has MOVED, which must fail.
///
/// Rule 4, spelled once and called by every family below. The perturbation is
/// three bf16 ulps of the row's own scale on one element that is neither the
/// first nor the last — a body that wrote only element zero, or only the last
/// word, is a different defect and one the main comparison already catches.
fn refuses_a_perturbed_reference(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        scale > 0.0 && want.len() > 2,
        "{what}: an all-zero or two-element reference cannot be perturbed, so \
         this control proves nothing; give the check something to bite on",
    );
    let at = want.len() / 3;
    let mut moved = want.to_vec();
    moved[at] += scale / 40.0;
    agrees(got, &moved, what).expect_err(
        "the device's own answer must be REFUSED against a reference whose \
         element was moved three bf16 ulps of the row's scale. A check that \
         cannot fail is not a check",
    );
}

/// A storage buffer holding `bytes`.
fn storage(gpu: &Gpu, bytes: &[u8]) -> wgpu::Buffer {
    assert!(!bytes.is_empty(), "a zero-length storage buffer is refused");
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes.len().next_multiple_of(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// A storage buffer of `words` words, every one of them [`SENTINEL`].
fn sentinelled(gpu: &Gpu, words: usize) -> wgpu::Buffer {
    storage(gpu, &SENTINEL.to_le_bytes().repeat(words))
}

/// A run of `i32` as a storage buffer.
fn i32s(gpu: &Gpu, values: &[i32]) -> wgpu::Buffer {
    storage(
        gpu,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

/// A run of `u32` as a storage buffer.
fn u32s(gpu: &Gpu, values: &[u32]) -> wgpu::Buffer {
    storage(
        gpu,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

/// A run of `f32` as bf16 words, plus what the shader will see.
fn bf16s(gpu: &Gpu, values: &[f32]) -> (wgpu::Buffer, Vec<f32>) {
    let (bytes, seen) = pack(values);
    (storage(gpu, &bytes), seen)
}

/// Every byte of a buffer, through a staging copy.
///
/// `wgpu` has no `vkMapMemory`: a `STORAGE` buffer cannot be mapped, so a
/// readback is copy -> `MAP_READ | COPY_DST` staging buffer -> `map_async` ->
/// poll -> read. The POLL is what runs the callback; `map_async` only queues
/// it, and waiting on the last submission is what makes the copy have
/// happened.
fn read(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<u8> {
    let span = buffer.size();
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: span,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, Some(span));
    gpu.queue.submit([encoder.finish()]);

    let answer: std::sync::Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
        std::sync::Arc::new(Mutex::new(None));
    let park = std::sync::Arc::clone(&answer);
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        *park.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
    });
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT),
        })
        .expect("the device answered the readback poll");
    match answer.lock().unwrap_or_else(PoisonError::into_inner).take() {
        Some(Ok(())) => {}
        other => panic!("the readback did not map: {other:?}"),
    }
    let view = staging
        .slice(..)
        .get_mapped_range()
        .expect("the staging buffer mapped");
    let bytes = view.to_vec();
    // Both, in this order: a `BufferView` borrows the mapping and `unmap`
    // takes it away.
    drop(view);
    staging.unmap();
    bytes
}

/// The `@group(1) @binding(0)` uniform block, filled BY NAME.
///
/// Rule 5, made unrepresentable-to-get-wrong. `Binding::Uniform(n)` is a field
/// INDEX and a shell needs an OFFSET, and turning one into the other is not
/// multiplication: WGSL aligns a member to its own alignment, so a
/// `vec2<u32>` after a lone `i32` starts at 8 and not at 4, and the block
/// rounds to 16. `kv_append` is the row that proves it — 24 bytes of fields
/// where the naive sum of widths says 20 — and a host that packed by
/// concatenation would write both strides four bytes low and the shader would
/// read two halves of two different numbers. Nothing at runtime reports that:
/// a uniform buffer is bytes.
///
/// So every write here goes through `uniform_layout()`, by the row's own
/// spelling of the field name, and [`Block::done`] refuses a block with a
/// field nobody wrote. Renaming or reordering a row's scalars therefore fails
/// a test instead of shifting every value after the change by four bytes.
struct Block {
    row: &'static str,
    fields: Vec<kernels_wgpu::UniformField>,
    bytes: Vec<u8>,
    written: Vec<bool>,
}

impl Block {
    fn of(entrypoint: &str) -> Self {
        let sig = kernels_wgpu::sig(entrypoint)
            .unwrap_or_else(|| panic!("the table has no row for `{entrypoint}`"));
        let fields = kernels_wgpu::uniform_layout(sig);
        Self {
            row: sig.symbol,
            bytes: vec![0u8; kernels_wgpu::uniform_size(sig) as usize],
            written: vec![false; fields.len()],
            fields,
        }
    }

    /// Where the row puts `name`, refusing a name the row does not state.
    fn at(&mut self, name: &str, width: u32) -> usize {
        let at = self
            .fields
            .iter()
            .position(|f| f.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "row `{}` states no scalar called `{name}`; it states {:?}",
                    self.row,
                    self.fields.iter().map(|f| f.name).collect::<Vec<_>>()
                )
            });
        let field = self.fields[at];
        assert_eq!(
            field.size, width,
            "row `{}`'s `{name}` is {} bytes wide and this writes {width}. A \
             64-bit operand crosses as `vec2<u32>`, low word first, and \
             writing four bytes into it leaves the high word holding whatever \
             was there",
            self.row, field.size,
        );
        assert!(
            !std::mem::replace(&mut self.written[at], true),
            "row `{}`'s `{name}` was written twice",
            self.row
        );
        field.offset as usize
    }

    fn i32(mut self, name: &str, v: i32) -> Self {
        let at = self.at(name, 4);
        self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        self
    }

    fn u32(mut self, name: &str, v: u32) -> Self {
        let at = self.at(name, 4);
        self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        self
    }

    fn f32(mut self, name: &str, v: f32) -> Self {
        let at = self.at(name, 4);
        self.bytes[at..at + 4].copy_from_slice(&v.to_bits().to_le_bytes());
        self
    }

    /// A `Ty::Usize` or `Ty::I64` field: `vec2<u32>`, LOW WORD FIRST.
    ///
    /// WGSL has `u32`, `i32`, `f32` and no 64-bit integer whatsoever, so the
    /// ABI splits one. Every shader that reads one of these reads `.x` and
    /// says why the high word cannot matter; writing only four bytes here
    /// would leave `.y` holding a stale value on a device that does not zero
    /// its uniforms.
    fn wide(mut self, name: &str, v: u64) -> Self {
        let at = self.at(name, 8);
        self.bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
        self
    }

    fn done(self) -> Vec<u8> {
        for (field, written) in self.fields.iter().zip(&self.written) {
            assert!(
                written,
                "row `{}`'s `{}` was never written, so the shader would read \
                 whatever the buffer held at byte {}",
                self.row, field.name, field.offset,
            );
        }
        self.bytes
    }
}

/// Whether a row's operand kind means the shader may WRITE through it.
///
/// `wgpu` compares a bind group layout's `read_only` against the module's own
/// address space for EQUALITY, so this has to be right or the pipeline does
/// not build. Read off the row's `Ty` and not off the shader, which is the
/// point: a row that says `Buf` where its shader says `read_write` is a
/// disagreement between the table and the tree, and it should fail here.
fn writable(ty: kernels::Ty) -> bool {
    use kernels::Ty;
    matches!(
        ty,
        Ty::BufMut
            | Ty::F32sMut
            | Ty::I32sMut
            | Ty::U32sMut
            | Ty::U8sMut
            | Ty::U16sMut
            | Ty::I8sMut
            | Ty::BufArrayMut
            | Ty::BufArrayOut
            | Ty::BufArrayOutMut
    )
}

/// One dispatch of one BASELINE entrypoint, bound the way the ROW says.
///
/// `buffers` is the row's buffer-kinded operands in ROW ORDER — which is the
/// `@group(0)` binding order, because [`kernels_wgpu::bindings`] numbers that
/// run densely from zero — and `uniform` is [`Block::done`]'s bytes, or empty
/// for a row with no scalars.
///
/// Nothing here reads the shader. The bind group layout is built from
/// `bindings()`, its width is checked against `storage_count()`, and the
/// uniform binding exists exactly when `uniform_size()` is nonzero. That is
/// what makes a dispatch a test of the ABI rather than of the body: a shader
/// that declared its bindings in Metal's numbering — scalars counted alongside
/// buffers, which is 60 entrypoints' worth of defect in `kernels-vulkan`'s
/// history — fails to create a pipeline against this layout instead of reading
/// a plausible number out of the wrong slot.
fn run(gpu: &Gpu, entrypoint: &str, buffers: &[&wgpu::Buffer], uniform: &[u8], groups: [u32; 3]) {
    let sig = kernels_wgpu::sig(entrypoint)
        .unwrap_or_else(|| panic!("the table has no row for `{entrypoint}`"));
    assert!(
        !sig.operands.is_empty(),
        "`{entrypoint}` is an UNSTATED row: it names no operands, so no layout \
         can be derived from it and this harness cannot launch it. See \
         `.wiki/new-driver/vulkan.md` §13",
    );
    assert_eq!(
        buffers.len(),
        kernels_wgpu::storage_count(sig) as usize,
        "`{entrypoint}` binds {} storage buffers and {} were handed over",
        kernels_wgpu::storage_count(sig),
        buffers.len(),
    );
    assert_eq!(
        uniform.len(),
        kernels_wgpu::uniform_size(sig) as usize,
        "`{entrypoint}`'s uniform block is {} bytes and {} were handed over",
        kernels_wgpu::uniform_size(sig),
        uniform.len(),
    );
    assert!(
        groups.iter().all(|n| *n > 0),
        "`{entrypoint}` was given the grid {groups:?}. A zero on any axis \
         dispatches NOTHING and completes successfully, which is the failure \
         `over()` exists to make impossible",
    );
    let limit = gpu.limits.max_compute_workgroups_per_dimension;
    // The other half of `every_stated_row_is_dispatched_or_named`. That test
    // says every stated row is claimed; this says every dispatch is one the
    // list knows about, so the two cannot drift in either direction.
    assert!(
        is_claimed(sig.symbol),
        "`{entrypoint}` is row `{}`, which `COVERAGE` does not claim this \
         suite dispatches. Classify it there in the same edit that dispatches \
         it, or the count that test pins stops meaning anything",
        sig.symbol,
    );
    assert!(
        groups.iter().all(|n| *n <= limit),
        "`{entrypoint}`'s grid {groups:?} is past this adapter's {limit} \
         workgroups per dimension",
    );

    // The `@group(0)` layout, one entry per buffer-kinded operand, in row
    // order, numbered from zero.
    let mut entries = Vec::new();
    for (op, place) in sig.operands.iter().zip(kernels_wgpu::bindings(sig)) {
        let kernels_wgpu::Binding::Storage(at) = place else {
            continue;
        };
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: at,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: !writable(op.ty),
                },
                has_dynamic_offset: false,
                // Left open: every tensor binding ends in a runtime array
                // whose length IS the binding's, so a minimum here would be
                // inventing one.
                min_binding_size: None,
            },
            count: None,
        });
    }
    let storage_layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(entrypoint),
            entries: &entries,
        });
    let uniform_layout = (!uniform.is_empty()).then(|| {
        gpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(entrypoint),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })
    });

    // Indexed by GROUP NUMBER, so the uniform block's `@group(1)` is position
    // one whatever `@group(0)` holds.
    let mut layouts: Vec<Option<&wgpu::BindGroupLayout>> = vec![Some(&storage_layout)];
    if let Some(block) = &uniform_layout {
        layouts.push(Some(block));
    }
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entrypoint),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });

    let source = kernels_wgpu::entrypoint_source(entrypoint, kernels_wgpu::Capability::Baseline)
        .unwrap_or_else(|why| panic!("`{entrypoint}` has no baseline source: {why}"));
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entrypoint),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entrypoint),
            layout: Some(&layout),
            module: &module,
            // `None` means "the module's only compute entry point", which
            // every expansion this crate produces has exactly one of.
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bound: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(at, buffer)| wgpu::BindGroupEntry {
            binding: at as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let storage_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entrypoint),
        layout: &storage_layout,
        entries: &bound,
    });
    let block = (!uniform.is_empty()).then(|| {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: uniform.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, uniform);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entrypoint),
            layout: uniform_layout.as_ref().expect("built beside the buffer"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        (buffer, group)
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(entrypoint),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entrypoint),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &storage_group, &[]);
        if let Some((_, group)) = &block {
            pass.set_bind_group(1, group, &[]);
        }
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT),
        })
        .expect("the device finished the dispatch");
}

/// D1. An adapter opens, and it is the numbers that are interesting.
///
/// Printed rather than asserted, because which adapter answered is what makes
/// every other result below mean something. The three assertions are the ones
/// that would make the rest of this file meaningless if they failed.
#[test]
fn an_adapter_opens_and_says_what_it_will_bind() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    println!("adapter      {}", gpu.info.name);
    println!("backend      {:?}", gpu.info.backend);
    println!("kind         {:?}", gpu.info.device_type);
    println!("driver       {} {}", gpu.info.driver, gpu.info.driver_info);
    println!(
        "storage      {} buffers per stage (WebGPU guarantees {})",
        gpu.limits.max_storage_buffers_per_shader_stage,
        kernels_wgpu::DOWNLEVEL_STORAGE_BUFFERS,
    );
    println!(
        "workgroups   {} per dimension, {} invocations",
        gpu.limits.max_compute_workgroups_per_dimension,
        gpu.limits.max_compute_invocations_per_workgroup,
    );
    println!(
        "f16          {}",
        gpu.features.contains(wgpu::Features::SHADER_F16)
    );
    println!(
        "subgroups    {}",
        gpu.features.contains(wgpu::Features::SUBGROUP)
    );
    let over = kernels_wgpu::over_downlevel_storage_limit();
    println!(
        "over floor   {} rows need more than {}: {:?}",
        over.len(),
        kernels_wgpu::DOWNLEVEL_STORAGE_BUFFERS,
        over.iter().map(|s| s.symbol).collect::<Vec<_>>(),
    );

    assert!(
        !over.is_empty(),
        "`sdpa_paged_decode` binds eleven storage buffers, so an empty answer \
         means `storage_count` stopped counting",
    );
    // The claim this harness is written on. Requesting
    // `Limits::downlevel_defaults()` instead would fail to create exactly the
    // attention pipelines below, on hardware that runs them.
    let widest = over
        .iter()
        .map(|sig| kernels_wgpu::storage_count(sig))
        .max()
        .expect("at least one row is over the floor");
    assert!(
        gpu.limits.max_storage_buffers_per_shader_stage >= widest,
        "this adapter offers {} storage buffers per stage and the table's \
         widest row binds {widest}. Those pipelines cannot be created here",
        gpu.limits.max_storage_buffers_per_shader_stage,
    );
}

/// D2. Every extent this file dispatches makes `div_ceil` and `/` differ.
///
/// `.wiki/new-driver/vulkan.md` §12 in one assertion. Three pointwise tests
/// there ran at n = 512 against a 256-wide workgroup and two GEMV tests at 16
/// rows against a kernel covering 8 — exact multiples, so the last partial
/// group never existed and the two expressions were the same one. Every pair
/// below is inexact, so replacing [`over`]'s body with `extent / workgroup`
/// undershoots a real grid and the tests that use it fail.
///
/// Needs no adapter: it is a claim about the SIZES, made where a reader looking
/// for them will find them all in one place.
#[test]
fn every_grid_here_is_ragged() {
    // (what, extent in lanes, the module's own `@workgroup_size` on that axis)
    let grids: &[(&str, u32, u32)] = &[
        ("residual_add / silu_mul / geglu / gptoss, words", 2990, 256),
        ("row_gather, words of a row", 22, 16),
        ("row_gather, gathered rows", 11, 16),
        ("split_qkv, channel pairs", 54, 256),
        ("geglu_tanh_strided, words of an output row", 23, 16),
        ("embed_gather, output words of one row", 192, 256),
        ("embed_gather_mb, rows", 13, 16),
        ("add_bias at an even width, words", 230, 256),
        ("add_bias at an odd width, words", 231, 256),
        ("affine_qmv_fast, output rows over a 8-row group", 13, 8),
        ("affine_qmv_routed, output rows over an 8-row block", 13, 8),
        ("affine_qmm_t, columns over a 16-wide tile", 47, 16),
        ("affine_qmm_t, columns over a 32-wide tile", 47, 32),
        ("affine_qmm_t, columns over a 64-wide tile", 47, 64),
        ("affine_qmm_t, rows over a 16-tall tile", 33, 16),
        ("affine_qmm_t, rows over a 32-tall tile", 33, 32),
        ("affine_qmm_t, rows over a 64-tall tile", 33, 64),
        ("kv_append, channel pairs", 6, 256),
        ("route_gather, columns", 13, 16),
        ("route_gather, permutation slots", 24, 16),
        ("combine_sorted, rows", 5, 16),
        ("shared_expert_combine, columns", 45, 16),
        ("shared_expert_combine, rows", 13, 16),
    ];
    for (what, extent, workgroup) in grids {
        assert_ne!(
            extent % workgroup,
            0,
            "{what}: {extent} lanes over a {workgroup}-wide workgroup divides \
             exactly, so `div_ceil` and `/` are the same expression here and \
             the tail this size was chosen for does not exist",
        );
        assert!(
            over(*extent, *workgroup) > extent / workgroup,
            "{what}: `over` must round UP",
        );
    }
}

/// D3. The comparison itself fails when a reference moves — on both claims.
///
/// Rule 4 for the CHECK, where every family test below runs rule 4 for its own
/// device answer. Needs no adapter.
#[test]
fn a_perturbed_reference_is_refused_by_the_same_check() {
    let want = spread(WIDTH as usize, 3);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    agrees(&want, &want, "itself").expect("a row agrees with itself");

    // The first claim: one element far out.
    let mut moved = want.clone();
    moved[17] += scale / 40.0;
    agrees(&moved, &want, "perturbed").expect_err(
        "a value three bf16 ulps of the row's scale away from its reference \
         must be refused, or the per-element bound is not measuring anything",
    );

    // And the scaling is by the ROW and not by a floor of one. A row whose
    // values are all small must still refuse a small error — which is exactly
    // what `max(|want|, 1.0)` would wave through.
    let small: Vec<f32> = want.iter().map(|v| v / 1000.0).collect();
    let mut nudged = small.clone();
    nudged[3] += scale / 40_000.0;
    agrees(&nudged, &small, "small and perturbed").expect_err(
        "a floor of 1.0 in the tolerance would accept this, which is the \
         defect `kernels-vulkan` found",
    );

    // The second claim: a SYSTEMATIC shift too small for any per-element bound
    // to see. Every value moved by a tenth of a percent — an `axis - 1` in a
    // norm — is inside the budget everywhere and must still be refused, on the
    // count.
    let drifted: Vec<f32> = want.iter().map(|v| v * 1.001).collect();
    let far = drifted
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        far < scale / 128.0,
        "this perturbation is supposed to be INSIDE the per-element budget, so \
         that it is the count that refuses it and not the bound",
    );
    agrees(&drifted, &want, "drifted").expect_err(
        "a systematic tenth of a percent is what an off-by-one in a reduction \
         width looks like, and no per-element bound scaled by a bf16 row can \
         see it",
    );

    // A NaN is refused whatever the budget: a kernel that produced one has not
    // produced a slightly wrong number.
    let mut bad = want.clone();
    bad[0] = f32::NAN;
    agrees(&bad, &want, "nan").expect_err("a NaN is not within any tolerance");

    // And the helper every family below calls does the same thing.
    refuses_a_perturbed_reference(&want, &want, "the shared control");
}

/// `RmsParams` as `norm/rms.wgsl` declares it: five words at binding 3.
///
/// A STORAGE block and not the `@group(1)` uniform, which is what the row means
/// by `params: Buf` — a struct is a struct, and moving it into the uniform
/// would be changing the kernel's ABI from the test.
fn rms_params(eps: f32, axis: u32, w_stride: u32, plus_one: u32, gain: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&eps.to_bits().to_le_bytes());
    out.extend_from_slice(&axis.to_le_bytes());
    out.extend_from_slice(&w_stride.to_le_bytes());
    out.extend_from_slice(&plus_one.to_le_bytes());
    out.extend_from_slice(&gain.to_bits().to_le_bytes());
    out
}

/// What `norm/rms.wgsl` computes for one row, from the bf16 the device saw.
fn rms_reference(
    x: &[f32],
    w: &[f32],
    w_stride: usize,
    plus_one: bool,
    gain: f32,
    eps: f32,
) -> Vec<f32> {
    let axis = x.len();
    let total: f32 = x.iter().map(|v| v * v).sum();
    let inv = (total / axis as f32 + eps).sqrt().recip();
    (0..axis)
        .map(|i| {
            let wv = w[w_stride * i];
            // Folded in FLOAT, before the bf16 round: MLX materialises
            // `add(weight, 1.0f)` in float and a parity walk has to agree.
            let g = gain * if plus_one { 1.0 + wv } else { wv };
            rounded(g * (x[i] * inv))
        })
        .collect()
}

/// D4. `rms_single_row_bfloat16` computes its closed form, both folds.
///
/// The clearest row in the table to check: `out = w * x / rms(x)` has an
/// unambiguous form, its params ride a STORAGE buffer rather than the uniform
/// block, and `LaunchRule::Rms` gives it one workgroup per ROW — so an
/// undershoot drops a whole row rather than a lane, and 13 of them is not a
/// multiple of anything.
///
/// Two dispatches, because the row has two arms nothing else reaches. The
/// second is gemma's: `plus_one` folds the weight as `1 + w` and `gain` scales
/// it, and `w_stride` is 2 rather than 1 — a shader that ignored the stride
/// would read the same 460 gains either way and pass the first dispatch.
#[test]
fn a_norm_is_its_closed_form_at_both_of_its_folds() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let name = "rms_single_row_bfloat16";
    let eps = 1e-6f32;
    let n = (ROWS * WIDTH) as usize;
    let (x, x_seen) = bf16s(gpu, &spread(n, 7));
    // Twice as long as the row, so `w_stride = 2` reads every other one.
    let (w, w_seen) = bf16s(gpu, &spread(2 * WIDTH as usize, 91));

    for (plus_one, gain, w_stride) in [(0u32, 1.0f32, 1usize), (1, 1.5, 2)] {
        let out = sentinelled(gpu, n / 2);
        let params = storage(
            gpu,
            &rms_params(eps, WIDTH, w_stride as u32, plus_one, gain),
        );
        // `LaunchRule::Rms` is one workgroup per axis, and the axis is the row.
        run(gpu, name, &[&x, &w, &out, &params], &[], [ROWS, 1, 1]);

        let got = unpack(&read(gpu, &out), n);
        for row in 0..ROWS as usize {
            let span = row * WIDTH as usize..(row + 1) * WIDTH as usize;
            let want = rms_reference(
                &x_seen[span.clone()],
                &w_seen,
                w_stride,
                plus_one != 0,
                gain,
                eps,
            );
            let what = format!("row {row} at plus_one={plus_one} stride={w_stride}");
            agrees(&got[span], &want, &what).expect("the norm agrees with its closed form");
            if row == ROWS as usize - 1 {
                // The last row, which is the one an undershot grid loses.
                refuses_a_perturbed_reference(&got[row * WIDTH as usize..], &want, &what);
            }
        }
    }
}

/// D5. `residual_add_bfloat16` is the sum of what it was given.
///
/// The other unambiguous form, and the one that says the bf16 PACKING is
/// right: every invocation writes a whole word of two values, so a half-index
/// off by one would shift the entire tensor by one element and still produce
/// finite numbers everywhere.
///
/// The grid is in WORDS. 13 x 460 elements is 2990 words over a 256-wide
/// workgroup, which is 11.68 — so plain division leaves the last 174 words
/// holding the sentinel they were born with and the dispatch still succeeds.
#[test]
fn a_residual_add_is_the_sum_of_what_it_was_given() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    let (x, x_seen) = bf16s(gpu, &spread(n, 13));
    let (r, r_seen) = bf16s(gpu, &spread(n, 29));
    let out = sentinelled(gpu, n / 2);

    run(
        gpu,
        "residual_add_bfloat16",
        &[&x, &r, &out],
        &[],
        [over(n as u32 / 2, 256), 1, 1],
    );

    let got = unpack(&read(gpu, &out), n);
    let want: Vec<f32> = x_seen
        .iter()
        .zip(&r_seen)
        .map(|(a, b)| rounded(a + b))
        .collect();
    agrees(&got, &want, "the whole tensor").expect("the add agrees");
    refuses_a_perturbed_reference(&got, &want, "the whole tensor");
}

/// D6. `row_gather_bfloat16` — the `Binding::Packed` row, on hardware.
///
/// `count` is `Ty::InPacked`: the row states it, and it takes NEITHER a
/// storage binding nor a uniform field, because it is the second FIELD of the
/// `RowGatherParams` struct binding 3 already carries. `layout/row_gather.wgsl`
/// declares no `@group(1)` at all and this test hands it none.
///
/// What makes the dispatch prove the rule rather than restate it: the index
/// list is FOURTEEN long and `count` is ELEVEN, and the output has room for
/// all fourteen. A `count` read from anywhere else — an unwritten uniform, a
/// stale word — bounds the gather by the wrong number, and the three trailing
/// rows stop holding their sentinel. Zero could not tell that story: it is
/// what the buffer was born with.
#[test]
fn a_packed_operand_bounds_the_gather_it_rides_in() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let name = "row_gather_bfloat16";
    let width = 44usize;
    let pitch = width / 2;
    let source_rows = 20usize;
    let count = 11usize;
    // Scrambled, so a gather that ignored the index list and copied a prefix
    // would be wrong everywhere rather than right by accident.
    let indices: Vec<u32> = vec![17, 3, 19, 0, 11, 8, 15, 2, 13, 6, 9, 1, 4, 5];
    assert_eq!(indices.len(), count + 3, "three entries past the count");

    let (input, input_seen) = bf16s(gpu, &spread(source_rows * width, 55));
    let rows = u32s(gpu, &indices);
    let out = sentinelled(gpu, indices.len() * pitch);
    // `[width, count]` — the statement states the width and the driver appends
    // the count, which is exactly the struct.
    let params = storage(
        gpu,
        &[(width as u32).to_le_bytes(), (count as u32).to_le_bytes()].concat(),
    );

    let sig = kernels_wgpu::sig(name).expect("the table covers the row gather");
    assert_eq!(
        kernels_wgpu::uniform_size(sig),
        0,
        "this row's only scalar is packed, so it declares no uniform block",
    );

    // `ElementwiseRows`, in WORDS on x because one invocation moves one word.
    run(
        gpu,
        name,
        &[&input, &out, &rows, &params],
        &[],
        [over(pitch as u32, 16), over(count as u32, 16), 1],
    );

    let back = read(gpu, &out);
    let got = unpack(&back, indices.len() * width);
    for (at, &row) in indices.iter().take(count).enumerate() {
        let from = row as usize * width;
        let want = &input_seen[from..from + width];
        let span = at * width..(at + 1) * width;
        let what = format!("gathered row {at} (source {row})");
        agrees(&got[span.clone()], want, &what).expect("the gather moved the row it was told to");
        if at == count - 1 {
            refuses_a_perturbed_reference(&got[span], want, &what);
        }
    }
    for at in count..indices.len() {
        for c in 0..width {
            assert_eq!(
                got[at * width + c].to_bits(),
                f32::from_bits((SENTINEL & 0xffff) << 16).to_bits(),
                "row {at} is past `count` = {count} and was written anyway, at \
                 column {c}. `count` is the packed field of `RowGatherParams`, \
                 so a gather that read it from a uniform block or from a stale \
                 word bounds itself by garbage",
            );
        }
    }
}

/// MLX's numerically stable sigmoid, as `mlp/gated.wgsl` spells it.
///
/// The exponent is taken of `-|x|` so it cannot overflow, and the branch puts
/// the reflection back. Not the same floating-point object as
/// `1/(1+exp(-x))`, which is why the two are written out separately here and
/// there: `shared_expert_combine` uses the plain one.
fn sigmoid_mlx(x: f32) -> f32 {
    let y = 1.0 / (1.0 + (-x.abs()).exp());
    if x < 0.0 { 1.0 - y } else { y }
}

/// `silu(g) * u`, with the two intermediate bf16 roundings Metal's has.
///
/// Metal rounds both intermediates to T, so this rounds through bf16 twice as
/// well: the sigmoid, and then the product with the gate. Doing the whole
/// thing in f32 and rounding once is a DIFFERENT number.
fn silu_mul_reference(g: f32, u: f32) -> f32 {
    let sg = rounded(sigmoid_mlx(g));
    let sil = rounded(g * sg);
    rounded(sil * u)
}

/// `gelu_tanh(g) * u` — the TANH approximation, not the erf one.
///
/// gemma's activation is specified as this closed form and the two differ by
/// more than rounding, which is exactly the substitution a `//#if` arm can
/// make silently.
fn geglu_tanh_reference(g: f32, u: f32) -> f32 {
    // `sqrt(2/pi)`. The shader spells it `0.7978845608028654`, which is the
    // same `f32` to the bit — WGSL's abstract float rounds to the same value
    // this literal does.
    let k = 0.797_884_6_f32;
    let inner = k * (g + 0.044_715 * g * g * g);
    rounded(0.5 * g * (1.0 + inner.tanh()) * u)
}

/// gpt-oss's clamped, alpha-scaled SwiGLU.
///
/// The gate is clamped ABOVE only; the linear branch is clamped both ways and
/// carries a `+1`. Both are gpt-oss's own, and dropping either produces a
/// model that runs and is wrong.
fn gptoss_reference(g: f32, u: f32, limit: f32, alpha: f32) -> f32 {
    let gc = g.min(limit);
    let uc = u.clamp(-limit, limit);
    let sig = 1.0 / (1.0 + (-alpha * gc).exp());
    rounded((gc * sig) * (uc + 1.0))
}

/// D7. The three gated activations each answer to their OWN closed form.
///
/// One file, one binding contract, five bodies chosen by `//#if` — which is
/// exactly the shape where a preprocessor slip compiles into the wrong
/// activation and produces plausible numbers everywhere. So each is checked
/// against its own form AND against its neighbours': the silu output must be
/// REFUSED by the geglu reference and the other way round, which is the claim
/// that the arms are distinct on this device and not merely distinct in the
/// text.
///
/// `gptoss_swiglu` is fed inputs past its limit in both directions and one
/// gate negative enough to overflow `exp`, where the result must be a finite
/// zero rather than a NaN out of `inf * 0`.
#[test]
fn each_gated_activation_answers_to_its_own_closed_form() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    let words = over(n as u32 / 2, 256);

    // The inputs every arm sees, with gpt-oss's edges written in. They are
    // harmless to the other two — silu and gelu are finite everywhere — and
    // having one input set is what makes the cross-checks below comparable.
    let mut gate = spread(n, 101);
    let mut up = spread(n, 103);
    let limit = 1.25f32;
    let alpha = 1.702f32;
    // Past the limit, above and below, on both operands.
    gate[5] = 3.5;
    gate[6] = -3.5;
    up[5] = 2.75;
    up[6] = -2.75;
    // Negative enough that `exp(-alpha * gc)` overflows to +inf, so the
    // sigmoid is exactly zero and the product must be a finite zero.
    gate[7] = -1024.0;
    up[7] = 1.5;
    let (gate_buf, gate_seen) = bf16s(gpu, &gate);
    let (up_buf, up_seen) = bf16s(gpu, &up);

    let silu = sentinelled(gpu, n / 2);
    run(
        gpu,
        "silu_mul_bfloat16",
        &[&gate_buf, &up_buf, &silu],
        &[],
        [words, 1, 1],
    );
    let silu_got = unpack(&read(gpu, &silu), n);

    // `GegluParams { unused: u32 }` — bound and never read, exactly as Metal's
    // is, because the row states `params: Buf` and the bind group layout a
    // shell builds from the row is the same on all three backends.
    let geglu = sentinelled(gpu, n / 2);
    let geglu_params = storage(gpu, &0u32.to_le_bytes());
    run(
        gpu,
        "geglu_tanh_bfloat16",
        &[&gate_buf, &up_buf, &geglu, &geglu_params],
        &[],
        [words, 1, 1],
    );
    let geglu_got = unpack(&read(gpu, &geglu), n);

    // `GptOssSwiGluParams { unused: u32, limit: f32, alpha: f32 }`.
    let gptoss = sentinelled(gpu, n / 2);
    let gptoss_params = storage(
        gpu,
        &[
            0u32.to_le_bytes(),
            limit.to_bits().to_le_bytes(),
            alpha.to_bits().to_le_bytes(),
        ]
        .concat(),
    );
    run(
        gpu,
        "gptoss_swiglu_bfloat16",
        &[&gate_buf, &up_buf, &gptoss, &gptoss_params],
        &[],
        [words, 1, 1],
    );
    let gptoss_got = unpack(&read(gpu, &gptoss), n);

    let silu_want: Vec<f32> = gate_seen
        .iter()
        .zip(&up_seen)
        .map(|(g, u)| silu_mul_reference(*g, *u))
        .collect();
    let geglu_want: Vec<f32> = gate_seen
        .iter()
        .zip(&up_seen)
        .map(|(g, u)| geglu_tanh_reference(*g, *u))
        .collect();
    let gptoss_want: Vec<f32> = gate_seen
        .iter()
        .zip(&up_seen)
        .map(|(g, u)| gptoss_reference(*g, *u, limit, alpha))
        .collect();

    agrees(&silu_got, &silu_want, "silu_mul").expect("silu_mul is silu");
    agrees(&geglu_got, &geglu_want, "geglu_tanh").expect("geglu_tanh is gelu");
    agrees(&gptoss_got, &gptoss_want, "gptoss_swiglu").expect("gptoss_swiglu is gpt-oss's");

    refuses_a_perturbed_reference(&silu_got, &silu_want, "silu_mul");
    refuses_a_perturbed_reference(&geglu_got, &geglu_want, "geglu_tanh");
    refuses_a_perturbed_reference(&gptoss_got, &gptoss_want, "gptoss_swiglu");

    // The neighbours. A `//#if` arm that selected the wrong activation would
    // pass its own check only if the two forms agreed, and they do not.
    agrees(&silu_got, &geglu_want, "silu against gelu").expect_err(
        "`silu_mul` and `geglu_tanh` are one file and one binding contract, so \
         the only thing separating them is the `//#if` arm. If the device's \
         silu output satisfies the GELU reference, the arms are not distinct",
    );
    agrees(&geglu_got, &silu_want, "gelu against silu").expect_err("and the other way round");
    agrees(&gptoss_got, &silu_want, "gptoss against silu").expect_err(
        "gpt-oss's SwiGLU bakes a clamp, an alpha and a `(up + 1)` that \
         nobody else's has",
    );

    // The overflow edge, called out by name because a finite zero and a NaN
    // are both "not the reference" and only one of them is a wrong number.
    assert!(
        gptoss_got[7].is_finite() && gptoss_got[7] == 0.0,
        "a gate of {} makes `exp(-alpha * gc)` overflow to +inf, so the \
         sigmoid is exactly zero and the product must be a finite zero. The \
         device returned {}",
        gate_seen[7],
        gptoss_got[7],
    );
}

/// One rotary pair turned by `theta` and scaled by `gain`.
fn rotate(x1: f32, x2: f32, theta: f32, gain: f32) -> (f32, f32) {
    let (s, c) = theta.sin_cos();
    (gain * (x1 * c - x2 * s), gain * (x1 * s + x2 * c))
}

/// The shape one `rope/neox.wgsl` launch runs over.
///
/// A struct rather than eight arguments, because most of them are counts and a
/// caller that transposed two would still compile. `pairs` is the ROTATED
/// half-count and `head_dim` is the whole head; when they disagree the rotary
/// is partial, which is the only case that separates `neox_prop_decode` from
/// `neox_decode`.
///
/// `freqs` selects the third spelling: a non-empty table is the `_freqs` rows,
/// where the angle comes from a BUFFER rather than from an exponent — llama-3's
/// piecewise interpolation and YaRN's are tables, not bases, so no exponent can
/// express them — and `mscale` is YaRN's attention-temperature correction,
/// which rides here because rotation is linear and scaling before or after is
/// the same thing.
#[derive(Clone, Copy)]
struct Neox<'a> {
    heads: usize,
    head_dim: usize,
    pairs: usize,
    scale: f32,
    base: f32,
    prop: bool,
    freqs: &'a [f32],
    mscale: f32,
}

/// What `rope/neox.wgsl` leaves in `x`, from the bf16 the device was given.
///
/// The whole tensor is returned rather than the rotated channels, because the
/// interesting claim is about what is NOT touched: a partial rotary leaves
/// channels past its range alone, and the invocation that owns the word they
/// share has to carry them through unchanged.
///
/// `prop` selects gemma's proportional exponent, which divides by the WHOLE
/// head while only `pairs` channels turn, and pairs across `head_dim / 2`
/// rather than across `pairs`. That is the ONLY thing separating
/// `neox_prop_decode` from `neox_decode`, and it separates them only when the
/// rotary is PARTIAL.
fn neox_reference(x: &[f32], positions: &[i32], shape: Neox<'_>) -> Vec<f32> {
    let Neox {
        heads,
        head_dim,
        pairs,
        scale,
        base,
        prop,
        freqs,
        mscale,
    } = shape;
    let mut out = x.to_vec();
    let dist = if prop { head_dim / 2 } else { pairs };
    let theta = |i: usize, pos: f32| -> f32 {
        if !freqs.is_empty() {
            return scale * pos * freqs[i];
        }
        let e = if prop {
            2.0 * i as f32 / head_dim as f32
        } else {
            i as f32 / pairs as f32
        };
        scale * pos * (-e * base).exp2()
    };
    for (row, position) in positions.iter().enumerate() {
        let pos = *position as f32;
        for h in 0..heads {
            // `t` is the invocation; it owns channels `2t` and `2t + 1` and
            // their partners, which is one whole word at each end.
            for t in 0..pairs.div_ceil(2) {
                let i0 = 2 * t;
                if i0 >= pairs {
                    break;
                }
                let at = row * heads * head_dim + h * head_dim + i0;
                let (a0, a1) = (x[at], x[at + 1]);
                let (b0, b1) = (x[at + dist], x[at + dist + 1]);
                let (r0x, r0y) = rotate(a0, b0, theta(i0, pos), mscale);
                // The odd tail of a partial rotary: channel `i0 + 1` is past
                // the rotated range, so it keeps its value — and is rewritten
                // with it, because the word is stored whole.
                let (r1x, r1y) = if i0 + 1 < pairs {
                    rotate(a1, b1, theta(i0 + 1, pos), mscale)
                } else {
                    (a1, b1)
                };
                out[at] = rounded(r0x);
                out[at + 1] = rounded(r1x);
                out[at + dist] = rounded(r0y);
                out[at + dist + 1] = rounded(r1y);
            }
        }
    }
    out
}

/// D8. `neox_decode` rotates in place, against values it has not yet changed.
///
/// RoPE writes its own INPUT, so a body that stored the first element of a
/// rotary pair before loading the second would rotate against a value it had
/// already changed. The reference is computed entirely from the ORIGINAL
/// tensor, so that body fails here.
///
/// `@workgroup_size(1)` and `LaunchRule::Rope` together make the grid EXACT:
/// the shader reads `num_workgroups.x` as the rotary pair count it strides
/// each pair's partner by and divides its exponent by, so a rounded-up grid
/// would not run a guarded lane — it would change the arithmetic every lane
/// does.
#[test]
fn a_rope_rotates_against_the_tensor_it_is_overwriting() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    // Multiples of four, which every checkpoint's rotary and head widths are,
    // and not multiples of each other or of any workgroup here.
    let head_dim = 12usize;
    let heads = 5usize;
    let pairs = head_dim / 2;
    let n = heads * head_dim;
    let scale = 0.75f32;
    let base = 13.287_712_f32; // log2(10000)
    let positions = [7i32];

    let (x, x_seen) = bf16s(gpu, &spread(n, 211));
    let position = i32s(gpu, &positions);
    let block = Block::of("neox_decode_bfloat16")
        .f32("scale", scale)
        .f32("base", base)
        .i32("head_dim", head_dim as i32)
        .done();
    run(
        gpu,
        "neox_decode_bfloat16",
        &[&x, &position],
        &block,
        [pairs as u32, heads as u32, 1],
    );

    let got = unpack(&read(gpu, &x), n);
    let want = neox_reference(
        &x_seen,
        &positions,
        Neox {
            heads,
            head_dim,
            pairs,
            scale,
            base,
            prop: false,
            freqs: &[],
            mscale: 1.0,
        },
    );
    agrees(&got, &want, "the rotated tensor").expect("neox_decode rotates");
    refuses_a_perturbed_reference(&got, &want, "the rotated tensor");
}

/// D9. `neox_prop_decode` is the PARTIAL rotary, which is the only place it
/// differs from `neox_decode`.
///
/// gemma's proportional slice: the exponent divides by the WHOLE head while
/// only `pairs` channels turn, and the partner is half a HEAD away rather than
/// half a rotary. At a full rotary the two bodies are arithmetically
/// IDENTICAL, so dispatching one is a test of nothing — this dispatches a
/// rotary of 6 over a head of 12, where the channels that move are [0, 3) and
/// [6, 9) rather than [0, 6).
///
/// The pair count is ODD on purpose. At `pairs = 3` the invocation that owns
/// channels 2 and 3 rotates only the first of them: channel 3 is past the
/// rotated range, and it is carried through unchanged in a word that is stored
/// whole. That branch has no other way to be reached.
#[test]
fn a_proportional_rope_turns_only_its_own_slice() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 12usize;
    let heads = 5usize;
    let rotary = 6usize;
    let pairs = rotary / 2;
    assert_eq!(
        pairs % 2,
        1,
        "the odd tail is the branch this dispatch is for"
    );
    let n = heads * head_dim;
    let scale = 0.75f32;
    let base = 13.287_712_f32;
    let positions = [7i32];

    let (x, x_seen) = bf16s(gpu, &spread(n, 211));
    let position = i32s(gpu, &positions);
    let block = Block::of("neox_prop_decode_bfloat16")
        .f32("scale", scale)
        .f32("base", base)
        .i32("head_dim", head_dim as i32)
        .done();
    run(
        gpu,
        "neox_prop_decode_bfloat16",
        &[&x, &position],
        &block,
        [pairs as u32, heads as u32, 1],
    );

    let got = unpack(&read(gpu, &x), n);
    let partial = Neox {
        heads,
        head_dim,
        pairs,
        scale,
        base,
        prop: true,
        freqs: &[],
        mscale: 1.0,
    };
    let want = neox_reference(&x_seen, &positions, partial);
    agrees(&got, &want, "the partially rotated tensor").expect("neox_prop_decode rotates");
    refuses_a_perturbed_reference(&got, &want, "the partially rotated tensor");

    // The two bodies must DISAGREE here, or this dispatch proves nothing about
    // which one ran. `neox_decode`'s answer for the same inputs pairs across 3
    // rather than across 6 and divides its exponent by 3 rather than by 6.
    let geometric = neox_reference(
        &x_seen,
        &positions,
        Neox {
            prop: false,
            ..partial
        },
    );
    agrees(&got, &geometric, "proportional against geometric").expect_err(
        "`neox_prop_decode` and `neox_decode` are arithmetically identical \
         unless the rotary is partial. If the device's proportional answer \
         satisfies the geometric reference, this dispatch was not partial and \
         the test is measuring nothing",
    );

    // And the channels past the rotary keep their values exactly: 4, 5, 10 and
    // 11 of every head are neither a pair's first element nor any pair's
    // partner.
    for h in 0..heads {
        for d in [4usize, 5, 10, 11] {
            let at = h * head_dim + d;
            assert_eq!(
                got[at].to_bits(),
                x_seen[at].to_bits(),
                "head {h} channel {d} is outside a rotary of {rotary} over a \
                 head of {head_dim} and must be untouched",
            );
        }
    }
}

/// D10. `affine_qmv_fast` at TWO quantization points.
///
/// Two because the pair is a COORDINATE and not a label: `gs_64/b_8` and
/// `gs_128/b_4` pack to IDENTICAL shapes — same word count, same scale plane
/// — so a module compiled for the wrong pair does not fail, it reads the
/// scales against the wrong weights and returns fluent nonsense.
/// `gs_64/b_4` and `gs_128/b_8` are the two ends of that square.
///
/// The output width is 13, which is neither a multiple of the 8 rows a
/// workgroup covers nor even. Odd is the interesting half: consecutive
/// outputs of one vector land in one 32-bit word while belonging to different
/// y-slots, and consecutive vectors land in one word while belonging to
/// different WORKGROUPS — which is the race `store_y`'s device-scoped
/// compare-exchange exists for, and no workgroup barrier could reach it.
///
/// # Why the reference reduces the way the kernel does
///
/// It splits K over 32 lanes, `PIE_QMV_VPT` values at a time, and folds the 32
/// partials in order. A flat left-to-right sum over 448 terms is a different
/// floating-point object by about `1e-4` relative here, which is inside the
/// per-element bf16 budget but not inside [`agrees`]'s rounding-noise count.
/// Mirroring the ORDER costs nothing in what is proven — every term, every
/// dequantisation and every address is still computed independently — and it
/// keeps the count meaning what it says.
#[test]
fn a_quantised_matvec_agrees_at_two_quantization_points() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    for (entrypoint, group, bits, k) in [
        (
            "affine_qmv_fast_bfloat16_gs_64_b_4",
            64usize,
            4u32,
            448usize,
        ),
        ("affine_qmv_fast_bfloat16_gs_128_b_8", 128, 8, 384),
    ] {
        let n_out = 13usize;
        let n_vec = 5usize;
        let plane = Affine::new(group, bits, n_out, k, 0x1234 ^ bits ^ group as u32);
        let vpt = plane.qmv_vpt();
        assert_ne!(k % (vpt * 32), 0, "the K sweep has a ragged last pass");

        let w = storage(gpu, &plane.words());
        let (scale_buf, _) = bf16s(gpu, &plane.scales);
        let (bias_buf, _) = bf16s(gpu, &plane.biases);
        let (x, x_seen) = bf16s(gpu, &spread(n_vec * k, 41 + bits));
        let y = sentinelled(gpu, (n_vec * n_out).div_ceil(2));

        let block = Block::of(entrypoint)
            .i32("in_vec_size", i32::try_from(k).expect("fits"))
            .i32("out_vec_size", i32::try_from(n_out).expect("fits"))
            .done();
        // `LaunchRule::Qmv`: the vector on x beside the 32 lanes of one
        // reduction, four outputs per y-slot and two slots per workgroup, so
        // `ceil(out / 8)` groups on y.
        run(
            gpu,
            entrypoint,
            &[&w, &scale_buf, &bias_buf, &x, &y],
            &block,
            [n_vec as u32, over(n_out as u32, 8), 1],
        );

        let mut want = Vec::with_capacity(n_vec * n_out);
        for vec_ in 0..n_vec {
            for row in 0..n_out {
                want.push(rounded(qmv_lane_sum(
                    &x_seen[vec_ * k..(vec_ + 1) * k],
                    |at| plane.value(row, at),
                    k,
                    vpt,
                )));
            }
        }

        let got = unpack(&read(gpu, &y), n_vec * n_out);
        agrees(&got, &want, entrypoint).expect("the quantised matvec agrees");
        refuses_a_perturbed_reference(&got, &want, entrypoint);
    }
}

/// D11. `kv_append_bfloat16` lands where the strides say and leaves the rest
/// alone.
///
/// A pure scatter, so the ADDRESS is the only thing it can get wrong. The
/// whole cache is checked against a SENTINEL, because zero cannot tell
/// "untouched" from "written zero" — a fresh `wgpu` buffer is already zero,
/// so a dispatch that ran nothing would satisfy a check written against it.
///
/// The two strides are DIFFERENT numbers and neither is derivable from the
/// other, so a body that swapped them writes somewhere real and wrong. They
/// are also the row's `Ty::Usize` operands, which cross as `vec2<u32>` at
/// offsets 8 and 16 rather than 4 and 12: a host packing the block by
/// concatenation writes both four bytes low and the shader reads two halves of
/// two different numbers, with nothing at runtime to report it.
///
/// The x extent is in channel PAIRS and is 6 over a 256-wide workgroup, so
/// `div_ceil` gives one group and plain division gives ZERO — a dispatch that
/// writes nothing and succeeds.
#[test]
fn a_kv_append_scatters_to_the_slot_its_strides_name() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 12usize;
    let kv_heads = 5usize;
    let max_seq = 7usize;
    let position = 5i32;
    let cache_len = kv_heads * max_seq * head_dim;
    // `[head][seq][channel]`: neither stride is the other's multiple by
    // accident, and both are read from the uniform block's low word.
    let k_head_stride = (max_seq * head_dim) as u64;
    let k_seq_stride = head_dim as u64;

    let (k_new, k_seen) = bf16s(gpu, &spread(kv_heads * head_dim, 401));
    let (v_new, v_seen) = bf16s(gpu, &spread(kv_heads * head_dim, 409));
    let k_cache = sentinelled(gpu, cache_len / 2);
    let v_cache = sentinelled(gpu, cache_len / 2);
    let pos = i32s(gpu, &[position]);

    let block = Block::of("kv_append_bfloat16")
        .i32("head_dim", head_dim as i32)
        .wide("k_head_stride", k_head_stride)
        .wide("k_seq_stride", k_seq_stride)
        .done();
    assert_eq!(block.len(), 32, "24 bytes of fields, rounded to 16 by WGSL");
    run(
        gpu,
        "kv_append_bfloat16",
        &[&k_new, &v_new, &k_cache, &v_cache, &pos],
        &block,
        [over(head_dim as u32 / 2, 256), kv_heads as u32, 1],
    );

    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    let mut want_k = vec![sentinel; cache_len];
    let mut want_v = vec![sentinel; cache_len];
    for h in 0..kv_heads {
        for d in 0..head_dim {
            let dst = h * k_head_stride as usize + position as usize * k_seq_stride as usize + d;
            want_k[dst] = k_seen[h * head_dim + d];
            want_v[dst] = v_seen[h * head_dim + d];
        }
    }

    let got_k = unpack(&read(gpu, &k_cache), cache_len);
    let got_v = unpack(&read(gpu, &v_cache), cache_len);
    // Exact, not within a tolerance: a scatter moves bits and rounds nothing.
    for (at, (g, w)) in got_k.iter().zip(&want_k).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "the key cache disagrees at element {at}: {g} where the strides \
             say {w}. head_stride {k_head_stride}, seq_stride {k_seq_stride}, \
             position {position}",
        );
    }
    for (at, (g, w)) in got_v.iter().zip(&want_v).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "the value cache disagrees at element {at}: {g} where {w} was \
             written. Both planes share the row's `k_*` strides",
        );
    }
    // And the check would notice: move one slot of the reference and it fails.
    let mut moved = want_k.clone();
    moved[k_head_stride as usize + position as usize * k_seq_stride as usize] = sentinel;
    assert_ne!(
        got_k
            .iter()
            .zip(&moved)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count(),
        0,
        "a reference that claims head 1's first channel was never written must \
         disagree with a device that wrote it",
    );
}

/// D12. `kv_append_paged_bfloat16` follows the page table and not the pool.
///
/// The pages are handed out in REVERSED order with NONZERO in-page offsets, so
/// a shader that ignored `w_page`/`w_off` and wrote the cache linearly lands in
/// the wrong page for every token but by construction inside the pool — which
/// is the failure a sentinel across the whole thing catches and a spot check
/// does not.
///
/// This row is also the launch ABI's other hard case: it names THIRTEEN buffer
/// operands, ten of which belong to a shared ring ABI this kernel does not
/// read, with three scalars interleaved between them. The write page and
/// offset land at bindings 10 and 11 rather than at the 13 and 14 a Metal
/// index would suggest, because the scalars left the buffer numbering.
/// `kernels-vulkan`'s copy of this shader carried 9 and 10 until an audit
/// compared its declared bindings against the row.
#[test]
fn a_paged_kv_append_writes_through_its_page_table() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 12usize;
    let n_kv_heads = 5usize;
    let page_size = 3usize;
    let pages = 5usize;
    let tokens = 4usize;
    let row_stride = n_kv_heads * head_dim;
    let pool = pages * page_size * row_stride;

    // Reversed, and none of them the token's own index; the offsets are
    // nonzero and not equal.
    let w_page = [4u32, 2, 3, 0];
    let w_off = [2u32, 1, 0, 2];

    let (k_new, k_seen) = bf16s(gpu, &spread(tokens * row_stride, 501));
    let (v_new, v_seen) = bf16s(gpu, &spread(tokens * row_stride, 509));
    let k_pages = sentinelled(gpu, pool / 2);
    let v_pages = sentinelled(gpu, pool / 2);
    let page = u32s(gpu, &w_page);
    let off = u32s(gpu, &w_off);
    // The ring operands this kernel does not declare. They are real entries of
    // the bind group layout — WGSL requires a shader's bindings to be a SUBSET
    // of the layout, not equal to it — so the group has to carry them.
    let rings: Vec<wgpu::Buffer> = (0..6).map(|_| storage(gpu, &[0u8; 4])).collect();

    let block = Block::of("kv_append_paged_bfloat16")
        .i32("head_dim", head_dim as i32)
        .i32("page_size", page_size as i32)
        .i32("n_kv_heads", n_kv_heads as i32)
        .done();
    run(
        gpu,
        "kv_append_paged_bfloat16",
        &[
            &k_new, &v_new, &k_pages, &v_pages, &rings[0], &rings[1], &rings[2], &rings[3],
            &rings[4], &rings[5], &page, &off, &rings[0],
        ],
        &block,
        [
            over(head_dim as u32 / 2, 256),
            n_kv_heads as u32,
            tokens as u32,
        ],
    );

    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    let mut want_k = vec![sentinel; pool];
    let mut want_v = vec![sentinel; pool];
    for i in 0..tokens {
        let slot = w_page[i] as usize * page_size + w_off[i] as usize;
        for h in 0..n_kv_heads {
            for d in 0..head_dim {
                let dst = slot * row_stride + h * head_dim + d;
                let src = i * row_stride + h * head_dim + d;
                want_k[dst] = k_seen[src];
                want_v[dst] = v_seen[src];
            }
        }
    }

    let got_k = unpack(&read(gpu, &k_pages), pool);
    let got_v = unpack(&read(gpu, &v_pages), pool);
    for (at, (g, w)) in got_k.iter().zip(&want_k).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "the key pool disagrees at element {at} (page {}, slot {}): {g} \
             where the page table says {w}",
            at / (page_size * row_stride),
            at / row_stride % page_size,
        );
    }
    for (at, (g, w)) in got_v.iter().zip(&want_v).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "the value pool disagrees at element {at}"
        );
    }
    // A shader that wrote the pool linearly — token `i` into slot `i` — would
    // land inside the pool and be wrong, which is what the reversed page list
    // makes a different answer rather than the same one.
    let mut linear = vec![sentinel; pool];
    for i in 0..tokens {
        for at in 0..row_stride {
            linear[i * row_stride + at] = k_seen[i * row_stride + at];
        }
    }
    assert_ne!(
        got_k
            .iter()
            .zip(&linear)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count(),
        0,
        "the device's answer must not also satisfy a linear write, or the page \
         table is not being read",
    );
}

/// A plain softmax attention over the keys a caller has already selected.
///
/// The reference for every SDPA body here: the maximum, the exponentials, the
/// weighted mean. `attn/sdpa_online.inc.wgsl` reaches the same numbers by a
/// running rescale — which is the point of computing this one the ordinary
/// way, since a recurrence that lost its history scale would agree with itself
/// and not with this.
///
/// `keys` is a list of `(score terms, value base)` pairs so the CALLER does the
/// addressing, which is what separates the two attention tests: the dense
/// decode's K and V have different strides and the paged one's come through a
/// page table.
fn softmax_attention(scores: &[f32], values: &[&[f32]], head_dim: usize) -> Vec<f32> {
    if scores.is_empty() {
        // A row with no keys at all: the denominator is exactly zero and the
        // numerator is too, so the kernel returns the zero rather than the NaN
        // a division would give.
        return vec![0.0; head_dim];
    }
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f32 = weights.iter().sum();
    (0..head_dim)
        .map(|d| {
            let acc: f32 = weights.iter().zip(values).map(|(w, v)| w * v[d]).sum();
            rounded(acc / z)
        })
        .collect()
}

/// D13. `sdpa_vector_decode_bfloat16_d_64` reads its values with the VALUE
/// stride.
///
/// K and V are given DIFFERENT layouts — `[head][seq][channel]` for the keys
/// and `[seq][head][channel]` for the values — which is the only way to catch
/// a body that reused the key stride for the value read. With one layout the
/// two are the same expression and the substitution is invisible.
///
/// The head width is 64 and the workgroup is THIRTY-TWO: bf16 crosses as
/// `array<u32>`, so a lane owns the channel PAIR and
/// `@workgroup_size(PIE_HEAD_DIM / 2)` is half what a Vulkan or Metal reading
/// gives. `driver-wgpu`'s `Rule::SdpaVector` doubles the module's width back
/// to the head width for exactly this reason; a grid built from the other
/// arithmetic would refuse every decode, and where it did not, mis-report the
/// head count through `num_workgroups.x`.
#[test]
fn a_dense_decode_attends_with_the_value_stride_and_not_the_key_stride() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 64usize;
    let q_heads = 6usize;
    let gqa = 3usize;
    let kv_heads = q_heads / gqa;
    let n = 11usize;
    let rows = 3usize;
    let scale = 0.125f32;

    // `[head][row][channel]`, which is the dense decode layout.
    let (queries, q_seen) = bf16s(gpu, &spread(q_heads * rows * head_dim, 601));
    // `[head][seq][channel]`.
    let k_head_stride = (n * head_dim) as u64;
    let k_seq_stride = head_dim as u64;
    let (keys, k_seen) = bf16s(gpu, &spread(kv_heads * n * head_dim, 607));
    // `[seq][head][channel]` — a different shape holding the same count, so a
    // body that read it with the key strides stays in bounds and is wrong.
    let v_seq_stride = (kv_heads * head_dim) as u64;
    let v_head_stride = head_dim as u64;
    let (values, v_seen) = bf16s(gpu, &spread(n * kv_heads * head_dim, 613));
    assert_ne!(
        (k_head_stride, k_seq_stride),
        (v_head_stride, v_seq_stride),
        "the two layouts must differ or this test cannot see the substitution",
    );
    let out = sentinelled(gpu, q_heads * rows * head_dim / 2);

    let block = Block::of("sdpa_vector_decode_bfloat16_d_64")
        .i32("gqa_factor", gqa as i32)
        .i32("n", n as i32)
        .wide("k_head_stride", k_head_stride)
        .wide("k_seq_stride", k_seq_stride)
        .wide("v_head_stride", v_head_stride)
        .wide("v_seq_stride", v_seq_stride)
        .f32("scale", scale)
        .done();
    assert_eq!(block.len(), 48, "two i32s, four vec2<u32>s and a float");
    run(
        gpu,
        "sdpa_vector_decode_bfloat16_d_64",
        &[&queries, &keys, &values, &out],
        &block,
        [q_heads as u32, rows as u32, 1],
    );

    let got = unpack(&read(gpu, &out), q_heads * rows * head_dim);
    for q_head in 0..q_heads {
        let kv_head = q_head / gqa;
        for row in 0..rows {
            let q_base = (q_head * rows + row) * head_dim;
            let scores: Vec<f32> = (0..n)
                .map(|i| {
                    let k_base = kv_head * k_head_stride as usize + i * k_seq_stride as usize;
                    // Scale per term, where both siblings put it: hoisting it
                    // out of the loop is a different rounding.
                    (0..head_dim)
                        .map(|d| scale * q_seen[q_base + d] * k_seen[k_base + d])
                        .sum()
                })
                .collect();
            let planes: Vec<&[f32]> = (0..n)
                .map(|i| {
                    let at = kv_head * v_head_stride as usize + i * v_seq_stride as usize;
                    &v_seen[at..at + head_dim]
                })
                .collect();
            let want = softmax_attention(&scores, &planes, head_dim);
            let what = format!("head {q_head} row {row}");
            agrees(&got[q_base..q_base + head_dim], &want, &what).expect("the decode attends");
            if q_head == q_heads - 1 && row == rows - 1 {
                refuses_a_perturbed_reference(&got[q_base..q_base + head_dim], &want, &what);
            }
        }
    }
}

/// D14. `sdpa_paged_decode_bfloat16_d_64` — the launch ABI's hardest row, on
/// hardware.
///
/// Its operands ALTERNATE between buffers and scalars, so the two runs
/// interleave: `Buffer(0..3)`, `Uniform(0)`, `Buffer(4..7)`, `Uniform(1..3)`,
/// `Buffer(8)`, `Uniform(4)`, `Buffer(9)`, `Uniform(5)`, `Buffer(10)`. A
/// backend numbering scalars alongside buffers would put `attention_mask` at
/// 12 rather than 8 and be wrong about everything after it, and nothing static
/// catches that — only a number does.
///
/// It also binds ELEVEN storage buffers against WebGPU's guaranteed floor of
/// eight, which is why [`open`] asks the ADAPTER for its limits.
///
/// Three things vary across the three rows of one dispatch, so one launch
/// covers three distinct key ranges: the window ends each row somewhere
/// different, the two requests own different physical pages, and the mask is
/// enabled for exactly one row. The physical pages are handed out REVERSED, so
/// a body reading the pool linearly fails.
#[test]
fn a_paged_decode_attends_through_its_page_table_and_its_mask() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 64usize;
    let q_heads = 6usize;
    let gqa = 3usize;
    let kv_heads = q_heads / gqa;
    let rows = 3usize;
    let page_size = 3usize;
    let pages = 7usize;
    let scale = 0.125f32;
    let window = 4i32;
    let mask_stride = 12u32;

    let positions = [6i32, 2, 9];
    let requests = [0i32, 1, 0];
    // Request 0 owns four pages and request 1 owns three, both REVERSED, so
    // logical page `p` is nowhere near physical page `p`.
    let indptr = [0u32, 4, 7];
    let indices = [6u32, 4, 2, 0, 5, 3, 1];

    // `[row][head][channel]`, which is NOT the dense decode's layout.
    let (queries, q_seen) = bf16s(gpu, &spread(rows * q_heads * head_dim, 701));
    let pool = pages * page_size * kv_heads * head_dim;
    let (k_pages, k_seen) = bf16s(gpu, &spread(pool, 709));
    let (v_pages, v_seen) = bf16s(gpu, &spread(pool, 719));
    let out = sentinelled(gpu, rows * q_heads * head_dim / 2);
    let position_ids = i32s(gpu, &positions);
    let req_of_token = i32s(gpu, &requests);
    let kv_page_indices = u32s(gpu, &indices);
    let kv_page_indptr = u32s(gpu, &indptr);

    // `U8s` in the row, and WGSL's smallest storage element is a `u32`, so a
    // byte is a shift. One row of `mask_stride` bytes each.
    let mut mask_bytes = vec![1u8; rows * mask_stride as usize];
    mask_bytes[mask_stride as usize + 1] = 0;
    let attention_mask = storage(gpu, &mask_bytes);
    // Enabled for row 1 alone: a mask every row obeyed would not tell an
    // ignored `attention_mask_enabled` from an honoured one.
    let attention_mask_enabled = storage(gpu, &[0u8, 1, 0, 0]);
    // Bound and unread: this entrypoint is the no-sink arm, and a row's bind
    // group layout does not change with a `//#if`.
    let (sinks, _) = bf16s(gpu, &spread(q_heads, 727));

    let block = Block::of("sdpa_paged_decode_bfloat16_d_64")
        .i32("gqa_factor", gqa as i32)
        .i32("page_size", page_size as i32)
        .i32("n_kv_heads", kv_heads as i32)
        .f32("scale", scale)
        .u32("attention_mask_stride", mask_stride)
        .i32("window", window)
        .done();
    run(
        gpu,
        "sdpa_paged_decode_bfloat16_d_64",
        &[
            &queries,
            &k_pages,
            &v_pages,
            &out,
            &position_ids,
            &req_of_token,
            &kv_page_indices,
            &kv_page_indptr,
            &attention_mask,
            &attention_mask_enabled,
            &sinks,
        ],
        &block,
        [q_heads as u32, rows as u32, 1],
    );

    let slot_of = |req: usize, kp: usize| -> usize {
        let phys = indices[indptr[req] as usize + kp / page_size] as usize;
        phys * page_size + kp % page_size
    };
    let got = unpack(&read(gpu, &out), rows * q_heads * head_dim);
    let mut ranges = Vec::new();
    for row in 0..rows {
        let req = requests[row] as usize;
        let q_pos = positions[row];
        let start = if window > 0 && q_pos >= window {
            q_pos - window + 1
        } else {
            0
        };
        let keeps: Vec<usize> = (start..=q_pos)
            .filter(|kp| {
                if row != 1 {
                    return true;
                }
                (*kp as u32) < mask_stride
                    && mask_bytes[row * mask_stride as usize + *kp as usize] != 0
            })
            .map(|kp| kp as usize)
            .collect();
        ranges.push(keeps.clone());
        for q_head in 0..q_heads {
            let kv_head = q_head / gqa;
            let q_base = (row * q_heads + q_head) * head_dim;
            let scores: Vec<f32> = keeps
                .iter()
                .map(|kp| {
                    let base = (slot_of(req, *kp) * kv_heads + kv_head) * head_dim;
                    (0..head_dim)
                        .map(|d| scale * q_seen[q_base + d] * k_seen[base + d])
                        .sum()
                })
                .collect();
            let planes: Vec<&[f32]> = keeps
                .iter()
                .map(|kp| {
                    let base = (slot_of(req, *kp) * kv_heads + kv_head) * head_dim;
                    &v_seen[base..base + head_dim]
                })
                .collect();
            let want = softmax_attention(&scores, &planes, head_dim);
            let what = format!("row {row} head {q_head} over keys {keeps:?}");
            agrees(&got[q_base..q_base + head_dim], &want, &what)
                .expect("the paged decode attends");
            if row == rows - 1 && q_head == q_heads - 1 {
                refuses_a_perturbed_reference(&got[q_base..q_base + head_dim], &want, &what);
            }
        }
    }
    assert_eq!(
        ranges,
        vec![vec![3, 4, 5, 6], vec![0, 2], vec![6, 7, 8, 9]],
        "one dispatch is supposed to cover three DIFFERENT key ranges — a \
         window that ends in a different place on each row, and a mask that is \
         enabled on exactly one of them. If they coincide, this test is one \
         case run three times",
    );
}

/// `out[r][c] += bias[c]`, from the bf16 the device was given.
fn add_bias_reference(value: &[f32], bias: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        for c in 0..width {
            out.push(rounded(value[r * width + c] + bias[c]));
        }
    }
    out
}

/// D15. `add_bias_bfloat16` at an EVEN width.
///
/// The Qwen-2 attention biases: one vector of `width`, broadcast down every
/// row, in place. `LaunchRule::RouteRows` dispatches `[width, rows, 1]` in
/// elements and this body wants `ceil(width / 2)` lanes on x, because one
/// invocation owns the PAIR of columns that share a 32-bit word — WGSL has no
/// sub-word atomic, so a read-modify-write of one half would race the lane
/// that owns the other.
///
/// At an even width every word of a row is that row's, so the two half-guards
/// coincide and the interesting case does not exist. That is what the next
/// test is for.
#[test]
fn a_bias_is_broadcast_down_every_row_at_an_even_width() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let width = WIDTH as usize;
    let rows = ROWS as usize;
    let n = rows * width;
    let (out, value_seen) = bf16s(gpu, &spread(n, 801));
    let (bias, bias_seen) = bf16s(gpu, &spread(width, 809));

    let block = Block::of("add_bias_bfloat16")
        .i32("width", i32::try_from(width).expect("fits"))
        .done();
    run(
        gpu,
        "add_bias_bfloat16",
        &[&out, &bias],
        &block,
        [over(width.div_ceil(2) as u32, 256), rows as u32, 1],
    );

    let got = unpack(&read(gpu, &out), n);
    let want = add_bias_reference(&value_seen, &bias_seen, rows, width);
    agrees(&got, &want, "the biased projection").expect("every row got its bias");
    refuses_a_perturbed_reference(&got, &want, "the biased projection");
}

/// D16. `add_bias_bfloat16` at an ODD width — the case its two half-guards
/// exist for.
///
/// **This is the shape the shader's own header calls out**, and it is the only
/// body in the tree where one invocation owns a PAIR of columns and the guard
/// differs per half:
///
/// > At an odd `width` the last word of a row holds that row's last column in
/// > its low half and the NEXT row's first column in its high half, so biasing
/// > both halves would apply the wrong column's bias to the next row.
/// > `hi < width` is what stops it, and it is the reason the two halves are
/// > guarded separately rather than together.
///
/// An even width cannot see it: there `lo` is even and `width` is even, so
/// `hi = lo + 1` is always inside the row and the early-out never fires.
///
/// 461 columns over 13 rows, so every ODD row starts in the upper half of a
/// word — `r * 461` is odd whenever `r` is — and every even row's last column
/// is the lower half of a word it shares with the row below.
#[test]
fn a_bias_is_broadcast_down_every_row_at_an_odd_width() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let width = 461usize;
    let rows = ROWS as usize;
    assert_eq!(width % 2, 1, "the whole point of this test");
    let n = rows * width;
    let (out, value_seen) = bf16s(gpu, &spread(n, 811));
    let (bias, bias_seen) = bf16s(gpu, &spread(width, 821));

    let block = Block::of("add_bias_bfloat16")
        .i32("width", i32::try_from(width).expect("fits"))
        .done();
    run(
        gpu,
        "add_bias_bfloat16",
        &[&out, &bias],
        &block,
        [over(width.div_ceil(2) as u32, 256), rows as u32, 1],
    );

    let got = unpack(&read(gpu, &out), n);
    let want = add_bias_reference(&value_seen, &bias_seen, rows, width);

    // Reported per row rather than as one failure, because WHICH rows
    // disagree is the diagnosis: an odd width puts every odd-numbered row on
    // an odd element boundary, and it is those rows whose word index and half
    // index come apart.
    let mut wrong = Vec::new();
    for r in 0..rows {
        let span = r * width..(r + 1) * width;
        if let Err(why) = agrees(&got[span.clone()], &want[span], &format!("row {r}")) {
            wrong.push(why);
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {rows} rows disagree at width {width}.\n\n{}\n\n\
         ── what a failure here means ───────────────────────────────────────\n\
         At an odd width, `row * width` is odd on every odd row, so a row's \
         span begins in the UPPER half of a word and its last column sits in \
         the LOWER half of a word the next row also occupies. If the rows that \
         fail are the odd-numbered ones at column 0 and the even-numbered ones \
         at column {}, the two halves of the straddling word have been swapped \
         or biased with each other's column — which is what the first draft of \
         `kernels/norm/add_bias.wgsl` did, and what the rewrite to one-word \
         ownership fixed: `biased` now takes each half's column from its own \
         ELEMENT index, so it does not matter which row the invocation \
         belongs to.\n\n\
         No tolerance here has been widened. A tolerance that had to be \
         widened is a defect report.",
        wrong.len(),
        wrong.join("\n"),
        width - 1,
    );
    refuses_a_perturbed_reference(&got, &want, "the biased projection");
}

/// `RouterParams` as `moe/params.inc.wgsl` declares it, in that order.
///
/// A STORAGE block and not the `@group(1)` uniform: these five rows state
/// `params: Buf`, a POINTER to where the numbers already are, because the
/// routing params are BUILT by the host plan rather than carried in the
/// statement. The field ORDER is the contract — a field moved here is a field
/// read at the wrong offset there, and everything after it shifts four bytes.
fn router_params(n_experts: u32, k: u32, softmax_over_all: u32, logits_pitch: u32) -> Vec<u8> {
    [n_experts, k, softmax_over_all, logits_pitch]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// `MoeRouteParams`: `n`, `n_experts`, `experts_per_token`, `tile_rows`,
/// `padded`, `width`, `x_pitch`.
///
/// `n` is the number of (row, slot) PAIRS and `padded` is the permutation's
/// length, `n` rounded up so every expert's span is a whole number of tiles.
/// They are different numbers and the sort reads both.
fn route_params(
    n: u32,
    n_experts: u32,
    k: u32,
    tile_rows: u32,
    padded: u32,
    width: u32,
    x_pitch: u32,
) -> Vec<u8> {
    [n, n_experts, k, tile_rows, padded, width, x_pitch]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// D17. `router_topk_bfloat16` picks the right experts and weights them.
///
/// The only kernel in the table whose output is partly INTEGER, so the ids are
/// compared EXACTLY and only the weights get a tolerance. A top-k that agreed
/// to within a tolerance would be a top-k that had chosen a different expert.
///
/// `k` is THREE, which is odd, and that is not decoration: lane 0 of a row's
/// workgroup writes that row's `k` weights, so at an odd `k` row `r`'s last
/// weight and row `r+1`'s first weight land in ONE 32-bit word written by two
/// DIFFERENT workgroups. Every checkpoint the tree has seen routes to an even
/// `k` and would never show it; the `atomicAnd`/`atomicOr` pair the shader
/// uses instead of a read-modify-write is what makes it come out right, and
/// this is the dispatch that asks.
///
/// `logits_pitch` is 9 against 7 experts, so the router is reading a SLICE of
/// a wider activation. A body that took the pitch to be the expert count reads
/// two of the next row's logits as its own.
#[test]
fn a_router_chooses_exactly_and_weights_approximately() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n_experts = 7usize;
    let k = 3usize;
    let pitch = 9usize;
    let rows = 5usize;
    assert_eq!(k % 2, 1, "the shared-word store is what an odd k reaches");

    let (logits, logits_seen) = bf16s(gpu, &spread(rows * pitch, 901));
    let expert_ids = i32s(gpu, &vec![-7i32; rows * k]);
    let expert_weights = sentinelled(gpu, (rows * k).div_ceil(2));
    let params = storage(
        gpu,
        &router_params(n_experts as u32, k as u32, 1, pitch as u32),
    );
    // Bound and unread by the unscaled arm; a row's bind group does not change
    // with a `//#if`.
    let (per_expert_scale, _) = bf16s(gpu, &positives(n_experts, 907));

    // `LaunchRule::RouterLane`: one workgroup per ROW, on y. The row axis was
    // once missing next door, and at `grid.y = 1` a mixture prefill routed row
    // 0 and left every other row's ids whatever the previous layer wrote.
    run(
        gpu,
        "router_topk_bfloat16",
        &[
            &logits,
            &expert_ids,
            &expert_weights,
            &params,
            &per_expert_scale,
        ],
        &[],
        [1, rows as u32, 1],
    );

    let got_ids = unpack_i32(&read(gpu, &expert_ids), rows * k);
    let got_weights = unpack(&read(gpu, &expert_weights), rows * k);
    let mut want_ids = Vec::with_capacity(rows * k);
    let mut want_weights = Vec::with_capacity(rows * k);
    for row in 0..rows {
        let mine = &logits_seen[row * pitch..row * pitch + n_experts];
        let mut sorted: Vec<usize> = (0..n_experts).collect();
        // The shader scans ascending and takes a strict `>`, so the FIRST
        // maximum wins a tie. Ties are refused outright below rather than
        // relied on.
        sorted.sort_by(|a, b| mine[*b].total_cmp(&mine[*a]).then(a.cmp(b)));
        for w in mine.windows(2) {
            assert_ne!(
                w[0], w[1],
                "row {row} has two equal logits after bf16 rounding, so its \
                 top-{k} is not a function of the values alone and an exact \
                 id comparison would be checking the tie-break",
            );
        }
        // `softmax_over_all = 1`: the denominator is every expert's, taken
        // BEFORE the selection eats the array.
        let m = mine.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = mine.iter().map(|v| (v - m).exp()).sum();
        for &e in sorted.iter().take(k) {
            want_ids.push(i32::try_from(e).expect("fits"));
            want_weights.push(rounded((mine[e] - m).exp() / z));
        }
    }

    assert_eq!(
        got_ids, want_ids,
        "the chosen experts are integers and are compared as integers: a \
         top-{k} that agreed to within a tolerance would be a top-{k} that had \
         chosen a different expert",
    );
    agrees(&got_weights, &want_weights, "the router weights").expect("the weights agree");
    refuses_a_perturbed_reference(&got_weights, &want_weights, "the router weights");
    // And the id comparison would notice, which the weights' control cannot
    // say for it.
    let mut moved = want_ids.clone();
    moved[k] = (moved[k] + 1) % i32::try_from(n_experts).expect("fits");
    assert_ne!(got_ids, moved, "an exact comparison must refuse a moved id");
}

/// D18. `route_sort` groups the rows by expert, and the STRUCTURE is what is
/// checkable.
///
/// The placement inside an expert's span is decided by the order the
/// `atomicAdd`s happen to land in — the sort is stable in neither sibling and
/// the 256-lane stripe here only changes which unstable order comes out — so
/// there is no reference for the permutation itself. What IS guaranteed, and
/// what this checks, is that every routed pair gets exactly one slot inside
/// its own expert's span, that `inv` is the exact inverse of `perm`, that the
/// unfilled slots keep their `-1`, and that `tile_expert` names the owner of
/// every tile. A race cannot hide behind that: a lost or doubled slot shows up
/// as a pair that never comes back.
///
/// `tile_rows` is THREE, which is not a power of two and does not divide the
/// 15 routed pairs, so every expert's span is rounded up by a different
/// amount and `padded` (24) is neither `n` nor a multiple of the 16-wide
/// workgroup `route_gather` reads it with.
#[test]
fn a_route_sort_gives_every_pair_one_slot_in_its_own_span() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n_experts = 7usize;
    let k = 3usize;
    let tile = 3usize;
    // Five rows of three choices, with a deliberately uneven histogram: expert
    // 2 takes five of the fifteen and experts 1 and 4 take one each.
    let ids: Vec<i32> = vec![2, 5, 0, 2, 3, 6, 1, 2, 5, 0, 4, 2, 6, 2, 3];
    let n = ids.len();

    let mut counts = vec![0usize; n_experts];
    for e in &ids {
        counts[usize::try_from(*e).expect("routed")] += 1;
    }
    let mut bases = Vec::with_capacity(n_experts);
    let mut at = 0usize;
    for c in &counts {
        bases.push(at);
        at += c.div_ceil(tile) * tile;
    }
    let padded = at;
    assert_eq!(
        padded, 24,
        "the span arithmetic is part of what is asserted"
    );
    let tiles = padded / tile;

    let expert_ids = i32s(gpu, &ids);
    let perm = i32s(gpu, &vec![-7i32; padded]);
    let row_expert = i32s(gpu, &vec![-7i32; padded]);
    let tile_expert = i32s(gpu, &vec![-7i32; tiles]);
    let inv = i32s(gpu, &vec![-7i32; n]);
    let params = storage(
        gpu,
        &route_params(
            n as u32,
            n_experts as u32,
            k as u32,
            tile as u32,
            padded as u32,
            13,
            17,
        ),
    );

    // `LaunchRule::RouterSort`: ONE workgroup for the whole routing, whatever
    // the row count. N copies of this would each clear and rewrite the
    // permutation the others are reading.
    run(
        gpu,
        "route_sort",
        &[&expert_ids, &perm, &row_expert, &tile_expert, &params, &inv],
        &[],
        [1, 1, 1],
    );

    let got_perm = unpack_i32(&read(gpu, &perm), padded);
    let got_row_expert = unpack_i32(&read(gpu, &row_expert), padded);
    let got_tile = unpack_i32(&read(gpu, &tile_expert), tiles);
    let got_inv = unpack_i32(&read(gpu, &inv), n);

    let mut seen = vec![false; n];
    for (slot, &pair) in got_perm.iter().enumerate() {
        if pair < 0 {
            continue;
        }
        let pair = usize::try_from(pair).expect("a slot holds a pair index");
        assert!(
            pair < n,
            "slot {slot} holds pair {pair}, past the {n} routed"
        );
        assert!(
            !std::mem::replace(&mut seen[pair], true),
            "pair {pair} was placed twice, at slot {slot} and earlier",
        );
        let e = usize::try_from(ids[pair]).expect("routed");
        assert!(
            slot >= bases[e] && slot < bases[e] + counts[e].div_ceil(tile) * tile,
            "pair {pair} chose expert {e}, whose span is [{}, {}), and landed \
             at slot {slot}. A routed GEMM reads one expert per row tile, so a \
             row outside its expert's span is multiplied by a neighbour's \
             weights",
            bases[e],
            bases[e] + counts[e].div_ceil(tile) * tile,
        );
        assert_eq!(
            got_row_expert[slot], ids[pair],
            "slot {slot} holds pair {pair}, which chose expert {}, and \
             `row_expert` says {}",
            ids[pair], got_row_expert[slot],
        );
        assert_eq!(
            got_inv[pair],
            i32::try_from(slot).expect("fits"),
            "`inv` must be the exact inverse of `perm`: pair {pair} is at slot \
             {slot} and `inv` says {}. `combine_sorted` reads back through it, \
             so an inverse that disagrees blends a different row's expert \
             output in",
            got_inv[pair],
        );
    }
    assert!(
        seen.iter().all(|s| *s),
        "{} of {n} routed pairs never got a slot",
        seen.iter().filter(|s| !**s).count(),
    );

    // The tiles. An expert whose rows half-fill a tile must still own the
    // whole tile, or its tail rows would be multiplied by its neighbour's
    // weights.
    let mut want_tiles = vec![-1i32; tiles];
    for (e, count) in counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        let span = bases[e] / tile..(bases[e] + count.div_ceil(tile) * tile) / tile;
        for owner in &mut want_tiles[span] {
            *owner = i32::try_from(e).expect("fits");
        }
    }
    assert_eq!(got_tile, want_tiles, "the tile ownership map");

    // The permutation's tail. `padded` is 24 against 15 routed pairs, so nine
    // slots are padding and must keep the `-1` the sort stamps, not the `-7`
    // the buffer was born with and not a stale index.
    let placed = got_perm.iter().filter(|p| **p >= 0).count();
    assert_eq!(placed, n, "exactly the routed pairs are placed");
    for (slot, &pair) in got_perm.iter().enumerate() {
        assert!(
            pair == -1 || pair >= 0,
            "slot {slot} holds {pair}, which is neither a pair nor the `-1` an \
             unfilled slot is stamped with — so the sort did not clear it and \
             the gather would read the previous routing's rows",
        );
    }

    // The control. There is no reference for the ORDER, so what has to be
    // shown is that the span check above can fail: an unsorted body — one that
    // left pair `i` at slot `i` — must violate it. If it did not, this
    // histogram would be one the check cannot distinguish from a sort.
    let unsorted_fits = (0..n).all(|pair| {
        let e = usize::try_from(ids[pair]).expect("routed");
        pair >= bases[e] && pair < bases[e] + counts[e].div_ceil(tile) * tile
    });
    assert!(
        !unsorted_fits,
        "a body that placed pair `i` at slot `i` would satisfy the span check \
         for this histogram, so the check is not measuring the grouping. Pick \
         an `ids` whose experts are not already in order",
    );
}

/// D19. `route_gather` compacts by the permutation, and pads with zero.
///
/// The permutation is handed in rather than taken from `route_sort`, because
/// that kernel's order is a race and a test that chained them would be
/// checking one unstable answer against another. This one has HOLES — three
/// slots that no pair claims — and its entries are out of order, so a gather
/// that copied a prefix or skipped the padding is wrong in a way the sentinel
/// can see.
///
/// `width` is 13 and ODD, which is the interesting half: row `r` ends at
/// element `13r + 12` and row `r+1` starts at `13r + 13`, so consecutive rows
/// share a 32-bit word and their two invocations are in different workgroups.
/// WGSL has no sub-word atomic, so the store is an `atomicAnd` that clears the
/// writer's half followed by an `atomicOr` that sets it — whatever order the
/// four operations of a shared word land in, each half ends at its own
/// writer's value.
#[test]
fn a_route_gather_compacts_by_its_permutation() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let width = 13usize;
    let x_pitch = 17usize;
    let k = 3usize;
    let padded = 24usize;
    let source_rows = 5usize;
    // Out of order, and with three slots nobody claims.
    let perm_values: Vec<i32> = vec![
        11, 2, 14, -1, 7, 0, 13, 5, -1, 9, 4, 1, 12, 8, -1, 3, 10, 6, -1, -1, -1, -1, -1, -1,
    ];
    assert_eq!(perm_values.len(), padded);

    let (x, x_seen) = bf16s(gpu, &spread(source_rows * x_pitch, 1001));
    let out = sentinelled(gpu, (padded * width).div_ceil(2));
    let perm = i32s(gpu, &perm_values);
    let params = storage(
        gpu,
        &route_params(
            (source_rows * k) as u32,
            7,
            k as u32,
            3,
            padded as u32,
            width as u32,
            x_pitch as u32,
        ),
    );

    run(
        gpu,
        "route_gather",
        &[&x, &out, &perm, &params],
        &[],
        [over(width as u32, 16), over(padded as u32, 16), 1],
    );

    let got = unpack(&read(gpu, &out), padded * width);
    let mut want = Vec::with_capacity(padded * width);
    for slot in 0..padded {
        for c in 0..width {
            // `perm` holds a (row, slot) PAIR index and `x` is indexed by ROW,
            // which is what the division by `k` is. The two are not the same
            // number and the gather is the only place that has to know it.
            want.push(if perm_values[slot] < 0 {
                0.0
            } else {
                x_seen[usize::try_from(perm_values[slot]).expect("routed") / k * x_pitch + c]
            });
        }
    }
    for (at, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "slot {} column {} is {g} where the permutation says {w}. A gather \
             moves BITS: the GLSL sibling assigns with no widening, and a \
             round trip through f32 would be a rounding step on a value that \
             is only being copied",
            at / width,
            at % width,
        );
    }
    // A padding slot is written with ZERO rather than skipped: the row exists
    // in the gathered tensor, the GEMM will read it, and whatever the buffer
    // held is not zero.
    for slot in 0..padded {
        if perm_values[slot] >= 0 {
            continue;
        }
        for c in 0..width {
            assert_eq!(
                got[slot * width + c],
                0.0,
                "slot {slot} is padding and holds {} at column {c}; it was \
                 born holding the sentinel, so an unwritten pad row is \
                 distinguishable from a zeroed one",
                got[slot * width + c],
            );
        }
    }

    // The control. A body that ignored `perm` and compacted the first rows in
    // order — the shape a gather degenerates into when the permutation is read
    // as an identity — must produce a DIFFERENT tensor, or the permutation is
    // not what is being checked.
    let identity: Vec<f32> = (0..padded * width)
        .map(|at| {
            let row = at / width / k;
            if row < source_rows {
                x_seen[row * x_pitch + at % width]
            } else {
                0.0
            }
        })
        .collect();
    assert!(
        got.iter()
            .zip(&identity)
            .any(|(g, w)| g.to_bits() != w.to_bits()),
        "the device's answer also satisfies an identity gather, so this \
         permutation is not distinguishing anything",
    );
}

/// D20. `combine_sorted` reads the sorted tensor back through `inv`.
///
/// The inverse permutation is the whole content: `y` is indexed by SORTED
/// slot and the output by ROW, and a body that read `y` at `row * k + e`
/// instead would read the sorted tensor at an unsorted index and blend a
/// different row's expert output in — plausible numbers, wrong rows.
///
/// One slot is UNROUTED (`inv` of `-1`), which must contribute nothing rather
/// than reading slot zero. And `out_pitch` is 19 against a width of 13, so the
/// six columns past each row's data must keep their sentinel: this body writes
/// one bf16 per invocation and the partner half of a boundary word is a row's
/// PADDING, which no invocation writes.
#[test]
fn a_combine_reads_back_through_the_inverse_permutation() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let width = 13usize;
    let out_pitch = 19usize;
    let k = 3usize;
    let rows = 5usize;
    let padded = 24usize;
    // The inverse of `a_route_gather_compacts_by_its_permutation`'s
    // permutation, with pair 6 left unrouted.
    let inv_values: Vec<i32> = vec![5, 11, 1, 15, 10, 7, -1, 4, 13, 9, 16, 0, 12, 6, 2];
    assert_eq!(inv_values.len(), rows * k);

    let (y, y_seen) = bf16s(gpu, &spread(padded * width, 1101));
    let (weights, weight_seen) = bf16s(gpu, &positives(rows * k, 1103));
    let out = sentinelled(gpu, (rows * out_pitch).div_ceil(2));
    let inv = i32s(gpu, &inv_values);
    // `ExpertCombineParams { width, experts_per_token, out_pitch }`.
    let params = storage(
        gpu,
        &[width as u32, k as u32, out_pitch as u32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );

    run(
        gpu,
        "combine_sorted",
        &[&y, &weights, &out, &params, &inv],
        &[],
        [over(width as u32, 16), over(rows as u32, 16), 1],
    );

    let got = unpack(&read(gpu, &out), rows * out_pitch);
    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    for row in 0..rows {
        let want: Vec<f32> = (0..width)
            .map(|c| {
                let mut acc = 0.0f32;
                for e in 0..k {
                    let slot = row * k + e;
                    if inv_values[slot] >= 0 {
                        let at = usize::try_from(inv_values[slot]).expect("placed") * width + c;
                        acc += weight_seen[slot] * y_seen[at];
                    }
                }
                rounded(acc)
            })
            .collect();
        let base = row * out_pitch;
        let what = format!("row {row}");
        agrees(&got[base..base + width], &want, &what).expect("the combine agrees");
        if row == rows - 1 {
            refuses_a_perturbed_reference(&got[base..base + width], &want, &what);
        }
        // The pitch's padding. Column 12 of a row is the low half of a word
        // whose high half is column 13, which nobody writes — so the
        // `atomicAnd`/`atomicOr` pair has to leave it exactly alone.
        for c in width..out_pitch {
            assert_eq!(
                got[base + c].to_bits(),
                sentinel.to_bits(),
                "row {row} column {c} is past the width and was written \
                 anyway. `out_pitch` is {out_pitch} against a width of \
                 {width}, so a store that took the whole word would clobber \
                 the padding it shares with column {}",
                width - 1,
            );
        }
    }
}

/// D21. `shared_expert_combine`'s gate is ONE NUMBER PER ROW.
///
/// `kernels-vulkan` records a real defect here: a port read `gate[r * width]`
/// where Metal reads `gate[r]`, collapsing a row's DATA BASE and its GATE
/// INDEX into one variable. The test that should have caught it allocated a
/// gate buffer `rows * width` long, which made the wrong index REPRESENTABLE
/// and the wrong answer plausible.
///
/// So the gate here is exactly `rows` elements. At `r * width` with 13 rows of
/// 45 that index leaves the buffer immediately, and a WGSL read past a bound
/// storage range is CLAMPED rather than trapped — so the wrong body returns
/// row 12's gate for every row and this test says which rows disagree.
///
/// The width is ODD, so consecutive rows share a word and each half has a
/// different writer in a different workgroup.
#[test]
fn a_shared_expert_gate_is_one_number_per_row() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let width = 45usize;
    let rows = ROWS as usize;
    assert_eq!(width % 2, 1, "an odd width puts two rows in one word");
    let n = rows * width;

    let (routed, routed_seen) = bf16s(gpu, &spread(n, 1201));
    let (shared, shared_seen) = bf16s(gpu, &spread(n, 1203));
    // `rows`, and not `rows * width`. This is the allocation that makes the
    // Vulkan port's index unrepresentable rather than merely wrong.
    let (gate, gate_seen) = bf16s(gpu, &spread(rows, 1207));
    assert_eq!(gate_seen.len(), rows);
    let out = sentinelled(gpu, n.div_ceil(2));

    let block = Block::of("shared_expert_combine")
        .u32("width", u32::try_from(width).expect("fits"))
        .done();
    run(
        gpu,
        "shared_expert_combine",
        &[&routed, &shared, &gate, &out],
        &block,
        [over(width as u32, 16), over(rows as u32, 16), 1],
    );

    let got = unpack(&read(gpu, &out), n);
    for (row, gate_value) in gate_seen.iter().enumerate() {
        // The plain logistic, not MLX's stable reflection: this is what the
        // shader writes and the two are different floating-point objects.
        let g = 1.0 / (1.0 + (-gate_value).exp());
        let want: Vec<f32> = (0..width)
            .map(|c| {
                let at = row * width + c;
                rounded(routed_seen[at] + g * shared_seen[at])
            })
            .collect();
        let base = row * width;
        let what = format!("row {row} at gate {gate_value}");
        agrees(&got[base..base + width], &want, &what)
            .expect("every row was blended with its OWN gate");
        if row == rows - 1 {
            refuses_a_perturbed_reference(&got[base..base + width], &want, &what);
        }
    }

    // And the gates differ enough that reading the wrong one is visible. If
    // every row's gate were nearly equal, the allocation above would be the
    // only thing standing between this test and the Vulkan port's defect.
    let (lo, hi) = gate_seen
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    assert!(
        hi - lo > 1.0,
        "the {rows} gates span only [{lo}, {hi}], so a body that read one row's \
         gate for every row would agree with this reference",
    );

    // The last word's high half is the tensor's own padding — 13 x 45 is odd,
    // so 585 elements occupy 293 words — and `LaunchRule::RouteRows` rounds
    // the row axis up to a whole 16-high workgroup. Row 13 of that grid
    // therefore exists, and its column 0 lands at element 585, which is inside
    // the BINDING and outside the tensor.
    //
    // That is the case the shader's `arrayLength(&out_)` guard is written for
    // and it is why the guard is on the buffer rather than on a row count the
    // block does not carry: the overshot row writes the pad half of the pad
    // word — `routed` and `shared` are both zero there, so the value is a zero
    // — and rows 14 and 15, and every column past 0 of row 13, are refused.
    // Asserted rather than waved at, because a guard that moved would show up
    // here as a nonzero pad or as a readback shorter than the allocation.
    let whole = read(gpu, &out);
    assert_eq!(
        whole.len(),
        n.div_ceil(2) * 4,
        "the write must not have grown the allocation",
    );
    let tail = unpack(&whole, n + 1);
    assert_eq!(
        tail[n], 0.0,
        "element {n} is the pad half of the pad word. The grid's 16-high \
         workgroup gives a row 13 that this tensor does not have, and its \
         column 0 is the one lane of it the `arrayLength` guard admits — over \
         a zero pad, so a zero. It holds {} instead, which means either the \
         guard or the padding moved",
        tail[n],
    );
}

/// D22. The uniform block refuses a field the row does not state.
///
/// [`Block`]'s teeth, checked rather than asserted in prose. Needs no adapter:
/// it is a claim about the ABI helper every dispatch above builds its scalars
/// with. A harness that let a typo through would write four bytes at offset
/// zero and leave the real field holding whatever the buffer had.
#[test]
#[should_panic(expected = "states no scalar called")]
fn a_uniform_field_the_row_does_not_state_is_refused() {
    // `kv_append` states `head_dim`, `k_head_stride` and `k_seq_stride`. It
    // does NOT state `v_head_stride` — the value plane rides the key strides,
    // which is the whole reason `sdpa_vector_decode` states four and this row
    // states two.
    let _ = Block::of("kv_append_bfloat16").wide("v_head_stride", 1);
}

/// D23. And it refuses a block with a field nobody wrote.
///
/// The failure this prevents is the quiet one: a shell that filled two of a
/// row's three scalars leaves the third reading whatever the uniform buffer
/// held, and a uniform buffer is bytes — nothing at runtime knows what they
/// were supposed to mean.
#[test]
#[should_panic(expected = "was never written")]
fn a_uniform_block_missing_a_field_is_refused() {
    let _ = Block::of("kv_append_bfloat16")
        .i32("head_dim", 12)
        .wide("k_head_stride", 84)
        .done();
}

/// D24. And it refuses a four-byte write into a 64-bit field.
///
/// `Ty::Usize` crosses as `vec2<u32>`, low word first, which gives it an
/// eight-byte alignment as well as an eight-byte width. A shell that wrote
/// four bytes would leave the high word stale — and on the offsets side, one
/// that packed the block by concatenation would put `k_head_stride` at 4
/// rather than at 8 and the shader would read two halves of two different
/// numbers.
#[test]
#[should_panic(expected = "bytes wide and this writes 4")]
fn a_narrow_write_into_a_split_field_is_refused() {
    let _ = Block::of("kv_append_bfloat16").u32("k_head_stride", 84);
}

/// An affine-quantised weight plane, and the dense values it decodes to.
///
/// `value = scale * code + bias`, per group of `group` elements, at `bits` bits
/// per code. The pair is a COORDINATE and not a label — g64/b8 and g128/b4 pack
/// to identical shapes — so a module compiled for the wrong one reads the
/// scales against the wrong weights and returns fluent nonsense rather than
/// failing.
///
/// Packing is MLX's, because the checkpoints are: codes little-endian within a
/// 32-bit word, lowest code in the lowest bits, `k * bits / 32` words to a row;
/// `scales` and `biases` one bf16 each per group, laid out `[rows, k / group]`.
///
/// The scale and bias planes are stored ALREADY ROUNDED to bf16, so
/// [`Affine::value`] is what the device decodes and not what the host meant —
/// rule 1, at the one place in this file where the input is a codec rather than
/// a tensor.
struct Affine {
    group: usize,
    bits: u32,
    rows: usize,
    k: usize,
    codes: Vec<u32>,
    packed: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

impl Affine {
    fn new(group: usize, bits: u32, rows: usize, k: usize, seed: u32) -> Self {
        let codes_per_word = 32 / bits as usize;
        assert_eq!(
            k % group,
            0,
            "a row is a whole number of groups or the checkpoint would not pack",
        );
        assert_eq!(k % codes_per_word, 0, "and a whole number of words");

        let mut state = seed | 1;
        let mut codes = vec![0u32; rows * k];
        for c in &mut codes {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *c = (state >> 13) & ((1u32 << bits) - 1);
        }
        let words_per_row = k / codes_per_word;
        let mut packed = vec![0u32; rows * words_per_row];
        for row in 0..rows {
            for at in 0..k {
                let word = row * words_per_row + at / codes_per_word;
                packed[word] |= codes[row * k + at] << ((at % codes_per_word) as u32 * bits);
            }
        }

        // Sized so `scale * max_code` is order one at either width, which keeps
        // the dot products comparable between quantization points instead of
        // 16x apart.
        let narrow = 4.0 / (1u32 << bits) as f32;
        let groups = rows * (k / group);
        let scales: Vec<f32> = positives(groups, seed ^ 0x5a5a)
            .iter()
            .map(|v| rounded(v * narrow))
            .collect();
        let biases: Vec<f32> = spread(groups, seed ^ 0x3c3c)
            .iter()
            .map(|v| rounded(v * 0.1))
            .collect();

        Self {
            group,
            bits,
            rows,
            k,
            codes,
            packed,
            scales,
            biases,
        }
    }

    /// Element `at` of row `row`, dequantised exactly as the device does.
    ///
    /// The bounds are asserted rather than left to the slice: a reference that
    /// walked past the plane would panic here with the row it wanted, where an
    /// unchecked one would read the next row's codes and disagree with the
    /// device by a plausible amount.
    fn value(&self, row: usize, at: usize) -> f32 {
        assert!(
            row < self.rows && at < self.k,
            "({row}, {at}) is off the plane"
        );
        let g = row * (self.k / self.group) + at / self.group;
        self.scales[g] * self.codes[row * self.k + at] as f32 + self.biases[g]
    }

    /// `PIE_QMV_VPT` — the values one lane of `quant/qmv.wgsl` pulls per pass.
    ///
    /// Two words' worth, so 16 codes at four bits and 8 at eight. Asked of the
    /// plane rather than recomputed at each call site, because it is a fact
    /// about the packing and the reduction has to agree with it.
    fn qmv_vpt(&self) -> usize {
        (32 / self.bits as usize) * 2
    }

    fn words(&self) -> Vec<u8> {
        self.packed.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// The sixteen E2M1 values, by code, as `common/mxfp4.inc.wgsl` lists them.
///
/// NOT linear in the code — they step by 0.5 up to 2 and then by 1, 2, 2 —
/// which is the whole reason MXFP4 cannot borrow the affine dot and why its
/// row has no separate bias plane for a per-group bias to live in.
const MXFP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Elements per E8M0 block. Fixed by the FORMAT, which is why the routed MXFP4
/// rows are compiled at `_gs_32` alone.
const MXFP4_BLOCK: usize = 32;

/// An MXFP4 weight plane: 4-bit E2M1 codes and one E8M0 byte per 32 of them.
///
/// Both planes cross as `array<u32>` four BYTES to a word, lowest byte first —
/// WGSL has no `u8` any more than it has a `u16` — so two readers of one plane
/// that disagree about byte order produce a transposed weight, which is wrong
/// by a plausible amount and looks like a bad fine-tune rather than a bug.
struct Mxfp4 {
    rows: usize,
    k: usize,
    codes: Vec<u32>,
    exponents: Vec<u32>,
}

impl Mxfp4 {
    fn new(rows: usize, k: usize, seed: u32) -> Self {
        assert_eq!(k % MXFP4_BLOCK, 0, "a row is a whole number of blocks");
        let mut state = seed | 1;
        let mut codes = vec![0u32; rows * k];
        for c in &mut codes {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *c = (state >> 13) & 0xf;
        }
        let mut exponents = vec![0u32; rows * (k / MXFP4_BLOCK)];
        for e in &mut exponents {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // 125..=128, so the block scale is 0.25, 0.5, 1 or 2. Never 0xff,
            // which is E8M0's NaN encoding and would poison a whole row.
            *e = 125 + ((state >> 11) & 3);
        }
        Self {
            rows,
            k,
            codes,
            exponents,
        }
    }

    fn value(&self, row: usize, at: usize) -> f32 {
        let code = self.codes[row * self.k + at] as usize;
        let e = self.exponents[row * (self.k / MXFP4_BLOCK) + at / MXFP4_BLOCK];
        MXFP4_LUT[code] * (e as f32 - 127.0).exp2()
    }

    /// Two codes per byte, low nibble first. Swapping them transposes every
    /// adjacent pair of weights.
    fn words(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.rows * self.k / 2];
        for row in 0..self.rows {
            for at in 0..self.k {
                let bi = row * (self.k / 2) + at / 2;
                let nibble = (self.codes[row * self.k + at] as u8) << ((at % 2) as u8 * 4);
                bytes[bi] |= nibble;
            }
        }
        bytes
    }

    fn scale_bytes(&self) -> Vec<u8> {
        self.exponents.iter().map(|e| *e as u8).collect()
    }
}

/// D25. `affine_qmm_t` over EVERY tile shape and EVERY quantization point.
///
/// 54 entrypoints: group `{32, 64, 128}` x bits `{4, 8}` x tile
/// `{16, 32, 64}^2`, all against ONE dense reference per quantization point
/// built from the codes. This is the table's largest kernel, and
/// `.wiki/new-driver/vulkan.md` records that on the Vulkan side exactly one of
/// its 54 had ever produced a number, at one quantization point — "the
/// thinnest possible evidence for the axis that matters most".
///
/// # Why `M = 33, N = 47`
///
/// So that EVERY tiling has a ragged edge on both axes. At a tile-aligned
/// shape the column overhang does not exist, and the column overhang is where
/// the sweep next door found a corrupting defect:
///
/// > `write_out` had no column bound check. A tile is `BN` wide whatever `N`
/// > is, so lanes at `col >= n` still wrote — and because the output is
/// > row-major, they did not write past the end of the buffer, they wrote over
/// > `(row + 1, col - n)`, a live element of the next row, with a value
/// > computed from weights that were themselves out of range. Every row after
/// > the first began with a zero.
///
/// This tree was ported with that guard, and `n` is in the uniform block so it
/// is exact. This is what says it is there: removing it fails all 54.
///
/// # The ROW overhang is a contract, not a defect
///
/// No entrypoint's uniform block carries `m`. The kernel cannot know where the
/// rows stop, so the only contract consistent with the block is that the
/// caller allocates a whole number of `BM` rows; the extra rows are written
/// with garbage and ignored. [`gemm_shape`] honours that — `x`, `y` and the
/// reference are all sized to `ceil(M / BM) * BM` — and the last real row's
/// last element survives an overhang row writing the other half of its word
/// because the store is a compare-exchange.
///
/// # Why K has no ragged edge and cannot be given one
///
/// `PIE_BK` is 16 and `K` must be a whole number of `PIE_GROUP` groups for the
/// codec's two quotients to be exact. Every group size here (32, 64, 128) is a
/// multiple of 16, so a legal `K` is always a whole number of `PIE_BK` blocks
/// and the `kn` tail in the staging loop is unreachable from a valid launch.
/// Said rather than left as an apparent gap.
#[test]
fn a_tiled_gemm_agrees_over_every_tile_shape_and_quantization_point() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let m = 33usize;
    let n = 47usize;
    // A multiple of 128, so one K serves all three group sizes.
    let k = 384usize;
    let widest_bm = 64usize;
    let padded = m.div_ceil(widest_bm) * widest_bm;

    let (x, x_seen) = bf16s(gpu, &spread(padded * k, 1301));
    let mut swept = 0usize;
    // Collected rather than panicked at the first, because the useful number
    // when this fails is HOW MANY of the 54 do: a defect in the shared
    // epilogue takes all of them, one in a single tile shape takes one.
    let mut wrong: Vec<String> = Vec::new();
    for (group, bits) in [
        (32usize, 4u32),
        (32, 8),
        (64, 4),
        (64, 8),
        (128, 4),
        (128, 8),
    ] {
        let plane = Affine::new(group, bits, n, k, 0x9e37 ^ bits ^ group as u32);
        let w = storage(gpu, &plane.words());
        let (scales, _) = bf16s(gpu, &plane.scales);
        let (biases, _) = bf16s(gpu, &plane.biases);

        // ONE dense reference, shared by all nine tilings. The device sums K
        // ascending in a single f32 accumulator — `PIE_BK` blocks in order,
        // and inside a block `kk` ascending — so a plain sequential sum is the
        // same floating-point object and not merely the same value.
        let mut want = Vec::with_capacity(m * n);
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for at in 0..k {
                    acc += x_seen[row * k + at] * plane.value(col, at);
                }
                want.push(rounded(acc));
            }
        }

        for bm in [16usize, 32, 64] {
            for bn in [16usize, 32, 64] {
                let entrypoint =
                    format!("affine_qmm_t_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}");
                // The row overhang contract: a whole number of BM rows.
                let rows_allocated = m.div_ceil(bm) * bm;
                assert!(rows_allocated <= padded, "x covers the widest overhang");
                let y = sentinelled(gpu, (rows_allocated * n).div_ceil(2));
                let block = Block::of(&entrypoint)
                    .i32("k", i32::try_from(k).expect("fits"))
                    .i32("n", i32::try_from(n).expect("fits"))
                    .done();
                // `LaunchRule::Qmm`: one workgroup per (column tile, row tile).
                run(
                    gpu,
                    &entrypoint,
                    &[&w, &scales, &biases, &x, &y],
                    &block,
                    [over(n as u32, bn as u32), over(m as u32, bm as u32), 1],
                );

                let back = read(gpu, &y);
                let got = unpack(&back, rows_allocated * n);
                // Only the first `m` rows are the caller's; the rest are the
                // overhang the contract allows.
                let real = &got[..m * n];
                if let Err(why) = agrees(real, &want, &entrypoint) {
                    wrong.push(why);
                } else if bm == 64 && bn == 64 {
                    refuses_a_perturbed_reference(real, &want, &entrypoint);
                }
                swept += 1;
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {swept} tile shapes disagree with the dense reference.\n\n{}\n\n\
         At M={m}, N={n} EVERY tiling has a ragged edge on both axes, which is \
         the whole reason for those two numbers. A disagreement that starts at \
         column 0 of row 1 is the COLUMN overhang: a tile is `BN` wide whatever \
         N is, and the output is row-major, so a lane at `col >= n` does not \
         write past the buffer — it writes over (row + 1, col - n), a live \
         element of the next row, with a value computed from weights that were \
         themselves out of range. `n` is in the uniform block, so the guard in \
         `write_out` is exact and costs nothing.",
        wrong.len(),
        wrong.join("\n"),
    );
    assert_eq!(
        swept, 54,
        "the row is 3 group sizes x 2 widths x 9 tile shapes and this swept \
         {swept} of them",
    );
}

/// D26. `affine_qmm_t_residual` folds its residual at the binding the ROW
/// names, which is 5 and not 7.
///
/// `.wiki/new-driver/vulkan.md` §3 is the record of what the other reading
/// costs: Metal numbers scalars in the same run as buffers, so its `residual`
/// is buffer 7 where the row puts it at 5, and across 54 entrypoints of this
/// file's Vulkan sibling that 7 was a descriptor the shell never wrote.
///
/// The residual values are drawn from `[10, 20)` — nowhere near `x`'s
/// `[-2, 2)` and nowhere near zero — so a wrong binding is not a small error.
/// A zero-filled residual would be indistinguishable from no residual at all,
/// and one filled from `x` would be indistinguishable from binding 3.
///
/// Two points rather than one, because the epilogue and the codec are
/// independent: the fold is `bf16(bf16(sum) + residual)`, rounded BEFORE the
/// add so that the fused variant is bit-identical to the two-kernel path
/// (project, then `residual_add`) it replaces.
#[test]
fn a_tiled_gemm_folds_its_residual_at_the_binding_the_row_names() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let m = 33usize;
    let n = 47usize;
    let k = 384usize;

    for (group, bits, bm, bn) in [(64usize, 4u32, 32usize, 32usize), (128, 8, 16, 64)] {
        let entrypoint =
            format!("affine_qmm_t_residual_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}");
        let rows_allocated = m.div_ceil(bm) * bm;
        let plane = Affine::new(group, bits, n, k, 0x51ed ^ bits ^ group as u32);
        let w = storage(gpu, &plane.words());
        let (scales, _) = bf16s(gpu, &plane.scales);
        let (biases, _) = bf16s(gpu, &plane.biases);
        let (x, x_seen) = bf16s(gpu, &spread(rows_allocated * k, 1401));
        // In [10, 20): a magnitude nothing else in the launch has.
        let residual_values: Vec<f32> = spread(rows_allocated * n, 1409)
            .iter()
            .map(|v| 15.0 + v * 2.5)
            .collect();
        let (residual, residual_seen) = bf16s(gpu, &residual_values);
        let y = sentinelled(gpu, (rows_allocated * n).div_ceil(2));

        let block = Block::of(&entrypoint)
            .i32("k", i32::try_from(k).expect("fits"))
            .i32("n", i32::try_from(n).expect("fits"))
            .done();
        run(
            gpu,
            &entrypoint,
            &[&w, &scales, &biases, &x, &y, &residual],
            &block,
            [over(n as u32, bn as u32), over(m as u32, bm as u32), 1],
        );

        let mut plain = Vec::with_capacity(m * n);
        let mut want = Vec::with_capacity(m * n);
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for at in 0..k {
                    acc += x_seen[row * k + at] * plane.value(col, at);
                }
                plain.push(rounded(acc));
                want.push(rounded(rounded(acc) + residual_seen[row * n + col]));
            }
        }

        let got = unpack(&read(gpu, &y), m * n);
        agrees(&got, &want, &entrypoint).expect("the fused projection agrees");
        refuses_a_perturbed_reference(&got, &want, &entrypoint);
        // And the residual is actually IN the answer: the un-fused reference
        // must be refused, or binding 5 was read as zeros.
        agrees(&got, &plain, &format!("{entrypoint} without its residual")).expect_err(
            "the residual is drawn from [10, 20), so an answer that satisfies \
             the plain matmul reference is an answer that never read binding 5",
        );
    }
}

/// The 32-lane split `quant/qmv.wgsl` reduces K with, mirrored.
///
/// Each of 32 lanes owns `vpt` consecutive values every `vpt * 32`, and lane 0
/// then folds the 32 partials in order. A flat left-to-right sum over 448
/// terms is a different floating-point object by about `1e-4` relative — inside
/// the per-element bf16 budget, but not inside [`agrees`]'s rounding-noise
/// count. Mirroring the ORDER costs nothing in what is proven: every term,
/// every dequantisation and every address is still computed independently.
fn qmv_lane_sum(x_row: &[f32], value: impl Fn(usize) -> f32, k: usize, vpt: usize) -> f32 {
    let mut partials = [0.0f32; 32];
    for (lid, partial) in partials.iter_mut().enumerate() {
        let mut k0 = lid * vpt;
        while k0 < k {
            for i in 0..vpt {
                if k0 + i < k {
                    *partial += x_row[k0 + i] * value(k0 + i);
                }
            }
            k0 += vpt * 32;
        }
    }
    partials.iter().sum()
}

/// D27. `affine_qmv_fast_residual` — the same operand, one binding lower down
/// a shorter row.
///
/// Five buffers then two scalars then the residual, so `residual` is
/// `@binding(5)` here exactly as it is in the GEMM. The output width is 13,
/// which is odd: consecutive vectors' outputs then land in one 32-bit word
/// while belonging to different WORKGROUPS, which is the race `store_y`'s
/// device-scoped compare-exchange exists for — and the residual read at
/// `vec * out_vec_size + row` shares that odd pitch.
#[test]
fn a_quantised_matvec_folds_its_residual_at_binding_five() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    for (group, bits, k) in [(64usize, 4u32, 448usize), (128, 8, 384)] {
        let entrypoint = format!("affine_qmv_fast_residual_bfloat16_gs_{group}_b_{bits}");
        let n_out = 13usize;
        let n_vec = 5usize;
        let vpt = (32 / bits as usize) * 2;

        let plane = Affine::new(group, bits, n_out, k, 0x2b1d ^ bits ^ group as u32);
        let w = storage(gpu, &plane.words());
        let (scales, _) = bf16s(gpu, &plane.scales);
        let (biases, _) = bf16s(gpu, &plane.biases);
        let (x, x_seen) = bf16s(gpu, &spread(n_vec * k, 1501));
        let residual_values: Vec<f32> = spread(n_vec * n_out, 1509)
            .iter()
            .map(|v| 15.0 + v * 2.5)
            .collect();
        let (residual, residual_seen) = bf16s(gpu, &residual_values);
        let y = sentinelled(gpu, (n_vec * n_out).div_ceil(2));

        let block = Block::of(&entrypoint)
            .i32("in_vec_size", i32::try_from(k).expect("fits"))
            .i32("out_vec_size", i32::try_from(n_out).expect("fits"))
            .done();
        run(
            gpu,
            &entrypoint,
            &[&w, &scales, &biases, &x, &y, &residual],
            &block,
            [n_vec as u32, over(n_out as u32, 8), 1],
        );

        let mut plain = Vec::with_capacity(n_vec * n_out);
        let mut want = Vec::with_capacity(n_vec * n_out);
        for vec_ in 0..n_vec {
            for row in 0..n_out {
                let sum = qmv_lane_sum(
                    &x_seen[vec_ * k..(vec_ + 1) * k],
                    |at| plane.value(row, at),
                    k,
                    vpt,
                );
                plain.push(rounded(sum));
                want.push(rounded(rounded(sum) + residual_seen[vec_ * n_out + row]));
            }
        }

        let got = unpack(&read(gpu, &y), n_vec * n_out);
        agrees(&got, &want, &entrypoint).expect("the fused matvec agrees");
        refuses_a_perturbed_reference(&got, &want, &entrypoint);
        agrees(&got, &plain, &format!("{entrypoint} without its residual")).expect_err(
            "an answer that satisfies the plain matvec reference never read \
             binding 5",
        );
    }
}

/// The 32-lane halving tree `moe/qmv_routed.wgsl` reduces with, mirrored.
///
/// Lane-strided over K — 32 lanes, `in_vec_size` elements, each lane owning
/// every 32nd — then a halving tree over the 32 partials. Same reasoning as
/// [`qmv_lane_sum`]: the ORDER is mirrored so that [`agrees`]'s rounding-noise
/// count keeps meaning what it says, and nothing else is.
fn routed_lane_sum(x: &[f32], x_base: usize, value: impl Fn(usize) -> f32, k: usize) -> f32 {
    let mut partial = [0.0f32; 32];
    for (lane, slot) in partial.iter_mut().enumerate() {
        let mut at = lane;
        while at < k {
            *slot += x[x_base + at] * value(at);
            at += 32;
        }
    }
    let mut step = 16usize;
    while step > 0 {
        for lane in 0..step {
            partial[lane] += partial[lane + step];
        }
        step >>= 1;
    }
    partial[0]
}

/// The routing one dispatch of `moe/qmv_routed.wgsl` runs over.
///
/// Written down once because three entrypoints share it and the interesting
/// content is that NONE of the three strides is derivable from another: the
/// weights are indexed by EXPERT, the input by a pair of strides neither of
/// which is `k`, and the output by flat SLOT.
struct Routed {
    rows: usize,
    slots_per_row: usize,
    n_experts: usize,
    out_vec_size: usize,
    in_vec_size: usize,
    x_slot_stride: usize,
    x_row_stride: usize,
    expert_ids: Vec<i32>,
}

impl Routed {
    fn new() -> Self {
        let rows = 5usize;
        let slots_per_row = 3usize;
        let in_vec_size = 192usize;
        // Neither is `k`, and the row stride is not `slots * slot_stride`
        // either: a routed decode packs `slots_per_row` copies of the
        // activation contiguously INSIDE a row, so walking slots is the small
        // stride and walking rows the large one, and a body that used one for
        // the other stays in bounds and reads a different token.
        let x_slot_stride = in_vec_size + 8;
        let x_row_stride = slots_per_row * x_slot_stride + 16;
        Self {
            rows,
            slots_per_row,
            n_experts: 4,
            out_vec_size: 13,
            in_vec_size,
            x_slot_stride,
            x_row_stride,
            // Varies per row, repeats experts across rows, and slot 1 of row 2
            // is UNROUTED. An unrouted slot must leave its output untouched,
            // which is checkable only against a sentinel.
            expert_ids: vec![2, 0, 3, 1, 1, 2, 0, -1, 3, 3, 2, 0, 1, 3, 2],
        }
    }

    fn slots(&self) -> usize {
        self.rows * self.slots_per_row
    }

    fn x_len(&self) -> usize {
        self.rows * self.x_row_stride
    }

    fn y_len(&self) -> usize {
        self.slots() * self.out_vec_size
    }

    fn weight_rows(&self) -> usize {
        self.n_experts * self.out_vec_size
    }

    fn block(&self, entrypoint: &str) -> Vec<u8> {
        Block::of(entrypoint)
            .i32(
                "in_vec_size",
                i32::try_from(self.in_vec_size).expect("fits"),
            )
            .i32(
                "out_vec_size",
                i32::try_from(self.out_vec_size).expect("fits"),
            )
            .i32(
                "x_slot_stride",
                i32::try_from(self.x_slot_stride).expect("fits"),
            )
            .i32(
                "x_row_stride",
                i32::try_from(self.x_row_stride).expect("fits"),
            )
            .i32(
                "slots_per_row",
                i32::try_from(self.slots_per_row).expect("fits"),
            )
            .done()
    }

    /// The grid `moe/qmv_routed.wgsl` needs: `(row, output block, slot)`, with
    /// EIGHT output rows to a block because the module is `@workgroup_size(32, 8)`
    /// and each y lane owns one output row.
    ///
    /// # This is not the grid `driver-wgpu`'s `Rule::RoutedQmv` computes
    ///
    /// That rule answers `[local.x * rows, width.div_ceil(4), slots]` LANES —
    /// Metal's extent, where a threadgroup is `[32, 2, 1]` and each y thread
    /// owns FOUR output rows. Divided by this module's 8-deep workgroup it
    /// gives `ceil(ceil(n / 4) / 8)` groups, which is `ceil(n / 32)` and not
    /// `ceil(n / 8)`. At `n = 13` that is one workgroup where six are needed;
    /// at a real 2048-wide expert projection it is 64 where 256 are needed.
    ///
    /// The assertion below is the arithmetic, stated here rather than argued
    /// about, because it is the same class of finding as `Rule::SdpaVector`
    /// being half the head width: an extent ported from a backend whose
    /// rows-per-lane differs. An undershot grid writes nothing, the gap reads
    /// back as the zeros the buffer was born with, and the dispatch completes.
    fn grid(&self) -> [u32; 3] {
        let n = self.out_vec_size as u32;
        assert!(
            over(over(n, 4), 8) < over(n, 8),
            "these two readings of the output extent are supposed to DISAGREE \
             at n = {n}; if they now agree, pick an n where they do not or the \
             note above has stopped being a finding",
        );
        [self.rows as u32, over(n, 8), self.slots_per_row as u32]
    }
}

/// D28. `affine_qmv_routed` and `affine_qmv_routed_bias` — three indices at
/// once, and one binding of difference.
///
/// The routed matvec indexes the weights by EXPERT (`[E, out_vec_size, K]`, so
/// a routed row is `e * out_vec_size + out_row` — folding the expert into the
/// element offset instead is the classic way to read expert 0's weights for
/// every expert), the input by a PAIR of strides neither of which is `k`, and
/// the output by flat SLOT.
///
/// The two entrypoints differ ONLY in whether binding 5 is read. So the bias
/// buffer is bound for both, with values large enough that reading it or not
/// is unmistakable, and each answer is required to be REFUSED by the other's
/// reference. A row lists the operand either way, because a row is positional
/// and dropping the slot would shift `expert_ids` down a binding.
#[test]
fn a_routed_matvec_indexes_weights_by_expert_and_input_by_two_strides() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let shape = Routed::new();
    let k = shape.in_vec_size;
    let plane = Affine::new(64, 4, shape.weight_rows(), k, 0x77c1);
    let w = storage(gpu, &plane.words());
    let (scales, _) = bf16s(gpu, &plane.scales);
    let (biases, _) = bf16s(gpu, &plane.biases);
    let (x, x_seen) = bf16s(gpu, &spread(shape.x_len(), 1601));
    // In [20, 28): comparable with the dot products themselves, so an answer
    // that dropped the bias is not a small error.
    let bias_values: Vec<f32> = spread(shape.weight_rows(), 1609)
        .iter()
        .map(|v| 24.0 + v * 2.0)
        .collect();
    let (bias, bias_seen) = bf16s(gpu, &bias_values);
    let ids = i32s(gpu, &shape.expert_ids);

    let mut answers = Vec::new();
    for entrypoint in [
        "affine_qmv_routed_bfloat16_gs_64_b_4",
        "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
    ] {
        let y = sentinelled(gpu, shape.y_len().div_ceil(2));
        let block = shape.block(entrypoint);
        run(
            gpu,
            entrypoint,
            &[&w, &scales, &biases, &x, &y, &bias, &ids],
            &block,
            shape.grid(),
        );
        answers.push(unpack(&read(gpu, &y), shape.y_len()));
    }

    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    let mut plain = vec![sentinel; shape.y_len()];
    let mut biased = vec![sentinel; shape.y_len()];
    for row in 0..shape.rows {
        for slot in 0..shape.slots_per_row {
            let sel = row * shape.slots_per_row + slot;
            let expert = shape.expert_ids[sel];
            if expert < 0 {
                // Left holding the sentinel: an unrouted slot still runs the
                // reduction — a `return` in front of a `workgroupBarrier()` is
                // a hang — but it must not STORE.
                continue;
            }
            let e = usize::try_from(expert).expect("routed");
            let x_base = row * shape.x_row_stride + slot * shape.x_slot_stride;
            for out_row in 0..shape.out_vec_size {
                let wrow = e * shape.out_vec_size + out_row;
                let sum = routed_lane_sum(&x_seen, x_base, |at| plane.value(wrow, at), k);
                plain[sel * shape.out_vec_size + out_row] = rounded(sum);
                biased[sel * shape.out_vec_size + out_row] =
                    rounded(sum + bias_seen[e * shape.out_vec_size + out_row]);
            }
        }
    }

    agrees(&answers[0], &plain, "affine_qmv_routed").expect("the routed matvec agrees");
    agrees(&answers[1], &biased, "affine_qmv_routed_bias").expect("the biased form agrees");
    refuses_a_perturbed_reference(&answers[0], &plain, "affine_qmv_routed");
    refuses_a_perturbed_reference(&answers[1], &biased, "affine_qmv_routed_bias");

    // The one binding of difference, both ways.
    agrees(
        &answers[0],
        &biased,
        "the plain form against the biased reference",
    )
    .expect_err(
        "`affine_qmv_routed` does not read binding 5, and the bias is drawn \
         from [20, 28), so an answer that satisfies the biased reference read \
         a buffer its `//#if` arm does not declare",
    );
    agrees(
        &answers[1],
        &plain,
        "the biased form against the plain reference",
    )
    .expect_err("and the `_bias` form must not agree with the unbiased one");

    // The unrouted slot, on both. Zero cannot say this: it is what the buffer
    // was born with, so only a sentinel distinguishes "not written" from
    // "written zero".
    let sel = 7usize;
    assert_eq!(shape.expert_ids[sel], -1, "slot 7 is the unrouted one");
    for answer in &answers {
        for out_row in 0..shape.out_vec_size {
            let at = sel * shape.out_vec_size + out_row;
            assert_eq!(
                answer[at].to_bits(),
                sentinel.to_bits(),
                "slot {sel} is unrouted and output row {out_row} holds {} \
                 rather than the sentinel it was born with",
                answer[at],
            );
        }
    }
}

/// D29. `mxfp4_qmv_routed_bias` — and the slot its codec has no use for.
///
/// The MXFP4 codec's codes are not linear in the code, so there is nothing for
/// a per-group bias PLANE to be, and `moe/qmv_routed.wgsl` simply does not
/// declare `@binding(2)` under `PIE_MXFP4`. The row still lists the operand,
/// because a row is positional and dropping it would shift `x` down a binding.
///
/// So rather than skip that slot, this binds it TWICE with two completely
/// different buffers and requires the answer to be identical to the bit. That
/// is a stronger statement than not binding it: it says the launch is
/// insensitive to the slot, which is what "unbound" has to mean on a device
/// where the bind group entry exists either way.
///
/// The real bias moved upstream to `Weight(2)` and rides binding 5 with the
/// affine forms.
#[test]
fn an_mxfp4_routed_matvec_does_not_depend_on_the_bias_plane_slot() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let entrypoint = "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4";
    let shape = Routed::new();
    let k = shape.in_vec_size;
    let plane = Mxfp4::new(shape.weight_rows(), k, 0x4d19);
    let w = storage(gpu, &plane.words());
    // E8M0: one unsigned power-of-two byte per 32-element block, four bytes to
    // a word, lowest byte first. Not a bf16 plane, on the same binding the
    // affine arm reads bf16 group scales from.
    let scales = storage(gpu, &plane.scale_bytes());
    let (x, x_seen) = bf16s(gpu, &spread(shape.x_len(), 1701));
    let bias_values: Vec<f32> = spread(shape.weight_rows(), 1709)
        .iter()
        .map(|v| 30.0 + v * 2.0)
        .collect();
    let (bias, bias_seen) = bf16s(gpu, &bias_values);
    let ids = i32s(gpu, &shape.expert_ids);

    // Two entirely different buffers for the slot the MXFP4 arm never reads.
    let unread_a = sentinelled(gpu, 64);
    let (unread_b, _) = bf16s(gpu, &spread(128, 1719));
    let mut answers = Vec::new();
    for unread in [&unread_a, &unread_b] {
        let y = sentinelled(gpu, shape.y_len().div_ceil(2));
        let block = shape.block(entrypoint);
        run(
            gpu,
            entrypoint,
            &[&w, &scales, unread, &x, &y, &bias, &ids],
            &block,
            shape.grid(),
        );
        answers.push(unpack(&read(gpu, &y), shape.y_len()));
    }

    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    let mut want = vec![sentinel; shape.y_len()];
    for row in 0..shape.rows {
        for slot in 0..shape.slots_per_row {
            let sel = row * shape.slots_per_row + slot;
            let expert = shape.expert_ids[sel];
            if expert < 0 {
                continue;
            }
            let e = usize::try_from(expert).expect("routed");
            let x_base = row * shape.x_row_stride + slot * shape.x_slot_stride;
            for out_row in 0..shape.out_vec_size {
                let wrow = e * shape.out_vec_size + out_row;
                let sum = routed_lane_sum(&x_seen, x_base, |at| plane.value(wrow, at), k);
                want[sel * shape.out_vec_size + out_row] =
                    rounded(sum + bias_seen[e * shape.out_vec_size + out_row]);
            }
        }
    }

    agrees(&answers[0], &want, entrypoint).expect("the MXFP4 routed matvec agrees");
    refuses_a_perturbed_reference(&answers[0], &want, entrypoint);
    for (at, (a, b)) in answers[0].iter().zip(&answers[1]).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {at} moved from {a} to {b} when binding 2 changed. The \
             MXFP4 arm declares no buffer there — its codec has no separate \
             bias plane — so the answer must not be a function of it",
        );
    }
    // And the E2M1 table is not linear, which is the reason MXFP4 cannot
    // borrow the affine dot: `scale * code + bias` factors the bias out of the
    // inner product and these values do not factor at all. A `const` block, so
    // a table edited into linearity fails to COMPILE rather than to run.
    const {
        assert!(
            MXFP4_LUT[5] - MXFP4_LUT[4] != MXFP4_LUT[4] - MXFP4_LUT[3],
            "the E2M1 steps are supposed to be uneven",
        );
    }
}

/// D30. `router_topk_scaled` weights by the EXPERT its row chose.
///
/// The scaled form differs from the plain one by a per-expert normalization,
/// and the scale is indexed by the EXPERT and not by the slot: it is a
/// property of the expert the row chose, and `r` is only where in this row's
/// top-k that expert came. A body that used `r` would scale every row's first
/// choice by scale 0.
///
/// Both entrypoints are dispatched over identical inputs and each is checked
/// against its OWN closed form, then required to be refused by the other's —
/// the same discipline the gated activations get, and for the same reason:
/// one file, one binding contract, two bodies chosen by a `//#if`.
#[test]
fn a_scaled_router_weights_by_the_expert_and_not_by_the_slot() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n_experts = 7usize;
    let k = 3usize;
    let pitch = 9usize;
    let rows = 5usize;

    let (logits, logits_seen) = bf16s(gpu, &spread(rows * pitch, 1801));
    let params = storage(
        gpu,
        &router_params(n_experts as u32, k as u32, 1, pitch as u32),
    );
    // Far from one, and far from each other, so a scale read at the wrong
    // index is not a small error.
    let scale_values: Vec<f32> = (0..n_experts).map(|e| 0.25 + 0.5 * (e as f32)).collect();
    let (per_expert_scale, scale_seen) = bf16s(gpu, &scale_values);

    let mut answers = Vec::new();
    for entrypoint in ["router_topk_bfloat16", "router_topk_scaled_bfloat16"] {
        let expert_ids = i32s(gpu, &vec![-7i32; rows * k]);
        let expert_weights = sentinelled(gpu, (rows * k).div_ceil(2));
        run(
            gpu,
            entrypoint,
            &[
                &logits,
                &expert_ids,
                &expert_weights,
                &params,
                &per_expert_scale,
            ],
            &[],
            [1, rows as u32, 1],
        );
        answers.push((
            unpack_i32(&read(gpu, &expert_ids), rows * k),
            unpack(&read(gpu, &expert_weights), rows * k),
        ));
    }

    let mut want_ids = Vec::with_capacity(rows * k);
    let mut plain = Vec::with_capacity(rows * k);
    let mut scaled = Vec::with_capacity(rows * k);
    for row in 0..rows {
        let mine = &logits_seen[row * pitch..row * pitch + n_experts];
        let mut sorted: Vec<usize> = (0..n_experts).collect();
        sorted.sort_by(|a, b| mine[*b].total_cmp(&mine[*a]).then(a.cmp(b)));
        for w in mine.windows(2) {
            assert_ne!(
                w[0], w[1],
                "row {row} has two equal logits after bf16 rounding, so an \
                 exact id comparison would be checking the tie-break",
            );
        }
        let m = mine.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = mine.iter().map(|v| (v - m).exp()).sum();
        for &e in sorted.iter().take(k) {
            want_ids.push(i32::try_from(e).expect("fits"));
            let w = (mine[e] - m).exp() / z;
            plain.push(rounded(w));
            scaled.push(rounded(w * scale_seen[e]));
        }
    }

    assert_eq!(answers[0].0, want_ids, "the plain form's chosen experts");
    assert_eq!(
        answers[1].0, want_ids,
        "the scale multiplies the WEIGHT and must not move the choice: the \
         selection happens before it and over the logits",
    );
    agrees(&answers[0].1, &plain, "router_topk").expect("the plain weights agree");
    agrees(&answers[1].1, &scaled, "router_topk_scaled").expect("the scaled weights agree");
    refuses_a_perturbed_reference(&answers[1].1, &scaled, "router_topk_scaled");

    agrees(
        &answers[1].1,
        &plain,
        "the scaled form against the plain reference",
    )
    .expect_err(
        "the per-expert scales run from 0.25 to 3.25, so an answer that \
         satisfies the unscaled reference never read binding 4",
    );
    agrees(
        &answers[0].1,
        &scaled,
        "the plain form against the scaled reference",
    )
    .expect_err("and the unscaled form must not have applied them");

    // The scale is indexed by the EXPERT. A reference that indexed by the SLOT
    // must disagree, or this dispatch cannot tell the two apart.
    let by_slot: Vec<f32> = (0..rows * k)
        .map(|at| rounded(plain[at] * scale_seen[at % k]))
        .collect();
    assert!(
        by_slot
            .iter()
            .zip(&scaled)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the routing chosen here makes `scale[slot]` and `scale[expert]` the \
         same numbers, so this test cannot see the substitution; vary the \
         logits until the top-k is not the identity",
    );
    agrees(&answers[1].1, &by_slot, "the scaled form indexed by slot")
        .expect_err("the scale belongs to the expert, not to where it placed");
}

/// D31. `shared_expert_combine_strided` reads its gate a full PITCH apart.
///
/// The unstrided form's gate is one number per ROW at `gate[r]`, and
/// allocating it `rows` long is what makes `gate[r * width]` unrepresentable —
/// `a_shared_expert_gate_is_one_number_per_row` is that test. This variant is
/// NOT a copy of it, and the difference is the whole point: here the gate
/// really does stride by the pitch, because `qmv_out_size` answers 1 for the
/// shared gate projection and so its single output column is written a full
/// pitch apart like every other projection's. Both halves are stated in
/// `route.metal`; the Vulkan port collapsed the one where they differ.
///
/// So the pitch is 61 against a width of 45 — the useless case to test is the
/// production one where they are equal — and both are odd, which puts each
/// row's data at a different parity and makes every row boundary a shared
/// 32-bit word between two workgroups.
#[test]
fn a_strided_shared_expert_combine_reads_its_gate_a_full_pitch_apart() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let entrypoint = "shared_expert_combine_strided";
    let width = 45usize;
    let row_pitch = 61usize;
    let rows = ROWS as usize;
    assert_ne!(
        row_pitch, width,
        "equal pitches are the case that proves nothing"
    );
    let n = rows * row_pitch;

    let (routed, routed_seen) = bf16s(gpu, &spread(n, 1901));
    let (shared, shared_seen) = bf16s(gpu, &spread(n, 1907));
    // A full plane, because THIS form's gate is a projection output written a
    // pitch apart — only `r * row_pitch` is read, and the columns between are
    // whatever that projection left there.
    let (gate, gate_seen) = bf16s(gpu, &spread(n, 1913));
    let out = sentinelled(gpu, n.div_ceil(2));

    let block = Block::of(entrypoint)
        .u32("width", u32::try_from(width).expect("fits"))
        .i32("row_pitch", i32::try_from(row_pitch).expect("fits"))
        .done();
    assert_eq!(
        block.len(),
        16,
        "two four-byte fields, rounded to 16 by WGSL"
    );
    run(
        gpu,
        entrypoint,
        &[&routed, &shared, &gate, &out],
        &block,
        [over(width as u32, 16), over(rows as u32, 16), 1],
    );

    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    let got = unpack(&read(gpu, &out), n);
    for row in 0..rows {
        let base = row * row_pitch;
        // The plain logistic, not MLX's stable reflection: this is what the
        // shader writes and the two are different floating-point objects.
        let g = 1.0 / (1.0 + (-gate_seen[base]).exp());
        let want: Vec<f32> = (0..width)
            .map(|c| rounded(routed_seen[base + c] + g * shared_seen[base + c]))
            .collect();
        let what = format!("row {row} at gate {}", gate_seen[base]);
        agrees(&got[base..base + width], &want, &what)
            .expect("every row was blended with the gate a pitch apart");
        if row == rows - 1 {
            refuses_a_perturbed_reference(&got[base..base + width], &want, &what);
        }
        // The columns between `width` and the pitch belong to nobody. This
        // body writes one bf16 per invocation through an atomicAnd/atomicOr
        // pair, so the partner half of the boundary word — column 44's, which
        // is column 45 — has to survive untouched.
        for c in width..row_pitch {
            assert_eq!(
                got[base + c].to_bits(),
                sentinel.to_bits(),
                "row {row} column {c} is between the width ({width}) and the \
                 pitch ({row_pitch}) and was written anyway",
            );
        }
    }

    // The gate is read at `r * row_pitch` and NOT at `r`, which is the
    // unstrided form's index. A reference built the other way must disagree,
    // or the two variants are not distinguishable here.
    let mut by_row = Vec::with_capacity(rows * width);
    let mut flat = Vec::with_capacity(rows * width);
    for (row, unstrided) in gate_seen.iter().enumerate().take(rows) {
        let base = row * row_pitch;
        let g = 1.0 / (1.0 + (-unstrided).exp());
        for c in 0..width {
            by_row.push(rounded(routed_seen[base + c] + g * shared_seen[base + c]));
            flat.push(got[base + c]);
        }
    }
    agrees(
        &flat,
        &by_row,
        "the strided form with the unstrided gate index",
    )
    .expect_err(
        "`gate[r]` is the UNSTRIDED form's index. If the device's answer also \
         satisfies it, the two variants are the same kernel here and this test \
         is not separating them",
    );

    // The overshot row the 16-high workgroup gives, exactly as the unstrided
    // form has it: `rows` is 13 and the grid covers 16, so row 13 column 0
    // lands on the pad half of the pad word — over zero pads, so a zero — and
    // the `arrayLength` guard refuses everything past it.
    let tail = unpack(&read(gpu, &out), n + 1);
    assert_eq!(
        tail[n], 0.0,
        "element {n} is the pad half of the pad word and holds {} rather than \
         the zero the overshot row computes over a zero pad",
        tail[n],
    );
}

/// D32. `split_qkv_bf16` writes three tensors, and the failure is a stride.
///
/// One packed row in, three rows out at three different offsets and TWO
/// different widths. So Q gets five heads and K and V get two — a body that
/// used the query head count for the key write is the defect this row has, and
/// equal counts cannot see it, because then every stride in the kernel is the
/// same number.
///
/// The two widths ride in `SplitQkvParams`, a STORAGE struct at binding 4:
/// the row states `params: Buf`, so there is no `@group(1)` here at all and
/// this test hands it none. `dump_layout` prints "0 bytes of uniform block".
///
/// The x extent is in channel PAIRS — 54 of them over a 256-wide workgroup —
/// so plain division dispatches nothing at all.
#[test]
fn a_qkv_split_gives_each_projection_its_own_width() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 12usize;
    let q_heads = 5usize;
    let kv_heads = 2usize;
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let packed_width = q_width + 2 * kv_width;
    let rows = ROWS as usize;
    assert_ne!(
        q_width, kv_width,
        "equal widths make every stride in this kernel the same number",
    );

    let (packed, packed_seen) = bf16s(gpu, &spread(rows * packed_width, 2001));
    let q = sentinelled(gpu, (rows * q_width).div_ceil(2));
    let k = sentinelled(gpu, (rows * kv_width).div_ceil(2));
    let v = sentinelled(gpu, (rows * kv_width).div_ceil(2));
    let params = storage(
        gpu,
        &[
            (q_width as u32).to_le_bytes(),
            (kv_width as u32).to_le_bytes(),
        ]
        .concat(),
    );

    // `LaunchRule::SplitPacked` is `[in_width, rows, 1]` in ELEMENTS; this body
    // owns a pair, so half that many lanes do the work and the rest exit at the
    // guard. Dispatched at the minimum, where an undershoot is visible.
    run(
        gpu,
        "split_qkv_bf16",
        &[&packed, &q, &k, &v, &params],
        &[],
        [over(packed_width.div_ceil(2) as u32, 256), rows as u32, 1],
    );

    let got_q = unpack(&read(gpu, &q), rows * q_width);
    let got_k = unpack(&read(gpu, &k), rows * kv_width);
    let got_v = unpack(&read(gpu, &v), rows * kv_width);
    // Exact: the body widens to f32 and repacks, which is a lossless round trip
    // for every finite bf16, so a split moves BITS.
    for row in 0..rows {
        for c in 0..packed_width {
            let from = packed_seen[row * packed_width + c];
            let (which, got, at) = if c < q_width {
                ("q", &got_q, row * q_width + c)
            } else if c < q_width + kv_width {
                ("k", &got_k, row * kv_width + (c - q_width))
            } else {
                ("v", &got_v, row * kv_width + (c - q_width - kv_width))
            };
            assert_eq!(
                got[at].to_bits(),
                from.to_bits(),
                "row {row} packed channel {c} belongs to {which}[{at}] and \
                 holds {} rather than {from}. q is {q_width} wide and k and v \
                 are {kv_width}, so a stride taken from the wrong projection \
                 lands inside another one",
                got[at],
            );
        }
    }

    // The control: a reference that gave K the QUERY width must disagree, or
    // the two head counts chosen here cannot tell them apart.
    let mut by_q_width = vec![0.0f32; rows * kv_width];
    for row in 0..rows {
        for c in 0..kv_width {
            // What a body striding K by `q_width` would have written.
            let at = row * q_width + c;
            if at < rows * kv_width {
                by_q_width[at] = packed_seen[row * packed_width + q_width + c];
            }
        }
    }
    assert!(
        got_k
            .iter()
            .zip(&by_q_width)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the key tensor also satisfies a reference strided by the query width, \
         so this shape is not separating them",
    );
}

/// D33. `logit_softcap` saturates rather than running away.
///
/// `cap * tanh(x / cap)`, and the interesting inputs are the ones a spread of
/// activations never produces: values well past the cap in both directions,
/// and one large enough that `x / cap` is 2.4e37 — where the answer must be
/// exactly the cap and not a NaN.
///
/// `SoftcapParams` is a STORAGE struct at binding 2 and its second field is
/// bound-and-unread, exactly as `GegluParams::unused` is. So it is filled with
/// a number that would be catastrophic as a bound — 3, against 5980 elements —
/// and every element is still required to be right. A future edit that started
/// reading it fails here rather than returning a tensor whose first three
/// elements are capped.
#[test]
fn a_logit_softcap_saturates_at_its_cap() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    let cap = 12.5f32;

    let mut values = spread(n, 2101);
    // Past the cap, both ways, at several magnitudes.
    values[3] = 40.0;
    values[4] = -40.0;
    values[5] = 512.0;
    values[6] = -512.0;
    // `x / cap` is 2.4e37 here: `tanh` must saturate to one and the product
    // must be the cap, not an infinity and not a NaN.
    values[7] = 3.0e38;
    values[8] = -3.0e38;
    let (logits, logits_seen) = bf16s(gpu, &values);
    let out = sentinelled(gpu, n / 2);
    // `cap`, then a field the body does not read — filled with a number that
    // would be a catastrophic bound.
    let params = storage(
        gpu,
        &[cap.to_bits().to_le_bytes(), 3u32.to_le_bytes()].concat(),
    );

    run(
        gpu,
        "logit_softcap_bfloat16",
        &[&logits, &out, &params],
        &[],
        [over(n as u32 / 2, 256), 1, 1],
    );

    let got = unpack(&read(gpu, &out), n);
    let want: Vec<f32> = logits_seen
        .iter()
        .map(|x| rounded(cap * (x / cap).tanh()))
        .collect();
    agrees(&got, &want, "the capped logits").expect("the softcap agrees");
    refuses_a_perturbed_reference(&got, &want, "the capped logits");

    // The saturation, named rather than folded into the comparison: a NaN and
    // a slightly wrong number are both "not the reference" and only one of
    // them is arithmetic.
    for at in [7usize, 8] {
        assert!(
            got[at].is_finite(),
            "element {at} was fed {} and came back {}, which is not a number",
            logits_seen[at],
            got[at],
        );
        assert_eq!(
            got[at].abs(),
            cap,
            "element {at} was fed {}, so `x / cap` is {} and `tanh` of it is \
             one: the answer must be exactly the cap",
            logits_seen[at],
            logits_seen[at] / cap,
        );
    }
    assert!(
        got.iter().all(|v| v.abs() <= cap),
        "a softcap that lets any logit past its cap has not capped anything",
    );
    // And every element is right, including the ones past the unread `n` = 3.
    assert!(
        n > 3,
        "the unread field is set to a number that would be a catastrophic \
         bound, which only means something if the tensor is longer than it",
    );
}

/// D34. `ple_combine` is `(proj + token) * inv_sqrt2`, in that order.
///
/// gemma's two per-layer-embedding streams, averaged in the root-mean-square
/// sense. The scale is the JOIN's and not a deployment's, so it arrives in
/// `PleCombineParams` — a STORAGE struct at binding 3 — beside an `n` the body
/// does not read.
///
/// # The trap, which is the rounding and not the addition
///
/// The add is in f32 and the multiply is in f32 and there is exactly ONE bf16
/// round, on the store. A body that rounded the sum first — which is what
/// `pie_bf16_to_f32(pie_f32_to_bf16(a + b)) * s` would be, and which is what
/// the two fused matmul epilogues in this tree deliberately DO — is a
/// different number, and at `inv_sqrt2` it is a different number by up to half
/// a bf16 ulp on every element. So the reference is checked against that
/// alternative too, and has to refuse it.
///
/// `n` is filled with 3 for the same reason `logit_softcap`'s unread field is.
#[test]
fn a_ple_combine_rounds_once_and_at_the_end() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    // Not 1.0 and not 0.5: a body that dropped the scale, or that used the
    // other obvious constant, has to be visible.
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let (proj, proj_seen) = bf16s(gpu, &spread(n, 2201));
    let (token, token_seen) = bf16s(gpu, &spread(n, 2203));
    let out = sentinelled(gpu, n / 2);
    let params = storage(
        gpu,
        &[inv_sqrt2.to_bits().to_le_bytes(), 3u32.to_le_bytes()].concat(),
    );

    run(
        gpu,
        "ple_combine_bfloat16",
        &[&proj, &token, &out, &params],
        &[],
        [over(n as u32 / 2, 256), 1, 1],
    );

    let got = unpack(&read(gpu, &out), n);
    let want: Vec<f32> = proj_seen
        .iter()
        .zip(&token_seen)
        .map(|(a, b)| rounded((a + b) * inv_sqrt2))
        .collect();
    agrees(&got, &want, "the combined embedding").expect("the combine agrees");
    refuses_a_perturbed_reference(&got, &want, "the combined embedding");

    // The alternative rounding, and the unscaled sum. Both are things a port
    // produces by accident and both are wrong.
    let rounded_first: Vec<f32> = proj_seen
        .iter()
        .zip(&token_seen)
        .map(|(a, b)| rounded(rounded(a + b) * inv_sqrt2))
        .collect();
    let moved = got
        .iter()
        .zip(&rounded_first)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        moved > 0,
        "rounding the sum before the scale is supposed to be a DIFFERENT \
         number on this data; if it is not, these inputs cannot see the \
         difference and the claim about the single round is untested",
    );
    let unscaled: Vec<f32> = proj_seen
        .iter()
        .zip(&token_seen)
        .map(|(a, b)| rounded(a + b))
        .collect();
    agrees(&got, &unscaled, "the combine without its scale")
        .expect_err("a body that dropped `inv_sqrt2` would satisfy this");
}

/// D35. `layer_scalar_mul` reads its scalar from a BUFFER.
///
/// Which layer is running is the FIRE's business, so gemma4's per-layer scale
/// is a resident `[1]` tensor rather than a number the statement carries — and
/// that makes the operand a `Buf` at binding 1, between `x` and `out`.
///
/// The scalar is 0.375, which is neither 1.0 (a body that ignored the buffer)
/// nor `x[0]` (a body that read the wrong binding). Element 1 of the same
/// buffer is -2.75, so a body that took the word's HIGH half — the other
/// obvious half-index slip — is wrong by a sign as well as a magnitude.
///
/// `LayerScalarParams.hidden` is bound and not read, and that is the Metal
/// port's finding kept rather than tidied away: the field is ONE ROW's width
/// while `LaunchRule::Elementwise` dispatches `width * rows`, so reading it as
/// a bound returned every row after the first holding whatever the arena had.
/// It is filled with 7 here.
#[test]
fn a_layer_scalar_multiply_reads_its_scale_from_the_buffer() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    let scale = 0.375f32;
    let (x, x_seen) = bf16s(gpu, &spread(n, 2301));
    assert_ne!(
        x_seen[0], scale,
        "the scale must not be a value `x` also holds at index zero",
    );
    let (scalar, scalar_seen) = bf16s(gpu, &[scale, -2.75]);
    let out = sentinelled(gpu, n / 2);
    let params = storage(gpu, &7u32.to_le_bytes());

    run(
        gpu,
        "layer_scalar_mul_bfloat16",
        &[&x, &scalar, &out, &params],
        &[],
        [over(n as u32 / 2, 256), 1, 1],
    );

    let got = unpack(&read(gpu, &out), n);
    let want: Vec<f32> = x_seen.iter().map(|v| rounded(v * scalar_seen[0])).collect();
    agrees(&got, &want, "the scaled layer").expect("the layer scalar agrees");
    refuses_a_perturbed_reference(&got, &want, "the scaled layer");

    // The three ways to get the scalar wrong, all refused.
    let unscaled: Vec<f32> = x_seen.iter().map(|v| rounded(*v)).collect();
    agrees(&got, &unscaled, "the layer without its scale")
        .expect_err("a body that ignored binding 1 would satisfy this");
    let high_half: Vec<f32> = x_seen.iter().map(|v| rounded(v * scalar_seen[1])).collect();
    agrees(&got, &high_half, "the layer scaled by the word's high half")
        .expect_err("`pie_bf16_at(scalar[0], 0u)` is the LOW half of word zero");
    // And every element past the unread `hidden` = 7 is right, which the
    // comparison above already covers and this names.
    assert!(
        n > 7 && got.len() == n,
        "the unread field is 7 against {n} elements, so a body that read it as \
         a bound would leave everything after element 7 untouched",
    );
}

/// D36. `vnorm_single_row` normalizes an AXIS that is not the row.
///
/// The weightless RMSNorm: the row divided by its own RMS and nothing else, so
/// the absence of a gain buffer is the whole difference from `rms_single_row`
/// and the operand list is three long rather than four.
///
/// # Why the axis is 92 and the width is 460
///
/// The row states `grid_param = Some(1)` — `VNormParams.axis_size` — and that
/// was a real gap on the Vulkan side, caught by the parity test on its first
/// run. A value norm's axis is the HEAD and its row is every head, so a launch
/// that took the fire's width for the axis reduces the whole row as one, which
/// is not a coarser normalization but a different number in every channel.
/// Five heads of 92 to a row is that case; one head per row could not see it.
///
/// 92 is not a multiple of the 1024-element chunk the body walks a row in, and
/// 46 words is not a multiple of its 256-lane store loop.
#[test]
fn a_vector_norm_reduces_over_its_axis_and_not_over_its_row() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let axis = 92usize;
    let width = WIDTH as usize;
    let rows = ROWS as usize;
    assert_eq!(width % axis, 0, "five heads to a row");
    assert_ne!(
        axis, width,
        "an axis equal to the width is the case that proves nothing"
    );
    let n = rows * width;
    let eps = 1e-6f32;

    let (x, x_seen) = bf16s(gpu, &spread(n, 2401));
    let out = sentinelled(gpu, n / 2);
    let params = storage(
        gpu,
        &[eps.to_bits().to_le_bytes(), (axis as u32).to_le_bytes()].concat(),
    );

    // `LaunchRule::Rms` is one workgroup per AXIS, not per row: `width / axis`
    // axes to a row and `rows` rows.
    let axes = (width / axis) * rows;
    run(
        gpu,
        "vnorm_single_row_bfloat16",
        &[&x, &out, &params],
        &[],
        [axes as u32, 1, 1],
    );

    let norm_over = |span: &[f32]| -> Vec<f32> {
        let total: f32 = span.iter().map(|v| v * v).sum();
        let inv = (total / span.len() as f32 + eps).sqrt().recip();
        span.iter().map(|v| rounded(v * inv)).collect()
    };
    let got = unpack(&read(gpu, &out), n);
    let mut want = Vec::with_capacity(n);
    for a in 0..axes {
        want.extend(norm_over(&x_seen[a * axis..(a + 1) * axis]));
    }
    agrees(&got, &want, "the value norm").expect("the value norm agrees");
    refuses_a_perturbed_reference(&got, &want, "the value norm");

    // The gap this shape exists for: a launch that took the fire's WIDTH for
    // the axis reduces five heads as one. Every channel then moves, so the
    // count claim refuses it as well as the per-element bound.
    let mut by_row = Vec::with_capacity(n);
    for row in 0..rows {
        by_row.extend(norm_over(&x_seen[row * width..(row + 1) * width]));
    }
    agrees(&got, &by_row, "the value norm reduced over the whole row").expect_err(
        "`axis_size` is the row's `grid_param` and a launch that used the \
         width instead normalizes five heads as one",
    );
    // And there IS a gain-free claim here: multiplying by anything would move
    // the answer, so a body that grew a weight buffer is visible.
    assert!(
        want.iter().any(|v| v.abs() > 0.5),
        "a norm with no gain leaves values of order one; if every element is \
         tiny the comparison has nothing to bite on",
    );
}

/// D37. `embed_gather` in all four of its corners, at the two quantization
/// points that pack IN STEP.
///
/// Four rows over 24 entrypoints: `{,_scaled}` x `{,_mb}` x `gs_{32,64,128}` x
/// `b_{4,8}`. The corners are the `_scaled`/`_mb` combinations, because those
/// are what change which buffers are read and which grid axis carries the row.
///
/// # Why `gs_64/b_8` and `gs_128/b_4` are the pair to choose
///
/// `common/affine.inc.wgsl` says the two numbers are one fact: `PIE_GROUP /
/// PIE_CODES_PER_WORD` is 16 for BOTH of them, so the two walk the packed plane
/// and the scale plane in exact step and a module compiled for the wrong one
/// does not fail — it reads the scales against the wrong weights and returns
/// fluent nonsense.
///
/// So both points are handed the SAME `w`, `scales` and `biases` bytes, sized
/// for the larger of the two readings, and each answer is required to satisfy
/// its own decode and to be REFUSED by the other's. That is the coordinate
/// claim done on hardware rather than argued about: nothing about the buffers
/// says which pair they were packed for.
///
/// # Why the MB x extent is not ragged, and cannot be made so
///
/// One invocation owns the output WORD, so the x extent is `hidden / 2`; and
/// `hidden` is a whole number of groups of at least 32 or the checkpoint would
/// not pack, so `hidden / 2` is always a multiple of 16 and the `_mb` body's
/// 16-wide x axis always divides. The ROW axis is where the round-up is real —
/// 13 over 16 — and that is the axis an undershoot loses whole tokens on. The
/// non-MB corners run at 192 words over 256, where plain division dispatches
/// nothing at all.
#[test]
fn an_embedding_gather_decodes_in_all_four_corners_at_two_packings() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let hidden = 384usize;
    let vocab = 20usize;
    let rows = ROWS as usize;
    let embed_scale = 19.593_75f32;

    // The two readings. Same words per group — 16 — which is what makes them
    // indistinguishable from the buffers alone.
    let points = [(64usize, 8u32), (128, 4)];
    for (group, bits) in points {
        assert_eq!(
            group / (32 / bits as usize),
            16,
            "this pair was chosen because both walk 16 words to a group",
        );
        assert_eq!(hidden % group, 0, "a row is a whole number of groups");
    }

    // Raw planes, sized for the WIDER reading of each, so one set of bytes
    // serves both modules.
    let mut state = 0xbeef_1234u32;
    let words_per_row = hidden / (32 / 8); // the b_8 reading, which is the larger
    let mut packed = vec![0u32; vocab * words_per_row];
    for w in &mut packed {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *w = state;
    }
    let groups_per_row = hidden / 64; // the gs_64 reading, which is the larger
    let scale_values: Vec<f32> = positives(vocab * groups_per_row, 2501)
        .iter()
        .map(|v| v * 0.02)
        .collect();
    let bias_values: Vec<f32> = spread(vocab * groups_per_row, 2503)
        .iter()
        .map(|v| v * 0.1)
        .collect();
    let w = storage(
        gpu,
        &packed
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let (scales, scale_seen) = bf16s(gpu, &scale_values);
    let (biases, bias_seen) = bf16s(gpu, &bias_values);
    // Scrambled, and none of them its own index, so a gather that ignored the
    // id list is wrong everywhere rather than right by accident.
    let ids: Vec<i32> = vec![17, 3, 19, 0, 11, 8, 15, 2, 13, 6, 9, 1, 4];
    assert_eq!(ids.len(), rows);
    let id = i32s(gpu, &ids);

    // One decoded element under one reading of the pair.
    let decode = |group: usize, bits: u32, row: usize, k: usize| -> f32 {
        let codes_per_word = 32 / bits as usize;
        let word = packed[row * (hidden / codes_per_word) + k / codes_per_word];
        let code = (word >> ((k % codes_per_word) as u32 * bits)) & ((1u32 << bits) - 1);
        let g = row * (hidden / group) + k / group;
        scale_seen[g] * code as f32 + bias_seen[g]
    };

    let mut covered = 0usize;
    for (scaled, mb) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut answers = Vec::new();
        for (group, bits) in points {
            let entrypoint = format!(
                "embed_gather{}{}_4bit_bfloat16_gs_{group}_b_{bits}",
                if scaled { "_scaled" } else { "" },
                if mb { "_mb" } else { "" },
            );
            // Allocated for every row either way, so the single-row corners
            // have somewhere to be wrong: rows 1.. must keep their sentinel.
            let out = sentinelled(gpu, (rows * hidden).div_ceil(2));
            let mut block =
                Block::of(&entrypoint).i32("hidden", i32::try_from(hidden).expect("fits"));
            if scaled {
                block = block.f32("embed_scale", embed_scale);
            }
            let grid = if mb {
                // `LaunchRule::ElementwiseRows`, in output WORDS on x.
                [over(hidden as u32 / 2, 16), over(rows as u32, 16), 1]
            } else {
                // `LaunchRule::Elementwise`, one row, in output WORDS.
                [over(hidden as u32 / 2, 256), 1, 1]
            };
            run(
                gpu,
                &entrypoint,
                &[&w, &scales, &biases, &id, &out],
                &block.done(),
                grid,
            );
            answers.push((entrypoint, unpack(&read(gpu, &out), rows * hidden)));
            covered += 1;
        }

        let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
        for (at, (group, bits)) in points.iter().enumerate() {
            let want: Vec<f32> = (0..rows * hidden)
                .map(|i| {
                    let m = i / hidden;
                    if !mb && m > 0 {
                        // The single-row corners read `id[0]` and write one
                        // row; everything after it was never theirs.
                        return sentinel;
                    }
                    let row = usize::try_from(ids[m]).expect("an embedding row");
                    let v = decode(*group, *bits, row, i % hidden);
                    rounded(if scaled { v * embed_scale } else { v })
                })
                .collect();
            let (name, got) = &answers[at];
            agrees(got, &want, name).expect("the gather decodes its own packing");
            refuses_a_perturbed_reference(got, &want, name);

            // The coordinate: the OTHER reading of the same bytes.
            let (other_group, other_bits) = points[1 - at];
            let other: Vec<f32> = (0..rows * hidden)
                .map(|i| {
                    let m = i / hidden;
                    if !mb && m > 0 {
                        return sentinel;
                    }
                    let row = usize::try_from(ids[m]).expect("an embedding row");
                    let v = decode(other_group, other_bits, row, i % hidden);
                    rounded(if scaled { v * embed_scale } else { v })
                })
                .collect();
            agrees(
                got,
                &other,
                &format!("{name} read as gs_{other_group}/b_{other_bits}"),
            )
            .expect_err(
                "these two packings walk 16 words to a group either way, \
                     so nothing about the buffers says which one they are. If \
                     one module's answer satisfies the other's decode, the \
                     entrypoint's `_gs_.._b_..` suffix is a label rather than \
                     a coordinate",
            );

            // And the scale is IN the answer, or out of it.
            let unscaled: Vec<f32> = (0..rows * hidden)
                .map(|i| {
                    let m = i / hidden;
                    if !mb && m > 0 {
                        return sentinel;
                    }
                    let row = usize::try_from(ids[m]).expect("an embedding row");
                    rounded(decode(*group, *bits, row, i % hidden))
                })
                .collect();
            if scaled {
                agrees(got, &unscaled, &format!("{name} without its embed_scale")).expect_err(
                    "gemma multiplies its embeddings by a number the statement \
                     carries; an answer that satisfies the unscaled decode \
                     never read `embed_scale`",
                );
            }
        }
    }
    assert_eq!(covered, 8, "four corners at two packings");
}

/// D38. `neox_mb` rotates a BATCH, one position per row.
///
/// The decode arms compile `row = 0` and read `position[0]`; the multi-batch
/// ones take the row from `workgroup_id.z` and index the position list with
/// it. So the three positions here are all different, and a body that read
/// `position[0]` for every row would turn rows 1 and 2 by row 0's angle —
/// which is a plausible tensor and the wrong one.
#[test]
fn a_batched_rope_gives_every_row_its_own_position() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 12usize;
    let heads = 5usize;
    let pairs = head_dim / 2;
    let rows = 3usize;
    let n = rows * heads * head_dim;
    let scale = 0.75f32;
    let base = 13.287_712_f32;
    let positions = [7i32, 2, 11];

    let (x, x_seen) = bf16s(gpu, &spread(n, 2601));
    let position = i32s(gpu, &positions);
    let block = Block::of("neox_mb_bfloat16")
        .f32("scale", scale)
        .f32("base", base)
        .i32("head_dim", head_dim as i32)
        .done();
    // `LaunchRule::Rope` is `[rotary / 2, heads, rows]`, and the module is
    // `@workgroup_size(1)` so the grid is EXACT — the body reads
    // `num_workgroups.x` as the pair count it strides each partner by.
    run(
        gpu,
        "neox_mb_bfloat16",
        &[&x, &position],
        &block,
        [pairs as u32, heads as u32, rows as u32],
    );

    let shape = Neox {
        heads,
        head_dim,
        pairs,
        scale,
        base,
        prop: false,
        freqs: &[],
        mscale: 1.0,
    };
    let got = unpack(&read(gpu, &x), n);
    let want = neox_reference(&x_seen, &positions, shape);
    agrees(&got, &want, "the batched rotation").expect("neox_mb rotates");
    refuses_a_perturbed_reference(&got, &want, "the batched rotation");

    // Every row turned by row 0's position: what reading `position[0]` gives.
    let by_first = neox_reference(&x_seen, &[positions[0]; 3], shape);
    agrees(&got, &by_first, "the batch turned by row 0's position").expect_err(
        "the three positions here are 7, 2 and 11; if the device's answer also \
         satisfies a reference that used 7 for all of them, the row axis is \
         not being read",
    );
}

/// D39. `neox_freqs_decode` and `neox_freqs_mb` — a table, and a REORDERED
/// uniform block.
///
/// This is the ABI half of the rope family and the reason it is worth a test
/// of its own. The geometric rows state `scale, base, head_dim`; these state
/// `scale, head_dim, mscale` and bind `inv_freq` as a THIRD storage buffer.
/// So `head_dim` moves from byte 8 to byte 4 and a shader — or a shell — that
/// transcribed Metal's numbering, where `inv_freq` is buffer 3 and `head_dim`
/// is 4, reads the frequency table's address as its head width.
///
/// [`Block`] refuses a field the row does not state, so writing `base` here
/// would panic rather than land somewhere plausible. The angles come from the
/// table and the rotation carries YaRN's `mscale`, which is 1.0 in every
/// deployment that has none — so it is 1.375 here, where dropping it is
/// visible.
#[test]
fn a_frequency_table_rope_reads_a_reordered_block() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 12usize;
    let heads = 5usize;
    let pairs = head_dim / 2;
    let scale = 0.75f32;
    let mscale = 1.375f32;
    // A ladder no exponent produces: llama-3's piecewise interpolation and
    // YaRN's are tables for exactly this reason.
    let inv_freq_values: Vec<f32> = vec![1.0, 0.61, 0.27, 0.089, 0.0201, 0.00374];
    assert_eq!(inv_freq_values.len(), pairs);

    for (entrypoint, positions) in [
        ("neox_freqs_decode_bfloat16", vec![7i32]),
        ("neox_freqs_mb_bfloat16", vec![7i32, 2, 11]),
    ] {
        let rows = positions.len();
        let n = rows * heads * head_dim;
        let (x, x_seen) = bf16s(gpu, &spread(n, 2701));
        let position = i32s(gpu, &positions);
        let inv_freq = storage(
            gpu,
            &inv_freq_values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        // No `base`: this row does not state one, and `Block` refuses a name
        // the row does not have.
        let block = Block::of(entrypoint)
            .f32("scale", scale)
            .i32("head_dim", head_dim as i32)
            .f32("mscale", mscale)
            .done();
        run(
            gpu,
            entrypoint,
            &[&x, &position, &inv_freq],
            &block,
            [pairs as u32, heads as u32, rows as u32],
        );

        let shape = Neox {
            heads,
            head_dim,
            pairs,
            scale,
            // Unused when a table is present, and set to a value that would be
            // catastrophic if it were: a body that fell through to the
            // geometric arm turns nothing at all.
            base: 0.0,
            prop: false,
            freqs: &inv_freq_values,
            mscale,
        };
        let got = unpack(&read(gpu, &x), n);
        let want = neox_reference(&x_seen, &positions, shape);
        agrees(&got, &want, entrypoint).expect("the frequency-table rope rotates");
        refuses_a_perturbed_reference(&got, &want, entrypoint);

        // Without the gain, which is the other half of this row's block.
        let ungained = neox_reference(
            &x_seen,
            &positions,
            Neox {
                mscale: 1.0,
                ..shape
            },
        );
        agrees(&got, &ungained, &format!("{entrypoint} without its mscale")).expect_err(
            "`mscale` is YaRN's attention-temperature correction and it is \
             1.375 here; an answer that satisfies a reference without it never \
             read the block's third field",
        );
    }
}

/// D40. `rms_residual` and `rms_residual_scaled` — the fold, and the buffer
/// that arrives AFTER the params struct.
///
/// The residual is `@binding(4)` and the per-layer gain `@binding(5)`, both
/// after `params` at 3, because the row's operand order is
/// `x, w, out, params, r[, s]` and the buffer run is dense in that order. This
/// is the shape `.wiki/new-driver/vulkan.md` §3 names: "a residual buffer that
/// sits at descriptor five because the two scalars before it moved to push
/// constants". Here there are no scalars at all — every one of this row's
/// operands is a buffer — so the numbering is the row's and Metal's is not.
///
/// The epilogue is `(gain * (x * inv) + r) * post`, with ONE bf16 round on the
/// store, so the residual is added in float and the scaled form's per-layer
/// gain multiplies the sum rather than the norm.
#[test]
fn a_norm_folds_its_residual_and_its_layer_gain() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let eps = 1e-6f32;
    let n = (ROWS * WIDTH) as usize;
    let (x, x_seen) = bf16s(gpu, &spread(n, 2801));
    let (w, w_seen) = bf16s(gpu, &spread(WIDTH as usize, 2803));
    let (r, r_seen) = bf16s(gpu, &spread(n, 2807));
    // One number for the whole launch, read by every lane as a broadcast. Not
    // 1.0, so dropping it is visible.
    let post = 1.25f32;
    let (s, s_seen) = bf16s(gpu, &[post, -3.5]);
    let params = storage(gpu, &rms_params(eps, WIDTH, 1, 0, 1.0));

    let plain = sentinelled(gpu, n / 2);
    run(
        gpu,
        "rms_residual_bfloat16",
        &[&x, &w, &plain, &params, &r],
        &[],
        [ROWS, 1, 1],
    );
    let got_plain = unpack(&read(gpu, &plain), n);

    let scaled = sentinelled(gpu, n / 2);
    run(
        gpu,
        "rms_residual_scaled_bfloat16",
        &[&x, &w, &scaled, &params, &r, &s],
        &[],
        [ROWS, 1, 1],
    );
    let got_scaled = unpack(&read(gpu, &scaled), n);

    let fold = |gain: f32| -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for row in 0..ROWS as usize {
            let span = &x_seen[row * WIDTH as usize..(row + 1) * WIDTH as usize];
            let total: f32 = span.iter().map(|v| v * v).sum();
            let inv = (total / WIDTH as f32 + eps).sqrt().recip();
            for i in 0..WIDTH as usize {
                let at = row * WIDTH as usize + i;
                out.push(rounded((w_seen[i] * (span[i] * inv) + r_seen[at]) * gain));
            }
        }
        out
    };
    let want_plain = fold(1.0);
    let want_scaled = fold(s_seen[0]);

    agrees(&got_plain, &want_plain, "rms_residual").expect("the folded norm agrees");
    agrees(&got_scaled, &want_scaled, "rms_residual_scaled").expect("the scaled fold agrees");
    refuses_a_perturbed_reference(&got_plain, &want_plain, "rms_residual");
    refuses_a_perturbed_reference(&got_scaled, &want_scaled, "rms_residual_scaled");

    // The residual is IN the answer: a norm that never read binding 4 would
    // satisfy `rms_single_row`'s reference instead.
    let unfolded: Vec<f32> = (0..ROWS as usize)
        .flat_map(|row| {
            let span = &x_seen[row * WIDTH as usize..(row + 1) * WIDTH as usize];
            rms_reference(span, &w_seen, 1, false, 1.0, eps)
        })
        .collect();
    agrees(&got_plain, &unfolded, "rms_residual without its residual")
        .expect_err("binding 4 is the residual and it is not zeros");
    // And the two arms differ by the gain, both ways.
    agrees(
        &got_scaled,
        &want_plain,
        "the scaled fold against the plain one",
    )
    .expect_err("`s[0]` is 1.25, so the two arms cannot agree");
    let by_high_half = fold(s_seen[1]);
    agrees(
        &got_scaled,
        &by_high_half,
        "the scaled fold with the word's high half",
    )
    .expect_err("`pie_bf16_at(s[0], 0u)` is the LOW half of word zero");
}

/// D41. `geglu_tanh_strided` — three pitches, and only one of them the width.
///
/// gemma4's per-layer-embedding GeGLU reads a NARROW gate out of a WIDE table:
/// the PLE table is `[rows, n_layers * ple_dim]`, so layer L's slice is
/// `ple_dim` wide with `n_layers * ple_dim` between rows, while the gate and
/// the output are densely `[rows, ple_dim]`. A byte offset cannot express
/// that, and the flat kernel reading one walks into the NEXT layers' slices
/// after the first row — not a crash and not even implausible numbers, since
/// those slices are the same table.
///
/// So all three pitches are DIFFERENT and none of them is the width. The
/// useless case to test is the production one where they are equal.
///
/// The output pitch is ODD, which is the only thing that reaches `store_half`:
/// a row then begins in the upper half of a word whose lower half is the
/// previous row's last element, written by a different workgroup at the same
/// moment, and the compare-exchange is what keeps both.
///
/// # Two things this dispatch is deliberately NOT doing, both measured
///
/// **The grid below is not the one this row's launch rule gives.** The row
/// states `LaunchRule::Elementwise`, which is `[width * rows, 1, 1]` LANES —
/// everything on x — while this variant alone is `@workgroup_size(16, 16)` and
/// reads `gid.y` as the ROW. Divided by a 16-high workgroup that is ONE group
/// on y, so only rows 0..15 ever launch. Dispatched that way at 21 rows on an
/// RTX 4090, row 16 column 0 comes back holding the sentinel it was born with
/// and the dispatch succeeds. `ElementwiseRows` is the shape this body wants
/// and is what is dispatched here; changing the row is a table edit and a
/// parity question, so it is reported rather than made.
///
/// **The header's stated x extent is one short at an odd `out_pitch`.**
/// `mlp/gated.wgsl` says "`gid.x` counts words of the output row, so the host's
/// x extent is `ceil(width / 2)`". At an odd pitch an ODD row's span starts in
/// a word's upper half and therefore straddles `ceil(width / 2) + 1` words.
/// Measured: at `width = 64` — where `ceil(width / 2)` is exactly two 16-wide
/// workgroups, so the round-up hides nothing — with pitches 89/71/67, row 1's
/// element 63 comes back as the sentinel. The shape below is saved only
/// because 23 rounds up to 32. That is the case `store_half` exists for, so
/// the extent that cannot cover it is the interesting half.
#[test]
fn a_strided_geglu_reads_three_different_pitches() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let width = 46usize;
    let rows = ROWS as usize;
    // Three pitches, all different, all wider than the width, and the output's
    // is odd.
    let gate_pitch = 79usize;
    let up_pitch = 53usize;
    let out_pitch = 61usize;
    for (a, b) in [
        (gate_pitch, up_pitch),
        (up_pitch, out_pitch),
        (gate_pitch, out_pitch),
    ] {
        assert_ne!(a, b, "equal pitches are the case that proves nothing");
    }
    assert_eq!(
        out_pitch % 2,
        1,
        "an odd output pitch is what reaches store_half"
    );

    let (gate, gate_seen) = bf16s(gpu, &spread(rows * gate_pitch, 2901));
    let (up, up_seen) = bf16s(gpu, &spread(rows * up_pitch, 2903));
    let out = sentinelled(gpu, (rows * out_pitch).div_ceil(2));
    // `GegluStridedParams { width, rows, gate_pitch, up_pitch, out_pitch }`,
    // a STORAGE struct at binding 3 — the row states `params: Buf`.
    let params = storage(
        gpu,
        &[width, rows, gate_pitch, up_pitch, out_pitch]
            .iter()
            .flat_map(|v| u32::try_from(*v).expect("fits").to_le_bytes())
            .collect::<Vec<u8>>(),
    );

    // The body is `@workgroup_size(16, 16)` and `gid.x` counts WORDS of the
    // output row, so the x extent is `ceil(width / 2)`.
    run(
        gpu,
        "geglu_tanh_strided_bfloat16",
        &[&gate, &up, &out, &params],
        &[],
        [over(width.div_ceil(2) as u32, 16), over(rows as u32, 16), 1],
    );

    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    let got = unpack(&read(gpu, &out), rows * out_pitch);
    for row in 0..rows {
        let base = row * out_pitch;
        let want: Vec<f32> = (0..width)
            .map(|k| {
                geglu_tanh_reference(gate_seen[row * gate_pitch + k], up_seen[row * up_pitch + k])
            })
            .collect();
        let what = format!("row {row}");
        agrees(&got[base..base + width], &want, &what).expect("the strided geglu agrees");
        if row == rows - 1 {
            refuses_a_perturbed_reference(&got[base..base + width], &want, &what);
        }
        // Between the width and the pitch is nobody's, and at an odd pitch the
        // boundary word is shared with the next row — so the compare-exchange
        // has to leave the padding exactly as it found it.
        for c in width..out_pitch {
            assert_eq!(
                got[base + c].to_bits(),
                sentinel.to_bits(),
                "row {row} column {c} is between the width ({width}) and the \
                 output pitch ({out_pitch}) and was written anyway",
            );
        }
    }

    // A body that read the gate and the up with the OUTPUT's pitch — the flat
    // kernel's mistake — must disagree.
    let mut by_out_pitch = Vec::with_capacity(rows * width);
    let mut flat = Vec::with_capacity(rows * width);
    for row in 0..rows {
        for k in 0..width {
            let at = row * out_pitch + k;
            by_out_pitch.push(geglu_tanh_reference(
                gate_seen[at.min(gate_seen.len() - 1)],
                up_seen[at.min(up_seen.len() - 1)],
            ));
            flat.push(got[at]);
        }
    }
    agrees(&flat, &by_out_pitch, "the strided geglu read at one pitch").expect_err(
        "three pitches that disagree is the whole point of this variant; if \
         reading all three at the output's also satisfies the device, the \
         shape is not separating them",
    );
}

/// A softmax attention with gpt-oss's learned SINK folded into the
/// denominator.
///
/// The sink is a per-head logit that joins the softmax with NO VALUE behind
/// it: it moves the running maximum and the denominator and contributes
/// nothing to the numerator. So it can only ever SHRINK the output, and by a
/// factor that is the same for every channel of a head — which is what makes a
/// body that divided by it instead of adding it to the denominator visible.
fn softmax_attention_with_sink(
    scores: &[f32],
    values: &[&[f32]],
    head_dim: usize,
    sink: f32,
) -> Vec<f32> {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let merged = m.max(sink);
    let weights: Vec<f32> = scores.iter().map(|s| (s - merged).exp()).collect();
    let z: f32 = weights.iter().sum::<f32>() + (sink - merged).exp();
    (0..head_dim)
        .map(|d| {
            let acc: f32 = weights.iter().zip(values).map(|(w, v)| w * v[d]).sum();
            rounded(acc / z)
        })
        .collect()
}

/// D42. `sdpa_paged_decode_sink` — the sink's value AND its direction.
///
/// The row is `sdpa_paged_decode`'s with one define, so every binding and
/// every offset is the same and the only difference is that `sinks` — bound
/// and unread by the no-sink arm — is now read. Both arms are dispatched over
/// identical inputs, so the comparison between them is exactly the sink's
/// contribution.
///
/// The DIRECTION is checked as well as the value, which rules out a body that
/// divided by the sink instead of adding its exponential to the denominator: a
/// sink can only shrink an output, by one factor per head, and never move it
/// away from zero.
#[test]
fn a_paged_decode_folds_its_sink_into_the_denominator() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let head_dim = 64usize;
    let q_heads = 6usize;
    let gqa = 3usize;
    let kv_heads = q_heads / gqa;
    let rows = 3usize;
    let page_size = 3usize;
    let pages = 7usize;
    let scale = 0.125f32;
    let window = 4i32;
    let mask_stride = 12u32;

    let positions = [6i32, 2, 9];
    let requests = [0i32, 1, 0];
    let indptr = [0u32, 4, 7];
    let indices = [6u32, 4, 2, 0, 5, 3, 1];

    let (queries, q_seen) = bf16s(gpu, &spread(rows * q_heads * head_dim, 3001));
    let pool = pages * page_size * kv_heads * head_dim;
    let (k_pages, k_seen) = bf16s(gpu, &spread(pool, 3009));
    let (v_pages, v_seen) = bf16s(gpu, &spread(pool, 3019));
    let position_ids = i32s(gpu, &positions);
    let req_of_token = i32s(gpu, &requests);
    let kv_page_indices = u32s(gpu, &indices);
    let kv_page_indptr = u32s(gpu, &indptr);
    let mask_bytes = vec![1u8; rows * mask_stride as usize];
    let attention_mask = storage(gpu, &mask_bytes);
    let attention_mask_enabled = storage(gpu, &[0u8; 4]);
    // One per HEAD, and spread wide enough that the shrink differs head to
    // head: a sink far below the scores changes almost nothing and one above
    // them halves the output.
    let sink_values: Vec<f32> = vec![-4.0, -1.0, 0.0, 0.75, 1.5, 3.0];
    let (sinks, sink_seen) = bf16s(gpu, &sink_values);

    let mut answers = Vec::new();
    for entrypoint in [
        "sdpa_paged_decode_bfloat16_d_64",
        "sdpa_paged_decode_sink_bfloat16_d_64",
    ] {
        let out = sentinelled(gpu, rows * q_heads * head_dim / 2);
        let block = Block::of(entrypoint)
            .i32("gqa_factor", gqa as i32)
            .i32("page_size", page_size as i32)
            .i32("n_kv_heads", kv_heads as i32)
            .f32("scale", scale)
            .u32("attention_mask_stride", mask_stride)
            .i32("window", window)
            .done();
        run(
            gpu,
            entrypoint,
            &[
                &queries,
                &k_pages,
                &v_pages,
                &out,
                &position_ids,
                &req_of_token,
                &kv_page_indices,
                &kv_page_indptr,
                &attention_mask,
                &attention_mask_enabled,
                &sinks,
            ],
            &block,
            [q_heads as u32, rows as u32, 1],
        );
        answers.push(unpack(&read(gpu, &out), rows * q_heads * head_dim));
    }

    let slot_of = |req: usize, kp: usize| -> usize {
        let phys = indices[indptr[req] as usize + kp / page_size] as usize;
        phys * page_size + kp % page_size
    };
    for row in 0..rows {
        let req = requests[row] as usize;
        let q_pos = positions[row];
        let start = if window > 0 && q_pos >= window {
            q_pos - window + 1
        } else {
            0
        };
        let keeps: Vec<usize> = (start..=q_pos).map(|kp| kp as usize).collect();
        for (q_head, sink) in sink_seen.iter().enumerate().take(q_heads) {
            let kv_head = q_head / gqa;
            let q_base = (row * q_heads + q_head) * head_dim;
            let scores: Vec<f32> = keeps
                .iter()
                .map(|kp| {
                    let base = (slot_of(req, *kp) * kv_heads + kv_head) * head_dim;
                    (0..head_dim)
                        .map(|d| scale * q_seen[q_base + d] * k_seen[base + d])
                        .sum()
                })
                .collect();
            let planes: Vec<&[f32]> = keeps
                .iter()
                .map(|kp| {
                    let base = (slot_of(req, *kp) * kv_heads + kv_head) * head_dim;
                    &v_seen[base..base + head_dim]
                })
                .collect();
            let plain = softmax_attention(&scores, &planes, head_dim);
            let want = softmax_attention_with_sink(&scores, &planes, head_dim, *sink);
            let what = format!("row {row} head {q_head} at sink {sink}");
            agrees(&answers[1][q_base..q_base + head_dim], &want, &what)
                .expect("the sunk decode attends");
            agrees(&answers[0][q_base..q_base + head_dim], &plain, &what)
                .expect("and the no-sink arm is unchanged by the same buffer");
            if row == rows - 1 && q_head == q_heads - 1 {
                refuses_a_perturbed_reference(&answers[1][q_base..q_base + head_dim], &want, &what);
            }

            // The DIRECTION. A sink adds a positive term to the denominator
            // and nothing to the numerator, so every channel shrinks toward
            // zero by ONE factor per head — never away from it, and never by a
            // different factor per channel.
            for d in 0..head_dim {
                let with = answers[1][q_base + d];
                let without = answers[0][q_base + d];
                assert!(
                    with.abs() <= without.abs() + (without.abs() / 64.0).max(1e-6),
                    "row {row} head {q_head} channel {d}: the sink moved {without} \
                     to {with}, which is AWAY from zero. A sink joins the \
                     softmax with no value behind it, so it can only shrink",
                );
            }
        }
    }
    // The largest sink must have shrunk its head noticeably, or this test is
    // comparing two tensors that were always going to agree.
    let hot = (q_heads - 1) * head_dim;
    let shrunk = (0..head_dim)
        .filter(|d| answers[1][hot + d].to_bits() != answers[0][hot + d].to_bits())
        .count();
    assert!(
        shrunk > head_dim / 2,
        "the head with the largest sink ({}) moved only {shrunk} of {head_dim} \
         channels; pick a sink comparable with the scores or this proves \
         nothing",
        sink_seen[q_heads - 1],
    );
}

/// D43. `sdpa_vector_decode_swa` — a window, two row pitches, and a causal end
/// that moves per row.
///
/// Three things this row has that `sdpa_vector_decode` does not, and one
/// dispatch reaches all three. The causal end is `n - (n_rows - 1 - row)`, so
/// a batch of query rows against one cache each stops at its OWN position
/// rather than at the last one; the window then moves the start forward, but
/// only where the row's history is longer than it; and `q_row_stride` and
/// `o_row_stride` are separate operands because gemma reads its query out of a
/// wider buffer than it writes.
///
/// So the window is 9 against ends of 9, 10 and 11: row 0 is UNCLAMPED and
/// rows 1 and 2 start one and two keys in. Three distinct key ranges, one of
/// which does not take the window branch at all.
///
/// The two pitches are 1600 and 1664 against a packed width of 1536, and both
/// differ from each other — a body that used one for the other stays in bounds
/// and reads the wrong row. The padding between the width and the output pitch
/// must keep its sentinel.
///
/// The block is 64 bytes, the widest in the table: two `i32`s, four
/// `vec2<u32>` strides that align to eight, then four more four-byte fields.
#[test]
fn a_sliding_window_decode_ends_each_row_at_its_own_position() {
    let Some((gpu, _held)) = adapter() else {
        return;
    };
    let entrypoint = "sdpa_vector_decode_swa_bfloat16_d_256";
    let head_dim = 256usize;
    let q_heads = 6usize;
    let gqa = 3usize;
    let kv_heads = q_heads / gqa;
    let n = 11usize;
    let rows = 3usize;
    let scale = 0.0625f32;
    let window = 9i32;
    let packed = q_heads * head_dim;
    let q_row_stride = packed + 64;
    let o_row_stride = packed + 128;
    assert_ne!(
        q_row_stride, o_row_stride,
        "one pitch for two tensors proves nothing"
    );

    let (queries, q_seen) = bf16s(gpu, &spread(rows * q_row_stride, 3101));
    // `[head][seq][channel]` for the keys.
    let k_head_stride = (n * head_dim) as u64;
    let k_seq_stride = head_dim as u64;
    let (keys, k_seen) = bf16s(gpu, &spread(kv_heads * n * head_dim, 3107));
    // `[seq][head][channel]` for the values — a different shape holding the
    // same count, so a body reading it with the key strides stays in bounds.
    let v_seq_stride = (kv_heads * head_dim) as u64;
    let v_head_stride = head_dim as u64;
    let (values, v_seen) = bf16s(gpu, &spread(n * kv_heads * head_dim, 3109));
    let out = sentinelled(gpu, (rows * o_row_stride).div_ceil(2));

    let block = Block::of(entrypoint)
        .i32("gqa_factor", gqa as i32)
        .i32("n", i32::try_from(n).expect("fits"))
        .wide("k_head_stride", k_head_stride)
        .wide("k_seq_stride", k_seq_stride)
        .wide("v_head_stride", v_head_stride)
        .wide("v_seq_stride", v_seq_stride)
        .f32("scale", scale)
        .i32("window", window)
        .i32("q_row_stride", i32::try_from(q_row_stride).expect("fits"))
        .i32("o_row_stride", i32::try_from(o_row_stride).expect("fits"))
        .done();
    assert_eq!(
        block.len(),
        64,
        "two i32s, four vec2<u32>s and four more words"
    );
    run(
        gpu,
        entrypoint,
        &[&queries, &keys, &values, &out],
        &block,
        [q_heads as u32, rows as u32, 1],
    );

    let sentinel = from_bf16((SENTINEL & 0xffff) as u16);
    let got = unpack(&read(gpu, &out), rows * o_row_stride);
    let mut ranges = Vec::new();
    for row in 0..rows {
        let n_row = n as i32 - (rows as i32 - 1 - row as i32);
        let kv_start = if window > 0 && n_row > window {
            n_row - window
        } else {
            0
        };
        ranges.push((kv_start, n_row));
        for q_head in 0..q_heads {
            let kv_head = q_head / gqa;
            let q_base = row * q_row_stride + q_head * head_dim;
            let o_base = row * o_row_stride + q_head * head_dim;
            let keeps: Vec<usize> = (kv_start..n_row).map(|i| i as usize).collect();
            let scores: Vec<f32> = keeps
                .iter()
                .map(|i| {
                    let k_base = kv_head * k_head_stride as usize + i * k_seq_stride as usize;
                    (0..head_dim)
                        .map(|d| scale * q_seen[q_base + d] * k_seen[k_base + d])
                        .sum()
                })
                .collect();
            let planes: Vec<&[f32]> = keeps
                .iter()
                .map(|i| {
                    let at = kv_head * v_head_stride as usize + i * v_seq_stride as usize;
                    &v_seen[at..at + head_dim]
                })
                .collect();
            let want = softmax_attention(&scores, &planes, head_dim);
            let what = format!("row {row} head {q_head} over keys [{kv_start}, {n_row})");
            agrees(&got[o_base..o_base + head_dim], &want, &what)
                .expect("the sliding decode attends");
            if row == rows - 1 && q_head == q_heads - 1 {
                refuses_a_perturbed_reference(&got[o_base..o_base + head_dim], &want, &what);
            }
        }
        // Between the packed width and the output pitch is nobody's.
        for c in packed..o_row_stride {
            assert_eq!(
                got[row * o_row_stride + c].to_bits(),
                sentinel.to_bits(),
                "row {row} column {c} is past the packed width ({packed}) and \
                 inside the output pitch ({o_row_stride}), so no head owns it",
            );
        }
    }
    assert_eq!(
        ranges,
        vec![(0, 9), (1, 10), (2, 11)],
        "one dispatch is supposed to cover three DIFFERENT key ranges, one of \
         them UNCLAMPED — row 0's history is exactly the window, so it does \
         not take the branch at all. If they coincide, this is one case run \
         three times",
    );
}

// ---------------------------------------------------------------------------
// The assertion that stops this suite quietly rotting.
// ---------------------------------------------------------------------------

/// This file, read back at compile time.
///
/// `tests/entrypoints.rs` scrapes `kernels-metal`'s SOURCE to hold two tables
/// against each other; this is the same trick turned inward. A list of "rows
/// this suite covers" that nothing checks is a list that keeps claiming
/// coverage after the test that provided it is deleted — so every claim below
/// names a string that must still BE here, and deleting a test deletes it.
const THIS_FILE: &str = include_str!("gpu.rs");

/// This file with every list that MENTIONS an entrypoint without dispatching
/// one cut out.
///
/// Searching the whole file is what the first draft did and it was worth
/// nothing. Two regions name entrypoints for reasons that are not a dispatch,
/// and both were enough on their own to make `contains` true whatever the rest
/// of the file said:
///
/// * [`COVERAGE`] itself, where every claimed name is a literal;
/// * the `let entrypoints = [...]` arrays of the four `*_modules_parse` tests,
///   which run `naga` over a module and never touch a device.
///
/// Deleting the value norm's dispatch left the check green against either of
/// them. With both cut it fails, which is what the check is for.
///
/// What remains is a dispatch or a mention in prose, and prose is a hole this
/// cannot close: a name written into a doc comment would stand in for the
/// dispatch it describes. That is not hypothetical — this very paragraph named
/// the entrypoint it was describing and had to stop. Said rather than hidden:
/// the check catches a deletion, not every possible way of lying about one, so
/// do not spell entrypoint names in the prose here.
fn body_without_the_lists() -> &'static str {
    /// Everything between `open` and the next `close`, removed, however many
    /// times it occurs.
    fn cut(text: &str, open: &str, close: &str, least: usize) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        loop {
            let Some(from) = rest.find(open) else {
                out.push_str(rest);
                return out;
            };
            out.push_str(&rest[..from]);
            let tail = &rest[from..];
            let to = tail.find(close).unwrap_or_else(|| {
                panic!("`{open}` is never closed by `{close}`; fix these markers")
            });
            assert!(
                to >= least,
                "the cut for `{open}` ran only {to} bytes before `{close}`, \
                 which is not the whole list — the search below would be \
                 reading the thing it is checking",
            );
            rest = &tail[to..];
        }
    }

    let without_table = cut(
        THIS_FILE,
        "const COVERAGE: &[(&str, Reached)] = &[",
        "\n];\n",
        2000,
    );
    // Leaked so the answer is `'static` like the input; a test binary owns it
    // until it exits, and this runs once.
    Box::leak(cut(&without_table, "    let entrypoints = [", "\n    ];\n", 100).into_boxed_str())
}

/// How this suite reaches one stated row.
enum Reached {
    /// Dispatched under a literal entrypoint name, which must appear in this
    /// file and must resolve to the row.
    By(&'static str),
    /// Dispatched under a `format!` template, because the row is a grid of
    /// entrypoints rather than one. The first string is the template's literal
    /// text, which must appear here; the second is one entrypoint it produces,
    /// which must resolve to the row.
    ByTemplate(&'static str, &'static str),
    /// Not dispatched, with the reason. There are none, and the variant stays
    /// because a row that genuinely cannot be dispatched should be NAMED
    /// rather than absent — an empty exclusion list is a claim, and a missing
    /// one is a silence.
    #[expect(dead_code, reason = "no stated row is currently unreachable")]
    Not(&'static str),
}

/// Every stated row, and where this file dispatches it.
///
/// The ORDER is the table's. A row added to `kernels-wgpu` and not classified
/// here fails `every_stated_row_is_dispatched_or_named` immediately, which is
/// the point: the failure mode this list exists for is a table that grows
/// while the suite does not.
const COVERAGE: &[(&str, Reached)] = &[
    ("add_bias", Reached::By("add_bias_bfloat16")),
    (
        "affine_qmm_t",
        Reached::ByTemplate(
            "affine_qmm_t_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32",
        ),
    ),
    (
        "affine_qmm_t_residual",
        Reached::ByTemplate(
            "affine_qmm_t_residual_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}",
            "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
        ),
    ),
    (
        "affine_qmv_fast",
        Reached::By("affine_qmv_fast_bfloat16_gs_64_b_4"),
    ),
    (
        "affine_qmv_fast_residual",
        Reached::ByTemplate(
            "affine_qmv_fast_residual_bfloat16_gs_{group}_b_{bits}",
            "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
        ),
    ),
    (
        "affine_qmv_routed",
        Reached::By("affine_qmv_routed_bfloat16_gs_64_b_4"),
    ),
    (
        "affine_qmv_routed_bias",
        Reached::By("affine_qmv_routed_bias_bfloat16_gs_64_b_4"),
    ),
    ("combine_sorted", Reached::By("combine_sorted")),
    (
        "embed_gather_4bit",
        Reached::ByTemplate(
            "embed_gather{}{}_4bit_bfloat16_gs_{group}_b_{bits}",
            "embed_gather_4bit_bfloat16_gs_64_b_8",
        ),
    ),
    (
        "embed_gather_mb_4bit",
        Reached::ByTemplate(
            "embed_gather{}{}_4bit_bfloat16_gs_{group}_b_{bits}",
            "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
        ),
    ),
    (
        "embed_gather_scaled_4bit",
        Reached::ByTemplate(
            "embed_gather{}{}_4bit_bfloat16_gs_{group}_b_{bits}",
            "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
        ),
    ),
    (
        "embed_gather_scaled_mb_4bit",
        Reached::ByTemplate(
            "embed_gather{}{}_4bit_bfloat16_gs_{group}_b_{bits}",
            "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
        ),
    ),
    ("geglu_tanh", Reached::By("geglu_tanh_bfloat16")),
    (
        "geglu_tanh_strided",
        Reached::By("geglu_tanh_strided_bfloat16"),
    ),
    ("gptoss_swiglu", Reached::By("gptoss_swiglu_bfloat16")),
    ("kv_append", Reached::By("kv_append_bfloat16")),
    ("kv_append_paged", Reached::By("kv_append_paged_bfloat16")),
    ("layer_scalar_mul", Reached::By("layer_scalar_mul_bfloat16")),
    ("logit_softcap", Reached::By("logit_softcap_bfloat16")),
    (
        "mxfp4_qmv_routed_bias",
        Reached::By("mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4"),
    ),
    ("neox_decode", Reached::By("neox_decode_bfloat16")),
    (
        "neox_freqs_decode",
        Reached::By("neox_freqs_decode_bfloat16"),
    ),
    ("neox_freqs_mb", Reached::By("neox_freqs_mb_bfloat16")),
    ("neox_mb", Reached::By("neox_mb_bfloat16")),
    ("neox_prop_decode", Reached::By("neox_prop_decode_bfloat16")),
    ("ple_combine", Reached::By("ple_combine_bfloat16")),
    ("residual_add", Reached::By("residual_add_bfloat16")),
    ("rms_residual", Reached::By("rms_residual_bfloat16")),
    (
        "rms_residual_scaled",
        Reached::By("rms_residual_scaled_bfloat16"),
    ),
    ("rms_single_row", Reached::By("rms_single_row_bfloat16")),
    ("route_gather", Reached::By("route_gather")),
    ("route_sort", Reached::By("route_sort")),
    ("router_topk", Reached::By("router_topk_bfloat16")),
    (
        "router_topk_scaled",
        Reached::By("router_topk_scaled_bfloat16"),
    ),
    ("row_gather", Reached::By("row_gather_bfloat16")),
    (
        "sdpa_paged_decode",
        Reached::By("sdpa_paged_decode_bfloat16_d_64"),
    ),
    (
        "sdpa_paged_decode_sink",
        Reached::By("sdpa_paged_decode_sink_bfloat16_d_64"),
    ),
    (
        "sdpa_vector_decode",
        Reached::By("sdpa_vector_decode_bfloat16_d_64"),
    ),
    (
        "sdpa_vector_decode_swa",
        Reached::By("sdpa_vector_decode_swa_bfloat16_d_256"),
    ),
    (
        "shared_expert_combine",
        Reached::By("shared_expert_combine"),
    ),
    (
        "shared_expert_combine_strided",
        Reached::By("shared_expert_combine_strided"),
    ),
    ("silu_mul", Reached::By("silu_mul_bfloat16")),
    ("split_qkv_bf16", Reached::By("split_qkv_bf16")),
    ("vnorm_single_row", Reached::By("vnorm_single_row_bfloat16")),
];

/// Whether [`COVERAGE`] says this suite dispatches `symbol`.
///
/// Called by [`run`] as well as by the test below, which is what closes the
/// other direction: a dispatch of a row nobody classified fails at the
/// dispatch, so the list cannot fall behind the file either way.
fn is_claimed(symbol: &str) -> bool {
    COVERAGE
        .iter()
        .any(|(row, how)| *row == symbol && !matches!(how, Reached::Not(_)))
}

/// D44. Every stated row is dispatched, or is named with a reason.
///
/// The number this suite covers has to be an ASSERTION and not a paragraph, or
/// it shrinks in silence: a test deleted in a refactor takes its row's coverage
/// with it and nothing says so. Four claims, and each catches a different way
/// for that to happen.
///
/// 1. [`COVERAGE`]'s rows are EXACTLY the table's stated rows, both directions.
///    A row added to `kernels-wgpu` fails here until somebody classifies it; a
///    row that lost its operands fails here too.
/// 2. Every entrypoint named resolves through `kernels_wgpu::sig` to the row it
///    is filed under. A copy-paste that filed `neox_freqs_mb` under `neox_mb`
///    is then a failure rather than a duplicate.
/// 3. Every name — or the `format!` template that builds it — still APPEARS in
///    this file. Deleting the test that dispatches a row deletes its string,
///    which is what makes this more than a list of intentions.
/// 4. The count, pinned. It is 44 of 44 with no exclusions; a row that
///    genuinely could not be dispatched would be `Reached::Not` with a reason,
///    and the count would have to move in the same edit.
///
/// Needs no adapter: it is a claim about the SUITE, and it should fail on the
/// build box too.
#[test]
fn every_stated_row_is_dispatched_or_named() {
    let mut stated: Vec<&str> = kernels_wgpu::KERNELS
        .iter()
        .filter(|row| !row.operands.is_empty())
        .map(|row| row.symbol)
        .collect();
    stated.sort_unstable();
    stated.dedup();

    let mut claimed: Vec<&str> = COVERAGE.iter().map(|(row, _)| *row).collect();
    claimed.sort_unstable();
    let mut once = claimed.clone();
    once.dedup();
    assert_eq!(claimed, once, "a row is classified twice");
    assert_eq!(
        claimed, stated,
        "this list and the table's stated rows have to be the same set. A row \
         that grew operands is a row this suite can now dispatch and does not; \
         a row that lost them is one it cannot",
    );

    let body = body_without_the_lists();
    let mut dispatched = Vec::new();
    let mut excluded = Vec::new();
    for (row, how) in COVERAGE {
        let (named, must_appear) = match how {
            Reached::By(entrypoint) => (*entrypoint, *entrypoint),
            Reached::ByTemplate(template, sample) => (*sample, *template),
            Reached::Not(why) => {
                excluded.push((*row, *why));
                continue;
            }
        };
        let sig = kernels_wgpu::sig(named)
            .unwrap_or_else(|| panic!("`{row}` names `{named}`, which is no entrypoint"));
        assert_eq!(
            sig.symbol, *row,
            "`{named}` belongs to row `{}` and is filed under `{row}`",
            sig.symbol,
        );
        assert!(
            body.contains(must_appear),
            "`{row}` is dispatched as `{must_appear}`, and that string appears \
             NOWHERE in this file outside the coverage table and the \
             parse-only lists — so whatever dispatched it has been deleted \
             and this list is the only thing still claiming coverage",
        );
        dispatched.push(*row);
    }

    assert!(
        excluded.is_empty(),
        "{} stated rows are not dispatched: {excluded:?}",
        excluded.len(),
    );
    assert_eq!(
        dispatched.len(),
        44,
        "this suite dispatches {} of the table's {} stated rows. The number is \
         pinned so that it SHRINKING is a failure rather than a silence — if a \
         row genuinely cannot be dispatched, say so with `Reached::Not` and a \
         reason, and change this number in the same edit",
        dispatched.len(),
        stated.len(),
    );
    // The other 56 rows of the table are UNSTATED: they carry axes and a name
    // and no operands, so no layout can be derived from them and this harness
    // cannot bind one. That is not a gap in the testing, it is a row with no
    // ABI — see `.wiki/new-driver/vulkan.md` §13.
    assert_eq!(
        kernels_wgpu::KERNELS.len() - stated.len(),
        56,
        "the unstated rows are the ones this suite structurally cannot reach, \
         and there are supposed to be 56 of them",
    );
}
