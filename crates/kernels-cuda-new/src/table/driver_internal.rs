//! Launchers the DRIVER reaches for directly — no DSL statement, no place
//! in the planner's vocabulary, and deliberately not rows of [`super::KERNELS`]:
//! `model`'s `kernels_table` holds that table and `dsl::cuda` to the same
//! set, and these have no statement a trace could record. The per-family
//! exhaustiveness tests classify them as `DriverInternal` for exactly this
//! reason.
//!
//! They are still LAUNCHES, and the Rust driver still has to make them —
//! which is what this second table is for. Same [`KernelSig`] rows, same
//! `kernels_cuda::abi::emit_c_shim` proof, same generated bindings; the only
//! difference is which invariant the table answers to. A row joins here when
//! a live seam or the executor needs a launcher the DSL surface correctly
//! lacks.

use kernels::kernel;
use kernels::{KernelSig, Source, operands};

#[rustfmt::skip]
pub static DRIVER_KERNELS: &[KernelSig] = &[
    // ── THE ENVELOPE TIER WAS HERE AND IT IS DELETED ─────────────────────
    //
    // Two rows: `envelope_seed` / `layout::launch_envelope_seed_empty_bf16`
    // and `envelope_merge_written` /
    // `layout::launch_envelope_merge_written_bf16`. They named the two
    // surviving launchers in `crates/kernels-cuda/csrc/src/layout/envelope.cu`
    // and `abi::emit_c_shim` generated a `pie_k_layout_*` body for each.
    //
    // **The file is gone and the launches are Rust.** They are
    // `driver-cuda/src/fire/envelope.rs`, firing
    // `families::layout::ENVELOPE`'s five `LaunchRule::Unstated` rows through
    // a driver-owned `kernels::Launch`. The three maintenance points the tier
    // has — seed, append-refresh, write-merge — are three functions there.
    //
    // WHY DELETION AND NOT `execution::RUST_SERVED`. That is the mechanism
    // the ported `table::attn` rows use, and it is unavailable here by
    // construction: `execution::tests::every_taken_over_row_is_stated`
    // resolves a taken-over symbol through `table::sig`, which scans
    // `super::TABLES`, and **`driver_internal` is deliberately not in
    // `TABLES`** (module header above). A `driver_internal` row has exactly
    // one close, which is to go, and `attn::copy_kv_cells_bf16` went the same
    // way in §59.1.
    //
    // THE CONSUMER EVIDENCE, on all five channels. `crates/model/src`: none,
    // for either symbol string or a `dsl::cuda` wrapper name — there can be
    // no wrapper, because `dsl::cuda` is generated from `TABLES`.
    // `model-compiler/src/dsl.rs` and `lower.rs::semantic()`: none, same
    // reason. Hand-written `ffi::pie_k_*` arms in `driver-cuda/src`: ONE,
    // `bind::abi::seed_envelopes_empty`, rewritten in the same change to call
    // `fire::envelope::seed_empty` — its own signature unchanged, so its
    // callers in `pools/kv_cache_live.rs` did not move. C++: `attn/kv_paged.cu`
    // called `launch_envelope_merge_written_bf16` (`:344`) and
    // `launch_envelope_update_appended_bf16` (`:145`), and both of those
    // launchers are ported in the same change; the three
    // `driver-cuda/tests/oracle/*/stub/layout/envelope.hpp` files define
    // their own `inline` bodies that LOG rather than launch and shadow the
    // real header, so they neither link against this nor notice it left.
    //
    // `layout::launch_envelope_update_appended_bf16` never had a row here at
    // all — it was reached only from C++ — which is why only two go.

    // The QKV split the generated bodies call ~390 times — the loud case
    // the attn exhaustiveness test names.
    kernel!(split_qkv "attn::split_qkv_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            k_out: BufMut <- Source::Out(1),
            v_out: BufMut <- Source::Out(2),
            n_tokens: I32 <- Source::Rows,
            // The two widths come off what is WRITTEN, not off the packed
            // operand: a `[N, q + 2*kv]` row cannot say where the cut
            // falls, and both results can.
            q_dim: I32 <- Source::OutWidth(0),
            kv_dim: I32 <- Source::OutWidth(1),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // ── `attn::split_qkv_bf16_devwin` WAS HERE and has MOVED to
    // `table::attn` ─────────────────────────────────────────────────────
    //
    // Moved, not deleted, and for `layout::embed_bf16`'s reason applied to
    // the same rule: this table holds *"launchers the driver fires with no
    // DSL statement"*, and `model-compiler/src/lower.rs:1503` names this
    // symbol from a statement. `execution::RUST_SERVED` is gated on
    // `table::sig` resolving and `table::TABLES` excludes this module, so a
    // row here can only ever be closed by deletion -- which is right for a
    // row nothing names and unavailable for a row a lowering does.
    // `driver-cuda/src/fire/split_packed.rs` is the launcher now, and with
    // it `attn/split_packed.cu` is deleted.

    // The page-mask packers `FirePageMask` fires.
    // ── `attn::pack_dense_mask` AND `attn::pack_structured_mask` WERE HERE ─
    //
    // Both rows are DELETED, with `attn/pack_dense_mask.cu` and its `.hpp`.
    // Not ported: §60.1's rule, *a port of a launcher with an empty consumer
    // set is a contract nobody signed*.
    //
    // THE SWEEP, all five channels, both symbols. `crates/model/src`: no hit
    // for either symbol string, and none for a `dsl::cuda` wrapper name —
    // there can BE no wrapper, because `dsl::cuda` is generated from `TABLES`
    // and `driver_internal` is deliberately not in it.
    // `model-compiler/src/dsl.rs`: none. `lower.rs::semantic()`: none.
    // Hand-written `ffi::pie_k_*` arms in `driver-cuda/src`: none — the eight
    // that exist name other symbols. C++ across all four extensions: the only
    // hits are `kernels-cuda-new/csrc/src/attn/{pack_dense_mask,qkv_fused,
    // attention_naive_paged}.cuh`, which are DEVICE text — the `__global__`s
    // themselves and two headers that mention the packed-bitmap format — and
    // not host callers.
    //
    // THE DEVICE ROWS STAY. `families/attn.rs:1206` and `:1222` are real
    // kernels with real device text, and a family row is a claim about what a
    // kernel IS rather than about who calls it. A caller that wants the
    // packed FlashInfer bitmap fires those two through `bind::jit::fire`; what
    // is gone is the claim that something already does.
    //
    // `bind::abi`'s `StructuredMaskParams` mirror and its `record!` entry are
    // left standing deliberately: the `static_assert`s they feed were in the
    // deleted `.cu`, so the layout agreement is now UNCHECKED on the C++
    // side, and the mirror is what a future caller would bind. Written down
    // rather than settled — see the report.

    // `attn::copy_kv_cells_bf16`'s row IS GONE, and it went with its
    // launcher in one edit — the rule this table's shape makes unavoidable.
    // A `driver_internal` row states `operands` and is in neither
    // `device::JIT_DISPATCHED` nor `execution::RUST_SERVED`, so `emit_c_shim`
    // writes a `pie_k_attn_copy_kv_cells_bf16` forwarder for it; once
    // `attn/kv_paged.cu`'s definition was deleted that forwarder would not
    // compile, and a shim entry pointing at a deleted launcher is the one
    // failure that stops the whole workspace.
    //
    // Routing was not an option and the reason is structural rather than a
    // judgement: a `driver_internal` row is not in `table::TABLES`, so
    // `table::sig` cannot resolve it and
    // `execution::tests::every_taken_over_row_is_stated` refuses to admit it
    // to `RUST_SERVED`; and its operands were all `Source::Unbound`, so
    // `abi::emit_dispatch` would have skipped it whole and left it with no
    // arm of either kind. Deletion is the only close, and the consumer set
    // makes it honest: one caller, `driver-cuda/src/serve/transfer.rs:321`,
    // which is Rust and now calls `fire::kv_paged::copy_kv_cells_bf16`.
    //
    // The two DEVICE rows are untouched — `families/attn.rs:3293`/`:3301` —
    // because they are what that Rust fires, and `SPECIALISATIONS`'
    // `COPY_KV_CELLS` still resolves `unit_of("attn::copy_kv_cells_bf16")`.
    // ── the ones a SEMANTIC op picks ──────────────────────────────
    //
    // No trace records a Launch naming these: the statement carries an
    // `OpKind`, and `lower()` reads the CUDA kernel off it. So the DSL
    // surface correctly lacks them and they are rows here, which is
    // what `every_lowered_kernel_has_a_bridge_row` found on its first
    // run and finds again whenever a semantic kind gains a reading.
    //
    // `norm::rmsnorm_bf16` stood here too and has left: the fan-out
    // pair is stated by `dsl::cuda::rmsnorm` now and its row moved to
    // `norm.rs`, where a text names it. That is the exit this table
    // wants for every row in this block — a row leaves when a statement
    // learns to say it.
    //
    // `gemm::act_x_w` also stood here — the quantized dispatch entry,
    // whose `WeightView` BY VALUE was the operand the handoff predicted
    // would be gemm's friction. It is gone, and the prediction was
    // answered rather than paid: the representation axis is FOUR named
    // rows now (`gemm.rs`'s tensor/channel/grouped/mxfp4 scaled entry
    // points), each one a symbol a statement chose, so nothing crosses
    // this ABI carrying a descriptor for the launcher to route on. What
    // the lowering still spells `gemm::act_x_w` is the DENSE matmul,
    // and the executor binds it to `gemm::act_x_wt_bf16` — which
    // `gemm.hpp` defines as `act_x_w` with `WeightView::raw(W, BF16)`,
    // the one view the dense arm ever built.
    // ── `layout::embed_bf16` WAS HERE and has MOVED to `table::layout` ───
    //
    // Not deleted — MOVED, and the difference is the whole point. This
    // table's own doc says its rows are *"launchers the driver fires with
    // no DSL statement"*, and that had stopped being true of this one:
    // `model-compiler/src/lower.rs:1462` lowers `Embed { .. }` to
    // `Semantic::Kernels(&["layout::embed_bf16"])`, so a statement DOES
    // name it, and `table::TABLES` excluding this module meant
    // `table::sig` could not resolve the symbol that statement names.
    //
    // It also unblocked the `.cu`. `execution::RUST_SERVED` requires
    // `table::sig(symbol)` to resolve, so a `driver_internal` row can only
    // ever be closed by DELETION — which is right for a row nothing names
    // (`attn::copy_kv_cells_bf16`, the two envelope rows,
    // `attn::pack_dense_mask`) and impossible for a row a lowering names.
    // In `table::layout` the row is takeable, and
    // `driver-cuda/src/fire/embed.rs` took it.

    // In place over the value it biases — one operand, one result, the
    // same bytes — so `out` binds from `Out(0)` and the staging comes off
    // the pair. The bias is the statement's named weight, like the
    // embedding's table.
    kernel!(add_bias "norm::add_bias_bf16", in_place = &[(0, 0)],
        operands = operands![
            out: BufMut <- Source::Out(0),
            bias: Buf <- Source::WeightNamed,
            num_rows: I32 <- Source::Rows,
            dim: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // qwen3_5's four, all read off a semantic kind the same way. They
    // arrived together because the family's declaration stopped naming
    // kernels and started naming what it MEANS — `GdnPrep`, `SplitQGate`,
    // `SigmoidGateMul`, `RmsnormGated` — which is the direction, and
    // leaves the reading to `lower()`.
    //
    // The post-conv prep, fused: q/k split and L2-normalized, v widened
    // to fp32, and g/beta gated — the three launches that used to sit
    // between the conv and the recurrent step. Its five fp32 outputs are
    // exactly the step's first five inputs, which is the shape of it.
    kernel!(gdn_post_conv_prep "ssm::qwen_gdn_post_conv_prep_bf16",
        operands = operands![
            qkv_post: Buf <- Source::In(0),
            a: Buf <- Source::In(1),
            b: Buf <- Source::In(2),
            a_log: Buf <- Source::WeightNamed,
            dt_bias: Buf <- Source::WeightNamed2,
            q_norm_kh: F32sMut <- Source::Out(0),
            k_norm_kh: F32sMut <- Source::Out(1),
            v_fp32: F32sMut <- Source::Out(2),
            g_log_out: F32sMut <- Source::Out(3),
            beta_out: F32sMut <- Source::Out(4),
            n: I32 <- Source::Rows,
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            conv_dim: I32 <- Source::Gdn("conv_dim"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // Full attention's q_proj packs the query and the per-token output
    // gate PER HEAD — `[N, heads, 2*head_dim]`, query first — so this is
    // strided by head, not a halves cut like `split_gate_up`. Three shape
    // arguments rather than one width, because the stride IS the layout.
    kernel!(split_q_gate "layout::split_q_gate_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            gate_out: BufMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            // Off the QUERY half, not the packed operand: `packed` is
            // `[N, heads, 2*head_dim]` and only the query's half of it
            // lands here, so the head count comes from what is written.
            num_heads: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::CtxNonZero("head_dim")),
            head_dim: I32 <- Source::Ctx("head_dim"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // That gate applied: `a' = a * σ(g)`, IN PLACE on operand 0 — the
    // header spells `x` "bf16, in-place" in as many words.
    kernel!(sigmoid_gate_inplace "mlp::sigmoid_gate_inplace_bf16",
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            gate: Buf <- Source::In(1),
            num_elements: I32 <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The gated norm with an FP32 `x`: the GDN recurrent step lands in
    // fp32, so this reads it there and the separate conversion launch
    // goes away. `x` and `weight` hold fp32 and are still `Buf` — the
    // header spells them `const void*`, and this table describes the
    // DECLARATION, not the contents. The shim initialises a function
    // pointer, so the spelling is what has to agree.
    // UNSOURCED, and the two numbers say why. The GDN landing norm runs
    // per (row, VALUE HEAD) over the trailing head width, so its rows are
    // `rows * gdn.v_h` and its width is `gdn.v_d` -- a PRODUCT of the
    // fire's rows and a context field, which no `Source::` spells. A row
    // that said `Rows` and `OutWidth(0)` would launch the right kernel
    // over the wrong rectangle, which is worse than having no row: the
    // hybrid's prefill found it immediately, and only because the walk
    // asserts every launch ran.
    kernel!(rmsnorm_gated_fp32_in "norm::rmsnorm_gated_fp32_in_bf16",
        operands = operands![
            x: Buf <- Source::In(0),
            gate: Buf <- Source::In(1),
            // BY NAME, not an arg slot: the hand arm resolved
            // `spec.weight` through the resolver.
            weight: Buf <- Source::WeightNamed,
            y: BufMut <- Source::Out(0),
            // ONE ROW PER (token, head), which is what the comment above
            // means by a rectangle no `Source::` spelled. It does now.
            num_rows: I32 <- Source::Mul(&Source::Rows, &Source::Gdn("v_h")),
            hidden: I32 <- Source::Gdn("v_d"),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // THE QWEN3-VL VISION TOWER IS GONE FROM THIS TABLE, and its absence is
    // the same point the gemma-4 pair below makes, made a third time.
    //
    // `vision::qwen3vl_scatter` stood here as a `whole = true` row: one row
    // that was a whole subgraph, bridged at tower granularity, whose wrapper
    // rebuilt a C++ weights struct from flat tables and whose walk and host
    // prep — bilinear pos-embed interpolation, 2-D rope ids, spatial-merge
    // reorder, the f32→bf16 pixel cast — were
    // `driver-cuda/csrc/vision/qwen3_vl_tower.cu`'s. It was the ONE row in
    // this file whose launcher lived in `driver-cuda/csrc/`, and the note
    // that stood here said what to do about that: **when the port lands,
    // this row is deleted in it — not routed and not served.**
    //
    // It landed. All three files (522 + 114 + 136 = 772 lines) went in one
    // commit, because they could not go separately: `qwen3_vl_tower.cu`
    // called `vis_helpers.cpp`'s two helpers and the C surface called the
    // walk. `driver-cuda/csrc/vision/` no longer exists and
    // `driver-cuda/build.rs` no longer runs nvcc for it.
    //
    // The mechanism is DELETION and not one of the other two, for the
    // reasons the note gave and which still hold:
    //
    //   * `RUST_SERVED` is for a symbol the generated dispatch must keep
    //     answering, and no model text ever named this one. The consumer set
    //     was a single hand-written call — `bridge_smoke.rs`'s
    //     `ffi::pie_k_vision_qwen3vl_scatter` — now a direct call to
    //     `driver_cuda::tower::qwen3_vl::scatter`, exactly as
    //     `the_gemma4_vision_tower_*` test was rewritten.
    //   * `JIT_DISPATCHED` is refused outright:
    //     `execution::tests::a_walk_is_only_a_walk` holds that a walk may
    //     DRIVE JIT'd kernels and may not BE one. The Rust walk drives
    //     nine of them.
    //
    // What replaced the row is not another row. It is sixteen fires of rows
    // that already existed in `families::vision` — `k_add_pe`, `k_bias`,
    // `k_f32_to_bf16`, `k_gelu_bias`, `k_gelu_erf`, `k_gelu_tanh`,
    // `k_layernorm`, `k_merge_gather`, `k_split_rope_qkv` — plus two cuBLAS
    // GEMM shapes and one FlashInfer prefill. That is the shape of every
    // tower under the owner's principle: the composition is Rust, and what a
    // table names is the kernels it composes.
    //
    // The stride-12 block table and stride-6 merger tables this row carried
    // are not lost: `tower::qwen3_vl::Weights::from_flat` reads them, with
    // the strides named rather than open-coded.
    // gemma-4's STANDALONE towers — the encode-ABI pair (host pixels /
    // log-mel in, HOST bf16 embedding rows out, anchor-segmented CSR).
    // Layer tables are `Ty::Bufs` at stride 41 (vision) / 62 (audio);
    // the field orders live in `vision/gemma4_towers_c.hpp`. The output
    // operands are HOST buffers — `PieEncodeDesc`'s own shape.
    // BOTH OF THE PAIR ARE GONE, and their absence is the point.
    // `vision::gemma4_vision_encode` and `vision::gemma4_audio_encode` stood
    // here. They were rows because a C++ host walk needed a shim entry to be
    // called from Rust; both walks are now Rust
    // (`driver-cuda/src/tower/gemma4_vision.rs` and `.../gemma4_audio.rs`),
    // so there is nothing for a shim entry to reach and no archive symbol to
    // resolve. A row kept "for the record" would be
    // `every_call_resolves_in_the_shim`'s 114-undefined-symbol failure
    // (§22.1) with extra steps: the shim would emit a call to
    // `kernels::vision::gemma4_*_encode`, and the translation units that
    // defined them are deleted.
    //
    // What replaced them is not one row but THIRTY-FOUR fires of rows that
    // already existed in `families::vision`, plus four the probe held and
    // one in `families::ssm`. That is the shape of every tower under the
    // owner's principle: the composition is Rust, and what a table names is
    // the kernels it composes.
    //
    // The stride-41 and stride-62 layer-table layouts these rows carried are
    // not lost — `model::shared::tower_names` owns the names and the order,
    // and `tower::gemma4_*::Weights::from_flat` reads them.
    // ── The W8A8 staging launchers `bind::quant_gemm` fires ──────────────
    //
    // Three rows added by §45's continuation, and the reason they are HERE
    // rather than in `super::KERNELS` is the header's rule read forwards: a
    // trace can state `gemm::act_x_wt_channel_scaled`, and it cannot state
    // "quantise the activation to int8 first". The staging steps are
    // interior to one row's body. The DSL surface correctly lacks them.
    //
    // They are rows AT ALL because the body that fires them is now Rust.
    // `gemm/gemm.cpp` called these three as C++-to-C++ and needed no entry
    // point; `driver-cuda/src/bind/quant_gemm.rs` calls them across the ABI
    // and needs one. Same launcher, same `.cu`, one generated forwarder.
    //
    // **These three are the ONLY reason the INT8 and FP8-blockwise arms are
    // in Rust today**, and they are a debt, not a destination: each has
    // migrated device text sitting in
    // `kernels-cuda-new/csrc/src/quant/quant_bf16_to_fp8.cuh`
    // (`quant_per_channel`, `w8a8_dequant`, `quant_act_fp8_per_group`) and
    // none has a JIT unit, because none of the three grids is a
    // `LaunchRule` this tree states:
    //
    //   * `quant_act_fp8_per_group` — `grid(ceil(k/gs), m)`, block 128.
    //   * `w8a8_dequant`            — `grid(ceil(N/32), ceil(M/8))`,
    //                                 block `(32, 8)`.
    //   * `quant_per_channel`       — `grid(rows)`, block `BLOCK`, with
    //                                 `ROW_REDUCE_SHMEM` dynamic shared
    //                                 memory, which no existing rule sizes.
    //
    // Give those three grids rules and all three rows leave this table for
    // `families::quant`, `bind::quant_gemm` fires them through
    // `bind::jit::fire` instead of through `ffi::pie_k_*`, and
    // `quant/quant_bf16_to_fp8.cu` loses its last consumer. Until then a
    // shim entry is what a Rust body has to reach them with.
    //
    // ALL THREE ARE GONE, AND ONLY ONE OF THEM NEEDED A RULE.
    //
    // * `quant::quantize_bf16_to_int8_per_token` never needed anything. It
    //   was a C++ forwarder onto `quantize_bf16_to_int8_per_channel`, whose
    //   `LaunchRule::Rms` row has been in `families::quant` since that family
    //   landed. `bind::quant_gemm` fires that row directly now, exactly as
    //   the paragraph above prescribes.
    // * `quant::dequant_int32_w8a8_to_bf16` and
    //   `quant::quantize_bf16_to_fp8_e4m3_per_token_group` did NOT get rules,
    //   because `new-horizon.md` §10.5 refuses vocabulary grown for one
    //   kernel and each of those two grids is one kernel — a 2-D block, and
    //   a grid axis that divides one operand by another. They are
    //   `LaunchRule::Unstated` rows of `families::quant` and
    //   `driver-cuda/src/fire/quant_int8.rs` states their rectangles, citing
    //   the `<<<>>>` each came from. That is `fire/attn_score.rs`'s escape
    //   hatch, which exists precisely so that "no rule states this" does not
    //   have to mean "invent a rule".
    //
    // The prediction the paragraph made held: with these three rows removed
    // the shim emits no `pie_k_quant_*` for the file and
    // `quant/quant_bf16_to_fp8.cu` had no consumer left in any language. It
    // is deleted, and `csrc/src/quant/` with it.
];
