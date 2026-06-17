//! `pie` CLI binary — thin wrapper over [`pie_server::cli::dispatch`].
//!
//! Subcommand layout (mirrors the legacy `pie_cli`):
//!
//! ```text
//! pie serve   [--config --host --port --no-auth --debug --no-snapshot --monitor]
//! pie run     <inferlet> [...]
//! pie new     <name> [--ts -o <dir>]      # forwards to `python3 -m bakery create`
//! pie build   <path> -o <output>          # forwards to `python3 -m bakery build`
//! pie config  init|show|set
//! pie auth    add|remove|list
//! pie model   list|download|remove
//! pie driver  list | <type> {install,doctor,set,unset,show,exec}
//! pie doctor
//! pie check   <toml> [--debug]
//! pie smoke   [--rpc]
//! ```
//!
//! All of the work happens in `pie_server::cli::dispatch`.

// mimalloc as the global allocator: thread-cached, low contention,
// good performance for the burst-allocation pattern the scheduler +
// chain-extender pool produce.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(windows)]
fn main() {
    let handle = std::thread::Builder::new()
        .name("pie-main".to_string())
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .expect("spawn pie main thread");

    match handle.join() {
        Ok(()) => {}
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[cfg(not(windows))]
fn main() {
    run();
}

fn run() {
    if let Err(e) = pie_server::cli::dispatch() {
        eprintln!("pie: {e:#}");
        std::process::exit(1);
    }
}

// Graph-build regression test for the portable driver's uniform-top-K
// sampling reshape. The five per-arch graph builders share
// `build_sampling_outputs`, whose gather reshape must key on the slot count
// (`n_slots`) rather than the request count (`n_req`): pass-level
// speculation samples more than one slot per request (draft verification),
// so `n_slots >= n_req`. A test-only C entry in the portable driver builds
// the sampling subgraph and reports the slot dimension of `top_k_probs`.
//
// This lives in the `pie` bin because CI runs `cargo test -p pie-server
// --bin pie`, and the driver/portable ctest targets never run in CI.
#[cfg(all(test, feature = "driver-portable"))]
mod portable_graph_sampling_tests {
    use std::os::raw::c_int;

    unsafe extern "C" {
        fn pie_portable_test_uniform_top_slot_dim(
            n_req: c_int,
            n_slots: c_int,
            k: c_int,
            vocab: c_int,
        ) -> c_int;
    }

    fn slot_dim(n_req: i32, n_slots: i32, k: i32, vocab: i32) -> i32 {
        unsafe { pie_portable_test_uniform_top_slot_dim(n_req, n_slots, k, vocab) }
    }

    // The speculation case: more sampling slots than requests. If a future
    // change keys the gather reshape on n_req again, GGML's nelements
    // assert aborts this process instead of returning n_slots.
    #[test]
    fn uniform_top_reshape_keys_on_n_slots_under_speculation() {
        assert_eq!(slot_dim(2, 5, 8, 64), 5);
    }

    // Plain decode: one slot per request. The invariant must still hold.
    #[test]
    fn uniform_top_reshape_handles_n_slots_eq_n_req() {
        assert_eq!(slot_dim(3, 3, 4, 32), 3);
    }
}

// Mapper-coverage + hparams regression tests for the hybrid Qwen 3.5 / 3.6
// ("qwen35" / "qwen35moe") load path. The qwen35 GGUF loader is otherwise
// guarded only by real 27B/35B model boots (#694); these assert the two pure
// pieces it depends on — `gguf_to_hf_name` covering the qwen35-UNIQUE tensors
// (gated-delta-rule linear attention + shared expert; the shared arms are
// covered broadly by other arches' loads), and `parse_gguf_hparams` deriving
// `layer_types` + linear-attn dims — without a model load. Test-only C entries
// live in the portable driver (`qwen35_mapping_testhook.cpp`); they are
// exercised here, in the `pie` bin, because CI runs `cargo test -p pie-server
// --bin pie` and the driver/portable ctest targets never run in CI.
#[cfg(all(test, feature = "driver-portable"))]
mod portable_qwen35_mapping_tests {
    use std::os::raw::c_int;

    unsafe extern "C" {
        fn pie_portable_test_qwen35_mapper_missing_count() -> c_int;
        fn pie_portable_test_qwen35_layer_type_at(
            interval: c_int,
            num_layers: c_int,
            idx: c_int,
        ) -> c_int;
        fn pie_portable_test_qwen35_parsed_linear_dim(which: c_int) -> c_int;
    }

    // Every qwen35-UNIQUE HF tensor name `build_qwen3_5_` /
    // `load_qwen3_5_moe_layer_` look up must be producible by
    // `gguf_to_hf_name`. Dropping a linear-attn (`ssm_*`/`attn_qkv`/`attn_gate`)
    // or shared-expert (`*_shexp`) mapper arm makes this nonzero.
    #[test]
    fn mapper_covers_qwen35_unique_loader_tensors() {
        let missing = unsafe { pie_portable_test_qwen35_mapper_missing_count() };
        assert_eq!(
            missing, 0,
            "{missing} qwen35 loader tensor(s) have no gguf_to_hf_name mapping \
             (see stderr for names)"
        );
    }

    // layer_types: layer i is full attention ('g') iff (i+1) % interval == 0,
    // else linear ('l'). For interval=4, num_layers=6 → "lllgll".
    #[test]
    fn layer_types_follow_full_attention_interval() {
        let at = |i| unsafe { pie_portable_test_qwen35_layer_type_at(4, 6, i) };
        let g = i32::from(b'g');
        let l = i32::from(b'l');
        let pattern: Vec<i32> = (0..6).map(at).collect();
        assert_eq!(pattern, vec![l, l, l, g, l, l]);
    }

    // Linear-attention + shared-expert dims read from the ssm.* / expert_*
    // GGUF keys must land in the matching Hparams fields.
    #[test]
    fn linear_attn_dims_parse_from_ssm_keys() {
        let dim = |w| unsafe { pie_portable_test_qwen35_parsed_linear_dim(w) };
        assert_eq!(dim(0), 2, "num K heads (ssm.group_count)");
        assert_eq!(dim(1), 16, "num V heads (ssm.time_step_rank)");
        assert_eq!(dim(2), 128, "K/V head dim (ssm.state_size)");
        assert_eq!(dim(3), 4, "conv kernel (ssm.conv_kernel)");
        assert_eq!(dim(4), 512, "shared-expert intermediate size");
    }
}

// F3 regression guard: the GGUF tokenizer minter must emit the leading
// metaspace Prepend in the gemma SPM normalizer exactly when
// tokenizer.ggml.add_space_prefix is set. Without the Prepend a
// non-space-leading input loses its leading ▁ and its first piece
// tokenizes to different IDs than the reference gemma tokenizer. A
// test-only C entry returns the serialized normalizer node for a given
// flag. Lives in the `pie` bin because CI runs `cargo test -p pie-server
// --bin pie`; driver/portable ctest never runs in CI.
#[cfg(all(test, feature = "driver-portable"))]
mod portable_gguf_tokenizer_tests {
    use std::os::raw::{c_char, c_int};

    unsafe extern "C" {
        fn pie_portable_test_spm_normalizer_json(
            add_space_prefix: c_int,
            out: *mut c_char,
            cap: c_int,
        ) -> c_int;
    }

    fn normalizer_json(add_space_prefix: bool) -> String {
        let mut buf = vec![0u8; 512];
        let n = unsafe {
            pie_portable_test_spm_normalizer_json(
                add_space_prefix as c_int,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            )
        };
        assert!(n > 0, "normalizer testhook returned {n}");
        String::from_utf8(buf[..n as usize].to_vec()).unwrap()
    }

    // ▁ = U+2581, the SentencePiece metaspace marker.
    const META: &str = "\u{2581}";

    // gemma-4 ships add_space_prefix=false → bare Replace (matches
    // google/gemma-4 tokenizer.json); NO Prepend, NO Sequence.
    #[test]
    fn no_space_prefix_emits_bare_replace() {
        let j = normalizer_json(false);
        assert!(j.contains("\"Replace\""), "{j}");
        assert!(j.contains(META), "{j}");
        assert!(!j.contains("Prepend"), "bare normalizer must not Prepend: {j}");
        assert!(!j.contains("Sequence"), "bare normalizer must not be a Sequence: {j}");
    }

    // add_space_prefix=true → Sequence[Prepend ▁, Replace], Prepend FIRST so
    // a non-space-leading input keeps its leading metaspace.
    #[test]
    fn space_prefix_prepends_metaspace_before_replace() {
        let j = normalizer_json(true);
        assert!(j.contains("Sequence"), "{j}");
        assert!(j.contains("Prepend"), "{j}");
        assert!(j.contains("\"Replace\""), "{j}");
        assert!(j.contains(META), "{j}");
        let prepend_at = j.find("Prepend").unwrap();
        let replace_at = j.find("Replace").unwrap();
        assert!(prepend_at < replace_at,
                "Prepend must precede Replace so the leading metaspace is added first: {j}");
    }
}
