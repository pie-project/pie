//! Differential proof for the `store/` layers ported in one pass:
//! recurrent-state addressing, swap copy planning, and the planner profile
//! key.
//!
//! TWO OF THE FIVE ARE RETIRED. `MLA geometry` and `DSV4 compressor geometry`
//! swept `layout::mla_geometry` and `layout::compressed_plane_geometry`, both
//! deleted with the MLA and compressed-plane pools they sized — no production
//! reader, and `Deployment::of` refuses every MLA/latent SKU by name. Their
//! rows are NOT re-blessed out of the golden, which the warning on
//! [`GOLDEN_FNV1A64`] forbids for good reason: they were the transcript's
//! PREFIX, and FNV-1a chains, so [`RETIRED_PREFIX_FNV1A64`] carries the hash
//! state they left and the surviving sweep still ends at the C++'s own value.
//!
//! Same protocol as the other parity harnesses -- build the real C++ as an
//! oracle, sweep both over one grid, require byte-identical output, pin the
//! **C++ side's** hash. One oracle binary covers all five so the whole batch
//! is proved by a single run.
//!
//! # The oracle
//!
//! None of these translation units compiles off-GPU as a whole, so `extract.sh`
//! lifts the relevant functions **verbatim** with `awk` and the oracle compiles
//! them against stub types carrying only the fields they read:
//!
//! | fragment | lifted from |
//! |---|---|
//! | `rec_strides.inc` | the three stride accessors in `recurrent_state_cache.hpp` |
//! | `rec_addr.inc` | `conv_state`, `recurrent_state_raw`, `mtp_pending_hidden` |
//! | `swap_helpers.inc` | `check_pairs`, `page_addr`, `submit_batch` |
//! | `swap_copy.inc` | the four `SwapPool::copy_*_async` |
//! | `profile.inc` | `key_to_json`, both `field_eq`, `key_matches` |
//!
//! The oracle sources live in `tests/oracle/store/` and regenerating the
//! golden is one command:
//!
//! ```text
//! crates/driver-cuda/tests/oracle/store/run.sh
//! ```
//!
//! which prints the three constants below. They are checked in rather than
//! described in prose because a golden nobody can regenerate is just a magic
//! number -- the pin is only worth having if a future reader can re-derive it
//! and find out *which* side moved.
//!
//! # The swap routines move real bytes
//!
//! The stub `cuda_runtime.h` implements `cudaMemcpyAsync` as a real `memcpy`
//! rather than a no-op, and the oracle allocates real host buffers for both
//! the "device" and pinned pools. So the extracted copy routines actually
//! move data, and the transcript records a hash of every pool afterwards.
//!
//! That is deliberately stronger than comparing computed offsets. An offset
//! comparison proves the plan matches; hashing the result proves the plan
//! *means* the same thing -- it would catch a src/dst transposition that
//! happened to produce a symmetric offset list, which is exactly the mistake
//! two same-typed `u32` index spaces invite.
//!
//! # Three outcomes, not two
//!
//! `RECADDR` and `RECMTP` encode `-2` for "the C++ threw", `-1` for "the C++
//! returned null", and an offset otherwise. Rust reproduces the split with a
//! panic and a `None` respectively, so [`RecurrentStateLayout`] has to
//! distinguish a full-attention layer (no state, an expected answer) from an
//! out-of-range index (a caller bug) exactly where the C++ does.

use driver_cuda::layout::profile_key::{KEY_FIELDS, ProfileKey, StoredField};
use driver_cuda::layout::recurrent_layout::{RecurrentShape, RecurrentStateLayout};
use driver_cuda::layout::swap_plan::{Direction, Pool, PoolGeometry, SwapPlan};
use std::fmt::Write as _;

/// FNV-1a 64 of the **C++ oracle's** whole stdout, all five layers.
const GOLDEN_FNV1A64: u64 = 0x05ae_9b51_0c01_1d57;
/// Byte count of the same output.
const GOLDEN_BYTES: usize = 2_132_663;
/// Row count, as a guard on the grid rather than on the values.
const GOLDEN_ROWS: usize = 80_235;

/// The hash state, bytes and rows the two retired layers left.
///
/// Not a second golden: the C++ printed `MLA` and `DSV4` first, so hashing
/// the surviving three FROM this state reaches [`GOLDEN_FNV1A64`] exactly and
/// they stay pinned to the C++ rather than to themselves. Taken by chaining
/// the full transcript this file rendered while all five layers were green.
const RETIRED_PREFIX_FNV1A64: u64 = 0x4f81_f3c1_f8fc_ed4e;
const RETIRED_PREFIX_BYTES: usize = 36_828;
const RETIRED_PREFIX_ROWS: usize = 1_322;

/// FNV-1a 64 continued from `h`, so a transcript can be hashed in pieces.
fn fnv1a64_from(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The same, from the empty state — what the swap sweep hashes its pools with.
fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_from(0xcbf2_9ce4_8422_2325, bytes)
}

// ---------------------------------------------------------------------------
// Recurrent state addressing
// ---------------------------------------------------------------------------
const STACKS: [&[bool]; 7] = [
    &[true],
    &[false],
    &[true, true],
    &[true, false, true],
    &[true, false, false, true, false, false, false, true],
    &[false, false, true, true],
    &[false, false, false, false],
];

/// Map a call into the oracle's three-way encoding: `-2` where the C++ throws
/// (Rust panics), `-1` where it returns null (Rust returns `None`), otherwise
/// the byte offset.
fn tri<F: FnOnce() -> Option<u64> + std::panic::UnwindSafe>(f: F) -> i64 {
    match std::panic::catch_unwind(f) {
        Err(_) => -2,
        Ok(None) => -1,
        Ok(Some(v)) => v as i64,
    }
}

fn render_recurrent(o: &mut String) {
    for st in STACKS {
        for bf16 in [false, true] {
            for slots in [1u32, 2, 16] {
                for ck in [2u32, 4] {
                    for cd in [8u32, 4096] {
                        for vh in [1u32, 32] {
                            for hk in [8u32, 128] {
                                for hv in [8u32, 128] {
                                    for hs in [0u32, 2048] {
                                        let l = RecurrentStateLayout::new(
                                            st,
                                            RecurrentShape {
                                                conv_dim: cd,
                                                conv_kernel: ck,
                                                v_heads: vh,
                                                head_k_dim: hk,
                                                head_v_dim: hv,
                                                hidden_size: hs,
                                                max_slots: slots,
                                                recurrent_is_bf16: bf16,
                                            },
                                        );
                                        let b = u8::from(bf16);
                                        let _ = writeln!(
                                            o,
                                            "RECSTRIDE\t{}\t{b}\t{slots}\t{ck}\t{cd}\t{vh}\t{hk}\t\
                                             {hv}\t{hs}\t{}\t{}\t{}",
                                            st.len(),
                                            l.conv_slot_stride_bytes(),
                                            l.recurrent_slot_stride_elems(),
                                            l.recurrent_slot_stride_bytes()
                                        );
                                        for layer in 0..st.len() {
                                            for s in 0..slots {
                                                let cs = tri(|| {
                                                    l.conv_state(layer as u32, s).map(|a| a.offset)
                                                });
                                                let rs = tri(|| {
                                                    l.recurrent_state(layer as u32, s)
                                                        .map(|a| a.offset)
                                                });
                                                let _ = writeln!(
                                                    o,
                                                    "RECADDR\t{}\t{b}\t{slots}\t{layer}\t{s}\t{cs}\t{rs}",
                                                    st.len()
                                                );
                                            }
                                        }
                                        for s in 0..=slots {
                                            let m =
                                                tri(|| l.mtp_pending_hidden(s).map(|a| a.offset));
                                            let _ = writeln!(o, "RECMTP\t{slots}\t{hs}\t{s}\t{m}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Out-of-bounds sweep, kept out of the main grid so it stays cheap. This is
/// the only place the throw-vs-null split is exercised for `conv_state` and
/// `recurrent_state`, and it pins the *order* of the two checks: the C++
/// validates the slot before it looks the layer up, so an out-of-range slot on
/// a full-attention layer throws rather than returning null.
///
/// The C++ also throws for negative indices. Those rows are absent because
/// both ports take `u32`, which makes a negative index a compile error rather
/// than a runtime one -- there is no Rust call to compare against.
fn render_recurrent_oob(o: &mut String) {
    for st in STACKS {
        let l = RecurrentStateLayout::new(
            st,
            RecurrentShape {
                conv_dim: 4096,
                conv_kernel: 4,
                v_heads: 32,
                head_k_dim: 128,
                head_v_dim: 128,
                hidden_size: 2048,
                max_slots: 4,
                recurrent_is_bf16: false,
            },
        );
        for layer in 0..=st.len() as u32 + 1 {
            for s in 0..=5u32 {
                let cs = tri(|| l.conv_state(layer, s).map(|a| a.offset));
                let rs = tri(|| l.recurrent_state(layer, s).map(|a| a.offset));
                let _ = writeln!(o, "RECOOB\t{}\t{layer}\t{s}\t{cs}\t{rs}", st.len());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Swap copy plans, executed on real memory
// ---------------------------------------------------------------------------
struct SwapCase {
    layers: u32,
    widths: &'static [u64],
    src: &'static [u32],
    dst: &'static [u32],
}

const SWAP_CASES: [SwapCase; 6] = [
    SwapCase {
        layers: 1,
        widths: &[64],
        src: &[0],
        dst: &[0],
    },
    SwapCase {
        layers: 1,
        widths: &[64],
        src: &[0, 1],
        dst: &[1, 0],
    },
    SwapCase {
        layers: 2,
        widths: &[64, 64],
        src: &[0, 2],
        dst: &[3, 1],
    },
    SwapCase {
        layers: 3,
        widths: &[32, 128],
        src: &[0, 1, 2],
        dst: &[4, 5, 6],
    },
    SwapCase {
        layers: 2,
        widths: &[16, 256],
        src: &[7],
        dst: &[0],
    },
    SwapCase {
        layers: 1,
        widths: &[64],
        src: &[],
        dst: &[],
    },
];

const NPAGES: u64 = 8;

/// Pool ids matching the oracle's registration: device `1000 + layer*10 + b`,
/// host `2000 + layer*10 + b`.
fn pool_id(p: Pool) -> i32 {
    match p {
        Pool::Device { layer, buffer } => 1000 + (layer as i32) * 10 + buffer as i32,
        Pool::Host { layer, buffer } => 2000 + (layer as i32) * 10 + buffer as i32,
    }
}

fn render_swap(o: &mut String) {
    for (ci, c) in SWAP_CASES.iter().enumerate() {
        for dir_code in 0..4 {
            let direction = match dir_code {
                0 => Direction::DeviceToHost,
                1 => Direction::HostToDevice,
                2 => Direction::DeviceToDevice,
                _ => Direction::HostToHost,
            };
            // Same fills as the oracle, so the hashes are comparable.
            let mut dev: Vec<Vec<Vec<u8>>> = Vec::new();
            let mut host: Vec<Vec<Vec<u8>>> = Vec::new();
            for layer in 0..c.layers {
                let mut dl = Vec::new();
                let mut hl = Vec::new();
                for (b, &w) in c.widths.iter().enumerate() {
                    let n = (w * NPAGES) as usize;
                    dl.push(
                        (0..n)
                            .map(|i| {
                                (0x10u32
                                    .wrapping_add(layer * 3)
                                    .wrapping_add(b as u32 * 7)
                                    .wrapping_add((i % 251) as u32))
                                    as u8
                            })
                            .collect(),
                    );
                    hl.push(
                        (0..n)
                            .map(|i| {
                                (0x80u32
                                    .wrapping_add(layer * 5)
                                    .wrapping_add(b as u32 * 11)
                                    .wrapping_add((i % 241) as u32))
                                    as u8
                            })
                            .collect(),
                    );
                }
                dev.push(dl);
                host.push(hl);
            }

            let _ = writeln!(o, "SWAPCASE\t{ci}\t{dir_code}");
            let geometry = PoolGeometry::new(vec![c.widths.to_vec(); c.layers as usize]);
            match SwapPlan::build(&geometry, direction, c.src, c.dst) {
                Ok(plan) => {
                    for op in plan.ops() {
                        // H2H goes through std::memcpy in the C++, which the
                        // stub cannot intercept, so it emits no SWAPOP rows --
                        // its correctness is carried entirely by the hashes.
                        if direction != Direction::HostToHost {
                            let _ = writeln!(
                                o,
                                "SWAPOP\t{}\t{}\t{}\t{}\t{}",
                                pool_id(op.dst),
                                op.dst_offset,
                                pool_id(op.src),
                                op.src_offset,
                                op.bytes
                            );
                        }
                        // Apply, one copy at a time, in plan order.
                        let take = |pools: &Vec<Vec<Vec<u8>>>, p: Pool| -> Vec<u8> {
                            let (l, b) = match p {
                                Pool::Device { layer, buffer } | Pool::Host { layer, buffer } => {
                                    (layer as usize, buffer as usize)
                                }
                            };
                            pools[l][b].clone()
                        };
                        let src_bytes = match op.src {
                            Pool::Device { .. } => take(&dev, op.src),
                            Pool::Host { .. } => take(&host, op.src),
                        };
                        let s = op.src_offset as usize;
                        let d = op.dst_offset as usize;
                        let n = op.bytes as usize;
                        let chunk = src_bytes[s..s + n].to_vec();
                        let target = match op.dst {
                            Pool::Device { layer, buffer } => {
                                &mut dev[layer as usize][buffer as usize]
                            }
                            Pool::Host { layer, buffer } => {
                                &mut host[layer as usize][buffer as usize]
                            }
                        };
                        target[d..d + n].copy_from_slice(&chunk);
                    }
                }
                Err(e) => {
                    let _ = writeln!(o, "SWAPERR\t{e}");
                }
            }
            for layer in 0..c.layers as usize {
                for b in 0..c.widths.len() {
                    let _ = writeln!(
                        o,
                        "SWAPHASH\t{layer}\t{b}\t{:016x}\t{:016x}",
                        fnv1a64(&dev[layer][b]),
                        fnv1a64(&host[layer][b])
                    );
                }
            }
        }
    }
    // The deliberate mismatch the oracle ends with.
    let g = PoolGeometry::uniform(1, 1, 64);
    let e = SwapPlan::build(&g, Direction::DeviceToHost, &[0, 1], &[0]).unwrap_err();
    let _ = writeln!(o, "SWAPERR\t{e}");
}

// ---------------------------------------------------------------------------
// Planner profile key
// ---------------------------------------------------------------------------
/// Two control-character cases, not one, because writing them in C++ is a
/// trap: `"ctrl\x01char"` there is *not* U+0001 followed by `char`. C++ lets
/// `\x` consume unboundedly many hex digits, so it reads `\x01c` -- U+001C --
/// and then `har`. Rust's `\x` takes exactly two. The first version of this
/// grid hit exactly that and the two sides disagreed on nine bytes; the fix
/// was to bound the escape with literal concatenation in the oracle and to
/// keep both code points, since both are now known to be interesting.
const GPU_NAMES: [&str; 10] = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA GeForce RTX 4090",
    "",
    "a\"b\\c",
    "tab\there",
    "line\nbreak",
    "ctrl\u{1}char",
    "ctrl\u{1c}har",
    "slash/es",
    "日本",
];

fn base_key() -> ProfileKey {
    ProfileKey {
        gpu_name: "NVIDIA H100 80GB HBM3".into(),
        compute_major: 9,
        compute_minor: 0,
        sm_count: 132,
        kv_cache_dtype: "bf16".into(),
        tp_size: 1,
        model_type: "llama".into(),
        hidden_size: 8192,
        num_hidden_layers: 80,
        num_attention_heads: 64,
        num_key_value_heads: 8,
        head_dim: 128,
    }
}

fn field_of(k: &ProfileKey, name: &str) -> StoredField {
    match name {
        "gpu_name" => StoredField::Str(k.gpu_name.clone()),
        "kv_cache_dtype" => StoredField::Str(k.kv_cache_dtype.clone()),
        "model_type" => StoredField::Str(k.model_type.clone()),
        "compute_major" => StoredField::Int(k.compute_major.into()),
        "compute_minor" => StoredField::Int(k.compute_minor.into()),
        "sm_count" => StoredField::Int(k.sm_count.into()),
        "tp_size" => StoredField::Int(k.tp_size.into()),
        "hidden_size" => StoredField::Int(k.hidden_size.into()),
        "num_hidden_layers" => StoredField::Int(k.num_hidden_layers.into()),
        "num_attention_heads" => StoredField::Int(k.num_attention_heads.into()),
        "num_key_value_heads" => StoredField::Int(k.num_key_value_heads.into()),
        "head_dim" => StoredField::Int(k.head_dim.into()),
        _ => StoredField::Missing,
    }
}

fn render_profile(o: &mut String) {
    for gn in GPU_NAMES {
        for major in [7, 9, 12] {
            for sm in [0, 132, -1] {
                let mut k = base_key();
                k.gpu_name = gn.into();
                k.compute_major = major;
                k.sm_count = sm;
                let _ = writeln!(o, "KEYJSON\t{}", k.to_json());
            }
        }
    }

    let k = base_key();
    let _ = writeln!(
        o,
        "KEYMATCH\tbaseline\t-\t{}",
        u8::from(k.matches(|n| field_of(&k, n)))
    );
    for field in KEY_FIELDS {
        for mode in 0..6 {
            let mutated = |name: &str| -> StoredField {
                if name != field {
                    return field_of(&k, name);
                }
                match mode {
                    0 => StoredField::Missing,
                    1 => StoredField::Null,
                    2 => StoredField::Bool(true),
                    3 => StoredField::Float(132.0),
                    4 => StoredField::Str("132".into()),
                    _ => StoredField::Int(132),
                }
            };
            let _ = writeln!(
                o,
                "KEYMATCH\t{field}\t{mode}\t{}",
                u8::from(k.matches(mutated))
            );
        }
    }
}

fn render() -> String {
    let mut o = String::with_capacity(GOLDEN_BYTES + 8192);
    render_recurrent(&mut o);
    render_recurrent_oob(&mut o);
    render_swap(&mut o);
    render_profile(&mut o);
    o
}

#[test]
fn every_store_layer_is_byte_identical_to_the_cpp_original() {
    // `tri` deliberately provokes panics; without this the transcript is
    // buried in backtrace noise.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = render();
    std::panic::set_hook(previous);

    assert_eq!(
        out.len() + RETIRED_PREFIX_BYTES,
        GOLDEN_BYTES,
        "output length drifted from the C++ oracle ({} vs {} bytes)",
        out.len() + RETIRED_PREFIX_BYTES,
        GOLDEN_BYTES
    );
    assert_eq!(
        fnv1a64_from(RETIRED_PREFIX_FNV1A64, out.as_bytes()),
        GOLDEN_FNV1A64,
        "output differs from the C++ oracle. Rebuild it per this file's module \
         docs and diff, but do NOT update the constant to match Rust -- the \
         golden is the C++'s output, and re-blessing it deletes the proof."
    );
}

#[test]
fn the_grid_still_covers_every_layer_it_claims_to() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = render();
    std::panic::set_hook(previous);

    assert_eq!(out.lines().count() + RETIRED_PREFIX_ROWS, GOLDEN_ROWS);
    let count = |k: &str| {
        out.lines()
            .filter(|l| l.starts_with(&format!("{k}\t")))
            .count()
    };
    for kind in [
        "RECSTRIDE",
        "RECADDR",
        "RECMTP",
        "RECOOB",
        "SWAPCASE",
        "SWAPOP",
        "SWAPHASH",
        "SWAPERR",
        "KEYJSON",
        "KEYMATCH",
    ] {
        assert!(
            count(kind) > 0,
            "no {kind} rows: a whole layer stopped being exercised"
        );
    }
    assert_eq!(count("SWAPCASE"), SWAP_CASES.len() * 4);
    assert_eq!(count("KEYJSON"), GPU_NAMES.len() * 3 * 3);
    assert_eq!(
        count("RECOOB"),
        STACKS.iter().map(|s| (s.len() + 2) * 6).sum::<usize>()
    );
    assert_eq!(count("KEYMATCH"), 1 + KEY_FIELDS.len() * 6);
}

/// Each layer's rows must actually vary. A transcript of constants would hash
/// consistently while proving nothing.
#[test]
fn every_layer_produces_varying_output() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = render();
    std::panic::set_hook(previous);

    let distinct = |k: &str| {
        let mut v: Vec<&str> = out
            .lines()
            .filter(|l| l.starts_with(&format!("{k}\t")))
            .collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    for (kind, min) in [("RECSTRIDE", 100), ("RECADDR", 100), ("SWAPHASH", 20)] {
        assert!(
            distinct(kind) >= min,
            "{kind} has only {} distinct rows",
            distinct(kind)
        );
    }

    // All three outcomes must appear, or the panic/null split is not being
    // exercised and the API distinction is unproven.
    let oob: Vec<&str> = out.lines().filter(|l| l.starts_with("RECOOB\t")).collect();
    assert!(
        oob.iter().any(|l| l.ends_with("\t-2\t-2")),
        "nothing ever went out of range"
    );
    assert!(
        oob.iter().any(|l| l.ends_with("\t-1\t-1")),
        "no full-attention layer probed"
    );
    assert!(
        oob.iter().any(|l| l
            .split('\t')
            .nth(4)
            .is_some_and(|v| v.parse::<i64>().unwrap_or(-1) > 0)),
        "no RECOOB row resolved to a real offset"
    );
    // Bounds win over the null: an out-of-range slot on a full-attention
    // layer is an error, not "this layer has no state". Getting that backwards
    // would silently swallow a bad slot index on exactly the layers where the
    // caller already expects a null.
    assert!(
        oob.iter().any(|l| l.starts_with("RECOOB\t3\t1\t4\t-2\t-2")),
        "an out-of-range slot on a full-attention layer must be an error"
    );

    // And the match sweep must produce both answers.
    let km: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("KEYMATCH\t"))
        .collect();
    assert!(
        km.iter().any(|l| l.ends_with("\t1")),
        "nothing ever matched"
    );
    assert!(
        km.iter().any(|l| l.ends_with("\t0")),
        "nothing ever failed to match"
    );
}

/// Ignored by default; run with `--ignored --nocapture` to regenerate the Rust
/// side for a byte diff against `cpp.txt`.
#[test]
#[ignore = "diagnostic dump, not an assertion"]
fn dump() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = render();
    std::panic::set_hook(previous);
    print!("{out}");
}
