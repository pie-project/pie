//===----------------------------------------------------------------------===//
//
// Every launch rule this backend added reproduces the `<<<>>>` it was
// derived from.
//
//===----------------------------------------------------------------------===//
//
// `runtime::launch`'s own test module proves the table is CONSISTENT: every
// rule some row states is ported, every variant of the vocabulary either
// evaluates or refuses in a sentence naming itself. Neither of those reads a
// `.cu` file. A rule can be listed, evaluated, agreed about by both halves of
// the crate and still launch a grid no kernel was ever launched with, and
// nothing in this repository would say so — the arithmetic would simply be
// wrong everywhere at once, consistently.
//
// So this file is the other half, and it is deliberately outside the module:
// these are transcriptions, not unit tests. Each case names a file and a
// kernel, quotes the launch expression it is checking, and asserts `eval`
// answers with that expression's three numbers. When a launcher changes, the
// case that disagrees names the launcher, and a reader who has never seen the
// rule can go and look.
//
// # Why the extents are what they are
//
// Two per rule at least, and one of the two is chosen to break something:
//
// * a rectangle whose element count is NOT a multiple of the block, because
//   `div_ceil` and a truncating divide agree on every extent that is;
// * a single row, because a rule that folded the row axis away answers
//   correctly for exactly one row and the fixture usually has more;
// * a head count the block width does not divide, because the recurrence's
//   shared allocation is a function of the head and not of the block, and a
//   rule that had crossed the two would only be caught where they differ.
//
// The numbers are model shapes rather than round ones. Qwen3-Next's GDN
// carries 32 value heads over a 128-channel key; DeepSeek's sparse indexer
// runs 2 048-token prefills; the hyper-connection expand is `hc_mult = 4` at
// hidden 2 048. A rule that passes at 128 x 128 and nowhere else has been
// tested against its own arithmetic.
//
// # What this file does not do
//
// It does not fire. `eval` is arithmetic over integers and wants no device —
// which is the property that lets a planner size a grid on a machine with no
// GPU, and the reason these cases run in CI where `tests/units.rs` cannot.
// The `#![cfg]` is there only because `runtime` is behind `_cuda`.
//
//===----------------------------------------------------------------------===//

#![cfg(feature = "_cuda")]

use kernels::LaunchRule as Rule;
use kernels_cuda_new::runtime::{Dims, Launch, Ungeometric, eval};

/// A rectangle with every field filled, so that a head-shaped rule under test
/// refuses for a reason the case chose rather than for an unset fixture.
///
/// The head counts differ — 32 query heads over 8 key heads — because a
/// fixture where they are equal cannot catch a rule reading the wrong one,
/// and three of the five rules below take a head count.
///
/// `stated_head_dim` is zero and `head_dim` is not, for the same reason: a
/// fixture where the STATEMENT's head width and the FIRE's are the same
/// number cannot catch [`Rule::RowsPerHead`] reading the wrong one. Zero is
/// also the honest default — `spec.per_head_dim` is `None` for every
/// `OpKind` in the tree but `RmsnormPerHead`.
///
/// `requests` and `altup_streams` are filled and are DIFFERENT from every
/// other count here, for the third time in the same sentence: `requests` is 4
/// against 32 query heads and 8 key heads, and `altup_streams` is 5 against
/// both. A fixture where a request count equals a row count cannot catch
/// [`Rule::PagedScores`] reading `rows` for its first axis, and a fixture
/// where a stream count equals `kv_heads` cannot catch the exact defect
/// [`Rule::AltUpStreams`] was written against. Cases that need the launcher's
/// own numbers override them.
fn dims(rows: u32, width: u32) -> Dims {
    Dims {
        rows,
        width,
        in_width: width,
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        stated_head_dim: 0,
        rotary_dims: 64,
        n_experts: 128,
        experts_per_token: 8,
        requests: 4,
        altup_streams: 5,
    }
}

/// `ssm/gated_delta_net.cu`, `gdn_recurrent_step_bf16`:
///
/// ```text
/// constexpr int BLOCK = 128;
/// dim3 grid(B, V_h);
/// dim3 block(BLOCK);
/// const int shmem_bytes = 2 * K_d * sizeof(float);
/// device::recurrent_step<device::bf16, true><<<grid, block, shmem_bytes, stream>>>(...);
/// ```
///
/// Three extents, and the second and third are the two ways this rule can be
/// wrong while looking right.
///
/// **Qwen3-Next's 32 value heads over a 128-channel key.** The head count
/// does not divide the block and the block does not divide the shared
/// allocation, so a rule that had derived either number from the other —
/// `2 * BLOCK * sizeof(float)`, which is 1 024 and would have matched at
/// `head_dim = 128` by coincidence — answers 1 024 here too and is caught by
/// the 256-channel case, where the contract is 2 048.
///
/// **The single row.** `B` is the token count and it reaches `grid.x`; a rule
/// that had put the head there instead is right about the block, right about
/// the shared memory, and launches a transposed grid that `recurrent_step`
/// reads as `b = blockIdx.x`.
#[test]
fn recurrent_scan_reproduces_the_gated_delta_net_launcher() {
    // 16 tokens, 8 value heads, 128-channel key: 2 * 128 * 4 = 1 024 bytes.
    assert_eq!(
        eval(Rule::RecurrentScan, dims(16, 2048)),
        Ok(Launch { grid: [16, 8, 1], block: [128, 1, 1], smem: 1024 })
    );

    // A 256-channel key: the allocation moves with `K_d` and not with the
    // block, which stays at the launcher's 128.
    let wide = Dims { kv_heads: 32, head_dim: 256, ..dims(16, 2048) };
    assert_eq!(
        eval(Rule::RecurrentScan, wide),
        Ok(Launch { grid: [16, 32, 1], block: [128, 1, 1], smem: 2048 })
    );

    // One token, the decode step this kernel exists for. The row is `grid.x`.
    assert_eq!(
        eval(Rule::RecurrentScan, dims(1, 2048)),
        Ok(Launch { grid: [1, 8, 1], block: [128, 1, 1], smem: 1024 })
    );
}

/// A recurrence over no value heads, or over a key of no channels, is
/// refused rather than launched.
///
/// Both are `Ungeometric::Empty` and they fail differently if they are not:
/// zero heads is a grid axis of zero, which CUDA accepts and reports success
/// for, and zero channels is a full grid of blocks whose stride loops run
/// zero times and whose `sq`/`sk` share one null offset. Neither raises
/// anything; both leave the recurrent state exactly as the previous fire left
/// it, which for a scan is a model that has stopped advancing.
#[test]
fn recurrent_scan_refuses_a_headless_rectangle() {
    assert_eq!(
        eval(Rule::RecurrentScan, Dims { kv_heads: 0, ..dims(16, 2048) }),
        Err(Ungeometric::Empty)
    );
    assert_eq!(
        eval(Rule::RecurrentScan, Dims { head_dim: 0, ..dims(16, 2048) }),
        Err(Ungeometric::Empty)
    );
}

/// `attn/kv_paged.cu`, `write_kv_to_pages_bf16`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// const int launch_tokens = total_tokens - first_token;
/// if (launch_tokens <= 0) return;
/// device::write_kv<true><<<launch_tokens, BLOCK, 0, stream>>>(...);
/// ```
///
/// The three numbers are one block per row, 256, and nothing shared, and the
/// case that matters is the last one: this grid is `Rms`'s and this block is
/// `Rms`'s, so the only observable difference between stating `PerRow` and
/// stating `Rms` today is the 32 bytes `Rms` requests for a fold that a
/// scatter does not perform. Asserting `smem: 0` is what makes the two rules
/// distinguishable by a test rather than only by a doc.
///
/// The width is varied across the two extents and deliberately does not move
/// anything. `write_kv` takes its row extent from `h_kv` and `head_dim` as
/// OPERANDS — the page geometry's, not the rectangle's — so a rule that had
/// sized the block on `Dims::width` in the manner of `route_rows` would pass
/// a single-width fixture and diverge here.
#[test]
fn per_row_reproduces_the_paged_kv_write_launcher() {
    assert_eq!(
        eval(Rule::PerRow, dims(2048, 1024)),
        Ok(Launch { grid: [2048, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // One token: the decode append, which is what this launcher runs for on
    // every step of every request.
    assert_eq!(
        eval(Rule::PerRow, dims(1, 4096)),
        Ok(Launch { grid: [1, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // A row count the block does not divide changes nothing, because the row
    // is the GRID here and not the thread.
    assert_eq!(
        eval(Rule::PerRow, dims(257, 1024)),
        Ok(Launch { grid: [257, 1, 1], block: [256, 1, 1], smem: 0 })
    );
}

/// `ssm/causal_conv1d.cu`, `causal_conv1d_prefill_bf16` through
/// `prefill_dispatch`:
///
/// ```text
/// if (N <= 0 || C <= 0 || K <= 0) return;
/// constexpr int BLOCK = 64;
/// dim3 grid(C);
/// dim3 block(BLOCK);
/// device::causal_conv1d_prefill<device::bf16, SILU><<<grid, block, 0, stream>>>(...);
/// ```
///
/// `C` is the WIDTH and it is the grid — the transpose of every other rule in
/// the file, and the assertion that catches a rule which put the row there
/// out of habit. The row count is varied by a factor of 1 500 across these
/// cases and the launch does not move: `N` is a loop bound inside the block,
/// `for (t = threadIdx.x; t < N; t += blockDim.x)`, because the trailing `K`
/// tokens of each channel are written back to `state_out` by thread zero
/// after a `__syncthreads()` the grid cannot provide.
///
/// The 64 is asserted rather than `BLOCK`'s 256 because the two are both
/// legal for this kernel and only one of them is what ran: the width is a
/// measurement here, not a contract, which is exactly the kind of number a
/// port silently rounds to the house default.
#[test]
fn per_channel_reproduces_the_causal_conv1d_prefill_launcher() {
    // A 2 048-token prefill over 5 120 conv channels.
    assert_eq!(
        eval(Rule::PerChannel, dims(2048, 5120)),
        Ok(Launch { grid: [5120, 1, 1], block: [64, 1, 1], smem: 0 })
    );

    // One token, and a channel count that is not a multiple of the block.
    assert_eq!(
        eval(Rule::PerChannel, dims(1, 100)),
        Ok(Launch { grid: [100, 1, 1], block: [64, 1, 1], smem: 0 })
    );
}

/// A convolution over no channels is refused, and so is one over no tokens.
///
/// The first is the launcher's `C <= 0` and would be a grid of nothing. The
/// second is its `N <= 0` and is the worse of the two: a full grid of blocks
/// whose stride loop runs zero times, which writes no output AND leaves
/// `state_out` holding the previous sequence's tail, so the next prefill
/// convolves against a history that belongs to another sequence.
#[test]
fn per_channel_refuses_an_empty_rectangle() {
    assert_eq!(eval(Rule::PerChannel, dims(2048, 0)), Err(Ungeometric::Empty));
    assert_eq!(eval(Rule::PerChannel, dims(0, 5120)), Err(Ungeometric::Empty));
}

/// `norm/dsv4_hc.cu`, `hc_post_bf16` and `hc_expand_bf16`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// const long long total = static_cast<long long>(N) * hidden_size;
/// if (total <= 0 || hc_mult > device::MAX_HC_MULT) return;
/// const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
/// device::hc_expand<device::bf16><<<grid, BLOCK, 0, stream>>>(...);
/// ```
///
/// `hidden_size` is the INPUT width — both kernels read `[N, H]`, write
/// `[N, M, H]`, and guard on `idx >= N * H` — so the fixture's `width` and
/// `in_width` are deliberately made to differ by the hyper-connection
/// multiplier. That is the whole of what this case tests: at `hc_mult = 4`,
/// stating `Elementwise` answers 128 blocks where the launcher ran 32, and
/// only a fixture whose two widths differ can tell them apart.
///
/// The partial tile is the second case and it is not decorative — a
/// truncating divide answers 8 blocks for 2 304 elements and leaves the last
/// 256 of them holding whatever the previous layer wrote into the residual.
#[test]
fn elementwise_in_reproduces_the_hyper_connection_launcher() {
    // 16 tokens, hidden 2048, hc_mult 4: the launcher covered 16 * 2048.
    let expand = Dims { width: 4 * 2048, in_width: 2048, ..dims(16, 2048) };
    assert_eq!(
        eval(Rule::ElementwiseIn, expand),
        Ok(Launch { grid: [128, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // A single row at a width the block does not divide: 2 304 elements is
    // nine blocks, the last of them nine-tenths idle.
    let ragged = Dims { width: 4 * 2304, in_width: 2304, ..dims(1, 2304) };
    assert_eq!(
        eval(Rule::ElementwiseIn, ragged),
        Ok(Launch { grid: [9, 1, 1], block: [256, 1, 1], smem: 0 })
    );
}

/// An input rectangle of no elements is refused.
///
/// `total <= 0` is the launcher's own guard and it returns without launching;
/// the rule says the same thing where a caller can see it, because a grid of
/// zero blocks is a `cuLaunchKernel` that succeeds.
#[test]
fn elementwise_in_refuses_an_empty_input_extent() {
    let no_input = Dims { width: 8192, in_width: 0, ..dims(16, 2048) };
    assert_eq!(eval(Rule::ElementwiseIn, no_input), Err(Ungeometric::Empty));
}

/// `attn/dsa_indexer.cu`, `dsa_index_topk_mask`:
///
/// ```text
/// if (tokens <= 0) return;
/// const std::size_t smem = static_cast<std::size_t>(tokens) * sizeof(float);
/// device::index_topk_mask<bf16><<<tokens, device::kBlock, smem, stream>>>(...);
/// ```
///
/// `kBlock` is 256, so this is `Rms`'s grid and `Rms`'s block with an
/// allocation that is a function of the ROW COUNT — `index_topk_mask` fills
/// `logit[0..blockIdx.x]`, one float per causal key, and every key is a token
/// of this same rectangle.
///
/// The two extents are chosen so that no constant can pass both. At 2 048
/// tokens the request is 8 192 bytes and at 4 096 it is 16 384: a rule that
/// had taken `Rms`'s 32 bytes, or the block's 1 024, or any fixed number,
/// answers the same at both and is caught. The 4 096 case is also a real
/// prefill — DeepSeek's indexer runs at that length — and its 16 KB is a
/// third of the 48 KB the rule deliberately does not clamp: a little under
/// 12 288 rows the launch starts being refused by the driver, loudly, which
/// is the only direction of shared-memory error that reports itself.
#[test]
fn row_scores_reproduces_the_sparse_indexer_launcher() {
    assert_eq!(
        eval(Rule::RowScores, dims(2048, 4096)),
        Ok(Launch { grid: [2048, 1, 1], block: [256, 1, 1], smem: 8192 })
    );
    assert_eq!(
        eval(Rule::RowScores, dims(4096, 4096)),
        Ok(Launch { grid: [4096, 1, 1], block: [256, 1, 1], smem: 16384 })
    );

    // One token, which is what every decode step of a sparse-attention model
    // fires: four bytes, and the block still 256.
    assert_eq!(
        eval(Rule::RowScores, dims(1, 4096)),
        Ok(Launch { grid: [1, 1, 1], block: [256, 1, 1], smem: 4 })
    );
}

/// The five new rules answer with a shared allocation the OLD rules would
/// have got wrong, and this states by how much.
///
/// It is a regression test for the near-miss that motivated the whole change
/// rather than for any one rule: `families/ssm.rs` recorded that `Rule::Rope`
/// gives the recurrence `K_d * sizeof(float)` — exactly half — and a
/// half-sized `extern __shared__` is not a launch failure. The kernels take
/// their second array as `smem + K_d`, so the key is staged over the query,
/// the step consumes a key it has already clobbered, and the recurrent state
/// absorbs the result for every token after it in the fire.
///
/// If a future edit ever makes these two agree, this fails here rather than
/// in a model whose GDN layers have quietly stopped being the model's.
#[test]
fn the_recurrence_allocation_is_twice_the_rotations() {
    let d = dims(16, 2048);
    let scan = eval(Rule::RecurrentScan, d).expect("ported");
    let rope = eval(Rule::Rope, d).expect("ported");
    assert_eq!(
        scan.smem,
        2 * d.head_dim * 4,
        "the scan stages `q` and `k`, two head-wide float arrays"
    );
    assert_eq!(rope.smem * 2, scan.smem, "`Rope` is half, which is the near miss");
}

/// The two smem-carrying rules size their allocation in `float`s at every
/// instantiation, whatever element the row names.
///
/// This is a geometry test for a `Ty`-shaped hazard, and it is here because
/// the offline device typecheck found four rows whose operand list disagrees
/// with the `__global__` on exactly this axis — a kernel with buffers of two
/// different element types, spelled with one row `elem`. Both rules below sit
/// on such kernels. `recurrent_step` is `template <typename StateT, bool
/// KLast>` with `float*` inputs and an `extern __shared__ float` slab, so
/// `StateT = bf16` changes the state's storage and nothing about the launch;
/// `index_topk_mask` is `template <class T>` over bf16 operands and stages
/// fp32 logits.
///
/// So `Dims` carries no element width, no rule reads one, and a future edit
/// that scaled either allocation by a row's `elem` would halve the slab for
/// precisely the bf16 instantiations — a launch that under-sizes shared
/// memory does not fail, it reads another block's floats. Stated as an
/// assertion because the doc that says it can be edited without anything
/// noticing.
#[test]
fn the_shared_allocations_are_counted_in_floats() {
    let d = dims(2048, 4096);
    assert_eq!(
        eval(Rule::RecurrentScan, d).expect("ported").smem,
        2 * d.head_dim * 4,
        "the recurrence stages `sq` and `sk` as floats in both StateT arms"
    );
    assert_eq!(
        eval(Rule::RowScores, d).expect("ported").smem,
        d.rows * 4,
        "the indexer stages fp32 logits whatever `T` its operands are"
    );

    // The same rectangle at a narrower nominal element must not move either
    // number, because nothing in `Dims` spells an element at all — this is
    // the assertion, not the fixture: if `Dims` ever gains an element width,
    // it fails here rather than in a fire that read half a slab.
    let same = Dims { rows: 2048, width: 4096, ..dims(2048, 4096) };
    assert_eq!(eval(Rule::RecurrentScan, same), eval(Rule::RecurrentScan, d));
    assert_eq!(eval(Rule::RowScores, same), eval(Rule::RowScores, d));
}

///
/// `RouterLane` is why this test exists at all: it dropped the row axis, a
/// mixture prefill routed row 0 and ran every other row through the first
/// row's experts, and every review passed because the launch was plausible.
/// Four of the five rules put the row on `grid.x` and `PerChannel` puts it in
/// a stride loop the launcher itself wrote — so for that one the assertion is
/// the opposite, and it is stated here rather than left implicit precisely
/// because "this rule ignores its rows" is the claim that needs a witness.
#[test]
fn every_new_rule_states_what_it_does_with_the_row_axis() {
    for rule in [Rule::RecurrentScan, Rule::PerRow, Rule::ElementwiseIn, Rule::RowScores] {
        let one = eval(rule, dims(1, 2048)).expect("ported");
        let many = eval(rule, dims(4096, 2048)).expect("ported");
        assert_ne!(one, many, "{rule:?} answers the same for 1 row and 4096");
    }

    // `PerChannel` is the deliberate exception: `C` is the grid and `N` is a
    // loop bound, because a channel's state crosses its tokens.
    assert_eq!(
        eval(Rule::PerChannel, dims(1, 5120)),
        eval(Rule::PerChannel, dims(4096, 5120)),
        "the convolution's grid is its channel count, at every sequence length"
    );
}

//===----------------------------------------------------------------------===//
//
// The seven rules that landed in the second pass.
//
// Same discipline as everything above: a file, a kernel, the launch
// expression quoted, and `eval`'s three numbers asserted against it. Two of
// them open ground the module header had recorded as closed — `Tile16`'s
// block is not one-dimensional and `AxialRope`'s grid has a third axis — so
// each of those carries an extra case asserting the axis the old shape would
// have collapsed.
//
//===----------------------------------------------------------------------===//

/// `norm/rmsnorm.cu`, `rmsnorm_gated_bf16` at `311-313`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// dim3 grid(num_rows);
/// dim3 block(BLOCK);
/// device::rmsnorm_gated<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
///     ..., num_rows, hidden, eps);
/// ```
///
/// The launcher takes `num_rows` as an ARGUMENT and `table/norm.rs:36` says
/// what the caller passes — `IfPresent(PerHeadDim, Rows * width/head_dim,
/// Rows)` — so the rule's whole content is that conditional, which is why
/// both arms are asserted here rather than only the interesting one.
///
/// **The conditional's subject is [`Dims::stated_head_dim`], not
/// [`Dims::head_dim`].** `Source::IfPresent` asks whether the STATEMENT named
/// a per-head width; the fire's attention head width cannot answer it, and
/// `the_absent_arm_is_not_the_fires_head_width` below is what happens when it
/// is asked to. The fixture leaves `head_dim: 128` set throughout so that a
/// rule reading the wrong field cannot pass by accident.
///
/// **gemma-4's per-head norm at 2 048 channels over 128-wide heads.** Sixteen
/// blocks per row, and the extent is chosen so that a rule which forgot the
/// multiply answers 16 for the wrong reason: `rows` is also 16. The 4 096-wide
/// case moves them apart.
///
/// **A width the head does not divide is refused.** The kernel norms
/// `head_dim` channels from `block * head_dim`, so a rounded-up grid runs its
/// last block off the end of the row and a rounded-down one leaves a head
/// unnormalised. 2 049 is prime to 128 and neither is a shape a statement can
/// have meant.
#[test]
fn rows_per_head_reproduces_the_rmsnorm_launcher() {
    // 16 rows of 2 048 over a STATED 128-wide head: 16 * 16 = 256 blocks.
    assert_eq!(
        eval(Rule::RowsPerHead, Dims { stated_head_dim: 128, ..dims(16, 2048) }),
        Ok(Launch { grid: [256, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // The same rectangle 4 096 wide: 32 heads a row, 512 blocks. A rule that
    // had dropped either factor answers 16 or 32 here.
    assert_eq!(
        eval(Rule::RowsPerHead, Dims { stated_head_dim: 128, ..dims(16, 4096) }),
        Ok(Launch { grid: [512, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // One token, a stated 64-wide head — a decode of gemma-4's q projection.
    let decode = Dims { stated_head_dim: 64, ..dims(1, 4096) };
    assert_eq!(
        eval(Rule::RowsPerHead, decode),
        Ok(Launch { grid: [64, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // THE ABSENT ARM. `stated_head_dim == 0` is `Source::IfPresent`'s false
    // branch reaching a rule instead of a binder: the plain norm, one block a
    // row. It is not `Ungeometric::Empty` and it is the one head-shaped rule
    // for which that is true. `head_dim` is still 128 here and must not move
    // it.
    assert_eq!(
        eval(Rule::RowsPerHead, dims(16, 2048)),
        Ok(Launch { grid: [16, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // A width the stated head does not divide, refused rather than rounded.
    assert_eq!(
        eval(Rule::RowsPerHead, Dims { stated_head_dim: 128, width: 2049, ..dims(16, 2048) }),
        Err(Ungeometric::Empty)
    );

    // And a stated head WIDER than the row, which is the same refusal read
    // the other way: a statement naming a per-head width the row's operands
    // contradict declines rather than launching a grid of zero blocks.
    assert_eq!(
        eval(Rule::RowsPerHead, Dims { stated_head_dim: 4096, ..dims(16, 2048) }),
        Err(Ungeometric::Empty)
    );
}

/// **The mutation check.** The defect the tenth field exists to remove,
/// reproduced on demand at the numbers it was found at.
///
/// `driver-cuda/src/bind/mod.rs` fills [`Dims::head_dim`] from the fire when
/// the statement named none — `spec.per_head_dim.unwrap_or_else(|| extent(
/// ctx.head_dim))` — which is right for that field and was, until this field
/// landed, also all [`Rule::RowsPerHead`] had to read. `mutant` below is that
/// reading: the absent case falling back to the fire's attention head width.
///
/// A plain `Rmsnorm` of gemma-4's 2 048 channels under 128-wide heads is
/// SIXTEEN TIMES the grid, and the failure is not a crash. `2048 % 128 == 0`,
/// so the multiple check passes and nothing refuses; the launch runs, every
/// one of the 256 blocks norms a whole 2 048-channel row from a sixteenth of
/// a row's offset, fifteen sixteenths of the writes land on rows other blocks
/// also wrote, and the tower answers. `tests/rows_per_head.rs`'s
/// `the_sixteen_times_grid_is_a_wrong_answer_and_not_a_crash` fires it and
/// shows the bytes differ.
#[test]
fn the_absent_arm_is_not_the_fires_head_width() {
    let absent = dims(16, 2048);
    assert_eq!(absent.stated_head_dim, 0, "the statement named no per-head width");
    assert_eq!(absent.head_dim, 128, "the fire's attention heads are 128 wide");

    let right = eval(Rule::RowsPerHead, absent).expect("ported");
    // The mutation: read the field the binder used to fill from the fire.
    let mutant = eval(Rule::RowsPerHead, Dims { stated_head_dim: absent.head_dim, ..absent })
        .expect("and it does not refuse, which is the whole problem");

    assert_eq!(right.grid, [16, 1, 1], "one block per row: 16 rows, 16 blocks");
    assert_eq!(mutant.grid, [256, 1, 1], "the fire's head width taken as a statement");
    assert_eq!(mutant.grid[0], right.grid[0] * 16, "sixteen times the grid");
    assert_eq!(mutant.grid[0] / right.grid[0], absent.width / absent.head_dim);
    assert_eq!(mutant.block, right.block, "and the same block, so no launch fails");
    assert_eq!(mutant.smem, right.smem);
}

/// **[`Rule::RowsPerHead`] reads the statement's head width and nothing
/// else**, and no other rule reads the statement's.
///
/// The two halves are one claim: the tenth field is a SECOND quantity, not a
/// rename of the ninth. If `head_dim` moved this rule, the binder's
/// fire-derived filler would still be reaching it. If `stated_head_dim` moved
/// a head-shaped rule, adding the field would have changed launches the
/// migration is A/B'ing against.
#[test]
fn the_two_head_widths_are_independent() {
    let stated = Dims { stated_head_dim: 128, ..dims(16, 2048) };
    for fire in [0, 32, 64, 128, 256, 2048] {
        assert_eq!(
            eval(Rule::RowsPerHead, Dims { head_dim: fire, ..stated }),
            Ok(Launch { grid: [256, 1, 1], block: [256, 1, 1], smem: 0 }),
            "the fire's head width moved a rule that reads the statement's (head_dim = {fire})"
        );
    }

    let d = dims(16, 2048);
    let restated = Dims { stated_head_dim: 64, ..d };
    for rule in [
        Rule::PerHead,
        Rule::PerHeadElementwise,
        Rule::GatedRms,
        Rule::Rope,
        Rule::RecurrentScan,
        Rule::AxialRope,
        Rule::WarpTiledScan,
        Rule::Rms,
        Rule::Elementwise,
    ] {
        assert_eq!(
            eval(rule, d),
            eval(rule, restated),
            "{rule:?} read the statement's head width; only RowsPerHead may"
        );
    }
}

/// The per-head reading and `Rule::Rms` are not the same launch, and the
/// distance is `width / stated_head_dim`.
///
/// `families/norm.rs` records the defect this catches in its own words: under
/// `Rule::Rms` a row for these five symbols *"would norm gemma-4's whole q
/// projection as one row"*. Both rules produce a 256-wide block and no third
/// number a reader would check, so the grid is the only witness — and at the
/// absent arm they agree except for the 32 bytes `Rms` requests for a fold.
#[test]
fn the_per_head_reading_is_not_the_plain_one() {
    let d = Dims { stated_head_dim: 128, ..dims(16, 2048) };
    let per_head = eval(Rule::RowsPerHead, d).expect("ported");
    let plain = eval(Rule::Rms, d).expect("ported");
    assert_eq!(per_head.grid[0], plain.grid[0] * (d.width / d.stated_head_dim));
    assert_ne!(per_head, plain, "the two readings of one symbol are two launches");

    // At the absent arm the grids coincide and the shared allocation is the
    // only difference — which is why the grid alone cannot be the test.
    let absent = dims(16, 2048);
    let flat = eval(Rule::RowsPerHead, absent).expect("ported");
    assert_eq!(flat.grid, eval(Rule::Rms, absent).expect("ported").grid);
    assert_eq!(flat.smem, 0, "RowsPerHead's kernels fold in registers, not shared");
}

/// `moe/dsv4_routing.cu`, `hash_route_lookup_bf16`:
///
/// ```text
/// // One thread per token, not one block: the kernel's whole body is a table
/// // read and a K-long gather.
/// const int grid = (tokens + kDsv4Block - 1) / kDsv4Block;
/// device::hash_route_lookup<device::bf16><<<grid, kDsv4Block, 0, stream>>>(...);
/// ```
///
/// with `kDsv4Block = 256`. The comment is the launcher's own.
///
/// **257 rows is the case that matters.** A truncating divide answers 1 and
/// routes 256 of the 257 tokens, leaving the last one's expert ids as
/// whatever the previous layer wrote — which is a mixture that runs, answers,
/// and is wrong for one token in every batch whose size is not a multiple of
/// 256.
///
/// The width is varied by a factor of 32 across the cases and moves nothing,
/// because it is not in the launcher: every other flat rule here multiplies
/// the rows by a width first, and one that did so would answer 4 096 blocks
/// for the first case instead of 8.
#[test]
fn rows_flat_reproduces_the_hash_route_launcher() {
    assert_eq!(
        eval(Rule::RowsFlat, dims(2048, 4096)),
        Ok(Launch { grid: [8, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // The round-up. 1 block covers 256 tokens and there are 257.
    assert_eq!(
        eval(Rule::RowsFlat, dims(257, 128)),
        Ok(Launch { grid: [2, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // One token: a decode step still costs a block, and 255 idle lanes.
    assert_eq!(
        eval(Rule::RowsFlat, dims(1, 4096)),
        Ok(Launch { grid: [1, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // The width is not in the launcher, at either extent.
    assert_eq!(eval(Rule::RowsFlat, dims(257, 128)), eval(Rule::RowsFlat, dims(257, 4096)));
}

/// `quant/dequant_wna16.cu`, `bf16_to_fp16`:
///
/// ```text
/// constexpr int BS = 256;
/// const long long n = static_cast<long long>(count);
/// const long long n_vec8 = n / 8;
/// const long long units = n_vec8 > 0 ? n_vec8 : n;
/// const int blocks = static_cast<int>(
///     std::min<long long>((units + BS - 1) / BS, 1024));
/// device::bf16_to_narrow<__half><<<std::max(blocks, 1), BS, 0, stream>>>(..., n);
/// ```
///
/// **`n = 4 097` is the whole point of the case list.** `ceil((n / 8) / 256)`
/// is 2 and `ceil(n / 2048)` is 3, so the reassociation a reader will reach
/// for — one divide instead of two — launches a block with nothing to do at
/// exactly the extents nobody picks. The tail those truncated loads leave is
/// the kernel's scalar loop's, not another block's.
///
/// **The cap is asserted from both sides.** At 2 098 176 elements the uncapped
/// arithmetic answers 1 025, and a rule that had dropped the `min` passes
/// every smaller case in this file. A grid short of the extent is CORRECT
/// here — `bf16_to_narrow` walks `i += gridDim.x * blockDim.x` — which is the
/// property that makes the cap a contract rather than a bug.
#[test]
fn slab_reproduces_the_bf16_to_fp16_launcher() {
    // 16 384 elements: 2 048 vec8 loads, 8 blocks.
    assert_eq!(
        eval(Rule::Slab, dims(2048, 8)),
        Ok(Launch { grid: [8, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // 4 097 elements. The launcher truncates to whole vectors FIRST.
    assert_eq!(
        eval(Rule::Slab, dims(1, 4097)),
        Ok(Launch { grid: [2, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // Exactly at the cap: 2 097 152 elements is 262 144 loads is 1 024 blocks.
    assert_eq!(
        eval(Rule::Slab, dims(2048, 1024)),
        Ok(Launch { grid: [1024, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // One row past it. Uncapped this is 1 025.
    assert_eq!(
        eval(Rule::Slab, dims(2049, 1024)),
        Ok(Launch { grid: [1024, 1, 1], block: [256, 1, 1], smem: 0 })
    );

    // Below one vector, `units = n` and the floor of one block is the
    // launcher's `std::max(blocks, 1)`.
    assert_eq!(
        eval(Rule::Slab, dims(1, 4)),
        Ok(Launch { grid: [1, 1, 1], block: [256, 1, 1], smem: 0 })
    );
}

/// `quant/dequant_wna16.cu`, `wna16_moe_gate_up_decode_bf16`:
///
/// ```text
/// const int routes = num_tokens * top_k;
/// constexpr int GU_WARPS = DECODE_BLOCK / 32;
/// const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);
/// device::wna16_gate_up_decode<<<grid, DECODE_BLOCK, 0, stream>>>(...);
/// ```
///
/// with `DECODE_BLOCK = 256`, so `GU_WARPS` is 8.
///
/// **The routed count is `rows * experts_per_token`.** A rule that took
/// `rows` for `grid.x` launches `top_k` times too few blocks and every token
/// past its first expert reads the previous layer's intermediate — which is a
/// mixture that runs and degrades rather than one that fails.
///
/// **The axes are asserted against their own transpose.** `wna16_down_decode`
/// in the same file is `dim3((hidden + DOWN_WARPS - 1) / DOWN_WARPS, routes)`
/// — these two numbers the other way round — and a transposed grid launches
/// the right blocks against the wrong cells.
///
/// `experts_per_token == 0` is what `driver-cuda`'s `jit_dims` fills today,
/// deliberately: absent, not zero-as-a-value. The refusal is asserted so that
/// the day it starts arriving filled, this file says what changed.
#[test]
fn routed_qmv_reproduces_the_wna16_moe_decode_launcher() {
    // A decode of one token into 8 experts over a 768-wide intermediate:
    // 8 routes, 96 warp tiles.
    let decode = Dims { experts_per_token: 8, ..dims(1, 768) };
    assert_eq!(
        eval(Rule::RoutedQmv, decode),
        Ok(Launch { grid: [8, 96, 1], block: [256, 1, 1], smem: 0 })
    );

    // Four tokens into 8, at twice the intermediate.
    let batch = Dims { experts_per_token: 8, ..dims(4, 1536) };
    assert_eq!(
        eval(Rule::RoutedQmv, batch),
        Ok(Launch { grid: [32, 192, 1], block: [256, 1, 1], smem: 0 })
    );

    // An intermediate the warp count does not divide: 769 needs 97 tiles and
    // a truncating divide drops a column of the projection.
    let ragged = Dims { experts_per_token: 8, ..dims(1, 769) };
    assert_eq!(
        eval(Rule::RoutedQmv, ragged),
        Ok(Launch { grid: [8, 97, 1], block: [256, 1, 1], smem: 0 })
    );

    // The transpose, named.
    let d = eval(Rule::RoutedQmv, decode).expect("ported");
    assert_ne!(d.grid, [96, 8, 1], "`wna16_down_decode` is this grid reversed");

    // The binder's current state.
    assert_eq!(
        eval(Rule::RoutedQmv, Dims { experts_per_token: 0, ..dims(1, 768) }),
        Err(Ungeometric::Empty)
    );
}

/// `vision/gemma4_vision.cu`, `k_addpos_grid2d`:
///
/// ```text
/// dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}
/// ...
/// vd::k_addpos_grid2d<bfd><<<G2(Hd,N),B2,0,S>>>(D(h),D(w.pos_table),pos,N,Hd,PT);
/// ```
///
/// — `B2` and `G2` at `117`, the launch at `144`, and all three towers
/// declare the same pair and launch eleven kernels through it.
///
/// **`X` is the WIDTH and `Y` is the ROW COUNT.** The first case is gemma-4's
/// vision tower — 768 channels over 4 096 patches — and it is chosen so the
/// two axes cannot be swapped without the numbers moving: 48 by 256, not 256
/// by 48.
///
/// **The block is asserted as `[16, 16, 1]` and that is the new ground.**
/// Every rule before this one produces `[n, 1, 1]`; the kernels here read
/// `threadIdx.y` for their row (`gemma4_naive_kernels.cuh:117`), so a
/// flattened block of 256 gives every thread of every block row zero — a full
/// grid of legal work against a sixteenth of the output.
#[test]
fn tile16_reproduces_the_vision_tower_launcher() {
    // 4 096 patches by 768 channels: 48 tiles across, 256 down.
    assert_eq!(
        eval(Rule::Tile16, dims(4096, 768)),
        Ok(Launch { grid: [48, 256, 1], block: [16, 16, 1], smem: 0 })
    );

    // Neither extent a multiple of the tile. 1 025 needs 65 columns of tiles
    // and 17 rows need 2: a truncating divide leaves a column and a row of
    // the output holding whatever the arena had.
    assert_eq!(
        eval(Rule::Tile16, dims(17, 1025)),
        Ok(Launch { grid: [65, 2, 1], block: [16, 16, 1], smem: 0 })
    );

    // One row, which is `k_matmul` at decode: one tile down, and the block
    // still square — fifteen of its sixteen rows idle, as the launcher's is.
    assert_eq!(
        eval(Rule::Tile16, dims(1, 768)),
        Ok(Launch { grid: [48, 1, 1], block: [16, 16, 1], smem: 0 })
    );

    // The transpose, named: `G2(X, Y)` is (width, rows) and not (rows, width).
    let wide = eval(Rule::Tile16, dims(16, 2048)).expect("ported");
    assert_eq!(wide.grid, [128, 1, 1]);
    assert_ne!(wide.grid, [1, 128, 1], "`G2`'s first argument is the width");
}

/// `vision/gemma4_vision.cu`, `k_rope_axial2d`, at `150` — ONE line, TWO
/// launches:
///
/// ```text
/// dim3 rg(1,NH,N);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(q),pos,N,NH,THETA);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(k),pos,N,NH,THETA);
/// ```
///
/// **`grid.x` is literally one and every case asserts it.** The kernel reads
/// `blockIdx.z` as its token and `blockIdx.y` as its head
/// (`gemma4_vision.cuh:106`); its head is 64 channels read as four 16-wide
/// quarters and it returns on `c >= 16`, so sixteen lanes of the warp work
/// and the channel axis is spent before it starts. A rule that packed the
/// counts into the first two axes out of habit launches a grid this kernel
/// indexes as one token.
///
/// Both launches take the same `rg` and one tensor each, which is why the
/// head count this rule reads is [`Dims::kv_heads`] — the addressed tensor's
/// — and not a sum of two.
///
/// **This is the first `grid.z` in the file.** Everything before it produces
/// a third axis of 1, which is why the assertion is on the whole `Launch` and
/// then on `grid[2]` by name.
#[test]
fn axial_rope_reproduces_the_vision_rope_launcher() {
    // gemma-4's vision tower: 12 heads, 4 096 patches, one warp each.
    let tower = Dims { kv_heads: 12, head_dim: 64, ..dims(4096, 768) };
    assert_eq!(
        eval(Rule::AxialRope, tower),
        Ok(Launch { grid: [1, 12, 4096], block: [32, 1, 1], smem: 0 })
    );

    // One token, and the head axis unchanged.
    let one = Dims { kv_heads: 12, head_dim: 64, ..dims(1, 768) };
    assert_eq!(
        eval(Rule::AxialRope, one),
        Ok(Launch { grid: [1, 12, 1], block: [32, 1, 1], smem: 0 })
    );

    // The rows are on the THIRD axis, which no rule before this one used.
    let launch = eval(Rule::AxialRope, tower).expect("ported");
    assert_eq!(launch.grid[0], 1, "`dim3 rg(1, NH, N)` opens with a literal one");
    assert_eq!(launch.grid[2], 4096, "the token count is `grid.z`");

    // A tensor with no heads is a launch of nothing, and refused.
    assert_eq!(
        eval(Rule::AxialRope, Dims { kv_heads: 0, ..tower }),
        Err(Ungeometric::Empty)
    );
}

/// `ssm/gated_delta_net.cu`,
/// `chunk_gated_delta_prefill_batched_warp_tiled_gqa`:
///
/// ```text
/// constexpr int WARPS = 4;
/// constexpr int BLOCK = WARPS * 32;
/// dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
/// dim3 block(BLOCK);
/// device::chunk_gated_delta_prefill_batched_warp_tiled_gqa<float, false>
///     <<<grid, block, 0, stream>>>(...);
/// ```
///
/// at `759-791`, and `..._state_bf16` at `816-850` over `__nv_bfloat16`.
///
/// **`V_d` is `width / kv_heads`.** The output row of both launchers is
/// `V_h * V_d` wide, so the value width is a quotient of two fields whose
/// meanings are fixed — and taking it from `Dims::head_dim` instead would
/// read as `V_d` the field `RecurrentScan` reads as `K_d`. The first case
/// sets those two to DIFFERENT numbers (a 128-channel key, a 128-channel
/// value, 32 value heads over a 4 096-wide row) and the second moves the
/// value width alone, so a rule that had crossed them is caught.
///
/// **Nothing shared, and that is the distance from `RecurrentScan`.** The
/// scan's block reads `2 * K_d` floats it must be given; this one holds its
/// tile in registers. A launch that inherited the allocation would pay for
/// staging it does not do — the harmless direction, and still a different
/// rule.
#[test]
fn warp_tiled_scan_reproduces_the_gated_delta_net_launcher() {
    // Qwen3-Next: 16 requests, 32 value heads, a 4 096-wide output row, so
    // `V_d` is 128 and the value channels split 32 ways across four warps.
    let gdn = Dims { kv_heads: 32, head_dim: 128, ..dims(16, 4096) };
    assert_eq!(
        eval(Rule::WarpTiledScan, gdn),
        Ok(Launch { grid: [16, 32, 32], block: [128, 1, 1], smem: 0 })
    );

    // The same key width over 8 value heads of 256 channels: the third axis
    // doubles and the block does not.
    assert_eq!(
        eval(Rule::WarpTiledScan, dims(16, 2048)),
        Ok(Launch { grid: [16, 8, 64], block: [128, 1, 1], smem: 0 })
    );

    // A value width the warps do not divide: 6 channels need 2 tiles, and the
    // second one runs two warps idle rather than dropping four channels.
    let ragged = Dims { kv_heads: 8, ..dims(1, 48) };
    assert_eq!(
        eval(Rule::WarpTiledScan, ragged),
        Ok(Launch { grid: [1, 8, 2], block: [128, 1, 1], smem: 0 })
    );

    // A row width the value heads do not divide is not a value width.
    assert_eq!(
        eval(Rule::WarpTiledScan, Dims { width: 2049, ..dims(16, 2048) }),
        Err(Ungeometric::Empty)
    );
}

/// The two recurrences are two launches, and the shared allocation is the
/// witness.
///
/// `RecurrentScan` and `WarpTiledScan` share `grid.x` and `grid.y` exactly,
/// so a reader checking one against the other sees two rules that agree about
/// everything a doc comment states first. They differ in the third axis, in
/// the block, and in the allocation — and it is the allocation that cannot be
/// got wrong quietly: a warp-tiled launch given the scan's slab wastes 1 KB
/// per block, and a scan given none reads its `sq`/`sk` staging out of
/// whatever the last kernel left in shared memory.
#[test]
fn the_two_recurrences_share_two_axes_and_nothing_else() {
    let d = Dims { kv_heads: 32, head_dim: 128, ..dims(16, 4096) };
    let scan = eval(Rule::RecurrentScan, d).expect("ported");
    let tiled = eval(Rule::WarpTiledScan, d).expect("ported");

    assert_eq!(scan.grid[0], tiled.grid[0], "both count the same rows");
    assert_eq!(scan.grid[1], tiled.grid[1], "both put the value heads on `grid.y`");
    assert_eq!(scan.grid[2], 1, "the scan gives each block the whole value width");
    assert_ne!(scan.grid[2], tiled.grid[2], "the tiled one cuts it four ways");
    assert_eq!(scan.smem, 2 * d.head_dim * 4, "the scan stages `q` and `k`");
    assert_eq!(tiled.smem, 0, "the tiled one holds its slice in registers");
}

/// Every rule of the second pass says what it does with the row axis.
///
/// The same witness `every_new_rule_states_what_it_does_with_the_row_axis`
/// keeps for the first pass, for the same reason: `RouterLane` dropped the
/// row axis and every review passed because the launch was plausible. Two of
/// these seven put the row somewhere no earlier rule did — `Tile16` on a
/// tiled `grid.y`, `AxialRope` on `grid.z` — which is exactly the kind of
/// move that makes a dropped axis look deliberate.
#[test]
fn every_second_pass_rule_states_what_it_does_with_the_row_axis() {
    let filled = |rows| Dims { experts_per_token: 8, ..dims(rows, 2048) };
    for rule in [
        Rule::RowsPerHead,
        Rule::RowsFlat,
        Rule::Slab,
        Rule::RoutedQmv,
        Rule::Tile16,
        Rule::AxialRope,
        Rule::WarpTiledScan,
    ] {
        let one = eval(rule, filled(1)).expect("ported");
        let many = eval(rule, filled(4096)).expect("ported");
        assert_ne!(one, many, "{rule:?} answers the same for 1 row and 4096");
    }
}

/// `vision/gemma4_audio.cu`, both SSCP layernorms:
///
/// ```text
/// vd::k_layernorm_relu<bfd><<<T1*F1,128,0,S>>>(
///     D(c0cl),D(w.sscp0_norm),D(c0cl),T1*F1,C0,EPS);
/// ```
///
/// at `189`, and the same line over `T2*F2` and `C1` at `196`.
///
/// **The block is the whole case.** This grid is `Rule::PerRow`'s and this
/// grid is `Rule::Rms`' — three rules, one `grid.x` — so the only witness to
/// stating the right one is the second number, and the second number is a
/// numerics contract: `gemma4_audio.cuh:141` records that the fold sums
/// `(blockDim.x + 31) / 32` warp partials serially in thread zero, so 128 and
/// 256 add the same values in a different order. A test that asserted only
/// the grid would pass under all three.
///
/// The width moves by a factor of eight across the cases and nothing moves
/// with it: `C` reaches the kernel as an operand and is the extent of the
/// fold, not of the launch.
#[test]
fn per_row_narrow_reproduces_the_audio_layernorm_launcher() {
    // The first SSCP layer: `T1 * F1` rows of `C0` channels.
    assert_eq!(
        eval(Rule::PerRowNarrow, dims(4096, 128)),
        Ok(Launch { grid: [4096, 1, 1], block: [128, 1, 1], smem: 0 })
    );

    // The second, narrower and deeper — the width does not reach the launch.
    assert_eq!(
        eval(Rule::PerRowNarrow, dims(1024, 1024)),
        Ok(Launch { grid: [1024, 1, 1], block: [128, 1, 1], smem: 0 })
    );

    // One row, and a row count the block does not divide: the row is the
    // GRID here, so neither rounds.
    assert_eq!(
        eval(Rule::PerRowNarrow, dims(1, 128)),
        Ok(Launch { grid: [1, 1, 1], block: [128, 1, 1], smem: 0 })
    );
    assert_eq!(
        eval(Rule::PerRowNarrow, dims(129, 128)),
        Ok(Launch { grid: [129, 1, 1], block: [128, 1, 1], smem: 0 })
    );
}

/// The three rules that share `grid.x = rows` are told apart by the two
/// numbers after it.
///
/// `Rms`, `PerRow` and `PerRowNarrow` produce the same first number for every
/// rectangle there is, so a row that states the wrong one launches a grid
/// nothing can distinguish from the right one by looking at it. What separates
/// them is a fold's scratch, a fold's width, and nothing else — which is
/// exactly the kind of distinction that is lost the first time someone reads
/// two rules as one.
#[test]
fn the_three_row_grids_differ_only_after_the_grid() {
    let d = dims(4096, 2048);
    let rms = eval(Rule::Rms, d).expect("ported");
    let wide = eval(Rule::PerRow, d).expect("ported");
    let narrow = eval(Rule::PerRowNarrow, d).expect("ported");

    assert_eq!(rms.grid, wide.grid);
    assert_eq!(wide.grid, narrow.grid, "all three are one block per row");
    assert_eq!(rms.block, wide.block, "and two of the three are 256 wide");
    assert_eq!(narrow.block, [128, 1, 1], "the layernorm is not");
    assert_eq!((rms.smem, wide.smem, narrow.smem), (32, 0, 0), "only `Rms` folds");
}

//===----------------------------------------------------------------------===//
//
// The citation, tested.
//
//===----------------------------------------------------------------------===//

/// Every case above compares `eval` against numbers a human read off a `.cu`
/// and typed here. This module compares it against the launcher's own
/// arithmetic, transcribed as code and ANCHORED to the source text it came
/// from.
///
/// # Why the difference matters
///
/// A hand-copied number tests the rule. It cannot test the CITATION, and this
/// file's law is that a rule with no cited launcher is a guess — from which
/// it follows that a rule with a WRONG cited launcher is worse than one with
/// none, because it reads as evidence and audits clean.
///
/// That is not hypothetical. [`Rule::AxialRope`]'s doc quoted
///
/// ```text
/// k_rope_axial2d<bfd><<<rg, 32, 0, S>>>(D(q), D(k), N, NH, gw, gh)
/// ```
///
/// for long enough to be copied into this file, and the live source at
/// `vision/gemma4_vision.cu:150` is `(D(q), pos, N, NH, THETA)` — different
/// parameters, and TWO launches rather than one. The kernel takes one tensor;
/// a reader who believed the quote would have concluded the launcher fires
/// once over both and that the head count is a sum. The GEOMETRY in the
/// citation was right the whole time, and every existing case here passed
/// throughout, because they all compare against `dim3 rg(1, NH, N)` — which
/// is exactly the number the stale quote still got right.
///
/// **So the geometry was checked by tests and the text was checked by
/// nobody.** These cases close that: each one pins the launcher's declaration
/// by its literal text AND by its line, transcribes the arithmetic from the
/// pinned text, and compares `eval` against the transcription. A `.cu` edit
/// that moves a line fails the pin and names the doc that must be revisited;
/// an edit that changes the arithmetic fails the comparison.
///
/// # What it is not
///
/// It is not a fire and it is not a substitute for one. Two grids agreeing on
/// three integers is a claim about the launch and not about the kernel — see
/// `tests/rows_per_head.rs` for the byte-identical half of the same argument.
mod transcribed {
    use super::{Rule, dims};
    use kernels_cuda_new::runtime::{Dims, eval};

    // The three towers moved to `driver-cuda/csrc/vision/`; a tower is a host
    // walk over device text the JIT already owns, not a kernel the archive was
    // holding. The transcriptions below still quote them, so this test follows
    // the files rather than the crate.
    const VISION: &str = include_str!("../../driver-cuda/csrc/vision/gemma4_vision.cu");
    const AUDIO: &str = include_str!("../../driver-cuda/csrc/vision/gemma4_audio.cu");
    const QWEN: &str = include_str!("../../driver-cuda/csrc/vision/qwen3_vl_tower.cu");
    const GDN: &str = include_str!("../../kernels-cuda/csrc/src/ssm/gated_delta_net.cu");
    const GDN_CUH: &str = include_str!("../csrc/src/ssm/gated_delta_net.cuh");
    const WNA16: &str = include_str!("../../kernels-cuda/csrc/src/quant/dequant_wna16.cu");
    const FP4: &str = include_str!("../../kernels-cuda/csrc/src/quant/dequant_fp4.cu");
    const FP4_CUH: &str = include_str!("../csrc/src/quant/dequant_fp4.cuh");
    const NAIVE: &str =
        include_str!("../../kernels-cuda/csrc/src/attn/attention_naive_paged.cu");
    const MLA: &str = include_str!("../../kernels-cuda/csrc/src/attn/mla_paged.cu");
    const QKV: &str = include_str!("../../kernels-cuda/csrc/src/attn/qkv_fused.cu");
    const ALTUP: &str = include_str!("../../kernels-cuda/csrc/src/norm/altup.cu");
    const ROPE: &str = include_str!("../../kernels-cuda/csrc/src/rope/rope.cu");
    const DSV4: &str = include_str!("../../kernels-cuda/csrc/src/attn/dsv4_compress.cu");
    const FLASHINFER: &str =
        include_str!("../../kernels-cuda/csrc/src/attn/attention_flashinfer.cu");
    const SLOT_OPS: &str = include_str!("../../kernels-cuda/csrc/src/layout/slot_ops.cu");
    const KV_PAGED: &str = include_str!("../../kernels-cuda/csrc/src/attn/kv_paged.cu");
    const NAIVE_UNPAGED: &str =
        include_str!("../../kernels-cuda/csrc/src/attn/attention_naive.cu");
    const PAGE_COMPACT: &str = include_str!("../../kernels-cuda/csrc/src/attn/page_compact.cu");
    const RMSNORM: &str = include_str!("../../kernels-cuda/csrc/src/norm/rmsnorm.cu");
    const NEMOTRON: &str = include_str!("../../kernels-cuda/csrc/src/ssm/nemotron_h.cu");

    /// Assert `text` appears in `src` at 1-based `line`, and say which
    /// citation goes stale if it has moved.
    ///
    /// Both halves are load-bearing. The TEXT catches an edit that rewrites
    /// the launch — the `(D(q), D(k), …)` case above. The LINE catches an
    /// edit that leaves the launch alone and moves it, which breaks every
    /// `at :NNN` in the rule's doc while every assertion in this file still
    /// passes.
    #[track_caller]
    fn pinned(src: &str, file: &str, line: usize, text: &str, cites: &str) {
        let found: Vec<usize> = src
            .lines()
            .enumerate()
            .filter(|(_, l)| l.trim() == text)
            .map(|(i, _)| i + 1)
            .collect();

        assert!(
            !found.is_empty(),
            "{file}: the line this transcription was written from is gone.\n  \
             expected: {text}\n  \
             {cites} cites it, and the transcription below is derived from it. \
             Re-read the launcher, correct the citation, then correct the \
             arithmetic — in that order, because a citation corrected to match \
             a transcription is how the wrong quote survived."
        );
        assert!(
            found.contains(&line),
            "{file}: the line is unchanged and has MOVED to {found:?}, cited as \
             {line}.\n  {cites} says `{line}` and now points at the wrong \
             place. Nothing about the grid changed; the evidence did."
        );
    }

    // ── The launchers' own arithmetic ────────────────────────────────
    //
    // Transcribed from the text `pinned()` checks, and from nothing else. No
    // number below was read off a rule.

    /// `inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}`
    ///
    /// X is the WIDTH and Y is the ROW COUNT, in that order.
    fn g2(x: i32, y: i32) -> [u32; 3] {
        [((x + 15) / 16) as u32, ((y + 15) / 16) as u32, 1]
    }

    /// `dim3 B2(16,16);`
    const B2: [u32; 3] = [16, 16, 1];

    /// `dim3 rg(1,NH,N);`
    fn rg(nh: i32, n: i32) -> [u32; 3] {
        [1, nh as u32, n as u32]
    }

    /// `vd::k_layernorm_relu<bfd><<<T1*F1,128,0,S>>>`
    fn sscp_ln(t: i32, f: i32) -> ([u32; 3], [u32; 3]) {
        ([(t * f) as u32, 1, 1], [128, 1, 1])
    }

    /// `dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS); dim3 block(BLOCK);`
    /// with `WARPS = 4` and `BLOCK = WARPS * 32`.
    fn gdn_warp_tiled(r: i32, v_h: i32, v_d: i32) -> ([u32; 3], [u32; 3]) {
        const WARPS: i32 = 4;
        const BLOCK: i32 = WARPS * 32;
        ([r as u32, v_h as u32, ((v_d + WARPS - 1) / WARPS) as u32], [BLOCK as u32, 1, 1])
    }

    /// `const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);`
    /// with `GU_WARPS = DECODE_BLOCK / 32` and `DECODE_BLOCK = 256`.
    fn wna16_gate_up(routes: i32, intermediate: i32) -> ([u32; 3], [u32; 3]) {
        const DECODE_BLOCK: i32 = 256;
        const GU_WARPS: i32 = DECODE_BLOCK / 32;
        (
            [routes as u32, ((intermediate + GU_WARPS - 1) / GU_WARPS) as u32, 1],
            [DECODE_BLOCK as u32, 1, 1],
        )
    }

    /// `const int grid = (tokens + kDsv4Block - 1) / kDsv4Block;` at
    /// `kDsv4Block = 256`.
    fn dsv4_route(tokens: i32) -> ([u32; 3], [u32; 3]) {
        const K: i32 = 256;
        ([((tokens + K - 1) / K) as u32, 1, 1], [K as u32, 1, 1])
    }

    /// `const dim3 grid((hidden + WARPS - 1) / WARPS, routes);` with
    /// `WARPS = BS / 32` and `BS = 256` — the TRANSPOSE of `wna16_gate_up`.
    ///
    /// Written as its own function rather than as a swap of that one's tuple,
    /// because the two launchers are two texts and this module transcribes
    /// texts. A transposition expressed as `let (a, b) = other(); [b, a]`
    /// would agree with itself if the original were misread.
    fn wna16_down(hidden: i32, routes: i32) -> ([u32; 3], [u32; 3]) {
        const BS: i32 = 256;
        const WARPS: i32 = BS / 32;
        (
            [((hidden + WARPS - 1) / WARPS) as u32, routes as u32, 1],
            [BS as u32, 1, 1],
        )
    }

    /// `dim3 grid(num_tokens * top_k, (intermediate + pairs_per_block - 1)
    /// / pairs_per_block);` with `pairs_per_block = warps * kMxfp4GateUpPairs`,
    /// `warps = kMxfp4DecodeBlock / 32`, `kMxfp4DecodeBlock = 128` and
    /// `kMxfp4GateUpPairs = 4`.
    ///
    /// **The tile is a PRODUCT of two constants and both are transcribed**,
    /// because `Rule::RoutedQmvQuad`'s claim is the product `4 * 4 = 16` and
    /// `Rule::RoutedQmv`'s is `8 * 1 = 8`. A transcription that folded either
    /// factor would agree with a rule that folded the other one.
    fn mxfp4_gate_up(routes: i32, intermediate: i32) -> ([u32; 3], [u32; 3]) {
        const BLOCK: i32 = 128;
        const PAIRS: i32 = 4;
        let warps = BLOCK / 32;
        let pairs_per_block = warps * PAIRS;
        (
            [routes as u32, ((intermediate + pairs_per_block - 1) / pairs_per_block) as u32, 1],
            [BLOCK as u32, 1, 1],
        )
    }

    /// `dim3 grid(num_tokens * top_k, (hidden + rows_per_block - 1)
    /// / rows_per_block);` with `rows_per_block = warps * kMxfp4DownRows`.
    ///
    /// **NOT the transpose of `mxfp4_gate_up`**, which is the whole point of
    /// writing it out: `dequant_wna16.cu` swaps the axes between its two legs
    /// and `dequant_fp4.cu` does not, so the two files need three rules
    /// between them and not two. Reading `routes` off `grid.x` in both is a
    /// claim about this file and it is checked here against this file's text.
    fn mxfp4_down(routes: i32, hidden: i32) -> ([u32; 3], [u32; 3]) {
        const BLOCK: i32 = 128;
        const ROWS: i32 = 4;
        let warps = BLOCK / 32;
        let rows_per_block = warps * ROWS;
        (
            [routes as u32, ((hidden + rows_per_block - 1) / rows_per_block) as u32, 1],
            [BLOCK as u32, 1, 1],
        )
    }

    /// `dim3 grid(num_requests, total_tokens, num_q_heads); dim3 block(BLOCK);`
    /// and `const std::size_t smem = (head_dim + BLOCK) * sizeof(float);`
    /// with `BLOCK = 128`.
    ///
    /// **The only transcription in this module that returns a shared-memory
    /// size**, and the only launcher whose `smem` is not a literal. `+ BLOCK`
    /// and not `+ head_dim`: the extra floats are the reduction scratch, one
    /// per THREAD, and `SdpaVector`'s `(rows + 256) * 4` adds a block width to
    /// a row count instead — the near-miss the refusal named.
    fn naive_paged(
        requests: i32,
        tokens: i32,
        q_heads: i32,
        head_dim: i32,
    ) -> ([u32; 3], [u32; 3], u32) {
        const BLOCK: i32 = 128;
        (
            [requests as u32, tokens as u32, q_heads as u32],
            [BLOCK as u32, 1, 1],
            ((head_dim + BLOCK) * 4) as u32,
        )
    }

    /// `dim3 grid(num_requests, num_q_heads); dim3 block(BLOCK);` and the same
    /// `(head_dim + BLOCK) * sizeof(float)`, at `attention_naive_paged.cu:147`.
    fn naive_paged_decode(requests: i32, q_heads: i32, head_dim: i32) -> ([u32; 3], [u32; 3], u32) {
        const BLOCK: i32 = 128;
        (
            [requests as u32, q_heads as u32, 1],
            [BLOCK as u32, 1, 1],
            ((head_dim + BLOCK) * 4) as u32,
        )
    }

    /// `dim3 grid(total_tokens, 1 + q_blocks);` at `BS = 256`, with
    /// `heads_per_block = half >= BS ? 1 : (BS / half)` and
    /// `q_blocks = (heads + heads_per_block - 1) / heads_per_block` and
    /// `half = rope / 2`.
    ///
    /// The `1 +` is transcribed as written. `mla_paged.cuh:236` reads
    /// `blockIdx.y - 1` and branches on `qb < 0`, so lane 0 is the KV lane —
    /// the norm, the `k_pe` rotation and the paged write — and it is an axis
    /// rather than padding.
    fn mla_prepare(tokens: i32, heads: i32, rope: i32) -> ([u32; 3], [u32; 3]) {
        const BS: i32 = 256;
        let half = rope / 2;
        let heads_per_block = if half >= BS { 1 } else { BS / half };
        let q_blocks = (heads + heads_per_block - 1) / heads_per_block;
        ([tokens as u32, (1 + q_blocks) as u32, 1], [BS as u32, 1, 1])
    }

    /// `dim3 grid(num_rows, num_q_heads + num_kv_heads);` at `BLOCK = 256`
    /// (`qkv_fused.cu:245-246`) and at `BLOCK = 128` (`:98-99`).
    ///
    /// One function for two launchers because they are the SAME expression at
    /// two widths — `qkv_fused.cu` writes `num_q_heads + num_kv_heads` twice,
    /// character for character, and both lines are pinned.
    fn rows_packed_heads(rows: i32, q_heads: i32, kv_heads: i32, block: i32) -> ([u32; 3], [u32; 3]) {
        ([rows as u32, (q_heads + kv_heads) as u32, 1], [block as u32, 1, 1])
    }

    /// `dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));`
    /// with `total_units = num_requests * (num_q_heads + num_kv_heads)` and
    /// `WARP_BLOCK = 256`.
    fn warp_packed_heads(requests: i32, q_heads: i32, kv_heads: i32) -> ([u32; 3], [u32; 3]) {
        const WARP_BLOCK: i32 = 256;
        let total_units = requests * (q_heads + kv_heads);
        (
            [((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32)) as u32, 1, 1],
            [WARP_BLOCK as u32, 1, 1],
        )
    }

    /// `const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);` at `BLOCK = 128`.
    ///
    /// `K` is the AltUp STREAM count and `H` is the PER-STREAM width, which is
    /// the whole reason `WarpTiledScan` — same block, same three axes — is not
    /// this launch. Its `z` is `ceil(V_d / 4)`, a thirty-second of these
    /// blocks, and its `y` comes from `Dims::kv_heads`.
    fn altup(t: i32, k: i32, h: i32) -> ([u32; 3], [u32; 3]) {
        const BLOCK: i32 = 128;
        ([t as u32, k as u32, ((h + BLOCK - 1) / BLOCK) as u32], [BLOCK as u32, 1, 1])
    }

    /// `dim3 grid(num_tokens, num_q_heads + num_kv_heads);` at
    /// `constexpr int BLOCK = 128;` — `rope/rope.cu`'s three `qk_rmsnorm`
    /// launchers, identical at `:45-47`, `:189-191` and `:213-215`.
    ///
    /// Deliberately a SEPARATE transcription from [`rows_packed_heads`], which
    /// is `qkv_fused.cu`'s. The two launchers compute the same three numbers
    /// out of differently-named locals, and collapsing them would make one
    /// launcher's pin stand as evidence for the other's arithmetic — which is
    /// how `RowsPackedHeadsNarrow` came to be ported from `attn` and refused
    /// in `rope` for four releases.
    ///
    /// `num_tokens` is the first argument and `num_q_heads + num_kv_heads` the
    /// second, in that order.
    fn rope_packed_heads(tokens: i32, q_heads: i32, kv_heads: i32) -> ([u32; 3], [u32; 3]) {
        const BLOCK: i32 = 128;
        ([tokens as u32, (q_heads + kv_heads) as u32, 1], [BLOCK as u32, 1, 1])
    }

    /// ```text
    /// dim3 grid(static_cast<unsigned>(total_tokens),
    ///           static_cast<unsigned>(num_q_heads));
    /// const std::size_t smem =
    ///     (static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);
    /// device::compressed_attn_paged<<<grid, ATTN_BLOCK, smem, stream>>>(
    /// ```
    ///
    /// with `constexpr int ATTN_BLOCK = 128;` at `dsv4_compress.cu:37`. The
    /// SMEM is the third number and the one that matters: it is computed from
    /// `head_dim`, an OPERAND, which is exactly what `dsv4_compress.cuh:50-52`
    /// says no ported rule does.
    fn compressed_paged(tokens: i32, q_heads: i32, head_dim: i32) -> ([u32; 3], [u32; 3], u32) {
        const ATTN_BLOCK: i32 = 128;
        (
            [tokens as u32, q_heads as u32, 1],
            [ATTN_BLOCK as u32, 1, 1],
            ((head_dim + ATTN_BLOCK) * 4) as u32,
        )
    }

    /// Every launcher declaration these transcriptions were written from, at
    /// the line the rule that cites it names.
    ///
    /// This runs first in effect — every comparison below calls it — so a
    /// source edit is reported as a stale citation rather than as arithmetic
    /// that mysteriously stopped agreeing.
    fn pins() {
        let g2b2 = "dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}";
        pinned(VISION, "driver-cuda csrc/vision/gemma4_vision.cu", 138, g2b2, "`tile16`");
        pinned(AUDIO, "driver-cuda csrc/vision/gemma4_audio.cu", 131, g2b2, "`tile16`");
        pinned(QWEN, "driver-cuda csrc/vision/qwen3_vl_tower.cu", 139, g2b2, "`tile16`");

        pinned(
            VISION,
            "driver-cuda csrc/vision/gemma4_vision.cu",
            195,
            "vd::k_addpos_grid2d<bfd><<<G2(Hd,N),B2,0,S>>>(D(h),D(w.pos_table),pos,N,Hd,PT);",
            "`tile16`",
        );

        // One line, two launches, one tensor each. This is the pin the stale
        // quote would have failed.
        pinned(
            VISION,
            "driver-cuda csrc/vision/gemma4_vision.cu",
            201,
            "dim3 rg(1,NH,N);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(q),pos,N,NH,THETA);\
             vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(k),pos,N,NH,THETA);",
            "`axial_rope`",
        );

        pinned(
            AUDIO,
            "driver-cuda csrc/vision/gemma4_audio.cu",
            189,
            "vd::k_layernorm_relu<bfd><<<T1*F1,128,0,S>>>(D(c0cl),D(w.sscp0_norm),D(c0cl),T1*F1,C0,EPS);",
            "`per_row_narrow`",
        );
        pinned(
            AUDIO,
            "driver-cuda csrc/vision/gemma4_audio.cu",
            196,
            "vd::k_layernorm_relu<bfd><<<T2*F2,128,0,S>>>(D(c1cl),D(w.sscp1_norm),D(c1cl),T2*F2,C1,EPS);",
            "`per_row_narrow`",
        );

        // `warp_tiled_scan`'s two HOST LAUNCHERS ARE GONE (§43). Nothing
        // called them, so the `.cu` no longer opens that grid at all, and a
        // `pinned` here would be a citation to text that does not exist.
        //
        // The rule is still exercised below, so the transcription still needs
        // a source — and the source is now the KERNEL rather than its
        // launcher. Two assertions replace the three pins, and they are
        // deliberately of opposite polarity:
        //
        //   * an ABSENCE, so that a launcher quietly coming back does not
        //     leave this transcription reading a `.cuh` while the `.cu`
        //     disagrees with it;
        //   * a live `pinned` into the `.cuh`, so the block width the
        //     arithmetic below divides by is still quoted from real text.
        //
        // Absence is the weaker claim of the two and that is exactly why it
        // is written out: `pinned` can only ever assert presence, so a
        // deleted line is the one thing it structurally cannot guard.
        assert!(
            !GDN.contains("dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);"),
            "ssm/gated_delta_net.cu opens `warp_tiled_scan`'s grid again.\n\
             The two launchers were deleted as unreached; if one is back, this\n\
             transcription must be re-pinned to it rather than to the .cuh."
        );
        pinned(GDN_CUH, "ssm/gated_delta_net.cuh", 436, "constexpr int WARPS = 4;", "`warp_tiled_scan`");

        pinned(
            WNA16,
            "quant/dequant_wna16.cu",
            74,
            "const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);",
            "`routed_qmv`",
        );
        pinned(WNA16, "quant/dequant_wna16.cu", 73, "constexpr int GU_WARPS = DECODE_BLOCK / 32;", "`routed_qmv`");

        // ── `routed_qmv_transposed`, the other half of the same file ──
        //
        // Pinned beside its twin ON PURPOSE. The two grids differ only in
        // which axis carries the routes, so the evidence that they differ has
        // to be two quotes and not one quote read twice.
        pinned(WNA16, "quant/dequant_wna16.cu", 102, "constexpr int WARPS = BS / 32;", "`routed_qmv_transposed`");
        pinned(
            WNA16,
            "quant/dequant_wna16.cu",
            103,
            "const dim3 grid((hidden + WARPS - 1) / WARPS, routes);",
            "`routed_qmv_transposed`",
        );

        // ── `routed_qmv_quad`, the OTHER MoE decode file ──
        //
        // Four pins for one rule, because the rule states a PRODUCT: the
        // block is 128 where `routed_qmv`'s is 256 and the rows per warp are
        // 4 where its are 1, so `128/32 * 4 = 16` against `256/32 * 1 = 8`.
        // Pinning only the `dim3` would leave both factors unwitnessed and
        // the product is what a near miss gets wrong.
        pinned(FP4, "quant/dequant_fp4.cu", 39, "constexpr int kMxfp4DecodeBlock = 128;", "`routed_qmv_quad`");
        pinned(FP4, "quant/dequant_fp4.cu", 42, "constexpr int kMxfp4GateUpPairs = 4;", "`routed_qmv_quad`");
        pinned(
            FP4,
            "quant/dequant_fp4.cu",
            44,
            "constexpr int kMxfp4DownRows = 4;  // four warps, one output row each",
            "`routed_qmv_quad`",
        );
        pinned(
            FP4,
            "quant/dequant_fp4.cu",
            70,
            "(intermediate + pairs_per_block - 1) / pairs_per_block);",
            "`routed_qmv_quad`",
        );
        pinned(
            FP4,
            "quant/dequant_fp4.cu",
            104,
            "(hidden + rows_per_block - 1) / rows_per_block);",
            "`routed_qmv_quad`",
        );
        // The two legs put the routes on the SAME axis, which is the claim
        // that makes one rule cover both. Pinned as two quotes of the same
        // text at two lines, because one quote read twice is not evidence
        // that two launchers agree.
        for line in [69, 103] {
            pinned(FP4, "quant/dequant_fp4.cu", line, "dim3 grid(num_tokens * top_k,", "`routed_qmv_quad`");
        }
        // And the grouped sibling, which used to be pinned for what it is
        // NOT: a `pairs_per_block` of `warps * 2`, half this rule's tile, on
        // a grid whose `x` is an EXPERT count.
        //
        // ITS HOST LAUNCHER IS GONE (§43) — nothing called it — so the
        // refusal no longer rests on a line of `.cu`. The kernel survives,
        // so the refusal survives with it, and it is re-pinned to the
        // instantiation constant that made the tile half-width in the first
        // place. An absence guards the direction the pin cannot: a launcher
        // coming back with the old grid and nobody noticing.
        assert!(
            !FP4.contains("const int pairs_per_block = warps * 2;"),
            "quant/dequant_fp4.cu opens the grouped sibling's half-width tile\n\
             again. The refusal in `families/quant.rs` is written against the\n\
             kernel; if there is a launcher once more, re-pin it to the .cu."
        );
        pinned(FP4_CUH, "quant/dequant_fp4.cuh", 469, "constexpr int kPairs = 2;", "`routed_qmv_quad` refusal");

        // ── `paged_scores` and `paged_scores_decode` ──
        //
        // FOUR launches, three of them the same three lines. Each is pinned
        // separately: `attention_naive_paged.cu` holds three prefill
        // launchers that differ only in how they reach the cache, and a
        // transcription written from one of them is evidence for that one.
        pinned(NAIVE, "attn/attention_naive_paged.cu", 35, "constexpr int BLOCK = 128;", "`paged_scores`");
        // Was three lines; one of the three prefill launchers has since been
        // deleted as unreached (§43), so this is two.
        for line in [108, 163] {
            pinned(
                NAIVE,
                "attn/attention_naive_paged.cu",
                line,
                "dim3 grid(num_requests, total_tokens, num_q_heads);",
                "`paged_scores`",
            );
        }
        pinned(
            NAIVE,
            "attn/attention_naive_paged.cu",
            110,
            "const std::size_t smem = (head_dim + BLOCK) * sizeof(float);",
            "`paged_scores`",
        );
        for line in [165] {
            pinned(
                NAIVE,
                "attn/attention_naive_paged.cu",
                line,
                "const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);",
                "`paged_scores`",
            );
        }
        pinned(
            NAIVE,
            "attn/attention_naive_paged.cu",
            111,
            "device::naive_paged_attn<BLOCK><<<grid, block, smem, stream>>>(",
            "`paged_scores`",
        );
        // `paged_scores_decode`'s LAUNCHER IS GONE (§43) — its three pins
        // with it. The rule is still stated and still transcribed below; what
        // is no longer true is that this `.cu` opens the grid. Asserted as an
        // absence for the direction `pinned` cannot see, since a pin proves
        // presence and can therefore never notice a deletion.
        assert!(
            !NAIVE.contains("device::naive_paged_decode<BLOCK><<<grid, block, smem, stream>>>("),
            "attn/attention_naive_paged.cu launches `naive_paged_decode` again.\n\
             `paged_scores_decode`'s transcription below was written from a\n\
             launcher that was deleted; if it is back, re-pin it."
        );

        // ── `mla_prepare`: the grid AND the three lines that build its y ──
        //
        // The `1 + q_blocks` is one line and the number it adds to is three
        // more, two of them a `?:` and a ceiling. All four are pinned, because
        // a rule that recomputed `heads_per_block` from a `half` that had
        // stopped being `rope / 2` would agree with a pin of the grid alone.
        pinned(MLA, "attn/mla_paged.cu", 50, "constexpr int BS = 256;", "`mla_prepare`");
        pinned(MLA, "attn/mla_paged.cu", 53, "const int half = rope / 2;", "`mla_prepare`");
        pinned(
            MLA,
            "attn/mla_paged.cu",
            58,
            "const int heads_per_block = half >= BS ? 1 : (BS / half);",
            "`mla_prepare`",
        );
        pinned(
            MLA,
            "attn/mla_paged.cu",
            59,
            "const int q_blocks = (heads + heads_per_block - 1) / heads_per_block;",
            "`mla_prepare`",
        );
        pinned(MLA, "attn/mla_paged.cu", 67, "dim3 grid(total_tokens, 1 + q_blocks);", "`mla_prepare`");
        pinned(
            MLA,
            "attn/mla_paged.cu",
            68,
            "device::mla_prepare<BS><<<grid, BS, 0, stream>>>(",
            "`mla_prepare`",
        );

        // ── the three `qkv_fused` grids ──
        pinned(QKV, "attn/qkv_fused.cu", 221, "constexpr int BLOCK = 256;", "`rows_packed_heads`");
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            222,
            "dim3 grid(num_rows, num_q_heads + num_kv_heads);",
            "`rows_packed_heads`",
        );
        pinned(QKV, "attn/qkv_fused.cu", 98, "constexpr int BLOCK = 128;", "`rows_packed_heads_narrow`");
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            99,
            "dim3 grid(num_requests, num_q_heads + num_kv_heads);",
            "`rows_packed_heads_narrow`",
        );
        pinned(QKV, "attn/qkv_fused.cu", 51, "constexpr int WARP_BLOCK = 256;", "`warp_packed_heads`");
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            52,
            "const int total_units = num_requests * (num_q_heads + num_kv_heads);",
            "`warp_packed_heads`",
        );
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            53,
            "dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));",
            "`warp_packed_heads`",
        );

        // ── the two `rope_table != nullptr` tests `Term::Present` reproduces ──
        //
        // Not geometry, and pinned anyway: `Specialisation`'s `because` field
        // cites these two lines, and a citation this file cannot check is one
        // nobody will. The trailing backslashes are the macro's; `:56` is
        // inside `LAUNCH_QKV_DECODE_POST_WARP` and `:100` is not.
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            56,
            "if (rope_table != nullptr) {                                         \\",
            "`QKV_DECODE_WARP`",
        );
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            58,
            "(HEAD_DIM_VALUE), true><<<warp_grid, WARP_BLOCK, 0, stream>>>( \\",
            "`QKV_DECODE_WARP`",
        );
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            71,
            "(HEAD_DIM_VALUE), false><<<warp_grid, WARP_BLOCK, 0, stream>>>( \\",
            "`QKV_DECODE_WARP`",
        );
        pinned(QKV, "attn/qkv_fused.cu", 100, "if (rope_table != nullptr) {", "`QKV_DECODE_BLOCK`");
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            101,
            "device::qkv_decode_qk_norm_rope_write_kv<BLOCK, true>",
            "`QKV_DECODE_BLOCK`",
        );
        pinned(
            QKV,
            "attn/qkv_fused.cu",
            126,
            "device::qkv_decode_qk_norm_rope_write_kv<BLOCK, false>",
            "`QKV_DECODE_BLOCK`",
        );

        // ── `altup_streams`, both launches, identical three lines ──
        for line in [17, 31] {
            pinned(ALTUP, "norm/altup.cu", line, "constexpr int BLOCK = 128;", "`altup_streams`");
        }
        for line in [18, 32] {
            pinned(
                ALTUP,
                "norm/altup.cu",
                line,
                "const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);",
                "`altup_streams`",
            );
        }
        pinned(
            ALTUP,
            "norm/altup.cu",
            19,
            "device::altup_predict<device::bf16><<<grid, BLOCK, 0, stream>>>(",
            "`altup_streams`",
        );
        pinned(
            ALTUP,
            "norm/altup.cu",
            33,
            "device::altup_correct<device::bf16><<<grid, BLOCK, 0, stream>>>(",
            "`altup_streams`",
        );

        // ── `rope/rope.cu`'s three `qk_rmsnorm` launchers ──
        //
        // THREE launchers, nine pins, and no shared line: `constexpr int
        // BLOCK = 128;` occurs four times in this file (`:45`, `:162`,
        // `:189`, `:213`) and `pinned` asserts the cited line is among the
        // matches, so each citation names its own. Two of the three get rows;
        // the third (`:213`, `_rounded`) is REFUSED, and it is pinned anyway
        // because `families::rope`'s refusal quotes it and a citation this
        // file cannot check is one nobody will.
        pinned(ROPE, "rope/rope.cu", 189, "constexpr int BLOCK = 128;", "`rope::qk_rmsnorm_rope`");
        pinned(
            ROPE,
            "rope/rope.cu",
            190,
            "dim3 grid(num_tokens, num_q_heads + num_kv_heads);",
            "`rope::qk_rmsnorm_rope`",
        );
        pinned(
            ROPE,
            "rope/rope.cu",
            191,
            "device::qk_rmsnorm_rotate<BLOCK><<<grid, BLOCK, 0, stream>>>(",
            "`rope::qk_rmsnorm_rope`",
        );
        pinned(ROPE, "rope/rope.cu", 45, "constexpr int BLOCK = 128;", "`rope::qk_rmsnorm_mrope`");
        pinned(
            ROPE,
            "rope/rope.cu",
            46,
            "dim3 grid(num_tokens, num_q_heads + num_kv_heads);",
            "`rope::qk_rmsnorm_mrope`",
        );
        pinned(
            ROPE,
            "rope/rope.cu",
            47,
            "device::qk_rmsnorm_rotate_mrope<BLOCK><<<grid, BLOCK, 0, stream>>>(",
            "`rope::qk_rmsnorm_mrope`",
        );
        pinned(ROPE, "rope/rope.cu", 213, "constexpr int BLOCK = 128;", "`families::rope`'s `_rounded` refusal");
        pinned(
            ROPE,
            "rope/rope.cu",
            214,
            "dim3 grid(num_tokens, num_q_heads + num_kv_heads);",
            "`families::rope`'s `_rounded` refusal",
        );
        pinned(
            ROPE,
            "rope/rope.cu",
            215,
            "device::qk_rmsnorm_rotate_rounded<BLOCK><<<grid, BLOCK, 0, stream>>>(",
            "`families::rope`'s `_rounded` refusal",
        );

        // ── `attn/dsv4_compress.cu`'s compressed paged attention ──
        //
        // FOUR pins for one launch, because this is the first row whose
        // shared-memory size comes off an operand: the grid is two lines, the
        // smem is two more, and `ATTN_BLOCK` is a file-scope constant 280
        // lines away. A citation that quoted only the `<<<>>>` would pin the
        // one line that names none of the three numbers.
        pinned(DSV4, "attn/dsv4_compress.cu", 37, "constexpr int ATTN_BLOCK = 128;", "`paged_scores_decode`");
        pinned(
            DSV4,
            "attn/dsv4_compress.cu",
            202,
            "dim3 grid(static_cast<unsigned>(total_tokens),",
            "`paged_scores_decode`",
        );
        pinned(
            DSV4,
            "attn/dsv4_compress.cu",
            205,
            "(static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);",
            "`paged_scores_decode`",
        );
        pinned(
            DSV4,
            "attn/dsv4_compress.cu",
            206,
            "device::compressed_attn_paged<<<grid, ATTN_BLOCK, smem, stream>>>(",
            "`paged_scores_decode`",
        );

        // ── `attn/attention_flashinfer.cu`'s score fold, pinned for a REFUSAL ──
        //
        // No rule states this launcher, so nothing below transcribes it. It is
        // pinned for the same reason `quant/dequant_fp4.cu:99` and
        // `rope/rope.cu:213-215` are: `attn/attention_flashinfer.cuh`'s header
        // refuses the row IN THESE TWO LINES and cites them by number, and a
        // citation this file cannot check is one nobody will. The whole content
        // of the refusal is that `grid.y` is a LITERAL 64 — a grid-stride
        // fanout, not a dimension of anything — over a REQUEST count on
        // `grid.x`, at 256 threads and no shared memory, and that no rule in
        // the vocabulary carries a literal grid axis. If either line changes,
        // the refusal has to be re-derived rather than re-worded.
        //
        // `LaunchRule::PerRequest` is `dim3(requests)` at 256 with no shared
        // memory and is the near miss that matters — the same `grid.x`, the
        // same block, `grid.y` of ONE. The body strides
        // `i += blockDim.x * gridDim.y`, so a `grid.y` of 1 computes THE SAME
        // FLOATS in 64x fewer blocks. A wrong rule visible only as latency is
        // one no test in this tree could fail on, which is exactly why the
        // refusal is written down instead of the row.
        pinned(
            FLASHINFER,
            "attn/attention_flashinfer.cu",
            782,
            "const dim3 grid(static_cast<unsigned>(num_requests), 64u);",
            "`attention_flashinfer.cuh`'s fold-heads refusal",
        );
        pinned(
            FLASHINFER,
            "attn/attention_flashinfer.cu",
            783,
            "device::attn_score_fold_heads<<<grid, 256, 0, stream>>>(",
            "`attention_flashinfer.cuh`'s fold-heads refusal",
        );

        // ── the three literal-grid launchers: `Single`, `SingleWarp` ──
        //
        // These are the first rules whose grid is a CONSTANT, so the pin is
        // the whole of the evidence: there is no arithmetic below to check
        // them against, only the text. `<<<1, 256>>>` and `<<<1, 32>>>` are
        // pinned as written rather than reconstructed, because the number a
        // reader must be able to verify is the `1`.
        pinned(
            SLOT_OPS,
            "layout/slot_ops.cu",
            39,
            "constexpr int kThreads = 256;",
            "`runtime::launch::single`",
        );
        pinned(
            SLOT_OPS,
            "layout/slot_ops.cu",
            40,
            "device::copy_if_valid_slot<<<1, kThreads, 0, stream>>>(",
            "`runtime::launch::single`",
        );
        pinned(
            KV_PAGED,
            "attn/kv_paged.cu",
            442,
            "device::build_window_page_view<<<1, 256, 0, stream>>>(",
            "`runtime::launch::single`",
        );
        pinned(
            KV_PAGED,
            "attn/kv_paged.cu",
            459,
            "device::build_full_split_view<<<1, 32, 0, stream>>>(",
            "`runtime::launch::single_warp`",
        );

        // ── `PerRequest`, and the three launchers whose verdict splits ──
        //
        // `attn/attention_naive.cu:174` gets the rule; `page_compact.cu:45`
        // and `:48` have the SAME text and keep `LaunchRule::PerRow`, because
        // `dsl::cuda::compact_page_csr`'s result is `Shape([Dim::Requests])`
        // and their fire's rectangle already IS the request count. Both are
        // pinned, and the second pair is pinned for a refusal-to-move rather
        // than for a rule — `runtime::launch::per_request` and
        // `families::attn::PAGE_COMPACT` both cite these two lines by number
        // to make that argument, and a citation this file cannot check is one
        // nobody will.
        pinned(
            NAIVE_UNPAGED,
            "attn/attention_naive.cu",
            89,
            "device::mtp_update_pending_hidden<bf16><<<num_requests, BLOCK, 0, stream>>>(",
            "`runtime::launch::per_request`",
        );
        pinned(
            NAIVE_UNPAGED,
            "attn/attention_naive.cu",
            24,
            "constexpr int BLOCK = device::BLOCK;",
            "`runtime::launch::per_request`",
        );
        pinned(
            PAGE_COMPACT,
            "attn/page_compact.cu",
            45,
            "<<<num_requests, device::kBlock, 0, stream>>>(",
            "`runtime::launch::per_request`'s argument for NOT moving these two",
        );
        pinned(
            PAGE_COMPACT,
            "attn/page_compact.cu",
            48,
            "<<<num_requests, device::kBlock, 0, stream>>>(",
            "`runtime::launch::per_request`'s argument for NOT moving these two",
        );

        // ── `norm/rmsnorm.cu`'s EMIT_FP16 arm ──
        //
        // `families::norm::RMSNORM_SIGS[10]`. Three lines, because the
        // template arguments and the `<<<>>>` are on separate lines here and
        // the row's claim is about both: the instantiation says
        // `EMIT_FP16=true`, and the launch says 512 where the row states
        // 256. Both halves are cited so neither can drift alone.
        pinned(
            RMSNORM,
            "norm/rmsnorm.cu",
            69,
            "constexpr int VBLOCK = 512;",
            "`families::norm::RMSNORM_SIGS[10]`",
        );
        pinned(
            RMSNORM,
            "norm/rmsnorm.cu",
            71,
            "device::rmsnorm_vec8<VBLOCK, /*WEIGHT_PLUS_ONE=*/false, /*EMIT_FP16=*/true>",
            "`families::norm::RMSNORM_SIGS[10]`",
        );
        pinned(
            RMSNORM,
            "norm/rmsnorm.cu",
            72,
            "<<<grid, VBLOCK, 0, stream>>>(",
            "`families::norm::RMSNORM_SIGS[10]`",
        );

        // ── `ssm/nemotron_h.cu`'s two `mamba_split` arms ──
        //
        // `families::ssm::NEMOTRON_H_SIGS[3]` states the SECOND. The first
        // is pinned too, and deliberately: the row's doc argues that
        // `ElementwiseIn` would OVER-LAUNCH it, and that argument is about
        // `:38`'s `conv_dim + num_heads` extent. A refusal's evidence gets
        // the same pin a claim's does.
        pinned(
            NEMOTRON,
            "ssm/nemotron_h.cu",
            36,
            "constexpr int BLOCK = 256;",
            "`families::ssm::NEMOTRON_H_SIGS[3]`",
        );
        pinned(
            NEMOTRON,
            "ssm/nemotron_h.cu",
            38,
            "const int conv_dt_total = N * (conv_dim + num_heads);",
            "`families::ssm::NEMOTRON_H_SIGS[3]`'s argument for holding the SIBLING arm",
        );
        pinned(
            NEMOTRON,
            "ssm/nemotron_h.cu",
            48,
            "const int grid = (total + BLOCK - 1) / BLOCK;",
            "`families::ssm::NEMOTRON_H_SIGS[3]`",
        );
        pinned(
            NEMOTRON,
            "ssm/nemotron_h.cu",
            49,
            "device::mamba_split<<<grid, BLOCK, 0, stream>>>(",
            "`families::ssm::NEMOTRON_H_SIGS[3]`",
        );

        // ── `rope/rope.cu`'s FOURTH `qk_rmsnorm` launcher ──
        //
        // The devwin form, and the fourth occurrence of `constexpr int BLOCK
        // = 128;` in this file. `pinned` asserts the cited line is AMONG the
        // matches, so this citation names `:162` and the three above name
        // `:45`, `:189` and `:213`; the grid line is likewise the third
        // `dim3 grid(...)` here and `n_max` is what tells it from the other
        // two.
        pinned(
            ROPE,
            "rope/rope.cu",
            162,
            "constexpr int BLOCK = 128;",
            "`rope::qk_rmsnorm_rope_devwin`",
        );
        pinned(
            ROPE,
            "rope/rope.cu",
            163,
            "dim3 grid(n_max, num_q_heads + num_kv_heads);",
            "`rope::qk_rmsnorm_rope_devwin`",
        );
        pinned(
            ROPE,
            "rope/rope.cu",
            164,
            "device::qk_rmsnorm_rotate_devwin<BLOCK><<<grid, BLOCK, 0, stream>>>(",
            "`rope::qk_rmsnorm_rope_devwin`",
        );
    }

    /// The launcher's numbers, and the rule's, at every shape the towers and
    /// the recurrence actually run.
    ///
    /// The rectangle handed to `eval` is the one the LAUNCHER walks, which is
    /// not always the one the statement would name. Three of the fifteen
    /// differ, and all three are here at the launcher's rectangle on purpose:
    ///
    /// * `k_av`'s width is one head's 64, not the 768 of the row it belongs
    ///   to — the host loops the twelve heads. A row inheriting the
    ///   statement's width launches 12x the blocks.
    /// * `k_rel_pos_enc`'s rows are 13 POSITIONS, not the token count.
    /// * `k_glu`'s width is the OUTPUT `Hd`; its input row is `2 * Hd`
    ///   (`gemma4_audio.cuh:133` reads `x[n * 2 * D + d]`). A row inheriting
    ///   `In(0)`'s width launches twice the grid.
    ///
    /// The rule is right about the arithmetic in all three and a ROW for any
    /// of them would still be wrong — a `Dims` field that is PRESENT and
    /// WRONG rather than absent, which no refusal catches because nothing is
    /// missing. That is a different defect in a different file, it is why
    /// `tile16`'s doc names them, and it is the same shape as
    /// [`Rule::RowsPerHead`]'s absent-versus-present case one file over.
    #[test]
    fn every_transcribed_launcher_agrees_with_its_rule() {
        /// One comparison: what it is, the launcher's grid, the launcher's
        /// block, the rectangle to evaluate at, and the rule under test.
        type Case = (&'static str, [u32; 3], [u32; 3], Dims, Rule);

        pins();

        // gemma-4 vision: hidden 768, 12 heads of 64, 4096 patches.
        // gemma-4 audio: hidden 1536 (12 x 128), 13 positions, sscp 128/32.
        // qwen3-vl: merge unit 4 over hidden 1024 -> W 4096.
        let cases: &[Case] = &[
            // ── tile16, `vision/gemma4_vision.cu` ──
            ("k_addpos_grid2d G2(Hd,N) :144", g2(768, 4096), B2, dims(4096, 768), Rule::Tile16),
            ("k_qk G2(N,N) :151", g2(4096, 4096), B2, dims(4096, 4096), Rule::Tile16),
            ("k_av G2(64,N) :151", g2(64, 4096), B2, dims(4096, 64), Rule::Tile16),
            ("k_pool G2(Hd,N) :165", g2(768, 4096), B2, dims(4096, 768), Rule::Tile16),
            // ── tile16, `vision/gemma4_audio.cu` ──
            ("k_matmul G2(Out,N) :165", g2(1536, 512), B2, dims(512, 1536), Rule::Tile16),
            ("k_sscp_flatten (FLAT,N) :201", g2(1024, 512), B2, dims(512, 1024), Rule::Tile16),
            ("k_matmul G2(Hd,N) :203", g2(1536, 512), B2, dims(512, 1536), Rule::Tile16),
            ("k_rel_pos_enc (Hd,P) :220", g2(1536, 13), B2, dims(13, 1536), Rule::Tile16),
            ("k_qkv_scale G2(Hd,N) :240", g2(1536, 512), B2, dims(512, 1536), Rule::Tile16),
            ("k_matmul G2(Hd,P) :242", g2(1536, 13), B2, dims(13, 1536), Rule::Tile16),
            ("k_glu G2(Hd,N) :250", g2(1536, 512), B2, dims(512, 1536), Rule::Tile16),
            ("k_matmul_bias G2(OPD,N) :283", g2(2048, 512), B2, dims(512, 2048), Rule::Tile16),
            ("k_matmul G2(TXT,N) :289", g2(2560, 512), B2, dims(512, 2560), Rule::Tile16),
            // ── tile16, `vision/qwen3_vl_tower.cu` ──
            ("k_merge_gather G2(W,n_token) :165", g2(4096, 256), B2, dims(256, 4096), Rule::Tile16),
            ("k_merge_gather G2(W,n_token) :168", g2(4096, 256), B2, dims(256, 4096), Rule::Tile16),
        ];

        let mut same = 0usize;
        for (what, grid, block, d, rule) in cases {
            let got = eval(*rule, *d).expect("ported");
            assert_eq!(
                (got.grid, got.block),
                (*grid, *block),
                "{what}: the rule and the launcher disagree"
            );
            same += 1;
        }

        // ── axial_rope, both launches of `:150` on the same `rg` ──
        let tower = Dims { kv_heads: 12, head_dim: 64, ..dims(4096, 768) };
        for over in ["q", "k"] {
            let got = eval(Rule::AxialRope, tower).expect("ported");
            assert_eq!(got.grid, rg(12, 4096), "k_rope_axial2d over {over}: `dim3 rg(1,NH,N)`");
            assert_eq!(got.block, [32, 1, 1], "k_rope_axial2d over {over}: 32 threads");
            same += 1;
        }

        // ── per_row_narrow, `:189` and `:196` ──
        for (t, f, c) in [(128, 32, 128), (64, 16, 256)] {
            let (grid, block) = sscp_ln(t, f);
            let got = eval(Rule::PerRowNarrow, dims((t * f) as u32, c)).expect("ported");
            assert_eq!((got.grid, got.block), (grid, block), "k_layernorm_relu T*F={}", t * f);
            same += 1;
        }

        // ── warp_tiled_scan, `:781` (f32) and `:840` (bf16), same shape ──
        // Qwen3.5 GDN: 32 requests, 32 value heads, 128 value channels.
        for ty in ["float", "__nv_bfloat16"] {
            let (grid, block) = gdn_warp_tiled(32, 32, 128);
            let d = Dims { kv_heads: 32, ..dims(32, 32 * 128) };
            let got = eval(Rule::WarpTiledScan, d).expect("ported");
            assert_eq!((got.grid, got.block), (grid, block), "warp_tiled_gqa<{ty}>");
            same += 1;
        }

        // ── routed_qmv, `:74` ──
        //
        // `routes` is the launcher's own `num_tokens * top_k` at `:70`, kept
        // as a product so the transcription reads as the line it came from.
        // `num_tokens * top_k` at `:70`, at one token and eight experts.
        let routes: i32 = 8;
        let (grid, block) = wna16_gate_up(routes, 768);
        let got = eval(Rule::RoutedQmv, Dims { experts_per_token: 8, ..dims(1, 768) })
            .expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "wna16_gate_up_decode");
        same += 1;

        // ── rows_flat, `moe/dsv4_routing.cu:59` ──
        let (grid, block) = dsv4_route(2048);
        let got = eval(Rule::RowsFlat, dims(2048, 8)).expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "hash_route_lookup");
        same += 1;

        // ── routed_qmv_transposed, `:103` — and the transposition, measured ──
        //
        // Same file, same divisor, same block, axes swapped. The second
        // assertion is the one that matters: it fires the OTHER rule at the
        // same numbers and requires the grids to differ, so a rule that had
        // been written as its twin would fail here rather than agree with a
        // quote of the wrong line.
        // `num_tokens * top_k` at `:70`, at one token and eight experts.
        let routes: i32 = 8;
        let (grid, block) = wna16_down(7168, routes);
        let d = Dims { experts_per_token: 8, ..dims(1, 7168) };
        let got = eval(Rule::RoutedQmvTransposed, d).expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "wna16_down_decode");
        assert_ne!(
            got.grid,
            eval(Rule::RoutedQmv, d).expect("ported").grid,
            "the two wna16 decode grids are transposed and the rules must be too"
        );
        same += 1;

        // ── routed_qmv_quad, `dequant_fp4.cu:69-70` and `:155-156` ──
        //
        // gpt-oss's own shape: 2 880-wide intermediate, top_k 4. The
        // STACKED width goes in — `k * intermediate` — because that is what
        // the statements declare and what `Dims::width` therefore is, and
        // the rule divides it back out before it slabs. A transcription
        // handed the per-route width would agree with a rule that forgot the
        // divide, so the case is written the way `jit_dims` fills it.
        let top_k: i32 = 4;
        let tokens: i32 = 2;
        let intermediate: i32 = 2880;
        let (grid, block) = mxfp4_gate_up(tokens * top_k, intermediate);
        let d = Dims {
            experts_per_token: top_k as u32,
            in_width: top_k as u32,
            ..dims(tokens as u32, (top_k * intermediate) as u32)
        };
        let got = eval(Rule::RoutedQmvQuad, d).expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "mxfp4_moe_gate_up_decode");
        // The near miss, at the shape the refusal named: `ceil(2880/8) = 360`
        // of 256 against `ceil(2880/16) = 180` of 128. `grid.x` agrees, which
        // is why this asserts on the whole launch and not on `grid.x`.
        assert_ne!(
            got.grid,
            eval(Rule::RoutedQmv, d).expect("ported").grid,
            "the two MoE decode tiles are 16 and 8 and the rules must differ"
        );
        same += 1;

        // The down leg, same rule, same axis order — the claim
        // `mxfp4_down`'s doc makes, checked against `dequant_wna16.cu`'s
        // opposite one.
        let hidden: i32 = 2880;
        let (grid, block) = mxfp4_down(tokens * top_k, hidden);
        let d = Dims {
            experts_per_token: top_k as u32,
            in_width: top_k as u32,
            ..dims(tokens as u32, (top_k * hidden) as u32)
        };
        let got = eval(Rule::RoutedQmvQuad, d).expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "mxfp4_moe_down_decode");
        assert_ne!(
            got.grid,
            eval(Rule::RoutedQmvTransposed, d).expect("ported").grid,
            "`dequant_fp4.cu` keeps the routes on `grid.x` in BOTH legs where \
             `dequant_wna16.cu` swaps them; if these agreed one file was read as \
             the other"
        );
        same += 1;

        // ── paged_scores, the three prefill launchers at `:111`, `:198`, `:248` ──
        //
        // 4 requests, 512 tokens, 32 query heads of 128 — and the shared
        // memory is compared too, which no other case in this module does
        // because no other launcher computes one.
        let (grid, block, smem) = naive_paged(4, 512, 32, 128);
        let d = Dims { requests: 4, q_heads: 32, head_dim: 128, ..dims(512, 4096) };
        for at in [":111", ":198", ":248"] {
            let got = eval(Rule::PagedScores, d).expect("ported");
            assert_eq!((got.grid, got.block), (grid, block), "naive_paged_attn {at}");
            assert_eq!(got.smem, smem, "naive_paged_attn {at}: (head_dim + BLOCK) * 4");
            same += 1;
        }

        // ── paged_scores_decode, `:150` ──
        let (grid, block, smem) = naive_paged_decode(4, 32, 128);
        let d = Dims { q_heads: 32, head_dim: 128, ..dims(4, 4096) };
        let got = eval(Rule::PagedScoresDecode, d).expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "naive_paged_decode :150");
        assert_eq!(got.smem, smem, "naive_paged_decode :150: (head_dim + BLOCK) * 4");
        same += 1;

        // ── mla_prepare, `:74` ──
        //
        // DeepSeek-V3: 128 query heads, `qk_rope_head_dim` 64, so `half` is 32
        // and `heads_per_block` is 8 — `q_blocks` 16 and `grid.y` 17. Kimi
        // K2's 64 heads at the same rope width give 9.
        for (tokens, heads, rope) in [(512, 128, 64), (1, 64, 64), (7, 16, 512)] {
            let (grid, block) = mla_prepare(tokens, heads, rope);
            let d = Dims { q_heads: heads as u32, rotary_dims: rope as u32, ..dims(tokens as u32, 576) };
            let got = eval(Rule::MlaPrepare, d).expect("ported");
            assert_eq!((got.grid, got.block), (grid, block), "mla_prepare heads={heads} rope={rope}");
            same += 1;
        }

        // ── rows_packed_heads, `:248`, and its narrow twin at `:102`/`:127` ──
        //
        // Qwen3-style: 32 query heads, 8 kv heads of 128. `grid.y` is 40 and
        // not 8, which is the entire content of the refusal `GatedRms` earned.
        let d = Dims { q_heads: 32, kv_heads: 8, head_dim: 128, ..dims(512, 4096) };
        let (grid, block) = rows_packed_heads(512, 32, 8, 256);
        let got = eval(Rule::RowsPackedHeads, d).expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "qkv_packed_… :248");
        same += 1;

        let (grid, block) = rows_packed_heads(512, 32, 8, 128);
        let got = eval(Rule::RowsPackedHeadsNarrow, d).expect("ported");
        assert_eq!((got.grid, got.block), (grid, block), "qkv_decode_… :102 / :127");
        same += 1;

        // ── warp_packed_heads, `:58` and `:71` on the same `warp_grid` ──
        let (grid, block) = warp_packed_heads(512, 32, 8);
        for arm in ["true", "false"] {
            let got = eval(Rule::WarpPackedHeads, d).expect("ported");
            assert_eq!(
                (got.grid, got.block),
                (grid, block),
                "qkv_decode_…_warp<HEAD_DIM, {arm}>"
            );
            same += 1;
        }

        // ── altup_streams, `:19` and `:33` on the same `grid` ──
        //
        // gemma-3n: 4 residual streams of 2048, so the value the statement
        // sees is 8192 wide and the launcher's `H` is 2048.
        let (grid, block) = altup(512, 4, 2048);
        let d = Dims { altup_streams: 4, ..dims(512, 4 * 2048) };
        for which in ["predict", "correct"] {
            let got = eval(Rule::AltUpStreams, d).expect("ported");
            assert_eq!((got.grid, got.block), (grid, block), "altup_{which} :18 / :32");
            same += 1;
        }

        // ── `rope/rope.cu`'s packed head axis, `:190-191` and `:46-47` ──
        //
        // The SAME rule as `:102`/`:127` above and a different launcher, which
        // is the point: `RowsPackedHeadsNarrow` was ported from `qkv_fused.cu`
        // and `families::rope` refused these two for a packed head axis no
        // rule stated. Both are transcribed from `rope.cu`'s own text.
        //
        // Three shapes, and the split varies in all three. §22.7's near miss
        // was byte-identical at one shape and 10 221 of 20 480 at another, so
        // a single q/kv ratio certifies nothing: Qwen3-style 32/8, a
        // multi-query 16/1 where `kv_heads` is the axis a rule reading only
        // `q_heads` would drop, and a 1/1 decode row where every axis is 1 or
        // 2 and a transposed grid is still a legal launch.
        for (tokens, q_heads, kv_heads, head_dim) in
            [(512, 32, 8, 128), (7, 16, 1, 64), (1, 1, 1, 256)]
        {
            let (grid, block) = rope_packed_heads(tokens, q_heads, kv_heads);
            let d = Dims {
                q_heads: q_heads as u32,
                kv_heads: kv_heads as u32,
                head_dim: head_dim as u32,
                ..dims(tokens as u32, (q_heads * head_dim) as u32)
            };
            for which in ["qk_rmsnorm_rotate :191", "qk_rmsnorm_rotate_mrope :47"] {
                let got = eval(Rule::RowsPackedHeadsNarrow, d).expect("ported");
                assert_eq!((got.grid, got.block), (grid, block), "{which} q={q_heads} kv={kv_heads}");
                assert_eq!(got.smem, 0, "{which}: the launcher passes 0");
                same += 1;
            }
        }

        // ── paged_scores_decode's THIRD launcher, `dsv4_compress.cu:319-323` ──
        //
        // The rule already agreed with `attention_naive_paged.cu:150` above.
        // This is the launcher that gives it a row, and the field under test
        // is `smem`: `(head_dim + 128) * sizeof(float)`, computed from an
        // OPERAND width. `head_dim` varies across the three shapes for that
        // reason and no other — it is the only axis a fixed-smem reading gets
        // wrong, and too little dynamic shared memory does not fault, it reads
        // another block's.
        for (tokens, q_heads, head_dim) in [(4, 32, 128), (3, 8, 64), (17, 4, 192)] {
            let (grid, block, smem) = compressed_paged(tokens, q_heads, head_dim);
            let d = Dims {
                q_heads: q_heads as u32,
                head_dim: head_dim as u32,
                ..dims(tokens as u32, (q_heads * head_dim) as u32)
            };
            let got = eval(Rule::PagedScoresDecode, d).expect("ported");
            assert_eq!((got.grid, got.block), (grid, block), "compressed_attn_paged :323");
            assert_eq!(got.smem, smem, "compressed_attn_paged :322: (head_dim + 128) * 4");
            same += 1;
        }

        println!("{same} / {same} launcher comparisons SAME");
        assert_eq!(same, 48, "a comparison was added or lost without the count moving");
    }

    /// **The refusal that holds `rope::qk_rmsnorm_rope_bf16_rounded`, measured.**
    ///
    /// The two rows that landed above and this one launch the same three
    /// numbers out of the same three lines of the same file. What separates
    /// them is not the rule's arithmetic — it is WHOSE `num_kv_heads` the
    /// second axis is.
    ///
    /// `rope/rope.cu:214` reads the launcher's ARGUMENT.
    /// `model-compiler/src/dsl.rs:6588` records
    /// `rope::qk_rmsnorm_rope_bf16_rounded` from
    /// `qk_rmsnorm_rope_rounded_q_only`, gemma-4's shared sliding layer
    /// (`model/src/gemma_4/forward/mod.rs:217`), and the ahead-of-time row
    /// binds `num_kv_heads <- Or(Div(Width(Out(1)), head_dim), Lit::I32(0))`
    /// (`table/rope.rs:191-218`) — zero, because there is no second result.
    ///
    /// `Dims::kv_heads` cannot be that zero. It is `extent(ctx.num_kv_heads)`
    /// (`driver-cuda/src/bind/mod.rs:1379`), a fire-wide fact, non-zero on
    /// every gemma-4 layer. So the rule opens a grid the launcher does not,
    /// and this test is that sentence as arithmetic: the same `eval` that
    /// AGREES with `:214` when the statement names both banks DISAGREES with
    /// it when the statement names one.
    ///
    /// Stated here rather than in prose because a refusal nothing evaluates is
    /// a refusal that goes stale silently — which is what the other six rows
    /// in this pass were.
    #[test]
    fn the_rounded_rope_grid_is_the_fires_head_count_and_not_the_statements() {
        pins();

        // gemma-4's shared sliding layer: 8 query heads, 4 kv heads of 256,
        // and a second one at a different ratio so a single shape cannot
        // certify the disagreement (hazard 1).
        for (tokens, q_heads, kv_heads, head_dim) in [(6, 8, 4, 256), (1, 16, 2, 128)] {
            let d = Dims {
                q_heads: q_heads as u32,
                kv_heads: kv_heads as u32,
                head_dim: head_dim as u32,
                ..dims(tokens as u32, (q_heads * head_dim) as u32)
            };
            let got = eval(Rule::RowsPackedHeadsNarrow, d).expect("ported");

            // Both banks stated: `:214` is `num_q_heads + num_kv_heads` and
            // the rule is `q_heads + kv_heads`. The same number.
            let (both, block) = rope_packed_heads(tokens, q_heads, kv_heads);
            assert_eq!(
                (got.grid, got.block),
                (both, block),
                "rope.cu:214 with both banks stated"
            );

            // Q-ONLY: the launcher is handed `num_kv_heads = 0`, so `:214`
            // computes `q + 0`. The rule cannot see that zero.
            let (q_only, _) = rope_packed_heads(tokens, q_heads, 0);
            assert_ne!(
                got.grid, q_only,
                "rope.cu:214 at the q-only site: if these agreed the refusal \
                 would be stale and `_rounded` would be a row"
            );
            assert_eq!(
                got.grid[1] - q_only[1],
                kv_heads as u32,
                "the excess is exactly the fire's kv bank — {kv_heads} columns \
                 of blocks addressing `k + (n * 0 + local) * head_dim` with \
                 `k == nullptr`"
            );
        }
    }

    /// The same transcriptions, mutated the four ways a reader most easily
    /// misreads them, so that agreeing is a measurement and not a tautology.
    ///
    /// Every one of these four produces a grid CUDA accepts and hardware
    /// reports success for. None is a crash, a bounds fault or an out-of-range
    /// dimension; each is a legal launch against the wrong cells, which is the
    /// only failure mode this whole file exists for.
    #[test]
    fn the_transcriptions_disagree_when_mutated() {
        pins();

        // 1. `G2(X, Y)`'s two arguments swapped. `k_av`'s launcher walks one
        //    head's 64 channels over 4096 patches; read the other way it is
        //    4096 channels over 64 rows. Same block, same total threads, and
        //    a sixteenth of the rectangle computed sixteen times.
        let av = eval(Rule::Tile16, dims(4096, 64)).expect("ported");
        assert_eq!(av.grid, g2(64, 4096));
        assert_ne!(av.grid, g2(4096, 64), "G2's first argument is the WIDTH");
        let transposed = g2(4096, 64);

        // 2. `B2` flattened to a 1-D block of the same 256 threads.
        //    `gemma4_naive_kernels.cuh:117` reads `threadIdx.y`, so with
        //    `blockDim.y == 1` every thread of every block computes row zero.
        assert_eq!(av.block, B2);
        assert_ne!(av.block, [256, 1, 1], "the tile is 16 by 16 and not 256 by 1");

        // 3. `rg(1, NH, N)` packed into two axes, which is what a rule
        //    written from habit produces — `Rule::PerHead`'s `[heads, rows]`.
        //    The kernel reads `blockIdx.z` as its token, so the packed grid
        //    launches 49 152 blocks that all address token zero.
        let tower = Dims { kv_heads: 12, head_dim: 64, ..dims(4096, 768) };
        let axial = eval(Rule::AxialRope, tower).expect("ported");
        assert_eq!(axial.grid, rg(12, 4096));
        let packed = eval(Rule::PerHead, tower).expect("ported").grid;
        assert_ne!(axial.grid, packed, "the counts are on y and z, not x and y");

        // 4. The SSCP layernorm at `Rule::PerRow`'s 256 instead of its own
        //    128. The grid is identical and only the block moves: the fold
        //    sums `(blockDim.x + 31) / 32` per-warp partials serially, so the
        //    two block widths add the same values in a different order and
        //    the encoder is no longer the checkpoint's.
        let sscp = dims(128 * 32, 128);
        let narrow = eval(Rule::PerRowNarrow, sscp).expect("ported");
        let wide = eval(Rule::PerRow, sscp).expect("ported");
        assert_eq!(narrow.grid, wide.grid, "the mutation is invisible in the grid");
        assert_ne!(narrow.block, wide.block);

        println!("mutation contrasts (mutant  vs  launcher):");
        println!("  k_av transposed          -> {:?}  vs  {:?}", transposed, av.grid);
        println!("  Tile16 block flattened   -> {:?}  vs  {:?}", [256, 1, 1], av.block);
        println!("  AxialRope packed into 2  -> {:?} vs {:?}", packed, axial.grid);
        println!("  PerRowNarrow at PerRow   -> {}        vs  {}", wide.block[0], narrow.block[0]);
    }

    /// The eight rules this round added, each perturbed the way its own
    /// launcher is most easily misread — and the wrong numbers printed.
    ///
    /// Separate from the four above because the perturbations are different in
    /// kind. Those four are misreadings of a transcription. These are the
    /// NEAREST PORTED RULE fired at the same `Dims` — the thing an author who
    /// decided "close enough" would actually have shipped — plus, where the
    /// refusal named one, the exact arithmetic slip it warned about. Every
    /// mutant here is a launch CUDA accepts.
    #[test]
    fn the_new_rules_disagree_with_their_near_misses() {
        pins();

        let mut lines: Vec<String> = Vec::new();
        // A macro rather than a closure: two of the twelve contrasts print a
        // shared-memory size or a thread count instead of a grid, and they
        // push to `lines` directly. A `FnMut` closure would hold the borrow
        // across those pushes.
        macro_rules! show {
            ($what:expr, $mutant:expr, $real:expr) => {{
                let (what, mutant, real): (&str, [u32; 3], [u32; 3]) = ($what, $mutant, $real);
                assert_ne!(mutant, real, "{what}: the mutation is invisible");
                lines.push(format!("  {what:<38} -> {mutant:?}  vs  {real:?}"));
            }};
        }

        // 1. `PagedScores` versus its refusal's own near-miss. `SdpaVector`'s
        //    smem is `(rows + 256) * 4` — the BLOCK added to the wrong extent
        //    — so at 512 tokens it asks for 3 072 bytes where the launcher
        //    asks for 1 024, and its grid is one axis where the launcher's is
        //    three.
        let d = Dims { requests: 4, q_heads: 32, head_dim: 128, ..dims(512, 4096) };
        let real = eval(Rule::PagedScores, d).expect("ported");
        let near = eval(Rule::SdpaVector, d).expect("ported");
        show!("PagedScores as SdpaVector", near.grid, real.grid);
        assert_ne!(near.smem, real.smem, "the smem is the refusal's whole point");
        lines.push(format!(
            "  {:<38} -> {} bytes  vs  {} bytes",
            "PagedScores smem as (rows+256)*4", near.smem, real.smem
        ));

        // 2. `PagedScores`' three axes read as two — requests folded into the
        //    token count, which is what `[rows, heads, 1]` looks like from a
        //    habit. The kernel reads `blockIdx.x` as its request and
        //    `blockIdx.y` as its token, so this launches 512 requests of one
        //    token each into a CSR with four.
        let folded = eval(Rule::PagedScoresDecode, Dims { rows: 512, ..d }).expect("ported");
        show!("PagedScores folded to 2 axes", folded.grid, real.grid);

        // 2b. And the same confusion the other way, which is the one that
        //    faults rather than merely lying. `naive_paged_decode` opens
        //    `dim3(num_requests, num_q_heads)` — ONE token a request, the one
        //    being decoded — and its `attn_out` is indexed
        //    `(request * num_q_heads + head) * head_dim`. Given the prefill's
        //    third axis it writes 512 tokens' worth of a buffer that holds
        //    four, 511/512 of it past the end.
        let decode = eval(Rule::PagedScoresDecode, Dims { rows: 4, ..d }).expect("ported");
        show!("PagedScoresDecode with a token axis", real.grid, decode.grid);
        assert_eq!(decode.smem, real.smem, "both forms share `(head_dim + 128) * 4`");

        // 3. `MlaPrepare` without the `1 +`. The KV lane is `blockIdx.y == 0`
        //    and every query lane is `blockIdx.y - 1`, so dropping it does not
        //    lose one block in seventeen — it SHIFTS every query head down by
        //    one block and drops the last, while the norm, the `k_pe` rotation
        //    and the paged write never run at all.
        let d = Dims { q_heads: 128, rotary_dims: 64, ..dims(512, 576) };
        let real = eval(Rule::MlaPrepare, d).expect("ported");
        let q_blocks = real.grid[1] - 1;
        show!("MlaPrepare without the KV lane", [512, q_blocks, 1], real.grid);

        // 4. `MlaPrepare` reading `head_dim` where it reads `rotary_dims`. An
        //    MLA head is `kv_lora_rank + qk_rope_head_dim` = 576, so `half` is
        //    288, `heads_per_block` is 1 and `q_blocks` is 128 — 129 lanes
        //    where the launcher opens 17, each block covering an eighth of the
        //    heads it was sized for.
        let as_head_dim = eval(Rule::MlaPrepare, Dims { rotary_dims: 576, ..d }).expect("ported");
        show!("MlaPrepare on head_dim (576)", as_head_dim.grid, real.grid);

        // 5. `RowsPackedHeads` as `GatedRms` — `[rows, kv_heads, 1]` at the
        //    same 256 threads and the same zero smem, which is the refusal's
        //    exact sentence. `grid.y` is short by every query head, so the Q
        //    lanes are never launched and `q_out` keeps whatever it held.
        let d = Dims { q_heads: 32, kv_heads: 8, head_dim: 128, ..dims(512, 4096) };
        let real = eval(Rule::RowsPackedHeads, d).expect("ported");
        let gated = eval(Rule::GatedRms, d).expect("ported");
        show!("RowsPackedHeads as GatedRms", gated.grid, real.grid);
        assert_eq!(gated.block, real.block, "the block is identical — only y moves");
        assert_eq!(gated.smem, real.smem, "and so is the shared memory");

        // 6. The two block widths of the same grid. `:246` launches 256 and
        //    `:99` launches 128, both over `dim3(rows, q + kv)`; the kernel
        //    sizes `__shared__ float buf[BLOCK]` and folds it by halving, so
        //    swapping them sums the same head in a different order.
        let narrow = eval(Rule::RowsPackedHeadsNarrow, d).expect("ported");
        assert_eq!(narrow.grid, real.grid, "the mutation is invisible in the grid");
        assert_ne!(narrow.block, real.block);
        lines.push(format!(
            "  {:<38} -> {} threads  vs  {} threads",
            "RowsPackedHeads at the narrow block", narrow.block[0], real.block[0]
        ));

        // 7. `WarpPackedHeads` divided by the BLOCK instead of by the warps
        //    per block — `256` where the launcher writes `WARP_BLOCK / 32`.
        //    An eighth of the blocks, so seven eighths of the (request, head)
        //    units are never opened.
        let real = eval(Rule::WarpPackedHeads, d).expect("ported");
        let units = 512 * (32 + 8);
        show!("WarpPackedHeads over 256 not 8", [(units as u32).div_ceil(256), 1, 1], real.grid);

        // 8. `RoutedQmvTransposed` as `RoutedQmv` — the transposition, at
        //    Kimi K2.6's decode shape. Neither grid contains the other.
        let d = Dims { experts_per_token: 8, ..dims(1, 7168) };
        let real = eval(Rule::RoutedQmvTransposed, d).expect("ported");
        show!("RoutedQmvTransposed as RoutedQmv", eval(Rule::RoutedQmv, d).expect("ported").grid, real.grid);

        // 9. `AltUpStreams` as `WarpTiledScan`, the near-miss the refusal
        //    named. Same 128-wide block, same three axes, and both of the
        //    other two wrong. `kv_heads` is set to 4 HERE so that only the z
        //    axis moves and the factor is exact; contrast 10 below is the y.
        //
        //    The z axes differ by 32 because the two kernels tile at different
        //    granularities: `warp_tiled_gqa` gives each block four value
        //    channels (`WARPS = 4`, one per warp) and `altup_predict` gives
        //    each block 128 (`blockIdx.z * blockDim.x + threadIdx.x`, one per
        //    THREAD). Fired this way the near-miss opens 32x the blocks, 31/32
        //    of which hit `if (h >= H) return` — the harmless direction. Fired
        //    the other way, which is the one the refusal measured, this rule's
        //    `ceil(H / 128)` on that kernel is a thirty-second of the blocks
        //    and leaves 31/32 of the value channels untouched.
        let d = Dims { altup_streams: 4, kv_heads: 4, head_dim: 128, ..dims(512, 4 * 2048) };
        let real = eval(Rule::AltUpStreams, d).expect("ported");
        let scan = eval(Rule::WarpTiledScan, d).expect("ported");
        show!("AltUpStreams as WarpTiledScan", scan.grid, real.grid);
        assert_eq!(scan.block, real.block, "the block width is the same 128");
        assert_eq!(
            scan.grid[2],
            32 * real.grid[2],
            "four channels a block against 128 is the factor between the two z axes"
        );

        // 10. `AltUpStreams` with the stream count taken from `kv_heads`, the
        //     hazard §22 records for `stated_head_dim`. Here the two happen to
        //     agree; at gemma-3n's real attention config they do not, and the
        //     rule reading the wrong field would not refuse — it would open a
        //     residual axis sized by the attention configuration.
        let skewed = Dims { altup_streams: 4, kv_heads: 8, ..dims(512, 4 * 2048) };
        let as_kv_heads = eval(Rule::AltUpStreams, Dims { altup_streams: 8, ..skewed })
            .expect("ported");
        show!("AltUpStreams on kv_heads (8)", as_kv_heads.grid, real.grid);

        println!("mutation contrasts (mutant  vs  launcher):");
        for line in &lines {
            println!("{line}");
        }
        assert_eq!(lines.len(), 12, "every new rule is perturbed and its numbers printed");
    }

    /// `PIE_QWEN35_GDN_SMEM_STEP` is gone, and cannot come back quietly.
    ///
    /// `ssm::recurrent_gated_delta_step_batched_gqa_state_bf16` used to pick
    /// between two kernels on `std::getenv("PIE_QWEN35_GDN_SMEM_STEP")`.
    /// That is the failure this whole crate exists to remove: same trace,
    /// same weights, same GPU, a different kernel, and NOTHING above the
    /// launcher able to see which one ran. A plan cannot record it, a replay
    /// cannot reproduce it, and no second backend can implement the same
    /// semantics, because the selector was invisible to everything but the
    /// process environment.
    ///
    /// §30 measured the two arms before moving them: byte-identical on the
    /// state slab AND on `out` at eight shapes, `written > 0` on every one.
    /// So the knob never chose semantics — only speed, and only downward
    /// (1.48x slower at R=511). An identical-but-slower arm is bring-up
    /// scaffolding, so the answer was DELETION, not relocation: no
    /// `Specialisation` (the geometries disagree and `agrees()` would refuse
    /// it), no `Choose` (there is nothing to choose between), no driver-side
    /// fact (nothing is left to configure).
    ///
    /// What replaces it is a SHAPE the fire already carries —
    /// `V_d == 128 && K_d == 128` — which costs zero vocabulary, is visible
    /// to a table, and is the form §26.10(b)'s `Term::IntIs` will read
    /// directly.
    ///
    /// This test is the ratchet. It is deliberately three assertions and not
    /// one, because each names a different way the knob could return: the
    /// call itself, its header, and the shape predicate that stands in its
    /// place. Comment lines are excluded from the `getenv` scan on purpose —
    /// the argument in the `.cu` has to be free to NAME the thing it
    /// deleted, or the record of why it went is unwritable.
    #[test]
    fn the_bf16_gqa_decode_step_reads_no_environment_variable() {
        let code: Vec<(usize, &str)> = GDN
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim_start().starts_with("//"))
            .map(|(i, l)| (i + 1, l))
            .collect();

        for (n, line) in &code {
            assert!(
                !line.contains("getenv"),
                "ssm/gated_delta_net.cu:{n} reads the environment in code: {line}\n\
                 a tuning knob is configuration; it is not a symbol and it is not\n\
                 geometry. If configuration must reach a launch it arrives as a\n\
                 fact the driver holds at load, never as a call inside a .cu."
            );
        }

        assert!(
            !GDN.contains("#include <cstdlib>"),
            "ssm/gated_delta_net.cu includes <cstdlib>; the only thing it was \
             ever pulled in for was `std::getenv`"
        );

        pinned(
            GDN,
            "ssm/gated_delta_net.cu",
            246,
            "if (!qwen_gdn_k_last_state_enabled() && V_d == 128 && K_d == 128) {",
            "`the deleted PIE_QWEN35_GDN_SMEM_STEP`",
        );
    }

    /// The pin's own two failure modes, fired.
    ///
    /// `pinned` is the entire guarantee of this module — every comparison
    /// below it is only as trustworthy as the claim that the text it was
    /// transcribed from is still there and still where the doc says. A check
    /// that cannot fail is not a check, and the stale `axial_rope` quote
    /// survived precisely because everything around it passed.
    ///
    /// So both arms are fired against a synthetic source, and the assertion
    /// is that each one PANICS. A `pinned` weakened to a no-op — by a
    /// `trim()` that swallows the comparison, an `is_empty()` guard that
    /// exits early, a refactor that stops splitting lines — fails here and
    /// nowhere else, because a weakened pin agrees with everything.
    #[test]
    fn the_pin_fails_when_the_citation_goes_stale() {
        let src = "alpha\nbeta\ngamma\n";

        // The text is gone: the launcher was rewritten.
        let rewritten = std::panic::catch_unwind(|| {
            pinned(src, "synthetic.cu", 2, "delta", "`nothing`");
        });
        assert!(rewritten.is_err(), "a rewritten launcher must fail the pin");

        // The text is intact and has moved: the launcher is unchanged and
        // every `at :NNN` in every doc that cites it is now wrong. This is
        // the arm no comparison of numbers can ever catch.
        let moved = std::panic::catch_unwind(|| {
            pinned(src, "synthetic.cu", 3, "beta", "`nothing`");
        });
        assert!(moved.is_err(), "a moved launcher must fail the pin");

        // And it passes when both hold, so the two above are measuring the
        // pin and not a helper that always panics.
        pinned(src, "synthetic.cu", 2, "beta", "`nothing`");
    }

    /// A citation this file cannot check is one nobody will.
    ///
    /// `pins()` is the whole evidence base and it is only as good as its
    /// coverage, so this asserts the count rather than letting a launcher
    /// quietly leave it. The number is small on purpose: these are the rules
    /// whose grids are not `[n, 1, 1]`, which is where a transposition or a
    /// spent axis hides.
    #[test]
    fn the_pins_cover_every_transcribed_launcher() {
        pins();

        for (src, name) in [
            (VISION, "gemma4_vision.cu"),
            (AUDIO, "gemma4_audio.cu"),
            (QWEN, "qwen3_vl_tower.cu"),
            (GDN, "gated_delta_net.cu"),
            (WNA16, "dequant_wna16.cu"),
            (FP4, "dequant_fp4.cu"),
            (NAIVE, "attention_naive_paged.cu"),
            (MLA, "mla_paged.cu"),
            (QKV, "qkv_fused.cu"),
            (ALTUP, "norm/altup.cu"),
            (ROPE, "rope/rope.cu"),
            (DSV4, "attn/dsv4_compress.cu"),
            (SLOT_OPS, "layout/slot_ops.cu"),
            (KV_PAGED, "attn/kv_paged.cu"),
            (NAIVE_UNPAGED, "attn/attention_naive.cu"),
            (PAGE_COMPACT, "attn/page_compact.cu"),
            (RMSNORM, "norm/rmsnorm.cu"),
            (NEMOTRON, "ssm/nemotron_h.cu"),
        ] {
            assert!(!src.is_empty(), "{name} is empty — `include_str!` found the wrong file");
        }

        // The one that went stale, spelled out. `k_rope_axial2d` takes ONE
        // tensor and a `pos` table; a quote naming `D(q), D(k)` in a single
        // launch describes a kernel that does not exist in this tree.
        assert!(
            VISION.contains("k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(q),pos,N,NH,THETA);"),
            "the corrected `axial_rope` citation no longer matches its source"
        );
        assert!(
            !VISION.contains("D(q),D(k),N,NH,gw,gh"),
            "the stale `axial_rope` quote has reappeared in the source"
        );
    }
}

//===----------------------------------------------------------------------===//
//
// The rules that reach a real kernel, fired.
//
//===----------------------------------------------------------------------===//
//
// Everything above is integers about integers. `mod transcribed` reads the
// launcher's own `G2`, `B2` and `rg` out of the `.cu` text and compares them
// to `eval`, which catches a rule that computes the wrong numbers. It cannot
// catch a rule that computes the RIGHT numbers for a kernel that indexes them
// differently, because no kernel is involved.
//
// This module is the other half, and it exists because of three measurements
// the file header cites and one this module makes for itself:
//
//   * §18: a wrong arm at 99.83% of the right answer — 7 of 4 095 values
//     moved, 0 of the 4 088 actually written.
//   * §21.14: a wrong arm that moved 34 273 of 55 200 cells WHILE WRITING THE
//     SAME NUMBER of non-zero values. A permutation, not a truncation.
//   * §21.13: five RMSNorm rows moving 35 266–61 757 of 65 536 bytes with the
//     same 32 768 values.
//
// No count, no norm and no tolerance flags any of those. So the bar here is
// byte-identical against a raw `cuLaunchKernel` at the launcher's own
// geometry, every comparison carries a `written > 0` guard, and the negative
// control is a PERMUTATION — a launch that writes the same number of values
// into the same buffer and puts different numbers in them.
//
// # The fourth measurement, which this module makes
//
// `altup_predict`'s first line is `if (t >= T_len || k >= K || h >= H)
// return;`. A grid LARGER than the rectangle is therefore invisible in the
// output: every excess block returns before it writes. `the_altup_near_miss`
// fires the refusal's own near-miss — `WarpTiledScan`'s `[rows, kv_heads,
// ceil(V_d/4)]` — and measures exactly that. It is byte-identical when the
// near-miss overshoots and it leaves whole streams unwritten when it
// undershoots, and BOTH readings are the point: the guard means an
// over-launch cannot be caught by looking at the output at all, so the
// argument for `Rule::AltUpStreams` over `Rule::WarpTiledScan` has to be made
// out of the source, which is what `mod transcribed`'s pins are for.
//
// # What is NOT fired here, and why
//
// `Rule::PagedScores` and `Rule::PagedScoresDecode` land no rows: their
// kernels take `device::KvScheme` and `device::KvDType` by value and
// `kernels::Ty` has no variant for an enum class. There is nothing to fire.
//
// `Rule::MlaPrepare`'s row does not claim the ahead-of-time symbol, because
// `attn::mla_prepare_bf16` takes `MlaCacheLayerView` by value — the same
// reason `attn::write_mla` does not claim its own.
//
// `Rule::RoutedQmv` and `Rule::RoutedQmvTransposed` reach real kernels and
// their rows are stated, but the kernels take `const std::int32_t* const*` —
// a device array of per-expert weight pointers — and `jit_dims` hard-codes
// `experts_per_token: 0`, so `eval` answers `Ungeometric::Empty` at every
// generated call site. Firing them would prove the arithmetic of a rule no
// production dispatch can reach; `mod transcribed` proves that arithmetic
// already, out of the launcher's own text, and the blocker is named in
// `families::quant`.
//
// `Rule::RowsPackedHeads`, `RowsPackedHeadsNarrow` and `WarpPackedHeads` are
// fired below through the one row that claims an ahead-of-time symbol.
//
//===----------------------------------------------------------------------===//

mod fires {
    use super::{Dims, Rule, dims, eval};
    use cudarc::driver::sys as dr;
    use kernels_cuda_new::runtime::{self, ArgValue, KernelModule, Stream, cache};
    use kernels_cuda_new::unit;
    use std::ffi::c_void;

    /// `norm/altup.cu:17` and `:31` both write `constexpr int BLOCK = 128;`.
    const ALTUP_BLOCK: u32 = 128;

    /// `sm_XY` for the current device, or a stated reason there is none.
    ///
    /// It also binds the thread, because this module makes its OWN driver-API
    /// calls: `cuMemAlloc_v2` is as much a driver-API call as
    /// `cuLaunchKernel`, and a test thread that has not forced the primary
    /// context cannot allocate the buffer it means to launch over.
    fn arch_or_skip(what: &str) -> Option<&'static str> {
        match cache::arch() {
            Some(arch) => match cache::bind_context() {
                Ok(()) => {
                    refuse_a_context_someone_else_poisoned(what);
                    Some(arch)
                }
                Err(why) => {
                    // A context that will not BIND is a skip only when
                    // nothing has corrupted it. `CUDA_ERROR_ILLEGAL_ADDRESS`
                    // is sticky and can surface here too, and skipping on it
                    // would turn a whole poisoned suite green — which is the
                    // §21.9 failure mode, arrived at from the other side.
                    let said = why.to_string();
                    assert!(
                        !said.contains("ILLEGAL_ADDRESS") && !said.contains("700"),
                        "{what}: the context cannot be bound because it is POISONED ({why}). \
                         An earlier fire in this process made an illegal access; this test \
                         never ran. Re-run with `--test-threads=1` to attribute it."
                    );
                    eprintln!("SKIP {what}: no usable context ({why})");
                    None
                }
            },
            None => {
                eprintln!("SKIP {what}: no CUDA device is current");
                None
            }
        }
    }

    /// Fail HERE if the primary context is already in a sticky error state.
    ///
    /// **A test that inherits a poisoned context must not report the
    /// inheritance as its own failure.** libtest runs tests in threads of one
    /// process, they share the primary context, and
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` is sticky: once any fire makes an illegal
    /// access, every later `cuMemAlloc`, `cuModuleLoadData` and
    /// `cuLaunchKernel` in the process returns it. The suite then prints a
    /// dozen failures of which exactly one is real, and — far worse — a
    /// GENUINE regression in a later fire is indistinguishable from the
    /// contagion. That is the same species as the gates §21.9 and §22.6
    /// caught green-while-blind: the signal decided by something other than
    /// the property under test.
    ///
    /// This turns the contagion into a sentence that says it is contagion and
    /// says what to do about it, so the one real failure is the only one that
    /// reads like a defect.
    fn refuse_a_context_someone_else_poisoned(what: &str) {
        // SAFETY: no arguments, and the context is bound.
        let code = unsafe { dr::cuCtxSynchronize() };
        assert_eq!(
            code,
            dr::CUresult::CUDA_SUCCESS,
            "{what} has not fired yet and the context is ALREADY {code:?}. An earlier fire in \
             this process made an illegal access and the error is sticky, so this test could \
             not have run and its result means nothing. Re-run with `--test-threads=1`: the \
             first test to fail there is the real one."
        );
    }

    /// `cuCtxSynchronize`, with the fire that is on the hook named.
    ///
    /// Every caller passes something that identifies the LAUNCH — the test,
    /// the symbol and the shape — because the alternative is what this file
    /// shipped with first: `"a control"`, on a fault whose blast radius was
    /// eleven other tests.
    fn synchronise(what: &str) {
        // SAFETY: no arguments, and the context is bound.
        let code = unsafe { dr::cuCtxSynchronize() };
        assert_eq!(
            code,
            dr::CUresult::CUDA_SUCCESS,
            "{what}: this launch left the context {code:?}. If that is \
             CUDA_ERROR_ILLEGAL_ADDRESS it is STICKY — every later test in this process will \
             now fail on `cuMemAlloc` or `cuModuleLoadData` for a reason that is not theirs."
        );
    }

    /// A device allocation, freed on drop.
    struct Buffer {
        ptr: u64,
        bytes: usize,
    }

    impl Buffer {
        fn of<T: Copy>(from: &[T]) -> Self {
            let bytes = std::mem::size_of_val(from).max(1);
            let mut ptr = 0u64;
            // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
            let code = unsafe { dr::cuMemAlloc_v2(&raw mut ptr, bytes) };
            if code != dr::CUresult::CUDA_SUCCESS {
                // Same reason as `module_of`: an allocation does not fail
                // because it is 4 096 bytes, it fails because someone else
                // already broke the context.
                refuse_a_context_someone_else_poisoned(&format!("allocating {bytes} bytes"));
            }
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "allocating {bytes} bytes");
            let me = Self { ptr, bytes };
            if !from.is_empty() {
                // SAFETY: the allocation is exactly `from`'s size.
                let code = unsafe { dr::cuMemcpyHtoD_v2(ptr, from.as_ptr().cast(), me.bytes) };
                if code != dr::CUresult::CUDA_SUCCESS {
                    refuse_a_context_someone_else_poisoned("upload");
                }
                assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "upload");
            }
            me
        }

        fn zeroed(bytes: usize) -> Self {
            let me = Self::of(&vec![0u8; bytes]);
            me.clear();
            me
        }

        fn clear(&self) {
            // SAFETY: the allocation is `bytes` long.
            let code = unsafe { dr::cuMemsetD8_v2(self.ptr, 0, self.bytes) };
            if code != dr::CUresult::CUDA_SUCCESS {
                refuse_a_context_someone_else_poisoned("memset");
            }
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "memset");
        }

        fn bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bytes];
            // SAFETY: same allocation, same length.
            let code =
                unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.bytes) };
            if code != dr::CUresult::CUDA_SUCCESS {
                refuse_a_context_someone_else_poisoned("download");
            }
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "download");
            out
        }

        fn arg(&self) -> ArgValue {
            ArgValue::Ptr(self.ptr as *mut c_void)
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            // SAFETY: the handle came from `cuMemAlloc_v2` and is freed once.
            unsafe { dr::cuMemFree_v2(self.ptr) };
        }
    }

    /// bf16 bit patterns that are not all the same and not symmetric.
    ///
    /// The generator is a 64-bit xorshift so the sequence is reproducible,
    /// the mantissa is never zero so no element of the rectangle can come out
    /// an accidental zero, and the exponents span sixteen binades so that two
    /// different sums cannot coincide.
    fn bf16_fill(n: usize, seed: u64) -> Vec<u16> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let exponent = 120 + u16::try_from((state >> 32) % 16).expect("small");
                let mantissa = u16::try_from((state >> 8) & 0x7F).expect("seven bits") | 1;
                let sign = u16::try_from((state >> 3) & 1).expect("one bit") << 15;
                sign | (exponent << 7) | mantissa
            })
            .collect()
    }

    fn f32_fill(n: usize, seed: u64) -> Vec<f32> {
        bf16_fill(n, seed).iter().map(|&h| f32::from_bits(u32::from(h) << 16)).collect()
    }

    fn differing(a: &[u8], b: &[u8]) -> usize {
        assert_eq!(a.len(), b.len(), "two buffers of different sizes are not comparable");
        a.iter().zip(b).filter(|(l, r)| l != r).count()
    }

    /// Non-zero bf16 VALUES, not non-zero bytes: a bf16 whose low byte is
    /// zero is still a written value, and counting bytes would flatter every
    /// comparison below.
    fn written(a: &[u8]) -> usize {
        a.chunks_exact(2).filter(|v| v != &[0, 0]).count()
    }

    /// The compiled unit that hosts `symbol`, with every row resolved.
    ///
    /// The `unwrap_or_else` reports a compile failure, and for most of this
    /// file's life that is what a failure here meant. It is NOT what it meant
    /// on the run that produced this comment: a sibling test made an illegal
    /// access, the error stuck to the process's primary context, and
    /// `cuModuleLoadData` returned 700 for eleven tests that had done nothing
    /// wrong. `rope/rope compiles: cuModuleLoadData failed with 700` is a
    /// true sentence and a completely misleading one.
    ///
    /// So the failure path asks the context whose fault it was first.
    fn module_of(symbol: &str, unit_name: &str) -> &'static KernelModule {
        let (index, unit) = unit::unit_of(symbol).expect("the row is hosted");
        assert_eq!(unit.name, unit_name);
        cache::module(index, unit).unwrap_or_else(|why| {
            refuse_a_context_someone_else_poisoned(&format!("loading {unit_name} for {symbol}"));
            panic!("{unit_name} compiles: {why}")
        })
    }

    /// A raw `cuLaunchKernel` at an explicit grid, with a hand-built cell
    /// array in the KERNEL's declared order.
    ///
    /// `<<<>>>` is this call with sugar, so a launch made this way and a
    /// launch made through `runtime::fire` differ in exactly what the row
    /// adds over the launcher — the rule, and the binding.
    ///
    /// # Safety
    ///
    /// `entry` must come from a module that outlives the call, and `cells`
    /// must be the kernel's declared parameters in order and at their
    /// declared widths.
    unsafe fn raw_launch(
        entry: dr::CUfunction,
        grid: [u32; 3],
        block: u32,
        cells: &mut [*mut c_void],
        what: &str,
    ) {
        // SAFETY: the caller's contract, forwarded, at no dynamic shared
        // memory.
        unsafe { raw_launch_smem(entry, grid, block, 0, cells, what) }
    }

    /// [`raw_launch`] with DYNAMIC shared memory, which two of the rules
    /// below need and which is not a detail: `attention_naive_paged.cu:197`
    /// sizes it `(head_dim + BLOCK) * sizeof(float)` and the kernel cuts that
    /// allocation into `[head_dim]` staged query values and `[BLOCK]`
    /// reduction slots. A launch one float short reduces through a slot
    /// nothing wrote, which is a number and not a fault.
    ///
    /// # Safety
    ///
    /// As [`raw_launch`], plus: `smem` must be at least what the kernel's own
    /// `extern __shared__` partitioning requires.
    unsafe fn raw_launch_smem(
        entry: dr::CUfunction,
        grid: [u32; 3],
        block: u32,
        smem: u32,
        cells: &mut [*mut c_void],
        what: &str,
    ) {
        // SAFETY: the caller's contract, plus the null stream, which is live.
        let code = unsafe {
            dr::cuLaunchKernel(
                entry,
                grid[0],
                grid[1],
                grid[2],
                block,
                1,
                1,
                smem,
                std::ptr::null_mut(),
                cells.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "{what}");
        synchronise(what);
    }

    // -----------------------------------------------------------------------
    // `Rule::AltUpStreams`
    // -----------------------------------------------------------------------

    /// A Gemma-3n AltUp rectangle: `K` streams of `T` tokens by `H` channels.
    ///
    /// `Dims::width` is `K * H` — the whole residual row as a fire sees it —
    /// and `Dims::altup_streams` is `K`, which is the field this rule was
    /// added for. The per-stream width the grid tiles is the QUOTIENT, and
    /// `Rule::AltUpStreams` refuses a width the stream count does not divide
    /// rather than flooring it.
    #[derive(Clone, Copy, Debug)]
    struct AltUp {
        k: u32,
        t: u32,
        h: u32,
    }

    impl AltUp {
        /// The rectangle as `driver-cuda`'s `jit_dims` builds it.
        ///
        /// `kv_heads` is deliberately NOT `k`: the whole second half of this
        /// rule's refusal is that `WarpTiledScan` fills `grid.y` from
        /// `Dims::kv_heads`, which is an attention head count, and a fixture
        /// where the two coincide cannot see the difference.
        fn dims(self, kv_heads: u32) -> Dims {
            Dims {
                rows: self.t,
                width: self.k * self.h,
                in_width: self.k * self.h,
                q_heads: 32,
                kv_heads,
                head_dim: 128,
                stated_head_dim: 0,
                rotary_dims: 64,
                n_experts: 0,
                experts_per_token: 0,
                requests: 0,
                altup_streams: self.k,
            }
        }

        /// `norm/altup.cu:18`, by hand:
        /// `const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);`
        fn grid(self) -> [u32; 3] {
            [self.t, self.k, self.h.div_ceil(ALTUP_BLOCK)]
        }

        /// Elements in the rectangle. Both kernels write exactly this many.
        fn elements(self) -> usize {
            self.k as usize * self.t as usize * self.h as usize
        }
    }

    /// Everything one AltUp row is fired with, allocated and uploaded.
    struct AltUpOps {
        _inputs: Vec<Buffer>,
        out: Buffer,
        values: Vec<ArgValue>,
        raw: Vec<u64>,
        scalars: Vec<i32>,
    }

    /// Build the operand set for `symbol` at `shape`.
    ///
    /// The scalars are the KERNEL's guard arguments — `(K, T_len, H)` for
    /// predict and `(K, T_len, H, active_idx)` for correct — and they are the
    /// SHAPE's, not the grid's. A rule that opened the wrong grid over the
    /// right guard writes a sub-rectangle; a rule that opened the right grid
    /// over the wrong guard walks off the end. Both are visible in `out`.
    fn altup_operands(symbol: &str, shape: AltUp) -> AltUpOps {
        let n = shape.elements();
        let mut inputs = Vec::new();
        let mut values = Vec::new();
        let mut raw = Vec::new();
        let mut push = |buffer: Buffer, values: &mut Vec<ArgValue>, raw: &mut Vec<u64>| {
            values.push(buffer.arg());
            raw.push(buffer.ptr);
            inputs.push(buffer);
        };

        let scalars: Vec<i32> = match symbol {
            "norm::altup_predict_bf16" => {
                // (streams, coefs, predictions, K, T, H)
                push(Buffer::of(&bf16_fill(n, 0xA17F_0001)), &mut values, &mut raw);
                let coefs = (shape.t * shape.k * shape.k) as usize;
                push(Buffer::of(&f32_fill(coefs, 0xA17F_0002)), &mut values, &mut raw);
                vec![shape.k as i32, shape.t as i32, shape.h as i32]
            }
            "norm::altup_correct_bf16" => {
                // (predictions, activated, correction_coefs_plus_one,
                //  corrected, K, T, H, active_idx)
                push(Buffer::of(&bf16_fill(n, 0xA17F_0003)), &mut values, &mut raw);
                push(
                    Buffer::of(&bf16_fill(shape.t as usize * shape.h as usize, 0xA17F_0004)),
                    &mut values,
                    &mut raw,
                );
                let coefs = (shape.t * shape.k) as usize;
                push(Buffer::of(&f32_fill(coefs, 0xA17F_0005)), &mut values, &mut raw);
                vec![shape.k as i32, shape.t as i32, shape.h as i32, 0]
            }
            other => panic!("{other} is not an AltUp row"),
        };

        let out = Buffer::zeroed(n * 2);
        values.push(out.arg());
        raw.push(out.ptr);
        for s in &scalars {
            values.push(ArgValue::I32(*s));
        }
        AltUpOps { _inputs: inputs, out, values, raw, scalars }
    }

    /// The two rows exist, are hosted, and state the rule this section proves.
    #[test]
    fn the_altup_rows_are_stated_and_hosted() {
        for symbol in ["norm::altup_predict_bf16", "norm::altup_correct_bf16"] {
            assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
            let row = runtime::row(symbol).expect("hosted");
            assert_eq!(row.sig.launch, Rule::AltUpStreams, "{symbol} states another rule");
            assert_eq!(row.sig.file, Some("norm/altup.cuh"));
        }
    }

    /// **The two AltUp rows are byte-identical to the launcher.**
    ///
    /// Four shapes each, and each of the four is chosen against a specific way
    /// of getting the rule wrong; the comments say which.
    #[test]
    fn the_altup_rows_reproduce_the_launcher() {
        let Some(_) = arch_or_skip("the_altup_rows_reproduce_the_launcher") else { return };
        let module = module_of("norm::altup_predict_bf16", "norm/altup");

        let shapes = [
            // Gemma-3n E2B's own: 4 streams, 2 048 channels, a prefill row
            // count. `H / 128` is exact, so the ragged tile is not the reason
            // anything here passes.
            AltUp { k: 4, t: 7, h: 2048 },
            // A decode step. One token is the extent at which a rule that
            // dropped `grid.x` altogether still answers 1.
            AltUp { k: 4, t: 1, h: 2048 },
            // A RAGGED third axis: 2 000 is not a multiple of 128, so the
            // last tile is partial and the `h >= H` guard is load-bearing.
            AltUp { k: 4, t: 5, h: 2000 },
            // A stream count that is not the usual 4 and a width the fire's
            // `head_dim` (128) also divides — so a rule that read `head_dim`
            // for the tile and `kv_heads` for the streams would produce a
            // plausible grid, and only the bytes would say.
            AltUp { k: 3, t: 4, h: 1024 },
        ];

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for symbol in ["norm::altup_predict_bf16", "norm::altup_correct_bf16"] {
            let entry = module.entry(symbol).expect("the row resolved");
            for shape in shapes {
                let mut ops = altup_operands(symbol, shape);

                // The rule's grid IS the launcher's, before anything fires.
                let launch = eval(Rule::AltUpStreams, shape.dims(8)).expect("ported");
                assert_eq!(launch.grid, shape.grid(), "{symbol} at {shape:?}");
                assert_eq!(launch.block, [ALTUP_BLOCK, 1, 1]);
                assert_eq!(launch.smem, 0, "the launcher asks for no dynamic shared memory");

                // SAFETY: every pointer addresses a live allocation of the
                // extent the row states, the values match the row's operand
                // list, and the null stream is live.
                unsafe { runtime::fire(symbol, shape.dims(8), &ops.values, Stream::NULL) }
                    .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
                synchronise(&format!("the shipped fire of {symbol} at {shape:?}"));
                let through_the_row = ops.out.bytes();

                ops.out.clear();
                synchronise("clearing between the two launches");
                let mut pointers = ops.raw.clone();
                let mut cells: Vec<*mut c_void> =
                    pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
                for s in ops.scalars.iter_mut() {
                    cells.push((&raw mut *s).cast());
                }
                // SAFETY: `entry` is from a module that outlives the call and
                // the cells are the kernel's declared parameters in order.
                unsafe {
                    raw_launch(entry, shape.grid(), ALTUP_BLOCK, &mut cells, "the launcher's own");
                }
                let through_the_launcher = ops.out.bytes();

                // §18's guard. Two buffers nothing wrote are equal, and a
                // comparison that cannot fail is not evidence.
                let live = written(&through_the_launcher);
                assert_eq!(
                    live,
                    shape.elements(),
                    "{symbol} at {shape:?}: the launcher wrote {live} of {} values, so the \
                     rest of this comparison is zeros against zeros",
                    shape.elements()
                );

                let differs = differing(&through_the_row, &through_the_launcher);
                assert_eq!(
                    differs, 0,
                    "{symbol} at {shape:?}: the row and the launcher disagree on {differs} of \
                     {} bytes (rule grid {:?}, launcher grid {:?})",
                    through_the_row.len(),
                    launch.grid,
                    shape.grid()
                );
                compared += through_the_row.len();
                live_total += live;
            }
        }
        eprintln!(
            "AltUpStreams: 2 rows x {} shapes, {compared} bytes compared, {live_total} values \
             written, 0 differing",
            shapes.len()
        );
    }

    /// **The negative control: a permutation, not a truncation.**
    ///
    /// §21.14's shape exactly. `(K, T) = (8, 4)` and `(4, 8)` factor the same
    /// 8 192-element buffer, open the same 64 blocks, and write every one of
    /// the same 8 192 values non-zero — and the numbers in them are
    /// different, because `stream_stride = T_len * H` and the coefficient
    /// index `t * K * K + j * K + k` both move. A test that counted outputs,
    /// summed them, or compared them to a tolerance would pass on the wrong
    /// one.
    ///
    /// The `assert_eq!` on the two counts is not decoration: it is the claim.
    /// If the two launches ever stop writing the same number of values, this
    /// control has become a truncation and proves less than it says.
    #[test]
    fn the_altup_control_is_a_permutation_and_bytes_catch_it() {
        let Some(_) = arch_or_skip("the_altup_control_is_a_permutation_and_bytes_catch_it") else {
            return;
        };
        let module = module_of("norm::altup_predict_bf16", "norm/altup");
        let entry = module.entry("norm::altup_predict_bf16").expect("resolved");

        let right = AltUp { k: 8, t: 4, h: 256 };
        let wrong = AltUp { k: 4, t: 8, h: 256 };
        assert_eq!(right.elements(), wrong.elements(), "one buffer, two factorisations");
        let blocks = |g: [u32; 3]| g[0] as usize * g[1] as usize * g[2] as usize;
        assert_eq!(
            blocks(right.grid()),
            blocks(wrong.grid()),
            "the control must open the same number of blocks, or it is a truncation"
        );

        // One operand set, fired twice: same `streams`, same `coefs`, same
        // `predictions`. Only the three guard scalars and the grid move, and
        // the coefficient buffer is sized for the larger of the two `T * K *
        // K` so neither launch reads past it.
        let n = right.elements();
        let coefs = (right.t * right.k * right.k).max(wrong.t * wrong.k * wrong.k) as usize;
        let streams = Buffer::of(&bf16_fill(n, 0xA17F_0001));
        let coef_buf = Buffer::of(&f32_fill(coefs, 0xA17F_0002));
        let out = Buffer::zeroed(n * 2);

        let fire = |shape: AltUp| {
            out.clear();
            synchronise("clearing");
            let mut pointers = [streams.ptr, coef_buf.ptr, out.ptr];
            let mut scalars = [shape.k as i32, shape.t as i32, shape.h as i32];
            let mut cells: Vec<*mut c_void> =
                pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for s in scalars.iter_mut() {
                cells.push((&raw mut *s).cast());
            }
            // SAFETY: the module outlives the call and the cells are the
            // kernel's six declared parameters in order.
            unsafe { raw_launch(entry, shape.grid(), ALTUP_BLOCK, &mut cells, "a control") };
            out.bytes()
        };

        let a = fire(right);
        let b = fire(wrong);

        let live_a = written(&a);
        let live_b = written(&b);
        assert!(live_a > 0, "the right factorisation wrote nothing");
        assert_eq!(
            live_a, live_b,
            "the control moved {live_a} values to {live_b} — it is a truncation, not a \
             permutation, and proves less than this test claims"
        );
        assert_eq!(live_a, n, "every element of the rectangle is written on both");

        let differs = differing(&a, &b);
        assert!(
            differs > 0,
            "the permutation is invisible in the bytes, so this control cannot fail"
        );
        eprintln!(
            "AltUp permutation control: (K,T) = (8,4) vs (4,8), {} blocks either way, \
             {live_a} of {n} values written on both, {differs} of {} bytes differ",
            blocks(right.grid()),
            a.len()
        );
    }

    /// **The refusal's own near-miss, fired — and the measurement that says
    /// why the argument for this rule cannot be made out of bytes alone.**
    ///
    /// `WarpTiledScan` produces `[rows, kv_heads, ceil(V_d / 4)]` where
    /// `V_d = width / kv_heads`, at this exact 128-wide block. Fired on the
    /// AltUp kernel it is wrong on both of the axes the refusal names, and
    /// what the output shows depends entirely on WHICH WAY it is wrong:
    ///
    ///   * `kv_heads > K`: `grid.y` overshoots, `grid.z` overshoots, and
    ///     every excess block hits `if (t >= T_len || k >= K || h >= H)
    ///     return;` — **byte-identical**. The wrong grid is invisible.
    ///   * `kv_heads < K`: `grid.y` undershoots and the streams above it are
    ///     never written — zeros, and detectable, but only because the buffer
    ///     was cleared first.
    ///
    /// This is the file header's second hazard in one kernel: a wrong grid
    /// can be byte-identical inside the rectangle. It is why `mod
    /// transcribed` pins `norm/altup.cu:18` rather than relying on a fire,
    /// and why `Dims::altup_streams` had to be a distinct field instead of a
    /// reading of `kv_heads`.
    #[test]
    fn the_altup_near_miss_is_invisible_one_way_and_partial_the_other() {
        let Some(_) = arch_or_skip("the_altup_near_miss_is_invisible_one_way_and_partial_the_other")
        else {
            return;
        };
        let module = module_of("norm::altup_predict_bf16", "norm/altup");
        let entry = module.entry("norm::altup_predict_bf16").expect("resolved");

        let shape = AltUp { k: 4, t: 5, h: 512 };
        let n = shape.elements();
        let streams = Buffer::of(&bf16_fill(n, 0xA17F_0001));
        let coef_buf =
            Buffer::of(&f32_fill((shape.t * shape.k * shape.k) as usize, 0xA17F_0002));
        let out = Buffer::zeroed(n * 2);

        let fire = |grid: [u32; 3]| {
            out.clear();
            synchronise("clearing");
            let mut pointers = [streams.ptr, coef_buf.ptr, out.ptr];
            let mut scalars = [shape.k as i32, shape.t as i32, shape.h as i32];
            let mut cells: Vec<*mut c_void> =
                pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for s in scalars.iter_mut() {
                cells.push((&raw mut *s).cast());
            }
            // SAFETY: as above; only the grid moves.
            unsafe { raw_launch(entry, grid, ALTUP_BLOCK, &mut cells, "a near-miss") };
            out.bytes()
        };

        let right = fire(shape.grid());
        assert_eq!(written(&right), n, "the launcher's grid writes the whole rectangle");

        let mut report = Vec::new();
        for kv_heads in [8u32, 2] {
            let near = eval(Rule::WarpTiledScan, shape.dims(kv_heads)).expect("ported");
            assert_ne!(near.grid, shape.grid(), "the near-miss must actually differ");
            assert_eq!(near.block, [ALTUP_BLOCK, 1, 1], "same block, which is the trap");

            let got = fire(near.grid);
            let live = written(&got);
            let differs = differing(&got, &right);
            report.push(format!(
                "  kv_heads={kv_heads}: WarpTiledScan {:?} vs AltUpStreams {:?} -> \
                 {live} of {n} values, {differs} of {} bytes differ",
                near.grid,
                shape.grid(),
                right.len()
            ));
            if kv_heads > shape.k {
                assert_eq!(
                    differs, 0,
                    "an over-large grid was expected to be invisible — if it is not, the \
                     guard in `altup.cuh` has changed and this test's argument with it"
                );
                assert_eq!(live, n);
            } else {
                assert!(differs > 0, "an under-large grid must leave part of the buffer unwritten");
                assert!(live < n, "and that part must be measurable as unwritten values");
                assert_eq!(
                    live,
                    n * kv_heads as usize / shape.k as usize,
                    "exactly `kv_heads` of the {} streams are written",
                    shape.k
                );
            }
        }
        eprintln!("AltUp near-miss (`WarpTiledScan`), fired:\n{}", report.join("\n"));
    }
    // -----------------------------------------------------------------------
    // `Rule::RowsPackedHeads`
    // -----------------------------------------------------------------------

    /// `attn/qkv_fused.cu:245` writes `constexpr int BLOCK = 256;`.
    const PACKED_BLOCK: u32 = 256;

    /// A prefill rectangle for `qkv_packed_qk_norm_rope_vnorm_write_kv`.
    ///
    /// `q_heads` and `kv_heads` are always DIFFERENT here: the refusal this
    /// rule closes is that `Rule::GatedRms`' `[rows, kv_heads, 1]` is
    /// *"grid.y short by every Q head"*, and a fixture where the two counts
    /// coincide cannot see that.
    #[derive(Clone, Copy, Debug)]
    struct Packed {
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        page_size: u32,
        hnd: bool,
    }

    impl Packed {
        fn dims(self) -> Dims {
            Dims {
                rows: self.rows,
                width: self.q_heads * self.head_dim,
                in_width: (self.q_heads + 2 * self.kv_heads) * self.head_dim,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                stated_head_dim: 0,
                rotary_dims: self.head_dim,
                n_experts: 0,
                experts_per_token: 0,
                requests: 0,
                altup_streams: 0,
            }
        }

        /// `attn/qkv_fused.cu:246`, by hand:
        /// `dim3 grid(num_rows, num_q_heads + num_kv_heads);`
        fn grid(self) -> [u32; 3] {
            [self.rows, self.q_heads + self.kv_heads, 1]
        }

        /// `packed` is `[rows, q_dim + 2 * kv_dim]`.
        fn packed_elems(self) -> usize {
            self.rows as usize
                * ((self.q_heads + 2 * self.kv_heads) * self.head_dim) as usize
        }

        /// `q_out` is `[rows, q_heads * head_dim]`.
        fn q_elems(self) -> usize {
            (self.rows * self.q_heads * self.head_dim) as usize
        }

        /// One page a row, so every row writes a distinct page and a launch
        /// that skipped a row shows up as a page of zeros rather than as a
        /// value another row would have written anyway.
        fn page_elems(self) -> usize {
            (self.rows * self.page_size * self.kv_heads * self.head_dim) as usize
        }
    }

    /// Everything the packed row is fired with.
    struct PackedOps {
        _inputs: Vec<Buffer>,
        q_out: Buffer,
        k_pages: Buffer,
        v_pages: Buffer,
        values: Vec<ArgValue>,
        raw: Vec<u64>,
    }

    /// `theta` and `eps` as a Gemma/Llama config spells them.
    const THETA: f32 = 10_000.0;
    const PACKED_EPS: f32 = 1e-6;

    fn packed_operands(shape: Packed, row_valid: bool) -> PackedOps {
        let mut inputs = Vec::new();
        let mut values = Vec::new();
        let mut raw = Vec::new();

        let packed = Buffer::of(&bf16_fill(shape.packed_elems(), 0x9CE7_0001));
        let q_out = Buffer::zeroed(shape.q_elems() * 2);
        let k_pages = Buffer::zeroed(shape.page_elems() * 2);
        let v_pages = Buffer::zeroed(shape.page_elems() * 2);
        let q_weight = Buffer::of(&bf16_fill(shape.head_dim as usize, 0x9CE7_0002));
        let k_weight = Buffer::of(&bf16_fill(shape.head_dim as usize, 0x9CE7_0003));
        // Positions spread over more than one rotary period, so a row that
        // took another row's angle is visible rather than merely rounded.
        let positions: Vec<i32> = (0..shape.rows).map(|r| (r * 37) as i32).collect();
        let positions = Buffer::of(&positions);
        // One page a row: `kv_page_indices[row] = row`, `indptr[row] = row`,
        // and one live slot in the last page. The kernel's arithmetic is
        // `abs_kv_pos = (num_pages_r - 1) * page_size + last_page_len - 1`,
        // which is 0 here, so each row writes slot 0 of its own page.
        let indices: Vec<u32> = (0..shape.rows).collect();
        let indices = Buffer::of(&indices);
        let indptr: Vec<u32> = (0..=shape.rows).collect();
        let indptr = Buffer::of(&indptr);
        let lens: Vec<u32> = vec![1; shape.rows as usize];
        let lens = Buffer::of(&lens);
        let valid: Vec<u8> = vec![1; shape.rows as usize];
        let valid = Buffer::of(&valid);

        for buffer in [&packed, &q_out, &k_pages, &v_pages, &q_weight, &k_weight, &positions,
                       &indices, &indptr, &lens] {
            values.push(buffer.arg());
            raw.push(buffer.ptr);
        }
        if row_valid {
            values.push(valid.arg());
            raw.push(valid.ptr);
        } else {
            // The operand is `U8s | null` and absence means *"every row is
            // valid"*. Firing it both ways is what makes the nullability a
            // measured property rather than a declared one.
            values.push(ArgValue::Ptr(std::ptr::null_mut()));
            raw.push(0);
        }
        values.push(ArgValue::I32(shape.q_heads as i32));
        values.push(ArgValue::I32(shape.kv_heads as i32));
        values.push(ArgValue::I32(shape.head_dim as i32));
        values.push(ArgValue::I32(shape.page_size as i32));
        values.push(ArgValue::Bool(shape.hnd));
        values.push(ArgValue::F32(THETA));
        values.push(ArgValue::F32(PACKED_EPS));

        inputs.push(packed);
        inputs.push(q_weight);
        inputs.push(k_weight);
        inputs.push(positions);
        inputs.push(indices);
        inputs.push(indptr);
        inputs.push(lens);
        inputs.push(valid);
        PackedOps { _inputs: inputs, q_out, k_pages, v_pages, values, raw }
    }

    /// The launcher's own cells, in the kernel's declared order.
    ///
    /// Returned as owned storage plus a cell array over it, because
    /// `cuLaunchKernel` reads through the pointers after this function
    /// returns and a temporary would be a dangling read.
    struct PackedCells {
        pointers: Vec<u64>,
        ints: Vec<i32>,
        flag: u8,
        floats: Vec<f32>,
    }

    impl PackedCells {
        fn new(shape: Packed, ops: &PackedOps) -> Self {
            Self {
                pointers: ops.raw.clone(),
                ints: vec![
                    shape.q_heads as i32,
                    shape.kv_heads as i32,
                    shape.head_dim as i32,
                    shape.page_size as i32,
                ],
                flag: u8::from(shape.hnd),
                floats: vec![THETA, PACKED_EPS],
            }
        }

        fn cells(&mut self) -> Vec<*mut c_void> {
            let mut cells: Vec<*mut c_void> =
                self.pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for i in self.ints.iter_mut() {
                cells.push((&raw mut *i).cast());
            }
            cells.push((&raw mut self.flag).cast());
            for f in self.floats.iter_mut() {
                cells.push((&raw mut *f).cast());
            }
            cells
        }
    }

    /// The packed row is stated, hosted, and claims the launcher's symbol.
    #[test]
    fn the_packed_row_is_stated_and_hosted() {
        let symbol = "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16";
        assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
        let row = runtime::row(symbol).expect("hosted");
        assert_eq!(row.sig.launch, Rule::RowsPackedHeads);
        assert_eq!(row.sig.file, Some("attn/qkv_fused.cuh"));
    }

    /// **The packed row is byte-identical to `qkv_fused.cu:245-248`.**
    ///
    /// Three output buffers a launch, all three compared: `q_out` catches a
    /// grid short in the Q heads, and `k_pages`/`v_pages` catch one short in
    /// the KV heads — which is the exact shape of the near-miss this rule
    /// closes, because `Rule::GatedRms`' `grid.y` is `kv_heads` and every
    /// block it opens lands in the Q range.
    #[test]
    fn the_packed_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_packed_row_reproduces_the_launcher") else { return };
        let symbol = "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16";
        let module = module_of(symbol, "attn/qkv_fused");
        let entry = module.entry(symbol).expect("the row resolved");

        let shapes = [
            // A Llama-shaped prefill: 32 Q over 8 KV, 128-wide heads.
            Packed { rows: 6, q_heads: 32, kv_heads: 8, head_dim: 128, page_size: 16, hnd: false },
            // The same in the head-major page layout, which moves every KV
            // write and nothing else.
            Packed { rows: 6, q_heads: 32, kv_heads: 8, head_dim: 128, page_size: 16, hnd: true },
            // One row: the extent at which a rule that dropped `grid.x`
            // altogether still answers 1.
            Packed { rows: 1, q_heads: 8, kv_heads: 2, head_dim: 64, page_size: 8, hnd: false },
            // Multi-query: one KV head, so `q_heads + kv_heads` and
            // `q_heads + 1` coincide and only the pages can tell a rule that
            // read the sum wrong.
            Packed { rows: 4, q_heads: 16, kv_heads: 1, head_dim: 256, page_size: 4, hnd: true },
        ];

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (case, shape) in shapes.into_iter().enumerate() {
            for row_valid in [true, false] {
                let ops = packed_operands(shape, row_valid);

                let launch = eval(Rule::RowsPackedHeads, shape.dims()).expect("ported");
                assert_eq!(launch.grid, shape.grid(), "case {case} at {shape:?}");
                assert_eq!(launch.block, [PACKED_BLOCK, 1, 1]);
                assert_eq!(launch.smem, 0, "the two `__shared__` arrays are STATIC");

                // SAFETY: every pointer addresses a live allocation of the
                // extent the row states, the values match the row's operand
                // list, and the null stream is live.
                unsafe { runtime::fire(symbol, shape.dims(), &ops.values, Stream::NULL) }
                    .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
                synchronise(&format!("the shipped fire of {symbol} at {shape:?}"));
                let row_q = ops.q_out.bytes();
                let row_k = ops.k_pages.bytes();
                let row_v = ops.v_pages.bytes();

                for buffer in [&ops.q_out, &ops.k_pages, &ops.v_pages] {
                    buffer.clear();
                }
                synchronise("clearing between the two launches");
                let mut cells = PackedCells::new(shape, &ops);
                let mut cells = cells.cells();
                // SAFETY: `entry` is from a module that outlives the call and
                // the cells are the kernel's nineteen declared parameters in
                // order.
                unsafe {
                    raw_launch(
                        entry,
                        shape.grid(),
                        PACKED_BLOCK,
                        &mut cells,
                        "the launcher's own",
                    );
                }
                let cu_q = ops.q_out.bytes();
                let cu_k = ops.k_pages.bytes();
                let cu_v = ops.v_pages.bytes();

                // §18's guard, on each of the three buffers separately: a
                // whole-buffer comparison over a buffer nothing wrote is two
                // fields of zeros and holds for a kernel that did nothing.
                let live = written(&cu_q) + written(&cu_k) + written(&cu_v);
                assert_eq!(
                    written(&cu_q),
                    shape.q_elems(),
                    "case {case}: the launcher wrote {} of {} query values",
                    written(&cu_q),
                    shape.q_elems()
                );
                assert!(
                    written(&cu_k) > 0 && written(&cu_v) > 0,
                    "case {case}: the launcher wrote no pages, so the page comparison \
                     below is zeros against zeros"
                );

                let differs = differing(&row_q, &cu_q)
                    + differing(&row_k, &cu_k)
                    + differing(&row_v, &cu_v);
                assert_eq!(
                    differs, 0,
                    "case {case} at {shape:?}, row_valid={row_valid}: the row and the \
                     launcher disagree on {differs} bytes (q {}, k {}, v {})",
                    differing(&row_q, &cu_q),
                    differing(&row_k, &cu_k),
                    differing(&row_v, &cu_v)
                );
                compared += row_q.len() + row_k.len() + row_v.len();
                live_total += live;
            }
        }
        eprintln!(
            "RowsPackedHeads: {} shapes x 2 row_valid arms, {compared} bytes compared, \
             {live_total} values written, 0 differing",
            shapes.len()
        );
    }

    /// **`Rule::GatedRms`, the refusal's own near-miss, fired — and a
    /// permutation control the bytes are the only thing that catches.**
    ///
    /// The refusal read: *"Nearest is `GatedRms` (`[rows, kv_heads, 1]`, 256,
    /// smem 0) — grid.y short by every Q head."* Fired, that is a truncation
    /// and it is measured here: `head_idx < num_q_heads` is true for every
    /// block `GatedRms` opens, so the KV pages are never written at all and
    /// `q_out` gets `kv_heads` of its `q_heads` heads.
    ///
    /// A truncation is the easy case. The hard one is §21.14's, and this
    /// kernel has it: at the launcher's OWN grid, flipping `hnd_layout` moves
    /// every KV write to a different address in the same buffer while writing
    /// **exactly the same number of values**. No count, no norm and no
    /// tolerance flags it; the byte comparison does. Both are measured
    /// because a control that is only ever a truncation proves less than this
    /// file claims.
    #[test]
    fn the_packed_near_miss_truncates_and_the_layout_flip_permutes() {
        let Some(_) =
            arch_or_skip("the_packed_near_miss_truncates_and_the_layout_flip_permutes")
        else {
            return;
        };
        let symbol = "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16";
        let module = module_of(symbol, "attn/qkv_fused");
        let entry = module.entry(symbol).expect("the row resolved");

        let shape = Packed {
            rows: 6,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            page_size: 16,
            hnd: false,
        };
        let ops = packed_operands(shape, true);
        let fire = |grid: [u32; 3], hnd: bool| {
            for buffer in [&ops.q_out, &ops.k_pages, &ops.v_pages] {
                buffer.clear();
            }
            synchronise("clearing");
            let mut cells = PackedCells::new(Packed { hnd, ..shape }, &ops);
            let mut cells = cells.cells();
            // SAFETY: as in the positive test; only the grid and the flag move.
            unsafe { raw_launch(entry, grid, PACKED_BLOCK, &mut cells, "a control") };
            (ops.q_out.bytes(), ops.k_pages.bytes(), ops.v_pages.bytes())
        };

        let (right_q, right_k, right_v) = fire(shape.grid(), shape.hnd);
        assert_eq!(written(&right_q), shape.q_elems(), "the launcher writes every query value");
        let right_pages = written(&right_k);
        assert!(right_pages > 0);

        // (a) the near-miss, as a grid.
        let near = eval(Rule::GatedRms, shape.dims()).expect("ported");
        assert_eq!(near.block, [PACKED_BLOCK, 1, 1], "same block width, which is the trap");
        assert_ne!(near.grid, shape.grid());
        let (near_q, near_k, near_v) = fire(near.grid, shape.hnd);
        let near_live = written(&near_q);
        assert_eq!(
            near_live,
            (shape.rows * shape.kv_heads * shape.head_dim) as usize,
            "`GatedRms` opens `kv_heads` head lanes and every one of them is a QUERY head"
        );
        assert_eq!(written(&near_k), 0, "and no block ever reaches the KV branch");
        assert_eq!(written(&near_v), 0);
        let near_differs = differing(&near_q, &right_q)
            + differing(&near_k, &right_k)
            + differing(&near_v, &right_v);
        assert!(near_differs > 0);

        // (b) the permutation, at the right grid.
        let (flip_q, flip_k, flip_v) = fire(shape.grid(), !shape.hnd);
        assert_eq!(
            differing(&flip_q, &right_q),
            0,
            "the layout flip must not touch `q_out` — if it does, this control is not the \
             permutation it claims to be"
        );
        assert_eq!(
            written(&flip_k),
            right_pages,
            "the flip moved {} page values to {} — a truncation, not a permutation",
            right_pages,
            written(&flip_k)
        );
        let flip_differs = differing(&flip_k, &right_k) + differing(&flip_v, &right_v);
        assert!(
            flip_differs > 0,
            "the layout flip is invisible in the bytes, so this control cannot fail"
        );

        eprintln!(
            "RowsPackedHeads controls:\n  \
             GatedRms {:?} vs RowsPackedHeads {:?} -> {near_live} of {} query values, \
             0 of {right_pages} page values, {near_differs} bytes differ (a TRUNCATION)\n  \
             hnd_layout flipped at the right grid -> {} of {right_pages} page values on both, \
             {flip_differs} of {} page bytes differ (a PERMUTATION)",
            near.grid,
            shape.grid(),
            shape.q_elems(),
            written(&flip_k),
            right_k.len() + right_v.len()
        );
    }

    // -----------------------------------------------------------------------
    // `Rule::PagedScores` and `Rule::PagedScoresDecode`
    //
    // The two rows that could not exist until `kernels::Ty` had a variant for
    // an `enum class ... : u8`. Both kernels take `device::KvScheme` and
    // `device::KvDType` BY VALUE and adjacently, so this pair of fires is
    // also the measurement that `Ty::KvScheme`/`Ty::KvDType` marshal — a
    // kind that renders in the emitters and not in `ArgValue::cell` binds a
    // cell the driver reads four bytes out of where the cubin declares one,
    // and every parameter after it shifts.
    // -----------------------------------------------------------------------

    /// `attention_naive_paged.cu:35`'s `constexpr int BLOCK = 128`.
    const PAGED_BLOCK: u32 = 128;

    /// A paged-attention rectangle.
    ///
    /// `tokens` is per REQUEST and `rows` is the total, which is the
    /// distinction `Dims::requests` exists for: the launcher spells
    /// `dim3(num_requests, total_tokens, num_q_heads)` and a rule that read
    /// `rows` for the request axis would open `total_tokens` requests.
    #[derive(Clone, Copy, Debug)]
    struct Paged {
        requests: u32,
        tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        page_size: u32,
        /// Pages backing each request, the last one partly full.
        pages: u32,
    }

    impl Paged {
        fn total_tokens(self) -> u32 {
            self.requests * self.tokens
        }

        fn total_pages(self) -> u32 {
            self.requests * self.pages
        }

        fn dims(self) -> Dims {
            Dims {
                rows: self.total_tokens(),
                width: self.q_heads * self.head_dim,
                in_width: self.q_heads * self.head_dim,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                stated_head_dim: 0,
                rotary_dims: 0,
                n_experts: 0,
                experts_per_token: 0,
                requests: self.requests,
                altup_streams: 0,
            }
        }

        /// `attention_naive_paged.cu:196-198`, transcribed.
        fn grid(self) -> [u32; 3] {
            [self.requests, self.total_tokens(), self.q_heads]
        }

        fn decode_grid(self) -> [u32; 3] {
            [self.total_tokens(), self.q_heads, 1]
        }

        fn smem(self) -> u32 {
            (self.head_dim + PAGED_BLOCK) * 4
        }

        fn q_elems(self) -> usize {
            (self.total_tokens() * self.q_heads * self.head_dim) as usize
        }

        fn page_elems(self) -> usize {
            (self.total_pages() * self.page_size * self.kv_heads * self.head_dim) as usize
        }
    }

    /// Every buffer and every cell one paged fire needs, built once and read
    /// by both the row's fire and the launcher's own.
    struct PagedOps {
        _q: Buffer,
        _k_pages: Buffer,
        _v_pages: Buffer,
        o: Buffer,
        lse: Buffer,
        _qo_indptr: Buffer,
        _kv_page_indices: Buffer,
        _kv_page_indptr: Buffer,
        _kv_last_page_lens: Buffer,
        values: Vec<ArgValue>,
        pointers: Vec<u64>,
    }

    /// `sm_scale` and `logits_soft_cap` the launcher's callers pass — a real
    /// scale rather than the kernel's own `1/sqrt(head_dim)` fallback, and a
    /// cap of zero, which `transform_logit` reads as "no cap".
    const PAGED_SM_SCALE: f32 = 0.088_388_35;
    const PAGED_SOFT_CAP: f32 = 0.0;
    /// `-1` is the launcher's "no sliding window".
    const PAGED_WINDOW: i32 = -1;

    fn paged_operands(shape: Paged) -> PagedOps {
        let q = Buffer::of(&bf16_fill(shape.q_elems(), 0x51A7_C0DE));
        let k_pages = Buffer::of(&bf16_fill(shape.page_elems(), 0x7E5D_11A2));
        let v_pages = Buffer::of(&bf16_fill(shape.page_elems(), 0x3C0F_B19E));
        let o = Buffer::zeroed(shape.q_elems() * 2);
        let lse = Buffer::zeroed((shape.total_tokens() * shape.q_heads) as usize * 4);

        let qo: Vec<u32> = (0..=shape.requests).map(|r| r * shape.tokens).collect();
        let pg_indptr: Vec<u32> = (0..=shape.requests).map(|r| r * shape.pages).collect();
        // A PERMUTED page table, so a kernel that assumed identity would be
        // visibly wrong: request `r`'s pages are the batch's in reverse.
        let indices: Vec<u32> = (0..shape.total_pages()).rev().collect();
        // The last page of each request is partly full, which is what makes
        // `kv_total` depend on `kv_last_page_lens` rather than on the page
        // count alone.
        let lens: Vec<u32> = (0..shape.requests).map(|_| shape.page_size.max(1) - 1).collect();

        let qo_indptr = Buffer::of(&qo);
        let kv_page_indices = Buffer::of(&indices);
        let kv_page_indptr = Buffer::of(&pg_indptr);
        let kv_last_page_lens = Buffer::of(&lens);

        let pointers = vec![
            q.ptr,
            k_pages.ptr,
            v_pages.ptr,
            0,
            0,
            o.ptr,
            qo_indptr.ptr,
            kv_page_indices.ptr,
            kv_page_indptr.ptr,
            kv_last_page_lens.ptr,
            0,
            0,
        ];
        let values = vec![
            q.arg(),
            k_pages.arg(),
            v_pages.arg(),
            ArgValue::Ptr(std::ptr::null_mut()),
            ArgValue::Ptr(std::ptr::null_mut()),
            o.arg(),
            qo_indptr.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            kv_last_page_lens.arg(),
            ArgValue::Ptr(std::ptr::null_mut()),
            ArgValue::Ptr(std::ptr::null_mut()),
            ArgValue::I32(shape.q_heads as i32),
            ArgValue::I32(shape.kv_heads as i32),
            ArgValue::I32(shape.head_dim as i32),
            ArgValue::I32(shape.page_size as i32),
            // THE TWO NEW KINDS. `KvScheme::Native` is 0 and `KvDType::BF16`
            // is 0, so a binder that dropped either would be invisible here —
            // which is why the second shape below fires `Fp8PerTensor` (1)
            // with `FP8_E4M3` (7), two DIFFERENT non-zero enumerators in
            // adjacent cells.
            ArgValue::U8(0),
            ArgValue::U8(0),
            ArgValue::I32(0),
            ArgValue::I32(PAGED_WINDOW),
            ArgValue::F32(PAGED_SM_SCALE),
            ArgValue::F32(PAGED_SOFT_CAP),
            lse.arg(),
        ];
        PagedOps {
            _q: q,
            _k_pages: k_pages,
            _v_pages: v_pages,
            o,
            lse,
            _qo_indptr: qo_indptr,
            _kv_page_indices: kv_page_indices,
            _kv_page_indptr: kv_page_indptr,
            _kv_last_page_lens: kv_last_page_lens,
            values,
            pointers,
        }
    }

    /// The launcher's own cells, owned, in the kernel's declared order.
    struct PagedCells {
        pointers: Vec<u64>,
        ints: [i32; 4],
        scheme: u8,
        storage_dtype: u8,
        tail_ints: [i32; 2],
        floats: [f32; 2],
        lse: u64,
    }

    impl PagedCells {
        fn new(shape: Paged, ops: &PagedOps, scheme: u8, storage_dtype: u8) -> Self {
            Self {
                pointers: ops.pointers.clone(),
                ints: [
                    shape.q_heads as i32,
                    shape.kv_heads as i32,
                    shape.head_dim as i32,
                    shape.page_size as i32,
                ],
                scheme,
                storage_dtype,
                tail_ints: [0, PAGED_WINDOW],
                floats: [PAGED_SM_SCALE, PAGED_SOFT_CAP],
                lse: ops.lse.ptr,
            }
        }

        /// The cells, in order. **The two enum cells are ONE BYTE each**,
        /// which is the whole of what this comparison tests on the binder's
        /// side: `cuLaunchKernel` copies `sizeof(param)` from the address in
        /// the cell array, so a `u8` here and a `u32` in `ArgValue::cell`
        /// would still agree on a little-endian machine — and a `u8` in the
        /// C++ against a four-byte read would not.
        fn cells(&mut self) -> Vec<*mut c_void> {
            let mut cells: Vec<*mut c_void> =
                self.pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for i in self.ints.iter_mut() {
                cells.push((&raw mut *i).cast());
            }
            cells.push((&raw mut self.scheme).cast());
            cells.push((&raw mut self.storage_dtype).cast());
            for i in self.tail_ints.iter_mut() {
                cells.push((&raw mut *i).cast());
            }
            for f in self.floats.iter_mut() {
                cells.push((&raw mut *f).cast());
            }
            cells.push((&raw mut self.lse).cast());
            cells
        }
    }

    /// Both paged rows are stated, hosted, and claim their rules.
    #[test]
    fn the_paged_rows_are_stated_and_hosted() {
        for (symbol, rule) in [
            ("attn::attention_naive_paged", Rule::PagedScores),
            ("attn::naive_paged_decode", Rule::PagedScoresDecode),
        ] {
            assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
            let row = runtime::row(symbol).expect("hosted");
            assert_eq!(row.sig.launch, rule, "{symbol}");
            assert_eq!(row.sig.file, Some("attn/attention_naive_paged.cuh"), "{symbol}");
        }
    }

    /// **The prefill row is byte-identical to `attention_naive_paged.cu:196-221`.**
    ///
    /// Four shapes, and hazard 1 is why there are four rather than one: a
    /// grid too large in ANY axis is invisible here, because the kernel's own
    /// first lines are `if (qo_off >= int(qo_hi - qo_lo)) return;` and its
    /// `kv` loops are bounded by `kv_lim`. Only a grid too SMALL shows in the
    /// output, so the shapes are chosen to make each axis the small one in
    /// turn: `requests != tokens` in every case, `q_heads != kv_heads` in
    /// three of the four, and one single-request case where a rule that read
    /// `rows` for the request axis would still open the right `grid.y`.
    ///
    /// Both output buffers are compared. `o` is bf16 and `lse_out` is fp32
    /// and only thread 0 of each block writes it, so a block that ran with
    /// the wrong `blockIdx` but the right data is visible in `lse` and not in
    /// `o`.
    #[test]
    fn the_paged_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_paged_row_reproduces_the_launcher") else { return };
        let symbol = "attn::attention_naive_paged";
        let module = module_of(symbol, "attn/attention_naive_paged");
        let entry = module.entry(symbol).expect("the row resolved");

        let shapes = [
            // Four requests of three tokens, grouped-query, 64-wide heads.
            Paged { requests: 4, tokens: 3, q_heads: 8, kv_heads: 2, head_dim: 64, page_size: 4, pages: 2 },
            // The AltUp near-miss's lesson applied: a SECOND shape where the
            // two head counts differ the other way and the page size does not
            // divide the token count.
            Paged { requests: 2, tokens: 5, q_heads: 4, kv_heads: 4, head_dim: 128, page_size: 8, pages: 2 },
            // One request: `requests` and a rule that dropped the axis both
            // answer 1, so this shape cannot catch that — it is here for the
            // opposite reason, that a rule reading `rows` for `grid.x` would
            // open 7 requests over a 2-entry `qo_indptr` and read past it.
            Paged { requests: 1, tokens: 7, q_heads: 6, kv_heads: 3, head_dim: 64, page_size: 4, pages: 3 },
            // Multi-query at a wide head: one KV head, so every Q head reads
            // the same bank and only the Q axis can be short.
            Paged { requests: 3, tokens: 2, q_heads: 8, kv_heads: 1, head_dim: 256, page_size: 2, pages: 2 },
        ];

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (case, shape) in shapes.into_iter().enumerate() {
            let ops = paged_operands(shape);

            let launch = eval(Rule::PagedScores, shape.dims()).expect("ported");
            assert_eq!(launch.grid, shape.grid(), "case {case} at {shape:?}");
            assert_eq!(launch.block, [PAGED_BLOCK, 1, 1], "case {case}");
            assert_eq!(launch.smem, shape.smem(), "case {case}: (head_dim + 128) * 4");

            // SAFETY: every pointer addresses a live allocation of the extent
            // the row states, the values match the row's operand list, and
            // the null stream is live.
            unsafe { runtime::fire(symbol, shape.dims(), &ops.values, Stream::NULL) }
                .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
            synchronise("the shipped fire");
            let row_o = ops.o.bytes();
            let row_lse = ops.lse.bytes();

            ops.o.clear();
            ops.lse.clear();
            synchronise("clearing between the two launches");
            let mut cells = PagedCells::new(shape, &ops, 0, 0);
            let mut cells = cells.cells();
            // SAFETY: `entry` is from a module that outlives the call and the
            // cells are the kernel's twenty-three declared parameters in
            // order and at their declared widths.
            unsafe {
                raw_launch_smem(
                    entry,
                    shape.grid(),
                    PAGED_BLOCK,
                    shape.smem(),
                    &mut cells,
                    "the launcher's own",
                );
            }
            let cu_o = ops.o.bytes();
            let cu_lse = ops.lse.bytes();

            // §18's guard: a whole-buffer comparison over a buffer nothing
            // wrote is two fields of zeros and holds for a kernel that did
            // nothing.
            let live = written(&cu_o);
            assert_eq!(
                live,
                shape.q_elems(),
                "case {case}: the launcher wrote {live} of {} output values",
                shape.q_elems()
            );
            assert!(
                cu_lse.chunks_exact(4).any(|w| w != [0, 0, 0, 0]),
                "case {case}: the launcher wrote no LSE, so that comparison is zeros"
            );

            let differs = differing(&row_o, &cu_o) + differing(&row_lse, &cu_lse);
            assert_eq!(
                differs, 0,
                "case {case} at {shape:?}: the row and the launcher disagree on {differs} \
                 bytes (o {}, lse {})",
                differing(&row_o, &cu_o),
                differing(&row_lse, &cu_lse)
            );
            compared += row_o.len() + row_lse.len();
            live_total += live;
        }
        eprintln!(
            "PagedScores: {} shapes, {compared} bytes compared, {live_total} values \
             written, 0 differing",
            shapes.len()
        );
    }

    /// **The decode row is byte-identical to `attention_naive_paged.cu:147-171`.**
    ///
    /// Its rule reads `Dims::rows` where the prefill's reads
    /// `Dims::requests`, and that identification is licensed by the decode
    /// contract alone — one token per request. The shapes below all satisfy
    /// it (`tokens: 1`), which is what makes the fire a test of the rule
    /// rather than of the identification.
    #[test]
    fn the_paged_decode_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_paged_decode_row_reproduces_the_launcher") else {
            return;
        };
        let symbol = "attn::naive_paged_decode";
        let module = module_of(symbol, "attn/attention_naive_paged");
        let entry = module.entry(symbol).expect("the row resolved");

        let shapes = [
            Paged { requests: 5, tokens: 1, q_heads: 8, kv_heads: 2, head_dim: 64, page_size: 4, pages: 2 },
            Paged { requests: 2, tokens: 1, q_heads: 4, kv_heads: 4, head_dim: 128, page_size: 8, pages: 3 },
            Paged { requests: 3, tokens: 1, q_heads: 6, kv_heads: 1, head_dim: 256, page_size: 2, pages: 2 },
        ];

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (case, shape) in shapes.into_iter().enumerate() {
            let ops = paged_operands(shape);

            let launch = eval(Rule::PagedScoresDecode, shape.dims()).expect("ported");
            assert_eq!(launch.grid, shape.decode_grid(), "case {case} at {shape:?}");
            assert_eq!(launch.block, [PAGED_BLOCK, 1, 1], "case {case}");
            assert_eq!(launch.smem, shape.smem(), "case {case}");

            // The decode kernel takes twenty operands, not twenty-three: no
            // `qo_indptr` and no mask pair. Dropping them from the middle of
            // the list is what a row states and what this slices.
            let decode_values: Vec<ArgValue> = ops.values[..3]
                .iter()
                .chain(&ops.values[3..6])
                .chain(&ops.values[7..10])
                .chain(&ops.values[12..])
                .copied()
                .collect();
            assert_eq!(decode_values.len(), 20, "the decode contract is twenty operands");

            // SAFETY: as above, at the decode contract's operand list.
            unsafe { runtime::fire(symbol, shape.dims(), &decode_values, Stream::NULL) }
                .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
            synchronise("the shipped fire");
            let row_o = ops.o.bytes();
            let row_lse = ops.lse.bytes();

            ops.o.clear();
            ops.lse.clear();
            synchronise("clearing between the two launches");
            let mut cells = PagedCells::new(shape, &ops, 0, 0);
            let cells = cells.cells();
            let mut cells: Vec<*mut c_void> = cells[..3]
                .iter()
                .chain(&cells[3..6])
                .chain(&cells[7..10])
                .chain(&cells[12..])
                .copied()
                .collect();
            // SAFETY: `entry` is from a module that outlives the call and the
            // cells are the decode kernel's twenty declared parameters.
            unsafe {
                raw_launch_smem(
                    entry,
                    shape.decode_grid(),
                    PAGED_BLOCK,
                    shape.smem(),
                    &mut cells,
                    "the launcher's own",
                );
            }
            let cu_o = ops.o.bytes();
            let cu_lse = ops.lse.bytes();

            let live = written(&cu_o);
            assert_eq!(
                live,
                shape.q_elems(),
                "case {case}: the launcher wrote {live} of {} output values",
                shape.q_elems()
            );

            let differs = differing(&row_o, &cu_o) + differing(&row_lse, &cu_lse);
            assert_eq!(
                differs, 0,
                "case {case} at {shape:?}: {differs} bytes differ (o {}, lse {})",
                differing(&row_o, &cu_o),
                differing(&row_lse, &cu_lse)
            );
            compared += row_o.len() + row_lse.len();
            live_total += live;
        }
        eprintln!(
            "PagedScoresDecode: {} shapes, {compared} bytes compared, {live_total} values \
             written, 0 differing",
            shapes.len()
        );
    }

    /// **The two enum cells are read as ONE BYTE EACH, measured.**
    ///
    /// `Ty::KvScheme` and `Ty::KvDType` render in five places and a variant
    /// that rendered in four would fail at the fifth. Four of the five are
    /// build-time — the C shim's parameter, the Rust binding, the dispatch's
    /// cast and `emit_device_typecheck`'s function-pointer type — and the
    /// fifth is `ArgValue`'s marshalling, which nothing but a fire can check.
    ///
    /// So this fires the SAME shape with `KvScheme::Fp8PerTensor` (1) and
    /// `KvDType::FP8_E4M3` (7) — two different non-zero enumerators in
    /// adjacent cells — against a raw launch that spells them as `u8`. Zero
    /// would have been invisible; equal values would not have caught a swap;
    /// and a marshalling that wrote four bytes per cell would shift
    /// `block_size`, `window_left` and both floats, which the bytes report.
    ///
    /// # Two controls, because the interesting one is the quiet one
    ///
    /// **The dtype control is a PERMUTATION.** `(1, 8)` — the same scheme
    /// under `FP8_E5M2` instead of `FP8_E4M3` — decodes every page byte at a
    /// different exponent bias. It writes exactly the same NUMBER of values
    /// into exactly the same addresses and different values in them, which is
    /// hazard 2's shape: no count, no norm and no occupancy figure moves, and
    /// only the bytes differ. That is what a `storage_dtype` cell dropped,
    /// widened or read at the wrong offset would look like.
    ///
    /// **The swap is TOTAL, and that is worth writing down too.** `(7, 1)`
    /// puts a dtype enumerator where the scheme goes, `load_kv_scalar`'s
    /// switch has no case for 7, and every output value comes out zero. So
    /// the adjacency confusion the two distinct `Ty`s exist to make
    /// unspellable is loud on THIS kernel — and that is a property of this
    /// switch and not of the vocabulary. `KvScheme` has five enumerators and
    /// `KvDType` ten; the four values they share would swap silently.
    #[test]
    fn the_kv_enums_cross_as_one_byte_each() {
        let Some(_) = arch_or_skip("the_kv_enums_cross_as_one_byte_each") else { return };
        let symbol = "attn::attention_naive_paged";
        let module = module_of(symbol, "attn/attention_naive_paged");
        let entry = module.entry(symbol).expect("the row resolved");

        // `Fp8PerTensor` needs no scale plane, which is why it is the scheme
        // this fires: the null `k_scales`/`v_scales` above stay null and the
        // only thing that changes is two bytes.
        const FP8_PER_TENSOR: u8 = 1;
        const FP8_E4M3: u8 = 7;
        const FP8_E5M2: u8 = 8;

        let shape =
            Paged { requests: 3, tokens: 2, q_heads: 4, kv_heads: 2, head_dim: 64, page_size: 4, pages: 2 };
        let ops = paged_operands(shape);

        let mut values = ops.values.clone();
        values[16] = ArgValue::U8(FP8_PER_TENSOR);
        values[17] = ArgValue::U8(FP8_E4M3);

        // SAFETY: as `the_paged_row_reproduces_the_launcher`.
        unsafe { runtime::fire(symbol, shape.dims(), &values, Stream::NULL) }
            .unwrap_or_else(|why| panic!("{symbol} would not fire: {why}"));
        synchronise("the shipped fire, fp8");
        let row_o = ops.o.bytes();

        ops.o.clear();
        ops.lse.clear();
        synchronise("clearing");
        let mut cells = PagedCells::new(shape, &ops, FP8_PER_TENSOR, FP8_E4M3);
        let mut cells = cells.cells();
        // SAFETY: as above, at the same twenty-three cells.
        unsafe {
            raw_launch_smem(
                entry,
                shape.grid(),
                PAGED_BLOCK,
                shape.smem(),
                &mut cells,
                "the launcher's own, fp8",
            );
        }
        let cu_o = ops.o.bytes();
        let live = written(&cu_o);
        assert!(live > 0, "the fp8 launch wrote nothing, so this comparison is zeros");
        let differs = differing(&row_o, &cu_o);
        assert_eq!(differs, 0, "the row and the launcher disagree on {differs} bytes at fp8");

        // CONTROL ONE: the dtype alone, at the launcher's own grid and cell
        // widths. Same scheme, same pages, a different exponent bias.
        ops.o.clear();
        ops.lse.clear();
        synchronise("clearing before the dtype control");
        let mut other = PagedCells::new(shape, &ops, FP8_PER_TENSOR, FP8_E5M2);
        let mut other = other.cells();
        // SAFETY: as above.
        unsafe {
            raw_launch_smem(
                entry,
                shape.grid(),
                PAGED_BLOCK,
                shape.smem(),
                &mut other,
                "the e5m2 control",
            );
        }
        let e5m2_o = ops.o.bytes();
        assert_eq!(
            written(&e5m2_o),
            live,
            "the dtype control moved a different NUMBER of values, so it is a truncation \
             and not the permutation this control claims to be"
        );
        let e5m2_differs = differing(&e5m2_o, &cu_o);
        assert!(
            e5m2_differs > 0,
            "reading the same pages as E5M2 rather than E4M3 is invisible in the bytes, \
             so this control cannot fail"
        );

        // CONTROL TWO: the adjacency swap.
        ops.o.clear();
        ops.lse.clear();
        synchronise("clearing before the swap");
        let mut swapped = PagedCells::new(shape, &ops, FP8_E4M3, FP8_PER_TENSOR);
        let mut swapped = swapped.cells();
        // SAFETY: as above.
        unsafe {
            raw_launch_smem(
                entry,
                shape.grid(),
                PAGED_BLOCK,
                shape.smem(),
                &mut swapped,
                "the swapped control",
            );
        }
        let swap_o = ops.o.bytes();
        assert_eq!(
            written(&swap_o),
            0,
            "7 names no `KvScheme`, so `load_kv_scalar` falls through and every value \
             is zero; a non-zero count here means the switch grew a case and this \
             control now needs re-reading"
        );

        eprintln!(
            "Ty::KvScheme/Ty::KvDType: (1, 7) row vs launcher -> {live} values written, \
             {differs} of {} bytes differ\n  \
             control (1, 8) E5M2 at the same grid -> {} values written, {e5m2_differs} \
             of {} bytes differ (a PERMUTATION)\n  \
             control (7, 1) the adjacency swap -> {} of {live} values written (TOTAL)",
            cu_o.len(),
            written(&e5m2_o),
            cu_o.len(),
            written(&swap_o)
        );
    }

    /// **`Dims::requests` filled from `rows`, fired — the substitution
    /// `jit_dims` refuses.**
    ///
    /// `bind/mod.rs`'s `jit_dims` fills `requests` from `AttnCtx` and not
    /// from `rows`, and its comment says what filling it from `rows` would
    /// do. This is that, measured: at four requests of three tokens the
    /// launcher opens `[4, 12, 8]` and the substitution opens `[12, 12, 8]`,
    /// so eight of the twelve request lanes index `qo_indptr` past its end.
    ///
    /// **It is not a crash and it is not a truncation.** `qo_indptr` is five
    /// entries and the read at `r = 11` is inside the allocation's page, so
    /// the launch succeeds and the excess blocks compute `qo_lo`/`qo_hi` from
    /// whatever follows — which for this allocator is zeros, so `qo_hi -
    /// qo_lo` is 0 and every excess block returns at the first `if`. The
    /// output is therefore BYTE-IDENTICAL, and that is the measurement: this
    /// near miss is hazard 1 exactly, and the only thing that catches it is
    /// the grid, which is why `Dims::requests` is a field and not a reading
    /// of `rows`.
    #[test]
    fn filling_requests_from_rows_states_a_grid_the_output_cannot_see() {
        let Some(_) = arch_or_skip("filling_requests_from_rows_states_a_grid_the_output_cannot_see")
        else {
            return;
        };
        let shape =
            Paged { requests: 4, tokens: 3, q_heads: 8, kv_heads: 2, head_dim: 64, page_size: 4, pages: 2 };

        let right = eval(Rule::PagedScores, shape.dims()).expect("ported");
        let mut substituted = shape.dims();
        substituted.requests = substituted.rows;
        let wrong = eval(Rule::PagedScores, substituted).expect("ported");

        assert_eq!(right.grid, shape.grid());
        assert_ne!(
            right.grid, wrong.grid,
            "if the substitution states the same grid there is nothing here to refuse"
        );
        assert_eq!(wrong.grid[0], shape.total_tokens());
        // THE RATIO, which is what `jit_dims`' comment names: `total_tokens /
        // num_requests` times the blocks, every extra one a request lane that
        // does not exist.
        let extra = u64::from(wrong.grid[0]) * u64::from(wrong.grid[1]) * u64::from(wrong.grid[2])
            - u64::from(right.grid[0]) * u64::from(right.grid[1]) * u64::from(right.grid[2]);
        assert!(extra > 0);

        eprintln!(
            "Dims::requests: the launcher's {:?} against `requests: rows`'s {:?} \
             -- {extra} excess blocks, each reading `qo_indptr[r]` for an `r` past \
             the CSR's {} entries",
            right.grid,
            wrong.grid,
            shape.requests + 1
        );
    }

    // -----------------------------------------------------------------------
    // `Rule::RoutedQmvQuad`
    //
    // The two MXFP4 MoE decode GEMVs. Their rows could not fire until a fire
    // carried an expert fanout — `jit_dims` filled `experts_per_token: 0` and
    // `eval` answered `Ungeometric::Empty` — and they are the rule's only
    // readers, so this is where its arithmetic meets a kernel.
    // -----------------------------------------------------------------------

    /// `dequant_fp4.cu:37`'s `kMxfp4DecodeBlock`.
    const MXFP4_BLOCK: u32 = 128;
    /// `dequant_fp4.cu:40` and `:42` — `kMxfp4GateUpPairs`, `kMxfp4DownRows`.
    const MXFP4_TILE: u32 = (MXFP4_BLOCK / 32) * 4;

    /// A routed MXFP4 decode rectangle.
    ///
    /// `hidden` and `intermediate` are both multiples of 32, which the two
    /// launchers require (`dequant_fp4.cu:63` and `:150`) and the kernels
    /// depend on: `groups_per_row` is `width / 32` and a remainder would be a
    /// tail no lane covers.
    #[derive(Clone, Copy, Debug)]
    struct Mxfp4 {
        tokens: u32,
        top_k: u32,
        experts: u32,
        hidden: u32,
        intermediate: u32,
    }

    impl Mxfp4 {
        fn routes(self) -> u32 {
            self.tokens * self.top_k
        }

        /// The STACKED dims a fire hands the rule.
        ///
        /// `width` is the first output's ROW width, and both statements
        /// declare `[Tokens, k, w]` — so it is `top_k * w` and the rule
        /// divides. `in_width` is the route-index row, `[Tokens, k]`.
        fn dims(self, per_route_width: u32) -> Dims {
            Dims {
                rows: self.tokens,
                width: self.top_k * per_route_width,
                in_width: self.top_k,
                q_heads: 0,
                kv_heads: 0,
                head_dim: 0,
                stated_head_dim: 0,
                rotary_dims: 0,
                n_experts: self.experts,
                experts_per_token: self.top_k,
                requests: 0,
                altup_streams: 0,
            }
        }

        /// `dequant_fp4.cu:67-70` and `:152-156`, transcribed.
        fn grid(self, per_route_width: u32) -> [u32; 3] {
            [self.routes(), per_route_width.div_ceil(MXFP4_TILE), 1]
        }
    }

    /// One expert bank: the packed nibbles and the block scales for `rows`
    /// rows of `width` codes.
    struct Bank {
        packed: Buffer,
        scales: Buffer,
    }

    impl Bank {
        fn new(rows: u32, width: u32, seed: u64) -> Self {
            let bytes = (rows as usize) * (width as usize) / 2;
            let groups = (rows as usize) * (width as usize) / 32;
            let mut state = seed | 1;
            let mut byte = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            };
            let packed: Vec<u8> = (0..bytes).map(|_| byte()).collect();
            // E8M0 exponents near the middle of the range, so no product
            // overflows and none flushes: `mxfp4_block_scale` reads
            // `2^(b - 127)`, and `b` in `[123, 130]` keeps every scale within
            // a factor of eight of one.
            let scales: Vec<u8> = (0..groups).map(|_| 123 + (byte() & 7)).collect();
            Self { packed: Buffer::of(&packed), scales: Buffer::of(&scales) }
        }
    }

    /// fp16 activations, as bit patterns: the same generator the bf16 fill
    /// uses, shifted into IEEE half's exponent field.
    fn f16_fill(n: usize, seed: u64) -> Vec<u16> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Exponent 12..=17 around half's bias of 15, so every value
                // is within a factor of eight of one and no accumulation
                // saturates half's narrow range.
                let exponent = u16::try_from(12 + (state >> 32) % 6).expect("small");
                let mantissa = u16::try_from((state >> 8) & 0x3FF).expect("ten bits") | 1;
                let sign = u16::try_from((state >> 3) & 1).expect("one bit") << 15;
                sign | (exponent << 10) | mantissa
            })
            .collect()
    }

    /// Both MXFP4 rows are stated, hosted, and claim `RoutedQmvQuad`.
    #[test]
    fn the_mxfp4_moe_rows_are_stated_and_hosted() {
        for symbol in
            ["quant::mxfp4_moe_gate_up_decode_bf16", "quant::mxfp4_moe_down_decode_bf16"]
        {
            assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
            let row = runtime::row(symbol).expect("hosted");
            assert_eq!(row.sig.launch, Rule::RoutedQmvQuad, "{symbol}");
            assert_eq!(row.sig.file, Some("quant/dequant_fp4.cuh"), "{symbol}");
        }
    }

    /// **Every row that states `RoutedQmvQuad` declares its first input as
    /// the route-index row.**
    ///
    /// The rule divides `Dims::width` by the fanout because its statements
    /// declare `[Tokens, k, w]`, and nothing in `eval` can see that: `Dims`
    /// carries numbers and not shapes, and `Ungeometric::Empty` is for a
    /// rectangle that collapsed rather than for a shape a rule dislikes. So
    /// the claim lives here, over the rows, where it is checkable.
    ///
    /// `Source::In(0)` being the `I32s` route row is the whole of it: both
    /// MXFP4 statements pass `vec![experts.id, x.id]`, so input 0 is
    /// `[Tokens, k]` of `i32` and its width IS the fanout. A row that took
    /// the ACTIVATION as input 0 — which is what
    /// `Rule::RoutedQmv`'s two rows do — would be a collapsed-shape
    /// statement under a stacked rule, and would slab `w / k` columns while
    /// reporting nothing.
    #[test]
    fn every_stacked_rule_reads_a_route_index_row() {
        let mut seen = 0usize;
        for k in unit::rows() {
            if k.sig.launch != Rule::RoutedQmvQuad {
                continue;
            }
            seen += 1;
            let first = k.sig.operands.first().expect("a row has operands");
            assert_eq!(
                first.source,
                kernels::Source::In(1),
                "{}: `RoutedQmvQuad` divides `Dims::width` by the fanout, so the \
                 statement's input 0 must be the route-index row and the kernel's \
                 first operand is therefore its ACTIVATION, `In(1)`",
                k.sig.symbol
            );
            let route = k
                .sig
                .operands
                .iter()
                .find(|o| o.source == kernels::Source::In(0))
                .unwrap_or_else(|| panic!("{}: no operand reads input 0", k.sig.symbol));
            assert_eq!(
                route.ty,
                kernels::Ty::I32s,
                "{}: input 0 must be the `[Tokens, k]` route row, whose width is the \
                 fanout `RoutedQmvQuad` divides by",
                k.sig.symbol
            );
        }
        assert!(seen >= 2, "only {seen} rows state RoutedQmvQuad; this check is vacuous");
    }

    /// **The gate/up row is byte-identical to `dequant_fp4.cu:67-77`.**
    ///
    /// Three shapes, because hazard 1 says one certifies a near miss, and
    /// they are chosen so each of the rule's three inputs is the one that
    /// varies: `top_k` 2 then 4 then 1, `intermediate` a multiple of the
    /// 16-row tile and then not, and an expert count that is smaller than
    /// the route count in two of the three so the route-to-expert map is
    /// genuinely many-to-one.
    #[test]
    fn the_mxfp4_gate_up_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_mxfp4_gate_up_row_reproduces_the_launcher") else {
            return;
        };
        let symbol = "quant::mxfp4_moe_gate_up_decode_bf16";
        let module = module_of(symbol, "quant/dequant_fp4");
        let entry = module.entry(symbol).expect("the row resolved");

        let shapes = [
            Mxfp4 { tokens: 3, top_k: 2, experts: 4, hidden: 64, intermediate: 32 },
            // `intermediate` is NOT a multiple of the 16-row tile, so the
            // last slab overhangs and the kernel's `min(row0 + p, ...)` clamp
            // runs — the branch a rule that floored `grid.y` would skip.
            Mxfp4 { tokens: 2, top_k: 4, experts: 3, hidden: 96, intermediate: 40 },
            // One expert per token: `routes == tokens`, so a rule that
            // dropped the fanout from `grid.x` would still be right here and
            // wrong in the other two.
            Mxfp4 { tokens: 5, top_k: 1, experts: 2, hidden: 32, intermediate: 64 },
        ];

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (case, shape) in shapes.into_iter().enumerate() {
            // The gate/up bank interleaves gate and up rows, so a bank is
            // `2 * intermediate` rows of `hidden` codes.
            let banks: Vec<Bank> = (0..shape.experts)
                .map(|e| Bank::new(2 * shape.intermediate, shape.hidden, 0xB00C_0001 + u64::from(e)))
                .collect();
            let packed_ptrs = Buffer::of(&banks.iter().map(|b| b.packed.ptr).collect::<Vec<_>>());
            let scale_ptrs = Buffer::of(&banks.iter().map(|b| b.scales.ptr).collect::<Vec<_>>());
            let act = Buffer::of(&f16_fill((shape.tokens * shape.hidden) as usize, 0xACC1_7E5A));
            // A route map that is neither identity nor constant.
            let topk: Vec<i32> = (0..shape.routes())
                .map(|r| ((r * 7 + 3) % shape.experts) as i32)
                .collect();
            let topk_idx = Buffer::of(&topk);
            let out_elems = (shape.routes() * shape.intermediate) as usize;
            let gate_out = Buffer::zeroed(out_elems * 2);
            let up_out = Buffer::zeroed(out_elems * 2);

            let dims = shape.dims(shape.intermediate);
            let launch = eval(Rule::RoutedQmvQuad, dims).expect("ported");
            assert_eq!(launch.grid, shape.grid(shape.intermediate), "case {case} at {shape:?}");
            assert_eq!(launch.block, [MXFP4_BLOCK, 1, 1], "case {case}");
            assert_eq!(launch.smem, 0, "case {case}: nothing shared");

            let values = vec![
                act.arg(),
                topk_idx.arg(),
                packed_ptrs.arg(),
                scale_ptrs.arg(),
                ArgValue::Ptr(std::ptr::null_mut()),
                ArgValue::Ptr(std::ptr::null_mut()),
                gate_out.arg(),
                up_out.arg(),
                ArgValue::Ptr(std::ptr::null_mut()),
                ArgValue::F32(0.0),
                ArgValue::F32(0.0),
                ArgValue::I32(shape.top_k as i32),
                ArgValue::I32(shape.hidden as i32),
                ArgValue::I32(shape.intermediate as i32),
            ];

            // SAFETY: every pointer addresses a live allocation of the extent
            // the row states, the values match the row's operand list, and
            // the null stream is live.
            unsafe { runtime::fire(symbol, dims, &values, Stream::NULL) }
                .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
            synchronise("the shipped fire");
            let row_gate = gate_out.bytes();
            let row_up = up_out.bytes();

            gate_out.clear();
            up_out.clear();
            synchronise("clearing between the two launches");
            let mut cells = Mxfp4Cells::gate_up(
                &[
                    act.ptr,
                    topk_idx.ptr,
                    packed_ptrs.ptr,
                    scale_ptrs.ptr,
                    0,
                    0,
                    gate_out.ptr,
                    up_out.ptr,
                    0,
                ],
                shape,
            );
            let mut cells = cells.cells();
            // SAFETY: `entry` is from a module that outlives the call and the
            // cells are the kernel's fourteen declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    shape.grid(shape.intermediate),
                    MXFP4_BLOCK,
                    &mut cells,
                    "the launcher's own",
                );
            }
            let cu_gate = gate_out.bytes();
            let cu_up = up_out.bytes();

            let live = written(&cu_gate) + written(&cu_up);
            assert_eq!(
                written(&cu_gate),
                out_elems,
                "case {case}: the launcher wrote {} of {out_elems} gate values",
                written(&cu_gate)
            );
            assert!(written(&cu_up) > 0, "case {case}: no up values, so that half is zeros");

            let differs = differing(&row_gate, &cu_gate) + differing(&row_up, &cu_up);
            assert_eq!(
                differs, 0,
                "case {case} at {shape:?}: the row and the launcher disagree on {differs} \
                 bytes (gate {}, up {})",
                differing(&row_gate, &cu_gate),
                differing(&row_up, &cu_up)
            );
            compared += row_gate.len() + row_up.len();
            live_total += live;
        }
        eprintln!(
            "RoutedQmvQuad gate/up: {} shapes, {compared} bytes compared, {live_total} \
             values written, 0 differing",
            shapes.len()
        );
    }

    /// **The down row is byte-identical to `dequant_fp4.cu:152-162`.**
    ///
    /// The same rule, and that is the claim: `dequant_fp4.cuh` takes
    /// `route = blockIdx.x` in BOTH kernels where `dequant_wna16.cuh` swaps
    /// them between its two, so this file needs one rule where that one needs
    /// `RoutedQmv` and `RoutedQmvTransposed`. A fire of both legs under one
    /// rule is what makes that a measurement.
    #[test]
    fn the_mxfp4_down_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_mxfp4_down_row_reproduces_the_launcher") else {
            return;
        };
        let symbol = "quant::mxfp4_moe_down_decode_bf16";
        let module = module_of(symbol, "quant/dequant_fp4");
        let entry = module.entry(symbol).expect("the row resolved");

        let shapes = [
            Mxfp4 { tokens: 3, top_k: 2, experts: 4, hidden: 32, intermediate: 64 },
            Mxfp4 { tokens: 2, top_k: 4, experts: 3, hidden: 40, intermediate: 96 },
            Mxfp4 { tokens: 5, top_k: 1, experts: 2, hidden: 64, intermediate: 32 },
        ];

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (case, shape) in shapes.into_iter().enumerate() {
            // The down bank is `hidden` rows of `intermediate` codes: the
            // transpose of the projection, not of the tensor.
            let banks: Vec<Bank> = (0..shape.experts)
                .map(|e| Bank::new(shape.hidden, shape.intermediate, 0xD0_0000_1 + u64::from(e)))
                .collect();
            let packed_ptrs = Buffer::of(&banks.iter().map(|b| b.packed.ptr).collect::<Vec<_>>());
            let scale_ptrs = Buffer::of(&banks.iter().map(|b| b.scales.ptr).collect::<Vec<_>>());
            // The down leg reads `act + route * intermediate`, so its
            // activation is ROUTED and not per token.
            let act =
                Buffer::of(&f16_fill((shape.routes() * shape.intermediate) as usize, 0x5EED_D0F0));
            let topk: Vec<i32> = (0..shape.routes())
                .map(|r| ((r * 5 + 1) % shape.experts) as i32)
                .collect();
            let topk_idx = Buffer::of(&topk);
            let out_elems = (shape.routes() * shape.hidden) as usize;
            let out = Buffer::zeroed(out_elems * 2);

            let dims = shape.dims(shape.hidden);
            let launch = eval(Rule::RoutedQmvQuad, dims).expect("ported");
            assert_eq!(launch.grid, shape.grid(shape.hidden), "case {case} at {shape:?}");
            assert_eq!(launch.block, [MXFP4_BLOCK, 1, 1], "case {case}");

            let values = vec![
                act.arg(),
                topk_idx.arg(),
                packed_ptrs.arg(),
                scale_ptrs.arg(),
                ArgValue::Ptr(std::ptr::null_mut()),
                out.arg(),
                ArgValue::I32(shape.hidden as i32),
                ArgValue::I32(shape.intermediate as i32),
            ];

            // SAFETY: as the gate/up fire, at the down contract's list.
            unsafe { runtime::fire(symbol, dims, &values, Stream::NULL) }
                .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
            synchronise("the shipped fire");
            let row_out = out.bytes();

            out.clear();
            synchronise("clearing between the two launches");
            let mut cells = Mxfp4Cells::down(
                &[act.ptr, topk_idx.ptr, packed_ptrs.ptr, scale_ptrs.ptr, 0, out.ptr],
                shape,
            );
            let mut cells = cells.cells();
            // SAFETY: `entry` is from a module that outlives the call and the
            // cells are the kernel's eight declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    shape.grid(shape.hidden),
                    MXFP4_BLOCK,
                    &mut cells,
                    "the launcher's own",
                );
            }
            let cu_out = out.bytes();

            let live = written(&cu_out);
            assert_eq!(
                live, out_elems,
                "case {case}: the launcher wrote {live} of {out_elems} values"
            );
            let differs = differing(&row_out, &cu_out);
            assert_eq!(differs, 0, "case {case} at {shape:?}: {differs} bytes differ");
            compared += row_out.len();
            live_total += live;
        }
        eprintln!(
            "RoutedQmvQuad down: {} shapes, {compared} bytes compared, {live_total} \
             values written, 0 differing",
            shapes.len()
        );
    }

    /// The launcher's own cells for the two MXFP4 legs, owned.
    struct Mxfp4Cells {
        pointers: Vec<u64>,
        floats: Vec<f32>,
        ints: Vec<i32>,
    }

    impl Mxfp4Cells {
        fn gate_up(pointers: &[u64], shape: Mxfp4) -> Self {
            Self {
                pointers: pointers.to_vec(),
                floats: vec![0.0, 0.0],
                ints: vec![
                    shape.top_k as i32,
                    shape.hidden as i32,
                    shape.intermediate as i32,
                ],
            }
        }

        fn down(pointers: &[u64], shape: Mxfp4) -> Self {
            Self {
                pointers: pointers.to_vec(),
                floats: Vec::new(),
                ints: vec![shape.hidden as i32, shape.intermediate as i32],
            }
        }

        fn cells(&mut self) -> Vec<*mut c_void> {
            let mut cells: Vec<*mut c_void> =
                self.pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for f in self.floats.iter_mut() {
                cells.push((&raw mut *f).cast());
            }
            for i in self.ints.iter_mut() {
                cells.push((&raw mut *i).cast());
            }
            cells
        }
    }

    /// **`RoutedQmv`'s grid fired against `RoutedQmvQuad`'s kernel — and it
    /// is INVISIBLE, which is the measurement.**
    ///
    /// Two constants differ and they push opposite ways: `RoutedQmv` is
    /// `dim3(routes, ceil(w / 8))` at 256 threads and `RoutedQmvQuad` is
    /// `dim3(routes, ceil(w / 16))` at 128. `grid.x` therefore AGREES and the
    /// wrong grid launches four times the blocks.
    ///
    /// It writes the same bytes. `row0` is
    /// `(blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows` and both
    /// factors are read from `blockDim.x` at RUN time, so a block twice as
    /// wide does not overrun a slab — it renumbers the warps, and on the
    /// warps that land inside the tensor the renumbering is the identity.
    /// The other three quarters take `if (row0 >= hidden) return`. This is
    /// hazard 1 exactly, on this rule, at this kernel: **the output cannot
    /// tell the two rules apart and only the block count can.**
    ///
    /// So the assertion here is the honest one — the block counts differ,
    /// the bytes do not — and the VISIBLE control is the other error, the
    /// one the stacked-width divide can make.
    ///
    /// # The wrong divide truncates, and that is what the bytes catch
    ///
    /// `RoutedQmvQuad` divides `Dims::width` by `Dims::experts_per_token`
    /// because its two statements declare `[Tokens, k, w]`. Applying that
    /// divide to a width that was already per-route makes `grid.y` a factor
    /// of `k` too small, and a short grid is not absorbed by any guard: the
    /// rows past `ceil(w / (16k)) * 16` are never claimed by any block and
    /// stay at whatever the output held. That is measured below, and it is
    /// why `every_stacked_rule_reads_a_route_index_row` checks the shape of
    /// the statement rather than `eval` checking the numbers.
    #[test]
    fn the_routed_qmv_near_miss_is_absorbed_and_the_wrong_divide_truncates() {
        let Some(_) = arch_or_skip("the_routed_qmv_near_miss_relabels_the_slabs") else {
            return;
        };
        let symbol = "quant::mxfp4_moe_down_decode_bf16";
        let module = module_of(symbol, "quant/dequant_fp4");
        let entry = module.entry(symbol).expect("the row resolved");

        // Wide enough that the two `grid.y` values genuinely differ: at
        // `hidden = 128` the right grid is 8 slabs and the near miss is 16.
        let shape = Mxfp4 { tokens: 4, top_k: 2, experts: 4, hidden: 128, intermediate: 64 };
        let banks: Vec<Bank> = (0..shape.experts)
            .map(|e| Bank::new(shape.hidden, shape.intermediate, 0xFACE_0001 + u64::from(e)))
            .collect();
        let packed_ptrs = Buffer::of(&banks.iter().map(|b| b.packed.ptr).collect::<Vec<_>>());
        let scale_ptrs = Buffer::of(&banks.iter().map(|b| b.scales.ptr).collect::<Vec<_>>());
        let act =
            Buffer::of(&f16_fill((shape.routes() * shape.intermediate) as usize, 0x11FE_0FF0));
        let topk: Vec<i32> =
            (0..shape.routes()).map(|r| ((r * 3 + 2) % shape.experts) as i32).collect();
        let topk_idx = Buffer::of(&topk);
        let out_elems = (shape.routes() * shape.hidden) as usize;
        let out = Buffer::zeroed(out_elems * 2);

        let dims = shape.dims(shape.hidden);
        let right = eval(Rule::RoutedQmvQuad, dims).expect("ported");
        let near = eval(Rule::RoutedQmv, dims).expect("ported");
        assert_eq!(right.grid[0], near.grid[0], "grid.x agrees, which is the trap");
        assert_ne!(right.grid[1], near.grid[1], "if grid.y agreed there is nothing to catch");
        assert_ne!(right.block[0], near.block[0]);

        let pointers = [act.ptr, topk_idx.ptr, packed_ptrs.ptr, scale_ptrs.ptr, 0, out.ptr];

        let mut cells = Mxfp4Cells::down(&pointers, shape);
        let mut cells = cells.cells();
        // SAFETY: `entry` outlives the call and the cells are the kernel's
        // eight declared parameters.
        unsafe { raw_launch(entry, right.grid, right.block[0], &mut cells, "the right grid") };
        let right_out = out.bytes();
        let right_live = written(&right_out);
        assert_eq!(right_live, out_elems, "the right grid must write every value");

        out.clear();
        synchronise("clearing before the near miss");
        let mut cells = Mxfp4Cells::down(&pointers, shape);
        let mut cells = cells.cells();
        // SAFETY: as above, at `RoutedQmv`'s grid and block.
        unsafe { raw_launch(entry, near.grid, near.block[0], &mut cells, "the near miss") };
        let near_out = out.bytes();
        let near_live = written(&near_out);
        let near_differs = differing(&near_out, &right_out);

        assert_eq!(
            near_differs, 0,
            "the `RoutedQmv` near miss became VISIBLE in the output ({near_differs} of {} \
             bytes). That is a better world than the measured one, but it means this \
             kernel's guard changed and both this control and \
             `LaunchRule::RoutedQmvQuad`'s doc need re-reading",
            right_out.len()
        );
        assert_eq!(near_live, right_live, "and it writes the same count, as it must");
        let blocks_right = right.grid[0] * right.grid[1];
        let blocks_near = near.grid[0] * near.grid[1];
        assert!(
            blocks_near > blocks_right,
            "if the two rules launched the same blocks they would be one rule"
        );

        // THE VISIBLE ERROR: the stacked divide applied to a per-route width.
        // `grid.y` comes out a factor of `top_k` short, and the rows past it
        // are claimed by no block at all.
        out.clear();
        synchronise("clearing before the wrong divide");
        let short = [right.grid[0], right.grid[1].div_ceil(shape.top_k), 1];
        assert!(short[1] < right.grid[1], "the wrong divide must actually shorten the grid");
        let mut cells = Mxfp4Cells::down(&pointers, shape);
        let mut cells = cells.cells();
        // SAFETY: as above; a SHORTER grid reads a subset of what the right
        // one reads, so nothing here can run out of bounds.
        unsafe { raw_launch(entry, short, right.block[0], &mut cells, "the wrong divide") };
        let short_out = out.bytes();
        let short_live = written(&short_out);
        let short_differs = differing(&short_out, &right_out);
        assert!(
            short_differs > 0,
            "dividing an already-per-route width left the output unchanged, so the \
             stacked/collapsed distinction this rule turns on is unmeasurable"
        );
        assert!(
            short_live < right_live,
            "the wrong divide must LOSE values ({short_live} against {right_live}); if it \
             wrote as many it is a permutation and this comment is wrong about why"
        );

        eprintln!(
            "RoutedQmvQuad vs RoutedQmv on `mxfp4_moe_down_decode<4>` at {shape:?}:\n  \
             right {:?} block {} -> {blocks_right} blocks, {right_live} of {out_elems} \
             values\n  \
             near  {:?} block {} -> {blocks_near} blocks ({}x), {near_live} of {out_elems} \
             values, {near_differs} of {} bytes differ (ABSORBED by the row0 guard)\n  \
             wrong divide {short:?} block {} -> {short_live} of {out_elems} values, \
             {short_differs} of {} bytes differ (a TRUNCATION)",
            right.grid,
            right.block[0],
            near.grid,
            near.block[0],
            blocks_near / blocks_right,
            right_out.len(),
            right.block[0],
            right_out.len()
        );
    }

    // -----------------------------------------------------------------------
    // `Rule::RowsPackedHeadsNarrow`, through `rope/rope.cu` — the two rows
    // this pass landed out of `migration_status`' class D.
    // -----------------------------------------------------------------------

    /// `rope/rope.cu:189`, `:213` and `:45` all write `constexpr int BLOCK = 128;`.
    const ROPE_BLOCK: u32 = 128;

    /// A `qk_rmsnorm_rotate` rectangle: `tokens` rows over a PACKED
    /// `q_heads + kv_heads` head axis of `head_dim`.
    ///
    /// `q` and `k` are SEPARATE banks with separate strides — the kernel takes
    /// two base pointers and picks between them on `blockIdx.y < num_q_heads`,
    /// which is the whole difference from `qkv_fused`'s single packed row and
    /// the reason a fixture where `q_heads == kv_heads` proves less.
    #[derive(Clone, Copy, Debug)]
    struct Rope {
        tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    }

    impl Rope {
        fn dims(self) -> Dims {
            Dims {
                rows: self.tokens,
                width: self.q_heads * self.head_dim,
                in_width: self.q_heads * self.head_dim,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                stated_head_dim: 0,
                rotary_dims: self.head_dim,
                n_experts: 0,
                experts_per_token: 0,
                requests: 0,
                altup_streams: 0,
            }
        }

        /// `rope/rope.cu:190`, by hand:
        /// `dim3 grid(num_tokens, num_q_heads + num_kv_heads);`
        fn grid(self) -> [u32; 3] {
            [self.tokens, self.q_heads + self.kv_heads, 1]
        }

        fn q_elems(self) -> usize {
            (self.tokens * self.q_heads * self.head_dim) as usize
        }

        fn k_elems(self) -> usize {
            (self.tokens * self.kv_heads * self.head_dim) as usize
        }
    }

    /// Everything a `qk_rmsnorm_rotate` fire needs, allocated and uploaded.
    ///
    /// `q` and `k` are IN PLACE — the kernel reads and writes the same bytes —
    /// so every launch has to start from the same upload. [`reset`] is what
    /// makes the two comparisons below comparisons of the launch rather than
    /// of whichever ran second.
    struct RopeOps {
        q: Buffer,
        k: Buffer,
        q_weight: Buffer,
        k_weight: Buffer,
        positions: Buffer,
        q_seed: Vec<u16>,
        k_seed: Vec<u16>,
    }

    impl RopeOps {
        fn new(shape: Rope) -> Self {
            let q_seed = bf16_fill(shape.q_elems(), 0x9E37_79B9);
            let k_seed = bf16_fill(shape.k_elems(), 0x2545_F491);
            // The positions are NOT `0..tokens`: an identity table makes
            // `pos * freq` agree with `blockIdx.x * freq` for any kernel that
            // read the wrong one, and the rotation is the only thing here that
            // reads the token index other than the addressing.
            let pos: Vec<i32> = (0..shape.tokens).map(|t| (t as i32) * 3 + 5).collect();
            Self {
                q: Buffer::of(&q_seed),
                k: Buffer::of(&k_seed),
                q_weight: Buffer::of(&bf16_fill(shape.head_dim as usize, 0x1234_5678)),
                k_weight: Buffer::of(&bf16_fill(shape.head_dim as usize, 0x8765_4321)),
                positions: Buffer::of(&pos),
                q_seed,
                k_seed,
            }
        }

        fn reset(&self) {
            // SAFETY: both allocations are exactly their seed's size.
            let code = unsafe {
                dr::cuMemcpyHtoD_v2(self.q.ptr, self.q_seed.as_ptr().cast(), self.q.bytes)
            };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "reset q");
            // SAFETY: as above.
            let code = unsafe {
                dr::cuMemcpyHtoD_v2(self.k.ptr, self.k_seed.as_ptr().cast(), self.k.bytes)
            };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "reset k");
            synchronise("resetting the in-place banks");
        }

        /// The row's operand list, in the row's declared order.
        fn values(&self, shape: Rope, mrope: bool) -> Vec<ArgValue> {
            let mut v = vec![
                self.q.arg(),
                self.k.arg(),
                self.q_weight.arg(),
                self.k_weight.arg(),
                self.positions.arg(),
                ArgValue::I32(shape.q_heads as i32),
                ArgValue::I32(shape.kv_heads as i32),
                ArgValue::I32(shape.head_dim as i32),
                ArgValue::F32(ROPE_THETA),
                ArgValue::F32(ROPE_EPS),
            ];
            if mrope {
                // `mrope_section` (t, h, w): the three halves of a vision
                // model's position triple. They sum to `head_dim / 2` at every
                // real checkpoint and the kernel indexes `positions` by which
                // section a pair falls in.
                let third = (shape.head_dim / 2) / 3;
                v.push(ArgValue::I32(third as i32));
                v.push(ArgValue::I32(third as i32));
                v.push(ArgValue::I32((shape.head_dim / 2 - 2 * third) as i32));
            }
            v
        }
    }

    const ROPE_THETA: f32 = 10_000.0;
    const ROPE_EPS: f32 = 1e-6;

    /// The kernel's own cells, owned, in its declared order.
    struct RopeCells {
        pointers: [u64; 5],
        ints: [i32; 3],
        floats: [f32; 2],
        sections: [i32; 3],
        mrope: bool,
    }

    impl RopeCells {
        fn new(shape: Rope, ops: &RopeOps, mrope: bool) -> Self {
            let third = ((shape.head_dim / 2) / 3) as i32;
            Self {
                pointers: [
                    ops.q.ptr,
                    ops.k.ptr,
                    ops.q_weight.ptr,
                    ops.k_weight.ptr,
                    ops.positions.ptr,
                ],
                ints: [shape.q_heads as i32, shape.kv_heads as i32, shape.head_dim as i32],
                floats: [ROPE_THETA, ROPE_EPS],
                sections: [third, third, (shape.head_dim / 2) as i32 - 2 * third],
                mrope,
            }
        }

        fn cells(&mut self) -> Vec<*mut c_void> {
            let mut cells: Vec<*mut c_void> =
                self.pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for i in self.ints.iter_mut() {
                cells.push((&raw mut *i).cast());
            }
            for f in self.floats.iter_mut() {
                cells.push((&raw mut *f).cast());
            }
            if self.mrope {
                for s in self.sections.iter_mut() {
                    cells.push((&raw mut *s).cast());
                }
            }
            cells
        }
    }

    /// The three shapes every rope comparison below runs at.
    ///
    /// Hazard 1 is why there are three rather than one. §22.7's `AltUpStreams`
    /// near miss was byte-identical at `kv_heads = 8` and 10 221 of 20 480 at
    /// `kv_heads = 2`, so the q/kv SPLIT is varied and not just the row count:
    /// a rule that opened `[rows, q_heads]` is right at 32/8 for the query
    /// bank and invisible in it, and only the kv bank says.
    const ROPE_SHAPES: [Rope; 3] = [
        // Qwen3-style grouped query: 32 q, 8 kv, 128-wide heads.
        Rope { tokens: 7, q_heads: 32, kv_heads: 8, head_dim: 128 },
        // Multi-query, and `tokens != q_heads != kv_heads` in every pair — so
        // a transposed or dropped axis cannot come out the same number.
        Rope { tokens: 5, q_heads: 16, kv_heads: 1, head_dim: 64 },
        // A single decode row at a head dim TWICE the block, so the reduction
        // strides `i += BLOCK` more than once and a block width the rule got
        // wrong is in the algebra rather than only in the grid.
        Rope { tokens: 1, q_heads: 4, kv_heads: 4, head_dim: 256 },
    ];

    /// Both rope rows are stated, hosted, and claim `RowsPackedHeadsNarrow`.
    #[test]
    fn the_rope_rows_are_stated_and_hosted() {
        for symbol in ["rope::qk_rmsnorm_rope_bf16", "rope::qk_rmsnorm_mrope_bf16"] {
            assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
            let row = runtime::row(symbol).expect("hosted");
            assert_eq!(row.sig.launch, Rule::RowsPackedHeadsNarrow, "{symbol}");
            assert_eq!(row.sig.file, Some("rope/rope.cuh"), "{symbol}");
        }
    }

    /// **Both rope rows are byte-identical to `rope/rope.cu:189-191` and
    /// `:45-47`.**
    ///
    /// Three shapes each, both banks compared. `k` is the one that matters:
    /// every wrong reading of this launcher that is not simply a smaller grid
    /// — `[rows, q_heads]`, `[rows, kv_heads]`, a transposed pair — writes the
    /// query bank correctly or not at all, and differs in the KV bank. So both
    /// buffers are downloaded and both are asserted written.
    #[test]
    fn the_rope_rows_reproduce_the_launcher() {
        let Some(_) = arch_or_skip("the_rope_rows_reproduce_the_launcher") else { return };
        let module = module_of("rope::qk_rmsnorm_rope_bf16", "rope/rope");

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (symbol, mrope) in
            [("rope::qk_rmsnorm_rope_bf16", false), ("rope::qk_rmsnorm_mrope_bf16", true)]
        {
            let entry = module.entry(symbol).expect("the row resolved");
            for shape in ROPE_SHAPES {
                let ops = RopeOps::new(shape);

                // The rule's launch IS the launcher's, before anything fires.
                let launch = eval(Rule::RowsPackedHeadsNarrow, shape.dims()).expect("ported");
                assert_eq!(launch.grid, shape.grid(), "{symbol} at {shape:?}");
                assert_eq!(launch.block, [ROPE_BLOCK, 1, 1]);
                assert_eq!(launch.smem, 0, "`rope.cu:191` passes 0");

                ops.reset();
                let values = ops.values(shape, mrope);
                // SAFETY: every pointer addresses a live allocation of the
                // extent the row states and the null stream is live.
                unsafe { runtime::fire(symbol, shape.dims(), &values, Stream::NULL) }
                    .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
                synchronise(&format!("the shipped fire of {symbol} at {shape:?}"));
                let (row_q, row_k) = (ops.q.bytes(), ops.k.bytes());

                ops.reset();
                let mut cells = RopeCells::new(shape, &ops, mrope);
                let mut cells = cells.cells();
                // SAFETY: `entry` is from a module that outlives the call and
                // the cells are the kernel's declared parameters in order.
                unsafe {
                    raw_launch(
                        entry,
                        shape.grid(),
                        ROPE_BLOCK,
                        &mut cells,
                        &format!("rope.cu:190 by hand, {symbol} at {shape:?}"),
                    );
                }
                let (raw_q, raw_k) = (ops.q.bytes(), ops.k.bytes());

                // §18's guard, and it is not vacuous for an IN-PLACE kernel:
                // the seed is non-zero everywhere, so `written` counts what is
                // there and not what was written. What proves a launch
                // happened is that the result DIFFERS from the seed.
                let q_moved = differing(&raw_q, bytemuck_le(&ops.q_seed).as_slice());
                let k_moved = differing(&raw_k, bytemuck_le(&ops.k_seed).as_slice());
                assert!(
                    q_moved > 0 && k_moved > 0,
                    "{symbol} at {shape:?}: the launcher moved {q_moved} q bytes and {k_moved} \
                     k bytes — a bank nothing wrote makes the comparison below zeros against zeros"
                );
                assert_eq!(written(&raw_q), shape.q_elems(), "{symbol}: q is dense");
                assert_eq!(written(&raw_k), shape.k_elems(), "{symbol}: k is dense");

                let differs = differing(&row_q, &raw_q) + differing(&row_k, &raw_k);
                assert_eq!(
                    differs, 0,
                    "{symbol} at {shape:?}: the row and the launcher disagree on {differs} of \
                     {} bytes (rule grid {:?}, launcher grid {:?})",
                    row_q.len() + row_k.len(),
                    launch.grid,
                    shape.grid()
                );
                compared += row_q.len() + row_k.len();
                live_total += q_moved + k_moved;
            }
        }
        eprintln!(
            "RowsPackedHeadsNarrow via rope: 2 rows x {} shapes, {compared} bytes compared, \
             {live_total} bytes moved, 0 differing",
            ROPE_SHAPES.len()
        );
    }

    /// **The negative controls: one permutation, one truncation.**
    ///
    /// (a) THE PERMUTATION. The two norm weights swapped, at the launcher's
    /// own grid. Exactly the same blocks run, exactly the same cells are
    /// written, and both weight vectors are drawn from the same generator at
    /// different seeds — so the result has the same shape, the same density
    /// and the same order of magnitude, and different numbers in it. No count,
    /// no norm and no tolerance distinguishes it. It is also the control that
    /// matters most for a TEN-operand row: `q_weight` and `k_weight` are
    /// adjacent cells of the same type, and a row that transposed them would
    /// compile, fire, and be wrong.
    ///
    /// (b) THE TRUNCATION, and it is the refusal this file used to carry.
    /// `GatedRms` is `[rows, kv_heads]` — the head axis this module said no
    /// rule stated — and at the launcher's own block it opens `kv_heads`
    /// columns where the launcher opens `q_heads + kv_heads`. The blocks it
    /// does open all take `is_q` (because `kv_heads <= q_heads` at every shape
    /// here), so the KV bank is untouched entirely, which is the reading that
    /// makes it a truncation rather than a permutation and is why both are
    /// here.
    #[test]
    fn the_rope_controls_permute_and_truncate() {
        let Some(_) = arch_or_skip("the_rope_controls_permute_and_truncate") else { return };
        let module = module_of("rope::qk_rmsnorm_rope_bf16", "rope/rope");
        let symbol = "rope::qk_rmsnorm_rope_bf16";
        let entry = module.entry(symbol).expect("the row resolved");

        for shape in [ROPE_SHAPES[0], ROPE_SHAPES[1]] {
            let ops = RopeOps::new(shape);

            let fire_at = |grid: [u32; 3], swap: bool| {
                ops.reset();
                let mut cells = RopeCells::new(shape, &ops, false);
                if swap {
                    cells.pointers.swap(2, 3);
                }
                let mut cells = cells.cells();
                // SAFETY: as above; `grid` is only ever <= the launcher's.
                unsafe { raw_launch(entry, grid, ROPE_BLOCK, &mut cells, "a control") };
                (ops.q.bytes(), ops.k.bytes())
            };

            let (right_q, right_k) = fire_at(shape.grid(), false);

            // (a) the permutation.
            let (swap_q, swap_k) = fire_at(shape.grid(), true);
            assert_eq!(
                written(&swap_q),
                written(&right_q),
                "the weight swap changed how many q values are live — a truncation, not a \
                 permutation"
            );
            assert_eq!(written(&swap_k), written(&right_k), "and the same for k");
            let swap_differs = differing(&swap_q, &right_q) + differing(&swap_k, &right_k);
            assert!(
                swap_differs > 0,
                "the weight swap is invisible in the bytes at {shape:?}, so this control cannot \
                 fail"
            );

            // (b) the truncation `GatedRms` would have been.
            let near = [shape.tokens, shape.kv_heads, 1];
            assert_ne!(near, shape.grid(), "the near miss must not BE the launch");
            let (near_q, near_k) = fire_at(near, false);
            assert_eq!(
                differing(&near_k, bytemuck_le(&ops.k_seed).as_slice()),
                0,
                "`[rows, kv_heads]` opens only query lanes, so the KV bank must be untouched"
            );
            let near_differs = differing(&near_q, &right_q) + differing(&near_k, &right_k);
            assert!(near_differs > 0);

            eprintln!(
                "RowsPackedHeadsNarrow controls at {shape:?}:\n  \
                 q_weight/k_weight swapped at {:?} -> {swap_differs} of {} bytes differ (a \
                 PERMUTATION: {} values live on both)\n  \
                 GatedRms {near:?} vs {:?} -> {near_differs} bytes differ, KV bank untouched (a \
                 TRUNCATION)",
                shape.grid(),
                right_q.len() + right_k.len(),
                written(&right_q) + written(&right_k),
                shape.grid(),
            );
        }
    }

    /// **`_rounded`'s refusal, fired: the rule's grid writes where the
    /// launcher's does not.**
    ///
    /// `mod transcribed`'s twin of this test proves the two grids differ as
    /// arithmetic. This one proves the difference is not absorbed. The
    /// kernel's only guards are `blockIdx`-derived addressing — there is no
    /// `if (head_idx >= total) return;` anywhere in `qk_rmsnorm_rotate` — so
    /// the extra `kv_heads` columns the rule opens at a q-only site are blocks
    /// that RUN, and they address the k bank the launcher was handed as
    /// `nullptr`.
    ///
    /// A null `k` would fault, which proves nothing about how much is wrong,
    /// so this fires the same disagreement with a LIVE k buffer and counts the
    /// bytes. That count is the refusal's size: it is what a reader who
    /// overturned the refusal would have shipped.
    #[test]
    fn the_rounded_rope_disagreement_is_not_absorbed() {
        let Some(_) = arch_or_skip("the_rounded_rope_disagreement_is_not_absorbed") else {
            return;
        };
        let module = module_of("rope::qk_rmsnorm_rope_bf16", "rope/rope");
        let entry = module.entry("rope::qk_rmsnorm_rope_bf16").expect("the row resolved");

        // gemma-4's shared sliding layers, and a second ratio: hazard 1 says
        // one shape certifies nothing.
        for shape in
            [Rope { tokens: 6, q_heads: 8, kv_heads: 4, head_dim: 256 }, Rope { tokens: 3, q_heads: 16, kv_heads: 2, head_dim: 128 }]
        {
            let ops = RopeOps::new(shape);

            let fire_at = |grid: [u32; 3], kv: u32| {
                ops.reset();
                let mut cells = RopeCells::new(shape, &ops, false);
                // The launcher's OWN `num_kv_heads` at a q-only site is 0.
                cells.ints[1] = kv as i32;
                let mut cells = cells.cells();
                // SAFETY: the k bank is a live allocation at the full kv
                // extent, so even the excess blocks address memory this test
                // owns — which is the whole reason it is fired this way.
                unsafe { raw_launch(entry, grid, ROPE_BLOCK, &mut cells, "the q-only pair") };
                (ops.q.bytes(), ops.k.bytes())
            };

            // What `rope.cu:214` opens when the statement names one result:
            // `num_q_heads + 0`.
            let launcher = [shape.tokens, shape.q_heads, 1];
            let (launcher_q, launcher_k) = fire_at(launcher, 0);
            assert_eq!(
                differing(&launcher_k, bytemuck_le(&ops.k_seed).as_slice()),
                0,
                "the launcher's q-only grid must not touch the k bank"
            );

            // What `RowsPackedHeadsNarrow` opens: `Dims::q_heads +
            // Dims::kv_heads`, and `Dims::kv_heads` is the FIRE's.
            let rule = eval(Rule::RowsPackedHeadsNarrow, shape.dims()).expect("ported");
            assert_eq!(rule.grid, shape.grid());
            assert_ne!(rule.grid, launcher, "if these agreed `_rounded` would be a row");
            let (rule_q, rule_k) = fire_at(rule.grid, 0);

            let q_differs = differing(&rule_q, &launcher_q);
            let k_differs = differing(&rule_k, &launcher_k);
            assert_eq!(
                q_differs, 0,
                "the query bank is the launcher's either way — which is exactly why no output \
                 check would have caught this"
            );
            assert!(
                k_differs > 0,
                "the rule's extra {} columns wrote nothing at {shape:?}: this refusal would be \
                 stale",
                shape.kv_heads
            );
            eprintln!(
                "_rounded at {shape:?}: launcher grid {launcher:?} vs rule grid {:?} -> \
                 {q_differs} of {} query bytes differ, {k_differs} of {} KV bytes differ. \
                 The KV bank is `nullptr` at the real call site.",
                rule.grid,
                rule_q.len(),
                rule_k.len(),
            );
        }
    }

    /// The bf16 seed as bytes, little-endian, for comparison against a
    /// download.
    fn bytemuck_le(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|h| h.to_le_bytes()).collect()
    }

    // -----------------------------------------------------------------------
    // `Rule::PagedScoresDecode`'s THIRD launcher — the row it finally lands.
    // -----------------------------------------------------------------------

    /// `attn/dsv4_compress.cu:37`: `constexpr int ATTN_BLOCK = 128;`
    const ATTN_BLOCK: u32 = 128;

    /// A DSV4 compressed-paged decode rectangle.
    ///
    /// **`pages` is DERIVED and not a field, and that is a bug fix.** The
    /// first draft of this harness let the caller state a page count, and
    /// every shape it stated was too small: `compressed_attn_paged` reads
    /// `kv_page_indices[indptr[req] + pos / page_size]` for `pos` up to
    /// `(num_visible) * ratio - 1`, then DEREFERENCES the page it finds. A
    /// short table returns a garbage page and the load lands wherever that
    /// page happens to point.
    ///
    /// It was invisible in isolation and intermittent in parallel — three
    /// runs in twenty — because whether an arbitrary page index is a mapped
    /// address depends on what ELSE the process has allocated, which under
    /// libtest is another thread's buffers. That made it a defect of exactly
    /// the shape this file exists to catch: the signal was being decided by
    /// something other than the property under test, and the *later* tests
    /// paid, because `CUDA_ERROR_ILLEGAL_ADDRESS` is sticky and poisons the
    /// primary context for every thread that shares it.
    ///
    /// So the count is computed from the shape, once, with the arithmetic
    /// written down — a caller cannot get it wrong because a caller cannot
    /// state it.
    #[derive(Clone, Copy, Debug)]
    struct Compressed {
        tokens: u32,
        q_heads: u32,
        head_dim: u32,
        ratio: u32,
        page_size: u32,
    }

    impl Compressed {
        fn dims(self) -> Dims {
            Dims {
                rows: self.tokens,
                width: self.q_heads * self.head_dim,
                in_width: self.q_heads * self.head_dim,
                q_heads: self.q_heads,
                kv_heads: 1,
                head_dim: self.head_dim,
                stated_head_dim: 0,
                rotary_dims: 0,
                n_experts: 0,
                experts_per_token: 0,
                requests: 0,
                altup_streams: 0,
            }
        }

        /// `dsv4_compress.cu:319-320`, by hand.
        fn grid(self) -> [u32; 3] {
            [self.tokens, self.q_heads, 1]
        }

        /// `:321-322`, by hand: `(head_dim + ATTN_BLOCK) * sizeof(float)`.
        fn smem(self) -> u32 {
            (self.head_dim + ATTN_BLOCK) * 4
        }

        /// The largest ABSOLUTE position any block of any launch below reads.
        ///
        /// # This method exists because a fixed `pages: 3` was a real defect
        ///
        /// The first draft of this harness carried `pages` as a field of the
        /// shape and set it to 3, which is enough for the page table to be
        /// *indexable* and not enough for it to be *covering*. At
        /// `tokens = 16, ratio = 4, page_size = 8` the kernel resolves
        /// position 75, asks for page `75 / 8 = 9`, and reads entry 9 of a
        /// three-entry table. The value it gets back is whatever follows that
        /// allocation; `paged_slot` multiplies it by `page_size` and the
        /// kernel loads from the result.
        ///
        /// The bug that makes is not a wrong number. Sometimes that address
        /// is mapped — another thread's buffer — and the test passes with
        /// garbage inputs. Sometimes it is not, and the launch takes
        /// `CUDA_ERROR_ILLEGAL_ADDRESS`, which is **sticky on the primary
        /// context that every libtest thread shares**. It failed about one
        /// run in six, always in parallel and never under
        /// `--test-threads=1`, and when it failed it took eleven unrelated
        /// tests with it — reported as `rope/rope compiles: cuModuleLoadData
        /// failed with 700`, which points at the wrong file, the wrong test
        /// and the wrong kind of problem.
        ///
        /// Two things were true at once and only one of them was mine to
        /// find: the harness read off the end of a table, and the harness had
        /// no way to say which fire did it. Both are fixed here — the count
        /// is DERIVED below, an assertion in [`compressed_operands`] holds it,
        /// and every launch in this module now synchronises under a name.
        ///
        /// The obvious suspect was wrong, and worth recording: the
        /// deliberately-short `smem` control looked like the culprit and an
        /// A/B over 20 runs each way put the failure rate at 3/20 with it and
        /// 3/20 without. One run would have "confirmed" it.
        ///
        /// # The derivation
        ///
        /// `positions[t] = (t + 4) * ratio`, so
        /// `num_visible = (positions[t] + 1) / ratio = t + 4`, and entry `c`
        /// ends at `(c + 1) * ratio - 1` (`dsv4_compress.cuh:686` and the
        /// `paged_slot` call at `:715`). The largest `c` is `num_visible - 1`
        /// at the largest `qi`, and the largest `qi` any launch here opens is
        /// `max(tokens, q_heads) - 1` — the TRANSPOSED control puts `q_heads`
        /// on the first axis, so the reference shape alone does not bound it.
        fn max_pos(self) -> u32 {
            let rows = if self.tokens > self.q_heads { self.tokens } else { self.q_heads };
            (rows + 3) * self.ratio - 1
        }

        /// Pages enough that `pos / page_size` is always a page the table
        /// has, with one page of deliberate slack.
        ///
        /// The slack is the §22.5/§22.7 pattern and it is load-bearing here:
        /// an overrun that lands in ALLOCATED memory is a number this test
        /// can compare, and an overrun that lands outside the mapping is a
        /// fault that destroys every later test in the process. One page is
        /// enough because the index is bounded above by `max_pos`, and the
        /// assertion in [`compressed_operands`] is what keeps it enough.
        fn pages(self) -> u32 {
            self.max_pos() / self.page_size + 2
        }

        fn q_elems(self) -> usize {
            (self.tokens * self.q_heads * self.head_dim) as usize
        }
    }

    /// The compressed row's operands, allocated and uploaded.
    struct CompressedOps {
        _q: Buffer,
        _kv: Buffer,
        o: Buffer,
        lse: Buffer,
        _positions: Buffer,
        _indices: Buffer,
        _indptr: Buffer,
        _req: Buffer,
        pointers: Vec<u64>,
        values: Vec<ArgValue>,
    }

    fn compressed_operands(shape: Compressed) -> CompressedOps {
        // The page table must cover every position the kernel resolves, or
        // `paged_slot` returns a page nobody wrote and the load that follows
        // it is an arbitrary address. Asserted rather than commented: this
        // is the defect the `pages` doc above records, and an assertion is
        // the only form of it that survives the next shape being added.
        let pages = shape.pages();
        assert!(
            shape.max_pos() / shape.page_size < pages,
            "{shape:?}: position {} needs page {} and the table has {pages}",
            shape.max_pos(),
            shape.max_pos() / shape.page_size
        );

        let q = Buffer::of(&bf16_fill(shape.q_elems(), 0x0DD5_EED1));
        let kv = Buffer::of(&bf16_fill(
            (pages * shape.page_size * shape.head_dim) as usize,
            0x0DD5_EED2,
        ));
        let o = Buffer::zeroed(shape.q_elems() * 2);
        let lse = Buffer::zeroed((shape.tokens * shape.q_heads) as usize * 4);

        // Every token is late enough in its sequence that `num_visible =
        // (qpos + 1) / ratio` is at least two — a zero-visible token skips the
        // loop and writes a constant, which would make the comparison agree
        // for the wrong reason.
        let positions: Vec<i32> =
            (0..shape.tokens).map(|t| (t as i32 + 4) * shape.ratio as i32).collect();
        // A PERMUTED page table, so a kernel assuming identity is visible.
        let indices: Vec<u32> = (0..pages).rev().collect();
        let indptr: Vec<u32> = vec![0, pages];
        // One request, so `kv_page_indptr` is `[0, pages]` and every token
        // resolves through it.
        let req: Vec<i32> = vec![0; shape.tokens as usize];

        let positions = Buffer::of(&positions);
        let indices = Buffer::of(&indices);
        let indptr = Buffer::of(&indptr);
        let req = Buffer::of(&req);

        let pointers =
            vec![q.ptr, kv.ptr, o.ptr, lse.ptr, positions.ptr, indices.ptr, indptr.ptr, req.ptr];
        let values = vec![
            q.arg(),
            kv.arg(),
            o.arg(),
            lse.arg(),
            positions.arg(),
            indices.arg(),
            indptr.arg(),
            req.arg(),
            ArgValue::I32(shape.q_heads as i32),
            ArgValue::I32(shape.head_dim as i32),
            ArgValue::I32(shape.ratio as i32),
            ArgValue::I32(shape.page_size as i32),
            ArgValue::F32(COMPRESSED_SCALE),
        ];
        CompressedOps {
            _q: q,
            _kv: kv,
            o,
            lse,
            _positions: positions,
            _indices: indices,
            _indptr: indptr,
            _req: req,
            pointers,
            values,
        }
    }

    const COMPRESSED_SCALE: f32 = 0.088_388_35;

    /// The compressed row is stated, hosted, and claims `PagedScoresDecode`.
    #[test]
    fn the_compressed_paged_row_is_stated_and_hosted() {
        let symbol = "attn::attention_compressed_paged_bf16";
        assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
        let row = runtime::row(symbol).expect("hosted");
        assert_eq!(row.sig.launch, Rule::PagedScoresDecode);
        assert_eq!(row.sig.file, Some("attn/dsv4_compress.cuh"));
    }

    /// **The compressed row is byte-identical to `dsv4_compress.cu:318-323`.**
    ///
    /// Three shapes, and `head_dim` is what varies across them, because
    /// `head_dim` is what makes the SHARED MEMORY dynamic — the field
    /// `dsv4_compress.cuh:50-52` says no ported rule computes. A rule that
    /// fixed the allocation would be right at exactly one head dim and would
    /// not fault at the others: the kernel cuts `extern __shared__ float
    /// smem[]` into `[head_dim]` staged query values and `[ATTN_BLOCK]`
    /// reduction slots, and a partition that runs off the end reduces through
    /// another block's bytes rather than trapping.
    ///
    /// Both outputs are compared. `lse_out` is fp32 and written by thread 0 of
    /// each block only, so a block that ran with the wrong `blockIdx` but the
    /// right data shows in the LSE and not in `o`.
    #[test]
    fn the_compressed_paged_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_compressed_paged_row_reproduces_the_launcher") else {
            return;
        };
        let symbol = "attn::attention_compressed_paged_bf16";
        let module = module_of(symbol, "attn/dsv4_compress");
        let entry = module.entry(symbol).expect("the row resolved");

        let shapes = [
            // DeepSeek-style: 128-wide heads, compression ratio 4.
            Compressed { tokens: 4, q_heads: 8, head_dim: 128, ratio: 4, page_size: 8 },
            // HALF the head dim, so the smem is half — and `tokens` and
            // `q_heads` swap which is larger, so a transposed grid is a
            // different rectangle here than it is above.
            Compressed { tokens: 9, q_heads: 4, head_dim: 64, ratio: 2, page_size: 16 },
            // A head dim that is NOT a power of two and is larger than the
            // block, so the staging loop strides more than once and the smem
            // partition is ragged.
            Compressed { tokens: 3, q_heads: 6, head_dim: 192, ratio: 4, page_size: 4 },
        ];

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for shape in shapes {
            let ops = compressed_operands(shape);

            let launch = eval(Rule::PagedScoresDecode, shape.dims()).expect("ported");
            assert_eq!(launch.grid, shape.grid(), "{shape:?}");
            assert_eq!(launch.block, [ATTN_BLOCK, 1, 1]);
            assert_eq!(launch.smem, shape.smem(), "{shape:?}: (head_dim + 128) * 4");

            // SAFETY: every pointer addresses a live allocation of the extent
            // the row states and the null stream is live.
            unsafe { runtime::fire(symbol, shape.dims(), &ops.values, Stream::NULL) }
                .unwrap_or_else(|why| panic!("{symbol} would not fire at {shape:?}: {why}"));
            synchronise(&format!("the shipped fire of {symbol} at {shape:?}"));
            let (row_o, row_lse) = (ops.o.bytes(), ops.lse.bytes());

            ops.o.clear();
            ops.lse.clear();
            synchronise(&format!("clearing between the two launches at {shape:?}"));

            let mut pointers = ops.pointers.clone();
            let mut ints = [
                shape.q_heads as i32,
                shape.head_dim as i32,
                shape.ratio as i32,
                shape.page_size as i32,
            ];
            let mut scale = COMPRESSED_SCALE;
            let mut cells: Vec<*mut c_void> =
                pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for i in ints.iter_mut() {
                cells.push((&raw mut *i).cast());
            }
            cells.push((&raw mut scale).cast());
            // SAFETY: `entry` is from a module that outlives the call, the
            // cells are the kernel's declared parameters in order, and `smem`
            // is the launcher's own partition size.
            unsafe {
                raw_launch_smem(
                    entry,
                    shape.grid(),
                    ATTN_BLOCK,
                    shape.smem(),
                    &mut cells,
                    &format!("dsv4_compress.cu:319 by hand at {shape:?}"),
                );
            }
            let (raw_o, raw_lse) = (ops.o.bytes(), ops.lse.bytes());

            let live = written(&raw_o);
            assert_eq!(
                live,
                shape.q_elems(),
                "{shape:?}: the launcher wrote {live} of {} values, so the rest of this \
                 comparison is zeros against zeros",
                shape.q_elems()
            );
            assert!(raw_lse.chunks_exact(4).any(|w| w != [0, 0, 0, 0]), "the LSE is written");

            let differs = differing(&row_o, &raw_o) + differing(&row_lse, &raw_lse);
            assert_eq!(
                differs, 0,
                "{shape:?}: the row and the launcher disagree on {differs} of {} bytes \
                 (rule {:?}/{} smem, launcher {:?}/{} smem)",
                row_o.len() + row_lse.len(),
                launch.grid,
                launch.smem,
                shape.grid(),
                shape.smem()
            );
            compared += row_o.len() + row_lse.len();
            live_total += live;
        }
        eprintln!(
            "PagedScoresDecode via dsv4_compress: {} shapes, {compared} bytes compared, \
             {live_total} values written, 0 differing",
            shapes.len()
        );
    }

    /// **Three negative controls: a permutation, a truncation, and a silent
    /// shared-memory allocation.**
    ///
    /// (a) THE PERMUTATION. `positions` reversed, at the launcher's own grid.
    /// Every block runs, every output row is written, the density and the
    /// magnitude are identical — and `num_visible = (qpos + 1) / ratio` moves,
    /// so each token attends over a different prefix. Same count, different
    /// numbers: §21.14's failure mode exactly, and the one no tolerance sees.
    /// The `assert_eq!` on the two live counts is the claim, not decoration.
    ///
    /// (b) THE TRUNCATION, and its measured shape is worth stating because the
    /// first draft of this test asserted it was a permutation and it is not.
    /// `[q_heads, tokens]` opens the same number of BLOCKS, but the row index
    /// is `qi * num_q_heads + q_head` and the transpose feeds `q_head` a range
    /// of `tokens`, so the image is `{qi * q_heads + t}` — overlapping, and
    /// short. At `16x8` it writes 9 216 of 16 384 values. A grid that opens
    /// the right number of blocks against the wrong cells is not automatically
    /// a permutation of the output, and assuming it is would have made this
    /// control weaker than it reads.
    ///
    /// (c) THE SMEM, in BOTH directions, and the conclusion is that this
    /// field is not output-observable at all.
    ///
    /// *Too much* is the probe that is provably legal: `smem + 4096` is a
    /// launch with defined behaviour under every CUDA rule, the kernel reads
    /// and writes only the `(head_dim + ATTN_BLOCK)` floats it partitions,
    /// and the output is byte-identical. So a rule that OVER-allocated could
    /// not be caught by any fire — established without leaning on anything
    /// undefined.
    ///
    /// *Too little* is the direction a wrong rule would actually produce —
    /// `ATTN_BLOCK * 4`, what a rule that never read `head_dim` would ask for,
    /// half of what `head_dim = 128` needs. The launch succeeds, hardware
    /// reports success, and the output is byte-identical here too.
    ///
    /// **An earlier draft of this comment said the hardware "absorbs" it, and
    /// that is a claim this test cannot make.** Writing past a block's
    /// declared dynamic allocation is undefined; the only reason the numbers
    /// come back right is that nothing else happened to be using those
    /// shared bytes. What that draft was really recording was one draw from a
    /// distribution, written down as a property — the same mistake in kind as
    /// the §22.7 near miss that was byte-identical at one shape and 10 221 of
    /// 20 480 at another. It is stated as an observation now, with 45
    /// consecutive parallel runs behind it and a `synchronise` that will name
    /// this launch by shape and by requested size if it ever stops holding.
    ///
    /// Either way the conclusion is the same and it is the useful one:
    /// `smem` is the one field of this rule **no fire can check**. That is not
    /// a reason to skip it; it is the reason `mod transcribed` pins
    /// `dsv4_compress.cu:322` by text and line, and the reason this row's
    /// evidence is a pin AND a fire rather than either alone.
    #[test]
    fn the_compressed_paged_controls_permute_and_the_wrong_smem_is_silent() {
        let Some(_) =
            arch_or_skip("the_compressed_paged_controls_permute_and_the_wrong_smem_is_silent")
        else {
            return;
        };
        let symbol = "attn::attention_compressed_paged_bf16";
        let module = module_of(symbol, "attn/dsv4_compress");
        let entry = module.entry(symbol).expect("the row resolved");

        // `tokens >= q_heads` at both, so the transpose stays in bounds.
        for shape in [
            Compressed { tokens: 16, q_heads: 8, head_dim: 128, ratio: 4, page_size: 8 },
            Compressed { tokens: 9, q_heads: 4, head_dim: 64, ratio: 2, page_size: 16 },
        ] {
            let ops = compressed_operands(shape);

            // The reversed position table the permutation control fires with.
            let flipped: Vec<i32> = (0..shape.tokens)
                .rev()
                .map(|t| (t as i32 + 4) * shape.ratio as i32)
                .collect();
            let flipped = Buffer::of(&flipped);

            let fire_at = |grid: [u32; 3], smem: u32, positions: u64| {
                ops.o.clear();
                ops.lse.clear();
                synchronise("clearing the compressed output between controls");
                let mut pointers = ops.pointers.clone();
                pointers[4] = positions;
                let mut ints = [
                    shape.q_heads as i32,
                    shape.head_dim as i32,
                    shape.ratio as i32,
                    shape.page_size as i32,
                ];
                let mut scale = COMPRESSED_SCALE;
                let mut cells: Vec<*mut c_void> =
                    pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
                for i in ints.iter_mut() {
                    cells.push((&raw mut *i).cast());
                }
                cells.push((&raw mut scale).cast());
                // SAFETY: as the test above — `grid` is bounded by the
                // buffers and the page table covers every position any of
                // these grids resolves, which `compressed_operands` asserts.
                //
                // `smem` is NOT covered by that sentence. One of the calls
                // below deliberately requests less than the kernel's own
                // partition, which is undefined behaviour; it is measured,
                // not assumed, and the doc comment above carries the
                // measurement and what to do when it stops holding.
                let what = format!(
                    "compressed_paged control at {shape:?}, grid {grid:?}, smem {smem}"
                );
                unsafe { raw_launch_smem(entry, grid, ATTN_BLOCK, smem, &mut cells, &what) };
                (ops.o.bytes(), ops.lse.bytes())
            };

            let (right_o, right_lse) =
                fire_at(shape.grid(), shape.smem(), ops.pointers[4]);
            let right_live = written(&right_o);
            assert_eq!(right_live, shape.q_elems(), "the reference launch is dense");

            // (a) the permutation.
            let (perm_o, perm_lse) = fire_at(shape.grid(), shape.smem(), flipped.ptr);
            assert_eq!(
                written(&perm_o),
                right_live,
                "the reversed position table moved {} values to {} — a truncation, not a \
                 permutation",
                right_live,
                written(&perm_o)
            );
            let perm_differs = differing(&perm_o, &right_o) + differing(&perm_lse, &right_lse);
            assert!(perm_differs > 0, "the position permutation is invisible at {shape:?}");

            // (b) the transpose, which is a truncation and is measured as one.
            let flip = [shape.q_heads, shape.tokens, 1];
            assert_ne!(flip, shape.grid(), "the transpose must not BE the launch");
            let (flip_o, flip_lse) = fire_at(flip, shape.smem(), ops.pointers[4]);
            let flip_live = written(&flip_o);
            assert!(
                flip_live < right_live,
                "the transpose wrote {flip_live} of {right_live} — if it were dense this would \
                 be a permutation and the comment above is wrong"
            );
            let flip_differs = differing(&flip_o, &right_o) + differing(&flip_lse, &right_lse);
            assert!(flip_differs > 0, "the transpose is invisible in the bytes at {shape:?}");

            // (c) the smem a fixed allocation would have asked for — and the
            //     measurement this test exists to record.
            let (fat_o, fat_lse) = fire_at(shape.grid(), shape.smem() + 4096, ops.pointers[4]);
            let fat_differs = differing(&fat_o, &right_o) + differing(&fat_lse, &right_lse);
            assert_eq!(
                fat_differs, 0,
                "{shape:?}: 4 096 bytes of EXCESS shared memory changed {fat_differs} bytes. \
                 Over-allocation is defined behaviour and the kernel touches only its own \
                 partition, so this must be zero; if it is not, the harness is feeding a \
                 different launch and not a different allocation."
            );

            let (thin_o, thin_lse) = fire_at(shape.grid(), ATTN_BLOCK * 4, ops.pointers[4]);
            let thin_differs = differing(&thin_o, &right_o) + differing(&thin_lse, &right_lse);
            assert_eq!(
                thin_differs, 0,
                "a launch {} bytes short of the kernel's own partition produced \
                 {thin_differs} differing bytes at {shape:?}. It was 0 across 45 consecutive \
                 parallel runs when this was written, and this assertion is an OBSERVATION \
                 rather than a guarantee — the launch is undefined behaviour and the paragraph \
                 above says so. If it has started differing, do not weaken this to a range: \
                 delete the launch and keep the over-allocation probe, which proves the same \
                 thing legally.",
                shape.smem() - ATTN_BLOCK * 4
            );

            eprintln!(
                "PagedScoresDecode controls at {shape:?}:\n  \
                 positions reversed at {:?} -> {perm_differs} of {} bytes differ (a \
                 PERMUTATION: {right_live} values live on both)\n  \
                 transposed grid {flip:?} -> {flip_live} of {right_live} values, \
                 {flip_differs} bytes differ (a TRUNCATION)\n  \
                 smem {} (short, undefined) and {} (excess, legal) vs the launcher's {} at \
                 the right grid -> {thin_differs} and {fat_differs} bytes differ. \
                 BYTE-IDENTICAL BOTH WAYS: `smem` is not output-observable in either \
                 direction, so it is checkable only against the source.",
                shape.grid(),
                right_o.len() + right_lse.len(),
                ATTN_BLOCK * 4,
                shape.smem() + 4096,
                shape.smem(),
            );
        }
    }

    // -----------------------------------------------------------------------
    // `Rule::PerRow`, via `attn::write_kv_explicit_bf16_devwin`
    // -----------------------------------------------------------------------

    /// `kv_paged.cu:282` writes `constexpr int BLOCK = 256;`.
    const DEVWIN_BLOCK: u32 = 256;

    /// A device-window KV write: `n_max` lanes into a `pages x page_size`
    /// arena of `h_kv` heads by `d` channels.
    #[derive(Clone, Copy, Debug)]
    struct Devwin {
        n_max: u32,
        h_kv: u32,
        d: u32,
        page_size: u32,
        pages: u32,
    }

    impl Devwin {
        /// `PerRow` reads `rows` and NOTHING else, so the other eleven fields
        /// are filled honestly rather than zeroed: a rule that started reading
        /// `head_dim` here must fail this test rather than pass it by seeing a
        /// zero it would have refused.
        fn dims(self) -> Dims {
            Dims {
                rows: self.n_max,
                width: self.h_kv * self.d,
                in_width: self.h_kv * self.d,
                q_heads: self.h_kv,
                kv_heads: self.h_kv,
                head_dim: self.d,
                stated_head_dim: 0,
                rotary_dims: 0,
                n_experts: 0,
                experts_per_token: 0,
                requests: self.n_max,
                altup_streams: 0,
            }
        }

        /// `kv_paged.cu:284` and `:292`, by hand: `<<<n_max, BLOCK, 0, s>>>`.
        fn grid(self) -> [u32; 3] {
            [self.n_max, 1, 1]
        }

        /// One lane of `k_curr`, in elements.
        fn row(self) -> usize {
            (self.h_kv * self.d) as usize
        }

        fn curr_elems(self) -> usize {
            self.n_max as usize * self.row()
        }

        fn arena_elems(self) -> usize {
            (self.pages * self.page_size) as usize * self.row()
        }
    }

    /// Everything a `write_kv_explicit_devwin` fire needs, allocated.
    struct DevwinOps {
        k_curr: Buffer,
        v_curr: Buffer,
        k_pages: Buffer,
        v_pages: Buffer,
        w_page: Buffer,
        w_off: Buffer,
        row_valid: Buffer,
        win: Buffer,
    }

    impl DevwinOps {
        /// The page/offset map is a PERMUTATION of distinct arena cells, not
        /// an identity, so a launch that dropped `w_page` or `w_off` — or read
        /// one for the other — lands somewhere else rather than somewhere
        /// coincidentally right.
        fn new(shape: Devwin) -> Self {
            let cells = (shape.pages * shape.page_size) as usize;
            let (mut page, mut off) = (Vec::new(), Vec::new());
            for b in 0..shape.n_max as usize {
                // 5 is coprime to every `cells` used below, so `b -> 5b + 1`
                // is injective over the lanes.
                let cell = (5 * b + 1) % cells;
                page.push(u32::try_from(cell / shape.page_size as usize).expect("small"));
                off.push(u32::try_from(cell % shape.page_size as usize).expect("small"));
            }
            Self {
                k_curr: Buffer::of(&bf16_fill(shape.curr_elems(), 0x51DE_0001)),
                v_curr: Buffer::of(&bf16_fill(shape.curr_elems(), 0x51DE_0002)),
                k_pages: Buffer::zeroed(shape.arena_elems() * 2),
                v_pages: Buffer::zeroed(shape.arena_elems() * 2),
                w_page: Buffer::of(&page),
                w_off: Buffer::of(&off),
                row_valid: Buffer::of(&vec![1u8; shape.n_max as usize]),
                win: Buffer::of(&[0u32, shape.n_max]),
            }
        }

        /// The row's thirteen operands: the kernel's twelve, then the flag.
        fn values(&self, shape: Devwin, hnd: bool) -> Vec<ArgValue> {
            vec![
                self.k_curr.arg(),
                self.v_curr.arg(),
                self.k_pages.arg(),
                self.v_pages.arg(),
                self.w_page.arg(),
                self.w_off.arg(),
                self.row_valid.arg(),
                self.win.arg(),
                ArgValue::I32(shape.n_max as i32),
                ArgValue::I32(shape.page_size as i32),
                ArgValue::I32(shape.h_kv as i32),
                ArgValue::I32(shape.d as i32),
                ArgValue::Bool(hnd),
            ]
        }

        fn clear(&self) {
            self.k_pages.clear();
            self.v_pages.clear();
            synchronise("clearing the devwin page arena between launches");
        }
    }

    /// The kernel's twelve cells, owned, in its declared order.
    struct DevwinCells {
        pointers: [u64; 8],
        ints: [i32; 4],
    }

    impl DevwinCells {
        fn new(shape: Devwin, ops: &DevwinOps) -> Self {
            Self {
                pointers: [
                    ops.k_curr.ptr,
                    ops.v_curr.ptr,
                    ops.k_pages.ptr,
                    ops.v_pages.ptr,
                    ops.w_page.ptr,
                    ops.w_off.ptr,
                    ops.row_valid.ptr,
                    ops.win.ptr,
                ],
                ints: [
                    shape.n_max as i32,
                    shape.page_size as i32,
                    shape.h_kv as i32,
                    shape.d as i32,
                ],
            }
        }

        fn cells(&mut self) -> Vec<*mut c_void> {
            let mut cells: Vec<*mut c_void> =
                self.pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
            for i in self.ints.iter_mut() {
                cells.push((&raw mut *i).cast());
            }
            cells
        }
    }

    /// The shapes every devwin comparison below runs at.
    ///
    /// Hazard 1 is why there are three. The two layouts differ only in where a
    /// lane's `(h, j)` lands inside a cell, so at `h_kv = 1` they are the SAME
    /// address for every element and the arm control would measure nothing —
    /// which is exactly `AltUpStreams`' near miss in a different coordinate.
    /// So `h_kv` is 4, 2 and 1: the first two make the flip visible, and the
    /// third is here to be the shape where it is not, stated rather than
    /// avoided.
    const DEVWIN_SHAPES: [Devwin; 3] = [
        Devwin { n_max: 6, h_kv: 4, d: 64, page_size: 8, pages: 3 },
        Devwin { n_max: 3, h_kv: 2, d: 128, page_size: 4, pages: 2 },
        Devwin { n_max: 5, h_kv: 1, d: 32, page_size: 8, pages: 2 },
    ];

    /// The devwin row and both of its arms are stated, hosted, and claim
    /// `PerRow`.
    #[test]
    fn the_devwin_rows_are_stated_and_hosted() {
        for symbol in [
            "attn::write_kv_explicit_bf16_devwin",
            "attn::write_kv_explicit_bf16_devwin#hnd",
            "attn::write_kv_explicit_bf16_devwin#nhd",
        ] {
            assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
            let row = runtime::row(symbol).expect("hosted");
            assert_eq!(row.sig.launch, Rule::PerRow, "{symbol}");
            assert_eq!(row.sig.file, Some("attn/kv_paged.cuh"), "{symbol}");
        }
        // The name the model text actually states — `dsl.rs:3415`. The rows
        // spelled `..._devwin_bf16` before this session and no statement could
        // ever have reached them.
        assert!(
            runtime::hosts("attn::write_kv_explicit_bf16_devwin"),
            "the AOT symbol is the JIT symbol"
        );
        assert!(
            !runtime::hosts("attn::write_kv_explicit_devwin_bf16"),
            "the transposed spelling must not survive as a second row"
        );
    }

    /// **The devwin row is byte-identical to `kv_paged.cu:284` and `:292`.**
    ///
    /// Three shapes, both arms, both banks. The row carries a THIRTEENTH
    /// operand the kernel has no parameter for — the layout flag — and
    /// `WRITE_KV_EXPLICIT_DEVWIN`'s two `Term::Is` arms consume it, so what
    /// this compares is a thirteen-cell fire against a twelve-cell
    /// `cuLaunchKernel`: if `TAKE_12` did not drop the flag, the flag's byte
    /// would arrive in the kernel's frame and every address after it would
    /// move.
    ///
    /// The grid is the claim that took the work. `n_max` is the fire's FULL
    /// row count and `Dims::rows` is the BOUND REGION's, and they are equal
    /// only because this kernel is `whole` — `model-compiler/src/kernels.rs:112`
    /// refuses a `whole` kernel inside a `Peel` statically and
    /// `lower.rs:1064-1073` refuses any window but `[0, rows.len())`
    /// dynamically. This test fires at the whole window because that is the
    /// only window that exists; `the_devwin_controls_permute_and_the_window_is_
    /// device_side` fires a narrower one to show what the guard is guarding.
    #[test]
    fn the_devwin_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_devwin_row_reproduces_the_launcher") else { return };
        let module = module_of("attn::write_kv_explicit_bf16_devwin", "attn/kv_paged");

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for shape in DEVWIN_SHAPES {
            for (hnd, arm) in [(true, "#hnd"), (false, "#nhd")] {
                let ops = DevwinOps::new(shape);
                let entry = module
                    .entry(&format!("attn::write_kv_explicit_bf16_devwin{arm}"))
                    .expect("the arm's row resolved");

                let launch = eval(Rule::PerRow, shape.dims()).expect("ported");
                assert_eq!(launch.grid, shape.grid(), "{shape:?}");
                assert_eq!(launch.block, [DEVWIN_BLOCK, 1, 1]);
                assert_eq!(launch.smem, 0, "`kv_paged.cu:284` passes 0");

                ops.clear();
                let values = ops.values(shape, hnd);
                // SAFETY: every pointer addresses a live allocation of the
                // extent the row states and the null stream is live.
                unsafe {
                    runtime::fire(
                        "attn::write_kv_explicit_bf16_devwin",
                        shape.dims(),
                        &values,
                        Stream::NULL,
                    )
                }
                .unwrap_or_else(|why| panic!("the devwin row would not fire at {shape:?}: {why}"));
                synchronise(&format!(
                    "the shipped fire of attn::write_kv_explicit_bf16_devwin at {shape:?} {arm}"
                ));
                let (row_k, row_v) = (ops.k_pages.bytes(), ops.v_pages.bytes());

                ops.clear();
                let mut cells = DevwinCells::new(shape, &ops);
                let mut cells = cells.cells();
                // SAFETY: `entry` is from a module that outlives the call and
                // the cells are the kernel's twelve declared parameters.
                unsafe {
                    raw_launch(
                        entry,
                        shape.grid(),
                        DEVWIN_BLOCK,
                        &mut cells,
                        &format!("kv_paged.cu:284 by hand at {shape:?} {arm}"),
                    );
                }
                let (raw_k, raw_v) = (ops.k_pages.bytes(), ops.v_pages.bytes());

                // §18's guard: the arena starts at zero, so "0 differing"
                // could otherwise mean "neither launch wrote".
                let live = written(&raw_k) + written(&raw_v);
                assert_eq!(
                    live,
                    2 * shape.curr_elems(),
                    "{shape:?} {arm}: the launcher wrote {live} values, not the \
                     {} the lane map covers",
                    2 * shape.curr_elems()
                );

                let differs = differing(&row_k, &raw_k) + differing(&row_v, &raw_v);
                assert_eq!(
                    differs, 0,
                    "{shape:?} {arm}: the row and the launcher disagree on {differs} of {} \
                     bytes (rule grid {:?}, launcher grid {:?})",
                    row_k.len() + row_v.len(),
                    launch.grid,
                    shape.grid()
                );
                compared += row_k.len() + row_v.len();
                live_total += live;
            }
        }
        eprintln!(
            "PerRow via write_kv_explicit_devwin: {} shapes x 2 arms, {compared} bytes \
             compared, {live_total} values written, 0 differing",
            DEVWIN_SHAPES.len()
        );
    }

    /// **Three negative controls: the wrong arm, a permuted lane map, and the
    /// device-side window.**
    ///
    /// (a) THE WRONG ARM, which is a PERMUTATION. `<false>` where the flag says
    /// `true` writes exactly the same VALUES into exactly as many cells — each
    /// lane's `h_kv * d` elements land once — and only the address of each
    /// element inside the arena moves. Same count, same multiset, same sums,
    /// same norms. At `h_kv = 1` the two layouts collapse to the same address
    /// and the control measures nothing, which is why `DEVWIN_SHAPES` carries
    /// three shapes and this test asserts the collapse rather than hiding it.
    ///
    /// (b) THE LANE MAP, and it is the operand a rule cannot see. `w_off`
    /// rotated by one is again a permutation of the same distinct arena cells,
    /// so the geometry is untouched and only the destination moves. A row that
    /// bound `w_page` and `w_off` in the wrong order, or dropped one, would be
    /// invisible to a grid check and visible here.
    ///
    /// (c) THE WINDOW, which is why this kernel exists. `win = {1, len - 2}` is
    /// a narrower device-side window at the SAME grid: every block still runs
    /// and the ones outside the window return before writing. That is the
    /// launcher's own contract — `kv_paged.cuh:757`, *"the launch shape is the
    /// FULL lane count and rows outside the window early-out"* — and it is the
    /// measured reason `PerRow` may read `Dims::rows` at all. If the grid
    /// tracked the WINDOW rather than the lane count, this launch and the full
    /// one would differ in the grid; they do not, and the difference is
    /// entirely in what the kernel declined to write.
    #[test]
    fn the_devwin_controls_permute_and_the_window_is_device_side() {
        let Some(_) = arch_or_skip("the_devwin_controls_permute_and_the_window_is_device_side")
        else {
            return;
        };
        let module = module_of("attn::write_kv_explicit_bf16_devwin", "attn/kv_paged");
        let hnd = module.entry("attn::write_kv_explicit_bf16_devwin#hnd").expect("resolved");
        let nhd = module.entry("attn::write_kv_explicit_bf16_devwin#nhd").expect("resolved");

        for shape in DEVWIN_SHAPES {
            let ops = DevwinOps::new(shape);
            let fire_at = |entry: dr::CUfunction, w_off: u64, win: u64| {
                ops.clear();
                let mut cells = DevwinCells::new(shape, &ops);
                cells.pointers[5] = w_off;
                cells.pointers[7] = win;
                let mut cells = cells.cells();
                // SAFETY: the substituted cells address live allocations of
                // the same extents as the ones they replace.
                let what = format!("write_kv_explicit_devwin control at {shape:?}");
                unsafe { raw_launch(entry, shape.grid(), DEVWIN_BLOCK, &mut cells, &what) };
                (ops.k_pages.bytes(), ops.v_pages.bytes())
            };

            let (right_k, right_v) = fire_at(hnd, ops.w_off.ptr, ops.win.ptr);
            let right_live = written(&right_k) + written(&right_v);
            assert_eq!(right_live, 2 * shape.curr_elems(), "the reference launch is dense");

            // (a) the wrong arm.
            let (flip_k, flip_v) = fire_at(nhd, ops.w_off.ptr, ops.win.ptr);
            assert_eq!(
                written(&flip_k) + written(&flip_v),
                right_live,
                "{shape:?}: the layout flip moved {right_live} values to {} — a truncation, not \
                 a permutation",
                written(&flip_k) + written(&flip_v)
            );
            let flip_differs = differing(&flip_k, &right_k) + differing(&flip_v, &right_v);
            if shape.h_kv == 1 {
                assert_eq!(
                    flip_differs, 0,
                    "{shape:?}: at ONE kv head the two layouts are the same address and this \
                     control is expected to measure nothing — if it now differs, the arithmetic \
                     in the comment above is wrong"
                );
            } else {
                assert!(
                    flip_differs > 0,
                    "{shape:?}: the layout flip is invisible in the bytes, so the arm this row \
                     picks is unmeasured at this shape"
                );
            }

            // (b) the lane map.
            let rotated: Vec<u32> = {
                let mut v: Vec<u32> = (0..shape.n_max as usize)
                    .map(|b| {
                        let cells = (shape.pages * shape.page_size) as usize;
                        u32::try_from(((5 * b + 1) % cells) % shape.page_size as usize)
                            .expect("small")
                    })
                    .collect();
                v.rotate_left(1);
                v
            };
            let rotated = Buffer::of(&rotated);
            let (map_k, map_v) = fire_at(hnd, rotated.ptr, ops.win.ptr);
            let map_differs = differing(&map_k, &right_k) + differing(&map_v, &right_v);
            assert!(
                map_differs > 0,
                "{shape:?}: rotating `w_off` produced the same arena, so the lane map is \
                 unmeasured"
            );

            // (c) the device-side window.
            let narrow = Buffer::of(&[1u32, shape.n_max - 2]);
            let (win_k, win_v) = fire_at(hnd, ops.w_off.ptr, narrow.ptr);
            let win_live = written(&win_k) + written(&win_v);
            assert_eq!(
                win_live,
                2 * (shape.n_max as usize - 2) * shape.row(),
                "{shape:?}: a window of {} lanes wrote {win_live} values",
                shape.n_max - 2
            );
            let win_differs = differing(&win_k, &right_k) + differing(&win_v, &right_v);
            assert!(
                win_differs > 0,
                "{shape:?}: the narrowed window wrote the same arena as the full one"
            );

            eprintln!(
                "PerRow devwin controls at {shape:?}, all at the launcher's grid {:?}:\n  \
                 layout flip -> {flip_differs} of {} bytes differ (a PERMUTATION: \
                 {right_live} values live on both)\n  \
                 `w_off` rotated -> {map_differs} bytes differ, {right_live} values still \
                 live (a PERMUTATION)\n  \
                 win {{1, {}}} vs {{0, {}}} -> {win_live} of {right_live} values written, \
                 {win_differs} bytes differ, SAME grid (the window is device-side)",
                shape.grid(),
                right_k.len() + right_v.len(),
                shape.n_max - 2,
                shape.n_max,
            );
        }
    }

    // -----------------------------------------------------------------------
    // `Rule::Single`, `Rule::SingleWarp` and `Rule::PerRequest`
    // -----------------------------------------------------------------------
    //
    // The three rules whose grid is not a function of the rectangle at all,
    // or is a function of the WRONG axis if read carelessly. All three carry
    // the same hazard and it is hazard 1's exactly: a `1` and a `ceil(rows /
    // 256)` are the same number for every fixture under 257 rows, and a
    // request count and a row count are the same number for every decode. So
    // every fire below runs at TWO shapes chosen to make the near miss
    // arithmetically visible, and asserts the near miss's grid explicitly.

    /// The `Single` rows, hosted, at the rules they claim.
    #[test]
    fn the_literal_grid_rows_are_stated_and_hosted() {
        for (symbol, rule, file) in [
            ("layout::copy_if_valid_slot", Rule::Single, "layout/slot_ops.cuh"),
            ("attn::build_window_page_view", Rule::Single, "attn/kv_paged.cuh"),
            ("attn::build_full_split_view", Rule::SingleWarp, "attn/kv_paged.cuh"),
            (
                "attn::mtp_update_pending_hidden_bf16",
                Rule::PerRequest,
                "attn/attention_naive.cuh",
            ),
        ] {
            assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
            let row = runtime::row(symbol).expect("hosted");
            assert_eq!(row.sig.launch, rule, "{symbol}");
            assert_eq!(row.sig.file, Some(file), "{symbol}");
        }

        // The two that KEEP `PerRow` against the same launcher text, asserted
        // rather than left to prose. `runtime::launch::per_request` and
        // `families::attn::PAGE_COMPACT` both argue these should not move;
        // this is the assertion that notices if someone moves them.
        for symbol in ["attn::count_kept", "attn::scan_and_scatter"] {
            let row = runtime::row(symbol).expect("hosted");
            assert_eq!(
                row.sig.launch,
                Rule::PerRow,
                "{symbol} has been moved to `PerRequest`. Its launcher is \
                 `<<<num_requests, kBlock>>>`, so that looks right and is not: \
                 `dsl::cuda::compact_page_csr` records `Shape([Dim::Requests])`, \
                 so `Dims::rows` IS the request count for this op's fire, while \
                 `Dims::requests` is filled from the ATTENTION context and is \
                 ZERO without one. The move trades a number that is always \
                 right for one that is usually absent."
            );
        }
    }

    /// **`Single` is a literal 1, and `RowsFlat` is not — at 300 rows.**
    ///
    /// `layout/slot_ops.cu:61` is `<<<1, kThreads>>>` with `kThreads = 256`,
    /// and the kernel strides `bytes` inside the one block. Two shapes, and
    /// the second is the whole point: at 200 rows `ceil(200 / 256)` is 1 and
    /// the near miss is INVISIBLE; at 300 rows it is 2, and a second block
    /// re-copies every byte the first copied.
    ///
    /// That second block is idempotent — this is a byte copy — so the near
    /// miss cannot be caught by comparing OUTPUTS at all. It is caught by
    /// comparing GRIDS, which this test does explicitly, and that is the
    /// honest limit of what a fire proves here: the row is byte-identical to
    /// its launcher, and the rule it does not state would have been too.
    #[test]
    fn the_single_row_reproduces_the_slot_copy_launcher() {
        let Some(_) = arch_or_skip("the_single_row_reproduces_the_slot_copy_launcher") else {
            return;
        };
        let module = module_of("layout::copy_if_valid_slot", "layout/slot_ops");
        let entry = module.entry("layout::copy_if_valid_slot").expect("the row resolved");

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (rows, bytes) in [(200u32, 4096usize), (300u32, 9000usize)] {
            let payload: Vec<u8> =
                bf16_fill(bytes.div_ceil(2), 0x5107_5C09).iter().flat_map(|v| v.to_le_bytes()).collect();
            let src = Buffer::of(&payload[..bytes]);
            let dst = Buffer::zeroed(bytes);
            // Request 2 of four slots, and the slot is VALID. Slot 1 is
            // negative so the guard has something to be true about in the
            // control below.
            let slots: Vec<u8> = [7i32, -1, 3, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
            let slot_ids = Buffer::of(&slots);

            let d = dims(rows, 128);
            let launch = eval(Rule::Single, d).expect("ported");
            assert_eq!(launch.grid, [1, 1, 1], "the grid is a literal, at {rows} rows");
            assert_eq!(launch.block, [256, 1, 1], "`slot_ops.cu:60` fixes 256");
            assert_eq!(launch.smem, 0, "`slot_ops.cu:61` passes 0");

            // THE NEAR MISS, stated at both shapes. This is the assertion the
            // output comparison cannot make.
            let flat = eval(Rule::RowsFlat, d).expect("ported").grid;
            assert_eq!(
                flat,
                [rows.div_ceil(256), 1, 1],
                "`RowsFlat` at {rows} rows"
            );
            if rows <= 256 {
                assert_eq!(flat, launch.grid, "at {rows} rows the near miss is INVISIBLE");
            } else {
                assert_ne!(flat, launch.grid, "at {rows} rows the near miss is 2 blocks");
            }

            let values = [
                src.arg(),
                dst.arg(),
                ArgValue::Usize(bytes),
                slot_ids.arg(),
                ArgValue::Usize(2),
            ];
            // SAFETY: every pointer addresses a live allocation at least
            // `bytes` long and the null stream is live.
            unsafe { runtime::fire("layout::copy_if_valid_slot", d, &values, Stream::NULL) }
                .unwrap_or_else(|why| panic!("the slot copy would not fire at {rows}: {why}"));
            synchronise(&format!(
                "the shipped fire of layout::copy_if_valid_slot at rows={rows} bytes={bytes}"
            ));
            let row_out = dst.bytes();

            dst.clear();
            let (mut p_src, mut p_dst, mut p_ids) = (src.ptr, dst.ptr, slot_ids.ptr);
            let (mut n, mut req) = (bytes, 2usize);
            let mut cells: Vec<*mut c_void> = vec![
                (&raw mut p_src).cast(),
                (&raw mut p_dst).cast(),
                (&raw mut n).cast(),
                (&raw mut p_ids).cast(),
                (&raw mut req).cast(),
            ];
            // SAFETY: `entry` outlives the call and the cells are the
            // kernel's five declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    [1, 1, 1],
                    256,
                    &mut cells,
                    &format!("slot_ops.cu:61 by hand at bytes={bytes}"),
                );
            }
            let raw_out = dst.bytes();

            let live = raw_out.iter().filter(|b| **b != 0).count();
            assert!(live > 0, "the launcher wrote nothing at bytes={bytes}");
            let differs = differing(&row_out, &raw_out);
            assert_eq!(
                differs, 0,
                "rows={rows} bytes={bytes}: the row and the launcher disagree on \
                 {differs} of {bytes} bytes (rule grid {:?})",
                launch.grid
            );
            compared += bytes;
            live_total += live;
        }
        eprintln!(
            "Single via layout::copy_if_valid_slot: 2 shapes, {compared} bytes compared, \
             {live_total} non-zero, 0 differing"
        );
    }

    /// **The `Single` control: the guard, and the block width.**
    ///
    /// (a) THE GUARD. `slot_ids[request] < 0` and the kernel writes nothing.
    /// The row and the launcher must BOTH write nothing, and the assertion is
    /// that the destination is still all zeroes — a `written > 0` guard read
    /// backwards, and the only negative control this kernel admits that is
    /// not a permutation, because a byte copy has no permutation: every
    /// destination byte has exactly one source.
    ///
    /// (b) THE BLOCK WIDTH, which is what `SingleWarp` differs from `Single`
    /// by. Firing the same kernel at 32 threads copies the same bytes — the
    /// stride loop covers `bytes` whatever `blockDim.x` is — so this control
    /// FAILS to differ, and that is the measurement: the two rules cannot be
    /// told apart by output on a stride loop, only by their launchers. Hazard
    /// 2's shape, stated rather than papered over.
    #[test]
    fn the_single_controls_are_the_guard_and_a_block_width_output_cannot_see() {
        let Some(_) =
            arch_or_skip("the_single_controls_are_the_guard_and_a_block_width_output_cannot_see")
        else {
            return;
        };
        let module = module_of("layout::copy_if_valid_slot", "layout/slot_ops");
        let entry = module.entry("layout::copy_if_valid_slot").expect("the row resolved");

        const BYTES: usize = 4096;
        let payload: Vec<u8> =
            bf16_fill(BYTES / 2, 0xA11C_E5).iter().flat_map(|v| v.to_le_bytes()).collect();
        let src = Buffer::of(&payload);
        let dst = Buffer::zeroed(BYTES);
        let slots: Vec<u8> = [7i32, -1, 3, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let slot_ids = Buffer::of(&slots);
        let d = dims(200, 128);

        // (a) request 1, whose slot id is -1.
        let guarded = [
            src.arg(),
            dst.arg(),
            ArgValue::Usize(BYTES),
            slot_ids.arg(),
            ArgValue::Usize(1),
        ];
        // SAFETY: as above; the guard makes it a no-op but the pointers are
        // live either way.
        unsafe { runtime::fire("layout::copy_if_valid_slot", d, &guarded, Stream::NULL) }
            .expect("the guarded fire is still a launch");
        synchronise("the guarded fire of layout::copy_if_valid_slot at request=1");
        let after_guard = dst.bytes();
        assert_eq!(
            after_guard.iter().filter(|b| **b != 0).count(),
            0,
            "the invalid-slot guard let {} bytes through",
            after_guard.iter().filter(|b| **b != 0).count()
        );

        // (b) the same launch at one warp.
        dst.clear();
        let valid = [
            src.arg(),
            dst.arg(),
            ArgValue::Usize(BYTES),
            slot_ids.arg(),
            ArgValue::Usize(2),
        ];
        // SAFETY: as above.
        unsafe { runtime::fire("layout::copy_if_valid_slot", d, &valid, Stream::NULL) }
            .expect("the valid fire");
        synchronise("the valid fire of layout::copy_if_valid_slot at request=2");
        let at_256 = dst.bytes();

        dst.clear();
        let (mut p_src, mut p_dst, mut p_ids) = (src.ptr, dst.ptr, slot_ids.ptr);
        let (mut n, mut req) = (BYTES, 2usize);
        let mut cells: Vec<*mut c_void> = vec![
            (&raw mut p_src).cast(),
            (&raw mut p_dst).cast(),
            (&raw mut n).cast(),
            (&raw mut p_ids).cast(),
            (&raw mut req).cast(),
        ];
        // SAFETY: `entry` outlives the call, the cells are the kernel's five
        // parameters, and 32 is a legal block width for a stride loop.
        unsafe {
            raw_launch(entry, [1, 1, 1], 32, &mut cells, "slot_ops by hand at ONE WARP");
        }
        let at_32 = dst.bytes();

        let live = at_256.iter().filter(|b| **b != 0).count();
        assert!(live > 0, "the 256-wide fire wrote nothing");
        let warp_differs = differing(&at_256, &at_32);
        assert_eq!(
            warp_differs, 0,
            "the warp-width control DIFFERS on {warp_differs} bytes. It is expected to \
             agree: a stride loop covers `bytes` at any block width, which is exactly \
             why `Single` and `SingleWarp` are told apart by their launchers and not \
             by their outputs."
        );
        eprintln!(
            "Single controls: guard let 0 of {BYTES} bytes through; \
             block 256 vs 32 differ on {warp_differs} of {BYTES} bytes ({live} non-zero) \
             — the block width is INVISIBLE to output, as stated"
        );
    }

    /// **`SingleWarp` is `<<<1, 32>>>` and the split view is byte-identical.**
    ///
    /// `attn/kv_paged.cu:533`. Two shapes — 3 splits over 6 pages and 5 over
    /// 5 — because `build_full_split_view`'s proportional boundaries take the
    /// `hi > lo` branch on the first and the EMPTY branch on the second when
    /// `splits > pages`, and a fixture that only ever divided evenly would
    /// never reach the `dst_last[i] = 0` arm.
    #[test]
    fn the_single_warp_row_reproduces_the_split_view_launcher() {
        let Some(_) = arch_or_skip("the_single_warp_row_reproduces_the_split_view_launcher")
        else {
            return;
        };
        let module = module_of("attn::build_full_split_view", "attn/kv_paged");
        let entry = module.entry("attn::build_full_split_view").expect("the row resolved");

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (splits, pages, page_size) in [(3u32, 6u32, 16u32), (5u32, 3u32, 8u32)] {
            let indptr: Vec<u8> = [0u32, pages].iter().flat_map(|v| v.to_le_bytes()).collect();
            let last: Vec<u8> = 5u32.to_le_bytes().to_vec();
            let indices: Vec<u8> =
                (0..pages).map(|p| 100 + p).flat_map(|v| v.to_le_bytes()).collect();
            let src_indptr = Buffer::of(&indptr);
            let src_last = Buffer::of(&last);
            let src_indices = Buffer::of(&indices);
            // `splits` slices, each of which may take the one-page empty arm,
            // so the index arena is at most `pages + splits` entries.
            let dst_indptr = Buffer::zeroed(4 * (splits as usize + 1));
            let dst_indices = Buffer::zeroed(4 * (pages as usize + splits as usize));
            let dst_last = Buffer::zeroed(4 * splits as usize);

            let d = dims(splits, 128);
            let launch = eval(Rule::SingleWarp, d).expect("ported");
            assert_eq!(launch.grid, [1, 1, 1], "the grid is a literal");
            assert_eq!(launch.block, [32, 1, 1], "`kv_paged.cu:533` fixes 32");
            assert_eq!(launch.smem, 0, "`kv_paged.cu:533` passes 0");
            assert_ne!(
                launch.block,
                eval(Rule::Single, d).expect("ported").block,
                "`Single` and `SingleWarp` must differ, and the block is the only \
                 field they can differ in"
            );

            let values = [
                src_indptr.arg(),
                src_last.arg(),
                ArgValue::I32(splits as i32),
                ArgValue::I32(page_size as i32),
                dst_indptr.arg(),
                dst_indices.arg(),
                dst_last.arg(),
                src_indices.arg(),
            ];
            // SAFETY: every pointer addresses a live allocation of the extent
            // the row states; the index arena is sized for the empty arm.
            unsafe { runtime::fire("attn::build_full_split_view", d, &values, Stream::NULL) }
                .unwrap_or_else(|why| panic!("the split view would not fire: {why}"));
            synchronise(&format!(
                "the shipped fire of attn::build_full_split_view at splits={splits} pages={pages}"
            ));
            let row_out =
                [dst_indptr.bytes(), dst_indices.bytes(), dst_last.bytes()].concat();

            dst_indptr.clear();
            dst_indices.clear();
            dst_last.clear();
            let (mut a, mut b) = (src_indptr.ptr, src_last.ptr);
            let (mut s, mut ps) = (splits as i32, page_size as i32);
            let (mut c, mut e, mut f, mut g) =
                (dst_indptr.ptr, dst_indices.ptr, dst_last.ptr, src_indices.ptr);
            let mut cells: Vec<*mut c_void> = vec![
                (&raw mut a).cast(),
                (&raw mut b).cast(),
                (&raw mut s).cast(),
                (&raw mut ps).cast(),
                (&raw mut c).cast(),
                (&raw mut e).cast(),
                (&raw mut f).cast(),
                (&raw mut g).cast(),
            ];
            // SAFETY: `entry` outlives the call and the cells are the
            // kernel's eight declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    [1, 1, 1],
                    32,
                    &mut cells,
                    &format!("kv_paged.cu:533 by hand at splits={splits}"),
                );
            }
            let raw_out =
                [dst_indptr.bytes(), dst_indices.bytes(), dst_last.bytes()].concat();

            let live = raw_out.iter().filter(|b| **b != 0).count();
            assert!(live > 0, "the launcher wrote nothing at splits={splits}");
            let differs = differing(&row_out, &raw_out);
            assert_eq!(
                differs, 0,
                "splits={splits} pages={pages}: the row and the launcher disagree on \
                 {differs} of {} bytes",
                row_out.len()
            );
            compared += row_out.len();
            live_total += live;
        }
        eprintln!(
            "SingleWarp via attn::build_full_split_view: 2 shapes, {compared} bytes \
             compared, {live_total} non-zero, 0 differing"
        );
    }

    /// **`PerRequest` is the request axis, and a decode cannot tell.**
    ///
    /// `attn/attention_naive.cu:174` is `<<<num_requests, BLOCK>>>` with
    /// `BLOCK = 256`. Two shapes, and they are chosen to be the two sides of
    /// hazard 1:
    ///
    /// * A DECODE — 4 requests, 4 tokens — where `rows == requests` and
    ///   `PerRow` and `PerRequest` are the same launch. Every single-token
    ///   fixture in this tree is this shape.
    /// * A PREFILL — 4 requests, 512 tokens — where `PerRow` opens 512 blocks
    ///   against a `slot_ids` of four entries. That is the substitution
    ///   `Dims::requests`' doc names, and it is asserted here as a GRID
    ///   difference rather than fired, because firing it is an out-of-bounds
    ///   read of `qo_indptr` and hazard 3 says a control that must fault is
    ///   reported, not left to poison eleven other tests.
    #[test]
    fn the_per_request_row_reproduces_the_mtp_launcher() {
        let Some(_) = arch_or_skip("the_per_request_row_reproduces_the_mtp_launcher") else {
            return;
        };
        let module =
            module_of("attn::mtp_update_pending_hidden_bf16", "attn/attention_naive");
        let entry = module
            .entry("attn::mtp_update_pending_hidden_bf16")
            .expect("the row resolved");

        const HIDDEN: usize = 256;
        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (requests, tokens) in [(4u32, 4u32), (4u32, 512u32)] {
            // A CSR that spends every token: request `r` owns a contiguous
            // run, and the kernel stashes each run's LAST row.
            let per = tokens / requests;
            let mut indptr = Vec::new();
            for r in 0..=requests {
                indptr.extend_from_slice(&(r * per).to_le_bytes());
            }
            let qo_indptr = Buffer::of(&indptr);
            // Slots out of order, so a fire that used `r` where it means
            // `slot_ids[r]` writes the wrong row rather than the same one.
            let slots: Vec<u8> =
                [3i32, 0, 2, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
            let slot_ids = Buffer::of(&slots);
            let payload = bf16_fill(tokens as usize * HIDDEN, 0x3E7_0BEE);
            let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
            let target = Buffer::of(&bytes);
            let pending = Buffer::zeroed(2 * requests as usize * HIDDEN);

            let mut d = dims(tokens, HIDDEN as u32);
            d.requests = requests;
            let launch = eval(Rule::PerRequest, d).expect("ported");
            assert_eq!(launch.grid, [requests, 1, 1], "the grid is the REQUEST count");
            assert_eq!(launch.block, [256, 1, 1], "`attention_naive.cuh:91` fixes 256");
            assert_eq!(launch.smem, 0, "`attention_naive.cu:174` passes 0");

            // THE NEAR MISS. Asserted, never fired: at 512 tokens `PerRow`
            // reads `qo_indptr[512]` off an array of five entries.
            let row_grid = eval(Rule::PerRow, d).expect("ported").grid;
            assert_eq!(row_grid, [tokens, 1, 1], "`PerRow` is the TOKEN count");
            if tokens == requests {
                assert_eq!(
                    row_grid, launch.grid,
                    "at {tokens} tokens and {requests} requests the two rules are the \
                     SAME launch — this is the shape that certifies nothing"
                );
            } else {
                assert_ne!(
                    row_grid, launch.grid,
                    "at {tokens} tokens the near miss is {}x the blocks",
                    tokens / requests
                );
            }

            let values = [
                target.arg(),
                pending.arg(),
                qo_indptr.arg(),
                slot_ids.arg(),
                ArgValue::I32(requests as i32),
                ArgValue::I32(HIDDEN as i32),
            ];
            // SAFETY: every pointer addresses a live allocation of the stated
            // extent, `qo_indptr` has `requests + 1` entries, and the null
            // stream is live.
            unsafe {
                runtime::fire(
                    "attn::mtp_update_pending_hidden_bf16",
                    d,
                    &values,
                    Stream::NULL,
                )
            }
            .unwrap_or_else(|why| panic!("the mtp row would not fire at {tokens}: {why}"));
            synchronise(&format!(
                "the shipped fire of attn::mtp_update_pending_hidden_bf16 at \
                 requests={requests} tokens={tokens}"
            ));
            let row_out = pending.bytes();

            pending.clear();
            let (mut a, mut b, mut c, mut e) =
                (target.ptr, pending.ptr, qo_indptr.ptr, slot_ids.ptr);
            let (mut r, mut h) = (requests as i32, HIDDEN as i32);
            let mut cells: Vec<*mut c_void> = vec![
                (&raw mut a).cast(),
                (&raw mut b).cast(),
                (&raw mut c).cast(),
                (&raw mut e).cast(),
                (&raw mut r).cast(),
                (&raw mut h).cast(),
            ];
            // SAFETY: `entry` outlives the call and the cells are the
            // kernel's six declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    [requests, 1, 1],
                    256,
                    &mut cells,
                    &format!("attention_naive.cu:174 by hand at tokens={tokens}"),
                );
            }
            let raw_out = pending.bytes();

            let live = written(&raw_out);
            assert_eq!(
                live,
                requests as usize * HIDDEN,
                "the launcher wrote {live} values, not the {} the CSR covers",
                requests as usize * HIDDEN
            );
            let differs = differing(&row_out, &raw_out);
            assert_eq!(
                differs, 0,
                "requests={requests} tokens={tokens}: the row and the launcher disagree \
                 on {differs} of {} bytes (rule grid {:?}, `PerRow` would be {row_grid:?})",
                row_out.len(),
                launch.grid
            );
            compared += row_out.len();
            live_total += live;
        }
        eprintln!(
            "PerRequest via attn::mtp_update_pending_hidden_bf16: 2 shapes (decode and \
             prefill), {compared} bytes compared, {live_total} values written, 0 differing"
        );
    }

    /// **The `PerRequest` control is a PERMUTATION of the slot map.**
    ///
    /// `slot_ids` rotated by one sends each request's last hidden row to a
    /// different slot. Same count of written values, same multiset of values,
    /// same sums — the buffer holds exactly the same four rows in a different
    /// order — so a check on statistics passes and a check on bytes does not.
    /// That is hazard 2's shape, in the operand a grid check cannot see.
    ///
    /// The second control is the CSR: `qo_indptr` shifted forward stashes a
    /// different row of `target_hidden` for each request. Not a permutation
    /// (the values are new), so it also proves the first control's agreement
    /// was not vacuous.
    #[test]
    fn the_per_request_controls_permute_the_slot_map_and_the_csr() {
        let Some(_) = arch_or_skip("the_per_request_controls_permute_the_slot_map_and_the_csr")
        else {
            return;
        };
        const HIDDEN: usize = 256;
        const REQUESTS: u32 = 4;
        const TOKENS: u32 = 512;

        let per = TOKENS / REQUESTS;
        let mut indptr = Vec::new();
        for r in 0..=REQUESTS {
            indptr.extend_from_slice(&(r * per).to_le_bytes());
        }
        let qo_indptr = Buffer::of(&indptr);
        let payload = bf16_fill(TOKENS as usize * HIDDEN, 0x5107_5);
        let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
        let target = Buffer::of(&bytes);
        let pending = Buffer::zeroed(2 * REQUESTS as usize * HIDDEN);

        let mut d = dims(TOKENS, HIDDEN as u32);
        d.requests = REQUESTS;

        let fire = |slots: &[i32], indptr: &Buffer, what: &str| {
            let map: Vec<u8> = slots.iter().flat_map(|v| v.to_le_bytes()).collect();
            let slot_ids = Buffer::of(&map);
            pending.clear();
            let values = [
                target.arg(),
                pending.arg(),
                indptr.arg(),
                slot_ids.arg(),
                ArgValue::I32(REQUESTS as i32),
                ArgValue::I32(HIDDEN as i32),
            ];
            // SAFETY: every pointer addresses a live allocation of the stated
            // extent and every slot id below is in `0..REQUESTS`.
            unsafe {
                runtime::fire(
                    "attn::mtp_update_pending_hidden_bf16",
                    d,
                    &values,
                    Stream::NULL,
                )
            }
            .expect("the mtp row fires");
            synchronise(what);
            pending.bytes()
        };

        let right = fire(&[3, 0, 2, 1], &qo_indptr, "the mtp reference fire");
        let rotated = fire(&[1, 3, 0, 2], &qo_indptr, "the mtp ROTATED-slots control");

        let live = written(&right);
        assert_eq!(live, REQUESTS as usize * HIDDEN, "the reference wrote {live} values");
        assert_eq!(
            written(&rotated),
            live,
            "the rotated map must write the same COUNT — that is what makes it a \
             permutation and not a truncation"
        );
        let mut a: Vec<u16> =
            right.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut b: Vec<u16> =
            rotated.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "the rotated map must be the same MULTISET of values");
        let slot_differs = differing(&right, &rotated);
        assert!(
            slot_differs > 0,
            "the rotated slot map is byte-identical: the fire is not reading \
             `slot_ids` at all"
        );

        // The CSR control: every request's run shifted forward by one token,
        // so each stashes a DIFFERENT row.
        let mut shifted = Vec::new();
        for r in 0..=REQUESTS {
            shifted.extend_from_slice(&(r * per + u32::from(r > 0)).to_le_bytes());
        }
        let shifted_indptr = Buffer::of(&shifted);
        let moved = fire(&[3, 0, 2, 1], &shifted_indptr, "the mtp SHIFTED-CSR control");
        let csr_differs = differing(&right, &moved);
        assert!(
            csr_differs > 0,
            "a shifted `qo_indptr` is byte-identical: the fire is not reading the CSR"
        );

        eprintln!(
            "PerRequest controls at requests={REQUESTS} tokens={TOKENS}: \
             rotated slots -> {slot_differs} of {} bytes differ, {live} values still \
             live (a PERMUTATION, same multiset); shifted CSR -> {csr_differs} bytes differ",
            right.len()
        );
    }

    // -----------------------------------------------------------------------
    // `Rule::RowsPackedHeadsNarrow` through the DEVICE-WINDOW rope
    // -----------------------------------------------------------------------

    /// A packed-heads rope rectangle with a device-side window.
    #[derive(Clone, Copy, Debug)]
    struct RopeWin {
        n_max: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    }

    impl RopeWin {
        fn dims(self) -> Dims {
            Dims {
                rows: self.n_max,
                width: self.q_heads * self.head_dim,
                in_width: self.q_heads * self.head_dim,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                stated_head_dim: 0,
                rotary_dims: self.head_dim,
                n_experts: 0,
                experts_per_token: 0,
                requests: 0,
                altup_streams: 0,
            }
        }

        /// `rope/rope.cu:163`, by hand:
        /// `dim3 grid(n_max, num_q_heads + num_kv_heads);`
        fn grid(self) -> [u32; 3] {
            [self.n_max, self.q_heads + self.kv_heads, 1]
        }

        fn q_elems(self) -> usize {
            (self.n_max * self.q_heads * self.head_dim) as usize
        }

        fn k_elems(self) -> usize {
            (self.n_max * self.kv_heads * self.head_dim) as usize
        }
    }

    /// Two shapes, and `kv_heads` differs on purpose.
    ///
    /// Hazard 1's lesson from `AltUpStreams`: at `q_heads == kv_heads` a rule
    /// that summed the wrong pair, or doubled one, lands on the same grid.
    /// 8+2 and 4+4 have different sums AND different products, so neither a
    /// transposition nor a doubling survives both.
    const ROPE_WIN_SHAPES: [RopeWin; 2] = [
        RopeWin { n_max: 6, q_heads: 8, kv_heads: 2, head_dim: 64 },
        RopeWin { n_max: 3, q_heads: 4, kv_heads: 4, head_dim: 128 },
    ];

    /// **The devwin rope row is byte-identical to `rope/rope.cu:164`.**
    ///
    /// The row states ELEVEN operands where the ahead-of-time twin states
    /// thirteen: `stream` goes because a stream is `cuLaunchKernel`'s sixth
    /// parameter, and `n_max` goes because the KERNEL never had it — the
    /// launcher spends it on the grid and `rope.cuh:483` reads `blockIdx.x`.
    /// So what this compares is an eleven-cell fire against an eleven-cell
    /// `cuLaunchKernel`: had `n_max` stayed, its four bytes would arrive in
    /// the kernel's frame where `win` is expected and every address after it
    /// would move.
    ///
    /// `Dims::rows` IS `n_max` and not by coincidence — the twin is `whole`
    /// (`table/rope.rs:86`) and `lower.rs:1064` refuses a `whole` statement
    /// any window but `[0, rows)` — so the fire runs at the full lane count,
    /// which is the only rectangle that exists for this row.
    #[test]
    fn the_devwin_rope_row_reproduces_the_launcher() {
        let Some(_) = arch_or_skip("the_devwin_rope_row_reproduces_the_launcher") else {
            return;
        };
        let module = module_of("rope::qk_rmsnorm_rope_bf16_devwin", "rope/rope");
        let entry =
            module.entry("rope::qk_rmsnorm_rope_bf16_devwin").expect("the row resolved");

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for shape in ROPE_WIN_SHAPES {
            let q_src = bf16_fill(shape.q_elems(), 0x0DE7_0001);
            let k_src = bf16_fill(shape.k_elems(), 0x0DE7_0002);
            let q = Buffer::of(&q_src);
            let k = Buffer::of(&k_src);
            let q_weight = Buffer::of(&bf16_fill(shape.head_dim as usize, 0x0DE7_0003));
            let k_weight = Buffer::of(&bf16_fill(shape.head_dim as usize, 0x0DE7_0004));
            let positions: Vec<i32> = (0..shape.n_max).map(|r| (r * 37) as i32).collect();
            let positions = Buffer::of(&positions);
            // The FULL window: offset 0, length `n_max`. Every lane is in.
            let win = Buffer::of(&[0u32, shape.n_max]);

            let launch = eval(Rule::RowsPackedHeadsNarrow, shape.dims()).expect("ported");
            assert_eq!(launch.grid, shape.grid(), "{shape:?}");
            assert_eq!(launch.block, [128, 1, 1], "`rope.cu:162` fixes 128");
            assert_eq!(launch.smem, 0, "`rope.cu:164` passes 0");

            let values = [
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                win.arg(),
                ArgValue::I32(shape.q_heads as i32),
                ArgValue::I32(shape.kv_heads as i32),
                ArgValue::I32(shape.head_dim as i32),
                ArgValue::F32(10_000.0),
                ArgValue::F32(1e-6),
            ];
            // SAFETY: every pointer addresses a live allocation of the extent
            // the row states, `win` is two `u32`s, and the null stream is live.
            unsafe {
                runtime::fire("rope::qk_rmsnorm_rope_bf16_devwin", shape.dims(), &values,
                              Stream::NULL)
            }
            .unwrap_or_else(|why| panic!("the devwin rope row would not fire at {shape:?}: {why}"));
            synchronise(&format!(
                "the shipped fire of rope::qk_rmsnorm_rope_bf16_devwin at {shape:?}"
            ));
            let (row_q, row_k) = (q.bytes(), k.bytes());

            // In-place, so restore the inputs before the hand launch.
            let q2 = Buffer::of(&q_src);
            let k2 = Buffer::of(&k_src);
            let (mut a, mut b) = (q2.ptr, k2.ptr);
            let (mut wq, mut wk) = (q_weight.ptr, k_weight.ptr);
            let (mut pos, mut w) = (positions.ptr, win.ptr);
            let (mut nq, mut nkv, mut hd) =
                (shape.q_heads as i32, shape.kv_heads as i32, shape.head_dim as i32);
            let (mut theta, mut eps) = (10_000.0f32, 1e-6f32);
            let mut cells: Vec<*mut c_void> = vec![
                (&raw mut a).cast(),
                (&raw mut b).cast(),
                (&raw mut wq).cast(),
                (&raw mut wk).cast(),
                (&raw mut pos).cast(),
                (&raw mut w).cast(),
                (&raw mut nq).cast(),
                (&raw mut nkv).cast(),
                (&raw mut hd).cast(),
                (&raw mut theta).cast(),
                (&raw mut eps).cast(),
            ];
            // SAFETY: `entry` outlives the call and the cells are the
            // kernel's eleven declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    shape.grid(),
                    128,
                    &mut cells,
                    &format!("rope.cu:164 by hand at {shape:?}"),
                );
            }
            let (raw_q, raw_k) = (q2.bytes(), k2.bytes());

            // In-place over a non-zero input, so `written` cannot be the
            // guard: the guard is that the fire CHANGED something.
            let live = written(&raw_q) + written(&raw_k);
            assert_eq!(
                live,
                shape.q_elems() + shape.k_elems(),
                "{shape:?}: the launcher left {live} live values"
            );
            let q_bytes: Vec<u8> = q_src.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert!(
                differing(&raw_q, &q_bytes) > 0,
                "{shape:?}: the launcher did not rotate anything"
            );

            let differs = differing(&row_q, &raw_q) + differing(&row_k, &raw_k);
            assert_eq!(
                differs, 0,
                "{shape:?}: the row and the launcher disagree on {differs} of {} bytes \
                 (rule grid {:?}, launcher grid {:?})",
                row_q.len() + row_k.len(),
                launch.grid,
                shape.grid()
            );
            compared += row_q.len() + row_k.len();
            live_total += live;
        }
        eprintln!(
            "RowsPackedHeadsNarrow via rope::qk_rmsnorm_rope_bf16_devwin: 2 shapes, \
             {compared} bytes compared, {live_total} values live, 0 differing"
        );

        the_devwin_rope_controls_truncate_and_the_head_split_is_invisible_to_the_grid();
    }

    /// **Two controls: the window is DEVICE-side, and the head split is not.**
    ///
    /// # Why this is called from the fire above and is not a `#[test]`
    ///
    /// It was one, and it raced — MEASURED, 1 run in 3, and the failure was
    /// not a fault but a WRONG ANSWER: the shipped fire's `q` came back
    /// differing from its input on 5,900 bytes where the hand launch and
    /// every serial run said 5,845. `--test-threads=1` was clean five times
    /// out of five, and each test alone was clean.
    ///
    /// This is the first IN-PLACE fire in this file — `q` and `k` are the
    /// row's inputs and its outputs — and it is the only pair of tests that
    /// fires one symbol from two threads. libtest runs tests in threads of
    /// one process sharing one primary context and one legacy null stream,
    /// and an in-place rotation read back while a second thread is uploading
    /// and rotating its own copy of the same rectangle is not a comparison of
    /// anything. Hazard 3's rule is that a control which cannot run cleanly
    /// is reported rather than left to poison; this one CAN run cleanly, in
    /// the same thread as the fire it is a control for, so it does.
    ///
    /// Nothing about the rule or the row is implicated: the hand launch was
    /// right on every run, including the ones where the shipped fire was not.
    ///
    /// (a) THE WINDOW. `win = {1, n_max - 2}` at the SAME grid. Every block
    /// still runs and the ones outside the window return before writing, so
    /// the grid is untouched and the OUTPUT is not — which is the measured
    /// reason `RowsPackedHeadsNarrow` may read `Dims::rows` for a kernel whose
    /// window it cannot see. The rows outside stay at their input values, so
    /// this is a TRUNCATION and the count of changed rows is the assertion.
    ///
    /// (b) THE HEAD SPLIT. `num_q_heads` and `num_kv_heads` swapped keeps the
    /// grid identical — the rule sums them — and moves the `is_q` boundary
    /// inside the kernel, so a different set of blocks writes `q` and `k`.
    /// The near miss a grid check cannot see, at the operand it hides in.
    /// Only run at 8+2, because 4+4 swapped is the same pair: hazard 1's
    /// shape, stated rather than avoided.
    fn the_devwin_rope_controls_truncate_and_the_head_split_is_invisible_to_the_grid() {
        let shape = ROPE_WIN_SHAPES[0];
        assert_ne!(shape.q_heads, shape.kv_heads, "control (b) needs an asymmetric split");

        let q_src = bf16_fill(shape.q_elems(), 0x0DE7_0001);
        let k_src = bf16_fill(shape.k_elems(), 0x0DE7_0002);
        let q_weight = Buffer::of(&bf16_fill(shape.head_dim as usize, 0x0DE7_0003));
        let k_weight = Buffer::of(&bf16_fill(shape.head_dim as usize, 0x0DE7_0004));
        let positions: Vec<i32> = (0..shape.n_max).map(|r| (r * 37) as i32).collect();
        let positions = Buffer::of(&positions);

        let fire = |window: &[u32; 2], nq: u32, nkv: u32, what: &str| {
            let q = Buffer::of(&q_src);
            let k = Buffer::of(&k_src);
            let win = Buffer::of(window);
            let values = [
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                win.arg(),
                ArgValue::I32(nq as i32),
                ArgValue::I32(nkv as i32),
                ArgValue::I32(shape.head_dim as i32),
                ArgValue::F32(10_000.0),
                ArgValue::F32(1e-6),
            ];
            // SAFETY: as the reference fire; `nq + nkv` is the shape's sum in
            // every call below, so no block addresses past the buffers.
            unsafe {
                runtime::fire("rope::qk_rmsnorm_rope_bf16_devwin", shape.dims(), &values,
                              Stream::NULL)
            }
            .expect("the devwin rope row fires");
            synchronise(what);
            (q.bytes(), k.bytes())
        };

        let (right_q, right_k) =
            fire(&[0, shape.n_max], shape.q_heads, shape.kv_heads, "the devwin rope reference");
        let (win_q, win_k) = fire(
            &[1, shape.n_max - 2],
            shape.q_heads,
            shape.kv_heads,
            "the devwin rope NARROW-WINDOW control",
        );
        let (swap_q, swap_k) = fire(
            &[0, shape.n_max],
            shape.kv_heads,
            shape.q_heads,
            "the devwin rope SWAPPED-HEADS control",
        );

        let q_bytes: Vec<u8> = q_src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let row_bytes = (shape.q_heads * shape.head_dim) as usize * 2;
        let rotated_rows = (0..shape.n_max as usize)
            .filter(|r| {
                let lo = r * row_bytes;
                differing(&right_q[lo..lo + row_bytes], &q_bytes[lo..lo + row_bytes]) > 0
            })
            .count();
        let windowed_rows = (0..shape.n_max as usize)
            .filter(|r| {
                let lo = r * row_bytes;
                differing(&win_q[lo..lo + row_bytes], &q_bytes[lo..lo + row_bytes]) > 0
            })
            .count();
        assert_eq!(
            rotated_rows, shape.n_max as usize,
            "the full window must rotate every lane"
        );
        assert_eq!(
            windowed_rows,
            (shape.n_max - 2) as usize,
            "the narrow window must rotate exactly {} lanes — the window is DEVICE-side \
             and the grid is unchanged, so this is the only place it can show",
            shape.n_max - 2
        );

        let win_differs = differing(&right_q, &win_q) + differing(&right_k, &win_k);
        assert!(win_differs > 0, "the narrow window changed nothing");
        let swap_differs = differing(&right_q, &swap_q) + differing(&right_k, &swap_k);
        assert!(
            swap_differs > 0,
            "swapping the head split is byte-identical: the fire is not reading \
             `num_q_heads`/`num_kv_heads` at all"
        );
        // The grid is the SAME for the swap — that is the whole point.
        let mut swapped = shape;
        swapped.q_heads = shape.kv_heads;
        swapped.kv_heads = shape.q_heads;
        assert_eq!(
            eval(Rule::RowsPackedHeadsNarrow, swapped.dims()).expect("ported").grid,
            eval(Rule::RowsPackedHeadsNarrow, shape.dims()).expect("ported").grid,
            "the swap must be invisible to the rule, or it is not this control"
        );

        eprintln!(
            "RowsPackedHeadsNarrow devwin controls at {shape:?}: narrow window -> \
             {windowed_rows} of {rotated_rows} lanes rotated, {win_differs} bytes differ \
             at the SAME grid; swapped head split -> {swap_differs} bytes differ at the \
             SAME grid"
        );
    }

    /// `norm::rmsnorm_bf16_with_fp16#vec8` against `norm/rmsnorm.cu:68-79`,
    /// and the DEFECT that firing it at two shapes found.
    ///
    /// # What this test measures, in the order it measured it
    ///
    /// The row was written to unblock `execution.rs`'s composition 6: the
    /// op's three predicates are all spellable and the `Choose` was refused
    /// only because the arm they select had no row. Firing that arm at two
    /// shapes, as the bar requires, did not certify it — it found that
    /// `rmsnorm_vec8<..., EMIT_FP16=true>` is **wrong for `num_rows > 1`**,
    /// in the shipping ahead-of-time kernel, reachable today.
    ///
    /// `rmsnorm.cuh:275-279` offsets both row pointers:
    ///
    /// ```text
    /// const float4* xr = ...(x + (long long)row * x_row_stride);
    /// float4*       yr = ...(y + (long long)row * y_row_stride);
    /// ```
    ///
    /// and `rmsnorm.cuh:318` does not:
    ///
    /// ```text
    /// y_fp16[i * 8 + j] = f32_to_f16(bf16_to_f32(ob[j]));
    /// ```
    ///
    /// `i` is the WITHIN-ROW `float4` index, so every block writes its fp16
    /// copy into row 0's slice. Measured at `rows = 3`, `hidden = 2048`:
    /// the fp16 head held 1 972 of 2 048 live values and the tail held
    /// **0 of 4 096** — rows 1 and 2 have no fp16 output at all — and two
    /// byte-identical hand launches of the same kernel differed on 247 fp16
    /// bytes, because three blocks race for one row's worth of slots. The
    /// bf16 output was byte-identical on every run, which is why nothing
    /// downstream of the bf16 path ever noticed.
    ///
    /// So this test does two things and says which is which:
    ///
    /// 1. **Certifies the row at `rows = 1`**, at two widths, where the
    ///    kernel is single-block and therefore correct and deterministic:
    ///    `hidden = 2048` is exactly one `float4` per lane at 256 threads,
    ///    and `hidden = 5376` is 672 vectors over 256 lanes — 2.625 passes,
    ///    a ragged tail. Both are `% 8 == 0` and `cuMemAlloc` is 256-byte
    ///    aligned, so `rmsnorm_vec8_ok`'s six clauses hold: this is the arm
    ///    the predicate selects, fired where the predicate is true.
    /// 2. **Pins the defect at `rows = 3`**, so that the day
    ///    `rmsnorm.cuh:318` grows its row offset, this test fails and says
    ///    the row can be widened. Until then the row exists, compiles and
    ///    resolves, and `families/norm.rs` says in its own text that it must
    ///    not be wired into a `Choose`.
    ///
    /// The two-shape bar is met in `hidden` rather than in `rows` for a
    /// stated reason: `Rule::Rms`'s grid is `rows` with no division, so
    /// varying `rows` cannot separate it from a near-miss rule, while
    /// varying `hidden` is what separates a right reduction from one folding
    /// shared memory no thread wrote. The row axis is pinned at 1 because
    /// the KERNEL is wrong above it, and that is measured below rather than
    /// assumed.
    #[test]
    fn the_emit_fp16_row_reproduces_the_with_fp16_launcher() {
        let Some(_) = arch_or_skip("the_emit_fp16_row_reproduces_the_with_fp16_launcher")
        else {
            return;
        };
        let module = module_of("norm::rmsnorm_bf16_with_fp16#vec8", "norm/rmsnorm");
        let entry =
            module.entry("norm::rmsnorm_bf16_with_fp16#vec8").expect("the row resolved");

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for hidden in [2048usize, 5376usize] {
            const ROWS: u32 = 1;
            let n = ROWS as usize * hidden;
            let x_src = bf16_fill(n, 0xF16_0001);
            let w_src = bf16_fill(hidden, 0xF16_0002);
            let x = Buffer::of(&x_src);
            let weight = Buffer::of(&w_src);
            let y = Buffer::zeroed(2 * n);
            let y_fp16 = Buffer::zeroed(2 * n);

            let d = dims(ROWS, u32::try_from(hidden).expect("small"));
            let launch = eval(Rule::Rms, d).expect("ported");
            assert_eq!(launch.grid, [ROWS, 1, 1], "`rmsnorm.cu:69` is `dim3 grid(num_rows)`");
            assert_eq!(
                launch.block, [256, 1, 1],
                "the row instantiates `rmsnorm_vec8<256, ...>` and `Rule::Rms` launches 256 \
                 — if these two ever disagree the reduction folds shared memory no thread \
                 wrote, which is finite and wrong"
            );

            let w = i32::try_from(hidden).expect("small");
            let values = [
                x.arg(),
                weight.arg(),
                y.arg(),
                y_fp16.arg(),
                ArgValue::I32(w),
                ArgValue::I32(w),
                ArgValue::I32(w),
                ArgValue::F32(1e-6),
            ];
            // SAFETY: every pointer addresses a live allocation of `hidden`
            // bf16, the strides are the width, and the null stream is live.
            unsafe {
                runtime::fire("norm::rmsnorm_bf16_with_fp16#vec8", d, &values, Stream::NULL)
            }
            .unwrap_or_else(|why| panic!("the EMIT_FP16 row would not fire at {hidden}: {why}"));
            synchronise(&format!(
                "the shipped fire of norm::rmsnorm_bf16_with_fp16#vec8 at hidden={hidden}"
            ));
            let row_y = y.bytes();
            let row_fp16 = y_fp16.bytes();

            y.clear();
            y_fp16.clear();
            let (mut a, mut b, mut c, mut e) = (x.ptr, weight.ptr, y.ptr, y_fp16.ptr);
            let (mut h, mut xs, mut ys) = (w, w, w);
            let mut eps = 1e-6f32;
            let mut cells: Vec<*mut c_void> = vec![
                (&raw mut a).cast(),
                (&raw mut b).cast(),
                (&raw mut c).cast(),
                (&raw mut e).cast(),
                (&raw mut h).cast(),
                (&raw mut xs).cast(),
                (&raw mut ys).cast(),
                (&raw mut eps).cast(),
            ];
            // SAFETY: `entry` outlives the call and the cells are the
            // kernel's eight declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    [ROWS, 1, 1],
                    256,
                    &mut cells,
                    &format!("rmsnorm.cu:68-79 by hand at hidden={hidden}"),
                );
            }
            let raw_y = y.bytes();
            let raw_fp16 = y_fp16.bytes();

            assert_eq!(written(&raw_y), n, "the bf16 output covers the row");
            let live16 = written(&raw_fp16);
            assert!(
                live16 > 0,
                "EMIT_FP16 is TRUE and the fp16 copy is entirely zero — the flag was compiled \
                 out and this row is the sibling row wearing a different name"
            );
            // NOT `== n`: `bf16_fill` spans 2^-7 to 2^9 and the norm
            // multiplies two of them, so a good fraction of the products
            // land below fp16's smallest normal and flush to zero. The bf16
            // count above is what proves the rectangle was covered.
            assert!(live16 < n, "the fixture no longer straddles fp16's exponent range");

            assert_eq!(
                differing(&row_y, &raw_y), 0,
                "hidden={hidden}: the row and the launcher disagree on the bf16 output"
            );
            assert_eq!(
                differing(&row_fp16, &raw_fp16), 0,
                "hidden={hidden}: the row and the launcher disagree on the FP16 output — \
                 this is the half of the contract the sibling row cannot carry"
            );
            compared += row_y.len() + row_fp16.len();
            live_total += written(&raw_y) + live16;

            // THE NEGATIVE CONTROL, in the same thread for hazard 3's
            // reason. `WEIGHT_PLUS_ONE` is the other flag on this template;
            // firing the same entry against a weight vector one binade up
            // must move bytes, or the comparison above is a norm compared
            // with itself. Done in the ARGUMENTS rather than the
            // instantiation, because the row states one instantiation and a
            // second would be a second row.
            let plus_one: Vec<u16> = w_src.iter().map(|w| bf16_plus_one(*w)).collect();
            let weight2 = Buffer::of(&plus_one);
            y.clear();
            y_fp16.clear();
            let mut b2 = weight2.ptr;
            cells[1] = (&raw mut b2).cast();
            // SAFETY: as above, with a different weight buffer of the same
            // extent.
            unsafe {
                raw_launch(
                    entry,
                    [ROWS, 1, 1],
                    256,
                    &mut cells,
                    &format!("rmsnorm.cu:68-79 control at hidden={hidden}"),
                );
            }
            let moved = differing(&y.bytes(), &raw_y);
            assert!(
                moved > 0,
                "hidden={hidden}: scaling every weight moved 0 of {} bytes — the comparison \
                 above is not measuring the weights at all",
                raw_y.len()
            );
            eprintln!(
                "  EMIT_FP16 control at hidden={hidden}: a scaled weight moved {moved} of {} \
                 bytes at the SAME grid",
                raw_y.len()
            );
        }
        eprintln!(
            "EMIT_FP16 via norm::rmsnorm_bf16_with_fp16#vec8: 2 shapes (1x2048, 1x5376), \
             {compared} bytes compared, {live_total} values live, 0 differing"
        );
        the_emit_fp16_kernel_is_wrong_above_one_row(entry);
    }

    /// The defect, pinned. Called from the reference test rather than being a
    /// `#[test]` of its own, for the reason
    /// `the_devwin_rope_row_reproduces_the_launcher` records: an in-place or
    /// racing fire run concurrently with its own reference produced a WRONG
    /// ANSWER rather than a fault, one run in three, and the resolution is
    /// that a control shares a thread with the thing it controls.
    ///
    /// `rmsnorm.cuh:318` writes `y_fp16[i * 8 + j]` where `i` is the
    /// within-row `float4` index, while `rmsnorm.cuh:277-279` offsets `yr`
    /// by `row * y_row_stride`. Three consequences, all asserted here:
    ///
    /// * rows 1..n have NO fp16 output — the tail is untouched;
    /// * row 0's slice is written by every block, so the value that survives
    ///   is a race;
    /// * the bf16 output is correct throughout, which is why the defect has
    ///   been invisible.
    ///
    /// This is not a JIT defect. The row reproduces `rmsnorm.cu:68-79`
    /// exactly; the ahead-of-time launcher fires the same kernel with the
    /// same arguments and gets the same wrong buffer, on every prefill whose
    /// rows are 16-byte aligned. Fixing it is a change to device text, which
    /// this work does not own — `new-horizon.md` §10.10 fixes the order:
    /// extract, add rows, measure, and only then change what was extracted.
    fn the_emit_fp16_kernel_is_wrong_above_one_row(entry: dr::CUfunction) {
        const ROWS: u32 = 3;
        const HIDDEN: usize = 2048;
        let n = ROWS as usize * HIDDEN;
        let x = Buffer::of(&bf16_fill(n, 0xF16_0011));
        let weight = Buffer::of(&bf16_fill(HIDDEN, 0xF16_0012));
        let y = Buffer::zeroed(2 * n);
        let y_fp16 = Buffer::zeroed(2 * n);

        let w = i32::try_from(HIDDEN).expect("small");
        let (mut a, mut b, mut c, mut e) = (x.ptr, weight.ptr, y.ptr, y_fp16.ptr);
        let (mut h, mut xs, mut ys) = (w, w, w);
        let mut eps = 1e-6f32;
        let mut cells: Vec<*mut c_void> = vec![
            (&raw mut a).cast(),
            (&raw mut b).cast(),
            (&raw mut c).cast(),
            (&raw mut e).cast(),
            (&raw mut h).cast(),
            (&raw mut xs).cast(),
            (&raw mut ys).cast(),
            (&raw mut eps).cast(),
        ];
        // SAFETY: three rows of `HIDDEN` bf16 in every buffer, strides at
        // the width, and the cells are the kernel's eight parameters.
        unsafe {
            raw_launch(entry, [ROWS, 1, 1], 256, &mut cells, "rmsnorm.cu:68-79 at 3 rows");
        }
        let first = y_fp16.bytes();
        let bf16_first = y.bytes();

        let per_row = HIDDEN * 2;
        let head = written(&first[..per_row]);
        let tail = written(&first[per_row..]);
        assert!(head > 0, "row 0's fp16 slice is empty — the flag is compiled out");
        assert_eq!(
            tail, 0,
            "rows 1..{ROWS} have {tail} live fp16 values. If this is non-zero the row \
             offset at `rmsnorm.cuh:318` has been ADDED and the defect is fixed: widen \
             `the_emit_fp16_row_reproduces_the_with_fp16_launcher` back to several rows \
             and delete this function."
        );

        y.clear();
        y_fp16.clear();
        // SAFETY: identical to the launch above, same buffers, same cells.
        unsafe {
            raw_launch(entry, [ROWS, 1, 1], 256, &mut cells, "rmsnorm.cu:68-79 at 3 rows again");
        }
        let second = y_fp16.bytes();
        let bf16_second = y.bytes();

        assert_eq!(
            differing(&bf16_first, &bf16_second), 0,
            "the BF16 output is not reproducible either — the defect is wider than \
             `y_fp16`'s missing row offset and this diagnosis is incomplete"
        );
        let raced = differing(&first, &second);
        assert!(
            raced > 0,
            "two launches agreed on {} fp16 bytes. Three blocks writing one row's slots \
             agreeing exactly would mean the slots are no longer shared — check whether \
             `rmsnorm.cuh:318` has been fixed",
            first.len()
        );
        eprintln!(
            "  EMIT_FP16 DEFECT pinned at rows={ROWS} hidden={HIDDEN}: fp16 head {head} of \
             {HIDDEN} live, TAIL {tail} of {} live; two identical launches differ on \
             {raced} fp16 bytes and 0 bf16 bytes (`rmsnorm.cuh:318` has no row offset)",
            (ROWS as usize - 1) * HIDDEN
        );
    }

    /// One bf16's exponent raised by one binade — a scale of 2, in bf16, with
    /// no rounding. Used as a negative control's perturbation because it
    /// cannot round back to the original value.
    fn bf16_plus_one(w: u16) -> u16 {
        let exponent = (w >> 7) & 0xFF;
        if exponent >= 0xFE { return w & 0x807F; }
        (w & 0x807F) | ((exponent + 1) << 7)
    }

    /// `ssm::nemotron_mamba_split_bf16#split` against `ssm/nemotron_h.cu:48-54`.
    ///
    /// # Two shapes, and the near miss between them
    ///
    /// `ElementwiseIn` is `ceil(rows * in_width / 256)`, so the two shapes
    /// are chosen either side of the block boundary:
    ///
    /// * `N = 2`, `projection_dim = 128` — 256 elements, exactly ONE block.
    ///   At this shape `Elementwise` over the OUTPUT width (`intermediate =
    ///   64`, so 128 elements) is *also* one block, and the two rules are
    ///   indistinguishable. This is the fixture that certifies nothing, and
    ///   it is here to be named as such.
    /// * `N = 5`, `projection_dim = 320` — 1600 elements, 7 blocks, where
    ///   `Elementwise` over the output width would open 3. That is the shape
    ///   that separates them, and it is asserted as a grid difference as
    ///   well as fired.
    #[test]
    fn the_elementwise_in_row_reproduces_the_mamba_split_launcher() {
        let Some(_) =
            arch_or_skip("the_elementwise_in_row_reproduces_the_mamba_split_launcher")
        else {
            return;
        };
        let module = module_of("ssm::nemotron_mamba_split_bf16#split", "ssm/nemotron_h");
        let entry = module
            .entry("ssm::nemotron_mamba_split_bf16#split")
            .expect("the row resolved");

        let mut compared = 0usize;
        let mut live_total = 0usize;
        for (n, intermediate, conv_dim, num_heads) in [(2u32, 64u32, 48u32, 16u32),
                                                       (5u32, 160u32, 128u32, 32u32)] {
            let projection_dim = intermediate + conv_dim + num_heads;
            let total = n * projection_dim;
            let src = bf16_fill(total as usize, 0x5017_7000);
            let projected = Buffer::of(&src);
            let gate = Buffer::zeroed(2 * (n * intermediate) as usize);
            let conv_in = Buffer::zeroed(2 * (n * conv_dim) as usize);
            let dt = Buffer::zeroed(2 * (n * num_heads) as usize);

            let mut d = dims(n, intermediate);
            d.in_width = projection_dim;
            let launch = eval(Rule::ElementwiseIn, d).expect("ported");
            assert_eq!(
                launch.grid,
                [total.div_ceil(256), 1, 1],
                "`nemotron_h.cu:49` is `(total + BLOCK - 1) / BLOCK` over `N * projection_dim`"
            );
            assert_eq!(launch.block, [256, 1, 1], "`nemotron_h.cu:36` fixes BLOCK = 256");
            assert_eq!(launch.smem, 0, "`nemotron_h.cu:50` passes 0");

            // THE NEAR MISS, asserted rather than fired: `Elementwise` sizes
            // on the OUTPUT width, which this file's own refusal named.
            let out_grid = eval(Rule::Elementwise, d).expect("ported").grid;
            if out_grid == launch.grid {
                assert_eq!(
                    total.div_ceil(256), 1,
                    "the two rules agreeing is only allowed inside one block"
                );
            } else {
                assert_ne!(
                    out_grid, launch.grid,
                    "at N={n} the output-width rule opens {out_grid:?} where the input-width \
                     rule opens {:?}", launch.grid
                );
            }

            let values = [
                projected.arg(),
                gate.arg(),
                conv_in.arg(),
                dt.arg(),
                ArgValue::I32(i32::try_from(projection_dim).expect("small")),
                ArgValue::I32(i32::try_from(intermediate).expect("small")),
                ArgValue::I32(i32::try_from(conv_dim).expect("small")),
                ArgValue::I32(i32::try_from(num_heads).expect("small")),
                ArgValue::I32(i32::try_from(total).expect("small")),
            ];
            // SAFETY: `projected` is `n * projection_dim` bf16 and the three
            // outputs are each `n` rows of their own width; the null stream
            // is live.
            unsafe {
                runtime::fire("ssm::nemotron_mamba_split_bf16#split", d, &values, Stream::NULL)
            }
            .unwrap_or_else(|why| panic!("the mamba_split row would not fire at N={n}: {why}"));
            synchronise(&format!(
                "the shipped fire of ssm::nemotron_mamba_split_bf16#split at N={n}"
            ));
            let row_out: Vec<u8> = [gate.bytes(), conv_in.bytes(), dt.bytes()].concat();

            gate.clear();
            conv_in.clear();
            dt.clear();
            let (mut a, mut b, mut c, mut e) =
                (projected.ptr, gate.ptr, conv_in.ptr, dt.ptr);
            let mut pd = i32::try_from(projection_dim).expect("small");
            let mut im = i32::try_from(intermediate).expect("small");
            let mut cd = i32::try_from(conv_dim).expect("small");
            let mut nh = i32::try_from(num_heads).expect("small");
            let mut tt = i32::try_from(total).expect("small");
            let mut cells: Vec<*mut c_void> = vec![
                (&raw mut a).cast(),
                (&raw mut b).cast(),
                (&raw mut c).cast(),
                (&raw mut e).cast(),
                (&raw mut pd).cast(),
                (&raw mut im).cast(),
                (&raw mut cd).cast(),
                (&raw mut nh).cast(),
                (&raw mut tt).cast(),
            ];
            // SAFETY: `entry` outlives the call and the cells are the
            // kernel's nine declared parameters in order.
            unsafe {
                raw_launch(
                    entry,
                    [total.div_ceil(256), 1, 1],
                    256,
                    &mut cells,
                    &format!("nemotron_h.cu:50 by hand at N={n}"),
                );
            }
            let raw_out: Vec<u8> = [gate.bytes(), conv_in.bytes(), dt.bytes()].concat();

            let live = written(&raw_out);
            assert_eq!(
                live, total as usize,
                "the split moves every element of the input rectangle exactly once, so {} \
                 values must be live and {live} are",
                total
            );
            assert_eq!(
                differing(&row_out, &raw_out), 0,
                "N={n}: the row and the launcher disagree across gate|conv_in|dt"
            );
            compared += row_out.len();
            live_total += live;

            // THE NEGATIVE CONTROL, in the same thread for hazard 3's
            // reason, and it is a PERMUTATION: swapping `conv_dim` and
            // `num_heads` moves the same multiset of values into different
            // slots. Sums, counts and the non-zero census are all identical
            // — only the addresses change — so anything that passes on
            // statistics passes this and only a byte comparison catches it.
            if conv_dim != num_heads {
                gate.clear();
                conv_in.clear();
                dt.clear();
                let (mut cd2, mut nh2) = (nh, cd);
                cells[6] = (&raw mut cd2).cast();
                cells[7] = (&raw mut nh2).cast();
                // SAFETY: as above; the two widths sum to the same total.
                unsafe {
                    raw_launch(
                        entry,
                        [total.div_ceil(256), 1, 1],
                        256,
                        &mut cells,
                        &format!("nemotron_h.cu:50 permuted control at N={n}"),
                    );
                }
                let control: Vec<u8> =
                    [gate.bytes(), conv_in.bytes(), dt.bytes()].concat();
                let moved = differing(&control, &raw_out);
                let still = written(&control);
                assert!(
                    moved > 0,
                    "N={n}: swapping conv_dim and num_heads moved 0 of {} bytes",
                    control.len()
                );
                eprintln!(
                    "  mamba_split control at N={n}: swapped conv/head widths -> {moved} of \
                     {} bytes differ, {still} values still live (a PERMUTATION, same multiset)",
                    control.len()
                );
            }
        }
        eprintln!(
            "ElementwiseIn via ssm::nemotron_mamba_split_bf16#split: 2 shapes (N=2/pd=128, \
             N=5/pd=320), {compared} bytes compared, {live_total} values live, 0 differing"
        );
    }
}
