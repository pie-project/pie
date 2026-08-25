//! The fire's own tables, staged once.
//!
//! A row names which table a slot wants; something has to put them on the
//! device. That was written twice — once at the engine seam and once in
//! `tests/device_real_weights.rs` — and the two drifted the moment the text
//! named a table only one of them knew: the gate held both fire classes to
//! MLX while the seam was still staging seven of nine, so a rotation with no
//! frequency ladder and a readout that answered the first token were invisible
//! from either side.
//!
//! One builder, and the resolver's index map beside it. What a caller supplies
//! is the FRAME's data; what this owns is the order.
//!
//! # THE ROTARY LADDER IS NOT ONE OF THEM
//!
//! `rope_frequencies` was the eighth table here and `FireTable` has no row
//! for it. A rotation is a statement of the trace and its base rides on that
//! statement, so a plain ladder is raised by the shader and a rescaled one is
//! a `Const` bank the text names -- neither is a plane a fire stages.
//! `baker::stage` carries the same note, and the three KV STRIDES that left
//! with it went to `baker::KvGeometry`, where a number the allocator settled
//! belongs.

use crate::baker::{FireTable, Slice};
use crate::device::{Context, Handle};
use crate::error::Result;
use crate::fire::scratch::{Lease, Scratch};
use crate::layout::region::Region as _;

/// The tables a fire states, in the order [`Staged::at`] indexes them.
///
/// Borrowed rather than owned: every one of these is the frame's, already
/// laid out, and copying them to describe them would be the third place they
/// live.
#[derive(Debug, Default, Clone, Copy)]
pub struct Frame<'a> {
    /// One token id per row.
    pub token_ids: &'a [u32],
    /// One position per row.
    pub position_ids: &'a [u32],
    /// Which request owns each token.
    pub req_of_token: &'a [u32],
    /// The request CSR: `qo_indptr[r]..qo_indptr[r + 1]` are request `r`'s
    /// rows. **ROWS ARE THE REQUEST COUNT AND THE WIDTH IS ONE MORE** — the
    /// shape `walk::fire`'s `runtime` states for it.
    ///
    /// The PREFILL lane's attention reads it and the decode lane's does not,
    /// which is why a driver that never staged it served every banked answer
    /// and refused its other lane at op 9.
    pub qo_indptr: &'a [u32],
    /// The page list, and the CSR into it.
    pub kv_page_indices: &'a [u32],
    /// See [`Self::kv_page_indices`].
    pub kv_page_indptr: &'a [u32],
    /// Per token: the physical page its KV row is written into, and the row
    /// within it.
    pub kv_write_page: &'a [u32],
    /// See [`Self::kv_write_page`].
    pub kv_write_offset: &'a [u32],
    /// Which rows of the FIRE the readout samples, in fire order.
    ///
    /// ABSOLUTE, and not what the wire carries. `Step::sampling_indices` is
    /// numbered inside the request that named it, and `row_gather` indexes
    /// the fire's stream with whatever is here -- so the two are different
    /// numbers wherever a fire has more than one request, and the wrong one
    /// reads a different token's hidden state without failing.
    /// [`crate::baker::frame::sampled_rows`] is the translation, and the
    /// only one; nothing else may fill this.
    pub sampling_indices: &'a [u32],
    /// How many KV rows one page holds.
    ///
    /// Not a table, and the only scalar here, because without it the tables
    /// above cannot be CHECKED. `kv_page_indices` is a run of physical pages
    /// per request and `position_ids` says how far into that run the fire
    /// reaches, but the two are in different units until something states
    /// the divisor. [`stage`] states the refusal that needs it.
    ///
    /// Zero means "not stated", and then the refusal does not run. That is
    /// not an escape hatch for a caller who finds the check inconvenient --
    /// it is for the fires that state no pages at all, where there is no run
    /// to fall short of.
    pub page_size: u32,
    /// Per ROW: which recurrent-state slot its request occupies.
    ///
    /// One entry per token, not per request: `gdn_prep_slotted` and
    /// `gdn_core_recurrent_slotted` both index it by the fire's row, and
    /// [`stage`] refuses any other length rather than let the shortfall
    /// become a device read past this region.
    ///
    /// Empty for every stack with no linear-attention layers, which is every
    /// row this backend serves today, and `Staged::at` answers `None` for an
    /// empty table rather than an address -- so a GDN symbol that reached a
    /// fire without one refuses instead of indexing a slab at zero.
    pub recurrent_slots: &'a [u32],
}

/// One region holding every table, and where each one starts.
#[derive(Debug)]
pub struct Staged {
    /// The region. **Held, not dropped** — every span points into it, and
    /// it is resident for exactly as long as this `Staged` lives, so it must
    /// outlive the fire that binds it rather than the loop that stages it.
    ///
    /// LEASED from the fire's [`Scratch`], for the reason that module's doc
    /// names: the tables are one of the three regions that vary between two
    /// fires of one shape, and a region allocated fresh per fire has a new
    /// address every fire. That is measured, not argued — a 32-step decode
    /// loop with a `Recordings` on the side made **32 recordings for 32
    /// decodes**, a 100 % miss rate, because `fingerprint` digests every
    /// operand's address and nine of them point in here. Pooling the region
    /// is what turns the recording cache from dead weight into the thing it
    /// was built to be, and it closes the residency-set leak in the same
    /// move.
    region: Lease,
    /// Each table's `(start, len)`, in u32s, indexed as [`Frame`] declares.
    spans: Vec<(usize, usize)>,
}

impl Staged {
    /// The region every span points into.
    ///
    /// What a caller registers with [`crate::device::Regions`], and what it
    /// names as the null stand-in.
    #[must_use]
    pub fn region(&self) -> &Handle {
        self.region.region()
    }

    /// Where `which` is, or `None` for a table this fire has none of.
    ///
    /// A zero-length table answers `None` rather than an empty region: a slot
    /// nobody fills is better than one pointed at nothing, because the second
    /// is a valid address the kernel will read.
    #[must_use]
    pub fn at(&self, which: FireTable) -> Option<Slice> {
        let i = match which {
            FireTable::TokenIds => 0,
            FireTable::Positions => 1,
            FireTable::RequestOfToken => 2,
            FireTable::KvPageIndices => 3,
            FireTable::KvPageIndptr => 4,
            FireTable::KvWritePage => 5,
            FireTable::KvWriteOffset => 6,
            FireTable::SamplingIndices => 7,
            // The enable flag is staged (zeros: causal), and the mask itself
            // is not -- a row whose enable is zero never indexes it. The
            // pool's numbers are answered by `Resolver::pool` rather than by
            // an address.
            FireTable::AttentionMaskEnabled => 8,
            FireTable::RecurrentSlots => 9,
            FireTable::AttentionMask => return None,
            // TEN AND ELEVEN, appended rather than inserted: this list IS the
            // order and renumbering it moves every table under every slot.
            //
            // They were `None` until the prefill lane was fired — *"serve::launch
            // builds a fire's frame from the wire's own plan and neither plane
            // is on it"*, which was true, and made every prefill of every tower
            // refuse at its first attention. Both are on it now.
            FireTable::QoIndptr => 10,
            FireTable::RowValid => 11,
        };
        let (at, len) = *self.spans.get(i)?;
        (len > 0).then(|| Slice {
            address: self.region.gpu_address() + (at * 4) as u64,
            bytes: (len * 4) as u64,
        })
    }
}

/// Stage every table of `frame` into one region, leased from `scratch`.
///
/// # Errors
///
/// A `recurrent_slots` that is neither empty nor one entry per token, and
/// then the allocation or the write.
pub fn stage(context: &Context, scratch: &Scratch, frame: Frame<'_>) -> Result<Staged> {
    // The seats are read PER TOKEN by both slotted GDN kernels, so a table
    // that is merely non-empty is not enough: one short by a row is a device
    // read past the end of this very region, and what it reads then indexes a
    // device write. Checked here because this is the one place the two
    // lengths are side by side, and because the alternative is not a wrong
    // answer but a GPU that never retires -- see
    // `model::qwen_3_5::forward::metal`.
    if !frame.recurrent_slots.is_empty() && frame.recurrent_slots.len() != frame.token_ids.len() {
        return Err(crate::error::Error::Create {
            what: "fire tables",
            message: format!(
                "the fire states {} recurrent seats for {} tokens; the slotted GDN kernels \
                 read one seat per token, so anything else is a read past this region",
                frame.recurrent_slots.len(),
                frame.token_ids.len()
            ),
        });
    }
    // THE PAGE RUN MUST REACH THE LAST POSITION, and this refusal exists
    // because a rig spent three commits blaming the driver for its absence.
    //
    // `kv_page_indices` is a per-request run of physical pages and the paged
    // attention kernels index it as `kv_page_indices[indptr[r] + p /
    // page_size]` for every KEY position `p` a query row may look back at.
    // A run one page long is right for a sequence that fits in one page and
    // is a read past the end of the list for anything longer. What it reads
    // then is a PAGE NUMBER, so the fire goes on to address a real page
    // belonging to nobody: nothing faults, nothing is uninitialised, and the
    // answer is simply drawn from the wrong keys.
    //
    // Measured, on `Qwen3-0.6B-4bit` with a 16-row page and a rig staging
    // `kv_page_indices: &[0]`: prefills of 16 tokens or fewer returned
    // bit-identical logits on every run, and 17 or more returned a different
    // answer in most processes -- five isolated runs of one 20-token prompt
    // gave `1124, 1124, 11, 220, 431`. The boundary was exactly the page
    // size, which is what named the defect after a kernel reading had
    // produced a plausible wrong answer instead.
    //
    // Nothing caught it for as long as it did because every reference prompt
    // in that rig was five to seven tokens. A rig-wide invariant can be
    // invisible simply because no fixture ever crossed it, which is why this
    // is checked where the tables are BUILT and not where they are used.
    //
    // `driver_api::Plan::validate` already refuses this on the serving path.
    // This is the same refusal for the path that does not build a `Plan` --
    // `serve::launch` and every hand-built test frame -- and it is stated
    // here because this is the one place all five tables are side by side.
    if frame.page_size > 0 && !frame.kv_page_indptr.is_empty() {
        let requests = frame.kv_page_indptr.len() - 1;
        for (t, (&r, &pos)) in frame
            .req_of_token
            .iter()
            .zip(frame.position_ids)
            .enumerate()
        {
            let r = r as usize;
            if r >= requests {
                return Err(crate::error::Error::Create {
                    what: "fire tables",
                    message: format!(
                        "token {t} says request {r}, but kv_page_indptr \
                         describes {requests} of them"
                    ),
                });
            }
            let (from, to) = (
                frame.kv_page_indptr[r] as usize,
                frame.kv_page_indptr[r + 1] as usize,
            );
            let run = to.saturating_sub(from);
            // The LAST position and not the token count, because a decode is
            // one token whose history is many pages: a fire of one row at
            // position 40 needs three 16-row pages staged, not one.
            let wanted = pos as usize / frame.page_size as usize + 1;
            if wanted > run {
                return Err(crate::error::Error::Create {
                    what: "fire tables",
                    message: format!(
                        "token {t} of request {r} sits at position {pos}, which a \
                         {}-row page puts in page {} of that request; the \
                         request states {run} page(s). The paged attention \
                         kernels read one page index per {} key positions, \
                         so a run this short is a read past kv_page_indices \
                         and an answer drawn from whichever page the number \
                         beyond it happened to be",
                        frame.page_size,
                        wanted - 1,
                        frame.page_size,
                    ),
                });
            }
            if to > frame.kv_page_indices.len() {
                return Err(crate::error::Error::Create {
                    what: "fire tables",
                    message: format!(
                        "request {r}'s page run ends at {to}, past the {} \
                         entries kv_page_indices holds",
                        frame.kv_page_indices.len()
                    ),
                });
            }
        }
    }
    let mut blob: Vec<u32> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    // The ORDER is the contract, and it is this list — `Staged::at` indexes
    // into it and nothing else may.
    for table in [
        frame.token_ids,
        frame.position_ids,
        frame.req_of_token,
        frame.kv_page_indices,
        frame.kv_page_indptr,
        frame.kv_write_page,
        frame.kv_write_offset,
        frame.sampling_indices,
    ] {
        spans.push((blob.len(), table.len()));
        blob.extend_from_slice(table);
    }
    // The attention-mask ENABLE flag, one zero byte per token, always staged.
    //
    // The shader reads `attention_mask_enabled[row]` unconditionally -- it is
    // what decides whether the mask applies -- and an unanswered fire table
    // binds ADDRESS ZERO. So the branch that decides "is this row masked" was
    // reading whatever page zero held: usually zero, and a fire that read a
    // non-zero byte would then index a mask at address zero too and drop
    // whatever keys it disagreed with. Nothing faults; the answer is just
    // wrong, occasionally.
    //
    // Staged as zeros rather than refused because this driver serves CAUSAL
    // attention, and "no row is masked" is the true answer for every fire it
    // can build. A user mask has no path to reach here at all -- `Frame` has
    // no field for one -- so the flag is not yet a variable.
    //
    // One `u32` per token covers `n_tokens` bytes with room to spare, and the
    // stride is the mask's, which is zero-width here.
    let enable_words = frame.token_ids.len().max(1);
    spans.push((blob.len(), enable_words));
    blob.extend(std::iter::repeat_n(0u32, enable_words));
    // Index NINE: the recurrent slot per token, after the enable flag because
    // the flag took eight and the order here is the contract `Staged::at`
    // reads. Empty for a stack with no linear layers.
    spans.push((blob.len(), frame.recurrent_slots.len()));
    blob.extend_from_slice(frame.recurrent_slots);
    // Index TEN: the request CSR, which only the prefill lane's attention
    // reads.
    spans.push((blob.len(), frame.qo_indptr.len()));
    blob.extend_from_slice(frame.qo_indptr);
    // Index ELEVEN: `row_valid`, ONE BYTE PER ROW and every row of a fire this
    // driver builds is valid.
    //
    // Packed four to a word, which is the whole subtlety: a word per row would
    // put `01 00 00 00` under row 0 and hand rows 1, 2 and 3 a zero — "this
    // row is not valid" — and the kernel would drop three quarters of a
    // prefill without failing. The declared element is `i32` and the buffer
    // must not be; `walk::fire`'s `runtime` states `Dt::U8` for exactly this.
    //
    // The span is in words, so the length `Staged::at` reports rounds UP to
    // the word — three bytes past the rows, inside a region that holds them.
    spans.push((blob.len(), frame.token_ids.len().div_ceil(4)));
    blob.extend(std::iter::repeat_n(
        0x0101_0101u32,
        frame.token_ids.len().div_ceil(4),
    ));
    let region = scratch.take(context, ((blob.len() * 4).max(4)) as u64, "fire tables")?;
    // SAFETY: leased for this fire, and no fire that could still be reading a
    // previous lease of it is in flight -- `Scratch` hands a region back only
    // after the `InFlight` holding it has dropped, which is after the step
    // that bound it retired. Every byte the spans name is written below.
    unsafe {
        let raw = core::slice::from_raw_parts(blob.as_ptr().cast::<u8>(), blob.len() * 4);
        region.write(0, raw)?;
    }
    Ok(Staged { region, spans })
}

#[cfg(test)]
// A device test that finds no device SAYS so and passes. Silence would make
// "the machine has no Metal 4" and "the table staged correctly" the same
// observation, which is the failure mode a skip is meant to avoid.
#[allow(clippy::print_stderr)]
mod tests {
    use super::*;

    /// **`row_valid` IS ONE BYTE PER ROW AND THE FOURTH ROW IS THE TEST.**
    ///
    /// A word per row is the plausible packing and it is wrong in a way no
    /// fire would report: rows 1, 2 and 3 would read the zero bytes of row 0's
    /// word and be dropped as invalid, so a five-row prefill would compute one
    /// row and answer fluently. Five rows is the shortest fire that crosses
    /// the boundary in both directions -- one full word and one partial.
    #[test]
    fn row_valid_is_a_byte_a_row_and_not_a_word_a_row() {
        let Ok(context) = Context::new() else {
            crate::skip::skipped("no Metal 4 device");
            return;
        };
        let ids = [7u32, 8, 9, 10, 11];
        let staged = stage(
            &context,
            &Scratch::new(),
            Frame {
                token_ids: &ids,
                qo_indptr: &[0, 5],
                ..Frame::default()
            },
        )
        .expect("it stages");
        let valid = staged.at(FireTable::RowValid).expect("five rows are valid");
        // Two words for five rows: the span rounds UP, inside a region that
        // holds them.
        assert_eq!(valid.bytes, 8);
        // THROUGH THE REGION AND NOT THROUGH THE ADDRESS. A `Slice::address`
        // is a GPU address; on this plane it is not a host pointer, and
        // dereferencing it is a SIGBUS rather than a wrong number.
        let at = (valid.address - staged.region().gpu_address()) as usize;
        // SAFETY: shared storage the line above just wrote, and `at + 5` is
        // inside the span `valid.bytes` names.
        let bytes = unsafe {
            core::slice::from_raw_parts(staged.region().contents().as_ptr().cast::<u8>().add(at), 5)
        };
        assert_eq!(bytes, &[1, 1, 1, 1, 1], "a row of this fire is not valid");

        let csr = staged.at(FireTable::QoIndptr).expect("one request");
        assert_eq!(csr.bytes, 8, "the CSR is one more than the request count");
    }

    #[test]
    fn a_table_this_fire_has_none_of_answers_nothing() {
        let Ok(context) = Context::new() else {
            crate::skip::skipped("no Metal 4 device");
            return;
        };
        let ids = [7u32, 8, 9];
        let staged = stage(
            &context,
            &Scratch::new(),
            Frame {
                token_ids: &ids,
                ..Frame::default()
            },
        )
        .expect("it stages");
        let tokens = staged.at(FireTable::TokenIds).expect("the ids are there");
        assert_eq!(tokens.bytes, 12);
        assert_eq!(tokens.address, staged.region.gpu_address());
        assert!(
            staged.at(FireTable::SamplingIndices).is_none(),
            "an empty table is a slot nobody fills, not an empty region"
        );

        // The attention-mask ENABLE flag always resolves, and reads zero.
        //
        // `sdpa_paged.metal` reads `attention_mask_enabled[row]`
        // unconditionally -- it is the branch that decides whether a row is
        // masked -- and an unanswered fire table binds ADDRESS ZERO. So this
        // was reading whatever page zero held; a non-zero byte there would
        // then index a mask at address zero too and drop whatever keys it
        // disagreed with. Nothing faults and the answer is just wrong,
        // occasionally, which is the shape of defect this crate keeps finding.
        let enable = staged
            .at(FireTable::AttentionMaskEnabled)
            .expect("the enable flag is always staged, or the shader reads address zero");
        assert!(
            enable.bytes >= ids.len() as u64,
            "one flag per token at least"
        );
        // SAFETY: the region is host-addressable and nothing is encoded
        // against it.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                (enable.address - staged.region.gpu_address()
                    + staged.region.contents().as_ptr() as u64) as *const u8,
                enable.bytes as usize,
            )
        };
        assert!(
            bytes.iter().all(|&b| b == 0),
            "this driver serves CAUSAL attention, so every row's flag must read \
             zero -- a non-zero byte here masks a row against a mask that does \
             not exist"
        );
    }

    #[test]
    fn every_table_starts_where_the_one_before_it_ended() {
        let Ok(context) = Context::new() else {
            crate::skip::skipped("no Metal 4 device");
            return;
        };
        let (a, b) = ([1u32, 2], [3u32, 4, 5]);
        let staged = stage(
            &context,
            &Scratch::new(),
            Frame {
                token_ids: &a,
                position_ids: &b,
                ..Frame::default()
            },
        )
        .expect("it stages");
        let t = staged.at(FireTable::TokenIds).expect("tokens");
        let p = staged.at(FireTable::Positions).expect("positions");
        assert_eq!(p.address, t.address + t.bytes, "packed, in order");
    }

    /// A page run that does not reach the last position is refused, and a
    /// `page_size` of zero disarms nothing that was armed.
    ///
    /// The bug this is the gate for: `kv_page_indices: &[0]`,
    /// `kv_page_indptr: &[0, 1]` — one page for a whole sequence — staged by
    /// a rig whose every fixture was five to seven tokens long. With a 16-row
    /// page it is exactly right up to sixteen tokens and a read past the end
    /// of the list at seventeen, and what lies past the end is a number the
    /// kernel spends as a PAGE. So the fire addresses a real page belonging
    /// to nobody: no fault, no uninitialised memory, just keys from somewhere
    /// else. Five isolated runs of one 20-token prompt answered `1124, 1124,
    /// 11, 220, 431`.
    ///
    /// Sixteen and seventeen are the two lengths here for that reason. One
    /// page is a correct staging at the first and a defect at the second, and
    /// a check that only refuses the obviously-too-short would have passed
    /// the fire that started this.
    #[test]
    fn a_page_run_short_of_the_last_position_is_refused() {
        let Ok(context) = Context::new() else {
            crate::skip::skipped("no Metal 4 device");
            return;
        };
        let one_page = [0u32];
        let ends = [0u32, 1];
        let frame = |positions: &'static [u32]| Frame {
            token_ids: positions,
            position_ids: positions,
            req_of_token: &[0; 17],
            kv_page_indices: &one_page,
            kv_page_indptr: &ends,
            page_size: 16,
            ..Frame::default()
        };

        // Sixteen rows fill the page exactly, and one page is the truth.
        let fits: &[u32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        stage(&context, &Scratch::new(), frame(fits)).expect("sixteen rows fit one 16-row page");

        // Seventeen do not, and this is the position the whole story turns on.
        let over: &[u32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let why = stage(&context, &Scratch::new(), frame(over))
            .expect_err("position 16 is the second page of a 16-row run")
            .to_string();
        assert!(
            why.contains("16") && why.contains("kv_page_indices"),
            "the refusal has to name the position and the table it would have \
             read past, or a reader cannot act on it: {why}"
        );

        // UNSTATED page size, and the check does not run. This is the arm
        // that makes the refusal adoptable by a caller that has no pages at
        // all, and it is asserted because a default of zero that silently
        // refused every fire would be the worse failure of the two.
        stage(
            &context,
            &Scratch::new(),
            Frame {
                page_size: 0,
                ..frame(over)
            },
        )
        .expect("no page size stated is no claim about pages to check");
    }

    #[test]
    fn a_seat_table_short_of_the_rows_is_refused_rather_than_read_past() {
        let Ok(context) = Context::new() else {
            crate::skip::skipped("no Metal 4 device");
            return;
        };
        let tokens = [1u32, 2, 3, 4];
        let one_seat = [0u32];
        let why = stage(
            &context,
            &Scratch::new(),
            Frame {
                token_ids: &tokens,
                recurrent_slots: &one_seat,
                ..Frame::default()
            },
        )
        .expect_err("one seat for four rows is a read past this region");
        let said = why.to_string();
        assert!(
            said.contains('1') && said.contains('4'),
            "the refusal has to carry both lengths, or it does not say what is wrong: {said}"
        );

        // Per token, and then it stages.
        let seats = [0u32; 4];
        stage(
            &context,
            &Scratch::new(),
            Frame {
                token_ids: &tokens,
                recurrent_slots: &seats,
                ..Frame::default()
            },
        )
        .expect("one seat per row stages");
    }
}
