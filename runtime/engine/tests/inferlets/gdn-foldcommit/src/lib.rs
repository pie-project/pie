//! **FOLD-COMMIT on a linear model** — the speculative-commit shape
//! `model.wit`'s `pass-kind()` exists to select.
//!
//! A linear/SSM model folds tokens into its recurrent state IRREVERSIBLY, so
//! the KV trick of discarding rejected slots does not exist for it. The
//! answer the runtime declares is fold-commit: run the uncertain tokens with
//! the fold SUPPRESSED, holding their pre-recurrence activations in the RS
//! working set's buffered slots, and afterwards fold only the accepted
//! prefix. A rejected tail is simply never folded — abandoning it costs one
//! `free-buffer`.
//!
//! This inferlet drives all three modes end to end:
//!
//!  1. **prefill — `fold`** (the default): materializes the folded state.
//!     Buffering and folding both read it, so this must come first.
//!  2. **speculate — `fold_len = 0`**: one chunk of `SPEC_TOKENS` tokens
//!     appended to the buffer; the folded boundary does not move, so the
//!     chunk stays abandonable.
//!  3. **commit — `fold_len = accepted`**: advances the boundary through the
//!     accepted prefix and drops the fully covered head slots.
//!
//! Input: the number of speculative tokens to accept (default all of them),
//! e.g. `"2"`. Output names the three phases so a harness can assert them.
//!
//! The special input `"chain"` accepts 2 and then buffers a SECOND chunk onto
//! the surviving unfolded tail, without emptying the buffer in between. That
//! second append is the buffer READ PATH: its tokens must recur from
//! `folded ⊕ replay(buffer)`, not from the folded state alone. The runtime
//! refuses it today, and `cuda_gdn_foldcommit.rs` pins that refusal — the mode
//! exists so the read path has an executable acceptance test rather than a
//! prose description.

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

/// Tokens written in the buffered (speculative) chunk.
const SPEC_TOKENS: u32 = 4;

/// One arm of the `inside` comparison: a fresh context, prefilled with the
/// prompt, then some number of fires over the SAME four continuation tokens.
/// Everything is parameterised so the two arms differ only in how the fires
/// are cut, never in what they compute.
struct Arm {
    ws: WorkingSet,
    rs: RsWorkingSet,
    page_size: u32,
    rs_page: u32,
    max_pages: u32,
    pos: u32,
}

impl Arm {
    /// Prefill `prompt` with the default fold, returning the greedy token.
    async fn open(prompt: &[u32], max_pages: u32, pipe: &Pipeline) -> Result<(Self, i32)> {
        let ws = WorkingSet::new();
        let rs = RsWorkingSet::new();
        let page_size = kv_page_size();
        ws.reserve(max_pages).context("ws.reserve")?;
        let mut arm = Arm {
            ws,
            rs,
            page_size,
            rs_page: inferlet::model::rs_buffer_page_size().max(1),
            max_pages,
            pos: 0,
        };
        let (g0, _) = arm.fire(prompt, None, pipe, "prefill").await?;
        arm.pos = prompt.len() as u32;
        Ok((arm, g0.unwrap_or(0)))
    }

    /// One fire over `toks` whose fold length comes from a CHANNEL the caller
    /// owns, with a caller-supplied epilogue.
    ///
    /// The channel is the point: `fold_len` may carry no host-known seed at
    /// all, in which case a stage computes it on device and only the driver
    /// ever resolves the value. `None` is the plain in-forward fold.
    /// CONSTRUCTS the pass without submitting it. A channel a later pass
    /// consumes as a descriptor must be claimed by that pass BEFORE the
    /// producing pass is submitted (the SDK's F8 eager claim), or the producer
    /// infers it as a terminal host-read output and the two declarations
    /// conflict at bind. So a device handoff needs the consumer BUILT first —
    /// construction order, not annotation.
    fn build_fire(
        &self,
        toks: &[u32],
        fold_len: Option<&Channel>,
        tag: &str,
        epilogue: impl Fn() + 'static,
    ) -> Result<ForwardPass> {
        let t = toks.len() as u32;
        let base = self.pos;
        let end = base + t;
        let ps = self.page_size;
        let ch = |v: Vec<u32>| Channel::from(v);

        let fwd = ForwardPass::new();
        fwd.embed(
            &Channel::from_iter(toks.iter().map(|&x| x as i32)),
            &ch(vec![0, t]),
        )?;
        let kv_len = ch(vec![end]);
        let pages = ch((0..self.max_pages).collect());
        let page_indptr = ch(vec![0, end.div_ceil(ps)]);
        let w_slot = ch((base..end).map(|p| p / ps).collect());
        let w_off = ch((base..end).map(|p| p % ps).collect());
        let positions = ch((base..end).collect());
        let kv = || {
            Some(KvBinding {
                working_set: &self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            })
        };
        match fold_len {
            None => fwd.attention(
                kv(),
                std::slice::from_ref(&self.rs),
                RsGeometry {
                    fold_len: None,
                    buffer: 0..0,
                },
            )?,
            Some(channel) => fwd
                .attention(
                    kv(),
                    std::slice::from_ref(&self.rs),
                    RsGeometry {
                        fold_len: Some(channel),
                        buffer: ..,
                    },
                )
                .with_context(|| format!("{tag} state binding"))?,
        }
        // An EMPTY epilogue, not an absent one: a pass with no stages has no
        // PTIR program at all. A commit samples nothing, so empty is right.
        fwd.epilogue(epilogue);
        Ok(fwd)
    }

    /// A fire whose row spans NO TOKENS.
    ///
    /// The IR has no zero-sized tensor (`Shape::new` refuses a 0 dim), so the
    /// emptiness cannot live in the channels. It lives where it belongs
    /// anyway: in the GEOMETRY. Every indptr declares a span of zero, and the
    /// channels carry one unreferenced element to stay well-formed.
    fn build_empty_fire(&self, fold_len: &Channel, tag: &str) -> Result<ForwardPass> {
        let base = self.pos;
        let ps = self.page_size;
        let ch = |v: Vec<u32>| Channel::from(v);

        let fwd = ForwardPass::new();
        fwd.embed(&Channel::from([0i32]), &ch(vec![0, 0]))?;
        let kv_len = ch(vec![base]);
        let pages = ch((0..self.max_pages).collect());
        let page_indptr = ch(vec![0, base.div_ceil(ps)]);
        let w_slot = ch(vec![0]);
        let w_off = ch(vec![0]);
        let positions = ch(vec![0]);
        let kv = || {
            Some(KvBinding {
                working_set: &self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            })
        };
        fwd.attention(
            kv(),
            std::slice::from_ref(&self.rs),
            RsGeometry {
                fold_len: Some(fold_len),
                buffer: ..,
            },
        )
        .with_context(|| format!("{tag} state binding"))?;
        fwd.epilogue(|| {});
        Ok(fwd)
    }

    /// One fire over `toks`.
    ///
    /// `buffered` is `Some((buffer_start, fold_len))`: the fire scatters its
    /// tokens into the buffer starting at logical token `buffer_start` and
    /// folds `fold_len` of the resulting buffer. `fold_len == 0` is a pure
    /// append; `0 < fold_len < buffer_start + toks.len()` is the boundary
    /// landing strictly INSIDE this fire's own tokens. `None` is the plain
    /// in-forward fold that touches no buffer at all.
    async fn fire(
        &self,
        toks: &[u32],
        buffered: Option<(u32, u32)>,
        pipe: &Pipeline,
        tag: &str,
    ) -> Result<(Option<i32>, f32)> {
        let t = toks.len() as u32;
        let base = self.pos;
        let end = base + t;
        let ps = self.page_size;
        let ch = |v: Vec<u32>| Channel::from(v);

        let fwd = ForwardPass::new();
        fwd.embed(
            &Channel::from_iter(toks.iter().map(|&x| x as i32)),
            &ch(vec![0, t]),
        )?;
        let kv_len = ch(vec![end]);
        let pages = ch((0..self.max_pages).collect());
        let page_indptr = ch(vec![0, end.div_ceil(ps)]);
        let w_slot = ch((base..end).map(|p| p / ps).collect());
        let w_off = ch((base..end).map(|p| p % ps).collect());
        let positions = ch((base..end).collect());
        let kv = || {
            Some(KvBinding {
                working_set: &self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            })
        };
        match buffered {
            None => fwd.attention(
                kv(),
                std::slice::from_ref(&self.rs),
                RsGeometry {
                    fold_len: None,
                    buffer: 0..0,
                },
            )?,
            Some((_start, fold_len)) => {
                // Where the tokens land in the buffer is not stated: new
                // tokens append at the tail, which only the runtime knows.
                fwd.attention(
                    kv(),
                    std::slice::from_ref(&self.rs),
                    RsGeometry {
                        fold_len: Some(&ch(vec![fold_len])),
                        buffer: ..,
                    },
                )
                .with_context(|| format!("{tag} state binding"))?;
            }
        }
        let out = Channel::new([1], dtype::i32).named("arm_out");
        let peak = Channel::new([1], dtype::f32).named("arm_peak");
        let sink = out.clone();
        let peak_sink = peak.clone();
        fwd.epilogue(move || {
            let logits = intrinsics::logits();
            sink.put(&reduce_argmax(&logits));
            // The greedy token is far too coarse to witness a state
            // difference on this model -- it has strong attractors and two
            // genuinely different states routinely pick the same argmax --
            // whereas the peak logit VALUE moves with the state continuously.
            peak_sink.put(&reduce_max(&logits));
        });
        fwd.submit(pipe).with_context(|| format!("{tag} submit"))?;
        let token = out
            .take_host::<i32>()
            .await
            .with_context(|| format!("{tag} take"))?;
        let peak_v = peak
            .take_host::<f32>()
            .await
            .with_context(|| format!("{tag} peak take"))?;
        Ok((Some(token), peak_v))
    }
}

/// TWO RS working sets in ONE fire, landing their folded boundaries in
/// DIFFERENT places: row 0 only appends to its buffer, row 1 folds outright.
///
/// This is the shape the planner used to refuse ("a fire folds uniformly
/// today") and the one real serving wants: a request committing while another
/// speculates. The two rows run the identical dispatch -- same initial state,
/// same layout, same outputs -- and disagree only about whether the recurrence
/// PERSISTS, which now travels as a per-row device mask instead of a per-pass
/// flag.
///
/// The check is an equivalence against the same two shapes run SOLO, drawn one
/// token PAST the mixed fire so it pins the resulting states rather than the
/// logits along the way. It is only meaningful because the two references
/// disagree with each other -- a buffered row's state does not move, a folded
/// row's does -- so the test asserts that first. A mask that was ignored in
/// either direction collapses one row onto the wrong reference.
/// How far two peak logits may drift and still count as the same state, as a
/// fraction of their magnitude. This is slack against reduction-order noise,
/// not against a real difference: the two sides run the same math on the same
/// inputs, but not always in the same BATCH SHAPE -- a solo reference fire and
/// a two-row mixed fire reduce in different orders.
///
/// It has to be relative because the activations are bf16, whose ULP at a peak
/// logit of ~14 is 0.0625. An absolute tolerance below that is finer than the
/// format can represent, so it fails on a single ULP of legitimate noise while
/// claiming a state divergence.
const TOL: f32 = 1e-2;

/// `a` and `b` are the same value up to `TOL` of their magnitude (floored at
/// 1, so small logits keep an absolute tolerance).
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL * a.abs().max(b.abs()).max(1.0)
}

/// TWO recurrent contexts in ONE fire, so a single pass can land their folded
/// boundaries in DIFFERENT places.
///
/// Both rows share a KV working set but own DISJOINT page ranges, so each row
/// is an independent context that happens to be batched. The rows carry the
/// same tokens at the same positions; the only thing that ever differs between
/// them is what the fire asks their recurrence to do.
struct Duo {
    ws: WorkingSet,
    rs: [RsWorkingSet; 2],
    page_size: u32,
    rs_page: u32,
    /// Pages per ROW. Row r owns `[r * span, (r + 1) * span)`.
    span: u32,
    pos: u32,
}

impl Duo {
    fn new(span: u32) -> Result<Self> {
        let ws = WorkingSet::new();
        let page_size = kv_page_size();
        ws.reserve(2 * span).context("duo reserve")?;
        let rs = [RsWorkingSet::new(), RsWorkingSet::new()];
        Ok(Duo {
            ws,
            rs_page: inferlet::model::rs_buffer_page_size().max(1),
            rs,
            page_size,
            span,
            pos: 0,
        })
    }

    /// One two-row fire over `toks` (the SAME tokens on both rows).
    ///
    /// `buffered[r]` is `Some((start, fold_len))` for a row that scatters into
    /// its buffer, `None` for the plain in-forward fold. A fire whose two
    /// entries disagree is the mixed pass this whole path exists for.
    ///
    /// Returns each row's peak logit. The greedy token is far too coarse to
    /// witness a state difference on this model -- it has strong attractors
    /// and two genuinely different states routinely pick the same argmax --
    /// whereas the peak logit VALUE moves with the state continuously.
    async fn fire(
        &self,
        toks: &[u32],
        buffered: [Option<(u32, u32)>; 2],
        pipe: &Pipeline,
        tag: &str,
    ) -> Result<[f32; 2]> {
        let t = toks.len() as u32;
        let base = self.pos;
        let end = base + t;
        let ps = self.page_size;
        let row_pages = end.div_ceil(ps);
        let ch = |v: Vec<u32>| Channel::from(v);

        let fwd = ForwardPass::new();
        let embed: Vec<i32> = (0..2)
            .flat_map(|_| toks.iter().map(|&x| x as i32))
            .collect();
        fwd.embed(&Channel::from(embed), &ch(vec![0, t, 2 * t]))?;

        // Row r reads and writes only its own page range; `w-slot` is the
        // index INTO THAT ROW'S page list, so both rows use the identical
        // row-relative descriptor and differ only in which pages the list
        // names.
        let pages: Vec<u32> = (0..2)
            .flat_map(|r| (0..row_pages).map(move |p| r * self.span + p))
            .collect();
        let slots: Vec<u32> = (base..end).map(|p| p / ps).collect();
        let offs: Vec<u32> = (base..end).map(|p| p % ps).collect();
        let kv_len = ch(vec![end, end]);
        let pages = Channel::from(pages);
        let page_indptr = ch(vec![0, row_pages, 2 * row_pages]);
        let w_slot = ch([slots.clone(), slots].concat());
        let w_off = ch([offs.clone(), offs].concat());
        let positions = ch([(base..end).collect::<Vec<_>>(), (base..end).collect()].concat());
        let kv = || {
            Some(KvBinding {
                working_set: &self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            })
        };

        if buffered.iter().all(Option::is_none) {
            fwd.attention(
                kv(),
                &self.rs,
                RsGeometry {
                    fold_len: None,
                    buffer: 0..0,
                },
            )?;
        } else {
            let rp = self.rs_page;
            let mut fold_len = Vec::with_capacity(2);
            let mut buffer_len = Vec::with_capacity(2);
            let mut buffer_pages: Vec<u32> = Vec::new();
            let mut buffer_indptr = vec![0u32];
            let mut w_slot: Vec<u32> = Vec::new();
            let mut w_off: Vec<u32> = Vec::new();
            for entry in buffered {
                match entry {
                    // A row that folds in-forward owns no buffer in this pass:
                    // it declares zero buffer tokens and no pages, and asks for
                    // the fire-invariant "fold everything".
                    None => {
                        fold_len.push(u32::MAX);
                        buffer_len.push(0);
                        buffer_indptr.push(buffer_pages.len() as u32);
                    }
                    Some((buf_start, fold)) => {
                        let first = buf_start / rp;
                        let last = (buf_start + t - 1) / rp;
                        fold_len.push(fold);
                        buffer_len.push(t);
                        for p in first..=last {
                            buffer_pages.push(p);
                        }
                        buffer_indptr.push(buffer_pages.len() as u32);
                        for x in buf_start..buf_start + t {
                            w_slot.push(x / rp - first);
                            w_off.push(x % rp);
                        }
                    }
                }
            }
            fwd.attention(
                kv(),
                &self.rs,
                RsGeometry {
                    fold_len: Some(&ch(fold_len)),
                    buffer: ..,
                },
            )
            .with_context(|| format!("{tag} state binding"))?;
        }

        // The default read-out is each lane's last row, so `logits` is
        // `[2, vocab]` and both reductions come back per row.
        let peak = Channel::new([2], dtype::f32).named("duo_peak");
        let sink = peak.clone();
        fwd.epilogue(move || {
            sink.put(&reduce_max(intrinsics::logits()));
        });
        fwd.submit(pipe).with_context(|| format!("{tag} submit"))?;
        let v = peak
            .take_host::<Vec<f32>>()
            .await
            .with_context(|| format!("{tag} take"))?;
        Ok([v[0], v[1]])
    }
}

/// One fire landing its two rows' folded boundaries in DIFFERENT places: row 0
/// only appends to its buffer, row 1 folds outright.
///
/// This is the shape the planner used to refuse ("a fire folds uniformly
/// today") and the one real serving wants -- a request committing while
/// another speculates. The two rows run the identical dispatch over the
/// identical layout and disagree only about whether the recurrence PERSISTS,
/// which now travels as a per-row device mask instead of a per-pass flag.
///
/// The check is an equivalence against the SAME two-row fire run uniformly:
/// one duo buffers on both rows, one folds on both rows, and the mixed duo
/// must match the buffering reference on row 0 and the folding reference on
/// row 1. Every shape is identical -- same batch, same pages, same tokens --
/// so the only variable is the mask.
///
/// It asserts up front that the two uniform references DISAGREE with each
/// other. Without that the equivalence holds vacuously and a mask ignored in
/// either direction would go unnoticed.
async fn mixed_positions(prompt: &[u32]) -> Result<String> {
    // The rows differ ONLY in their recurrent state -- same tokens, same KV
    // positions -- so the continuation must be long and varied enough that the
    // difference moves the logits. A short repeated one does not.
    let cont: Vec<u32> = {
        let c = wit_model::encode(" quick brown fox jumps over the lazy dog and runs away");
        if c.is_empty() {
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        } else {
            c
        }
    };
    let t = cont.len() as u32;
    let n = prompt.len() as u32;
    let pipe = Pipeline::new();

    let span = (n + t + 2).div_ceil(kv_page_size()).max(1);
    let buf_pages = |d: &Duo| t.div_ceil(d.rs_page);

    // Three duos, identical in every way except what the middle fire asks of
    // each row: uniformly buffering, uniformly folding, and mixed.
    let mut duos = Vec::new();
    for _ in 0..3 {
        duos.push(Duo::new(span)?);
    }
    let next = {
        let mut peaks: Vec<[f32; 2]> = Vec::new();
        for (i, duo) in duos.iter_mut().enumerate() {
            duo.fire(prompt, [None, None], &pipe, "prefill").await?;
            duo.pos = n;
            let shapes: [Option<(u32, u32)>; 2] = match i {
                0 => [Some((0, 0)), Some((0, 0))],
                1 => [None, None],
                _ => [Some((0, 0)), None],
            };
            for (row, shape) in shapes.iter().enumerate() {
                if shape.is_some() {
                    duo.rs[row]
                        .alloc_buffer(buf_pages(duo))
                        .with_context(|| format!("duo {i} row {row} alloc_buffer"))?;
                }
            }
            duo.fire(&cont, shapes, &pipe, "middle").await?;
            duo.pos += t;
            // Abandon whatever was buffered, so the follow-up fire sees an
            // empty buffer on every row of every duo.
            for (row, shape) in shapes.iter().enumerate() {
                if shape.is_some() {
                    duo.rs[row]
                        .free_buffer(&[0])
                        .with_context(|| format!("duo {i} row {row} free_buffer"))?;
                }
            }
            peaks.push(duo.fire(&[cont[0]], [None, None], &pipe, "next").await?);
        }
        peaks
    };
    pipe.close();

    let (buf_ref, fold_ref, mixed) = (next[0], next[1], next[2]);
    let spread = (fold_ref[0] - buf_ref[0]).abs();
    // The references must disagree by much more than the agreement tolerance,
    // or "agree" carries no information. 8x, measured on the same relative
    // scale `close` uses.
    if spread <= 8.0 * TOL * fold_ref[0].abs().max(buf_ref[0].abs()).max(1.0) {
        return Err(format!(
            "buffering and folding leave indistinguishable states \
             (buffer={:.4} fold={:.4}); this prompt cannot witness a per-row fold \
             and the test would pass vacuously",
            buf_ref[0], fold_ref[0]
        ));
    }
    let agree = close(mixed[0], buf_ref[0]) && close(mixed[1], fold_ref[1]);
    let result = format!(
        "mixed tokens={t} row0={:.4} buf_ref={:.4} row1={:.4} fold_ref={:.4} \
         spread={spread:.4} agree={}",
        mixed[0],
        buf_ref[0],
        mixed[1],
        fold_ref[1],
        if agree { "yes" } else { "no" }
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    if !agree {
        return Err(result);
    }
    Ok(result)
}

/// A fold running THROUGH a non-empty buffer, in the same fire that fills it.
///
/// Arm A appends two tokens onto a buffer that already holds two, and folds
/// all four in that one fire. The buffered pair has to be replayed ahead of
/// the new pair (the read path), and the folded boundary has to land past
/// both -- so this is a write and a fold at once, over the extended layout.
///
/// Arm B computes the same four tokens the long way: fold the first two
/// outright, then fold the last two outright. Both arms must agree, and
/// because the comparison is drawn AFTER the fold -- each arm continues from
/// its own resulting state -- agreement pins the folded state itself, not
/// merely the logits along the way.
async fn fold_inside_new_tokens(prompt: &[u32]) -> Result<String> {
    const T: usize = 4;
    const HALF: u32 = 2;

    let n = prompt.len() as u32;
    let pipe = Pipeline::new();
    let probe = WorkingSet::new();
    let max_pages = (n + 2 * T as u32 + 1).div_ceil(kv_page_size());
    drop(probe);

    let (mut a, g0) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut b, g0b) = Arm::open(prompt, max_pages, &pipe).await?;
    if g0 != g0b {
        return Err(format!("arms disagree on the prefill token: {g0} vs {g0b}"));
    }
    let cont: Vec<u32> = vec![g0 as u32; T];

    // Arm A: buffer the first pair, then append the second pair AND fold all
    // four -- the write-and-fold through a non-empty buffer.
    a.rs.alloc_buffer((T as u32).div_ceil(a.rs_page))
        .context("A alloc_buffer")?;
    a.fire(&cont[..HALF as usize], Some((0, 0)), &pipe, "A-buffer")
        .await?;
    a.pos += HALF;
    let a_last = a
        .fire(
            &cont[HALF as usize..],
            Some((HALF, T as u32)),
            &pipe,
            "A-append-and-fold",
        )
        .await?
        .0
        .unwrap_or(0);
    a.pos += HALF;

    // Arm B: the same four tokens, folded two fires at a time.
    b.fire(&cont[..HALF as usize], None, &pipe, "B-fold-1")
        .await?;
    b.pos += HALF;
    let b_last = b
        .fire(&cont[HALF as usize..], None, &pipe, "B-fold-2")
        .await?
        .0
        .unwrap_or(0);
    b.pos += HALF;

    // Now compare the STATES the two arms arrived at, by continuing each one
    // token further. If arm A's fold had missed the buffered pair (or replayed
    // it twice) this is where it shows.
    let next = vec![b_last as u32];
    let a_next = a.fire(&next, None, &pipe, "A-next").await?.0.unwrap_or(0);
    let b_next = b.fire(&next, None, &pipe, "B-next").await?.0.unwrap_or(0);

    pipe.close();

    let agree = a_last == b_last && a_next == b_next;
    let result = format!(
        "foldthrough tokens={T} buffered={HALF} a_last={a_last} b_last={b_last} \
         a_next={a_next} b_next={b_next} agree={}",
        if agree { "yes" } else { "no" }
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    if !agree {
        return Err(result);
    }
    Ok(result)
}

/// The fold boundary lands BEHIND the fire's own new tokens.
///
/// `0 < n < b` with `t > 0` — the shape the north star needs. A speculative
/// decoder learns window `k`'s accepted length only after window `k`'s verify
/// has run, so it cannot fold that prefix in the fire that produced it. It
/// CAN fold it in the fire that writes window `k + 1`, which collapses the
/// steady state from two fires per window to one.
///
/// The planner used to refuse this, not deliberately but by accident: it
/// classified a row as a pure commit whenever `n <= b`, and this shape
/// satisfies that condition while meaning the opposite. Now that a commit is
/// a row carrying NO TOKENS, `n <= b` is free to mean what it says, and the
/// driver's 2R-segment split already handles a cut anywhere in `(0, b + t)`.
///
/// Arm A folds behind while writing ahead in ONE fire. Arm B does the same
/// two things in two fires — an empty commit, then a plain append. Their
/// written tokens' logits and their resulting state must both agree. Arm C
/// folds a DIFFERENT prefix in its single fire; folding is a no-op, so it too
/// must converge, and the assertion is that the boundary's placement never
/// changes what the context means.
async fn fold_behind_new_tokens(prompt: &[u32]) -> Result<String> {
    const W: usize = 4;
    const T: usize = 3;
    const BEHIND: u32 = 2;

    let n = prompt.len() as u32;
    let pipe = Pipeline::new();
    let probe = WorkingSet::new();
    let max_pages = (n + 3 * (W + T) as u32).div_ceil(kv_page_size());
    drop(probe);

    let (mut a, g0) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut b, _) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut c, _) = Arm::open(prompt, max_pages, &pipe).await?;

    // Distinct tokens: a window of repeats makes a fold LENGTH nearly
    // unobservable, which has silently emptied three earlier tests.
    let window = wit_model::encode("alpha beta gamma delta");
    let window: Vec<u32> = if window.len() < W {
        vec![g0 as u32; W]
    } else {
        window[..W].to_vec()
    };
    let ahead = wit_model::encode("epsilon zeta eta");
    let ahead: Vec<u32> = if ahead.len() < T {
        vec![g0 as u32; T]
    } else {
        ahead[..T].to_vec()
    };
    let slabs = ((W + T) as u32 + 1).div_ceil(a.rs_page).max(1);

    for (arm, tag) in [(&a, "A"), (&b, "B"), (&c, "C")] {
        arm.rs
            .alloc_buffer(slabs)
            .with_context(|| format!("{tag} alloc_buffer"))?;
    }

    // Every arm buffers the same window, folding nothing.
    for (arm, tag) in [(&mut a, "A"), (&mut b, "B"), (&mut c, "C")] {
        arm.fire(&window, Some((0, 0)), &pipe, tag).await?;
        arm.pos += W as u32;
    }

    // ── Arm A: fold BEHIND while writing AHEAD, in one fire ───────────────
    let (_, a_ahead) = a.fire(&ahead, Some((0, BEHIND)), &pipe, "A-behind").await?;
    a.pos += T as u32;

    // ── Arm B: the same two things, in two fires ──────────────────────────
    b.build_empty_fire(&Channel::from([BEHIND]), "B-commit")?
        .submit(&pipe)
        .context("B-commit submit")?;
    let (_, b_ahead) = b.fire(&ahead, Some((0, 0)), &pipe, "B-append").await?;
    b.pos += T as u32;

    // ── Arm C: a different boundary behind ────────────────────────────────
    let (_, c_ahead) = c.fire(&ahead, Some((0, 1)), &pipe, "C-behind").await?;
    c.pos += T as u32;

    let next = vec![g0 as u32];
    let mut nexts = Vec::new();
    for (arm, tag) in [(&mut a, "A"), (&mut b, "B"), (&mut c, "C")] {
        arm.fire(&next, Some((0, u32::MAX)), &pipe, tag).await?;
        arm.pos += 1;
        let live = arm.rs.buffer_size();
        if live > 0 {
            arm.rs
                .free_buffer(&(0..live).collect::<Vec<_>>())
                .with_context(|| format!("{tag} free_buffer"))?;
        }
        let (_, peak) = arm.fire(&next, None, &pipe, tag).await?;
        arm.pos += 1;
        nexts.push(peak);
    }
    pipe.close();

    let agree = close(a_ahead, b_ahead)
        && close(a_ahead, c_ahead)
        && close(nexts[0], nexts[1])
        && close(nexts[0], nexts[2]);
    let result = format!(
        "behind a_ahead={a_ahead:.4} b_ahead={b_ahead:.4} c_ahead={c_ahead:.4} \
a_next={:.4} b_next={:.4} c_next={:.4} agree={}",
        nexts[0],
        nexts[1],
        nexts[2],
        if agree { "yes" } else { "no" }
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    if !agree {
        return Err(result);
    }
    Ok(result)
}

/// The fold boundary lands STRICTLY INSIDE the fire's own new tokens.
///
/// `b < n < b + t` — the last RS position the planner refused. `commit_len`
/// cannot express it: the kernels implement it as `if (c < Nr) Nr = c`, which
/// TRUNCATES the sequence, so the tokens past the boundary would get no
/// outputs at all. The driver cuts the row into two segments instead and runs
/// the recurrence twice on one stream — the head persisting its
/// end-of-sequence state onto the boundary, the tail continuing from that
/// state to produce the remaining outputs without moving it again.
///
/// The test pins BOTH halves of that claim, because either can fail alone:
///
///  * **the outputs past the boundary are still correct.** Arm A runs all four
///    continuation tokens in ONE fire folding only the first two. Arm B runs
///    the same four as a two-token in-forward fold followed by a two-token
///    pure append. Their LAST token is the same token at the same position
///    continuing from the same folded state, so its peak logit must match. A
///    truncating implementation produces nothing here; one that folded the
///    whole row produces the wrong thing.
///
///  * **the boundary landed on token 2, not token 4.** Both arms then drain
///    their buffer with a fold-everything fire and probe one token further,
///    which pins the STATE rather than the logits along the way. Arm C folds
///    all four in its single fire — the boundary at the END — and must
///    DISAGREE, or the equivalence is vacuous and an implementation that
///    ignored `n` entirely would pass.
async fn fold_interior_boundary(prompt: &[u32]) -> Result<String> {
    const T: usize = 4;
    const HALF: u32 = 2;

    let n = prompt.len() as u32;
    let pipe = Pipeline::new();
    let probe = WorkingSet::new();
    let max_pages = (n + 2 * T as u32 + 2).div_ceil(kv_page_size());
    drop(probe);

    let (mut a, g0) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut b, g0b) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut c, g0c) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut d, _) = Arm::open(prompt, max_pages, &pipe).await?;
    if g0 != g0b || g0 != g0c {
        return Err(format!(
            "arms disagree on the prefill token: {g0} vs {g0b} vs {g0c}"
        ));
    }
    let cont: Vec<u32> = vec![g0 as u32; T];
    let slabs = (T as u32 + 1).div_ceil(a.rs_page).max(1);

    // ── Arm A: ONE fire over four tokens, folding only the first two ──────
    a.rs.alloc_buffer(slabs).context("A alloc_buffer")?;
    let (_, a_inside) = a
        .fire(&cont, Some((0, HALF)), &pipe, "A-fold-inside")
        .await?;
    a.pos += T as u32;

    // ── Arm C: the SAME fire, boundary one token earlier ──────────────────
    // A different interior cut. `fold(n)` is a no-op for every legal n, so C
    // must converge on A: the boundary may land anywhere inside the fire's
    // own tokens without changing what the context means.
    c.rs.alloc_buffer(slabs).context("C alloc_buffer")?;
    let (_, c_inside) = c.fire(&cont, Some((0, 1)), &pipe, "C-fold-one").await?;
    c.pos += T as u32;

    // ── Arm B: two folded, then two appended — the reference ──────────────
    b.fire(&cont[..HALF as usize], None, &pipe, "B-fold")
        .await?;
    b.pos += HALF;
    b.rs.alloc_buffer(slabs).context("B alloc_buffer")?;
    let (_, b_inside) = b
        .fire(&cont[HALF as usize..], Some((0, 0)), &pipe, "B-append")
        .await?;
    b.pos += HALF;

    // ── Arm D: A's exact STORE state reached the trusted way ─────────────
    // Buffer all four with the boundary held still, then move it two forward
    // with an ordinary commit. D ends with the same `fill=4, head=2` A does,
    // but never uses the split. It separates "the split is wrong" from "a
    // buffer holding its already-folded prefix behaves differently from one
    // that never held it" -- which is what arm B would otherwise conflate.
    d.rs.alloc_buffer(slabs).context("D alloc_buffer")?;
    d.fire(&cont, Some((0, 0)), &pipe, "D-append").await?;
    d.pos += T as u32;
    // An ordinary COMMIT moves the boundary two tokens into an ALREADY
    // buffered span. Its own token is a placeholder at a position the drain
    // fire immediately overwrites; the recurrence ignores those rows and
    // replays the buffer instead.
    d.build_empty_fire(&Channel::from([HALF]), "D-commit")?
        .submit(&pipe)
        .context("D-commit submit")?;

    // Drain: fold whatever is still buffered, then free the slabs so the
    // probe below is a plain in-forward fold on both arms.
    let next = vec![g0 as u32];
    let mut peaks = Vec::new();
    for (arm, tag) in [(&mut a, "A"), (&mut b, "B"), (&mut c, "C"), (&mut d, "D")] {
        // `u32::MAX` is the fire-invariant "fold everything" — it is clamped
        // to this row's own `b + t`, which only the runtime knows.
        arm.fire(&next, Some((0, u32::MAX)), &pipe, tag).await?;
        arm.pos += 1;
        let live = arm.rs.buffer_size();
        if live > 0 {
            arm.rs
                .free_buffer(&(0..live).collect::<Vec<_>>())
                .with_context(|| format!("{tag} free_buffer"))?;
        }
        let (_, peak) = arm.fire(&next, None, &pipe, tag).await?;
        arm.pos += 1;
        peaks.push(peak);
    }
    let (a_next, b_next, c_next, d_next) = (peaks[0], peaks[1], peaks[2], peaks[3]);

    pipe.close();

    // The tail's outputs, and then the state the boundary left behind.
    //
    // There is deliberately NO "different fold lands a different state"
    // control here, unlike the other arms. Folding IS a no-op: every legal
    // boundary has to converge, so any arm that disagreed would be evidence
    // against the model rather than for the test. What gives these assertions
    // teeth is that they caught a real defect — the wire's `buffer_head` was
    // sampled AFTER the fold advance, so a fire whose rows straddle the
    // boundary was handed a head from the other side of it. Before the fix A
    // read `a_next=17.2500` against `b_next=16.8750`, and D — which reaches
    // A's exact store state by a path that predates the split — was wrong too.
    let outputs_agree = close(a_inside, b_inside) && close(a_inside, c_inside);
    let states_agree = close(a_next, d_next) && close(a_next, b_next) && close(a_next, c_next);
    let ok = outputs_agree && states_agree;
    let result = format!(
        "interior tokens={T} fold={HALF} a_inside={a_inside:.4} b_inside={b_inside:.4} \
         c_inside={c_inside:.4} a_next={a_next:.4} b_next={b_next:.4} \
         c_next={c_next:.4} d_next={d_next:.4} outputs={} states={} agree={}",
        if outputs_agree { "ok" } else { "BAD" },
        if states_agree { "ok" } else { "BAD" },
        if ok { "yes" } else { "no" }
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    if !ok {
        return Err(result);
    }
    Ok(result)
}

/// A commit that carries NO TOKENS OF ITS OWN.
///
/// The pure replay -- "I have nothing to compute, only move the boundary" --
/// said directly instead of inferred. The planner used to read it off the
/// incidental fact that `fold-len <= buffered`, which is why a fire could not
/// fold BEHIND its own new tokens while writing them: the same condition
/// meant two different things. A row spanning zero tokens says it
/// unambiguously, and it is the only way to say it -- the IR has no
/// zero-sized tensor, so the emptiness lives in the token CSR and the
/// channels carry an unreferenced tail.
///
/// Arm A folds half its buffer through an empty row; arm B folds nothing at
/// all. Folding is a no-op, so they must decode identically.
async fn empty_commit(prompt: &[u32]) -> Result<String> {
    const T: usize = 4;
    const HALF: u32 = 2;

    let n = prompt.len() as u32;
    let pipe = Pipeline::new();
    let probe = WorkingSet::new();
    let max_pages = (n + 2 * T as u32 + 4).div_ceil(kv_page_size());
    drop(probe);

    let (mut a, g0) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut b, _) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut c, _) = Arm::open(prompt, max_pages, &pipe).await?;
    let cont: Vec<u32> = vec![g0 as u32; T];
    let slabs = (T as u32 + 1).div_ceil(a.rs_page).max(1);

    for arm in [&a, &b, &c] {
        arm.rs.alloc_buffer(slabs).context("alloc_buffer")?;
    }
    a.fire(&cont, Some((0, 0)), &pipe, "A-append").await?;
    a.pos += T as u32;
    b.fire(&cont, Some((0, 0)), &pipe, "B-append").await?;
    b.pos += T as u32;
    c.fire(&cont, Some((0, 0)), &pipe, "C-append").await?;
    c.pos += T as u32;

    // The whole point: a fire whose row carries no tokens at all.
    a.build_empty_fire(&Channel::from([HALF]), "A-commit")?
        .submit(&pipe)
        .context("A-commit submit")?;
    // Arm C folds EVERYTHING through an empty row. `u32::MAX` is the
    // fire-invariant spelling of "fold everything", and the WIT promises it
    // is CLAMPED to the tail. Under the old rule a commit was selected by
    // `fold-len <= buffered`, so the value reaching the plan was clamped by
    // construction; a row that carries no tokens is a commit whatever its
    // fold length says, so nothing but an explicit clamp keeps this legal.
    c.build_empty_fire(&Channel::from([u32::MAX]), "C-commit")?
        .submit(&pipe)
        .context("C-commit submit")?;
    // Arm B does not fold. Folding is a no-op -- it trades optionality for
    // memory, never meaning -- so a context that folded half its buffer and
    // one that folded none of it must decode identically. If the empty row
    // were doing anything other than moving the boundary, this is where it
    // would show.

    let next = vec![g0 as u32];
    let mut peaks = Vec::new();
    for (arm, tag) in [(&mut a, "A"), (&mut b, "B"), (&mut c, "C")] {
        arm.fire(&next, Some((0, u32::MAX)), &pipe, tag).await?;
        arm.pos += 1;
        let live = arm.rs.buffer_size();
        if live > 0 {
            arm.rs
                .free_buffer(&(0..live).collect::<Vec<_>>())
                .with_context(|| format!("{tag} free_buffer"))?;
        }
        let (_, peak) = arm.fire(&next, None, &pipe, tag).await?;
        arm.pos += 1;
        peaks.push(peak);
    }
    pipe.close();

    let agree = close(peaks[0], peaks[1]) && close(peaks[0], peaks[2]);
    let result = format!(
        "emptycommit folded={HALF} a_next={:.4} b_next={:.4} c_next={:.4} agree={}",
        peaks[0],
        peaks[1],
        peaks[2],
        if agree { "yes" } else { "no" }
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    if !agree {
        return Err(result);
    }
    Ok(result)
}

/// A fold whose LENGTH the host never learns.
///
/// The point of `t15`: a speculative decode's accepted count is produced by
/// the verify fire, on device. Reading it back to choose the commit's
/// `fold-len` puts a host round-trip between two fires that could otherwise
/// have been enqueued together. So the count stays in a channel, the guest
/// hands that channel to `recurrent_with`, and the host plans against its own
/// UPPER BOUND -- the row's whole live buffer -- while the driver resolves the
/// real value and CLAMPS it to that bound.
///
/// The test computes a genuinely device-derived count: `1 + (argmax % W)` over
/// the append fire's own logits. The host cannot know it without awaiting, and
/// arm A never does. Arm B awaits the same argmax and passes the identical
/// count as a plain constant, which is the path that already worked. The two
/// commits must land the folded boundary in the same place, and the check is
/// drawn one token PAST the commit so it pins the folded STATE rather than the
/// logits along the way.
///
/// The negative control is the disagreement assertion: folding a DIFFERENT
/// count must produce a different continuation, or the equivalence is vacuous
/// and a dropped device value would pass unnoticed.
async fn device_fold_length(prompt: &[u32]) -> Result<String> {
    const W: u32 = 4;

    let n = prompt.len() as u32;
    let pipe = Pipeline::new();
    let probe = WorkingSet::new();
    let max_pages = (n + 2 * W + 4).div_ceil(kv_page_size());
    drop(probe);

    let (mut a, g0) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut b, g0b) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut c, _) = Arm::open(prompt, max_pages, &pipe).await?;
    if g0 != g0b {
        return Err(format!("arms disagree on the prefill token: {g0} vs {g0b}"));
    }
    // DISTINCT tokens, unlike the other arms' repeated `g0`. The negative
    // control below turns on folding `expected` tokens versus `other` leaving
    // materially different contexts, and `free_buffer` discards whatever the
    // fold did not absorb -- so a window of four identical tokens makes the
    // difference between folding one of them and two nearly unobservable.
    let mut window: Vec<u32> = wit_model::encode("alpha beta gamma delta")
        .into_iter()
        .take(W as usize)
        .collect();
    while window.len() < W as usize {
        window.push(g0 as u32 + window.len() as u32);
    }

    for arm in [&a, &b, &c] {
        arm.rs
            .alloc_buffer(W.div_ceil(arm.rs_page).max(1))
            .context("alloc_buffer")?;
    }

    // ── 1. APPEND: fill the buffer, and compute the count ON DEVICE ────────
    // `n_dev` is never awaited by arm A. `raw` carries the same argmax back to
    // the host so arms B and C can name the same number as a constant -- that
    // is the CONTROL, not the mechanism.
    //
    // Arm A's COMMIT is constructed here, before the append is submitted: the
    // commit's claim on `n_dev` is what stops the append from inferring it as
    // a terminal host-read output, which is a different endpoint role and a
    // conflicting declaration.
    let n_dev = Channel::new([1], dtype::u32).named("n_dev");
    let raw = Channel::new([1], dtype::i32).named("raw_argmax");
    let a_append = {
        let sink = n_dev.clone();
        let raw_sink = raw.clone();
        a.build_fire(
            &window,
            Some(&Channel::from([0u32])),
            "A-append",
            move || {
                let t = reduce_argmax(&intrinsics::logits());
                raw_sink.put(&t);
                // In [1, W]: a real function of the logits, bounded by the window.
                sink.put(&cast(t % W + 1u32, dtype::u32));
            },
        )?
    };
    // The commit sits one window PAST the append, so build it at that
    // position and put the arm back until the append has actually run.
    a.pos += W;
    let a_commit = a.build_empty_fire(&n_dev, "A-commit")?;
    a.pos -= W;
    a_append.submit(&pipe).context("A-append submit")?;

    let b_append = b.build_fire(&window, Some(&Channel::from([0u32])), "B-append", || {})?;
    b_append.submit(&pipe).context("B-append submit")?;
    let c_append = c.build_fire(&window, Some(&Channel::from([0u32])), "C-append", || {})?;
    c_append.submit(&pipe).context("C-append submit")?;

    let argmax = raw.take_host::<i32>().await?;
    let expected = 1 + (argmax.rem_euclid(W as i32)) as u32;
    // As far from `expected` as the window allows: the control has to be a
    // clearly different context, not an adjacent one.
    let other = if expected == 1 { W } else { 1 };

    for arm in [&mut a, &mut b, &mut c] {
        arm.pos += W;
    }

    // ── 2. COMMIT: A from the device channel, B and C from constants ───────
    // A commit replays buffered activations and stops at the recurrence, so it
    // produces no logits and carries an empty epilogue. Arm A's `n_dev` has no
    // host-known seed, which is exactly what routes it down the device path.
    a_commit.submit(&pipe).context("A-commit submit")?;
    b.build_empty_fire(&Channel::from([expected]), "B-commit")?
        .submit(&pipe)
        .context("B-commit submit")?;
    c.build_empty_fire(&Channel::from([other]), "C-commit")?
        .submit(&pipe)
        .context("C-commit submit")?;
    for arm in [&mut a, &mut b, &mut c] {
        arm.pos += 1;
        // The commit settles the boundary the only way the host can: the guest
        // says it is done with the window. Until then arm A's occupancy is a
        // bound, and a fire that needed the exact value would be refused.
        arm.rs
            .free_buffer(&(0..W.div_ceil(arm.rs_page).max(1)).collect::<Vec<_>>())
            .context("free_buffer")?;
    }

    // ── 3. PROBE: one token past the commit pins the folded STATE ──────────
    let next = vec![g0 as u32];
    // PEAK LOGITS, not greedy tokens. The argmax is far too coarse to witness
    // a fold length on this model: two genuinely different folded states pick
    // the same token, and the control below then reads as vacuous when it is
    // really just under-resolved. Same lesson as the mixed-position arm.
    let a_next = a.fire(&next, None, &pipe, "A-next").await?.1;
    let b_next = b.fire(&next, None, &pipe, "B-next").await?.1;
    let c_next = c.fire(&next, None, &pipe, "C-next").await?.1;

    pipe.close();

    if close(b_next, c_next) {
        return Err(format!(
            "folding {expected} and {other} token(s) leave indistinguishable \
             continuations ({b_next:.4} vs {c_next:.4}); this prompt cannot witness a \
             fold length and the test would pass vacuously"
        ));
    }
    let agree = close(a_next, b_next);
    let result = format!(
        "devicefold window={W} expected={expected} other={other} a_next={a_next:.4} \
         b_next={b_next:.4} c_next={c_next:.4} agree={}",
        if agree { "yes" } else { "no" }
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    if !agree {
        return Err(result);
    }
    Ok(result)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // `chain` folds a strict prefix so the buffer keeps an unfolded tail, then
    // appends a second chunk onto it.
    let chain = input.trim() == "chain";
    let accepted: u32 = if chain {
        SPEC_TOKENS / 2
    } else {
        input.trim().parse().unwrap_or(SPEC_TOKENS).min(SPEC_TOKENS)
    };
    let chunks: u32 = if chain { 2 } else { 1 };

    if wit_model::pass_kind() == wit_model::ForwardKind::Attention {
        return Ok("skipped: fold-commit needs a linear model".to_string());
    }

    if input.trim() == "mixed" {
        let prompt = wit_model::encode("hello world");
        let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
        return mixed_positions(&prompt).await;
    }

    if input.trim() == "behind" {
        let prompt = wit_model::encode("hello world");
        let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
        return fold_behind_new_tokens(&prompt).await;
    }

    if input.trim() == "empty" {
        let prompt = wit_model::encode("hello world");
        let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
        return empty_commit(&prompt).await;
    }

    if input.trim() == "device" {
        let prompt = wit_model::encode("hello world");
        let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
        return device_fold_length(&prompt).await;
    }

    if input.trim() == "interior" {
        let prompt = wit_model::encode("hello world");
        let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
        return fold_interior_boundary(&prompt).await;
    }

    if input.trim() == "inside" {
        let prompt = wit_model::encode("hello world");
        let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
        return fold_inside_new_tokens(&prompt).await;
    }

    let ws = WorkingSet::new();
    let page_size = kv_page_size();
    let rs = RsWorkingSet::new();
    let buffer_page = rs.buffer_page_size();
    if buffer_page == 0 {
        return Err("linear model reports no RS buffer page size".to_string());
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + chunks * SPEC_TOKENS + 1).div_ceil(page_size);
    ws.reserve(max_pages).context("ws.reserve")?;

    // Buffered slots for the speculative chunk. Reserved logically here;
    // the buffering fire materializes them on first write.
    let slabs = (chunks * SPEC_TOKENS).div_ceil(buffer_page);
    rs.alloc_buffer(slabs).context("rs.alloc_buffer")?;

    // ───────────────── 1. PREFILL — mode `fold` (default) ─────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    let kv = || {
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len_p,
                pages: &pages_p,
                page_indptr: &page_indptr_p,
                w_slot: &w_slot_p,
                w_off: &w_off_p,
                positions: &positions_p,
                mask: None,
            },
        })
    };
    fwd_p.attention(
        kv(),
        std::slice::from_ref(&rs),
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        g0_ch.put(&t);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;
    let g0 = g0_ch.take_host::<i32>().await?;

    // ────────────── 2. SPECULATE — `fold_len = 0`, nothing folds ──────────
    // One SPEC_TOKENS-wide fire. Its activations land in the buffered slots;
    // the folded state does not move, so nothing here is committed yet.
    let spec_toks = Channel::from(vec![g0; SPEC_TOKENS as usize]).named("spec_toks");
    let spec_indptr = Channel::from([0u32, SPEC_TOKENS]).named("spec_indptr");
    let spec_positions = Channel::from_iter(n..n + SPEC_TOKENS).named("spec_positions");
    let spec_pages = Channel::from_iter(0..max_pages).named("spec_pages");
    let spec_page_indptr =
        Channel::from([0u32, (n + SPEC_TOKENS).div_ceil(page_size)]).named("spec_page_indptr");
    let spec_w_slot =
        Channel::from_iter((n..n + SPEC_TOKENS).map(|p| p / page_size)).named("spec_w_slot");
    let spec_w_off =
        Channel::from_iter((n..n + SPEC_TOKENS).map(|p| p % page_size)).named("spec_w_off");
    let spec_out = Channel::new([1], dtype::i32).named("spec_out");

    let fwd_s = ForwardPass::new();
    fwd_s.embed(&spec_toks, &spec_indptr)?;
    let spec_kv_len = Channel::from([n + SPEC_TOKENS]).named("spec_kv_len");
    let kv = || {
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &spec_kv_len,
                pages: &spec_pages,
                page_indptr: &spec_page_indptr,
                w_slot: &spec_w_slot,
                w_off: &spec_w_off,
                positions: &spec_positions,
                mask: None,
            },
        })
    };
    // The buffered geometry: `SPEC_TOKENS` tokens appended at the buffer tail,
    // page-major from slab zero. `fold_len = 0` holds the folded boundary
    // still, so nothing here is committed -- that is what makes this fire
    // abandonable.
    let spec_fold_len = Channel::from([0u32]).named("spec_fold_len");
    fwd_s
        .attention(
            kv(),
            std::slice::from_ref(&rs),
            RsGeometry {
                fold_len: Some(&spec_fold_len),
                buffer: ..,
            },
        )
        .context("speculative state binding")?;
    fwd_s.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        spec_out.put(&t);
    });
    fwd_s.submit(&pipe).context("speculative submit")?;
    let drafted = spec_out.take_host::<i32>().await?;

    // ─────────────── 3. COMMIT — `fold_len = accepted` ────────────────────
    // Replays only the accepted prefix into the folded state. No logits, and
    // none are possible: a fold fire returns from the linear layers before the
    // output projection, so the driver refuses one that declares sample rows.
    // Hence the empty readout and the absent epilogue — and nothing to await,
    // since pipeline order already puts the fold ahead of whatever reads next.
    let mut committed = 0u32;
    if accepted > 0 {
        let commit_toks = Channel::from(vec![g0; accepted as usize]).named("commit_toks");
        let commit_indptr = Channel::from([0u32, accepted]).named("commit_indptr");
        let commit_positions = Channel::from_iter(n..n + accepted).named("commit_positions");
        let commit_pages = Channel::from_iter(0..max_pages).named("commit_pages");
        let commit_page_indptr =
            Channel::from([0u32, (n + accepted).div_ceil(page_size)]).named("commit_page_indptr");
        let commit_w_slot =
            Channel::from_iter((n..n + accepted).map(|p| p / page_size)).named("commit_w_slot");
        let commit_w_off =
            Channel::from_iter((n..n + accepted).map(|p| p % page_size)).named("commit_w_off");

        let fwd_c = ForwardPass::new();
        fwd_c.embed(&commit_toks, &commit_indptr)?;
        let commit_kv_len = Channel::from([n + accepted]).named("commit_kv_len");
        let kv = || {
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &commit_kv_len,
                    pages: &commit_pages,
                    page_indptr: &commit_page_indptr,
                    w_slot: &commit_w_slot,
                    w_off: &commit_w_off,
                    positions: &commit_positions,
                    mask: None,
                },
            })
        };
        // The commit replays `accepted` buffered tokens. WHICH tokens is not
        // stated: the replay reaches back over the row's own occupancy, which
        // the runtime tracks.
        let commit_fold_len = Channel::from([accepted]).named("commit_fold_len");
        fwd_c
            .attention(
                kv(),
                std::slice::from_ref(&rs),
                RsGeometry {
                    fold_len: Some(&commit_fold_len),
                    buffer: ..,
                },
            )
            .context("commit state binding")?;
        // An EMPTY epilogue, not an absent one. The commit samples nothing —
        // it replays buffered activations and stops at the recurrence — but a
        // pass with no stages has no PTIR program at all, and registration
        // fails with PIE_STATUS_INVALID_ARGUMENT. An empty epilogue is the
        // minimal well-formed program that produces no logits.
        fwd_c.epilogue(|| {});
        fwd_c.submit(&pipe).context("commit submit")?;
        committed = accepted;
    }

    // ────────── 4. CHAIN — a SECOND chunk onto the unfolded tail ──────────
    // The commit folded `accepted` of the `SPEC_TOKENS` buffered tokens, so the
    // buffer still holds `SPEC_TOKENS - accepted` of them. Appending here means
    // the new tokens sit at [F + tail, ...), and their recurrence has to start
    // from `folded ⊕ replay(buffer)` — the buffer READ PATH. Every recurrence
    // today initializes from `recurrent_state[slot]`, the state at F, so the
    // runtime refuses this fire rather than running it and silently pretending
    // the tail is not there.
    //
    // When the read path lands this becomes a real two-chunk speculation and
    // the harness flips from asserting the refusal to asserting the value.
    let mut chained = "n/a";
    if chain {
        let base = n + SPEC_TOKENS;
        let c2_toks = Channel::from(vec![drafted; SPEC_TOKENS as usize]).named("c2_toks");
        let c2_indptr = Channel::from([0u32, SPEC_TOKENS]).named("c2_indptr");
        let c2_positions = Channel::from_iter(base..base + SPEC_TOKENS).named("c2_positions");
        let c2_pages = Channel::from_iter(0..max_pages).named("c2_pages");
        let c2_page_indptr =
            Channel::from([0u32, (base + SPEC_TOKENS).div_ceil(page_size)]).named("c2_page_indptr");
        let c2_w_slot = Channel::from_iter((base..base + SPEC_TOKENS).map(|p| p / page_size))
            .named("c2_w_slot");
        let c2_w_off =
            Channel::from_iter((base..base + SPEC_TOKENS).map(|p| p % page_size)).named("c2_w_off");
        let c2_out = Channel::new([1], dtype::i32).named("c2_out");

        let fwd2 = ForwardPass::new();
        fwd2.embed(&c2_toks, &c2_indptr)?;
        let c2_kv_len = Channel::from([base + SPEC_TOKENS]).named("c2_kv_len");
        let kv = || {
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &c2_kv_len,
                    pages: &c2_pages,
                    page_indptr: &c2_page_indptr,
                    w_slot: &c2_w_slot,
                    w_off: &c2_w_off,
                    positions: &c2_positions,
                    mask: None,
                },
            })
        };
        // The new chunk lands at buffer token `tail`, immediately after what
        // the commit left unfolded — which is what makes this the read path
        // rather than a fresh chunk. The guest does not say so; the runtime
        // appends at the tail because there is nowhere else to append.
        let c2_fold_len = Channel::from([0u32]).named("c2_fold_len");
        fwd2.attention(
            kv(),
            std::slice::from_ref(&rs),
            RsGeometry {
                fold_len: Some(&c2_fold_len),
                buffer: ..,
            },
        )
        .context("chain state binding")?;
        fwd2.epilogue(move || {
            let t = reduce_argmax(intrinsics::logits());
            c2_out.put(&t);
        });
        fwd2.submit(&pipe).context("chain submit")?;
        c2_out.take_host::<Vec<i32>>().await?;
        chained = "ok";
    }

    // Whatever was buffered but not folded is abandoned by dropping its
    // slots — the reject half of fold-commit, and the whole point of the
    // buffer: no folded state was ever perturbed by the rejected tail.
    let remaining = rs.buffer_size();
    if remaining > 0 {
        rs.free_buffer(&(0..remaining).collect::<Vec<_>>())
            .context("free_buffer")?;
    }

    pipe.close();

    let result = format!(
        "foldcommit prefill=1 buffered={SPEC_TOKENS} committed={committed} \
         abandoned={} g0={g0} drafted={drafted} chained={chained}",
        SPEC_TOKENS - committed
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    Ok(result)
}
