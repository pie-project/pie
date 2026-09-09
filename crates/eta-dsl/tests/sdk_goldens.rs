use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Traced};

const VOCAB: u32 = 151_936;
const PAGE: u32 = 32;

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn s3() -> Traced {
    let vocab = 32_000u32;
    let ctr1: &'static Tensor = leak(Tensor::constant([0u32, 1]));
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    let mask: &'static Channel = leak(Channel::new([vocab], dtype::bool).named("mask"));
    let len: &'static Channel = leak(Channel::from([1u32]).named("len"));
    let rng_ch: &'static Channel = leak(Channel::from([7u32, 0]).named("rng"));
    tok.put([1i32]);
    let mut b = Builder::new(vocab, 16);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
    b.bind_port(Port::KvLen, len);
    b.stage(Stage::Epilogue, move || {
        let logits = intrinsics::logits();
        let r = rng_ch.take();
        let g = gumbel(&r, [intrinsics::vocab()]);
        let t = reduce_argmax(add(mask_apply(logits, mask.take()), g));
        rng_ch.put(add(&r, ctr1));
        tok.put(&t);
        len.put(add(len.take(), 1u32));
        out.put(t);
    });
    mask.put(vec![true; vocab as usize]);
    b.build().unwrap()
}

fn text_completion_decode() -> Traced {
    let n = 5u32;
    let page_size = PAGE;
    let tok_in: &'static Channel = leak(Channel::from([42i32]).named("tok_in"));
    let embed_indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("embed_indptr"));
    let positions: &'static Channel = leak(Channel::from([n]).named("positions"));
    let pages: &'static Channel = leak(Channel::from((0..3u32).collect::<Vec<_>>()).named("pages"));
    let page_indptr: &'static Channel =
        leak(Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr"));
    let w_slot: &'static Channel = leak(Channel::from([n / page_size]).named("w_slot"));
    let w_off: &'static Channel = leak(Channel::from([n % page_size]).named("w_off"));
    let kv_len: &'static Channel = leak(Channel::from([n + 1]).named("kv_len"));
    let tok_out: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok_out"));
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok_in);
    b.bind_port(Port::EmbedIndptr, embed_indptr);
    b.bind_port(Port::KvLen, kv_len);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_indptr);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::Positions, positions);
    b.stage(Stage::Epilogue, move || {
        let length = kv_len.take();
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(page_size);
        kv_len.put(&next_length);
        positions.put(&length);
        w_slot.put(&length / page_size);
        w_off.put(&length % page_size);
        page_indptr.put(indptr(1, &page_count));
        tok_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });
    tok_out.note_host_take();
    b.build().unwrap()
}

fn naive_decode() -> Traced {
    let n = 7u32;
    let page_size = PAGE;
    let temperature = 0.7f32;
    let cap = 8u32;
    let tok_in: &'static Channel = leak(Channel::from([3i32]).named("tok_in"));
    let rng: &'static Channel = leak(Channel::from([0x7ce1u32 ^ 0x5bd1, 0]).named("rng"));
    let tok_out: &'static Channel =
        leak(Channel::new([1], dtype::i32).capacity(cap).named("tok_out"));
    let s1_out: &'static Channel =
        leak(Channel::new([1], dtype::f32).capacity(cap).named("s1_out"));
    let s2_out: &'static Channel =
        leak(Channel::new([1], dtype::f32).capacity(cap).named("s2_out"));
    let lane1: &'static Channel = leak(Channel::from([0u32, 1u32]).named("embed_indptr"));
    let positions: &'static Channel = leak(Channel::from([n]).named("positions"));
    let pages: &'static Channel = leak(Channel::from((0..4u32).collect::<Vec<_>>()).named("pages"));
    let page_indptr: &'static Channel =
        leak(Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr"));
    let w_slot: &'static Channel = leak(Channel::from([n / page_size]).named("w_slot"));
    let w_off: &'static Channel = leak(Channel::from([n % page_size]).named("w_off"));
    let kv_len: &'static Channel = leak(Channel::from([n + 1]).named("kv_len"));
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok_in);
    b.bind_port(Port::EmbedIndptr, lane1);
    b.bind_port(Port::KvLen, kv_len);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_indptr);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::Positions, positions);
    b.stage(Stage::Epilogue, move || {
        let length = kv_len.take();
        let r = rng.take();
        let logits = intrinsics::logits();
        let scaled = &logits / temperature;
        let token = gumbel_max(scaled, &r);
        let r_next = &r + iota(2);
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(page_size);
        tok_in.put(&token);
        kv_len.put(&next_length);
        positions.put(&length);
        w_slot.put(&length / page_size);
        w_off.put(&length % page_size);
        page_indptr.put(indptr(1, &page_count));
        tok_out.put(&token);
        let mirror = reshape(cast(&token, dtype::f32), [1]);
        s1_out.put(&mirror);
        s2_out.put(&mirror);
        rng.put(&r_next);
    });
    tok_out.note_host_take();
    s1_out.note_host_take();
    s2_out.note_host_take();
    b.build().unwrap()
}

fn coverage() -> Traced {
    let k = 8u32;
    let tok: &'static Channel = leak(Channel::from([1i32]).named("tok"));
    let indptr_ch: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let rng_ch: &'static Channel = leak(Channel::from([1u32, 2]).named("rng"));
    let top_p: &'static Channel = leak(Channel::from([0.9f32]).named("top_p"));
    let bias: &'static Channel = leak(Channel::from(vec![0.0f32, -1.5, 2.25, 0.0]).named("bias"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    let stat: &'static Channel = leak(Channel::new([1], dtype::f32).named("stat"));
    let stat2: &'static Channel = leak(Channel::new([4], dtype::f32).named("stat2"));
    let flag: &'static Channel = leak(Channel::new([1], dtype::bool).named("flag"));
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr_ch);
    b.stage(Stage::Epilogue, move || {
        let logits = intrinsics::logits();
        let r = rng_ch.take();
        let p = softmax(&logits);
        let lp = log_softmax(&logits);
        let h = entropy(&p);
        let h2 = entropy_from_logprobs(&p, &lp);
        let (tv, ti) = top_k(&logits, k);
        let keep = pivot_threshold(&p, cummass_le(top_p.read()));
        let keep2 = pivot_threshold(&p, rank_le(40u32));
        let keep3 = pivot_threshold(&p, prob_ge(0.01f32));
        let both = and(and(&keep, &keep2), not(keep3));
        let masked = mask_apply(&logits, or(both, lt(&logits, 0.0f32)));
        let t1 = nucleus_sample(&masked, top_p.read(), &r);
        let t2 = masked_argmax(&logits, &keep);
        let t3 = gumbel_max(&logits, &r);
        let g = gather(&logits, cast(&ti, dtype::u32));
        let g2 = scalar_gather(&logits, cast(&t1, dtype::u32));
        let ssum = reduce_sum(&tv) + reduce_max(&tv) - reduce_min(&tv);
        let cs = cumsum(&tv) * cumprod(exp(&tv));
        let (sv, _si) = sort_desc(&logits);
        let l2 = l2norm(reshape(&sv, [VOCAB]));
        let m = matmul(reshape(&tv, [1, k]), transpose(reshape(&tv, [1, k])));
        let sel = select(gt(&t1, &t2), &t1, &t3);
        let sc = scatter_set(&logits, cast(&t2, dtype::u32), -1.0f32);
        let sa = scatter_add(&sc, cast(&t3, dtype::u32), 1.0f32);
        let ge_ = ge(&sa, 0.5f32);
        let cm = causal_mask(iota(4), 8);
        let sw = sliding_window_mask(iota(4), 8, 3);
        let sk = sink_window_mask(iota(4), 8, 1, 3);
        let mem = row_membership(reshape(iota(8), [2, 4]), iota(3));
        let u = rng(&r, [4]);
        let bb = bias.read() + u;
        let extra = abs(recip(sign(neg(&bb)))) + log(exp(&bb)) + max_elem(&bb, 1.0f32)
            - min_elem(&bb, 2.0f32)
            + rem(&bb, 3.0f32);
        let flag_v = reduce_sum(cast(cm, dtype::u32))
            + reduce_sum(cast(sw, dtype::u32))
            + reduce_sum(cast(sk, dtype::u32))
            + reduce_sum(cast(mem, dtype::u32))
            + reduce_sum(cast(ge_, dtype::u32));
        let total = h
            + h2
            + ssum
            + reduce_sum(cs)
            + reduce_sum(l2)
            + reduce_sum(reshape(m, [1]))
            + reduce_sum(g)
            + g2
            + reduce_sum(&extra)
            + cast(flag_v, dtype::f32)
            + cast(eq(&t1, &t2), dtype::f32)
            + cast(ne(&t1, &t3), dtype::f32)
            + cast(le(&t2, &t3), dtype::f32);
        stat.put(reshape(&total, [1]));
        stat2.put(extra);
        flag.put(reshape(gt(&total, 0.0f32), [1]));
        out.put(reshape(sel, [1]));
        rng_ch.put(&r + iota(2));
    });
    out.note_host_take();
    stat.note_host_take();
    stat2.note_host_take();
    flag.note_host_take();
    b.build().unwrap()
}

fn sinks() -> Traced {
    let tok: &'static Channel = leak(Channel::from([1i32]).named("tok"));
    let indptr_ch: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let a: &'static Channel = leak(Channel::new([2, 4, 8], dtype::f32).named("a"));
    let bch: &'static Channel = leak(Channel::new([2, 8, 4], dtype::f32).named("b"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    a.put(vec![0.0f32; 64]);
    bch.put(vec![0.0f32; 64]);
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr_ch);
    b.stage(Stage::Prologue, move || {
        intrinsics::kernel::lora(a.read(), bch.read(), Tensor::constant(1u32 | 4u32));
        intrinsics::kernel::attn_page_mask(iota(4));
    });
    b.stage(Stage::OnAttnProj, move || {
        let q = intrinsics::query(16);
        let s = intrinsics::kernel::envelope_dot(4);
        let _ = (q, s);
        intrinsics::kernel::attn_page_mask(
            cast(gt(intrinsics::kernel::envelope_dot(4), 0.0f32), dtype::u32) + intrinsics::layer(),
        );
    });
    b.stage(Stage::Epilogue, move || {
        out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });
    out.note_host_take();
    b.build().unwrap()
}

fn diffusion_step() -> Traced {
    let length = 8u32;
    let taps = 4u32;
    let base = 5u32;
    let end = base + length;
    let page_size = PAGE;
    let max_pages = 2u32;
    let bound = 0.5f32;
    let confidence = 0.1f32;
    let toks: &'static Channel = leak(
        Channel::from(
            (0..length)
                .map(|i| (i * 7919 % 1000) as i32)
                .collect::<Vec<_>>(),
        )
        .named("canvas"),
    );
    let embed_indptr: &'static Channel = leak(Channel::from([0u32, length]).named("embed_indptr"));
    let positions: &'static Channel =
        leak(Channel::from((base..end).collect::<Vec<_>>()).named("positions"));
    let pages: &'static Channel =
        leak(Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages"));
    let page_indptr: &'static Channel =
        leak(Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr"));
    let w_slot: &'static Channel =
        leak(Channel::from((base..end).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot"));
    let w_off: &'static Channel =
        leak(Channel::from((base..end).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off"));
    let kv_len: &'static Channel = leak(Channel::from([end]).named("kv_len"));
    let readout: &'static Channel =
        leak(Channel::from((0..length).collect::<Vec<_>>()).named("readout"));
    let temp: &'static Channel = leak(Channel::from([1.0f32]).named("temperature"));
    let rng_state: &'static Channel = leak(Channel::from([7u32, 0]).named("rng"));
    let history: &'static Channel =
        leak(Channel::from(vec![-1i32; length as usize]).named("argmax_history"));
    let canvas_out: &'static Channel = leak(Channel::new([length], dtype::i32).named("canvas_out"));
    let argmax_out: &'static Channel = leak(Channel::new([length], dtype::i32).named("argmax_out"));
    let stop: &'static Channel = leak(Channel::new([1], dtype::bool).named("stop"));
    let mean_out: &'static Channel = leak(Channel::new([1], dtype::f32).named("mean_entropy"));
    let tap_ids_out: &'static Channel =
        leak(Channel::new([length, taps], dtype::u32).named("tap_ids"));
    let tap_weights_out: &'static Channel =
        leak(Channel::new([length, taps], dtype::f32).named("tap_weights"));
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, embed_indptr);
    b.bind_port(Port::KvLen, kv_len);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_indptr);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::Positions, positions);
    b.bind_port(Port::Readout, readout);
    b.stage(Stage::Epilogue, move || {
        positions.put(positions.take());
        w_slot.put(w_slot.take());
        w_off.put(w_off.take());

        let r = rng_state.take();
        let t = reshape(temp.read(), []);
        let logits = intrinsics::logits();
        let scaled = div(&logits, &t);
        let probs = softmax(&scaled);
        let h = entropy(&probs);
        let sampled = gumbel_max(&scaled, &r);
        let argmax = reduce_argmax(&scaled);

        let accept = {
            let n = h.shape().dims()[0];
            let (neg_sorted, order) = sort_desc(neg(&h));
            let sorted = neg(&neg_sorted);
            let below = le(sub(cumsum(&sorted), &sorted), bound);
            let none = lt(iota(n), 0u32);
            scatter_set(&none, &order, &below)
        };
        let r_noise = add(&r, iota(2));
        let noise = cast(mul(rng(&r_noise, [length]), VOCAB as f32), dtype::i32);
        let next = select(&accept, &sampled, &noise);

        let previous = history.take();
        history.put(&argmax);
        let done = {
            let n = argmax.shape().dims()[0];
            let unchanged = reduce_sum(cast(eq(&argmax, &previous), dtype::i32));
            let stable = eq(&unchanged, n as i32);
            let mean = div(reduce_sum(&h), n as f32);
            and(&stable, lt(&mean, confidence))
        };

        let (tap_weights, tap_ids) = top_k(&probs, taps);
        tap_ids_out.put(&tap_ids);
        tap_weights_out.put(&tap_weights);

        canvas_out.put(&next);
        argmax_out.put(&argmax);
        stop.put(reshape(done, [1]));
        mean_out.put(reshape(div(reduce_sum(&h), length as f32), [1]));
        rng_state.put(add(&r_noise, iota(2)));
    });
    for ch in [
        canvas_out,
        argmax_out,
        stop,
        mean_out,
        tap_ids_out,
        tap_weights_out,
    ] {
        ch.note_host_take();
    }
    b.build().unwrap()
}

fn beam_step() -> Traced {
    #[allow(non_snake_case)]
    let B = 2u32;
    let pool_pages = 8u32;
    let page_t = PAGE;
    let pool_len = pool_pages * page_t;
    let v = VOCAB;
    let pool_ids: Vec<u32> = (0..pool_pages).collect();
    let tiled: Vec<u32> = (0..B * pool_pages).map(|_| pool_ids[0]).collect();
    let init_mask: Vec<bool> = (0..B).flat_map(|_| (0..pool_len).map(|p| p == 0)).collect();
    let mask: &'static Channel = leak(Channel::from_shaped([B, pool_len], init_mask).named("mask"));
    let scores: &'static Channel =
        leak(Channel::from(vec![0.0f32, f32::NEG_INFINITY]).named("scores"));
    let toks: &'static Channel = leak(Channel::from(vec![1i32; B as usize]).named("toks"));
    let pos: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("pos"));
    let fill: &'static Channel = leak(Channel::from([1u32]).named("fill"));
    let klen: &'static Channel = leak(Channel::from(vec![1u32; B as usize]).named("klen"));
    let w_slot: &'static Channel =
        leak(Channel::from(vec![pool_ids[0]; B as usize]).named("w_slot"));
    let w_off: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("w_off"));
    let pages: &'static Channel = leak(Channel::from(tiled).named("pages"));
    let page_indptr: &'static Channel =
        leak(Channel::from_shaped([B + 1], (0..=B).collect::<Vec<_>>()).named("page_indptr"));
    let lanes_b: &'static Channel =
        leak(Channel::from((0..=B).collect::<Vec<_>>()).named("embed_indptr"));
    let pool_ids_ch: &'static Channel = leak(Channel::from(pool_ids).named("pool_ids"));
    let out: &'static Channel = leak(Channel::new([B], dtype::i32).capacity(8).named("out"));
    let out_par: &'static Channel =
        leak(Channel::new([B], dtype::u32).capacity(8).named("out_par"));
    let out_scr: &'static Channel =
        leak(Channel::new([B], dtype::f32).capacity(8).named("out_scr"));
    let out_greedy: &'static Channel = leak(
        Channel::new([B], dtype::i32)
            .capacity(8)
            .named("out_greedy"),
    );
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_indptr);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::Positions, pos);
    b.bind_port(Port::AttnMask, mask);
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes_b);
    b.stage(Stage::Epilogue, move || {
        let logits = reshape(intrinsics::logits(), [B, v]);
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, v]),
            log_softmax(&logits),
        );
        let (s, i) = top_k(reshape(cand, [B * v]), B);
        let parent = div(&i, v);
        let tok_i = cast(rem(&i, v), dtype::i32);

        let base = fill.take();
        let lane = iota(B);
        let base_b = broadcast(reshape(&base, [1]), [B]);
        let wpos = add(&base_b, &lane);

        let inherited = gather(mask.take(), &parent);
        let col = broadcast(reshape(iota(pool_len), [1, pool_len]), [B, pool_len]);
        let wpos_b = broadcast(reshape(&wpos, [B, 1]), [B, pool_len]);
        let newpos = eq(col, wpos_b);
        let new_mask = or(inherited, &newpos);
        mask.put(&new_mask);

        let pids = pool_ids_ch.take();
        let logical_slot = div(&wpos, page_t);
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = rem(&wpos, page_t);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);

        let filled = add(&base, B);
        klen.put(broadcast(reshape(&filled, [1]), [B]));

        pos.put(add(pos.take(), 1u32));
        fill.put(&filled);
        scores.put(&s);
        toks.put(&tok_i);
        let page_count = filled.div_ceil(page_t);
        let pages_ig = gather(
            &pids,
            rem(
                iota(B * pool_pages),
                broadcast(&page_count, [B * pool_pages]),
            ),
        );
        pages.put(&pages_ig);
        page_indptr.put(mul(iota(B + 1), broadcast(&page_count, [B + 1])));

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        out_greedy.put(reshape(reduce_argmax(&logits), [B]));
        pool_ids_ch.put(&pids);
    });
    b.build().unwrap()
}

fn latent_step() -> Traced {
    let rows = 8u32;
    let channels = 16u32;
    let tok: &'static Channel = leak(Channel::from([1i32]).named("tok"));
    let indptr_ch: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let readout: &'static Channel =
        leak(Channel::from((0..rows).collect::<Vec<_>>()).named("readout"));
    let kv_len: &'static Channel = leak(Channel::from([rows]).named("kv_len"));
    let positions: &'static Channel =
        leak(Channel::from((0..rows).collect::<Vec<_>>()).named("positions"));
    let pages: &'static Channel = leak(Channel::from([0u32]).named("pages"));
    let page_indptr: &'static Channel =
        leak(Channel::from([0u32, rows.div_ceil(PAGE)]).named("page_indptr"));
    let w_slot: &'static Channel =
        leak(Channel::from((0..rows).map(|p| p / PAGE).collect::<Vec<_>>()).named("w_slot"));
    let w_off: &'static Channel =
        leak(Channel::from((0..rows).map(|p| p % PAGE).collect::<Vec<_>>()).named("w_off"));
    let latent: &'static Channel = leak(Channel::new([rows, channels], dtype::f32).named("latent"));
    let dsigma: &'static Channel = leak(Channel::from([-0.25f32]).named("dsigma"));
    let rng_ch: &'static Channel = leak(Channel::from([9u32, 0]).named("rng"));
    let out: &'static Channel =
        leak(Channel::new([rows, channels], dtype::f32).named("latent_out"));
    let norm_out: &'static Channel = leak(Channel::new([rows], dtype::f32).named("norms"));
    latent.put(vec![0.0f32; (rows * channels) as usize]);
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr_ch);
    b.bind_port(Port::KvLen, kv_len);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_indptr);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::Positions, positions);
    b.bind_port(Port::Readout, readout);
    b.stage(Stage::Epilogue, move || {
        positions.put(positions.take());
        w_slot.put(w_slot.take());
        w_off.put(w_off.take());
        let v = intrinsics::velocity(channels);
        let x = latent.take();
        let d = reshape(dsigma.read(), []);

        let square = mul(&v, &v);
        let energy = reduce_sum(&square);
        let norm = sqrt(&energy);
        let guarded = add(&energy, 1.0e-12f32);
        let inverse = rsqrt(&guarded);
        let column = reshape(&inverse, [rows, 1]);
        let spread = broadcast(&column, [rows, channels]);
        let unit = mul(&v, &spread);

        let ramp = cast(iota(channels), dtype::f32);
        let ramp_row = reshape(&ramp, [1, channels]);
        let ramp_plane = broadcast(&ramp_row, [rows, channels]);
        let angle = mul(&ramp_plane, &d);
        let sine = sin(&angle);
        let cosine = cos(&angle);
        let embedding = add(&sine, &cosine);

        let r = rng_ch.take();
        let z = normal(&r, [rows, channels]);
        let drift = add(&unit, &embedding);
        let step = mul(&drift, &d);
        let jitter = mul(&z, &d);
        let moved = add(&x, &step);
        let stepped = add(&moved, &jitter);

        latent.put(&stepped);
        out.put(&stepped);
        norm_out.put(&norm);
        rng_ch.put(add(&r, iota(2)));
    });
    out.note_host_take();
    norm_out.note_host_take();
    b.build().unwrap()
}

fn vae_readback() -> Traced {
    let (rows, rgb) = (16u32, 3u32);
    let out: &'static Channel = leak(Channel::new([rows, rgb], dtype::f32).named("pixels_out"));
    let mut b = Builder::new(VOCAB, PAGE);
    b.stage(Stage::Epilogue, move || {
        let px = intrinsics::pixels(rows, rgb);
        let shifted = add(&px, 1.0f32);
        let unit = mul(&shifted, 0.5f32);
        out.put(&unit);
    });
    out.note_host_take();
    b.build().unwrap()
}

fn programs() -> Vec<(&'static str, Traced)> {
    vec![
        ("s3", s3()),
        ("text_completion_decode", text_completion_decode()),
        ("naive_decode", naive_decode()),
        ("coverage", coverage()),
        ("sinks", sinks()),
        ("diffusion_step", diffusion_step()),
        ("beam_step", beam_step()),
        ("latent_step", latent_step()),
        ("vae_readback", vae_readback()),
    ]
}

const GOLDENS: &str = "tests/goldens/sdk_containers.txt";

#[test]
fn sdk_goldens_every_case() {
    sdk_port_goldens_are_pinned();
    the_latent_step_binds_against_a_denoising_model();
    the_vae_readback_binds_against_a_model_that_lands_pixels();
}

fn sdk_port_goldens_are_pinned() {
    let rendered: String = programs()
        .iter()
        .map(|(name, t)| format!("{name} {} {}\n", t.identity_hash(), hex(&t.encode())))
        .collect();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDENS);
    if std::env::var_os("UPDATE_SDK_GOLDENS").is_some() {
        std::fs::write(&path, &rendered).expect("write goldens");
        return;
    }
    let pinned = std::fs::read_to_string(&path).expect("goldens file present");
    for (want, got) in pinned.lines().zip(rendered.lines()) {
        let (wn, wrest) = want.split_once(' ').unwrap();
        let (gn, grest) = got.split_once(' ').unwrap();
        assert_eq!(wn, gn, "program order");
        assert_eq!(
            wrest, grest,
            "container bytes of `{wn}` moved: the SDK ports' goldens must be regenerated \
             (UPDATE_SDK_GOLDENS=1) and the ports re-verified"
        );
    }
    assert_eq!(pinned.lines().count(), rendered.lines().count());
}

fn the_latent_step_binds_against_a_denoising_model() {
    let profile = eta_ir::registry::ModelProfile {
        vocab: VOCAB,
        page_size: PAGE,
        has_velocity: true,
        velocity_width: 16,
        ..eta_ir::registry::ModelProfile::dummy()
    };
    eta_ir::validate::bind(latent_step().container().clone(), profile)
        .expect("the latent step binds against a model that predicts a velocity");
}

fn the_vae_readback_binds_against_a_model_that_lands_pixels() {
    let profile = eta_ir::registry::ModelProfile {
        vocab: VOCAB,
        page_size: PAGE,
        has_pixels: true,
        pixels_width: 0,
        ..eta_ir::registry::ModelProfile::dummy()
    };
    eta_ir::validate::bind(vae_readback().container().clone(), profile.clone())
        .expect("the readback binds against a model whose VAE lands pixels");
    let vaeless = eta_ir::registry::ModelProfile {
        has_pixels: false,
        ..profile
    };
    eta_ir::validate::bind(vae_readback().container().clone(), vaeless)
        .expect_err("a model with no VAE refuses the readback at bind");
}
