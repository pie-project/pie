//! The SDK-port goldens: the canonical container bytes of five programs,
//! pinned in `goldens/sdk_containers.txt` and rebuilt byte for byte by the
//! Python (`sdk/inferlet/python/tests/test_eta_goldens.py`) and JavaScript
//! (`sdk/inferlet/javascript/src/__tests__/eta_goldens.test.ts`) ports of
//! this crate. A change to the encoder moves the bytes here first; run with
//! `UPDATE_SDK_GOLDENS=1` to rewrite the file, then copy it to
//! `sdk/inferlet/python/tests/goldens/eta_containers.txt` (the JS test reads
//! that copy) and re-run both port suites.
use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Traced};

const VOCAB: u32 = 151_936;
const PAGE: u32 = 32;

fn leak<T>(v: T) -> &'static T { Box::leak(Box::new(v)) }

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

/// lowering.rs `build_s3` verbatim (golden hash 4213522552817221928 with VOCAB 32000, PAGE 16).
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

/// text-completion's decode pass (host-driven token).
fn text_completion_decode() -> Traced {
    let n = 5u32;
    let page_size = PAGE;
    let tok_in: &'static Channel = leak(Channel::from([42i32]).named("tok_in"));
    let embed_indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("embed_indptr"));
    let positions: &'static Channel = leak(Channel::from([n]).named("positions"));
    let pages: &'static Channel = leak(Channel::from((0..3u32).collect::<Vec<_>>()).named("pages"));
    let page_indptr: &'static Channel = leak(Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr"));
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

/// naive-baseline's decode pass (device-carried token, gumbel-max, stats mirrors).
fn naive_decode() -> Traced {
    let n = 7u32;
    let page_size = PAGE;
    let temperature = 0.7f32;
    let cap = 8u32;
    let tok_in: &'static Channel = leak(Channel::from([3i32]).named("tok_in"));
    let rng: &'static Channel = leak(Channel::from([0x7ce1u32 ^ 0x5bd1, 0]).named("rng"));
    let tok_out: &'static Channel = leak(Channel::new([1], dtype::i32).capacity(cap).named("tok_out"));
    let s1_out: &'static Channel = leak(Channel::new([1], dtype::f32).capacity(cap).named("s1_out"));
    let s2_out: &'static Channel = leak(Channel::new([1], dtype::f32).capacity(cap).named("s2_out"));
    let lane1: &'static Channel = leak(Channel::from([0u32, 1u32]).named("embed_indptr"));
    let positions: &'static Channel = leak(Channel::from([n]).named("positions"));
    let pages: &'static Channel = leak(Channel::from((0..4u32).collect::<Vec<_>>()).named("pages"));
    let page_indptr: &'static Channel = leak(Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr"));
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

/// Op coverage: touches most of the op surface in one epilogue.
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
        let extra = abs(recip(sign(neg(&bb)))) + log(exp(&bb)) + max_elem(&bb, 1.0f32) - min_elem(&bb, 2.0f32) + rem(&bb, 3.0f32);
        let flag_v = reduce_sum(cast(cm, dtype::u32)) + reduce_sum(cast(sw, dtype::u32)) + reduce_sum(cast(sk, dtype::u32)) + reduce_sum(cast(mem, dtype::u32)) + reduce_sum(cast(ge_, dtype::u32));
        let total = h + h2 + ssum + reduce_sum(cs) + reduce_sum(l2) + reduce_sum(reshape(m, [1])) + reduce_sum(g) + g2 + reduce_sum(&extra) + cast(flag_v, dtype::f32) + cast(eq(&t1, &t2), dtype::f32) + cast(ne(&t1, &t3), dtype::f32) + cast(le(&t2, &t3), dtype::f32);
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

/// Prologue sinks + names table + kernel call.
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
        intrinsics::kernel::attn_page_mask(cast(gt(intrinsics::kernel::envelope_dot(4), 0.0f32), dtype::u32) + intrinsics::layer());
    });
    b.stage(Stage::Epilogue, move || {
        out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
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
    ]
}

const GOLDENS: &str = "tests/goldens/sdk_containers.txt";

#[test]
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
