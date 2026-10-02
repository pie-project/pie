use inferlet::eta::hybrid::prelude::*;
use serde::Deserialize;

const TOP_P: f32 = 0.9;
const K: u32 = 4;

#[derive(Deserialize, Default)]
struct Input {
    #[serde(default)]
    light: bool,
}

fn desc_order(values: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    order
}

fn first_mismatch<T: PartialEq + std::fmt::Debug>(
    name: &str,
    got: &[T],
    want: &[T],
    report: &mut Vec<String>,
) {
    if got.len() != want.len() {
        report.push(format!("{name}: length {} vs {}", got.len(), want.len()));
        return;
    }
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if bad.is_empty() {
        report.push(format!("{name}: ok ({} entries)", got.len()));
    } else {
        let i = bad[0];
        report.push(format!(
            "{name}: {} of {} differ; first at {i}: got {:?} want {:?}",
            bad.len(),
            got.len(),
            got[i],
            want[i]
        ));
    }
}

fn close(name: &str, got: &[f32], want: &[f32], tol: f32, report: &mut Vec<String>) {
    if got.len() != want.len() {
        report.push(format!("{name}: length {} vs {}", got.len(), want.len()));
        return;
    }
    let bad: Vec<usize> = (0..got.len())
        .filter(|&i| (got[i] - want[i]).abs() > tol * (1.0 + want[i].abs()))
        .collect();
    if bad.is_empty() {
        report.push(format!("{name}: ok ({} entries)", got.len()));
    } else {
        let i = bad[0];
        report.push(format!(
            "{name}: {} of {} differ; first at {i}: got {} want {}",
            bad.len(),
            got.len(),
            got[i],
            want[i]
        ));
    }
}

#[derive(Clone, Copy)]
struct DryCfg {
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    max_ngram: u32,
    capacity: u32,
}

fn suffix_match(hist: &Tensor, hlen: &Tensor, cfg: DryCfg) -> Tensor {
    let l = cfg.capacity;
    let pos = cast(iota(l), dtype::i32);
    let zero_i = broadcast(0i32, [l]);
    let last = cast(hlen, dtype::i32) - 1i32;
    let mut alive = lt(&pos, broadcast(&last, [l]));
    let mut m = broadcast(0.0f32, [l]);
    for d in 0..cfg.max_ngram {
        let d_i = d as i32;
        let d_row = broadcast(d_i, [l]);
        let src = max_elem(&pos - &d_row, &zero_i);
        let tail = &last - d as i32;
        let window = gather(hist, &src);
        let target = gather(hist, max_elem(&tail, 0i32));
        let matches = eq(&window, broadcast(&target, [l]));
        let in_window = ge(&pos, &d_row);
        let tail_ok = broadcast(ge(&tail, 0i32), [l]);
        alive = and(and(alive, matches), and(in_window, tail_ok));
        m = &m + cast(&alive, dtype::f32);
    }
    m
}

fn dry_penalty(hist: &Tensor, hlen: &Tensor, vocab: u32, cfg: DryCfg) -> (Tensor, Tensor, Tensor) {
    let l = cfg.capacity;
    let m = suffix_match(hist, hlen, cfg);
    let next_idx = min_elem(
        cast(iota(l), dtype::i32) + 1i32,
        broadcast(l as i32 - 1, [l]),
    );
    let next_tok = max_elem(gather(hist, &next_idx), broadcast(0i32, [l]));
    let vocab_zero = broadcast(0.0f32, [vocab]);
    let mut penalty = broadcast(0.0f32, [vocab]);
    for n in cfg.allowed_length..=cfg.max_ngram {
        let hit = ge(&m, broadcast(n as f32, [l]));
        let votes = scatter_add(&vocab_zero, &next_tok, cast(&hit, dtype::f32));
        let charge = cfg.multiplier * cfg.base.powi((n - cfg.allowed_length) as i32);
        penalty = select(
            gt(&votes, &vocab_zero),
            broadcast(charge, [vocab]),
            &penalty,
        );
    }
    let count = reshape(
        reduce_sum(cast(&gt(&penalty, &vocab_zero), dtype::f32)),
        [1],
    );
    let peak = reshape(reduce_max(&penalty), [1]);
    (penalty, count, peak)
}

fn host_suffix_match(hist: &[i32], hlen: usize, cfg: DryCfg) -> Vec<f32> {
    let l = hist.len();
    let last = hlen as i32 - 1;
    let mut m = vec![0f32; l];
    for pos in 0..l {
        let mut alive = (pos as i32) < last;
        for d in 0..cfg.max_ngram as i32 {
            let src = (pos as i32 - d).max(0) as usize;
            let tail = last - d;
            let target = hist[tail.max(0) as usize];
            alive = alive && hist[src] == target && pos as i32 >= d && tail >= 0;
            if alive {
                m[pos] += 1.0;
            }
        }
    }
    m
}

fn host_dry_penalty(hist: &[i32], hlen: usize, vocab: usize, cfg: DryCfg) -> Vec<f32> {
    let l = hist.len();
    let m = host_suffix_match(hist, hlen, cfg);
    let mut penalty = vec![0f32; vocab];
    for n in cfg.allowed_length..=cfg.max_ngram {
        let mut votes = vec![0f32; vocab];
        for pos in 0..l {
            let next = hist[(pos + 1).min(l - 1)].max(0) as usize;
            if m[pos] >= n as f32 {
                votes[next] += 1.0;
            }
        }
        let charge = cfg.multiplier * cfg.base.powi((n - cfg.allowed_length) as i32);
        for t in 0..vocab {
            if votes[t] > 0.0 {
                penalty[t] = charge;
            }
        }
    }
    penalty
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let light = input.light;
    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let rs_ws: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        model::ForwardKind::Recurrent => {
            return Err(
                "this program has no recurrent-only path (it measures the sampling epilogue of a \
                 KV-paged fire)"
                    .into(),
            );
        }
        model::ForwardKind::Diffusion => {
            return Err(
                "this program decodes a token at a time; a diffusion model wants a canvas loop"
                    .into(),
            );
        }
    };
    let page_size = kv_page_size();

    let mut prompt = model::encode("The capital of France is");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = n.div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let toks = Channel::from_iter(prompt.iter().map(|&token| token as i32));
    let embed_indptr = Channel::from([0u32, n]).named("embed_indptr");
    let positions = Channel::from_iter(0..n).named("positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let page_indptr = Channel::from([0u32, max_pages]).named("page_indptr");
    let w_slot = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot");
    let w_off = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off");
    let kv_len = Channel::from([n]).named("kv_len");
    let logits_out = Channel::new([vocab], dtype::f32).named("logits");
    let probs_out = Channel::new([vocab], dtype::f32).named("probabilities");
    let keep_out = Channel::new([vocab], dtype::i32).named("nucleus_keep");
    let cumsum_out = Channel::new([vocab], dtype::f32).named("cumsum");
    let sorted_out = Channel::new([vocab], dtype::f32).named("sorted");
    let order_out = Channel::new([vocab], dtype::i32).named("order");
    let top1_v = Channel::new([K], dtype::f32).named("top1_v");
    let top1_i = Channel::new([K], dtype::i32).named("top1_i");
    let top2_v = Channel::new([K], dtype::f32).named("top2_v");
    let top2_i = Channel::new([K], dtype::i32).named("top2_i");
    let argmax2 = Channel::new([2], dtype::i32).named("argmax2");
    let bc_out = Channel::new([5], dtype::f32).named("bcast_samples");
    let bc_idx = Channel::from(vec![
        0i32,
        1,
        vocab as i32,
        vocab as i32 + 1,
        2 * vocab as i32 - 1,
    ])
    .named("bcast_idx");
    let offs = Channel::from(vec![0.0f32, -0.5]).named("offs");
    const HIST: [i32; 16] = [5, 6, 7, 5, 6, 7, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    const HLEN: u32 = 8;
    let dry = DryCfg {
        multiplier: 0.8,
        base: 1.75,
        allowed_length: 2,
        max_ngram: 8,
        capacity: 16,
    };
    let hist_ch = Channel::from(HIST.to_vec()).named("hist");
    let hlen_ch = Channel::from([HLEN]).named("hlen");
    let m_out = Channel::new([16], dtype::f32).named("dry_m");
    let cnt_out = Channel::new([1], dtype::f32).named("dry_count");
    let peak_out = Channel::new([1], dtype::f32).named("dry_peak");
    let argpen_out = Channel::new([1], dtype::i32).named("dry_argmax");
    let histn_out = Channel::new([16], dtype::i32).named("dry_hist_next");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    fwd.attention(
        Some(KvBinding {
            working_set: &ws,
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
        }),
        &rs_ws,
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    fwd.epilogue(move || {
        let logits = intrinsics::logits();
        let logprobs = log_softmax(&logits);
        let probabilities = exp(&logprobs);
        let keep = pivot_threshold(&probabilities, cummass_le(TOP_P));
        keep_out.put(&reshape(cast(&keep, dtype::i32), [vocab]));
        cumsum_out.put(&reshape(cumsum(&probabilities), [vocab]));
        let cand = broadcast(reshape(offs.take(), [2, 1]), [2, vocab])
            + broadcast(reshape(&logprobs, [1, vocab]), [2, vocab]);
        bc_out.put(&reshape(
            gather(reshape(&cand, [2 * vocab]), bc_idx.take()),
            [5],
        ));
        argmax2.put(&reshape(cast(reduce_argmax(&cand), dtype::i32), [2]));
        if !light {
            let (sorted, order) = sort_desc(&probabilities);
            sorted_out.put(&reshape(&sorted, [vocab]));
            order_out.put(&reshape(cast(&order, dtype::i32), [vocab]));
            let (v1, i1) = top_k(reshape(&logits, [vocab]), K);
            top1_v.put(&reshape(&v1, [K]));
            top1_i.put(&reshape(cast(&i1, dtype::i32), [K]));
            let (v2, i2) = top_k(reshape(&cand, [2 * vocab]), K);
            top2_v.put(&reshape(&v2, [K]));
            top2_i.put(&reshape(cast(&i2, dtype::i32), [K]));
        }
        let hist = hist_ch.take();
        let hlen = hlen_ch.take();
        m_out.put(&reshape(suffix_match(&hist, &hlen, dry), [16]));
        let (penalty, count, peak) = dry_penalty(&hist, &hlen, vocab, dry);
        cnt_out.put(&count);
        peak_out.put(&peak);
        let pen_tok = cast(reshape(reduce_argmax(&penalty), [1]), dtype::i32);
        argpen_out.put(&pen_tok);
        histn_out.put(&reshape(scatter_set(&hist, &hlen, &pen_tok), [16]));
        logits_out.put(&logits);
        probs_out.put(&probabilities);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline).context("sort-probe submit")?;

    let logits = logits_out.take_host::<Vec<f32>>().await?;
    let probs = probs_out.take_host::<Vec<f32>>().await?;
    let keep = keep_out.take_host::<Vec<i32>>().await?;
    let cums = cumsum_out.take_host::<Vec<f32>>().await?;
    let am2 = argmax2.take_host::<Vec<i32>>().await?;
    let bc = bc_out.take_host::<Vec<f32>>().await?;
    let sorts = if light {
        None
    } else {
        Some((
            sorted_out.take_host::<Vec<f32>>().await?,
            order_out.take_host::<Vec<i32>>().await?,
            top1_v.take_host::<Vec<f32>>().await?,
            top1_i.take_host::<Vec<i32>>().await?,
            top2_v.take_host::<Vec<f32>>().await?,
            top2_i.take_host::<Vec<i32>>().await?,
        ))
    };
    let dry_m = m_out.take_host::<Vec<f32>>().await?;
    let dry_count = cnt_out.take_host::<f32>().await?;
    let dry_peak = peak_out.take_host::<f32>().await?;
    let dry_arg = argpen_out.take_host::<i32>().await?;
    let dry_hist = histn_out.take_host::<Vec<i32>>().await?;

    let v = vocab as usize;
    let mut report = Vec::new();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = logits.iter().map(|x| (x - max).exp()).sum::<f32>().ln() + max;
    let logprobs: Vec<f32> = logits.iter().map(|x| x - lse).collect();
    let ref_probs: Vec<f32> = logprobs.iter().map(|x| x.exp()).collect();
    close("probabilities", &probs, &ref_probs, 1e-3, &mut report);
    let mut acc = 0.0f32;
    let ref_cums: Vec<f32> = probs
        .iter()
        .map(|p| {
            acc += p;
            acc
        })
        .collect();
    close("cumsum", &cums, &ref_cums, 1e-2, &mut report);
    let ord = desc_order(&probs);
    let lord = desc_order(&logits);
    let cand: Vec<f32> = logprobs
        .iter()
        .cloned()
        .chain(logprobs.iter().map(|x| x - 0.5))
        .collect();
    let ref_bc: Vec<f32> = [0usize, 1, v, v + 1, 2 * v - 1]
        .iter()
        .map(|&i| cand[i])
        .collect();
    close("broadcast [2,v] samples", &bc, &ref_bc, 1e-3, &mut report);
    let ref_am2 = vec![lord[0] as i32, lord[0] as i32];
    first_mismatch("reduce_argmax [2,v]", &am2, &ref_am2, &mut report);
    if let Some((sorted, order, t1v, t1i, t2v, t2i)) = &sorts {
        let ref_sorted: Vec<f32> = ord.iter().map(|&i| probs[i]).collect();
        close("sort_desc values", sorted, &ref_sorted, 1e-6, &mut report);
        let ref_order: Vec<i32> = ord.iter().map(|&i| i as i32).collect();
        first_mismatch("sort_desc order", order, &ref_order, &mut report);
        let ref_t1i: Vec<i32> = lord[..K as usize].iter().map(|&i| i as i32).collect();
        let ref_t1v: Vec<f32> = lord[..K as usize].iter().map(|&i| logits[i]).collect();
        report.push(format!("top_k [v] got {:?} {:?}", t1i, t1v));
        first_mismatch("top_k [v] indices", t1i, &ref_t1i, &mut report);
        close("top_k [v] values", t1v, &ref_t1v, 1e-5, &mut report);
        let cord = desc_order(&cand);
        let ref_t2i: Vec<i32> = cord[..K as usize].iter().map(|&i| i as i32).collect();
        let ref_t2v: Vec<f32> = cord[..K as usize].iter().map(|&i| cand[i]).collect();
        first_mismatch("top_k [2v] indices", t2i, &ref_t2i, &mut report);
        close("top_k [2v] values", t2v, &ref_t2v, 1e-3, &mut report);
        report.push(format!("top_k [2v] got {:?} {:?}", t2i, t2v));
    } else {
        report.push("sorts skipped (light)".to_string());
    }
    let mut excl = 0.0f32;
    let mut want_kept = 0usize;
    for &i in &ord {
        if !(excl < TOP_P) {
            break;
        }
        want_kept += 1;
        excl += probs[i];
    }
    let kept = keep.iter().filter(|&&k| k != 0).count();
    let prefix_ok = ord[..kept.min(v)].iter().all(|&i| keep[i] != 0);
    report.push(format!(
        "pivot keep: kept {kept} want {want_kept} prefix_ok={prefix_ok}"
    ));
    let want_m = host_suffix_match(&HIST, HLEN as usize, dry);
    close("dry suffix_match", &dry_m, &want_m, 1e-6, &mut report);
    let want_pen = host_dry_penalty(&HIST, HLEN as usize, v, dry);
    let want_count = want_pen.iter().filter(|&&p| p > 0.0).count() as f32;
    let want_peak = want_pen.iter().cloned().fold(0.0, f32::max);
    let want_arg = (0..v)
        .max_by(|&a, &b| {
            want_pen[a]
                .partial_cmp(&want_pen[b])
                .unwrap()
                .then(b.cmp(&a))
        })
        .unwrap_or(0) as i32;
    report.push(format!(
        "dry penalty: count {dry_count} want {want_count}, peak {dry_peak} want {want_peak}, argmax {dry_arg} want {want_arg}"
    ));
    let mut want_hist = HIST.to_vec();
    want_hist[HLEN as usize] = want_arg;
    first_mismatch("dry scatter_set", &dry_hist, &want_hist, &mut report);
    Ok(report.join("\n"))
}
