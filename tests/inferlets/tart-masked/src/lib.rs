//! tart: the 0.3 naive-masked — a CUSTOM (dense-packed) attention mask
//! whose numerics are exactly causal. The prefill mask is a host bool
//! tensor (a literal the structured recognizer cannot match), and the
//! decode evolution is `and(causal, causal)` — same trick, device-side.
//! This is the mask axis's parity probe AND the spatial-split trigger:
//! co-fired with plain lanes it must produce a MASK region in the
//! scheduler's region table and the driver's planned mask split.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
use serde::Deserialize;

const PAGE_T: u32 = 16;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    max_layers: Option<u32>,
    /// BISECT: 0 = full (mask everywhere), 1 = no decode mask,
    /// 2 = no masks at all (prefill causal channel still bound? no — none).
    #[serde(default)]
    bisect: u32,
    /// CO-FIRE: run two lanes (masked + plain) through one pipeline, both
    /// fires of each step submitted into the same frame.
    #[serde(default)]
    co: bool,
    /// The plain lane's prompt in co mode.
    #[serde(default = "default_prompt_b")]
    prompt_b: String,
    /// NON-CAUSAL probe: additionally hide this KV position from every
    /// query. Distinguishes "mask applied" from "mask elided as causal" —
    /// output must differ from the maskless run.
    #[serde(default)]
    blind: Option<u32>,
    /// WIRE-ROUTE LoRA: attach a Q-site adapter (rank 8, 0.6B geometry)
    /// through `Pass::adapter` at this amplitude. Wire fires then carry
    /// the LORA region signature — the axis rides the same co-batchable
    /// class as the mask.
    #[serde(default)]
    adapter_scale: Option<f32>,
    /// WIRE-ROUTE hook: attach an `on_attn` score-fold tap to the decode
    /// pass, so wire fires carry the HOOK region signature.
    #[serde(default)]
    hook: bool,
}

/// Q-site LoRA channels at the 0.6B geometry (lora-probe's pattern seed,
/// alpha folded into B).
const LORA_RANK: u32 = 8;
const LORA_LAYERS: u32 = 28;
const LORA_D_IN: u32 = 1024;
const LORA_D_OUT: u32 = 2048;

fn lora_pattern(i: u32, salt: u32, amp: f32) -> f32 {
    let h = (i ^ salt).wrapping_mul(0x9e37_79b9) >> 8;
    ((h % 10_000) as f32 / 10_000.0 - 0.5) * 2.0 * amp
}

fn make_lora_channels(scale: f32) -> (Channel, Channel) {
    let a_len = (LORA_LAYERS * LORA_RANK * LORA_D_IN) as usize;
    let b_len = (LORA_LAYERS * LORA_D_OUT * LORA_RANK) as usize;
    let a_host: Vec<f32> = (0..a_len as u32)
        .map(|i| lora_pattern(i, 0x0a0a_a0a0, 0.05))
        .collect();
    let b_host: Vec<f32> = (0..b_len as u32)
        .map(|i| lora_pattern(i, 0x0c0c_c0c0, 0.5) * scale)
        .collect();
    (
        Channel::from_shaped([LORA_LAYERS, LORA_RANK, LORA_D_IN], a_host).named("lora_a"),
        Channel::from_shaped([LORA_LAYERS, LORA_D_OUT, LORA_RANK], b_host).named("lora_b"),
    )
}

fn default_prompt_b() -> String {
    "Name three rivers.".into()
}

fn default_prompt() -> String {
    "Tell me a story about a clockmaker.".into()
}

fn default_max_tokens() -> usize {
    32
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Ok(String::new());
    }
    if input.co {
        let prompt_b = input.prompt_b.clone();
        return run_co(&input, &prompt_b).await;
    }

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();
    let pool_pages = (n + input.max_tokens as u32 + 2).div_ceil(PAGE_T);
    let pool_len = pool_pages * PAGE_T;

    let ws = WorkingSet::new();
    let slots = ws.reserve(pool_pages).context("reserve tart-masked KV")?;
    let pool_ids = slots.ids().to_vec();

    let prompt_tokens = Channel::from_iter(prompt.iter().map(|&token| token as i32));
    let prefill_embed_indptr = Channel::from([0u32, n]).named("prefill_embed_indptr");
    let prefill_positions = Channel::from_iter(0..n).named("prefill_positions");
    let prefill_slots =
        Channel::from_iter((0..n).map(|position| pool_ids[(position / PAGE_T) as usize]));
    let prefill_offsets = Channel::from_iter((0..n).map(|position| position % PAGE_T));
    let prefill_klen = Channel::from([n]);
    let prefill_pages = Channel::from(pool_ids.clone());
    let prefill_indptr = Channel::from([0u32, n.div_ceil(PAGE_T)]);
    // The DENSE causal literal: byte-for-byte causal numerics, packed as
    // a custom mask because a host literal has no structured form.
    let blind = input.blind;
    let causal = Channel::from_shaped(
        [n, pool_len],
        (0..n)
            .flat_map(|query| {
                (0..pool_len).map(move |key| key <= query && Some(key) != blind)
            })
            .collect::<Vec<_>>(),
    );
    let first_out = Channel::new([1], dtype::i32).named("first_token");

    let prefill = ForwardPass::new();
    if let Some(k) = input.max_layers {
        prefill.set_max_layers(k)?;
    }
    if let Some(scale) = input.adapter_scale {
        let (a, b) = make_lora_channels(scale);
        use inferlet::ptir::adapter::{mm, Site};
        prefill.adapter(Site::Q, |x, y| y + mm(&b, mm(&a, x)))?;
    }
    prefill.embed(&prompt_tokens, &prefill_embed_indptr)?;
    prefill.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &prefill_klen,
            pages: &prefill_pages,
            page_indptr: &prefill_indptr,
            w_slot: &prefill_slots,
            w_off: &prefill_offsets,
            positions: &prefill_positions,
            mask: if input.bisect >= 2 { None } else { Some(&causal) },
        },
    )?;
    prefill.epilogue(move || {
        first_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
    });

    let pipeline = Pipeline::new();
    prefill.submit(&pipeline).context("tart-masked prefill")?;
    let first = first_out.take_host::<i32>().await? as u32;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        pipeline.close();
        return model::decode(&generated);
    }

    // HOST-DRIVEN decode (the 0.2 naive-masked posture): every geometry
    // channel is put from the host each step, because a HOST wire mask
    // (dense BRLE — the tart spatial path) cannot mix with
    // device-evolved geometry. Sequential, one fire per submit.
    let token_in = Channel::from([first as i32]).named("token_in");
    let decode_indptr = Channel::from([0u32, 1]).named("decode_indptr");
    let position = Channel::from([n]).named("position");
    let klen = Channel::from([n + 1]).named("klen");
    let write_slot = Channel::from([pool_ids[(n / PAGE_T) as usize]]);
    let write_offset = Channel::from([n % PAGE_T]);
    let mask = Channel::from_shaped(
        [1, pool_len],
        (0..pool_len)
            .map(|key| key <= n && Some(key) != blind)
            .collect::<Vec<_>>(),
    );
    let pages = Channel::from(pool_ids.clone());
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(PAGE_T)]);
    let token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("token_out");
    // Shadow-plan anchor: a take-less (publish-only) epilogue classifies
    // as Unknown in the host shadow planner and stalls host-route
    // submits; a loop-carried counter gives the stage the standard
    // Fold shape. (Filed as an upstream shadow-classifier gap.)
    let step_counter = Channel::from([0u32]).named("step_counter");

    let decode = ForwardPass::new();
    if let Some(k) = input.max_layers {
        decode.set_max_layers(k)?;
    }
    if let Some(scale) = input.adapter_scale {
        let (a, b) = make_lora_channels(scale);
        use inferlet::ptir::adapter::{mm, Site};
        decode.adapter(Site::Q, |x, y| y + mm(&b, mm(&a, x)))?;
    }
    if input.hook {
        // A per-layer score-fold tap: enough to make the fire a genuine
        // hook program (Stage::OnAttn) without changing the numerics.
        let score_acc =
            Channel::from(vec![0.0f32; pool_len as usize]).named("score_acc");
        decode.on_attn(move || {
            let prev = score_acc.take();
            let scores = intrinsics::attn_score(pool_len);
            score_acc.put(&(&prev + &scores));
        });
    }
    decode.embed(&token_in, &decode_indptr)?;
    decode.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &klen,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &write_slot,
            w_off: &write_offset,
            positions: &position,
            mask: if input.bisect >= 1 { None } else { Some(&mask) },
        },
    )?;
    decode.epilogue(move || {
        let step = step_counter.take();
        token_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
        step_counter.put(&step + 1u32);
    });

    let budget = input.max_tokens.saturating_sub(generated.len());
    let mut filled = n + 1; // tokens in KV after the prefill+first write
    for _ in 0..budget {
        decode.submit(&pipeline).context("tart-masked decode")?;
        let token = token_out.take_host::<i32>().await? as u32;
        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
        if generated.len() >= input.max_tokens {
            break;
        }
        // Host-advance the geometry for the next fire. Two channel
        // postures here (engine program.rs channel_accesses): channels
        // bound to EmbedTokens/Positions/WSlot/WOff (or ChanTake'd in a
        // stage) are CONSUMED per fire — queue the next value with `put`.
        // Everything else (KvLen/Pages/PageIndptr/EmbedIndptr/AttnMask)
        // is latest-value: the device never advances its head, so a
        // second `put` blocks forever on a full ring — REPLACE the
        // committed front with `set` instead, and leave constant
        // channels (pages, decode_indptr) untouched.
        let pos = filled;
        token_in.put([token as i32]);
        position.put([pos]);
        klen.set([pos + 1])?;
        write_slot.put([pool_ids[(pos / PAGE_T) as usize]]);
        write_offset.put([pos % PAGE_T]);
        if input.bisect == 0 {
            mask.set(
                (0..pool_len)
                    .map(|key| key <= pos && Some(key) != blind)
                    .collect::<Vec<bool>>(),
            )?;
        }
        page_indptr.set([0u32, (pos + 1).div_ceil(PAGE_T)])?;
        filled += 1;
    }
    pipeline.close();
    model::decode(&generated)
}

/// One sequential decode lane of the co-fired pair: its own KV pages,
/// geometry channels, and decode pass; masked lanes re-`set` the dense
/// causal row before every fire.
struct Lane {
    pool_ids: Vec<u32>,
    pool_len: u32,
    decode: ForwardPass,
    token_in: Channel,
    position: Channel,
    klen: Channel,
    write_slot: Channel,
    write_offset: Channel,
    mask: Option<Channel>,
    page_indptr: Channel,
    token_out: Channel,
    filled: u32,
    primed: bool,
    blind: Option<u32>,
    generated: Vec<u32>,
}

impl Lane {
    /// Stage the next fire's geometry. The first call SEEDS the empty
    /// channels with `put`; later calls `put` only the consumed channels
    /// (EmbedTokens/Positions/WSlot/WOff) and `set` the latest-value ones
    /// (KvLen/AttnMask/PageIndptr), whose device head never advances.
    fn advance(&mut self, token: u32) -> Result<()> {
        self.generated.push(token);
        let pos = self.filled;
        self.token_in.put([token as i32]);
        self.position.put([pos]);
        self.write_slot
            .put([self.pool_ids[(pos / PAGE_T) as usize]]);
        self.write_offset.put([pos % PAGE_T]);
        let klen = [pos + 1];
        let indptr = [0u32, (pos + 1).div_ceil(PAGE_T)];
        let blind = self.blind;
        let mask_row = self.mask.as_ref().map(|_| {
            (0..self.pool_len)
                .map(|key| key <= pos && Some(key) != blind)
                .collect::<Vec<bool>>()
        });
        if self.primed {
            self.klen.set(klen)?;
            if let (Some(mask), Some(row)) = (&self.mask, mask_row) {
                mask.set(row)?;
            }
            self.page_indptr.set(indptr)?;
        } else {
            self.klen.put(klen);
            if let (Some(mask), Some(row)) = (&self.mask, mask_row) {
                mask.put(row);
            }
            self.page_indptr.put(indptr);
            self.primed = true;
        }
        self.filled += 1;
        Ok(())
    }
}

/// CO-FIRE: one process, one pipeline, two lanes per step — lane A carries
/// the dense custom mask (exactly-causal numerics), lane B is plain causal.
/// Both fires of a step are submitted back-to-back before either take, so
/// they ride one frame: the scheduler's region table must show a MASK
/// region next to a plain region, and the driver's planned mask split must
/// engage. Stop tokens are ignored so the lanes stay in lockstep.
async fn run_co(input: &Input, prompt_b: &str) -> Result<String> {
    let ws = WorkingSet::new();
    let pipeline = Pipeline::new();

    let build = |text: &str, masked: bool| -> Result<(Lane, Channel, ForwardPass)> {
        let mut prompt = chat::system_user("You are a helpful assistant.", text);
        prompt.extend(chat::cue());
        if prompt.is_empty() {
            prompt.push(0);
        }
        let n = prompt.len() as u32;
        let pool_pages = (n + input.max_tokens as u32 + 2).div_ceil(PAGE_T);
        let pool_len = pool_pages * PAGE_T;
        let slots = ws.reserve(pool_pages).context("reserve co-lane KV")?;
        let pool_ids = slots.ids().to_vec();

        let prompt_tokens = Channel::from_iter(prompt.iter().map(|&token| token as i32));
        let prefill_embed_indptr = Channel::from([0u32, n]);
        let prefill_positions = Channel::from_iter(0..n);
        let prefill_slots =
            Channel::from_iter((0..n).map(|position| pool_ids[(position / PAGE_T) as usize]));
        let prefill_offsets = Channel::from_iter((0..n).map(|position| position % PAGE_T));
        let prefill_klen = Channel::from([n]);
        let prefill_pages = Channel::from(pool_ids.clone());
        let prefill_indptr = Channel::from([0u32, n.div_ceil(PAGE_T)]);
        let blind = input.blind;
        let prefill_mask = masked.then(|| {
            Channel::from_shaped(
                [n, pool_len],
                (0..n)
                    .flat_map(|query| {
                        (0..pool_len).map(move |key| key <= query && Some(key) != blind)
                    })
                    .collect::<Vec<_>>(),
            )
        });
        let first_out = Channel::new([1], dtype::i32);

        let prefill = ForwardPass::new();
        prefill.embed(&prompt_tokens, &prefill_embed_indptr)?;
        prefill.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &prefill_klen,
                pages: &prefill_pages,
                page_indptr: &prefill_indptr,
                w_slot: &prefill_slots,
                w_off: &prefill_offsets,
                positions: &prefill_positions,
                mask: prefill_mask.as_ref(),
            },
        )?;
        {
            let first_out = first_out.clone();
            prefill.epilogue(move || {
                first_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
            });
        }

        let token_in = Channel::new([1], dtype::i32);
        let decode_indptr = Channel::from([0u32, 1]);
        let position = Channel::new([1], dtype::u32);
        let klen = Channel::new([1], dtype::u32);
        let write_slot = Channel::new([1], dtype::u32);
        let write_offset = Channel::new([1], dtype::u32);
        let mask = masked.then(|| Channel::new([1, pool_len], dtype::bool));
        let pages = Channel::from(pool_ids.clone());
        let page_indptr = Channel::new([2], dtype::u32);
        let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        // Fold-shaped anchor for the host shadow classifier (see solo path).
        let step_counter = Channel::from([0u32]);

        let decode = ForwardPass::new();
        decode.embed(&token_in, &decode_indptr)?;
        decode.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &klen,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &write_slot,
                w_off: &write_offset,
                positions: &position,
                mask: mask.as_ref(),
            },
        )?;
        {
            let token_out = token_out.clone();
            decode.epilogue(move || {
                let step = step_counter.take();
                token_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
                step_counter.put(&step + 1u32);
            });
        }

        let lane = Lane {
            pool_ids,
            pool_len,
            decode,
            token_in,
            position,
            klen,
            write_slot,
            write_offset,
            mask,
            page_indptr,
            token_out,
            filled: n,
            primed: false,
            blind: masked.then_some(input.blind).flatten(),
            generated: Vec::with_capacity(input.max_tokens),
        };
        Ok((lane, first_out, prefill))
    };

    let (mut lane_a, first_a, prefill_a) = build(&input.prompt, true)?;
    let (mut lane_b, first_b, prefill_b) = build(prompt_b, false)?;

    // Both prefills before either take: one frame, MULTI_TOKEN|MASK next
    // to MULTI_TOKEN.
    println!("[co] submit prefills");
    prefill_a.submit(&pipeline).context("co prefill A")?;
    prefill_b.submit(&pipeline).context("co prefill B")?;
    println!("[co] take first A");
    let first_a = first_a.take_host::<i32>().await? as u32;
    println!("[co] take first B");
    let first_b = first_b.take_host::<i32>().await? as u32;
    lane_a.advance(first_a)?;
    lane_b.advance(first_b)?;

    for step in 1..input.max_tokens {
        println!("[co] submit {step}");
        lane_a.decode.submit(&pipeline).context("co decode A")?;
        lane_b.decode.submit(&pipeline).context("co decode B")?;
        println!("[co] take {step} B");
        let token_b = lane_b.token_out.take_host::<i32>().await? as u32;
        println!("[co] take {step} A");
        let token_a = lane_a.token_out.take_host::<i32>().await? as u32;
        lane_a.advance(token_a)?;
        lane_b.advance(token_b)?;
    }
    pipeline.close();

    let text_a = model::decode(&lane_a.generated)?;
    let text_b = model::decode(&lane_b.generated)?;
    Ok(format!("{text_a}\n=====\n{text_b}"))
}
