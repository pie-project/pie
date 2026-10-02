use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use inferlet::session;
use serde::{Deserialize, Serialize};

const PARK_AFTER: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_user")]
    user: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    raw_tokens: Option<Vec<u32>>,
    #[serde(default)]
    debug: bool,
    #[serde(default)]
    chunk: Option<u32>,
}

#[derive(Deserialize)]
struct Turn {
    user: String,
}

#[derive(Serialize)]
struct Event<'a> {
    e: &'a str,
    turn: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<&'a [u32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<&'a [u32]>,
    fed: u32,
    prompt: u32,
    resident: u32,
    prefill_us: u64,
    decode_us: u64,
}

fn default_system() -> String {
    "You are a helpful agent.".into()
}

fn default_user() -> String {
    "start".into()
}

fn default_max_tokens() -> usize {
    24
}

fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

struct Session {
    ws: WorkingSet,
    rs_ws: Vec<RsWorkingSet>,
    pipe: Pipeline,
    pool_ids: Vec<u32>,
    resident: u32,
    prompt: u32,
    prefill_us: u64,
    decode_us: u64,
    pending: Vec<u32>,
    page_t: u32,
    max_tokens: usize,
    chunk: Option<u32>,
}

impl Session {
    fn new(max_tokens: usize, chunk: Option<u32>) -> Result<Self> {
        let rs_ws: Vec<RsWorkingSet> = match model::pass_kind() {
            model::ForwardKind::Attention => Vec::new(),
            model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
            model::ForwardKind::Recurrent => {
                return Err("this program has no recurrent-only path (it needs a KV cache)".into());
            }
            model::ForwardKind::Diffusion => {
                return Err("this program decodes a token at a time".into());
            }
        };
        Ok(Self {
            ws: WorkingSet::new(),
            rs_ws,
            pipe: Pipeline::new(),
            pool_ids: Vec::new(),
            resident: 0,
            prompt: 0,
            prefill_us: 0,
            decode_us: 0,
            pending: Vec::new(),
            page_t: kv_page_size(),
            max_tokens,
            chunk,
        })
    }

    fn ensure_pages(&mut self, tokens: u32) -> Result<()> {
        let need = tokens.div_ceil(self.page_t) as usize;
        if need > self.pool_ids.len() {
            let slots = self
                .ws
                .reserve((need - self.pool_ids.len()) as u32)
                .context("reserve KV pages")?;
            self.pool_ids.extend_from_slice(slots.ids());
        }
        Ok(())
    }

    async fn turn(&mut self, new_tokens: &[u32], on_first: &mut dyn FnMut()) -> Result<Vec<u32>> {
        let stop = chat::stop_tokens();
        let began = inferlet::monotonic_now_ns();
        let mut feed = std::mem::take(&mut self.pending);
        feed.extend_from_slice(new_tokens);
        if feed.is_empty() {
            feed.push(0);
        }
        let n0 = self.resident;
        let n = n0 + feed.len() as u32;
        self.prompt = n;
        self.ensure_pages(n + self.max_tokens as u32 + 2)?;
        let page_t = self.page_t;
        let pool_ids = self.pool_ids.clone();
        let pool_pages = pool_ids.len() as u32;
        let feed_i32: Vec<i32> = feed.iter().map(|&t| t as i32).collect();

        let spans = prefill_chunks(feed.len() as u32, self.chunk);
        let mut heads = Vec::with_capacity(spans.len());
        for &(base, end) in spans.iter() {
            let len = end - base;
            let abs_base = n0 + base;
            let abs_end = n0 + end;
            let toks = Channel::from(&feed_i32[base as usize..end as usize]).named("toks_p");
            let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_p");
            let positions = Channel::from_iter(abs_base..abs_end).named("positions_p");
            let w_slot = Channel::from_iter(
                (abs_base..abs_end).map(|c| pool_ids[(c / page_t) as usize]),
            )
            .named("w_slot_p");
            let w_off = Channel::from_iter((abs_base..abs_end).map(|c| c % page_t)).named("w_off_p");
            let klen = Channel::from([abs_end]).named("klen_p");
            let pages = Channel::from(pool_ids.clone()).named("pages_p");
            let page_indptr = Channel::from([0u32, abs_end.div_ceil(page_t)]).named("pidx_p");

            let fwd = ForwardPass::new();
            fwd.embed(&toks, &embed_indptr)?;
            fwd.attention(
                Some(KvBinding {
                    working_set: &self.ws,
                    geometry: KvGeometry {
                        readable_pages: ..,
                        writable_pages: ..,
                        kv_len: &klen,
                        pages: &pages,
                        page_indptr: &page_indptr,
                        w_slot: &w_slot,
                        w_off: &w_off,
                        positions: &positions,
                        mask: None,
                    },
                }),
                &self.rs_ws,
                RsGeometry {
                    fold_len: None,
                    buffer: 0..0,
                },
            )?;
            let head = Channel::new([1], dtype::i32).named("g0");
            let head_out = head.clone();
            fwd.epilogue(move || {
                head_out.put(greedy(intrinsics::logits()));
            });
            fwd.submit(&self.pipe).context("prefill submit")?;
            heads.push(head);
        }
        let mut g0 = 0i32;
        for head in heads {
            g0 = head.take_host::<i32>().await.context("prefill drain")?;
        }

        let mut generated = Vec::new();
        self.resident = n;
        let primed = inferlet::monotonic_now_ns();
        self.prefill_us = (primed - began) / 1000;
        if stop.contains(&(g0 as u32)) {
            on_first();
            return Ok(generated);
        }
        generated.push(g0 as u32);
        on_first();
        if self.max_tokens <= 1 {
            self.pending = vec![g0 as u32];
            return Ok(generated);
        }

        let slot_n = pool_ids[(n / page_t) as usize];
        let tok_in = Channel::from([g0]).named("tok_in");
        let pos = Channel::from([n]).named("pos");
        let fill = Channel::from([n + 1]).named("fill");
        let klen = Channel::from([n + 1]).named("klen");
        let w_slot = Channel::from([slot_n]).named("w_slot");
        let w_off = Channel::from([n % page_t]).named("w_off");
        let pages = Channel::from(pool_ids.clone()).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_t)]).named("page_indptr");
        let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
        let out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (n / page_t)..,
                    kv_len: &klen,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &pos,
                    mask: None,
                },
            }),
            &self.rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        fwd.epilogue(move || {
            let base = fill.take();
            let pids = pool_ids_ch.take();
            let token = greedy(intrinsics::logits());
            let logical_slot = &base / page_t;
            let w_slot_v = gather(&pids, &logical_slot);
            let w_off_v = &base % page_t;
            let klen_v = &base + 1u32;
            let next_free = &base + 1u32;
            let pages_v = reshape(&pids, [pool_pages]);
            let page_count = klen_v.div_ceil(page_t);
            let pidx_v = indptr(1, &page_count);

            tok_in.put(&token);
            out.put(&token);
            w_slot.put(&w_slot_v);
            w_off.put(&w_off_v);
            klen.put(&klen_v);
            pos.put(&base);
            fill.put(&next_free);
            pages.put(&pages_v);
            page_indptr.put(&pidx_v);
            pool_ids_ch.put(&pids);
        });

        let budget = self.max_tokens - 1;
        let mut fired = 0usize;
        let mut stopped = false;
        while fired < budget {
            fwd.submit(&self.pipe).context("decode submit")?;
            fired += 1;
            let t = out.take_host::<Vec<i32>>().await?;
            let token = *t.first().unwrap_or(&0) as u32;
            if stop.contains(&token) {
                stopped = true;
                break;
            }
            generated.push(token);
        }
        self.decode_us = (inferlet::monotonic_now_ns() - primed) / 1000;
        self.resident = n + fired as u32;
        if !stopped {
            self.pending = vec![*generated.last().unwrap()];
        }
        Ok(generated)
    }
}

fn emit(e: &str, turn: u32, tokens: Option<&[u32]>, context: Option<&[u32]>, fed: u32, sess: &Session) {
    let text = tokens.and_then(|t| model::decode(t).ok());
    let event = Event {
        e,
        turn,
        tokens,
        text,
        context,
        fed,
        prompt: sess.prompt,
        resident: sess.resident,
        prefill_us: sess.prefill_us,
        decode_us: sess.decode_us,
    };
    session::send(&serde_json::to_string(&event).unwrap_or_default());
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mut sess = Session::new(input.max_tokens, input.chunk)?;
    let mut history: Vec<u32> = Vec::new();

    if let Some(raw) = input.raw_tokens {
        let generated = sess.turn(&raw, &mut || {}).await?;
        emit("done", 0, Some(&generated), None, raw.len() as u32, &sess);
        sess.pipe.close();
        return Ok(String::new());
    }

    let mut new_tokens = chat::system_user(&input.system, &input.user);
    new_tokens.extend(chat::cue());
    let mut turn = 0u32;
    loop {
        let fed = new_tokens.len() as u32;
        if input.debug {
            history.extend_from_slice(&new_tokens);
        }
        let generated = {
            let mut on_first = || {
                session::send(&format!("{{\"e\":\"first\",\"turn\":{turn}}}"));
            };
            sess.turn(&new_tokens, &mut on_first).await?
        };
        let context = input.debug.then_some(history.as_slice());
        emit("done", turn, Some(&generated), context, fed, &sess);
        if input.debug {
            history.extend_from_slice(&generated);
        }
        let incoming = {
            let receive = std::pin::pin!(session::receive());
            let grace = std::pin::pin!(inferlet::sleep(PARK_AFTER));
            match futures::future::select(receive, grace).await {
                futures::future::Either::Left((message, _)) => message,
                futures::future::Either::Right((_, receive)) => {
                    sess.pipe.park();
                    receive.await
                }
            }
        };
        let Some(message) = incoming else {
            break;
        };
        let Ok(next) = serde_json::from_str::<Turn>(&message) else {
            break;
        };
        new_tokens = chat::seal();
        new_tokens.extend(chat::user(&next.user));
        new_tokens.extend(chat::cue());
        turn += 1;
    }
    sess.pipe.close();
    Ok(String::new())
}
