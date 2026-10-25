//! Text to speech through Qwen3-TTS's talker and code predictor: the talker
//! speaks a frame's first codebook from projected text and codec rows, the
//! predictor fills in the other fifteen, and the frame's codes, embedded and
//! summed, are the talker's next row. The codec's readings then turn the
//! frames into 24 kHz audio, sent as `speech.wav`: once at the end, or
//! streamed a piece at a time as the frames come. The codec's codes, its
//! control tokens and its speakers are Qwen3-TTS's, stated here.

use inferlet::eta::prelude::*;
use inferlet::frames::{AudioFormat, Pcm};
use inferlet::{model, session};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const TALKER_CODES: u32 = 3072;
/// The codec table's zero row, a seed row's token.
const ZERO: u32 = TALKER_CODES + 15 * BOOK;
const BOOK: u32 = 2048;
const BOOKS: usize = 16;
const PAD: u32 = 2148;
const BOS: u32 = 2149;
const EOS: u32 = 2150;
const THINK: u32 = 2154;
const NO_THINK: u32 = 2155;
const THINK_BOS: u32 = 2156;
const THINK_EOS: u32 = 2157;

#[derive(Deserialize)]
struct Input {
    text: String,
    #[serde(default = "default_speaker")]
    speaker: String,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default = "default_max_frames")]
    max_frames: usize,
    /// Argmax every code instead of sampling (for checking against a reference).
    #[serde(default)]
    greedy: bool,
    #[serde(default)]
    seed: u64,
    /// Answer the codes too.
    #[serde(default)]
    codes: bool,
    /// Send the audio a `speech.wav` at a time as it is decoded, rather
    /// than once at the end.
    #[serde(default)]
    stream: bool,
    /// Seconds of audio the listener already has queued when a stream
    /// starts, which its pieces may take their time behind.
    #[serde(default)]
    ahead: f32,
    /// Frames to decode as they are, skipping the talker: a list, or that
    /// list as JSON text (what `pie run --frames` hands over).
    #[serde(default, deserialize_with = "frames")]
    frames: Option<Vec<[u32; BOOKS]>>,
}

fn frames<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<Vec<[u32; BOOKS]>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Given {
        List(Vec<[u32; BOOKS]>),
        Text(String),
    }
    Ok(match Option::<Given>::deserialize(d)? {
        None => None,
        Some(Given::List(frames)) => Some(frames),
        Some(Given::Text(text)) => {
            Some(inferlet::serde_json::from_str(&text).map_err(serde::de::Error::custom)?)
        }
    })
}

fn default_speaker() -> String {
    "ryan".into()
}
fn default_language() -> String {
    "auto".into()
}
fn default_max_frames() -> usize {
    750
}

#[derive(Serialize)]
struct Output {
    frames: usize,
    seconds: f32,
    /// Seconds spent: making the frames, and decoding them.
    spent: [f32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    codes: Option<Vec<[u32; BOOKS]>>,
}

const RATE: u32 = 24_000;
const LATENT: usize = 512;

/// Frames already decoded that a decode reads ahead of its new ones: the
/// reach of the decoder's causal convs, 18009 samples.
const CONTEXT: usize = 10;
/// A stream's first piece, in frames, and its shortest after that: shorter
/// pieces spend more on the frames they decode again than the audio they add.
const FIRST: usize = 18;
const LEAST: usize = 20;

/// When a stream's next piece is due: the first once `FIRST` frames are in,
/// each later one only when what was sent is about to run out, so every
/// piece is as long as the device's pace allows.
struct Pace {
    runs_out: Option<Instant>,
    /// A frame's decode and a frame's generation, as last seen.
    decode: Duration,
    frame: Duration,
    last: Instant,
}

impl Pace {
    fn new(ahead: f32) -> Pace {
        let now = Instant::now();
        Pace {
            runs_out: (ahead > 0.0).then(|| now + Duration::from_secs_f32(ahead)),
            // Until a piece is timed, a frame's decode at a phone's pace.
            decode: Duration::from_millis(20),
            frame: Duration::ZERO,
            last: now,
        }
    }

    fn due(&mut self, pending: usize, first: bool) -> bool {
        let now = Instant::now();
        self.frame = now - self.last;
        self.last = now;
        let least = if first { FIRST } else { LEAST };
        pending >= least
            && self.runs_out.is_none_or(|end| {
                let need = self.decode * (pending + CONTEXT) as u32
                    + self.frame
                    + Duration::from_millis(150);
                end.saturating_duration_since(now) < need
            })
    }

    fn sent(&mut self, decoded: usize, took: Duration, samples: usize) {
        self.decode = took / decoded.max(1) as u32;
        let now = Instant::now();
        let heard = Duration::from_secs_f64(samples as f64 / f64::from(RATE));
        self.runs_out = Some(self.runs_out.map_or(now, |end| end.max(now)) + heard);
        self.last = now;
    }
}

/// The codec turning frames into audio as they come: each frame's latent and
/// prenet row once, then the decoder over the frames not yet heard behind
/// `CONTEXT` already decoded.
struct Voice {
    quantize: Lane,
    prenet: Kv,
    decode: Lane,
    latents: Vec<f32>,
    hidden: Vec<f32>,
    width: u32,
    heard: usize,
}

impl Voice {
    fn new() -> Result<Voice> {
        let decode = Lane::of("codec.decode")?;
        let width = decode.input("frames")?.width;
        Ok(Voice {
            quantize: Lane::of("codec.quantize")?,
            prenet: Kv::new(Lane::of("codec.prenet")?, 0)?,
            decode,
            latents: Vec::new(),
            hidden: Vec::new(),
            width,
            heard: 0,
        })
    }

    fn frames(&self) -> usize {
        self.latents.len() / LATENT
    }

    /// The prenet's rows for `frames`, the next after those taken.
    async fn take(&mut self, frames: &[[u32; BOOKS]], pipe: &Pipeline) -> Result<()> {
        if frames.is_empty() {
            return Ok(());
        }
        let from = self.frames();
        let t = frames.len();
        // A fire reads out a few hundred rows at most; 32 frames is 512.
        let width = self.quantize.width()? as usize;
        let mut rows = Vec::with_capacity(t * BOOKS * width);
        for part in frames.chunks(32) {
            let codes: Vec<u32> = part
                .iter()
                .flat_map(|f| f.iter().enumerate().map(|(k, &c)| k as u32 * BOOK + c))
                .collect();
            rows.extend(embed(&self.quantize, &codes, pipe).await?);
        }
        let mut latents = vec![0f32; t * LATENT];
        for (f, latent) in latents.as_chunks_mut::<LATENT>().0.iter_mut().enumerate() {
            for k in 0..BOOKS {
                let row = &rows[(f * BOOKS + k) * width..][..width];
                let half = if k == 0 {
                    &row[..LATENT]
                } else {
                    &row[LATENT..]
                };
                for (l, x) in latent.iter_mut().zip(half) {
                    *l += x;
                }
            }
        }
        self.latents.extend(latents);
        // Each frame beside the two before it: the causal conv's three taps.
        let mut taps = vec![0f32; t * 3 * LATENT];
        for f in 0..t {
            for j in 0..3 {
                if let Some(src) = (from + f + j).checked_sub(2) {
                    taps[(f * 3 + j) * LATENT..][..LATENT]
                        .copy_from_slice(&self.latents[src * LATENT..][..LATENT]);
                }
            }
        }
        self.prenet.grow((from + t) as u32)?;
        let taps = Channel::from_shaped([t as u32, (3 * LATENT) as u32], taps.as_slice());
        let hidden = self
            .prenet
            .fire(&vec![0; t], from as u32, &[("taps", &taps)], pipe)?
            .all()
            .await?;
        self.hidden.extend(hidden);
        Ok(())
    }

    /// The audio of the frames taken and not yet heard, clamped to [-1, 1].
    async fn hear(&mut self, pipe: &Pipeline) -> Result<Vec<f32>> {
        let to = self.frames();
        if to == self.heard {
            return Ok(Vec::new());
        }
        let from = self.heard.saturating_sub(CONTEXT);
        let t = (to - from) as u32;
        let w = self.width as usize;
        let samples = t * 1920;
        let pass = self.decode.pass()?;
        pass.input(
            0,
            "frames",
            &Channel::from_shaped([t, 1, 1, self.width], &self.hidden[from * w..to * w]),
        )?;
        // The audio is the decoder's first channel.
        let channels = self.decode.width()?;
        let out = Channel::new([samples, 1], dtype::f32);
        pass.epilogue(move || {
            let all = reshape(intrinsics::pixels(0, samples, channels), [samples * channels]);
            out.put(reshape(gather(&all, iota(samples) * channels), [samples, 1]));
        });
        pass.submit(pipe).context("the codec decode")?;
        let wave: Vec<f32> = out.take_host::<Vec<f32>>().await?;
        let skip = (self.heard - from) * 1920;
        self.heard = to;
        Ok(wave[skip..].iter().map(|x| x.clamp(-1.0, 1.0)).collect())
    }
}

/// `samples` out as `speech.wav`.
fn send(samples: &[f32]) -> Result<()> {
    let pcm = Pcm::from_f32(samples, RATE, 1)?;
    session::send_pcm(&pcm, AudioFormat::Wav, "speech.wav")?;
    Ok(())
}

fn speaker(name: &str) -> Result<(u32, Option<u32>)> {
    Ok(match name.to_lowercase().as_str() {
        "serena" => (3066, None),
        "vivian" => (3065, None),
        "uncle_fu" => (3010, None),
        "ryan" => (3061, None),
        "aiden" => (2861, None),
        "ono_anna" => (2873, None),
        "sohee" => (2864, None),
        "eric" => (2875, Some(2062)),
        "dylan" => (2878, Some(2074)),
        other => return Err(format!("no speaker `{other}`")),
    })
}

fn language(name: &str) -> Result<Option<u32>> {
    Ok(Some(match name.to_lowercase().as_str() {
        "auto" => return Ok(None),
        "chinese" => 2055,
        "english" => 2050,
        "german" => 2053,
        "italian" => 2070,
        "portuguese" => 2071,
        "spanish" => 2054,
        "japanese" => 2058,
        "korean" => 2064,
        "french" => 2061,
        "russian" => 2069,
        other => return Err(format!("no language `{other}`")),
    }))
}

/// A lane of reading `reading`.
#[derive(Clone)]
struct Lane {
    reading: &'static str,
    decl: model::LaneDecl,
}

impl Lane {
    fn of(reading: &'static str) -> Result<Lane> {
        let decl = model::lane_of(&[("reading", reading)])
            .map_err(|why| format!("this model has no `{reading}` lane: {why}"))?;
        Ok(Lane { reading, decl })
    }

    fn input(&self, name: &str) -> Result<&model::InputDecl> {
        self.decl
            .inputs
            .iter()
            .find(|i| i.name == name)
            .ok_or_else(|| format!("the `{}` lane takes no input `{name}`", self.reading))
    }

    fn width(&self) -> Result<u32> {
        self.decl
            .readout
            .as_ref()
            .map(|r| r.width)
            .ok_or_else(|| format!("the `{}` lane reads nothing out", self.reading))
    }

    fn pass(&self) -> Result<ForwardPass> {
        let pass = ForwardPass::new();
        pass.switch(0, "reading", self.reading)?;
        Ok(pass)
    }
}

fn one(text: &str) -> Result<u32> {
    match model::encode(text).as_slice() {
        [id] => Ok(*id),
        ids => Err(format!("`{text}` encodes to {ids:?}, not one marker")),
    }
}

/// A reading's KV, its slots reserved as its rows need them.
struct Kv {
    lane: Lane,
    ws: WorkingSet,
    reserved: u32,
}

impl Kv {
    fn new(lane: Lane, rows: u32) -> Result<Kv> {
        let mut kv = Kv {
            lane,
            ws: WorkingSet::new(),
            reserved: 0,
        };
        kv.grow(rows)?;
        Ok(kv)
    }

    /// Slots for `rows` rows in all.
    fn grow(&mut self, rows: u32) -> Result<()> {
        if rows > self.reserved {
            self.ws.reserve(rows).context("reserve KV slots")?;
            self.reserved = rows;
        }
        Ok(())
    }

    /// A pass over `tokens` at rows `from..`, every row read out, its
    /// `inputs` bound, for the caller to give its epilogue.
    fn pass(
        &self,
        tokens: &Channel,
        count: u32,
        from: u32,
        inputs: &[(&str, &Channel)],
    ) -> Result<ForwardPass> {
        let to = from + count;
        let indptr = Channel::from([0u32, count]);
        let kv_len = Channel::from([to]);
        let slots = Channel::from_iter(from..to);
        let pass = self.lane.pass()?;
        pass.embed(tokens, &indptr)?;
        for (name, ch) in inputs {
            pass.input(0, name, ch)?;
        }
        pass.state(
            &[&self.ws],
            StateGeometry {
                positions: Some(&slots),
                slots: Some(&slots),
                kv_len: Some(&kv_len),
                fold: Fold::All,
            },
        )?;
        pass.readout(&Channel::from_iter(0..count))?;
        Ok(pass)
    }

    /// One fire over `tokens` at rows `from..`, its `inputs` bound,
    /// answering every row's readout.
    fn fire(
        &self,
        tokens: &[u32],
        from: u32,
        inputs: &[(&str, &Channel)],
        pipe: &Pipeline,
    ) -> Result<Fired> {
        let count = tokens.len() as u32;
        let toks = Channel::from_iter(tokens.iter().map(|&t| t as i32));
        let pass = self.pass(&toks, count, from, inputs)?;
        let width = self.lane.width()?;
        let out = Channel::new([count, width], dtype::f32);
        pass.epilogue(move || out.put(intrinsics::hidden(0, count, width)));
        pass.submit(pipe)
            .with_context(|| format!("the `{}` fire", self.lane.reading))?;
        Ok(Fired { out })
    }
}

/// A fire's readout, every row of it.
struct Fired {
    out: Channel,
}

impl Fired {
    async fn all(self) -> Result<Vec<f32>> {
        self.out.take_host::<Vec<f32>>().await
    }
}

/// The rows `lane` reads out for `codes`, one a code.
async fn embed(lane: &Lane, codes: &[u32], pipe: &Pipeline) -> Result<Vec<f32>> {
    let width = lane.width()?;
    let n = codes.len() as u32;
    let pass = lane.pass()?;
    pass.embed(
        &Channel::from_iter(codes.iter().map(|&t| t as i32)),
        &Channel::from([0u32, n]),
    )?;
    pass.readout(&Channel::from_iter(0..n))?;
    let out = Channel::new([n, width], dtype::f32);
    pass.epilogue(move || out.put(intrinsics::hidden(0, n, width)));
    pass.submit(pipe)
        .with_context(|| format!("the `{}` fire", lane.reading))?;
    out.take_host::<Vec<f32>>().await
}

/// A code off `logits`: their argmax, or a draw at temperature 0.9 from
/// the top 50 (Qwen3-TTS's sampling), on the device.
fn draw(logits: &Tensor, greedy: bool, rng: &Tensor) -> Tensor {
    if greedy {
        return reduce_argmax(logits);
    }
    let (top, index) = top_k(logits, 50);
    let pick = gumbel_max(&top / 0.9f32, rng);
    cast(
        gather(&index, reshape(cast(&pick, dtype::u32), [1])),
        dtype::i32,
    )
}

/// Frames whose codes the host has yet to collect while later ones run.
const LOOKAHEAD: usize = 2;

/// What one frame's passes hand each other on the device. The host reads
/// only the codes, which land while later frames run.
struct Links {
    /// The predictor's next token: the first code, then each code drawn.
    token: Channel,
    /// The talker's hidden, the predictor's seed row.
    seed: Channel,
    /// The frame's codes as rows of the codec table: the talker's next row
    /// is their sum.
    joined: Channel,
    /// The first code, and the other fifteen.
    first: Channel,
    rest: Channel,
    /// The KV the frame's predictor passes write, held until they land.
    predicted: Option<Kv>,
}

impl Links {
    fn new(hidden: u32) -> Links {
        Links {
            token: Channel::new([1], dtype::i32).queued(),
            seed: Channel::new([1, hidden], dtype::f32).queued(),
            joined: Channel::new([1, BOOKS as u32], dtype::i32).queued(),
            first: Channel::new([1], dtype::i32),
            rest: Channel::new([1], dtype::i32).capacity(BOOKS as u32),
            predicted: None,
        }
    }

    /// The frame's codes, or none at the end of speech.
    async fn collect(&self) -> Result<Option<[u32; BOOKS]>> {
        let first = *self
            .first
            .take_host::<Vec<i32>>()
            .await?
            .first()
            .unwrap_or(&0) as u32;
        let mut frame = [0u32; BOOKS];
        frame[0] = first;
        for code in frame[1..].iter_mut() {
            *code = *self
                .rest
                .take_host::<Vec<i32>>()
                .await?
                .first()
                .unwrap_or(&0) as u32;
        }
        Ok((first != EOS).then_some(frame))
    }
}

/// The talker's draw of a frame's first code, on the device: its logits
/// with each code said before penalized once (transformers' repetition
/// penalty) and the codec's control codes masked (end of speech allowed from
/// the third frame), then the next frame's links filled in.
struct Talk {
    width: u32,
    hidden: u32,
    greedy: bool,
    seen: Channel,
    rng: Channel,
}

impl Talk {
    /// The draw after `pass`; the prefill's (`fresh`, the sampler's seed)
    /// starts the penalty and the random stream the later ones carry on.
    fn draw(&self, pass: &ForwardPass, rows: u32, frame: usize, links: &Links, fresh: Option<u32>) {
        let (width, hidden, greedy) = (self.width, self.hidden, self.greedy);
        let (seen, rng) = (self.seen, self.rng);
        let (token, seed, joined, first) = (links.token, links.seed, links.joined, links.first);
        pass.epilogue(move || {
            let all = reshape(intrinsics::hidden(0, rows, width), [rows * width]);
            let last = gather(&all, iota(width) + (rows - 1) * width);
            let logits = gather(&last, iota(TALKER_CODES));
            let state = gather(&last, iota(hidden) + TALKER_CODES);
            let said = match fresh {
                Some(_) => broadcast(0.0f32, [TALKER_CODES]),
                None => seen.take(),
            };
            let penalized = select(
                gt(&said, 0.0f32),
                select(lt(&logits, 0.0f32), &logits * 1.05f32, &logits / 1.05f32),
                &logits,
            );
            let index = iota(TALKER_CODES);
            let speech = lt(&index, BOOK);
            let allowed = if frame >= 2 {
                or(speech, eq(&index, EOS))
            } else {
                speech
            };
            let masked = mask_apply(&penalized, allowed);
            let r = match fresh {
                Some(start) => select(
                    eq(iota(2), 0u32),
                    broadcast(start, [2]),
                    broadcast(1u32, [2]),
                ),
                None => rng.take(),
            };
            let drawn = draw(&masked, greedy, &r);
            let code = reshape(drawn, [1]);
            seen.put(scatter_set(
                &said,
                cast(&code, dtype::u32),
                broadcast(1.0f32, [1]),
            ));
            rng.put(&r + iota(2));
            token.put(&code);
            first.put(&code);
            joined.put(reshape(
                scatter_set(broadcast(0i32, [BOOKS as u32]), broadcast(0u32, [1]), &code),
                [1, BOOKS as u32],
            ));
            seed.put(reshape(state, [1, hidden]));
        });
    }
}

/// A frame's predictor passes over a KV of the frame's own, drawing its
/// other fifteen books on the device: one seeds the KV with the talker's
/// hidden, the other runs fifteen times, each epilogue drawing the next code
/// from its book's logits and feeding it straight back as the next row's
/// token.
fn draw_books(
    kv: &Kv,
    links: &Links,
    greedy: bool,
    seed: u32,
) -> Result<(ForwardPass, ForwardPass)> {
    let first = kv.pass(&Channel::from([ZERO as i32]), 1, 0, &[("seed", &links.seed)])?;
    let settled = Channel::new([1], dtype::f32).queued();
    let books = kv.lane.width()?;
    first.epilogue(move || {
        settled.put(gather(
            reshape(intrinsics::hidden(0, 1, books), [books]),
            broadcast(0u32, [1]),
        ))
    });

    let tok_in = links.token;
    let lane = Channel::from([0u32, 1]);
    let pos = Channel::from([1u32]);
    let kv_len = Channel::from([2u32]);
    let base = Channel::from([0u32]);
    let rng = Channel::from([seed, 1u32]);
    let width = kv.lane.input("seed")?.width;
    let zeros = Channel::from_shaped([1, width], vec![0f32; width as usize].as_slice());
    let (out, joined) = (links.rest, links.joined);

    let pass = kv.lane.pass()?;
    pass.embed(&tok_in, &lane)?;
    pass.input(0, "seed", &zeros)?;
    pass.state(
        &[&kv.ws],
        StateGeometry {
            positions: Some(&pos),
            slots: Some(&pos),
            kv_len: Some(&kv_len),
            fold: Fold::All,
        },
    )?;
    pass.epilogue(move || {
        let at = base.take();
        let r = rng.take();
        let here = pos.take();
        let codes = reshape(joined.take(), [BOOKS as u32]);
        let book = reshape(intrinsics::hidden(0, 1, books), [books]);
        let drawn = draw(&book, greedy, &r);
        let drawn = reshape(drawn, [1]);
        out.put(&drawn);
        let row = &drawn + cast(&at, dtype::i32) + TALKER_CODES as i32;
        tok_in.put(&row);
        joined.put(reshape(
            scatter_set(&codes, &at / BOOK + 1u32, &row),
            [1, BOOKS as u32],
        ));
        let next = &here + 1u32;
        pos.put(&next);
        kv_len.put(&next + 1u32);
        base.put(&at + BOOK);
        rng.put(&r + iota(2));
    });
    Ok((first, pass))
}

/// The seeds of the device's draws.
struct Sampler {
    state: u64,
}

impl Sampler {
    fn next(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if let Some(frames) = &input.frames {
        let pipe = Pipeline::new();
        let mut voice = Voice::new()?;
        voice.take(frames, &pipe).await?;
        let audio = voice.hear(&pipe).await?;
        pipe.close();
        send(&audio)?;
        return Ok(Output {
            frames: frames.len(),
            seconds: audio.len() as f32 / RATE as f32,
            spent: [0.0; 2],
            codes: None,
        });
    }
    let talker = Lane::of("talker")?;
    let predictor = Lane::of("predictor")?;
    let pipe = Pipeline::new();

    let (spk, dialect) = speaker(&input.speaker)?;
    let mut lang = language(&input.language)?;
    if dialect.is_some() && matches!(input.language.to_lowercase().as_str(), "chinese" | "auto") {
        lang = dialect;
    }
    let ids = model::encode(&format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        input.text
    ));
    if ids.len() < 9 {
        return Err("nothing to say".into());
    }
    let (role, text) = (&ids[..3], &ids[3..ids.len() - 5]);
    let (tts_pad, tts_bos, tts_eos) = (
        one("<tts_pad>")?,
        one("<tts_text_bos>")?,
        one("<tts_text_eod>")?,
    );

    // The prefill: the role's text alone, then the codec prefix under pads,
    // tts_bos, the text, tts_eos, each beside codec_pad, and tts_pad beside
    // codec_bos.
    let mut prefix = match lang {
        Some(l) => vec![THINK, THINK_BOS, l, THINK_EOS],
        None => vec![NO_THINK, THINK_BOS, THINK_EOS],
    };
    prefix.extend([spk, PAD]);
    let mut tokens: Vec<u32> = role.to_vec();
    let mut codec: Vec<Option<u32>> = vec![None; role.len()];
    tokens.extend(std::iter::repeat_n(tts_pad, prefix.len() - 1));
    tokens.push(tts_bos);
    codec.extend(prefix.iter().map(|&c| Some(c)));
    tokens.extend_from_slice(text);
    tokens.push(tts_eos);
    codec.extend(std::iter::repeat_n(Some(PAD), text.len() + 1));
    tokens.push(tts_pad);
    codec.push(Some(BOS));

    // A row's codec part: its one code, or none.
    let mut codes = vec![0i32; tokens.len() * BOOKS];
    let mut weights = vec![0f32; tokens.len() * BOOKS];
    for (row, code) in codec.iter().enumerate() {
        if let Some(code) = code {
            codes[row * BOOKS] = *code as i32;
            weights[row * BOOKS] = 1.0;
        }
    }

    let n = tokens.len() as u32;
    let mut talk = Kv::new(talker, n)?;
    let mut voice = Voice::new()?;
    let mut audio: Vec<f32> = Vec::new();
    let mut pace = Pace::new(input.ahead);
    let started = Instant::now();
    let mut sampler = Sampler {
        state: input.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
    };
    let mut spent = [0f32; 2];
    let mut frames: Vec<[u32; BOOKS]> = Vec::new();

    // Every frame's passes go in back to back, each reading what the one
    // before it left on the device; the host only collects the codes, a
    // few frames behind, and stops at the end of speech.
    let talk_out = Talk {
        width: talk.lane.width()?,
        hidden: predictor.input("seed")?.width,
        greedy: input.greedy,
        seen: Channel::new([TALKER_CODES], dtype::f32).queued(),
        rng: Channel::new([2], dtype::u32).queued(),
    };
    let mut links = Links::new(talk_out.hidden);
    let books = BOOKS as u32;
    let prefill = talk.pass(
        &Channel::from_iter(tokens.iter().map(|&t| t as i32)),
        n,
        0,
        &[
            ("codes", &Channel::from_shaped([n, books], codes.as_slice())),
            (
                "code_weights",
                &Channel::from_shaped([n, books], weights.as_slice()),
            ),
        ],
    )?;
    // A frame's row sums all sixteen of its codes.
    let summed = Channel::from_shaped([1, books], [1f32; BOOKS].as_slice());
    talk_out.draw(
        &prefill,
        n,
        0,
        &links,
        Some(sampler.next().to_bits() as u32),
    );
    prefill.submit(&pipe).context("the talker's prefill")?;
    let mut inflight: std::collections::VecDeque<Links> = std::collections::VecDeque::new();
    let mut ended = false;
    for (f, at) in (0..input.max_frames).zip(n..) {
        let predict = Kv::new(predictor.clone(), BOOKS as u32 + 1)?;
        let (seed, chain) = draw_books(
            &predict,
            &links,
            input.greedy,
            sampler.next().to_bits() as u32,
        )?;
        let next = Links::new(talk_out.hidden);
        talk.grow(at + 1)?;
        let step = talk.pass(
            &Channel::from([tts_pad as i32]),
            1,
            at,
            &[("codes", &links.joined), ("code_weights", &summed)],
        )?;
        talk_out.draw(&step, 1, f + 1, &next, None);
        // The frame's passes in order, as many to a submitted frame as the
        // runtime takes: the seed, the fifteen draws, the talker.
        let passes: Vec<Option<&ForwardPass>> = std::iter::once(&seed)
            .chain(std::iter::repeat_n(&chain, BOOKS - 1))
            .chain([&step])
            .map(Some)
            .collect();
        for slots in passes.chunks(frame_size()) {
            submit_frame(&pipe, slots).context("a frame's passes")?;
        }
        links.predicted = Some(predict);
        inflight.push_back(std::mem::replace(&mut links, next));
        while inflight.len() > LOOKAHEAD || (f + 1 == input.max_frames && !inflight.is_empty()) {
            let done = inflight.pop_front().expect("a frame in flight");
            let Some(frame) = done.collect().await? else {
                ended = true;
                break;
            };
            frames.push(frame);
            let taken = voice.frames();
            if input.stream && pace.due(frames.len() - taken, taken == 0) {
                let clock = Instant::now();
                let decoded = frames.len() - taken + voice.heard.min(CONTEXT);
                voice.take(&frames[taken..], &pipe).await?;
                let piece = voice.hear(&pipe).await?;
                spent[1] += clock.elapsed().as_secs_f32();
                pace.sent(decoded, clock.elapsed(), piece.len());
                send(&piece)?;
                audio.extend(piece);
            }
        }
        if ended {
            break;
        }
    }
    spent[0] = started.elapsed().as_secs_f32() - spent[1];
    let clock = Instant::now();
    voice.take(&frames[voice.frames()..], &pipe).await?;
    let piece = voice.hear(&pipe).await?;
    spent[1] += clock.elapsed().as_secs_f32();
    pipe.close();
    if input.stream {
        if !piece.is_empty() {
            send(&piece)?;
        }
        audio.extend(piece);
    } else {
        audio = piece;
        send(&audio)?;
    }
    Ok(Output {
        frames: frames.len(),
        seconds: audio.len() as f32 / RATE as f32,
        spent,
        codes: input.codes.then_some(frames),
    })
}
