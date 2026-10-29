//! Speech to text through Qwen3-ASR's readings: a WAV clip's log-mel
//! through `audio.conv`, its frame rows through `audio.encode`, and those
//! rows spoken into the `text` reading in place of the audio tokens'
//! embeddings, which then writes the transcript. The prompt's markers, the
//! log-mel and the encoder's chunking are Qwen3-ASR's, stated here: no host
//! fact yet carries a clip into a model as a still is carried.
//!
//! The log-mel is Whisper's (`WhisperFeatureExtractor`): a periodic Hann
//! window of 400 samples every 160, a 128-band Slaney mel bank over 0–8 kHz,
//! `log10`, floored 8 below the clip's peak and mapped by `(x + 4) / 4`.
//!
//! The encoder's convolutions read the log-mel in chunks of 100 frames and
//! its attention in windows of 104 rows (8 chunks after the convolutions).
//! A voxel port takes one clip a pass, so whole chunks go a few to a clip, a
//! chunk a frame, and a short last chunk as a clip of its own; the windows
//! likewise.

use inferlet::eta::prelude::*;
use inferlet::model;
use serde::{Deserialize, Serialize};

const RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const MELS: usize = 128;
const CHUNK: usize = 100;
const WINDOW_CHUNKS: usize = 8;
/// Chunks a convolution pass takes: its clip is `MELS * CHUNK` rows a chunk,
/// and the engine's voxel ceiling (`max_voxels`) must hold them.
const CHUNKS_PER_PASS: usize = 2;

#[derive(Deserialize)]
struct Input {
    /// A WAV file (PCM 16-bit or float 32), base64.
    audio_b64: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Serialize)]
struct Output {
    text: String,
    language: String,
    frames: u32,
    audio_tokens: u32,
    generated: u32,
}

fn base64_decode(text: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return Err("audio_b64 is not base64".into()),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

/// A WAV file's samples, mono, at 16 kHz.
fn wav(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("the audio is not a WAV file".into());
    }
    let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (mut format, mut channels, mut rate, mut depth) = (0u16, 0usize, 0u32, 0u16);
    let mut at = 12;
    let mut data: Option<&[u8]> = None;
    while at + 8 <= bytes.len() {
        let size = u32_at(at + 4) as usize;
        let body = &bytes[at + 8..(at + 8 + size).min(bytes.len())];
        match &bytes[at..at + 4] {
            b"fmt " if body.len() >= 16 => {
                format = u16_at(at + 8);
                channels = usize::from(u16_at(at + 10));
                rate = u32_at(at + 12);
                depth = u16_at(at + 22);
            }
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + size + (size & 1);
    }
    let data = data.ok_or("the WAV file has no data chunk")?;
    if channels == 0 {
        return Err("the WAV file has no format chunk".into());
    }
    let frames: Vec<f32> = match (format, depth) {
        (1, 16) => data
            .chunks_exact(2 * channels)
            .map(|f| {
                f.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|s| f32::from(i16::from_le_bytes(*s)) / 32768.0)
                    .sum::<f32>()
                    / channels as f32
            })
            .collect(),
        (3, 32) => data
            .chunks_exact(4 * channels)
            .map(|f| {
                f.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|s| f32::from_le_bytes(*s))
                    .sum::<f32>()
                    / channels as f32
            })
            .collect(),
        _ => {
            return Err(format!(
                "a WAV of format {format} at {depth} bits is not read here"
            ));
        }
    };
    if rate == RATE || frames.is_empty() {
        return Ok(frames);
    }
    let out = (frames.len() as u64 * u64::from(RATE) / u64::from(rate)) as usize;
    Ok((0..out)
        .map(|i| {
            let x = i as f64 * f64::from(rate) / f64::from(RATE);
            let j = x as usize;
            let t = (x - j as f64) as f32;
            let a = frames[j.min(frames.len() - 1)];
            let b = frames[(j + 1).min(frames.len() - 1)];
            a + (b - a) * t
        })
        .collect())
}

fn hz_to_mel(f: f64) -> f64 {
    if f < 1000.0 {
        3.0 * f / 200.0
    } else {
        15.0 + (f / 1000.0).ln() * 27.0 / 6.4f64.ln()
    }
}

fn mel_to_hz(m: f64) -> f64 {
    if m < 15.0 {
        200.0 * m / 3.0
    } else {
        1000.0 * ((m - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

/// The Slaney-normalized triangular bank, `[MELS][bins]`.
fn mel_bank(bins: usize) -> Vec<Vec<f64>> {
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(f64::from(RATE) / 2.0));
    let edges: Vec<f64> = (0..MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64))
        .collect();
    let freq = |k: usize| f64::from(RATE) / 2.0 * k as f64 / (bins - 1) as f64;
    (0..MELS)
        .map(|m| {
            let norm = 2.0 / (edges[m + 2] - edges[m]);
            (0..bins)
                .map(|k| {
                    let f = freq(k);
                    let down = (f - edges[m]) / (edges[m + 1] - edges[m]);
                    let up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
                    down.min(up).max(0.0) * norm
                })
                .collect()
        })
        .collect()
}

/// Whisper's log-mel, `[MELS][frames]` row-major, and the frame count.
fn log_mel(samples: &[f32]) -> (Vec<f32>, usize) {
    let n = samples.len();
    let frames = n / HOP;
    let bins = N_FFT / 2 + 1;
    let window: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
        .collect();
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..N_FFT)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64;
            (a.cos(), a.sin())
        })
        .unzip();
    let bank = mel_bank(bins);
    let half = (N_FFT / 2) as isize;
    let at = |i: isize| -> f64 {
        let last = n as isize - 1;
        let j = if i < 0 {
            -i
        } else if i > last {
            2 * last - i
        } else {
            i
        };
        f64::from(samples[j.clamp(0, last.max(0)) as usize])
    };
    let mut out = vec![0f32; MELS * frames];
    let mut frame = vec![0f64; N_FFT];
    let mut power = vec![0f64; bins];
    for t in 0..frames {
        let start = (t * HOP) as isize - half;
        for (i, x) in frame.iter_mut().enumerate() {
            *x = at(start + i as isize) * window[i];
        }
        for (k, p) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0f64, 0f64);
            for (i, x) in frame.iter().enumerate() {
                let w = (i * k) % N_FFT;
                re += x * cos[w];
                im -= x * sin[w];
            }
            *p = re * re + im * im;
        }
        for (m, filter) in bank.iter().enumerate() {
            let e: f64 = filter.iter().zip(&power).map(|(f, p)| f * p).sum();
            out[m * frames + t] = e.max(1e-10).log10() as f32;
        }
    }
    let peak = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut out {
        *v = (v.max(peak - 8.0) + 4.0) / 4.0;
    }
    (out, frames)
}

/// Frames left of `len` after the three stride-2 convolutions.
fn after_convs(len: usize) -> usize {
    (0..3).fold(len, |l, _| (l - 1) / 2 + 1)
}

/// The sinusoid the encoder adds to a chunk's `at`-th row.
fn position(at: usize, width: usize) -> Vec<f32> {
    let half = width / 2;
    let step = 10_000f64.ln() / (half - 1) as f64;
    let mut row = vec![0f32; width];
    for i in 0..half {
        let angle = at as f64 * (-step * i as f64).exp();
        row[i] = angle.sin() as f32;
        row[half + i] = angle.cos() as f32;
    }
    row
}

/// A lane of reading `reading`, on `stream` if it has one.
struct Lane {
    reading: &'static str,
    stream: Option<&'static str>,
    decl: model::LaneDecl,
}

impl Lane {
    fn of(reading: &'static str, stream: Option<&'static str>) -> Result<Lane> {
        let mut switches = vec![("reading", reading)];
        switches.extend(stream.map(|s| ("stream", s)));
        let decl = model::lane_of(&switches)
            .map_err(|why| format!("this model has no `{reading}` lane: {why}"))?;
        Ok(Lane {
            reading,
            stream,
            decl,
        })
    }

    /// Its one input on the `axis`.
    fn input(&self, axis: model::InputAxis) -> Result<&model::InputDecl> {
        self.decl
            .inputs
            .iter()
            .find(|i| i.axis == axis)
            .ok_or_else(|| format!("the `{}` lane takes no {axis:?} input", self.reading))
    }

    /// The width of its output `name`.
    fn width(&self, name: &str) -> Result<u32> {
        self.decl
            .outputs
            .iter()
            .find(|o| o.name == name)
            .map(|o| o.width)
            .ok_or_else(|| format!("the `{}` lane writes no `{name}`", self.reading))
    }

    fn pass(&self) -> Result<ForwardPass> {
        let pass = ForwardPass::new();
        pass.switch(0, "reading", self.reading)?;
        if let Some(stream) = self.stream {
            pass.switch(0, "stream", stream)?;
        }
        Ok(pass)
    }
}

/// One pass of a voxel reading over the clip `x`, `rows` rows read out.
async fn voxels(
    lane: &Lane,
    output: &'static str,
    x: Channel,
    rows: u32,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let width = lane.width(output)?;
    let pass = lane.pass()?;
    pass.input(0, &lane.input(model::InputAxis::Voxel)?.name, &x)?;
    let out = Channel::new([rows, width], dtype::f32);
    pass.epilogue(move || out.put(intrinsics::output_rows(0, output, rows)));
    pass.submit(pipe)
        .with_context(|| format!("the `{}` pass", lane.reading))?;
    out.take_host::<Vec<f32>>().await
}

/// The encoder's rows for the clip's log-mel, `[rows][out]`.
async fn hear(mel: &[f32], frames: usize, pipe: &Pipeline) -> Result<(Vec<f32>, usize)> {
    let conv = Lane::of("audio.conv", None)?;
    let encode = Lane::of("audio.encode", None)?;
    let width = conv.width("features")? as usize;

    // The convolutions: whole chunks as one clip, a chunk a frame, then the
    // short last chunk alone.
    let full = frames / CHUNK;
    let tail = frames % CHUNK;
    let mut pieces: Vec<(usize, usize)> = (0..full)
        .step_by(CHUNKS_PER_PASS)
        .map(|c| ((full - c).min(CHUNKS_PER_PASS), CHUNK))
        .collect();
    if tail > 0 {
        pieces.push((1, tail));
    }
    // Each chunk's rows, its positions added.
    let mut rows: Vec<f32> = Vec::new();
    let mut start = 0;
    for &(count, len) in &pieces {
        let mut clip = Vec::with_capacity(count * MELS * len);
        for c in 0..count {
            for m in 0..MELS {
                let row = &mel[m * frames + start + c * len..][..len];
                clip.extend_from_slice(row);
            }
        }
        let x = if count > 1 {
            Channel::from_shaped([count as u32, MELS as u32, len as u32, 1], clip.as_slice())
        } else {
            Channel::from_shaped([MELS as u32, len as u32, 1], clip.as_slice())
        };
        let steps = after_convs(len);
        let out = voxels(&conv, "features", x, (count * steps) as u32, pipe).await?;
        for (i, step) in out.chunks_exact(width).enumerate() {
            rows.extend(
                step.iter()
                    .zip(position(i % steps, width))
                    .map(|(x, p)| x + p),
            );
        }
        start += count * len;
    }
    let row = width;
    let total = rows.len() / row;

    // The encoder: whole windows as one clip, a window a frame, then the
    // short last window alone.
    let window = after_convs(CHUNK) * WINDOW_CHUNKS;
    let whole = total / window;
    let rest = total % window;
    let out_width = encode.width("hidden")? as usize;
    let mut heard = Vec::with_capacity(total * out_width);
    if whole > 0 {
        let n = whole * window;
        let x = Channel::from_shaped(
            [whole as u32, window as u32, 1, row as u32],
            &rows[..n * row],
        );
        heard.extend(voxels(&encode, "hidden", x, n as u32, pipe).await?);
    }
    if rest > 0 {
        let from = whole * window * row;
        let x = Channel::from_shaped([rest as u32, 1, row as u32], &rows[from..]);
        heard.extend(voxels(&encode, "hidden", x, rest as u32, pipe).await?);
    }
    Ok((heard, total))
}

struct Kv {
    text: Lane,
    heard: Lane,
    ws: WorkingSet,
}

impl Kv {
    /// One fire of the text reading over `tokens` at rows `from..`; with
    /// `audio`, the rows are the encoder's. With `sample`, answers the
    /// argmax of the last row's logits.
    fn fire(
        &self,
        tokens: &[u32],
        from: u32,
        audio: Option<&[f32]>,
        pipe: &Pipeline,
        tag: &str,
        sample: bool,
    ) -> Result<Option<Channel>> {
        let to = from + tokens.len() as u32;
        let kv_len = Channel::from([to]);
        let slots = Channel::from_iter(from..to);
        let lane = if audio.is_some() {
            &self.heard
        } else {
            &self.text
        };
        let pass = lane.pass()?;
        if lane.decl.embeds {
            let toks = Channel::from_iter(tokens.iter().map(|&t| t as i32));
            let indptr = Channel::from([0u32, tokens.len() as u32]);
            pass.embed(&toks, &indptr)?;
        }
        if let Some(rows) = audio {
            let input = lane.input(model::InputAxis::Row)?;
            let x = Channel::from_shaped([tokens.len() as u32, input.width], rows);
            pass.input(0, &input.name, &x)?;
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
        let next = sample.then(|| Channel::new([1], dtype::i32));
        match next {
            Some(out) => pass.epilogue(move || {
                out.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]))
            }),
            // A fire with nothing to sample still runs a stage.
            None => {
                let done = Channel::new([1], dtype::u32);
                pass.epilogue(move || done.put(iota(1)));
            }
        }
        pass.submit(pipe)
            .with_context(|| format!("the `{tag}` fire"))?;
        Ok(next)
    }
}

fn one(text: &str) -> Result<u32> {
    match model::encode(text).as_slice() {
        [id] => Ok(*id),
        ids => Err(format!("`{text}` encodes to {ids:?}, not one marker")),
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let samples = wav(&base64_decode(&input.audio_b64)?)?;
    let (mel, frames) = log_mel(&samples);
    if frames == 0 {
        return Err("the clip is shorter than one frame".into());
    }
    let text = Lane::of("text", Some("text"))?;
    let spoken = Lane::of("text", Some("audio"))?;
    let pipe = Pipeline::new();
    let (heard, audio_tokens) = hear(&mel, frames, &pipe).await?;

    let pad = one("<|audio_pad|>")?;
    let stops = [one("<|im_end|>")?, one("<|endoftext|>")?];
    let head = model::encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>");
    let tail = model::encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    let n = (head.len() + audio_tokens + tail.len()) as u32;

    let ws = WorkingSet::new();
    ws.reserve(n + input.max_tokens as u32 + 1)
        .context("reserve KV slots")?;
    let kv = Kv {
        text,
        heard: spoken,
        ws,
    };

    let mut at = 0u32;
    kv.fire(&head, at, None, &pipe, "head", false)?;
    at += head.len() as u32;
    kv.fire(
        &vec![pad; audio_tokens],
        at,
        Some(&heard),
        &pipe,
        "audio",
        false,
    )?;
    at += audio_tokens as u32;
    let mut next = kv
        .fire(&tail, at, None, &pipe, "tail", true)?
        .ok_or("no readout")?;
    at += tail.len() as u32;

    let mut out: Vec<u32> = Vec::new();
    for step in 0..input.max_tokens {
        let token = next.take_host::<i32>().await? as u32;
        if stops.contains(&token) {
            break;
        }
        out.push(token);
        if step + 1 == input.max_tokens {
            break;
        }
        next = kv
            .fire(&[token], at, None, &pipe, &format!("d{step}"), true)?
            .ok_or("no readout")?;
        at += 1;
    }
    pipe.close();

    let raw = model::decode(&out).map_err(|e| format!("decode: {e:?}"))?;
    let (language, transcript) = match raw.split_once("<asr_text>") {
        Some((head, body)) => (
            head.trim_start_matches("language").trim().to_string(),
            body.trim().to_string(),
        ),
        None => (String::new(), raw.trim().to_string()),
    };
    Ok(Output {
        text: transcript,
        language,
        frames: frames as u32,
        audio_tokens: audio_tokens as u32,
        generated: out.len() as u32,
    })
}
