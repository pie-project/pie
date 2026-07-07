//! Non-sampling prefill mechanics — the KV-materialization twin of the
//! run-ahead decode carrier ([`crate::carrier`]).
//!
//! A prefill pass writes the KV cache for a span (text tokens, or a media clip's
//! soft tokens) at the current cursor and advances it, WITHOUT sampling — it
//! produces no output token. It is the raw-WIT form of `Context::flush` /
//! `Context::append_image` / `append_audio`, factored here (In Gim's
//! SDK-minimize thesis) so the geometry + cursor-advance is proven ONCE and the
//! inferlet composes an explicit, visible prompt: weave text prefills and media
//! prefills in order, then hand the trailing prompt tail to the decode loop's
//! first [`carrier::submit_pass`](crate::carrier::submit_pass).
//!
//! These do NOT touch the run-ahead carrier `fresh` flag — the #26 dangling
//! carrier clear (`fresh_generate`) belongs to the first *sampling* pass
//! (`submit_pass`), which runs after all prefill. Composes the KV page-geometry
//! keep-core primitive ([`crate::geometry`]).

use crate::geometry;
use crate::inference::ForwardPass;
use crate::media::{Audio, Image};
use crate::working_set::KvWorkingSet;
use crate::Result;

/// Materialize the KV for `tokens` at the current cursor `*seq_len` (a pure
/// prefill forward — geometry + `input_tokens` + `execute`, no sampler), then
/// advance the cursor by `tokens.len()`. A no-op on an empty span.
///
/// Mirrors `Context::flush` for an explicit token span. Use for the system /
/// user / cue prompt text that precedes (or is interleaved with) media, when
/// that text is not the trailing tail sampled by the first decode pass.
pub fn tokens(kv: &KvWorkingSet, seq_len: &mut u32, tokens: &[u32]) -> Result<()> {
    let n = tokens.len() as u32;
    if n == 0 {
        return Ok(());
    }
    let pass = ForwardPass::new();
    let geom = geometry::ensure_pages(
        kv,
        geometry::kv_write_geometry(*seq_len, n, kv.page_size()),
    )?;
    geometry::attach_kv_write(&pass, kv, &geom);
    let positions: Vec<u32> = (*seq_len..*seq_len + n).collect();
    pass.input_tokens(tokens, &positions);
    pass.execute();
    *seq_len += n;
    Ok(())
}

/// Materialize the soft-token KV for an encoded `image` at the current cursor
/// (the host vision encoder runs driver-side; `token_count()` soft tokens occupy
/// KV rows), then advance the cursor past them. A no-op when the image encodes
/// to zero soft tokens.
///
/// Mirrors the media forward inside `Context::append_image` — the model's own
/// span delimiters (`image.prefix_tokens()` / `suffix_tokens()`) are the
/// inferlet's to weave in via [`tokens`] before / after this call, so ordering
/// (text → image → text) stays explicit and model-agnostic.
pub fn image(kv: &KvWorkingSet, seq_len: &mut u32, image: &Image) -> Result<()> {
    let n = image.token_count();
    if n == 0 {
        return Ok(());
    }
    let pass = ForwardPass::new();
    let geom =
        geometry::ensure_pages(kv, geometry::kv_write_geometry(*seq_len, n, kv.page_size()))?;
    geometry::attach_kv_write(&pass, kv, &geom);
    pass.input_image(image, *seq_len);
    pass.execute();
    *seq_len += n;
    Ok(())
}

/// Materialize the soft-token KV for an encoded `audio` clip at the current
/// cursor (the host audio encoder runs driver-side), then advance the cursor.
/// A no-op when the clip encodes to zero soft tokens. The twin of [`image`].
pub fn audio(kv: &KvWorkingSet, seq_len: &mut u32, audio: &Audio) -> Result<()> {
    let n = audio.token_count();
    if n == 0 {
        return Ok(());
    }
    let pass = ForwardPass::new();
    let geom =
        geometry::ensure_pages(kv, geometry::kv_write_geometry(*seq_len, n, kv.page_size()))?;
    geometry::attach_kv_write(&pass, kv, &geom);
    pass.input_audio(audio, *seq_len);
    pass.execute();
    *seq_len += n;
    Ok(())
}
