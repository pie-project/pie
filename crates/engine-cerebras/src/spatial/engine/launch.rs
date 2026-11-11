//! The engine's requests to the wafer program ([`wafer::Wafer`]): the token
//! loop and the forward.

use engine::error::{Error, Result as EngineResult};

use crate::spatial::wafer;
pub(super) use wafer::MODEL_NS;

/// A wafer model error as the engine's.
fn device(e: String) -> Error {
    Error::Device(format!("the wafer model: {e}"))
}

/// The token loop: `tokens` at `start..`, then up to `extra` steps each fed
/// the argmax before. The tokens the cache then holds past `start` and the
/// argmax after each where known (the feed's last row on).
pub(super) fn greedy(
    w: &mut wafer::Wafer,
    tokens: &[u32],
    start: u32,
    extra: usize,
) -> EngineResult<(Vec<u32>, Vec<Option<u32>>)> {
    let n = tokens.len();
    let argmax = w.greedy(tokens, start as usize, extra).map_err(device)?;
    let new = argmax[n - 1..].to_vec();
    if new.is_empty() {
        return Err(Error::Device(
            "the model's token loop answered nothing".into(),
        ));
    }
    // Fed: the tokens, then every new one but the last; the argmax after
    // the feed's last row is new[0], after new[i] it is new[i + 1].
    let mut fed = tokens.to_vec();
    fed.extend_from_slice(&new[..new.len() - 1]);
    let mut argmax = vec![None; n - 1];
    argmax.extend(new.iter().map(|&t| Some(t)));
    Ok((fed, argmax))
}

/// The logits `which` asks for (one row for `Last`, a row a token for
/// `All`) of `tokens` at `start..`.
pub(super) fn forward(
    w: &mut wafer::Wafer,
    tokens: &[u32],
    start: u32,
    which: wafer::Logits,
) -> EngineResult<Vec<Vec<f32>>> {
    w.forward(tokens, start as usize, which)
        .map(|(_, logits)| logits)
        .map_err(device)
}
