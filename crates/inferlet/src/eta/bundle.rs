//! An adapter bundle: the bytes of one trained `adapter.safetensors`, checked
//! and packed into a channel the engine reads. The guest never learns where
//! the adapter attaches; the engine fills each registered bank the file
//! names and zeroes the rest.

use serde_json::Value;

use super::Channel;

/// A trained adapter, ready to attach to forward passes with
/// [`ForwardPass::adapter`](super::ForwardPass::adapter). Passes that attach
/// the same bytes share one engine slot.
#[derive(Clone, Copy)]
pub struct Adapter {
    /// The safetensors bytes, zero-padded to 4 and packed little-endian into
    /// u32 words.
    pub(crate) bundle: Channel,
}

impl Adapter {
    /// Checks that `bytes` is a safetensors file whose tensors are F32, BF16
    /// or F16 and lie inside it, then packs it into one u32 channel.
    pub fn from_bundle(bytes: &[u8]) -> Result<Adapter, String> {
        check(bytes)?;
        let words: Vec<u32> = bytes
            .chunks(4)
            .map(|chunk| {
                let mut word = [0u8; 4];
                word[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(word)
            })
            .collect();
        Ok(Adapter {
            bundle: Channel::from(&words[..]),
        })
    }
}

/// Refuses bytes that are not a safetensors file of F32, BF16 or F16 tensors,
/// naming the tensor at fault.
fn check(bytes: &[u8]) -> Result<(), String> {
    let bad = |why: &str| format!("adapter bundle: not a safetensors file: {why}");
    let len = bytes
        .get(..8)
        .map(|head| u64::from_le_bytes(head.try_into().expect("eight bytes")))
        .ok_or_else(|| bad("shorter than its header length"))?;
    let end = usize::try_from(len)
        .ok()
        .and_then(|len| len.checked_add(8))
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| bad("its header runs past the file"))?;
    let header: Value = serde_json::from_slice(&bytes[8..end])
        .map_err(|e| bad(&format!("its header is not JSON: {e}")))?;
    let tensors = header
        .as_object()
        .ok_or_else(|| bad("its header is not a JSON object"))?;
    let data = bytes.len() - end;
    for (name, tensor) in tensors.iter().filter(|(name, _)| *name != "__metadata__") {
        let refuse = |why: &str| format!("adapter bundle: tensor `{name}` {why}");
        let width = match tensor.get("dtype").and_then(Value::as_str) {
            Some("F32") => 4,
            Some("BF16" | "F16") => 2,
            other => return Err(refuse(&format!("is {other:?}; a bank is F32, BF16 or F16"))),
        };
        let numel = tensor
            .get("shape")
            .and_then(Value::as_array)
            .and_then(|dims| dims.iter().map(Value::as_u64).product::<Option<u64>>())
            .ok_or_else(|| refuse("has no shape"))?;
        let span = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .and_then(|at| Some((at.first()?.as_u64()?, at.get(1)?.as_u64()?)))
            .ok_or_else(|| refuse("has no data offsets"))?;
        if span.0 > span.1 || span.1 > data as u64 || span.1 - span.0 != numel * width {
            return Err(refuse(&format!(
                "spans bytes {}..{} of {data} for {numel} elements of {width} bytes",
                span.0, span.1
            )));
        }
    }
    Ok(())
}
