pub(crate) fn bytes(len: u64) -> wasmtime::Result<Vec<u8>> {
    let len =
        usize::try_from(len).map_err(|_| wasmtime::format_err!("random request too large"))?;
    let mut buf = vec![0u8; len];
    getrandom::fill(&mut buf).map_err(|e| wasmtime::format_err!("getrandom failed: {e}"))?;
    Ok(buf)
}

pub(crate) fn u64() -> wasmtime::Result<u64> {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).map_err(|e| wasmtime::format_err!("getrandom failed: {e}"))?;
    Ok(u64::from_le_bytes(buf))
}

pub(crate) fn seed_pair() -> Option<(u64, u64)> {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).ok()?;
    let (a, b) = buf.split_at(8);
    Some((
        u64::from_le_bytes(a.try_into().ok()?),
        u64::from_le_bytes(b.try_into().ok()?),
    ))
}
