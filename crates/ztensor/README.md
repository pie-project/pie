# ztensor

The `.zt` container format: aligned, verifiable tensor storage.

```rust
use ztensor::{Leaf, Source, Term, Writer};

let mut w = Writer::create("model.zt")?;
w.add("weights", [2u64, 2], Leaf::F32, &[0u8; 16])?;
let affine = Term::parse("g64_u4_bf16_b_bf16")?;
w.object("q", |o| {
    o.shape([64u64, 64])
        .term(affine)
        .planes([&codes[..], &scales[..], &biases[..]])
})?;
w.finish()?;

let src = Source::open("model.zt")?;
let t = src.tensor("q")?;
let bytes = t.map()?;              // zero-copy borrow, or error
for p in t.planes()? {             // 256-byte aligned planes
    let _ = &bytes[p.range()];
}
# Ok::<(), ztensor::Error>(())
```

Each tensor in a canonical `.zt` file is 64 KiB-aligned and carries an XXH3
digest, with its planes aligned to 256-byte boundaries. Access tensor data via
`map()` (zero-copy borrow), `bytes()` (borrow or copy), or `locate()` (raw byte
range).

- Foreign formats (`.safetensors`, `.gguf`, `.npz`, `.pt`, HDF5, ONNX): [`ztensor-compat`](../ztensor-compat)
- Spec: [`ztensor-v3-spec.md`](https://github.com/pie-project/ztensor/blob/main/spec/ztensor-v3-spec.md)
- License: MIT
