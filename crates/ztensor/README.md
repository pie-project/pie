# ztensor

The `.zt` container format: reading, writing, and the whole of its validation.

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
let bytes = t.map()?;              // borrowed, or an error; never a copy
for p in t.planes()? {             // code, gain, offset: each 256-byte aligned
    let _ = &bytes[p.range()];
}
# Ok::<(), ztensor::Error>(())
```

An object is a shape, a type and one blob. The type is a term: a leaf
(`bf16`, `u8`, `e4m3`) or a group form naming the code leaf, the group size
and the scale and offset terms. The blob holds the term's planes at 256-byte
boundaries, in an order derived from the type, unless a named `layout`
(`gguf.q4_k/2`, `zt.sparse_csr/2`) says otherwise.

Every tensor in a canonical file begins on a 64 KiB boundary and carries an
XXH3 digest, so one weight can be mapped, verified, and evicted without
touching its neighbours. Two canonical writes of the same tensors produce
byte-identical files.

Three ways to get at a tensor's bytes, one per intent:

| | |
| --- | --- |
| `bytes()` | the best the file can do, saying whether it borrowed or copied |
| `map()` | a borrow, or an error |
| `locate()` | the exact byte range, for a caller doing its own I/O |

`caps()` reports which will work, per tensor, and each of its fields is
computed by the very precondition the matching method checks.

- **Foreign formats** (safetensors, GGUF, `.npz`, `.pt`, HDF5, ONNX) are read
  through [`ztensor-compat`](https://crates.io/crates/ztensor-compat), which
  projects each into this same object model.
- **Command line**: [`ztensor-cli`](https://crates.io/crates/ztensor-cli)
  installs `zt`, for inspecting, verifying, converting and diffing.
- **Specification**: the normative rules live in
  [`spec/ztensor-v3-spec.md`](https://github.com/pie-project/ztensor/blob/main/spec/ztensor-v3-spec.md).

Features: `zstd` enables the `zt.zstd-seekable/1` encoding profile.

MIT licensed.
