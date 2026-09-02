# ztensor-compat

Foreign tensor formats, projected into the zTensor object model.

```rust
let src = ztensor_compat::open("model.safetensors")?;
let bytes = src.tensor("layer.weight")?.map()?;
# Ok::<(), ztensor::Error>(())
```

Reads `.safetensors`, `.gguf`, `.npz`, PyTorch `.pt`, HDF5 and ONNX, detected
by magic, returned as an ordinary [`ztensor::Source`]. There is no per-format
type to learn, because after projection there is nothing per-format left to
say.

Projections are read-only and honest. Each tensor's payload is one of three
shapes: a raw addressable range, something stored under an encoding, or bytes
only the format's own reader can produce. The capability report is a
direct reading of which. Nothing is silently reinterpreted, dequantized, or
degraded to a copy.

Some things the projections refuse rather than guess at: a safetensors header
whose tensor ranges do not tile the data section exactly (which is how the
duplicate-key aliasing attack works), a big-endian numpy descr, a
`fortran_order` array, an unknown GGUF type id, an external-data ONNX
initializer, an AMD `fnuz` fp8 tensor that no leaf denotes. A quantized GGUF
tensor keeps its blocks under `gguf.<type>/2` with the term the profile gives
it; a complex tensor becomes a `[…, 2]` tensor of `f32` or `f64`. `.pt` pickle is evaluated by a restricted VM that executes no
code, and is behind the non-default `pickle` feature.

To turn any of them into a file with digests and page-aligned tensors, convert
it: `Writer::ingest` copies any source into a canonical `.zt`.

Features: `safetensors` and `gguf` by default; `npz`, `pickle`, `hdf5`, `onnx`
opt-in.

[`ztensor::Source`]: https://docs.rs/ztensor/latest/ztensor/struct.Source.html

MIT licensed.
