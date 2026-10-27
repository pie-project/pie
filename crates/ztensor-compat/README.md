# ztensor-compat

Projects foreign tensor formats into a [`ztensor::Source`](../ztensor).

```rust
let src = ztensor_compat::open("model.safetensors")?;
let bytes = src.tensor("layer.weight")?.map()?;
# Ok::<(), ztensor::Error>(())
```

Detects and reads `.safetensors`, `.gguf`, `.npz`, PyTorch `.pt`, HDF5, and
ONNX without silent reinterpretation, dequantization, or hidden copies. Use
`Writer::ingest` to convert any source into a canonical, page-aligned `.zt`
file with XXH3 digests.

Features: `safetensors` and `gguf` enabled by default; `npz`, `pickle`, `hdf5`,
and `onnx` are opt-in. MIT licensed.
