# Semantic Groups

Built-in semantic groups are closed Rust enum variants in `SemanticGroupKind`.
Experimental groups may use `Extension(String)`, but paper/evidence claims
should rely on built-in variants.

Initial built-ins:

- `AttentionQkv`
- `MlpGateUp`
- `ExpertBank`
- `ExpertGateUpInterleaved`
- `GptOssMxfp4`
- `QuantizedTensor`
