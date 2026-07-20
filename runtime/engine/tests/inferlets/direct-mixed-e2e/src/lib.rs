use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

/// One-page working set + a single seeded token.
const MAX_PAGES: u32 = 1;

fn geometry() -> Result<(WorkingSet, Channel)> {
    let ws = WorkingSet::new();
    ws.reserve(MAX_PAGES)
        .map_err(|error| format!("ws.reserve: {error}"))?;
    Ok((ws, Channel::from(vec![1i32]).named("token")))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    model::configure(wit_model::output_vocab_size(), 16, 1);

    let (ws, token) = geometry()?;
    let mixed_kv_len = Channel::from(vec![1u32]).named("mixed_kv_len");
    let mixed_token = Channel::new([1], dtype::u32).named("mixed_token");
    let mixed_scalar = Channel::new([1], dtype::f32).named("mixed_scalar");
    let vector = Channel::new([4], dtype::u32).named("vector");
    let prefix_len = Channel::new([1], dtype::u32).named("prefix_len");
    let sampler_a = Channel::new([1], dtype::u32).named("sampler_a");
    let sampler_b = Channel::new([1], dtype::u32).named("sampler_b");
    let sampler_c = Channel::new([1], dtype::u32).named("sampler_c");
    let sampler_d = Channel::new([1], dtype::u32).named("sampler_d");
    let mixed_token_source = Channel::from(vec![7u32]).named("mixed_token_source");
    let mixed_scalar_source = Channel::from(vec![1.25f32]).named("mixed_scalar_source");
    let vector_source = Channel::from(vec![3u32, 5, 8, 13]).named("vector_source");
    let prefix_source = Channel::from(vec![0u32]).named("prefix_source");
    let sampler_a_source = Channel::from(vec![11u32]).named("sampler_a_source");
    let sampler_b_source = Channel::from(vec![12u32]).named("sampler_b_source");
    let sampler_c_source = Channel::from(vec![13u32]).named("sampler_c_source");
    let sampler_d_source = Channel::from(vec![14u32]).named("sampler_d_source");

    let mixed = ForwardPass::new();
    mixed.embed(&token, Tensor::constant(vec![0u32, 1]));
    mixed.port_channel(Port::KvLen, &mixed_kv_len);
    mixed.attn_working_set(&ws, .., ..)?;
    mixed.derive_dense_geometry();
    mixed.epilogue(move || {
        let mixed_token_value = mixed_token_source.take().tensor();
        let mixed_scalar_value = mixed_scalar_source.take().tensor();
        let vector_value = vector_source.take().tensor();
        let prefix_value = prefix_source.take().tensor();
        let sampler_a_value = sampler_a_source.take().tensor();
        let sampler_b_value = sampler_b_source.take().tensor();
        let sampler_c_value = sampler_c_source.take().tensor();
        let sampler_d_value = sampler_d_source.take().tensor();

        mixed_token_source.put(&mixed_token_value);
        mixed_scalar_source.put(&mixed_scalar_value);
        vector_source.put(&vector_value);
        prefix_source.put(&prefix_value);
        sampler_a_source.put(&sampler_a_value);
        sampler_b_source.put(&sampler_b_value);
        sampler_c_source.put(&sampler_c_value);
        sampler_d_source.put(&sampler_d_value);

        mixed_token.put(&mixed_token_value);
        mixed_scalar.put(&mixed_scalar_value);
        vector.put(&vector_value);
        prefix_len.put(&prefix_value);
        sampler_a.put(&sampler_a_value);
        sampler_b.put(&sampler_b_value);
        sampler_c.put(&sampler_c_value);
        sampler_d.put(&sampler_d_value);
    });

    let pipeline = Pipeline::new();
    mixed
        .submit(&pipeline)
        .map_err(|error| format!("mixed submit: {error}"))?;
    let token_value = mixed_token
        .take()
        .get::<u32>()
        .await
        .map_err(|error| format!("mixed token: {error}"))?[0];
    let scalar_value = mixed_scalar
        .take()
        .get::<f32>()
        .await
        .map_err(|error| format!("mixed scalar: {error}"))?[0];
    let vector_value = vector
        .take()
        .get::<u32>()
        .await
        .map_err(|error| format!("vector: {error}"))?;
    let empty_prefix = prefix_len
        .take()
        .get::<u32>()
        .await
        .map_err(|error| format!("prefix: {error}"))?[0] as usize;
    let samplers = [
        sampler_a
            .take()
            .get::<u32>()
            .await
            .map_err(|error| error.to_string())?[0],
        sampler_b
            .take()
            .get::<u32>()
            .await
            .map_err(|error| error.to_string())?[0],
        sampler_c
            .take()
            .get::<u32>()
            .await
            .map_err(|error| error.to_string())?[0],
        sampler_d
            .take()
            .get::<u32>()
            .await
            .map_err(|error| error.to_string())?[0],
    ];
    pipeline.close();

    let (entropy_ws, entropy_token) = geometry()?;
    let entropy_kv_len = Channel::from(vec![1u32]).named("entropy_kv_len");
    let entropy = Channel::new([1], dtype::f32).named("entropy");
    let entropy_source = Channel::from(vec![0.5f32]).named("entropy_source");
    let entropy_pass = ForwardPass::new();
    entropy_pass.embed(&entropy_token, Tensor::constant(vec![0u32, 1]));
    entropy_pass.port_channel(Port::KvLen, &entropy_kv_len);
    entropy_pass.attn_working_set(&entropy_ws, .., ..)?;
    entropy_pass.derive_dense_geometry();
    entropy_pass.epilogue(move || {
        let entropy_value = entropy_source.take().tensor();
        entropy_source.put(&entropy_value);
        entropy.put(&entropy_value);
    });
    let entropy_pipeline = Pipeline::new();
    entropy_pass
        .submit(&entropy_pipeline)
        .map_err(|error| format!("entropy submit: {error}"))?;
    let entropy_value = entropy
        .take()
        .get::<f32>()
        .await
        .map_err(|error| format!("entropy: {error}"))?[0];
    entropy_pipeline.close();

    let mixed_ok = token_value == 7 && (scalar_value - 1.25).abs() < f32::EPSILON;
    let vector_ok = vector_value == [3, 5, 8, 13];
    let empty_prefix_ok = vector_value[..empty_prefix].is_empty();
    let entropy_ok = entropy_value.is_finite() && entropy_value > 0.0;
    let multisampler_ok = samplers == [11, 12, 13, 14];
    Ok(format!(
        "MIXED_OK={mixed_ok} ENTROPY_OK={entropy_ok} VECTOR_OK={vector_ok} \
         EMPTY_PREFIX_OK={empty_prefix_ok} MULTISAMPLER_OK={multisampler_ok}"
    ))
}
