use poem::Platform;

/// Qwen3.5's MTP head is fed, at a prompt's rows, the lane's next tokens
/// (`inputs.next_tokens()`, vLLM's shifted ids): every row its MTP drafter
/// ships, on every GPU platform the text builds for, reads them; a row
/// without the drafter reads none.
#[test]
fn an_mtp_head_reads_the_next_tokens() {
    let rows: Vec<_> = poem_compiler::catalog::deployments()
        .filter(|row| row.model.id == "qwen35-d0.8b")
        .collect();
    let mtp: Vec<_> = rows
        .iter()
        .filter(|row| row.deploy.drafter.as_deref() == Some("mtp"))
        .collect();
    assert!(!mtp.is_empty(), "the catalog ships Qwen3.5 0.8B's MTP rows");
    for row in &mtp {
        for platform in [
            Platform::Cuda,
            Platform::Vulkan,
            Platform::Wgpu,
            Platform::Metal,
            Platform::Xla,
        ] {
            assert!(
                row.trace(platform).reads_next_tokens(),
                "{} on {platform:?} feeds its head no next tokens",
                row.name
            );
        }
    }
    let plain = rows
        .iter()
        .find(|row| row.deploy.drafter.is_none() && row.deploy.parts.is_empty())
        .expect("a plain Qwen3.5 0.8B row");
    assert!(
        !plain.trace(Platform::Cuda).reads_next_tokens(),
        "{} drafts nothing and reads no next tokens",
        plain.name
    );
}
