use poem::Platform;

/// An engine refuses to bake a plan whose request classes leave a merge
/// unwritten (`this plan does not bake: Classes([Uncovered ..])`), so a
/// deployment that fails here ships but never boots.
#[test]
fn every_deployment_resolves_its_request_classes() {
    let mut faults = Vec::new();
    for deployment in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits())
    {
        for platform in [Platform::Cuda, Platform::Metal] {
            let trace = deployment.trace(platform);
            if let Err(found) = poem_ir::check::classes::resolve_classes(&trace) {
                faults.push(format!(
                    "`{}` on {platform:?}: {} fault(s), first: {}",
                    deployment.name,
                    found.len(),
                    found[0].say(&trace)
                ));
            }
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
