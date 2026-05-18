#[test]
fn generated_header_uses_stable_cxx_names() {
    let header_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("include/weight_loader.h");
    let header = std::fs::read_to_string(header_path).expect("generated header exists");
    assert!(header.contains("struct PieLoaderStorageProgramView"));
    assert!(header.contains("struct PieLoaderStorageInstrView"));
    assert!(header.contains("struct PieLoaderOptimizerReportView"));
    assert!(header.contains("enum class PieLoaderSemanticRole"));
    assert!(header.contains("PieLoaderSourceExtentView source"));
    assert!(header.contains("int32_t shard_axis"));
    assert!(!header.contains("PiePieLoader"));
}

#[test]
fn cuda_raii_wrapper_header_compiles() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("repo root");
    let source = root.join("target/weight_loader_header_probe.cpp");
    let object = root.join("target/weight_loader_header_probe.o");
    std::fs::write(
        &source,
        "#include \"driver/cuda/src/loader/rust_storage_program.hpp\"\nint main() { return 0; }\n",
    )
    .unwrap();
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let status = std::process::Command::new(&compiler)
        .arg("-std=c++20")
        .arg("-I")
        .arg(root)
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status();
    match status {
        Ok(status) => assert!(
            status.success(),
            "{compiler} failed to compile wrapper header"
        ),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("failed to run {compiler}: {err}"),
    }
}

#[test]
fn cuda_input_builder_header_compiles() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("repo root");
    let source = root.join("target/weight_loader_input_probe.cpp");
    let object = root.join("target/weight_loader_input_probe.o");
    std::fs::write(
        &source,
        "#include \"driver/cuda/src/loader/rust_loader_input.hpp\"\nint main() { return 0; }\n",
    )
    .unwrap();
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let status = std::process::Command::new(&compiler)
        .arg("-std=c++20")
        .arg("-I")
        .arg(root)
        .arg("-I")
        .arg(root.join("driver/cuda/src"))
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status();
    match status {
        Ok(status) => assert!(
            status.success(),
            "{compiler} failed to compile input header"
        ),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("failed to run {compiler}: {err}"),
    }
}

#[test]
fn cuda_bridge_header_compiles() {
    compile_header_probe(
        "weight_loader_bridge_probe.cpp",
        "#include \"driver/cuda/src/loader/rust_loader_bridge.hpp\"\nint main() { return 0; }\n",
    );
}

#[test]
fn cuda_rust_executor_header_compiles() {
    compile_header_probe(
        "weight_loader_executor_probe.cpp",
        "#include \"driver/cuda/src/loader/rust_storage_executor.hpp\"\nint main() { return 0; }\n",
    );
}

fn compile_header_probe(file_name: &str, contents: &str) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("repo root");
    let source = root.join("target").join(file_name);
    let object = source.with_extension("o");
    std::fs::write(&source, contents).unwrap();
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let status = std::process::Command::new(&compiler)
        .arg("-std=c++20")
        .arg("-I")
        .arg(root)
        .arg("-I")
        .arg(root.join("driver/cuda/src"))
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status();
    match status {
        Ok(status) => assert!(status.success(), "{compiler} failed to compile {file_name}"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("failed to run {compiler}: {err}"),
    }
}
