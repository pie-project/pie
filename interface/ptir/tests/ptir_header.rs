//! Keeps the checked-in `include/ptir_abi.h` byte-identical to the generator
//! (the op table is the single source of truth; the header is a projection).
//! Regenerate with `PTIR_REGEN=1 cargo test -p pie-sampling-ir --test ptir_header`.

use pie_ptir::header::generate_c_header;

#[test]
fn ptir_header_uptodate() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/include/ptir_abi.h");
    let expected = generate_c_header();
    if std::env::var("PTIR_REGEN").is_ok() {
        std::fs::create_dir_all(concat!(env!("CARGO_MANIFEST_DIR"), "/include")).unwrap();
        std::fs::write(path, &expected).unwrap();
        return;
    }
    let on_disk = std::fs::read_to_string(path)
        .expect("include/ptir_abi.h missing — run with PTIR_REGEN=1 to generate");
    assert_eq!(
        on_disk, expected,
        "include/ptir_abi.h is stale — regenerate with PTIR_REGEN=1 cargo test -p pie-sampling-ir --test ptir_header"
    );
}
