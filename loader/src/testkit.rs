//! The part of this crate that exists to *check* the loader rather than to be
//! the loader.
//!
//! One module, one feature gate, one reason: [`mod@reference`] — the
//! differential oracle plans are replayed against — is not on the load path,
//! so a driver that links this crate has no use for it. `worker/Cargo.toml`
//! turns `testkit` off, which is also the check — if the load path ever grows
//! a route into the oracle, the worker stops compiling.
//!
//! Two things used to sit beside it and left for opposite reasons. The host
//! executor was *promoted*: `pie model convert` made it a production caller,
//! so it lives ungated at [`crate::executor::host`]. The POD contract fixture
//! builder followed the ABI into `pie-loader-capi` and then out of existence:
//! contracts stopped crossing the ABI when the last C++ author was harvested
//! (`plan/model-in-rust.md` §8-5), so there is no POD graph left to write.

pub mod reference;

/// Write a zero-filled BF16 safetensors file the checkpoint reader parses.
///
/// One header assembler for every test that needs a real file on disk rather
/// than in-memory metadata: `pie_loader_open_checkpoint` parses the header,
/// so a fixture whose header did not exist would test nothing. Zero-filled
/// because these fixtures exercise plans, not arithmetic — a test that cares
/// about values writes its own bytes.
///
/// Published by rename so a parallel reader never sees a partial header.
pub fn write_safetensors_fixture(dir: &std::path::Path, tensors: &[(String, Vec<i64>)]) {
    std::fs::create_dir_all(dir).expect("create fixture directory");
    let mut header = String::from("{");
    let mut offset = 0u64;
    for (index, (name, shape)) in tensors.iter().enumerate() {
        let elements: i64 = shape.iter().product();
        let nbytes = u64::try_from(elements).expect("fixture shape") * 2;
        if index > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
            dims.join(","),
            offset + nbytes
        ));
        offset += nbytes;
    }
    header.push('}');

    let mut file = Vec::with_capacity(8 + header.len() + offset as usize);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend(std::iter::repeat_n(0u8, offset as usize));
    let staging = dir.join(format!("model.{:?}.partial", std::thread::current().id()));
    std::fs::write(&staging, &file).expect("write fixture checkpoint");
    std::fs::rename(&staging, dir.join("model.safetensors")).expect("publish fixture checkpoint");
}
