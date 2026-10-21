//! How many data words one PE takes: compiles phases holding big arrays
//! and reports where `cslc` fails. Run with `--nocapture` to read the
//! boundary; the assertion pins the budget the kernels plan with
//! (`kernels_cerebras::linear::gemm::pe_words()`).

use dtype::Dtype;
use engine_cerebras::sdk::{Arch, Compile, Cslc};
use kernels_cerebras::Tensor;
use kernels_cerebras::cx::Tracer;
use kernels_cerebras::linear::gemm::pe_words;

/// Compiles a phase holding `arrays` arrays of `words` f32 each.
fn compiles(cslc: &Cslc, words: u32, arrays: u32) -> Result<(), String> {
    let tracer = Tracer::default();
    let xs: Vec<Tensor> = (0..arrays)
        .map(|i| Tensor::new(10 + i, 1, words, Dtype::F32))
        .collect();
    let y = Tensor::new(2, 1, 1, Dtype::F32);
    kernels_cerebras::cx::Emit::emit(&tracer, &mut |cx| {
        let yb = cx.write(y)?;
        for x in &xs {
            let xb = cx.read(*x)?;
            cx.line(format!(
                "{}[0] = {}[0] + {}[{}];",
                yb.name,
                yb.name,
                xb.name,
                words - 1
            ));
        }
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    let r = tracer.into_program().render();
    let dir = tempfile::Builder::new()
        .prefix("pie-cerebras-probe-")
        .tempdir()
        .map_err(|e| e.to_string())?;
    std::fs::write(dir.path().join("layout.csl"), &r.layout).map_err(|e| e.to_string())?;
    std::fs::write(dir.path().join("pe.csl"), &r.pe).map_err(|e| e.to_string())?;
    let out = dir.path().join("out");
    cslc.compile(&Compile::memcpy(
        Arch::Wse3,
        dir.path().join("layout.csl"),
        1,
        1,
        &out,
    ))
    .map_err(|e| e.to_string())
}

#[test]
fn a_pe_holds_the_budget_the_kernels_plan_with() {
    if !Cslc::available() {
        return;
    }
    let cslc = Cslc::find().unwrap();
    let mut last_ok = 0;
    for (words, arrays) in [
        (8192u32, 1u32),
        (8448, 1),
        (4096, 2),
        (4608, 2),
        (5120, 2),
        (5632, 2),
        (6144, 2),
        (3072, 4),
        (3584, 4),
    ] {
        let r = compiles(&cslc, words, arrays);
        eprintln!(
            "{arrays} x {words} words: {}",
            match &r {
                Ok(()) => "compiles".to_string(),
                Err(e) => e
                    .lines()
                    .find(|l| l.contains("error"))
                    .unwrap_or("fails")
                    .to_string(),
            }
        );
        if r.is_ok() {
            last_ok = last_ok.max(words * arrays);
        }
    }
    assert!(
        u64::from(last_ok) >= pe_words(),
        "a PE holds {last_ok} words, under the planned {}",
        pe_words()
    );
}
