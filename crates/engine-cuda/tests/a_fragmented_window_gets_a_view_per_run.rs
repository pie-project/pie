use engine_cuda::window::Windows;
use model_compiler::{Budget, CompiledModel, DeviceProfile, compile};
use model_dsl::Platform;
use model_exec::fire::{WindowTable, fallback};
use model_ir::Trace;

fn test_slots() -> engine_cuda::window::Slots {
    engine_cuda::window::Slots::new(8, 512, 8, 1, 4096, 2, 512)
}

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

fn budget() -> Budget {
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 8,
    }
}

fn sku() -> (Trace, CompiledModel) {
    let trace = models::sku(SKU)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"))
        .trace;
    let trace = trace(Platform::Cuda);
    let compiled = compile(&trace, &budget(), &DeviceProfile::default())
        .unwrap_or_else(|refusal| panic!("`{SKU}` bakes: {refusal:?}"));
    assert!(
        !compiled.fallback.rows.is_empty(),
        "`{SKU}` owes fallback rows — that is the premise of this file",
    );
    (trace, compiled)
}

fn indptr(rows: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in rows {
        out.push(out[out.len() - 1] + n as i32);
    }
    out
}

#[test]
fn a_window_p4_promised_whole_is_still_a_bake_integrity_refusal() {
    let (plan, compiled) = sku();

    let count = compiled.classes.classes.len();
    let ascending = WindowTable::new(
        (0..count)
            .map(|at| model_exec::fire::ClassWindow {
                row_offset: at as u32,
                rows: 1,
                lane_offset: at as u32,
                lanes: 1,
            })
            .collect(),
    );

    let seated = compiled
        .template()
        .iter()
        .find(|region| {
            fallback::promised(&compiled, model_ir::RowAxis::Tokens, region)
                && ascending.span(&region.mask).is_err()
        })
        .expect("some seated window is not an interval of the ascending order");
    assert_eq!(
        fallback::bound(&compiled, model_ir::RowAxis::Tokens, &seated.mask),
        1
    );

    let refusal = Windows::of(
        &plan,
        &compiled,
        model_ir::PerAxis::new([&ascending, &WindowTable::default(), &WindowTable::default()]),
        &indptr(&vec![1; count]),
        engine_cuda::window::Copies::off(),
        test_slots(),
    )
    .expect_err("a promise P4 made and this table broke");
    let said = refusal.to_string();
    assert!(said.contains("no fallback row"), "{said}");
}
