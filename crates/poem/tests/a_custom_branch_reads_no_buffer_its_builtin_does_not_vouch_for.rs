//! A buffer a lane may not carry is read only on rows a builtin fact says
//! carry it: an inferlet sets a custom fact, so a branch on one alone says
//! nothing about which lanes hold adapter routes or a mask.

use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, Predicate, Value, Weight, fact, ops, switch,
    trace_hybrid,
};

const WIDTH: u64 = 64;

struct Probe {
    rows: fn() -> Predicate,
}

impl ForwardHybrid for Probe {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        let x = inputs.latents(0, WIDTH as u32, Dtype::Bf16);
        let y = ops::linear::matmul(&x, &Weight::sym("w", [WIDTH, WIDTH], Dtype::Bf16));
        let (a, b) = banks("w", 2, 8, WIDTH, Dtype::Bf16);
        let routes = inputs.adapter_routes();
        switch(&y)
            .case((self.rows)(), |y| {
                ops::linear::lora_correct(&x.on((self.rows)()), &a, &b, &routes, y)
            })
            .otherwise(Value::clone)
    }
}

fn traces(rows: fn() -> Predicate) -> Result<(), String> {
    std::panic::catch_unwind(|| {
        trace_hybrid("probe", &Probe { rows }, Platform::Cuda);
    })
    .map_err(|why| {
        why.downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| "a panic with no message".to_string())
    })
}

#[test]
fn a_custom_branch_reads_no_buffer_its_builtin_does_not_vouch_for() {
    let why = traces(fact::scores).expect_err("a flag alone does not vouch for adapter routes");
    assert!(why.contains("reads Adapted input"), "{why}");
    assert!(
        why.contains("fact.has(fact.Adapter)"),
        "and names the fact that would: {why}"
    );

    traces(|| fact::scores() & fact::has(fact::Adapter))
        .expect("the flag narrowed to the lanes that carry routes reads them");
    traces(|| fact::has(fact::Adapter)).expect("so does the builtin alone");
}

/// A layer's two LoRA banks under `prefix`: `slots` adapters of `rank` over
/// a `hidden`-wide residual, registered by the host.
fn banks(
    prefix: &str,
    slots: u64,
    rank: u64,
    hidden: u64,
    dense: Dtype,
) -> (poem::Weight, poem::Weight) {
    (
        poem::Weight::sym(format!("{prefix}.lora_a"), [slots, rank, hidden], dense).registered(),
        poem::Weight::sym(format!("{prefix}.lora_b"), [slots, hidden, rank], dense).registered(),
    )
}
