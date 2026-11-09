//! A layout declares what its model remembers as `state(shape, dtype)`,
//! as it does a weight; the forward hands the state to the ops that read
//! and write it. The state is named by its path in the layout, and what
//! its rows are (a kv row's space, window, head width and cut, a slab's
//! cut) is what the ops reading it say of it.

use poem::star::{Deploy, Package};
use poem::{Dtype, Platform};
use poem_ir::{CacheRow, Shard};

fn deploy() -> Deploy {
    Deploy {
        weights: vec![Dtype::Bf16.into()],
        kv: Dtype::Bf16,
        tp: 1,
        ep: 1,
        parts: vec![],
        drafter: None,
        yarn: None,
        tap: None,
        layers: None,
        experts: None,
    }
}

const MODEL: &str = r#"
def layout(id, deploy):
    def layer(window):
        return struct(
            q = weight([64, 64]),
            k = weight([32, 64]),
            v = weight([32, 64]),
            o = weight([64, 64]),
            conv = weight([64, 4]),
            window = window,
            kv = state([tokens, 2, 2, 16], deploy.kv),
            taps = state([4, 64], dtype.bf16),
        )
    return struct(
        embed = weight([16, 64]),
        head = weight([16, 64]),
        layers = [layer(None), layer(8), layer(None)],
        spare = state([tokens, 2, 2, 16], deploy.kv),
    )

def forward(m, inputs):
    y = ops.gather.embed(inputs.tokens(), m.embed, 16)

    def block(l, w, y):
        q, k, v = ops.linear.matmul(y, w.q), ops.linear.matmul(y, w.k), ops.linear.matmul(y, w.v)
        ops.attn.kv_append(k, v, w.kv)
        o = ops.attn.attend(q, inputs, w.kv, 4, 2, 16, 0.25, window = w.window)
        y = ops.elemwise.residual_add(ops.linear.matmul(o, w.o), y)
        return ops.elemwise.residual_add(ops.ssm.causal_conv1d(y, w.conv, w.taps, 4), y)

    y = inputs.fold_layers(m.layers, y, block)
    output("logits", ops.linear.lm_head(ops.gather.gather_rows(y, inputs.readout_rows()), m.head))
"#;

fn traced(source: &str) -> anyhow::Result<poem_ir::Trace> {
    Package::new("toy", &[("toy.poem", source)])?.trace(
        "toy",
        &deploy(),
        "toy-bf16",
        Platform::Cuda,
    )
}

#[test]
fn a_state_is_named_by_its_path_and_laid_out_by_its_readers() {
    let trace = traced(MODEL).unwrap_or_else(|e| panic!("{e:#}"));
    let rows: Vec<String> = trace
        .caches
        .iter()
        .map(|row| match row {
            CacheRow::Kv {
                name,
                planes,
                space,
                window,
                head_dim,
                shard,
                ..
            } => format!(
                "{name} kv {planes:?} space {space} window {window:?} head {head_dim} {}",
                matches!(shard, Shard::Cut { axis: 0, .. })
            ),
            CacheRow::State {
                name, slab, shard, ..
            } => format!(
                "{name} slab {slab:?} {}",
                match shard {
                    Shard::Cut { axis, .. } => format!("cut {axis}"),
                    _ => "whole".to_string(),
                }
            ),
        })
        .collect();
    assert_eq!(
        rows,
        [
            "layers.0.kv kv [32, 32] space 0 window None head 16 true",
            "layers.0.taps slab [4, 64] cut 1",
            "layers.1.kv kv [32, 32] space 1 window Some(8) head 16 true",
            "layers.1.taps slab [4, 64] cut 1",
            "layers.2.kv kv [32, 32] space 0 window None head 16 true",
            "layers.2.taps slab [4, 64] cut 1",
            // no op reads it: laid out as the rows of its shape one does
            "spare kv [32, 32] space 0 window None head 16 true",
        ],
        "{rows:#?}"
    );
}

#[test]
fn a_layout_holding_state_declares_no_caches_too() {
    let source = format!("{MODEL}\ndef caches(m, c):\n    pass\n");
    let why = format!("{:#}", traced(&source).expect_err("two declarations"));
    assert!(why.contains("declares `caches` too"), "{why}");
}

#[test]
fn tokens_is_a_states_first_axis() {
    let source = MODEL.replace("state([4, 64]", "state([4, tokens]");
    let why = format!("{:#}", traced(&source).expect_err("tokens second"));
    assert!(why.contains("`tokens` is a state's first axis"), "{why}");
}

#[test]
fn a_pool_keeps_its_share_of_a_tokens_row() {
    let source = MODEL.replace("spare = state([tokens,", "spare = state([tokens // 4,");
    let why = format!("{:#}", traced(&source).expect_err("no reader of a pool"));
    // a pool is no row of the layers' shape: nothing says its head width
    assert!(
        why.contains("no attention reads the per-token state `spare`"),
        "{why}"
    );
}
