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
            k_conv = weight([32, 4]),
            window = window,
            kv = state([tokens, 2, 2, 16], deploy.kv),
            k_taps = state([4, 32], dtype.bf16),
            taps = state([4, 64], dtype.bf16),
        )
    return struct(
        embed = weight([16, 64]),
        head = weight([16, 64]),
        layers = [layer(None), layer(8), layer(None)],
        spare = state([tokens, [32, 32]], deploy.kv),
    )

def forward(m, inputs):
    y = ops.gather.embed(inputs.tokens(), m.embed, 16)

    def block(l, w, y):
        q, k, v = ops.linear.matmul(y, w.q), ops.linear.matmul(y, w.k), ops.linear.matmul(y, w.v)
        k = ops.ssm.causal_conv1d(k, w.k_conv, w.k_taps, 4)
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
            // its channels feed the heads: cut as they are
            "layers.0.k_taps slab [4, 32] cut 1",
            // over the residual stream: whole on every rank
            "layers.0.taps slab [4, 64] whole",
            "layers.1.kv kv [32, 32] space 1 window Some(8) head 16 true",
            "layers.1.k_taps slab [4, 32] cut 1",
            "layers.1.taps slab [4, 64] whole",
            "layers.2.kv kv [32, 32] space 0 window None head 16 true",
            "layers.2.k_taps slab [4, 32] cut 1",
            "layers.2.taps slab [4, 64] whole",
            // no op reads it: laid out as the rows of its planes one does
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

#[test]
fn a_row_format_states_the_dims_it_encodes() {
    let source = MODEL.replace(
        "spare = state([tokens, [32, 32]], deploy.kv)",
        "spare = state([tokens // 4, 256], kv_rows.v4)",
    );
    let why = format!("{:#}", traced(&source).expect_err("v4 rows of 256 dims"));
    assert!(
        why.contains("kv_rows.v4 rows encode 512 dims, not 256"),
        "{why}"
    );
}

#[test]
fn only_a_per_token_state_lists_its_planes() {
    let source = MODEL.replace("state([4, 64]", "state([[4, 64]]");
    let why = format!("{:#}", traced(&source).expect_err("a listed slab"));
    assert!(why.contains("a per-token state lists its planes"), "{why}");
}

const PADDED: &str = r#"
def layout(id, deploy):
    return struct(
        embed = weight([16, 64]),
        head = weight([16, 64]),
        qkv = weight([4 * 12 + 2 * (12 + 8), 64]),
        o = weight([64, 4 * 8]),
        kv = state([tokens, 2, 2, 16], deploy.kv),
    )

def forward(m, inputs):
    y = ops.gather.embed(inputs.tokens(), m.embed, 16)
    q, k, v = ops.gather.split_qkv_padded(ops.linear.matmul(y, m.qkv), 4, 2, 12, 8, 16)
    q, k = ops.elemwise.rope(q, inputs.positions(), 16, 10000.0, rotary.partial(8), k = k)
    ops.attn.kv_append(k, v * f32(0.5), m.kv)
    a = ops.attn.attend(q, inputs, m.kv, 4, 2, 16, 0.25)
    a, _ = ops.gather.split_q_gate(a, 8)
    y = ops.elemwise.residual_add(ops.linear.matmul(a, m.o), y)
    output("logits", ops.linear.lm_head(ops.gather.gather_rows(y, inputs.readout_rows()), m.head))
"#;

/// The live widths of `trace`'s kv rows.
fn live(trace: &poem_ir::Trace) -> Vec<Option<[u32; 2]>> {
    trace
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { live, .. } => Some(*live),
            CacheRow::State { .. } => None,
        })
        .collect()
}

#[test]
fn a_padded_split_says_the_widths_its_heads_carry() {
    // keys 12 and values 8 wide, each head zero-padded to 16: the split
    // cutting out what the row is appended says so, through the rotation
    // and the scale between
    let trace = traced(PADDED).unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(live(&trace), [Some([12, 8])]);
    // keys and values a projection cuts out whole carry their every dim
    let trace = traced(MODEL).unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(live(&trace), [None; 4]);
}
