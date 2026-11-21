//! The shell's rs layout is the chunk plan's: the convs it buffers are
//! `poem_compiler::rs::buffered_convs`, each plane as wide as the rule's
//! input, and the recipe's rs runs land on the shell's planes, a run's
//! planes one after another at the offsets the recipe states. Over the
//! qwen_4 package's recipes (Qwen3.8-Flash-Next, its miniature and its MTP
//! deployment; Qwen4's PLE conv is the package's), whose PLE conv no
//! recurrence reads, and Qwen3.5's (gated delta alone). No device: the
//! layout and the recipe are read off the trace.

use engine_gpu::rs::{Layout, Role};
use engine_gpu::store::Rows;
use poem_compiler::chunk::ChunkRecipe;
use poem_compiler::rs::{Feeds, buffered_convs};
use poem_ir::{Dim, Trace, Ty, ValueId};

use crate::backend;

fn width(trace: &Trace, value: ValueId) -> u64 {
    let Ty::Tensor { shape, .. } = &trace.values[value.0 as usize].ty else {
        panic!("value {} is no tensor", value.0)
    };
    shape
        .iter()
        .skip(1)
        .map(|dim| match dim {
            Dim::Const(n) => *n,
            _ => 1,
        })
        .product()
}

fn recipe(trace: &Trace) -> ChunkRecipe {
    let budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(4, 256));
    engine_gpu::chunks::recipe(trace, &budgets, 16, 4096, 1 << 30, None)
        .expect("the GPU chunk recipe plans")
}

/// The shell's layout of `trace` against the rule and the recipe; the lone
/// convs it found (by their state: a conv's decode and prompt forms are one).
fn agrees(name: &str, trace: &Trace) -> usize {
    let layout = Layout::read(trace)
        .expect("the rs layout reads")
        .expect("a recurrent trace buffers rs rows");
    let convs: Vec<_> = buffered_convs(trace, false).collect();
    assert!(!convs.is_empty(), "{name}: the rule buffers its convs");
    // Every conv the rule buffers has its plane, as wide as its input; every
    // conv plane is the rule's.
    for conv in &convs {
        let plane = layout.in_of.get(&conv.x.0).map(|&at| layout.planes[at]);
        let plane = plane.unwrap_or_else(|| {
            panic!("{name}: no plane buffers conv {} ({:?})", conv.node, conv.feeds)
        });
        assert_eq!(plane.role, Role::ConvX, "{name}: conv {}", conv.node);
        assert_eq!(u64::from(plane.width), width(trace, conv.x), "{name}: conv {}", conv.node);
        assert!(layout.out_of.contains_key(&conv.y.0), "{name}: conv {} lands", conv.node);
    }
    let conv_planes = layout.planes.iter().filter(|p| p.role == Role::ConvX).count();
    let mut states: Vec<_> = convs.iter().map(|conv| conv.state.0).collect();
    states.sort_unstable();
    states.dedup();
    assert_eq!(conv_planes, states.len(), "{name}: one conv plane a conv state");

    // The recipe's runs land on the shell's planes: each run's planes one
    // after another from its offset, at the bytes the shell stores.
    let recipe = recipe(trace);
    let facts = engine_gpu::store::kv::probe(trace).expect("the kv facts");
    let rows = Rows::plan(trace, &recipe, &facts, Some(&layout))
        .unwrap_or_else(|why| panic!("{name}: the recipe's rs rows are refused: {why}"));
    let rs = rows.rs.expect("the pool lays an rs list");
    let tokens = u64::from(recipe.rs.tokens);
    let mut covered = 0usize;
    for (id, row) in &recipe.rs.rows {
        let run = recipe.rs.run(*id).expect("each rs row has its run");
        assert_eq!(run.per_token, row.per_token(), "{name}: {id:?}");
        for (k, plane) in row.planes.iter().enumerate() {
            let mine = layout.in_of[&plane.value.0];
            assert_eq!(
                rs.plane_at[mine],
                run.offset + row.plane_at(k, tokens),
                "{name}: {id:?} plane {k}"
            );
            assert_eq!(layout.planes[mine].row_bytes, plane.bytes, "{name}: {id:?} plane {k}");
            covered += 1;
        }
    }
    assert_eq!(covered, layout.planes.len(), "{name}: every shell plane is a recipe plane");

    // A lone conv is an rs row of its input alone: its run is sized by its
    // own rows, its plane the shell's extended rows of it.
    let lone: Vec<_> = convs.iter().filter(|conv| conv.feeds == Feeds::Alone).collect();
    for conv in &lone {
        let (id, row) = recipe
            .rs
            .rows
            .iter()
            .find(|(_, row)| row.planes.iter().any(|p| p.value == conv.x))
            .unwrap_or_else(|| panic!("{name}: the recipe buffers lone conv {}", conv.node));
        assert_eq!(row.planes.len(), 1, "{name}: lone conv {} buffers its input alone", conv.node);
        let plane = layout.planes[layout.in_of[&conv.x.0]];
        assert_eq!(recipe.rs.run(*id).unwrap().per_token, plane.row_bytes);
        let rows_ext = 32;
        let (_, _, planes) = layout.pack(rows_ext);
        assert!(
            planes >= u64::from(rows_ext) * plane.row_bytes,
            "{name}: the seat's planes extend the lone conv's rows"
        );
    }
    let mut lone_states: Vec<u32> = lone.iter().map(|conv| conv.state.0).collect();
    lone_states.sort_unstable();
    lone_states.dedup();
    eprintln!(
        "{name}: {} conv form(s) buffered, {} lone conv(s); {} rs row(s) on {} plane(s), {} bytes a slot of {} tokens",
        convs.len(),
        lone_states.len(),
        recipe.rs.rows.len(),
        layout.planes.len(),
        rs.bytes,
        rs.tokens
    );

    // Negative control: the recipe as the rule stood before lone convs were
    // buffered (no rs row for a conv no recurrence reads) is refused by
    // name, where the shell's PLE conv would otherwise have no rs place.
    if !lone.is_empty() {
        let mut old = recipe.clone();
        old.rs.rows.retain(|_, row| {
            !row.planes.iter().any(|p| lone.iter().any(|conv| conv.x == p.value))
        });
        let why = Rows::plan(trace, &old, &facts, Some(&layout))
            .err()
            .unwrap_or_else(|| panic!("{name}: a recipe without the PLE conv's row is served"));
        assert!(why.to_string().contains("keeps no rs row"), "{name}: {why}");
    }
    lone_states.len()
}

fn traced(model: &str, deploy: &poem::star::Deploy) -> Trace {
    poem_compiler::catalog::package_of(model)
        .expect("the model is listed")
        .trace(model, deploy, model, backend::PLATFORM)
        .unwrap_or_else(|why| panic!("`{model}` traces: {why:#}"))
}

#[test]
fn the_rs_layout_is_the_rule_and_the_recipe_for_the_qwen_4_package() {
    let bf16 = poem_compiler::catalog::one_rank(poem::Dtype::Bf16, poem::Dtype::Bf16);
    for model in ["qwen38-flash-next-micro", "qwen38-flash-next-mini", "qwen38-flash-next"] {
        assert_eq!(agrees(model, &traced(model, &bf16)), 1, "{model}: the PLE conv is lone");
    }
    // The MTP deployment: the drafter's layers beside the trunk's.
    let mtp = poem_compiler::catalog::deployments()
        .find(|d| d.name.starts_with("qwen38-flash-next-") && d.name.contains("mtp"))
        .expect("the package deploys its MTP drafter");
    let name = mtp.name.clone();
    assert!(agrees(&name, &mtp.trace(backend::PLATFORM)) >= 1, "{name}");
}

#[test]
fn the_rs_layout_is_the_rule_and_the_recipe_for_qwen35() {
    let bf16 = poem_compiler::catalog::one_rank(poem::Dtype::Bf16, poem::Dtype::Bf16);
    assert_eq!(agrees("qwen35-tiny", &traced("qwen35-tiny", &bf16)), 0);
}
