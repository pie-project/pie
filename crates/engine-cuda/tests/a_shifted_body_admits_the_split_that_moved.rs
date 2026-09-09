use engine_cuda::record::BodyKey;
use engine_cuda::window::{Copies, Windows};
use model_compiler::{Budget, CompiledModel, DeviceProfile, compile};
use model_exec::fire::{Composition, Lane, compose};
use model_ir::ops::Elementwise;

fn test_slots() -> engine_cuda::window::Slots {
    engine_cuda::window::Slots::new(8, 512, 8, 1, 4096, 4, 4096)
}
use model_ir::{
    CacheRow, Def, Dim, Dtype, Guard, Node, Operands, Platform, RuntimeInput, Seam, Trace, Ty,
    ValueDecl, ValueId,
};

const WIDTH: u64 = 8;

fn budget() -> Budget {
    Budget::new(4, 64)
}

fn no_decode_class() -> model_ir::ClassSet {
    model_ir::ClassSet::default()
}

const LANES: u32 = 4;

fn act() -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(WIDTH)],
        dtype: Dtype::Bf16,
    }
}

struct Build {
    trace: Trace,
}

impl Build {
    fn new() -> Build {
        Build {
            trace: Trace {
                name: "hand-built shifted split".to_string(),
                platform: Platform::Cuda,
                params: Vec::new(),
                caches: vec![CacheRow::State {
                    name: "state".to_string(),
                    slab: vec![1],
                    dtype: Dtype::Bf16,
                }],
                values: Vec::new(),
                nodes: Vec::new(),
                seams: Vec::new(),
                drafter: None,
            },
        }
    }

    fn value(&mut self, def: Def, ty: Ty) -> ValueId {
        self.trace.values.push(ValueDecl { def, ty });
        ValueId((self.trace.values.len() - 1) as u32)
    }

    fn op(&mut self, x: ValueId, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y = self.value(Def::Op(node), act());
        self.trace.nodes.push(Node {
            op: Elementwise::LayernormNoScale { x, eps: 1e-6, y }.into(),
            guard,
            layer: None,
        });
        y
    }
}

fn subject() -> Trace {
    let mut b = Build::new();
    let tokens = b.value(Def::Input(RuntimeInput::Tokens), act());
    let head = b.op(tokens, Guard::Always);
    let hot = b.op(head, Guard::Fact(0));
    let cold = b.op(head, Guard::not(Guard::Fact(0)));
    let merged = b.value(
        Def::Merge(vec![
            (hot, Guard::Fact(0)),
            (cold, Guard::not(Guard::Fact(0))),
        ]),
        act(),
    );
    let y = b.op(merged, Guard::Always);
    b.trace.seams.push(Seam {
        seam: "out".to_string(),
        values: vec![y],
        layer: None,
    });
    b.trace
}

fn baked() -> (Trace, CompiledModel) {
    let trace = subject();
    let compiled =
        compile(&trace, &budget(), &DeviceProfile::default()).expect("the subject bakes");
    (trace, compiled)
}

fn shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region.nodes.clone().all(|node| {
                trace
                    .nodes
                    .get(node as usize)
                    .is_some_and(|node| engine_cuda::shifted(Operands::name(&node.op)))
            })
        })
        .collect()
}

fn lane_shifting(compiled: &CompiledModel) -> Vec<bool> {
    vec![true; compiled.template().len()]
}

fn boundaries(fire: &Composition) -> Vec<i32> {
    let mut lanes: Vec<(u32, u32)> = fire
        .lanes()
        .iter()
        .map(|lane| (lane.row_offset, lane.rows))
        .collect();
    lanes.sort_unstable();
    let mut out = vec![0i32];
    for (_, rows) in lanes {
        out.push(out[out.len() - 1] + rows as i32);
    }
    out
}

fn windows(trace: &Trace, compiled: &CompiledModel, fire: &Composition) -> Windows {
    Windows::of(
        trace,
        compiled,
        model_ir::PerAxis::new([fire.classes(), fire.patch_classes(), fire.voxel_classes()]),
        &boundaries(fire),
        Copies::off(),
        test_slots(),
    )
    .expect("every region seats a window")
}

fn launch_rows(compiled: &CompiledModel, table: &Windows) -> Vec<u32> {
    let mut rows = Vec::new();
    for region in 0..compiled.template().len() as u32 {
        for run in 0..table.runs(region) {
            rows.push(table.at(region, run).span().rows);
        }
    }
    rows
}

fn split(compiled: &CompiledModel, hot: u32, cold: u32) -> Composition {
    compose(compiled, &budget(), &[Lane::new(1, hot), Lane::new(0, cold)])
        .expect("the two-class fire composes")
}

#[test]
fn a_shifted_body_admits_the_split_that_moved_every_case() {
    every_region_of_the_subject_addresses_off_the_seat();
    the_gate_the_narrow_reading_refuses_is_one_the_wide_reading_admits();
    two_splits_of_one_key_move_a_launch_the_total_does_not();
}

fn every_region_of_the_subject_addresses_off_the_seat() {
    let (trace, compiled) = baked();
    let shifted = shifting(&trace, &compiled);
    assert!(
        shifted.iter().all(|&moves| moves),
        "the subject was built out of `SHIFTED` names and the reading says \
         otherwise: {shifted:?}",
    );
    assert!(
        !compiled.template().is_empty(),
        "a template of no regions would make every claim below vacuous",
    );

    let fire = split(&compiled, 5, 3);
    let table = windows(&trace, &compiled, &fire);
    let offset = (0..compiled.template().len() as u32).any(|region| {
        (0..table.runs(region)).any(|run| {
            let window = table.at(region, run);
            window.span().rows > 0 && window.span().row_offset > 0
        })
    });
    assert!(
        offset,
        "no region of this fire begins above row zero, so it is not the \
         windowed fire this file is about",
    );
}

fn the_gate_the_narrow_reading_refuses_is_one_the_wide_reading_admits() {
    let (trace, compiled) = baked();
    let shifted = shifting(&trace, &compiled);
    let fire = split(&compiled, 5, 3);
    let table = windows(&trace, &compiled, &fire);

    assert!(
        !table.covers_fire(fire.rows()),
        "the narrow reading admitted a two-class fire, so this subject is not \
         windowed and proves nothing",
    );
    assert!(
        table.covers_fire_shifted(fire.rows(), &shifted, &lane_shifting(&compiled)),
        "every region of this fire is either the whole fire or one whose ops \
         all read the seat's start, and the wide gate refused it anyway",
    );

    let mut crippled = shifted.clone();
    let windowed = (0..compiled.template().len())
        .find(|&region| {
            (0..table.runs(region as u32)).any(|run| {
                let window = table.at(region as u32, run);
                window.span().rows > 0 && window.span().row_offset > 0
            })
        })
        .expect("(a) found a windowed region");
    crippled[windowed] = false;
    assert!(
        !table.covers_fire_shifted(fire.rows(), &crippled, &lane_shifting(&compiled)),
        "a windowed region that does NOT move its own base was admitted; the \
         launch plane would hand it pre-shifted pointers under a disarmed seat",
    );

    let table_admits = table.admits(fire.rows(), &crippled, &lane_shifting(&compiled));
    assert_eq!(
        table_admits[windowed],
        engine_cuda::window::Admit::Island,
        "the region whose shift was taken away is the one that cannot be in a \
         graph, and the table has to say so at ITS index",
    );
    assert_eq!(
        table_admits
            .iter()
            .filter(|admit| **admit == engine_cuda::window::Admit::Island)
            .count(),
        1,
        "crippling one region made more than one island, so the table is not \
         answering per region: {table_admits:?}",
    );
    assert!(
        table
            .admits(fire.rows(), &shifted, &lane_shifting(&compiled))
            .iter()
            .all(|admit| *admit == engine_cuda::window::Admit::Captured),
        "the uncrippled slice left an island behind, so the collapsed reading \
         and the table disagree",
    );
}

fn two_splits_of_one_key_move_a_launch_the_total_does_not() {
    let (trace, compiled) = baked();
    let first = split(&compiled, 5, 3);
    let second = split(&compiled, 3, 5);

    assert_eq!(first.rows(), second.rows(), "the two splits share a total");
    assert_eq!(
        first.bucket(),
        second.bucket(),
        "and therefore a lattice point",
    );
    assert_eq!(
        BodyKey::of_axes(first.classes(), first.bucket(), &no_decode_class(), LANES, None),
        BodyKey::of_axes(second.classes(), second.bucket(), &no_decode_class(), LANES, None),
        "the two fires must reach the SAME body, or there is no hazard here",
    );

    let a = launch_rows(&compiled, &windows(&trace, &compiled, &first));
    let b = launch_rows(&compiled, &windows(&trace, &compiled, &second));
    assert_eq!(
        a.len(),
        b.len(),
        "one key, one launch count — the comparison in `record` walks these \
         pairwise and a length that moved would be a key that lied",
    );
    assert!(
        a.iter().zip(&b).any(|(&was, &now)| now > was),
        "no launch of the second split asks for more rows than the first \
         recorded ({a:?} then {b:?}), so this pair is not the hazard the \
         per-launch check exists for",
    );
    assert!(
        a.iter().sum::<u32>() == b.iter().sum::<u32>(),
        "the launches' rows sum the same both ways ({a:?}, {b:?}) — which is \
         exactly why summing them, as `Body::rows: u32` effectively did, \
         cannot see the move",
    );
}
