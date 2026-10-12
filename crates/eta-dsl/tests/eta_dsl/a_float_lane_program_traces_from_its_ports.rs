use eta_ir::container::HostRole;
use eta_ir::registry::ModelProfile;
use eta_ir::validate::bind;

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Traced};

const ROWS: u32 = 16;
const WIDTH: u32 = 8;

fn profile(has_velocity: bool) -> ModelProfile {
    ModelProfile {
        vocab: 32,
        page_size: 2,
        num_layers: 2,
        has_velocity,
        velocity_width: WIDTH,
        ..ModelProfile::dummy()
    }
}

fn float_lane() -> (Traced, [Channel; 4]) {
    let x = Channel::from(vec![0f32; (ROWS * WIDTH) as usize]).named("latents");
    let t = Channel::from([1000f32]).named("timestep");
    let ctx = Channel::from(vec![0f32; 4 * WIDTH as usize]).named("context");
    let out = Channel::new([ROWS, WIDTH], eta_dsl::dtype::f32).named("out");
    let step = Channel::from([0u32]).named("step");
    let rng = Channel::from([7u32, 0u32]).named("rng");
    let dts = Channel::from(vec![0f32, -0.25, -0.25, -0.25, -0.25, 0.0]).named("dts");
    let ts = Channel::from(vec![1000f32, 1000.0, 750.0, 500.0, 250.0, 0.0]).named("ts");
    let (px, pt, pc) = (x.clone(), t.clone(), ctx.clone());
    let (ex, et, eo) = (x.clone(), t.clone(), out.clone());
    let mut b = Builder::new(32, 2);
    b.rows_hint(ROWS);
    b.stage(Stage::Prologue, move || {
        let _ = px.read();
        let _ = pt.read();
        let _ = pc.read();
    });
    b.stage(Stage::Epilogue, move || {
        let k = step.take();
        let x_cur = reshape(ex.take(), [ROWS, WIDTH]);
        let v = intrinsics::velocity(WIDTH);
        assert_eq!(
            v.shape().dims(),
            &[ROWS, WIDTH],
            "velocity sized by the rows hint"
        );
        let dt = gather(dts.read(), &k);
        let next = &x_cur + &(&v * &broadcast(&dt, [ROWS, WIDTH]));
        let r = rng.take();
        let fresh = normal(&r, [ROWS, WIDTH]);
        let first = broadcast(reshape(eq(&k, 0u32), [1, 1]), [ROWS, WIDTH]);
        let x_next = select(&first, &fresh, &next);
        ex.put(reshape(&x_next, [ROWS * WIDTH]));
        eo.put(&x_next);
        et.take();
        et.put(gather(ts.read(), &(&k + 1u32)));
        step.put(&k + 1u32);
        rng.put(&r + &Tensor::constant([0u32, 1u32]));
    });
    let traced = b.build().expect("a float lane's program traces");
    (traced, [x, t, ctx, out])
}

#[test]
fn a_float_lane_program_traces_and_binds_under_a_velocity_profile() {
    let (traced, [x, t, ctx, out]) = float_lane();
    let container = traced.container();
    assert!(
        container.ports.is_empty(),
        "a float lane binds no descriptor port"
    );
    let decl = |ch: &Channel| {
        let dense = traced
            .channel_order()
            .iter()
            .position(|gid| *gid == ch.gid())
            .expect("declared");
        &container.channels[dense]
    };
    assert_eq!(decl(&x).host_role, HostRole::None);
    assert!(decl(&x).seeded);
    assert_eq!(decl(&ctx).host_role, HostRole::Writer);
    assert!(decl(&ctx).seeded);
    assert_eq!(decl(&t).host_role, HostRole::None);
    assert_eq!(decl(&out).host_role, HostRole::Reader);

    bind(container.clone(), profile(true)).expect("binds under a profile stating the velocity");
    assert!(
        bind(container.clone(), profile(false)).is_err(),
        "a text profile refuses `velocity()` at bind"
    );
}
