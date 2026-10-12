//! A toy backend registry, declared the way a kernel crate declares its own,
//! and the files that read against it.

use profile::{Candidate, Decision, Domain, Facts, Registry, Source, Strict, Var};

const FACTS: &[Var] = &[
    Var::fixed("family", Domain::COUNT),
    Var::fixed("ane", Domain::Bool),
];

/// Dense affine-quantised matmul.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, PartialEq, Decision)]
#[decision(
    id = "kernel.qmm",
    shape(rows; k, n, bits: set(2, 4, 8), dtype: enum(bf16, f16)),
    default = r#"[{ when = "rows < 5", use = "qmv", rows_max = 2 }, { use = "qmm_t", bm = 32 }]"#,
    fits = Qmm::fits_shape,
)]
enum Qmm {
    /// One threadgroup per output row.
    Qmv {
        #[constant(set(1, 2, 4, 8), default = 2)]
        rows_max: u32,
    },
    /// Tiled.
    QmmT {
        #[constant(set(8, 16, 32, 64), default = 32)]
        bm: u32,
    },
    /// The tensor-core path.
    #[requires("family >= 10")]
    #[partial]
    #[consistent_with("layout.weights = mpp")]
    QmmMpp {
        #[constant(default = true)]
        packed: bool,
    },
}

impl Qmm {
    fn fits_shape(&self, at: &QmmAt) -> bool {
        match self {
            Qmm::QmmMpp { .. } => at.k % 64 == 0,
            _ => true,
        }
    }
}

/// How weights are laid out at import.
#[derive(Clone, Debug, PartialEq, Decision)]
#[decision(id = "layout.weights", default = "plain")]
enum Weights {
    Plain,
    #[requires("family >= 10")]
    Mpp,
}

/// A matmul split with the Neural Engine.
#[derive(Clone, Debug, PartialEq, Decision)]
#[decision(id = "kernel.matmul", shape(rows; n), default = "gpu")]
enum Matmul {
    Gpu,
    #[requires("ane")]
    AneColumns {
        #[constant(range(1..=99), default = 47)]
        share: u32,
    },
}

/// Tokens per prefill chunk.
#[derive(Clone, Copy, Debug, PartialEq, Decision)]
#[decision(id = "engine.prefill.chunk", constant(range(64..=65_536)), default = 4096)]
struct PrefillChunk(u32);

/// The frame's launch cost, in microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Decision)]
#[decision(id = "compiler.launch_us", constant(range(0..=1000)), default = 4.0)]
struct LaunchUs(f64);

#[derive(Clone, Copy, Debug, PartialEq, Decision)]
#[decision(id = "fuse.mlp_swiglu", flag, shape(width))]
struct MlpSwiglu;

#[derive(Clone, Copy, Debug, PartialEq, Decision)]
#[decision(id = "fuse.residual_add_rmsnorm", flag, default = true)]
struct ResidualAddRmsnorm;

/// A flag beside constants: written as its own key.
#[derive(Clone, Copy, Debug, PartialEq, Decision)]
#[decision(id = "engine.copies", flag, default = true)]
struct Copies;

/// Two flags alone under one path: written as a list there.
#[derive(Clone, Copy, Debug, PartialEq, Decision)]
#[decision(id = "kernel.sdpa.split_small", flag)]
struct SplitSmall;

#[derive(Clone, Copy, Debug, PartialEq, Decision)]
#[decision(id = "kernel.sdpa.split_window", flag)]
struct SplitWindow;

fn registry() -> Registry {
    let mut r = Registry::new("toy", FACTS, &["rows"]);
    r.add::<Qmm>()
        .add::<Weights>()
        .add::<Matmul>()
        .add::<PrefillChunk>()
        .add::<LaunchUs>()
        .add::<MlpSwiglu>()
        .add::<ResidualAddRmsnorm>()
        .add::<Copies>()
        .add::<SplitSmall>()
        .add::<SplitWindow>();
    r
}

const M4: &str = r#"
api = 1
fuse = ["residual_add_rmsnorm", { mlp_swiglu = "width <= 16_384" }]

[target]
device  = "Apple M4 Max"
assumes = { family = 9, ane = true }

[provenance]
note = "hand"

[kernel]
qmm = [
  { when = "rows < 5 and n >= 65_536", use = "qmv", rows_max = 8 },
  { when = "rows < 5",                 use = "qmv", rows_max = 2 },
  {                                    use = "qmm_t", bm = 64 },
]
matmul = [{ when = "rows >= 64 and n >= 8192", use = "ane_columns", share = 47 }, "gpu"]
sdpa = ["split_small"]

[layout]
weights = "plain"

[engine]
prefill.chunk = 4096
copies = false

[compiler]
launch_us = 4
"#;

fn qmm(rows: i64, k: i64, n: i64) -> QmmAt {
    QmmAt {
        rows,
        k,
        n,
        bits: 4,
        dtype: "bf16",
    }
}

#[test]
fn the_registry_holds_together() {
    let problems = registry().check();
    assert!(problems.is_empty(), "{problems:?}");
}

#[test]
fn a_checked_in_file_reads_and_picks_by_rung() {
    let p = registry().load(M4, "m4-max.toml").expect("reads");
    for (at, want) in [
        (qmm(1, 4096, 131_072), Qmm::Qmv { rows_max: 8 }),
        (qmm(4, 4096, 4096), Qmm::Qmv { rows_max: 2 }),
        (qmm(5, 4096, 4096), Qmm::QmmT { bm: 64 }),
    ] {
        assert_eq!(*p.pick::<Qmm>(&at), want, "{at:?}");
    }
    assert_eq!(p.get::<PrefillChunk>().0, 4096);
    assert!(
        (p.get::<LaunchUs>().0 - 4.0).abs() < f64::EPSILON,
        "an integer reads as a float"
    );
    assert!(p.on::<MlpSwiglu>(&MlpSwigluAt { width: 16_384 }));
    assert!(!p.on::<MlpSwiglu>(&MlpSwigluAt { width: 16_385 }));
    assert!(p.on::<ResidualAddRmsnorm>(&profile::Anywhere));
    assert_eq!(
        *p.pick::<Matmul>(&MatmulAt { rows: 64, n: 8192 }),
        Matmul::AneColumns { share: 47 }
    );
    assert_eq!(
        *p.pick::<Matmul>(&MatmulAt { rows: 63, n: 8192 }),
        Matmul::Gpu
    );
}

fn refusal(text: &str) -> String {
    registry()
        .parse(text, "p.toml", Strict::CheckedIn)
        .expect_err("refused")
        .to_string()
}

/// `file` (M4 or a variant of it) with `kernel.qmm` replaced.
fn with_qmm(file: &str, qmm: &str) -> String {
    let old = &M4[M4.find("qmm = [").unwrap()..M4.find("matmul =").unwrap()];
    file.replace(old, &format!("{qmm}\n"))
}

/// M4 on a family-10 chip whose qmm leads with the partial `qmm_mpp`.
fn m5() -> String {
    with_qmm(
        &M4.replace("family = 9", "family = 10"),
        r#"qmm = [{ use = "qmm_mpp", packed = false }, { use = "qmm_t", bm = 32 }]"#,
    )
}

#[test]
fn an_unknown_name_is_refused_with_the_vocabulary_and_its_place() {
    let e = refusal(&with_qmm(
        M4,
        r#"qmm = [{ when = "keys < 256", use = "qmv", rows_max = 2 }, { use = "qmm_t", bm = 32 }]"#,
    ));
    assert!(
        e.contains(
            "`keys` is not a variable here; declared: rows, k, n, bits, dtype, family, ane (in `kernel.qmm`)"
        ),
        "{e}"
    );
    let line = M4.lines().position(|l| l.starts_with("qmm = [")).unwrap() + 1;
    assert!(e.starts_with(&format!("p.toml:{line}:")), "{e}");

    for (file, want) in [
        (
            with_qmm(M4, r#"qmm = "qmm_tt""#),
            "`qmm_tt` is not an alternative of `kernel.qmm`; declared: qmv, qmm_t, qmm_mpp",
        ),
        (
            with_qmm(M4, r#"qmm = { use = "qmm_t", bn = 32 }"#),
            "`bn` is not a constant of `qmm_t`; declared: bm",
        ),
        (
            with_qmm(M4, r#"qmm = { use = "qmm_t", bm = 48 }"#),
            "`kernel.qmm.qmm_t.bm` is one of 8, 16, 32, 64, found 48",
        ),
        (
            M4.replace("[layout]", "[layot]"),
            "`layot` is not a decision of toy; domains of toy: compiler, engine, fuse, kernel, layout",
        ),
        (
            M4.replace("matmul =", "matmull ="),
            "under `kernel`: matmul, qmm",
        ),
        (
            M4.replace("\"residual_add_rmsnorm\"", "\"rms\""),
            "`rms` is not a flag of `fuse`; declared: mlp_swiglu, residual_add_rmsnorm",
        ),
        (
            M4.replace("api = 1", "api = 2"),
            "this pie reads profile api 1, the file is api 2",
        ),
    ] {
        let e = refusal(&file);
        assert!(e.contains(want), "{e}");
    }
}

#[test]
fn a_ladder_must_be_total_and_every_rung_reachable() {
    for (file, want) in [
        (
            with_qmm(
                M4,
                r#"qmm = [{ when = "rows < 5", use = "qmv", rows_max = 2 }]"#,
            ),
            "`kernel.qmm` picks nothing where `rows >= 5`",
        ),
        (
            with_qmm(
                M4,
                r#"qmm = [{ when = "rows < 5", use = "qmv", rows_max = 2 }, { when = "rows < 4", use = "qmv", rows_max = 8 }, { use = "qmm_t", bm = 32 }]"#,
            ),
            "rung 2 of `kernel.qmm` is unreachable",
        ),
        (
            with_qmm(
                &M4.replace("family = 9", "family = 10"),
                r#"qmm = [{ when = "rows < 5", use = "qmv", rows_max = 2 }, { use = "qmm_mpp", packed = true }]"#,
            ),
            "ends on an alternative that fits only some shapes",
        ),
    ] {
        let e = refusal(&file);
        assert!(e.contains(want), "{e}");
    }
}

#[test]
fn a_partial_alternative_falls_through_where_it_does_not_fit() {
    let file = m5().replace("weights = \"plain\"", "weights = \"mpp\"");
    let p = registry().load(&file, "m5.toml").expect("reads");
    assert_eq!(
        *p.pick::<Qmm>(&qmm(8, 4096, 4096)),
        Qmm::QmmMpp { packed: false }
    );
    assert_eq!(*p.pick::<Qmm>(&qmm(8, 4095, 4096)), Qmm::QmmT { bm: 32 });
}

#[test]
fn a_rung_that_compares_an_unknown_variable_does_not_hold() {
    /// A tile by rows and how full the device is, which a device that
    /// does not say its size leaves unknown.
    #[derive(Clone, Copy, Debug, PartialEq, Decision)]
    #[decision(
        id = "kernel.tile",
        shape(rows, fill: derived optional;),
        constant(set(8, 16, 32, 64)),
        default = r#"[
            { when = "fill >= 8", value = 64 },
            { when = "fill < 8 and rows < 4", value = 8 },
            { when = "rows >= 100 or not (fill < 2)", value = 32 },
            { value = 16 },
        ]"#
    )]
    struct Tile(u32);
    let mut r = registry();
    r.add::<Tile>();
    assert!(r.check().is_empty(), "{:?}", r.check());
    let p = r.defaults();
    let tile = |rows, fill| p.pick::<Tile>(&TileAt { rows, fill }).0;
    assert_eq!(tile(2, Some(9)), 64);
    assert_eq!(tile(2, Some(1)), 8);
    assert_eq!(tile(50, Some(3)), 32);
    assert_eq!(tile(2, None), 16, "neither side of `fill < 8` holds");
    assert_eq!(tile(50, None), 16);
    assert_eq!(
        tile(200, None),
        32,
        "a rung holds where it does whatever `fill` is"
    );
}

#[test]
fn what_an_alternative_requires_the_file_must_assume() {
    let e = refusal(&M4.replace("ane = true", "ane = false"));
    assert!(
        e.contains("`kernel.matmul` picks `ane_columns`, which requires `ane`; `target.assumes` does not promise it"),
        "{e}"
    );
    let e = refusal(&M4.replace("ane = true }", "ane = true, cores = 40 }"));
    assert!(
        e.contains("`cores` is not a fact of toy; declared: family, ane"),
        "{e}"
    );
}

#[test]
fn a_coupled_alternative_needs_its_partner() {
    let e = refusal(&m5());
    assert!(
        e.contains("`kernel.qmm` picks `qmm_mpp`, which needs `layout.weights = mpp`, and no rung of `layout.weights` picks `mpp`"),
        "{e}"
    );
}

#[test]
fn a_flag_cannot_read_a_dynamic_variable_and_a_dynamic_variable_is_an_axis() {
    #[derive(Clone, Debug, Decision)]
    #[decision(id = "kernel.attend", shape(keys; head_dim), default = "vector")]
    enum Sdpa {
        Vector,
        Tiled,
    }
    let mut r = registry();
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        r.add::<Sdpa>();
    }));
    let why = refused
        .expect_err("refused")
        .downcast::<String>()
        .map(|s| *s)
        .unwrap_or_default();
    assert!(
        why.contains("`keys` is dynamic but not a recording axis of toy (rows)"),
        "{why}"
    );
}

#[test]
fn a_checked_in_file_spells_its_constants_and_may_leave_decisions_out() {
    let short = M4.replace("[compiler]\nlaunch_us = 4\n", "");
    let file = registry()
        .parse(&short, "p.toml", Strict::CheckedIn)
        .expect("a checked-in file may leave a decision to the layer below");
    assert!(!file.entries.contains_key("compiler.launch_us"));
    let p = registry().load(&short, "p.toml").unwrap();
    assert!((p.get::<LaunchUs>().0 - 4.0).abs() < f64::EPSILON);

    let bare = with_qmm(
        M4,
        r#"qmm = [{ when = "rows < 5", use = "qmv" }, { use = "qmm_t", bm = 32 }]"#,
    );
    let e = refusal(&bare);
    assert!(e.contains("`qmv` leaves `rows_max` unspecified"), "{e}");
    let p = registry()
        .load_user(&bare, "user.toml")
        .expect("a user file fills a constant from its default");
    assert_eq!(
        *p.pick::<Qmm>(&qmm(1, 4096, 4096)),
        Qmm::Qmv { rows_max: 2 }
    );
}

#[test]
fn measured_names_decisions_and_nothing_else() {
    let file = registry()
        .parse(
            &M4.replace(
                "note = \"hand\"",
                "measured = [\"kernel.qmm\", \"engine.copies\"]",
            ),
            "p.toml",
            Strict::CheckedIn,
        )
        .unwrap();
    assert_eq!(file.measured, ["kernel.qmm", "engine.copies"]);
    for (bad, why) in [
        (
            "measured = true",
            "`provenance.measured` is a list of decision ids",
        ),
        (
            "measured = [\"kernel.qmv\"]",
            "`kernel.qmv` is not a decision of toy",
        ),
        (
            "measured = [\"kernel.qmm\", \"kernel.qmm\"]",
            "`kernel.qmm` is listed twice",
        ),
    ] {
        let e = refusal(&M4.replace("note = \"hand\"", bad));
        assert!(e.contains(why), "{bad}: {e}");
    }
}

fn facts(family: i64, ane: bool) -> Facts {
    Facts::of(&registry())
        .int("family", family)
        .flag("ane", ane)
}

fn layer(source: Source, text: &str) -> Candidate {
    Candidate {
        source,
        text: text.to_string(),
    }
}

#[test]
fn a_higher_layer_replaces_what_it_spells_and_a_refused_one_leaves_the_rest() {
    let r = registry();
    let seeded = layer(Source::Seeded("m4-max.toml".into()), M4);
    let user = layer(
        Source::User("user/m4.toml".into()),
        "api = 1\n[provenance]\nmeasured = [\"kernel.qmm\"]\n[kernel]\nqmm = [{ when = \"rows < 5\", use = \"qmv\", rows_max = 4 }, { use = \"qmm_t\", bm = 8 }]\n",
    );
    let decode = qmm(1, 4096, 131_072);

    let (p, report) = r.resolve(&[seeded.clone(), user.clone()], &facts(9, true));
    assert_eq!(report.used, user.source);
    assert_eq!(report.layers, [seeded.source.clone(), user.source.clone()]);
    assert_eq!(report.measured, ["kernel.qmm"]);
    assert!(!report.untuned());
    assert_eq!(*p.pick::<Qmm>(&decode), Qmm::Qmv { rows_max: 4 });
    assert!(!p.on::<Copies>(&profile::Anywhere), "the seeded layer's");
    assert_eq!(
        report.describe(),
        "user profile user/m4.toml over checked-in profile m4-max.toml"
    );

    let (p, report) = r.resolve(&[user.clone(), seeded.clone()], &facts(9, true));
    assert_eq!(
        *p.pick::<Qmm>(&decode),
        Qmm::Qmv { rows_max: 8 },
        "the later layer wins"
    );
    assert_eq!(report.measured, Vec::<&str>::new());

    let broken = layer(
        user.source.clone(),
        &user.text.replace("rows_max = 4", "rows_max = 3"),
    );
    let (p, report) = r.resolve(&[seeded.clone(), broken], &facts(9, true));
    assert_eq!(report.used, seeded.source);
    assert_eq!(report.rejected.len(), 1);
    assert!(report.rejected[0].1.contains("found 3"), "{report:?}");
    assert!(report.untuned(), "a rejected file is worth saying");
    assert_eq!(*p.pick::<Qmm>(&decode), Qmm::Qmv { rows_max: 8 });

    let (p, report) = r.resolve(&[seeded.clone(), user.clone()], &facts(9, false));
    assert_eq!(
        report.layers,
        std::slice::from_ref(&user.source),
        "the whole seeded layer goes, not the one rung"
    );
    assert!(
        report.rejected[0]
            .1
            .contains("it assumes ane = true, and this device has false"),
        "{report:?}"
    );
    assert_eq!(*p.pick::<Qmm>(&decode), Qmm::Qmv { rows_max: 4 });
    assert!(
        !p.on::<MlpSwiglu>(&MlpSwigluAt { width: 1 }),
        "a flag defaults off"
    );
    assert!(p.on::<Copies>(&profile::Anywhere), "the default again");

    let (_, report) = r.resolve(std::slice::from_ref(&seeded), &facts(9, true));
    assert!(
        report.untuned(),
        "a file that measured nothing is still worth tuning"
    );

    let unknown = Facts::of(&r).flag("ane", true);
    let (_, report) = r.resolve(&[seeded], &unknown);
    assert_eq!(report.used, Source::Defaults);
    assert!(
        report.rejected[0]
            .1
            .contains("this device does not report family"),
        "{report:?}"
    );
}

#[test]
fn a_layer_that_measures_a_decision_it_leaves_out_says_the_one_below_held() {
    let r = registry();
    let family = layer(Source::Seeded("f.toml".into()), M4);
    let device = layer(
        Source::Seeded("f/d.toml".into()),
        "api = 1\n[provenance]\nmeasured = [\"kernel.qmm\"]\n",
    );
    let (_, report) = r.resolve(&[family, device], &facts(9, true));
    assert_eq!(report.measured, ["kernel.qmm"]);
}

/// A family file whose Neural Engine share is settled by the `ane` fact,
/// and whose launch cost reads the family.
const FAMILY: &str = r#"
api = 1

[target]
assumes = { family = 9 }

[kernel]
matmul = [
  { when = "ane and rows >= 64", use = "ane_columns", share = 47 },
  "gpu",
]

[compiler]
launch_us = [{ when = "family >= 9", value = 2 }, { value = 3 }]
"#;

#[test]
fn a_when_may_read_a_fact_and_the_device_settles_it() {
    let r = registry();
    let file = r
        .parse(FAMILY, "family.toml", Strict::CheckedIn)
        .expect("`ane_columns` requires `ane`, and its rung holds only where it is");
    assert_eq!(file.entries["kernel.matmul"].len(), 2);
    let e = refusal(&FAMILY.replace("ane and rows >= 64", "rows >= 64"));
    assert!(
        e.contains("`kernel.matmul` picks `ane_columns`, which requires `ane`; `target.assumes` does not promise it"),
        "{e}"
    );
    let e = refusal(&FAMILY.replace(", { value = 3 }", ""));
    assert!(
        e.contains("`compiler.launch_us` picks nothing where `family <= 8`"),
        "{e}"
    );

    let wide = MatmulAt { rows: 64, n: 8192 };
    let family = layer(Source::Seeded("apple9.toml".into()), FAMILY);
    let (p, report) = r.resolve(std::slice::from_ref(&family), &facts(9, true));
    assert!(report.rejected.is_empty(), "{report:?}");
    assert_eq!(*p.pick::<Matmul>(&wide), Matmul::AneColumns { share: 47 });
    assert_eq!(
        *p.pick::<Matmul>(&MatmulAt { rows: 63, n: 8192 }),
        Matmul::Gpu
    );
    assert!((p.get::<LaunchUs>().0 - 2.0).abs() < f64::EPSILON);
    let dumped = profile::render::dump(&p);
    assert!(
        dumped.contains(r#"{ when = "rows >= 64", use = "ane_columns", share = 47 }"#),
        "a settled `when` names no fact:\n{dumped}"
    );
    assert!(dumped.contains("launch_us = 2.0"), "{dumped}");

    let (p, report) = r.resolve(std::slice::from_ref(&family), &facts(9, false));
    assert!(
        report.rejected.is_empty(),
        "a device without the ANE takes the file: {report:?}"
    );
    assert_eq!(*p.pick::<Matmul>(&wide), Matmul::Gpu);

    let (p, report) = r.resolve(
        std::slice::from_ref(&family),
        &Facts::of(&r).int("family", 9),
    );
    assert!(report.rejected.is_empty(), "{report:?}");
    assert_eq!(
        *p.pick::<Matmul>(&wide),
        Matmul::Gpu,
        "a rung on a fact the device does not report does not hold"
    );
    let all_on_ane = layer(
        family.source.clone(),
        &FAMILY.replace(
            "launch_us = [{ when = \"family >= 9\", value = 2 }, { value = 3 }]",
            "launch_us = [{ when = \"ane\", value = 2 }, { when = \"not ane\", value = 3 }]",
        ),
    );
    let (_, report) = r.resolve(&[all_on_ane], &Facts::of(&r).int("family", 9));
    assert!(
        report.rejected[0].1.contains(
            "`compiler.launch_us` reads `ane`, which this device does not report, and picks nothing without it"
        ),
        "{report:?}"
    );

    let (p, _) = r.read(std::slice::from_ref(&family));
    let dumped = profile::render::dump(&p);
    assert!(
        dumped.contains(r#"when = "ane and rows >= 64""#),
        "without facts the rungs stay as written:\n{dumped}"
    );
}

#[test]
fn a_default_reads_no_fact_and_no_variable_shares_a_facts_name() {
    #[derive(Clone, Debug, Decision)]
    #[decision(id = "kernel.odd", default = r#"[{ when = "ane", use = "a" }, "b"]"#)]
    enum Odd {
        A,
        B,
    }
    #[derive(Clone, Debug, Decision)]
    #[decision(id = "kernel.clash", shape(family), default = "a")]
    enum Clash {
        A,
        B,
    }
    let refused = |add: &dyn Fn(&mut Registry)| {
        let mut r = registry();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| add(&mut r)))
            .expect_err("refused")
            .downcast::<String>()
            .map(|s| *s)
            .unwrap_or_default()
    };
    let why = refused(&|r| {
        r.add::<Odd>();
    });
    assert!(why.contains("a default must run anywhere"), "{why}");
    let why = refused(&|r| {
        r.add::<Clash>();
    });
    assert!(
        why.contains("variable `family` is named as a fact"),
        "{why}"
    );
}

#[test]
fn a_dump_reads_back_to_the_same_profile() {
    let r = registry();
    for p in [r.load(M4, "m4-max.toml").unwrap(), r.defaults()] {
        let text = profile::render::dump(&p);
        let file = r
            .parse(&text, "dump.toml", Strict::User)
            .unwrap_or_else(|e| panic!("{e}\n---\n{text}"));
        for (spec, rungs, _) in p.entries() {
            assert_eq!(
                file.entries.get(spec.id).map(Vec::as_slice),
                Some(rungs),
                "{}\n{text}",
                spec.id
            );
        }
    }
}

#[test]
fn the_schema_names_every_decision_alternative_and_constant() {
    let text = profile::render::schema(&registry());
    for needle in [
        "kernel.qmm  [choice]  varies with rows; fixed per op: k, n, bits (one of 2, 4, 8), dtype (one of bf16, f16)",
        "Dense affine-quantised matmul.",
        "use = \"qmm_mpp\"  (requires family >= 10; fits some shapes; with layout.weights = mpp)",
        "rows_max           one of 1, 2, 4, 8, compiled, default 2",
        "share              integer in 1..99, per launch, default 47",
        "engine.prefill.chunk  [constant]",
        "fuse.mlp_swiglu  [flag]",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
}

#[test]
fn the_census_is_what_a_profile_can_compile() {
    let p = registry().load(M4, "m4-max.toml").unwrap();
    let census: Vec<String> = p
        .census()
        .iter()
        .map(|c| {
            let baked: Vec<String> = c.baked.iter().map(|(k, v)| format!("{k}={v}")).collect();
            format!("{}:{}[{}]", c.decision, c.alternative, baked.join(","))
        })
        .collect();
    assert_eq!(
        census,
        [
            "kernel.matmul:ane_columns[]",
            "kernel.matmul:gpu[]",
            "kernel.qmm:qmv[rows_max=8]",
            "kernel.qmm:qmv[rows_max=2]",
            "kernel.qmm:qmm_t[bm=64]",
            "layout.weights:plain[]",
        ],
        "a launch constant (share) is not compiled"
    );
}

const APPLE9: &str = r#"
api = 1
[target]
assumes = { family = 9 }
[kernel]
qmm = [{ when = "rows < 5", use = "qmv", rows_max = 4 }, { use = "qmm_t", bm = 16 }]
[engine]
copies = false
"#;

const M4_MAX: &str = r#"
api = 1
[target]
device = "Apple M4 Max"
assumes = { ane = true }
[provenance]
measured = ["kernel.matmul"]
[kernel]
matmul = { use = "ane_columns", share = 60 }
"#;

const APPLE10: &str = "api = 1\n[target]\nassumes = { family = 10 }\n";

static SHELF: profile::Shelf = profile::Shelf {
    backend: "toy",
    seeded: &[
        ("apple9.toml", APPLE9),
        ("apple9/apple-m4-max.toml", M4_MAX),
        ("apple10.toml", APPLE10),
    ],
    derived: Some(|name, _| {
        format!("api = 1\n[target]\ndevice = \"{name}\"\n[engine]\nprefill.chunk = 128\n")
    }),
};

#[test]
fn a_device_runs_its_family_then_its_device_file_then_its_user_file() {
    let r = registry();
    let wide = MatmulAt { rows: 64, n: 8192 };
    let decode = qmm(1, 4096, 4096);

    let (p, report) = SHELF.resolve(&r, "Apple M4 Max", &facts(9, true), None);
    assert_eq!(
        report.layers,
        [
            Source::Seeded("apple9.toml".into()),
            Source::Seeded("apple9/apple-m4-max.toml".into())
        ]
    );
    assert_eq!(report.measured, ["kernel.matmul"]);
    assert_eq!(*p.pick::<Matmul>(&wide), Matmul::AneColumns { share: 60 });
    assert_eq!(
        *p.pick::<Qmm>(&decode),
        Qmm::Qmv { rows_max: 4 },
        "the family's"
    );
    assert_eq!(p.get::<PrefillChunk>().0, 4096, "the registry's");

    let (p, report) = SHELF.resolve(&r, "Apple A17 Pro GPU", &facts(9, true), None);
    assert_eq!(report.layers, [Source::Seeded("apple9.toml".into())]);
    assert!(report.untuned());
    assert_eq!(*p.pick::<Qmm>(&decode), Qmm::Qmv { rows_max: 4 });
    assert_eq!(*p.pick::<Matmul>(&wide), Matmul::Gpu);

    let dir = std::env::temp_dir().join(format!("pie-shelf-{}", std::process::id()));
    let path = SHELF.user_path(&dir, "Apple M4 Max");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "api = 1\n[engine]\nprefill.chunk = 1024\n").unwrap();
    let (p, report) = SHELF.resolve(&r, "Apple M4 Max", &facts(9, true), Some(&dir));
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(report.layers.len(), 3, "{report:?}");
    assert!(matches!(report.used, Source::User(_)));
    assert_eq!(p.get::<PrefillChunk>().0, 1024);
    assert_eq!(*p.pick::<Matmul>(&wide), Matmul::AneColumns { share: 60 });
    assert!(!p.on::<Copies>(&profile::Anywhere));

    let (p, report) = SHELF.resolve(&r, "Apple M5", &facts(10, true), None);
    assert_eq!(report.layers, [Source::Seeded("apple10.toml".into())]);
    assert_eq!(*p.pick::<Qmm>(&decode), Qmm::Qmv { rows_max: 2 });

    let (p, report) = SHELF.resolve(&r, "Apple M9", &facts(11, true), None);
    assert_eq!(report.layers, [Source::Derived], "no family applies");
    assert_eq!(p.get::<PrefillChunk>().0, 128);

    let (p, report) = SHELF.read(&r, "Apple M4 Max", None);
    assert_eq!(report.layers.len(), 2, "by name, without facts");
    assert_eq!(*p.pick::<Matmul>(&wide), Matmul::AneColumns { share: 60 });
}

#[test]
fn a_refused_device_file_leaves_its_family() {
    let r = registry();
    let (p, report) = SHELF.resolve(&r, "Apple M4 Max", &facts(9, false), None);
    assert_eq!(report.used, Source::Seeded("apple9.toml".into()));
    assert_eq!(report.rejected.len(), 1);
    assert_eq!(
        report.rejected[0].0,
        Source::Seeded("apple9/apple-m4-max.toml".into())
    );
    assert!(
        report.rejected[0]
            .1
            .contains("it assumes ane = true, and this device has false"),
        "{report:?}"
    );
    assert!(report.untuned());
    assert_eq!(
        *p.pick::<Qmm>(&qmm(1, 4096, 4096)),
        Qmm::Qmv { rows_max: 4 }
    );
    assert_eq!(
        *p.pick::<Matmul>(&MatmulAt { rows: 64, n: 8192 }),
        Matmul::Gpu
    );
}

#[test]
fn the_shelf_check_keeps_one_family_and_one_device_file_per_device() {
    fn check(seeded: &'static [(&'static str, &'static str)]) -> Result<(), String> {
        let dir = std::env::temp_dir().join(format!(
            "pie-shelf-check-{}-{}",
            std::process::id(),
            seeded.len()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        for (path, text) in seeded {
            let at = dir.join(path);
            std::fs::create_dir_all(at.parent().unwrap()).unwrap();
            std::fs::write(at, text).unwrap();
        }
        let shelf = profile::Shelf {
            backend: "toy",
            seeded,
            derived: None,
        };
        let got = shelf.check(&registry(), &dir);
        std::fs::remove_dir_all(&dir).unwrap();
        got
    }
    check(SHELF.seeded).unwrap();
    let e = check(&[
        ("apple9.toml", APPLE9),
        ("ane.toml", "api = 1\n[target]\nassumes = { ane = true }\n"),
    ])
    .unwrap_err();
    assert!(
        e.contains("apple9.toml and ane.toml can both apply to one device"),
        "{e}"
    );
    let e = check(&[
        ("apple9.toml", APPLE9),
        ("apple9/apple-m4-max.toml", M4_MAX),
        (
            "apple9/any.toml",
            "api = 1\n[target]\nassumes = { ane = true }\n",
        ),
    ])
    .unwrap_err();
    assert!(e.contains("can both apply to one device"), "{e}");
    let e = check(&[("apple9.toml", APPLE9), ("apple9/m4.toml", M4_MAX)]).unwrap_err();
    assert!(
        e.contains("names `Apple M4 Max`, filed as apple9/apple-m4-max.toml"),
        "{e}"
    );
    let e = check(&[(
        "apple9.toml",
        "api = 1\n[target]\ndevice = \"x\"\nassumes = { family = 9 }\n",
    )])
    .unwrap_err();
    assert!(e.contains("is a family file, and names a device"), "{e}");
    let e = check(&[
        ("apple9.toml", APPLE9),
        ("apple8/apple-m4-max.toml", M4_MAX),
    ])
    .unwrap_err();
    assert!(
        e.contains("is not in the directory of a family file"),
        "{e}"
    );
}
