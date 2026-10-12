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
    assert!(registry().check().is_empty(), "{:?}", registry().check());
}

#[test]
fn a_checked_in_file_reads_and_picks_by_rung() {
    let p = registry().load(M4, "m4-max.toml").expect("reads");
    assert_eq!(
        *p.pick::<Qmm>(&qmm(1, 4096, 131_072)),
        Qmm::Qmv { rows_max: 8 }
    );
    assert_eq!(
        *p.pick::<Qmm>(&qmm(4, 4096, 4096)),
        Qmm::Qmv { rows_max: 2 }
    );
    assert_eq!(*p.pick::<Qmm>(&qmm(5, 4096, 4096)), Qmm::QmmT { bm: 64 });
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

fn with(kernel_qmm: &str) -> String {
    M4.replace(
        &M4[M4.find("qmm = [").unwrap()..M4.find("matmul =").unwrap()],
        &format!("{kernel_qmm}\n"),
    )
}

#[test]
fn an_unknown_name_is_refused_with_the_vocabulary_and_its_place() {
    let e = refusal(&with(
        r#"qmm = [{ when = "keys < 256", use = "qmv", rows_max = 2 }, { use = "qmm_t", bm = 32 }]"#,
    ));
    assert!(
        e.contains(
            "`keys` is not a variable here; declared: rows, k, n, bits, dtype (in `kernel.qmm`)"
        ),
        "{e}"
    );
    let line = M4.lines().position(|l| l.starts_with("qmm = [")).unwrap() + 1;
    assert!(e.starts_with(&format!("p.toml:{line}:")), "{e}");

    let e = refusal(&with(r#"qmm = "qmm_tt""#));
    assert!(
        e.contains("`qmm_tt` is not an alternative of `kernel.qmm`; declared: qmv, qmm_t, qmm_mpp"),
        "{e}"
    );

    let e = refusal(&with(r#"qmm = { use = "qmm_t", bn = 32 }"#));
    assert!(
        e.contains("`bn` is not a constant of `qmm_t`; declared: bm"),
        "{e}"
    );

    let e = refusal(&with(r#"qmm = { use = "qmm_t", bm = 48 }"#));
    assert!(
        e.contains("`kernel.qmm.qmm_t.bm` is one of 8, 16, 32, 64, found 48"),
        "{e}"
    );

    let e = refusal(&M4.replace("[layout]", "[layot]"));
    assert!(e.contains("`layot` is not a decision of toy; domains of toy: compiler, engine, fuse, kernel, layout"), "{e}");

    let e = refusal(&M4.replace("matmul =", "matmull ="));
    assert!(e.contains("under `kernel`: matmul, qmm"), "{e}");

    let e = refusal(&M4.replace("\"residual_add_rmsnorm\"", "\"rms\""));
    assert!(
        e.contains("`rms` is not a flag of `fuse`; declared: mlp_swiglu, residual_add_rmsnorm"),
        "{e}"
    );

    let e = refusal(&M4.replace("api = 1", "api = 2"));
    assert!(
        e.contains("this pie reads profile api 1, the file is api 2"),
        "{e}"
    );
}

#[test]
fn a_ladder_must_be_total_and_every_rung_reachable() {
    let e = refusal(&with(
        r#"qmm = [{ when = "rows < 5", use = "qmv", rows_max = 2 }]"#,
    ));
    assert!(
        e.contains("`kernel.qmm` picks nothing where `rows >= 5`"),
        "{e}"
    );

    let e = refusal(&with(
        r#"qmm = [{ when = "rows < 5", use = "qmv", rows_max = 2 }, { when = "rows < 4", use = "qmv", rows_max = 8 }, { use = "qmm_t", bm = 32 }]"#,
    ));
    assert!(e.contains("rung 2 of `kernel.qmm` is unreachable"), "{e}");

    let file = M4.replace(
        "assumes = { family = 9, ane = true }",
        "assumes = { family = 10, ane = true }",
    );
    let e = refusal(&file.replace(
        &M4[M4.find("qmm = [").unwrap()..M4.find("matmul =").unwrap()],
        "qmm = [{ when = \"rows < 5\", use = \"qmv\", rows_max = 2 }, { use = \"qmm_mpp\", packed = true }]\n",
    ));
    assert!(
        e.contains("ends on an alternative that fits only some shapes"),
        "{e}"
    );
}

#[test]
fn a_partial_alternative_falls_through_where_it_does_not_fit() {
    let file = M4
        .replace(
            "assumes = { family = 9, ane = true }",
            "assumes = { family = 10, ane = true }",
        )
        .replace("weights = \"plain\"", "weights = \"mpp\"");
    let file = file.replace(
        &M4[M4.find("qmm = [").unwrap()..M4.find("matmul =").unwrap()],
        "qmm = [{ use = \"qmm_mpp\", packed = false }, { use = \"qmm_t\", bm = 32 }]\n",
    );
    let p = registry().load(&file, "m5.toml").expect("reads");
    assert_eq!(
        *p.pick::<Qmm>(&qmm(8, 4096, 4096)),
        Qmm::QmmMpp { packed: false }
    );
    assert_eq!(*p.pick::<Qmm>(&qmm(8, 4095, 4096)), Qmm::QmmT { bm: 32 });
}

#[test]
fn what_an_alternative_requires_the_file_must_assume() {
    let e = refusal(&M4.replace("ane = true", "ane = false"));
    assert!(
        e.contains("`kernel.matmul` picks `ane_columns`, which requires `ane`; `target.assumes` does not promise it"),
        "{e}"
    );
    let e = refusal(&M4.replace(
        "assumes = { family = 9, ane = true }",
        "assumes = { family = 9, ane = true, cores = 40 }",
    ));
    assert!(
        e.contains("`cores` is not a fact of toy; declared: family, ane"),
        "{e}"
    );
}

#[test]
fn a_coupled_alternative_needs_its_partner() {
    let file = M4.replace(
        "assumes = { family = 9, ane = true }",
        "assumes = { family = 10, ane = true }",
    );
    let file = file.replace(
        &M4[M4.find("qmm = [").unwrap()..M4.find("matmul =").unwrap()],
        "qmm = [{ use = \"qmm_mpp\", packed = false }, { use = \"qmm_t\", bm = 32 }]\n",
    );
    let e = refusal(&file);
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
fn a_checked_in_file_spells_everything_a_user_file_may_not() {
    let short = M4.replace("[compiler]\nlaunch_us = 4\n", "");
    let e = refusal(&short);
    assert!(e.contains("`compiler.launch_us` is unspecified"), "{e}");
    let file = registry()
        .parse(&short, "user.toml", Strict::User)
        .expect("a user file fills from defaults");
    assert_eq!(file.defaulted, vec!["compiler.launch_us".to_string()]);

    let bare = with(r#"qmm = [{ when = "rows < 5", use = "qmv" }, { use = "qmm_t", bm = 32 }]"#);
    assert!(refusal(&bare).contains("`qmv` leaves `rows_max` unspecified"));
    let file = registry()
        .parse(&bare, "user.toml", Strict::User)
        .expect("reads");
    assert_eq!(file.defaulted, vec!["kernel.qmm.qmv.rows_max".to_string()]);
}

fn facts(family: i64, ane: bool) -> Facts {
    Facts::of(&registry())
        .int("family", family)
        .flag("ane", ane)
}

#[test]
fn resolution_takes_the_first_file_that_holds_and_says_why_others_did_not() {
    let user = Candidate {
        source: Source::User("user/m4.toml".into()),
        text: M4.replace("rows_max = 8", "rows_max = 4"),
    };
    let seeded = Candidate {
        source: Source::Seeded("m4-max.toml".into()),
        text: M4.to_string(),
    };
    let r = registry();

    let (p, report) = r.resolve(&[user.clone(), seeded.clone()], &facts(9, true));
    assert_eq!(report.used, user.source);
    assert!(!report.untuned());
    assert_eq!(
        *p.pick::<Qmm>(&qmm(1, 4096, 131_072)),
        Qmm::Qmv { rows_max: 4 }
    );

    let broken = Candidate {
        text: user.text.replace("rows_max = 4", "rows_max = 3"),
        ..user.clone()
    };
    let (p, report) = r.resolve(&[broken, seeded.clone()], &facts(9, true));
    assert_eq!(report.used, seeded.source);
    assert_eq!(report.rejected.len(), 1);
    assert!(
        report.rejected[0].1.contains("found 3"),
        "{:?}",
        report.rejected
    );
    assert!(report.untuned(), "a rejected file is worth saying");
    assert_eq!(
        *p.pick::<Qmm>(&qmm(1, 4096, 131_072)),
        Qmm::Qmv { rows_max: 8 }
    );

    let (p, report) = r.resolve(std::slice::from_ref(&seeded), &facts(9, false));
    assert_eq!(
        report.used,
        Source::Defaults,
        "the whole file goes, not the one rung"
    );
    assert!(
        report.rejected[0]
            .1
            .contains("it assumes ane = true, and this device has false"),
        "{:?}",
        report.rejected
    );
    assert_eq!(
        *p.pick::<Qmm>(&qmm(1, 4096, 131_072)),
        Qmm::Qmv { rows_max: 2 }
    );
    assert_eq!(*p.pick::<Qmm>(&qmm(9, 4096, 131_072)), Qmm::QmmT { bm: 32 });
    assert!(
        !p.on::<MlpSwiglu>(&MlpSwigluAt { width: 1 }),
        "a flag defaults off"
    );
    assert!(p.on::<ResidualAddRmsnorm>(&profile::Anywhere));

    let carried = Candidate {
        text: M4.replace("note = \"hand\"", "measured = false"),
        ..seeded.clone()
    };
    let (_, report) = r.resolve(std::slice::from_ref(&carried), &facts(9, true));
    assert_eq!(report.used, carried.source);
    assert!(
        report.untuned(),
        "a file carried over from other numbers is still worth tuning"
    );

    let unknown = Facts::of(&r).flag("ane", true);
    let (_, report) = r.resolve(&[seeded], &unknown);
    assert!(
        report.rejected[0]
            .1
            .contains("this device does not report family"),
        "{:?}",
        report.rejected
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
        let again = r.parse(&text, "dump.toml", Strict::User).unwrap();
        assert_eq!(file.entries, again.entries);
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
