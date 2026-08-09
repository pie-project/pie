//! **A wall in front of a door nobody opens is not a wall.**
//!
//! # What this file is for
//!
//! Sixteen symbols were classified as hard migration problems — *"the
//! launcher returns `bool` and declines"*, *"a tuning table picks 1 of 15"*,
//! *"needs sm90"* — and then found to be reached by nothing at all. Twenty-one
//! more were deleted as duplicates. Every one of those statements was TRUE.
//! Every one of them was a statement about a **launcher**, and none of them
//! was a statement about whether anything **calls** it.
//!
//! §28's archaeology found the mechanism: the DSL surface was generated FROM
//! the launcher headers (`6d02452de`, `c0e57c7f1` — *"read the HEADERS to
//! learn what a launcher IS"*), so **a wrapper exists whether or not a model
//! asked**, and the wrapper then reads as demand to any tool that stops at
//! it. Both tools stopped at it: `crates/model/tests/kernels_table.rs` checks
//! table ⊇ dsl, and `assert_total` check 1 holds that every classification
//! names a live ROW. **Neither traverses wrapper → caller.**
//!
//! `migration_status.rs` now carries a `consumer:` field beside `why`, and
//! `assert_consumers` refuses a `Structural` verdict for a symbol no text
//! names. This file is the proof that that gate can fail. See
//! `new-horizon.md` §34.
//!
//! # Why the proof is the point
//!
//! This tree has now found seven gates that passed while measuring nothing:
//! a filter over an empty set, a parity test whose two sides were the same
//! expression, a manifest check that skipped every row it could not parse.
//! The rule that came out of it has two halves, and both are asserted below:
//!
//! * **A gate that filters must assert its own denominator** —
//!   [`the_gate_counts_what_it_checked`].
//! * **Something must prove it can fail on demand** —
//!   [`a_live_symbol_marked_consumerless_is_caught`] and
//!   [`a_dead_symbol_marked_consumed_is_caught`], which perturb the table in
//!   BOTH directions and require the failure to name the symbol.
//!
//! # Why the example is red and this file is green
//!
//! The gate turns three rows red today, and they are left red: deletion is a
//! separate task with its own evidence (§10.10 — a launcher goes only when
//! its WHOLE consumer set has gone). So `cargo run --example migration_status`
//! exits 101 by design, and [`the_gate_is_red_for_exactly_the_pinned_rows`]
//! PINS that set instead of tolerating it. Pinning is the stronger assertion:
//! it fails if a fourth row goes dark, and it fails just as loudly if one of
//! the three is resolved or deleted without this pin being updated.

#[path = "../examples/migration_status.rs"]
mod status;

use status::{Cite, Class, Consumer, Refusal, Wall, CLASSIFIED};

/// The `Structural` rows that nothing reaches, as measured.
///
/// Not a tolerance list, in either direction. A row appearing here requires a
/// measurement; a row leaving requires a consumer stated or a deletion
/// argued with its own evidence.
/// [`the_gate_is_red_for_exactly_the_pinned_rows`] fails if the tree and this
/// list disagree by one either way.
const REACHED_BY_NOTHING: &[&str] = &[
    "attn::attention_mtp_paged_history_bf16",
    "attn::merge_attention_states_bf16",
    "gemm::batched_act_x_wt_bf16",
];

/// A `Structural` row that is genuinely reached, used as the live half of the
/// perturbation — `write_kv_to_pages` is named by 27 of the 73 golden traces
/// and by `llama_like/forward/mod.rs`, which is every model with a KV cache.
const LIVE: &str = "attn::write_kv_to_pages";

/// The table with the known-dark rows removed, so that a perturbation below
/// produces **exactly one** red symbol and cannot be confused with an ambient
/// one. A perturbation proof that hides inside an existing failure proves
/// nothing.
fn green_table() -> Vec<Refusal> {
    CLASSIFIED.iter().filter(|row| !REACHED_BY_NOTHING.contains(&row.symbol)).copied().collect()
}

/// [`green_table`] with one row's consumer replaced by a lie.
fn perturb(symbol: &str, consumer: Consumer) -> Vec<Refusal> {
    let mut rows: Vec<Refusal> = green_table().into_iter().filter(|r| r.symbol != symbol).collect();
    let mut doctored = row_for(symbol);
    doctored.consumer = consumer;
    rows.push(doctored);
    rows
}

fn row_for(symbol: &str) -> Refusal {
    *CLASSIFIED.iter().find(|row| row.symbol == symbol).expect("a classified symbol")
}

/// The symbols the gate turned red, so a perturbation can assert it caused
/// exactly one — the count of FAILURES is larger, because a lie about
/// reachability is usually caught by more than one of the gate's instruments
/// at once, which is itself the point.
fn red_symbols(why: &str) -> std::collections::BTreeSet<String> {
    why.lines()
        .filter_map(|l| l.trim_start().strip_prefix("* `"))
        .filter_map(|l| l.split('`').next())
        .map(str::to_owned)
        .collect()
}

/// **The gate, on the real table, is red for exactly the pinned rows.**
///
/// Written to hold whether the pin is empty or not, because the pin is
/// expected to empty as the deletion tasks land, and a test that only works
/// while something is broken is a test that gets deleted with the breakage.
#[test]
fn the_gate_is_red_for_exactly_the_pinned_rows() {
    let outcome = status::check_consumers(CLASSIFIED, &status::refused_set());

    match (outcome, REACHED_BY_NOTHING.is_empty()) {
        (Ok(_), true) => {}
        (Ok(_), false) => panic!(
            "{} row(s) are pinned as reached-by-nothing and the gate passed. Either a \
             consumer was found for them — state it in the row and empty the pin — or the \
             rows were deleted and the pin was not moved: {REACHED_BY_NOTHING:?}",
            REACHED_BY_NOTHING.len()
        ),
        (Err(why), true) => panic!(
            "a `Structural` row is reached by nothing and the pin is empty. This is the gate \
             doing its job on a NEW row: measure it, state a consumer or pin it here with the \
             sweep, and do not silence it.\n{why}"
        ),
        (Err(why), false) => {
            for symbol in REACHED_BY_NOTHING {
                assert!(
                    why.contains(symbol),
                    "`{symbol}` is pinned as reached-by-nothing and the gate did not name \
                     it.\n{why}"
                );
            }
            // And nothing ELSE is red: the pin is a set, not a floor, or a
            // fourth dark row slips in behind the pinned ones.
            assert_eq!(
                red_symbols(&why).len(),
                REACHED_BY_NOTHING.len(),
                "the gate turned {} row(s) red and {} are pinned. A row changed state; \
                 re-measure it and move the pin.\n{why}",
                red_symbols(&why).len(),
                REACHED_BY_NOTHING.len()
            );
        }
    }
}

/// **Perturbation one: a live symbol marked consumer-less is caught, by name.**
///
/// This is the assertion that the gate is not simply green-by-construction on
/// `Nothing`. Marking `attn::write_kv_to_pages` unreached is a lie a naive
/// gate would happily accept, since `Consumer::Nothing` carries no citation to
/// check. It is caught because `Nothing` is MEASURED: its stated wrapper is
/// looked for under `crates/model/src`, and `write_kv_to_pages` is right
/// there.
#[test]
fn a_live_symbol_marked_consumerless_is_caught() {
    let rows = perturb(
        LIVE,
        Consumer::Nothing {
            wrapper: "write_kv_to_pages",
            swept: "a fabrication: this symbol is named by 27 goldens and by every KV-cache model",
        },
    );

    let why = status::check_consumers(&rows, &status::refused_set())
        .expect_err("a live symbol claiming no consumer must be refused");
    assert!(why.contains(LIVE), "the failure did not name `{LIVE}`:\n{why}");
    assert_eq!(
        red_symbols(&why),
        [LIVE.to_owned()].into(),
        "the perturbation must be the only symbol turned red:\n{why}"
    );
    assert!(
        why.contains("crates/model/src"),
        "the failure must name the file that DOES call it, or the reader has to go find it \
         themselves:\n{why}"
    );
    // Caught THREE ways, independently: the wall rule, the golden corpus, and
    // the sweep of `crates/model/src` for the wrapper. Any one of them alone
    // would do; three is what makes the field hard to lie in by accident.
    assert!(why.contains("27"), "the golden corpus did not speak:\n{why}");
    assert!(why.matches(LIVE).count() >= 3, "only one instrument spoke:\n{why}");
}

/// **Perturbation two: a dead symbol marked consumed is caught, by name.**
///
/// The direction the taxonomy actually failed in, and the whole reason the
/// field is an enum with a citation rather than free text: `consumer:
/// "reachable"` would pass any checker, because there is nothing in it to
/// check. A [`Consumer::ModelText`] must name a FILE and a TOKEN, and
/// `merge_attention_states` is in no file under `crates/model/src` — the claim
/// is refused when the file is opened and read.
#[test]
fn a_dead_symbol_marked_consumed_is_caught() {
    let dead = "attn::merge_attention_states_bf16";
    let mut rows = green_table();
    let mut doctored = row_for(dead);
    doctored.consumer = Consumer::ModelText {
        cite: Cite {
            at: "crates/model/src/shared/llama_like/forward/mod.rs:1198",
            names: "merge_attention_states",
        },
        goldens: 0,
    };
    rows.push(doctored);

    let why = status::check_consumers(&rows, &status::refused_set())
        .expect_err("a fabricated model-text consumer must be refused");
    assert!(why.contains(dead), "the failure did not name `{dead}`:\n{why}");
    assert_eq!(red_symbols(&why), [dead.to_owned()].into(), "only the fabrication:\n{why}");
    assert!(
        why.contains("merge_attention_states"),
        "the failure must name the token it could not find, so the fix is one grep:\n{why}"
    );
    assert!(
        why.contains("does not contain that token"),
        "the citation was not opened and read — which is the only thing separating this enum \
         from free text:\n{why}"
    );
}

/// **Perturbation three: a citation that has drifted is caught, and the
/// failure names the true line.**
///
/// A `consumer:` is worth something because a human can check it in seconds; a
/// line number forty screens off costs that back. The failure prints the line
/// the token is REALLY on, so the repair is one character.
#[test]
fn a_drifted_citation_is_caught_and_says_where_it_went() {
    let live = "gemm::act_x_wt_bf16";
    let rows = perturb(
        live,
        Consumer::ModelText {
            cite: Cite { at: "crates/model/src/glm_5/forward/mod.rs:99127", names: "gemm_xwt" },
            goldens: 2,
        },
    );

    let why = status::check_consumers(&rows, &status::refused_set())
        .expect_err("a citation pointing at line 99127 must be refused");
    assert!(why.contains(live), "the failure did not name `{live}`:\n{why}");
    assert_eq!(red_symbols(&why), [live.to_owned()].into(), "only the perturbation:\n{why}");
    assert!(
        why.contains("drifted"),
        "a citation whose token exists but is nowhere near the cited line is a different \
         failure from a missing token, and should read as one:\n{why}"
    );
    let true_line = "write `crates/model/src/glm_5/forward/mod.rs:";
    assert!(
        why.contains(true_line) && !why.contains(&format!("{true_line}99127`")),
        "the failure must print the TRUE line, not echo the wrong one, or the repair is a \
         search:\n{why}"
    );
}

/// **Perturbation four: a golden count taken on faith is caught.**
///
/// `goldens:` is the one number in a consumer that is not a citation, and it
/// is corroborated by a second instrument — the 73 traces under
/// `crates/model/tests/golden`, read directly. A row may not simply assert a
/// popularity.
#[test]
fn a_wrong_golden_count_is_caught() {
    let live = "attn::dequant_kv_cache_layer_to_bf16_active";
    let Consumer::ModelText { cite, goldens } = row_for(live).consumer else {
        panic!("`{live}` is a model-text consumer");
    };
    assert_eq!(goldens, 9, "the fixture moved; re-measure before trusting this test");
    let rows = perturb(live, Consumer::ModelText { cite, goldens: goldens + 1 });

    let why = status::check_consumers(&rows, &status::refused_set())
        .expect_err("a golden count nobody measured must be refused");
    assert!(why.contains(live), "the failure did not name `{live}`:\n{why}");
    assert!(why.contains("10 golden") && why.contains("9 of the"), "{why}");
    assert_eq!(red_symbols(&why), [live.to_owned()].into(), "only the perturbation:\n{why}");
}

/// **Perturbation five: a `Cpp` consumer does not hold up a wall.**
///
/// A C++-internal caller keeps the `.cu` — §10.10 — and says NOTHING about
/// whether the row is wanted. Twelve of §28's sixty-two unreached rows are
/// exactly there, and reading *"`csrc` calls it"* as demand is the same
/// one-hop error in a different accent. The variant exists so that fact can be
/// written down; it is deliberately not green.
#[test]
fn a_cpp_only_consumer_does_not_hold_up_a_wall() {
    let live = "gemm::act_x_wt_bf16_out_fp32";
    let rows = perturb(
        live,
        Consumer::Cpp {
            cite: Cite {
                at: "crates/kernels-cuda/csrc/src/gemm/gemm.cpp:2148",
                names: "act_x_wt_bf16_out_fp32",
            },
        },
    );

    let why = status::check_consumers(&rows, &status::refused_set())
        .expect_err("a `Structural` verdict resting only on a C++ caller must be refused");
    assert!(why.contains(live), "the failure did not name `{live}`:\n{why}");
    assert!(
        why.contains("says nothing about the ROW"),
        "the refusal must say WHY a C++ caller is not demand, or the next reader adds one to \
         silence it:\n{why}"
    );
    assert_eq!(red_symbols(&why), [live.to_owned()].into(), "only the perturbation:\n{why}");
}

/// **The gate counts what it checked.**
///
/// Half of the rule that came out of seven hollow gates: *a gate that filters
/// must assert its own denominator*. Every number below could be silently
/// zero — a table that lost its rows, a corpus directory that moved, a
/// `Consumer` variant that cites nothing — and each zero would make some
/// check above pass vacuously.
#[test]
fn the_gate_counts_what_it_checked() {
    let rows = green_table();
    let checked = status::check_consumers(&rows, &status::refused_set())
        .expect("with the dark rows removed the rest hold");

    assert_eq!(checked.rows, rows.len(), "the gate did not visit every row it was given");
    assert_eq!(
        checked.rows,
        CLASSIFIED.len() - REACHED_BY_NOTHING.len(),
        "the classification changed size; re-derive the pin"
    );
    assert!(
        checked.traces >= 60,
        "the gate read {} golden traces — every `goldens:` claim in the table is checked \
         against that number, and a small one is a vacuous check",
        checked.traces
    );

    let citations: usize = rows.iter().map(|r| r.consumer.cites().len()).sum();
    assert_eq!(checked.citations, citations, "a citation went unopened");
    assert!(
        checked.citations >= rows.len() - checked.nothing,
        "every consumer other than `Nothing` cites at least one file — that is the enum's \
         entire claim over free text"
    );
    assert!(citations >= rows.len(), "only {citations} citation(s) in {} rows", rows.len());
}

/// **Every citation in the real table opens and reads — including the red
/// rows'.**
///
/// The reds fail the WALL rule, not the citation rule, and their `Nothing`
/// sweeps are checked too: each names a `dsl.rs` wrapper that exists and that
/// no model text mentions. Run separately from
/// [`the_gate_is_red_for_exactly_the_pinned_rows`] so a citation rotting
/// cannot hide inside the expected failure.
#[test]
fn every_citation_in_the_table_opens_and_reads() {
    let mut opened = 0;
    for row in CLASSIFIED {
        for cite in row.consumer.cites() {
            status::resolve(cite).unwrap_or_else(|why| panic!("`{}`'s consumer {why}", row.symbol));
            opened += 1;
        }
    }
    let nothings =
        CLASSIFIED.iter().filter(|r| matches!(r.consumer, Consumer::Nothing { .. })).count();
    let fact_gated =
        CLASSIFIED.iter().filter(|r| matches!(r.consumer, Consumer::FactGated { .. })).count();
    assert_eq!(
        opened,
        CLASSIFIED.len() - nothings + fact_gated,
        "one citation per row, none for `Nothing`, two for each fact-gated row: {opened} opened \
         across {} rows ({nothings} `Nothing`, {fact_gated} fact-gated)",
        CLASSIFIED.len()
    );
    assert!(opened >= 25, "only {opened} citation(s) opened — the gate is thinning out");
}

/// **A `Structural` verdict cannot be spelled without a consumer at all.**
///
/// The type-level half. `Refusal` has four fields and none of them is
/// optional, so the one thing this session cannot happen again is a row
/// landing with `why` filled in and demand never asked about. There is no
/// `Consumer::default()`, and `Consumer::Nothing` requires a written sweep.
#[test]
fn every_classified_row_states_a_consumer_and_a_sweep() {
    let mut nothings = 0;
    for row in CLASSIFIED {
        if let Consumer::Nothing { wrapper, swept } = row.consumer {
            nothings += 1;
            assert!(
                swept.len() > 40,
                "`{}` claims nothing reaches it in {} characters. `Nothing` is the most \
                 expensive claim in this table — it is what a deletion is argued from — and it \
                 must say WHICH channels were swept",
                row.symbol,
                swept.len()
            );
            assert!(
                !wrapper.is_empty(),
                "`{}` states `Nothing` behind no wrapper at all; say `\"\"` only if the symbol \
                 has no DSL surface, and then say so in `swept`",
                row.symbol
            );
        }
        assert!(row.why.len() > 10, "`{}` has no stated reason", row.symbol);
    }
    assert_eq!(nothings, 4, "the `Nothing` population changed; re-measure and move the pin");

    // Three of the four are `Structural` and red; the fourth is `Stale`,
    // which the gate lets past as amber because a stale refusal is already an
    // admission that the row needs re-deriving.
    let amber: Vec<&str> = CLASSIFIED
        .iter()
        .filter(|r| matches!(r.consumer, Consumer::Nothing { .. }))
        .filter(|r| !matches!(r.class, Class::Structural(_)))
        .map(|r| r.symbol)
        .collect();
    assert_eq!(amber, ["ssm::build_nemotron_moe_ptrs_decode_batched_bf16"]);
}

/// A shape assertion, so the enum cannot quietly grow a way to say
/// *"reachable"*.
///
/// §21.14's test is *does the shape make a wrong claim well-formed?* The
/// answer here is that the only variant carrying no citation is
/// [`Consumer::Nothing`], and it is the one variant the gate MEASURES rather
/// than believes. Every other variant must name a file and a token.
#[test]
fn only_the_measured_variant_may_cite_nothing() {
    for row in CLASSIFIED {
        let cites = row.consumer.cites().len();
        match row.consumer {
            Consumer::Nothing { .. } => assert_eq!(cites, 0),
            Consumer::FactGated { .. } => assert_eq!(
                cites, 2,
                "`{}` is fact-gated: reachability is `does a checkpoint publish this repr`, \
                 which takes TWO citations — the selector arm and the one place the fact is \
                 built",
                row.symbol
            ),
            _ => assert_eq!(cites, 1, "`{}` cites nothing", row.symbol),
        }
        // A non-model-text consumer claiming goldens is a category error: a
        // golden trace is a model text, run.
        if !matches!(row.consumer, Consumer::ModelText { .. } | Consumer::Lowering { .. }) {
            assert_eq!(
                row.consumer.goldens(),
                0,
                "`{}` claims goldens through a channel that cannot produce one",
                row.symbol
            );
        }
    }
    // And the walls the gate accepts are exactly the reaching ones.
    assert!(!Consumer::Nothing { wrapper: "x", swept: "y" }.holds_up_a_wall());
    assert!(!Consumer::Cpp { cite: Cite { at: "a", names: "b" } }.holds_up_a_wall());
    assert!(Consumer::Driver { cite: Cite { at: "a", names: "b" } }.holds_up_a_wall());
    let _ = Wall::Library;
}
