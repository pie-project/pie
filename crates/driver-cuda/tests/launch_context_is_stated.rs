//! Every scalar a bind arm reads off the launch context comes from the
//! model, or says why it does not.
//!
//! An arm asks the driver for a fact — `cx.rms_eps()?`, `cx.theta()?`,
//! `cx.moe_norm_topk()?` — and `fire::launch` builds the context those
//! queries read. When it fills one of those fields with a bare literal,
//! the kernel gets a number nobody measured, and every one of them is a
//! scalar, so the model does not fault, it degrades.
//!
//! Four had gone that way before this test existed:
//!
//! - `moe_norm_topk: false` — DeepSeek-V3 and Kimi-K2 publish `false`,
//!   GLM-4.5 publishes `true`. Every mixture routed on the first answer.
//! - `moe_routed_scaling: 1.0` — DeepSeek and GLM publish 2.5, Kimi-K2
//!   2.0. A routed token arrived at two-fifths of its trained size.
//! - `glu_limit`/`glu_alpha: 0.0` — `Deployment::mlp_gate` had stated
//!   gpt-oss's `SiluClamped { limit: 7.0, alpha: 1.702 }` all along, and
//!   `alpha` scales the gate INSIDE the sigmoid, so zero collapses
//!   `silu(a*x)` to `x/2` on every routed expert.
//! - `yarn: [0.0; 4]` — gpt-oss states `factor: 32.0, beta_fast: 32.0,
//!   beta_slow: 1.0, original_max_position: 4096` as its generation's
//!   `ROPE_SCALING`, and `driver-metal` REFUSES a load rather than zero
//!   it: "zeroing YaRN's factor would serve the model with an unrescaled
//!   ladder rather than refuse it".
//!
//! None of those was a hard question. Each was a field the deployment
//! already carried, next to a dozen the launch read off it correctly.
//! That is what this test is for: not to judge the value, but to notice
//! when a name stops being connected to the thing that states it.
//!
//! # Why it reads the FILL now, and used to read the ask
//!
//! It used to enumerate the names from the other end. A kernel row bound
//! an operand with `Source::Ctx("name")`, the scan collected those names
//! out of the kernels crate, and every one of them had to be filled by
//! `fire::launch` or argued for here. That vocabulary is gone: kernel-x
//! deleted the rows, an arm asks by CALLING, and the scan went on
//! searching a tree for a constructor no code spells any more. Both of
//! its tests then failed on their own vacuity floor rather than passing
//! over an empty set, which is the only reason the hole was visible at
//! all. A scan is a claim about a directory, and a claim about a
//! directory expires when the code moves.
//!
//! The replacement is not that scan re-pointed, because the two ends are
//! no longer symmetric. The MISSING half of the old question — a name a
//! kernel reads and the launch never fills — is the type system's now:
//! every `Fire` answer is an `Option`, every `Cx` query turns a `None`
//! into a `Refusal` that names the fact, and the arm refuses at the call
//! site. That half is loud. What stayed silent is the other half — a
//! field that IS filled, with a number the launch invented — and it can
//! only enter at one place. So this reads that place: the three context
//! structs `fire::launch` builds, and every bare literal in them.
//!
//! `AttnCtx` is in scope for the same reason `Source::Attn` was. Its
//! `logits_soft_cap` was the literal `0.0` while gemma-2 published
//! `attn_logit_softcapping: 50.0`, so a capped gemma-2 and an uncapped
//! one attended identically — the same defect as the four above, in the
//! struct beside the one they were in.
//!
//! # The three lists, and why they are three
//!
//! A literal is excused by exactly one of them, and they do not claim the
//! same thing:
//!
//! - [`CONSTANT_BY_ARGUMENT`] — an arm can be handed this, and no
//!   checkpoint varies it.
//! - [`VARIED_BY_A_ROW_WITH_NO_TEXT`] — an arm can be handed it, a
//!   catalog row DOES vary it, and that row cannot dispatch the kernel
//!   that reads it. A debt, not an argument, and its premise is read
//!   rather than restated.
//! - [`NOTHING_DELIVERS_IT`] — no arm can be handed it at all.
//!
//! Which list a name is allowed in is COMPUTED, not declared: the test
//! works out whether the field can reach an arm and rejects an entry
//! filed under a claim stronger or weaker than the reachability supports.
//! That is the whole safety margin — `moe_routed_scaling: 1.0` was also
//! "safe" right up until a routed text arrived.
//!
//! It is also what the previous version of these lists lacked. The names
//! were checked and the REASONS were not, so while the scan was dead four
//! of its eight entries went on arguing from something that had stopped
//! being true: `situ_beta` said no text names a situ kernel, and kimi-k3's
//! every MLP activation is one; `altup_streams` said gemma-3n has no CUDA
//! text, and it is the METAL half that refuses that row; `wna16_group_size`
//! said no row is loaded from a w4a16 checkpoint, and kimi-k2's decode
//! states both wna16 kernels; `rope_interleaved` said one arm reads it, and
//! three do. Each entry is rewritten below against what its claim rests on
//! now, and every one of them still lands on "the literal is not read", by
//! a different route than the one it used to claim.
//!
//! # What this cannot see
//!
//! It is a text scan and it should be read as one.
//!
//! - It asks whether the right-hand side is a LITERAL, not whether it is
//!   the right value. `eps: model.deployment.rope_theta` would pass.
//! - It follows a bare identifier through ONE `let` in the same file and
//!   no further, and it reads that `let`'s first binding.
//! - Pointers are out of scope. `core::ptr::null_mut()` is not a literal
//!   here, because null is how this driver spells absence and a fire that
//!   reads one refuses — deliberate nulls outnumber the fields at risk.
//! - A per-layer VECTOR is judged at its fill line. An empty `Vec` is a
//!   literal; a vector built from the wrong table is not visible.
//! - The reader it asks about is a BIND ARM. The driver reads this struct
//!   too — `ctx.stream`, `ctx.cublas`, `ctx.lora` — and a literal only
//!   those reach is outside the question, which is safe only while no
//!   field is in both sets. None is today.

use std::collections::{BTreeMap, BTreeSet};

/// The names a literal is the right answer for, and why.
///
/// A name here is a claim that no checkpoint in the catalog varies it,
/// made about a field a bind arm CAN be handed. Adding one is cheap and
/// wrong is silent, so each carries the reason it is not a measurement.
const CONSTANT_BY_ARGUMENT: &[(&str, &str)] = &[
    (
        "write_state",
        "whether the fire advances recurrent state, and every class that \
         still exists does. `FireClass` says THE REPAIR CLASSES ARE GONE: \
         `FrozenVerify`, `CommitAdvance` and `StateOnly` were retired when \
         the driver accepted PIE_RS_FLAG_FOLD, because a speculative \
         decode writes to a buffer and folds only the accepted prefix, so \
         nothing is ever wrong and nothing needs freezing. Decode and \
         Prefill remain and both advance. THIS ENTRY MOVED HERE FROM \
         `NOTHING_DELIVERS_IT`, and the move is what that list being \
         COMPUTED is for: it sat there arguing that the ssm arms' \
         `cx.gdn()?` was a constant `None` because the `GdnCtx`-to-`Gdn` \
         conversion was never written. The conversion is written, so an \
         arm can be handed this now — and the reason it stays a literal \
         is the one above, which never depended on that plumbing. A false \
         here would need a class to come back.",
    ),
    (
        "first_token",
        "`attn::write_kv_to_pages`' write origin, and `0` is a fact about \
         the REGION rather than about a checkpoint: a fire's rows begin at \
         its own row zero. The one context that begins elsewhere does not \
         take this literal — `peel_tail_ctx` fills the same field with \
         `i32::try_from(split)`, the boundary the fire already computed \
         for its device peel word — so the two fills are the two answers \
         and a model states neither.",
    ),
    (
        "window_left",
        "the FIRE-WIDE fallback beneath a per-layer table that IS read off \
         the deployment. `window_of` prefers the statement's own param, \
         then `window_left_by_layer` — `dep.windows()`, one entry per \
         attention layer — and only then this. `-1` is how this driver \
         spells UNBOUNDED: `attn_plan` picks a family's full-attention \
         decode plan on `window_of(..) == -1`. So what this answers is a \
         layer past the end of that table, which is a stack stating no \
         attention at all. The value that could be wrong is the table, and \
         it is filled on the line below this literal.",
    ),
];

/// Names that a CATALOG ROW varies, kept literal because that row has
/// no text to dispatch.
///
/// This is a DEBT, not an argument, and the two must not share a list:
/// [`CONSTANT_BY_ARGUMENT`] means "no checkpoint varies it", and an
/// entry that is merely unreachable makes a weaker claim which reads
/// like the stronger one.
///
/// Each entry names the generation that varies it and the fact whose
/// UNREAD status is the premise, so the excuse EXPIRES BY ITSELF: the
/// day someone writes that text, the fact leaves
/// `model/tests/facts_are_read.rs`'s `DECLARED_BUT_UNREAD` and
/// [`the_deferred_names_are_still_out_of_reach`] fails here.
const VARIED_BY_A_ROW_WITH_NO_TEXT: &[(&str, &str, &str, &str)] = &[(
    "rope_interleaved",
    "kimi_k2",
    "rope_yarn_original",
    "the rotation pairs adjacent elements rather than halves, and NOTHING \
     ON `Deployment` STATES IT — this is not a name that came loose from a \
     statement, it is one the model side does not carry, which is what \
     makes the entry a debt rather than a fix. Three arms read it: \
     `rope::rope_bf16`, `rope::rope_partial_last_bf16` and \
     `rope::rope_yarn_original_bf16`. (This entry claimed only the last \
     one did — true of the row world, and never re-measured against the \
     arms.) The dispatchers are llama_like's `cuda::rope`, admitted only \
     for `RopeKind::Standard`, plus gemma-2, gemma-3n and nemotron-h, and \
     gpt-oss through `rope_yarn_original`; HF's shared \
     `apply_rotary_pos_emb` is the half-split `rotate_half`, and a family \
     that differs overrides it in its own `modeling_*.py`. \
     `rope_partial_last` is deepseek-v4's alone, and that row refuses at \
     the door — `deployment()` answers `Refusal::Unsupported` because no \
     build here provisions its compressed store — so it dispatches \
     nothing. kimi-k2 is the row that overrides, measured against the \
     modeling code `moonshotai/Kimi-K2-Instruct` publishes rather than \
     assumed from the shared kernel name: its `rotate_half` is the \
     ordinary half-split, but `apply_rotary_pos_emb` first does \
     `q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)` on \
     both `q_pe` and `k_pe` — a DE-INTERLEAVE — so with respect to the \
     tensor as stored the ladder is interleaved and gpt-oss's is not. A \
     literal here would rotate every K2 position against the wrong pair. \
     It cannot today: no MLA text reads `rope_yarn_original`.",
)];

/// Literals no arm can be handed, and the reason each field exists.
///
/// The claim is not "this value is right", it is "nothing reads it", and
/// the test admits two ways for that to be true — see [`delivered`]:
/// either `bind/facts.rs` never names the field, so no query could
/// return it, or the `Fire` method named for it answers a constant
/// `None`, so every arm that asks REFUSES and never sees the literal.
///
/// Both expire by themselves, which is the point of keeping these here
/// rather than deleting the fields: the day a query is written, or the
/// day that `None` becomes a read of the context, the name has to move
/// to one of the two lists above and someone has to write the argument.
const NOTHING_DELIVERS_IT: &[(&str, &str)] = &[
    (
        "gate_second",
        "the ORDER of the two halves in a fused gate/up tensor. \
         `contract.rs` decides it per family when it stacks the experts \
         (`hf_moe_expert_stacks(b, gate_second = true)`), so by the time a \
         tensor reaches a kernel it is already in this driver's order. The \
         context copy is the dense path's, and it stacks the other way \
         round. `mlp::chunked_situ_bf16`'s `unbound` row names it as half \
         of that binding's FLOOR, which is where a query for it would come \
         from.",
    ),
    (
        "situ_beta",
        "SITU's gate constant. kimi-k3 states the kernel — its every MLP \
         activation is `mlp::situ_bf16` — so this is NOT the unreached \
         family it was described as. What holds is one step later: both \
         situ arms are `unbound`, naming this field and its pair as the \
         FLOOR of a binding nobody has written, and `Deployment` states \
         neither beta, so the arm that gets written will need the model \
         side to carry them first.",
    ),
    ("situ_linear_beta", "as `situ_beta` — the linear half of the same unwritten binding."),
    (
        "wna16_group_size",
        "the AWQ/GPTQ group width, and the ONE name here that is asked \
         for: kimi-k2's decode text states `wna16_gate_up_decode` and \
         `wna16_down_decode`, and both arms read `cx.wna16_group_size()?`. \
         They refuse. `Fire::wna16_group_size` is a constant `None` — the \
         driver holds the width on `WeightView::group_size` and nothing \
         hands it to a fire — so the zero is unreachable rather than \
         unread, and a w4a16 checkpoint DOES state a group size. Wiring \
         the query to this field is one line, and this literal goes live \
         and quiet the moment it happens.",
    ),
    (
        "altup_streams",
        "gemma-3n's AltUp width. It has a CUDA text — `gemma3n_cuda`, the \
         four-way hidden bundle — and it is METAL that refuses the row, \
         which is what this entry used to have backwards. The three `norm` \
         arms its AltUp block states are `unbound` and each names this \
         field as the fact `Cx` has no query for: the streams arrive \
         interleaved in one `[t, k*h]` row, so nothing but the fire knows \
         how it divides.",
    ),
    (
        "altup_active",
        "as `altup_streams` — the active stream's index, and \
         `norm::altup_correct_bf16`'s `unbound` row says it is the one \
         extent on that statement which does not come off a value's own \
         width.",
    ),
    (
        "altup_std_mult_by_layer",
        "the per-layer `gaussian_inverse_cdf(activation_sparsity)` behind \
         `DispatchCtx::altup_std_mult`, which carries `#[expect(dead_code)]` \
         because `mlp::gaussian_topk_bf16` is `unbound` and names the \
         accessor as the half that already exists. Empty means every layer \
         reads zero, which that accessor's doc gives as \"keep everything\" \
         — the neutral answer, not a degraded one.",
    ),
];

/// A literal, as far as this test is concerned.
///
/// The `[0.0; N]` arm has no filler today — `yarn` reads a `match` on
/// `Deployment::rope_scaling` — and it stays because that arm is exactly
/// the shape the fourth bug in this file's header had.
fn is_literal(rhs: &str) -> bool {
    let r = rhs.trim().trim_end_matches(',').trim();
    matches!(r, "true" | "false" | "None" | "Vec::new()")
        || r.parse::<f64>().is_ok()
        || (r.starts_with('[') && r.trim_matches(['[', ']'].as_ref()).starts_with("0.0;"))
}

fn read(rel: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// `src` with every `//` tail removed.
///
/// Everything below reads code, and this crate ARGUES in prose about the
/// facts it does not fill: `bind/arms/mlp.rs` writes `DispatchCtx::situ_beta`
/// into an `unbound` row's reason, and a scan over raw text would read the
/// road not taken as a name something delivers.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The text between the braces that open at `after` and the one that
/// closes them.
///
/// Brace-MATCHED rather than read line by line, because the fills nest:
/// `o_out` is a `match` with two arms and an `unsafe` block inside one of
/// them, and a scan that split on every comma would call `None =>
/// d_attn_out` a field of the struct.
fn balanced(src: &str, after: usize) -> Option<&str> {
    let mut depth = 0usize;
    for (j, c) in src[after..].char_indices() {
        match c {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => {
                if depth == 0 {
                    return Some(&src[after..after + j]);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Every `Name { .. }` literal's body, in source order.
fn bodies<'a>(src: &'a str, header: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = src[at..].find(header) {
        let start = at + i + header.len();
        let Some(body) = balanced(src, start) else { break };
        at = start + body.len();
        out.push(body);
    }
    out
}

/// One struct literal's fields, as `(name, right-hand side)`.
///
/// A shorthand `sm_scale,` answers with its own name as the right-hand
/// side, so [`one_hop`] resolves it exactly as it resolves `field: local`
/// — the two spellings mean the same thing and used to be read
/// differently, the shorthand being waved through as "the launch computed
/// this" without anyone checking what it computed.
fn fields(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    let flush = |entry: &str, out: &mut Vec<(String, String)>| {
        let e = entry.trim();
        // `..fire.clone()`: a base, not a field. The tail context takes
        // one, and what it inherits is judged where it was filled.
        if e.is_empty() || e.starts_with("..") {
            return;
        }
        let (name, rhs) = e.split_once(':').map_or((e, e), |(n, r)| (n.trim(), r.trim()));
        if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            out.push((name.to_string(), rhs.split_whitespace().collect::<Vec<_>>().join(" ")));
        }
    };
    for c in body.chars() {
        match c {
            '{' | '(' | '[' => {
                depth += 1;
                cur.push(c);
            }
            '}' | ')' | ']' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            ',' if depth == 0 => {
                flush(&cur, &mut out);
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    flush(&cur, &mut out);
    out
}

/// What the launch binds `ident` to, one `let` deep.
///
/// One hop and no further, and it reads the FIRST binding of the name.
/// The hop is worth having — `field: local` was a free pass before, and a
/// literal is as easy to hide in a `let` as in a struct — but two hops
/// would be an evaluator, and this file should not become one.
fn one_hop(launch: &str, ident: &str) -> Option<String> {
    let mut at = 0;
    while let Some(i) = launch[at..].find("let ") {
        let s = at + i + 4;
        at = s;
        let rest = &launch[s..];
        let rest = rest.strip_prefix("mut ").unwrap_or(rest);
        let Some(tail) = rest.strip_prefix(ident) else { continue };
        if tail.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        // `let x = ..` or `let x: T = ..`, and nothing that runs past the
        // end of the line before its `=` — a `let` pattern spanning lines
        // is not a name this reads.
        let Some(eq) = tail.find('=') else { continue };
        if tail[..eq].contains('\n') {
            continue;
        }
        let rhs = &tail[eq + 1..];
        let line = rhs.split('\n').next().unwrap_or(rhs);
        return Some(line.trim().trim_end_matches(';').trim().to_string());
    }
    None
}

/// One context literal's field, as this test judges it.
struct Fill {
    /// Which struct fills it.
    owner: &'static str,
    /// The field.
    name: String,
    /// What the launch fills it with, after [`one_hop`].
    rhs: String,
}

/// Every field of every context literal `fire::launch` builds.
///
/// The three structs are NAMED rather than discovered, which is what makes
/// the floors below possible: a scan that finds nothing has to fail, and it
/// can only fail if it knows what it was looking for. The previous version
/// of this file died on exactly this assert, and dying is what told anyone
/// that it had stopped reading anything at all.
fn context_fills(launch: &str) -> Vec<Fill> {
    let mut out = Vec::new();
    for (owner, least_bodies, least_fields) in
        [("DispatchCtx", 1, 20), ("AttnCtx", 2, 8), ("GdnCtx", 1, 8)]
    {
        let bodies = bodies(launch, &format!("{owner} {{"));
        assert!(
            bodies.len() >= least_bodies,
            "`fire::launch` builds {} `{owner}` literals and this test \
             expects at least {least_bodies} — either the context moved \
             out of this file, or it is built by a function now and the \
             fields are its arguments",
            bodies.len(),
        );
        for body in bodies {
            let fields = fields(body);
            assert!(
                fields.len() >= least_fields,
                "a `{owner}` literal parsed to {} fields — the scan broke",
                fields.len(),
            );
            for (name, rhs) in fields {
                let resolved = if rhs.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    one_hop(launch, &rhs).unwrap_or(rhs)
                } else {
                    rhs
                };
                out.push(Fill { owner, name, rhs: resolved });
            }
        }
    }
    out
}

/// Every identifier in `src`.
fn idents(src: &str) -> BTreeSet<String> {
    src.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether `bind/facts.rs` answers `name` with a bare `None`.
///
/// Seven methods do, under the banner "the seven a fire cannot answer",
/// and they are the reason a query being ASKED is not the same as a field
/// being delivered: an arm that asks one gets a `Refusal` naming the
/// fact, every time, and never reaches the context.
fn answers_a_constant_none(facts: &str, name: &str) -> bool {
    let needle = format!("fn {name}(");
    let Some(i) = facts.find(&needle) else { return false };
    let Some(open) = facts[i..].find('{').map(|j| i + j + 1) else { return false };
    balanced(facts, open).is_some_and(|body| body.trim() == "None")
}

/// The `DispatchCtx` fields each of its accessors folds together.
///
/// `Facts::theta` reads `DispatchCtx::theta(layer)`, and the fields that
/// decides between — `rope_theta_by_layer`, then `rope_theta` — are named
/// only in ITS body. Without this hop a per-layer table could be emptied
/// and read as untouchable, which is the hole the retired
/// `ACCESSOR_NOT_FIELD` list papered over by listing the accessors and
/// excusing them.
fn accessor_fields(binds: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut out = BTreeMap::new();
    let Some(i) = binds.find("impl DispatchCtx {") else { return out };
    let Some(block) = balanced(binds, i + "impl DispatchCtx {".len()) else { return out };
    let mut at = 0;
    while let Some(j) = block[at..].find("fn ") {
        let s = at + j + 3;
        let name: String =
            block[s..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        let Some(open) = block[s..].find('{').map(|k| s + k + 1) else { break };
        let Some(body) = balanced(block, open) else { break };
        at = open + body.len();
        let mut reads = BTreeSet::new();
        let mut rest = body;
        while let Some(k) = rest.find("self.") {
            rest = &rest[k + 5..];
            let f: String =
                rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            if !f.is_empty() {
                reads.insert(f);
            }
        }
        out.insert(name, reads);
    }
    out
}

/// Whether a bind arm can be handed this field.
///
/// Three answers, and they are not the same question asked three ways:
///
/// - An `AttnCtx` field always can. `Cx::attn_ctx` hands the struct over
///   WHOLE — the FA2 arms read a dozen of its fields together and the
///   query's own doc argues why — so every field of it is one `.` away
///   from an arm whether or not anything names it today. The test asserts
///   that `Cx` does the same for neither of the other two.
/// - A `DispatchCtx` or `GdnCtx` field can if `bind/facts.rs` names it,
///   directly or through one of the context's own accessors, since that
///   file is the only thing that turns this state into an answer. NAMES
///   it, not reads it: this asks whether the token appears there, so a
///   local sharing a field's name would call that field delivered. The
///   error is toward demanding the stronger argument, which is the
///   direction to make it in.
/// - Except that a method answering a constant `None` delivers nothing,
///   however loudly it is asked for.
fn delivered(fill: &Fill, facts: &str, named: &BTreeSet<String>) -> bool {
    if fill.owner == "AttnCtx" {
        return true;
    }
    named.contains(&fill.name) && !answers_a_constant_none(facts, &fill.name)
}

/// Every name `bind/facts.rs` can put in front of an arm.
fn deliverable_names(facts: &str, binds: &str) -> BTreeSet<String> {
    let mut named = idents(facts);
    for (accessor, reads) in accessor_fields(binds) {
        if named.contains(&accessor) {
            named.extend(reads);
        }
    }
    named
}

#[test]
fn every_context_scalar_is_read_off_the_model_or_argued_for() {
    let launch = strip_line_comments(&read("src/fire/launch.rs"));
    let facts = strip_line_comments(&read("src/bind/facts.rs"));
    let binds = strip_line_comments(&read("src/bind/mod.rs"));
    let cx = strip_line_comments(&read("src/bind/cx.rs"));

    // [`delivered`]'s premise, checked rather than assumed: the whole
    // struct crosses to an arm for exactly one of the three, and a `Cx`
    // that learned to hand over another would make half this test's
    // reasoning silently too generous.
    assert!(
        cx.contains("AttnCtx"),
        "`Cx` no longer hands an `AttnCtx` over, so `delivered` is \
         excusing that struct's literals for a reason that has expired",
    );
    for whole in ["DispatchCtx", "GdnCtx"] {
        assert!(
            !cx.contains(whole),
            "`Cx` names `{whole}`. If it hands the struct over the way it \
             hands over `AttnCtx`, then every field of it is reachable and \
             `delivered` must say so — it currently asks `bind/facts.rs` \
             instead, which would be reading the wrong seam.",
        );
    }

    let named = deliverable_names(&facts, &binds);
    let argued: BTreeSet<&str> = CONSTANT_BY_ARGUMENT.iter().map(|(n, _)| *n).collect();
    let deferred: BTreeSet<&str> = VARIED_BY_A_ROW_WITH_NO_TEXT.iter().map(|(n, ..)| *n).collect();
    let inert: BTreeSet<&str> = NOTHING_DELIVERS_IT.iter().map(|(n, _)| *n).collect();

    let mut unstated = BTreeSet::new();
    let mut too_weak = BTreeSet::new();
    let mut too_strong = BTreeSet::new();
    for fill in context_fills(&launch) {
        if !is_literal(&fill.rhs) {
            continue;
        }
        let name = fill.name.as_str();
        let excused = argued.contains(name) || deferred.contains(name) || inert.contains(name);
        let one = format!("{}::{name} = {}", fill.owner, fill.rhs);
        if !excused {
            unstated.insert(one);
        } else if delivered(&fill, &facts, &named) {
            if inert.contains(name) {
                too_weak.insert(one);
            }
        } else if !inert.contains(name) {
            too_strong.insert(one);
        }
    }

    assert!(
        too_weak.is_empty(),
        "these are excused by `NOTHING_DELIVERS_IT`, and something \
         delivers them now:\n  {}\n\nThe literal is live. Read the field \
         off `model.deployment` — which is where `eps`, `vocab`, \
         `moe_norm_topk` and the rest of that struct come from — or move \
         the name to `CONSTANT_BY_ARGUMENT` with the reason no checkpoint \
         varies it.",
        too_weak.into_iter().collect::<Vec<_>>().join("\n  "),
    );
    assert!(
        too_strong.is_empty(),
        "these argue about a field no arm can be handed:\n  {}\n\nThat is \
         a weaker situation than the list they are in claims, and the \
         weaker claim reads like the stronger one. Move them to \
         `NOTHING_DELIVERS_IT` with what stops the value crossing.",
        too_strong.into_iter().collect::<Vec<_>>().join("\n  "),
    );
    assert!(
        unstated.is_empty(),
        "a bind arm reads these off the launch context and the launch \
         invents them:\n  {}\n\nEither read the field off \
         `model.deployment` — which is where `eps`, `vocab`, \
         `moe_norm_topk` and the rest of this struct come from — or add \
         the name to `CONSTANT_BY_ARGUMENT` with the reason no checkpoint \
         varies it, or, if one DOES and merely cannot dispatch yet, to \
         `VARIED_BY_A_ROW_WITH_NO_TEXT` with the row and the fact that \
         holds it back, or, if nothing can reach it, to \
         `NOTHING_DELIVERS_IT` with what stops it.",
        unstated.into_iter().collect::<Vec<_>>().join("\n  "),
    );
}

/// No list outlives the literal it excuses.
///
/// An excuse for a field the launch has stopped inventing is an argument
/// about nothing, which reads, to the next person, as a live constant
/// somebody thought about. Both directions matter: the field may be gone,
/// or it may be read off the model now — and the second is a line someone
/// should delete while they still remember writing it.
#[test]
fn nothing_is_argued_for_that_the_launch_no_longer_invents() {
    let launch = strip_line_comments(&read("src/fire/launch.rs"));
    let literal: BTreeSet<String> = context_fills(&launch)
        .into_iter()
        .filter(|f| is_literal(&f.rhs))
        .map(|f| f.name)
        .collect();

    let listed: Vec<&str> = CONSTANT_BY_ARGUMENT
        .iter()
        .map(|(n, _)| *n)
        .chain(VARIED_BY_A_ROW_WITH_NO_TEXT.iter().map(|(n, ..)| *n))
        .chain(NOTHING_DELIVERS_IT.iter().map(|(n, _)| *n))
        .collect();

    // One name, one list. Two entries for a field is two claims about it,
    // and the test would then check whichever it reached first — which is
    // the failure the three lists exist to prevent, spelled as a typo.
    let mut once = BTreeSet::new();
    let twice: Vec<&&str> = listed.iter().filter(|n| !once.insert(**n)).collect();
    assert!(twice.is_empty(), "these are excused by more than one list: {twice:?}");

    let stale: Vec<&str> = listed.into_iter().filter(|n| !literal.contains(*n)).collect();
    assert!(
        stale.is_empty(),
        "the launch does not invent these any more: {stale:?}. Delete the \
         entry — or, if the field was renamed, rename it here too, because \
         an excuse that names nothing excuses nothing.",
    );
}

/// A deferred literal stops being excused the day its row gets a text.
///
/// The premise is not restated here, it is READ: `model`'s own
/// `facts_are_read` keeps `DECLARED_BUT_UNREAD`, the list of per-backend
/// facts a family declares and no text consumes. While `kimi_k2`'s
/// `rope_yarn_original` sits on it, nothing dispatches the kernel that
/// reads `rope_interleaved` with a K2 ladder. Writing that text means
/// deleting the entry there, which fails this test HERE — which is the
/// point, because the person writing an MLA rope pass is exactly the
/// person who must decide what `rope_interleaved` is.
///
/// The excuses this crate's siblings deleted named a future reader their
/// own layering ruled out, so they could never expire. This one names a
/// line someone has to delete to finish the work it defers.
#[test]
fn the_deferred_names_are_still_out_of_reach() {
    let unread = read("../model/tests/facts_are_read.rs");
    let table = unread
        .find("DECLARED_BUT_UNREAD")
        .map(|i| &unread[i..])
        .expect("`model` still keeps a `DECLARED_BUT_UNREAD` list");

    for (name, family, fact, _) in VARIED_BY_A_ROW_WITH_NO_TEXT {
        let entry = table
            .find(&format!("(\"{family}\""))
            .map(|i| &table[i..])
            .and_then(|e| e.find("])").map(|j| &e[..j]));
        assert!(
            entry.is_some_and(|e| e.contains(&format!("\"{fact}\""))),
            "`{name}` is a literal in `fire::launch` because {family} \
             cannot dispatch the kernel that reads it, and the proof of \
             that was `{fact}` sitting in `DECLARED_BUT_UNREAD`. It is \
             not there now. Either a text reads it — in which case fill \
             `{name}` off the model, because that text's checkpoint \
             varies it — or the fact was deleted, in which case say so \
             here.",
        );
    }
}

