use gpugrammar_ir::grammar::Grammar;
use gpugrammar_lex::lexicon::{Lexicon, extract};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lr::cfg::{Cfg, Symbol, flatten};
use gpugrammar_lr::parser::Parse;
use gpugrammar_lr::tables::{Tables, build};

fn compile(source: &str, root: &str) -> (Lexicon, Cfg, Tables) {
    let grammar = Grammar::from_ebnf(source, root).unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let cfg = flatten(&lexicon);
    let tables = build(&cfg).expect("grammar should be LALR(1)");
    (lexicon, cfg, tables)
}

fn terminal(lexicon: &Lexicon, name: &str) -> Symbol {
    let index = lexicon
        .terminals
        .iter()
        .position(|terminal| terminal.name == name)
        .unwrap_or_else(|| {
            panic!(
                "no terminal {name}, have {:?}",
                lexicon
                    .terminals
                    .iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>()
            )
        });
    Symbol::Terminal(gpugrammar_lex::TerminalId(index as u32))
}

const JSON: &str = r#"
value ::= object | array | string | number
object ::= "{" "}" | "{" members "}"
members ::= pair | pair "," members
pair ::= string ":" value
array ::= "[" "]" | "[" items "]"
items ::= value | value "," items
string ::= "\"" chars
chars ::= "\"" | [^"\\] chars
number ::= "-"? [0-9]+
"#;

#[test]
fn a_json_skeleton_is_lalr1() {
    let (_, cfg, tables) = compile(JSON, "value");
    assert!(tables.num_states() > 1);
    assert_eq!(tables.num_terminals, cfg.num_terminals);
}

#[test]
fn the_tables_accept_well_formed_sentences() {
    let (lexicon, _, tables) = compile(JSON, "value");
    let lb = terminal(&lexicon, "'{'");
    let rb = terminal(&lexicon, "'}'");
    let lk = terminal(&lexicon, "'['");
    let rk = terminal(&lexicon, "']'");
    let comma = terminal(&lexicon, "','");
    let colon = terminal(&lexicon, "':'");
    let string = terminal(&lexicon, "string");
    let number = terminal(&lexicon, "number");
    // `object ::= "{" "}"` is regular in full, so the empty object is one
    // terminal rather than two. The same holds for the empty array.
    let empty_object = terminal(&lexicon, "(. b{ b})");
    let empty_array = terminal(&lexicon, "(. b[ b])");

    assert!(Parse::accepts(&tables, &[empty_object]));
    assert!(Parse::accepts(&tables, &[lb, string, colon, number, rb]));
    assert!(Parse::accepts(
        &tables,
        &[lb, string, colon, number, comma, string, colon, string, rb]
    ));
    assert!(Parse::accepts(&tables, &[empty_array]));
    assert!(Parse::accepts(&tables, &[lk, number, comma, number, rk]));
    assert!(Parse::accepts(
        &tables,
        &[lb, string, colon, lk, empty_object, rk, rb]
    ));
}

#[test]
fn the_tables_reject_malformed_sentences() {
    let (lexicon, _, tables) = compile(JSON, "value");
    let lb = terminal(&lexicon, "'{'");
    let rb = terminal(&lexicon, "'}'");
    let comma = terminal(&lexicon, "','");
    let colon = terminal(&lexicon, "':'");
    let string = terminal(&lexicon, "string");
    let number = terminal(&lexicon, "number");

    assert!(!Parse::accepts(&tables, &[lb]));
    assert!(!Parse::accepts(&tables, &[rb]));
    assert!(!Parse::accepts(&tables, &[lb, comma, rb]));
    assert!(!Parse::accepts(&tables, &[lb, number, colon, number, rb]));
    assert!(!Parse::accepts(&tables, &[lb, string, colon, rb]));
}

#[test]
fn admissibility_follows_from_the_stack_top() {
    // The premise of the whole runtime: what may come next is a property of
    // the top state, so one lookup per token group serves the whole batch.
    let (lexicon, _, tables) = compile(JSON, "value");
    let lb = terminal(&lexicon, "'{'");
    let string = terminal(&lexicon, "string");

    let mut parse = Parse::new(&tables);
    let at_start = parse.admissible();
    assert!(at_start.contains(&match lb {
        Symbol::Terminal(t) => t.0,
        _ => unreachable!(),
    }));

    let Symbol::Terminal(open) = lb else {
        unreachable!()
    };
    parse.feed(open.0);
    let inside = parse.admissible();
    let Symbol::Terminal(key) = string else {
        unreachable!()
    };
    assert!(inside.contains(&key.0));
    assert_ne!(at_start, inside);
}

#[test]
fn an_ambiguous_grammar_is_reported_not_silently_resolved() {
    let grammar = Grammar::from_ebnf(
        r#"
stmt ::= "i" stmt | "i" stmt "e" stmt | "x"
"#,
        "stmt",
    )
    .unwrap();
    let lexicon = extract(&grammar, &analyze(&grammar));
    let cfg = flatten(&lexicon);
    let error = build(&cfg).unwrap_err().to_string();
    assert!(error.contains("not LALR(1)"), "unexpected error: {error}");
}

#[test]
fn a_recursive_grammar_keeps_the_table_small() {
    let (_, cfg, tables) = compile(JSON, "value");
    // Canonical LR(1) multiplies states by lookahead; merging keeps the table
    // proportional to the LR(0) automaton.
    assert!(
        tables.num_states() < cfg.productions.len() * 8,
        "{} states for {} productions",
        tables.num_states(),
        cfg.productions.len()
    );
}
