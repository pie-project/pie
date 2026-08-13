//! Compiles the Qwen tool-call grammars this branch emits, so a syntax error in
//! them is found here rather than on a rented GPU.
use pie_grammar::grammar::Grammar;

fn check(label: &str, source: &str) -> bool {
    match Grammar::from_ebnf(source, "root") {
        Ok(_) => {
            println!("PASS  {label}");
            true
        }
        Err(error) => {
            println!("FAIL  {label}: {error}");
            false
        }
    }
}

fn main() {
    let xml = r#"root ::= reasoning-block? tool-call ("\n" tool-call)*
reasoning-block ::= "<think>" reasoning-content "</think>" "\n"*
reasoning-content ::= reasoning-piece*
reasoning-piece ::= [^<] | "<" [^/] | "</" [^t] | "</t" [^h] | "</th" [^i] | "</thi" [^n] | "</thin" [^k] | "</think" [^>]
tool-call ::= "<tool_call>\n<function=" tool-name ">\n" parameter* "</function>\n</tool_call>"
tool-name ::= "bash"
parameter ::= "<parameter=" parameter-name ">\n" parameter-value "\n</parameter>\n"
parameter-name ::= [A-Za-z_][A-Za-z0-9_-]*
parameter-value ::= parameter-char*
parameter-char ::= [^<]
"#;
    let json = r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
tool-json ::= "{"  "\"name\": \"" tool-name "\", \"arguments\": " json-object "}"
tool-name ::= "bash"
json-object ::= "{" json-members? "}"
json-members ::= json-pair ("," json-pair)*
json-pair ::= json-string ":" json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" (json-value ("," json-value)*)? "]"
"#;
    let ok = check("qwen3.5-xml + reasoning prefix (new)", xml)
        & check("qwen3-json, no prefix (pre-existing)", json);
    if !ok {
        std::process::exit(1);
    }
}
