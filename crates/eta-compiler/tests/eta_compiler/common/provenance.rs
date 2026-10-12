#![allow(dead_code)]

use std::path::Path;

const STAMP: &str = "# REGENERATED: the body below is this compiler's output, not the recorded output of the source of truth above";
const REASON_PREFIX: &str = "#   reason: ";

fn regenerate_foreign(path: &Path, header: &str, body: &str) {
    if let Some(existing) = Golden::read(path)
        && existing.body == body
    {
        return;
    }
    let reason = std::env::var("PTIR_REGEN_REASON").unwrap_or_default();
    let reason = reason.trim();
    assert!(
        !reason.is_empty(),
        "{} records the output of a source of truth that no longer exists, so \
         overwriting it discards evidence that cannot be recovered. Set \
         PTIR_REGEN_REASON=\"why the new bytes are correct\" to regenerate it \
         anyway; the reason is written into the file.",
        path.display()
    );
    let mut lines: Vec<String> = header
        .lines()
        .filter(|line| *line != STAMP && !line.starts_with(REASON_PREFIX))
        .map(str::to_string)
        .collect();
    lines.push(STAMP.to_string());
    lines.push(format!("{REASON_PREFIX}{reason}"));
    let text = lines.join("\n") + "\n" + body;
    std::fs::write(path, text)
        .unwrap_or_else(|error| panic!("{} could not be rewritten ({error})", path.display()));
}

struct Golden {
    header: String,
    body: String,
}

impl Golden {
    fn read(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let header: String = text
            .lines()
            .take_while(|line| line.starts_with('#'))
            .map(|line| format!("{line}\n"))
            .collect();
        let body = text[header.len()..].to_string();
        Some(Self { header, body })
    }

    fn expect(path: &Path, advice: &str) -> Self {
        Self::read(path).unwrap_or_else(|| panic!("{} missing; {advice}", path.display()))
    }
}

pub enum Regenerate<'a> {
    Own { header: &'a str },
    Foreign,
}

pub fn body_to_diff(path: &Path, body: &str, how: Regenerate) -> Option<String> {
    const MISSING: &str = "re-pin it with PTIR_REGEN=1";
    const MISSING_FOREIGN: &str =
        "it records a source of truth that is gone, so it cannot simply be re-pinned";
    if std::env::var("PTIR_REGEN").is_ok() {
        match how {
            Regenerate::Own { header } => {
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir)
                        .unwrap_or_else(|e| panic!("{} could not be created ({e})", dir.display()));
                }
                std::fs::write(path, format!("{header}{body}"))
                    .unwrap_or_else(|e| panic!("{} could not be written ({e})", path.display()));
            }
            Regenerate::Foreign => {
                let golden = Golden::expect(path, MISSING_FOREIGN);
                regenerate_foreign(path, &golden.header, body);
            }
        }
        return None;
    }
    let golden = Golden::expect(
        path,
        match how {
            Regenerate::Own { .. } => MISSING,
            Regenerate::Foreign => MISSING_FOREIGN,
        },
    );
    (golden.body != body).then_some(golden.body)
}

pub fn assert_same_lines(mine: &str, theirs: &str, what: &str, hint: &str) {
    let mut case = String::from("<before the first case>");
    for (index, (mine, theirs)) in mine.lines().zip(theirs.lines()).enumerate() {
        if let Some(id) = theirs.strip_prefix("=== ") {
            case = id.to_string();
        }
        assert_eq!(
            mine,
            theirs,
            "{what} diverged at line {} (case `{case}`){hint}",
            index + 1
        );
    }
    assert_eq!(
        mine.lines().count(),
        theirs.lines().count(),
        "{what} produced a different number of lines"
    );
}
