
use std::path::Path;

struct Golden {
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
        Some(Self { body })
    }

    fn expect(path: &Path, advice: &str) -> Self {
        Self::read(path).unwrap_or_else(|| panic!("{} missing; {advice}", path.display()))
    }
}

pub enum Regenerate<'a> {
    Own { header: &'a str },
}

pub fn body_to_diff(path: &Path, body: &str, how: Regenerate) -> Option<String> {
    const MISSING: &str = "re-pin it with PTIR_REGEN=1";
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
        }
        return None;
    }
    let golden = Golden::expect(path, MISSING);
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
