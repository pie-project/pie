use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostics {
    pub golden_probe: bool,
    pub golden_skip: bool,
    pub arm_trace: bool,
    pub ptr_trace: Option<String>,
    pub graph_dot: Option<PathBuf>,
    pub grid_trace: Option<String>,
    pub plan_trace: Option<String>,
    pub capture_serial: bool,
    pub boundary_trace: bool,
    pub reap_trace: bool,
    pub trace_census: bool,
    pub nan_check: bool,
    pub fuse_chains: bool,
    pub gumbel_direct: bool,
}

impl Default for Diagnostics {
    fn default() -> Diagnostics {
        Diagnostics {
            golden_probe: false,
            golden_skip: false,
            arm_trace: false,
            ptr_trace: None,
            graph_dot: None,
            grid_trace: None,
            plan_trace: None,
            capture_serial: false,
            boundary_trace: false,
            reap_trace: false,
            trace_census: false,
            nan_check: false,
            fuse_chains: true,
            gumbel_direct: true,
        }
    }
}

impl Diagnostics {
    #[must_use]
    pub fn any(&self) -> bool {
        *self != Diagnostics::default()
    }

    #[must_use]
    pub fn words(&self) -> Option<String> {
        let mut words: Vec<String> = Vec::new();
        for (word, on) in [
            ("golden-probe", self.golden_probe),
            ("golden-skip", self.golden_skip),
            ("arm-trace", self.arm_trace),
            ("capture-serial", self.capture_serial),
            ("boundary-trace", self.boundary_trace),
            ("reap-trace", self.reap_trace),
            ("trace-census", self.trace_census),
            ("nan-check", self.nan_check),
        ] {
            if on {
                words.push(word.to_string());
            }
        }
        for (word, value) in [
            ("ptr-trace", self.ptr_trace.clone()),
            ("grid-trace", self.grid_trace.clone()),
            ("plan-trace", self.plan_trace.clone()),
            (
                "graph-dot",
                self.graph_dot.as_ref().map(|dir| dir.display().to_string()),
            ),
        ] {
            if let Some(value) = value {
                words.push(format!("{word}={value}"));
            }
        }
        if !self.fuse_chains {
            words.push("fuse-chains=off".to_string());
        }
        if !self.gumbel_direct {
            words.push("gumbel-direct=off".to_string());
        }
        (!words.is_empty()).then(|| words.join(","))
    }
}

fn switch(word: &str, value: &str) -> std::result::Result<bool, String> {
    match value {
        "" | "on" | "1" | "true" | "yes" => Ok(true),
        "off" | "0" | "false" | "no" => Ok(false),
        other => Err(format!(
            "`{word}={other}` is not a switch; write `{word}` or `{word}=off`"
        )),
    }
}

const WORDS: &str = "`golden-probe`, `golden-skip`, `arm-trace`, \
     `ptr-trace=<key substring>`, `graph-dot=<dir>`, `grid-trace=<key substring>`, \
     `plan-trace=<rows|all>`, `capture-serial`, `boundary-trace`, `reap-trace`, \
     `trace-census`, `nan-check`, `fuse-chains=off`, `gumbel-direct=off`";

impl std::str::FromStr for Diagnostics {
    type Err = String;

    fn from_str(list: &str) -> std::result::Result<Diagnostics, String> {
        let mut diag = Diagnostics::default();
        for term in list.split(',') {
            let term = term.trim();
            if term.is_empty() {
                continue;
            }
            let (word, value) = match term.split_once('=') {
                Some((word, value)) => (word.trim(), value.trim()),
                None => (term, ""),
            };
            match word {
                "golden-probe" => diag.golden_probe = switch(word, value)?,
                "golden-skip" => diag.golden_skip = switch(word, value)?,
                "arm-trace" => diag.arm_trace = switch(word, value)?,
                "capture-serial" => diag.capture_serial = switch(word, value)?,
                "boundary-trace" => diag.boundary_trace = switch(word, value)?,
                "reap-trace" => diag.reap_trace = switch(word, value)?,
                "trace-census" => diag.trace_census = switch(word, value)?,
                "nan-check" => diag.nan_check = switch(word, value)?,
                "fuse-chains" => diag.fuse_chains = switch(word, value)?,
                "gumbel-direct" => diag.gumbel_direct = switch(word, value)?,
                "ptr-trace" | "grid-trace" | "plan-trace" | "graph-dot" => {
                    if value.is_empty() {
                        return Err(format!(
                            "`{word}` takes a value: `{word}=<{}>`",
                            match word {
                                "graph-dot" => "directory",
                                "plan-trace" => "rows|all",
                                _ => "substring of a body key",
                            }
                        ));
                    }
                    match word {
                        "ptr-trace" => diag.ptr_trace = Some(value.to_string()),
                        "grid-trace" => diag.grid_trace = Some(value.to_string()),
                        "plan-trace" => diag.plan_trace = Some(value.to_string()),
                        _ => diag.graph_dot = Some(PathBuf::from(value)),
                    }
                }
                other => {
                    return Err(format!(
                        "`{other}` does not name a diagnostic; the words are {WORDS}"
                    ));
                }
            }
        }
        Ok(diag)
    }
}

static PUBLISHED: OnceLock<Diagnostics> = OnceLock::new();

pub(crate) fn publish(stated: &Diagnostics) {
    let published = PUBLISHED.get_or_init(|| stated.clone());
    if published != stated {
        eprintln!(
            "engine-cuda: this process already published a diagnostics record \
             ({}), so the one this boot states ({}) is ignored; every rank of a \
             group traces the same thing",
            published.words().unwrap_or_else(|| "none".to_string()),
            stated.words().unwrap_or_else(|| "none".to_string()),
        );
    }
}

#[must_use]
pub fn on() -> &'static Diagnostics {
    static OFF: OnceLock<Diagnostics> = OnceLock::new();
    PUBLISHED
        .get()
        .unwrap_or_else(|| OFF.get_or_init(Diagnostics::default))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diag_every_case() {
        an_empty_list_turns_nothing_on();
        a_word_list_names_each_knob();
        an_unknown_word_refuses_by_name();
        a_valued_word_with_no_value_refuses();
        the_words_round_trip();
    }

    fn an_empty_list_turns_nothing_on() {
        let diag: Diagnostics = "".parse().expect("silence parses");
        assert_eq!(diag, Diagnostics::default());
        assert!(!diag.any());
        assert_eq!(diag.words(), None);
    }

    fn a_word_list_names_each_knob() {
        let diag: Diagnostics = "golden-probe, ptr-trace=decode:c0 ,fuse-chains=off"
            .parse()
            .expect("the three words parse");
        assert!(diag.golden_probe);
        assert_eq!(diag.ptr_trace.as_deref(), Some("decode:c0"));
        assert!(!diag.fuse_chains);
        assert!(diag.gumbel_direct, "an unnamed arm keeps its default");
    }

    fn an_unknown_word_refuses_by_name() {
        let why = "golden-probes".parse::<Diagnostics>().expect_err("refused");
        assert!(why.contains("golden-probes"), "{why}");
        assert!(
            why.contains("golden-probe`"),
            "and lists the vocabulary: {why}"
        );
    }

    fn a_valued_word_with_no_value_refuses() {
        assert!("ptr-trace".parse::<Diagnostics>().is_err());
        assert!("graph-dot=".parse::<Diagnostics>().is_err());
    }

    fn the_words_round_trip() {
        let list = "arm-trace,capture-serial,plan-trace=all,gumbel-direct=off";
        let diag: Diagnostics = list.parse().expect("parses");
        let again: Diagnostics = diag
            .words()
            .expect("something is on")
            .parse()
            .expect("its own spelling parses");
        assert_eq!(diag, again);
    }
}
