use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    #[default]
    Off,
    On,
    Shaped,
}

impl Profile {
    #[must_use]
    pub fn on(self) -> bool {
        !matches!(self, Profile::Off)
    }

    #[must_use]
    pub fn shaped(self) -> bool {
        matches!(self, Profile::Shaped)
    }

    #[must_use]
    pub fn rows(self) -> usize {
        match self {
            Profile::Shaped => 60,
            Profile::Off | Profile::On => 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamedKind {
    Wide,
    Partial,
    Single,
    Reduce,
    Argmax,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    pub cut_trace: bool,
    pub tier_trace: bool,
    pub rs_trace: bool,
    pub fire_trace: bool,
    pub region_trace: bool,
    pub streamed_trace: bool,
    pub kernel_profile: Profile,
    pub kernel_dump: Option<PathBuf>,
    pub route_dump: Option<PathBuf>,
    pub pass_half: bool,
    pub route_prefetch: bool,
    pub expert_passes: bool,
    pub keepalive: bool,
    pub scratch_no_zero: bool,
    pub host_rows: bool,
    pub nan_check: bool,

    pub nan_limit: f32,
    pub prefault: bool,
    pub copy_resident: bool,
    pub prefetch_k: Option<usize>,
    pub seat_threads: Option<usize>,
    pub keepalive_iters: Option<u32>,
    pub window_ceiling: Option<u64>,
    pub streamed_groups: Option<u32>,
    pub streamed_repeat: usize,
    pub streamed_repeat_kind: Option<StreamedKind>,
    pub streamed_limit: Option<usize>,
    pub streamed_nop: usize,
}

impl Default for Diagnostics {
    fn default() -> Diagnostics {
        Diagnostics {
            cut_trace: false,
            tier_trace: false,
            rs_trace: false,
            fire_trace: false,
            region_trace: false,
            streamed_trace: false,
            kernel_profile: Profile::Off,
            kernel_dump: None,
            route_dump: None,
            pass_half: true,
            route_prefetch: true,
            expert_passes: true,
            keepalive: true,
            scratch_no_zero: false,
            host_rows: false,
            nan_check: false,
            nan_limit: 3.0e38,
            prefault: false,
            copy_resident: false,
            prefetch_k: None,
            seat_threads: None,
            keepalive_iters: None,
            window_ceiling: None,
            streamed_groups: None,
            streamed_repeat: 1,
            streamed_repeat_kind: None,
            streamed_limit: None,
            streamed_nop: 0,
        }
    }
}

impl Diagnostics {
    #[must_use]
    pub fn any(&self) -> bool {
        *self != Diagnostics::default()
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

fn number<T: std::str::FromStr>(word: &str, value: &str) -> std::result::Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("`{word}` takes a number; `{value}` is not one"))
}

const WORDS: &str = "`cut-trace`, `tier-trace`, `rs-trace`, `fire-trace`, \
     `region-trace`, `streamed-trace`, `kernel-profile[=1|2]`, \
     `nan-check`, `nan-limit=<float>`, `kernel-dump=<dir>`, `route-dump=<file>`, `pass-half=off`, \
     `route-prefetch=off`, `expert-passes=off`, `keepalive=off`, \
     `scratch-no-zero`, `host-rows`, `prefault`, `copy-resident`, \
     `prefetch-k=<n>`, `seat-threads=<n>`, `keepalive-iters=<n>`, \
     `window-ceiling=<bytes>`, `streamed-groups=<n>`, `streamed-repeat=<n>`, \
     `streamed-repeat-kind=<wide|partial|single|reduce|argmax>`, \
     `streamed-limit=<n>`, `streamed-nop=<n>`";

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
                "cut-trace" => diag.cut_trace = switch(word, value)?,
                "tier-trace" => diag.tier_trace = switch(word, value)?,
                "rs-trace" => diag.rs_trace = switch(word, value)?,
                "fire-trace" => diag.fire_trace = switch(word, value)?,
                "region-trace" => diag.region_trace = switch(word, value)?,
                "streamed-trace" => diag.streamed_trace = switch(word, value)?,
                "pass-half" => diag.pass_half = switch(word, value)?,
                "route-prefetch" => diag.route_prefetch = switch(word, value)?,
                "expert-passes" => diag.expert_passes = switch(word, value)?,
                "keepalive" => diag.keepalive = switch(word, value)?,
                "scratch-no-zero" => diag.scratch_no_zero = switch(word, value)?,
                "host-rows" => diag.host_rows = switch(word, value)?,
                "nan-check" => diag.nan_check = switch(word, value)?,
                "nan-limit" => {
                    diag.nan_limit = value.parse::<f32>().map_err(|_| {
                        format!("`nan-limit` takes a float magnitude; `{value}` is not one")
                    })?;
                }
                "prefault" => diag.prefault = switch(word, value)?,
                "copy-resident" => diag.copy_resident = switch(word, value)?,
                "kernel-profile" => {
                    diag.kernel_profile = match value {
                        "" | "1" | "on" | "true" | "yes" => Profile::On,
                        "2" | "shaped" => Profile::Shaped,
                        "off" | "0" | "false" | "no" => Profile::Off,
                        other => {
                            return Err(format!(
                                "`kernel-profile={other}` is not a mode; write \
                                 `kernel-profile` (per entrypoint), \
                                 `kernel-profile=2` (per entrypoint and shape), \
                                 or `kernel-profile=off`"
                            ));
                        }
                    };
                }
                "streamed-repeat-kind" => {
                    diag.streamed_repeat_kind = Some(match value {
                        "wide" => StreamedKind::Wide,
                        "partial" => StreamedKind::Partial,
                        "single" => StreamedKind::Single,
                        "reduce" => StreamedKind::Reduce,
                        "argmax" => StreamedKind::Argmax,
                        other => {
                            return Err(format!(
                                "`streamed-repeat-kind={other}` does not name a step \
                                 kind; the spellings are `wide`, `partial`, \
                                 `single`, `reduce`, `argmax`"
                            ));
                        }
                    });
                }
                "kernel-dump" | "route-dump" => {
                    if value.is_empty() {
                        return Err(format!("`{word}` takes a path: `{word}=<path>`"));
                    }
                    match word {
                        "kernel-dump" => diag.kernel_dump = Some(PathBuf::from(value)),
                        _ => diag.route_dump = Some(PathBuf::from(value)),
                    }
                }
                "prefetch-k" => diag.prefetch_k = Some(number(word, value)?),
                "seat-threads" => {
                    let threads: usize = number(word, value)?;
                    if threads == 0 {
                        return Err("`seat-threads` must be > 0".to_string());
                    }
                    diag.seat_threads = Some(threads);
                }
                "keepalive-iters" => diag.keepalive_iters = Some(number(word, value)?),
                "window-ceiling" => {
                    let bytes: u64 = number(word, value)?;
                    if bytes == 0 {
                        return Err(
                            "`window-ceiling` must be > 0; omit it for the device's own \
                             `maxBufferLength`"
                                .to_string(),
                        );
                    }
                    diag.window_ceiling = Some(bytes);
                }
                "streamed-groups" => {
                    let groups: u32 = number(word, value)?;
                    if groups == 0 {
                        return Err("`streamed-groups` must be > 0".to_string());
                    }
                    diag.streamed_groups = Some(groups);
                }
                "streamed-repeat" => diag.streamed_repeat = number::<usize>(word, value)?.max(1),
                "streamed-limit" => diag.streamed_limit = Some(number(word, value)?),
                "streamed-nop" => diag.streamed_nop = number(word, value)?,
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
            "engine-metal: this process already published a diagnostics record, \
             so the different one this boot states is ignored"
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
        a_number_that_is_not_one_refuses();
        the_streamed_kind_takes_one_of_five_words();
    }

    fn an_empty_list_turns_nothing_on() {
        let diag: Diagnostics = "".parse().expect("silence parses");
        assert_eq!(diag, Diagnostics::default());
        assert!(!diag.any());
    }

    fn a_word_list_names_each_knob() {
        let diag: Diagnostics = "tier-trace, kernel-profile=2 ,pass-half=off,streamed-nop=3"
            .parse()
            .expect("the four words parse");
        assert!(diag.tier_trace);
        assert_eq!(diag.kernel_profile, Profile::Shaped);
        assert!(diag.kernel_profile.shaped());
        assert_eq!(diag.kernel_profile.rows(), 60);
        assert!(!diag.pass_half);
        assert_eq!(diag.streamed_nop, 3);
        assert!(diag.keepalive, "an unnamed arm keeps its default");
    }

    fn an_unknown_word_refuses_by_name() {
        let why = "tier-traces".parse::<Diagnostics>().expect_err("refused");
        assert!(why.contains("tier-traces"), "{why}");
        assert!(
            why.contains("tier-trace`"),
            "and lists the vocabulary: {why}"
        );
    }

    fn a_number_that_is_not_one_refuses() {
        assert!("prefetch-k=many".parse::<Diagnostics>().is_err());
        assert!("seat-threads=0".parse::<Diagnostics>().is_err());
        assert!("window-ceiling=0".parse::<Diagnostics>().is_err());
    }

    fn the_streamed_kind_takes_one_of_five_words() {
        let diag: Diagnostics = "streamed-repeat=4,streamed-repeat-kind=reduce"
            .parse()
            .expect("parses");
        assert_eq!(diag.streamed_repeat, 4);
        assert_eq!(diag.streamed_repeat_kind, Some(StreamedKind::Reduce));
        assert!(
            "streamed-repeat-kind=narrow"
                .parse::<Diagnostics>()
                .is_err()
        );
    }
}
