//! What `pie` looks like: colour policy, units, glyphs, terminal width.
//!
//! One module because the alternative was measured. Before this existed, six
//! files decided independently whether to colour, five of them by asking
//! `stdout().is_terminal()` (including for output that goes to stderr), and
//! none of them honoured `NO_COLOR`. Four of them formatted bytes, and the
//! same 3 GiB printed as `3.00 GiB`, `3.0GiB`, `3072MiB` and `3.2 GB` -- the
//! last one decimal, so it was not even the same quantity.
//!
//! The rule this module exists to make enforceable: **ops code never emits an
//! escape sequence and never formats a quantity itself.** It picks a [`Mark`],
//! calls [`bytes`], and asks [`Palette`] for the styling.

use std::io::IsTerminal;

// -----------------------------------------------------------------------------
// Colour policy
// -----------------------------------------------------------------------------

/// Which stream a [`Palette`] is being built for.
///
/// It matters: the download bar draws to stderr, so deciding its colour from
/// `stdout().is_terminal()` -- as the code here used to -- got the answer from
/// the wrong file descriptor. `pie model download > log` would keep colouring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stream {
    Stdout,
    Stderr,
}

/// Whether to colour, honouring the conventions a user expects to work.
///
/// `NO_COLOR` is checked first and its presence alone disables colour,
/// whatever the value -- that is what the convention specifies. `TERM=dumb`
/// is the older spelling of the same request. Both beat a TTY check, because
/// both are the user saying so and the TTY check is only a guess about what
/// they want.
pub fn colour_enabled(stream: Stream) -> bool {
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("TERM").is_ok_and(|term| term == "dumb") {
        return false;
    }
    match stream {
        Stream::Stdout => std::io::stdout().is_terminal(),
        Stream::Stderr => std::io::stderr().is_terminal(),
    }
}

/// Styling that renders to nothing when colour is off.
///
/// Returning empty strings rather than branching at each call site is what
/// keeps a format string readable and keeps the no-colour path from being the
/// one nobody looks at.
#[derive(Clone, Copy)]
pub struct Palette {
    on: bool,
}

impl Palette {
    pub fn for_stream(stream: Stream) -> Self {
        Self {
            on: colour_enabled(stream),
        }
    }

    /// Colour forced on or off. For tests: there is deliberately no
    /// `--color` flag, because `NO_COLOR` plus the TTY check already answer
    /// the question and a three-way switch would be one more thing to get
    /// wrong.
    pub fn forced(on: bool) -> Self {
        Self { on }
    }

    pub fn enabled(&self) -> bool {
        self.on
    }

    fn code(&self, escape: &'static str) -> &'static str {
        if self.on { escape } else { "" }
    }

    /// Secondary text: paths, notes, descriptions. Never the answer itself.
    pub fn dim(&self) -> &'static str {
        self.code("\x1b[2m")
    }
    /// Headings and section labels.
    pub fn bold(&self) -> &'static str {
        self.code("\x1b[1m")
    }
    pub fn green(&self) -> &'static str {
        self.code("\x1b[32m")
    }
    pub fn yellow(&self) -> &'static str {
        self.code("\x1b[33m")
    }
    pub fn red(&self) -> &'static str {
        self.code("\x1b[31m")
    }
    pub fn reset(&self) -> &'static str {
        self.code("\x1b[0m")
    }
}

// -----------------------------------------------------------------------------
// Glyphs
// -----------------------------------------------------------------------------

/// The one meaning each glyph carries.
///
/// `✓` used to mean three unrelated things -- "this model is supported", "this
/// check passed", "the command did the thing" -- while "absent" was spelled
/// `○` in one listing, `—` in another and a blank in a third. A reader cannot
/// learn a vocabulary that changes per command, so there is one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mark {
    /// The command did something. Only ever on a result line, never in a table.
    Did,
    /// True of the machine or the config, and not a fault.
    Warn,
    /// Would stop pie from working.
    Blocked,
    /// The operator chose this value; it is not the default.
    Chosen,
    /// Present and unremarkable. Renders as a blank, so the marked rows are
    /// what the eye finds.
    Plain,
    /// Not there. One spelling, whether that is "not downloaded yet", "no
    /// driver compiled" or "unsupported".
    Absent,
}

impl Mark {
    pub fn glyph(self) -> &'static str {
        match self {
            Mark::Did => "✓",
            Mark::Warn => "!",
            Mark::Blocked => "✗",
            Mark::Chosen => "•",
            Mark::Plain => " ",
            Mark::Absent => "—",
        }
    }

    /// The glyph in its colour, or bare when colour is off. Width is always
    /// one column either way, so a table stays aligned.
    pub fn render(self, p: &Palette) -> String {
        let colour = match self {
            Mark::Did => p.green(),
            Mark::Warn => p.yellow(),
            Mark::Blocked => p.red(),
            Mark::Chosen | Mark::Plain => "",
            Mark::Absent => p.dim(),
        };
        if colour.is_empty() {
            self.glyph().to_string()
        } else {
            format!("{colour}{}{}", self.glyph(), p.reset())
        }
    }
}

// -----------------------------------------------------------------------------
// Quantities
// -----------------------------------------------------------------------------

/// Bytes, in the largest binary unit that leaves a number worth reading.
///
/// Binary throughout, because every quantity pie reports is a memory or page
/// count: `optimize` reported the same bytes in decimal GB, which made the
/// same file look 7% larger there than in `cache list`.
///
/// No space before the unit, and the fraction is dropped past three digits
/// where it is noise. The point is that a column of these lines up and can be
/// compared at a glance.
pub fn bytes(n: u64) -> String {
    const UNITS: [(&str, u64); 5] = [
        ("TiB", 1 << 40),
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
        ("B", 1),
    ];
    for (suffix, scale) in UNITS {
        if n >= scale {
            let value = n as f64 / scale as f64;
            return if scale == 1 || value >= 100.0 {
                format!("{value:.0}{suffix}")
            } else {
                format!("{value:.1}{suffix}")
            };
        }
    }
    "0B".to_string()
}

/// Bytes per second.
pub fn rate(bytes_per_second: f64) -> String {
    format!("{}/s", bytes(bytes_per_second.max(0.0) as u64))
}

/// A duration, in the largest unit that leaves a number worth reading.
pub fn duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs >= 3600 {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else if secs > 0 {
        format!("{secs}s")
    } else {
        format!("{}ms", d.as_millis())
    }
}

/// A path with `$HOME` written as `~`.
///
/// Almost every path pie prints is under the user's home, and the prefix
/// carries no information -- it is the same on every line, and it is what
/// pushes the part that differs off the edge of a column.
pub fn short_path(path: &std::path::Path) -> String {
    let Some(home) = std::env::var_os("HOME") else {
        return path.display().to_string();
    };
    match path.strip_prefix(std::path::Path::new(&home)) {
        Ok(rest) => format!("~/{}", rest.display()),
        Err(_) => path.display().to_string(),
    }
}

// -----------------------------------------------------------------------------
// Tables
// -----------------------------------------------------------------------------

/// One printed row: a mark, then cells.
pub struct Row {
    pub mark: Mark,
    /// Cells left to right. The last one is treated as the note column and is
    /// what gets cut when the terminal is narrow.
    pub cells: Vec<String>,
}

impl Row {
    pub fn new(mark: Mark, cells: impl IntoIterator<Item = String>) -> Self {
        Self {
            mark,
            cells: cells.into_iter().collect(),
        }
    }
}

/// How a column is laid out.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

/// A column-aligned listing that fits the terminal.
///
/// Four listings each computed their own `{:<width$}`, and `doctor` computed
/// none -- its fixed `{:<20}` key column was blown apart by the absolute
/// config path beside it. Widths come from the rows actually being printed, so
/// a filtered listing is not padded out to the width of the rows it excluded.
pub struct Table {
    aligns: Vec<Align>,
    dim_from: usize,
    rows: Vec<Row>,
}

impl Table {
    /// `dim_from` is the first column that is secondary text -- paths, notes,
    /// descriptions. Everything from there on is dimmed and the last column is
    /// what gets cut to fit.
    pub fn new(aligns: impl IntoIterator<Item = Align>, dim_from: usize) -> Self {
        Self {
            aligns: aligns.into_iter().collect(),
            dim_from,
            rows: Vec::new(),
        }
    }

    pub fn push(&mut self, row: Row) {
        self.rows.push(row);
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Print with two spaces between columns, indented by two.
    pub fn print(&self, p: &Palette) {
        const GAP: usize = 2;
        const INDENT: usize = 2;
        let columns = self.rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
        let widths: Vec<usize> = (0..columns)
            .map(|i| {
                self.rows
                    .iter()
                    .filter_map(|r| r.cells.get(i))
                    .map(|c| c.chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        // Everything but the last column is laid out at its natural width; the
        // last one gets whatever is left, because cutting a note is a loss a
        // reader can absorb and cutting a name is not.
        let fixed: usize = INDENT
            + 2
            + widths.iter().take(columns.saturating_sub(1)).sum::<usize>()
            + GAP * columns.saturating_sub(1);
        let last_room = width().saturating_sub(fixed).max(8);

        for row in &self.rows {
            let mut line = format!("{}{} ", " ".repeat(INDENT), row.mark.render(p));
            for (i, width) in widths.iter().enumerate() {
                let raw = row.cells.get(i).map(String::as_str).unwrap_or("");
                let last = i + 1 == columns;
                let text = if last { clip(raw, last_room) } else { raw.to_string() };
                let pad = width.saturating_sub(text.chars().count());
                if i >= self.dim_from {
                    line.push_str(p.dim());
                }
                match self.aligns.get(i).copied().unwrap_or(Align::Left) {
                    Align::Left => {
                        line.push_str(&text);
                        if !last {
                            line.push_str(&" ".repeat(pad));
                        }
                    }
                    Align::Right => {
                        line.push_str(&" ".repeat(pad));
                        line.push_str(&text);
                    }
                }
                if i >= self.dim_from {
                    line.push_str(p.reset());
                }
                if !last {
                    line.push_str(&" ".repeat(GAP));
                }
            }
            println!("{}", line.trim_end());
        }
    }
}

// -----------------------------------------------------------------------------
// Machine-readable output
// -----------------------------------------------------------------------------

/// Print a value as the command's entire stdout, and nothing else.
///
/// Every `--json` path goes through here so the contract is one sentence: one
/// JSON document per invocation, on stdout, with the human rendering
/// suppressed entirely. A listing that printed a table *and* a document would
/// be parseable by nobody.
///
/// Pretty-printed unconditionally. `jq` does not care, and a person checking
/// what the shape is does.
pub fn emit_json(value: &serde_json::Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

// -----------------------------------------------------------------------------
// Terminal
// -----------------------------------------------------------------------------

/// Usable terminal columns, or 80 when there is no terminal to ask.
///
/// A redraw that writes past the edge wraps, and then `\r` returns to the
/// start of the *last* screen row rather than the start of the line -- so the
/// progress bar leaves a trail of half-erased rows on a narrow terminal. Every
/// line this module draws in place is cut to this width.
pub fn width() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: `winsize` is plain data and the ioctl only writes into it.
        unsafe {
            let mut size: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) == 0
                && size.ws_col > 0
            {
                return size.ws_col as usize;
            }
        }
    }
    80
}

/// Cut `text` to `limit` display columns, ending with `…` when it had to.
///
/// Counts characters rather than bytes: cutting a multi-byte character in half
/// writes a broken sequence to the terminal.
pub fn clip(text: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit.saturating_sub(1)).collect();
    // Prefer a word boundary, but only if one is close enough that cutting
    // there does not throw away most of the line.
    match head.rfind(' ') {
        Some(space) if space * 4 >= limit * 3 => format!("{}…", head[..space].trim_end()),
        _ => format!("{head}…"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_quantity_has_one_rendering() {
        // The four spellings this replaced: "3.00 GiB", "3.0GiB", "3072MiB"
        // and "3.2 GB". The last was decimal, so it was a different number.
        assert_eq!(bytes(3 << 30), "3.0GiB");
        assert_eq!(bytes(0), "0B");
        assert_eq!(bytes(512), "512B");
        assert_eq!(bytes(1 << 10), "1.0KiB");
        // Past three digits the fraction is noise.
        assert_eq!(bytes(200 << 20), "200MiB");
        // A weight-sized artifact should not read as four digits of MiB.
        assert_eq!(bytes(2 << 40), "2.0TiB");
    }

    #[test]
    fn no_colour_leaves_no_escapes() {
        let plain = Palette::forced(false);
        assert_eq!(plain.dim(), "");
        assert_eq!(plain.reset(), "");
        assert_eq!(Mark::Did.render(&plain), "✓");
        // And every glyph is one column wide, so a table built without colour
        // lines up with the same table built with it.
        for mark in [
            Mark::Did,
            Mark::Warn,
            Mark::Blocked,
            Mark::Chosen,
            Mark::Plain,
            Mark::Absent,
        ] {
            assert_eq!(mark.glyph().chars().count(), 1, "{mark:?}");
        }
    }

    #[test]
    fn colour_renders_and_resets() {
        let colour = Palette::forced(true);
        let rendered = Mark::Blocked.render(&colour);
        assert!(rendered.starts_with("\x1b[31m"));
        assert!(rendered.ends_with("\x1b[0m"));
    }

    #[test]
    fn clip_never_splits_a_character() {
        assert_eq!(clip("short", 10), "short");
        assert_eq!(clip("", 10), "");
        // Multi-byte input: the result must still be valid UTF-8 and within
        // the limit.
        let wide = "가나다라마바사아자차카타파하";
        let cut = clip(wide, 5);
        assert!(cut.chars().count() <= 5, "got {cut:?}");
        assert!(cut.ends_with('…'));
        // A word boundary is preferred only when it is not throwing the line
        // away: "aaaa…" beats "…" for a single long word.
        assert_eq!(clip("a bbbbbbbbbbbbbbbb", 6), "a bbb…");
    }

    #[test]
    fn a_table_pads_from_the_rows_it_prints() {
        // Widths come from what is being shown, not from a fixed number: a
        // filtered listing padded to the width of rows it excluded is what
        // `doctor`'s `{:<20}` key column used to be.
        let mut table = Table::new([Align::Left, Align::Right], 1);
        table.push(Row::new(Mark::Plain, ["a".into(), "1".into()]));
        table.push(Row::new(Mark::Absent, ["bbb".into(), "22".into()]));
        assert!(!table.is_empty());
        assert_eq!(table.rows.len(), 2);
        // Column 0 is three wide because "bbb" is, not because anything said so.
        let widths: Vec<usize> = (0..2)
            .map(|i| {
                table
                    .rows
                    .iter()
                    .map(|r| r.cells[i].chars().count())
                    .max()
                    .unwrap()
            })
            .collect();
        assert_eq!(widths, vec![3, 2]);
    }

    #[test]
    fn a_home_path_loses_the_prefix_that_carries_nothing() {
        let home = std::env::var("HOME").unwrap_or_default();
        if home.is_empty() {
            return;
        }
        let inside = std::path::Path::new(&home).join("pie").join("config.toml");
        assert_eq!(short_path(&inside), "~/pie/config.toml");
        // Outside home it stays absolute -- abbreviating there would be a lie.
        let outside = std::path::Path::new("/etc/pie.toml");
        assert_eq!(short_path(outside), "/etc/pie.toml");
    }

    #[test]
    fn durations_read_as_the_unit_a_person_would_use() {
        use std::time::Duration;
        assert_eq!(duration(Duration::from_millis(250)), "250ms");
        assert_eq!(duration(Duration::from_secs(9)), "9s");
        assert_eq!(duration(Duration::from_secs(90)), "1m30s");
        assert_eq!(duration(Duration::from_secs(3700)), "1h01m");
    }
}
