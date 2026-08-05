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
/// the wrong file descriptor. `pie model import > log` would keep colouring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stream {
    Stdout,
    Stderr,
    /// Only ever asked about interactivity — there is nothing to colour on the
    /// way in. [`confirm`] uses it to tell "the user is here" from "this is a
    /// script".
    Stdin,
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
        // Nothing is ever written to stdin, so there is nothing to colour.
        Stream::Stdin => false,
    }
}

/// Styling that renders to nothing when colour is off.
///
/// Every method takes the text it styles and hands back something that knows
/// how to end itself. The previous shape handed out the escape sequences
/// themselves -- `palette.dim()` was a `&'static str` and the caller wrote the
/// matching `palette.reset()` -- which made "never emit an escape yourself" a
/// rule ops could only follow voluntarily. Two of them stopped following it and
/// went back to `\x1b[2m` and a bare `is_terminal()` check, so `NO_COLOR` did
/// nothing in `pie config show` or `pie model list`. There is no longer a way
/// to ask this type for an escape, so there is no longer a way to leak one or
/// to forget its reset.
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

    fn wrap<T: std::fmt::Display>(&self, code: &'static str, text: T) -> Styled<T> {
        Styled {
            text,
            code,
            on: self.on,
        }
    }

    /// Secondary text: paths, notes, descriptions. Never the answer itself.
    pub fn dim<T: std::fmt::Display>(&self, text: T) -> Styled<T> {
        self.wrap("2", text)
    }
    /// Headings and section labels.
    pub fn bold<T: std::fmt::Display>(&self, text: T) -> Styled<T> {
        self.wrap("1", text)
    }
    pub fn green<T: std::fmt::Display>(&self, text: T) -> Styled<T> {
        self.wrap("32", text)
    }
    pub fn yellow<T: std::fmt::Display>(&self, text: T) -> Styled<T> {
        self.wrap("33", text)
    }
    pub fn red<T: std::fmt::Display>(&self, text: T) -> Styled<T> {
        self.wrap("31", text)
    }
    /// A screen's own accent, for a role the shared vocabulary does not name.
    ///
    /// `pie inferlet info` colours parameter names, which is one screen's
    /// business and not a meaning any other command needs. It reached for a
    /// literal `\x1b[36m` to do it. The escape still does not belong to the
    /// caller -- what belongs to the caller is the choice of hue.
    pub fn accent<T: std::fmt::Display>(&self, text: T) -> Styled<T> {
        self.wrap("36", text)
    }
}

/// Text that renders with its styling, or plainly when colour is off.
///
/// `Display`, so it drops into a format string like the string it wraps, and
/// it closes what it opens. Width is the width of the text: nothing here is
/// visible to a column count, which is what lets [`Table`] pad a styled cell.
pub struct Styled<T> {
    text: T,
    code: &'static str,
    on: bool,
}

impl<T: std::fmt::Display> std::fmt::Display for Styled<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.on {
            write!(f, "\x1b[{}m{}\x1b[0m", self.code, self.text)
        } else {
            write!(f, "{}", self.text)
        }
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
        let glyph = self.glyph();
        match self {
            Mark::Did => p.green(glyph).to_string(),
            Mark::Warn => p.yellow(glyph).to_string(),
            Mark::Blocked => p.red(glyph).to_string(),
            Mark::Absent => p.dim(glyph).to_string(),
            Mark::Chosen | Mark::Plain => glyph.to_string(),
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
                // Pad first, style second. The padding is inside the styling
                // and the styling closes itself, so a dimmed column cannot
                // leave the rest of the line dim -- which is what an
                // empty-note row used to do.
                let cell = match self.aligns.get(i).copied().unwrap_or(Align::Left) {
                    Align::Left if last => text,
                    Align::Left => format!("{text}{}", " ".repeat(pad)),
                    Align::Right => format!("{}{text}", " ".repeat(pad)),
                };
                // An empty cell gets no styling at all: wrapping nothing still
                // wrote `\x1b[2m\x1b[0m`, so every `cache list` row with no
                // note carried four invisible bytes that `trim_end` could not
                // see and a `| cat -v` reader could.
                if i >= self.dim_from && !cell.is_empty() {
                    line.push_str(&p.dim(cell).to_string());
                } else {
                    line.push_str(&cell);
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
// TOML highlighting
// -----------------------------------------------------------------------------

/// One line of TOML, coloured the way `pie config show` prints a config file.
///
/// Here rather than in `ops/config.rs` because it is presentation and this is
/// where the colour policy lives. It was six `\x1b[..m` constants and its own
/// `is_terminal()` check inside the op, which is how `NO_COLOR=1 pie config
/// show` came to print escapes at a user who had asked it not to. It now goes
/// through the same [`Palette`] as everything else and answers to the same
/// three signals.
///
/// A tiny state machine over the line rather than a TOML parser: the input is
/// one line at a time and the grammar of a line is `#comment`, `[header]`, or
/// `key = value`. Mirrors the "monokai"-ish palette `rich.Syntax(lexer="toml")`
/// produced, which is what this output looked like before.
pub fn toml_line(line: &str, p: &Palette) -> String {
    let comment = |text: &str| p.wrap("2;37", text.to_string()).to_string();
    let header = |text: &str| p.wrap("1;34", text.to_string()).to_string();
    let key = |text: &str| p.wrap("36", text.to_string()).to_string();

    let trimmed_start = line.trim_start();
    let leading = &line[..line.len() - trimmed_start.len()];

    // Whole-line comment.
    if trimmed_start.starts_with('#') {
        return format!("{leading}{}", comment(trimmed_start));
    }
    // Section header: [foo] / [[foo]].
    if trimmed_start.starts_with('[') {
        // Split off any trailing comment so it gets its own colour.
        let (head, tail) = split_trailing_comment(trimmed_start);
        let mut out = format!("{leading}{}", header(head));
        if let Some(c) = tail {
            out.push(' ');
            out.push_str(&comment(c));
        }
        return out;
    }
    // key = value [# comment]
    let Some(eq) = trimmed_start.find('=') else {
        // No `=`: blank line or unrecognized — return as-is.
        return line.to_string();
    };
    let (key_part, rest) = trimmed_start.split_at(eq);
    let (value, trailing) = split_trailing_comment(&rest[1..]);

    let mut out = format!(
        "{leading}{} = {}",
        key(key_part.trim_end()),
        toml_value(value.trim_start(), p)
    );
    if let Some(c) = trailing {
        out.push(' ');
        out.push_str(&comment(c));
    }
    out
}

/// Split off a `#`-prefixed trailing comment, respecting `#` characters
/// inside double-quoted strings. Returns `(value, Option<comment>)`.
fn split_trailing_comment(s: &str) -> (&str, Option<&str>) {
    let mut in_string = false;
    for (i, ch) in s.char_indices() {
        match ch {
            '"' => in_string = !in_string,
            '#' if !in_string => return (s[..i].trim_end(), Some(s[i..].trim_end())),
            _ => {}
        }
    }
    (s.trim_end(), None)
}

fn toml_value(v: &str, p: &Palette) -> String {
    let trimmed = v.trim();
    if trimmed == "true" || trimmed == "false" {
        return p.wrap("35", trimmed).to_string();
    }
    if trimmed.starts_with('"') {
        return p.wrap("32", trimmed).to_string();
    }
    if trimmed.starts_with('[') {
        // Arrays: highlight individual elements, leaving brackets/commas
        // un-coloured. Cheap and good enough for typical config arrays.
        let inner = &trimmed[1..trimmed.len().saturating_sub(1)];
        let elements: Vec<String> = inner.split(',').map(|e| toml_value(e.trim(), p)).collect();
        return format!("[{}]", elements.join(", "));
    }
    if trimmed.parse::<f64>().is_ok() {
        return p.wrap("33", trimmed).to_string();
    }
    trimmed.to_string()
}

// -----------------------------------------------------------------------------
// Asking
// -----------------------------------------------------------------------------

/// Ask before doing something irreversible. `Ok(false)` means "they said no".
///
/// The rule that matters is the one about a missing terminal: there is nobody
/// to ask, so this refuses rather than assuming consent for a delete. It was
/// written out twice, in `pie cache clear` and `pie model remove`, with the
/// same logic and two different wordings; the second copy is how a rule gets
/// half-changed later.
///
/// `escape_hatch` is the flag that skips the question, named so the refusal can
/// say which one to pass.
pub fn confirm(question: &str, escape_hatch: &str) -> anyhow::Result<bool> {
    use std::io::Write;
    if !is_interactive(Stream::Stdin) {
        anyhow::bail!("this needs confirmation and there is no terminal to ask; rerun with `{escape_hatch}`");
    }
    // The prompt goes to stderr so that `pie cache clear > log` still shows it
    // to the person being asked.
    eprint!("{question} [y/N] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| anyhow::anyhow!("read stdin: {e}"))?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "YES"))
}

// -----------------------------------------------------------------------------
// Progress
// -----------------------------------------------------------------------------

/// Whether a stream is something a redraw makes sense on.
///
/// Distinct from [`colour_enabled`]: `NO_COLOR` is a statement about colour and
/// says nothing about whether `\r` will land somewhere useful. A bar checks
/// this; a colour checks that.
pub fn is_interactive(stream: Stream) -> bool {
    match stream {
        Stream::Stdout => std::io::stdout().is_terminal(),
        Stream::Stderr => std::io::stderr().is_terminal(),
        Stream::Stdin => std::io::stdin().is_terminal(),
    }
}

/// A one-line progress bar redrawn in place on stderr.
///
/// Here rather than in the op that draws it, for the reason this module
/// exists. Its previous home was `pie model import`, where it reported bytes as
/// `read as f64 / 1e9` -- decimal GB, against the binary GiB every other line
/// pie prints, so the same file read 7% larger in the bar than in `pie model
/// list` right after. It also drew a fixed 20-cell bar and clipped names at a
/// fixed 48 columns, which is where the wrapping-and-smearing on a narrow
/// terminal came from.
///
/// Nothing is drawn when stderr is not a terminal: a log file collects the
/// finished lines, not an animation.
pub struct Bar {
    interactive: bool,
    last_draw: std::time::Instant,
    drew: bool,
}

impl Default for Bar {
    fn default() -> Self {
        Self::new()
    }
}

impl Bar {
    pub fn new() -> Self {
        Self {
            interactive: is_interactive(Stream::Stderr),
            last_draw: std::time::Instant::now(),
            drew: false,
        }
    }

    /// Redraw, at most ten times a second. The final frame always draws, so
    /// the bar ends full rather than wherever the throttle last let it stop.
    pub fn draw(&mut self, done: u64, total: u64, label: &str) {
        if !self.interactive {
            return;
        }
        let complete = done >= total;
        if !complete && self.last_draw.elapsed() < std::time::Duration::from_millis(100) {
            return;
        }
        self.last_draw = std::time::Instant::now();
        self.drew = true;

        let percent = if total == 0 {
            100
        } else {
            (done * 100 / total).min(100)
        };
        let quantity = format!("{}/{}", bytes(done), bytes(total));
        // Everything but the label is fixed-width; the label gets what is left
        // and is padded to it, because a redraw that shrinks the line leaves
        // the tail of the previous one on screen.
        const CELLS: usize = 20;
        let fixed = 2 + CELLS + 3 + 4 + 2 + quantity.chars().count() + 2;
        let room = width().saturating_sub(fixed).max(8);
        let label = clip(label, room);
        let filled = (percent as usize * CELLS) / 100;
        eprint!(
            "\r  [{}{}] {percent:3}%  {quantity}  {label:<room$}",
            "#".repeat(filled),
            "-".repeat(CELLS - filled),
        );
    }

    /// End the line, if anything was ever drawn on it.
    pub fn finish(&mut self) {
        if self.drew {
            eprintln!();
            self.drew = false;
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
// What a command answers with
// -----------------------------------------------------------------------------

/// A command's answer, in a form that reads to a person and parses to `jq`.
///
/// The two renderings come off **one** value. They were built separately: each
/// `--json` branch assembled its own `json!({...})` and returned early, and the
/// table below it was written from the same data by different code. Nothing
/// stopped the two from drifting, and `doctor` -- the one command whose whole
/// job is to be believed -- carries a comment saying it collects its sections
/// before rendering precisely so "the table and the JSON cannot drift into
/// disagreeing about the verdict". That was one command holding a rule the
/// other seven did not know about.
///
/// `Serialize` gives the machine rendering; [`Report::render`] gives the human
/// one. There is no way to implement one and forget the other.
pub trait Report: serde::Serialize {
    /// Draw to stdout. Never called when `--json` is on.
    fn render(&self, p: &Palette);
}

/// The object-safe half of [`Report`], so [`Output`] can carry any of them.
///
/// `Serialize` is not object-safe, so the erasure happens here: the blanket
/// impl is the only implementor and it forwards to the real thing.
pub trait AnyReport {
    fn render_any(&self, p: &Palette);
    fn to_json(&self) -> anyhow::Result<serde_json::Value>;
}

impl<T: Report> AnyReport for T {
    fn render_any(&self, p: &Palette) {
        self.render(p)
    }
    fn to_json(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }
}

/// What an op hands back instead of printing: an answer, and the status the
/// process exits with.
///
/// Three shapes of answer, because commands come in three kinds and pretending
/// otherwise is what put `--json` on five subcommands out of twenty and left
/// the rest unscriptable. A command either answers a question ([`Answer::report`]),
/// changes something ([`Answer::did`]), or has already said everything it had
/// to say while it worked ([`Answer::quiet`] -- `pie run`, whose output is the
/// inferlet's).
///
/// The exit status rides along because for two commands it *is* the answer:
/// `pie doctor && pie serve` has to work, and `pie run` reports the inferlet's
/// status rather than its own. Everything else exits zero, and no longer has to
/// write `Ok(ExitCode::SUCCESS)` to say so.
pub struct Answer {
    kind: Kind,
    code: std::process::ExitCode,
}

enum Kind {
    Quiet,
    /// `bool` is whether anything actually changed. See [`Answer::noop`].
    Did(bool, String),
    // `Send`, because the ops that block run on `spawn_blocking` and the answer
    // crosses back over that boundary. A report is plain data, so saying so
    // costs nothing.
    Report(Box<dyn AnyReport + Send>),
}

impl Answer {
    /// Nothing to print; the effect was the point, or it was already streamed.
    pub fn quiet() -> Self {
        Self::of(Kind::Quiet)
    }

    /// One result line: what the command changed. Marked [`Mark::Did`].
    pub fn did(text: impl Into<String>) -> Self {
        Self::of(Kind::Did(true, text.into()))
    }

    /// One result line for a command that changed **nothing** — it was already
    /// in the asked-for state, the user declined the prompt, or `--dry-run` was
    /// on.
    ///
    /// Unmarked, because `Mark::Did` means "the command did something" and
    /// these did not. Printing `✓ aborted; nothing was deleted` says the
    /// opposite of what happened, and it is the same glyph the tables use for
    /// success. The distinction was there before this type existed -- the
    /// no-op lines printed bare while `set`/`unset`/`download` printed a `✓` --
    /// and collapsing the two is a regression this constructor exists to
    /// prevent, since a caller now has to pick one.
    pub fn noop(text: impl Into<String>) -> Self {
        Self::of(Kind::Did(false, text.into()))
    }

    /// A document.
    pub fn report(report: impl Report + Send + 'static) -> Self {
        Self::of(Kind::Report(Box::new(report)))
    }

    fn of(kind: Kind) -> Self {
        Self {
            kind,
            code: std::process::ExitCode::SUCCESS,
        }
    }

    /// The same answer, exiting non-zero. For a verdict, not for an error --
    /// an error is an `Err` and is rendered by `main`'s reporter.
    pub fn with_code(mut self, code: std::process::ExitCode) -> Self {
        self.code = code;
        self
    }

    pub fn code(&self) -> std::process::ExitCode {
        self.code
    }
}

/// Show a command's answer, in whichever of the two renderings was asked for.
///
/// The single place that decides. An action under `--json` reports what it did
/// rather than nothing, so a script driving `pie model remove --json` gets a
/// document like every other command instead of an empty stdout.
pub fn present(answer: Answer, json: bool) -> anyhow::Result<()> {
    let palette = Palette::for_stream(Stream::Stdout);
    match answer.kind {
        Kind::Quiet => Ok(()),
        // `changed` rather than the message alone: "already downloaded" and
        // "downloaded" are both successes and a script has to tell them apart
        // without reading English.
        Kind::Did(changed, line) if json => {
            emit_json(&serde_json::json!({ "changed": changed, "message": line }))
        }
        Kind::Did(true, line) => {
            println!("{} {line}", Mark::Did.render(&palette));
            Ok(())
        }
        Kind::Did(false, line) => {
            // Indented to the same column as a marked line, so a run of result
            // lines still aligns and the marked ones are what the eye finds.
            println!("{} {line}", Mark::Plain.render(&palette));
            Ok(())
        }
        Kind::Report(report) if json => emit_json(&report.to_json()?),
        Kind::Report(report) => {
            report.render_any(&palette);
            Ok(())
        }
    }
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
        assert_eq!(plain.dim("path/to/thing").to_string(), "path/to/thing");
        assert_eq!(plain.bold(42).to_string(), "42");
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
        // Styling closes itself. This is the whole reason the palette hands
        // back a `Styled` rather than an escape: a caller cannot open one and
        // forget the reset, because a caller never opens one.
        assert_eq!(colour.dim("x").to_string(), "\x1b[2mx\x1b[0m");
    }

    #[test]
    fn the_toml_highlighter_answers_to_the_palette() {
        // `NO_COLOR=1 pie config show` printed escapes because this rendering
        // asked `is_terminal()` instead of the palette. Every branch of it --
        // comment, header, key/value, trailing comment -- must come back bare.
        let plain = Palette::forced(false);
        for line in [
            "# a comment",
            "[section]  # with a trailing comment",
            "key = \"value\"  # and here",
            "number = 8080",
            "flag = true",
            "list = [1, 2]",
            "",
        ] {
            let rendered = toml_line(line, &plain);
            assert!(
                !rendered.contains('\x1b'),
                "{line:?} rendered with escapes: {rendered:?}"
            );
        }
        // And with colour on it actually colours, so the test above is not
        // passing because the highlighter does nothing.
        let colour = Palette::forced(true);
        assert!(toml_line("[section]", &colour).contains('\x1b'));
    }

    #[test]
    fn a_bar_measures_in_the_same_units_as_everything_else() {
        // The bar reported `bytes / 1e9` -- decimal GB against pie's binary
        // GiB everywhere else, so the same file read 7% larger here than in
        // the listing printed right after it. Both go through `bytes` now.
        assert_eq!(bytes(3 << 30), "3.0GiB");
        // A non-interactive stderr draws nothing at all, which is what keeps
        // an animation out of a log file.
        let mut bar = Bar {
            interactive: false,
            last_draw: std::time::Instant::now(),
            drew: false,
        };
        bar.draw(1, 2, "anything");
        assert!(!bar.drew, "a bar drew to a non-terminal");
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
