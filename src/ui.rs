#![allow(unsafe_code)]
use std::io::IsTerminal;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stream {
    Stdout,
    Stderr,
    Stdin,
}

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
        Stream::Stdin => false,
    }
}

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

    pub fn dim<T: std::fmt::Display>(&self, text: T) -> Styled<T> {
        self.wrap("2", text)
    }
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
}

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mark {
    Did,
    Warn,
    Blocked,
    Chosen,
    Plain,
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

pub fn rate(bytes_per_second: f64) -> String {
    format!("{}/s", bytes(bytes_per_second.max(0.0) as u64))
}

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

pub fn short_path(path: &std::path::Path) -> String {
    let Some(home) = std::env::var_os("HOME") else {
        return path.display().to_string();
    };
    match path.strip_prefix(std::path::Path::new(&home)) {
        Ok(rest) => format!("~/{}", rest.display()),
        Err(_) => path.display().to_string(),
    }
}

pub struct Row {
    pub mark: Mark,
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

pub struct Table {
    aligns: Vec<Align>,
    dim_from: usize,
    rows: Vec<Row>,
}

impl Table {
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
                let text = if last {
                    clip(raw, last_room)
                } else {
                    raw.to_string()
                };
                let pad = width.saturating_sub(text.chars().count());
                let cell = match self.aligns.get(i).copied().unwrap_or(Align::Left) {
                    Align::Left if last => text,
                    Align::Left => format!("{text}{}", " ".repeat(pad)),
                    Align::Right => format!("{}{text}", " ".repeat(pad)),
                };
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

pub fn toml_line(line: &str, p: &Palette) -> String {
    let comment = |text: &str| p.wrap("2;37", text.to_string()).to_string();
    let header = |text: &str| p.wrap("1;34", text.to_string()).to_string();
    let key = |text: &str| p.wrap("36", text.to_string()).to_string();

    let trimmed_start = line.trim_start();
    let leading = &line[..line.len() - trimmed_start.len()];

    if trimmed_start.starts_with('#') {
        return format!("{leading}{}", comment(trimmed_start));
    }
    if trimmed_start.starts_with('[') {
        let (head, tail) = split_trailing_comment(trimmed_start);
        let mut out = format!("{leading}{}", header(head));
        if let Some(c) = tail {
            out.push(' ');
            out.push_str(&comment(c));
        }
        return out;
    }
    let Some(eq) = trimmed_start.find('=') else {
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
        let inner = &trimmed[1..trimmed.len().saturating_sub(1)];
        let elements: Vec<String> = inner.split(',').map(|e| toml_value(e.trim(), p)).collect();
        return format!("[{}]", elements.join(", "));
    }
    if trimmed.parse::<f64>().is_ok() {
        return p.wrap("33", trimmed).to_string();
    }
    trimmed.to_string()
}

pub fn confirm(question: &str, escape_hatch: &str) -> anyhow::Result<bool> {
    use std::io::Write;
    if !is_interactive(Stream::Stdin) {
        anyhow::bail!(
            "this needs confirmation and there is no terminal to ask; rerun with `{escape_hatch}`"
        );
    }
    eprint!("{question} [y/N] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| anyhow::anyhow!("read stdin: {e}"))?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "YES"))
}

pub fn is_interactive(stream: Stream) -> bool {
    match stream {
        Stream::Stdout => std::io::stdout().is_terminal(),
        Stream::Stderr => std::io::stderr().is_terminal(),
        Stream::Stdin => std::io::stdin().is_terminal(),
    }
}

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

        let percent = (done * 100).checked_div(total).map_or(100, |p| p.min(100));
        let quantity = format!("{}/{}", bytes(done), bytes(total));
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

    pub fn finish(&mut self) {
        if self.drew {
            eprintln!();
            self.drew = false;
        }
    }
}

pub fn emit_json(value: &serde_json::Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub trait Report: serde::Serialize {
    fn render(&self, p: &Palette);
}

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

pub struct Answer {
    kind: Kind,
    code: std::process::ExitCode,
}

enum Kind {
    Quiet,
    Did(bool, String),
    Report(Box<dyn AnyReport + Send>),
}

impl Answer {
    pub fn quiet() -> Self {
        Self::of(Kind::Quiet)
    }

    pub fn did(text: impl Into<String>) -> Self {
        Self::of(Kind::Did(true, text.into()))
    }

    pub fn noop(text: impl Into<String>) -> Self {
        Self::of(Kind::Did(false, text.into()))
    }

    pub fn report(report: impl Report + Send + 'static) -> Self {
        Self::of(Kind::Report(Box::new(report)))
    }

    fn of(kind: Kind) -> Self {
        Self {
            kind,
            code: std::process::ExitCode::SUCCESS,
        }
    }

    pub fn with_code(mut self, code: std::process::ExitCode) -> Self {
        self.code = code;
        self
    }

    pub fn code(&self) -> std::process::ExitCode {
        self.code
    }
}

pub fn present(answer: Answer, json: bool) -> anyhow::Result<()> {
    let palette = Palette::for_stream(Stream::Stdout);
    match answer.kind {
        Kind::Quiet => Ok(()),
        Kind::Did(changed, line) if json => {
            emit_json(&serde_json::json!({ "changed": changed, "message": line }))
        }
        Kind::Did(true, line) => {
            println!("{} {line}", Mark::Did.render(&palette));
            Ok(())
        }
        Kind::Did(false, line) => {
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

pub fn width() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: `winsize` is plain data and the ioctl only writes into it.
        unsafe {
            let mut size: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) == 0 && size.ws_col > 0
            {
                return size.ws_col as usize;
            }
        }
    }
    80
}

pub fn clip(text: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit.saturating_sub(1)).collect();
    match head.rfind(' ') {
        Some(space) if space * 4 >= limit * 3 => format!("{}…", head[..space].trim_end()),
        _ => format!("{head}…"),
    }
}
