//! The startup box — pure presentation.
//!
//! Renders the three public facts of a boot (model, backend, device) and the
//! client URL into the box `pie serve` prints. Split from `serve.rs` because
//! column arithmetic is not part of booting anything.

use crate::config;

pub(super) struct StartupBanner {
    model: String,
    /// The execution shell the box's third row names. Called `backend`, not
    /// `engine`, since the box is headed `Pie Engine` and a field sharing
    /// that word would collide with it.
    backend: String,
    device: String,
}

impl StartupBanner {
    pub(super) fn from_config(cfg: &config::Config) -> Self {
        let m = &cfg.model;
        let model = format!("{} ({})", m.name, m.model);
        let backend = m.engine.kind.as_str().to_string();
        let device = {
            let device = m.engine.device.join(", ");
            if device.is_empty() {
                "-".to_string()
            } else {
                device
            }
        };

        Self {
            model,
            backend,
            device,
        }
    }

    pub(super) fn render(&self, url: &str) -> String {
        let host = url
            .strip_prefix("ws://")
            .or_else(|| url.strip_prefix("edge://"))
            .unwrap_or(url);
        let rows = [
            ("Host", host),
            ("Model", self.model.as_str()),
            ("Backend", self.backend.as_str()),
            ("Device", self.device.as_str()),
        ];
        let label_width = 12;
        let header = "─ Pie Engine ";
        // character counts, not `str::len()`: `header` opens with `─`
        // (U+2500, three bytes), so byte and character lengths disagree.
        let header_cols = header.chars().count();
        let content_width = rows
            .iter()
            .map(|(_, value)| label_width + 1 + value.chars().count())
            .max()
            .unwrap_or(0)
            .max(header_cols - 2);
        let inner_width = content_width + 2;
        let mut out = String::new();

        out.push_str(&format!(
            "╭{}{}╮\n",
            header,
            "─".repeat(inner_width - header_cols)
        ));
        for (label, value) in rows {
            let content = format!("{label:<label_width$} {value}");
            out.push_str(&format!(
                "│ {:<content_width$} │\n",
                content,
                content_width = content_width
            ));
        }
        out.push_str(&format!("╰{}╯", "─".repeat(inner_width)));
        out
    }
}

/// The one line a supervisor waits for. Not part of [`StartupBanner::render`]
/// because the box is presentation and this is a readiness contract: the box
/// may be suppressed, this may not.
pub(super) fn ready_line(url: &str) -> String {
    format!("✓ Server ready at {url}")
}

