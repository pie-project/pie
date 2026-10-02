use crate::config;

pub(super) struct StartupBanner {
    model: String,
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
