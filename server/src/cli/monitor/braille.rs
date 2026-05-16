//! Braille-character multi-series line graph. Mirrors
//! `pie_cli/monitor/app.py::GraphCanvas::_render_braille_graph`.
//!
//! Each terminal cell is a 2-wide × 4-tall braille glyph (8 dots);
//! we pack `width * 2` data points into `width` cells per row, and
//! `height * 4` vertical resolution per column. Series are drawn in
//! input order; later series only fill cells the earlier ones left
//! empty (so the dominant series wins under overlap).

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

const BRAILLE_BASE: u32 = 0x2800;
/// Bit at row r, col c. Row 0 = top, row 3 = bottom; col 0 = left.
const BRAILLE_DOTS: [[u32; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

#[derive(Debug, Clone)]
pub struct Series<'a> {
    /// Series identifier; surfaced in the legend by the parent app.
    /// Unused inside the renderer but kept here so the caller can
    /// pass a single struct to both braille::render and the legend.
    #[allow(dead_code)]
    pub name: &'a str,
    pub color: Color,
    pub values: &'a [f64],
    pub min_val: f64,
    pub max_val: f64,
    /// Format hint for the legend; renderer ignores it.
    #[allow(dead_code)]
    pub is_integer: bool,
}

fn normalize(v: f64, min: f64, max: f64) -> f64 {
    let range = max - min;
    if range == 0.0 {
        return 0.5;
    }
    let n = (v - min) / range;
    n.clamp(0.0, 1.0)
}

/// Render `series` as a stack of `height` lines of width `width`,
/// returning ratatui `Line<'static>`s that the caller paints into a
/// Paragraph. The bottom line reserves `axis_label_width` cells on
/// the right for the caller-provided axis label (rendered separately
/// — this fn just leaves whitespace).
pub fn render(
    series: &[Series<'_>],
    width: usize,
    height: usize,
    axis_label_width: usize,
) -> Vec<Line<'static>> {
    if width < 5 || height < 1 {
        return Vec::new();
    }

    let data_points_needed = width * 2;
    let total_dot_rows = height * 4;
    // Per-cell ownership grid: -1 if empty, else the index of the
    // series that owns that dot. Stored row-major.
    let mut grid: Vec<i32> = vec![-1; data_points_needed * total_dot_rows];

    for (s_idx, s) in series.iter().enumerate() {
        if s.values.is_empty() {
            continue;
        }
        let n = s.values.len().min(data_points_needed);
        let start_col = data_points_needed - n;
        let display = &s.values[s.values.len() - n..];

        let mut prev_row: Option<i32> = None;
        for (i, &value) in display.iter().enumerate() {
            let col = start_col + i;
            let normalized = normalize(value, s.min_val, s.max_val);
            let dot_row = ((1.0 - normalized) * (total_dot_rows as f64 - 1.0)).round() as i32;
            let dot_row = dot_row.clamp(0, total_dot_rows as i32 - 1);

            // Mark this cell. Lower-index series claims first.
            let idx = dot_row as usize * data_points_needed + col;
            if grid[idx] == -1 {
                grid[idx] = s_idx as i32;
            }

            // Draw a vertical line between this point and the previous
            // when they're separated by more than one row, so the line
            // doesn't visually disconnect. Only fills cells that are
            // still empty.
            if let Some(p) = prev_row {
                if (dot_row - p).abs() > 1 {
                    let (lo, hi) = if dot_row < p {
                        (dot_row, p)
                    } else {
                        (p, dot_row)
                    };
                    for r in lo..=hi {
                        let idx = r as usize * data_points_needed + col;
                        if grid[idx] == -1 {
                            grid[idx] = s_idx as i32;
                        }
                    }
                }
            }
            prev_row = Some(dot_row);
        }
    }

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(height);
    for char_row in 0..height {
        let row_width = if char_row == height - 1 {
            width.saturating_sub(axis_label_width)
        } else {
            width
        }
        .max(1);

        let mut spans: Vec<Span<'static>> = Vec::new();
        // Coalesce contiguous same-color glyphs into one Span to keep
        // the resulting Paragraph small.
        let mut run = String::new();
        let mut run_color: Option<Color> = None;
        for char_col in 0..row_width {
            let mut braille = BRAILLE_BASE;
            let mut owner: i32 = -1;
            for dr in 0..4usize {
                for dc in 0..2usize {
                    let grid_row = char_row * 4 + dr;
                    let grid_col = char_col * 2 + dc;
                    if grid_col >= data_points_needed {
                        continue;
                    }
                    let idx = grid_row * data_points_needed + grid_col;
                    let cell = grid[idx];
                    if cell >= 0 {
                        braille |= BRAILLE_DOTS[dr][dc];
                        if owner == -1 {
                            owner = cell;
                        }
                    }
                }
            }

            let (ch, color) = if braille == BRAILLE_BASE {
                (' ', Color::Reset)
            } else {
                let c = if owner >= 0 {
                    series[owner as usize].color
                } else {
                    Color::Reset
                };
                (char::from_u32(braille).unwrap_or(' '), c)
            };

            if Some(color) != run_color && !run.is_empty() {
                spans.push(Span::styled(
                    std::mem::take(&mut run),
                    Style::default().fg(run_color.unwrap_or(Color::Reset)),
                ));
            }
            run.push(ch);
            run_color = Some(color);
        }
        if !run.is_empty() {
            spans.push(Span::styled(
                run,
                Style::default().fg(run_color.unwrap_or(Color::Reset)),
            ));
        }
        lines.push(Line::from(spans));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s<'a>(values: &'a [f64]) -> Series<'a> {
        Series {
            name: "x",
            color: Color::Reset,
            values,
            min_val: 0.0,
            max_val: 100.0,
            is_integer: false,
        }
    }

    #[test]
    fn empty_when_too_small() {
        // Width < 5 or height < 1 → empty (caller draws nothing).
        let out = render(&[s(&[1.0, 2.0])], 4, 1, 0);
        assert!(out.is_empty());
        let out = render(&[s(&[1.0, 2.0])], 10, 0, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn produces_one_line_per_row() {
        let values = (0..20).map(|i| i as f64 * 5.0).collect::<Vec<_>>();
        let out = render(&[s(&values)], 20, 4, 0);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn handles_single_point() {
        let out = render(&[s(&[42.0])], 20, 3, 0);
        assert_eq!(out.len(), 3);
        // Last row has a non-empty span (the dot landed somewhere).
        let nonempty: usize = out
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .filter(|s| !s.content.trim().is_empty())
                    .count()
            })
            .sum();
        assert!(nonempty > 0);
    }

    #[test]
    fn flat_values_dont_panic_on_zero_range() {
        let values = vec![50.0; 10];
        let out = render(&[s(&values)], 20, 3, 0);
        assert_eq!(out.len(), 3);
    }
}
