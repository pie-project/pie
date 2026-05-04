# Pie figure style guide

This file governs SVG diagrams under `website/docs/`. When in doubt, optimize for a reader who is glancing — figures should communicate the structure at a glance, not reward close inspection.

Companion to [STYLE.md](./STYLE.md), which governs prose.

## Authoring approach

Hand-drawn SVG. We don't use diagram DSLs (Mermaid, D2) — the figures are few enough that per-pixel control beats automation, and auto-layout output never matches the editorial register of the prose.

Place each SVG next to the page that uses it (`docs/<section>/<page>/<figure>.svg`). The same file is reused only when the same diagram appears on multiple pages.

## Canvas

- `viewBox` width: max 720 (matches Docusaurus prose width).
- `viewBox` height: whatever the diagram needs, typically 200–280.
- Outer margin: 24px on all sides.
- Set `width` and `height` attributes equal to the viewBox values so the figure renders at intrinsic size by default.

## Typography

- **One family**: IBM Plex Serif throughout. Fallback chain: `Charter, Georgia, serif`.
- **One size**: 13px. Use it for every text element regardless of role.
- **No monospace fonts.** The diagrams render as part of the prose flow, not as code; mixing JetBrains Mono into figures fights the editorial tone.

Differentiation between text roles comes from color and style, not size.

## Type roles

Three text classes, all 13px Plex Serif:

| Class | Style | Color | Use |
|---|---|---|---|
| `.ink` | regular | `#2c2722` | Primary labels naming a concrete element (`Client`, `Pie`, `Inferlet`, `Model`). |
| `.sub` | regular | `#54473b` | Parenthetical subtitle for environment / location info (`(Wasm)`, `(on GPU)`). |
| `.muted` | italic | `#7a6a55` | Ambient / flow labels: arrow names, relationship descriptions (`WebSocket`, `events`). |

The italic of `.muted` is the only stylistic variation. It marks "this label describes a relationship, not an element."

## Color

A figure uses **one** stroke color and **one** fill color. No per-element accents, no color-coding by category.

| Token | Value | Use |
|---|---|---|
| Stroke | `#54473b` | Every box border, every arrow, every arrowhead fill. |
| Fill | `#fdfbf3` | Every box and container interior. |

If a future figure needs to draw attention to one specific element ("this is the focal point of the page"), reach for stroke weight (`stroke-width: 2.5`) before reaching for color.

## Geometry

Two spacing tokens. Prefer multiples of these.

| Token | Value | Use |
|---|---|---|
| Tight | 24px | Outer canvas margin, container inner padding (sides + bottom). |
| Loose | 48px | Inter-box gap between sibling elements; container top label area. |

Standard dimensions:

- Box corner radius: 8px.
- Container corner radius: 10px.
- Stroke width: 1.5px.
- Stroke line cap: rounded (`stroke-linecap="round"`).
- Standard box height for paired primary elements: 90px.
- Standard box width: 144 (one-line label) or 180 (two-line label, side-by-side with a sibling).

## Containers vs elements

| Stroke style | Meaning |
|---|---|
| Solid | A concrete element. Its outline is its real boundary. |
| Dashed (`6 4`) | A logical grouping. The outline is conceptual, not physical. |

Both use the same stroke color. The dashing alone is enough to read as "different kind of thing."

A container's name sits in its top-left corner, inset by 24px from the left edge and 24px from the top edge. Use the `.ink` text class.

## Arrows

A single marker definition for the whole figure:

```xml
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 Z" fill="#54473b"/>
</marker>
```

`orient="auto-start-reverse"` lets the same marker serve both `marker-start` and `marker-end`, which is what bidirectional arrows need.

| Direction | Markers |
|---|---|
| Unidirectional | `marker-end="url(#ah)"`. The line's `x1, y1` is the source; `x2, y2` is the target. |
| Bidirectional | `marker-start="url(#ah)" marker-end="url(#ah)"` on the same line. |

Leave a 6–8px visible gap between the arrowhead tip and the box it points to (or away from). Don't let the arrow visually merge with the box border.

## Labels

- **Capitalize concept names in figures**: `Client`, `Inferlet`, `Model`, `Pie`, `Generator`, `Forward`. Even when the prose convention is lowercase (STYLE.md keeps `inferlet` lowercase in body text). Inside a figure, every concept is a proper noun.
- **Subtitles in parentheses**: `(Wasm)`, `(on GPU)`, `(Py / JS / Rust)`. Use the `.sub` class.
- **Arrow / flow labels are lowercase**: `WebSocket`, `events`, `pull on first launch`. Use the `.muted` class (italic). Capitalize only when the underlying noun is a proper one (`WebSocket` is a protocol name).

## Embedding

In MDX:

```mdx
![A description of what the figure shows.](./figure.svg)
```

The alt text describes the *content* of the diagram, not the file's filename. A reader using a screen reader should learn what the figure communicates, not that there's an image of one.

For dark-mode adaptation later, switch to the `<picture>` pattern (`./figure.svg` + `./figure-dark.svg`); the existing `pie-light.svg` / `pie-dark.svg` in `website/static/img/` is the reference. Don't pre-build dark variants until you've confirmed the light figure is right.

## What to avoid

- Monospace fonts (use Plex Serif for everything).
- Multiple font sizes (use 13px everywhere).
- Per-element color coding (one stroke color across the figure).
- Gradients, drop shadows, blurs, glows.
- More than two stroke styles (solid and dashed are the only ones).
- Decorative elements that don't carry meaning (icons, badges, ornament).
- Diagram-DSL tools (Mermaid, D2, Excalidraw). All SVGs are written by hand.

## Worked example

The first figure built to this style: [`docs/overview/how-pieces-fit.svg`](./docs/overview/how-pieces-fit.svg). Reference it when starting any new figure: it shows the canvas geometry, the marker definition, all three text classes, the solid-vs-dashed container distinction, and the bidirectional arrow pattern in one place.
