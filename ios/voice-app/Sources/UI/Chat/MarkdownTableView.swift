import SwiftUI

/// A pipe table drawn as a grid that scrolls sideways when it is wider
/// than the screen: the header in bold over a rule, hairlines between
/// rows.
struct MarkdownTableView: View {
    let table: MarkdownTable

    /// Cells longer than this wrap at `wrapWidth` instead of widening their
    /// column without bound.
    private static let wrapThreshold = 28
    private static let wrapWidth: CGFloat = 220

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(Array(table.header.enumerated()), id: \.offset) { column, text in
                        cell(text, column: column)
                            .font(.subheadline.weight(.semibold))
                    }
                }
                ForEach(Array(table.rows.enumerated()), id: \.offset) { index, _ in
                    divider(height: index == 0 ? 1 : 0.5)
                    GridRow {
                        ForEach(Array(table.rows[index].enumerated()), id: \.offset) { column, text in
                            cell(text, column: column)
                                .font(.subheadline)
                        }
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.hairline, lineWidth: 0.5)
            }
        }
    }

    /// A rule across the whole grid. It must not size the grid itself, or
    /// inside a sideways scroll view it would ask for infinite width.
    private func divider(height: CGFloat) -> some View {
        Rectangle()
            .fill(Theme.hairline)
            .frame(height: height)
            .gridCellUnsizedAxes(.horizontal)
    }

    @ViewBuilder
    private func cell(_ text: String, column: Int) -> some View {
        let alignment = column < table.alignments.count ? table.alignments[column] : .leading
        let content = Text(MarkdownInline.render(text))
            .multilineTextAlignment(Self.textAlignment(alignment))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        Group {
            if text.count > Self.wrapThreshold {
                content
                    .frame(width: Self.wrapWidth, alignment: Self.frameAlignment(alignment))
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                content.fixedSize()
            }
        }
        .gridColumnAlignment(Self.horizontalAlignment(alignment))
    }

    private static func textAlignment(_ alignment: MarkdownTable.Alignment) -> TextAlignment {
        switch alignment {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    private static func frameAlignment(_ alignment: MarkdownTable.Alignment) -> Alignment {
        switch alignment {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    private static func horizontalAlignment(_ alignment: MarkdownTable.Alignment) -> HorizontalAlignment {
        switch alignment {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }
}
