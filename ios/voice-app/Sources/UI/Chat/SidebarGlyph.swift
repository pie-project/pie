import SwiftUI

/// ChatGPT's sidebar button: two strokes, the lower one shorter, rather
/// than the three-line menu symbol.
struct SidebarGlyph: View {
    var body: some View {
        SidebarGlyphShape()
            .stroke(style: StrokeStyle(lineWidth: 2, lineCap: .round))
            .frame(width: 19, height: 9)
    }
}

private struct SidebarGlyphShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX + rect.width * 0.58, y: rect.maxY))
        return path
    }
}
