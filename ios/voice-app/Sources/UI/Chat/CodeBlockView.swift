import SwiftUI
import UIKit

/// A fenced code block as ChatGPT draws it: a header with the language
/// and a Copy button over monospaced text that scrolls sideways instead of
/// wrapping.
struct CodeBlockView: View {
    let language: String
    let code: String

    @State private var didCopy = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language)
                    .font(.caption.weight(.medium))
                Spacer(minLength: 8)
                Button {
                    UIPasteboard.general.string = code
                    didCopy = true
                } label: {
                    Label(didCopy ? "Copied" : "Copy", systemImage: didCopy ? "checkmark" : "doc.on.doc")
                        .font(.caption.weight(.medium))
                        .frame(minHeight: 32)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(didCopy ? "Copied" : "Copy code")
            }
            .foregroundStyle(Theme.secondaryInk)
            .padding(.horizontal, 14)
            .padding(.vertical, 2)
            .background(Theme.surface)

            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(Theme.ink)
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .padding(14)
            }
        }
        .background(Theme.codeBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Theme.hairline, lineWidth: 0.5)
        }
        .task(id: didCopy) {
            guard didCopy else { return }
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            didCopy = false
        }
    }
}
