import SwiftUI
import UIKit

/// A fenced code block as ChatGPT draws it: a header with the language
/// and a Copy button over monospaced text that scrolls sideways instead of
/// wrapping.
///
/// While the reply streams, the frame fades in when the fence opens and
/// the code's newest words fade in like the rest of the reply.
struct CodeBlockView: View {
    let language: String
    let code: String
    var isLive = false
    /// The block appeared just now in a streaming reply.
    var startsFresh = false

    @EnvironmentObject private var settings: AppSettings
    @State private var didCopy = false
    /// Bumped by every copy, so a second tap restarts the two seconds.
    @State private var copies = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language)
                    .font(.caption.weight(.medium))
                Spacer(minLength: 8)
                copyButton
            }
            .foregroundStyle(Theme.secondaryInk)
            .padding(.horizontal, 14)
            .padding(.vertical, 2)
            .background(Theme.surface)

            ScrollView(.horizontal, showsIndicators: false) {
                RevealText(AttributedString(code), isLive: isLive, startsFresh: startsFresh)
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
        .modifier(FadeInWhenNew(isNew: isLive && startsFresh))
        .task(id: copies) {
            guard copies > 0 else { return }
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(Motion.control) { didCopy = false }
        }
    }

    /// "Copy" becomes "Copied" with a checkmark for two seconds. The label
    /// keeps the width of the longer one, so nothing beside it shifts.
    private var copyButton: some View {
        Button {
            UIPasteboard.general.string = code
            Haptics.tap(enabled: settings.haptics)
            withAnimation(Motion.control) { didCopy = true }
            copies += 1
        } label: {
            ZStack(alignment: .leading) {
                copyLabel(copied: true).hidden()
                copyLabel(copied: didCopy)
            }
            .font(.caption.weight(.medium))
            .frame(minHeight: 32)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressDimButtonStyle())
        .accessibilityLabel(didCopy ? "Copied" : "Copy code")
    }

    private func copyLabel(copied: Bool) -> some View {
        HStack(spacing: 4) {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .contentTransition(.symbolEffect(.replace))
            Text(copied ? "Copied" : "Copy")
                .contentTransition(.opacity)
        }
    }
}
