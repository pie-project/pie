import SwiftUI
import UIKit

/// The row of small icons under a finished reply: copy, thumbs up and
/// down, read aloud, regenerate, share.
struct MessageActionBar: View {
    let message: StoredMessage
    let isReadingAloud: Bool
    let canRegenerate: Bool
    let actions: MessageActions

    @State private var didCopy = false

    var body: some View {
        HStack(spacing: 0) {
            icon(didCopy ? "checkmark" : "doc.on.doc", label: didCopy ? "Copied" : "Copy") {
                UIPasteboard.general.string = message.text
                didCopy = true
            }
            icon(
                message.feedback == .good ? "hand.thumbsup.fill" : "hand.thumbsup",
                label: "Good response",
                isSelected: message.feedback == .good
            ) {
                actions.setFeedback(message.feedback == .good ? nil : .good, message.id)
            }
            icon(
                message.feedback == .bad ? "hand.thumbsdown.fill" : "hand.thumbsdown",
                label: "Bad response",
                isSelected: message.feedback == .bad
            ) {
                actions.setFeedback(message.feedback == .bad ? nil : .bad, message.id)
            }
            icon(isReadingAloud ? "stop.fill" : "speaker.wave.2", label: isReadingAloud ? "Stop reading aloud" : "Read aloud") {
                actions.toggleReadAloud(message.id)
            }
            Menu {
                RegenerateMenuItems { mode in actions.regenerate(message.id, mode) }
            } label: {
                glyph("arrow.clockwise")
            }
            .disabled(!canRegenerate)
            .accessibilityLabel("Regenerate")
            ShareLink(item: message.text) {
                glyph("square.and.arrow.up")
            }
            .accessibilityLabel("Share")
            Spacer(minLength: 0)
        }
        .foregroundStyle(Theme.secondaryInk)
        // The 44 pt targets are wider than their glyphs; pull the first one
        // back so its glyph lines up with the reply's text.
        .padding(.leading, -13)
        .task(id: didCopy) {
            guard didCopy else { return }
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            didCopy = false
        }
    }

    private func icon(
        _ symbol: String,
        label: String,
        isSelected: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            glyph(symbol)
                .foregroundStyle(isSelected ? Theme.accent : Theme.secondaryInk)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private func glyph(_ symbol: String) -> some View {
        Image(systemName: symbol)
            .font(.system(size: 15, weight: .regular))
            .frame(width: 44, height: 44)
            .contentShape(Rectangle())
    }
}
