import SwiftUI
import UIKit

/// The row of small icons under a finished reply: copy, thumbs up and
/// down, read aloud, regenerate, share.
///
/// Every glyph change is a symbol replace, so it morphs rather than
/// flickers: copy becomes a checkmark for two seconds (a second copy
/// restarts the two seconds), a thumb fills and the other thumb shrinks
/// away (as in ChatGPT), and the speaker fills and its waves animate while
/// the reply is read aloud. Each tap has a light haptic. A reply with no
/// text (stopped before its first word) offers nothing to copy, read or
/// share.
struct MessageActionBar: View {
    let message: StoredMessage
    let isReadingAloud: Bool
    let actions: MessageActions

    @Environment(\.canRegenerate) private var canRegenerate
    @EnvironmentObject private var settings: AppSettings
    @State private var didCopy = false
    /// Bumped by every copy, so a second tap restarts the two seconds.
    @State private var copies = 0

    var body: some View {
        HStack(spacing: 0) {
            if hasText {
                copyButton
            }
            if message.feedback != .bad {
                thumb(.good).transition(Self.otherThumb)
            }
            if message.feedback != .good {
                thumb(.bad).transition(Self.otherThumb)
            }
            if hasText {
                readAloudButton
            }
            Menu {
                RegenerateMenuItems { mode in
                    Haptics.tap(enabled: settings.haptics)
                    actions.regenerate(message.id, mode)
                }
            } label: {
                glyph("arrow.clockwise")
            }
            .disabled(!canRegenerate)
            .accessibilityLabel("Regenerate")
            if hasText {
                ShareLink(item: message.text) {
                    glyph("square.and.arrow.up")
                }
                .buttonStyle(PressDimButtonStyle())
                .accessibilityLabel("Share")
            }
            Spacer(minLength: 0)
        }
        .foregroundStyle(Theme.secondaryInk)
        // The 44 pt targets are wider than their glyphs; pull the first one
        // back so its glyph lines up with the reply's text.
        .padding(.leading, -13)
        .task(id: copies) {
            guard copies > 0 else { return }
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(Motion.control) { didCopy = false }
        }
    }

    /// The thumb not chosen shrinks to 80% as it fades, and is gone in
    /// 0.1 s, before the icons to its right (sliding left over 0.2 s) reach
    /// its place; at the slide's own pace it was still half there when the
    /// speaker passed over it (recorded). Coming back, it waits a moment
    /// for the icons to move out of its way.
    private static var otherThumb: AnyTransition {
        let shrunk = AnyTransition.opacity.combined(with: .scale(scale: Motion.prefersReduced ? 1 : 0.8))
        return .asymmetric(
            insertion: shrunk.animation(.easeOut(duration: 0.18).delay(0.08)),
            removal: shrunk.animation(.easeOut(duration: 0.1))
        )
    }

    private var hasText: Bool {
        !message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The checkmark comes first and the pasteboard after, and the swap
    /// takes about 0.2 s (see `Clipboard`).
    private var copyButton: some View {
        Button {
            Haptics.tap(enabled: settings.haptics)
            withAnimation(Motion.control) { didCopy = true }
            copies += 1
            Clipboard.copy(message.text)
        } label: {
            glyph(didCopy ? "checkmark" : "doc.on.doc")
                .contentTransition(Clipboard.glyphSwap)
        }
        .buttonStyle(PressDimButtonStyle())
        .accessibilityLabel(didCopy ? "Copied" : "Copy")
        .accessibilityIdentifier("doc.on.doc")
    }

    /// The controller animates the change, so the glyph morphs to its
    /// filled form and the other thumb fades out (or back in).
    private func thumb(_ kind: Feedback) -> some View {
        let isSelected = message.feedback == kind
        let symbol = kind == .good ? "hand.thumbsup" : "hand.thumbsdown"
        return Button {
            Haptics.selection(enabled: settings.haptics)
            actions.setFeedback(isSelected ? nil : kind, message.id)
        } label: {
            glyph(isSelected ? symbol + ".fill" : symbol)
                .foregroundStyle(isSelected ? Theme.accent : Theme.secondaryInk)
                .contentTransition(Clipboard.glyphSwap)
        }
        .buttonStyle(PressDimButtonStyle())
        .accessibilityLabel(kind == .good ? "Good response" : "Bad response")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    /// While the reply is read aloud the speaker fills and its waves
    /// sweep; tapping it again stops.
    private var readAloudButton: some View {
        Button {
            Haptics.tap(enabled: settings.haptics)
            actions.toggleReadAloud(message.id)
        } label: {
            glyph(isReadingAloud ? "speaker.wave.2.fill" : "speaker.wave.2")
                .foregroundStyle(isReadingAloud ? Theme.accent : Theme.secondaryInk)
                .contentTransition(Clipboard.glyphSwap)
                .symbolEffect(.variableColor.iterative, options: .repeating, isActive: isReadingAloud)
        }
        .buttonStyle(PressDimButtonStyle())
        .accessibilityLabel(isReadingAloud ? "Stop reading aloud" : "Read aloud")
    }

    private func glyph(_ symbol: String) -> some View {
        Image(systemName: symbol)
            .font(.system(size: 15, weight: .regular))
            .frame(width: 44, height: 44)
            .contentShape(Rectangle())
    }
}
