import SwiftUI
import UIKit

/// A message the user sent: right-aligned in a rounded bubble, with its
/// attachments above it and a microphone glyph when it was spoken.
///
/// A message sent a moment ago rises into place as it appears, fading in
/// from 12 points lower, as if out of the composer (with Reduce Motion it
/// only fades). It does this itself,
/// on appearing, so it happens the same way whether the list was already
/// on screen or arrives with this first message; a chat opened later shows
/// its messages where they are. The space above the bubble is part of the
/// row, so when the list lands a new question at the top it sits a little
/// below the top bar.
///
/// Equatable on its values, so a streamed token elsewhere in the
/// conversation does not redraw it.
struct UserMessageRow: View, Equatable {
    let message: StoredMessage
    /// Space always left empty to the bubble's left, so it never runs
    /// wider than about 78% of the row but still hugs short text.
    let minimumLeadingSpace: CGFloat
    /// Whether the engine can answer an edit; false while it is loading.
    let canEdit: Bool
    let sheet: Binding<MessageSheet?>

    /// The gap above the bubble, inside the row.
    static let topSpace: CGFloat = 12
    /// How far a new bubble rises.
    private static let rise: CGFloat = 12
    /// A message younger than this when its row appears was just sent.
    private static let newMessageAge: TimeInterval = 1

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hasRisen: Bool

    init(message: StoredMessage, minimumLeadingSpace: CGFloat, canEdit: Bool, sheet: Binding<MessageSheet?>) {
        self.message = message
        self.minimumLeadingSpace = minimumLeadingSpace
        self.canEdit = canEdit
        self.sheet = sheet
        _hasRisen = State(initialValue: -message.createdAt.timeIntervalSinceNow > Self.newMessageAge)
    }

    static func == (lhs: UserMessageRow, rhs: UserMessageRow) -> Bool {
        lhs.message == rhs.message
            && lhs.minimumLeadingSpace == rhs.minimumLeadingSpace
            && lhs.canEdit == rhs.canEdit
    }

    var body: some View {
        VStack(alignment: .trailing, spacing: 6) {
            if !message.attachments.isEmpty {
                MessageAttachmentsView(attachments: message.attachments)
            }
            if !message.text.isEmpty {
                HStack(alignment: .center, spacing: 8) {
                    Spacer(minLength: minimumLeadingSpace)
                    if message.viaVoice {
                        Image(systemName: "mic.fill")
                            .font(.caption)
                            .foregroundStyle(Theme.tertiaryInk)
                            .accessibilityLabel("Spoken")
                    }
                    bubble
                }
            }
        }
        .opacity(hasRisen ? 1 : 0)
        .offset(y: hasRisen || reduceMotion ? 0 : Self.rise)
        .frame(maxWidth: .infinity, alignment: .trailing)
        .padding(.top, Self.topSpace)
        .onAppear {
            guard !hasRisen else { return }
            withMotion(Motion.content) { hasRisen = true }
        }
    }

    private var bubble: some View {
        Text(message.text)
            .font(.body)
            .foregroundStyle(Theme.onUserBubble)
            .lineSpacing(2)
            // An edit's new text crossfades as the bubble resizes.
            .contentTransition(.opacity)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Theme.userBubble, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .contentShape(.contextMenuPreview, RoundedRectangle(cornerRadius: 20, style: .continuous))
            .contextMenu {
                Button {
                    UIPasteboard.general.string = message.text
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                Button {
                    sheet.wrappedValue = .edit(message)
                } label: {
                    Label("Edit", systemImage: "pencil")
                }
                .disabled(!canEdit)
                Button {
                    sheet.wrappedValue = .selectText(message.text)
                } label: {
                    Label("Select Text", systemImage: "selection.pin.in.out")
                }
            }
            .accessibilityLabel("You said: \(message.text)")
    }
}
