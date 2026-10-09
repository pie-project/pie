import SwiftUI
import UIKit

/// A message the user sent: right-aligned in a rounded bubble, with its
/// attachments above it and a microphone glyph when it was spoken.
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
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    private var bubble: some View {
        Text(message.text)
            .font(.body)
            .foregroundStyle(Theme.onUserBubble)
            .lineSpacing(2)
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
