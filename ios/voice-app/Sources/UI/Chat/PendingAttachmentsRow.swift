import SwiftUI

/// Attachments waiting to go out with the next message, each with a
/// remove button, and a spinner while one is still being read.
struct PendingAttachmentsRow: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(chat.pendingAttachments) { attachment in
                    preview(attachment)
                        .overlay(alignment: .topTrailing) { removeButton(attachment) }
                }
                if chat.isImportingAttachment {
                    ProgressView()
                        .tint(Theme.secondaryInk)
                        .frame(width: 56, height: 56)
                        .background(Theme.surfaceStrong, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .accessibilityLabel("Reading attachment")
                }
            }
            .padding(.top, 10)
            .padding(.horizontal, 8)
        }
    }

    @ViewBuilder
    private func preview(_ attachment: Attachment) -> some View {
        switch attachment.kind {
        case .photo:
            AttachmentThumbnail(attachment: attachment, side: 56)
        case .file:
            AttachmentFileChip(attachment: attachment)
        }
    }

    private func removeButton(_ attachment: Attachment) -> some View {
        Button {
            chat.removePendingAttachment(attachment.id)
        } label: {
            Image(systemName: "xmark")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(Theme.background)
                .frame(width: 20, height: 20)
                .background(Theme.ink, in: Circle())
                .overlay { Circle().strokeBorder(Theme.background, lineWidth: 1.5) }
                .frame(width: 30, height: 30)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .offset(x: 9, y: -9)
        .accessibilityLabel("Remove \(attachment.name)")
    }
}
