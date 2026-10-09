import SwiftUI

/// What a user attached to a sent message, shown above its bubble:
/// photos as thumbnails, documents as file chips.
struct MessageAttachmentsView: View {
    let attachments: [Attachment]

    var body: some View {
        VStack(alignment: .trailing, spacing: 6) {
            let photos = attachments.filter { $0.kind == .photo }
            if !photos.isEmpty {
                HStack(spacing: 6) {
                    ForEach(photos) { photo in
                        AttachmentThumbnail(attachment: photo, side: photos.count == 1 ? 160 : 96)
                    }
                }
            }
            ForEach(attachments.filter { $0.kind == .file }) { file in
                AttachmentFileChip(attachment: file)
            }
        }
    }
}
