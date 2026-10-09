import SwiftUI

/// A document attachment: an icon tile, the file name and its type.
struct AttachmentFileChip: View {
    let attachment: Attachment

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "doc.text.fill")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(Theme.onAccent)
                .frame(width: 36, height: 36)
                .background(Theme.orange, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.name)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(kind)
                    .font(.caption)
                    .foregroundStyle(Theme.secondaryInk)
            }
        }
        .padding(.vertical, 8)
        .padding(.leading, 8)
        .padding(.trailing, 14)
        .frame(maxWidth: 240, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Theme.hairline, lineWidth: 0.5)
        }
        .accessibilityElement(children: .combine)
    }

    private var kind: String {
        let ext = (attachment.name as NSString).pathExtension
        return ext.isEmpty ? "Document" : ext.uppercased()
    }
}
