import SwiftUI
import UIKit

/// A photo attachment's preview, square with rounded corners.
struct AttachmentThumbnail: View {
    let attachment: Attachment
    let side: CGFloat

    var body: some View {
        Group {
            if let image = ThumbnailCache.image(for: attachment) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Theme.surfaceStrong
                    .overlay {
                        Image(systemName: "photo").foregroundStyle(Theme.secondaryInk)
                    }
            }
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .accessibilityElement()
        .accessibilityLabel(attachment.name)
    }
}

/// Decoded thumbnails by attachment. The views that show them redraw on
/// every streamed token, and decoding the JPEG each time would be wasted
/// work on the main thread.
enum ThumbnailCache {
    private static let cache = NSCache<NSUUID, UIImage>()

    static func image(for attachment: Attachment) -> UIImage? {
        let key = attachment.id as NSUUID
        if let image = cache.object(forKey: key) { return image }
        guard let data = attachment.thumbnailJPEG, let image = UIImage(data: data) else { return nil }
        cache.setObject(image, forKey: key)
        return image
    }
}
