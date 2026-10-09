import SwiftUI

/// Attachments waiting to go out with the next message, each with a
/// remove button, and a spinner tile while one is still being read.
///
/// The composer passes in its animated copy of the list (see
/// `ComposerView`), so a new tile pops in from 80 % and a removed one
/// shrinks and fades while its neighbours slide over.
struct PendingAttachmentsRow: View {
    let attachments: [Attachment]
    let showsImportTile: Bool

    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Photos are square tiles of this side; file chips are centred in a
    /// row this tall, so swapping the spinner for either does not change
    /// the composer's height.
    private static let tileHeight: CGFloat = 56
    private static let importTileID = UUID()

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(attachments) { attachment in
                        preview(attachment)
                            .overlay(alignment: .topTrailing) { removeButton(attachment) }
                            .id(attachment.id)
                            .transition(tileTransition)
                    }
                    if showsImportTile {
                        ProgressView()
                            .tint(Theme.secondaryInk)
                            .frame(width: Self.tileHeight, height: Self.tileHeight)
                            .background(Theme.surfaceStrong, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .accessibilityLabel("Reading attachment")
                            .id(Self.importTileID)
                            .transition(tileTransition)
                    }
                }
                .padding(.top, 10)
                .padding(.horizontal, 8)
            }
            // The newest tile is the last one; bring it into view when it
            // arrives, in the same motion as its pop.
            .onChange(of: newestTileID) { _, id in
                guard let id else { return }
                withMotion(Motion.scroll) { proxy.scrollTo(id, anchor: .trailing) }
            }
        }
    }

    private var newestTileID: UUID? {
        showsImportTile ? Self.importTileID : attachments.last?.id
    }

    /// In: a pop from 80 %. Out: shrinking to 80 % while fading. Reduce
    /// Motion keeps only the fades.
    private var tileTransition: AnyTransition {
        .asymmetric(
            insertion: .popIn(from: 0.8),
            removal: reduceMotion ? .opacity : .scale(scale: 0.8).combined(with: .opacity)
        )
    }

    @ViewBuilder
    private func preview(_ attachment: Attachment) -> some View {
        switch attachment.kind {
        case .photo:
            AttachmentThumbnail(attachment: attachment, side: Self.tileHeight)
        case .file:
            AttachmentFileChip(attachment: attachment)
                .frame(height: Self.tileHeight)
        }
    }

    private func removeButton(_ attachment: Attachment) -> some View {
        Button {
            Haptics.tap(enabled: settings.haptics)
            // The composer animates the tile away when the list changes.
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
        // A 20 pt badge needs a deeper press than the send button's to show.
        .buttonStyle(PressScaleButtonStyle(pressedScale: 0.85))
        .offset(x: 9, y: -9)
        .accessibilityLabel("Remove \(attachment.name)")
    }
}
