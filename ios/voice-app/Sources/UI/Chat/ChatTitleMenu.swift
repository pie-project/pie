import SwiftUI
import UIKit

/// The top bar's title: "Pie" in the site's serif and the current mode,
/// with "Temporary chat" under it while the open chat is one. It opens
/// ChatGPT's model menu: a Mode section (Instant, Thinking), a Model
/// section listing the ladder, where picking another installed rung takes
/// effect on the next launch, and sharing the open chat.
///
/// This view only picks out of the controller what the menu shows and
/// hands it to `TitleMenu`, which compares equal unless one of those
/// changed. So the menu is not rebuilt on every streamed token or
/// keystroke, and a menu held open while a reply streams keeps its shape:
/// none of what it shows changes when the reply ends. (It used to offer
/// "Share chat" only once the reply had finished, and the open menu
/// jumped and grew a row at that moment, recorded.)
struct ChatTitleMenu: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        TitleMenu(
            chat: chat,
            mode: chat.mode,
            isTemporary: chat.conversation.isTemporary,
            canShare: !chat.conversation.isEmpty
        )
        .equatable()
    }
}

private struct TitleMenu: View, Equatable {
    /// For changing the mode; not observed here (see `ChatTitleMenu`).
    let chat: ChatController
    let mode: ReplyMode
    let isTemporary: Bool
    /// The chat has something to share.
    let canShare: Bool

    @State private var chosenDirectory = PieRuntimeConfig.selected.directory
    @State private var restartTarget: PieRuntimeConfig.Model?

    // The title follows Dynamic Type up to what the bar's 44 points can
    // hold, the caption line included; bars do not grow past that.
    @ScaledMetric(relativeTo: .title3) private var titleSize: CGFloat = 21
    @ScaledMetric(relativeTo: .body) private var modeSize: CGFloat = 17
    @ScaledMetric(relativeTo: .caption) private var captionSize: CGFloat = 12

    static func == (lhs: TitleMenu, rhs: TitleMenu) -> Bool {
        lhs.mode == rhs.mode && lhs.isTemporary == rhs.isTemporary && lhs.canShare == rhs.canShare
    }

    var body: some View {
        Menu {
            Section("Mode") {
                ForEach(ReplyMode.allCases) { mode in
                    Toggle(isOn: modeBinding(mode)) {
                        Text(mode.title)
                        Text(mode.subtitle)
                    }
                }
            }
            Section("Model") {
                ForEach(PieRuntimeConfig.ladder, id: \.directory) { model in
                    if BootedModel.installedDirectories.contains(model.directory) {
                        Toggle(isOn: modelBinding(model)) {
                            Text(model.label)
                            Text(model.directory == BootedModel.current.directory ? "Running now" : "Loads on restart")
                        }
                    } else {
                        Button {} label: {
                            Text(model.label)
                            Text("Not installed")
                        }
                        .disabled(true)
                    }
                }
            }
            if canShare {
                Section {
                    // The text is made when the share sheet asks for it,
                    // not each time the bar redraws.
                    ShareLink(item: ChatShare(chat: chat), preview: SharePreview("Pie chat")) {
                        Label("Share chat", systemImage: "square.and.arrow.up")
                    }
                }
            }
        } label: {
            label
        }
        .accessibilityLabel(accessibilityTitle)
        .accessibilityHint("Chooses the mode and the model")
        .alert(
            "Restart Pie Voice to load \(restartTarget?.label ?? "")",
            isPresented: Binding(get: { restartTarget != nil }, set: { if !$0 { restartTarget = nil } })
        ) {
            // The engine maps a model's weights once per process; quitting
            // is the only way to load another, and the choice is already
            // saved for the next launch.
            Button("Restart") { quit() }
            Button("Later", role: .cancel) {}
        } message: {
            Text("The model loads when Pie Voice starts. Your chats are saved.")
        }
    }

    /// The mode's name crossfades to the new one, and "Temporary chat"
    /// fades in or out under the title while the title eases up or down
    /// to make room.
    ///
    /// The caption is always there, only its opacity changes. The bar hosts
    /// this label in UIKit, and there a caption inserted with a transition
    /// popped in at full strength while the title was still sliding (about
    /// 0.1 s ahead of it, recorded); property changes animate fine. The
    /// label's height is fixed at the bar's 44 points: without the caption
    /// the title row is centred in them (`titleRowCenter`), with it the
    /// two lines are, and the change of alignment is what moves the title.
    private var label: some View {
        VStack(spacing: 0) {
            // The row is laid out at the width of its widest mode, with
            // the current one centred inside: the bar centres the item by
            // its width and would move it in one jump if that changed, so
            // the width stays put and the row slides within it instead.
            ZStack {
                ForEach(ReplyMode.allCases) { mode in
                    titleRow(mode).hidden()
                }
                titleRow(mode)
            }
            .alignmentGuide(.titleRowCenter) { $0[VerticalAlignment.center] }
            Text("Temporary chat")
                .font(.system(size: min(captionSize, 13), weight: .medium))
                .opacity(isTemporary ? 0.78 : 0)
                // Going away, it is gone before the title has settled
                // back onto its line.
                .animation(isTemporary ? Motion.crossfade : Motion.fadeOut, value: isTemporary)
        }
        .lineLimit(1)
        .foregroundStyle(Theme.onTopBar)
        .padding(.horizontal, 10)
        .frame(height: 44, alignment: Alignment(horizontal: .center, vertical: isTemporary ? .center : .titleRowCenter))
        .contentShape(Rectangle())
        .animation(Motion.crossfade, value: mode)
        .animation(Motion.crossfade, value: isTemporary)
    }

    private func titleRow(_ mode: ReplyMode) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text("Pie")
                .font(Theme.serif(min(titleSize, 23)))
            Text(mode.title)
                .font(.system(size: min(modeSize, 20), weight: .regular))
                .opacity(0.78)
                .contentTransition(.opacity)
            Image(systemName: "chevron.down")
                .font(.system(size: min(titleSize, 23) * 0.52, weight: .bold))
                .opacity(0.78)
        }
    }

    private var accessibilityTitle: String {
        let title = "Pie, \(mode.title)"
        return isTemporary ? title + ", temporary chat" : title
    }

    private func modeBinding(_ mode: ReplyMode) -> Binding<Bool> {
        Binding(
            get: { self.mode == mode },
            set: { isOn in
                guard isOn else { return }
                withMotion(Motion.crossfade) { chat.mode = mode }
            }
        )
    }

    private func modelBinding(_ model: PieRuntimeConfig.Model) -> Binding<Bool> {
        Binding(
            get: { chosenDirectory == model.directory },
            set: { isOn in
                guard isOn, model.directory != chosenDirectory else { return }
                PieRuntimeConfig.select(model)
                chosenDirectory = model.directory
                if model.directory != BootedModel.current.directory {
                    restartTarget = model
                }
            }
        )
    }

    /// Quits so the next launch loads the chosen model. A reply still
    /// streaming is stopped and saved as it stands first, and the app
    /// fades out rather than vanishing, so the quit reads as deliberate
    /// rather than as a crash. The chat files are flushed at the very end,
    /// right before the exit, so a save made during the fade is not lost.
    private func quit() {
        chat.stop()
        let window = UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow }
            .first
        guard let window else {
            ChatStore.finishPendingWrites()
            exit(0)
        }
        UIView.animate(withDuration: 0.25, animations: { window.alpha = 0 }) { _ in
            ChatStore.finishPendingWrites()
            exit(0)
        }
    }
}

/// The open chat as Markdown text for the share sheet. It is read from the
/// controller when the share sheet asks for it, so it is the chat as it
/// is then (the menu itself is not rebuilt as the chat grows), and the
/// bar does not render the whole chat each time it redraws.
private struct ChatShare: Transferable {
    let chat: ChatController

    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: .utf8PlainText) { (share: ChatShare) in
            Data(await share.text().utf8)
        }
    }

    @MainActor
    private func text() -> String {
        MessageRendering.shareText(chat.conversation)
    }
}

private extension VerticalAlignment {
    /// The middle of the title's row (not counting the caption under it).
    enum TitleRowCenter: AlignmentID {
        static func defaultValue(in dimensions: ViewDimensions) -> CGFloat {
            dimensions[VerticalAlignment.center]
        }
    }

    static let titleRowCenter = VerticalAlignment(TitleRowCenter.self)
}
