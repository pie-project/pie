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
/// changed. So the menu (and the whole chat rendered for sharing) is not
/// rebuilt on every streamed token or keystroke, and a menu held open
/// while a reply streams is not swapped under the finger.
struct ChatTitleMenu: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        TitleMenu(
            chat: chat,
            mode: chat.mode,
            isTemporary: chat.conversation.isTemporary,
            // Rendering the whole chat is left until it has stopped
            // changing, as before: nothing to share while it streams.
            shareable: chat.conversation.isEmpty || chat.isGenerating ? nil : chat.conversation
        )
        .equatable()
    }
}

private struct TitleMenu: View, Equatable {
    /// For changing the mode; not observed here (see `ChatTitleMenu`).
    let chat: ChatController
    let mode: ReplyMode
    let isTemporary: Bool
    /// The conversation, once it can be shared.
    let shareable: Conversation?

    @State private var chosenDirectory = PieRuntimeConfig.selected.directory
    @State private var restartTarget: PieRuntimeConfig.Model?

    // The title follows Dynamic Type up to what the bar's 44 points can
    // hold, the caption line included; bars do not grow past that.
    @ScaledMetric(relativeTo: .title3) private var titleSize: CGFloat = 21
    @ScaledMetric(relativeTo: .body) private var modeSize: CGFloat = 17
    @ScaledMetric(relativeTo: .caption) private var captionSize: CGFloat = 12

    static func == (lhs: TitleMenu, rhs: TitleMenu) -> Bool {
        lhs.mode == rhs.mode && lhs.isTemporary == rhs.isTemporary && lhs.shareable == rhs.shareable
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
            if let shareable {
                Section {
                    ShareLink(item: MessageRendering.shareText(shareable)) {
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
            Button("Restart") { Self.quit() }
            Button("Later", role: .cancel) {}
        } message: {
            Text("The model loads when Pie Voice starts. Your chats are saved.")
        }
    }

    /// The mode's name crossfades to the new one, and "Temporary chat"
    /// fades in or out under the title while the title eases up or down
    /// to make room.
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
            if isTemporary {
                Text("Temporary chat")
                    .font(.system(size: min(captionSize, 13), weight: .medium))
                    .opacity(0.78)
                    .transition(.opacity)
            }
        }
        .lineLimit(1)
        .foregroundStyle(Theme.onTopBar)
        .padding(.horizontal, 10)
        .frame(minHeight: 44)
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

    /// Quits so the next launch loads the chosen model. Chats still being
    /// written are finished first, and the app fades out rather than
    /// vanishing, so the quit reads as deliberate rather than as a crash.
    private static func quit() {
        ChatStore.finishPendingWrites()
        let window = UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow }
            .first
        guard let window else { exit(0) }
        UIView.animate(withDuration: 0.25, animations: { window.alpha = 0 }) { _ in
            exit(0)
        }
    }
}
