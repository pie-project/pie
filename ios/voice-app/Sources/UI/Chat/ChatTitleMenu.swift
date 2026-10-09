import SwiftUI

/// The top bar's title: "Pie" in the site's serif and the current mode,
/// with "Temporary chat" under it while the open chat is one. It opens
/// ChatGPT's model menu: a Mode section (Instant, Thinking), a Model
/// section listing the ladder, where picking another installed rung takes
/// effect on the next launch, and sharing the open chat.
struct ChatTitleMenu: View {
    @EnvironmentObject private var chat: ChatController

    @State private var chosenDirectory = PieRuntimeConfig.selected.directory
    @State private var restartTarget: PieRuntimeConfig.Model?

    // The title follows Dynamic Type up to what the bar's 44 points can
    // hold, the caption line included; bars do not grow past that.
    @ScaledMetric(relativeTo: .title3) private var titleSize: CGFloat = 21
    @ScaledMetric(relativeTo: .body) private var modeSize: CGFloat = 17
    @ScaledMetric(relativeTo: .caption) private var captionSize: CGFloat = 12

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
            // Rendering the whole chat is left until it has stopped
            // changing: this menu is rebuilt on every streamed token.
            if !chat.conversation.isEmpty && !chat.isGenerating {
                Section {
                    ShareLink(item: chat.shareText()) {
                        Label("Share chat", systemImage: "square.and.arrow.up")
                    }
                }
            }
        } label: {
            VStack(spacing: 0) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("Pie")
                        .font(Theme.serif(min(titleSize, 23)))
                    Text(chat.mode.title)
                        .font(.system(size: min(modeSize, 20), weight: .regular))
                        .opacity(0.78)
                    Image(systemName: "chevron.down")
                        .font(.system(size: min(titleSize, 23) * 0.52, weight: .bold))
                        .opacity(0.78)
                }
                if chat.conversation.isTemporary {
                    Text("Temporary chat")
                        .font(.system(size: min(captionSize, 13), weight: .medium))
                        .opacity(0.78)
                }
            }
            .lineLimit(1)
            .foregroundStyle(Theme.onTopBar)
            .padding(.horizontal, 10)
            .frame(minHeight: 44)
            .contentShape(Rectangle())
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
            Button("Restart") { exit(0) }
            Button("Later", role: .cancel) {}
        } message: {
            Text("The model loads when Pie Voice starts. Your chats are saved.")
        }
    }

    private var accessibilityTitle: String {
        let title = "Pie, \(chat.mode.title)"
        return chat.conversation.isTemporary ? title + ", temporary chat" : title
    }

    private func modeBinding(_ mode: ReplyMode) -> Binding<Bool> {
        Binding(
            get: { chat.mode == mode },
            set: { isOn in if isOn { chat.mode = mode } }
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
}
