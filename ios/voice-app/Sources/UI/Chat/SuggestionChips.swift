import PhotosUI
import SwiftUI

/// ChatGPT's starter chips above the composer of an empty chat, in two
/// rows that scroll sideways together. Most send a complete prompt; "Read
/// a photo" opens the photo picker and "Talk it through" opens voice mode.
///
/// When the empty chat appears, the chips fade in rising, one after
/// another from the left. They fade away the moment the user starts a
/// message of their own, and come back when the composer is emptied.
struct SuggestionChips: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var settings: AppSettings

    /// Whether the rows show, copied from `wantsRows` inside an animation:
    /// the draft changes with each keystroke outside any animation, and
    /// the rows (and the greeting re-centring above them) would jump.
    /// Nil until the first change, when `wantsRows` is used as it is.
    @State private var showsRows: Bool?
    @State private var isPickingPhoto = false
    @State private var photoItem: PhotosPickerItem?

    var body: some View {
        VStack(spacing: 0) {
            if showsRows ?? wantsRows {
                rows
                    // Arriving, each chip runs its own staggered entrance
                    // (`ChipEntrance`); leaving, the rows fade as one.
                    .transition(.asymmetric(insertion: .identity, removal: .opacity))
            }
        }
        .onChange(of: wantsRows) { _, wants in
            withMotion(wants ? Motion.fadeIn : Motion.fadeOut) { showsRows = wants }
        }
        .photosPicker(isPresented: $isPickingPhoto, selection: $photoItem, matching: .images, photoLibrary: .shared())
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            photoItem = nil
            let chat = self.chat
            Task {
                guard let data = try? await item.loadTransferable(type: Data.self) else {
                    chat.banner = "Couldn't load that photo"
                    return
                }
                await chat.addPhoto(data)
            }
        }
    }

    private var rows: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 8) {
                chipRow(Suggestion.all.enumerated().filter { $0.offset % 2 == 0 })
                chipRow(Suggestion.all.enumerated().filter { $0.offset % 2 == 1 })
            }
            .padding(.horizontal, 16)
        }
        .padding(.bottom, 8)
    }

    /// Only while the composer is empty, as in ChatGPT: they make way as
    /// soon as the user starts a message of their own. Spaces alone do
    /// not count, as they do not for the send button.
    private var wantsRows: Bool {
        chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && chat.pendingAttachments.isEmpty
            && !chat.isImportingAttachment
    }

    private var canSendPrompt: Bool {
        chat.engineState == .ready && !chat.isGenerating
    }

    private func chipRow(_ suggestions: [EnumeratedSequence<[Suggestion]>.Element]) -> some View {
        HStack(spacing: 8) {
            ForEach(suggestions, id: \.element.title) { offset, suggestion in
                chip(suggestion, tint: Suggestion.tints[offset % Suggestion.tints.count])
                    // Left to right: the rows alternate, so the offset order
                    // runs down each column, then on to the next.
                    .modifier(ChipEntrance(delay: Double(offset) * 0.04))
            }
        }
    }

    @ViewBuilder
    private func chip(_ suggestion: Suggestion, tint: Color) -> some View {
        switch suggestion.action {
        case .prompt(let prompt):
            Button {
                Haptics.tap(enabled: settings.haptics)
                // As the send button does, so the composer lands with the
                // keyboard before the message goes out.
                KeyboardDismissal.dismiss()
                chat.send(text: prompt)
            } label: {
                chipLabel(suggestion, tint: tint)
            }
            .buttonStyle(ChipButtonStyle())
            .disabled(!canSendPrompt)
            .opacity(canSendPrompt ? 1 : 0.5)
            .animation(Motion.crossfade, value: canSendPrompt)
        case .photo:
            Button {
                Haptics.tap(enabled: settings.haptics)
                KeyboardDismissal.dismiss()
                isPickingPhoto = true
            } label: {
                chipLabel(suggestion, tint: tint)
            }
            .buttonStyle(ChipButtonStyle())
        case .voice:
            Button {
                Haptics.tap(enabled: settings.haptics)
                KeyboardDismissal.dismiss()
                router.isVoiceModePresented = true
            } label: {
                chipLabel(suggestion, tint: tint)
            }
            .buttonStyle(ChipButtonStyle())
        }
    }

    private func chipLabel(_ suggestion: Suggestion, tint: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: suggestion.symbol)
                .foregroundStyle(tint)
            Text(suggestion.title)
                .foregroundStyle(Theme.ink)
        }
        .font(.subheadline.weight(.medium))
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Theme.background, in: RoundedRectangle(cornerRadius: ChipButtonStyle.cornerRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: ChipButtonStyle.cornerRadius, style: .continuous)
                .strokeBorder(Theme.hairline, lineWidth: 1)
        }
        .contentShape(RoundedRectangle(cornerRadius: ChipButtonStyle.cornerRadius, style: .continuous))
    }
}

/// A chip pressed: it shrinks a touch and its fill darkens (lightens in
/// dark mode), then springs back, as ChatGPT's suggestion cards do.
private struct ChipButtonStyle: ButtonStyle {
    static let cornerRadius: CGFloat = 18

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .overlay {
                RoundedRectangle(cornerRadius: Self.cornerRadius, style: .continuous)
                    .fill(Theme.ink.opacity(configuration.isPressed ? 0.06 : 0))
            }
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.97 : 1)
            .animation(.snappy(duration: 0.18), value: configuration.isPressed)
    }
}

/// A chip's entrance when its rows appear: a fade while rising 10 pt,
/// `delay` after the first chip's. Reduce Motion keeps the fade only.
private struct ChipEntrance: ViewModifier {
    let delay: Double

    @State private var hasAppeared = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content
            .opacity(hasAppeared ? 1 : 0)
            .offset(y: hasAppeared || reduceMotion ? 0 : 10)
            .onAppear {
                withAnimation(Animation.smooth(duration: 0.35).delay(delay)) { hasAppeared = true }
            }
    }
}

/// One starter chip.
private struct Suggestion {
    enum Action {
        case prompt(String)
        case photo
        case voice
    }

    let title: String
    let symbol: String
    let action: Action

    /// The site's three accents, in rotation along each row.
    static let tints: [Color] = [Theme.accent, Theme.orange, Theme.red]

    static let all: [Suggestion] = [
        Suggestion(
            title: "Explain a concept",
            symbol: "graduationcap",
            action: .prompt("Explain how a neural network learns, in plain words, with one everyday example.")
        ),
        Suggestion(
            title: "Help me write",
            symbol: "pencil.line",
            action: .prompt("Help me write a short, friendly email asking my professor for a meeting next week.")
        ),
        Suggestion(
            title: "Brainstorm ideas",
            symbol: "lightbulb",
            action: .prompt("Brainstorm eight ideas for a weekend project I could finish in two days.")
        ),
        Suggestion(
            title: "Summarize text",
            symbol: "text.alignleft",
            action: .prompt("Summarize the main ideas of the theory of evolution in five short bullet points.")
        ),
        Suggestion(
            title: "Make a plan",
            symbol: "checklist",
            action: .prompt("Make a simple four-week plan to start running three times a week.")
        ),
        Suggestion(
            title: "Quiz me",
            symbol: "questionmark.bubble",
            action: .prompt("Quiz me on world capitals: ask five questions one at a time and tell me if I'm right.")
        ),
        Suggestion(title: "Read a photo", symbol: "photo", action: .photo),
        Suggestion(title: "Talk it through", symbol: "waveform", action: .voice),
    ]
}
