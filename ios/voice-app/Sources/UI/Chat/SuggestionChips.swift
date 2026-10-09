import PhotosUI
import SwiftUI

/// ChatGPT's starter chips above the composer of an empty chat, in two
/// rows that scroll sideways together. Most send a complete prompt; "Read
/// a photo" opens the photo picker and "Talk it through" opens voice mode.
struct SuggestionChips: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var settings: AppSettings

    @State private var photoItem: PhotosPickerItem?

    var body: some View {
        if isShown {
            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 8) {
                    chipRow(Suggestion.all.enumerated().filter { $0.offset % 2 == 0 })
                    chipRow(Suggestion.all.enumerated().filter { $0.offset % 2 == 1 })
                }
                .padding(.horizontal, 16)
            }
            .padding(.bottom, 8)
            .transition(.opacity)
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
    }

    /// Only while the composer is empty, as in ChatGPT: they make way
    /// as soon as the user starts a message of their own.
    private var isShown: Bool {
        chat.draft.isEmpty && chat.pendingAttachments.isEmpty && !chat.isImportingAttachment
    }

    private var canSendPrompt: Bool {
        chat.engineState == .ready && !chat.isGenerating
    }

    private func chipRow(_ suggestions: [EnumeratedSequence<[Suggestion]>.Element]) -> some View {
        HStack(spacing: 8) {
            ForEach(suggestions, id: \.element.title) { offset, suggestion in
                chip(suggestion, tint: Suggestion.tints[offset % Suggestion.tints.count])
            }
        }
    }

    @ViewBuilder
    private func chip(_ suggestion: Suggestion, tint: Color) -> some View {
        switch suggestion.action {
        case .prompt(let prompt):
            Button {
                ChatHaptics.messageSent(enabled: settings.haptics)
                chat.send(text: prompt)
            } label: {
                chipLabel(suggestion, tint: tint)
            }
            .buttonStyle(.plain)
            .disabled(!canSendPrompt)
            .opacity(canSendPrompt ? 1 : 0.5)
        case .photo:
            PhotosPicker(selection: $photoItem, matching: .images, photoLibrary: .shared()) {
                chipLabel(suggestion, tint: tint)
            }
            .buttonStyle(.plain)
        case .voice:
            Button {
                router.isVoiceModePresented = true
            } label: {
                chipLabel(suggestion, tint: tint)
            }
            .buttonStyle(.plain)
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
        .background(Theme.background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(Theme.hairline, lineWidth: 1)
        }
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
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
