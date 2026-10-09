import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// The composer's "+" sheet, laid out like ChatGPT's: Camera, Photos and
/// Files tiles, then the thinking toggle and the ways into voice mode.
struct AttachmentSheet: View {
    /// Runs once the sheet has gone, for choices that present something of
    /// their own (voice mode is a full-screen cover on the root view).
    @Binding var afterDismiss: (() -> Void)?

    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var voice: VoiceModeController

    @State private var photoItem: PhotosPickerItem?
    @State private var isImportingFile = false
    @State private var isCameraPresented = false

    /// What the importer can turn into text for the model.
    private static let fileTypes: [UTType] = [.pdf, .plainText, .text, .json, .commaSeparatedText, .sourceCode]

    var body: some View {
        // Scrolls so that at the larger text sizes the rows wrap instead of
        // being squeezed into the medium detent and truncated.
        ScrollView {
            content
        }
        .scrollBounceBehavior(.basedOnSize)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        .presentationBackground(Theme.background)
        .fileImporter(isPresented: $isImportingFile, allowedContentTypes: Self.fileTypes) { result in
            importFile(result)
        }
        .fullScreenCover(isPresented: $isCameraPresented) {
            CameraPicker { image in
                let chat = self.chat
                isCameraPresented = false
                router.isAttachmentSheetPresented = false
                Task {
                    guard let data = await CameraPicker.jpegData(from: image) else {
                        chat.banner = "Couldn't read that photo"
                        return
                    }
                    await chat.addPhoto(data)
                }
            } onCancel: {
                isCameraPresented = false
            }
            .ignoresSafeArea()
        }
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            importPhoto(item)
        }
    }

    private var content: some View {
        VStack(spacing: 14) {
            HStack(spacing: 10) {
                if CameraPicker.isAvailable {
                    tile("Camera", symbol: "camera") { isCameraPresented = true }
                }
                PhotosPicker(selection: $photoItem, matching: .images, photoLibrary: .shared()) {
                    tileLabel("Photos", symbol: "photo.on.rectangle")
                }
                .buttonStyle(Self.tileStyle)
                .accessibilityLabel("Photos")
                tile("Files", symbol: "folder") { isImportingFile = true }
            }

            VStack(spacing: 0) {
                Toggle(isOn: thinkLonger) {
                    rowLabel("Think longer", subtitle: ReplyMode.thinking.subtitle, symbol: "lightbulb")
                }
                .tint(Theme.accentFill)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)

                divider
                row("Talk to Pie", subtitle: "Voice mode, entirely on this iPhone", symbol: "waveform") {
                    let router = self.router
                    leave { router.isVoiceModePresented = true }
                }

                if voice.hasSampleQuestions {
                    divider
                    row("Ask a sample question", subtitle: "Plays a recorded question, then answers it", symbol: "play.circle") {
                        let router = self.router
                        let voice = self.voice
                        leave {
                            router.isVoiceModePresented = true
                            // A sample is played into an active session
                            // only. Voice mode's own start on appearing is
                            // then a no-op, and does not cut the sample off.
                            voice.start()
                            voice.askSampleQuestion()
                        }
                    }
                }
            }
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            // Keeps a pressed row's highlight inside the rounded corners.
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .padding(.horizontal, 16)
        .padding(.top, 28)
        .padding(.bottom, 16)
    }

    // MARK: - Actions

    private var thinkLonger: Binding<Bool> {
        Binding(
            get: { chat.mode == .thinking },
            set: { chat.mode = $0 ? .thinking : .instant }
        )
    }

    private func leave(then action: @escaping () -> Void) {
        afterDismiss = action
        router.isAttachmentSheetPresented = false
    }

    private func importPhoto(_ item: PhotosPickerItem) {
        let chat = self.chat
        router.isAttachmentSheetPresented = false
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self) else {
                chat.banner = "Couldn't load that photo"
                return
            }
            await chat.addPhoto(data)
        }
    }

    private func importFile(_ result: Result<URL, Error>) {
        let chat = self.chat
        router.isAttachmentSheetPresented = false
        switch result {
        case .success(let url):
            Task {
                // A file picked from outside the app's container can only be
                // read inside a security-scoped access window.
                let isScoped = url.startAccessingSecurityScopedResource()
                await chat.addFile(at: url)
                if isScoped { url.stopAccessingSecurityScopedResource() }
            }
        case .failure:
            chat.banner = "Couldn't open that file"
        }
    }

    // MARK: - Pieces

    /// Tiles are cards: pressed, they shrink a touch, as ChatGPT's do.
    private static let tileStyle = PressScaleButtonStyle(pressedScale: 0.97)

    private func tile(_ title: String, symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            tileLabel(title, symbol: symbol)
        }
        .buttonStyle(Self.tileStyle)
        .accessibilityLabel(title)
    }

    private func tileLabel(_ title: String, symbol: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: symbol)
                .font(.system(size: 22))
                .foregroundStyle(Theme.accent)
                .frame(height: 28)
            Text(title)
                .font(.footnote.weight(.medium))
                .foregroundStyle(Theme.ink)
        }
        .frame(maxWidth: .infinity, minHeight: 84)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func row(_ title: String, subtitle: String, symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                rowLabel(title, subtitle: subtitle, symbol: symbol)
                Spacer(minLength: 8)
                Image(systemName: "chevron.right")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Theme.tertiaryInk)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .contentShape(Rectangle())
        }
        // The gray pressed fill of a list row.
        .buttonStyle(PressHighlightButtonStyle(cornerRadius: 0))
    }

    private func rowLabel(_ title: String, subtitle: String, symbol: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: symbol)
                .font(.system(size: 18))
                .foregroundStyle(Theme.accent)
                .frame(width: 26)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body)
                    .foregroundStyle(Theme.ink)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(Theme.secondaryInk)
            }
        }
    }

    private var divider: some View {
        Rectangle()
            .fill(Theme.hairline)
            .frame(height: 0.5)
            .padding(.leading, 52)
    }
}
