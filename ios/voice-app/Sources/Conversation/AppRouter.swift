import Foundation

/// Which screens are showing. Owned by the composition root and shared
/// through the environment, so the sidebar, the top bar, the composer and
/// the screenshot tour all drive navigation the same way.
final class AppRouter: ObservableObject {
    @Published var isSidebarOpen = false
    @Published var isSettingsPresented = false
    @Published var isVoiceModePresented = false
    /// The "+" sheet in the composer: camera, photos, files, modes.
    @Published var isAttachmentSheetPresented = false
}
