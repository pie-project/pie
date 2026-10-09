import Foundation

/// Which screens are showing. Owned by the composition root and shared
/// through the environment, so the sidebar, the top bar, the composer and
/// the screenshot tour all drive navigation the same way.
final class AppRouter: ObservableObject {
    /// Whether the sidebar is open or opening. The drawer slides to match
    /// whoever sets it (see `RootView`), so callers just set it.
    @Published var isSidebarOpen = false {
        didSet {
            // Opened again before it finished closing: whatever was
            // waiting for the close is no longer wanted.
            if isSidebarOpen { afterSidebarCloses = nil }
        }
    }
    @Published var isSettingsPresented = false
    @Published var isVoiceModePresented = false
    /// The "+" sheet in the composer: camera, photos, files, modes.
    @Published var isAttachmentSheetPresented = false

    /// Run once the drawer has slid shut; see `closeSidebar(then:)`.
    private var afterSidebarCloses: (() -> Void)?

    /// Closes the sidebar, then runs `action` once the drawer is shut. For
    /// presenting something that covers the screen from the sidebar (voice
    /// mode): rising while the drawer still slides sideways under it, the
    /// two motions fight.
    func closeSidebar(then action: @escaping () -> Void) {
        guard isSidebarOpen else {
            action()
            return
        }
        isSidebarOpen = false
        afterSidebarCloses = action
    }

    /// The drawer calls this each time it arrives at the shut position.
    func sidebarDidClose() {
        let action = afterSidebarCloses
        afterSidebarCloses = nil
        action?()
    }
}
