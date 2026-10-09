import Foundation
import SwiftUI

/// Which screens are showing. Owned by the composition root and shared
/// through the environment, so the sidebar, the top bar, the composer and
/// the screenshot tour all drive navigation the same way.
final class AppRouter: ObservableObject {
    /// Whether the sidebar is open or opening. The drawer slides to match
    /// whoever sets it (see `RootView`), so callers just set it. A tap
    /// should go through `setSidebarOpen(_:)`, which drops the keyboard
    /// first.
    @Published var isSidebarOpen = false {
        didSet {
            // Opened again before it finished closing: whatever was
            // waiting for the close is no longer wanted.
            if isSidebarOpen { afterSidebarCloses = nil }
        }
    }
    @Published var isSettingsPresented = false
    /// The "+" sheet in the composer: camera, photos, files, modes.
    @Published var isAttachmentSheetPresented = false

    /// Whether voice mode is up. Setting it crossfades voice mode in or
    /// out over the chat, whoever sets it.
    ///
    /// Voice mode is a layer over the chat (`RootView`), not a
    /// presentation, so it moves only if the change is made inside an
    /// animation. The animation lives here rather than at each of the
    /// half-dozen places that open or close it (the composer, the chips,
    /// the "+" sheet, the sidebar, voice mode's own End button, the tour),
    /// so none of them can forget it. A caller's own animation around the
    /// change is replaced by this one, which is the same 0.35 s ease.
    var isVoiceModePresented: Bool {
        get { isVoiceModeShown }
        set {
            guard newValue != isVoiceModeShown else { return }
            withMotion(Self.voiceModeCrossfade) { isVoiceModeShown = newValue }
        }
    }
    @Published private(set) var isVoiceModeShown = false

    /// ChatGPT's separate voice mode fades in and out over the chat
    /// (about 0.35 s) rather than sliding up as a cover.
    static let voiceModeCrossfade = Animation.easeInOut(duration: 0.35)

    /// Run once the drawer has slid shut; see `closeSidebar(then:)`.
    private var afterSidebarCloses: (() -> Void)?

    /// Closes the sidebar, then runs `action` once the drawer is shut. For
    /// putting something over the whole screen from the sidebar (voice
    /// mode): arriving while the drawer still slides sideways under it,
    /// the two motions fight.
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

extension AppRouter {
    /// Opens or closes the sidebar from a tap. The keyboard (the
    /// composer's, or the sidebar search's) goes first, at the tap, as
    /// ChatGPT drops it: hidden from here, the layout lands with it in the
    /// same frame. Hidden later, from the drawer's reaction to the flag,
    /// SwiftUI moved the layout on its own slower curve, and the composer
    /// and the sidebar's footer floated over empty space for a third of a
    /// second (recorded).
    @MainActor
    func setSidebarOpen(_ open: Bool) {
        KeyboardDismissal.dismiss()
        isSidebarOpen = open
    }
}
