import UIKit

/// The status bar's style while one of the app's light screens is under
/// it: the open sidebar, and voice mode.
///
/// The chat's blue top bar makes the clock and battery white (see
/// `ChatTopBar`). The sidebar and voice mode are light in light mode,
/// where white text disappears, so over them it has to be dark. SwiftUI's
/// only switch for that is the navigation bar's color scheme, and
/// flipping it restyles the whole bar: every drawer slide stalled for
/// 35-70 ms at the halfway point, where the flip happened (recorded).
///
/// UIKit has another switch: the frontmost window decides the status
/// bar's style. So while the drawer is out or voice mode is up, this puts
/// an empty, transparent window over the app whose only job is to answer
/// "which style". It takes no touches (they fall through to the app's
/// window) and VoiceOver does not see it. Changing its answer crossfades
/// the clock and battery and redraws nothing of the app.
///
/// The rest of the time the window is down, and the app's own bars and
/// presentations decide as before: white under the blue bar, white over a
/// sheet's dark backdrop, no status bar under the camera.
@MainActor
enum StatusBarOverride {
    enum Style: Equatable {
        /// No override: the window is down and the app decides.
        case app
        /// White text, as the blue bar has it. The window is up already
        /// (the drawer has started to move), so the sidebar reaching the
        /// clock is only a change of answer.
        case light
        /// Dark text, for a light screen under the status bar.
        case dark
    }

    /// How long the clock and battery take to change color. Short: the
    /// change happens as the sidebar's edge passes the clock, and at 0.25 s
    /// the clock was half-faded, so nearly invisible, for about 0.1 s.
    private static let fade: TimeInterval = 0.08

    private static var current: Style = .app
    private static var window: UIWindow?
    private static let controller = StyleController()

    static func apply(_ style: Style) {
        guard style != current else { return }
        current = style
        switch style {
        case .app:
            window?.isHidden = true
            // The app's own window decides again.
            let appRoot = appWindow()?.rootViewController
            UIView.animate(withDuration: fade) { appRoot?.setNeedsStatusBarAppearanceUpdate() }
        case .light, .dark:
            guard let window = window ?? makeWindow() else { return }
            controller.style = style == .dark ? .darkContent : .lightContent
            window.isHidden = false
            UIView.animate(withDuration: fade) { controller.setNeedsStatusBarAppearanceUpdate() }
        }
    }

    /// Above the app's window, below the keyboard and the system's own.
    /// Never made key, so typing and focus stay with the app.
    private static func makeWindow() -> UIWindow? {
        guard let scene = appWindow()?.windowScene else { return nil }
        let window = PassThroughWindow(windowScene: scene)
        window.windowLevel = UIWindow.Level(rawValue: UIWindow.Level.normal.rawValue + 1)
        window.backgroundColor = .clear
        window.isUserInteractionEnabled = false
        window.accessibilityElementsHidden = true
        window.rootViewController = controller
        self.window = window
        return window
    }

    /// The app's own window: the key one, which this one never is.
    private static func appWindow() -> UIWindow? {
        UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow }
            .first { $0 !== window }
    }
}

/// Lets every touch through to the window below.
private final class PassThroughWindow: UIWindow {
    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        nil
    }
}

/// The override window's only content: an empty view, and the answer to
/// "which status bar style".
private final class StyleController: UIViewController {
    var style: UIStatusBarStyle = .lightContent

    override var preferredStatusBarStyle: UIStatusBarStyle { style }

    override func loadView() {
        let view = UIView()
        view.backgroundColor = .clear
        view.isUserInteractionEnabled = false
        view.accessibilityElementsHidden = true
        self.view = view
    }
}
