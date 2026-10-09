import UIKit

/// Hides the keyboard so the composer lands with it, never after it.
///
/// Measured in the iOS 26 Simulator (Xcode 26.0.1): when a SwiftUI text
/// field loses focus, by `@FocusState` or by `resignFirstResponder`, UIKit
/// posts its will-hide notification with a duration of zero and the
/// keyboard is gone in one frame. SwiftUI's keyboard avoidance still moves
/// the layout down on a ~0.45 s curve of its own, so for a third of a
/// second the composer floated in mid-screen over nothing and then slid
/// down. On iOS 18 the same dismissal reports 0.25 s, the keyboard slides,
/// and SwiftUI rides it exactly; there this changes nothing.
///
/// Two things make the composer land in the keyboard's frame on iOS 26:
/// resigning inside `performWithoutAnimation`, so SwiftUI applies the new
/// safe area without its fallback animation, and laying the window out
/// right away. Without that second step the safe-area change waited for
/// the next update and was swept into whatever the tap animated next (the
/// send's `withAnimation`), and the composer slid on that curve instead.
@MainActor
enum KeyboardDismissal {
    static func dismiss() {
        startListening()
        if zeroLengthHides {
            // The keyboard leaves in one frame here: resign inside
            // performWithoutAnimation so SwiftUI applies the new safe area
            // without its fallback animation, and lay out at once so the
            // change is not swept into the caller's next animation.
            UIView.performWithoutAnimation {
                if resign() { layOutNow() }
            }
        } else if resign() {
            // Applies the keyboard's own animation now, before the caller's
            // state changes, so the composer follows the keyboard's curve.
            layOutNow()
        }
    }

    // MARK: - How this device hides the keyboard

    /// Whether to use the one-frame workaround above. Only in the
    /// Simulator, where the zero-length hide was measured. It cannot be
    /// learned at run time: a hide made inside performWithoutAnimation is
    /// itself reported with zero duration, so the workaround would teach
    /// itself and a phone whose keyboard slides would then snap forever.
    /// On a device the keyboard keeps the system's own animation and
    /// SwiftUI rides it; the first hide's duration is logged, so the
    /// phone's real behaviour can be read off `pie-console.log`.
    private static var zeroLengthHides: Bool {
        #if targetEnvironment(simulator)
        if #available(iOS 26, *) { return true }
        #endif
        return false
    }

    private static var didLogHide = false
    private static var observer: NSObjectProtocol?

    /// Logs the first keyboard hide of the process, whatever caused it.
    static func startListening() {
        guard observer == nil else { return }
        observer = NotificationCenter.default.addObserver(
            forName: UIResponder.keyboardWillHideNotification, object: nil, queue: .main
        ) { note in
            let duration = note.userInfo?[UIResponder.keyboardAnimationDurationUserInfoKey] as? Double
            MainActor.assumeIsolated {
                guard !didLogHide, let duration else { return }
                didLogHide = true
                print("[ui] keyboard hide reported \(duration) s (workaround \(zeroLengthHides ? "on" : "off"))")
            }
        }
    }

    /// False when nothing took the action, so there was nothing to lay out.
    private static func resign() -> Bool {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    private static func layOutNow() {
        let window = UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow }
            .first
        window?.layoutIfNeeded()
    }
}
