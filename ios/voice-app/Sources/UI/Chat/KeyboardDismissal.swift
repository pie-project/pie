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

    /// Whether hides on this device are reported with zero duration. The
    /// workaround above is only right where the keyboard really leaves in
    /// one frame; where it slides (iOS 18, and possibly a real iOS 26
    /// phone), removing SwiftUI's animation would drop the composer ahead
    /// of the keyboard. The path has to be chosen before the field
    /// resigns, so it follows the last hide UIKit reported, from any
    /// cause. Until a hide has been seen, iOS 26 assumes the one-frame
    /// hide measured in its Simulator.
    private static var zeroLengthHides: Bool {
        if let lastHideDuration { return lastHideDuration == 0 }
        if #available(iOS 26, *) { return true }
        return false
    }

    private static var lastHideDuration: Double?
    private static var observer: NSObjectProtocol?

    /// Listens for every keyboard hide, once per process, and logs the
    /// first duration so a phone's behaviour can be read off
    /// `pie-console.log`.
    static func startListening() {
        guard observer == nil else { return }
        observer = NotificationCenter.default.addObserver(
            forName: UIResponder.keyboardWillHideNotification, object: nil, queue: .main
        ) { note in
            let duration = note.userInfo?[UIResponder.keyboardAnimationDurationUserInfoKey] as? Double
            MainActor.assumeIsolated {
                if lastHideDuration == nil, let duration {
                    print("[ui] keyboard hide reported \(duration) s")
                }
                lastHideDuration = duration ?? lastHideDuration
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
