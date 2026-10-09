import SwiftUI
import UIKit

/// The drawer's pan, as ChatGPT has it: a sideways drag almost anywhere on
/// the chat opens the sidebar, and one to the left on the sidebar or the
/// dimmed chat closes it, with the drawer under the finger the whole way.
///
/// It is a UIKit recognizer because it has to share the screen with
/// scroll views. A SwiftUI drag on a view that contains a scroll view
/// either blocks its scrolling or is blocked by it; a UIKit recognizer is
/// asked, before it starts, whether this touch is its to take. It takes a
/// drag only when the movement is clearly sideways (more than 1.2 times as
/// wide as it is tall) and in a direction the drawer can go, so scrolling
/// the chat or the history up and down always wins. It never takes a touch
/// that lands in something that scrolls sideways (code blocks, tables, the
/// suggestion chips), so those keep their own drags.
@available(iOS 18.0, *)
struct SidebarPanGesture: UIGestureRecognizerRepresentable {
    /// The drawer's position when a drag is about to start (0 shut, 1
    /// open), to tell which ways it can go.
    let position: () -> CGFloat
    let onBegan: () -> Void
    /// The finger's sideways travel since the drag began, in points.
    let onChanged: (CGFloat) -> Void
    /// The finger's sideways speed at release, in points per second; 0
    /// when the system cancelled the drag.
    let onEnded: (CGFloat) -> Void

    func makeCoordinator(converter: CoordinateSpaceConverter) -> Coordinator {
        Coordinator()
    }

    func makeUIGestureRecognizer(context: Context) -> UIPanGestureRecognizer {
        let pan = UIPanGestureRecognizer()
        pan.maximumNumberOfTouches = 1
        pan.delegate = context.coordinator
        return pan
    }

    func updateUIGestureRecognizer(_ recognizer: UIPanGestureRecognizer, context: Context) {
        context.coordinator.position = position
    }

    func handleUIGestureRecognizerAction(_ pan: UIPanGestureRecognizer, context: Context) {
        switch pan.state {
        case .began:
            // The recognizer starts only after the finger has moved a few
            // points. Counting from here rather than from touch-down means
            // the drawer does not jump by that distance on the first frame.
            pan.setTranslation(.zero, in: pan.view)
            onBegan()
        case .changed:
            onChanged(pan.translation(in: pan.view).x)
        case .ended:
            onEnded(pan.velocity(in: pan.view).x)
        case .cancelled, .failed:
            onEnded(0)
        default:
            break
        }
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var position: () -> CGFloat = { 0 }

        func gestureRecognizerShouldBegin(_ recognizer: UIGestureRecognizer) -> Bool {
            guard let pan = recognizer as? UIPanGestureRecognizer else { return false }
            var movement = pan.translation(in: pan.view)
            if movement == .zero { movement = pan.velocity(in: pan.view) }
            guard abs(movement.x) > abs(movement.y) * 1.2 else { return false }
            let position = position()
            return movement.x > 0 ? position < 1 : position > 0
        }

        func gestureRecognizer(_ recognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
            var view = touch.view
            while let current = view, current !== recognizer.view {
                if let scrollView = current as? UIScrollView,
                   scrollView.contentSize.width > scrollView.bounds.width + 1 {
                    return false
                }
                view = current.superview
            }
            return true
        }
    }
}
