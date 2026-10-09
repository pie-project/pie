import XCTest

/// Drives Pie Voice through every animated interaction with real touches,
/// on the demo backend (`-PieDemoBackend 1`), so a screen recording of the
/// Simulator shows each animation as a person would see it. Not a test of
/// correctness: a step whose element is missing is logged and skipped.
///
/// Each step prints `MOTION <unix time> <name>`; `motion-tour.sh` records
/// the Simulator while this runs and cuts the video at those marks.
///
///   bash ios/PieVoice/motion-tour.sh
final class MotionTourUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUp() {
        continueAfterFailure = true
        app = XCUIApplication()
        app.launchArguments = ["-PieDemoBackend", "1"]
    }

    private func mark(_ name: String) {
        print(String(format: "MOTION %.3f %@", Date().timeIntervalSince1970, name))
    }

    private func pause(_ seconds: TimeInterval) {
        Thread.sleep(forTimeInterval: seconds)
    }

    @discardableResult
    private func tap(_ element: XCUIElement, _ name: String, wait: TimeInterval = 4) -> Bool {
        guard element.waitForExistence(timeout: wait), element.isHittable else {
            print("MOTION-SKIP \(name): not found or not hittable")
            return false
        }
        mark(name)
        element.tap()
        return true
    }

    /// Taps the first on-screen button matching `predicate` (the sidebar
    /// keeps off-screen buttons with the same labels as the chat's).
    @discardableResult
    private func tapButton(_ predicate: String, _ name: String, wait: TimeInterval = 4) -> Bool {
        let query = app.buttons.matching(NSPredicate(format: predicate))
        let deadline = Date().addingTimeInterval(wait)
        repeat {
            if let hit = query.allElementsBoundByIndex.first(where: { $0.exists && $0.isHittable }) {
                mark(name)
                hit.tap()
                return true
            }
            pause(0.2)
        } while Date() < deadline
        print("MOTION-SKIP \(name): no hittable button for \(predicate)")
        return false
    }

    /// Accepts a permission prompt (speech recognition cannot be granted
    /// ahead of time in the Simulator) if one is showing.
    private func allowSystemAlert() {
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        for label in ["Allow", "OK", "Allow While Using App"] {
            let button = springboard.alerts.buttons[label].firstMatch
            if button.waitForExistence(timeout: 1.5) {
                button.tap()
                pause(0.8)
            }
        }
    }

    private func button(_ label: String) -> XCUIElement {
        app.buttons[label].firstMatch
    }

    /// A drag at a finger's pace from one point of the window to another.
    private func drag(from: CGVector, to: CGVector, velocity: XCUIGestureVelocity = .default, _ name: String) {
        let window = app.windows.firstMatch
        let start = window.coordinate(withNormalizedOffset: from)
        let end = window.coordinate(withNormalizedOffset: to)
        mark(name)
        start.press(forDuration: 0.05, thenDragTo: end, withVelocity: velocity, thenHoldForDuration: 0.05)
    }

    /// Prints the accessibility tree at the screens the tour visits, for
    /// fixing its element queries.
    func testDumpHierarchy() {
        app.launch()
        pause(2.5)
        print("DUMP-BEGIN empty\n\(app.debugDescription)\nDUMP-END")
        let field = app.textFields["Ask anything"].exists ? app.textFields["Ask anything"] : app.textViews.firstMatch
        field.tap()
        field.typeText("Explain this")
        button("Send").tap()
        pause(9)
        print("DUMP-BEGIN replied\n\(app.debugDescription)\nDUMP-END")
    }

    /// Just the drawer, for quick iterations on it.
    func testSidebar() {
        mark("launch")
        app.launch()
        pause(2.0)
        if tapButton("label == 'Open sidebar'", "sidebar-open-button") { pause(1.2) }
        if tapButton("label == 'Close sidebar'", "sidebar-close-tap") { pause(1.2) }
        drag(from: CGVector(dx: 0.03, dy: 0.5), to: CGVector(dx: 0.75, dy: 0.5), velocity: 600, "sidebar-swipe-open")
        pause(1.2)
        drag(from: CGVector(dx: 0.95, dy: 0.5), to: CGVector(dx: 0.2, dy: 0.5), velocity: 900, "sidebar-swipe-close")
        pause(1.2)
        mark("end")
    }

    func testMotionTour() {
        mark("launch")
        app.launch()
        pause(2.5)

        // Sidebar: button, tap-out, then an interactive swipe each way.
        if tapButton("label == 'Open sidebar'", "sidebar-open-button") { pause(1.2) }
        if tapButton("label == 'Close sidebar'", "sidebar-close-tap") { pause(1.2) }
        drag(from: CGVector(dx: 0.03, dy: 0.5), to: CGVector(dx: 0.75, dy: 0.5), velocity: 600, "sidebar-swipe-open")
        pause(1.2)
        drag(from: CGVector(dx: 0.95, dy: 0.5), to: CGVector(dx: 0.2, dy: 0.5), velocity: 900, "sidebar-swipe-close")
        pause(1.2)
        drag(from: CGVector(dx: 0.03, dy: 0.5), to: CGVector(dx: 0.35, dy: 0.5), velocity: 300, "sidebar-swipe-partial-cancel")
        pause(1.2)

        // Typing: keyboard, field growth over several lines, send button.
        let field = app.textFields["Ask anything"].exists ? app.textFields["Ask anything"] : app.textViews.firstMatch
        if tap(field, "composer-focus") {
            pause(1.0)
            mark("composer-type")
            field.typeText("Explain how running a language model on the phone works, step by step, with a small example I can try")
            pause(1.0)
        }
        if tapButton("label == 'Send'", "send-long") { pause(9.0) }

        // Scroll up, the scroll-to-bottom button, and back down.
        drag(from: CGVector(dx: 0.5, dy: 0.25), to: CGVector(dx: 0.5, dy: 0.8), velocity: 2500, "scroll-up")
        pause(1.2)
        if tapButton("label == 'Scroll to bottom'", "scroll-to-bottom") { pause(1.2) }

        // Copy feedback on the finished reply.
        if tapButton("identifier == 'doc.on.doc'", "copy") { pause(1.5) }

        // Stop a reply part way.
        if tap(field, "composer-focus-2") {
            field.typeText("Tell me more")
            pause(0.6)
        }
        if tapButton("label == 'Send'", "send-then-stop") {
            pause(1.6)
            if tapButton("label == 'Stop generating'", "stop") { pause(1.5) }
        }

        // Dismiss the keyboard by dragging the list.
        drag(from: CGVector(dx: 0.5, dy: 0.4), to: CGVector(dx: 0.5, dy: 0.7), velocity: 800, "keyboard-interactive-dismiss")
        pause(1.0)

        // Long press for the context menu, then dismiss it.
        let lastText = app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] 'Tell me more'")).firstMatch
        if lastText.waitForExistence(timeout: 2) {
            mark("context-menu")
            lastText.press(forDuration: 0.8)
            pause(1.5)
            app.windows.firstMatch.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
            pause(1.0)
        }

        // New chat, then a suggestion chip.
        if tapButton("label == 'New chat'", "new-chat") { pause(1.5) }
        if tapButton("label == 'Explain a concept'", "suggestion-chip") { pause(1.5) }
        if tapButton("label == 'Send'", "send-chip", wait: 2) { pause(8.0) }

        // Thinking mode from the title menu.
        if tapButton("label BEGINSWITH 'Pie,'", "title-menu") {
            pause(1.2)
            if tapButton("label CONTAINS[c] 'Thinking'", "pick-thinking") { pause(1.0) }
        }
        if tap(field, "composer-focus-3") {
            field.typeText("Plan a short study session")
            pause(0.5)
        }
        if tapButton("label == 'Send'", "send-thinking") { pause(9.0) }
        if tapButton("label BEGINSWITH[c] 'Thought'", "reasoning-expand") {
            pause(1.2)
            tapButton("label BEGINSWITH[c] 'Thought'", "reasoning-collapse")
            pause(1.2)
        }

        // Dictation and voice mode.
        if tapButton("label == 'Dictate'", "dictation-start") {
            allowSystemAlert()
            pause(2.0)
            if tapButton("label == 'Cancel dictation'", "dictation-cancel") { pause(1.2) }
        }
        if tapButton("label == 'Start voice mode'", "voice-enter") {
            allowSystemAlert()
            pause(4.0)
            if tapButton("label == 'End voice mode'", "voice-leave") { pause(1.5) }
        }

        // Settings sheet from the sidebar.
        if tapButton("label == 'Open sidebar'", "sidebar-open-for-settings") {
            pause(1.0)
            if tapButton("label == 'Pie Voice settings'", "settings-open") {
                pause(1.5)
                drag(from: CGVector(dx: 0.5, dy: 0.12), to: CGVector(dx: 0.5, dy: 0.9), velocity: 1500, "settings-dismiss-swipe")
                pause(1.5)
            }
        }
        mark("end")
    }
}
