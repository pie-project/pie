import SwiftUI
import UIKit

/// Copy buttons in the transcript (a reply's copy icon, a code block's
/// Copy): what they put on the pasteboard, and when, and how their glyph
/// turns into a checkmark.
@MainActor
enum Clipboard {
    /// Puts `text` on the pasteboard once the checkmark has played.
    ///
    /// The pasteboard is shared with the user's other devices (and in the
    /// Simulator with the Mac), so writing it can hold the main thread up
    /// for a moment. Written before the checkmark, it delayed the
    /// checkmark (recorded: 0.13 s after the finger lifted); written while
    /// the swap plays, it could make the swap stutter. Nobody pastes
    /// within a third of a second of tapping Copy.
    static func copy(_ text: String) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            UIPasteboard.general.string = text
        }
    }

    /// Copy becoming a checkmark and back: the plain down-up replace, sped
    /// up so it takes about 0.2 s, as ChatGPT's does. iOS 26's default
    /// replace drew the checkmark on over about 0.3 s more.
    static let glyphSwap = ContentTransition.symbolEffect(.replace.downUp, options: .speed(1.5))
}
