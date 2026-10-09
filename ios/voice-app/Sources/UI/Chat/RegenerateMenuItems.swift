import SwiftUI

/// ChatGPT's "Try again" choices for a reply: as it was, or switched to a
/// different mode.
struct RegenerateMenuItems: View {
    let regenerate: (ReplyMode?) -> Void

    var body: some View {
        Button {
            regenerate(nil)
        } label: {
            Label("Try again", systemImage: "arrow.clockwise")
        }
        Button {
            regenerate(.instant)
        } label: {
            Label("Try again with \(ReplyMode.instant.title)", systemImage: "bolt")
        }
        Button {
            regenerate(.thinking)
        } label: {
            Label("Try again with \(ReplyMode.thinking.title)", systemImage: "lightbulb")
        }
    }
}
