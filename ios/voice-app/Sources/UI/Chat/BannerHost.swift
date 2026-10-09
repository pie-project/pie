import SwiftUI
import UIKit

/// The controller's transient notice ("Couldn't read that file") as a
/// toast just under the top bar. It clears itself after three seconds, or
/// on a tap.
///
/// As in ChatGPT, it drops a few points into place while fading in, and
/// fades out (a little faster) when it goes. A new notice replaces the old
/// one by the same entrance, even when its words are the same, and gets
/// its own full three seconds.
struct BannerHost: View {
    @EnvironmentObject private var chat: ChatController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The notice on screen. A copy of `chat.banner` with an identity of
    /// its own, so the same words posted again count as a new notice.
    @State private var notice: Notice?

    /// The reference's toast timings: in on a quick spring, out on a short
    /// fade. Slower than `Motion.control` and `Motion.fadeOut`, since a
    /// toast arrives on its own and has to be noticed.
    private static let arrival = Animation.snappy(duration: 0.3)
    private static let departure = Animation.easeOut(duration: 0.2)
    /// Longer than ChatGPT's two-word toasts stay: these are sentences.
    private static let holdTime: UInt64 = 3_000_000_000

    var body: some View {
        ZStack {
            if let notice {
                Text(notice.text)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.background)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(Theme.ink, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .onTapGesture { chat.banner = nil }
                    .id(notice.id)
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .offset(y: reduceMotion ? 0 : -12)),
                        removal: .opacity
                    ))
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 8)
        .frame(maxWidth: .infinity)
        .onReceive(chat.$banner) { banner in
            if let banner {
                withAnimation(Self.arrival) { notice = Notice(text: banner) }
                UIAccessibility.post(notification: .announcement, argument: banner)
            } else if notice != nil {
                withAnimation(Self.departure) { notice = nil }
            }
        }
        .task(id: notice?.id) {
            guard let notice else { return }
            try? await Task.sleep(nanoseconds: Self.holdTime)
            guard !Task.isCancelled, chat.banner == notice.text else { return }
            chat.banner = nil
        }
    }
}

private struct Notice: Equatable {
    let id = UUID()
    let text: String
}
