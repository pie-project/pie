import SwiftUI
import UIKit

/// The controller's transient notice ("Couldn't read that file") as a
/// toast under the top bar. It clears itself after three seconds, or on a
/// tap.
struct BannerHost: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        VStack {
            if let banner = chat.banner {
                Text(banner)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.background)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(Theme.ink, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .padding(.horizontal, 24)
                    .padding(.top, 60)
                    .onTapGesture { chat.banner = nil }
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity)
        .animation(.spring(response: 0.35, dampingFraction: 0.85), value: chat.banner)
        .task(id: chat.banner) {
            guard let banner = chat.banner else { return }
            UIAccessibility.post(notification: .announcement, argument: banner)
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            guard !Task.isCancelled, chat.banner == banner else { return }
            chat.banner = nil
        }
    }
}
