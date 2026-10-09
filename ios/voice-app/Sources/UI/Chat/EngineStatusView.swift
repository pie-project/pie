import SwiftUI

/// The engine's boot state under the top bar: a pill while loading, a card
/// if loading failed, nothing once it is ready.
struct EngineStatusView: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        Group {
            switch chat.engineState {
            case .booting(let seconds):
                EngineBootPill(seconds: seconds)
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
            case .failed(let message):
                EngineFailedCard(message: message)
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .transition(.opacity)
            case .ready:
                EmptyView()
            }
        }
        .animation(.easeInOut(duration: 0.25), value: chat.engineState)
    }
}
