import SwiftUI

/// The engine's boot state under the top bar: a pill while loading, a card
/// if loading failed, nothing once it is ready.
///
/// The pill floats over the chat rather than taking a line of its own, so
/// nothing moves when loading finishes: it only fades away. The failure
/// card does take room, and the chat below slides down to make it.
struct EngineStatusView: View {
    @EnvironmentObject private var chat: ChatController

    /// The engine state as shown. The engine's boot task publishes its
    /// state with no animation; this copy changes inside one whenever the
    /// phase changes (booting, ready, failed), so the pill fades and the
    /// card arrives smoothly. The boot pill's seconds, ticking within one
    /// phase, are copied as they are, so they don't crossfade every
    /// second. Nil until the first change.
    @State private var shown: ChatController.EngineState?

    var body: some View {
        let state = shown ?? chat.engineState
        VStack(spacing: 0) {
            if case .failed(let message) = state {
                EngineFailedCard(message: message)
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .transition(.fadeRise(8))
            }
        }
        .frame(maxWidth: .infinity)
        .overlay(alignment: .top) {
            if case .booting(let seconds) = state {
                EngineBootPill(seconds: seconds)
                    .fixedSize()
                    .padding(.top, 8)
                    .transition(.popIn(from: 0.9))
            }
        }
        // Drawn above the chat that follows it, which the pill overlaps.
        .zIndex(1)
        .onChange(of: chat.engineState) { old, new in
            if Self.phase(of: old) == Self.phase(of: new) {
                shown = new
            } else {
                withMotion(new == .ready ? Motion.fadeOut : Motion.content) { shown = new }
            }
        }
    }

    private static func phase(of state: ChatController.EngineState) -> Int {
        switch state {
        case .booting: return 0
        case .ready: return 1
        case .failed: return 2
        }
    }
}
