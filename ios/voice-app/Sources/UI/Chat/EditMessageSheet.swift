import SwiftUI

/// Rewrites a sent message; sending answers again from that point.
///
/// Motion: the sheet, its detents and the keyboard are the system's. The
/// editor takes focus only once the sheet has risen, so the keyboard comes
/// up as its own step on the system curve instead of lifting the sheet a
/// second time partway through its spring. Sending runs the edit inside
/// `withMotion`, so the turns after the message fade out and the bubble
/// settles to its new text while the sheet goes down, as in ChatGPT,
/// rather than everything changing in one frame.
struct EditMessageSheet: View {
    let onSend: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var text: String
    @FocusState private var isFocused: Bool

    /// About how long the system sheet takes to rise to its detent. Not a
    /// `Motion` token: it times the system's animation, not one of ours.
    private static let sheetSettleDelay: Duration = .milliseconds(450)

    init(original: String, onSend: @escaping (String) -> Void) {
        self.onSend = onSend
        _text = State(initialValue: original)
    }

    var body: some View {
        NavigationStack {
            TextEditor(text: $text)
                .focused($isFocused)
                .font(.body)
                .foregroundStyle(Theme.ink)
                .scrollContentBackground(.hidden)
                .padding(12)
                .background(Theme.background)
                .navigationTitle("Edit message")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Cancel") { dismiss() }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Send") {
                            // Not wrapped in an animation: the controller
                            // animates the change itself, and a scroll
                            // issued inside an outer animation loses the
                            // landing glide (measured on iOS 26).
                            onSend(trimmed)
                            dismiss()
                        }
                        .fontWeight(.semibold)
                        .disabled(trimmed.isEmpty)
                    }
                }
        }
        .presentationDetents([.medium, .large])
        .task {
            // Cancelled with the view, so a sheet closed before it settles
            // never raises the keyboard.
            try? await Task.sleep(for: Self.sheetSettleDelay)
            guard !Task.isCancelled else { return }
            isFocused = true
        }
    }

    private var trimmed: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
