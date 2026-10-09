import SwiftUI

/// Rewrites a sent message; sending answers again from that point.
struct EditMessageSheet: View {
    let onSend: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var text: String
    @FocusState private var isFocused: Bool

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
                            onSend(trimmed)
                            dismiss()
                        }
                        .fontWeight(.semibold)
                        .disabled(trimmed.isEmpty)
                    }
                }
        }
        .presentationDetents([.medium, .large])
        .onAppear { isFocused = true }
    }

    private var trimmed: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
