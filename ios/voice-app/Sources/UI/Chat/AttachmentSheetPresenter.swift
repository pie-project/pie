import SwiftUI

/// Presents the composer's "+" sheet from the router's flag.
///
/// UIKit will not present voice mode's full-screen cover while this sheet
/// is still animating away, so a choice that leads there is handed back
/// as `afterDismiss` and carried out once the sheet is gone.
struct AttachmentSheetPresenter: ViewModifier {
    @EnvironmentObject private var router: AppRouter

    @State private var afterDismiss: (() -> Void)?

    func body(content: Content) -> some View {
        content.sheet(isPresented: $router.isAttachmentSheetPresented, onDismiss: runAfterDismiss) {
            AttachmentSheet(afterDismiss: $afterDismiss)
        }
    }

    private func runAfterDismiss() {
        let action = afterDismiss
        afterDismiss = nil
        action?()
    }
}
