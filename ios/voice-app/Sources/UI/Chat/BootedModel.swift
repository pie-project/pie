import Foundation

/// The rung of the model ladder this process booted with.
///
/// `PieRuntimeConfig.select` records the rung for the next launch and
/// changes `selected` at once, but the engine loads weights once per
/// process, so what is running is whatever was selected when the chat
/// first asked. Nothing selects a rung before the chat screen is up.
enum BootedModel {
    static let current = PieRuntimeConfig.selected

    /// Rungs whose artifacts were on disk at launch. Models are added to
    /// the container between launches, never while the app runs, and the
    /// check reads directories, so it is done once rather than on every
    /// redraw of the menu.
    static let installedDirectories = Set(PieRuntimeConfig.available.map(\.directory))
}
