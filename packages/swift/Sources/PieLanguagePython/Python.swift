import Foundation
import PieServer

extension PieServer.Language {
    /// Python inferlets: CPython and the `inferlet` package in one component.
    public static let python = PieServer.Language(name: "python") {
        guard let url = Bundle.module.url(forResource: "python", withExtension: "wasm", subdirectory: "Resources") else {
            throw PieError.server("python.wasm was not bundled; scripts/build-languages.sh builds it")
        }
        return try Data(contentsOf: url)
    }
}
