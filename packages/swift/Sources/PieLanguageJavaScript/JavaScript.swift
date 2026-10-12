import Foundation
import PieServer

extension PieServer.Language {
    /// JavaScript inferlets: StarlingMonkey and `@pie-project/inferlet` in one component.
    public static let javascript = PieServer.Language(name: "javascript") {
        guard let url = Bundle.module.url(forResource: "javascript", withExtension: "wasm", subdirectory: "Resources") else {
            throw PieError.server("javascript.wasm was not bundled; scripts/build-languages.sh builds it")
        }
        return try Data(contentsOf: url)
    }
}
