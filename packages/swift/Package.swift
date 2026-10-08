// swift-tools-version:6.2
// `./build-xcframework.sh` builds the PieServerCore binary target.

import PackageDescription

let package = Package(
    name: "Pie",
    platforms: [.iOS(.v26), .macOS(.v26)],
    products: [
        .library(name: "PieClient", targets: ["PieClient"]),
        .library(name: "PieServer", targets: ["PieServer"]),
    ],
    targets: [
        .binaryTarget(name: "PieServerCore", path: "build/PieServerCore.xcframework"),
        .target(name: "PieClient", path: "client/Sources/PieClient"),
        .target(
            name: "PieServer",
            dependencies: ["PieServerCore", "PieClient"],
            path: "server/Sources/PieServer",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Security"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("Foundation"),
                .linkedLibrary("objc"),
                .linkedLibrary("iconv"),
            ]
        ),
        .executableTarget(
            name: "pie-smoke",
            dependencies: ["PieServer"],
            path: "server/Sources/pie-smoke"
        ),
    ],
    swiftLanguageModes: [.v5]
)
