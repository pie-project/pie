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
        .target(name: "PieClient"),
        .target(
            name: "PieServer",
            dependencies: ["PieServerCore", "PieClient"],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Security"),
                .linkedLibrary("iconv"),
            ]
        ),
        .executableTarget(name: "pie-smoke", dependencies: ["PieServer"]),
        .testTarget(name: "PieClientTests", dependencies: ["PieClient"]),
        .testTarget(name: "PieServerTests", dependencies: ["PieServer"]),
    ]
)
