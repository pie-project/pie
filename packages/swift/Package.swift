// swift-tools-version:6.2
// `./build-xcframework.sh` builds the PieServerCore binary target;
// `scripts/build-languages.sh` puts the language components in Resources/.

import PackageDescription

let package = Package(
    name: "Pie",
    platforms: [.iOS(.v26), .macOS(.v26)],
    products: [
        .library(name: "PieClient", targets: ["PieClient"]),
        .library(name: "PieServer", targets: ["PieServer"]),
        .library(name: "PieLanguagePython", targets: ["PieLanguagePython"]),
        .library(name: "PieLanguageJavaScript", targets: ["PieLanguageJavaScript"]),
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
        .target(name: "PieLanguagePython", dependencies: ["PieServer"], resources: [.copy("Resources")]),
        .target(name: "PieLanguageJavaScript", dependencies: ["PieServer"], resources: [.copy("Resources")]),
        .executableTarget(name: "pie-smoke", dependencies: ["PieServer", "PieLanguagePython", "PieLanguageJavaScript"]),
        .testTarget(name: "PieClientTests", dependencies: ["PieClient"]),
        .testTarget(name: "PieServerTests", dependencies: ["PieServer"]),
    ]
)
