plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "org.pieproject.server"
    compileSdk = 35
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    defaultConfig {
        minSdk = 29
        ndk { abiFilters += "arm64-v8a" }
        consumerProguardFiles("consumer-rules.pro")
    }
    testOptions.unitTests.all {
        it.useJUnitPlatform()
        // `PIE_NATIVE_DIR` holds a host build of the core (`cargo build -p pie-kotlin-core`)
        // and `PIE_MODEL` an artifact; ServerTest boots it when both are set.
        it.systemProperty("java.library.path", System.getenv("PIE_NATIVE_DIR") ?: "")
        it.environment("PIE_MODEL", System.getenv("PIE_MODEL") ?: "")
    }
}

kotlin {
    compilerOptions.jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    explicitApi()
}

dependencies {
    api(project(":client"))
    implementation(libs.serialization.json)
    testImplementation(libs.kotlin.test)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.launcher)
    testImplementation(project(":language-python"))
}
