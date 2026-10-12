plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "org.pieproject.language.python"
    compileSdk = 35
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    defaultConfig { minSdk = 29 }
}

kotlin {
    compilerOptions.jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    explicitApi()
}

dependencies {
    api(project(":server"))
}
