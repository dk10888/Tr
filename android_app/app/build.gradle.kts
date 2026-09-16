plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.example.receiptscanner"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.example.receiptscanner"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
      compose = true
      aidl = false
      buildConfig = false
      shaders = false
    }

    packaging {
      resources {
        excludes += "/META-INF/{AL2.0,LGPL2.1}"
        // ONNX Runtime bundles multiple JNI .so — pick first to avoid conflicts
        pickFirsts += listOf(
          "lib/x86/libonnxruntime.so",
          "lib/x86_64/libonnxruntime.so",
          "lib/armeabi-v7a/libonnxruntime.so",
          "lib/arm64-v8a/libonnxruntime.so"
        )
      }
      // Keep ONNX models uncompressed for fast mmap access at runtime
      jniLibs {
        useLegacyPackaging = false
      }
    }

    // Prevent ONNX model files from being compressed in the APK
    androidResources {
        noCompress += listOf("onnx")
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
  val composeBom = platform(libs.androidx.compose.bom)
  implementation(composeBom)
  androidTestImplementation(composeBom)

  // Core Android
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)

  // Compose
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.compose.material3)
  implementation(libs.compose.material.icons.extended)
  debugImplementation(libs.androidx.compose.ui.tooling)
  debugImplementation(libs.androidx.compose.ui.test.manifest)

  // Navigation
  implementation(libs.androidx.navigation3.ui)
  implementation(libs.androidx.navigation3.runtime)
  implementation(libs.androidx.lifecycle.viewmodel.navigation3)

  // Coroutines
  implementation(libs.kotlinx.coroutines.android)

  // ── ML Kit Text Recognition ──────────────────────────────────────
  implementation(libs.mlkit.text.recognition)         // Latin (English)
  implementation(libs.mlkit.text.recognition.korean)  // 한글 (Korean)

  // ── ONNX Runtime (NNAPI / GPU delegate built-in) ─────────────────
  implementation(libs.onnxruntime.android)

  // ── CameraX ──────────────────────────────────────────────────────
  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.androidx.camera.lifecycle)
  implementation(libs.androidx.camera.view)

  // ── Image loading ─────────────────────────────────────────────────
  implementation(libs.coil.compose)

  // Tests
  testImplementation(libs.junit)
  testImplementation(libs.kotlinx.coroutines.test)
  androidTestImplementation(libs.androidx.test.core)
  androidTestImplementation(libs.androidx.test.ext.junit)
  androidTestImplementation(libs.androidx.test.runner)
  androidTestImplementation(libs.androidx.test.espresso.core)
  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
}
