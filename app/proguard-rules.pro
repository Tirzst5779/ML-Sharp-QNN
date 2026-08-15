# ============================================================
# ProGuard / R8 Rules — SHARP QNN Android App
# ============================================================

# ---- QNN JNI 桥接 ----
# QNN JNI bridge
-keep class com.sharp.qnn.pipeline.QnnJni { *; }
-keepclassmembers class com.sharp.qnn.pipeline.QnnJni {
    native <methods>;
}
-keep class com.sharp.qnn.pipeline.PipelineManager { *; }

# ---- 数据模型 (JSON 序列化/反序列化) ----
# Data models (JSON serialization/deserialization)
-keep class com.sharp.qnn.data.ModelEntry { *; }
-keep class com.sharp.qnn.data.ModelType { *; }
-keep class com.sharp.qnn.data.ModelFormat { *; }
-keep class com.sharp.qnn.data.ModelStatus { *; }
-keep class com.sharp.qnn.data.ModelStore { *; }

# ---- ViewModel 构造器 (AndroidViewModel) ----
# ViewModel constructors (AndroidViewModel)
-keepclassmembers class com.sharp.qnn.ui.**ViewModel {
    public <init>(android.app.Application);
}

# ---- DataStore Preferences ----
# DataStore Preferences
-keep class androidx.datastore.preferences.** { *; }
-keep class androidx.datastore.core.** { *; }
-keepclassmembers class * extends androidx.datastore.preferences.protobuf.GeneratedMessageLite {
    <fields>;
}

# ---- Compose ----
# Keep Compose functions and classes
-dontwarn androidx.compose.**
-keep class androidx.compose.** { *; }
-keepclassmembers class androidx.compose.** { *; }

# Compose Runtime
-keep class androidx.compose.runtime.** { *; }

# Material3
-keep class androidx.compose.material3.** { *; }

# Navigation Compose
-keep class androidx.navigation.** { *; }

# Compose UI tooling (preview 等)
-dontwarn androidx.compose.ui.tooling.**

# ---- Lifecycle / ViewModel ----
-keep class androidx.lifecycle.** { *; }
-keep class * extends androidx.lifecycle.ViewModel { *; }
-keep class * extends androidx.lifecycle.AndroidViewModel { *; }

# ---- Activity Compose ----
-keep class androidx.activity.compose.** { *; }

# ---- ExifInterface ----
-keep class androidx.exifinterface.media.ExifInterface { *; }

# ---- DocumentFile ----
-keep class androidx.documentfile.** { *; }

# ---- Kotlin ----
# Keep Kotlin metadata
-keepattributes *Annotation*, InnerClasses, Signature, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations

# Keep Kotlin coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}

# Keep Kotlin serialization (即使未用 @Serializable, 保留以防未来使用)
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}
-keep,includedescriptorclasses class com.sharp.qnn.**$$serializer { *; }
-keepclassmembers class com.sharp.qnn.** {
    *** Companion;
}
-keepclasseswithmembers class com.sharp.qnn.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# ---- R 类 ----
# Keep R classes (资源引用)
-keepclassmembers class **.R$* {
    public static <fields>;
}

# ---- Application ----
-keep class com.sharp.qnn.SHARPApplication { *; }

# ---- 通用 Android ----
# Keep parcelable
-keepclassmembers class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator CREATOR;
}

# Keep enums
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# ---- 优化 ----
# 允许优化, 但保留行号便于调试
-keepattributes SourceFile, LineNumberTable
-renamesourcefileattribute SourceFile

# 移除废弃代码的 obfuscation 警告
-dontwarn org.codehaus.mojo.animal_sniffer.**
-dontwarn javax.annotation.**
-dontwarn kotlin.Unit
-dontwarn retrofit2.**