package com.sharp.qnn.data

import androidx.annotation.StringRes
import com.sharp.qnn.R
import org.json.JSONObject

/**
 * 模型类型: 5 个模型部分。
 * Model types: the 5 model parts.
 *
 * - PE     : Patch Encoder / 图块编码
 * - PE     : Patch Encoder
 * - IE     : Image Encoder / 图像编码
 * - IE     : Image Encoder
 * - REST_A : REST Seg A / 特征融合 (6 输入 → 6 上采样特征)
 * - REST_A : REST Seg A / feature fusion (6 inputs → 6 upsampled features)
 * - REST_B : REST Seg B / 视差估计 (3 特征 → disparity)
 * - REST_B : REST Seg B / disparity estimation (3 features → disparity)
 * - REST_C : REST Seg C / 高斯增量 (image+disparity+特征 → delta)
 * - REST_C : REST Seg C / gaussian delta (image+disparity+features → delta)
 */
enum class ModelType(
    /** 双语显示名 (日志/消息用) / bilingual display name (logs & messages) */
    val displayName: String,
    /** 槽位编码 / slot code */
    val code: String,
    /** 本地化名称资源 (UI 用) / localized name string-res (UI) */
    @StringRes val nameRes: Int
) {
    PE("图块编码 (Patch Encoder)", "pe", R.string.model_pe),
    IE("图像编码 (Image Encoder)", "ie", R.string.model_ie),
    REST_A("特征融合 (Feature Fusion)", "rest_a", R.string.model_rest_a),
    REST_B("视差估计 (Disparity Estimation)", "rest_b", R.string.model_rest_b),
    REST_C("高斯增量 (Gaussian Delta)", "rest_c", R.string.model_rest_c);

    companion object {
        fun fromCode(code: String): ModelType? = entries.firstOrNull { it.code == code }
    }
}

/** 模型格式: 预编译 .bin / DLC */
/** Model format: precompiled .bin / DLC */
enum class ModelFormat(val ext: String) {
    BIN("bin"),
    DLC("dlc")
}

/** 模型状态 */
/** Model status */
enum class ModelStatus(
    /** 双语标签 (日志用) / bilingual label (logs) */
    val label: String,
    /** 本地化标签资源 (UI 用) / localized label string-res (UI) */
    @StringRes val labelRes: Int
) {
    NOT_IMPORTED("未导入", R.string.status_not_imported),
    COMPILED("已编译", R.string.status_compiled),
    UNCOMPILED("未编译", R.string.status_uncompiled),
    COMPILING("编译中", R.string.status_compiling)
}

/**
 * 模型元数据。
 * Model metadata.
 *
 * @param type            模型类型 (PE / IE / REST_A / REST_B / REST_C)
 * @param type            model type (PE / IE / REST_A / REST_B / REST_C)
 * @param format          用户导入的原始格式 (BIN / DLC)
 * @param format          original format as imported (BIN / DLC)
 * @param sourcePath      原始文件路径 (.bin 或 .dlc)
 * @param sourcePath      path of the source file (.bin or .dlc)
 * @param sourceName      原始文件名
 * @param sourceName      original file name
 * @param compiledBinPath 编译后 .bin 路径 (DLC 编译后才有；BIN 直接指向 sourcePath)
 * @param compiledBinPath compiled .bin path (only after DLC compilation; BIN points directly to sourcePath)
 * @param status          当前状态
 * @param status          current status
 * @param fileSize        原始文件大小 (字节)
 * @param fileSize        source file size (bytes)
 * @param importTime      导入时间戳 (毫秒)
 * @param importTime      import timestamp (milliseconds)
 */
data class ModelEntry(
    val type: ModelType,
    val format: ModelFormat,
    val sourcePath: String,
    val sourceName: String,
    val compiledBinPath: String?,
    val status: ModelStatus,
    val fileSize: Long,
    val importTime: Long
) {

    /** 返回运行时实际加载的 .bin 路径 (优先编译产物，其次原始 .bin) */
    /** The .bin path actually loaded at runtime (compiled artifact first, raw .bin second) */
    val runtimeBinPath: String? get() = compiledBinPath ?: sourcePath.takeIf { format == ModelFormat.BIN }

    /** 序列化为 JSON */
    /** Serialize to JSON */
    fun toJson(): JSONObject = JSONObject().apply {
        put("type", type.code)
        put("format", format.name)
        put("sourcePath", sourcePath)
        put("sourceName", sourceName)
        put("compiledBinPath", compiledBinPath)
        put("status", status.name)
        put("fileSize", fileSize)
        put("importTime", importTime)
    }

    companion object {
        /** 从 JSON 反序列化 (格式错误返回 null) */
        /** Deserialize from JSON (returns null on malformed input) */
        fun fromJson(json: JSONObject): ModelEntry? {
            return try {
                val type = ModelType.fromCode(json.getString("type")) ?: return null
                ModelEntry(
                    type = type,
                    format = ModelFormat.valueOf(json.getString("format")),
                    sourcePath = json.getString("sourcePath"),
                    sourceName = json.getString("sourceName"),
                    compiledBinPath = json.optString("compiledBinPath").ifBlank { null },
                    status = ModelStatus.valueOf(json.getString("status")),
                    fileSize = json.getLong("fileSize"),
                    importTime = json.getLong("importTime")
                )
            } catch (e: Exception) {
                null
            }
        }
    }
}