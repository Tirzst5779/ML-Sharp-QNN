package com.sharp.qnn.data

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.provider.DocumentsContract
import androidx.annotation.StringRes
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.sharp.qnn.R
import com.sharp.qnn.SHARPApplication
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/** 应用级 DataStore (Preferences)，保证进程内单例 */
/** App-level DataStore (Preferences), a process-wide singleton */
private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(name = "sharp_settings")

/**
 * 设置仓库：基于 DataStore Preferences 管理用户偏好。
 * Settings repository: manages user preferences via DataStore Preferences.
 *
 * 管理项:
 * Managed items:
 * - plySaveLocation:      PLY 保存目录 (SAF tree Uri 字符串, 空 = 未设置)
 * - plySaveLocation:      PLY save directory (SAF tree Uri string; empty = not set)
 * - showImageDetails:     是否显示图片详细信息 (焦距、格式等)
 * - showImageDetails:     whether to show detailed image info (focal length, format, ...)
 * - logRecording:         是否记录日志到下载目录 sharp_log/
 * - logRecording:         whether to record logs into Download/sharp_log/
 * - language:             界面语言 (SYSTEM / ZH / EN, 默认跟随系统)
 * - language:             UI language (SYSTEM / ZH / EN; SYSTEM follows the device by default)
 * - downloadSource:      模型下载源 (HG / HM, 默认HG; 中国用户可选择HM镜像站加速)
 * - downloadSource:      model download source (HG / HM, defaults to HG; Chinese users may prefer HM mirror for faster speed)
 * - HTP 性能调度 (perfType/lockedCorner/rangeMin/rangeTarget/rangeMax/dcvsMode)
 * - HTP performance scheduling (perfType/lockedCorner/rangeMin/rangeTarget/rangeMax/dcvsMode)
 */
class SettingsRepository(private val context: Context) {

    private object Keys {
        val PLY_SAVE = stringPreferencesKey("ply_save_location")
        val SHOW_IMAGE_DETAILS = booleanPreferencesKey("show_image_details")
        val LOG_RECORDING = booleanPreferencesKey("log_recording")
        val DYNAMIC_COLOR = booleanPreferencesKey("dynamic_color")
        val LANGUAGE = stringPreferencesKey("language")
        val PERF_TYPE = intPreferencesKey("perf_type")
        val PERF_LOCKED_CORNER = intPreferencesKey("perf_locked_corner")
        val PERF_RANGE_MIN = intPreferencesKey("perf_range_min")
        val PERF_RANGE_TARGET = intPreferencesKey("perf_range_target")
        val PERF_RANGE_MAX = intPreferencesKey("perf_range_max")
        val PERF_DCVS_MODE = intPreferencesKey("perf_dcvs_mode")
        val DOWNLOAD_SOURCE = stringPreferencesKey("download_source")
    }

    /** HTP 调度类型 */
    /** HTP scheduling type */
    object PerfType {
        const val LOCKED = 0    // 锁角模式 (默认): 电压角固定, 频率恒定
                                // Locked-corner mode (default): fixed voltage corner, constant frequency
        const val RANGE = 1     // 自动调角模式: DCVS 在最小~最大角区间内动态调节
                                // Range mode: DCVS adjusts dynamically within the min~max corner range
    }

    /**
     * HTP 电压角 (取值与 QNN QnnHtpPerfInfrastructure_VoltageCorner_t 一致, 0x20~0xA0;
     * 不含 DISABLE 0x10 与 UNKNOWN)。列表按电压从低到高排列。
     * HTP voltage corners (values match QNN QnnHtpPerfInfrastructure_VoltageCorner_t,
     * 0x20~0xA0; DISABLE 0x10 and UNKNOWN excluded). Listed from lowest to highest voltage.
     */
    object VoltageCorner {
        const val MIN = 0x20    // MIN (SVS2, 平台最低) / MIN (SVS2, lowest platform corner)
        const val SVS2 = 0x30
        const val SVS = 0x40
        const val SVS_PLUS = 0x50
        const val NOM = 0x60
        const val NOM_PLUS = 0x70
        const val TURBO = 0x80
        const val TURBO_PLUS = 0x90
        const val TURBO_L2 = 0x92
        const val TURBO_L3 = 0x93
        const val TURBO_L4 = 0x94
        const val TURBO_L5 = 0x95
        const val MAX = 0xA0    // MAX (平台最高) / MAX (highest platform corner)

        /** 全部可选角, 从低到高 (与下拉列表共用) */
        /** All selectable corners, low to high (shared with dropdowns) */
        val ALL: List<Int> = listOf(
            MIN, SVS2, SVS, SVS_PLUS, NOM, NOM_PLUS,
            TURBO, TURBO_PLUS, TURBO_L2, TURBO_L3, TURBO_L4, TURBO_L5, MAX
        )

        fun name(corner: Int): String = when (corner) {
            MIN -> "MIN (最低)"
            SVS2 -> "SVS2"
            SVS -> "SVS"
            SVS_PLUS -> "SVS_PLUS"
            NOM -> "NOM"
            NOM_PLUS -> "NOM_PLUS"
            TURBO -> "TURBO"
            TURBO_PLUS -> "TURBO_PLUS"
            TURBO_L2 -> "TURBO_L2"
            TURBO_L3 -> "TURBO_L3"
            TURBO_L4 -> "TURBO_L4"
            TURBO_L5 -> "TURBO_L5"
            MAX -> "MAX (最高)"
            else -> "0x%02X".format(corner)
        }
    }

    /**
     * DCVS 调节模式 (取值与 QNN QnnHtpPerfInfrastructure_PowerMode_t 一致, 0x1~0x20;
     * 官方语义见 HAP_DCVS_V2)。
     * DCVS adjustment modes (values match QNN QnnHtpPerfInfrastructure_PowerMode_t,
     * 0x1~0x20; official semantics in HAP_DCVS_V2).
     */
    object DcvsMode {
        const val ADJUST_UP_DOWN = 0x1             // 允许上下双向调节 (默认) / allow both up and down (default)
        const val ADJUST_ONLY_UP = 0x2             // 只升不降, 避免抖动 / up only, avoids oscillation
        const val POWER_SAVER = 0x4                // 省电: 爬升阈值高, 倾向低频 / power saver: high ramp threshold, low-frequency bias
        const val POWER_SAVER_AGGRESSIVE = 0x8     // 激进省电: 降频更快 / aggressive power saver: faster frequency reduction
        const val PERFORMANCE = 0x10               // 性能: 爬升阈值低, 尽量保持高频 / performance: low ramp threshold, stays high-frequency
        const val DUTY_CYCLE = 0x20                // 占空比: 检测 HVX 活动周期调频 (流式负载) / duty cycle: scales with HVX activity (streaming workloads)

        /** 全部可选 DCVS 模式 */
        /** All selectable DCVS modes */
        val ALL: List<Int> = listOf(
            ADJUST_UP_DOWN, ADJUST_ONLY_UP, POWER_SAVER,
            POWER_SAVER_AGGRESSIVE, PERFORMANCE, DUTY_CYCLE
        )

        fun name(mode: Int): String = when (mode) {
            ADJUST_UP_DOWN -> "ADJUST_UP_DOWN (上下双向)"
            ADJUST_ONLY_UP -> "ADJUST_ONLY_UP (只升不降)"
            POWER_SAVER -> "POWER_SAVER (省电)"
            POWER_SAVER_AGGRESSIVE -> "POWER_SAVER_AGGRESSIVE (激进省电)"
            PERFORMANCE -> "PERFORMANCE_MODE (性能优先)"
            DUTY_CYCLE -> "DUTY_CYCLE (占空比/流式)"
            else -> "0x%02X".format(mode)
        }
    }

    /** 默认性能配置: 锁角模式 + 最高角 (用户指定默认) */
    /** Default performance config: locked mode + highest corner (user-specified default) */
    object PerfDefaults {
        const val TYPE = PerfType.LOCKED
        const val LOCKED_CORNER = VoltageCorner.MAX
        const val RANGE_MIN = VoltageCorner.NOM
        const val RANGE_TARGET = VoltageCorner.MAX
        const val RANGE_MAX = VoltageCorner.MAX
        const val DCVS_MODE = DcvsMode.ADJUST_UP_DOWN
    }

    /** 界面语言: 跟随系统 / 中文 / English */
    /** UI language: follow system / Chinese / English */
    enum class Language(@StringRes val labelRes: Int, val key: String) {
        SYSTEM(R.string.settings_language_system, "system"),
        ZH(R.string.settings_language_zh, "zh"),
        EN(R.string.settings_language_en, "en");

        companion object {
            fun fromKey(key: String?): Language =
                entries.firstOrNull { it.key == key } ?: SYSTEM
        }
    }

    /**
     * 模型下载源: HuggingFace 官方 (HG) 或国内镜像站 (HM)。
     * Model download source: HuggingFace official (HG) or HF Mirror (HM) for Chinese users.
     * 镜像站地址: https://hf-mirror.com
     * Mirror URL: https://hf-mirror.com
     */
    enum class DownloadSource(val key: String) {
        HG("hg"),   // HuggingFace 官方 (默认) / HuggingFace official (default)
        HM("hm");   // HF 镜像站 (国内用户推荐) / HF Mirror (recommended for Chinese users)

        companion object {
            fun fromKey(key: String?): DownloadSource =
                entries.firstOrNull { it.key == key } ?: HG
        }
    }

    /**
     * 设置快照。
     * Settings snapshot.
     */
    data class Settings(
        val plySaveLocation: String,
        val showImageDetails: Boolean,
        val logRecording: Boolean,
        val perfType: Int,
        val perfLockedCorner: Int,
        val perfRangeMin: Int,
        val perfRangeTarget: Int,
        val perfRangeMax: Int,
        val perfDcvsMode: Int,
        val dynamicColor: Boolean,
        val language: Language = Language.SYSTEM,
        val downloadSource: DownloadSource = DownloadSource.HG
    )

    val settingsFlow: Flow<Settings> = context.settingsDataStore.data.map { prefs ->
        Settings(
            plySaveLocation = prefs[Keys.PLY_SAVE] ?: DEFAULTS.plySaveLocation,
            showImageDetails = prefs[Keys.SHOW_IMAGE_DETAILS] ?: DEFAULTS.showImageDetails,
            logRecording = prefs[Keys.LOG_RECORDING] ?: DEFAULTS.logRecording,
            perfType = prefs[Keys.PERF_TYPE] ?: PerfDefaults.TYPE,
            perfLockedCorner = prefs[Keys.PERF_LOCKED_CORNER] ?: PerfDefaults.LOCKED_CORNER,
            perfRangeMin = prefs[Keys.PERF_RANGE_MIN] ?: PerfDefaults.RANGE_MIN,
            perfRangeTarget = prefs[Keys.PERF_RANGE_TARGET] ?: PerfDefaults.RANGE_TARGET,
            perfRangeMax = prefs[Keys.PERF_RANGE_MAX] ?: PerfDefaults.RANGE_MAX,
            perfDcvsMode = prefs[Keys.PERF_DCVS_MODE] ?: PerfDefaults.DCVS_MODE,
            dynamicColor = prefs[Keys.DYNAMIC_COLOR] ?: DEFAULTS.dynamicColor,
            language = Language.fromKey(prefs[Keys.LANGUAGE]),
            downloadSource = DownloadSource.fromKey(prefs[Keys.DOWNLOAD_SOURCE])
        )
    }

    suspend fun setPlySaveLocation(uriString: String) {
        context.settingsDataStore.edit { it[Keys.PLY_SAVE] = uriString }
    }

    suspend fun setShowImageDetails(show: Boolean) {
        context.settingsDataStore.edit { it[Keys.SHOW_IMAGE_DETAILS] = show }
    }

    suspend fun setLogRecording(enable: Boolean) {
        context.settingsDataStore.edit { it[Keys.LOG_RECORDING] = enable }
    }

    suspend fun setDynamicColor(enable: Boolean) {
        context.settingsDataStore.edit { it[Keys.DYNAMIC_COLOR] = enable }
    }

    /** 设置界面语言 (SYSTEM 表示跟随系统) */
    /** Set the UI language (SYSTEM follows the device) */
    suspend fun setLanguage(language: Language) {
        context.settingsDataStore.edit { it[Keys.LANGUAGE] = language.key }
        // 同步写 SharedPreferences, 供 attachBaseContext 冷启动读取
        // Also write to SharedPreferences for cold-start attachBaseContext
        context.getSharedPreferences(SHARPApplication.LANG_PREFS, Context.MODE_PRIVATE)
            .edit().putString(SHARPApplication.KEY_LANGUAGE, language.key).apply()
    }

    /**
     * 设置模型下载源。
     * Set the model download source.
     * @param source 下载源 (HG=官方, HM=国内镜像) / download source (HG=official, HM=mirror)
     */
    suspend fun setDownloadSource(source: DownloadSource) {
        context.settingsDataStore.edit { it[Keys.DOWNLOAD_SOURCE] = source.key }
    }

    suspend fun setPerfType(type: Int) {
        context.settingsDataStore.edit { it[Keys.PERF_TYPE] = type }
    }

    suspend fun setPerfLockedCorner(corner: Int) {
        context.settingsDataStore.edit { it[Keys.PERF_LOCKED_CORNER] = corner }
    }

    suspend fun setPerfRangeMin(corner: Int) {
        context.settingsDataStore.edit { it[Keys.PERF_RANGE_MIN] = corner }
    }

    suspend fun setPerfRangeTarget(corner: Int) {
        context.settingsDataStore.edit { it[Keys.PERF_RANGE_TARGET] = corner }
    }

    suspend fun setPerfRangeMax(corner: Int) {
        context.settingsDataStore.edit { it[Keys.PERF_RANGE_MAX] = corner }
    }

    suspend fun setPerfDcvsMode(mode: Int) {
        context.settingsDataStore.edit { it[Keys.PERF_DCVS_MODE] = mode }
    }

    companion object {
        /** 默认设置快照 (页面 collectAsState 的 initial 值统一取此) */
        /** Default settings snapshot (used as the initial value for collectAsState) */
        val DEFAULTS = Settings(
            plySaveLocation = "",
            showImageDetails = false,
            logRecording = false,
            perfType = PerfDefaults.TYPE,
            perfLockedCorner = PerfDefaults.LOCKED_CORNER,
            perfRangeMin = PerfDefaults.RANGE_MIN,
            perfRangeTarget = PerfDefaults.RANGE_TARGET,
            perfRangeMax = PerfDefaults.RANGE_MAX,
            perfDcvsMode = PerfDefaults.DCVS_MODE,
            dynamicColor = true,
            language = Language.SYSTEM,
            downloadSource = DownloadSource.HG
        )

        /** 将 SAF tree Uri 转为人眼可读路径 (如 /storage/emulated/0/Download/sharp_ply) */
        /** Convert a SAF tree Uri to a human-readable path (e.g. /storage/emulated/0/Download/sharp_ply) */
        fun plySaveDisplayPath(uriString: String): String {
            if (uriString.isBlank()) return ""
            return try {
                val uri = Uri.parse(uriString)
                val docId = DocumentsContract.getTreeDocumentId(uri)
                val decoded = Uri.decode(docId)
                val primary = decoded.substringAfter("primary:", "")
                if (primary != decoded) "/storage/emulated/0/$primary" else decoded
            } catch (e: Exception) {
                uriString
            }
        }

        /** 通用显示: content:// 转可读路径, 其余 (本地路径) 原样返回 */
        /** Generic display: content:// becomes a readable path; others (local paths) pass through */
        fun displayPath(value: String): String =
            if (value.startsWith("content://")) plySaveDisplayPath(value) else value
    }
}