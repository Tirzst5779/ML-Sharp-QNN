package com.sharp.qnn.ui.settings

import android.app.Application
import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CleaningServices
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RangeSlider
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.sharp.qnn.R
import com.sharp.qnn.SHARPApplication
import com.sharp.qnn.data.SettingsRepository
import com.sharp.qnn.data.SettingsRepository.DownloadSource
import com.sharp.qnn.data.SettingsRepository.Language
import com.sharp.qnn.service.LogRecorderService
import com.sharp.qnn.util.FileUtil
import com.sharp.qnn.util.FileUtil.formatFileSize
import com.sharp.qnn.util.MsgKey
import com.sharp.qnn.util.i18nMessage
import com.sharp.qnn.ui.theme.Spacing
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.roundToInt

/**
 * 设置页 ViewModel。
 * Settings view model.
 */
class SettingsViewModel(app: Application) : AndroidViewModel(app) {
    private val sharpApp = app as SHARPApplication

    val settingsFlow = sharpApp.settingsRepository.settingsFlow

    fun setPlySaveLocation(uriString: String) {
        viewModelScope.launch { sharpApp.settingsRepository.setPlySaveLocation(uriString) }
    }

    fun setShowImageDetails(show: Boolean) {
        viewModelScope.launch { sharpApp.settingsRepository.setShowImageDetails(show) }
    }

    /** 界面语言 (运行时生效, 无需重启) */
    /** UI language (applies at runtime, no restart needed). */
    fun setLanguage(language: Language) {
        viewModelScope.launch { sharpApp.settingsRepository.setLanguage(language) }
    }

    /** 模型下载源 (HG=官方, HM=国内镜像) */
    /** Model download source (HG=official, HM=mirror for China). */
    fun setDownloadSource(source: DownloadSource) {
        viewModelScope.launch { sharpApp.settingsRepository.setDownloadSource(source) }
    }

    /** 日志记录开关: 开启启动前台记录服务, 关闭停止 */
    /** Log recording toggle: starts the foreground recording service, or stops it. */
    fun setLogRecording(enable: Boolean) {
        viewModelScope.launch {
            sharpApp.settingsRepository.setLogRecording(enable)
            if (enable) LogRecorderService.start(getApplication())
            else LogRecorderService.stop(getApplication())
        }
    }

    /** 动态色彩开关 (Android 12+ 生效) */
    /** Dynamic color toggle (effective on Android 12+). */
    fun setDynamicColor(enable: Boolean) {
        viewModelScope.launch { sharpApp.settingsRepository.setDynamicColor(enable) }
    }

    /** HTP 调度类型: 0=锁角, 1=自动调角 */
    /** HTP scheduling type: 0=locked corner, 1=auto range. */
    fun setPerfType(type: Int) {
        viewModelScope.launch { sharpApp.settingsRepository.setPerfType(type) }
    }

    /** 锁角模式的电压角 */
    /** Voltage corner for locked mode. */
    fun setPerfLockedCorner(corner: Int) {
        viewModelScope.launch { sharpApp.settingsRepository.setPerfLockedCorner(corner) }
    }

    /**
     * 规范化自动调角三元组, 保持不变量 min < target <= max 且 min != max:
     * Normalizes the auto-range triple, keeping the invariant min < target <= max and min != max:
     * target <= min 时推到 min 的上一档; max < target 时 max 提升到 target。
     * If target <= min, target is pushed to the step above min; if max < target, max is raised to target.
     * 返回 null 表示 min 已是最高档、无法满足不变量 (调用方应放弃本次修改)。
     * Returns null when min is already the top step and the invariant cannot hold (callers should discard the change).
     */
    private fun fixRange(min: Int, target: Int, max: Int): Triple<Int, Int, Int>? {
        var m = min
        var t = target
        var x = max
        if (m >= t) {
            val h = higherCorner(m) ?: return null
            t = h
        }
        if (t > x) x = t
        if (t <= m) {
            val h = higherCorner(m) ?: return null
            t = h
        }
        if (t > x) x = t
        return Triple(m, t, x)
    }

    /** 自动调角最小角 (联动 target/max 保持不变量) */
    /** Auto-range min corner (keeps the invariant with target/max). */
    fun setPerfRangeMin(corner: Int) {
        viewModelScope.launch {
            val repo = sharpApp.settingsRepository
            val s = repo.settingsFlow.first()
            val r = fixRange(corner, s.perfRangeTarget, s.perfRangeMax) ?: return@launch
            repo.setPerfRangeMin(r.first)
            if (r.second != s.perfRangeTarget) repo.setPerfRangeTarget(r.second)
            if (r.third != s.perfRangeMax) repo.setPerfRangeMax(r.third)
        }
    }

    /** 自动调角目标角 (限制在 (min, max] 内, 联动 max 保持不变量) */
    /** Auto-range target corner (clamped to (min, max]; keeps the invariant with max). */
    fun setPerfRangeTarget(corner: Int) {
        viewModelScope.launch {
            val repo = sharpApp.settingsRepository
            val s = repo.settingsFlow.first()
            val r = fixRange(s.perfRangeMin, corner, s.perfRangeMax) ?: return@launch
            repo.setPerfRangeTarget(r.second)
            if (r.third != s.perfRangeMax) repo.setPerfRangeMax(r.third)
        }
    }

    /** 自动调角最大角 (联动 min/target 保持不变量) */
    /** Auto-range max corner (keeps the invariant with min/target). */
    fun setPerfRangeMax(corner: Int) {
        viewModelScope.launch {
            val repo = sharpApp.settingsRepository
            val s = repo.settingsFlow.first()
            val r = fixRange(s.perfRangeMin, s.perfRangeTarget, corner) ?: return@launch
            repo.setPerfRangeMax(r.third)
            if (r.first != s.perfRangeMin) repo.setPerfRangeMin(r.first)
            if (r.second != s.perfRangeTarget) repo.setPerfRangeTarget(r.second)
        }
    }

    /** 自动调角的 DCVS 调节模式 */
    /** DCVS adjustment mode for auto-range. */
    fun setPerfDcvsMode(mode: Int) {
        viewModelScope.launch { sharpApp.settingsRepository.setPerfDcvsMode(mode) }
    }

    /** 文件管理器操作后重新扫描目录对齐槽位 */
    /** Re-scans the directory after file manager actions to realign slots. */
    fun scanModels() {
        viewModelScope.launch { sharpApp.modelStore.scanModelDirectory() }
    }

    /** 模型根目录 (sharp_models) */
    /** Model root directory (sharp_models). */
    fun modelRootDir(): File = sharpApp.modelStore.modelRootDir()

    fun clearCache(onResult: (Pair<Int, Long>) -> Unit) {
        viewModelScope.launch {
            val result = sharpApp.modelStore.clearAppCache()
            onResult(result.getOrElse { 0 to 0L })
        }
    }
}

@Composable
fun SettingsScreen(vm: SettingsViewModel = viewModel()) {
    val context = LocalContext.current
    val settings by vm.settingsFlow.collectAsState(initial = SettingsRepository.DEFAULTS)

    // 模型文件管理器弹窗开关
    // Model file manager dialog toggle
    var showModelManager by remember { mutableStateOf(false) }

    // PLY 目录选择器
    // PLY directory picker
    val plyDirPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        if (uri != null) {
            val granted = runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
                android.util.Log.i("SharpQnn", "PLY persist grant OK: $uri")
                true
            }.getOrDefault(false)
            if (granted) {
                // 持久化保存 tree Uri 本身, 导出时经 contentResolver 写入 (SAF)
                // Persist the tree Uri; exports write through the contentResolver (SAF)
                vm.setPlySaveLocation(uri.toString())
            }
        }
    }

    var cacheMessage by remember { mutableStateOf<String?>(null) }

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.md)
    ) {
        // ====== 界面语言 ======
        // ====== UI language ======
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                    Text(stringResource(R.string.settings_language), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        text = stringResource(R.string.settings_language_desc),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    LanguagePanel(
                        current = settings.language,
                        onChange = { vm.setLanguage(it) }
                    )
                }
            }
        }

        // ====== 模型下载源 ======
        // ====== Model download source ======
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                    Text(stringResource(R.string.settings_download_source), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        text = stringResource(R.string.settings_download_source_desc),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    DownloadSourcePanel(
                        current = settings.downloadSource,
                        onChange = { vm.setDownloadSource(it) }
                    )
                }
            }
        }

        // ====== 模型文件管理器 ======
        // ====== Model file manager ======
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                    Text(stringResource(R.string.settings_model_manager), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        text = stringResource(R.string.settings_model_manager_desc),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    FilledTonalButton(onClick = { showModelManager = true }) {
                        Icon(Icons.Filled.Folder, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.settings_open_manager))
                    }
                }
            }
        }

        // ====== PLY 保存位置 ======
        // ====== PLY save location ======
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                    Text(stringResource(R.string.settings_ply_location), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        text = if (settings.plySaveLocation.isBlank())
                            stringResource(R.string.settings_ply_not_set)
                        else
                            stringResource(R.string.settings_ply_current, SettingsRepository.plySaveDisplayPath(settings.plySaveLocation)),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    FilledTonalButton(onClick = { plyDirPicker.launch(null) }) {
                        Icon(Icons.Filled.Folder, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.settings_choose_dir))
                    }
                }
            }
        }

        // ====== 图片详细信息 ======
        // ====== Image details ======
        item {
            SwitchSettingRow(
                title = stringResource(R.string.settings_image_details),
                description = stringResource(R.string.settings_image_details_desc),
                checked = settings.showImageDetails,
                onCheckedChange = { vm.setShowImageDetails(it) }
            )
        }

        // ====== HTP 性能调度 ======
        // ====== HTP performance scheduling ======
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(Spacing.lg).fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(stringResource(R.string.settings_htp_perf), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        text = stringResource(R.string.settings_htp_perf_desc),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    val haptic = LocalHapticFeedback.current
                    // ----- 锁角模式 (默认) -----
                    // ----- Locked corner mode (default) -----
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = settings.perfType == SettingsRepository.PerfType.LOCKED,
                            onClick = {
                                haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                                vm.setPerfType(SettingsRepository.PerfType.LOCKED)
                            }
                        )
                        Column(modifier = Modifier.weight(1f)) {
                            Text(stringResource(R.string.settings_locked_mode), style = MaterialTheme.typography.bodyLarge)
                            Text(
                                text = stringResource(R.string.settings_locked_mode_desc),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    if (settings.perfType == SettingsRepository.PerfType.LOCKED) {
                        LockedCornerPanel(
                            corner = settings.perfLockedCorner,
                            onChange = { vm.setPerfLockedCorner(it) }
                        )
                    }
                    // ----- 自动调角模式 -----
                    // ----- Auto-range mode -----
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = settings.perfType == SettingsRepository.PerfType.RANGE,
                            onClick = {
                                haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                                vm.setPerfType(SettingsRepository.PerfType.RANGE)
                            }
                        )
                        Column(modifier = Modifier.weight(1f)) {
                            Text(stringResource(R.string.settings_range_mode), style = MaterialTheme.typography.bodyLarge)
                            Text(
                                text = stringResource(R.string.settings_range_mode_desc),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    if (settings.perfType == SettingsRepository.PerfType.RANGE) {
                        RangePanel(
                            minCorner = settings.perfRangeMin,
                            targetCorner = settings.perfRangeTarget,
                            maxCorner = settings.perfRangeMax,
                            dcvsMode = settings.perfDcvsMode,
                            onMinChange = { vm.setPerfRangeMin(it) },
                            onTargetChange = { vm.setPerfRangeTarget(it) },
                            onMaxChange = { vm.setPerfRangeMax(it) },
                            onDcvsModeChange = { vm.setPerfDcvsMode(it) }
                        )
                    }
                }
            }
        }

        // ====== 动态色彩 ======
        // ====== Dynamic color ======
        item {
            SwitchSettingRow(
                title = stringResource(R.string.settings_dynamic_color),
                description = stringResource(R.string.settings_dynamic_color_desc),
                checked = settings.dynamicColor,
                onCheckedChange = { vm.setDynamicColor(it) }
            )
        }

        // ====== 日志记录 ======
        // ====== Log recording ======
        item {
            SwitchSettingRow(
                title = stringResource(R.string.settings_logging),
                description = stringResource(R.string.settings_logging_desc),
                checked = settings.logRecording,
                onCheckedChange = { vm.setLogRecording(it) }
            )
        }

        // ====== 清除缓存 ======
        // ====== Clear cache ======
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                    Text(stringResource(R.string.settings_cache), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        text = stringResource(R.string.settings_cache_desc),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    OutlinedButton(onClick = {
                        vm.clearCache { (count, bytes) ->
                            cacheMessage = if (count > 0)
                                MsgKey.k(MsgKey.MSG_CACHE_CLEARED, count.toString(), FileUtil.formatFileSize(bytes))
                            else
                                MsgKey.MSG_CACHE_EMPTY
                        }
                    }) {
                        Icon(Icons.Filled.CleaningServices, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.settings_clear_cache))
                    }
                    cacheMessage?.let {
                        Text(i18nMessage(it), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                    }
                }
            }
        }
    }

    // 模型文件管理器弹窗 (独立窗口, 置于 LazyColumn 外)
    // Model file manager dialog (standalone window, outside the LazyColumn)
    if (showModelManager) {
        ModelFileManagerDialog(
            root = vm.modelRootDir(),
            onDismiss = { showModelManager = false },
            onDeleted = { vm.scanModels() }
        )
    }
}

/** 电压角列表中的下一档 (更高电压); 已是最高档返回 null */
/** Next step up the voltage corner list (higher voltage); null at the top. */
private fun higherCorner(corner: Int): Int? {
    val idx = SettingsRepository.VoltageCorner.ALL.indexOf(corner)
    if (idx < 0 || idx >= SettingsRepository.VoltageCorner.ALL.size - 1) return null
    return SettingsRepository.VoltageCorner.ALL[idx + 1]
}

/** 电压角列表中的上一档 (更低电压); 已是最低档返回 null */
/** Previous step down the voltage corner list (lower voltage); null at the bottom. */
private fun lowerCorner(corner: Int): Int? {
    val idx = SettingsRepository.VoltageCorner.ALL.indexOf(corner)
    if (idx <= 0) return null
    return SettingsRepository.VoltageCorner.ALL[idx - 1]
}

/** 电压角在列表中的下标 (用于 Slider 离散映射) */
/** Index of a corner in the list (for discrete slider mapping). */
private fun idxOfCorner(corner: Int): Int {
    val idx = SettingsRepository.VoltageCorner.ALL.indexOf(corner)
    return if (idx >= 0) idx else SettingsRepository.VoltageCorner.ALL.size - 1
}

/**
 * MD3 面板容器: 以 tonal surface (surfaceContainerLow) 分层,
 * MD3 sub-panel container: layered with tonal surfaces (surfaceContainerLow),
 * 遵循 MD3 "以色调表面表达层级, 而非阴影" 的规范
 * following the MD3 guideline "express elevation with tonal surfaces, not shadows".
 */
@Composable
private fun PerfSubPanel(
    verticalArrangement: Arrangement.Vertical = Arrangement.spacedBy(4.dp),
    content: @Composable ColumnScope.() -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.medium)
            .background(MaterialTheme.colorScheme.surfaceContainerLow)
            .padding(Spacing.md),
        verticalArrangement = verticalArrangement
    ) { content() }
}

/** 离散电压角 Slider (13 档, MIN~MAX) */
/** Discrete voltage corner slider (13 steps, MIN~MAX). */
@Composable
private fun LockedCornerPanel(
    corner: Int,
    onChange: (Int) -> Unit
) {
    val corners = SettingsRepository.VoltageCorner.ALL
    val maxIdx = corners.size - 1
    var draft by remember { mutableStateOf(idxOfCorner(corner)) }
    val haptic = LocalHapticFeedback.current
    LaunchedEffect(corner) { draft = idxOfCorner(corner) }

    PerfSubPanel(verticalArrangement = Arrangement.spacedBy(0.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(stringResource(R.string.settings_corner_step), style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
            Text(
                text = SettingsRepository.VoltageCorner.name(corners[draft]),
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.SemiBold
            )
        }
        Slider(
            value = draft.toFloat(),
            onValueChange = { raw ->
                val newIdx = raw.roundToInt()
                if (newIdx != draft) {
                    draft = newIdx
                    // 每跳过一档触发一次轻触觉反馈 (M3 离散 Slider 规范)
                    // Haptic tick per step skipped (M3 discrete slider spec)
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                }
            },
            onValueChangeFinished = { onChange(corners[draft]) },
            valueRange = 0f..maxIdx.toFloat(),
            steps = maxIdx - 1,
            modifier = Modifier.fillMaxWidth()
        )
        Row(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = "MIN",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.weight(1f))
            Text(
                text = "MAX",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Text(
            text = stringResource(R.string.settings_corner_default_hint),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp)
        )
    }
}

/** 自动调角面板: 区间 RangeSlider + 目标角 Slider + DCVS 策略 FilterChip */
/** Auto-range panel: RangeSlider for the interval + target slider + DCVS FilterChips. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RangePanel(
    minCorner: Int,
    targetCorner: Int,
    maxCorner: Int,
    dcvsMode: Int,
    onMinChange: (Int) -> Unit,
    onTargetChange: (Int) -> Unit,
    onMaxChange: (Int) -> Unit,
    onDcvsModeChange: (Int) -> Unit
) {
    val corners = SettingsRepository.VoltageCorner.ALL
    val maxIdx = corners.size - 1
    var rangeDraft by remember {
        mutableStateOf(idxOfCorner(minCorner).toFloat()..idxOfCorner(maxCorner).toFloat())
    }
    var targetDraft by remember { mutableStateOf(idxOfCorner(targetCorner).toFloat()) }
    val haptic = LocalHapticFeedback.current

    // 外部持久值变化 (如切换模式后恢复) 时同步本地草稿
    // Sync the local drafts when external persisted values change (e.g. after switching modes)
    LaunchedEffect(minCorner, maxCorner, targetCorner) {
        rangeDraft = idxOfCorner(minCorner).toFloat()..idxOfCorner(maxCorner).toFloat()
        targetDraft = idxOfCorner(targetCorner).toFloat().coerceIn(rangeDraft.start, rangeDraft.endInclusive)
    }

    PerfSubPanel {
        Text(stringResource(R.string.settings_freq_range), style = MaterialTheme.typography.labelLarge)
        RangeSlider(
            value = rangeDraft,
            onValueChange = { raw ->
                // 任一端跳过档位时触发轻触觉反馈
                // Haptic tick whenever either handle skips a step
                if (raw.start.roundToInt() != rangeDraft.start.roundToInt() ||
                    raw.endInclusive.roundToInt() != rangeDraft.endInclusive.roundToInt()
                ) {
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                }
                rangeDraft = raw
            },
            onValueChangeFinished = {
                var lo = rangeDraft.start.roundToInt()
                var hi = rangeDraft.endInclusive.roundToInt()
                // 保证 min != max: 两端重叠时推开一格
                // Enforce min != max: push a handle apart when they overlap
                if (lo == hi) {
                    if (lo >= maxIdx) lo-- else hi++
                }
                onMinChange(corners[lo])
                onMaxChange(corners[hi])
            },
            valueRange = 0f..maxIdx.toFloat(),
            steps = maxIdx - 1,
            modifier = Modifier.fillMaxWidth()
        )
        Row(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = stringResource(R.string.settings_min, SettingsRepository.VoltageCorner.name(corners[rangeDraft.start.roundToInt()])),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = stringResource(R.string.settings_max, SettingsRepository.VoltageCorner.name(corners[rangeDraft.endInclusive.roundToInt()])),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(stringResource(R.string.settings_target_corner), style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
            Text(
                text = SettingsRepository.VoltageCorner.name(corners[targetDraft.roundToInt()]),
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.SemiBold
            )
        }
        Slider(
            value = targetDraft.coerceIn(rangeDraft.start, rangeDraft.endInclusive),
            onValueChange = { raw ->
                val newIdx = raw.roundToInt()
                if (newIdx != targetDraft.roundToInt()) {
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                }
                targetDraft = raw
            },
            onValueChangeFinished = { onTargetChange(corners[targetDraft.roundToInt()]) },
            // 跟随草稿区间: 范围拖动与目标角拖动互不越界 (持久值尚未提交)
            // Track the draft interval so the range and target sliders never cross (persisted values not yet committed)
            valueRange = rangeDraft.start..rangeDraft.endInclusive,
            steps = (rangeDraft.endInclusive - rangeDraft.start).toInt().coerceAtLeast(0),
            modifier = Modifier.fillMaxWidth()
        )
        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
        Text(stringResource(R.string.settings_dcvs_policy), style = MaterialTheme.typography.labelLarge)
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            DcvsModeChips.forEach { (mode, labelRes) ->
                FilterChip(
                    selected = dcvsMode == mode,
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        onDcvsModeChange(mode)
                    },
                    label = { Text(stringResource(labelRes)) }
                )
            }
        }
        Text(
            text = dcvsModeHint(dcvsMode),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = stringResource(R.string.settings_dcvs_rule),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

/** DCVS 模式短名 (FilterChip 标签, 资源 id) 与官方语义提示 */
/** DCVS mode short labels (FilterChip text, resource ids) with official semantics below. */
private val DcvsModeChips = listOf(
    SettingsRepository.DcvsMode.ADJUST_UP_DOWN to R.string.settings_dcvs_chip_up_down,
    SettingsRepository.DcvsMode.ADJUST_ONLY_UP to R.string.settings_dcvs_chip_only_up,
    SettingsRepository.DcvsMode.POWER_SAVER to R.string.settings_dcvs_chip_saver,
    SettingsRepository.DcvsMode.POWER_SAVER_AGGRESSIVE to R.string.settings_dcvs_chip_saver_aggressive,
    SettingsRepository.DcvsMode.PERFORMANCE to R.string.settings_dcvs_chip_performance,
    SettingsRepository.DcvsMode.DUTY_CYCLE to R.string.settings_dcvs_chip_duty
)

@Composable
private fun dcvsModeHint(mode: Int): String = when (mode) {
    SettingsRepository.DcvsMode.ADJUST_UP_DOWN -> stringResource(R.string.settings_dcvs_hint_up_down)
    SettingsRepository.DcvsMode.ADJUST_ONLY_UP -> stringResource(R.string.settings_dcvs_hint_only_up)
    SettingsRepository.DcvsMode.POWER_SAVER -> stringResource(R.string.settings_dcvs_hint_saver)
    SettingsRepository.DcvsMode.POWER_SAVER_AGGRESSIVE -> stringResource(R.string.settings_dcvs_hint_saver_aggressive)
    SettingsRepository.DcvsMode.PERFORMANCE -> stringResource(R.string.settings_dcvs_hint_performance)
    SettingsRepository.DcvsMode.DUTY_CYCLE -> stringResource(R.string.settings_dcvs_hint_duty)
    else -> stringResource(R.string.settings_dcvs_unknown, "0x%02X".format(mode))
}

/** 语言选择: 跟随系统 / 中文 / English (FilterChip 行) */
/** Language picker: System / Chinese / English (FilterChip row). */
@Composable
private fun LanguagePanel(
    current: Language,
    onChange: (Language) -> Unit
) {
    val haptic = LocalHapticFeedback.current
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Language.entries.forEach { lang ->
            FilterChip(
                selected = current == lang,
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                    onChange(lang)
                },
                label = { Text(stringResource(lang.labelRes)) }
            )
        }
    }
}

/**
 * 模型下载源选择面板 (HG 官方 / HM 国内镜像)。
 * Model download source selection panel (HG official / HM mirror for China).
 *
 * 中国用户可选择 HM 镜像站 (hf-mirror.com) 获得更快的下载速度。
 * Chinese users can select the HM mirror (hf-mirror.com) for faster download speeds.
 */
@Composable
private fun DownloadSourcePanel(
    current: DownloadSource,
    onChange: (DownloadSource) -> Unit
) {
    val haptic = LocalHapticFeedback.current
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        DownloadSource.entries.forEach { source ->
            FilterChip(
                selected = current == source,
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                    onChange(source)
                },
                label = {
                    Text(
                        when (source) {
                            DownloadSource.HG -> stringResource(R.string.settings_download_source_hg)
                            DownloadSource.HM -> stringResource(R.string.settings_download_source_hm)
                        }
                    )
                }
            )
        }
    }
}

/**
 * MD3 设置卡片中的开关行: 标题 + 描述 + Switch。
 * MD3 switch row for settings cards: title + description + Switch.
 *
 * 抽象重复的 Row + Column + Switch 模式, 统一排版与色调。
 * Abstracts the repeated Row + Column + Switch pattern with consistent layout and tones.
 * 卡片容器使用 surfaceContainerLow (MD3 tonal surface 层级规范)。
 * Card containers use surfaceContainerLow (MD3 tonal surface hierarchy).
 */
@Composable
private fun SwitchSettingRow(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(Spacing.lg).fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Switch(
                checked = checked,
                onCheckedChange = onCheckedChange
            )
        }
    }
}