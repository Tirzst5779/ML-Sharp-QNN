package com.sharp.qnn.ui.models

import android.app.Application
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.sharp.qnn.R
import com.sharp.qnn.SHARPApplication
import com.sharp.qnn.data.ModelEntry
import com.sharp.qnn.data.ModelStatus
import com.sharp.qnn.data.ModelType
import com.sharp.qnn.data.ModelDownloader
import com.sharp.qnn.data.SettingsRepository.DownloadSource
import com.sharp.qnn.util.FileUtil.formatFileSize
import com.sharp.qnn.util.LocaleUtil
import com.sharp.qnn.util.MsgKey
import com.sharp.qnn.util.i18nMessage
import com.sharp.qnn.util.resolveMessage
import com.sharp.qnn.util.resolveMessage
import com.sharp.qnn.ui.theme.Spacing
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 模型管理 ViewModel。
 * Model management view model.
 */
class ModelsViewModel(app: Application) : AndroidViewModel(app) {
    private val sharpApp = app as SHARPApplication

    val models = sharpApp.modelStore.models

    private val _busy = androidx.compose.runtime.mutableStateOf<ModelType?>(null)
    val busy: androidx.compose.runtime.State<ModelType?> = _busy

    private val _message = androidx.compose.runtime.mutableStateOf<String?>(null)
    val message: androidx.compose.runtime.State<String?> = _message

    // ====== 模型下载状态 ======
    // ====== Model download state ======
    private val _downloadProgress = androidx.compose.runtime.mutableStateOf<Pair<String, Pair<Int, Int>>?>(null)
    val downloadProgress: androidx.compose.runtime.State<Pair<String, Pair<Int, Int>>?> = _downloadProgress

    private val _downloading = androidx.compose.runtime.mutableStateOf(false)
    val downloading: androidx.compose.runtime.State<Boolean> = _downloading

    private val downloader = ModelDownloader(app)

    /**
     * 开始下载预转换模型 (一键下载全部5个DLC文件)。
     * Start downloading pre-converted models (one-tap download all 5 DLC files).
     */
    fun startDownload() {
        if (_downloading.value) return
        _downloading.value = true
        _downloadProgress.value = null

        viewModelScope.launch {
            val source = sharpApp.settingsRepository.settingsFlow.first().downloadSource

            downloader.downloadAll(
                source = source,
                onProgress = { fileName, current, total ->
                    _downloadProgress.value = fileName to (current to total)
                },
                onComplete = { successCount, totalCount ->
                    // onComplete 始终被调用 (包括取消时), 统一在此重置状态
                    // onComplete is always called (including on cancel), reset state here
                    _downloading.value = false
                    _downloadProgress.value = null
                    _message.value = MsgKey.k(MsgKey.MSG_DOWNLOAD_COMPLETE, successCount.toString(), totalCount.toString())
                    // 下载完成后重新扫描模型目录
                    // Re-scan model directory after download completes
                    scanModels()
                },
                onError = { errorMsg ->
                    // onError 仅报告单文件错误, 不重置整体下载状态
                    // onError only reports single-file errors, does NOT reset overall download state
                    _message.value = MsgKey.k(MsgKey.ERR_DOWNLOAD_FAIL, errorMsg)
                }
            )
        }
    }

    /**
     * 取消当前下载 (仅设置取消令牌, 状态由 onComplete 统一重置)。
     * Cancel the current download (only sets the cancellation token; state reset by onComplete).
     */
    fun cancelDownload() {
        downloader.cancel()
    }

    // 当前语言下的本地化 Context (消息键参数用)
    // Locale-wrapped context for the current language (message-key arguments)
    private suspend fun localeCtx() =
        LocaleUtil.wrap(sharpApp, sharpApp.settingsRepository.settingsFlow.first().language)

    /** 本地化模型名 (嵌入消息键参数) */
    /** Localized model name (embedded into message-key arguments) */
    private suspend fun modelName(type: ModelType): String =
        LocaleUtil.modelName(localeCtx(), type)

    /** 异常 → 本地化错误文本 (异常 message 本身可能是消息键) */
    /** Exception → localized error text (the exception message may itself be a key) */
    private suspend fun errorText(e: Throwable?): String =
        resolveMessage(localeCtx(), e?.message ?: "")

    fun importModel(type: ModelType, uri: Uri) {
        _busy.value = type
        viewModelScope.launch {
            // 文件复制 + JNI 均为阻塞 IO, 不能跑在主线程
            // File copy + JNI are blocking IO and must not run on the main thread
            val result = withContext(Dispatchers.IO) { sharpApp.modelStore.importModel(type, uri) }
            _message.value = if (result.isSuccess) MsgKey.k(MsgKey.MSG_IMPORT_OK, modelName(type))
            else MsgKey.k(MsgKey.ERR_IMPORT_FAIL, errorText(result.exceptionOrNull()))
            _busy.value = null
        }
    }

    fun compileModel(type: ModelType) {
        _busy.value = type
        viewModelScope.launch {
            // nativeInit / compileDlc 是阻塞 JNI (编译可能数十秒), 必须在 IO 线程
            // nativeInit / compileDlc are blocking JNI calls (compilation can take tens of seconds); must run on the IO thread
            val result = withContext(Dispatchers.IO) {
                val initResult = kotlin.runCatching { sharpApp.pipelineManager.ensureQnnInitialized() }
                if (initResult.isFailure) {
                    Result.failure(initResult.exceptionOrNull() ?: RuntimeException(MsgKey.ERR_QNN_INIT_DEFAULT))
                } else {
                    sharpApp.modelStore.compileModel(type)
                }
            }
            _message.value = if (result.isSuccess) MsgKey.k(MsgKey.MSG_COMPILE_OK, modelName(type))
            else MsgKey.k(MsgKey.ERR_COMPILE_FAIL, errorText(result.exceptionOrNull()))
            _busy.value = null
        }
    }

    /** 取消编译: 通知 native 中断并放弃本次编译结果 */
    /** Cancels compilation: asks native to abort and discards the in-progress result. */
    fun cancelCompile(type: ModelType) {
        sharpApp.modelStore.cancelCompile(type)
    }

    fun removeModel(type: ModelType) {
        _busy.value = type
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { sharpApp.modelStore.removeModel(type) }
            _message.value = if (result.isSuccess) MsgKey.k(MsgKey.MSG_REMOVE_OK, modelName(type))
            else MsgKey.k(MsgKey.ERR_REMOVE_FAIL, errorText(result.exceptionOrNull()))
            _busy.value = null
        }
    }

    fun clearMessage() { _message.value = null }

    /** 重新扫描模型目录 (进入模型页时调用), 目录为权威来源 */
    /** Re-scans the model directory (called on entry); the directory is the source of truth. */
    fun scanModels() {
        viewModelScope.launch(Dispatchers.IO) { sharpApp.modelStore.scanModelDirectory() }
    }
}

@Composable
fun ModelsScreen(
    vm: ModelsViewModel = viewModel(),
    snackbarHostState: SnackbarHostState = androidx.compose.runtime.remember { SnackbarHostState() }
) {
    val models by vm.models.collectAsState()
    val busy by vm.busy
    val message by vm.message
    val downloadProgress by vm.downloadProgress
    val downloading by vm.downloading
    val context = LocalContext.current

    // 每次切换到模型选项卡时重新扫描目录 (目录为权威来源, 自动识别 .bin/.dlc)
    // Re-scan the directory on every visit; the directory is the source of truth (.bin/.dlc auto-detected)
    LaunchedEffect(Unit) { vm.scanModels() }

    // 导入/编译/删除结果 → Snackbar (短时长, 避免长时间占据底部; 键在此解析为当前语言)
    // Import/compile/remove result -> snackbar (short duration so it does not linger; keys resolved in the current language)
    LaunchedEffect(message) {
        message?.let { msg ->
            snackbarHostState.showSnackbar(resolveMessage(context, msg), duration = SnackbarDuration.Short)
            vm.clearMessage()
        }
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.md)
    ) {
        item {
            Text(
                text = stringResource(R.string.models_subtitle),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // ====== 预转换模型下载 (P2P 分块并行) ======
        // ====== Pre-converted model download (P2P chunked parallel) ======
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                )
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(Spacing.lg),
                    verticalArrangement = Arrangement.spacedBy(Spacing.sm)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = stringResource(R.string.models_download_title),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                text = stringResource(R.string.models_download_desc),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    if (downloading) {
                        // 下载进度 / Download progress
                        downloadProgress?.let { (fileName, progress) ->
                            val (current, total) = progress
                            Text(
                                text = stringResource(R.string.models_download_downloading, fileName, current, total),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary
                            )
                            LinearProgressIndicator(
                                progress = { current.toFloat() / total },
                                modifier = Modifier.fillMaxWidth()
                            )
                        }

                        OutlinedButton(
                            onClick = { vm.cancelDownload() },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Filled.Cancel, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.size(Spacing.sm))
                            Text(stringResource(R.string.models_download_cancel))
                        }
                    } else {
                        // 下载按钮 / Download button
                        Button(
                            onClick = { vm.startDownload() },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Filled.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.size(Spacing.sm))
                            Text(stringResource(R.string.models_download_btn))
                        }
                    }
                }
            }
        }

        // 是否有模型正在编译 (编译期间其余模型的编译按钮禁用, 同一时刻只允许一个编译任务)
        // Any model compiling? (while compiling, other compile buttons are disabled so only one compile runs at a time)
        val anyCompiling = models.values.any { it.status == ModelStatus.COMPILING }

        items(ModelType.entries.toList()) { type ->
            ModelSlotCard(
                type = type,
                entry = models[type],
                isBusy = busy == type,
                compileLocked = anyCompiling && models[type]?.status != ModelStatus.COMPILING,
                onImport = { uri -> vm.importModel(type, uri) },
                onCompile = { vm.compileModel(type) },
                onCancelCompile = { vm.cancelCompile(type) },
                onRemove = { vm.removeModel(type) }
            )
        }
    }
}

/** 单个模型槽位卡片 */
/** A single model slot card. */
@Composable
private fun ModelSlotCard(
    type: ModelType,
    entry: ModelEntry?,
    isBusy: Boolean,
    compileLocked: Boolean,
    onImport: (Uri) -> Unit,
    onCompile: () -> Unit,
    onCancelCompile: () -> Unit,
    onRemove: () -> Unit
) {
    // 文件选择器 (支持 .bin / .dlc)
    // File picker (.bin / .dlc)
    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) onImport(uri)
    }

    val status = entry?.status ?: ModelStatus.NOT_IMPORTED
    val isCompiling = status == ModelStatus.COMPILING

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        )
    ) {
        Column(modifier = Modifier.padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
            // 标题行: 类型 + 状态标签
            // Title row: type + status chip
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = stringResource(type.nameRes),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f)
                )
                StatusChip(status = status)
            }

            // 详情
            // Details
            if (entry != null) {
                InfoLine(stringResource(R.string.models_file_name), entry.sourceName)
                InfoLine(stringResource(R.string.models_format), entry.format.name)
                InfoLine(stringResource(R.string.models_size), formatFileSize(entry.fileSize))
                entry.compiledBinPath?.let { InfoLine(stringResource(R.string.models_artifact), File(it).name) }
            } else {
                Text(
                    text = stringResource(R.string.models_not_imported_hint),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // 编译中: 显示进度提示 + 取消按钮
            // Compiling: progress hints + cancel button
            if (isCompiling) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(Spacing.md),
                        verticalArrangement = Arrangement.spacedBy(Spacing.sm)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Spacer(Modifier.size(Spacing.sm))
                            Text(
                                text = stringResource(R.string.models_compiling),
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                                modifier = Modifier.weight(1f)
                            )
                        }
                        // 不确定进度条 (native graphFinalize 无法报告进度)
                        // Indeterminate progress bar (native graphFinalize reports no progress)
                        LinearProgressIndicator(
                            modifier = Modifier.fillMaxWidth().height(4.dp),
                            color = MaterialTheme.colorScheme.primary,
                            trackColor = MaterialTheme.colorScheme.surface
                        )
                        Text(
                            text = stringResource(R.string.models_compile_cancel_hint),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                        )
                        Button(
                            onClick = onCancelCompile,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.errorContainer,
                                contentColor = MaterialTheme.colorScheme.onErrorContainer
                            )
                        ) {
                            Icon(Icons.Filled.Cancel, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.size(Spacing.sm))
                            Text(stringResource(R.string.models_cancel_compile))
                        }
                    }
                }
            } else {
                // 操作按钮 (非编译中状态)
                // Action buttons (not compiling)
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(Spacing.sm)
                ) {
                    // 导入按钮
                    // Import button
                    OutlinedButton(
                        onClick = { filePicker.launch("*/*") },
                        enabled = !isBusy,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Filled.UploadFile, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.models_import))
                    }

                    // 编译按钮
                    // Compile button
                    val canCompile = entry != null &&
                            entry.format == com.sharp.qnn.data.ModelFormat.DLC &&
                            (entry.status == ModelStatus.UNCOMPILED)
                    OutlinedButton(
                        onClick = onCompile,
                        enabled = canCompile && !isBusy && !compileLocked,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Filled.Build, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.models_compile))
                    }

                    // 删除按钮
                    // Remove button
                    OutlinedButton(
                        onClick = onRemove,
                        enabled = entry != null && !isBusy,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Filled.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.models_remove))
                    }

                    if (isBusy) {
                        Spacer(Modifier.size(Spacing.xs))
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    }
                }

                // 编译期间: 其余模型按钮禁用原因提示 (独立一行, 避免挤压按钮/撑高卡片)
                // Locked reason while another model compiles (its own line so buttons stay compact)
                if (compileLocked) {
                    Text(
                        text = stringResource(R.string.models_compile_locked),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            }
        }
    }
}

/**
 * 状态标签: 使用 MD3 tonal container 配对色 (容器色 + 内容色),
 * Status chip: uses MD3 tonal container pairs (container + on-container colors),
 * 而非单一文字色 + 中性背景, 提供更清晰的状态视觉反馈。
 * instead of plain text color on a neutral background, for clearer state feedback.
 *
 * - 已编译   → tertiaryContainer / onTertiaryContainer
 * - 未编译   → secondaryContainer / onSecondaryContainer
 * - 编译中   → primaryContainer / onPrimaryContainer
 * - 未导入   → surfaceContainerHigh / onSurfaceVariant
 */
@Composable
private fun StatusChip(status: ModelStatus) {
    val (containerColor, contentColor) = when (status) {
        ModelStatus.COMPILED -> MaterialTheme.colorScheme.tertiaryContainer to MaterialTheme.colorScheme.onTertiaryContainer
        ModelStatus.UNCOMPILED -> MaterialTheme.colorScheme.secondaryContainer to MaterialTheme.colorScheme.onSecondaryContainer
        ModelStatus.COMPILING -> MaterialTheme.colorScheme.primaryContainer to MaterialTheme.colorScheme.onPrimaryContainer
        ModelStatus.NOT_IMPORTED -> MaterialTheme.colorScheme.surfaceContainerHigh to MaterialTheme.colorScheme.onSurfaceVariant
    }
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = containerColor
    ) {
        Text(
            text = stringResource(status.labelRes),
            style = MaterialTheme.typography.labelMedium,
            color = contentColor,
            modifier = Modifier.padding(horizontal = Spacing.sm, vertical = Spacing.xs)
        )
    }
}

@Composable
private fun InfoLine(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = "$label: ",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium
        )
    }
}