package com.sharp.qnn.ui.home

import android.app.Application
import android.content.Intent
import android.graphics.BitmapFactory
import android.media.ExifInterface
import android.net.Uri
import android.provider.DocumentsContract
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.BrokenImage
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.runtime.collectAsState
import com.sharp.qnn.SHARPApplication
import com.sharp.qnn.R
import com.sharp.qnn.data.ModelType
import com.sharp.qnn.data.ModelFormat
import com.sharp.qnn.data.ModelStatus
import com.sharp.qnn.data.SettingsRepository
import com.sharp.qnn.ui.components.ProgressCard
import com.sharp.qnn.util.FileUtil
import com.sharp.qnn.util.FileUtil.formatDuration
import com.sharp.qnn.util.FileUtil.formatFileSize
import com.sharp.qnn.util.MsgKey
import com.sharp.qnn.util.i18nMessage
import com.sharp.qnn.util.resolveMessage
import com.sharp.qnn.ui.theme.Spacing
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 主页 ViewModel：持有 Pipeline 状态、模型就绪情况与选中的图片。
 * Home view model: holds pipeline state, model readiness and the selected image.
 *
 * 图片 Uri 持有在 ViewModel 中, 切换页面不会丢失。
 * The image Uri lives in the ViewModel, so it survives page switches.
 */
class HomeViewModel(app: Application) : AndroidViewModel(app) {
    private val sharpApp = app as SHARPApplication

    val pipelineState = sharpApp.pipelineManager.state
    val models = sharpApp.modelStore.models
    val settingsFlow = sharpApp.settingsRepository.settingsFlow

    // 选中的图片 Uri 与文件名 (持久于 ViewModel, 切页不丢失)
    // Selected image Uri and file name (kept in the ViewModel across page switches)
    private val _selectedImageUri = mutableStateOf<Uri?>(null)
    val selectedImageUri: State<Uri?> = _selectedImageUri

    private val _selectedImageName = mutableStateOf<String?>(null)
    val selectedImageName: State<String?> = _selectedImageName

    // 图片详细信息
    // Detailed image info
    data class ImageDetails(
        val width: Int,
        val height: Int,
        val format: String,
        val focalLength: Float? = null,  // mm
        val fileSize: Long = 0
    )

    private val _imageDetails = mutableStateOf<ImageDetails?>(null)
    val imageDetails: State<ImageDetails?> = _imageDetails

    // 选图代数: 每次选图自增, 用于丢弃过期异步加载结果
    // Image-load generation: incremented on each selection, used to drop stale async results
    @Volatile
    private var imageLoadGeneration = 0L

    /** 设置选中的图片 */
    /** Sets the selected image. */
    fun setSelectedImage(uri: Uri?) {
        _selectedImageUri.value = uri
        _selectedImageName.value = uri?.let { FileUtil.getFileNameFromUri(sharpApp, it) }
        _imageDetails.value = null
        val gen = ++imageLoadGeneration
        if (uri != null) {
            viewModelScope.launch(Dispatchers.IO) {
                val details = loadImageDetails(uri)
                // 仅当本次仍是"最新选择"时才发布详情, 过期结果直接丢弃
                // Publish details only if this selection is still the latest; stale results are dropped
                if (gen == imageLoadGeneration) _imageDetails.value = details
            }
        }
    }

    private suspend fun loadImageDetails(uri: Uri): ImageDetails? = withContext(Dispatchers.IO) {
        try {
            val resolver = sharpApp.contentResolver

            // 尺寸 + MIME 类型
            // Size + MIME type
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
            val width = opts.outWidth
            val height = opts.outHeight
            val format = opts.outMimeType?.substringAfter("/")?.uppercase() ?: "UNKNOWN"

            // 文件大小
            // File size
            val size = FileUtil.getFileSize(sharpApp, uri)

            // EXIF 焦距 (仅 JPEG 支持)
            // EXIF focal length (JPEG only)
            var focal: Float? = null
            if (format == "JPEG" || format == "JPG") {
                resolver.openInputStream(uri)?.use { input ->
                    val exif = ExifInterface(input)
                    val focalVal = exif.getAttributeDouble(ExifInterface.TAG_FOCAL_LENGTH, 0.0)
                    if (focalVal > 0.0) focal = focalVal.toFloat()
                }
            }

            ImageDetails(width, height, format, focal, size)
        } catch (e: Exception) {
            null
        }
    }

    /** 运行完整推理流程 */
    /** Runs the full inference pipeline. */
    fun runPipeline(imageUri: Uri) {
        viewModelScope.launch {
            sharpApp.pipelineManager.runPipeline(imageUri)
        }
    }

    // PLY 导出消息
    // PLY export message
    private val _exportMessage = mutableStateOf<String?>(null)
    val exportMessage: State<String?> = _exportMessage

    /** 设置 PLY 保存目录 (SAF tree Uri 字符串) */
    /** Sets the PLY save directory (SAF tree Uri string). */
    fun setPlySaveLocation(uriString: String) {
        viewModelScope.launch { sharpApp.settingsRepository.setPlySaveLocation(uriString) }
    }

    /** 导出 PLY 到 SAF 所选目录 (经 contentResolver 写入, 无需存储权限) */
    /** Exports the PLY into the chosen SAF directory (written via contentResolver, no storage permission needed). */
    fun exportPly(treeUriString: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val src = sharpApp.pipelineManager.getLastPlyFile()
                ?: run {
                    _exportMessage.value = MsgKey.ERR_PLY_MISSING
                    return@launch
                }
            try {
                val resolver = sharpApp.contentResolver
                val treeUri = Uri.parse(treeUriString)
                val rootDoc = DocumentsContract.buildDocumentUriUsingTree(
                    treeUri,
                    DocumentsContract.getTreeDocumentId(treeUri)
                )

                val baseName = _selectedImageName.value
                    ?.substringBeforeLast('.')
                    ?.takeIf { it.isNotBlank() }
                    ?: "sharp"
                val name = "${baseName}_ply.ply"
                val doc = DocumentsContract.createDocument(resolver, rootDoc, "application/octet-stream", name)
                    ?: run {
                        _exportMessage.value = MsgKey.ERR_EXPORT_CREATE
                        return@launch
                    }

                resolver.openOutputStream(doc, "w")?.use { out ->
                    src.inputStream().use { it.copyTo(out) }
                } ?: run {
                    _exportMessage.value = MsgKey.ERR_EXPORT_STREAM
                    return@launch
                }

                _exportMessage.value = MsgKey.k(MsgKey.MSG_EXPORT_OK, name)
            } catch (e: Exception) {
                _exportMessage.value = MsgKey.k(MsgKey.ERR_EXPORT_FAIL, e.message ?: "")
            }
        }
    }

    fun clearExportMessage() {
        _exportMessage.value = null
    }
}

@Composable
fun HomeScreen(
    vm: HomeViewModel = viewModel(),
    snackbarHostState: SnackbarHostState = androidx.compose.runtime.remember { SnackbarHostState() }
) {
    val pipelineState by vm.pipelineState.collectAsState()
    val models by vm.models.collectAsState()
    val settings by vm.settingsFlow.collectAsState(initial = SettingsRepository.DEFAULTS)
    val selectedImageUri by vm.selectedImageUri
    val selectedImageName by vm.selectedImageName
    val imageDetails by vm.imageDetails
    val exportMessage by vm.exportMessage
    val context = LocalContext.current

    // 导出结果 → Snackbar (短时长, 避免长时间占据底部; 键在此处解析为当前语言)
    // Export result -> snackbar (short duration so it does not linger; keys are resolved in the current language)
    LaunchedEffect(exportMessage) {
        exportMessage?.let { msg ->
            snackbarHostState.showSnackbar(resolveMessage(context, msg), duration = SnackbarDuration.Short)
            vm.clearExportMessage()
        }
    }

    // 图片选择器
    // Image picker
    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri -> vm.setSelectedImage(uri) }

    // PLY 保存目录选择器 (SAF tree, 持久授权)
    // PLY save directory picker (SAF tree, persisted permission)
    val plyDirPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        if (uri != null) {
            val granted = runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
                true
            }.getOrDefault(false)
            if (granted) {
                vm.setPlySaveLocation(uri.toString())
            }
        }
    }

    val canRun = ModelType.entries.all { type ->
        val model = models[type]
        model != null && (model.format == ModelFormat.BIN || model.status == ModelStatus.COMPILED)
    } && !pipelineState.isRunning && selectedImageUri != null

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.md)
    ) {
        // 副标题 (主标题已固定在顶部 TopAppBar)
        // Subtitle (the main title lives in the top app bar)
        item {
            Text(
                text = stringResource(R.string.pipeline_subtitle),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // 图片选择与预览
        // Image selection and preview
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
                    Text(
                        text = stringResource(R.string.home_input_image),
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        FilledTonalButton(onClick = { imagePicker.launch("image/*") }) {
                            Icon(Icons.Filled.AddPhotoAlternate, contentDescription = null)
                            Spacer(Modifier.size(Spacing.sm))
                            Text(stringResource(R.string.home_select_image))
                        }
                        Spacer(Modifier.size(Spacing.md))
                        Text(
                            text = selectedImageName ?: stringResource(R.string.home_no_image),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    // AnimatedContent: 选择/未选择图片之间的淡入淡出过渡 (MD3 emphasized)
                    // AnimatedContent: cross-fade between selected/unselected states (MD3 emphasized)
                    AnimatedContent(
                        targetState = selectedImageUri,
                        transitionSpec = {
                            fadeIn(animationSpec = tween(300)) togetherWith
                                fadeOut(animationSpec = tween(200))
                        },
                        label = "imagePreview"
                    ) { uri ->
                        if (uri != null) {
                            Column(verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                                ImagePreview(
                                    uri = uri,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(260.dp)
                                        .clip(MaterialTheme.shapes.medium)
                                        .background(MaterialTheme.colorScheme.surfaceContainerHighest)
                                )
                                // 图片详细信息 (设置开启后显示)
                                // Image details (shown when enabled in settings)
                                if (settings.showImageDetails) {
                                    imageDetails?.let { d ->
                                        Card(
                                            colors = CardDefaults.cardColors(
                                                containerColor = MaterialTheme.colorScheme.secondaryContainer
                                            ),
                                            modifier = Modifier.fillMaxWidth()
                                        ) {
                                            Column(
                                                modifier = Modifier.padding(Spacing.md),
                                                verticalArrangement = Arrangement.spacedBy(2.dp)
                                            ) {
                                                Text(
                                                    text = stringResource(R.string.home_image_info),
                                                    style = MaterialTheme.typography.labelLarge,
                                                    fontWeight = FontWeight.SemiBold,
                                                    color = MaterialTheme.colorScheme.onSecondaryContainer
                                                )
                                                Text(
                                                    text = stringResource(R.string.home_image_size, d.width, d.height),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSecondaryContainer
                                                )
                                                Text(
                                                    text = stringResource(R.string.home_image_format, d.format),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSecondaryContainer
                                                )
                                                d.focalLength?.let { f ->
                                                    Text(
                                                        text = stringResource(R.string.home_image_focal, "%.1f".format(f)),
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSecondaryContainer
                                                    )
                                                }
                                                if (d.fileSize > 0) {
                                                    Text(
                                                        text = stringResource(R.string.home_image_file_size, formatFileSize(d.fileSize)),
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSecondaryContainer
                                                    )
                                                }
                                            }
                                        }
                                    } ?: Text(
                                        text = stringResource(R.string.home_loading_image_info),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        } else {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(260.dp)
                                    .clip(MaterialTheme.shapes.medium)
                                    .background(MaterialTheme.colorScheme.surfaceContainerHighest),
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Icon(
                                        imageVector = Icons.Filled.AddPhotoAlternate,
                                        contentDescription = null,
                                        modifier = Modifier.size(48.dp),
                                        tint = MaterialTheme.colorScheme.outline
                                    )
                                    Spacer(Modifier.size(Spacing.sm))
                                    Text(
                                        text = stringResource(R.string.home_no_image),
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // 运行按钮
        // Run button
        item {
            Text(
                text = stringResource(R.string.home_run_section),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Spacer(Modifier.size(Spacing.xs))
            Button(
                onClick = { selectedImageUri?.let { vm.runPipeline(it) } },
                enabled = canRun,
                modifier = Modifier.fillMaxWidth().height(48.dp)
            ) {
                Icon(Icons.Filled.PlayArrow, contentDescription = null)
                Spacer(Modifier.size(Spacing.sm))
                Text(stringResource(R.string.home_run))
            }
            val missing = ModelType.entries.filter { !models.containsKey(it) }
            if (missing.isNotEmpty()) {
                Text(
                    text = stringResource(
                        R.string.home_missing_models,
                        missing.joinToString { it.displayName }
                    ),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(top = Spacing.xs)
                )
            }
        }

        // 错误提示 (消息键在此解析为当前语言)
        // Error banner (message keys are resolved in the current language)
        pipelineState.errorMessage?.let { msg ->
            item {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(Spacing.lg),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(Spacing.sm)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.BrokenImage,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onErrorContainer
                        )
                        Text(
                            text = i18nMessage(msg),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                    }
                }
            }
        }

        // 推理进度
        // Inference progress
        if (pipelineState.stages.isNotEmpty() &&
            (pipelineState.isRunning || pipelineState.totalElapsedMs > 0 || pipelineState.errorMessage != null)) {
            item {
                Text(
                    text = stringResource(R.string.home_progress_section),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }

        // 各阶段进度卡片
        // Per-stage progress cards
        items(pipelineState.stages) { stage ->
            ProgressCard(stage = stage)
        }

        // 总耗时 (仅在推理完成后显示)
        // Total time (shown only after inference completes)
        if (!pipelineState.isRunning && pipelineState.totalElapsedMs > 0 && pipelineState.errorMessage == null) {
            item {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(Spacing.lg),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = stringResource(R.string.home_done_total),
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                        Text(
                            text = formatDuration(pipelineState.totalElapsedMs),
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                    }
                }
            }
        }

        // PLY 导出 (推理完成后显示导出按钮, 导出到设置中的保存位置)
        // PLY export (available after inference; exports to the directory set in settings)
        item {
            Text(
                text = stringResource(R.string.home_export_section),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Spacer(Modifier.size(Spacing.xs))
            val plyReady = !pipelineState.isRunning &&
                    pipelineState.totalElapsedMs > 0 &&
                    pipelineState.errorMessage == null

            // 导出到设置中的 SAF 保存目录; 未设置时先让用户选目录
            // Export to the SAF directory from settings; pick a directory first if unset
            val hasSaveDir = settings.plySaveLocation.startsWith("content://")

            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(Spacing.lg),
                    verticalArrangement = Arrangement.spacedBy(Spacing.sm)
                ) {
                    FilledTonalButton(
                        onClick = {
                            if (hasSaveDir) vm.exportPly(settings.plySaveLocation)
                            else plyDirPicker.launch(null)
                        },
                        enabled = plyReady,
                        modifier = Modifier.fillMaxWidth().height(48.dp)
                    ) {
                        Icon(Icons.Filled.Save, contentDescription = null)
                        Spacer(Modifier.size(Spacing.sm))
                        Text(if (hasSaveDir) stringResource(R.string.home_export_ply) else stringResource(R.string.home_choose_dir))
                    }
                    Text(
                        text = if (hasSaveDir)
                            stringResource(R.string.home_save_to, SettingsRepository.plySaveDisplayPath(settings.plySaveLocation))
                        else
                            stringResource(R.string.home_no_save_dir),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    if (!plyReady) {
                        Text(
                            text = stringResource(R.string.home_export_after_done),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}

/**
 * 图片预览: 从 Uri 解码 (含采样) 并以 fit 模式显示在矩形框内。
 * Image preview: decodes from the Uri (with sampling) and fits it inside the box.
 * 比例不同时留黑边 (letterbox), 不裁剪填满。
 * Aspect mismatches are letterboxed instead of cropped.
 */
@Composable
private fun ImagePreview(uri: Uri, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val bitmap by produceState<android.graphics.Bitmap?>(initialValue = null, uri) {
        value = withContext(Dispatchers.IO) {
            runCatching {
                context.contentResolver.openInputStream(uri)?.use { input ->
                    val opts = BitmapFactory.Options().apply { inSampleSize = 4 }
                    BitmapFactory.decodeStream(input, null, opts)
                }
            }.getOrNull()
        }
    }

    Box(
        modifier = modifier,
        contentAlignment = Alignment.Center
    ) {
        val bmp = bitmap
        if (bmp != null) {
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = stringResource(R.string.home_image_preview_cd),
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize()
            )
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Filled.BrokenImage,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.outline
                )
                Spacer(Modifier.size(Spacing.sm))
                Text(
                    text = stringResource(R.string.home_image_load_failed),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }
    }
}