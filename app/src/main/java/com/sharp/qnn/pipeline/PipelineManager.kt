package com.sharp.qnn.pipeline

import android.content.Context
import android.net.Uri
import com.sharp.qnn.R
import com.sharp.qnn.data.ModelEntry
import com.sharp.qnn.data.ModelFormat
import com.sharp.qnn.data.ModelStatus
import com.sharp.qnn.data.ModelType
import com.sharp.qnn.data.SettingsRepository
import com.sharp.qnn.data.ModelStore
import com.sharp.qnn.util.FileUtil
import com.sharp.qnn.util.FileUtil.ensureDir
import com.sharp.qnn.util.LocaleUtil
import com.sharp.qnn.util.MsgKey
import com.sharp.qnn.util.SkelExtractor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

/**
 * Pipeline 编排器。
 * Pipeline orchestrator.
 *
 * 串联完整推理流程:
 * Chains the full inference flow:
 * prepImage → runPre → PE → IE → Merge → REST(A/B/C) → Post(PLY)
 *
 * 通过 [ProgressCallback] 接收 native 层进度，并以 [StateFlow] 暴露给 UI。
 * Receives native progress through [ProgressCallback] and exposes it to the
 * UI as a [StateFlow].
 * 每步检查模型是否已编译并加载，DLC 需先编译。
 * Each step checks that models are compiled and loaded; DLCs must be compiled first.
 */
class PipelineManager(
    private val context: Context,
    private val modelStore: ModelStore,
    private val settings: SettingsRepository
) : ProgressCallback {

    private val _state = MutableStateFlow(PipelineState())
    val state: StateFlow<PipelineState> = _state.asStateFlow()

    private val runMutex = Mutex()

    // 运行时状态
    // Runtime state
    @Volatile
    private var initialized = false
    private val loadedModels = mutableSetOf<ModelType>()
    private val loadedModelsLock = Any()
    private var pipelineStartTime = 0L
    private var stageStartTime = 0L
    private var compileStageStartTime = 0L

    // 本次运行的本地化 Context (按用户语言包装; 语言切换在一次运行内保持一致)
    // Locale-wrapped context for this run (consistent for the whole run,
    // even if the language setting changes mid-run)
    private var langCtx: Context = context

    // prepImage 输出 (供后续 Post 阶段使用)
    // prepImage output (used by the later Post stage)
    private var prepMeta: FloatArray? = null

    /** 本次运行语言的本地化模型名 */
    /** Localized model name for the current run's language */
    private fun modelName(type: ModelType): String = LocaleUtil.modelName(langCtx, type)

    /**
     * 运行完整 Pipeline。
     * Run the full pipeline.
     * @param imageUri 用户选择的图片 URI
     * @param imageUri the image URI selected by the user
     * @return 成功与否
     * @return success/failure
     */
    suspend fun runPipeline(imageUri: Uri): Result<Unit> = runMutex.withLock {
        withContext(Dispatchers.IO) {
            try {
                // 锁定本次运行的语言 (消息细节按此语言本地化)
                // Pin the language for this run (detail messages use it)
                langCtx = LocaleUtil.wrap(context, settings.settingsFlow.first().language)

                // 重置状态 (loadedModels 保留, 已加载模型无需重新加载)
                // Reset state (loadedModels is kept; already-loaded models are reused)
                pipelineStartTime = System.currentTimeMillis()
                val hasUncompiled = ModelType.entries.any { type ->
                    modelStore.getModel(type)?.let {
                        it.format == ModelFormat.DLC && it.status != ModelStatus.COMPILED
                    } == true
                }
                val stages = if (hasUncompiled) {
                    listOf(StageState(id = COMPILE_STAGE_ID, name = "Model Compilation", nameRes = R.string.stage_compile)) + DEFAULT_STAGES
                } else {
                    DEFAULT_STAGES
                }
                _state.value = PipelineState(stages = stages, isRunning = true)

                // 注册回调
                // Register the callback
                runCatching { QnnJni.setProgressCallback(this@PipelineManager) }

                // 1. 初始化 QNN 运行时 + 确保模型就绪
                // 1. Initialize the QNN runtime and ensure models are ready
                ensureReady()

                // 2. 准备工作目录
                // 2. Prepare the work directory
                val workDir = File(context.cacheDir, "sharp_work").also {
                    if (it.exists()) FileUtil.deleteRecursively(it)
                    ensureDir(it)
                }

                // 3. 复制输入图片
                // 3. Copy the input image
                val fileName = FileUtil.getFileNameFromUri(context, imageUri)
                android.util.Log.i(TAG, "=== Pipeline started, image=$fileName ===")
                val ext = fileName.substringAfterLast('.', "jpg").lowercase()
                val inputFile = File(workDir, "input.$ext")
                if (!FileUtil.copyUriToFile(context, imageUri, inputFile)) {
                    fail(MsgKey.ERR_COPY_IMAGE_FAILED)
                    return@withContext Result.failure(IOException("copy image failed"))
                }

                // 4. Stage 0: 解码图片 (prepImage)
                // 4. Stage 0: decode the image (prepImage)
                val imageRaw = File(workDir, "image.raw")
                val meta = runStage(0) {
                    QnnJni.prepImage(inputFile.absolutePath, imageRaw.absolutePath)
                } ?: run {
                    fail(MsgKey.ERR_PREP_NULL)
                    return@withContext Result.failure(IOException("prepImage failed"))
                }
                prepMeta = meta

                // 5. Stage 1: 预处理切 Patch (runPre)
                // 5. Stage 1: pre-process and split patches (runPre)
                if (!runStage(1) { QnnJni.runPre(imageRaw.absolutePath, workDir.absolutePath) }) {
                    fail(MsgKey.ERR_PRE_FAILED)
                    return@withContext Result.failure(IOException("runPre failed"))
                }

                // 6. Stage 2: PE 推理
                // 6. Stage 2: PE inference
                if (!runStage(2) { QnnJni.runPatchEncoder(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_PE_INFER_FAILED)
                    return@withContext Result.failure(IOException("runPatchEncoder failed"))
                }
                // PE 输出已全部落盘, 模型不再需要: 立即释放 context 内存 (QnnContext_free)
                // All PE outputs are on disk and the model is no longer needed:
                // release its context memory immediately (QnnContext_free)
                android.util.Log.i(TAG, "PE done, releasing PE model")
                runCatching { QnnJni.freeModel(ModelType.PE.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.PE) }

                // 7. Stage 3: IE 推理
                // 7. Stage 3: IE inference
                if (!runStage(3) { QnnJni.runImageEncoder(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_IE_INFER_FAILED)
                    return@withContext Result.failure(IOException("runImageEncoder failed"))
                }
                android.util.Log.i(TAG, "IE done, releasing IE model")
                runCatching { QnnJni.freeModel(ModelType.IE.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.IE) }

                // 8. Stage 4: Merge
                // 8. Stage 4: Merge
                // 注意: 目录名必须与 JNI 层 (sharp_jni.cpp) 一致: out_pe / out_ie
                // Note: directory names must match the JNI layer (sharp_jni.cpp): out_pe / out_ie
                val peOutDir = File(workDir, "out_pe")
                val ieOutDir = File(workDir, "out_ie")
                if (!runStage(4) { QnnJni.runMerge(workDir.absolutePath, peOutDir.absolutePath, ieOutDir.absolutePath) }) {
                    fail(MsgKey.ERR_MERGE_FAILED)
                    return@withContext Result.failure(IOException("runMerge failed"))
                }

                // 9. Stage 5: REST Seg A (特征融合)
                // 9. Stage 5: REST Seg A (feature fusion)
                if (!runStage(5) { QnnJni.runRestSegA(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_REST_A_FAILED)
                    return@withContext Result.failure(IOException("runRestSegA failed"))
                }
                android.util.Log.i(TAG, "REST A done, releasing REST_A model")
                runCatching { QnnJni.freeModel(ModelType.REST_A.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.REST_A) }

                // 10. Stage 6: REST Seg B (视差估计)
                // 10. Stage 6: REST Seg B (disparity estimation)
                if (!runStage(6) { QnnJni.runRestSegB(workDir.absolutePath) }) {
                    fail(MsgKey.ERR_REST_B_FAILED)
                    return@withContext Result.failure(IOException("runRestSegB failed"))
                }
                android.util.Log.i(TAG, "REST B done, releasing REST_B model")
                runCatching { QnnJni.freeModel(ModelType.REST_B.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.REST_B) }

                // 11. Stage 7: REST Seg C (高斯增量)
                // 11. Stage 7: REST Seg C (gaussian delta)
                val fpx = prepMeta?.getOrNull(0) ?: 0f
                val origW = (prepMeta?.getOrNull(2) ?: 0f).toInt()
                val origH = (prepMeta?.getOrNull(3) ?: 0f).toInt()
                if (!runStage(7) { QnnJni.runRestSegC(workDir.absolutePath, fpx, origW) }) {
                    fail(MsgKey.ERR_REST_C_FAILED)
                    return@withContext Result.failure(IOException("runRestSegC failed"))
                }
                // REST C 输出已落盘 (delta.raw), 模型不再需要
                // REST C output is on disk (delta.raw); the model is no longer needed
                android.util.Log.i(TAG, "REST C done, releasing REST_C model")
                runCatching { QnnJni.freeModel(ModelType.REST_C.code) }
                synchronized(loadedModelsLock) { loadedModels.remove(ModelType.REST_C) }

                // 12. Stage 8: Post (PLY 生成)
                // 12. Stage 8: Post (PLY generation)
                val plyPath = File(workDir, "output.ply").absolutePath
                if (!runStage(8) { QnnJni.runPost(workDir.absolutePath, fpx, origW, origH, plyPath) }) {
                    fail(MsgKey.ERR_POST_FAILED)
                    return@withContext Result.failure(IOException("runPost failed"))
                }

                // 完成
                // Done
                val totalElapsed = System.currentTimeMillis() - pipelineStartTime
                android.util.Log.i(TAG, "=== Pipeline done, total ${totalElapsed}ms, PLY=$plyPath ===")
                _state.value = _state.value.copy(
                    isRunning = false,
                    totalElapsedMs = totalElapsed,
                    errorMessage = null
                )
                Result.success(Unit)
            } catch (e: Exception) {
                fail(e.message ?: e.toString())
                Result.failure(e)
            }
        }
    }

    /** 取消并重置当前 Pipeline 状态 */
    /** Cancel and reset the current pipeline state */
    fun reset() {
        _state.value = PipelineState()
        // 释放已加载模型的 HTP 内存
        // Release the HTP memory of loaded models
        synchronized(loadedModelsLock) {
            for (modelType in loadedModels) {
                runCatching { QnnJni.freeModel(modelType.code) }
            }
            loadedModels.clear()
        }
        runCatching { QnnJni.clearRestCache() }
    }

    /** 获取最近一次推理的 PLY 文件路径 (供导出使用) */
    /** Path of the PLY file from the latest run (for export) */
    fun getLastPlyFile(): File? {
        val workDir = File(context.cacheDir, "sharp_work")
        val ply = File(workDir, "output.ply")
        return if (ply.exists()) ply else null
    }

    // ====== 内部: 初始化与模型就绪 ======
    // ====== Internals: initialization and model readiness ======

    /** 确保 QNN 运行时已初始化 (幂等)。模型页单独编译前也需调用。 */
    /** Ensure the QNN runtime is initialized (idempotent). Also required before
     * standalone compilation from the models tab. */
    suspend fun ensureQnnInitialized() {
        if (initialized) return
        // 锁定本次语言 (编译进度细节按此本地化)
        // Pin the language for this call (compile progress details use it)
        langCtx = LocaleUtil.wrap(context, settings.settingsFlow.first().language)
        val libDir = context.applicationInfo.nativeLibraryDir
        // 按探测结果选择 Skel: skel 必须与设备架构精确匹配, 探测失败即报错
        // Pick the Skel from the probe result: the skel must match the device
        // architecture exactly, so a failed probe is a hard error
        val arch = QnnJni.probeHtpArch(libDir)
        if (arch.isBlank()) {
            throw IOException(MsgKey.ERR_QNN_INIT)
        }
        // HTP 性能配置必须在 nativeInit 前下发 (创建共享 backend/device 时生效)
        // The HTP perf config must be sent before nativeInit (it applies when
        // the shared backend/device is created)
        val s = settings.settingsFlow.first()
        QnnJni.setPerfConfig(
            type = s.perfType,
            lockedCorner = s.perfLockedCorner,
            minCorner = s.perfRangeMin,
            targetCorner = s.perfRangeTarget,
            maxCorner = s.perfRangeMax,
            dcvsMode = s.perfDcvsMode
        )
        val skelDir = SkelExtractor.extractSkel(context, arch)
            ?: throw IOException(MsgKey.k(MsgKey.ERR_SKEL, arch))
        val ok = QnnJni.nativeInit(libDir, skelDir, arch)
        if (!ok) throw IOException(MsgKey.ERR_QNN_NATIVE_INIT)
        initialized = true
    }

    private suspend fun ensureReady() {
        ensureQnnInitialized()

        // 待编译模型列表 (DLC 且未编译)
        // Models to compile (DLC and not yet compiled)
        val toCompile = ModelType.entries.filter { type ->
            val e = modelStore.getModel(type)
            e != null && e.format == ModelFormat.DLC && e.status != ModelStatus.COMPILED
        }
        val total = toCompile.size

        // 确保 PE / IE 均已编译并加载
        // Ensure every model is compiled and loaded
        if (total > 0) {
            compileStageStartTime = System.currentTimeMillis()
            onStageStart(COMPILE_STAGE_ID, "Model Compilation")
        }
        for (type in ModelType.entries) {
            val compileIndex = toCompile.indexOf(type) // -1 = 无需编译 / -1 = nothing to compile
            ensureModelReady(type, compileIndex, total)
        }
        if (total > 0) {
            onStageComplete(COMPILE_STAGE_ID, "Model Compilation", System.currentTimeMillis() - compileStageStartTime)
        }
    }

    private suspend fun ensureModelReady(type: ModelType, compileIndex: Int, compileTotal: Int) {
        synchronized(loadedModelsLock) { if (loadedModels.contains(type)) return }

        val entry: ModelEntry = modelStore.getModel(type)
            ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_NOT_IMPORTED, modelName(type)))

        // DLC 未编译则先编译
        // Compile first if the DLC is not compiled yet
        if (entry.status != ModelStatus.COMPILED) {
            // 编译开始: 进度显示已完成 x/总 (当前模型尚未计入)
            // Compilation start: progress shows x/total done (current model not counted yet)
            onProgress(
                COMPILE_STAGE_ID,
                compileIndex, compileTotal,
                0,
                MsgKey.k(MsgKey.DETAIL_COMPILING, modelName(type))
            )
            val compileResult = modelStore.compileModel(type)
            if (compileResult.isFailure) {
                throw IOException(
                    MsgKey.k(MsgKey.ERR_MODEL_COMPILE_FAILED, modelName(type), compileResult.exceptionOrNull()?.message ?: "")
                )
            }
            // 编译完成: 更新完成数与时耗
            // Compilation done: update the count and elapsed time
            onProgress(
                COMPILE_STAGE_ID,
                compileIndex + 1, compileTotal,
                System.currentTimeMillis() - compileStageStartTime,
                MsgKey.k(MsgKey.DETAIL_COMPILED, modelName(type))
            )
        }

        // 重新读取 (编译后 entry 可能已更新)
        // Re-read the entry (it may have been updated by compilation)
        val ready = modelStore.getModel(type) ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_MISSING, modelName(type)))
        val binPath = ready.runtimeBinPath
            ?: throw IllegalStateException(MsgKey.k(MsgKey.ERR_MODEL_NO_BIN, modelName(type)))

        val ok = QnnJni.loadContextBinary(type.code, binPath)
        if (!ok) throw IOException(MsgKey.k(MsgKey.ERR_MODEL_LOAD_FAILED, modelName(type)))
        android.util.Log.i(TAG, "Model loaded: ${type.displayName} (${type.code})")
        synchronized(loadedModelsLock) { loadedModels.add(type) }
    }

    // ====== 内部: 阶段执行 ======
    // ====== Internals: stage execution ======

    /** 执行单个阶段，自动发出 start / complete 回调 */
    /** Run a single stage, emitting start / complete callbacks automatically */
    private fun <T> runStage(stageId: Int, block: () -> T): T {
        val stage = _state.value.stages.first { it.id == stageId }
        stageStartTime = System.currentTimeMillis()
        android.util.Log.i(TAG, "Stage ${stage.id} start: ${stage.name}")
        onStageStart(stageId, stage.name)
        try {
            val result = block()
            val elapsed = System.currentTimeMillis() - stageStartTime
            android.util.Log.i(TAG, "Stage ${stage.id} done: ${stage.name} (${elapsed}ms)")
            // 仅当 pipeline 仍在运行时才标记阶段完成
            // Only mark the stage complete while the pipeline is still running
            if (_state.value.isRunning) {
                onStageComplete(stageId, stage.name, elapsed)
            }
            return result
        } catch (e: Exception) {
            // 异常时标记阶段失败
            // Mark the stage as failed on exception
            if (_state.value.isRunning) {
                fail(MsgKey.k(MsgKey.ERR_STAGE_EXCEPTION, stageId, e.message ?: e.toString()))
            }
            throw e
        }
    }

    private fun fail(message: String) {
        android.util.Log.e(TAG, "Pipeline failed: $message")
        // 释放已加载模型的 HTP 内存 (推理失败后清理, 避免泄漏)
        // Release HTP memory of loaded models after failure to avoid leaks
        synchronized(loadedModelsLock) {
            for (modelType in loadedModels) {
                runCatching { QnnJni.freeModel(modelType.code) }
            }
            loadedModels.clear()
        }
        runCatching { QnnJni.clearRestCache() }
        // 复位所有运行中的 stage (取消/异常后保持状态一致)
        // Reset every running stage so the state stays consistent after
        // cancellation or an exception
        val current = _state.value
        _state.value = current.copy(
            isRunning = false,
            errorMessage = message,
            stages = current.stages.map { s ->
                if (s.isRunning) s.copy(isRunning = false, detail = if (s.id == COMPILE_STAGE_ID) MsgKey.DETAIL_CANCELLED else s.detail)
                else s
            }
        )
    }

    // ====== ProgressCallback 实现 ======
    // ====== ProgressCallback implementation ======

    override fun onStageStart(stageId: Int, stageName: String) {
        updateStage(stageId) { it.copy(isRunning = true, isComplete = false, detail = "") }
    }

    override fun onProgress(stageId: Int, current: Int, total: Int, elapsedMs: Long, detail: String) {
        updateStage(stageId) {
            it.copy(
                current = current,
                total = if (total > 0) total else it.total,
                elapsedMs = if (elapsedMs > 0) elapsedMs else it.elapsedMs,
                detail = detail
            )
        }
    }

    override fun onStageComplete(stageId: Int, stageName: String, elapsedMs: Long) {
        updateStage(stageId) {
            val total = if (it.total > 0) it.total else it.current
            it.copy(
                isRunning = false,
                isComplete = true,
                current = if (total > 0) total else it.current,
                total = total,
                elapsedMs = if (elapsedMs > 0) elapsedMs else it.elapsedMs,
                detail = ""
            )
        }
    }

    override fun onLog(level: Int, message: String) {
        // 日志可在此转发到 Logcat 或 UI 日志面板 (预留)
        // Log lines may be forwarded to Logcat or a UI log panel here (reserved)
        android.util.Log.println(logPriority(level), TAG, message)
    }

    override fun onError(stageId: Int, message: String) {
        fail(MsgKey.k(MsgKey.ERR_STAGE_ERROR, stageId, message))
    }

    // ====== 内部: 状态更新 ======
    // ====== Internals: state updates ======

    private fun updateStage(stageId: Int, transform: (StageState) -> StageState) {
        val current = _state.value
        val newStages = current.stages.map { if (it.id == stageId) transform(it) else it }
        _state.value = current.copy(stages = newStages)
    }

    private fun logPriority(level: Int): Int = when (level) {
        0 -> android.util.Log.DEBUG
        1 -> android.util.Log.INFO
        2 -> android.util.Log.WARN
        3 -> android.util.Log.ERROR
        else -> android.util.Log.INFO
    }

    companion object {
        private const val TAG = "PipelineManager"

        // 模型编译阶段 (动态插入 stages 头部, 不与 0-8 冲突)
        // Model compilation stage (inserted at the head of stages; does not collide with 0-8)
        const val COMPILE_STAGE_ID = -1
    }
}