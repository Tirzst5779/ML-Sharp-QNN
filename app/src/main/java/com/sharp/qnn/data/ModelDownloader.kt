package com.sharp.qnn.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * 预转换模型下载器 (P2P 分块并行下载)。
 * Pre-converted model downloader (P2P chunked parallel download).
 *
 * 从 HuggingFace 或国内镜像站 (HF Mirror) 下载 DLC 模型文件。
 * Downloads DLC model files from HuggingFace or HF Mirror (for Chinese users).
 *
 * 文件列表 (5个模型):
 * File list (5 models):
 *   pe.dlc, ie.dlc, rest_a.dlc, rest_b.dlc, rest_c.dlc
 *
 * 仓库路径: kjcpc/ML-Sharp-QNN
 * Repo path: kjcpc/ML-Sharp-QNN
 */
class ModelDownloader(private val context: Context) {

    companion object {
        /** 每个文件的分块数 (并发下载) */
        /** Chunk count per file (parallel download) */
        private const val CHUNK_COUNT = 4

        /** 连接超时 / Connection timeout */
        private const val CONNECT_TIMEOUT_MS = 15_000

        /** 读取超时 / Read timeout */
        private const val READ_TIMEOUT_MS = 30_000

        /** 下载缓冲区大小 / Download buffer size */
        private const val BUFFER_SIZE = 8192

        /** 分块下载阈值: 小于此值的文件直接用简单流式下载 */
        /** Chunked download threshold: files smaller than this use simple streaming */
        private const val CHUNK_THRESHOLD = 4 * 1024 * 1024L // 4 MB

        /** HuggingFace 官方源 / HuggingFace official URL */
        private const val HG_BASE_URL = "https://huggingface.co/kjcpc/ML-Sharp-QNN/resolve/main"

        /** HF 镜像站 (国内用户推荐) / HF Mirror (recommended for Chinese users) */
        private const val HM_BASE_URL = "https://hf-mirror.com/kjcpc/ML-Sharp-QNN/resolve/main"

        /** 模型精度目录 / Model precision directory */
        private const val PRECISION_DIR = "dlc/w8a16"

        /** 5 个 DLC 模型文件名 */
        /** 5 DLC model file names */
        val MODEL_FILES = listOf(
            "pe.dlc",
            "ie.dlc",
            "rest_a.dlc",
            "rest_b.dlc",
            "rest_c.dlc"
        )

        /**
         * 根据下载源获取基础 URL。
         * Get base URL based on download source.
         */
        fun getBaseUrl(source: SettingsRepository.DownloadSource): String =
            when (source) {
                SettingsRepository.DownloadSource.HG -> HG_BASE_URL
                SettingsRepository.DownloadSource.HM -> HM_BASE_URL
            }

        /**
         * 获取模型文件的完整 URL。
         * Get the full URL for a model file.
         */
        fun getModelUrl(source: SettingsRepository.DownloadSource, fileName: String): String =
            "${getBaseUrl(source)}/$PRECISION_DIR/$fileName"
    }

    /**
     * 取消令牌 (volatile, 确保跨协程可见)。
     * Cancellation token (volatile for cross-coroutine visibility).
     */
    @Volatile
    private var cancelled = false

    /**
     * 取消当前下载。
     * Cancel the current download.
     */
    fun cancel() {
        cancelled = true
    }

    /**
     * 下载所有 5 个模型文件到 dlc/ 目录。
     * Download all 5 model files to the dlc/ directory.
     *
     * 注意: onComplete 始终会被调用 (即使取消), 调用方在此统一重置状态。
     * Note: onComplete is always called (even on cancel), so callers reset state there.
     *
     * @param source 下载源 / download source
     * @param onProgress 进度回调 (文件名, 当前索引, 总数) / progress callback (fileName, currentIndex, total)
     * @param onComplete 完成回调 (成功数, 总数) — 始终调用 / completion callback (successCount, totalCount) — always called
     * @param onError 单文件错误回调 (错误信息) — 不重置整体状态 / single-file error callback (error message) — does NOT reset overall state
     */
    suspend fun downloadAll(
        source: SettingsRepository.DownloadSource,
        onProgress: (fileName: String, current: Int, total: Int) -> Unit = { _, _, _ -> },
        onComplete: (successCount: Int, totalCount: Int) -> Unit = { _, _ -> },
        onError: (message: String) -> Unit = {}
    ) {
        cancelled = false
        // 使用与 ModelStore 一致的模型根目录 (外部私有目录优先)
        // Use the same model root directory as ModelStore (external private dir preferred)
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        val dlcDir = File(base, "sharp_models/dlc")
        if (!dlcDir.exists()) {
            dlcDir.mkdirs()
        }

        val total = MODEL_FILES.size
        var successCount = 0
        var cancelledByUser = false

        for ((index, fileName) in MODEL_FILES.withIndex()) {
            if (cancelled) {
                cancelledByUser = true
                break // 跳出循环, 确保 onComplete 被调用
                // break to ensure onComplete is called
            }

            onProgress(fileName, index + 1, total)

            val url = getModelUrl(source, fileName)
            val destFile = File(dlcDir, fileName)

            try {
                downloadFile(url, destFile)
                successCount++
            } catch (e: Exception) {
                if (cancelled) {
                    cancelledByUser = true
                    break
                }
                onError("${e.message ?: "Unknown error"}")
                // 删除失败的部分下载文件
                // Delete partial/failed download file
                destFile.delete()
            }
        }

        // 始终调用 onComplete, 确保调用方重置状态
        // Always call onComplete so the caller resets state
        onComplete(successCount, total)
    }

    /**
     * 下载单个文件 (HEAD 探测 + 流式/分块下载)。
     * Download a single file (HEAD probe + streaming/chunked download).
     *
     * 先用 HEAD 请求获取文件大小, 再决定用简单流式还是分块并行下载。
     * First uses HEAD request to get file size, then decides simple or chunked download.
     */
    private suspend fun downloadFile(urlStr: String, destFile: File): Unit = withContext(Dispatchers.IO) {
        // 检查取消令牌
        // Check cancellation token
        if (cancelled) throw RuntimeException("Download cancelled")

        // HEAD 请求获取文件大小 (不下载正文, 避免浪费)
        // HEAD request to get file size (no body, avoids waste)
        val headConn = URL(urlStr).openConnection() as HttpURLConnection
        headConn.connectTimeout = CONNECT_TIMEOUT_MS
        headConn.readTimeout = READ_TIMEOUT_MS
        headConn.requestMethod = "HEAD"
        headConn.connect()

        val responseCode = headConn.responseCode
        val contentLength = headConn.contentLength
        headConn.disconnect()

        if (responseCode != HttpURLConnection.HTTP_OK) {
            throw RuntimeException("HTTP $responseCode for $urlStr")
        }

        // 小文件 (< 4MB) 或无法获取大小的文件 → 简单流式下载
        // Small files (< 4MB) or unknown size → simple streaming download
        if (contentLength <= 0 || contentLength < CHUNK_THRESHOLD) {
            downloadSimple(urlStr, destFile)
            return@withContext
        }

        // 大文件 → 尝试分块并行下载; 若服务器不支持 Range 则回退流式
        // Large files → try chunked parallel download; fall back to streaming if server doesn't support Range
        val chunkedSuccess = tryDownloadChunked(urlStr, destFile, contentLength)
        if (!chunkedSuccess) {
            // 服务器不支持 Range (返回 200 而非 206), 回退流式下载
            // Server doesn't support Range (returned 200 instead of 206), fall back to streaming
            downloadSimple(urlStr, destFile)
        }
    }

    /**
     * 尝试分块并行下载, 返回是否成功 (false 表示服务器不支持 Range)。
     * Try chunked parallel download, returns whether successful (false = server doesn't support Range).
     */
    private suspend fun tryDownloadChunked(
        urlStr: String, destFile: File, contentLength: Int
    ): Boolean = withContext(Dispatchers.IO) {
        // 先发一个 Range 探测请求, 确认服务器支持 206 Partial Content
        // Send a Range probe request first to confirm server supports 206 Partial Content
        val probeConn = URL(urlStr).openConnection() as HttpURLConnection
        probeConn.connectTimeout = CONNECT_TIMEOUT_MS
        probeConn.readTimeout = READ_TIMEOUT_MS
        probeConn.requestMethod = "GET"
        probeConn.setRequestProperty("Range", "bytes=0-0")
        probeConn.connect()

        val probeCode = probeConn.responseCode
        probeConn.disconnect()

        // 服务器不支持 Range → 回退流式下载
        // Server doesn't support Range → fall back to streaming
        if (probeCode != HttpURLConnection.HTTP_PARTIAL) {
            return@withContext false
        }

        // 服务器支持 Range → 分块并行下载
        // Server supports Range → chunked parallel download
        val tmpDir = File(destFile.parentFile, ".tmp_${destFile.name}")
        if (tmpDir.exists()) tmpDir.deleteRecursively()
        tmpDir.mkdirs()

        try {
            val chunkSize = contentLength / CHUNK_COUNT
            val chunkFiles = mutableListOf<File>()

            coroutineScope {
                for (i in 0 until CHUNK_COUNT) {
                    val start = (i * chunkSize).toLong()
                    val end = (if (i == CHUNK_COUNT - 1) contentLength - 1L else (i + 1L) * chunkSize - 1L)
                    val chunkFile = File(tmpDir, "chunk_$i")
                    chunkFiles.add(chunkFile)

                    launch {
                        downloadChunk(urlStr, chunkFile, start, end)
                    }
                }
            }

            mergeChunks(chunkFiles, destFile)
            return@withContext true
        } finally {
            // 清理临时目录
            // Clean up temp directory
            tmpDir.deleteRecursively()
        }
    }

    /**
     * 简单流式下载 (单线程, 全量)。
     * Simple streaming download (single thread, full file).
     */
    private suspend fun downloadSimple(urlStr: String, destFile: File): Unit = withContext(Dispatchers.IO) {
        if (cancelled) throw RuntimeException("Download cancelled")

        val conn = URL(urlStr).openConnection() as HttpURLConnection
        conn.connectTimeout = CONNECT_TIMEOUT_MS
        conn.readTimeout = READ_TIMEOUT_MS
        conn.requestMethod = "GET"
        conn.connect()

        val responseCode = conn.responseCode
        if (responseCode != HttpURLConnection.HTTP_OK) {
            conn.disconnect()
            throw RuntimeException("HTTP $responseCode for $urlStr")
        }

        try {
            conn.inputStream.use { input ->
                FileOutputStream(destFile).use { output ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        if (cancelled) {
                            throw RuntimeException("Download cancelled")
                        }
                        output.write(buffer, 0, bytesRead)
                    }
                }
            }
        } finally {
            conn.disconnect()
        }
    }

    /**
     * 下载单个分块 (Range 请求)。
     * Download a single chunk (Range request).
     */
    private suspend fun downloadChunk(urlStr: String, chunkFile: File, start: Long, end: Long): Unit =
        withContext(Dispatchers.IO) {
            if (cancelled) throw RuntimeException("Download cancelled")

            val conn = URL(urlStr).openConnection() as HttpURLConnection
            conn.connectTimeout = CONNECT_TIMEOUT_MS
            conn.readTimeout = READ_TIMEOUT_MS
            conn.requestMethod = "GET"
            conn.setRequestProperty("Range", "bytes=$start-$end")
            conn.connect()

            val responseCode = conn.responseCode
            if (responseCode != HttpURLConnection.HTTP_PARTIAL) {
                conn.disconnect()
                throw RuntimeException("HTTP $responseCode for chunk $urlStr (expected 206)")
            }

            try {
                conn.inputStream.use { input ->
                    FileOutputStream(chunkFile).use { output ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            if (cancelled) {
                                throw RuntimeException("Download cancelled")
                            }
                            output.write(buffer, 0, bytesRead)
                        }
                    }
                }
            } finally {
                conn.disconnect()
            }
        }

    /**
     * 合并分块文件为一个完整文件。
     * Merge chunk files into a single complete file.
     */
    private suspend fun mergeChunks(chunkFiles: List<File>, destFile: File): Unit = withContext(Dispatchers.IO) {
        if (cancelled) throw RuntimeException("Download cancelled")

        FileOutputStream(destFile).use { output ->
            val buffer = ByteArray(BUFFER_SIZE)
            for (chunkFile in chunkFiles) {
                if (cancelled) {
                    throw RuntimeException("Download cancelled")
                }
                chunkFile.inputStream().use { input ->
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                    }
                }
            }
        }
    }
}