package com.sharp.qnn.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.InsertDriveFile
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material3.BasicAlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import com.sharp.qnn.R
import com.sharp.qnn.data.ModelType
import com.sharp.qnn.util.FileUtil.formatFileSize
import com.sharp.qnn.util.MsgKey
import com.sharp.qnn.util.i18nMessage
import com.sharp.qnn.ui.theme.Spacing
import java.io.File

/**
 * 模型文件管理器弹窗 (MD3 BasicAlertDialog)。
 * Model file manager dialog (MD3 BasicAlertDialog).
 *
 * 根目录为 sharp_models, 可进入子文件夹 (bin/ / dlc/), 支持多选删除文件。
 * Rooted at sharp_models; can drill into sub-folders (bin/ / dlc/) and multi-delete files.
 * 删除为真实删除 (应用私有目录, 无需权限)。
 * Deletion is final (app-private directory, no permission needed).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelFileManagerDialog(
    root: File,
    onDismiss: () -> Unit,
    onDeleted: () -> Unit
) {
    // 导航栈: 根 → bin/|dlc/ ...
    // Navigation stack: root -> bin/|dlc/ ...
    val stack = remember { mutableStateListOf<File>(root) }
    val current = stack.last()

    // 多选状态
    // Multi-select state
    var selectMode by remember { mutableStateOf(false) }
    val selected = remember { mutableStateListOf<String>() }

    // 删除结果提示
    // Delete result message
    var message by remember { mutableStateOf<String?>(null) }

    // 列表刷新标记: 删除文件后强制重列目录
    // Refresh tick: forces re-listing after deletions
    var refreshTick by remember { mutableIntStateOf(0) }

    val entries = remember(current, refreshTick) {
        (current.listFiles() ?: emptyArray<File>())
            .sortedWith(compareBy({ !it.isDirectory }, { it.name.lowercase() }))
            .toList()
    }

    fun enter(dir: File) {
        selectMode = false
        selected.clear()
        stack.add(dir)
    }

    fun back() {
        selectMode = false
        selected.clear()
        if (stack.size > 1) stack.removeAt(stack.size - 1)
    }

    fun toggle(file: File, checked: Boolean) {
        if (checked) {
            if (!selected.contains(file.absolutePath)) selected.add(file.absolutePath)
        } else {
            selected.remove(file.absolutePath)
        }
    }

    fun deleteSelected() {
        var deleted = 0
        for (path in selected) {
            runCatching {
                val f = File(path)
                if (f.exists() && !f.isDirectory && f.delete()) deleted++
            }
        }
        selected.clear()
        selectMode = false
        message = MsgKey.k(MsgKey.MSG_DELETED, deleted.toString())
        refreshTick++
        if (deleted > 0) onDeleted()
    }

    BasicAlertDialog(
        onDismissRequest = onDismiss,
        modifier = Modifier.fillMaxWidth(0.96f).fillMaxHeight(0.9f),
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            shape = MaterialTheme.shapes.extraLarge,
            color = MaterialTheme.colorScheme.surfaceContainerLow
        ) {
            Column(modifier = Modifier.fillMaxSize().padding(Spacing.md)) {
                // 标题栏
                // Title bar
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                    if (stack.size > 1) {
                        IconButton(onClick = { back() }) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.fm_back_cd))
                        }
                    } else {
                        Spacer(Modifier.width(48.dp))
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = stringResource(R.string.fm_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = relativePath(root, current),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    TextButton(onClick = {
                        selectMode = !selectMode
                        if (!selectMode) selected.clear()
                    }) {
                        Icon(
                            imageVector = if (selectMode) Icons.Filled.SelectAll else Icons.Filled.CheckCircle,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(Modifier.width(Spacing.xs))
                        Text(if (selectMode) stringResource(R.string.fm_done) else stringResource(R.string.fm_multiselect))
                    }
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Filled.Close, contentDescription = stringResource(R.string.fm_close_cd))
                    }
                }

                HorizontalDivider(modifier = Modifier.padding(vertical = Spacing.xs))

                // 文件列表
                // File list
                if (entries.isEmpty()) {
                    Box(modifier = Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                        Text(
                            text = stringResource(R.string.fm_empty),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                } else {
                    LazyColumn(modifier = Modifier.weight(1f).fillMaxWidth()) {
                        items(entries, key = { it.absolutePath }) { file ->
                            FileRow(
                                file = file,
                                selectMode = selectMode,
                                selected = selected.contains(file.absolutePath),
                                onClick = {
                                    if (file.isDirectory) {
                                        // 文件夹不可删除: 多选模式下不可选中, 仅普通模式可进入
                                        // Folders cannot be deleted: not selectable in multi-select; enterable only in normal mode
                                        if (!selectMode) enter(file)
                                    } else {
                                        selectMode = true
                                        toggle(file, !selected.contains(file.absolutePath))
                                    }
                                },
                                onCheckedChange = { toggle(file, it) }
                            )
                        }
                    }
                }

                // 底部消息 + 删除按钮
                // Bottom message + delete button
                if (message != null) {
                    Text(
                        text = i18nMessage(message.orEmpty()),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(vertical = Spacing.xs)
                    )
                }
                if (selectMode) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(top = Spacing.sm),
                        horizontalArrangement = Arrangement.End
                    ) {
                        Button(
                            onClick = { deleteSelected() },
                            enabled = selected.isNotEmpty(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.errorContainer,
                                contentColor = MaterialTheme.colorScheme.onErrorContainer
                            )
                        ) {
                            Text(stringResource(R.string.fm_delete_selected, selected.size.toString()))
                        }
                    }
                }
            }
        }
    }
}

/** 单行文件/目录 (MD3 ListItem) */
/** A single file/directory row (MD3 ListItem). */
@Composable
private fun FileRow(
    file: File,
    selectMode: Boolean,
    selected: Boolean,
    onClick: () -> Unit,
    onCheckedChange: (Boolean) -> Unit
) {
    ListItem(
        leadingContent = {
            Icon(
                imageVector = if (file.isDirectory) Icons.Filled.Folder
                else Icons.AutoMirrored.Filled.InsertDriveFile,
                contentDescription = null,
                tint = if (file.isDirectory) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(24.dp)
            )
        },
        headlineContent = {
            Text(
                text = file.name,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = if (file.isDirectory) FontWeight.SemiBold else FontWeight.Normal,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        },
        supportingContent = if (!file.isDirectory) {
            {
                Text(
                    text = when {
                        file.extension.equals("dlc", ignoreCase = true) -> stringResource(R.string.fm_dlc_hint)
                        nameStartsWithModelCode(file) -> stringResource(R.string.fm_artifact)
                        else -> stringResource(R.string.fm_bin_hint)
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else null,
        trailingContent = if (!file.isDirectory) {
            {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = formatFileSize(file.length()),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    if (selectMode) {
                        Spacer(Modifier.width(Spacing.sm))
                        Checkbox(checked = selected, onCheckedChange = { onCheckedChange(it) })
                    }
                }
            }
        } else null,
        modifier = Modifier.clickable(onClick = onClick)
    )
}

/** 文件是否为编译产物 (pe_xxx.bin 等 <code>_ 前缀) */
/** Whether a file is a compiled artifact (pe_xxx.bin, i.e. a <code>_ prefix). */
private fun nameStartsWithModelCode(file: File): Boolean {
    val base = file.name.substringBeforeLast('.').lowercase()
    return ModelType.entries.any { base.startsWith("${it.code}_") }
}

/** 相对根目录的显示路径 */
/** Display path relative to the root. */
private fun relativePath(root: File, current: File): String {
    val rel = root.toPath().relativize(current.toPath()).toString()
    return if (rel.isEmpty()) root.name else "${root.name}/$rel"
}