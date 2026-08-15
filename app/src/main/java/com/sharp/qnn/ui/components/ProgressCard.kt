package com.sharp.qnn.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.sharp.qnn.R
import com.sharp.qnn.pipeline.StageState
import com.sharp.qnn.util.FileUtil.formatDuration
import com.sharp.qnn.util.i18nMessage
import com.sharp.qnn.ui.theme.SHARPQNNTheme
import com.sharp.qnn.ui.theme.Spacing

/**
 * 单阶段进度卡片 (MD3 Card)。
 * A progress card for a single stage (MD3 Card).
 *
 * 展示: 阶段名、LinearProgressIndicator、当前/总数、耗时、完成勾号。
 * Shows: stage name, LinearProgressIndicator, current/total, elapsed time, done check.
 * 使用 animateColorAsState 在 idle/running/complete 之间平滑过渡强调色。
 * Uses animateColorAsState to smoothly transition the accent color between
 * idle / running / complete states.
 */
@Composable
fun ProgressCard(
    stage: StageState,
    modifier: Modifier = Modifier
) {
    // 强调色: 完成=tertiary, 运行中=primary, 待命=outline; animateColorAsState 平滑过渡
    // Accent: done=tertiary, running=primary, idle=outline; animated via animateColorAsState
    val targetAccent = when {
        stage.isComplete -> MaterialTheme.colorScheme.tertiary
        stage.isRunning -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.outline
    }
    val accent by animateColorAsState(
        targetValue = targetAccent,
        animationSpec = tween(durationMillis = 300),
        label = "accentColor"
    )

    // 运行中卡片使用更高层级的 tonal surface (surfaceContainerHigh), 完成后回退到 surfaceContainerLow
    // Running cards use a higher tonal surface (surfaceContainerHigh); done cards drop back to surfaceContainerLow
    val targetContainer = when {
        stage.isRunning -> MaterialTheme.colorScheme.surfaceContainerHigh
        else -> MaterialTheme.colorScheme.surfaceContainerLow
    }
    val containerColor by animateColorAsState(
        targetValue = targetContainer,
        animationSpec = tween(durationMillis = 300),
        label = "containerColor"
    )

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor)
    ) {
        Column(
            modifier = Modifier.padding(Spacing.md),
            verticalArrangement = Arrangement.spacedBy(Spacing.sm)
        ) {
            // 标题行: 阶段名 + 完成勾号
            // Title row: stage name + done check
            // 阶段名优先取本地化资源 (nameRes), 无资源时回退英文规范名
            // Stage name prefers the localized resource (nameRes) and falls back to the English name
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = if (stage.nameRes != 0) stringResource(stage.nameRes) else stage.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f)
                )
                AnimatedVisibility(visible = stage.isComplete) {
                    Icon(
                        imageVector = Icons.Filled.CheckCircle,
                        contentDescription = stringResource(R.string.fm_done),
                        tint = MaterialTheme.colorScheme.tertiary,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            // 进度条 (运行中或已完成时显示)
            // Progress bar (visible while running or done)
            if (stage.isRunning || stage.isComplete) {
                LinearProgressIndicator(
                    progress = { stage.progress },
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                    color = accent,
                    trackColor = MaterialTheme.colorScheme.surfaceContainerHighest
                )
            }

            // 详情行: 当前/总数 · 耗时
            // Detail row: current/total · elapsed time
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                val progressText = if (stage.total > 0) {
                    "${stage.current} / ${stage.total}"
                } else if (stage.detail.isNotBlank()) {
                    stage.detail
                } else {
                    ""
                }
                if (progressText.isNotBlank()) {
                    Text(
                        text = progressText,
                        style = MaterialTheme.typography.bodySmall,
                        color = accent
                    )
                }
                if (stage.elapsedMs > 0) {
                    Spacer(Modifier.width(Spacing.sm))
                    Text(
                        text = formatDuration(stage.elapsedMs),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // 运行中详情 (跳过默认空详情)
            // Running detail (skips the empty-detail placeholder)
            if (stage.isRunning && stage.detail.isNotBlank()) {
                Text(
                    text = i18nMessage(stage.detail),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

/** 预览: 运行中的阶段卡片 */
/** Preview: a running stage card */
@Preview(showBackground = true, widthDp = 360)
@Composable
private fun ProgressCardRunningPreview() {
    SHARPQNNTheme(dynamicColor = false) {
        ProgressCard(
            stage = StageState(
                id = 2, name = "图块编码",
                current = 21, total = 35, elapsedMs = 84_532,
                detail = "Patch 21/35 · 峰值 1.2s/patch",
                isRunning = true
            )
        )
    }
}

/** 预览: 已完成阶段卡片 */
/** Preview: a completed stage card */
@Preview(showBackground = true, widthDp = 360)
@Composable
private fun ProgressCardDonePreview() {
    SHARPQNNTheme(dynamicColor = false) {
        ProgressCard(
            stage = StageState(
                id = 4, name = "特征合并",
                current = 6, total = 6, elapsedMs = 210,
                isComplete = true
            )
        )
    }
}