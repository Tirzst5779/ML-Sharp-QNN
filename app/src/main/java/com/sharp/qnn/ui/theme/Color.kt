package com.sharp.qnn.ui.theme

import androidx.compose.ui.graphics.Color

// ====== 蓝紫色调色板 (动态色彩不可用时的 fallback) ======
// ====== Blue-purple palette (fallback when dynamic color is unavailable) ======

// 主色调 - 蓝紫
// Primary - blue purple
val BluePurple10 = Color(0xFF1A1B4B)
val BluePurple20 = Color(0xFF2E2F6E)
val BluePurple30 = Color(0xFF45468F)
val BluePurple40 = Color(0xFF5C5DB0)
val BluePurple80 = Color(0xFFC4C4FF)
val BluePurple90 = Color(0xFFE3E0FF)

// 次要色 - 柔和紫灰
// Secondary - soft purple grey
val Secondary40 = Color(0xFF5B5D72)
val Secondary80 = Color(0xFFC4C5DC)
val Secondary90 = Color(0xFFE0E1F9)

// 第三色 - 蓝青
// Tertiary - blue cyan
val Tertiary40 = Color(0xFF3A5BA0)
val Tertiary80 = Color(0xFFA6C8FF)
val Tertiary90 = Color(0xFFD5E4FF)

// 错误色
// Error colors
val Error40 = Color(0xFFBA1A1A)
val Error80 = Color(0xFFFFB4AB)
val Error90 = Color(0xFFFFDAD6)

// 中性色
// Neutral colors
val Neutral10 = Color(0xFF1B1B21)
val Neutral20 = Color(0xFF303036)
val Neutral90 = Color(0xFFE4E1E9)
val Neutral95 = Color(0xFFF3F0F7)
val Neutral99 = Color(0xFFFDFBFF)

// 中性容器色阶 (surfaceContainer 系, 同一蓝紫中性家族)
// Neutral container scale (surfaceContainer family, same blue-purple neutral family)
// --- 亮色模式 (升序: lowest < low < container < high < highest) ---
// --- Light mode (ascending: lowest < low < container < high < highest) ---
val LightSurfaceContainerLowest = Color(0xFFFFFFFF)
val LightSurfaceContainerLow = Color(0xFFF7F5FA)
val LightSurfaceContainer = Color(0xFFF2F0F6)
val LightSurfaceContainerHigh = Color(0xFFEDEBF2)
val LightSurfaceContainerHighest = Color(0xFFE7E5ED)
val LightSurfaceDim = Color(0xFFD9D7DF)
val LightSurfaceBright = Color(0xFFFDFBFF)

// --- 暗色模式 (在 Neutral10 表面基础上抬升) ---
// --- Dark mode (lifted from the Neutral10 surface) ---
val DarkSurfaceContainerLowest = Color(0xFF121218)
val DarkSurfaceContainerLow = Color(0xFF232329)
val DarkSurfaceContainer = Color(0xFF272730)
val DarkSurfaceContainerHigh = Color(0xFF32323B)
val DarkSurfaceContainerHighest = Color(0xFF3D3C46)
val DarkSurfaceDim = Color(0xFF1B1B21)
val DarkSurfaceBright = Color(0xFF41414C)

val NeutralVariant30 = Color(0xFF47464F)
val NeutralVariant50 = Color(0xFF787680)
val NeutralVariant60 = Color(0xFF918F9A)
val NeutralVariant70 = Color(0xFFC9C6D2)
val NeutralVariant80 = Color(0xFFC9C5D0)
val NeutralVariant90 = Color(0xFFE5E1E9)