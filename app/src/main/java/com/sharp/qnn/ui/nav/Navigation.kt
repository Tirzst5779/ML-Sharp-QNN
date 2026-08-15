package com.sharp.qnn.ui.nav

import android.app.Activity
import android.content.Context
import androidx.activity.compose.LocalActivityResultRegistryOwner
import androidx.annotation.StringRes
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.sharp.qnn.R
import com.sharp.qnn.SHARPApplication
import com.sharp.qnn.data.SettingsRepository
import com.sharp.qnn.ui.home.HomeScreen
import com.sharp.qnn.ui.models.ModelsScreen
import com.sharp.qnn.ui.settings.SettingsScreen
import com.sharp.qnn.ui.theme.SHARPQNNTheme
import com.sharp.qnn.util.LocaleUtil

/** 顶部导航项定义 */
/** Top-level navigation destination definitions. */
private sealed class Dest(
    val route: String,
    val label: String,
    @StringRes val labelRes: Int,
    val icon: ImageVector
) {
    object Home : Dest("home", "Home", R.string.nav_home, Icons.Filled.Home)
    object Models : Dest("models", "Models", R.string.nav_models, Icons.Filled.ModelTraining)
    object Settings : Dest("settings", "Settings", R.string.nav_settings, Icons.Filled.Settings)
}

private val destinations = listOf(Dest.Home, Dest.Models, Dest.Settings)

/** 中等屏幕宽度阈值: >=600dp 改用 NavigationRail (MD3 自适应导航) */
/** Medium-screen width threshold: >=600dp switches to NavigationRail (MD3 adaptive navigation) */
private const val RAIL_BREAKPOINT_DP = 600

/** 内容区最大宽度 (Large 屏避免全宽拉伸) */
/** Max content width (avoids full-width stretching on large screens) */
private val CONTENT_MAX_WIDTH = 840.dp

/**
 * 应用根 Composable: 主题 + Scaffold (CenterAlignedTopAppBar + 自适应导航) + NavHost。
 * App root composable: theme + Scaffold (CenterAlignedTopAppBar + adaptive nav) + NavHost.
 *
 * 含 3 个目的地: Home / Models / Settings。
 * Contains three destinations: Home / Models / Settings.
 * Screen 组的 snackbarHostState 由顶层 Scaffold 持有 (消息统一浮于底部)。
 * The snackbarHostState is owned by the top-level Scaffold so all messages
 * float at the bottom consistently.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SHARPApp() {
    val app = LocalContext.current.applicationContext as SHARPApplication
    val settings by app.settingsRepository.settingsFlow.collectAsState(initial = SettingsRepository.DEFAULTS)

    // 按用户语言设置包装 Context, 覆盖 CompositionLocal 使其下所有 stringResource 跟随所选语言
    // Wrap the Context with the user language and override LocalContext so every
    // stringResource below follows the selected language
    val baseContext = LocalContext.current
    val activityResultRegistryOwner = LocalActivityResultRegistryOwner.current
        ?: error("ActivityResultRegistryOwner not available")
    val localizedContext: Context = remember(baseContext, settings.language) {
        LocaleUtil.wrap(baseContext, settings.language)
    }
    CompositionLocalProvider(
        LocalContext provides localizedContext,
        LocalActivityResultRegistryOwner provides activityResultRegistryOwner
    ) {
        SHARPQNNTheme(dynamicColor = settings.dynamicColor) {
            val navController = rememberNavController()
            val backStackEntry by navController.currentBackStackEntryAsState()
            val currentRoute = backStackEntry?.destination?.route
            val snackbarHostState = remember { SnackbarHostState() }

            // 切换目的地时立即收起当前 Snackbar, 避免跨页残留
            // Dismiss the current snackbar on destination change so it never lingers across pages
            LaunchedEffect(currentRoute) {
                snackbarHostState.currentSnackbarData?.dismiss()
            }

            // 自适应: 宽屏 (>600dp) 用导航 rail, 手机用底部导航栏
            // Adaptive: wide screens (>600dp) use a navigation rail, phones use a bottom bar
            val configuration = LocalConfiguration.current
            val useRail = configuration.screenWidthDp >= RAIL_BREAKPOINT_DP

            // TopAppBar 标题跟随页面
            // TopAppBar title follows the current page
            val topBarTitle = when (currentRoute) {
                Dest.Models.route -> stringResource(R.string.title_models)
                Dest.Settings.route -> stringResource(R.string.title_settings)
                else -> stringResource(R.string.title_home)
            }

            // 系统栏图标明暗与当前主题一致 (edge-to-edge 下图标浮于内容之上)
            // Match status/navigation bar icon appearance to the theme (edge-to-edge floats icons over content)
            // isAppearanceLightStatusBars = true 表示浅色(白色)图标, 适合深色背景
            // isAppearanceLightStatusBars = true means light (white) icons for dark backgrounds
            val darkTheme = isSystemInDarkTheme()
            val view = LocalView.current
            if (!view.isInEditMode) {
                SideEffect {
                    val window = (view.context as Activity).window
                    val insetsController = WindowCompat.getInsetsController(window, view)
                    insetsController.isAppearanceLightStatusBars = darkTheme
                    insetsController.isAppearanceLightNavigationBars = darkTheme
                }
            }

        val scrollBehavior = TopAppBarDefaults.pinnedScrollBehavior()

        val navItem = { dest: Dest ->
            val selected = currentRoute == dest.route ||
                    backStackEntry?.destination?.hierarchy?.any { it.route == dest.route } == true
            val onClick = {
                if (!selected) {
                    navController.navigate(dest.route) {
                        popUpTo(navController.graph.findStartDestination().id) {
                            saveState = true
                        }
                        launchSingleTop = true
                        restoreState = true
                    }
                }
            }
            Triple(dest, selected, onClick)
        }

        Scaffold(
            containerColor = MaterialTheme.colorScheme.surface,
            snackbarHost = { SnackbarHost(snackbarHostState) },
            topBar = {
                // MD3 CenterAlignedTopAppBar: 标题居中, 滚动时容器色从 surface 提升为 surfaceContainer
                // MD3 CenterAlignedTopAppBar: centered title; container lifts from surface to surfaceContainer on scroll
                CenterAlignedTopAppBar(
                    title = { Text(topBarTitle, style = MaterialTheme.typography.titleLarge) },
                    colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surface,
                        scrolledContainerColor = MaterialTheme.colorScheme.surfaceContainer
                    ),
                    scrollBehavior = scrollBehavior
                )
            },
            bottomBar = {
                if (!useRail) {
                    NavigationBar {
                        destinations.forEach { dest ->
                            val (d, selected, onClick) = navItem(dest)
                            NavigationBarItem(
                                selected = selected,
                                onClick = onClick,
                                icon = { Icon(d.icon, contentDescription = stringResource(d.labelRes)) },
                                label = { Text(stringResource(d.labelRes)) }
                            )
                        }
                    }
                }
            }
        ) { innerPadding ->
            Row(Modifier.fillMaxSize().padding(innerPadding)) {
                if (useRail) {
                    NavigationRail(
                        modifier = Modifier.fillMaxHeight(),
                        header = { /* 预留: 品牌图标 */ /* Reserved: brand icon */ }
                    ) {
                        destinations.forEach { dest ->
                            val (d, selected, onClick) = navItem(dest)
                            NavigationRailItem(
                                selected = selected,
                                onClick = onClick,
                                icon = { Icon(d.icon, contentDescription = stringResource(d.labelRes)) },
                                label = { Text(stringResource(d.labelRes)) }
                            )
                        }
                    }
                }

                // 内容区: 大屏限制最大宽度并居中 (MD3 反拉伸规范)
                // Content area: capped width, centered on large screens (MD3 anti-stretch spec)
                Box(
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                    contentAlignment = Alignment.TopCenter
                ) {
                    NavHost(
                        navController = navController,
                        startDestination = Dest.Home.route,
                        modifier = Modifier
                            .widthIn(max = CONTENT_MAX_WIDTH)
                            .fillMaxSize()
                            .nestedScroll(scrollBehavior.nestedScrollConnection),
                        // MD3 过渡动画: emphasized decelerate 入场, emphasized accelerate 出场
                        // MD3 transitions: emphasized decelerate on enter, emphasized accelerate on exit
                        enterTransition = {
                            slideInHorizontally(initialOffsetX = { it / 4 }, animationSpec = tween(300)) +
                                fadeIn(animationSpec = tween(300))
                        },
                        exitTransition = {
                            fadeOut(animationSpec = tween(200))
                        },
                        popEnterTransition = {
                            fadeIn(animationSpec = tween(300))
                        },
                        popExitTransition = {
                            slideOutHorizontally(targetOffsetX = { it / 4 }, animationSpec = tween(200)) +
                                fadeOut(animationSpec = tween(200))
                        }
                    ) {
                        composable(Dest.Home.route) { HomeScreen(snackbarHostState = snackbarHostState) }
                        composable(Dest.Models.route) { ModelsScreen(snackbarHostState = snackbarHostState) }
                        composable(Dest.Settings.route) { SettingsScreen() }
                    }
                }
            }
            }
        }
    }
}