package com.irspace.imageto3d.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.irspace.imageto3d.ui.screen.gallery.GalleryScreen
import com.irspace.imageto3d.ui.screen.result.ResultScreen
import com.irspace.imageto3d.ui.screen.upload.UploadScreen

sealed class Screen(val route: String) {
    object Gallery : Screen("gallery")
    object Upload : Screen("upload")
    object Result : Screen("result/{jobId}") {
        fun createRoute(jobId: String) = "result/$jobId"
    }
}

@Composable
fun AppNavigation(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = Screen.Gallery.route
    ) {
        composable(Screen.Gallery.route) {
            GalleryScreen(
                onNavigateToResult = { jobId ->
                    navController.navigate(Screen.Result.createRoute(jobId))
                },
                onNavigateToUpload = {
                    navController.navigate(Screen.Upload.route)
                }
            )
        }
        
        composable(Screen.Upload.route) {
            UploadScreen(
                onNavigateToResult = { jobId ->
                    navController.navigate(Screen.Result.createRoute(jobId)) {
                        popUpTo(Screen.Gallery.route)
                    }
                }
            )
        }
        
        composable(
            route = Screen.Result.route,
            arguments = listOf(
                navArgument("jobId") { type = NavType.StringType }
            )
        ) {
            ResultScreen(
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }
    }
}
