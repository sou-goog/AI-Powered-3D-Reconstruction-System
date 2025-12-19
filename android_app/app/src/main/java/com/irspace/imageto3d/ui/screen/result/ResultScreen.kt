package com.irspace.imageto3d.ui.screen.result

import android.content.Intent
import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import coil.compose.AsyncImage
import com.irspace.imageto3d.BuildConfig

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    viewModel: ResultViewModel = hiltViewModel(),
    onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    val resultState by viewModel.resultState.collectAsState()
    val downloadState by viewModel.downloadState.collectAsState()
    
    LaunchedEffect(downloadState) {
        when (val state = downloadState) {
            is DownloadState.Success -> {
                Toast.makeText(
                    context,
                    "Downloaded: ${state.filename}",
                    Toast.LENGTH_SHORT
                ).show()
                
                // Open file
                val uri = FileProvider.getUriForFile(
                    context,
                    "${context.packageName}.provider",
                    state.file
                )
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    setDataAndType(uri, "application/octet-stream")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                context.startActivity(Intent.createChooser(intent, "Open with"))
                
                viewModel.resetDownloadState()
            }
            is DownloadState.Error -> {
                Toast.makeText(context, state.message, Toast.LENGTH_SHORT).show()
                viewModel.resetDownloadState()
            }
            else -> {}
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("3D Model Result") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        when (val state = resultState) {
            is ResultState.Loading -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(padding),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }
            
            is ResultState.Success -> {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(padding)
                        .verticalScroll(rememberScrollState())
                ) {
                    // Video Preview
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        elevation = CardDefaults.cardElevation(4.dp)
                    ) {
                        Column {
                            AsyncImage(
                                model = BuildConfig.BASE_URL.replace("/api/", "") + 
                                        state.detail.previewImages.firstOrNull()?.url,
                                contentDescription = null,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(300.dp),
                                contentScale = ContentScale.Fit
                            )
                            
                            // Preview Images Row
                            if (state.detail.previewImages.size > 1) {
                                LazyRow(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    contentPadding = PaddingValues(16.dp)
                                ) {
                                    items(state.detail.previewImages) { preview ->
                                        AsyncImage(
                                            model = BuildConfig.BASE_URL.replace("/api/", "") + 
                                                    preview.url,
                                            contentDescription = null,
                                            modifier = Modifier.size(80.dp),
                                            contentScale = ContentScale.Crop
                                        )
                                    }
                                }
                            }
                        }
                    }
                    
                    // Download Buttons
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp)
                        ) {
                            Text(
                                text = "Download 3D Model",
                                style = MaterialTheme.typography.titleMedium
                            )
                            
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            // OBJ Download
                            DownloadButton(
                                text = "Download OBJ",
                                subtitle = formatFileSize(state.detail.files.obj.size),
                                icon = Icons.Default.Download,
                                onClick = {
                                    val file = context.getExternalFilesDir(null)
                                        ?.resolve("mesh.obj")
                                    file?.let { viewModel.downloadFile("mesh.obj", it) }
                                },
                                enabled = downloadState !is DownloadState.Downloading
                            )
                            
                            Spacer(modifier = Modifier.height(8.dp))
                            
                            // STL Download
                            DownloadButton(
                                text = "Download STL (3D Printing)",
                                subtitle = formatFileSize(state.detail.files.stl.size),
                                icon = Icons.Default.Download,
                                onClick = {
                                    val file = context.getExternalFilesDir(null)
                                        ?.resolve("mesh.stl")
                                    file?.let { viewModel.downloadFile("mesh.stl", it) }
                                },
                                enabled = downloadState !is DownloadState.Downloading
                            )
                            
                            Spacer(modifier = Modifier.height(8.dp))
                            
                            // Video Download
                            DownloadButton(
                                text = "Download Video",
                                subtitle = formatFileSize(state.detail.files.video.size),
                                icon = Icons.Default.VideoLibrary,
                                onClick = {
                                    val file = context.getExternalFilesDir(null)
                                        ?.resolve("render.mp4")
                                    file?.let { viewModel.downloadFile("render.mp4", it) }
                                },
                                enabled = downloadState !is DownloadState.Downloading
                            )
                        }
                    }
                    
                    // Info Card
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp)
                        ) {
                            Text(
                                text = "Information",
                                style = MaterialTheme.typography.titleMedium
                            )
                            
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            InfoRow("Job ID", state.detail.jobId)
                            InfoRow("Images Used", "${state.detail.imageCount}")
                            InfoRow("Status", state.detail.status.uppercase())
                            InfoRow("Render Frames", "${state.detail.renderFrames.size}")
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
            
            is ResultState.Error -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(padding),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(32.dp)
                    ) {
                        Icon(
                            Icons.Default.Error,
                            contentDescription = null,
                            modifier = Modifier.size(64.dp),
                            tint = MaterialTheme.colorScheme.error
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(text = state.message)
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(onClick = { viewModel.loadResult() }) {
                            Text("Retry")
                        }
                    }
                }
            }
        }
        
        // Download Progress
        if (downloadState is DownloadState.Downloading) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentAlignment = Alignment.Center
            ) {
                Card {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("Downloading...")
                    }
                }
            }
        }
    }
}

@Composable
fun DownloadButton(
    text: String,
    subtitle: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    onClick: () -> Unit,
    enabled: Boolean
) {
    OutlinedButton(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        enabled = enabled
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(text)
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Icon(icon, contentDescription = null)
        }
    }
}

@Composable
fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

private fun formatFileSize(bytes: Long): String {
    return when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024 * 1024 -> "${bytes / 1024} KB"
        else -> "${bytes / (1024 * 1024)} MB"
    }
}
