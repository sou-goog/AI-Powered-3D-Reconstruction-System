package com.irspace.imageto3d.ui.screen.upload

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
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
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import coil.compose.rememberAsyncImagePainter

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UploadScreen(
    viewModel: UploadViewModel = hiltViewModel(),
    onNavigateToResult: (String) -> Unit
) {
    val uploadState by viewModel.uploadState.collectAsState()
    val selectedImages by viewModel.selectedImages.collectAsState()
    val jobStatus by viewModel.jobStatus.collectAsState()
    
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        viewModel.addImages(uris)
    }
    
    LaunchedEffect(uploadState) {
        if (uploadState is UploadState.Completed) {
            val jobId = (uploadState as UploadState.Completed).jobId
            onNavigateToResult(jobId)
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Image to 3D") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Upload Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(4.dp)
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.CloudUpload,
                        contentDescription = null,
                        modifier = Modifier.size(64.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    Text(
                        text = "Upload Images",
                        style = MaterialTheme.typography.headlineSmall
                    )
                    
                    Text(
                        text = "Select 1-5 images for 3D reconstruction",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    Button(
                        onClick = { imagePickerLauncher.launch("image/*") },
                        enabled = selectedImages.size < 5 && uploadState !is UploadState.Uploading,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Default.AddPhotoAlternate, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Select Images (${selectedImages.size}/5)")
                    }
                }
            }
            
            // Selected Images
            if (selectedImages.isNotEmpty()) {
                Spacer(modifier = Modifier.height(16.dp))
                
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Selected Images",
                                style = MaterialTheme.typography.titleMedium
                            )
                            
                            if (uploadState !is UploadState.Uploading) {
                                TextButton(onClick = { viewModel.clearImages() }) {
                                    Text("Clear All")
                                }
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        LazyRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(selectedImages) { uri ->
                                SelectedImageItem(
                                    uri = uri,
                                    onRemove = { viewModel.removeImage(uri) },
                                    enabled = uploadState !is UploadState.Uploading
                                )
                            }
                        }
                    }
                }
            }
            
            // Upload Button
            if (selectedImages.isNotEmpty() && uploadState !is UploadState.Uploading) {
                Spacer(modifier = Modifier.height(16.dp))
                
                Button(
                    onClick = { viewModel.uploadImages() },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = selectedImages.isNotEmpty()
                ) {
                    Icon(Icons.Default.Upload, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Generate 3D Model")
                }
            }
            
            // Processing Status
            when (uploadState) {
                is UploadState.Uploading -> {
                    Spacer(modifier = Modifier.height(24.dp))
                    ProcessingCard(
                        message = "Uploading images...",
                        progress = null
                    )
                }
                is UploadState.Success -> {
                    Spacer(modifier = Modifier.height(24.dp))
                    jobStatus?.let { job ->
                        ProcessingCard(
                            message = job.lastMessage ?: job.message,
                            progress = job.progress / 100f
                        )
                    }
                }
                is UploadState.Error -> {
                    Spacer(modifier = Modifier.height(24.dp))
                    ErrorCard(
                        message = (uploadState as UploadState.Error).message,
                        onRetry = { viewModel.resetState() }
                    )
                }
                else -> {}
            }
        }
    }
}

@Composable
fun SelectedImageItem(
    uri: Uri,
    onRemove: () -> Unit,
    enabled: Boolean
) {
    Card(
        modifier = Modifier.size(100.dp)
    ) {
        Box {
            Image(
                painter = rememberAsyncImagePainter(uri),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
            
            if (enabled) {
                IconButton(
                    onClick = onRemove,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .size(32.dp)
                ) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Remove",
                        tint = MaterialTheme.colorScheme.error
                    )
                }
            }
        }
    }
}

@Composable
fun ProcessingCard(
    message: String,
    progress: Float?
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (progress != null) {
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth(),
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Text(
                    text = "${(progress * 100).toInt()}%",
                    style = MaterialTheme.typography.titleLarge
                )
            } else {
                CircularProgressIndicator()
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}

@Composable
fun ErrorCard(
    message: String,
    onRetry: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                Icons.Default.Error,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = MaterialTheme.colorScheme.error
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onErrorContainer
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Button(onClick = onRetry) {
                Text("Try Again")
            }
        }
    }
}
