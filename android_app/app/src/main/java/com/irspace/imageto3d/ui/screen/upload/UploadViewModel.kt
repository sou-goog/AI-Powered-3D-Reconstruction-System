package com.irspace.imageto3d.ui.screen.upload

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.irspace.imageto3d.data.model.JobData
import com.irspace.imageto3d.data.repository.TripoSRRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class UploadViewModel @Inject constructor(
    private val repository: TripoSRRepository
) : ViewModel() {
    
    private val _uploadState = MutableStateFlow<UploadState>(UploadState.Idle)
    val uploadState: StateFlow<UploadState> = _uploadState.asStateFlow()
    
    private val _selectedImages = MutableStateFlow<List<Uri>>(emptyList())
    val selectedImages: StateFlow<List<Uri>> = _selectedImages.asStateFlow()
    
    private val _jobStatus = MutableStateFlow<JobData?>(null)
    val jobStatus: StateFlow<JobData?> = _jobStatus.asStateFlow()
    
    fun addImages(uris: List<Uri>) {
        val current = _selectedImages.value.toMutableList()
        val newImages = uris.filter { it !in current }.take(5 - current.size)
        _selectedImages.value = (current + newImages).take(5)
    }
    
    fun removeImage(uri: Uri) {
        _selectedImages.value = _selectedImages.value.filter { it != uri }
    }
    
    fun clearImages() {
        _selectedImages.value = emptyList()
    }
    
    fun uploadImages() {
        if (_selectedImages.value.isEmpty()) return
        
        viewModelScope.launch {
            _uploadState.value = UploadState.Uploading
            
            repository.uploadImages(_selectedImages.value)
                .onSuccess { response ->
                    _uploadState.value = UploadState.Success(response.jobId)
                    startPolling(response.jobId)
                }
                .onFailure { error ->
                    _uploadState.value = UploadState.Error(
                        error.message ?: "Upload failed"
                    )
                }
        }
    }
    
    private fun startPolling(jobId: String) {
        viewModelScope.launch {
            repository.pollJobStatus(jobId)
                .catch { error ->
                    _uploadState.value = UploadState.Error(
                        error.message ?: "Status polling failed"
                    )
                }
                .collect { jobData ->
                    _jobStatus.value = jobData
                    
                    when (jobData.status) {
                        "completed" -> {
                            _uploadState.value = UploadState.Completed(jobId, jobData)
                        }
                        "failed" -> {
                            _uploadState.value = UploadState.Error(
                                jobData.message
                            )
                        }
                    }
                }
        }
    }
    
    fun resetState() {
        _uploadState.value = UploadState.Idle
        _jobStatus.value = null
        clearImages()
    }
}

sealed class UploadState {
    object Idle : UploadState()
    object Uploading : UploadState()
    data class Success(val jobId: String) : UploadState()
    data class Completed(val jobId: String, val result: JobData) : UploadState()
    data class Error(val message: String) : UploadState()
}
