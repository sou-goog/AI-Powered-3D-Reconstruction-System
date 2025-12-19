package com.irspace.imageto3d.ui.screen.result

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.irspace.imageto3d.data.model.GalleryItemDetail
import com.irspace.imageto3d.data.repository.TripoSRRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject

@HiltViewModel
class ResultViewModel @Inject constructor(
    private val repository: TripoSRRepository,
    savedStateHandle: SavedStateHandle
) : ViewModel() {
    
    private val jobId: String = checkNotNull(savedStateHandle["jobId"])
    
    private val _resultState = MutableStateFlow<ResultState>(ResultState.Loading)
    val resultState: StateFlow<ResultState> = _resultState.asStateFlow()
    
    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState.asStateFlow()
    
    init {
        loadResult()
    }
    
    fun loadResult() {
        viewModelScope.launch {
            _resultState.value = ResultState.Loading
            
            repository.getGalleryItem(jobId)
                .onSuccess { detail ->
                    _resultState.value = ResultState.Success(detail)
                }
                .onFailure { error ->
                    _resultState.value = ResultState.Error(
                        error.message ?: "Failed to load result"
                    )
                }
        }
    }
    
    fun downloadFile(filename: String, outputFile: File) {
        viewModelScope.launch {
            _downloadState.value = DownloadState.Downloading(filename)
            
            repository.downloadFile(jobId, filename, outputFile)
                .onSuccess { file ->
                    _downloadState.value = DownloadState.Success(filename, file)
                }
                .onFailure { error ->
                    _downloadState.value = DownloadState.Error(
                        error.message ?: "Download failed"
                    )
                }
        }
    }
    
    fun resetDownloadState() {
        _downloadState.value = DownloadState.Idle
    }
}

sealed class ResultState {
    object Loading : ResultState()
    data class Success(val detail: GalleryItemDetail) : ResultState()
    data class Error(val message: String) : ResultState()
}

sealed class DownloadState {
    object Idle : DownloadState()
    data class Downloading(val filename: String) : DownloadState()
    data class Success(val filename: String, val file: File) : DownloadState()
    data class Error(val message: String) : DownloadState()
}
