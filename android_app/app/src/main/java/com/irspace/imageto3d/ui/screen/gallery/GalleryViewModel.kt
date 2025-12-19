package com.irspace.imageto3d.ui.screen.gallery

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.irspace.imageto3d.data.model.GalleryItem
import com.irspace.imageto3d.data.repository.TripoSRRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class GalleryViewModel @Inject constructor(
    private val repository: TripoSRRepository
) : ViewModel() {
    
    private val _galleryState = MutableStateFlow<GalleryState>(GalleryState.Loading)
    val galleryState: StateFlow<GalleryState> = _galleryState.asStateFlow()
    
    init {
        loadGallery()
    }
    
    fun loadGallery(offset: Int = 0) {
        viewModelScope.launch {
            _galleryState.value = GalleryState.Loading
            
            repository.getGallery(limit = 50, offset = offset)
                .onSuccess { response ->
                    _galleryState.value = GalleryState.Success(
                        items = response.items,
                        total = response.total,
                        hasMore = response.total > (offset + response.count)
                    )
                }
                .onFailure { error ->
                    _galleryState.value = GalleryState.Error(
                        error.message ?: "Failed to load gallery"
                    )
                }
        }
    }
    
    fun deleteItem(jobId: String) {
        viewModelScope.launch {
            repository.deleteJob(jobId)
                .onSuccess {
                    loadGallery() // Refresh
                }
        }
    }
}

sealed class GalleryState {
    object Loading : GalleryState()
    data class Success(
        val items: List<GalleryItem>,
        val total: Int,
        val hasMore: Boolean
    ) : GalleryState()
    data class Error(val message: String) : GalleryState()
}
