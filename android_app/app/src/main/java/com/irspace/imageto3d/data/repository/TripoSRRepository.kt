package com.irspace.imageto3d.data.repository

import android.content.Context
import android.net.Uri
import com.irspace.imageto3d.data.api.TripoSRApiService
import com.irspace.imageto3d.data.model.*
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TripoSRRepository @Inject constructor(
    private val apiService: TripoSRApiService,
    @ApplicationContext private val context: Context
) {
    
    suspend fun healthCheck(): Result<HealthResponse> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.healthCheck()
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                Result.failure(Exception("Health check failed: ${response.message()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun uploadImages(imageUris: List<Uri>): Result<UploadResponse> = withContext(Dispatchers.IO) {
        try {
            val parts = imageUris.mapIndexed { index, uri ->
                val file = getFileFromUri(uri)
                val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
                MultipartBody.Part.createFormData("images", file.name, requestFile)
            }
            
            val response = apiService.uploadImages(parts)
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                val errorBody = response.errorBody()?.string() ?: "Upload failed"
                Result.failure(Exception(errorBody))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getJobStatus(jobId: String): Result<JobData> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.getJobStatus(jobId)
            if (response.isSuccessful && response.body()?.success == true) {
                response.body()?.data?.let {
                    Result.success(it)
                } ?: Result.failure(Exception("No data received"))
            } else {
                Result.failure(Exception(response.body()?.error ?: "Status check failed"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    fun pollJobStatus(jobId: String, intervalMs: Long = 2000): Flow<JobData> = flow {
        while (true) {
            val result = getJobStatus(jobId)
            result.onSuccess { jobData ->
                emit(jobData)
                if (jobData.status == "completed" || jobData.status == "failed") {
                    return@flow
                }
            }.onFailure {
                throw it
            }
            delay(intervalMs)
        }
    }.flowOn(Dispatchers.IO)
    
    suspend fun getPreviewImages(jobId: String): Result<List<PreviewImage>> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.getPreviewImages(jobId)
            if (response.isSuccessful && response.body()?.success == true) {
                Result.success(response.body()!!.previews)
            } else {
                Result.failure(Exception(response.body()?.error ?: "Failed to get previews"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getRenderFrames(jobId: String): Result<List<ImageInfo>> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.getRenderFrames(jobId)
            if (response.isSuccessful && response.body()?.success == true) {
                Result.success(response.body()!!.frames)
            } else {
                Result.failure(Exception("Failed to get render frames"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getGallery(
        limit: Int = 50,
        offset: Int = 0,
        sort: String = "newest"
    ): Result<GalleryResponse> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.getGallery(limit, offset, sort)
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                val errorMsg = "Failed to load gallery: ${response.code()} - ${response.message()}"
                android.util.Log.e("TripoSRRepository", errorMsg)
                Result.failure(Exception(errorMsg))
            }
        } catch (e: Exception) {
            android.util.Log.e("TripoSRRepository", "Gallery error", e)
            Result.failure(e)
        }
    }
    
    suspend fun getGalleryItem(jobId: String): Result<GalleryItemDetail> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.getGalleryItem(jobId)
            if (response.isSuccessful && response.body()?.success == true) {
                response.body()?.data?.let {
                    Result.success(it)
                } ?: Result.failure(Exception("No data received"))
            } else {
                Result.failure(Exception(response.body()?.error ?: "Failed to load item"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun deleteJob(jobId: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.deleteJob(jobId)
            if (response.isSuccessful && response.body()?.success == true) {
                Result.success(response.body()!!.message)
            } else {
                Result.failure(Exception("Failed to delete job"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun downloadFile(jobId: String, filename: String, outputFile: File): Result<File> = 
        withContext(Dispatchers.IO) {
            try {
                val response = apiService.downloadFile(jobId, filename)
                if (response.isSuccessful && response.body() != null) {
                    response.body()!!.byteStream().use { input ->
                        outputFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    Result.success(outputFile)
                } else {
                    Result.failure(Exception("Download failed"))
                }
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    
    private fun getFileFromUri(uri: Uri): File {
        val contentResolver = context.contentResolver
        val file = File(context.cacheDir, "upload_${System.currentTimeMillis()}.jpg")
        
        contentResolver.openInputStream(uri)?.use { input ->
            file.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        
        return file
    }
}
