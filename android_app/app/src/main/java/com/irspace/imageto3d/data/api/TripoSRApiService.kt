package com.irspace.imageto3d.data.api

import com.irspace.imageto3d.data.model.*
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*

interface TripoSRApiService {
    
    @GET("api/health")
    suspend fun healthCheck(): Response<HealthResponse>
    
    @Multipart
    @POST("api/upload")
    suspend fun uploadImages(
        @Part images: List<MultipartBody.Part>
    ): Response<UploadResponse>
    
    @GET("api/status/{jobId}")
    suspend fun getJobStatus(
        @Path("jobId") jobId: String
    ): Response<StatusResponse>
    
    @Streaming
    @GET("api/download/{jobId}/{filename}")
    suspend fun downloadFile(
        @Path("jobId") jobId: String,
        @Path("filename") filename: String
    ): Response<ResponseBody>
    
    @GET("api/preview/{jobId}")
    suspend fun getPreviewImages(
        @Path("jobId") jobId: String
    ): Response<PreviewResponse>
    
    @GET("api/renders/{jobId}")
    suspend fun getRenderFrames(
        @Path("jobId") jobId: String
    ): Response<RenderFramesResponse>
    
    @GET("api/gallery")
    suspend fun getGallery(
        @Query("limit") limit: Int = 50,
        @Query("offset") offset: Int = 0,
        @Query("sort") sort: String = "newest"
    ): Response<GalleryResponse>
    
    @GET("api/gallery/{jobId}")
    suspend fun getGalleryItem(
        @Path("jobId") jobId: String
    ): Response<GalleryItemDetailResponse>
    
    @GET("api/jobs")
    suspend fun listJobs(
        @Query("status") status: String? = null,
        @Query("limit") limit: Int? = 50
    ): Response<JobListResponse>
    
    @GET("api/logs/{jobId}")
    suspend fun getJobLogs(
        @Path("jobId") jobId: String
    ): Response<LogsResponse>
    
    @DELETE("api/delete/{jobId}")
    suspend fun deleteJob(
        @Path("jobId") jobId: String
    ): Response<DeleteResponse>
}
