package com.irspace.imageto3d.data.model

import com.google.gson.annotations.SerializedName

// ==================== API Responses ====================

data class HealthResponse(
    val status: String,
    val message: String,
    val device: String,
    val model: String,
    val version: String
)

data class UploadResponse(
    val success: Boolean,
    @SerializedName("job_id")
    val jobId: String,
    val status: String,
    val message: String,
    @SerializedName("image_count")
    val imageCount: Int,
    val filenames: List<String>? = null,
    @SerializedName("progress_stream")
    val progressStream: String? = null,
    val error: String? = null
)

data class StatusResponse(
    val success: Boolean,
    val data: JobData? = null,
    val error: String? = null
)

data class JobData(
    @SerializedName("job_id")
    val jobId: String,
    val status: String, // queued, processing, completed, failed
    val progress: Int, // 0-100
    val message: String,
    @SerializedName("created_at")
    val createdAt: Long,
    @SerializedName("image_count")
    val imageCount: Int,
    val result: JobResult? = null,
    val logs: List<LogEntry>? = null,
    val filenames: List<String>? = null,
    @SerializedName("last_message")
    val lastMessage: String? = null,
    val error: String? = null
)

data class JobResult(
    @SerializedName("job_id")
    val jobId: String,
    @SerializedName("obj_file")
    val objFile: String,
    @SerializedName("stl_file")
    val stlFile: String,
    @SerializedName("video_file")
    val videoFile: String,
    @SerializedName("preview_images")
    val previewImages: List<String>,
    @SerializedName("render_frames")
    val renderFrames: List<String>,
    @SerializedName("input_images")
    val inputImages: List<String>,
    @SerializedName("file_sizes")
    val fileSizes: FileSizes,
    val timestamp: Long
)

data class FileSizes(
    val obj: Long,
    val stl: Long,
    val video: Long
)

data class LogEntry(
    val message: String,
    val timestamp: String,
    val step: Int? = null,
    @SerializedName("total_steps")
    val totalSteps: Int? = null
)

data class PreviewResponse(
    val success: Boolean,
    val previews: List<PreviewImage>,
    val error: String? = null
)

data class PreviewImage(
    val index: Int,
    val data: String // base64 encoded
)

data class GalleryResponse(
    val success: Boolean,
    val total: Int,
    val limit: Int,
    val offset: Int,
    val count: Int,
    val items: List<GalleryItem>
)

data class GalleryItem(
    @SerializedName("job_id")
    val jobId: String,
    val thumbnail: String?,
    @SerializedName("preview_images")
    val previewImages: List<String>,
    @SerializedName("input_images")
    val inputImages: List<String>,
    @SerializedName("video_url")
    val videoUrl: String,
    @SerializedName("obj_url")
    val objUrl: String,
    @SerializedName("stl_url")
    val stlUrl: String,
    @SerializedName("created_at")
    val createdAt: Long,
    @SerializedName("image_count")
    val imageCount: Int,
    val status: String,
    @SerializedName("file_sizes")
    val fileSizes: FileSizes,
    val filenames: List<String>? = null
)

data class GalleryItemDetailResponse(
    val success: Boolean,
    val data: GalleryItemDetail? = null,
    val error: String? = null
)

data class GalleryItemDetail(
    @SerializedName("job_id")
    val jobId: String,
    @SerializedName("created_at")
    val createdAt: Long,
    val status: String,
    @SerializedName("image_count")
    val imageCount: Int,
    val filenames: List<String>? = null,
    val files: FileInfo,
    @SerializedName("input_images")
    val inputImages: List<ImageInfo>,
    @SerializedName("preview_images")
    val previewImages: List<ImageInfo>,
    @SerializedName("render_frames")
    val renderFrames: List<ImageInfo>,
    val logs: List<LogEntry>? = null
)

data class FileInfo(
    val obj: FileDetail,
    val stl: FileDetail,
    val video: FileDetail
)

data class FileDetail(
    val url: String,
    val size: Long,
    val exists: Boolean
)

data class ImageInfo(
    val index: Int,
    val url: String,
    val size: Long
)

data class RenderFramesResponse(
    val success: Boolean,
    @SerializedName("job_id")
    val jobId: String,
    @SerializedName("total_frames")
    val totalFrames: Int,
    val frames: List<ImageInfo>
)

data class DeleteResponse(
    val success: Boolean,
    val message: String
)

data class JobListResponse(
    val success: Boolean,
    val count: Int,
    val jobs: List<JobData>
)

data class LogsResponse(
    val success: Boolean,
    @SerializedName("job_id")
    val jobId: String,
    @SerializedName("log_count")
    val logCount: Int,
    val logs: List<LogEntry>
)
