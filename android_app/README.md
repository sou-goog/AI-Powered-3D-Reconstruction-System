# Image to 3D - Android App

Complete Android application in Kotlin Compose that integrates with the TripoSR REST API for converting images to 3D models.

## Package Structure

```
com.irspace.imageto3d/
├── MainActivity.kt                    # Entry point with Hilt and Compose setup
├── MyApplication.kt                   # Application class with @HiltAndroidApp
├── data/
│   ├── model/
│   │   └── ApiModels.kt              # Data models for all API responses
│   ├── api/
│   │   └── TripoSRApiService.kt      # Retrofit interface
│   └── repository/
│       └── TripoSRRepository.kt      # Repository with suspend functions
├── di/
│   └── NetworkModule.kt              # Hilt DI for network dependencies
└── ui/
    ├── navigation/
    │   └── Navigation.kt             # Navigation graph
    ├── theme/
    │   └── Theme.kt                  # Material3 theme
    └── screen/
        ├── upload/
        │   ├── UploadViewModel.kt    # Upload state management
        │   └── UploadScreen.kt       # Image picker and upload UI
        ├── gallery/
        │   ├── GalleryViewModel.kt   # Gallery state management
        │   └── GalleryScreen.kt      # Grid of 3D models
        └── result/
            ├── ResultViewModel.kt    # Result state management
            └── ResultScreen.kt       # Preview and download UI
```

## Features

### 1. Upload Screen
- Select 1-5 images from gallery
- Preview selected images in grid
- Upload to server with progress tracking
- Real-time progress updates (0-100%)
- Automatic navigation to result on completion

### 2. Gallery Screen
- Grid view of all generated 3D models
- Thumbnail previews
- Creation timestamp
- Delete functionality with confirmation
- Pagination support (load more)
- Click to view details

### 3. Result Screen
- Image preview carousel
- Download buttons for OBJ, STL, and MP4 files
- File size display
- Download progress tracking
- Open downloaded files with FileProvider
- Model information (creation time, processing time)

## Tech Stack

- **Language**: Kotlin
- **UI**: Jetpack Compose with Material3
- **Architecture**: MVVM with Repository pattern
- **DI**: Hilt (Dagger)
- **Network**: Retrofit 2.9.0 + OkHttp 4.12.0
- **Image Loading**: Coil 2.5.0
- **Async**: Kotlin Coroutines 1.7.3
- **Navigation**: Navigation Compose 2.7.5
- **Min SDK**: 24 (Android 7.0)
- **Target SDK**: 34 (Android 14)

## Dependencies

```kotlin
// Compose BOM
implementation(platform("androidx.compose:compose-bom:2023.10.01"))

// Hilt
implementation("com.google.dagger:hilt-android:2.48")
kapt("com.google.dagger:hilt-android-compiler:2.48")
implementation("androidx.hilt:hilt-navigation-compose:1.1.0")

// Retrofit
implementation("com.squareup.retrofit2:retrofit:2.9.0")
implementation("com.squareup.retrofit2:converter-gson:2.9.0")

// OkHttp
implementation("com.squareup.okhttp3:okhttp:4.12.0")
implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

// Coil
implementation("io.coil-kt:coil-compose:2.5.0")

// Navigation
implementation("androidx.navigation:navigation-compose:2.7.5")

// Coroutines
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
```

## Setup Instructions

### 1. Update Server URL

In `app/build.gradle.kts`, update the BASE_URL with your server's IP address:

```kotlin
buildConfigField("String", "BASE_URL", "\"http://YOUR_SERVER_IP:5002\"")
```

Replace `YOUR_SERVER_IP` with:
- Local testing: `10.0.2.2` (Android emulator) or `192.168.x.x` (actual device)
- Production: Your server's public IP or domain

### 2. Build the Project

```bash
cd android_app
./gradlew build
```

### 3. Run on Device/Emulator

```bash
./gradlew installDebug
```

Or open in Android Studio and click Run.

## API Integration

The app connects to the following endpoints:

- `GET /api/health` - Server health check
- `POST /api/upload` - Upload 1-5 images
- `GET /api/status/{jobId}` - Check job status
- `GET /api/gallery` - List all 3D models
- `GET /api/gallery/{jobId}` - Get specific model details
- `GET /api/download/{jobId}/mesh.obj` - Download OBJ file
- `GET /api/download/{jobId}/mesh.stl` - Download STL file
- `GET /api/download/{jobId}/render.mp4` - Download video
- `GET /api/preview/{jobId}` - Get preview images
- `DELETE /api/delete/{jobId}` - Delete a model

## Permissions

The app requires the following permissions:

- `INTERNET` - Network access for API calls
- `READ_MEDIA_IMAGES` - Access photos (Android 13+)
- `READ_EXTERNAL_STORAGE` - Access photos (Android 12 and below)

Permissions are requested at runtime when needed.

## File Provider

FileProvider is configured to share downloaded 3D model files:

- Authority: `com.irspace.imageto3d.fileprovider`
- Paths: External files directory, cache, and external cache
- Configuration: `res/xml/file_provider_paths.xml`

## State Management

### Upload Flow
1. User selects images → `UploadViewModel.selectedImages` updated
2. User clicks upload → `UploadViewModel.uploadImages()` called
3. Repository uploads files → Progress tracked via `_uploadState`
4. On success → Navigate to Result screen with jobId

### Gallery Flow
1. Screen loads → `GalleryViewModel.loadGallery()` called
2. Repository fetches gallery → Items displayed in LazyVerticalGrid
3. User clicks item → Navigate to Result screen
4. User deletes item → Confirmation dialog → API call → Refresh gallery

### Result Flow
1. Screen receives jobId → `ResultViewModel.loadResult()` called
2. Repository polls status every 2 seconds via Flow
3. When complete → Display previews and download buttons
4. User clicks download → `ResultViewModel.downloadFile()` called
5. File saved → User can open with external app

## Error Handling

All screens handle three states:
- **Loading**: Shows CircularProgressIndicator
- **Success**: Displays content
- **Error**: Shows error message with retry button

Network errors are caught in Repository and propagated as sealed Result types.

## Testing the App

### With Local Server

1. Start the API server:
```bash
cd /teamspace/studios/this_studio
python api.py
```

2. Get your machine's IP address:
```bash
ip addr show | grep inet
```

3. Update `BASE_URL` in build.gradle.kts

4. Ensure firewall allows port 5002

5. Run the app on a device connected to the same network

### With Emulator

Use `10.0.2.2:5002` as the BASE_URL (emulator's host loopback).

## Troubleshooting

### Cannot connect to server
- Check server is running: `curl http://localhost:5002/api/health`
- Verify BASE_URL is correct
- Check firewall settings
- Ensure device and server are on same network

### Upload fails
- Check image file size (max 10MB recommended)
- Verify internet permission in manifest
- Check server logs: `tail -f server.log`

### Download fails
- Ensure external storage permission granted
- Check available storage space
- Verify FileProvider configuration

### Build errors
- Clean project: `./gradlew clean`
- Invalidate caches in Android Studio
- Sync Gradle files
- Check Kotlin/Gradle plugin versions

## Next Steps

- Add video player for render.mp4 preview (ExoPlayer)
- Implement 3D model viewer (Sceneform/Filament)
- Add dark theme toggle
- Implement offline caching with Room
- Add animation between screens
- Support landscape orientation
- Add unit and UI tests

## License

Same as parent project.
