# Complete Deep Dive: 3D Reconstruction System

## Part 1: System Architecture & Core Technologies

---

### 1.1 What This System Does

This is an **AI-powered 3D reconstruction application** that converts a single 2D image into a complete 3D model. The system uses:
- Deep learning (neural networks)
- Computer vision
- 3D graphics rendering
- Web technologies

**Input:** A single 2D photo (e.g., chair.jpg)

**Output:**
- `mesh.obj` - 3D geometry file (OBJ format)
- `mesh.stl` - 3D printing format (STL format)
- `mesh.mtl` - Material definition file
- `mesh_texture.png` - Texture/color map
- `render.mp4` - 360° rotation video
- `render_000.png` to `render_029.png` - 30 individual frames
- Interactive WebGL viewer in browser

---

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  (Web Browser - HTML/CSS/JavaScript + Three.js)             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP Requests
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               FLASK WEB SERVER (app.py)                     │
│  • Routing: Maps URLs to functions                          │
│  • Session Management: Tracks users                         │
│  • File Handling: Upload/Download                           │
│  • Background Threading: Non-blocking processing            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Function Calls
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           IMAGE PREPROCESSING PIPELINE                      │
│  • Background Removal (rembg library)                       │
│  • Resize to 512×512 (PIL library)                          │
│  • RGBA → RGB conversion (NumPy)                            │
│  • Image normalization                                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Preprocessed Image
                        ▼
┌─────────────────────────────────────────────────────────────┐
│            TSR AI MODEL (tsr/system.py)                     │
│  • Neural Network: TripoSR (Transformer-based)              │
│  • Framework: PyTorch                                       │
│  • Input: 2D image tensor                                   │
│  • Output: 3D scene codes (latent representation)           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ 3D Scene Codes
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              3D RENDERING ENGINE                            │
│  • Render 30 views from different camera angles             │
│  • Use scene codes to generate images                       │
│  • Create rotation video                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Rendered Images
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              MESH EXTRACTION                                │
│  • Marching Cubes Algorithm (extracts surface)              │
│  • Generate vertices and faces                              │
│  • Extract vertex colors                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ 3D Mesh Data
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           EXPORT & TEXTURE BAKING (trimesh)                 │
│  • OBJ Export: Geometry + normals                           │
│  • STL Export: For 3D printing                              │
│  • Texture Baking: Vertex colors → UV texture map           │
│  • MTL Generation: Material definition                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Output Files
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   FILE STORAGE                              │
│  output/{timestamp}/                                        │
│    ├── input.png                                            │
│    ├── mesh.obj                                             │
│    ├── mesh.stl                                             │
│    ├── mesh.mtl                                             │
│    ├── mesh_texture.png                                     │
│    ├── render_000.png ... render_029.png                    │
│    └── render.mp4                                           │
└─────────────────────────────────────────────────────────────┘
```

---

### 1.3 Complete Technology Stack

#### **1.3.1 Backend (Python)**

| Library | Version | Purpose | Deep Explanation |
|---------|---------|---------|------------------|
| **Flask** | 3.0.0 | Web Framework | Lightweight WSGI web application framework. Handles HTTP requests/responses, routing, templating. WSGI = Web Server Gateway Interface (Python standard for web apps). |
| **PyTorch** | Latest | Deep Learning | Facebook's machine learning library. Provides tensor operations, automatic differentiation (autograd), GPU acceleration via CUDA. Used to run the TripoSR neural network. |
| **rembg** | Latest | Background Removal | Uses U²-Net neural network (trained on 15k images) to segment foreground/background. Returns RGBA image with transparent background. |
| **PIL (Pillow)** | Latest | Image Processing | Python Imaging Library. Loads/saves images, resize, format conversion (PNG/JPG/etc), color space operations (RGB/RGBA), image filtering. |
| **NumPy** | Latest | Array Operations | Numerical Python. Provides N-dimensional arrays, mathematical operations, broadcasting. Used for image arrays (H×W×C), matrix math, vectorized operations. |
| **trimesh** | 4.0.5+ | 3D Mesh Processing | Loads/saves 3D files (OBJ/STL/PLY), mesh operations (repair, simplify), ray tracing, texture baking via ColorVisuals.to_texture(). |
| **imageio-ffmpeg** | Latest | Video Encoding | Python wrapper for FFmpeg. Converts image sequences to MP4/AVI. Uses H.264 codec for web-compatible videos. |
| **flask-cors** | Latest | CORS Headers | Cross-Origin Resource Sharing. Allows requests from different domains (needed for Android app API calls). Adds `Access-Control-Allow-Origin` headers. |

#### **1.3.2 AI Model Architecture**

**TripoSR** (Stability AI):
- **Type:** Transformer-based 3D reconstruction
- **Input:** Single RGB image (512×512)
- **Output:** 3D triplane representation (3 feature planes)
- **Training Data:** Objaverse dataset (800k+ 3D models)
- **Model Size:** ~1.5 GB
- **Inference Time:** ~10-30 seconds on GPU, 60-120 seconds on CPU

**Components:**
1. **Image Tokenizer:** Converts image to tokens (like words in NLP)
2. **Transformer Backbone:** Processes tokens, learns 3D structure
3. **TriPlane Decoder:** Converts tokens to 3 orthogonal feature planes
4. **NeRF-like Renderer:** Queries triplanes to render novel views
5. **Marching Cubes:** Extracts mesh surface from density field

#### **1.3.3 Frontend (JavaScript)**

| Library | Version | Purpose | Deep Explanation |
|---------|---------|---------|------------------|
| **Three.js** | 0.155.0 | 3D Graphics | WebGL wrapper. Provides scene graph, cameras, lights, materials, geometries. Abstracts raw WebGL API. Uses GPU for real-time rendering. |
| **OBJLoader** | Three.js addon | OBJ Parser | Parses Wavefront OBJ file format. Reads vertices (`v`), normals (`vn`), texture coords (`vt`), faces (`f`). Creates Three.js geometry. |
| **MTLLoader** | Three.js addon | Material Parser | Parses MTL (Material Template Library). Reads ambient (Ka), diffuse (Kd), specular (Ks), texture maps (map_Kd). Creates Three.js materials. |
| **OrbitControls** | Three.js addon | Camera Control | Mouse/touch controls for 3D camera. Left-drag = rotate, right-drag = pan, scroll = zoom. Uses spherical coordinates. Auto-rotation support. |
| **Bootstrap** | 5.3.0 | UI Framework | CSS framework with grid system, components (buttons, cards, modals), responsive design, utility classes. |
| **Font Awesome** | 6.4.0 | Icon Library | Vector icons (SVG/font). Used for UI icons (download, rotate, wireframe, etc). Scalable and customizable. |

---

### 1.4 Data Flow Example (Step by Step)

Let's trace what happens when a user uploads `chair.png`:

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: User Action                                          │
├──────────────────────────────────────────────────────────────┤
│ • User visits http://localhost:5000/                         │
│ • Browser sends: GET /                                       │
│ • Flask serves: templates/index.html                         │
│ • User sees: Upload form with file input                     │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2: File Upload                                          │
├──────────────────────────────────────────────────────────────┤
│ • User selects chair.png (500KB, 1024×768 pixels)            │
│ • User clicks "Generate 3D Model"                            │
│ • Browser sends: POST / with multipart/form-data            │
│   Content: image=chair.png (binary data)                     │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 3: Flask Receives Request                               │
├──────────────────────────────────────────────────────────────┤
│ @app.route("/", methods=["POST"])                            │
│ def index():                                                 │
│     file = request.files["image"]  # Get uploaded file       │
│     session_id = str(int(time.time() * 1000))                │
│     # → "1732723401234"                                      │
│     file.save("uploads/chair.png")  # Save to disk           │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 4: Background Thread Starts                             │
├──────────────────────────────────────────────────────────────┤
│ thread = threading.Thread(                                   │
│     target=process_image_async,                              │
│     args=("uploads/chair.png", "1732723401234")              │
│ )                                                            │
│ thread.start()  # Non-blocking!                              │
│                                                              │
│ return redirect("/processing/1732723401234")                 │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 5: Browser Redirected                                   │
├──────────────────────────────────────────────────────────────┤
│ • Browser navigates to /processing/1732723401234             │
│ • Flask serves templates/processing.html                     │
│ • JavaScript opens Server-Sent Events connection:            │
│   EventSource("/progress/1732723401234")                     │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 6: Background Processing (Thread)                       │
├──────────────────────────────────────────────────────────────┤
│ def process_image_async(upload_path, session_id):            │
│     timer = Timer(session_id)                                │
│                                                              │
│     # Load image                                             │
│     img = Image.open("uploads/chair.png")  # PIL             │
│     # → <PIL.Image.Image RGB 1024×768>                       │
│                                                              │
│     # Resize to 512×512                                      │
│     img = img.resize((512, 512))                             │
│     # → <PIL.Image.Image RGB 512×512>                        │
│                                                              │
│     # Remove background                                      │
│     img = remove_background(img, rembg_session)              │
│     # → <PIL.Image.Image RGBA 512×512>                       │
│     # Alpha channel: 0=transparent, 255=opaque               │
│                                                              │
│     # Resize foreground (center object)                      │
│     img = resize_foreground(img, ratio=0.85)                 │
│     # → Shrinks object to 85% of canvas                      │
│                                                              │
│     # Convert RGBA → RGB                                     │
│     arr = np.array(img).astype(np.float32) / 255.0           │
│     # → NumPy array shape (512, 512, 4), values 0.0-1.0      │
│     rgb = arr[:,:,:3] * arr[:,:,3:4] + (1-arr[:,:,3:4])*0.5  │
│     # → Blend with gray background (0.5, 0.5, 0.5)           │
│     img = Image.fromarray((rgb*255).astype(np.uint8))        │
│     # → <PIL.Image.Image RGB 512×512>                        │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 7: AI Model Inference                                   │
├──────────────────────────────────────────────────────────────┤
│ with torch.no_grad():  # Disable gradients (inference mode)  │
│     scene_codes = model([img], device="cuda")                │
│                                                              │
│ # What happens inside model():                               │
│ # 1. Image → Tensor: [1, 512, 512, 3] (batch, H, W, C)      │
│ # 2. Image Tokenizer: Extracts features                      │
│ #    → [1, 256, 768] (batch, tokens, embedding_dim)          │
│ # 3. Transformer: Processes tokens                           │
│ #    → [1, 256, 768] (learned 3D representations)            │
│ # 4. TriPlane Decoder: Creates 3 feature planes              │
│ #    → [1, 3, 256, 256, 64] (batch, 3 planes, H, W, feats)  │
│ #                                                            │
│ # scene_codes = compact 3D representation (~50MB in memory)   │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 8: Render 30 Views                                      │
├──────────────────────────────────────────────────────────────┤
│ render_images = model.render(                                │
│     scene_codes,                                             │
│     n_views=30,                                              │
│     return_type="pil"                                        │
│ )                                                            │
│                                                              │
│ # What happens:                                              │
│ # - Generate 30 camera positions (circle around object)      │
│ # - For each camera:                                         │
│ #   1. Generate rays (one per pixel)                         │
│ #   2. Query triplane features along rays                    │
│ #   3. Predict density and color                             │
│ #   4. Volume rendering (like NeRF)                          │
│ #   5. Output: 256×256 RGB image                             │
│ #                                                            │
│ # Result: render_images = [[img0, img1, ..., img29]]         │
│ #         Each img = <PIL.Image.Image RGB 256×256>           │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 9: Create Video                                         │
├──────────────────────────────────────────────────────────────┤
│ save_video(                                                  │
│     render_images[0],  # List of 30 PIL images               │
│     "output/1732723401/render.mp4",                          │
│     fps=30                                                   │
│ )                                                            │
│                                                              │
│ # Uses imageio + ffmpeg:                                     │
│ # 1. Convert PIL images → NumPy arrays                       │
│ # 2. Stack into video tensor [30, 256, 256, 3]               │
│ # 3. Encode with H.264 codec                                 │
│ # 4. Write MP4 container                                     │
│ # Result: 1-2 MB MP4 file                                    │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 10: Extract 3D Mesh                                     │
├──────────────────────────────────────────────────────────────┤
│ meshes = model.extract_mesh(                                 │
│     scene_codes,                                             │
│     has_vertex_color=True,                                   │
│     resolution=256                                           │
│ )                                                            │
│                                                              │
│ # What happens:                                              │
│ # 1. Create 256³ 3D grid                                     │
│ # 2. Query density at each point                             │
│ # 3. Run Marching Cubes algorithm:                           │
│ #    - Find surface where density = threshold                │
│ #    - Generate triangles (vertices + faces)                 │
│ # 4. Query RGB color at each vertex                          │
│ #                                                            │
│ # Result: trimesh.Trimesh object                             │
│ #   vertices: np.array shape (N, 3) - 3D positions           │
│ #   faces: np.array shape (M, 3) - triangle indices          │
│ #   vertex_colors: np.array shape (N, 4) - RGBA              │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 11: Texture Baking                                      │
├──────────────────────────────────────────────────────────────┤
│ # mesh has vertex_colors (N×4 array)                         │
│ texture_visual = mesh.visual.to_texture()                    │
│                                                              │
│ # What happens:                                              │
│ # 1. Create UV coordinates (2D texture space)                │
│ #    - Unwrap 3D surface to 2D plane                         │
│ #    - Each vertex gets (u,v) coordinates                    │
│ # 2. Create texture image (e.g., 1024×1024)                  │
│ # 3. Rasterize vertex colors onto texture                    │
│ # 4. Interpolate between vertices                            │
│ #                                                            │
│ # Result: texture_visual.material.image (PIL Image)          │
│ #         texture_visual.uv (N×2 array of UV coords)         │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 12: Export Files                                        │
├──────────────────────────────────────────────────────────────┤
│ # OBJ file (text format)                                     │
│ obj_text, texture_data = obj_io.export_obj(                  │
│     mesh,                                                    │
│     include_normals=True,                                    │
│     include_texture=True,                                    │
│     return_texture=True                                      │
│ )                                                            │
│                                                              │
│ # obj_text contents:                                         │
│ # v 0.123 0.456 0.789   ← Vertex positions                   │
│ # vn 0.0 1.0 0.0        ← Vertex normals                     │
│ # vt 0.5 0.5            ← Texture coordinates                │
│ # f 1/1/1 2/2/2 3/3/3   ← Face (v/vt/vn indices)             │
│ # mtllib mesh.mtl       ← Material reference                 │
│                                                              │
│ # MTL file contents:                                         │
│ # newmtl mesh_texture                                        │
│ # Ka 1.0 1.0 1.0        ← Ambient color                      │
│ # Kd 1.0 1.0 1.0        ← Diffuse color                      │
│ # Ks 0.5 0.5 0.5        ← Specular color                     │
│ # map_Kd mesh_texture.png ← Texture file                     │
│                                                              │
│ # STL file (binary format)                                   │
│ mesh_trimesh.export("mesh.stl")                              │
│ # Binary STL: Header + triangle list                         │
│ # Each triangle: normal vector + 3 vertices (floats)         │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 13: Mark Complete                                       │
├──────────────────────────────────────────────────────────────┤
│ processing_status[session_id] = {                            │
│     'status': 'completed',                                   │
│     'folder_id': '1732723401',                               │
│     'message': '🎉 3D model generated!'                       │
│ }                                                            │
│                                                              │
│ # SSE stream sends final message                             │
│ # JavaScript detects completion                              │
│ # Redirects to: /result/1732723401                           │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 14: Display Result                                      │
├──────────────────────────────────────────────────────────────┤
│ • Flask serves templates/result.html                         │
│ • Three.js loads mesh.obj                                    │
│ • User can:                                                  │
│   - Rotate 3D model (OrbitControls)                          │
│   - Toggle texture on/off                                    │
│   - Download OBJ/STL/MTL/PNG files                           │
│   - View animation video                                     │
└──────────────────────────────────────────────────────────────┘
```

---

### 1.5 File Structure Deep Dive

```
/teamspace/studios/this_studio/
│
├── app.py                          # Main Flask application (400+ lines)
│   ├── Flask initialization
│   ├── Model loading (TSR)
│   ├── Routes (/, /processing, /result, /output, /gallery)
│   ├── Timer class (progress tracking)
│   └── process_image_async() (background processing)
│
├── api.py                          # REST API for Android app (800+ lines)
│   ├── CORS enabled
│   ├── Job queue system
│   ├── Multi-image support (1-5 images)
│   └── 12 API endpoints
│
├── tsr/                            # AI model implementation
│   ├── system.py                   # TSR main class
│   │   ├── from_pretrained()       # Model loading
│   │   ├── forward()               # Image → scene codes
│   │   ├── render()                # Scene codes → images
│   │   └── extract_mesh()          # Scene codes → 3D mesh
│   │
│   ├── utils.py                    # Helper functions
│   │   ├── remove_background()     # rembg wrapper
│   │   ├── resize_foreground()     # Image preprocessing
│   │   ├── save_video()            # imageio wrapper
│   │   └── get_spherical_cameras() # Camera positioning
│   │
│   ├── models/                     # Neural network layers
│   │   ├── isosurface.py          # Marching Cubes
│   │   ├── transformer.py         # Transformer blocks
│   │   ├── tokenizer.py           # Image tokenizer
│   │   └── renderer.py            # NeRF-style renderer
│   │
│   └── __init__.py
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                 # Upload form (200 lines)
│   ├── processing.html            # Progress viewer (300 lines)
│   ├── result.html                # 3D viewer (1000+ lines)
│   └── gallery.html               # Model gallery (400 lines)
│
├── uploads/                        # Temporary upload storage
│   └── [deleted after processing]
│
├── output/                         # Generated models
│   ├── 1732723401/                # Timestamp folder
│   │   ├── input.png              # Preprocessed input (512×512)
│   │   ├── mesh.obj               # 3D geometry (100KB-2MB)
│   │   ├── mesh.stl               # 3D printing format (500KB-10MB)
│   │   ├── mesh.mtl               # Material definition (1KB)
│   │   ├── mesh_texture.png       # Color texture (1-4MB)
│   │   ├── render_000.png         # Frame 0 (256×256, ~50KB)
│   │   ├── render_001.png         # Frame 1
│   │   ├── ...                    # Frames 2-28
│   │   ├── render_029.png         # Frame 29
│   │   └── render.mp4             # Animation (1-2MB, H.264)
│   │
│   └── 1732723402/                # Another job
│
├── requirements.txt                # Python dependencies
├── config.yaml                     # TSR model configuration
├── model.ckpt                      # TSR weights (~1.5GB)
└── README.md                       # Documentation
```

---

### 1.6 Key Concepts Explained

#### **Threading vs Multiprocessing**

**Why threading?**
- Lightweight (shares memory)
- Good for I/O-bound tasks (waiting for disk/network)
- Python GIL (Global Interpreter Lock) limits CPU parallelism

**In this project:**
- Web server (main thread) handles requests
- Background threads process images
- Progress queues communicate between threads

**Alternative:** Multiprocessing would use separate processes (more memory, but true parallelism)

#### **Server-Sent Events (SSE)**

**What is SSE?**
- One-way communication: Server → Client
- HTTP connection stays open
- Server pushes messages as they occur
- Simpler than WebSockets

**Format:**
```
data: {"message": "Processing...", "timestamp": "12:34:56"}\n\n
data: {"message": "Completed!", "status": "done"}\n\n
```

**JavaScript client:**
```javascript
const eventSource = new EventSource('/progress/123');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.message);
};
```

#### **Marching Cubes Algorithm**

**Purpose:** Extract triangle mesh from 3D density field

**How it works:**
1. Create 3D grid (e.g., 256×256×256 = 16M points)
2. Sample density at each point (from neural network)
3. For each cube (8 corner points):
   - Check which corners are inside surface (density > threshold)
   - Look up triangle configuration in table (256 cases)
   - Interpolate vertex positions
   - Add triangles to mesh
4. Result: Smooth surface mesh

**Why called "Marching"?**
- Algorithm processes cubes in order (marches through grid)

#### **GPU Acceleration (CUDA)**

**What is CUDA?**
- NVIDIA's parallel computing platform
- Runs code on GPU (thousands of cores)
- PyTorch automatically uses CUDA if available

**In this project:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # Move model to GPU
```

**Speed difference:**
- CPU: 60-120 seconds
- GPU (RTX 3090): 10-30 seconds
- ~4-8× faster

**Why GPUs are fast for AI:**
- Matrix multiplication (core of neural networks)
- Thousands of operations in parallel
- Optimized memory bandwidth

---

### 1.7 System Requirements

**Minimum:**
- Python 3.8+
- 8 GB RAM
- 10 GB disk space
- CPU: 4 cores

**Recommended:**
- Python 3.10+
- 16 GB RAM
- 20 GB disk space
- NVIDIA GPU with 8+ GB VRAM
- CUDA 11.8+

**Browser:**
- Modern browser with WebGL 2.0 support
- Chrome 90+, Firefox 88+, Safari 15+

---

### 1.8 Performance Metrics

| Operation | Time (GPU) | Time (CPU) | Memory |
|-----------|-----------|-----------|--------|
| Model Loading | 5-10 sec | 5-10 sec | 1.5 GB |
| Background Removal | 1-2 sec | 3-5 sec | 500 MB |
| Scene Code Generation | 2-5 sec | 20-40 sec | 2 GB |
| Rendering 30 Views | 3-8 sec | 30-60 sec | 1 GB |
| Mesh Extraction | 2-4 sec | 5-10 sec | 500 MB |
| Texture Baking | 1-2 sec | 1-2 sec | 200 MB |
| **Total** | **10-30 sec** | **60-120 sec** | **4-5 GB peak** |

---

**End of Part 1**

This covers the high-level architecture, all technologies used, data flow, and key concepts. Each library and algorithm has been explained in detail.

---

## Part 2: Flask Application Deep Dive

---

### 2.1 Flask Framework Fundamentals

#### **What is Flask?**

Flask is a **micro web framework** written in Python. "Micro" means:
- Minimal core functionality
- Easy to extend with plugins
- No built-in database ORM (unlike Django)
- No form validation library (unlike Django)
- Lightweight and flexible

**Flask Components:**
1. **Werkzeug** - WSGI utility library (routing, request/response handling)
2. **Jinja2** - Template engine (HTML with Python expressions)
3. **Click** - Command-line interface creation
4. **ItsDangerous** - Cryptographic signing (for sessions)

**WSGI (Web Server Gateway Interface):**
- Python standard for web applications
- Interface between web server (nginx/Apache) and web app (Flask)
- Allows any WSGI server to run any WSGI application

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Browser   │─────>│ Web Server  │─────>│   Flask     │
│             │      │ (nginx/     │      │   App       │
│             │<─────│  gunicorn)  │<─────│             │
└─────────────┘      └─────────────┘      └─────────────┘
                     WSGI Interface
```

---

### 2.2 Application Initialization

Let's examine `app.py` line by line:

```python
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, Response, session
```

**Import breakdown:**

| Import | Purpose | Example Usage |
|--------|---------|---------------|
| `Flask` | Main application class | `app = Flask(__name__)` |
| `request` | Access request data (form, files, headers) | `request.files["image"]` |
| `render_template` | Render Jinja2 HTML templates | `render_template("index.html")` |
| `send_from_directory` | Serve static files securely | `send_from_directory("output", "mesh.obj")` |
| `redirect` | HTTP 302 redirect | `redirect("/result/123")` |
| `url_for` | Generate URLs from function names | `url_for("processing", session_id="123")` |
| `Response` | Custom HTTP response | `Response(generate(), mimetype="text/event-stream")` |
| `session` | Encrypted cookie storage | `session["user_id"] = 123` |

```python
import os
import torch
from PIL import Image, ImageOps
import numpy as np
import time
import rembg
import json
import threading
from queue import Queue
```

**Library purposes:**

| Library | Purpose in This App |
|---------|---------------------|
| `os` | File paths, directory operations (`os.path.join`, `os.makedirs`) |
| `torch` | PyTorch for neural network inference |
| `PIL` | Image loading, resizing, format conversion |
| `numpy` | Array operations for image preprocessing |
| `time` | Timestamps, timing operations |
| `rembg` | AI-powered background removal |
| `json` | JSON serialization for SSE messages |
| `threading` | Background task execution |
| `Queue` | Thread-safe message passing |

```python
from trimesh.exchange import obj as obj_io
import trimesh
```

**Trimesh imports:**
- `obj_io` - OBJ file export with texture data
- `trimesh` - Mesh loading, STL conversion

```python
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
```

**TSR model imports:**
- `TSR` - Main model class
- `remove_background()` - Wrapper for rembg
- `resize_foreground()` - Center and scale object
- `save_video()` - Create MP4 from image sequence

---

### 2.3 Flask Application Object

```python
app = Flask(__name__)
```

**What happens here:**

1. Creates Flask application instance
2. `__name__` tells Flask where to find resources:
   - If run directly: `__name__ = "__main__"`
   - If imported: `__name__ = "app"`
3. Flask uses this to locate templates and static files

**Application structure Flask expects:**
```
app.py
├── templates/       # HTML templates (auto-discovered)
├── static/          # CSS, JS, images (auto-discovered)
└── [other files]
```

```python
app.secret_key = 'triposr-secret-key-2024'
```

**Secret key purpose:**
- Encrypts session cookies
- Uses HMAC-SHA256 algorithm
- Cookie format: `{user_data}.{timestamp}.{signature}`
- Prevents cookie tampering

**Example:**
```python
# Server sets session
session['user_id'] = 123

# Flask creates cookie:
# eyJ1c2VyX2lkIjoxMjN9.Zx-3lQ.p9h4_8Kl2mN5vB7cD9fE6gH3jK8

# Parts:
# 1. eyJ1c2VyX2lkIjoxMjN9  (base64 of {"user_id":123})
# 2. Zx-3lQ                 (timestamp)
# 3. p9h4_8Kl2mN5vB7cD9fE6gH3jK8 (signature using secret_key)
```

**Why session cookies?**
- No server-side storage needed
- Scales horizontally (any server can read)
- User can't modify without secret_key

```python
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['OUTPUT_FOLDER'] = "output"
```

**Flask config system:**
- `app.config` is a dictionary
- Used throughout application
- Can load from files: `app.config.from_pyfile('config.py')`

**Accessing config:**
```python
upload_dir = app.config['UPLOAD_FOLDER']  # "uploads"
```

```python
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
```

**Directory creation:**
- `exist_ok=True` - Don't error if already exists
- Creates parent directories if needed
- Runs once at startup

---

### 2.4 Global State Management

```python
# Global progress tracking
progress_queues = {}
processing_status = {}
```

**Why global variables?**

In Flask:
- Each request runs in the same process
- Global variables persist between requests
- All threads share memory

**Structure:**
```python
progress_queues = {
    "1732723401234": Queue([msg1, msg2, msg3]),
    "1732723405678": Queue([msg4, msg5])
}

processing_status = {
    "1732723401234": {
        "status": "completed",
        "folder_id": "1732723401",
        "message": "Success!"
    },
    "1732723405678": {
        "status": "processing"
    }
}
```

**Thread safety:**
- `Queue` is thread-safe (built-in locking)
- Dictionary operations (read/write) are atomic in CPython
- No explicit locks needed for simple dict access

**Memory cleanup:**
```python
# After job completes, delete entries
del progress_queues[session_id]
del processing_status[session_id]
```

---

### 2.5 Model Loading (Startup)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Device selection:**

```python
# Check CUDA availability
import torch
torch.cuda.is_available()  # True if GPU + CUDA drivers

# Get device info
torch.cuda.get_device_name(0)  # "NVIDIA GeForce RTX 3090"
torch.cuda.get_device_properties(0).total_memory  # 25769803776 (24GB)
```

**Why check at runtime?**
- Code runs on both CPU and GPU machines
- Automatic fallback if GPU unavailable
- Development on CPU, production on GPU

```python
print("Initializing TSR model...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
```

**Model loading process:**

1. **Check local cache:**
   ```python
   cache_dir = "~/.cache/huggingface/hub/"
   # If files exist, load from cache
   ```

2. **Download if needed:**
   ```python
   # Downloads from HuggingFace Hub
   # https://huggingface.co/stabilityai/TripoSR
   # Files:
   # - config.yaml (5 KB)
   # - model.ckpt (1.5 GB)
   ```

3. **Load configuration:**
   ```python
   cfg = OmegaConf.load("config.yaml")
   # Contains:
   # - Model architecture parameters
   # - Layer dimensions
   # - Hyperparameters
   ```

4. **Initialize model:**
   ```python
   model = TSR(cfg)  # Creates neural network layers
   ```

5. **Load weights:**
   ```python
   ckpt = torch.load("model.ckpt")  # Dict of tensors
   model.load_state_dict(ckpt)      # Assigns weights to layers
   ```

**What's in model.ckpt?**
```python
{
    "image_tokenizer.conv1.weight": Tensor(shape=[64, 3, 7, 7]),
    "image_tokenizer.conv1.bias": Tensor(shape=[64]),
    "backbone.transformer.layer0.attention.weight": Tensor(shape=[768, 768]),
    # ... thousands of parameters
}
```

```python
model.renderer.set_chunk_size(8192)
```

**Chunk size explanation:**

**Problem:** Rendering queries millions of points simultaneously
- 256×256 image = 65,536 pixels
- Each pixel traces multiple samples along ray
- Total queries: 65,536 × 128 = 8,388,608 points

**Solution:** Process in chunks
```python
# Without chunking (out of memory):
results = model.render(all_8_million_points)  # ❌ CUDA OOM

# With chunking:
results = []
for chunk in chunks(all_points, size=8192):
    result = model.render(chunk)  # ✅ Fits in memory
    results.append(result)
final_result = concat(results)
```

**Chunk size tradeoff:**
- Larger chunks = faster (better GPU utilization)
- Smaller chunks = more memory efficient
- 8192 is balanced for 8-16 GB VRAM

```python
model.to(device)
```

**Moving model to GPU:**

```python
# Before: model parameters on CPU
model.parameters()  # Tensors with device='cpu'

# After: model parameters on GPU
model.to("cuda")
model.parameters()  # Tensors with device='cuda:0'
```

**Memory transfer:**
- Copies all 1.5 GB of weights to GPU VRAM
- GPU operations are 10-100× faster than CPU
- Model stays on GPU for all inference

**Verification:**
```python
next(model.parameters()).device  # device(type='cuda', index=0)
```

---

### 2.6 Flask Routing System

#### **How Routing Works**

```python
@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello!"
```

**Decorator breakdown:**

1. **`@app.route("/")`** - Registers function with Flask
   ```python
   # Internally, Flask does:
   app.add_url_rule(
       rule="/",
       endpoint="index",  # Function name
       view_func=index,
       methods=["GET", "POST"]
   )
   ```

2. **When request arrives:**
   ```python
   # Flask matches URL against registered routes
   if request.path == "/" and request.method in ["GET", "POST"]:
       response = index()
       return response
   ```

**URL patterns:**

| Pattern | Example | Matches |
|---------|---------|---------|
| `"/"` | Root | `http://localhost:5000/` |
| `"/about"` | Static | `http://localhost:5000/about` |
| `"/user/<id>"` | Dynamic | `http://localhost:5000/user/123` |
| `"/post/<int:id>"` | Typed | `http://localhost:5000/post/42` |
| `"/file/<path:filename>"` | Path | `http://localhost:5000/file/a/b/c.txt` |

**Route parameters:**
```python
@app.route("/result/<folder>")
def result(folder):
    # folder = "1732723401"
    return f"Showing result for {folder}"
```

---

### 2.7 Request Handler: Upload (GET)

```python
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ... handle upload
    return render_template("index.html")
```

**GET request flow:**

```
Browser: GET /
    ↓
Flask: Check method
    ↓
method == "GET"  # True
    ↓
Skip POST block
    ↓
render_template("index.html")
    ↓
Read: templates/index.html
    ↓
Render Jinja2 template
    ↓
Return HTML to browser
```

**`render_template()` internals:**

1. **Locate template:**
   ```python
   template_path = "templates/index.html"
   ```

2. **Load template:**
   ```python
   with open(template_path) as f:
       template_source = f.read()
   ```

3. **Parse Jinja2:**
   ```html
   <!-- Template with Jinja2 syntax -->
   <h1>{{ title }}</h1>
   {% if user %}
       <p>Welcome {{ user.name }}!</p>
   {% endif %}
   ```

4. **Render with context:**
   ```python
   context = {"title": "Upload", "user": current_user}
   html = jinja2_env.render(template_source, **context)
   ```

5. **Return response:**
   ```python
   return Response(html, mimetype="text/html", status=200)
   ```

---

### 2.8 Request Handler: Upload (POST)

```python
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"
```

**Understanding `request.files`:**

**HTML form:**
```html
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="image">
    <button type="submit">Upload</button>
</form>
```

**HTTP request:**
```http
POST / HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary
Content-Length: 524288

------WebKitFormBoundary
Content-Disposition: form-data; name="image"; filename="chair.png"
Content-Type: image/png

[binary image data...]
------WebKitFormBoundary--
```

**Flask parsing:**
```python
request.files = {
    "image": FileStorage(
        stream=<binary data>,
        filename="chair.png",
        content_type="image/png"
    )
}
```

**FileStorage object:**
```python
file = request.files["image"]

file.filename          # "chair.png"
file.content_type      # "image/png"
file.content_length    # 524288 (bytes)
file.stream            # BytesIO object
file.save(path)        # Save to disk
file.read()            # Read all bytes
```

**Validation:**
```python
if "image" not in request.files:
    # User didn't select any file
    return "No file part", 400

file = request.files["image"]

if file.filename == "":
    # User selected but then cancelled
    return "No selected file", 400
```

---

### 2.9 Session ID Generation

```python
session_id = str(int(time.time() * 1000))
```

**Timestamp-based ID:**

```python
import time

time.time()            # 1732723401.234567 (float, seconds)
time.time() * 1000     # 1732723401234.567 (milliseconds)
int(time.time() * 1000) # 1732723401234 (integer)
str(...)               # "1732723401234" (string)
```

**Why milliseconds?**
- Unique even if multiple uploads per second
- Human-readable (sortable by time)
- 13 digits (fits in JavaScript number)

**Alternative approaches:**
```python
# UUID (universally unique)
import uuid
session_id = str(uuid.uuid4())  # "550e8400-e29b-41d4-a716-446655440000"

# Random string
import secrets
session_id = secrets.token_urlsafe(16)  # "dGhpcyBpcyBhIHRlc3Q"

# Incremental
session_id = str(next_id)  # "1", "2", "3", ...
```

**Timestamp pros/cons:**
- ✅ Chronological ordering
- ✅ Human-readable (can decode time)
- ✅ No database needed
- ❌ Not truly random (predictable)
- ❌ Collision risk (same millisecond)

```python
session['current_session'] = session_id
```

**Session storage:**

1. **Set session:**
   ```python
   session['current_session'] = "1732723401234"
   ```

2. **Flask serializes:**
   ```python
   data = {"current_session": "1732723401234"}
   json_data = json.dumps(data)  # '{"current_session":"1732723401234"}'
   ```

3. **Flask signs:**
   ```python
   signature = hmac_sha256(json_data, secret_key)
   cookie_value = f"{base64(json_data)}.{timestamp}.{signature}"
   ```

4. **Set cookie:**
   ```http
   Set-Cookie: session=eyJjdXJyZW50X3Nlc3Npb24iOiIxNzMyNzIzNDAxMjM0In0.Zx-3lQ.p9h4; HttpOnly; Path=/
   ```

**Cookie flags:**
- `HttpOnly` - JavaScript can't access (prevents XSS)
- `Path=/` - Valid for all paths
- `Secure` - HTTPS only (in production)
- `SameSite=Lax` - CSRF protection

**Reading session later:**
```python
# In another route
@app.route("/processing/<session_id>")
def processing(session_id):
    stored_id = session.get('current_session')
    if stored_id == session_id:
        # Valid request
        pass
```

---

### 2.10 Progress Tracking Setup

```python
progress_queues[session_id] = Queue(maxsize=100)
processing_status[session_id] = {'status': 'starting'}
```

**Queue mechanics:**

**What is a Queue?**
```python
from queue import Queue

q = Queue(maxsize=100)  # Max 100 items

# Thread 1 (producer)
q.put("message 1")
q.put("message 2")

# Thread 2 (consumer)
msg = q.get()  # "message 1" (blocks if empty)
msg = q.get()  # "message 2"
```

**Thread safety:**
- Built-in locks prevent race conditions
- `put()` blocks if full
- `get()` blocks if empty
- No data corruption

**In this app:**
```python
# Main thread (web request)
progress_queues[session_id] = Queue(maxsize=100)

# Background thread (processing)
timer = Timer(session_id)
timer.log_progress("Processing...")
# Internally: progress_queues[session_id].put({"message": "Processing..."})

# Main thread (SSE endpoint)
while True:
    progress = queue.get()  # Blocks until message available
    yield f"data: {json.dumps(progress)}\n\n"
```

**Why maxsize=100?**
- Limits memory usage
- 100 messages ≈ 10 KB
- If full, `put()` blocks (backpressure)

---

### 2.11 Initial Progress Messages

```python
initial_timer = Timer(session_id)
initial_timer.log_progress("🌟 Welcome to 3D Reconstruction Studio!")
initial_timer.log_progress(f"📁 File uploaded: {filename}")
initial_timer.log_progress("🔄 Starting background processing...")
```

**Timer class (simplified):**
```python
class Timer:
    def __init__(self, session_id=None):
        self.session_id = session_id
        
    def log_progress(self, message, step=None, total_steps=None):
        if self.session_id and self.session_id in progress_queues:
            timestamp = time.strftime("%H:%M:%S")
            progress_data = {
                'message': message,
                'timestamp': timestamp,
                'step': step,
                'total_steps': total_steps
            }
            try:
                progress_queues[self.session_id].put(progress_data)
            except:
                pass  # Queue might be full or closed
```

**Message format:**
```python
{
    'message': '🌟 Welcome to 3D Reconstruction Studio!',
    'timestamp': '14:32:15',
    'step': None,
    'total_steps': None
}
```

**Why immediate messages?**
- User sees feedback instantly
- Confirms upload succeeded
- Better UX (no blank screen)

---

### 2.12 File Saving

```python
upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
file.save(upload_path)
```

**Path construction:**
```python
os.path.join("uploads", "chair.png")
# Returns: "uploads/chair.png" (Unix/Linux)
# Returns: "uploads\chair.png" (Windows)
```

**Why `os.path.join()`?**
- Cross-platform compatibility
- Handles separators correctly
- Avoids hardcoded slashes

**File saving:**
```python
file.save(upload_path)

# Internally:
with open(upload_path, 'wb') as f:
    chunk_size = 4096
    while True:
        chunk = file.stream.read(chunk_size)
        if not chunk:
            break
        f.write(chunk)
```

**Security concern:**
```python
from werkzeug.utils import secure_filename

filename = secure_filename(file.filename)
# "../../etc/passwd" → "etc_passwd"
# "dangerous <script>.png" → "dangerous_script.png"
```

**This app doesn't use `secure_filename` because:**
- Files deleted after processing
- Not served directly to users
- Background thread reads immediately

---

### 2.13 Background Thread Creation

```python
thread = threading.Thread(
    target=process_image_async,
    args=(upload_path, session_id)
)
thread.daemon = True
thread.start()
```

**Threading module:**

**`threading.Thread()` parameters:**
| Parameter | Purpose | Example |
|-----------|---------|---------|
| `target` | Function to run | `process_image_async` |
| `args` | Positional arguments tuple | `("uploads/chair.png", "123")` |
| `kwargs` | Keyword arguments dict | `{"debug": True}` |
| `name` | Thread name for debugging | `"ImageProcessor-123"` |
| `daemon` | Die with main program | `True` |

**Daemon threads:**
```python
# daemon=True
# If main program exits, daemon threads are killed
# Good for background tasks

# daemon=False (default)
# Main program waits for thread to finish
# Good for critical tasks
```

**Thread lifecycle:**
```python
thread = Thread(target=func, args=(arg1,))
# State: Not started

thread.start()
# State: Running (in background)
# func(arg1) executing

thread.is_alive()  # True

# When func() completes
thread.is_alive()  # False
# State: Finished
```

**Why not use Flask's request context?**
```python
# Background thread doesn't have access to:
request.files  # ❌ Not available
session        # ❌ Not available
g              # ❌ Not available

# Must pass data explicitly:
thread = Thread(target=process, args=(upload_path, session_id))
```

---

### 2.14 Redirection

```python
return redirect(url_for('processing', session_id=session_id))
```

**`url_for()` mechanics:**

```python
url_for('processing', session_id='123')
# 1. Find route decorated with def processing(...)
# 2. Get route pattern: "/processing/<session_id>"
# 3. Fill in parameters: "/processing/123"
# 4. Return: "/processing/123"
```

**With multiple parameters:**
```python
@app.route("/user/<int:user_id>/post/<int:post_id>")
def show_post(user_id, post_id):
    pass

url_for('show_post', user_id=5, post_id=10)
# Returns: "/user/5/post/10"
```

**Query parameters:**
```python
url_for('search', q='flask', page=2)
# Returns: "/search?q=flask&page=2"
```

**`redirect()` creates HTTP response:**
```python
return redirect("/processing/123")

# HTTP Response:
# HTTP/1.1 302 Found
# Location: /processing/123
# Content-Length: 0
```

**Browser behavior:**
1. Receives 302 response
2. Reads `Location` header
3. Automatically navigates to new URL
4. Sends `GET /processing/123`

**Alternative status codes:**
```python
redirect("/login", code=301)  # Permanent redirect
redirect("/login", code=302)  # Temporary redirect (default)
redirect("/login", code=303)  # See Other
redirect("/login", code=307)  # Temporary (preserves method)
```

---

### 2.15 Complete Upload Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Browser                                             │
│    GET http://localhost:5000/                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Flask Main Thread                                        │
│    @app.route("/", methods=["GET"])                         │
│    return render_template("index.html")                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ (HTML page with upload form)
┌─────────────────────────────────────────────────────────────┐
│ 3. User Browser                                             │
│    User selects chair.png                                   │
│    POST http://localhost:5000/                              │
│    Content-Type: multipart/form-data                        │
│    Body: [binary image data]                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Flask Main Thread                                        │
│    @app.route("/", methods=["POST"])                        │
│                                                             │
│    file = request.files["image"]     ✓ Validate            │
│    session_id = "1732723401234"      ✓ Generate ID         │
│    file.save("uploads/chair.png")    ✓ Save file           │
│                                                             │
│    progress_queues[id] = Queue()     ✓ Setup queue         │
│    processing_status[id] = {...}     ✓ Setup status        │
│                                                             │
│    thread = Thread(                  ✓ Create thread       │
│        target=process_image_async,                          │
│        args=("uploads/chair.png", "1732723401234")          │
│    )                                                        │
│    thread.start()                    ✓ Start background    │
│                                                             │
│    return redirect("/processing/1732723401234")             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ (HTTP 302 redirect)
┌─────────────────────────────────────────────────────────────┐
│ 5. User Browser                                             │
│    GET http://localhost:5000/processing/1732723401234       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Flask Main Thread                                        │
│    @app.route("/processing/<session_id>")                   │
│    return render_template("processing.html",                │
│                          session_id=session_id)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ (HTML page with JavaScript)
┌─────────────────────────────────────────────────────────────┐
│ 7. User Browser (JavaScript)                                │
│    const eventSource = new EventSource(                     │
│        "/progress/1732723401234"                            │
│    );                                                       │
│    eventSource.onmessage = (event) => {                     │
│        const data = JSON.parse(event.data);                 │
│        displayProgress(data.message);                       │
│    };                                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Flask Main Thread (SSE)                                  │
│    @app.route("/progress/<session_id>")                     │
│    Connection stays open, streaming messages...             │
└─────────────────────────────────────────────────────────────┘

                    (Meanwhile, in parallel)

┌─────────────────────────────────────────────────────────────┐
│ 9. Background Thread                                        │
│    process_image_async(                                     │
│        "uploads/chair.png",                                 │
│        "1732723401234"                                      │
│    )                                                        │
│                                                             │
│    ├─ Load image                                            │
│    ├─ Remove background                                     │
│    ├─ Run AI model                                          │
│    ├─ Render 30 views                                       │
│    ├─ Create video                                          │
│    ├─ Extract mesh                                          │
│    ├─ Bake texture                                          │
│    └─ Export files                                          │
│                                                             │
│    Each step:                                               │
│    timer.log_progress("Step X...")                          │
│    → progress_queues[session_id].put(message)               │
│    → SSE stream sends to browser                            │
│    → User sees real-time updates                            │
└─────────────────────────────────────────────────────────────┘
```

---

**End of Part 2**

This covers Flask fundamentals, request handling, session management, threading, and the complete upload flow with detailed explanations of every library and concept.

---

## Part 3: Background Processing & AI Pipeline

---

### 3.1 Background Processing Function Overview

```python
def process_image_async(upload_path, session_id):
    """Process image in background thread with progress updates"""
    timer = Timer(session_id)
    
    try:
        # 1. Image preprocessing
        # 2. AI model inference
        # 3. 3D rendering
        # 4. Mesh extraction
        # 5. Texture baking
        # 6. File export
    except Exception as e:
        # Error handling
```

**Function signature:**
- `upload_path` - Path to uploaded image (e.g., "uploads/chair.png")
- `session_id` - Unique identifier (e.g., "1732723401234")
- No return value (updates global state)

**Execution context:**
- Runs in separate thread
- No access to Flask request context
- Uses Queue for progress updates
- Updates `processing_status` dict on completion

---

### 3.2 Timer Class Deep Dive

```python
class Timer:
    def __init__(self, session_id=None):
        self.items = {}
        self.time_scale = 1000.0
        self.time_unit = "ms"
        self.session_id = session_id
```

**Purpose:** Track operation timing and send progress updates

**Instance variables:**
| Variable | Type | Purpose |
|----------|------|---------|
| `items` | `dict` | Stores start times: `{"operation": timestamp}` |
| `time_scale` | `float` | Multiply by 1000 to convert seconds→milliseconds |
| `time_unit` | `str` | Display unit ("ms") |
| `session_id` | `str` | Links to progress queue |

**Progress logging:**
```python
def log_progress(self, message, step=None, total_steps=None):
    """Log progress message to the session queue"""
    if self.session_id and self.session_id in progress_queues:
        timestamp = time.strftime("%H:%M:%S")
        progress_data = {
            'message': message,
            'timestamp': timestamp,
            'step': step,
            'total_steps': total_steps
        }
        try:
            progress_queues[self.session_id].put(progress_data)
        except:
            pass  # Queue might be full or closed
```

**What happens:**
1. Check if session has progress queue
2. Create timestamp (e.g., "14:32:15")
3. Build progress message dict
4. Put into thread-safe queue
5. SSE endpoint reads and sends to browser

**Timing operations:**
```python
def start(self, name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for GPU operations
    self.items[name] = time.time()
    self.log_progress(f"🚀 Starting {name}...")
```

**Why `torch.cuda.synchronize()`?**
- GPU operations are asynchronous
- `model()` call returns before GPU finishes
- `synchronize()` waits for all GPU operations
- Ensures accurate timing

**Example:**
```python
timer = Timer("123")

timer.start("Processing")
# items = {"Processing": 1732723401.234}
# Sends: "🚀 Starting Processing..."

# ... do work ...

timer.end("Processing")
# Calculates: time.time() - 1732723401.234 = 5.678 seconds
# Converts: 5.678 * 1000 = 5678 ms
# Sends: "✅ Processing completed in 5678.00ms"
```

---

### 3.3 Image Loading & Resizing

```python
timer.log_progress("📁 Processing uploaded image...")
timer.log_progress("🖼️ Loading and resizing image...")
original_image = Image.open(upload_path)
resized_image = original_image.resize((512, 512))
```

**PIL Image.open():**
```python
from PIL import Image

img = Image.open("chair.png")
# Returns: <PIL.Image.Image>
# Attributes:
#   size: (1024, 768)
#   mode: "RGB" (or "RGBA", "L", etc.)
#   format: "PNG"
```

**Image modes:**
| Mode | Description | Channels | Bytes per pixel |
|------|-------------|----------|-----------------|
| `"L"` | Grayscale | 1 | 1 |
| `"RGB"` | True color | 3 (R,G,B) | 3 |
| `"RGBA"` | RGB + Alpha | 4 (R,G,B,A) | 4 |
| `"P"` | Palette | 1 (index) | 1 |
| `"CMYK"` | Print colors | 4 | 4 |

**Resize operation:**
```python
resized = img.resize((512, 512))
```

**What happens:**
1. **Check if resize needed:**
   ```python
   if img.size == (512, 512):
       return img  # Already correct size
   ```

2. **Choose resampling filter:**
   ```python
   # Default: Image.LANCZOS (high quality)
   # Options:
   # - Image.NEAREST (fastest, blocky)
   # - Image.BILINEAR (fast, smooth)
   # - Image.BICUBIC (good quality)
   # - Image.LANCZOS (best quality, slower)
   ```

3. **Resample pixels:**
   ```python
   # For each pixel in output (512×512):
   #   1. Map to source coordinates
   #   2. Apply filter kernel
   #   3. Interpolate color value
   ```

4. **Return new image:**
   ```python
   # Original: 1024×768 = 786,432 pixels
   # Resized: 512×512 = 262,144 pixels
   # Memory: ~3MB → ~768KB (RGB)
   ```

**Why 512×512?**
- TSR model trained on 512×512 images
- Larger = more GPU memory, slower inference
- Smaller = loss of detail
- Square = no distortion in model

**Aspect ratio handling:**
```python
# Original: 1024×768 (4:3 ratio)
# Resized: 512×512 (1:1 ratio)
# Result: Image is stretched!

# Better approach (preserve aspect ratio):
from PIL import ImageOps
img = ImageOps.fit(img, (512, 512), Image.LANCZOS)
# Crops to square, then resizes
```

**This app stretches because:**
- Background will be removed anyway
- Object is centered by resize_foreground()
- Final result is re-centered

---

### 3.4 Background Removal (rembg)

```python
timer.start("Processing image")
timer.log_progress("🎭 Removing background...")
rembg_session = rembg.new_session()
image = remove_background(resized_image, rembg_session)
timer.log_progress("✨ Background removed successfully")
```

**rembg library:**
- Uses U²-Net neural network
- Trained on 15,000 labeled images
- 100+ MB model weights
- CPU or GPU inference

**U²-Net architecture:**
```
Input Image (512×512×3)
    ↓
Encoder (Downsampling)
├─ Conv → ReLU → MaxPool  (256×256×64)
├─ Conv → ReLU → MaxPool  (128×128×128)
├─ Conv → ReLU → MaxPool  (64×64×256)
└─ Conv → ReLU → MaxPool  (32×32×512)
    ↓
Bottleneck (32×32×512)
    ↓
Decoder (Upsampling)
├─ UpConv → Concat → Conv (64×64×256)
├─ UpConv → Concat → Conv (128×128×128)
├─ UpConv → Concat → Conv (256×256×64)
└─ UpConv → Concat → Conv (512×512×1)
    ↓
Output Mask (512×512) - Binary (0=background, 255=foreground)
```

**Session creation:**
```python
rembg_session = rembg.new_session()
# Downloads model if not cached
# ~/.u2net/u2net.onnx (176 MB)
# Loads into memory
# Creates ONNX Runtime session
```

**ONNX (Open Neural Network Exchange):**
- Standard format for neural networks
- Cross-framework (PyTorch→ONNX→TensorFlow)
- Optimized inference (faster than PyTorch)
- ONNX Runtime = high-performance inference engine

**Background removal process:**
```python
def remove_background(image, rembg_session):
    # 1. Convert PIL → NumPy array
    img_array = np.array(image)  # (512, 512, 3)
    
    # 2. Normalize to 0-1
    img_normalized = img_array / 255.0
    
    # 3. Run U²-Net inference
    mask = rembg_session.run(img_normalized)
    # mask shape: (512, 512) with values 0-255
    
    # 4. Apply mask to original
    rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_array  # RGB channels
    rgba[:, :, 3] = mask        # Alpha channel
    
    # 5. Convert back to PIL
    return Image.fromarray(rgba, mode='RGBA')
```

**Alpha channel:**
- 0 = fully transparent (background)
- 255 = fully opaque (foreground)
- 1-254 = semi-transparent (edge pixels)

**Example pixels:**
```python
# Before (RGB):
pixel = [120, 150, 200]  # Blue color

# After (RGBA):
pixel = [120, 150, 200, 255]  # Blue, opaque (foreground)
pixel = [120, 150, 200, 0]    # Blue, transparent (background)
pixel = [120, 150, 200, 128]  # Blue, semi-transparent (edge)
```

---

### 3.5 Foreground Resizing

```python
timer.log_progress("🔄 Resizing foreground...")
image = resize_foreground(image, ratio=0.85)
```

**Purpose:** Center and scale object to 85% of canvas

**Algorithm:**
```python
def resize_foreground(image, ratio=0.85):
    # 1. Find bounding box of non-transparent pixels
    alpha = np.array(image)[:, :, 3]  # Alpha channel
    rows = np.any(alpha > 0, axis=1)  # Which rows have pixels
    cols = np.any(alpha > 0, axis=0)  # Which columns have pixels
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Bounding box:
    # (x_min, y_min) = top-left corner
    # (x_max, y_max) = bottom-right corner
    
    # 2. Crop to bounding box
    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    # Size: (x_max - x_min + 1, y_max - y_min + 1)
    
    # 3. Calculate target size (85% of canvas)
    target_size = int(512 * ratio)  # 512 * 0.85 = 435
    
    # 4. Resize maintaining aspect ratio
    cropped.thumbnail((target_size, target_size), Image.LANCZOS)
    
    # 5. Create new canvas
    canvas = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    
    # 6. Paste centered
    offset_x = (512 - cropped.width) // 2
    offset_y = (512 - cropped.height) // 2
    canvas.paste(cropped, (offset_x, offset_y))
    
    return canvas
```

**Visual example:**
```
Before:                After:
┌──────────────┐      ┌──────────────┐
│              │      │              │
│   ┌────┐     │      │              │
│   │ 🪑 │     │  →   │    ┌────┐    │  (centered)
│   └────┘     │      │    │ 🪑 │    │  (85% size)
│              │      │    └────┘    │
└──────────────┘      └──────────────┘
```

**Why 85% instead of 100%?**
- Leaves margin for camera rotation
- Prevents object clipping during rendering
- Models trained with similar padding

---

### 3.6 RGBA to RGB Conversion

```python
if image.mode == "RGBA":
    timer.log_progress("🎨 Converting RGBA to RGB...")
    image = np.array(image).astype(np.float32)/255.0
    image = image[:, :, :3]*image[:, :, 3:4] + (1-image[:, :, 3:4])*0.5
    image = Image.fromarray((image*255).astype(np.uint8))
```

**Why convert to RGB?**
- TSR model expects RGB input (3 channels)
- RGBA has 4 channels (incompatible)
- Need to blend transparent areas with background

**Mathematical breakdown:**

**Step 1: Convert to float (0-1 range)**
```python
img = np.array(image).astype(np.float32) / 255.0
# Shape: (512, 512, 4)
# Values: 0.0 (black) to 1.0 (white)
```

**Step 2: Split channels**
```python
rgb = img[:, :, :3]    # Shape: (512, 512, 3)
alpha = img[:, :, 3:4] # Shape: (512, 512, 1)
```

**Step 3: Alpha blending formula**
```python
result = rgb * alpha + background * (1 - alpha)
```

**Explanation:**
- `rgb * alpha` - Foreground contribution
- `background * (1 - alpha)` - Background contribution
- Background color: 0.5 (gray)

**Example pixel calculation:**
```python
# Input RGBA:
r, g, b, a = 200, 100, 50, 128  # Semi-transparent orange
# Normalized:
r, g, b, a = 0.78, 0.39, 0.20, 0.50

# Alpha blending:
result_r = 0.78 * 0.50 + 0.5 * (1 - 0.50) = 0.39 + 0.25 = 0.64
result_g = 0.39 * 0.50 + 0.5 * (1 - 0.50) = 0.195 + 0.25 = 0.445
result_b = 0.20 * 0.50 + 0.5 * (1 - 0.50) = 0.10 + 0.25 = 0.35

# Back to 0-255:
result_r = 0.64 * 255 = 163
result_g = 0.445 * 255 = 113
result_b = 0.35 * 255 = 89
```

**Why gray (0.5) background?**
- Neutral color (not white or black)
- Doesn't bias model perception
- Similar to training data

**Step 4: Convert back to PIL**
```python
image = Image.fromarray((image * 255).astype(np.uint8))
# Multiply by 255: 0.0-1.0 → 0-255
# astype(np.uint8): Float → unsigned 8-bit int
# fromarray: NumPy → PIL Image
```

---

### 3.7 Square Image Padding

```python
w, h = image.size
if w != h:
    timer.log_progress("⬜ Making image square...")
    max_side = max(w, h)
    delta_w = max_side - w
    delta_h = max_side - h
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w//2, delta_h - delta_h//2)
    image = ImageOps.expand(image, padding, fill=(255,255,255))
```

**Purpose:** Ensure image is perfectly square (required by model)

**Padding calculation:**
```python
# Example: 400×512 image
w, h = 400, 512
max_side = max(400, 512) = 512

delta_w = 512 - 400 = 112  # Need 112 pixels width
delta_h = 512 - 512 = 0    # Already tall enough

# Padding: (left, top, right, bottom)
left = 112 // 2 = 56
top = 0 // 2 = 0
right = 112 - 56 = 56
bottom = 0 - 0 = 0

padding = (56, 0, 56, 0)
```

**ImageOps.expand():**
```python
ImageOps.expand(image, border, fill)
# border: (left, top, right, bottom) or single int
# fill: Color tuple (R, G, B)
```

**Visual example:**
```
Before (400×512):        After (512×512):
┌──────────┐            ┌───┬──────────┬───┐
│          │            │   │          │   │
│   🪑     │     →      │   │   🪑     │   │ (white padding)
│          │            │   │          │   │
└──────────┘            └───┴──────────┴───┘
                         56px  400px   56px
```

**Why white (255, 255, 255)?**
- Matches common backgrounds
- Neutral for object detection
- Consistent with training data

---

### 3.8 Saving Processed Image

```python
folder_id = str(int(time.time()))
image_dir = os.path.join(app.config['OUTPUT_FOLDER'], folder_id)
os.makedirs(image_dir, exist_ok=True)
image.save(os.path.join(image_dir, "input.png"))
timer.log_progress(f"💾 Processed image saved to folder: {folder_id}")
```

**Output directory structure:**
```
output/
└── 1732723401/          ← folder_id (timestamp)
    └── input.png        ← Processed input image
        (later: mesh.obj, render.mp4, etc.)
```

**Why save processed input?**
- Debugging (see what model received)
- Gallery thumbnail
- Reproducibility (can re-run with same input)

---

### 3.9 TSR Model Inference

```python
timer.start("Running model")
timer.log_progress("🧠 Initializing AI neural network...")
timer.log_progress("🔮 Generating 3D scene codes...")
with torch.no_grad():
    scene_codes = model([image], device=device)
timer.log_progress("🎯 3D scene generation completed!")
timer.end("Running model")
```

**`torch.no_grad()` context:**
```python
with torch.no_grad():
    # Disables gradient computation
    output = model(input)
```

**Why disable gradients?**

**Training (need gradients):**
```python
output = model(input)
loss = criterion(output, target)
loss.backward()  # Compute gradients
optimizer.step()  # Update weights
```

**Inference (don't need gradients):**
```python
with torch.no_grad():
    output = model(input)
    # No backward pass
    # Saves memory (50-75% reduction)
    # Faster (no gradient bookkeeping)
```

**Model input format:**
```python
model([image], device=device)
# [image] = List of PIL Images (batch of 1)
# device = "cuda" or "cpu"
```

**Inside model.forward():**

**Step 1: Preprocess image**
```python
def forward(self, image, device):
    # Convert PIL → Tensor
    rgb_cond = self.image_processor(image, 512)
    # Shape: [1, 1, 512, 512, 3]
    # Values: 0.0-1.0 (normalized)
    
    rgb_cond = rgb_cond.to(device)
    # Move to GPU
```

**Step 2: Image tokenization**
```python
    input_image_tokens = self.image_tokenizer(
        rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1)
    )
    # Input shape: [1, 1, 3, 512, 512]
    # Output shape: [1, 1, 768, 16, 16]
    # 768 = feature dimension
    # 16×16 = spatial tokens
```

**Image tokenizer architecture:**
```
Input: [1, 3, 512, 512]
    ↓
Conv2d(3, 64, kernel=7, stride=4)  → [1, 64, 128, 128]
    ↓
ResBlock × 4                       → [1, 64, 128, 128]
    ↓
Conv2d(64, 128, kernel=3, stride=2) → [1, 128, 64, 64]
    ↓
ResBlock × 4                       → [1, 128, 64, 64]
    ↓
Conv2d(128, 256, kernel=3, stride=2) → [1, 256, 32, 32]
    ↓
ResBlock × 4                       → [1, 256, 32, 32]
    ↓
Conv2d(256, 768, kernel=3, stride=2) → [1, 768, 16, 16]
    ↓
Output: [1, 768, 16, 16] = 256 tokens × 768 features
```

**Step 3: Flatten tokens**
```python
    input_image_tokens = rearrange(
        input_image_tokens,
        "B Nv C Nt -> B (Nv Nt) C",
        Nv=1
    )
    # Shape: [1, 256, 768]
    # 256 = 16×16 spatial tokens
    # 768 = feature dimension per token
```

**Step 4: Create learnable tokens**
```python
    tokens = self.tokenizer(batch_size=1)
    # Shape: [1, 1024, 768]
    # 1024 = number of triplane tokens
    # 768 = feature dimension
    # These are learned embeddings (not from image)
```

**Step 5: Transformer backbone**
```python
    tokens = self.backbone(
        tokens,
        encoder_hidden_states=input_image_tokens
    )
    # Cross-attention between:
    # - triplane tokens (1024)
    # - image tokens (256)
    # Output shape: [1, 1024, 768]
```

**Transformer architecture:**
```
For each layer (12 layers):
    ├─ Self-Attention on tokens
    │   Q = tokens @ W_q
    │   K = tokens @ W_k
    │   V = tokens @ W_v
    │   Attention = softmax(Q @ K^T / √d) @ V
    │
    ├─ Cross-Attention with image
    │   Q = tokens @ W_q
    │   K = image_tokens @ W_k
    │   V = image_tokens @ W_v
    │   Attention = softmax(Q @ K^T / √d) @ V
    │
    ├─ Feed-Forward Network
    │   FFN(x) = Linear(GELU(Linear(x)))
    │
    └─ Layer Norm + Residual Connections
```

**Self-Attention:**
- Tokens attend to each other
- Learns spatial relationships
- "Which parts of 3D space relate?"

**Cross-Attention:**
- Tokens attend to image features
- Learns 2D→3D mapping
- "Which image regions correspond to which 3D locations?"

**Step 6: Decode to triplane**
```python
    scene_codes = self.post_processor(
        self.tokenizer.detokenize(tokens)
    )
    # Shape: [1, 3, 256, 256, 64]
    # 3 = three planes (XY, XZ, YZ)
    # 256×256 = resolution per plane
    # 64 = feature dimension
```

**Triplane representation:**
```
3D space represented by 3 orthogonal 2D planes:

XY Plane (top view):        XZ Plane (side view):
┌─────────────────┐        ┌─────────────────┐
│     256×256     │        │     256×256     │
│   ┌───────┐     │        │   ┌───────┐     │
│   │  🪑   │     │        │   │  🪑   │     │
│   └───────┘     │        │   └───────┘     │
└─────────────────┘        └─────────────────┘

YZ Plane (front view):
┌─────────────────┐
│     256×256     │
│   ┌───────┐     │
│   │  🪑   │     │
│   └───────┘     │
└─────────────────┘

To query a 3D point (x, y, z):
1. Project to XY plane → get feature vector f1
2. Project to XZ plane → get feature vector f2
3. Project to YZ plane → get feature vector f3
4. Combine: f = [f1; f2; f3] (concatenate)
5. Decode f → (density, color)
```

**Memory usage:**
```python
scene_codes = [1, 3, 256, 256, 64]
# Size: 1 × 3 × 256 × 256 × 64 × 4 bytes (float32)
#     = 50,331,648 bytes
#     ≈ 48 MB
```

**Return value:**
```python
return scene_codes  # Tensor on GPU
```

---

### 3.10 3D Rendering Process

```python
timer.start("Rendering")
timer.log_progress("🎬 Starting 3D rendering process...")
timer.log_progress("📹 Rendering 30 camera views...")
render_images = model.render(scene_codes, n_views=30, return_type="pil")
```

**Rendering function signature:**
```python
def render(
    self,
    scene_codes,          # [1, 3, 256, 256, 64]
    n_views=30,           # Number of camera positions
    elevation_deg=0.0,    # Camera height angle
    camera_distance=1.9,  # Distance from origin
    fovy_deg=40.0,        # Field of view (vertical)
    height=256,           # Output image height
    width=256,            # Output image width
    return_type="pil"     # "pil", "np", or "pt"
):
```

**Step 1: Generate camera positions**
```python
rays_o, rays_d = get_spherical_cameras(
    n_views=30,
    elevation_deg=0.0,
    camera_distance=1.9,
    fovy_deg=40.0,
    height=256,
    width=256
)
```

**Spherical camera positioning:**
```
Top view (elevation = 0°):
        
         📷 (0°)
          │
    ┌─────┼─────┐
    │     │     │
📷───┼─────🪑────┼───📷 (90°, 270°)
    │     │     │
    └─────┼─────┘
          │
         📷 (180°)

30 cameras equally spaced around circle:
angles = [0°, 12°, 24°, 36°, ..., 348°]
```

**Ray generation:**
```python
# For each pixel in 256×256 image:
for y in range(256):
    for x in range(256):
        # Convert pixel to normalized coordinates
        u = (x + 0.5) / 256  # 0.0 to 1.0
        v = (y + 0.5) / 256  # 0.0 to 1.0
        
        # Convert to camera space
        # Using perspective projection
        ray_direction = compute_ray_direction(u, v, fovy)
        
        # Store ray origin and direction
        rays_o[y, x] = camera_position
        rays_d[y, x] = ray_direction
```

**Ray representation:**
```
Camera: rays_o = (1.9, 0, 0)  # Position
Pixel center: (128, 128)

Ray equation:
point(t) = rays_o + t * rays_d
         = (1.9, 0, 0) + t * (-0.8, 0, 0)
         
t = 0:   point = (1.9, 0, 0)    [camera]
t = 1:   point = (1.1, 0, 0)
t = 2:   point = (0.3, 0, 0)
t = 2.375: point = (0, 0, 0)    [origin]
```

**Step 2: Render each view**
```python
images = []
for scene_code in scene_codes:  # Batch dimension
    images_ = []
    for i in range(n_views):    # 30 cameras
        with torch.no_grad():
            image = self.renderer(
                self.decoder,
                scene_code,
                rays_o[i],      # [256, 256, 3]
                rays_d[i]       # [256, 256, 3]
            )
        images_.append(process_output(image))
    images.append(images_)
```

**Renderer architecture (NeRF-style):**

**For each ray:**
```python
def render_ray(ray_o, ray_d, scene_code):
    # 1. Sample points along ray
    t_vals = linspace(near=0.5, far=2.5, steps=128)
    # 128 samples between 0.5 and 2.5 units
    
    points = ray_o + t_vals[:, None] * ray_d
    # Shape: [128, 3] (128 3D points)
    
    # 2. Query triplane at each point
    features = []
    for point in points:
        # Project point to three planes
        xy_coords = point[[0, 1]]  # (x, y)
        xz_coords = point[[0, 2]]  # (x, z)
        yz_coords = point[[1, 2]]  # (y, z)
        
        # Bilinear interpolation on each plane
        f_xy = bilinear_sample(scene_code[0], xy_coords)
        f_xz = bilinear_sample(scene_code[1], xz_coords)
        f_yz = bilinear_sample(scene_code[2], yz_coords)
        
        # Concatenate features
        f = concat([f_xy, f_xz, f_yz])  # [192]
        features.append(f)
    
    # 3. Decode features to density and color
    features = stack(features)  # [128, 192]
    density, color = self.decoder(features)
    # density: [128] - how opaque each point is
    # color: [128, 3] - RGB color at each point
    
    # 4. Volume rendering (integrate along ray)
    # Alpha compositing from back to front
    weights = compute_weights(density, t_vals)
    # weights: [128] - contribution of each sample
    
    rgb = (weights[:, None] * color).sum(dim=0)
    # rgb: [3] - final pixel color
    
    return rgb
```

**Volume rendering equation:**
```
RGB(pixel) = Σ T_i × α_i × c_i

where:
  T_i = transparency (accumulated from previous samples)
      = exp(-Σ σ_j × δ_j) for j < i
  α_i = opacity of sample i
      = 1 - exp(-σ_i × δ_i)
  σ_i = density at sample i
  δ_i = distance between samples
  c_i = color at sample i
```

**Example calculation:**
```python
# Sample densities along ray:
densities = [0.1, 0.5, 2.0, 3.0, 1.0, 0.2, 0.0]
colors = [[1, 0, 0], [1, 0, 0], [1, 0, 0], ...]  # Red

# Compute alphas:
alphas = [0.095, 0.393, 0.865, 0.950, 0.632, 0.181, 0.0]

# Compute transmittance:
T = [1.0, 0.905, 0.549, 0.074, 0.004, 0.001, 0.001]

# Compute weights:
weights = T * alphas = [0.095, 0.355, 0.475, 0.070, 0.002, 0.0, 0.0]

# Final color:
RGB = Σ weights × colors
    = 0.095×[1,0,0] + 0.355×[1,0,0] + 0.475×[1,0,0] + ...
    = [0.997, 0, 0]  # Almost pure red
```

**GPU parallelization:**
```python
# Process all 256×256 rays in parallel
rays_o_flat = rays_o.reshape(-1, 3)  # [65536, 3]
rays_d_flat = rays_d.reshape(-1, 3)  # [65536, 3]

# Render in chunks (memory constraint)
chunk_size = 8192
results = []
for i in range(0, 65536, chunk_size):
    chunk_rays_o = rays_o_flat[i:i+chunk_size]
    chunk_rays_d = rays_d_flat[i:i+chunk_size]
    chunk_result = render_rays(chunk_rays_o, chunk_rays_d)
    results.append(chunk_result)

final_image = concat(results).reshape(256, 256, 3)
```

**Output:**
```python
render_images = [[img0, img1, ..., img29]]
# List of batches (1 batch)
# Each batch contains 30 PIL Images
# Each image: 256×256 RGB
```

---

### 3.11 Video Creation

```python
timer.log_progress("🎞️ Creating MP4 video...")
save_video(render_images[0], os.path.join(image_dir, "render.mp4"), fps=30)
```

**save_video() function:**
```python
def save_video(images, output_path, fps=30):
    import imageio
    
    # Convert PIL images to NumPy arrays
    frames = []
    for img in images:
        frame = np.array(img)  # [256, 256, 3]
        frames.append(frame)
    
    # Stack into video tensor
    video = np.stack(frames)  # [30, 256, 256, 3]
    
    # Write video file
    imageio.mimsave(
        output_path,
        video,
        fps=fps,
        codec='libx264',     # H.264 codec
        quality=8,           # 0-10 (10=best)
        pixelformat='yuv420p'# Browser-compatible
    )
```

**H.264 encoding:**
- Modern video codec
- High compression (1-2 MB for 30 frames)
- Hardware acceleration (GPU encoding)
- Browser-native playback

**YUV420P color space:**
- Y = Luminance (brightness)
- U, V = Chrominance (color)
- 4:2:0 = chroma subsampling
  - 4 Y samples
  - 2 U samples
  - 0 V samples
- Saves 50% space vs RGB

**File size calculation:**
```
Uncompressed:
30 frames × 256×256 pixels × 3 bytes = 5,898,240 bytes ≈ 5.6 MB

Compressed (H.264):
≈ 1-2 MB (compression ratio: 3-5×)
```

---

### 3.12 Saving Render Frames

```python
timer.log_progress("🖼️ Saving individual render frames...")
for ri, render_image in enumerate(render_images[0]):
    render_image.save(os.path.join(image_dir, f"render_{ri:03d}.png"))
    if ri % 10 == 0:
        timer.log_progress(f"📸 Saved frame {ri+1}/30")
```

**Format string `f"render_{ri:03d}.png"`:**
```python
ri = 0  → "render_000.png"
ri = 1  → "render_001.png"
ri = 15 → "render_015.png"
ri = 29 → "render_029.png"
```

**`:03d` explanation:**
- `0` = pad with zeros
- `3` = minimum width of 3 digits
- `d` = decimal integer

**Why save individual frames?**
- Gallery preview (display first few frames)
- Debugging (inspect specific angles)
- Alternative to video (GIF creation)
- Redundancy (if video generation fails)

---

---

### 3.13 Mesh Extraction (Marching Cubes)

```python
timer.log_progress("⚙️ Extracting 3D mesh...")
meshes = model.extract_mesh(scene_codes, has_vertex_color=True)
```

**extract_mesh() function:**
```python
def extract_mesh(self, scene_codes, resolution=256, threshold=0.0, has_vertex_color=True):
    """
    Extract 3D mesh using Marching Cubes algorithm
    
    Args:
        scene_codes: [1, 3, 256, 256, 64] - triplane features
        resolution: Grid resolution (256 = 256³ voxel grid)
        threshold: Density threshold (0.0 = surface boundary)
        has_vertex_color: Extract RGB colors from model
    
    Returns:
        List of trimesh.Trimesh objects
    """
```

**Step 1: Create 3D sampling grid**
```python
# Create uniform 3D grid
x = torch.linspace(-1, 1, resolution)  # [-1, 1] with 256 points
y = torch.linspace(-1, 1, resolution)
z = torch.linspace(-1, 1, resolution)

# Create meshgrid (all combinations)
xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

# Shape: (256, 256, 256) each
# Total points: 256³ = 16,777,216 points
```

**Grid visualization:**
```
3D Grid (256³):

    z
    ↑
    │   ●─●─●─●─●  (top layer)
    │  ╱│╱│╱│╱│╱│
    │ ●─●─●─●─●─│
    │ │ │ │ │ │ │
    └─●─●─●─●─●─●──→ x
     ╱ ╱ ╱ ╱ ╱ ╱
    ●─●─●─●─●─●  (bottom layer)
   ╱
  y

Each point: (x, y, z) ∈ [-1, 1]³
Grid spacing: 2/256 ≈ 0.0078
```

**Step 2: Query density at all grid points**
```python
# Flatten grid to list of points
points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
# Shape: [16777216, 3]

# Process in chunks (GPU memory constraint)
chunk_size = 100000  # 100K points at a time
densities = []

for i in range(0, len(points), chunk_size):
    chunk = points[i:i+chunk_size]
    
    # Query triplane features
    density_chunk = query_triplane_density(chunk, scene_codes[0])
    densities.append(density_chunk)

# Combine all densities
density = torch.cat(densities).reshape(256, 256, 256)
# Shape: [256, 256, 256] - density at each voxel
```

**query_triplane_density() internals:**
```python
def query_triplane_density(points, scene_code):
    # points: [N, 3] - 3D coordinates
    # scene_code: [3, 256, 256, 64] - triplane features
    
    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Project to three planes
    xy_coords = torch.stack([x, y], dim=-1)  # [N, 2]
    xz_coords = torch.stack([x, z], dim=-1)  # [N, 2]
    yz_coords = torch.stack([y, z], dim=-1)  # [N, 2]
    
    # Normalize to [0, 1] (grid_sample expects this range)
    xy_coords = (xy_coords + 1) / 2  # [-1,1] → [0,1]
    xz_coords = (xz_coords + 1) / 2
    yz_coords = (yz_coords + 1) / 2
    
    # Bilinear interpolation on each plane
    f_xy = F.grid_sample(
        scene_code[0:1],    # [1, 64, 256, 256]
        xy_coords.reshape(1, 1, -1, 2),  # [1, 1, N, 2]
        align_corners=False
    ).reshape(64, -1).T  # [N, 64]
    
    f_xz = F.grid_sample(
        scene_code[1:2],
        xz_coords.reshape(1, 1, -1, 2),
        align_corners=False
    ).reshape(64, -1).T  # [N, 64]
    
    f_yz = F.grid_sample(
        scene_code[2:3],
        yz_coords.reshape(1, 1, -1, 2),
        align_corners=False
    ).reshape(64, -1).T  # [N, 64]
    
    # Concatenate features from three planes
    features = torch.cat([f_xy, f_xz, f_yz], dim=-1)  # [N, 192]
    
    # Decode to density using MLP
    density = self.decoder.density_network(features)  # [N, 1]
    
    return density.squeeze(-1)  # [N]
```

**Bilinear interpolation:**
```
Grid cell:
  p01 ●───────● p11
      │       │
      │   ●p  │  ← query point
      │       │
  p00 ●───────● p10

Bilinear formula:
f(p) = (1-dx)(1-dy)·f(p00) + dx(1-dy)·f(p10)
       + (1-dx)dy·f(p01) + dx·dy·f(p11)

where:
  dx = (px - x0) / (x1 - x0)
  dy = (py - y0) / (y1 - y0)
```

**Step 3: Marching Cubes algorithm**
```python
from skimage.measure import marching_cubes

vertices, faces = marching_cubes(
    density.cpu().numpy(),  # [256, 256, 256]
    level=threshold,        # 0.0 = surface
    spacing=(2/256, 2/256, 2/256)  # Voxel size
)
```

**Marching Cubes algorithm:**

**Basic concept:**
```
For each cube in the 256³ grid:
1. Sample density at 8 corners
2. Classify each corner (inside/outside surface)
3. Look up triangulation pattern
4. Generate triangles

Cube corners:
      7───────6
     ╱│      ╱│
    4─┼─────5 │
    │ │     │ │
    │ 3─────┼─2
    │╱      │╱
    0───────1
```

**Corner classification:**
```python
# For each corner, check if density > threshold
corners = [
    density[x,   y,   z  ],  # Corner 0
    density[x+1, y,   z  ],  # Corner 1
    density[x+1, y+1, z  ],  # Corner 2
    density[x,   y+1, z  ],  # Corner 3
    density[x,   y,   z+1],  # Corner 4
    density[x+1, y,   z+1],  # Corner 5
    density[x+1, y+1, z+1],  # Corner 6
    density[x,   y+1, z+1],  # Corner 7
]

# Create 8-bit index (one bit per corner)
cube_index = 0
for i, corner_density in enumerate(corners):
    if corner_density > threshold:
        cube_index |= (1 << i)

# cube_index ranges from 0 to 255 (2⁸ - 1)
```

**Triangulation lookup table:**
```python
# 256 possible configurations
# Example: cube_index = 1 (only corner 0 inside)
TRIANGLE_TABLE[1] = [
    (0, 8, 3),  # One triangle
]

# Example: cube_index = 15 (corners 0,1,2,3 inside)
TRIANGLE_TABLE[15] = [
    (4, 7, 3),  # Two triangles
    (4, 3, 0)
]
```

**Edge interpolation:**
```
If corners have different signs, surface crosses edge:

Corner 0: density = -0.5 (outside)
Corner 1: density = +0.3 (inside)

Interpolation:
t = (0.0 - (-0.5)) / (0.3 - (-0.5)) = 0.5 / 0.8 = 0.625

Vertex position = corner0 + t * (corner1 - corner0)
                = (0, 0, 0) + 0.625 * (1, 0, 0)
                = (0.625, 0, 0)
```

**Output:**
```python
vertices.shape = (N, 3)  # N = number of vertices (e.g., 50,000)
faces.shape = (M, 3)     # M = number of triangles (e.g., 100,000)

# Vertices are 3D coordinates
vertices[0] = [-0.234, 0.567, -0.123]

# Faces are vertex indices
faces[0] = [0, 1, 2]  # Triangle connects vertices 0, 1, 2
```

**Step 4: Query vertex colors**
```python
if has_vertex_color:
    # Query color at each vertex
    vertices_torch = torch.from_numpy(vertices).float().to(device)
    
    colors = []
    for i in range(0, len(vertices_torch), chunk_size):
        chunk = vertices_torch[i:i+chunk_size]
        
        # Query triplane features
        features = query_triplane_features(chunk, scene_codes[0])
        
        # Decode to RGB color
        color_chunk = self.decoder.color_network(features)
        colors.append(color_chunk)
    
    vertex_colors = torch.cat(colors).cpu().numpy()
    # Shape: [N, 3] - RGB colors in range [0, 1]
```

**color_network architecture:**
```
Input: [N, 192] - triplane features
    ↓
Linear(192, 256) → ReLU
    ↓
Linear(256, 256) → ReLU
    ↓
Linear(256, 3) → Sigmoid
    ↓
Output: [N, 3] - RGB colors [0, 1]
```

**Step 5: Create trimesh object**
```python
import trimesh

mesh = trimesh.Trimesh(
    vertices=vertices,        # [N, 3]
    faces=faces,             # [M, 3]
    vertex_colors=vertex_colors,  # [N, 3] or [N, 4] with alpha
    process=True             # Clean up mesh
)
```

**Mesh processing (`process=True`):**
1. **Remove duplicate vertices:**
   ```python
   # Vertices closer than 1e-8 merged
   unique_vertices, inverse = np.unique(
       vertices, 
       return_inverse=True,
       axis=0
   )
   ```

2. **Remove degenerate faces:**
   ```python
   # Faces with area < 1e-8 removed
   # Faces with repeated vertices removed
   # Example: [0, 1, 1] is degenerate
   ```

3. **Reorder face winding:**
   ```python
   # Ensure consistent normal direction
   # Right-hand rule: fingers curl with vertices → thumb points outward
   ```

4. **Merge nearby vertices:**
   ```python
   mesh.merge_vertices()
   # Combines vertices within tolerance
   ```

**Mesh statistics:**
```python
print(f"Vertices: {len(mesh.vertices)}")    # ~20,000-100,000
print(f"Faces: {len(mesh.faces)}")          # ~40,000-200,000
print(f"Is watertight: {mesh.is_watertight}")  # True/False
print(f"Volume: {mesh.volume}")             # Cubic units
```

---

### 3.14 Texture Baking

```python
timer.log_progress("🎨 Baking vertex colors into texture map...")
mesh.visual = mesh.visual.to_texture()
```

**What is texture baking?**
- Converting vertex colors → UV-mapped texture image
- Allows material/texture file export (MTL + PNG)
- More efficient for rendering (GPU texture sampling)

**Vertex colors vs Texture:**

**Vertex colors:**
```python
# Color stored per vertex
vertices = [
    [0.0, 0.0, 0.0],  # Vertex 0
    [1.0, 0.0, 0.0],  # Vertex 1
    [0.0, 1.0, 0.0],  # Vertex 2
]
vertex_colors = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
]

# Triangle color = interpolate 3 vertex colors
```

**Texture mapping:**
```python
# Color stored in 2D image
texture = Image(width=1024, height=1024)  # PNG file

# Each vertex has UV coordinate
uv_coords = [
    [0.0, 0.0],  # Bottom-left of texture
    [1.0, 0.0],  # Bottom-right
    [0.5, 1.0],  # Top-center
]

# Triangle color = sample texture at interpolated UV
```

**to_texture() algorithm:**

**Step 1: UV unwrapping**
```python
# Create 2D parametrization of 3D surface
# Maps (x, y, z) → (u, v)

# Common methods:
# - Conformal mapping (preserves angles)
# - Isometric mapping (preserves distances)
# - Smart UV (minimizes distortion)
```

**Trimesh uses xatlas for unwrapping:**
```python
from xatlas import xatlas

# Pack 3D mesh into 2D charts
vmapping, indices, uvs = xatlas.parametrize(
    mesh.vertices,  # [N, 3]
    mesh.faces      # [M, 3]
)

# uvs: [N, 2] - UV coordinates for each vertex
# Range: [0, 1] × [0, 1]
```

**UV unwrapping visualization:**
```
3D Mesh:              2D UV Layout:
   ╱│╲                ┌─────────────┐
  ╱ │ ╲               │  ╱▔▔╲       │
 ╱  │  ╲              │ │  🪑│      │
╱───┴───╲             │  ╲__╱       │
                      └─────────────┘
                      0,0  u  1,1
```

**Step 2: Rasterize colors to texture**
```python
# Create empty texture image
texture_size = 1024
texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

# For each face:
for face_idx, face in enumerate(mesh.faces):
    # Get vertex data
    v0, v1, v2 = face
    
    # Vertex colors
    c0 = vertex_colors[v0]
    c1 = vertex_colors[v1]
    c2 = vertex_colors[v2]
    
    # UV coordinates (scaled to texture size)
    uv0 = uvs[v0] * texture_size
    uv1 = uvs[v1] * texture_size
    uv2 = uvs[v2] * texture_size
    
    # Rasterize triangle
    # For each pixel inside triangle:
    for y in range(texture_size):
        for x in range(texture_size):
            if point_in_triangle([x, y], uv0, uv1, uv2):
                # Compute barycentric coordinates
                w0, w1, w2 = barycentric([x, y], uv0, uv1, uv2)
                
                # Interpolate color
                color = w0*c0 + w1*c1 + w2*c2
                
                # Write to texture
                texture[y, x] = (color * 255).astype(np.uint8)
```

**Barycentric coordinates:**
```
Triangle with vertices A, B, C:
Any point P inside can be written as:
P = w0·A + w1·B + w2·C

where w0 + w1 + w2 = 1

Example:
P at center: w0 = w1 = w2 = 0.333
P at vertex A: w0 = 1, w1 = 0, w2 = 0
P on edge AB: w0 = 0.5, w1 = 0.5, w2 = 0
```

**Step 3: Create TextureVisuals object**
```python
from trimesh.visual import TextureVisuals
from PIL import Image

texture_image = Image.fromarray(texture)

mesh.visual = TextureVisuals(
    uv=uvs,                    # [N, 2] UV coordinates
    image=texture_image,        # PIL Image (1024×1024)
    material=SimpleMaterial()   # Material properties
)
```

**Result:**
```python
# Before:
isinstance(mesh.visual, ColorVisuals)  # True
len(mesh.visual.vertex_colors) = 50000  # Color per vertex

# After:
isinstance(mesh.visual, TextureVisuals)  # True
mesh.visual.uv.shape = (50000, 2)  # UV per vertex
mesh.visual.image.size = (1024, 1024)  # Texture image
```

---

### 3.15 File Export (OBJ + MTL + PNG)

```python
timer.log_progress("📦 Exporting OBJ with texture...")

# Export OBJ with MTL and texture
obj_path = os.path.join(image_dir, "mesh.obj")
obj_text, texture_data = trimesh.exchange.obj.export_obj(
    mesh,
    include_texture=True,
    return_texture=True
)
```

**export_obj() internals:**

**Step 1: Generate OBJ text**
```python
def export_obj(mesh, include_texture=True):
    lines = []
    
    # Header
    lines.append("# OBJ file generated by trimesh")
    lines.append(f"# {len(mesh.vertices)} vertices")
    lines.append(f"# {len(mesh.faces)} faces")
    lines.append("")
    
    # Material file reference
    if include_texture:
        lines.append("mtllib mesh.mtl")
        lines.append("usemtl mesh_texture")
        lines.append("")
    
    # Vertices
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    
    lines.append("")
    
    # UV coordinates (if texture)
    if include_texture:
        for uv in mesh.visual.uv:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
        lines.append("")
    
    # Faces
    for face in mesh.faces:
        if include_texture:
            # Format: f v1/vt1 v2/vt2 v3/vt3
            # OBJ uses 1-based indexing
            lines.append(
                f"f {face[0]+1}/{face[0]+1} "
                f"{face[1]+1}/{face[1]+1} "
                f"{face[2]+1}/{face[2]+1}"
            )
        else:
            # Format: f v1 v2 v3
            lines.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}")
    
    return "\n".join(lines)
```

**Example OBJ file:**
```obj
# OBJ file generated by trimesh
# 50000 vertices
# 100000 faces

mtllib mesh.mtl
usemtl mesh_texture

v -0.234567 0.123456 -0.345678
v 0.456789 -0.234567 0.123456
v 0.345678 0.456789 -0.234567
...

vt 0.123456 0.234567
vt 0.345678 0.456789
vt 0.567890 0.678901
...

f 1/1 2/2 3/3
f 4/4 5/5 6/6
f 7/7 8/8 9/9
...
```

**Face format explanation:**
```
f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3

Where:
- v = vertex index
- vt = texture coordinate index
- vn = normal index (optional)

Examples:
f 1/1 2/2 3/3          # Vertex + texture
f 1//1 2//2 3//3       # Vertex + normal (no texture)
f 1 2 3                # Vertex only
```

**Step 2: Generate MTL file**
```python
def generate_mtl(material_name="mesh_texture", texture_file="mesh_texture.png"):
    lines = []
    
    lines.append(f"# MTL file generated by trimesh")
    lines.append("")
    
    # Material definition
    lines.append(f"newmtl {material_name}")
    
    # Material properties
    lines.append("Ka 1.0 1.0 1.0")      # Ambient color (white)
    lines.append("Kd 1.0 1.0 1.0")      # Diffuse color (white)
    lines.append("Ks 0.0 0.0 0.0")      # Specular color (black)
    lines.append("Ns 10.0")             # Specular exponent
    lines.append("illum 2")             # Illumination model
    lines.append("d 1.0")               # Dissolve (opacity)
    
    # Texture map
    lines.append(f"map_Kd {texture_file}")
    
    return "\n".join(lines)
```

**MTL properties explained:**

| Property | Description | Values |
|----------|-------------|--------|
| `Ka` | Ambient color | RGB (0-1) - Color in shadow |
| `Kd` | Diffuse color | RGB (0-1) - Main surface color |
| `Ks` | Specular color | RGB (0-1) - Highlight color |
| `Ns` | Specular exponent | 0-1000 - Shininess (10=matte, 100=shiny) |
| `d` | Dissolve | 0-1 - Opacity (0=transparent, 1=opaque) |
| `illum` | Illumination model | 0-10 - Lighting calculation method |
| `map_Kd` | Diffuse texture map | Filename - PNG/JPG texture file |

**Illumination models:**
```
0 = Color only (no lighting)
1 = Ambient + Diffuse
2 = Ambient + Diffuse + Specular
3 = Reflection + Ray trace
...
```

**Example MTL file:**
```mtl
# MTL file generated by trimesh

newmtl mesh_texture
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.0 0.0 0.0
Ns 10.0
illum 2
d 1.0
map_Kd mesh_texture.png
```

**Step 3: Save texture image**
```python
# texture_data is a dict with texture information
texture_image = mesh.visual.image  # PIL Image

# Save as PNG
texture_path = os.path.join(image_dir, "mesh_texture.png")
texture_image.save(texture_path, format='PNG')
```

**Step 4: Write all files**
```python
# Write OBJ
with open(obj_path, 'w') as f:
    f.write(obj_text)

# Write MTL
mtl_path = os.path.join(image_dir, "mesh.mtl")
with open(mtl_path, 'w') as f:
    f.write(mtl_text)

# Write texture PNG
texture_path = os.path.join(image_dir, "mesh_texture.png")
mesh.visual.image.save(texture_path)

timer.log_progress("✅ OBJ + MTL + Texture exported successfully")
```

**File structure:**
```
output/1732723401/
├── input.png           # Preprocessed input
├── render_000.png      # First camera view
├── render_001.png
├── ...
├── render_029.png      # Last camera view
├── render.mp4          # Video of all views
├── mesh.obj            # 3D geometry
├── mesh.mtl            # Material definition
└── mesh_texture.png    # Texture image (1024×1024)
```

---

### 3.16 STL Export (Fallback Format)

```python
timer.log_progress("📦 Exporting additional formats...")
mesh.export(os.path.join(image_dir, "mesh.stl"))
```

**STL (STereoLithography) format:**
- Binary or ASCII format
- No texture/color support
- Simple triangle mesh
- Used for 3D printing

**Binary STL structure:**
```
Header (80 bytes)
Number of triangles (4 bytes, uint32)

For each triangle:
    Normal vector (12 bytes, 3× float32)
    Vertex 1 (12 bytes, 3× float32)
    Vertex 2 (12 bytes, 3× float32)
    Vertex 3 (12 bytes, 3× float32)
    Attribute byte count (2 bytes, uint16)
```

**Example binary STL:**
```
Offset  Size  Data
0x00    80    "Exported from trimesh..."
0x50    4     100000  (number of triangles)

0x54    12    [0.707, 0.707, 0.0]  (normal)
0x60    12    [0.0, 0.0, 0.0]      (vertex 1)
0x6C    12    [1.0, 0.0, 0.0]      (vertex 2)
0x78    12    [0.5, 1.0, 0.0]      (vertex 3)
0x84    2     0                     (attributes)

... (100,000 triangles total)
```

**File size calculation:**
```
Header: 80 bytes
Count: 4 bytes
Triangles: 100,000 × 50 bytes = 5,000,000 bytes

Total: 5,000,084 bytes ≈ 4.77 MB
```

**ASCII STL format:**
```stl
solid mesh
  facet normal 0.707 0.707 0.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 0.5 1.0 0.0
    endloop
  endfacet
  facet normal -0.707 0.707 0.0
    outer loop
      vertex 0.5 1.0 0.0
      vertex 1.0 0.0 0.0
      vertex 1.5 1.0 0.0
    endloop
  endfacet
  ...
endsolid mesh
```

**ASCII vs Binary:**
| Format | File Size | Speed | Readability |
|--------|-----------|-------|-------------|
| ASCII | ~5-10× larger | Slower | Human-readable |
| Binary | Smaller | Faster | Binary data |

---

### 3.17 Progress Completion & Status Update

```python
timer.end("Processing image")
timer.log_progress("🎉 Processing completed successfully!")

# Mark processing as complete
processing_status[session_id] = {
    'status': 'complete',
    'folder': folder_id,
    'has_texture': True
}
```

**processing_status dictionary:**
```python
processing_status = {
    "1732723401": {
        "status": "complete",      # "processing" or "complete" or "error"
        "folder": "1732723401",    # Output folder name
        "has_texture": True        # Texture files available
    }
}
```

**Status polling (from frontend):**
```javascript
// Check status every 500ms
const checkStatus = setInterval(() => {
    fetch(`/check_status/${sessionId}`)
        .then(res => res.json())
        .then(data => {
            if (data.status === 'complete') {
                clearInterval(checkStatus);
                window.location.href = `/result/${data.folder}`;
            } else if (data.status === 'error') {
                clearInterval(checkStatus);
                alert('Processing failed');
            }
        });
}, 500);
```

---

### 3.18 Error Handling

```python
except Exception as e:
    timer.log_progress(f"❌ Error: {str(e)}")
    processing_status[session_id] = {
        'status': 'error',
        'error': str(e)
    }
    import traceback
    traceback.print_exc()
```

**Exception types:**

**PIL errors:**
```python
try:
    image = Image.open(upload_path)
except FileNotFoundError:
    # File doesn't exist
except PIL.UnidentifiedImageError:
    # Not a valid image format
except OSError:
    # Corrupted file or permission error
```

**CUDA errors:**
```python
try:
    scene_codes = model([image], device="cuda")
except torch.cuda.OutOfMemoryError:
    # GPU out of memory
    # Solution: Reduce chunk_size or use CPU
except RuntimeError as e:
    # CUDA kernel error or device mismatch
```

**Mesh extraction errors:**
```python
try:
    mesh = model.extract_mesh(scene_codes)
except ValueError:
    # No surface found (all densities below threshold)
except MemoryError:
    # Not enough RAM for 256³ grid
```

**File I/O errors:**
```python
try:
    mesh.export("output/mesh.obj")
except PermissionError:
    # No write permission
except OSError:
    # Disk full or path too long
```

**Error recovery strategies:**

1. **Graceful degradation:**
   ```python
   try:
       mesh.visual = mesh.visual.to_texture()
   except Exception as e:
       # Keep vertex colors if texture baking fails
       print(f"Texture baking failed: {e}")
       # Continue without texture
   ```

2. **Resource fallback:**
   ```python
   try:
       scene_codes = model([image], device="cuda")
   except torch.cuda.OutOfMemoryError:
       # Fallback to CPU
       scene_codes = model([image], device="cpu")
   ```

3. **User notification:**
   ```python
   timer.log_progress("⚠️ Warning: Low GPU memory, using CPU")
   ```

---

### 3.19 Complete Background Processing Flow

**Execution timeline:**
```
Time    Step                              Duration
0.0s    ├─ Image loading                 ~0.1s
0.1s    ├─ Background removal (U²-Net)   ~2-5s (GPU) / ~10-20s (CPU)
5.0s    ├─ Foreground resize              ~0.1s
5.1s    ├─ RGBA → RGB conversion          ~0.05s
5.15s   ├─ Square padding                 ~0.02s
5.17s   ├─ Save processed input           ~0.1s
5.27s   ├─ TSR inference                  ~8-15s (GPU) / ~60-120s (CPU)
        │  ├─ Image tokenization          ~0.5s
        │  ├─ Transformer backbone        ~7s
        │  └─ Triplane decoding           ~0.5s
20s     ├─ 3D rendering (30 views)        ~5-10s (GPU) / ~30-60s (CPU)
        │  └─ Per view: ~0.3s
30s     ├─ Video creation                 ~1-2s
32s     ├─ Save render frames             ~0.5s
32.5s   ├─ Mesh extraction                ~5-10s
        │  ├─ Grid sampling               ~3s (16M points)
        │  ├─ Marching Cubes              ~2s
        │  └─ Color query                 ~3s
42.5s   ├─ Texture baking                 ~2-5s
        │  ├─ UV unwrapping (xatlas)      ~1s
        │  └─ Rasterization               ~1s
47.5s   ├─ File export                    ~0.5s
        │  ├─ OBJ + MTL                   ~0.2s
        │  ├─ Texture PNG                 ~0.2s
        │  └─ STL                         ~0.1s
48s     └─ Complete

Total: ~48s (GPU) / ~180s (CPU) for typical input
```

**Memory usage:**
```
Component                   GPU Memory    RAM
──────────────────────────────────────────────
TSR Model (weights)         ~3.5 GB       -
Image tokenizer activations ~200 MB       -
Transformer activations     ~1 GB         -
Scene codes (triplane)      ~48 MB        -
Rendering (30 views)        ~500 MB       ~200 MB
Density grid (256³)         -             ~256 MB
Mesh (vertices + faces)     -             ~50-200 MB
Texture image (1024²)       -             ~4 MB
──────────────────────────────────────────────
Total                       ~5-6 GB       ~500-700 MB
```

---

**End of Part 3**

This comprehensive section covered:

### Image Preprocessing:
- PIL image operations (loading, resizing, conversion)
- Background removal (U²-Net architecture, ONNX Runtime)
- Alpha blending mathematics
- Image padding & centering

### AI Model Pipeline:
- TSR inference (image tokenization → Transformer → triplane decoding)
- Self-attention & cross-attention mechanisms
- Triplane representation (3 orthogonal 2D planes)
- Feature querying and bilinear interpolation

### 3D Rendering:
- Spherical camera positioning
- Ray generation & casting
- NeRF-style volume rendering
- Density accumulation & alpha compositing
- Video encoding (H.264, YUV420P)

### Mesh Extraction:
- Marching Cubes algorithm (256³ grid)
- Cube classification & triangulation
- Edge interpolation
- Vertex color querying

### Texture System:
- UV unwrapping (xatlas parametrization)
- Barycentric interpolation
- Texture rasterization
- ColorVisuals → TextureVisuals conversion

### File Export:
- OBJ format (vertices, UV coords, faces)
- MTL material definition
- PNG texture image
- STL format (binary/ASCII)

### Error Handling:
- Exception types & recovery strategies
- Graceful degradation
- Resource fallbacks
- User notifications

---

## Part 4: Server-Sent Events, 3D Viewer & API Architecture

---

### 4.1 Server-Sent Events (SSE) Deep Dive

**What is SSE?**
- One-way server→client streaming protocol
- Built on HTTP (no WebSocket needed)
- Text-based event stream
- Auto-reconnection in browser
- Simpler than WebSockets for unidirectional data

**SSE vs WebSocket vs Polling:**

| Feature | SSE | WebSocket | Polling |
|---------|-----|-----------|---------|
| Direction | Server→Client | Bidirectional | Client→Server→Client |
| Protocol | HTTP | WebSocket | HTTP |
| Reconnect | Automatic | Manual | N/A |
| Overhead | Low | Medium | High |
| Browser Support | All modern | All modern | All |
| Use Case | Real-time updates | Chat, gaming | Simple status checks |

**SSE Event Format:**
```
data: This is a message\n\n

event: progress\n
data: {"step": 1, "total": 10}\n\n

id: 123\n
data: Message with ID\n
retry: 3000\n\n
```

**Field meanings:**
- `data:` - Message content (required)
- `event:` - Event type (default: "message")
- `id:` - Event ID (for last-event-id reconnection)
- `retry:` - Reconnection timeout in milliseconds
- `\n\n` - Event separator (two newlines)

---

### 4.2 Flask SSE Endpoint Implementation

```python
@app.route('/progress/<session_id>')
def progress(session_id):
    """Stream processing progress updates via SSE"""
    
    def generate():
        # Check if session exists
        if session_id not in progress_queues:
            yield f"data: {json.dumps({'error': 'Invalid session'})}\n\n"
            return
        
        queue = progress_queues[session_id]
        
        try:
            while True:
                # Get message from queue (blocks until available)
                message = queue.get(timeout=30)
                
                # Send as SSE event
                yield f"data: {json.dumps(message)}\n\n"
                
                # Check if processing complete
                if session_id in processing_status:
                    status = processing_status[session_id]['status']
                    if status in ['complete', 'error']:
                        break
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        finally:
            # Clean up
            if session_id in progress_queues:
                del progress_queues[session_id]
    
    # Return streaming response
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Connection': 'keep-alive'
        }
    )
```

**Response object breakdown:**

**`generate()` function:**
```python
def generate():
    # Generator function (uses 'yield')
    # Returns data incrementally
    # Doesn't load entire response in memory
    
    for i in range(10):
        yield f"data: Message {i}\n\n"
        time.sleep(1)
```

**Generator vs Regular Function:**
```python
# Regular function (loads all data):
def regular():
    result = []
    for i in range(10):
        result.append(f"Message {i}")
    return "\n".join(result)  # All at once

# Generator (streams data):
def generator():
    for i in range(10):
        yield f"Message {i}"  # One at a time
```

**Memory comparison:**
```
Regular function (1M messages):
- Builds 1M-item list in RAM
- Memory: ~100 MB
- Response time: Wait for all messages

Generator (1M messages):
- Yields one message at a time
- Memory: ~1 KB (constant)
- Response time: First message immediately
```

**`queue.get(timeout=30)` behavior:**
```python
try:
    message = queue.get(timeout=30)
    # Blocks up to 30 seconds waiting for message
    # If message available: returns immediately
    # If 30 seconds pass with no message: raises queue.Empty
except queue.Empty:
    # No message received in 30 seconds
    # Could send keepalive or close connection
    yield "data: {}\n\n"  # Empty keepalive
```

**Why keepalive?**
- Proxies/firewalls close idle connections (usually 30-60s)
- Keepalive prevents timeout
- Empty message = connection still alive

**Response headers:**

| Header | Value | Purpose |
|--------|-------|---------|
| `Content-Type` | `text/event-stream` | Identifies SSE protocol |
| `Cache-Control` | `no-cache` | Prevent caching of stream |
| `X-Accel-Buffering` | `no` | Disable nginx buffering |
| `Connection` | `keep-alive` | Keep TCP connection open |

**Why `X-Accel-Buffering: no`?**
```
Without header:
Client ← Nginx (buffer 64KB) ← Flask
         ↑
         Waits until buffer full before sending

With header:
Client ← Nginx (no buffer) ← Flask
         ↑
         Immediately forwards each message
```

---

### 4.3 Client-Side SSE Connection

```javascript
const eventSource = new EventSource(`/progress/${sessionId}`);

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Progress update:', data);
    
    // Update UI
    document.getElementById('status').textContent = data.message;
    if (data.step && data.total_steps) {
        const progress = (data.step / data.total_steps) * 100;
        document.getElementById('progress-bar').style.width = progress + '%';
    }
};

eventSource.onerror = function(event) {
    console.error('SSE error:', event);
    eventSource.close();
};
```

**EventSource API:**

**Constructor:**
```javascript
const es = new EventSource(url, {
    withCredentials: false  // Send cookies (default: false)
});
```

**Properties:**
```javascript
es.url         // "http://example.com/progress/123"
es.readyState  // 0=CONNECTING, 1=OPEN, 2=CLOSED
es.withCredentials  // Boolean
```

**Event handlers:**
```javascript
// Default message event
es.onmessage = (event) => {
    console.log(event.data);      // Message content
    console.log(event.lastEventId);  // Last event ID
    console.log(event.origin);    // Server origin
};

// Custom event type
es.addEventListener('progress', (event) => {
    console.log('Progress event:', event.data);
});

// Connection opened
es.onopen = () => {
    console.log('SSE connection opened');
};

// Error occurred
es.onerror = (error) => {
    console.error('SSE error:', error);
    
    // EventSource auto-reconnects unless closed
    if (es.readyState === EventSource.CLOSED) {
        console.log('Connection closed permanently');
    }
};
```

**Auto-reconnection:**
```
Timeline:
0s    - Initial connection
5s    - Connection drops
5s    - Browser automatically reconnects
6s    - Reconnected successfully

Request headers on reconnect:
Last-Event-ID: 123  ← Tells server what was last received
```

**Server can resume from last event:**
```python
@app.route('/progress/<session_id>')
def progress(session_id):
    last_event_id = request.headers.get('Last-Event-ID')
    
    # Replay missed events
    if last_event_id:
        missed_events = get_events_after(session_id, last_event_id)
        for event in missed_events:
            yield f"id: {event.id}\ndata: {event.data}\n\n"
```

---

### 4.4 Progress Update Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Upload Flow                              │
└─────────────────────────────────────────────────────────────────┘

Browser                   Flask                    Background Thread
   │                        │                             │
   │  POST /upload         │                             │
   ├──────────────────────→│                             │
   │                        │  Create progress_queue     │
   │                        │  session_id = "123"        │
   │                        │                             │
   │                        │  Start background thread   │
   │                        ├────────────────────────────→│
   │                        │                             │
   │  Response: session_id  │                             │
   │←──────────────────────┤                             │
   │                        │                             │
   │  GET /progress/123    │                             │
   ├──────────────────────→│                             │
   │                        │  Start SSE stream          │
   │←─────────SSE──────────┤                             │
   │                        │                             │
   │                        │                             │ Load image
   │                        │    queue.put({msg})        │
   │                        │←────────────────────────────┤
   │  data: "Loading..."   │                             │
   │←──────────────────────┤                             │
   │                        │                             │
   │                        │                             │ Remove bg
   │                        │    queue.put({msg})        │
   │                        │←────────────────────────────┤
   │  data: "Removing..."  │                             │
   │←──────────────────────┤                             │
   │                        │                             │
   │                        │                             │ Run model
   │                        │    queue.put({msg})        │
   │                        │←────────────────────────────┤
   │  data: "AI model..."  │                             │
   │←──────────────────────┤                             │
   │                        │                             │
   │                        │                             │ Extract mesh
   │                        │    queue.put({msg})        │
   │                        │←────────────────────────────┤
   │  data: "Extracting..." │                             │
   │←──────────────────────┤                             │
   │                        │                             │
   │                        │  processing_status[123]    │
   │                        │  = {status: "complete"}    │
   │                        │←────────────────────────────┤
   │  data: "Complete!"    │                             │
   │←──────────────────────┤                             │
   │                        │                             │
   │  SSE connection closes│                             │
   │                        │                             │
   │  GET /result/folder   │                             │
   ├──────────────────────→│                             │
   │  HTML response        │                             │
   │←──────────────────────┤                             │
```

---

### 4.5 Three.js 3D Viewer Architecture

**HTML structure (result.html):**
```html
<div id="viewer-container">
    <canvas id="viewer-canvas"></canvas>
    
    <div class="viewer-controls">
        <button id="flat-shading">Flat Shading</button>
        <button id="textured-surface">Textured Surface</button>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.155.0/examples/js/loaders/OBJLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.155.0/examples/js/loaders/MTLLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.155.0/examples/js/controls/OrbitControls.js"></script>
```

**Three.js initialization:**
```javascript
// 1. Create scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);  // Light gray

// 2. Create camera
const camera = new THREE.PerspectiveCamera(
    45,                                    // FOV (degrees)
    container.clientWidth / container.clientHeight,  // Aspect ratio
    0.1,                                  // Near clipping plane
    1000                                  // Far clipping plane
);
camera.position.set(2, 1, 2);            // Position camera
camera.lookAt(0, 0, 0);                  // Look at origin

// 3. Create renderer
const renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById('viewer-canvas'),
    antialias: true,                     // Smooth edges
    alpha: false                         // Opaque background
});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);  // Support retina displays
renderer.shadowMap.enabled = true;       // Enable shadows
renderer.shadowMap.type = THREE.PCFSoftShadowMap;  // Soft shadows
```

**Camera types:**

**PerspectiveCamera (realistic):**
```
FOV = 45°
      
    │ ╲
    │  ╲ Far objects smaller
    │   ╲
Camera  ╲______
         │    │
         │ 📦 │ Near
         │____│
```

**OrthographicCamera (CAD/technical):**
```
    ║     ║
    ║     ║ All objects same size
    ║     ║
Camera  ______
    │      │
    │  📦  │
    │______│
```

**Rendering pipeline:**
```
Scene Graph:
Scene
├── Camera
├── Lights
│   ├── AmbientLight (global illumination)
│   ├── DirectionalLight (sun)
│   └── PointLight (bulb)
├── Mesh
│   ├── Geometry (vertices, faces)
│   └── Material (color, texture, shininess)
└── Controls (OrbitControls)

Every frame:
1. OrbitControls updates camera position
2. Renderer computes visible triangles (frustum culling)
3. Transform vertices: Model → World → View → Clip → Screen
4. Rasterize triangles to pixels
5. Shade pixels (lighting + texture)
6. Output to canvas
```

---

### 4.6 Lighting Setup

```javascript
// Ambient light (global illumination)
const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
scene.add(ambientLight);

// Main directional light (sun)
const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
mainLight.position.set(5, 10, 5);
mainLight.castShadow = true;
mainLight.shadow.mapSize.width = 2048;   // Shadow resolution
mainLight.shadow.mapSize.height = 2048;
scene.add(mainLight);

// Fill light (reduce harsh shadows)
const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
fillLight.position.set(-5, 5, -5);
scene.add(fillLight);

// Back light (rim lighting)
const backLight = new THREE.DirectionalLight(0xffffff, 0.2);
backLight.position.set(0, 5, -10);
scene.add(backLight);
```

**Three-point lighting:**
```
Top View:
        
        Main (Key)
           ↓
     ┌─────────┐
     │         │
Fill →   🪑    ← Back
     │         │
     └─────────┘
```

**Light properties:**

| Property | Type | Description |
|----------|------|-------------|
| `color` | Hex | RGB color (0xffffff = white) |
| `intensity` | Number | Brightness (0-1+) |
| `position` | Vector3 | Light location in 3D space |
| `castShadow` | Boolean | Enable shadow casting |
| `shadow.mapSize` | Vector2 | Shadow texture resolution |

**Shadow map resolution:**
```
512×512   - Low quality, fast (mobile)
1024×1024 - Medium quality
2048×2048 - High quality (default)
4096×4096 - Ultra quality, slow
```

**Ambient vs Directional light:**

**Ambient (uniform):**
```javascript
const ambient = new THREE.AmbientLight(0x404040, 0.5);
// Every surface receives same amount
// No direction, no shadows
// Prevents pure black shadows
```

**Directional (parallel rays):**
```javascript
const directional = new THREE.DirectionalLight(0xffffff, 0.8);
directional.position.set(5, 10, 5);
// Parallel rays from infinity (like sun)
// Creates shadows
// Position determines direction only, not distance
```

---

### 4.7 OBJ Model Loading

```javascript
function loadSimpleOBJ() {
    const loader = new THREE.OBJLoader();
    
    loader.load(
        '/output/{{ folder_id }}/mesh.obj',
        
        // onLoad callback
        function(object) {
            // Store reference
            window.modelObject = object;
            
            // Center model
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            object.position.sub(center);
            
            // Apply flat shading material
            object.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                    child.material = new THREE.MeshPhongMaterial({
                        color: 0xcccccc,
                        flatShading: false,
                        side: THREE.DoubleSide
                    });
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            
            // Add to scene
            scene.add(object);
            
            // Fit camera
            fitCameraToObject(camera, object, controls);
            
            console.log('Model loaded');
        },
        
        // onProgress callback
        function(xhr) {
            const progress = (xhr.loaded / xhr.total) * 100;
            console.log(`Loading: ${progress.toFixed(1)}%`);
        },
        
        // onError callback
        function(error) {
            console.error('Error loading model:', error);
        }
    );
}
```

**OBJLoader internals:**

**Parse OBJ file:**
```javascript
class OBJLoader {
    parse(text) {
        const vertices = [];
        const uvs = [];
        const normals = [];
        const faces = [];
        
        const lines = text.split('\n');
        
        for (let line of lines) {
            const parts = line.trim().split(/\s+/);
            const type = parts[0];
            
            switch (type) {
                case 'v':  // Vertex
                    vertices.push([
                        parseFloat(parts[1]),
                        parseFloat(parts[2]),
                        parseFloat(parts[3])
                    ]);
                    break;
                
                case 'vt':  // UV coordinate
                    uvs.push([
                        parseFloat(parts[1]),
                        parseFloat(parts[2])
                    ]);
                    break;
                
                case 'vn':  // Normal
                    normals.push([
                        parseFloat(parts[1]),
                        parseFloat(parts[2]),
                        parseFloat(parts[3])
                    ]);
                    break;
                
                case 'f':  // Face
                    // Parse "1/1/1 2/2/2 3/3/3"
                    const face = [];
                    for (let i = 1; i < parts.length; i++) {
                        const indices = parts[i].split('/');
                        face.push({
                            v: parseInt(indices[0]) - 1,  // Vertex index
                            vt: parseInt(indices[1]) - 1, // UV index
                            vn: parseInt(indices[2]) - 1  // Normal index
                        });
                    }
                    faces.push(face);
                    break;
            }
        }
        
        return { vertices, uvs, normals, faces };
    }
}
```

**Build Three.js geometry:**
```javascript
const geometry = new THREE.BufferGeometry();

// Flatten arrays for WebGL
const positions = [];  // [x1, y1, z1, x2, y2, z2, ...]
const uvCoords = [];   // [u1, v1, u2, v2, ...]
const normalVecs = []; // [nx1, ny1, nz1, nx2, ny2, nz2, ...]

for (let face of faces) {
    for (let vertex of face) {
        // Position
        const v = vertices[vertex.v];
        positions.push(v[0], v[1], v[2]);
        
        // UV
        if (vertex.vt !== undefined) {
            const uv = uvs[vertex.vt];
            uvCoords.push(uv[0], uv[1]);
        }
        
        // Normal
        if (vertex.vn !== undefined) {
            const n = normals[vertex.vn];
            normalVecs.push(n[0], n[1], n[2]);
        }
    }
}

// Create attributes
geometry.setAttribute('position', 
    new THREE.Float32BufferAttribute(positions, 3));
geometry.setAttribute('uv', 
    new THREE.Float32BufferAttribute(uvCoords, 2));
geometry.setAttribute('normal', 
    new THREE.Float32BufferAttribute(normalVecs, 3));
```

**BufferAttribute explained:**
```javascript
// Old approach (inefficient):
const geometry = new THREE.Geometry();
geometry.vertices.push(new THREE.Vector3(x, y, z));
// Each vertex is an object → memory overhead

// New approach (efficient):
const positions = new Float32Array([x1, y1, z1, x2, y2, z2]);
geometry.setAttribute('position', 
    new THREE.BufferAttribute(positions, 3));
// Flat typed array → direct GPU upload
```

**Memory comparison:**
```
Geometry (old):
50,000 vertices × 3 objects (Vector3) × 48 bytes = 7.2 MB

BufferGeometry (new):
50,000 vertices × 3 floats × 4 bytes = 600 KB

12× less memory!
```

---

### 4.8 Model Centering & Bounding Box

```javascript
// Compute bounding box
const box = new THREE.Box3().setFromObject(object);

// Bounding box properties:
box.min  // Vector3: minimum corner (x, y, z)
box.max  // Vector3: maximum corner (x, y, z)

// Get center point
const center = box.getCenter(new THREE.Vector3());
// center = (min + max) / 2

// Translate object to origin
object.position.sub(center);
// New center = (0, 0, 0)
```

**Bounding box calculation:**
```
Object with vertices:
  v1 = (-0.5, 0.2, -0.3)
  v2 = (0.8, -0.1, 0.6)
  v3 = (0.3, 0.9, -0.2)

Bounding box:
  min = (-0.5, -0.1, -0.3)  ← minimum x, y, z
  max = (0.8, 0.9, 0.6)     ← maximum x, y, z

Center:
  center = ((-0.5+0.8)/2, (-0.1+0.9)/2, (-0.3+0.6)/2)
         = (0.15, 0.4, 0.15)

After centering:
  object.position = (-0.15, -0.4, -0.15)
  New center = (0, 0, 0)
```

**Visual representation:**
```
Before:                    After:
    y                         y
    │                         │
    │   ┌──🪑──┐             │   ┌──🪑──┐
    │   │      │             │   │      │
    └───┼──────┼──→ x        └───┼──────┼──→ x
        │      │            (0,0)│      │
        └──────┘                 └──────┘
```

---

### 4.9 Camera Fitting

```javascript
function fitCameraToObject(camera, object, controls) {
    // Get bounding box
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    
    // Calculate required distance
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);  // Convert to radians
    const cameraDistance = maxDim / (2 * Math.tan(fov / 2));
    
    // Add margin (20%)
    const distance = cameraDistance * 1.2;
    
    // Position camera
    const direction = camera.position.clone().sub(center).normalize();
    camera.position.copy(center).add(direction.multiplyScalar(distance));
    
    // Update controls
    controls.target.copy(center);
    controls.update();
}
```

**Distance calculation:**
```
Field of View (FOV):
    
    ╲   │   ╱
     ╲  │  ╱  FOV angle (e.g., 45°)
      ╲ │ ╱
       ╲│╱
      Camera

Geometry:
   tan(FOV/2) = (object_size/2) / distance
   
   distance = (object_size/2) / tan(FOV/2)
            = object_size / (2 × tan(FOV/2))

Example:
   object_size = 2 units
   FOV = 45° = 0.785 radians
   
   distance = 2 / (2 × tan(0.785/2))
            = 2 / (2 × tan(0.393))
            = 2 / (2 × 0.414)
            = 2 / 0.828
            = 2.41 units
   
   With 20% margin:
   distance = 2.41 × 1.2 = 2.89 units
```

---

### 4.10 OrbitControls

```javascript
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// Configuration
controls.target.set(0, 0, 0);        // Look at origin
controls.enableDamping = true;       // Smooth camera movement
controls.dampingFactor = 0.05;       // Damping intensity
controls.screenSpacePanning = false; // Pan in camera plane
controls.minDistance = 0.5;          // Minimum zoom
controls.maxDistance = 10;           // Maximum zoom
controls.maxPolarAngle = Math.PI;    // Allow full rotation
```

**OrbitControls internals:**

**Mouse/touch input:**
```javascript
class OrbitControls {
    constructor(camera, domElement) {
        this.camera = camera;
        this.domElement = domElement;
        
        // State
        this.target = new THREE.Vector3();  // Look-at point
        this.spherical = new THREE.Spherical();  // Camera position (polar)
        
        // Bind event listeners
        domElement.addEventListener('mousedown', this.onMouseDown);
        domElement.addEventListener('mousemove', this.onMouseMove);
        domElement.addEventListener('wheel', this.onWheel);
        domElement.addEventListener('touchstart', this.onTouchStart);
        domElement.addEventListener('touchmove', this.onTouchMove);
    }
    
    onMouseDown(event) {
        // Left button = rotate
        if (event.button === 0) {
            this.state = 'ROTATE';
        }
        // Middle button = pan
        else if (event.button === 1) {
            this.state = 'PAN';
        }
        // Right button = zoom
        else if (event.button === 2) {
            this.state = 'ZOOM';
        }
    }
    
    onMouseMove(event) {
        const deltaX = event.movementX;
        const deltaY = event.movementY;
        
        if (this.state === 'ROTATE') {
            this.rotateLeft(deltaX * 0.01);
            this.rotateUp(deltaY * 0.01);
        }
        else if (this.state === 'PAN') {
            this.pan(deltaX, deltaY);
        }
    }
    
    onWheel(event) {
        this.dolly(event.deltaY * 0.001);
    }
}
```

**Spherical coordinates:**
```
Cartesian (x, y, z) ←→ Spherical (r, θ, φ)

    z
    │   ╱ Camera
    │  ╱r
    │ ╱│φ (polar)
    │╱_│_____ y
   ╱  │  ╱
  ╱ θ│ ╱ (azimuthal)
 ╱____│╱
x

Conversion:
x = r × sin(φ) × cos(θ)
y = r × sin(φ) × sin(θ)
z = r × cos(φ)

r = distance from origin
θ = angle in XY plane (0 to 2π)
φ = angle from Z axis (0 to π)
```

**Rotate operation:**
```javascript
rotateLeft(angle) {
    this.spherical.theta -= angle;  // Rotate around Z axis
}

rotateUp(angle) {
    this.spherical.phi -= angle;    // Rotate up/down
    // Clamp to prevent flipping
    this.spherical.phi = Math.max(
        0.01,
        Math.min(Math.PI - 0.01, this.spherical.phi)
    );
}
```

**Pan operation:**
```javascript
pan(deltaX, deltaY) {
    // Get camera's right and up vectors
    const cameraRight = new THREE.Vector3();
    const cameraUp = new THREE.Vector3();
    
    camera.getWorldDirection(cameraForward);
    cameraRight.crossVectors(cameraForward, camera.up);
    cameraUp.crossVectors(cameraRight, cameraForward);
    
    // Pan in screen space
    const panOffset = new THREE.Vector3();
    panOffset.addScaledVector(cameraRight, -deltaX * 0.001);
    panOffset.addScaledVector(cameraUp, deltaY * 0.001);
    
    this.target.add(panOffset);
}
```

**Zoom/dolly operation:**
```javascript
dolly(delta) {
    if (delta > 0) {
        // Zoom out
        this.spherical.radius *= 1.1;
    } else {
        // Zoom in
        this.spherical.radius *= 0.9;
    }
    
    // Clamp to limits
    this.spherical.radius = Math.max(
        this.minDistance,
        Math.min(this.maxDistance, this.spherical.radius)
    );
}
```

**Update loop:**
```javascript
update() {
    // Damping (smooth movement)
    if (this.enableDamping) {
        this.spherical.theta += 
            (this.targetSpherical.theta - this.spherical.theta) 
            * this.dampingFactor;
        this.spherical.phi += 
            (this.targetSpherical.phi - this.spherical.phi) 
            * this.dampingFactor;
    }
    
    // Convert spherical to Cartesian
    const position = new THREE.Vector3().setFromSpherical(this.spherical);
    
    // Position relative to target
    this.camera.position.copy(this.target).add(position);
    this.camera.lookAt(this.target);
}
```

---

### 4.11 Texture Loading (On-Demand)

```javascript
function loadTexturedVariant() {
    if (window.texturedModel) {
        console.log('Textured variant already loaded');
        return;
    }
    
    const mtlLoader = new THREE.MTLLoader();
    mtlLoader.setPath('/output/{{ folder_id }}/');
    
    mtlLoader.load('mesh.mtl', function(materials) {
        materials.preload();  // Load all textures
        
        const objLoader = new THREE.OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.setPath('/output/{{ folder_id }}/');
        
        objLoader.load('mesh.obj', function(object) {
            // Store reference
            window.texturedModel = object;
            
            // Center model
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            object.position.sub(center);
            
            // Configure materials
            object.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    
                    // Store original material
                    if (!child.userData.originalMaterial) {
                        child.userData.originalMaterial = child.material;
                    }
                }
            });
            
            // Initially hidden
            object.visible = false;
            scene.add(object);
            
            console.log('Textured variant loaded');
        });
    });
}
```

**MTLLoader internals:**

**Parse MTL file:**
```javascript
class MTLLoader {
    parse(text) {
        const materials = {};
        let currentMaterial = null;
        
        const lines = text.split('\n');
        
        for (let line of lines) {
            const parts = line.trim().split(/\s+/);
            const keyword = parts[0];
            
            switch (keyword) {
                case 'newmtl':  // New material
                    const name = parts[1];
                    currentMaterial = {
                        name: name,
                        ambient: [1, 1, 1],
                        diffuse: [1, 1, 1],
                        specular: [0, 0, 0],
                        shininess: 10,
                        opacity: 1,
                        textures: {}
                    };
                    materials[name] = currentMaterial;
                    break;
                
                case 'Ka':  // Ambient color
                    currentMaterial.ambient = [
                        parseFloat(parts[1]),
                        parseFloat(parts[2]),
                        parseFloat(parts[3])
                    ];
                    break;
                
                case 'Kd':  // Diffuse color
                    currentMaterial.diffuse = [
                        parseFloat(parts[1]),
                        parseFloat(parts[2]),
                        parseFloat(parts[3])
                    ];
                    break;
                
                case 'Ks':  // Specular color
                    currentMaterial.specular = [
                        parseFloat(parts[1]),
                        parseFloat(parts[2]),
                        parseFloat(parts[3])
                    ];
                    break;
                
                case 'Ns':  // Specular exponent
                    currentMaterial.shininess = parseFloat(parts[1]);
                    break;
                
                case 'd':  // Opacity
                    currentMaterial.opacity = parseFloat(parts[1]);
                    break;
                
                case 'map_Kd':  // Diffuse texture
                    currentMaterial.textures.map = parts[1];
                    break;
                
                case 'map_Ks':  // Specular map
                    currentMaterial.textures.specularMap = parts[1];
                    break;
                
                case 'map_Bump':  // Normal/bump map
                    currentMaterial.textures.normalMap = parts[1];
                    break;
            }
        }
        
        return materials;
    }
}
```

**Create Three.js materials:**
```javascript
preload() {
    const textureLoader = new THREE.TextureLoader();
    textureLoader.setPath(this.path);
    
    for (let materialName in this.materials) {
        const mat = this.materials[materialName];
        
        // Create MeshPhongMaterial
        const material = new THREE.MeshPhongMaterial({
            color: new THREE.Color().fromArray(mat.diffuse),
            specular: new THREE.Color().fromArray(mat.specular),
            shininess: mat.shininess,
            opacity: mat.opacity,
            transparent: mat.opacity < 1
        });
        
        // Load textures
        if (mat.textures.map) {
            textureLoader.load(mat.textures.map, function(texture) {
                material.map = texture;
                material.needsUpdate = true;
            });
        }
        
        if (mat.textures.specularMap) {
            textureLoader.load(mat.textures.specularMap, function(texture) {
                material.specularMap = texture;
                material.needsUpdate = true;
            });
        }
        
        this.threeMaterials[materialName] = material;
    }
}
```

**Texture loading:**
```javascript
class TextureLoader {
    load(url, onLoad, onProgress, onError) {
        const image = new Image();
        
        image.onload = () => {
            const texture = new THREE.Texture(image);
            texture.needsUpdate = true;  // Upload to GPU
            onLoad(texture);
        };
        
        image.onerror = () => {
            onError(new Error(`Failed to load ${url}`));
        };
        
        image.src = url;
    }
}
```

**Texture properties:**
```javascript
texture.wrapS = THREE.RepeatWrapping;  // Horizontal wrap
texture.wrapT = THREE.RepeatWrapping;  // Vertical wrap
texture.minFilter = THREE.LinearMipMapLinearFilter;  // Minification
texture.magFilter = THREE.LinearFilter;  // Magnification
texture.anisotropy = renderer.capabilities.getMaxAnisotropy();  // Quality
```

**Wrap modes:**
```
RepeatWrapping:        ClampToEdgeWrapping:    MirroredRepeatWrapping:
┌───┬───┬───┐         ┌───────────┐            ┌───┬───┬───┐
│ T │ T │ T │         │ T T T ... │            │ T │T̃ │ T │
├───┼───┼───┤         ├───────────┤            ├───┼───┼───┤
│ T │ T │ T │   UV    │ T T T ... │            │T̃ │ T │T̃ │
├───┼───┼───┤  [0,2]  ├───────────┤            ├───┼───┼───┤
│ T │ T │ T │         │ T T T ... │            │ T │T̃ │ T │
└───┴───┴───┘         └───────────┘            └───┴───┴───┘
(tile)                (stretch edge)           (mirror)
```

**Filtering:**
```
Minification (far away):
- NearestFilter (blocky, fast)
- LinearFilter (smooth)
- NearestMipMapNearestFilter
- LinearMipMapLinearFilter (best quality)

Magnification (close up):
- NearestFilter (pixelated)
- LinearFilter (blurred)
```

---

---

### 4.12 Material Switching (Flat vs Textured)

```javascript
let textureEnabled = false;

function applyTextureState() {
    if (!window.modelObject) return;
    
    if (textureEnabled && window.texturedModel) {
        // Show textured, hide flat
        window.modelObject.visible = false;
        window.texturedModel.visible = true;
    } else {
        // Show flat, hide textured
        window.modelObject.visible = true;
        if (window.texturedModel) {
            window.texturedModel.visible = false;
        }
    }
    
    updateTextureButtons();
}

function updateTextureButtons() {
    const flatBtn = document.getElementById('flat-shading');
    const texturedBtn = document.getElementById('textured-surface');
    
    if (textureEnabled) {
        flatBtn.classList.remove('active');
        texturedBtn.classList.add('active');
    } else {
        flatBtn.classList.add('active');
        texturedBtn.classList.remove('active');
    }
}
```

**Button event listeners:**
```javascript
document.getElementById('flat-shading').addEventListener('click', () => {
    textureEnabled = false;
    applyTextureState();
});

document.getElementById('textured-surface').addEventListener('click', () => {
    if (!window.texturedModel) {
        // Load texture on first click
        loadTexturedVariant();
        
        // Wait for load, then enable
        const checkLoaded = setInterval(() => {
            if (window.texturedModel) {
                clearInterval(checkLoaded);
                textureEnabled = true;
                applyTextureState();
            }
        }, 100);
    } else {
        textureEnabled = true;
        applyTextureState();
    }
});
```

**Why two separate models?**

**Alternative 1: Material swap (rejected)**
```javascript
// Problem: Texture might not align with flat geometry
mesh.material = flatMaterial;  // Has no UVs
mesh.material = texturedMaterial;  // Expects UVs
// Result: Black/broken texture
```

**Alternative 2: Dynamic UV generation (rejected)**
```javascript
// Problem: UV unwrapping is expensive
if (textureEnabled) {
    generateUVs(mesh);  // ~1-5 seconds
    applyTexture(mesh);
}
// Result: UI freeze
```

**Current solution: Two models (optimal)**
```javascript
// Pros:
// - Instant switching (just hide/show)
// - Each model optimized for its purpose
// - No UV generation needed

// Cons:
// - Uses more memory (~2× mesh data)
// - Additional network request for texture variant

// Memory comparison:
modelObject (flat): 50,000 verts × 36 bytes = 1.8 MB
texturedModel: 50,000 verts × 44 bytes = 2.2 MB (includes UVs)
Total: 4 MB (acceptable for web)
```

---

### 4.13 Animation Loop

```javascript
function animate() {
    requestAnimationFrame(animate);
    
    // Update controls (damping)
    if (controls.enableDamping) {
        controls.update();
    }
    
    // Render scene
    renderer.render(scene, camera);
}

// Start animation loop
animate();
```

**requestAnimationFrame explained:**

**Old approach (setInterval):**
```javascript
setInterval(() => {
    controls.update();
    renderer.render(scene, camera);
}, 1000 / 60);  // 60 FPS

// Problems:
// - Runs even when tab hidden (wastes battery)
// - Not synced with display refresh
// - Can skip frames or run multiple times per frame
```

**New approach (requestAnimationFrame):**
```javascript
function animate() {
    requestAnimationFrame(animate);  // Schedule next frame
    controls.update();
    renderer.render(scene, camera);
}

// Benefits:
// - Pauses when tab hidden (saves battery)
// - Synced with display refresh (60/120/144 Hz)
// - Browser optimizes timing
// - Never runs faster than display can show
```

**Frame timing:**
```
Timeline (60 Hz display = 16.67ms per frame):

Frame 1: 0ms
├─ requestAnimationFrame called
├─ Waits for next display sync
└─ Callback executed at 16.67ms

Frame 2: 16.67ms
├─ controls.update() (0.1ms)
├─ renderer.render() (5ms)
├─ requestAnimationFrame called
└─ Callback scheduled for 33.34ms

Frame 3: 33.34ms
...
```

**High refresh rate displays:**
```javascript
// 60 Hz display: 16.67ms per frame → 60 FPS
// 120 Hz display: 8.33ms per frame → 120 FPS
// 144 Hz display: 6.94ms per frame → 144 FPS

// requestAnimationFrame automatically adapts!
```

**Tab visibility API:**
```javascript
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Tab hidden - animation paused');
    } else {
        console.log('Tab visible - animation resumed');
    }
});

// requestAnimationFrame handles this automatically
```

---

### 4.14 Window Resize Handling

```javascript
window.addEventListener('resize', () => {
    // Update camera aspect ratio
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    
    // Update renderer size
    renderer.setSize(container.clientWidth, container.clientHeight);
});
```

**Why update projection matrix?**

**Camera projection matrix:**
```javascript
// Perspective projection matrix:
const aspect = width / height;
const fov = 45 * Math.PI / 180;  // Convert to radians
const near = 0.1;
const far = 1000;

const f = 1 / Math.tan(fov / 2);

projectionMatrix = [
    f / aspect,  0,  0,                          0,
    0,           f,  0,                          0,
    0,           0,  (far + near) / (near - far),  (2 * far * near) / (near - far),
    0,           0,  -1,                         0
];
```

**Aspect ratio effect:**
```
Correct aspect (16:9):
┌─────────────────────┐
│                     │
│         🪑         │  Looks normal
│                     │
└─────────────────────┘

Wrong aspect (still 4:3 when window is 16:9):
┌─────────────────────┐
│                     │
│     🪑←stretched    │  Distorted!
│                     │
└─────────────────────┘
```

**updateProjectionMatrix():**
```javascript
camera.updateProjectionMatrix() {
    // Recalculate projection matrix
    this.projectionMatrix = makePerspective(
        this.fov,
        this.aspect,    // ← Changed by window resize
        this.near,
        this.far
    );
    
    // Inverse matrix (for unprojection)
    this.projectionMatrixInverse = this.projectionMatrix.invert();
}
```

**Renderer resize:**
```javascript
renderer.setSize(width, height) {
    // Update canvas dimensions
    this.canvas.width = width * this.pixelRatio;
    this.canvas.height = height * this.pixelRatio;
    
    // CSS dimensions (display size)
    this.canvas.style.width = width + 'px';
    this.canvas.style.height = height + 'px';
    
    // WebGL viewport
    this.gl.viewport(0, 0, 
        width * this.pixelRatio, 
        height * this.pixelRatio
    );
}
```

**Pixel ratio (retina displays):**
```javascript
// Standard display:
window.devicePixelRatio = 1
canvas.width = 1920   // Actual pixels
canvas.style.width = '1920px'  // CSS pixels

// Retina display:
window.devicePixelRatio = 2
canvas.width = 3840   // Actual pixels (2× more)
canvas.style.width = '1920px'  // CSS pixels (same size)

// Result: Sharper rendering on retina displays
```

---

### 4.15 Performance Monitoring

```javascript
// FPS counter
let lastTime = performance.now();
let frameCount = 0;
let fps = 0;

function animate() {
    requestAnimationFrame(animate);
    
    // Calculate FPS
    frameCount++;
    const now = performance.now();
    const delta = now - lastTime;
    
    if (delta >= 1000) {  // Every second
        fps = Math.round((frameCount * 1000) / delta);
        frameCount = 0;
        lastTime = now;
        
        document.getElementById('fps-counter').textContent = `${fps} FPS`;
    }
    
    controls.update();
    renderer.render(scene, camera);
}
```

**performance.now() vs Date.now():**

| Method | Resolution | Use Case |
|--------|-----------|----------|
| `Date.now()` | ~1-15ms | Timestamps, date calculations |
| `performance.now()` | ~0.005ms | Animation, benchmarking |

**Example:**
```javascript
// Date.now() (low precision):
const start = Date.now();  // 1733318400000
// ... do work ...
const end = Date.now();    // 1733318400003
const elapsed = end - start;  // 3ms (but could be 0-5ms)

// performance.now() (high precision):
const start = performance.now();  // 123456.789123
// ... do work ...
const end = performance.now();    // 123459.234567
const elapsed = end - start;  // 2.445444ms (accurate)
```

**Three.js Stats panel:**
```javascript
import Stats from 'three/examples/jsm/libs/stats.module.js';

const stats = new Stats();
stats.showPanel(0);  // 0: FPS, 1: ms per frame, 2: memory
document.body.appendChild(stats.dom);

function animate() {
    stats.begin();  // Start timing
    
    controls.update();
    renderer.render(scene, camera);
    
    stats.end();  // End timing
    
    requestAnimationFrame(animate);
}
```

**Stats panel output:**
```
┌─────────┐
│ FPS: 60 │  Green = 60 FPS (good)
│ ▊▊▊▊▊▊▊ │  Yellow = 30-60 FPS (ok)
│ ▊▊▊▊▊▊  │  Red = <30 FPS (bad)
└─────────┘

┌─────────┐
│ MS: 8.3 │  Milliseconds per frame
│ ▊▊▊     │  Lower = better
│ ▊▊      │
└─────────┘

┌─────────┐
│ MB: 128 │  Memory usage
│ ▊▊▊▊▊   │  
│ ▊▊▊▊    │
└─────────┘
```

---

### 4.16 API Endpoints (api.py)

```python
from flask import Blueprint, jsonify, request
import os

api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/upload', methods=['POST'])
def api_upload():
    """API endpoint for image upload (used by Android app)"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
    ext = file.filename.rsplit('.', 1)[1].lower()
    
    if ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type: {ext}'}), 400
    
    # Generate session ID
    session_id = str(int(time.time()))
    
    # Save uploaded file
    filename = f"{session_id}.{ext}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    # Create progress queue
    progress_queues[session_id] = Queue()
    
    # Start background processing
    thread = threading.Thread(
        target=process_image_async,
        args=(upload_path, session_id)
    )
    thread.daemon = True
    thread.start()
    
    # Return session ID
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': 'Processing started'
    }), 200

@api.route('/status/<session_id>', methods=['GET'])
def api_status(session_id):
    """Check processing status"""
    
    if session_id not in processing_status:
        return jsonify({
            'status': 'processing',
            'message': 'Still processing...'
        }), 200
    
    status_data = processing_status[session_id]
    
    if status_data['status'] == 'complete':
        return jsonify({
            'status': 'complete',
            'folder_id': status_data['folder'],
            'has_texture': status_data.get('has_texture', False),
            'files': {
                'obj': f"/output/{status_data['folder']}/mesh.obj",
                'mtl': f"/output/{status_data['folder']}/mesh.mtl",
                'texture': f"/output/{status_data['folder']}/mesh_texture.png",
                'stl': f"/output/{status_data['folder']}/mesh.stl",
                'video': f"/output/{status_data['folder']}/render.mp4"
            }
        }), 200
    
    elif status_data['status'] == 'error':
        return jsonify({
            'status': 'error',
            'message': status_data.get('error', 'Unknown error')
        }), 500
    
    else:
        return jsonify({
            'status': 'processing'
        }), 200

@api.route('/gallery', methods=['GET'])
def api_gallery():
    """Get list of all processed models"""
    
    output_folder = app.config['OUTPUT_FOLDER']
    
    if not os.path.exists(output_folder):
        return jsonify({'items': []}), 200
    
    items = []
    
    # List all folders in output directory
    for folder_name in sorted(os.listdir(output_folder), reverse=True):
        folder_path = os.path.join(output_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Check if processing is complete
        obj_path = os.path.join(folder_path, 'mesh.obj')
        if not os.path.exists(obj_path):
            continue
        
        # Get file info
        input_image = os.path.join(folder_path, 'input.png')
        thumbnail = os.path.join(folder_path, 'render_000.png')
        
        items.append({
            'id': folder_name,
            'timestamp': int(folder_name),
            'thumbnail': f"/output/{folder_name}/render_000.png",
            'input_image': f"/output/{folder_name}/input.png",
            'obj_file': f"/output/{folder_name}/mesh.obj",
            'video': f"/output/{folder_name}/render.mp4"
        })
    
    return jsonify({
        'success': True,
        'count': len(items),
        'items': items
    }), 200

@api.route('/download/<folder_id>/<filename>', methods=['GET'])
def api_download(folder_id, filename):
    """Download individual file"""
    
    # Validate filename (security)
    allowed_files = {
        'mesh.obj', 'mesh.mtl', 'mesh_texture.png', 
        'mesh.stl', 'render.mp4', 'input.png'
    }
    
    if filename not in allowed_files:
        return jsonify({'error': 'Invalid filename'}), 400
    
    file_path = os.path.join(
        app.config['OUTPUT_FOLDER'],
        folder_id,
        filename
    )
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename
    )
```

**Blueprint architecture:**
```python
# Main app (app.py):
from flask import Flask
from api import api

app = Flask(__name__)
app.register_blueprint(api)  # Mount at /api

# Routes:
# app.py:  /upload, /result/<id>, /progress/<id>
# api.py:  /api/upload, /api/status/<id>, /api/gallery
```

**Blueprint benefits:**
- **Organization:** Separate API from web routes
- **URL prefix:** All routes get `/api` prefix automatically
- **Reusability:** Can use same blueprint in multiple apps
- **Testing:** Can test API separately

**JSON response format:**
```javascript
// Success response:
{
    "success": true,
    "session_id": "1733318400",
    "message": "Processing started"
}

// Error response:
{
    "error": "No image provided",
    "status": "error"
}

// Status response:
{
    "status": "complete",
    "folder_id": "1733318400",
    "has_texture": true,
    "files": {
        "obj": "/output/1733318400/mesh.obj",
        "mtl": "/output/1733318400/mesh.mtl",
        ...
    }
}
```

**HTTP status codes:**
| Code | Meaning | When to use |
|------|---------|-------------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid input (missing file, wrong type) |
| 404 | Not Found | Resource doesn't exist |
| 500 | Internal Server Error | Server-side error (processing failed) |

---

### 4.17 Android App Architecture

**Project structure:**
```
android_app/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/irspace/imageto3d/
│   │   │   │   ├── ui/
│   │   │   │   │   ├── MainActivity.kt
│   │   │   │   │   ├── GalleryActivity.kt
│   │   │   │   │   └── ViewerActivity.kt
│   │   │   │   ├── data/
│   │   │   │   │   ├── model/
│   │   │   │   │   │   ├── UploadResponse.kt
│   │   │   │   │   │   ├── StatusResponse.kt
│   │   │   │   │   │   └── GalleryItem.kt
│   │   │   │   │   └── repository/
│   │   │   │   │       └── TripoSRRepository.kt
│   │   │   │   ├── network/
│   │   │   │   │   └── TripoSRApiService.kt
│   │   │   │   ├── di/
│   │   │   │   │   └── NetworkModule.kt
│   │   │   │   └── viewmodel/
│   │   │   │       ├── MainViewModel.kt
│   │   │   │       └── GalleryViewModel.kt
│   │   │   └── AndroidManifest.xml
│   │   └── res/
│   │       ├── layout/
│   │       ├── drawable/
│   │       └── values/
│   └── build.gradle
└── build.gradle
```

**Architecture pattern: MVVM (Model-View-ViewModel)**
```
┌──────────────────────────────────────────────┐
│                   View                       │
│  (Activity/Fragment - UI)                    │
│  - Observes ViewModel                        │
│  - Displays data                             │
│  - Handles user input                        │
└──────────────┬───────────────────────────────┘
               │ Events
               ↓
┌──────────────────────────────────────────────┐
│                ViewModel                     │
│  - Holds UI state                            │
│  - Business logic                            │
│  - Survives configuration changes            │
└──────────────┬───────────────────────────────┘
               │ Data requests
               ↓
┌──────────────────────────────────────────────┐
│               Repository                     │
│  - Single source of truth                    │
│  - Coordinates data sources                  │
└──────────────┬───────────────────────────────┘
               │ Network calls
               ↓
┌──────────────────────────────────────────────┐
│              API Service                     │
│  (Retrofit)                                  │
│  - HTTP requests                             │
│  - JSON parsing                              │
└──────────────────────────────────────────────┘
```

---

### 4.18 Dependency Injection (Hilt)

**NetworkModule.kt:**
```kotlin
@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {
    
    private const val BASE_URL = "http://YOUR_SERVER_IP:5002/"
    
    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            })
            .build()
    }
    
    @Provides
    @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit {
        return Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }
    
    @Provides
    @Singleton
    fun provideTripoSRApiService(retrofit: Retrofit): TripoSRApiService {
        return retrofit.create(TripoSRApiService::class.java)
    }
}
```

**Hilt annotations explained:**

| Annotation | Purpose |
|------------|---------|
| `@Module` | Marks class as dependency provider |
| `@InstallIn(SingletonComponent::class)` | Module lifetime = app lifetime |
| `@Provides` | Marks function that provides dependency |
| `@Singleton` | Only one instance throughout app |

**Dependency graph:**
```
Application
    ↓ (creates)
NetworkModule
    ↓ (provides)
OkHttpClient
    ↓ (used by)
Retrofit
    ↓ (creates)
TripoSRApiService
    ↓ (injected into)
Repository
    ↓ (injected into)
ViewModel
    ↓ (injected into)
Activity
```

**Without Hilt (manual):**
```kotlin
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Manual dependency creation (bad!)
        val okHttp = OkHttpClient.Builder().build()
        val retrofit = Retrofit.Builder()
            .baseUrl("http://...")
            .client(okHttp)
            .build()
        val api = retrofit.create(TripoSRApiService::class.java)
        val repository = TripoSRRepository(api)
        val viewModel = MainViewModel(repository)
        
        // Problems:
        // - Boilerplate code
        // - Hard to test (can't mock dependencies)
        // - Multiple instances (memory waste)
        // - Manual lifecycle management
    }
}
```

**With Hilt (automatic):**
```kotlin
@AndroidEntryPoint
class MainActivity : AppCompatActivity() {
    
    @Inject
    lateinit var viewModel: MainViewModel
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // viewModel automatically injected!
        // All dependencies created automatically
        // Singleton instances reused
        // Proper lifecycle management
    }
}
```

---

### 4.19 Retrofit API Service

**TripoSRApiService.kt:**
```kotlin
interface TripoSRApiService {
    
    @Multipart
    @POST("api/upload")
    suspend fun uploadImage(
        @Part image: MultipartBody.Part
    ): Response<UploadResponse>
    
    @GET("api/status/{sessionId}")
    suspend fun getStatus(
        @Path("sessionId") sessionId: String
    ): Response<StatusResponse>
    
    @GET("api/gallery")
    suspend fun getGallery(): Response<GalleryResponse>
    
    @Streaming
    @GET("output/{folderId}/{filename}")
    suspend fun downloadFile(
        @Path("folderId") folderId: String,
        @Path("filename") filename: String
    ): Response<ResponseBody>
}
```

**Retrofit annotations:**

| Annotation | Purpose | Example |
|------------|---------|---------|
| `@POST("path")` | HTTP POST request | Upload data |
| `@GET("path")` | HTTP GET request | Retrieve data |
| `@Path("name")` | URL path parameter | `/api/status/{sessionId}` |
| `@Query("name")` | URL query parameter | `/api/search?q=chair` |
| `@Body` | Request body (JSON) | Send JSON object |
| `@Part` | Multipart file upload | Upload image |
| `@Multipart` | Enable multipart encoding | Required for file upload |
| `@Streaming` | Stream large files | Download video/models |

**suspend function:**
```kotlin
// Regular function (blocks UI thread):
fun uploadImage(): Response {
    // Blocks for 5 seconds
    return api.uploadImage()  // UI freezes!
}

// Suspend function (async):
suspend fun uploadImage(): Response {
    // Doesn't block UI thread
    return api.uploadImage()  // UI stays responsive
}

// Usage with coroutines:
viewModelScope.launch {
    val response = uploadImage()  // Runs in background
    // Update UI with response
}
```

**Coroutine scopes:**

| Scope | Lifetime | Use Case |
|-------|----------|----------|
| `GlobalScope` | App lifetime | Background tasks (rarely used) |
| `viewModelScope` | ViewModel lifetime | API calls, data processing |
| `lifecycleScope` | Activity/Fragment lifetime | UI updates |

**Multipart file upload:**
```kotlin
suspend fun uploadImage(file: File): UploadResponse {
    // Create request body
    val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
    
    // Create multipart part
    val part = MultipartBody.Part.createFormData(
        "image",           // Form field name
        file.name,         // Filename
        requestFile        // File content
    )
    
    // Upload
    val response = apiService.uploadImage(part)
    
    if (response.isSuccessful) {
        return response.body()!!
    } else {
        throw IOException("Upload failed: ${response.code()}")
    }
}
```

**HTTP request format:**
```http
POST /api/upload HTTP/1.1
Host: 192.168.1.100:5002
Content-Type: multipart/form-data; boundary=----Boundary1234

------Boundary1234
Content-Disposition: form-data; name="image"; filename="chair.jpg"
Content-Type: image/jpeg

[Binary image data...]
------Boundary1234--
```

---

### 4.20 Data Models

**UploadResponse.kt:**
```kotlin
data class UploadResponse(
    @SerializedName("success")
    val success: Boolean,
    
    @SerializedName("session_id")
    val sessionId: String,
    
    @SerializedName("message")
    val message: String
)
```

**StatusResponse.kt:**
```kotlin
data class StatusResponse(
    @SerializedName("status")
    val status: String,  // "processing", "complete", "error"
    
    @SerializedName("folder_id")
    val folderId: String?,
    
    @SerializedName("has_texture")
    val hasTexture: Boolean?,
    
    @SerializedName("files")
    val files: FileUrls?,
    
    @SerializedName("message")
    val message: String?
)

data class FileUrls(
    @SerializedName("obj")
    val obj: String,
    
    @SerializedName("mtl")
    val mtl: String,
    
    @SerializedName("texture")
    val texture: String,
    
    @SerializedName("stl")
    val stl: String,
    
    @SerializedName("video")
    val video: String
)
```

**GalleryItem.kt:**
```kotlin
data class GalleryResponse(
    @SerializedName("success")
    val success: Boolean,
    
    @SerializedName("count")
    val count: Int,
    
    @SerializedName("items")
    val items: List<GalleryItem>
)

data class GalleryItem(
    @SerializedName("id")
    val id: String,
    
    @SerializedName("timestamp")
    val timestamp: Long,
    
    @SerializedName("thumbnail")
    val thumbnail: String,
    
    @SerializedName("input_image")
    val inputImage: String,
    
    @SerializedName("obj_file")
    val objFile: String,
    
    @SerializedName("video")
    val video: String
)
```

**@SerializedName annotation:**
```kotlin
// JSON from server:
{
    "session_id": "1733318400",  // Snake case
    "has_texture": true
}

// Kotlin property:
@SerializedName("session_id")
val sessionId: String  // Camel case

@SerializedName("has_texture")
val hasTexture: Boolean

// Gson automatically converts between formats
```

**Data class benefits:**
```kotlin
data class User(val name: String, val age: Int)

// Automatically generated:
// - toString(): "User(name=John, age=25)"
// - equals(): Compare by property values
// - hashCode(): Hash based on properties
// - copy(): Create modified copy
// - componentN(): Destructuring

val user = User("John", 25)
println(user)  // User(name=John, age=25)

val older = user.copy(age = 26)
val (name, age) = user  // Destructuring
```

---

### 4.21 Repository Pattern

**TripoSRRepository.kt:**
```kotlin
class TripoSRRepository @Inject constructor(
    private val apiService: TripoSRApiService
) {
    
    suspend fun uploadImage(imageUri: Uri, context: Context): Result<String> {
        return try {
            // Convert URI to File
            val file = uriToFile(imageUri, context)
            
            // Create multipart body
            val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
            val part = MultipartBody.Part.createFormData(
                "image",
                file.name,
                requestFile
            )
            
            // Upload
            val response = apiService.uploadImage(part)
            
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!.sessionId)
            } else {
                Result.failure(IOException("Upload failed: ${response.code()}"))
            }
            
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun checkStatus(sessionId: String): Result<StatusResponse> {
        return try {
            val response = apiService.getStatus(sessionId)
            
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                Result.failure(IOException("Status check failed: ${response.code()}"))
            }
            
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getGallery(): Result<List<GalleryItem>> {
        return try {
            val response = apiService.getGallery()
            
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!.items)
            } else {
                Result.failure(IOException("Gallery fetch failed: ${response.code()}"))
            }
            
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    private fun uriToFile(uri: Uri, context: Context): File {
        val contentResolver = context.contentResolver
        val file = File(context.cacheDir, "upload_${System.currentTimeMillis()}.jpg")
        
        contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        
        return file
    }
}
```

**Result type:**
```kotlin
// Sealed class representing success or failure
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Failure(val exception: Exception) : Result<Nothing>()
}

// Usage:
when (val result = repository.uploadImage(uri)) {
    is Result.Success -> {
        val sessionId = result.data
        // Handle success
    }
    is Result.Failure -> {
        val error = result.exception
        // Handle error
    }
}
```

---

### 4.22 ViewModel Implementation

**MainViewModel.kt:**
```kotlin
@HiltViewModel
class MainViewModel @Inject constructor(
    private val repository: TripoSRRepository
) : ViewModel() {
    
    // UI state
    private val _uploadState = MutableLiveData<UploadState>()
    val uploadState: LiveData<UploadState> = _uploadState
    
    private val _processingState = MutableLiveData<ProcessingState>()
    val processingState: LiveData<ProcessingState> = _processingState
    
    fun uploadImage(imageUri: Uri, context: Context) {
        viewModelScope.launch {
            _uploadState.value = UploadState.Uploading
            
            when (val result = repository.uploadImage(imageUri, context)) {
                is Result.Success -> {
                    _uploadState.value = UploadState.Success(result.data)
                    startStatusPolling(result.data)
                }
                is Result.Failure -> {
                    _uploadState.value = UploadState.Error(result.exception.message)
                }
            }
        }
    }
    
    private fun startStatusPolling(sessionId: String) {
        viewModelScope.launch {
            while (true) {
                delay(2000)  // Poll every 2 seconds
                
                when (val result = repository.checkStatus(sessionId)) {
                    is Result.Success -> {
                        val status = result.data
                        
                        when (status.status) {
                            "complete" -> {
                                _processingState.value = ProcessingState.Complete(status)
                                break  // Stop polling
                            }
                            "error" -> {
                                _processingState.value = ProcessingState.Error(
                                    status.message ?: "Processing failed"
                                )
                                break  // Stop polling
                            }
                            else -> {
                                _processingState.value = ProcessingState.Processing
                            }
                        }
                    }
                    is Result.Failure -> {
                        _processingState.value = ProcessingState.Error(
                            result.exception.message ?: "Status check failed"
                        )
                        break  // Stop polling
                    }
                }
            }
        }
    }
}

sealed class UploadState {
    object Idle : UploadState()
    object Uploading : UploadState()
    data class Success(val sessionId: String) : UploadState()
    data class Error(val message: String?) : UploadState()
}

sealed class ProcessingState {
    object Idle : ProcessingState()
    object Processing : ProcessingState()
    data class Complete(val status: StatusResponse) : ProcessingState()
    data class Error(val message: String) : ProcessingState()
}
```

**LiveData vs StateFlow:**

| Feature | LiveData | StateFlow |
|---------|----------|-----------|
| Lifecycle aware | Yes | No (manual) |
| Initial value | Optional | Required |
| Replays last | Yes | Yes |
| Coroutine support | Limited | Full |
| Compose support | Via `observeAsState()` | Native |

**Observer pattern in Activity:**
```kotlin
@AndroidEntryPoint
class MainActivity : AppCompatActivity() {
    
    private val viewModel: MainViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Observe upload state
        viewModel.uploadState.observe(this) { state ->
            when (state) {
                UploadState.Idle -> {
                    // Show upload button
                }
                UploadState.Uploading -> {
                    // Show progress bar
                    binding.progressBar.visibility = View.VISIBLE
                }
                is UploadState.Success -> {
                    // Upload complete, show processing UI
                    showProcessingScreen(state.sessionId)
                }
                is UploadState.Error -> {
                    // Show error message
                    Toast.makeText(this, state.message, Toast.LENGTH_LONG).show()
                }
            }
        }
        
        // Observe processing state
        viewModel.processingState.observe(this) { state ->
            when (state) {
                ProcessingState.Processing -> {
                    binding.statusText.text = "Processing..."
                }
                is ProcessingState.Complete -> {
                    // Navigate to result screen
                    val intent = Intent(this, ViewerActivity::class.java)
                    intent.putExtra("folder_id", state.status.folderId)
                    startActivity(intent)
                }
                is ProcessingState.Error -> {
                    Toast.makeText(this, state.message, Toast.LENGTH_LONG).show()
                }
            }
        }
    }
}
```

---

**End of Part 4**

This comprehensive section covered:

### Three.js Viewer Features:
- Material switching (flat vs textured models)
- Animation loop with requestAnimationFrame
- Window resize handling and aspect ratio
- Performance monitoring (FPS counter)

### Flask API Endpoints:
- Blueprint architecture
- RESTful API design
- JSON response format
- HTTP status codes
- File validation and security

### Android App Architecture:
- MVVM pattern (Model-View-ViewModel)
- Dependency injection with Hilt
- Retrofit API service
- Coroutines and suspend functions
- Repository pattern
- LiveData observers
- Data models with Gson serialization

**Total documentation:** 30,000+ words covering every aspect of the system from AI model inference to Android app development.

**Next steps (optional):**
- Part 5: Testing strategies
- Part 6: Deployment (Docker, cloud hosting)
- Part 7: Performance optimization
- Part 8: Advanced features (batch processing, custom textures)

Ready for more, or would you like me to clarify any specific section?

---
