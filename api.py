from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import threading
import torch
import rembg
import trimesh
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import json
from queue import Queue

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video

app = Flask(__name__)
CORS(app)  # Enable CORS for Android app
app.secret_key = 'triposr-api-secret-key-2024'

# Configuration
app.config['UPLOAD_FOLDER'] = "api_uploads"
app.config['OUTPUT_FOLDER'] = "api_output"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global storage for job status and progress
jobs = {}
job_lock = threading.Lock()
progress_queues = {}
processing_status = {}

# Device and Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading TSR model on {device}...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)
print("✅ Model loaded successfully!")

# Timer class with progress tracking
class Timer:
    def __init__(self, job_id=None):
        self.items = {}
        self.time_scale = 1000.0
        self.time_unit = "ms"
        self.job_id = job_id
        
    def log_progress(self, message, step=None, total_steps=None):
        """Log progress message to the job queue"""
        if self.job_id:
            timestamp = time.strftime("%H:%M:%S")
            progress_data = {
                'message': message,
                'timestamp': timestamp,
                'step': step,
                'total_steps': total_steps
            }
            
            # Update job progress logs
            with job_lock:
                if self.job_id in jobs:
                    if 'logs' not in jobs[self.job_id]:
                        jobs[self.job_id]['logs'] = []
                    jobs[self.job_id]['logs'].append(progress_data)
                    jobs[self.job_id]['last_message'] = message
            
            # Add to progress queue if it exists
            if self.job_id in progress_queues:
                try:
                    progress_queues[self.job_id].put(progress_data)
                except:
                    pass  # Queue might be full or closed
                
    def start(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        self.log_progress(f"🚀 Starting {name}...")
        
    def end(self, name):
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        delta = time.time() - self.items.pop(name)
        duration_msg = f"{delta * self.time_scale:.2f}{self.time_unit}"
        self.log_progress(f"✅ {name} completed in {duration_msg}")
        print(f"{name} finished in {duration_msg}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_3d_generation(job_id, image_paths):
    """Background task for 3D model generation with detailed progress tracking"""
    timer = Timer(job_id)
    
    try:
        # Update status
        with job_lock:
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['progress'] = 5
        
        timer.log_progress("🌟 Starting 3D reconstruction process...")
        timer.log_progress(f"📁 Processing {len(image_paths)} image(s)...")
        
        processed_images = []
        scene_codes_list = []
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            with job_lock:
                jobs[job_id]['progress'] = 10 + (20 * (i + 1) // len(image_paths))
            
            timer.log_progress(f"🖼️ Processing image {i+1}/{len(image_paths)}...")
            
            # Load and preprocess
            original_image = Image.open(image_path)
            resized_image = original_image.resize((512, 512))
            timer.log_progress(f"📐 Image {i+1} resized to 512x512 pixels")
            
            # TSR processing
            timer.start(f"Processing image {i+1}")
            timer.log_progress(f"🎭 Removing background from image {i+1}...")
            rembg_session = rembg.new_session()
            image = remove_background(resized_image, rembg_session)
            timer.log_progress(f"✨ Background removed from image {i+1}")
            
            timer.log_progress(f"🔄 Resizing foreground of image {i+1}...")
            image = resize_foreground(image, ratio=0.85)
            
            # Convert RGBA to RGB
            if image.mode == "RGBA":
                timer.log_progress(f"🎨 Converting RGBA to RGB for image {i+1}...")
                image = np.array(image).astype(np.float32) / 255.0
                image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
                image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Make square
            w, h = image.size
            if w != h:
                timer.log_progress(f"⬜ Making image {i+1} square...")
                max_side = max(w, h)
                delta_w = max_side - w
                delta_h = max_side - h
                padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
                image = ImageOps.expand(image, padding, fill=(255, 255, 255))
            
            processed_images.append(image)
            timer.end(f"Processing image {i+1}")
            
            # Generate scene codes
            timer.log_progress(f"🧠 Running neural network on image {i+1}...")
            with torch.no_grad():
                scene_code = model([image], device=device)
                scene_codes_list.append(scene_code)
            timer.log_progress(f"🎯 Scene codes generated for image {i+1}")
        
        # Create output directory
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed images
        for i, img in enumerate(processed_images):
            img.save(os.path.join(output_dir, f'input_{i}.png'))
        timer.log_progress(f"💾 Saved {len(processed_images)} processed image(s)")
        
        # Fuse scene codes if multiple images
        with job_lock:
            jobs[job_id]['progress'] = 35
        
        if len(scene_codes_list) > 1:
            timer.log_progress("🔄 Fusing scene codes from multiple images...")
            stacked_codes = torch.stack(scene_codes_list, dim=0)
            fused_scene_codes = torch.mean(stacked_codes, dim=0)
            timer.log_progress("✅ Scene codes fused successfully")
        else:
            fused_scene_codes = scene_codes_list[0]
        
        # Run TSR model rendering
        timer.start("Rendering")
        with job_lock:
            jobs[job_id]['progress'] = 45
        timer.log_progress("🎬 Starting 3D rendering process...")
        timer.log_progress("📹 Rendering 30 camera views...")
        
        render_images = model.render(fused_scene_codes, n_views=30, return_type="pil")
        
        with job_lock:
            jobs[job_id]['progress'] = 60
        timer.log_progress("🎞️ Creating MP4 video...")
        save_video(render_images[0], os.path.join(output_dir, "render.mp4"), fps=30)
        timer.log_progress("✅ Video created successfully")
        
        # Save all render frames
        timer.log_progress("🖼️ Saving individual render frames...")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, f"render_{ri:03d}.png"))
            if ri % 10 == 0:  # Update every 10 frames
                timer.log_progress(f"📸 Saved frame {ri+1}/30")
        timer.log_progress("✅ All 30 frames saved")
        
        # Save preview frames (first 8)
        for ri in range(min(8, len(render_images[0]))):
            render_images[0][ri].save(os.path.join(output_dir, f"preview_{ri}.png"))
        
        timer.end("Rendering")
        
        # Export mesh
        timer.start("Exporting mesh")
        with job_lock:
            jobs[job_id]['progress'] = 75
        timer.log_progress("🏗️ Extracting 3D mesh geometry...")
        
        meshes = model.extract_mesh(fused_scene_codes, has_vertex_color=False)
        mesh_obj = os.path.join(output_dir, "mesh.obj")
        meshes[0].export(mesh_obj)
        timer.log_progress("📦 OBJ file exported successfully")
        
        # Convert to STL using trimesh
        stl_file = os.path.join(output_dir, "mesh.stl")
        try:
            timer.log_progress("🔄 Converting to STL format...")
            mesh_trimesh = trimesh.load(mesh_obj)
            mesh_trimesh.export(stl_file)
            
            # Verify the STL file was created and has content
            if os.path.exists(stl_file) and os.path.getsize(stl_file) > 0:
                file_size = os.path.getsize(stl_file)
                timer.log_progress(f"✅ STL file created successfully ({file_size:,} bytes)")
            else:
                timer.log_progress("⚠️ STL file creation failed - file is empty or missing")
        except Exception as e:
            timer.log_progress(f"⚠️ STL conversion failed: {str(e)}")
            print(f"⚠️ STL conversion failed: {e}")
        
        timer.end("Exporting mesh")
        
        # Get file sizes
        obj_size = os.path.getsize(mesh_obj)
        stl_size = os.path.getsize(stl_file) if os.path.exists(stl_file) else 0
        video_size = os.path.getsize(os.path.join(output_dir, "render.mp4"))
        
        # Count render frames
        render_frames = [f'/api/download/{job_id}/render_{i:03d}.png' for i in range(30)]
        
        # Update job as completed
        with job_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['message'] = '🎉 3D model generated successfully!'
            jobs[job_id]['result'] = {
                'job_id': job_id,
                'obj_file': f'/api/download/{job_id}/mesh.obj',
                'stl_file': f'/api/download/{job_id}/mesh.stl',
                'video_file': f'/api/download/{job_id}/render.mp4',
                'preview_images': [f'/api/download/{job_id}/preview_{i}.png' for i in range(8)],
                'render_frames': render_frames,
                'input_images': [f'/api/download/{job_id}/input_{i}.png' for i in range(len(image_paths))],
                'file_sizes': {
                    'obj': obj_size,
                    'stl': stl_size,
                    'video': video_size
                },
                'timestamp': int(time.time())
            }
        
        timer.log_progress("🎉 All processing completed successfully!")
        
    except Exception as e:
        error_msg = f"❌ Error during processing: {str(e)}"
        timer.log_progress(error_msg)
        print(f"Error in job {job_id}: {e}")
        
        with job_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['progress'] = 0
            jobs[job_id]['message'] = error_msg
            jobs[job_id]['error'] = str(e)

# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'TripoSR API is running',
        'device': device,
        'model': 'stabilityai/TripoSR',
        'version': '1.0.0'
    }), 200

@app.route('/api/upload', methods=['POST'])
def upload_images():
    """
    Upload images and start 3D generation
    
    Request:
    - Form data with 1-5 images under 'images' field
    
    Response:
    - job_id: Unique identifier for tracking the job
    - status: 'queued' or 'processing'
    - message: Status message
    """
    if 'images' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No images provided'
        }), 400
    
    files = request.files.getlist('images')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({
            'success': False,
            'error': 'No valid images selected'
        }), 400
    
    if len(files) > 5:
        return jsonify({
            'success': False,
            'error': 'Maximum 5 images allowed'
        }), 400
    
    # Validate all files
    for file in files:
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type: {file.filename}'
            }), 400
    
    # Generate job ID
    job_id = f"job_{int(time.time() * 1000)}"
    
    # Save uploaded files
    image_paths = []
    filenames = []
    for i, file in enumerate(files):
        if file and file.filename:
            filename = secure_filename(f"{job_id}_{i}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_paths.append(filepath)
            filenames.append(file.filename)
    
    # Initialize job tracking with logs
    with job_lock:
        jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0,
            'message': 'Job queued for processing',
            'created_at': int(time.time()),
            'image_count': len(image_paths),
            'filenames': filenames,
            'logs': []
        }
    
    # Setup progress queue for SSE
    progress_queues[job_id] = Queue(maxsize=100)
    
    # Start background processing
    thread = threading.Thread(target=process_3d_generation, args=(job_id, image_paths))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'status': 'queued',
        'message': f'Processing {len(image_paths)} image(s)',
        'image_count': len(image_paths),
        'filenames': filenames,
        'progress_stream': f'/api/progress/{job_id}'
    }), 202

@app.route('/api/progress/<job_id>')
def progress_stream(job_id):
    """
    Server-Sent Events endpoint for real-time progress updates
    
    This endpoint streams progress updates in real-time as the job processes.
    Android clients can use EventSource or OkHttp SSE to receive updates.
    """
    def generate():
        print(f"SSE connection opened for job: {job_id}")
        
        if job_id not in progress_queues:
            print(f"Job {job_id} not found in progress_queues")
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return
            
        queue = progress_queues[job_id]
        
        # Send initial connection confirmation
        yield f"data: {json.dumps({'connected': True, 'job_id': job_id})}\n\n"
        
        while True:
            try:
                # Check if processing is done
                with job_lock:
                    if job_id in jobs:
                        status = jobs[job_id]['status']
                        if status in ['completed', 'failed']:
                            final_data = jobs[job_id].copy()
                            print(f"Processing finished for job {job_id}: {status}")
                            yield f"data: {json.dumps(final_data)}\n\n"
                            break
                
                # Get progress updates
                try:
                    progress = queue.get(timeout=1)
                    print(f"Sending progress: {progress}")
                    yield f"data: {json.dumps(progress)}\n\n"
                except:
                    # Send heartbeat
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                    
            except Exception as e:
                print(f"Error in SSE stream: {e}")
                break
                
        # Cleanup
        print(f"Cleaning up SSE session: {job_id}")
        if job_id in progress_queues:
            del progress_queues[job_id]
    
    return Response(generate(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*',
                           'X-Accel-Buffering': 'no'})

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get job status and progress
    
    Response:
    - status: 'queued', 'processing', 'completed', 'failed'
    - progress: 0-100
    - message: Current status message
    - logs: Array of progress log messages
    - result: Download URLs (if completed)
    """
    with job_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        job_data = jobs[job_id].copy()
    
    return jsonify({
        'success': True,
        'data': job_data
    }), 200

@app.route('/api/download/<job_id>/<filename>', methods=['GET'])
def download_file(job_id, filename):
    """
    Download generated files
    
    Parameters:
    - job_id: Job identifier
    - filename: File to download (mesh.obj, mesh.stl, render.mp4, preview_N.png)
    """
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({
            'success': False,
            'error': 'File not found'
        }), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/preview/<job_id>', methods=['GET'])
def get_preview_base64(job_id):
    """
    Get preview images as base64 for quick display in Android app
    
    Response:
    - Array of base64 encoded preview images
    """
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    if not os.path.exists(output_dir):
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    previews = []
    for i in range(8):
        preview_path = os.path.join(output_dir, f'preview_{i}.png')
        if os.path.exists(preview_path):
            with open(preview_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
                previews.append({
                    'index': i,
                    'data': f'data:image/png;base64,{img_data}'
                })
    
    return jsonify({
        'success': True,
        'previews': previews
    }), 200

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """
    List all jobs (for debugging/admin)
    
    Optional query parameters:
    - status: Filter by status
    - limit: Limit number of results (default: 50)
    """
    status_filter = request.args.get('status')
    limit = int(request.args.get('limit', 50))
    
    with job_lock:
        job_list = list(jobs.values())
    
    # Filter by status if provided
    if status_filter:
        job_list = [j for j in job_list if j['status'] == status_filter]
    
    # Sort by creation time (newest first)
    job_list.sort(key=lambda x: x.get('created_at', 0), reverse=True)
    
    # Limit results
    job_list = job_list[:limit]
    
    return jsonify({
        'success': True,
        'count': len(job_list),
        'jobs': job_list
    }), 200

@app.route('/api/renders/<job_id>', methods=['GET'])
def get_render_frames(job_id):
    """
    Get all 30 render frame URLs for a completed job
    
    Response:
    - Array of render frame URLs
    """
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    if not os.path.exists(output_dir):
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    render_frames = []
    for i in range(30):
        frame_path = os.path.join(output_dir, f'render_{i:03d}.png')
        if os.path.exists(frame_path):
            render_frames.append({
                'index': i,
                'url': f'/api/download/{job_id}/render_{i:03d}.png',
                'size': os.path.getsize(frame_path)
            })
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'total_frames': len(render_frames),
        'frames': render_frames
    }), 200

@app.route('/api/logs/<job_id>', methods=['GET'])
def get_job_logs(job_id):
    """
    Get detailed processing logs for a job
    
    Response:
    - Array of log messages with timestamps
    """
    with job_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        logs = jobs[job_id].get('logs', [])
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'log_count': len(logs),
        'logs': logs
    }), 200

@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    """
    Get all completed 3D models for gallery view
    
    Optional query parameters:
    - limit: Maximum number of results (default: 50)
    - offset: Pagination offset (default: 0)
    - sort: Sort order - 'newest' or 'oldest' (default: 'newest')
    
    Response:
    - Array of completed jobs with thumbnails and metadata
    """
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    sort_order = request.args.get('sort', 'newest')
    
    gallery_items = []
    
    # Scan output directory for completed jobs
    output_folder = app.config['OUTPUT_FOLDER']
    if os.path.exists(output_folder):
        for folder_name in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Check if required files exist
            obj_file = os.path.join(folder_path, 'mesh.obj')
            video_file = os.path.join(folder_path, 'render.mp4')
            preview_0 = os.path.join(folder_path, 'preview_0.png')
            
            if not (os.path.exists(obj_file) and os.path.exists(video_file)):
                continue
            
            # Get job metadata
            with job_lock:
                job_data = jobs.get(folder_name, {})
            
            # Get file sizes
            obj_size = os.path.getsize(obj_file) if os.path.exists(obj_file) else 0
            stl_file = os.path.join(folder_path, 'mesh.stl')
            stl_size = os.path.getsize(stl_file) if os.path.exists(stl_file) else 0
            video_size = os.path.getsize(video_file) if os.path.exists(video_file) else 0
            
            # Count input images
            input_images = []
            for i in range(10):  # Check up to 10 input images
                input_file = os.path.join(folder_path, f'input_{i}.png')
                if os.path.exists(input_file):
                    input_images.append(f'/api/download/{folder_name}/input_{i}.png')
                else:
                    break
            
            # Count preview images
            preview_images = []
            for i in range(8):
                preview_file = os.path.join(folder_path, f'preview_{i}.png')
                if os.path.exists(preview_file):
                    preview_images.append(f'/api/download/{folder_name}/preview_{i}.png')
            
            # Get creation timestamp from folder name or file
            try:
                if folder_name.startswith('job_'):
                    timestamp = int(folder_name.split('_')[1]) // 1000
                else:
                    timestamp = int(folder_name)
            except:
                timestamp = int(os.path.getctime(folder_path))
            
            gallery_items.append({
                'job_id': folder_name,
                'thumbnail': preview_images[0] if preview_images else None,
                'preview_images': preview_images,
                'input_images': input_images,
                'video_url': f'/api/download/{folder_name}/render.mp4',
                'obj_url': f'/api/download/{folder_name}/mesh.obj',
                'stl_url': f'/api/download/{folder_name}/mesh.stl',
                'created_at': timestamp,
                'image_count': len(input_images),
                'status': job_data.get('status', 'completed'),
                'file_sizes': {
                    'obj': obj_size,
                    'stl': stl_size,
                    'video': video_size
                },
                'filenames': job_data.get('filenames', [])
            })
    
    # Sort gallery items
    reverse = (sort_order == 'newest')
    gallery_items.sort(key=lambda x: x['created_at'], reverse=reverse)
    
    # Apply pagination
    total_count = len(gallery_items)
    gallery_items = gallery_items[offset:offset + limit]
    
    return jsonify({
        'success': True,
        'total': total_count,
        'limit': limit,
        'offset': offset,
        'count': len(gallery_items),
        'items': gallery_items
    }), 200

@app.route('/api/gallery/<job_id>', methods=['GET'])
def get_gallery_item(job_id):
    """
    Get detailed information for a specific gallery item
    
    Response:
    - Complete job information including all files and metadata
    """
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    if not os.path.exists(output_dir):
        return jsonify({
            'success': False,
            'error': 'Gallery item not found'
        }), 404
    
    # Get job data
    with job_lock:
        job_data = jobs.get(job_id, {})
    
    # Gather all files
    obj_file = os.path.join(output_dir, 'mesh.obj')
    stl_file = os.path.join(output_dir, 'mesh.stl')
    video_file = os.path.join(output_dir, 'render.mp4')
    
    # Input images
    input_images = []
    for i in range(10):
        input_file = os.path.join(output_dir, f'input_{i}.png')
        if os.path.exists(input_file):
            input_images.append({
                'index': i,
                'url': f'/api/download/{job_id}/input_{i}.png',
                'size': os.path.getsize(input_file)
            })
        else:
            break
    
    # Preview images
    preview_images = []
    for i in range(8):
        preview_file = os.path.join(output_dir, f'preview_{i}.png')
        if os.path.exists(preview_file):
            preview_images.append({
                'index': i,
                'url': f'/api/download/{job_id}/preview_{i}.png',
                'size': os.path.getsize(preview_file)
            })
    
    # Render frames
    render_frames = []
    for i in range(30):
        render_file = os.path.join(output_dir, f'render_{i:03d}.png')
        if os.path.exists(render_file):
            render_frames.append({
                'index': i,
                'url': f'/api/download/{job_id}/render_{i:03d}.png',
                'size': os.path.getsize(render_file)
            })
    
    # Get timestamp
    try:
        if job_id.startswith('job_'):
            timestamp = int(job_id.split('_')[1]) // 1000
        else:
            timestamp = int(job_id)
    except:
        timestamp = int(os.path.getctime(output_dir))
    
    item_data = {
        'job_id': job_id,
        'created_at': timestamp,
        'status': job_data.get('status', 'completed'),
        'image_count': len(input_images),
        'filenames': job_data.get('filenames', []),
        'files': {
            'obj': {
                'url': f'/api/download/{job_id}/mesh.obj',
                'size': os.path.getsize(obj_file) if os.path.exists(obj_file) else 0,
                'exists': os.path.exists(obj_file)
            },
            'stl': {
                'url': f'/api/download/{job_id}/mesh.stl',
                'size': os.path.getsize(stl_file) if os.path.exists(stl_file) else 0,
                'exists': os.path.exists(stl_file)
            },
            'video': {
                'url': f'/api/download/{job_id}/render.mp4',
                'size': os.path.getsize(video_file) if os.path.exists(video_file) else 0,
                'exists': os.path.exists(video_file)
            }
        },
        'input_images': input_images,
        'preview_images': preview_images,
        'render_frames': render_frames,
        'logs': job_data.get('logs', [])
    }
    
    return jsonify({
        'success': True,
        'data': item_data
    }), 200

@app.route('/api/delete/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """
    Delete a job and its associated files
    """
    with job_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        del jobs[job_id]
    
    # Clean up progress queue
    if job_id in progress_queues:
        del progress_queues[job_id]
    
    # Delete files
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    # Delete upload files
    upload_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            if filename.startswith(job_id):
                try:
                    os.remove(os.path.join(upload_dir, filename))
                except:
                    pass
    
    return jsonify({
        'success': True,
        'message': 'Job deleted successfully'
    }), 200

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 TripoSR API Server with Real-time Progress Tracking")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: stabilityai/TripoSR")
    print(f"API Base URL: http://0.0.0.0:5002/api")
    print("\n📡 Available Endpoints:")
    print("  GET    /api/health                    - Health check")
    print("  POST   /api/upload                    - Upload images (1-5)")
    print("  GET    /api/progress/<job_id>         - Real-time progress stream (SSE)")
    print("  GET    /api/status/<job_id>           - Get job status")
    print("  GET    /api/logs/<job_id>             - Get detailed logs")
    print("  GET    /api/download/<job_id>/<file>  - Download files")
    print("  GET    /api/preview/<job_id>          - Get preview images (base64)")
    print("  GET    /api/renders/<job_id>          - Get all 30 render frames")
    print("  GET    /api/gallery                   - Get all completed 3D models")
    print("  GET    /api/gallery/<job_id>          - Get detailed gallery item")
    print("  GET    /api/jobs                      - List all jobs")
    print("  DELETE /api/delete/<job_id>           - Delete job")
    print("\n📂 Output Files Per Job:")
    print("  • mesh.obj                - 3D model (OBJ format)")
    print("  • mesh.stl                - 3D model (STL format for 3D printing)")
    print("  • render.mp4              - 360° rotation video")
    print("  • render_000.png to render_029.png - Individual frames")
    print("  • preview_0.png to preview_7.png   - Preview images")
    print("  • input_0.png, input_1.png, ...    - Processed input images")
    print("\n🔧 Features:")
    print("  ✓ Multi-image support (up to 5 images)")
    print("  ✓ Real-time progress tracking via SSE")
    print("  ✓ Detailed logging with timestamps")
    print("  ✓ Background processing with threading")
    print("  ✓ Automatic STL conversion for 3D printing")
    print("  ✓ CORS enabled for mobile apps")
    print("=" * 70)
    
    app.run(debug=True, host="0.0.0.0", port=5002, threaded=True)