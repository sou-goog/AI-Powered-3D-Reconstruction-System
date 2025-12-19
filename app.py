from flask import Flask, request, render_template, send_from_directory, redirect, url_for, Response, session
import os
import torch
from PIL import Image, ImageOps
import numpy as np
import time
import rembg
import json
import threading
from queue import Queue

# Import trimesh for STL conversion
import trimesh


from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video

app = Flask(__name__)
app.secret_key = 'triposr-secret-key-2024'  # Required for sessions
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['OUTPUT_FOLDER'] = "output"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global progress tracking
progress_queues = {}
processing_status = {}

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once on startup
print("Initializing TSR model...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)
print("Model loaded.")

# Timer class with progress tracking
class Timer:
    def __init__(self, session_id=None):
        self.items = {}
        self.time_scale = 1000.0
        self.time_unit = "ms"
        self.session_id = session_id
        
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

timer = Timer()

def process_image_async(upload_path, session_id):
    """Process image in background thread with progress updates"""
    timer = Timer(session_id)
    
    try:
        timer.log_progress("📁 Processing uploaded image...")
        
        # Open and resize
        timer.log_progress("🖼️ Loading and resizing image...")
        original_image = Image.open(upload_path)
        resized_image = original_image.resize((512, 512))
        timer.log_progress(f"📐 Image resized to 512x512 pixels")

        # TSR processing
        timer.start("Processing image")
        timer.log_progress("🎭 Removing background...")
        rembg_session = rembg.new_session()
        image = remove_background(resized_image, rembg_session)
        timer.log_progress("✨ Background removed successfully")
        
        timer.log_progress("🔄 Resizing foreground...")
        image = resize_foreground(image, ratio=0.85)

        if image.mode == "RGBA":
            timer.log_progress("🎨 Converting RGBA to RGB...")
            image = np.array(image).astype(np.float32)/255.0
            image = image[:, :, :3]*image[:, :, 3:4] + (1-image[:, :, 3:4])*0.5
            image = Image.fromarray((image*255).astype(np.uint8))

        # Make square
        w, h = image.size
        if w != h:
            timer.log_progress("⬜ Making image square...")
            max_side = max(w, h)
            delta_w = max_side - w
            delta_h = max_side - h
            padding = (delta_w // 2, delta_h // 2, delta_w - delta_w//2, delta_h - delta_h//2)
            image = ImageOps.expand(image, padding, fill=(255,255,255))

        # Unique output folder
        folder_id = str(int(time.time()))
        image_dir = os.path.join(app.config['OUTPUT_FOLDER'], folder_id)
        os.makedirs(image_dir, exist_ok=True)
        image.save(os.path.join(image_dir, "input.png"))
        timer.log_progress(f"💾 Processed image saved to folder: {folder_id}")
        timer.end("Processing image")

        # Run TSR model
        timer.start("Running model")
        timer.log_progress("🧠 Initializing AI neural network...")
        timer.log_progress("🔮 Generating 3D scene codes...")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        timer.log_progress("🎯 3D scene generation completed!")
        timer.end("Running model")

        # Render video
        timer.start("Rendering")
        timer.log_progress("🎬 Starting 3D rendering process...")
        timer.log_progress("📹 Rendering 30 camera views...")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        
        timer.log_progress("🎞️ Creating MP4 video...")
        save_video(render_images[0], os.path.join(image_dir, "render.mp4"), fps=30)
        
        timer.log_progress("🖼️ Saving individual frames...")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(image_dir, f"render_{ri:03d}.png"))
            if ri % 10 == 0:  # Update every 10 frames
                timer.log_progress(f"📸 Saved frame {ri+1}/30")
        timer.end("Rendering")

        # Export mesh
        timer.start("Exporting mesh")
        timer.log_progress("🏗️ Extracting 3D mesh geometry...")
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
        mesh_file = os.path.join(image_dir, "mesh.obj")
        meshes[0].export(mesh_file)
        timer.log_progress("📦 OBJ file exported successfully")

        # Convert to STL using trimesh
        stl_file = os.path.join(image_dir, "mesh.stl")
        try:
            timer.log_progress("🔄 Converting to STL format...")
            # Load the OBJ file with trimesh
            mesh_trimesh = trimesh.load(mesh_file)
            # Export as STL
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

        # Mark as completed
        processing_status[session_id] = {
            'status': 'completed',
            'folder_id': folder_id,
            'message': '🎉 3D model generation completed successfully!'
        }
        timer.log_progress("🎉 All processing completed! Redirecting to results...")

    except Exception as e:
        error_msg = f"❌ Error during processing: {str(e)}"
        timer.log_progress(error_msg)
        processing_status[session_id] = {
            'status': 'error',
            'message': error_msg
        }

@app.route("/progress/<session_id>")
def progress_stream(session_id):
    """Server-Sent Events endpoint for progress updates"""
    def generate():
        print(f"SSE connection opened for session: {session_id}")
        
        if session_id not in progress_queues:
            print(f"Session {session_id} not found in progress_queues")
            yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
            return
            
        queue = progress_queues[session_id]
        
        # Send initial connection confirmation
        yield f"data: {json.dumps({'connected': True, 'session_id': session_id})}\n\n"
        
        while True:
            try:
                # Check if processing is done
                if session_id in processing_status:
                    status = processing_status[session_id]
                    if status['status'] in ['completed', 'error']:
                        print(f"Processing finished for session {session_id}: {status['status']}")
                        yield f"data: {json.dumps(status)}\n\n"
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
        print(f"Cleaning up session: {session_id}")
        if session_id in progress_queues:
            del progress_queues[session_id]
        if session_id in processing_status:
            del processing_status[session_id]
    
    return Response(generate(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'})

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"

        # Generate session ID
        session_id = str(int(time.time() * 1000))  # Unique session ID
        session['current_session'] = session_id
        
        # Get filename
        filename = file.filename
        
        # Setup progress tracking
        progress_queues[session_id] = Queue(maxsize=100)
        processing_status[session_id] = {'status': 'starting'}
        
        # Add immediate progress update
        initial_timer = Timer(session_id)
        initial_timer.log_progress("🌟 Welcome to 3D Reconstruction Studio!")
        initial_timer.log_progress(f"📁 File uploaded: {filename}")
        initial_timer.log_progress("🔄 Starting background processing...")

        # Save uploaded image
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Start background processing
        thread = threading.Thread(target=process_image_async, args=(upload_path, session_id))
        thread.daemon = True
        thread.start()

        # Redirect to processing page
        return redirect(url_for('processing', session_id=session_id))

    return render_template("index.html")

@app.route("/processing/<session_id>")
def processing(session_id):
    """Processing page with real-time terminal interface"""
    return render_template("processing.html", session_id=session_id)

@app.route("/result/<folder>")
def result(folder):
    # Get list of all output folders for gallery
    output_folders = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        for item in os.listdir(app.config['OUTPUT_FOLDER']):
            folder_path = os.path.join(app.config['OUTPUT_FOLDER'], item)
            if os.path.isdir(folder_path) and item != folder:
                # Check if required files exist
                if (os.path.exists(os.path.join(folder_path, "input.png")) and 
                    os.path.exists(os.path.join(folder_path, "mesh.obj")) and
                    os.path.exists(os.path.join(folder_path, "render.mp4"))):
                    output_folders.append(item)
    
    # Sort by timestamp (newest first)
    output_folders.sort(reverse=True)
    
    return render_template(
        "result.html",
        folder=folder,
        video_file="render.mp4",
        obj_file="mesh.obj",
        stl_file="mesh.stl",
        previous_outputs=output_folders[:10]  # Limit to 10 most recent
    )

@app.route("/output/<folder>/<filename>")
def output_files(folder, filename):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], folder), filename)

@app.route("/gallery")
def gallery():
    """Gallery view of all previous outputs"""
    output_folders = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        for item in os.listdir(app.config['OUTPUT_FOLDER']):
            folder_path = os.path.join(app.config['OUTPUT_FOLDER'], item)
            if os.path.isdir(folder_path):
                # Check if required files exist
                if (os.path.exists(os.path.join(folder_path, "input.png")) and 
                    os.path.exists(os.path.join(folder_path, "mesh.obj")) and
                    os.path.exists(os.path.join(folder_path, "render.mp4"))):
                    output_folders.append({
                        'id': item,
                        'timestamp': int(item) if item.isdigit() else 0
                    })
    
    # Sort by timestamp (newest first)
    output_folders.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template("gallery.html", outputs=output_folders)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
