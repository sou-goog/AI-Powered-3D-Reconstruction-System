#!/usr/bin/env python3
"""
Test script to verify STL generation functionality
"""

import os
import time
import torch
from PIL import Image
import trimesh
import tempfile
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import rembg

def test_stl_generation():
    """Test the complete pipeline including STL generation"""
    print("🧪 Testing STL Generation Pipeline")
    print("=" * 50)
    
    # Test parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print("📦 Loading TSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    print("✅ Model loaded successfully")
    
    # Test with an example image
    test_image_path = "examples/chair.png"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
        
    print(f"🖼️  Loading test image: {test_image_path}")
    
    # Process image
    original_image = Image.open(test_image_path)
    resized_image = original_image.resize((512, 512))
    
    # Remove background
    print("🎭 Removing background...")
    rembg_session = rembg.new_session()
    image = remove_background(resized_image, rembg_session)
    print("✨ Background removed")
    
    # Resize foreground
    image = resize_foreground(image, ratio=0.85)
    
    # Convert RGBA to RGB if needed
    if image.mode == "RGBA":
        import numpy as np
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Make square
    w, h = image.size
    if w != h:
        from PIL import ImageOps
        max_side = max(w, h)
        delta_w = max_side - w
        delta_h = max_side - h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w//2, delta_h - delta_h//2)
        image = ImageOps.expand(image, padding, fill=(255, 255, 255))
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Save processed image
        processed_image_path = os.path.join(temp_dir, "input.png")
        image.save(processed_image_path)
        print("💾 Processed image saved")
        
        # Run TSR model
        print("🧠 Running TSR model...")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        print("🎯 3D scene generation completed")
        
        # Extract mesh
        print("🏗️  Extracting 3D mesh...")
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
        
        # Export OBJ
        obj_path = os.path.join(temp_dir, "mesh.obj")
        meshes[0].export(obj_path)
        print("📦 OBJ file exported")
        
        # Test STL conversion with trimesh
        print("🔄 Converting to STL format...")
        stl_path = os.path.join(temp_dir, "mesh.stl")
        
        try:
            # Load OBJ with trimesh
            mesh_trimesh = trimesh.load(obj_path)
            print(f"📊 Mesh info: {len(mesh_trimesh.vertices)} vertices, {len(mesh_trimesh.faces)} faces")
            
            # Export as STL
            mesh_trimesh.export(stl_path)
            
            # Verify STL file
            if os.path.exists(stl_path) and os.path.getsize(stl_path) > 0:
                file_size = os.path.getsize(stl_path)
                print(f"✅ STL file created successfully!")
                print(f"📏 STL file size: {file_size:,} bytes")
                print(f"📍 STL file location: {stl_path}")
                
                # Additional validation: try to load the STL back
                try:
                    stl_mesh = trimesh.load(stl_path)
                    print(f"✅ STL validation: {len(stl_mesh.vertices)} vertices, {len(stl_mesh.faces)} faces")
                    return True
                except Exception as e:
                    print(f"⚠️  STL validation failed: {e}")
                    return False
            else:
                print("❌ STL file creation failed - file is empty or missing")
                return False
                
        except Exception as e:
            print(f"❌ STL conversion failed: {e}")
            return False

if __name__ == "__main__":
    success = test_stl_generation()
    if success:
        print("\n🎉 STL generation test PASSED!")
        print("✨ The Flask app should now generate STL files correctly")
    else:
        print("\n❌ STL generation test FAILED!")
        print("🔧 Please check the error messages above")