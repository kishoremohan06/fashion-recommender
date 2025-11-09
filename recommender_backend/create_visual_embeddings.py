import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---

# 1. This is the ML model that understands images
MODEL_NAME = 'clip-ViT-B-32'

# 2. This is the path to your folder of 261,000+ images
#    We assume your backend and frontend folders are next to each other:
#    RS_Project/
#    ├── recommender_backend/  <-- (This script is in here)
#    └── recommender_frontend/
#        └── images/           <-- (This is the folder it reads)
IMAGE_FOLDER_PATH = '../recommender_frontend/images/'

# 3. This is the data file we will create
OUTPUT_FILE = 'visual_embeddings.json'

# --- END CONFIGURATION ---

def check_for_gpu():
    """Checks if a CUDA-compatible GPU is available for PyTorch."""
    if torch.cuda.is_available():
        print("--- 🚀 SUCCESS: GPU DETECTED! ---")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        print("--- 🐢 WARNING: No GPU detected. ---")
        print("Using CPU. This will be very, very slow (potentially hours).")
        return 'cpu'

def create_visual_embeddings():
    """
    Loops through all images, runs them through the CLIP model,
    and saves the resulting vectors to a JSON file.
    """
    
    device = check_for_gpu()
    
    # 1. Load the CLIP model onto the GPU (if available)
    print(f"\nLoading CLIP model '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This often happens if your internet connection is down.")
        print("Or if you are having CUDA driver issues (like the WinError 1114).")
        print("If you get the 'WinError 1114', see our previous chat for the fix.")
        return

    print("Model loaded successfully.")

    # 2. Find all images to process
    print(f"Scanning for images in '{IMAGE_FOLDER_PATH}'...")
    try:
        image_files = [f for f in os.listdir(IMAGE_FOLDER_PATH) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print(f"--- ERROR: No images found in '{IMAGE_FOLDER_PATH}' ---")
            print("Please check the IMAGE_FOLDER_PATH variable in this script.")
            return
    except FileNotFoundError:
        print(f"--- ERROR: Image folder not found at '{IMAGE_FOLDER_PATH}' ---")
        print("Please check the IMAGE_FOLDER_PATH variable in this script.")
        print("Make sure your 'recommender_frontend/images' folder exists.")
        return
        
    print(f"Found {len(image_files)} images to process.")

    # 3. Process all images and get their vectors
    # We will process in batches for efficiency
    batch_size = 32  # You can lower this if you run out of GPU memory
    all_embeddings = {}

    print(f"Starting embedding generation (Batch size: {batch_size})...")
    
    # Use tqdm for a live progress bar
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing Batches"):
        
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_item_ids = []

        for file_name in batch_files:
            try:
                # Get the item ID from the file name (e.g., "12345.jpg" -> "12345")
                item_id = os.path.splitext(file_name)[0]
                image_path = os.path.join(IMAGE_FOLDER_PATH, file_name)
                
                # Open the image and add it to our batch
                img = Image.open(image_path)
                batch_images.append(img)
                batch_item_ids.append(item_id)
            
            except Exception as e:
                # This catches corrupted or unreadable images
                print(f"\nWarning: Could not process image '{file_name}'. Error: {e}")
                continue
        
        if not batch_images:
            continue

        # --- This is the ML magic ---
        # The model processes the entire batch of images at once
        try:
            with torch.no_grad(): # Disables gradient calculations, saving memory/speed
                vectors = model.encode(batch_images, convert_to_tensor=True, device=device)
            
            # Convert vectors to simple Python lists for JSON saving
            vectors_list = vectors.cpu().numpy().tolist()
            
            # Add the vectors to our main dictionary
            for item_id, vector in zip(batch_item_ids, vectors_list):
                all_embeddings[item_id] = vector
                
        except Exception as e:
            print(f"\n--- ERROR during batch {i} encoding ---")
            print(f"This can happen if you run out of GPU memory.")
            print(f"Try lowering the 'batch_size' variable in this script (e.g., to 16 or 8).")
            print(f"Error details: {e}")
            continue # Skip this batch

    # 4. Save the final file
    print("\nEmbedding generation complete!")
    print(f"Successfully processed {len(all_embeddings)} images.")
    
    print(f"Saving all {len(all_embeddings)} vectors to '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_embeddings, f)
        print("--- Save successful! ---")
        print(f"\nYou are now ready to run 'python server_visual.py'")
        
    except Exception as e:
        print(f"--- ERROR saving file: {e} ---")

if __name__ == "__main__":
    create_visual_embeddings()