import json
from sentence_transformers import SentenceTransformer
import time
import torch  # --- NEW: Import PyTorch ---

# --- NEW: Function to check for GPU ---
def check_gpu():
    """
    Checks if PyTorch can access a CUDA-enabled GPU.
    """
    print("--- Checking for GPU ---")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"--- \N{ROCKET} SUCCESS: GPU DETECTED! ---")
        print(f"Using GPU: {device_name}")
        print("--- This will be MUCH faster. ---")
        return "cuda"  # Use the GPU
    else:
        print("--- \N{TURTLE} WARNING: No GPU detected. ---")
        print("Using CPU. This will be very, very slow.")
        print("(This is normal if you don't have an NVIDIA GPU or if PyTorch/CUDA is not set up correctly.)")
        print("-------------------------------")
        return "cpu"  # Use the CPU

# --- CONFIGURATION ---
METADATA_FILE = 'polyvore_item_metadata.json'
OUTPUT_FILE = 'item_embeddings.json'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- 0. CHECK FOR GPU FIRST ---
# Call our new function to get the device (e.g., "cuda" or "cpu")
device_to_use = check_gpu()

# --- 1. LOAD THE PRE-TRAINED ML MODEL ---
print(f"Loading ML model '{MODEL_NAME}'... (This may take a moment)")
# This will download the model the first time you run it.
# We now tell the model to use the device we found.
model = SentenceTransformer(MODEL_NAME, device=device_to_use)
print("Model loaded.")

# --- 2. LOAD THE ITEM METADATA ---
print(f"Loading metadata from {METADATA_FILE}...")
try:
    with open(METADATA_FILE, 'r') as f:
        item_metadata = json.load(f)
    print(f"Loaded {len(item_metadata)} items.")
except FileNotFoundError:
    print(f"ERROR: Could not find {METADATA_FILE}.")
    print("Please make sure it's in the same directory.")
    exit()

# --- 3. CREATE EMBEDDINGS ---
print("Starting embedding generation... This will take some time.")
item_embeddings = {}
start_time = time.time()
count = 0

# Loop through every single item in our metadata
for item_id, details in item_metadata.items():
    title = details.get('title', '')
    description = details.get('description', '')
    
    if not title and not description:
        continue

    text_to_embed = f"{title} {description}"
    
    # Use the ML model to convert this text into a vector
    # The model is already on the correct device (GPU or CPU)
    vector = model.encode(text_to_embed).tolist()
    
    item_embeddings[item_id] = vector
    
    count += 1
    if count % 1000 == 0:
        print(f"  ...processed {count} / {len(item_metadata)} items")

end_time = time.time()
print(f"Embedding generation complete! Took {end_time - start_time:.2f} seconds.")

# --- 4. SAVE THE EMBEDDINGS TO A FILE ---
print(f"Saving {len(item_embeddings)} embeddings to {OUTPUT_FILE}...")
try:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(item_embeddings, f)
    print("Save successful!")
    print("\n--- You are now ready to run 'server.py'! ---")
except IOError as e:
    print(f"ERROR: Could not write to file {OUTPUT_FILE}. Error: {e}")