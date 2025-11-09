import json
import numpy as np
import random
import io
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# --- 1. SETUP & MODEL LOADING ---
print("--- Starting Visual Outfit Completer Server (v4 - DType Fix) ---")

# --- Configuration ---
METADATA_FILE = 'polyvore_item_metadata.json'
EMBEDDINGS_FILE = 'visual_embeddings.json'
MODEL_NAME = 'clip-ViT-B-32'

# --- Global Variables ---
app = Flask(__name__)
CORS(app) 
item_metadata = {}
item_embeddings = {} 
category_buckets = {} 
item_id_to_matrix_index = {}
all_item_vectors = np.array([])
clip_model = None
device = 'cpu' 
CATEGORY_TEXT_VECTORS = {} # Will hold our pre-computed text vectors

# --- Context Rules ---
CONTEXT_RULES = {
    "any": ["tops", "bottoms", "shoes", "bags", "outerwear"],
    "job_interview": ["tops", "bottoms", "shoes", "bags", "outerwear"],
    "casual_weekend": ["tops", "bottoms", "shoes", "bags", "sunglasses", "hats"],
    "party": ["all-body", "shoes", "bags", "jewellery"],
    "sporty": ["tops", "bottoms", "shoes", "hats"]
}

# --- Category Guesser Config ---
CATEGORY_TEXT_QUERIES = [
    "a photo of a top, like a t-shirt, blouse, or shirt",
    "a photo of bottoms, like pants, jeans, or a skirt",
    "a photo of shoes, like sneakers, heels, or boots",
    "a photo of outerwear, like a jacket or coat",
    "a photo of an all-body outfit, like a dress or jumpsuit",
    "a photo of a bag or purse",
    "a photo of sunglasses",
    "a photo of a hat or cap",
    "a photo of jewellery, like a necklace or earrings"
]
CATEGORY_TEXT_MAP = {
    "a photo of a top, like a t-shirt, blouse, or shirt": "tops",
    "a photo of bottoms, like pants, jeans, or a skirt": "bottoms",
    "a photo of shoes, like sneakers, heels, or boots": "shoes",
    "a photo of outerwear, like a jacket or coat": "outerwear",
    "a photo of an all-body outfit, like a dress or jumpsuit": "all-body",
    "a photo of a bag or purse": "bags",
    "a photo of sunglasses": "sunglasses",
    "a photo of a hat or cap": "hats",
    "a photo of jewellery, like a necklace or earrings": "jewellery"
}

def check_for_gpu():
    global device
    if torch.cuda.is_available():
        print(f"--- 🚀 SUCCESS: GPU DETECTED! Using device: {torch.cuda.get_device_name(0)} ---")
        device = 'cuda'
    else:
        print("--- 🐢 WARNING: No GPU detected. Using CPU. ---")
        device = 'cpu'

def load_all_data():
    global item_metadata, item_embeddings, category_buckets
    global item_id_to_matrix_index, all_item_vectors, clip_model, CATEGORY_TEXT_VECTORS
    
    check_for_gpu()
    
    try:
        # 1. Load CLIP model
        print(f"Loading CLIP model '{MODEL_NAME}' into memory...")
        clip_model = SentenceTransformer(MODEL_NAME, device=device)
        print("CLIP model loaded successfully.")

        # 2. Load metadata
        print(f"Loading metadata from '{METADATA_FILE}'...")
        with open(METADATA_FILE, 'r') as f:
            item_metadata = json.load(f)
        print(f"Loaded {len(item_metadata)} item details.")

        # 3. Load visual embeddings
        print(f"Loading item embeddings (Visual ML model) from '{EMBEDDINGS_FILE}'...")
        with open(EMBEDDINGS_FILE, 'r') as f:
            item_embeddings = json.load(f)
        print(f"Loaded {len(item_embeddings)} visual ML vectors.")

        # 4. Build buckets and vector matrix
        print("Building category buckets for outfit completion...")
        
        temp_vectors = []
        current_index = 0
        for item_id, vector in item_embeddings.items():
            # We explicitly cast to float32 here to prevent dtype errors later
            temp_vectors.append(np.array(vector, dtype=np.float32)) 
            item_id_to_matrix_index[item_id] = current_index
            
            if item_id in item_metadata:
                details = item_metadata[item_id]
                category = details.get('semantic_category', 'unknown').strip() 
                if category not in category_buckets:
                    category_buckets[category] = []
                category_buckets[category].append(item_id)
            
            current_index += 1
            
        all_item_vectors = np.array(temp_vectors)
        print(f"Created {len(category_buckets)} category buckets.")
        print(f"Vector matrix shape: {all_item_vectors.shape}")

        # 5. Pre-compute category text vectors
        print("Pre-computing text vectors for category sanity checks...")
        with torch.no_grad():
            # Encode and ensure they are float32
            text_vectors = clip_model.encode(CATEGORY_TEXT_QUERIES, convert_to_tensor=True, device=device).to(torch.float32)
        
        for i, text_query in enumerate(CATEGORY_TEXT_QUERIES):
            category_name = CATEGORY_TEXT_MAP[text_query]
            CATEGORY_TEXT_VECTORS[category_name] = text_vectors[i]
        print("Category text vectors are ready.")
        
        print("\n--- Server is ready and 'VISUAL Brain' is online. ---")

    except FileNotFoundError as e:
        print(f"\n--- !!! FATAL ERROR !!! ---\nCould not find a data file: {e.filename}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        import traceback
        traceback.print_exc()
        exit()

# --- 2. API ENDPOINTS ---

@app.route('/get_all_items', methods=['GET'])
def get_all_items():
    print("Request received for /get_all_items (visual catalog)")
    valid_item_ids = list(item_embeddings.keys())
    valid_item_ids_with_metadata = [item_id for item_id in valid_item_ids if item_id in item_metadata]
    sample_size = min(len(valid_item_ids_with_metadata), 50)
    if sample_size == 0:
        return jsonify([])
    random_item_ids = random.sample(valid_item_ids_with_metadata, sample_size)
    items_list = []
    for item_id in random_item_ids:
        details = item_metadata[item_id]
        items_list.append({
            "id": item_id,
            "title": details.get('title', 'Untitled'),
            "category": details.get('semantic_category', 'unknown').strip()
        })
    print(f"Sending a random visual catalog of {len(items_list)} items.")
    return jsonify(items_list)


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations_from_catalog():
    data = request.json
    seed_item_id = data.get('seed_item_id')
    context_key = data.get('context_key', 'any')
    print(f"\n--- Catalog Outfit Request ---")
    print(f"  > Seed Item: {seed_item_id}")
    try:
        # We load the vector as a float32 numpy array
        seed_vector_np = all_item_vectors[item_id_to_matrix_index[seed_item_id]]

        seed_details = item_metadata[seed_item_id]
        seed_category = seed_details.get('semantic_category', 'unknown').strip()
        
        final_outfit = find_complete_outfit(seed_vector_np, seed_category, context_key)
        
        seed_item_data = {
            "id": seed_item_id,
            "title": seed_details.get('title', 'Untitled'),
            "category": seed_category,
            "image_url": f"images/{seed_item_id}.jpg"
        }
        final_outfit_with_seed = [seed_item_data] + final_outfit
        print(f"  > Found {len(final_outfit)} items. Sending full outfit.")
        return jsonify(final_outfit_with_seed)
        
    except KeyError as e:
        print(f"Error: Seed item '{seed_item_id}' not in embeddings or metadata. {e}")
        return jsonify({"error": f"Seed item not found: {seed_item_id}"}), 404
    except Exception as e:
        print(f"\n--- !!! UNEXPECTED SERVER ERROR !!! ---")
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc() 
        print(f"---------------------------------------\n")
        return jsonify({"error": "Error finding outfit"}), 500


@app.route('/recommend_by_upload', methods=['POST'])
def recommend_by_upload():
    print("\n--- Image Upload Outfit Request ---")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    context_key = request.form.get('context_key', 'any')
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image_bytes = file.read()
        uploaded_image = Image.open(io.BytesIO(image_bytes))
        print("  > Encoding uploaded image...")
        with torch.no_grad():
            # Ensure the output is float32
            seed_vector_tensor = clip_model.encode([uploaded_image], convert_to_tensor=True, device=device).to(torch.float32)
        
        print("  > Guessing image category...")
        seed_category = guess_image_category(seed_vector_tensor)
        print(f"  > Guessed category: {seed_category}")

        final_outfit = find_complete_outfit(seed_vector_tensor.cpu().numpy(), seed_category, context_key)
        
        print(f"  > Found {len(final_outfit)} items. Sending outfit.")
        return jsonify(final_outfit)

    except Exception as e:
        print(f"\n--- !!! UNEXPECTED SERVER ERROR !!! ---")
        print(f"Error processing upload: {e}")
        import traceback
        traceback.print_exc()
        print(f"---------------------------------------\n")
        return jsonify({"error": "Error processing uploaded image"}), 500

# --- 3. CORE LOGIC (THE "BRAIN") ---

def guess_image_category(image_vector_tensor):
    """
    Compares an image vector to our pre-computed text vectors
    to find the most likely category.
    """
    # text_vectors is already float32 from load_all_data
    text_vectors = torch.stack(list(CATEGORY_TEXT_VECTORS.values())).to(device)
    
    # image_vector_tensor is now also float32
    similarities = util.cos_sim(image_vector_tensor, text_vectors)
    similarity_scores = similarities[0].cpu().numpy()

    best_match_index = np.argmax(similarity_scores)
    best_category_name = list(CATEGORY_TEXT_VECTORS.keys())[best_match_index]
    
    return best_category_name


def find_complete_outfit(seed_vector_np, seed_category, context_key):
    """
    This is the core "Outfit Completer" logic.
    It now includes a "SANITY CHECK" to filter mislabeled items.
    """
    print(f"  > Finding outfit for seed category '{seed_category}' and context '{context_key}'")
    
    required_slots = CONTEXT_RULES.get(context_key, [])
    final_outfit = []
    
    if seed_vector_np.ndim == 1:
        seed_vector_np = seed_vector_np.reshape(1, -1)
    
    # Ensure seed_vector is float32, just in case
    seed_vector_np = seed_vector_np.astype(np.float32)

    for slot_category in required_slots:
        
        if slot_category == seed_category:
            continue 
            
        item_ids_in_bucket = category_buckets.get(slot_category)
        
        if not item_ids_in_bucket:
            print(f"  > Warning: No items found in bucket '{slot_category}'. Skipping.")
            continue
            
        valid_item_ids_in_bucket = [item_id for item_id in item_ids_in_bucket if item_id in item_id_to_matrix_index]
        if not valid_item_ids_in_bucket:
            print(f"  > Warning: Items in bucket '{slot_category}' but not in vector matrix. Skipping.")
            continue

        indices = [item_id_to_matrix_index[item_id] for item_id in valid_item_ids_in_bucket]
        # all_item_vectors is already float32 from load_all_data
        bucket_vectors = all_item_vectors[indices]
        
        # This is now float32 vs float32
        similarities = cosine_similarity(seed_vector_np, bucket_vectors)
        similarity_scores = similarities[0]
        
        # --- NEW SANITY CHECK LOOP ---
        top_k_indices = np.argsort(similarity_scores)[-5:][::-1] 
        
        found_item = False
        for i in top_k_indices:
            best_item_id = valid_item_ids_in_bucket[i]
            best_item_score = similarity_scores[i]
            
            # --- THIS IS THE SANITY CHECK ---
            # Get the vector, which is already float32
            item_vector_np = all_item_vectors[item_id_to_matrix_index[best_item_id]]
            
            # --- THIS IS THE FIX ---
            # Convert the numpy array to a float32 tensor
            item_vector_tensor = torch.tensor(item_vector_np, dtype=torch.float32).to(device).reshape(1, -1)
            # --- END OF FIX ---

            guessed_category = guess_image_category(item_vector_tensor)
            
            if guessed_category == slot_category:
                # SUCCESS!
                print(f"  > Found best '{slot_category}': {best_item_id} (Score: {best_item_score:.4f})")
                details = item_metadata.get(best_item_id)
                final_outfit.append({
                    "id": best_item_id,
                    "title": details.get('title', 'Untitled'),
                    "category": slot_category,
                    "image_url": f"images/{best_item_id}.jpg",
                    "score": float(best_item_score)
                })
                found_item = True
                break 
            else:
                # FAILURE!
                print(f"  > Discarding mislabeled item: {best_item_id}. Wanted '{slot_category}', but image looks like '{guessed_category}'.")
        
        if not found_item:
            print(f"  > Warning: Could not find a sane match for '{slot_category}' after checking top 5.")

    return final_outfit


# --- 4. RUN THE SERVER ---
if __name__ == '__main__':
    load_all_data()
    app.run(debug=True, port=5000)