import json
import numpy as np
import random
import io
import torch
import re # Import regex for parsing gender

from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# --- 1. SETUP & MODEL LOADING ---
print("--- Starting Multimodal Outfit Completer Server (v3 - GENDER AWARE) ---")

# --- Configuration ---
METADATA_FILE = 'polyvore_item_metadata.json'
# We now use the COMBINED embeddings file
COMBINED_EMBEDDINGS_FILE = 'combined_embeddings.json' 
TEXT_MODEL_NAME = 'all-MiniLM-L6-v2' # For text part of upload
VISUAL_MODEL_NAME = 'clip-ViT-B-32'  # For visual part of upload

# --- Global Variables ---
app = Flask(__name__)
CORS(app) 

item_metadata = {}
combined_embeddings = {} # Holds ID -> Vector (list)
item_id_to_matrix_index = {}
all_item_vectors = np.array([])
text_model = None # This will hold the text model
clip_model = None # This will hold the visual (CLIP) model
device = 'cpu' 
CATEGORY_TEXT_VECTORS = {} # For sanity checks

# --- NEW GENDER-AWARE BUCKETS ---
# This is the fix for the gender bug.
# We will sort all item IDs by both category AND gender.
# Example: category_buckets['shoes_men'] = ['id1', 'id2']
category_buckets = {}

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

def get_gender_from_categories(category_list):
    """Parses the category list from metadata to find a gender."""
    for cat in category_list:
        c = cat.lower()
        if "men's" in c or "male" in c:
            return "men"
        if "women's" in c or "female" in c:
            return "women"
    return "unisex" # Default if no gender is specified

def load_all_data():
    global item_metadata, combined_embeddings, category_buckets
    global item_id_to_matrix_index, all_item_vectors, clip_model, text_model, CATEGORY_TEXT_VECTORS
    
    check_for_gpu()
    
    try:
        # 1. Load BOTH models
        print(f"Loading CLIP model '{VISUAL_MODEL_NAME}'...")
        clip_model = SentenceTransformer(VISUAL_MODEL_NAME, device=device)
        print(f"Loading Text model '{TEXT_MODEL_NAME}'...")
        text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
        print("Models loaded successfully.")

        # 2. Load metadata
        print(f"Loading metadata from '{METADATA_FILE}'...")
        with open(METADATA_FILE, 'r') as f:
            item_metadata = json.load(f)
        print(f"Loaded {len(item_metadata)} item details.")

        # 3. Load COMBINED embeddings
        print(f"Loading item embeddings (Multimodal) from '{COMBINED_EMBEDDINGS_FILE}'...")
        with open(COMBINED_EMBEDDINGS_FILE, 'r') as f:
            combined_embeddings = json.load(f)
        print(f"Loaded {len(combined_embeddings)} multimodal vectors.")

        # 4. Build GENDERED category buckets and vector matrix
        print("Building GENDERED category buckets for outfit completion...")
        
        temp_vectors = []
        current_index = 0
        # We iterate through the combined_embeddings, as this is our "high-quality" set
        for item_id, vector in combined_embeddings.items():
            
            # A) Add to vector matrix
            temp_vectors.append(np.array(vector, dtype=np.float32)) 
            item_id_to_matrix_index[item_id] = current_index
            current_index += 1
            
            if item_id not in item_metadata:
                continue 
            
            # B) Add to GENDERED bucket
            details = item_metadata[item_id]
            category = details.get('semantic_category', 'unknown').strip()
            
            # Find the gender from the full category list
            gender = get_gender_from_categories(details.get('categories', [])) # e.g., "men", "women", "unisex"
            
            # Create the gendered bucket key, e.g., "shoes_men"
            bucket_key = f"{category}_{gender}" 
            
            if bucket_key not in category_buckets:
                category_buckets[bucket_key] = []
            category_buckets[bucket_key].append(item_id)

            # Also add to a generic "unisex" bucket for that category
            unisex_bucket_key = f"{category}_unisex"
            if gender != "unisex" and unisex_bucket_key not in category_buckets:
                 category_buckets[unisex_bucket_key] = []
            if gender != "unisex":
                 category_buckets[unisex_bucket_key].append(item_id)


        all_item_vectors = np.array(temp_vectors)
        print(f"Created {len(category_buckets)} GENDERED category buckets.")
        print(f"Vector matrix shape: {all_item_vectors.shape}")

        # 5. Pre-compute category text vectors for sanity checks
        print("Pre-computing text vectors for category sanity checks...")
        with torch.no_grad():
            text_vectors = clip_model.encode(CATEGORY_TEXT_QUERIES, convert_to_tensor=True, device=device).to(torch.float32)
        
        for i, text_query in enumerate(CATEGORY_TEXT_QUERIES):
            category_name = CATEGORY_TEXT_MAP[text_query]
            CATEGORY_TEXT_VECTORS[category_name] = text_vectors[i]
        print("Category text vectors are ready.")
        
        print("\n--- Server is ready and 'Multimodal GENDER-AWARE Brain' is online. ---")

    except FileNotFoundError as e:
        print(f"\n--- !!! FATAL ERROR !!! ---\nCould not find a data file: {e.filename}")
        print(f"Please make sure '{METADATA_FILE}' and '{COMBINED_EMBEDDINGS_FILE}' are in this folder.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        import traceback
        traceback.print_exc()
        exit()

# --- 2. API ENDPOINTS ---

@app.route('/get_all_items', methods=['GET'])
def get_all_items():
    print("Request received for /get_all_items (multimodal catalog)")
    # We only show items that have combined embeddings
    valid_item_ids = list(combined_embeddings.keys())
    
    valid_item_ids_with_metadata = [item_id for item_id in valid_item_ids if item_id in item_metadata]
    sample_size = min(len(valid_item_ids_with_metadata), 50) # Still send 50
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
    print(f"Sending a random multimodal catalog of {len(items_list)} items.")
    return jsonify(items_list)


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations_from_catalog():
    data = request.json
    seed_item_id = data.get('seed_item_id')
    context_key = data.get('context_key', 'any')
    gender_key = data.get('gender_key', 'women') # Default to women if not provided
    
    print(f"\n--- Catalog Outfit Request ---")
    print(f"  > Seed Item: {seed_item_id}")
    print(f"  > Gender:    {gender_key}")

    try:
        seed_vector_np = all_item_vectors[item_id_to_matrix_index[seed_item_id]]
        seed_details = item_metadata[seed_item_id]
        seed_category = seed_details.get('semantic_category', 'unknown').strip()
        
        final_outfit = find_complete_outfit(seed_vector_np, seed_category, context_key, gender_key)
        
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
    gender_key = request.form.get('gender_key', 'women') # Get gender from form
    
    print(f"  > Gender: {gender_key}")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        uploaded_image = Image.open(io.BytesIO(image_bytes))

        # 1. Get Visual Vector for upload
        print("  > Encoding uploaded image (Visual)...")
        with torch.no_grad():
            visual_vector_tensor = clip_model.encode([uploaded_image], convert_to_tensor=True, device=device).to(torch.float32)
        
        # 2. Get Text Vector for upload (by guessing category)
        print("  > Guessing image category...")
        seed_category = guess_image_category(visual_vector_tensor)
        print(f"  > Guessed category: {seed_category}")
        
        # We use a generic text for the uploaded item
        upload_text = f"a photo of {seed_category}"
        print(f"  > Encoding text: '{upload_text}'...")
        with torch.no_grad():
            text_vector_tensor = text_model.encode([upload_text], convert_to_tensor=True, device=device).to(torch.float32)

        # 3. Create the "Super-Vector" for the upload
        combined_vector_np = np.concatenate([
            text_vector_tensor.cpu().numpy(),
            visual_vector_tensor.cpu().numpy()
        ], axis=1)

        # 4. Find the complete outfit
        final_outfit = find_complete_outfit(combined_vector_np, seed_category, context_key, gender_key)
        
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
    """Compares an image vector to our pre-computed text vectors."""
    text_vectors = torch.stack(list(CATEGORY_TEXT_VECTORS.values())).to(device)
    similarities = util.cos_sim(image_vector_tensor, text_vectors)
    similarity_scores = similarities[0].cpu().numpy()
    best_match_index = np.argmax(similarity_scores)
    best_category_name = list(CATEGORY_TEXT_VECTORS.keys())[best_match_index]
    return best_category_name


def find_complete_outfit(seed_vector_np, seed_category, context_key, gender_key):
    """
    Core "Outfit Completer" logic.
    NOW GENDER-AWARE.
    """
    print(f"  > Finding outfit for: [Category: {seed_category}], [Context: {context_key}], [Gender: {gender_key}]")
    
    required_slots = CONTEXT_RULES.get(context_key, [])
    final_outfit = []
    
    if seed_vector_np.ndim == 1:
        seed_vector_np = seed_vector_np.reshape(1, -1)
    
    seed_vector_np = seed_vector_np.astype(np.float32)

    for slot_category in required_slots:
        
        if slot_category == seed_category:
            continue 
        
        # --- NEW GENDER LOGIC ---
        # 1. Try to find the specific gender bucket (e.g., "shoes_men")
        gender_bucket_key = f"{slot_category}_{gender_key}"
        item_ids_in_bucket = category_buckets.get(gender_bucket_key)
        
        # 2. If "shoes_men" is empty or doesn't exist, fall back to "shoes_unisex"
        if not item_ids_in_bucket:
            print(f"  > No items in '{gender_bucket_key}', falling back to 'unisex' bucket...")
            unisex_bucket_key = f"{slot_category}_unisex"
            item_ids_in_bucket = category_buckets.get(unisex_bucket_key)
        # --- END GENDER LOGIC ---

        if not item_ids_in_bucket:
            print(f"  > Warning: No items found in bucket '{gender_bucket_key}' or '{unisex_bucket_key}'. Skipping.")
            continue
            
        valid_item_ids_in_bucket = [item_id for item_id in item_ids_in_bucket if item_id in item_id_to_matrix_index]
        if not valid_item_ids_in_bucket:
            print(f"  > Warning: Items in bucket but not in vector matrix. Skipping.")
            continue

        indices = [item_id_to_matrix_index[item_id] for item_id in valid_item_ids_in_bucket]
        bucket_vectors = all_item_vectors[indices]
        
        similarities = cosine_similarity(seed_vector_np, bucket_vectors)
        similarity_scores = similarities[0]
        
        # --- SANITY CHECK LOOP ---
        top_k_indices = np.argsort(similarity_scores)[-5:][::-1] 
        
        found_item = False
        for i in top_k_indices:
            best_item_id = valid_item_ids_in_bucket[i]
            best_item_score = similarity_scores[i]
            
            item_vector_np = all_item_vectors[item_id_to_matrix_index[best_item_id]]
            # Get the VISUAL part of the vector for the sanity check
            # Text vector is 384, Visual is 512. Visual is the *second* half.
            visual_part_of_vector = item_vector_np[384:] 
            
            item_vector_tensor = torch.tensor(visual_part_of_vector, dtype=torch.float32).to(device).reshape(1, -1)

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
                print(f"  > Discarding mislabeled item: {best_item_id}. Wanted '{slot_category}', but image looks like '{guessed_category}'.")
        
        if not found_item:
            print(f"  > Warning: Could not find a sane match for '{slot_category}' after checking top 5.")

    return final_outfit


# --- 4. RUN THE SERVER ---
if __name__ == '__main__':
    load_all_data()
    app.run(debug=True, port=5000)