import json
import numpy as np
import random
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP ---
app = Flask(__name__)
CORS(app)  # Allow our frontend to call this server

print("--- Starting Outfit Completer Server ---")

# --- 2. LOAD OUR DATA (THE "BRAIN'S MEMORY") ---
METADATA_FILE = 'polyvore_item_metadata.json'
EMBEDDINGS_FILE = 'item_embeddings.json' # Using the TEXT embeddings you created
item_metadata = {}
item_embeddings = {} # Holds ID -> Vector (list)

# --- NEW: Category Buckets ---
# This is the key to the new logic. We sort all items by category first.
# It will look like:
# {
#   "tops": ["id1", "id2", ...],
#   "bottoms": ["id3", "id4", ...]
# }
category_buckets = {}

# We also store all vectors in a big matrix for fast similarity search
# This maps an item_id to its row index in the matrix
item_id_to_matrix_index = {}
all_item_vectors = []

try:
    print(f"Loading metadata from '{METADATA_FILE}'...")
    with open(METADATA_FILE, 'r') as f:
        item_metadata = json.load(f)
    print(f"Loaded {len(item_metadata)} item details.")

    print(f"Loading item embeddings (ML model) from '{EMBEDDINGS_FILE}'...")
    with open(EMBEDDINGS_FILE, 'r') as f:
        item_embeddings = json.load(f) # This is a dict: { "id": [vector], ... }
    print(f"Loaded {len(item_embeddings)} ML vectors.")

    # --- NEW: Build the category "buckets" and vector matrix ---
    print("Building category buckets for outfit completion...")
    
    current_index = 0
    for item_id, vector in item_embeddings.items():
        # 1. Store the vector and its index
        all_item_vectors.append(vector)
        item_id_to_matrix_index[item_id] = current_index
        current_index += 1

        # 2. Add the item_id to its category bucket
        if item_id not in item_metadata:
            continue # Skip if item has no metadata
            
        details = item_metadata[item_id]
        # .strip() is important to fix "jewellery "
        category = details.get('semantic_category', 'unknown').strip() 
        
        if category not in category_buckets:
            category_buckets[category] = []
        category_buckets[category].append(item_id)
        
    # Convert the vector list into a giant, fast NumPy matrix
    all_item_vectors = np.array(all_item_vectors)
    
    print(f"Created {len(category_buckets)} category buckets.")
    print("\n--- Server is ready and 'Outfit Completer Brain' is online. ---")

except FileNotFoundError:
    print("\n--- !!! FATAL ERROR !!! ---")
    print(f"Could not find data files. Make sure '{METADATA_FILE}' and '{EMBEDDINGS_FILE}' are in the same folder as 'server.py'.")
    exit()
except Exception as e:
    print(f"An error occurred during startup: {e}")
    exit()


# --- 3. THE "BRAIN" API ENDPOINTS ---

@app.route('/get_all_items', methods=['GET'])
def get_all_items():
    """
    Called by the frontend ONCE on page load to populate the dropdown.
    This sends 50 random items for speed.
    """
    print("Request received for /get_all_items")
    
    # Get items that have an embedding
    valid_item_ids = list(item_embeddings.keys())
    
    sample_size = min(len(valid_item_ids), 50)
    random_item_ids = random.sample(valid_item_ids, sample_size)
    
    items_list = []
    for item_id in random_item_ids:
        if item_id in item_metadata:
            details = item_metadata[item_id]
            items_list.append({
                "id": item_id,
                "title": details.get('title', 'Untitled'),
                "category": details.get('semantic_category', 'unknown').strip()
            })

    print(f"Sending a random catalog of {len(items_list)} items.")
    return jsonify(items_list)


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """
    Called by the frontend when the user clicks "Build Outfit".
    This is the NEW "outfit completer" logic.
    """
    data = request.json
    seed_item_id = data.get('seed_item_id')
    context_key = data.get('context_key', 'any')

    print(f"\nOutfit Completion request:")
    print(f"  > Seed Item: {seed_item_id}")
    print(f"  > Context:     {context_key}")

    # --- a) Find the Seed Item's Vector & Category ---
    try:
        seed_vector = item_embeddings[seed_item_id]
        seed_details = item_metadata[seed_item_id]
        seed_category = seed_details.get('semantic_category', 'unknown').strip()
        # Reshape for scikit-learn: from (384,) to (1, 384)
        seed_vector_np = np.array(seed_vector).reshape(1, -1)
    except KeyError:
        print(f"Error: Seed item {seed_item_id} not in embeddings or metadata.")
        return jsonify({"error": "Seed item not found"}), 404

    # --- b) Find the "Slots" We Need to Fill ---
    # Get the list of required categories for this context
    required_slots = CONTEXT_RULES.get(context_key, [])
    
    final_outfit = [] # This will hold our final [top, bottom, shoes, ...]
    
    print(f"  > Seed category is: {seed_category}")
    print(f"  > Context requires slots: {required_slots}")

    # --- c) Find the Best Match for Each Slot ---
    for slot_category in required_slots:
        
        # 1. Don't find a match for the slot we already have
        # (e.g., if our seed is "bottoms", don't find another "bottoms")
        if slot_category == seed_category:
            continue 
            
        # 2. Get all item IDs for the category we're looking for (e.g., "tops")
        item_ids_in_bucket = category_buckets.get(slot_category)
        
        if not item_ids_in_bucket:
            print(f"  > Warning: No items found in bucket '{slot_category}'. Skipping.")
            continue
            
        # 3. Get the pre-computed vectors for just those items
        # We find the matrix index for each item_id in our bucket
        indices = [item_id_to_matrix_index[item_id] for item_id in item_ids_in_bucket]
        bucket_vectors = all_item_vectors[indices]
        
        # 4. Calculate similarity between our seed item and ALL items in the bucket
        similarities = cosine_similarity(seed_vector_np, bucket_vectors)
        similarity_scores = similarities[0]
        
        # 5. Find the single best item in this bucket
        best_item_index_in_bucket = np.argmax(similarity_scores)
        best_item_id = item_ids_in_bucket[best_item_index_in_bucket]
        best_item_score = similarity_scores[best_item_index_in_bucket]
        
        # 6. Add this best item to our outfit
        details = item_metadata.get(best_item_id)
        final_outfit.append({
            "id": best_item_id,
            "title": details.get('title', 'Untitled'),
            "category": slot_category,
            "image_url": f"images/{best_item_id}.jpg",
            "score": float(best_item_score)
        })
        
        print(f"  > Found best '{slot_category}': {best_item_id} (Score: {best_item_score:.4f})")

    # --- d) Add the seed item to the front and send back the complete outfit ---
    seed_item_data = {
        "id": seed_item_id,
        "title": seed_details.get('title', 'Untitled'),
        "category": seed_category,
        "image_url": f"images/{seed_item_id}.jpg",
        "score": 1.0
    }
    
    # Add seed item to the start
    final_outfit_with_seed = [seed_item_data] + final_outfit
    
    print(f"  > Found {len(final_outfit)} complementary items. Sending full outfit.")
    return jsonify(final_outfit_with_seed)


# --- 4. CONTEXT RULES (The "Logic" part of the Brain) ---
# This dictionary defines the "slots" to fill for a complete outfit.
# We can easily add new rules!
CONTEXT_RULES = {
    # 'any' context tries to build a full outfit
    "any": ["tops", "bottoms", "shoes", "bags", "outerwear"],
    
    "job_interview": [
        "tops", "bottoms", "shoes", "bags", "outerwear"
    ],
    
    "casual_weekend": [
        "tops", "bottoms", "shoes", "bags", "sunglasses", "hats"
    ],
    
    "party": [
        "all-body", "shoes", "bags", "jewellery "
        # Note: If seed is "all-body", it will skip this.
        # If seed is "shoes", it will find an "all-body" (dress), bag, and jewellery
    ],

    "sporty": [
        "tops", "bottoms", "shoes", "hats"
    ]
}


# --- 5. RUN THE SERVER ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)