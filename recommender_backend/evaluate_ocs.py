import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys
import os # Included for robustness, though not strictly required by core logic

# --- 1. CONFIGURATION & GLOBAL VARIABLES (From server_visual.py) ---
METADATA_FILE = 'polyvore_item_metadata.json'
EMBEDDINGS_FILE = 'visual_embeddings.json'
VISUAL_MODEL_NAME = 'clip-ViT-B-32'
TEST_OUTFIT_FILE_PATH = '../polyvore_outfits/disjoint/test.json' # Path to your ground truth file

# Global data containers
item_metadata = {}
item_id_to_matrix_index = {}
all_item_vectors = np.array([])
clip_model = None
device = 'cpu'
CATEGORY_TEXT_VECTORS = {}
GENDER_TEXT_VECTORS = {}
category_buckets = {}

# --- Context Rules (From server_visual.py) ---
CONTEXT_RULES = {
    "any": ["tops", "bottoms", "shoes", "bags", "outerwear"],
    "job_interview": ["tops", "bottoms", "shoes", "bags", "outerwear"],
    "casual_weekend": ["tops", "bottoms", "shoes", "bags", "sunglasses", "hats"],
    "party": ["all-body", "shoes", "bags", "jewellery"],
    "sporty": ["tops", "bottoms", "shoes", "bags", "hats"]
}

# --- Category Guesser Config (From server_visual.py) ---
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

# --- Gender Guesser Config (From server_visual.py) ---
GENDER_TEXT_QUERIES = [
    "men's fashion, clothing for men, masculine style",
    "women's fashion, clothing for women, feminine style"
]
GENDER_TEXT_MAP = {
    "men's fashion, clothing for men, masculine style": "men",
    "women's fashion, clothing for women, feminine style": "women"
}


# --- 2. CORE HELPER FUNCTIONS (From server_visual.py) ---

def check_for_gpu():
    global device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

def get_gender_from_categories(category_list):
    """Parses the category list from metadata to find a gender."""
    has_men = False
    has_women = False
    for cat in category_list:
        c = cat.lower()
        if "men's" in c or "male" in c:
            has_men = True
        if "women's" in c or "female" in c:
            has_women = True
    
    if has_men and not has_women:
        return "men"
    if has_women and not has_men:
        return "women"
    return "unisex" 

def load_all_data():
    global item_metadata, category_buckets
    global item_id_to_matrix_index, all_item_vectors, clip_model, CATEGORY_TEXT_VECTORS, GENDER_TEXT_VECTORS
    
    check_for_gpu()
    print(f"Loading data using device: {device}")
    
    try:
        # 1. Load CLIP model
        clip_model = SentenceTransformer(VISUAL_MODEL_NAME, device=device)

        # 2. Load metadata
        with open(METADATA_FILE, 'r') as f:
            item_metadata = json.load(f)

        # 3. Load FULL visual embeddings
        with open(EMBEDDINGS_FILE, 'r') as f:
            item_embeddings = json.load(f) 

        # 4. Build GENDERED category buckets and vector matrix
        temp_vectors = []
        current_index = 0
        
        for item_id, vector in item_embeddings.items():
            
            temp_vectors.append(np.array(vector, dtype=np.float32)) 
            item_id_to_matrix_index[item_id] = current_index
            current_index += 1
            
            if item_id not in item_metadata:
                continue 
            
            details = item_metadata[item_id]
            category = details.get('semantic_category', 'unknown').strip()
            gender = get_gender_from_categories(details.get('categories', []))
            
            bucket_key = f"{category}_{gender}" 
            
            if bucket_key not in category_buckets:
                category_buckets[bucket_key] = []
            category_buckets[bucket_key].append(item_id)

        all_item_vectors = np.array(temp_vectors, dtype=np.float32)

        # 5. Pre-compute text vectors
        with torch.no_grad():
            text_vectors = clip_model.encode(CATEGORY_TEXT_QUERIES, convert_to_tensor=True, device=device).to(torch.float32)
            gender_vecs = clip_model.encode(GENDER_TEXT_QUERIES, convert_to_tensor=True, device=device).to(torch.float32)
        
        for i, text_query in enumerate(CATEGORY_TEXT_QUERIES):
            category_name = CATEGORY_TEXT_MAP[text_query]
            CATEGORY_TEXT_VECTORS[category_name] = text_vectors[i]
        
        for i, text_query in enumerate(GENDER_TEXT_QUERIES):
            gender_name = GENDER_TEXT_MAP[text_query]
            GENDER_TEXT_VECTORS[gender_name] = gender_vecs[i]
        
        print(f"Data loading complete. Vector matrix shape: {all_item_vectors.shape}")

    except FileNotFoundError as e:
        print(f"\n--- !!! FATAL ERROR !!! ---\nCould not find a data file: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        sys.exit(1)


def guess_image_category(image_vector_tensor):
    """Compares an image vector to our pre-computed category text vectors."""
    text_vectors = torch.stack(list(CATEGORY_TEXT_VECTORS.values())).to(device)
    similarities = util.cos_sim(image_vector_tensor.to(torch.float32), text_vectors)
    similarity_scores = similarities[0].cpu().numpy()
    best_match_index = np.argmax(similarity_scores)
    best_category_name = list(CATEGORY_TEXT_VECTORS.keys())[best_match_index]
    return best_category_name

def guess_image_gender(image_vector_tensor):
    """Compares an image vector to our pre-computed gender text vectors."""
    gender_vecs = torch.stack(list(GENDER_TEXT_VECTORS.values())).to(device)
    similarities = util.cos_sim(image_vector_tensor.to(torch.float32), gender_vecs)
    similarity_scores = similarities[0].cpu().numpy()
    
    men_score = similarity_scores[0]
    women_score = similarity_scores[1]

    if abs(men_score - women_score) < 0.05:
        return "unisex"
    elif men_score > women_score:
        return "men"
    else:
        return "women"

# --- 3. CORE LOGIC (From server_visual.py) ---

def find_complete_outfit(seed_vector_np, seed_category, context_key, gender_key):
    """
    Core "Outfit Completer" logic. Returns the list of recommended items.
    """
    
    required_slots = CONTEXT_RULES.get(context_key, [])
    final_outfit = []
    
    if seed_vector_np.ndim == 1:
        seed_vector_np = seed_vector_np.reshape(1, -1)
    
    seed_vector_np = seed_vector_np.astype(np.float32)

    for slot_category in required_slots:
        
        if slot_category == seed_category:
            continue 
            
        # --- GENDER LOGIC ---
        search_bucket_keys = []
        if gender_key == 'men':
            search_bucket_keys = [f"{slot_category}_men", f"{slot_category}_unisex"]
        elif gender_key == 'women':
            search_bucket_keys = [f"{slot_category}_women", f"{slot_category}_unisex"]
        else: # 'unisex'
            search_bucket_keys = [f"{slot_category}_unisex", f"{slot_category}_men", f"{slot_category}_women"]

        combined_item_ids_in_bucket = []
        for key in search_bucket_keys:
            if key in category_buckets:
                combined_item_ids_in_bucket.extend(category_buckets[key])
        
        if not combined_item_ids_in_bucket:
            continue
            
        valid_item_ids_in_bucket = [item_id for item_id in list(set(combined_item_ids_in_bucket)) if item_id in item_id_to_matrix_index]
        if not valid_item_ids_in_bucket:
            continue

        indices = [item_id_to_matrix_index[item_id] for item_id in valid_item_ids_in_bucket]
        bucket_vectors = all_item_vectors[indices]
        
        similarities = cosine_similarity(seed_vector_np, bucket_vectors)
        similarity_scores = similarities[0]
        
        # --- SANITY CHECK LOOP (Top 10) ---
        top_k_indices = np.argsort(similarity_scores)[-10:][::-1] 
        
        found_item = False
        for i in top_k_indices:
            best_item_id = valid_item_ids_in_bucket[i]
            best_item_score = similarity_scores[i]
            
            item_vector_np = all_item_vectors[item_id_to_matrix_index[best_item_id]]
            item_vector_tensor = torch.tensor(item_vector_np, dtype=torch.float32).to(device).reshape(1, -1)

            # CHECK 1: CATEGORY SANITY CHECK
            guessed_category = guess_image_category(item_vector_tensor)
            if guessed_category != slot_category:
                continue

            # CHECK 2: GENDER SANITY CHECK
            if slot_category not in ["bags", "sunglasses", "hats", "jewellery"]:
                guessed_gender = guess_image_gender(item_vector_tensor)
                
                if gender_key == 'men' and guessed_gender == 'women':
                    continue
                if gender_key == 'women' and guessed_gender == 'men':
                    continue
            
            # --- SUCCESS! ---
            final_outfit.append({
                "id": best_item_id,
                "title": item_metadata.get(best_item_id).get('title', 'Untitled'),
                "category": slot_category,
                "image_url": f"images/{best_item_id}.jpg",
                "score": float(best_item_score)
            })
            found_item = True
            break
        
    return final_outfit


# --- 4. OCS EVALUATION FUNCTIONS (NEW) ---

def load_ground_truth_outfits(filepath):
    """Loads ground truth outfit sets from the Polyvore JSON files."""
    try:
        with open(filepath, 'r') as f:
            outfit_data = json.load(f)
        
        all_outfits = []
        for outfit in outfit_data:
            # The Polyvore outfit structure has an 'items' list where each item is a dict
            item_ids = [item['item_id'] for item in outfit['items']]
            all_outfits.append(item_ids)
            
        return all_outfits
        
    except FileNotFoundError:
        print(f"--- ERROR: Ground truth file not found at {filepath} ---")
        return []
    except Exception as e:
        print(f"Error loading ground truth outfits: {e}")
        return []

def run_ocs_evaluation(all_outfits, context_key, target_gender, num_test_outfits):
    """Runs the Outfit Completeness Score (OCS) evaluation."""
    all_similarity_scores = []
    
    random.shuffle(all_outfits)
    outfits_to_test = all_outfits[:num_test_outfits]

    for outfit_item_ids in tqdm(outfits_to_test, desc=f"Evaluating OCS ({context_key}/{target_gender})"):
        
        if len(outfit_item_ids) < 2:
            continue

        seed_item_id = outfit_item_ids[0]
        ground_truth_ids = outfit_item_ids[1:]
        
        try:
            # Get seed properties
            seed_details = item_metadata[seed_item_id]
            seed_vector_np = all_item_vectors[item_id_to_matrix_index[seed_item_id]]
            seed_category = seed_details.get('semantic_category', 'unknown').strip()
            
            # Filter seed by target gender to ensure a meaningful test
            seed_gender = get_gender_from_categories(seed_details.get('categories', []))
            if (target_gender == 'men' and seed_gender == 'women') or \
               (target_gender == 'women' and seed_gender == 'men'):
                continue
                
            # Build ground truth lookup map for comparison
            ground_truth_map = {}
            for item_id in ground_truth_ids:
                if item_id in item_metadata:
                    cat = item_metadata[item_id].get('semantic_category', 'unknown').strip()
                    if cat != 'unknown':
                        # Ensure the true item is in the vector matrix
                        if item_id in item_id_to_matrix_index:
                            ground_truth_map[cat] = item_id

            if not ground_truth_map:
                continue

        except KeyError:
            continue 

        # 1. Generate Recommendations
        recommended_items = find_complete_outfit(
            seed_vector_np, seed_category, context_key, target_gender
        )

        # 2. Compare Recommendations to Ground Truth
        for rec_item in recommended_items:
            rec_category = rec_item['category']
            
            # We only compare slots that are present in BOTH the Context Rules AND the Ground Truth set
            if rec_category in ground_truth_map:
                try:
                    rec_vector = all_item_vectors[item_id_to_matrix_index[rec_item['id']]]
                    true_item_id = ground_truth_map[rec_category]
                    true_vector = all_item_vectors[item_id_to_matrix_index[true_item_id]]
                    
                    # Calculate similarity using Cosine Similarity
                    score = cosine_similarity(
                        rec_vector.reshape(1, -1), 
                        true_vector.reshape(1, -1)
                    )[0][0]
                    
                    all_similarity_scores.append(score)
                    
                except KeyError:
                    continue

    if not all_similarity_scores:
        return 0.0
        
    return np.mean(all_similarity_scores)

def plot_results(results_dict, gender):
    """Generates and saves a bar chart of the OCS results."""
    df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['OCS'])
    df = df.sort_values('OCS', ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = df['OCS'].plot(kind='bar', color='#007bff', zorder=2)
    plt.title(f'Outfit Completeness Score (OCS) by Context ({gender.title()})', fontsize=16)
    plt.ylabel('Mean Cosine Similarity (Higher is Better)', fontsize=12)
    plt.xlabel('Outfit Context', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    
    for bar in bars.patches:
        plt.annotate(f'{bar.get_height():.3f}',
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom',
                     size=11, xytext=(0, 5),
                     textcoords='offset points')
                     
    plt.tight_layout()
    filename = f'ocs_evaluation_plot_{gender}.png'
    plt.savefig(filename)
    print(f"\nPlot saved as {filename}")

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    
    print("--- Starting OCS Evaluation Script ---")
    
    # 1. Load all models and data
    load_all_data() 
    
    # 2. Define your test parameters
    CONTEXTS_TO_TEST = ["casual_weekend", "sporty", "party", "job_interview"]
    GENDERS_TO_TEST = ['women', 'men'] # Test both genders for a complete picture
    NUM_OUTFITS_TO_TEST = 1000 # Number of outfits to sample from the test set

    # 3. Load ground truth outfits
    all_outfits = load_ground_truth_outfits(TEST_OUTFIT_FILE_PATH)

    if not all_outfits:
        print("Evaluation failed: No outfits loaded.")
        sys.exit(1)
        
    print(f"Loaded {len(all_outfits)} outfits from test set.")
    
    # 4. Run evaluation for each context/gender combination
    for gender in GENDERS_TO_TEST:
        ocs_results = {}
        print(f"\n=============================================")
        print(f"  RUNNING EVALUATION FOR GENDER: {gender.upper()}")
        print(f"=============================================")

        for context in CONTEXTS_TO_TEST:
            score = run_ocs_evaluation(
                all_outfits, 
                context_key=context, 
                target_gender=gender,
                num_test_outfits=NUM_OUTFITS_TO_TEST
            )
            ocs_results[context] = score
        
        # 5. Plot and display results
        if ocs_results:
            print("\n--- Final OCS Scores ---")
            for k, v in ocs_results.items():
                print(f"  {k.ljust(18)}: {v:.4f}")
            plot_results(ocs_results, gender)
        else:
            print(f"No results generated for {gender}.")

    print("\n--- FULL OCS EVALUATION FINISHED ---")