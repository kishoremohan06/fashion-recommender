import json
import numpy as np 
from tqdm import tqdm # To show progress

# --- CONFIGURATION ---
# These are the files we are reading
TEXT_EMBEDDINGS_FILE = 'item_embeddings.json' # Your 74k text vectors
VISUAL_EMBEDDINGS_FILE = 'visual_embeddings.json' # Your 261k visual vectors

# This is the new file we will create
OUTPUT_FILE = 'combined_embeddings.json'
# --- END CONFIGURATION ---

def combine_embeddings():
    """
    Loads text and visual embeddings, finds the intersection,
    and saves a new file with combined 'super-vectors'.
    """
    print("--- Starting Multimodal Embedding Combination ---")
    
    # 1. Load the text embeddings (the smaller file)
    print(f"Loading text embeddings from '{TEXT_EMBEDDINGS_FILE}'...")
    try:
        with open(TEXT_EMBEDDINGS_FILE, 'r') as f:
            text_embeddings = json.load(f)
        print(f"Loaded {len(text_embeddings)} text vectors.")
    except FileNotFoundError:
        print(f"--- ERROR: File not found: '{TEXT_EMBEDDINGS_FILE}' ---")
        print("Please run 'create_embeddings.py' first.")
        return
    except Exception as e:
        print(f"Error loading {TEXT_EMBEDDINGS_FILE}: {e}")
        return

    # 2. Load the visual embeddings (the larger file)
    print(f"Loading visual embeddings from '{VISUAL_EMBEDDINGS_FILE}'...")
    try:
        with open(VISUAL_EMBEDDINGS_FILE, 'r') as f:
            visual_embeddings = json.load(f)
        print(f"Loaded {len(visual_embeddings)} visual vectors.")
    except FileNotFoundError:
        print(f"--- ERROR: File not found: '{VISUAL_EMBEDDINGS_FILE}' ---")
        print("Please run 'create_visual_embeddings.py' first.")
        return
    except Exception as e:
        print(f"Error loading {VISUAL_EMBEDDINGS_FILE}: {e}")
        return

    # 3. Find the intersection and combine vectors
    print("Finding intersection and combining vectors...")
    
    combined_embeddings = {}
    
    # We iterate over the smaller set (text_embeddings) for speed
    # and look for each item in the larger set (visual_embeddings)
    
    # Use tqdm for a progress bar
    for item_id, text_vector in tqdm(text_embeddings.items(), desc="Combining vectors"):
        
        # Check if this item ALSO has a visual vector
        if item_id in visual_embeddings:
            visual_vector = visual_embeddings[item_id]
            
            # Concatenate them into one "super-vector"
            # We convert to numpy arrays for fast concatenation
            combined_vector = np.concatenate([
                np.array(text_vector, dtype=np.float32),
                np.array(visual_vector, dtype=np.float32)
            ])
            
            # Add to our new dictionary (convert back to list for JSON)
            combined_embeddings[item_id] = combined_vector.tolist()

    if not combined_embeddings:
        print("--- ERROR: No items in common between the two embedding files! ---")
        print("Something went wrong. Cannot create combined file.")
        return

    print(f"\nFound {len(combined_embeddings)} items that have BOTH text and visual data.")
    
    # 4. Save the new combined file
    print(f"Saving combined 'super-vectors' to '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(combined_embeddings, f)
        print("--- Save successful! ---")
        print(f"\nYou are now ready to run 'python server_multimodal.py'")
    except Exception as e:
        print(f"--- ERROR saving file: {e} ---")

if __name__ == "__main__":
    combine_embeddings()