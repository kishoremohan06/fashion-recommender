import json
import csv
import os
from collections import defaultdict

# --- Configuration ---
# --- (You may need to change these paths) ---

# Path to the root folder of the dataset
DATASET_ROOT = 'polyvore_outfits' 

# Metadata file paths
ITEM_METADATA_FILE = os.path.join(DATASET_ROOT, 'polyvore_item_metadata.json')
OUTFIT_TITLES_FILE = os.path.join(DATASET_ROOT, 'polyvore_outfit_titles.json')
CATEGORIES_FILE = os.path.join(DATASET_ROOT, 'categories.csv')

# Folders containing the outfit definition files (train.json, etc.)
OUTFIT_FOLDERS = [
    os.path.join(DATASET_ROOT, 'disjoint'),
    os.path.join(DATASET_ROOT, 'nondisjoint')
]

# Our final output file
OUTPUT_DATABASE_FILE = 'app_database.json'

# --- Step 1: Define Our "Context" Keyword Mapper ---
# We will mine outfit titles for these keywords.
CONTEXT_MAPPER = {
    "job_interview": ["work", "office", "business", "interview", "career", "professional"],
    "casual_weekend": ["casual", "weekend", "errands", "relax", "comfy", "day off"],
    "party": ["party", "night out", "club", "formal", "event", "wedding", "dance"],
    "sporty": ["sport", "gym", "workout", "athletic", "active"]
}

print("Starting data processing...")

def load_json_file(filepath):
    """Safely loads a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_csv_to_dict(filepath, key_field):
    """Loads a CSV file into a dictionary, keyed by the specified field."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return {}
    
    data_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get(key_field)
                if key:
                    data_dict[key] = row
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}
    return data_dict

# --- Step 2: Load All Metadata ---
print("Loading metadata files...")
item_metadata = load_json_file(ITEM_METADATA_FILE)
outfit_titles = load_json_file(OUTFIT_TITLES_FILE)
category_map = load_csv_to_dict(CATEGORIES_FILE, key_field='category_id')

if not all([item_metadata, outfit_titles, category_map]):
    print("Critical metadata files are missing. Exiting.")
    exit()

print("Metadata loaded successfully.")

# --- Step 3: Find and Load All Outfit Files ---
all_outfits = []
for folder in OUTFIT_FOLDERS:
    for filename in os.listdir(folder):
        # We only care about the training/validation data
        if filename.startswith('train') and filename.endswith('.json') or \
           filename.startswith('valid') and filename.endswith('.json'):
            
            filepath = os.path.join(folder, filename)
            print(f"Loading outfit file: {filepath}")
            outfit_data = load_json_file(filepath)
            if outfit_data:
                all_outfits.extend(outfit_data)

print(f"Loaded a total of {len(all_outfits)} outfits.")

# --- Step 4: Build the Final app_database ---
# This will be a dictionary where keys are item_ids for fast lookups
app_database = {}

# Use defaultdict to easily append to lists
item_compatibility_map = defaultdict(set)
item_context_map = defaultdict(set)

print("Processing outfits to build compatibility and context maps...")
for outfit in all_outfits:
    set_id = outfit.get('set_id')
    if not set_id:
        continue
        
    item_ids_in_outfit = [item.get('item_id') for item in outfit.get('items', []) if item.get('item_id')]
    
    # 1. Find the Context for this outfit
    outfit_context_tags = set()
    if set_id in outfit_titles:
        title = outfit_titles[set_id].get('title', '').lower()
        for context_name, keywords in CONTEXT_MAPPER.items():
            if any(keyword in title for keyword in keywords):
                outfit_context_tags.add(context_name)
    
    # 2. Add compatibility and context to each item
    for item_id in item_ids_in_outfit:
        # Add context tags to this item
        item_context_map[item_id].update(outfit_context_tags)
        
        # Add compatibility links (item -> other items)
        for other_item_id in item_ids_in_outfit:
            if item_id != other_item_id:
                item_compatibility_map[item_id].add(other_item_id)

print("Building final database from item metadata...")
# Now, create the final database by combining metadata with our new maps
final_database_list = []
for item_id, details in item_metadata.items():
    
    # Find the best category name
    category_name = details.get('semantic_category')
    category_id = details.get('category_id')
    if category_id and category_id in category_map:
        # Prefer the 'main_category' from the CSV (e.g., "tops")
        category_name = category_map[category_id].get('main_category', category_name)
    
    if not category_name:
        category_name = "unknown" # Fallback

    # Create the final, clean item object
    clean_item = {
        "id": item_id,
        "title": details.get('title', 'No Title'),
        "category": category_name.lower(),
        "image_url": f"images/{item_id}.jpg", # Assumes images are named by item_id
        "contexts": list(item_context_map[item_id]),
        "compatible_with": list(item_compatibility_map[item_id])
    }
    
    final_database_list.append(clean_item)

# --- Step 5: Save the Final File ---
print(f"Saving {len(final_database_list)} items to {OUTPUT_DATABASE_FILE}...")
try:
    with open(OUTPUT_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_database_list, f, indent=2)
    print("---")
    print(f"Success! Your clean database is ready at: {OUTPUT_DATABASE_FILE}")
    print("---")
except Exception as e:
    print(f"Error saving final database: {e}")
    
    