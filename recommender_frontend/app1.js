// --- 1. CONSTANTS & DOM ELEMENTS ---
// This is the address of your Python server.
const API_URL = "http://127.0.0.1:5000";

// Get references to all the HTML elements we need
const seedItemSelect = document.getElementById("seed-item-select");
const contextSelect = document.getElementById("context-select");
const buildBtn = document.getElementById("build-btn");
const outfitDisplay = document.getElementById("outfit-display");
const loader = document.getElementById("loader");

// This will store the data from /get_all_items to avoid re-fetching
let allItemData = [];

// --- 2. MAIN APP INITIALIZATION ---

// Wait for the HTML document to be fully loaded before running our script
document.addEventListener("DOMContentLoaded", () => {
    // 1. Fetch all items from the server to fill the dropdown
    populateSeedItems();

    // 2. Add a click listener to our "Build Outfit" button
    buildBtn.addEventListener('click', handleBuildOutfit);
});

// --- 3. CORE FUNCTIONS ---

/**
 * Fetches the complete list of items from our server's /get_all_items endpoint
 * and populates the "Seed Item" dropdown menu.
 */
async function populateSeedItems() {
    console.log("Connecting to server to get item list...");
    showLoader("Connecting to server...");
    
    try {
        const response = await fetch(`${API_URL}/get_all_items`);
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        
        let items = await response.json();
        
        // Store the items so we can use them later if needed
        allItemData = items;

        // Sort items alphabetically by title
        items.sort((a, b) => a.title.localeCompare(b.title));

        // Clear the "Connecting..." message
        seedItemSelect.innerHTML = '<option value="" disabled selected>Select a "Seed" Item</option>';
        
        // Create an <option> for each item and add it to the dropdown
        items.forEach(item => {
            const option = document.createElement('option');
            option.value = item.id;
            option.textContent = item.title;
            seedItemSelect.appendChild(option);
        });
        
        console.log(`Successfully loaded ${items.length} items.`);

    } catch (error) {
        console.error("Error populating items:", error);
        seedItemSelect.innerHTML = '<option value="" disabled selected>Error: Could not connect to server</option>';
        // If we fail, show the error in the UI
        outfitDisplay.innerHTML = `<p class="error"><strong>Failed to load data from server.</strong><br>Is your 'server.py' terminal running?</p>`;
    } finally {
        // Whether it succeeded or failed, hide the loader
        hideLoader();
    }
}

/**
 * Called when the "Build Outfit" button is clicked.
 * It sends the selected seed item and context to the backend API.
 */
async function handleBuildOutfit() {
    const seed_item_id = seedItemSelect.value;
    const context_key = contextSelect.value;

    // Simple validation
    if (!seed_item_id) {
        alert("Please select a seed item first.");
        return;
    }

    console.log(`Building outfit. Seed: ${seed_item_id}, Context: ${context_key}`);
    showLoader("Finding the perfect items...");
    outfitDisplay.innerHTML = ""; // Clear the previous outfit

    try {
        // Call the /get_recommendations endpoint
        const response = await fetch(`${API_URL}/get_recommendations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                seed_item_id: seed_item_id,
                context_key: context_key
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const recommendedItems = await response.json();
        
        // Pass the recommended items to the display function
        displayOutfit(recommendedItems);

    } catch (error) {
        console.error("Error building outfit:", error);
        outfitDisplay.innerHTML = `<p class="error"><strong>An error occurred.</strong><br>Could not get recommendations. Please check the console.</p>`;
    } finally {
        hideLoader();
    }
}

/**
 * Takes an array of item objects and renders them as cards in the
 * #outfit-display grid.
 */
function displayOutfit(items) {
    // Clear any previous items or error messages
    outfitDisplay.innerHTML = "";

    if (!items || items.length === 0) {
        outfitDisplay.innerHTML = "<p>No items were found for this combination.</p>";
        return;
    }

    // Create a card for each item
    items.forEach(item => {
        const itemCard = document.createElement('div');
        itemCard.className = 'outfit-item';

        const itemImage = document.createElement('img');
        // We construct the image path. Assumes 'images' folder is in the same
        // directory as index.html
        itemImage.src = item.image_url;
        
        // Add a fallback image in case the file is missing
        itemImage.onerror = () => {
            itemImage.src = `https://placehold.co/300x400/f0f2f5/ccc?text=Image+Missing`;
            itemImage.title = `Image not found at path: ${item.image_url}`;
        };

        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'item-details';

        const itemTitle = document.createElement('p');
        itemTitle.className = 'item-title';
        itemTitle.textContent = item.title || "Untitled"; // Handle missing titles

        const itemCategory = document.createElement('p');
        itemCategory.className = 'item-category';
        itemCategory.textContent = item.category;

        // Assemble the card
        detailsDiv.appendChild(itemTitle);
        detailsDiv.appendChild(itemCategory);
        itemCard.appendChild(itemImage);
        itemCard.appendChild(detailsDiv);

        // Add the new card to the display grid
        outfitDisplay.appendChild(itemCard);
    });
}

// --- 4. HELPER FUNCTIONS ---

/** Shows the loading spinner and disables the button */
function showLoader(message) {
    loader.querySelector('p').textContent = message;
    loader.classList.remove('hidden');
    buildBtn.disabled = true;
    seedItemSelect.disabled = true;
    contextSelect.disabled = true;
}

/** Hides the loading spinner and enables the button */
function hideLoader() {
    loader.classList.add('hidden');
    buildBtn.disabled = false;
    seedItemSelect.disabled = false;
    contextSelect.disabled = false;
}