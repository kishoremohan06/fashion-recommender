// --- 1. CONSTANTS & GLOBAL STATE ---
const API_URL = "http://127.0.0.1:5000";
let allItemData = [];
let activeTab = 'catalog-tab'; // 'catalog-tab' or 'upload-tab'
let uploadedFile = null;

// --- 2. DOM ELEMENTS ---
const seedItemSelect = document.getElementById("seed-item");
const contextSelect = document.getElementById("context");
const genderSelect = document.getElementById("gender"); // <-- NEW GENDER ELEMENT
const buildBtn = document.getElementById("build-outfit-btn");
const outfitDisplay = document.getElementById("outfit-display");
const outfitTitle = document.getElementById("outfit-title");
const loader = document.getElementById("loader");

// Tab elements
const tabButtons = document.querySelectorAll(".tab-button");
const tabContents = document.querySelectorAll(".tab-content");

// Upload elements
const dropZone = document.getElementById("file-drop-zone");
const fileInput = document.getElementById("file-input");
const filePreview = document.getElementById("file-preview");
const previewImage = document.getElementById("preview-image");
const previewFilename = document.getElementById("preview-filename");

// --- 3. MAIN APP INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
    populateSeedItems();
    
    // Main button listener
    buildBtn.addEventListener('click', handleBuildOutfit);

    // Tab switching logic
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            showTab(tabId);
        });
    });

    // File Uploader listeners
    setupFileUploader();
});

// --- 4. CORE LOGIC ---

/**
 * Main function called when "Build Outfit" is clicked.
 * It checks which tab is active and calls the correct API.
 */
async function handleBuildOutfit() {
    const context_key = contextSelect.value;
    const gender_key = genderSelect.value; // <-- NEW: Get selected gender
    
    // Clear previous results
    outfitDisplay.innerHTML = "";
    outfitTitle.classList.add('hidden');
    showLoader("Finding the perfect items...");

    try {
        let recommendedItems = [];
        let seedItem = null;

        if (activeTab === 'catalog-tab') {
            // --- OPTION 1: Get from Catalog ---
            const seed_item_id = seedItemSelect.value;
            if (!seed_item_id) {
                alert("Please select a seed item first.");
                hideLoader();
                return;
            }
            // Pass the gender_key to the API
            recommendedItems = await recommendFromCatalog(seed_item_id, context_key, gender_key);
            
            // The server sends the seed item back in the list
            seedItem = recommendedItems.shift(); // Remove the first item (seed)

        } else if (activeTab === 'upload-tab') {
            // --- OPTION 2: Get from Upload ---
            if (!uploadedFile) {
                alert("Please upload an image first.");
                hideLoader();
                return;
            }
            // Pass the gender_key to the API
            recommendedItems = await recommendFromUpload(uploadedFile, context_key, gender_key);
            
            // For uploads, the "seed item" is the preview image
            seedItem = {
                title: "Your Uploaded Item",
                category: "Seed Item",
                image_url: previewImage.src // Use the local preview URL
            };
        }

        // Display the results
        displayOutfit(seedItem, recommendedItems);

    } catch (error) {
        console.error("Error building outfit:", error);
        outfitDisplay.innerHTML = `<p class="text-red-500 col-span-full"><strong>An error occurred.</strong><br>Could not get recommendations. Check the console and make sure your server is running.</p>`;
    } finally {
        hideLoader();
    }
}

/**
 * Calls the API for a catalog-based recommendation.
 */
async function recommendFromCatalog(seed_item_id, context_key, gender_key) {
    console.log(`Building outfit from catalog. Seed: ${seed_item_id}, Context: ${context_key}, Gender: ${gender_key}`);
    const response = await fetch(`${API_URL}/get_recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            seed_item_id: seed_item_id,
            context_key: context_key,
            gender_key: gender_key // <-- NEW: Send gender
        })
    });
    if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
    return await response.json();
}

/**
 * Calls the API for an image-upload-based recommendation.
 */
async function recommendFromUpload(file, context_key, gender_key) {
    console.log(`Building outfit from upload. File: ${file.name}, Context: ${context_key}, Gender: ${gender_key}`);
    
    // We use FormData to send a file and other data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('context_key', context_key);
    formData.append('gender_key', gender_key); // <-- NEW: Send gender

    const response = await fetch(`${API_URL}/recommend_by_upload`, {
        method: 'POST',
        body: formData // No 'Content-Type' header needed, browser sets it
    });
    if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
    return await response.json();
}

// --- 5. UI & DISPLAY FUNCTIONS ---

/**
 * Fetches the 50-item catalog from the server to fill the dropdown.
 */
async function populateSeedItems() {
    console.log("Connecting to server for catalog...");
    try {
        const response = await fetch(`${API_URL}/get_all_items`);
        if (!response.ok) throw new Error("Server not responding");
        
        let items = await response.json();
        allItemData = items; 
        items.sort((a, b) => a.title.localeCompare(b.title));

        seedItemSelect.innerHTML = '<option value="" disabled selected>Select a "Seed" Item</option>';
        items.forEach(item => {
            const option = document.createElement('option');
            option.value = item.id;
            option.textContent = item.title;
            seedItemSelect.appendChild(option);
        });
    } catch (error) {
        console.error("Error populating items:", error);
        seedItemSelect.innerHTML = '<option value="" disabled selected>Error: Could not connect</option>';
    }
}

/**
 * Renders the final outfit (seed item + recommendations) to the page.
 */
function displayOutfit(seedItem, recommendedItems) {
    outfitDisplay.innerHTML = ""; // Clear
    outfitTitle.classList.remove('hidden'); // Show title

    if ((!recommendedItems || recommendedItems.length === 0) && !seedItem) {
        outfitDisplay.innerHTML = "<p class='col-span-full text-gray-500'>No items were found for this combination.</p>";
        return;
    }

    // 1. Create a card for the seed item
    if (seedItem) {
        const seedCard = createItemCard(seedItem, true);
        outfitDisplay.appendChild(seedCard);
    }

    // 2. Create cards for all recommended items
    recommendedItems.forEach(item => {
        const itemCard = createItemCard(item, false);
        outfitDisplay.appendChild(itemCard);
    });
}

/**
 * Helper function to create a single HTML card for an item.
 */
function createItemCard(item, isSeedItem) {
    const itemCard = document.createElement('div');
    itemCard.className = 'outfit-item';
    
    let imageUrl = item.image_url;
    // For catalog items, we must construct the full path
    // For uploaded items, the URL is a local blob: URL
    if (!isSeedItem || (isSeedItem && !item.image_url.startsWith('blob:'))) {
         imageUrl = `images/${item.id}.jpg`;
    }

    itemCard.innerHTML = `
        <img src="${imageUrl}" alt="${item.title}" onerror="this.src='https://placehold.co/300x400/f0f2f5/ccc?text=Image+Missing'; this.onerror=null;">
        <div class="item-details">
            <p class="item-title" title="${item.title}">${item.title}</p>
            <p class="item-category">${item.category}</p>
        </div>
    `;
    return itemCard;
}

/**
 * Handles the logic for switching between the "Catalog" and "Upload" tabs.
 */
function showTab(tabId) {
    activeTab = tabId;
    
    // Update button styles
    tabButtons.forEach(button => {
        if (button.getAttribute('data-tab') === tabId) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });

    // Show/hide tab content
    tabContents.forEach(content => {
        if (content.id === tabId) {
            content.classList.remove('hidden');
        } else {
            content.classList.add('hidden');
        }
    });
}

// --- 6. FILE UPLOADER LOGIC ---

/**
 * Sets up all the event listeners for the drag-and-drop box.
 */
function setupFileUploader() {
    // Click to upload
    fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    // Drag and drop listeners
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drop-zone--over');
    });
    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('drop-zone--over');
        });
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drop-zone--over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    // Allow clicking the dropzone to open the file dialog
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
}

/**
 * Handles the selected file (from click or drop).
 * It validates the file and shows a preview.
 */
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert("Please upload a valid image file (jpg, png).");
        return;
    }
    
    uploadedFile = file; // Save the file for when we click "Build"

    // Show preview
    previewFilename.textContent = file.name;
    previewImage.src = URL.createObjectURL(file); // Create a temporary local URL
    filePreview.classList.remove('hidden');
}

// --- 7. LOADER HELPER FUNCTIONS ---

function showLoader(message) {
    loader.querySelector('p').textContent = message;
    loader.classList.remove('hidden');
    buildBtn.disabled = true;
    seedItemSelect.disabled = true;
    contextSelect.disabled = true;
    genderSelect.disabled = true; // <-- NEW
}

function hideLoader() {
    loader.classList.add('hidden');
    buildBtn.disabled = false;
    seedItemSelect.disabled = false;
    contextSelect.disabled = false;
    genderSelect.disabled = false; // <-- NEW
}