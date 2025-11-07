document.addEventListener('DOMContentLoaded', () => {
    
    // --- CONFIG & ELEMENTS ---
    const API_URL = "http://127.0.0.1:5000"; // Our Python "Brain" server
    const seedItemSelect = document.getElementById('seed-item');
    const contextSelect = document.getElementById('context');
    const buildOutfitBtn = document.getElementById('build-outfit-btn');
    const outfitDisplay = document.getElementById('outfit-display');
    const resultsContainer = document.getElementById('results');
    const loader = document.getElementById('loader');
    const loaderText = document.getElementById('loader-text');
    const errorBox = document.getElementById('error-box');
    const errorMessage = document.getElementById('error-message');
    const seedLoadingOption = document.getElementById('seed-loading-option');

    // --- 1. INITIALIZATION ---

    /**
     * Fetches the 50-item catalog from the server to fill the dropdown.
     */
    async function populateSeedItems() {
        console.log("Connecting to server to get item catalog...");
        try {
            // This endpoint name /get_all_items matches the server.py
            const response = await fetch(`${API_URL}/get_all_items`);
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            const items = await response.json();

            // Clear the "loading" message
            seedLoadingOption.remove(); 

            // Add the 50 random items to the dropdown
            items.forEach(item => {
                const option = document.createElement('option');
                option.value = item.id;
                // Use title, or "Untitled" if title is missing
                option.textContent = item.title || "Untitled Item";
                option.dataset.category = item.category; // Store category
                seedItemSelect.appendChild(option);
            });

            // Enable the build button
            buildOutfitBtn.disabled = false;

        } catch (error) {
            console.error("Failed to populate seed items:", error);
            showError("Could not connect to server. Please ensure the backend is running.");
            seedLoadingOption.textContent = "Error: Could not load items";
        }
    }

    // --- 2. EVENT LISTENERS ---
    buildOutfitBtn.addEventListener('click', handleBuildOutfit);

    // --- 3. CORE LOGIC ---

    /**
     * Called when "Build Outfit" is clicked.
     */
    async function handleBuildOutfit() {
        const seed_item_id = seedItemSelect.value;
        const context_key = contextSelect.value;

        if (!seed_item_id) {
            showError("Please select a seed item.");
            return;
        }

        console.log(`Building outfit for: ${seed_item_id}, Context: ${context_key}`);
        
        showLoader("Finding the perfect items...");
        
        try {
            // This endpoint /get_recommendations matches the server.py
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
                const err = await response.json();
                throw new Error(err.error || `Server error: ${response.statusText}`);
            }

            const outfitItems = await response.json();
            
            displayOutfit(outfitItems);

        } catch (error) {
            console.error("Error building outfit:", error);
            showError(error.message);
        } finally {
            // This 'finally' block will run *no matter what*,
            // guaranteeing the spinner is hidden and buttons are re-enabled.
            // This is the fix for your bug.
            hideLoader();
        }
    }

    // --- 4. UI HELPER FUNCTIONS ---

    /**
     * Takes a list of outfit items and renders them to the page.
     */
    function displayOutfit(items) {
        // Clear previous results
        outfitDisplay.innerHTML = '';
        resultsContainer.classList.remove('hidden');
        hideError();

        if (!items || items.length === 0) {
            showError("No matching items found for this combination.");
            return;
        }

        items.forEach(item => {
            const card = document.createElement('div');
            card.className = 'bg-white rounded-lg shadow-md overflow-hidden transform transition-all duration-300 hover:shadow-xl hover:-translate-y-1';
            
            let cardTitle = item.title || "Untitled";
            // Truncate long titles
            if (cardTitle.length > 50) {
                cardTitle = cardTitle.substring(0, 50) + '...';
            }

            card.innerHTML = `
                <div class="aspect-w-1 aspect-h-1 bg-gray-200">
                    <img src="${item.image_url}" alt="${cardTitle}" class="w-full h-full object-cover object-center" onerror="this.src='https://placehold.co/300x300/e2e8f0/94a3b8?text=Image+Not+Found'">
                </div>
                <div class="p-4">
                    <h3 class="text-sm font-semibold text-gray-800 truncate" title="${item.title || 'Untitled'}">${cardTitle}</h3>
                    <p class="text-xs text-gray-500 uppercase font-medium">${item.category}</p>
                </div>
            `;
            outfitDisplay.appendChild(card);
        });
    }

    function showLoader(text) {
        loaderText.textContent = text;
        loader.classList.remove('hidden');
        resultsContainer.classList.add('hidden'); // Hide old results
        hideError();
        
        // Disable controls
        buildOutfitBtn.disabled = true;
        contextSelect.disabled = true;
        seedItemSelect.disabled = true;
    }

    function hideLoader() {
        loader.classList.add('hidden');
        
        // Re-enable controls
        buildOutfitBtn.disabled = false;
        contextSelect.disabled = false;
        seedItemSelect.disabled = false;
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorBox.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    }

    function hideError() {
        errorBox.classList.add('hidden');
    }

    // --- 5. START THE APP ---
    populateSeedItems();
});