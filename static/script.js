// ============================================================
// static/script.js — JavaScript frontend logic
// ============================================================
// This file runs in the BROWSER (not on the server).
// It does 3 things:
//   1. Reads values the user typed into the form
//   2. Sends those values to the FastAPI backend using fetch()
//   3. Displays the prediction result (or error) on the page
//
// The fetch() function is a modern browser API for making HTTP requests.
// It's like an AJAX call — it sends/receives data without refreshing the page.
// ============================================================

// ============================================================
// WAIT FOR THE PAGE TO FULLY LOAD
// ============================================================
// DOMContentLoaded fires when the HTML is fully parsed.
// We wrap everything in this so our code runs AFTER the HTML exists.
// If we don't do this, JavaScript might try to find elements
// that haven't been created yet.

document.addEventListener("DOMContentLoaded", function () {

    // ============================================================
    // GET REFERENCES TO HTML ELEMENTS
    // ============================================================
    // document.getElementById("id") finds an HTML element by its id attribute.
    // We store them in variables so we can use them easily.

    // INPUT FIELDS — where the user types values
    const areaInput      = document.getElementById("area");
    const bedroomsInput  = document.getElementById("bedrooms");
    const bathroomsInput = document.getElementById("bathrooms");
    const storiesInput   = document.getElementById("stories");
    const parkingInput   = document.getElementById("parking");

    // BUTTON — the "Predict Price" button
    const predictBtn     = document.getElementById("predict-btn");
    const btnText        = document.getElementById("btn-text");
    const btnSpinner     = document.getElementById("btn-spinner");

    // RESULT ELEMENTS — shown after prediction
    const resultSection  = document.getElementById("result-section");
    const predictedPrice = document.getElementById("predicted-price");
    const resultMetaText = document.getElementById("result-meta-text");

    // ERROR ELEMENT — shown when something goes wrong
    const errorSection   = document.getElementById("error-section");
    const errorText      = document.getElementById("error-text");

    // ============================================================
    // ADD CLICK EVENT TO THE PREDICT BUTTON
    // ============================================================
    // addEventListener("click", function) means:
    // "When this button is clicked, run this function."

    predictBtn.addEventListener("click", handlePrediction);

    // Also allow pressing Enter in any input to trigger prediction
    // This improves user experience
    [areaInput, bedroomsInput, bathroomsInput, storiesInput, parkingInput].forEach(function(input) {
        input.addEventListener("keypress", function(event) {
            if (event.key === "Enter") {
                handlePrediction();
            }
        });
        
        // Remove red border when user starts typing again
        input.addEventListener("input", function() {
            this.classList.remove("is-invalid");
        });
    });

    // ============================================================
    // MAIN FUNCTION: handlePrediction()
    // ============================================================
    // This runs when the user clicks "Predict Price".
    // 'async' means this function can use 'await' to wait for API responses.

    async function handlePrediction() {

        // ----------------------------------------------------------
        // STEP 1: Read and validate input values
        // ----------------------------------------------------------
        
        // parseFloat() converts a string like "7420" to a number 7420
        // parseInt() converts a string like "4" to an integer 4
        const area      = parseFloat(areaInput.value);
        const bedrooms  = parseInt(bedroomsInput.value);
        const bathrooms = parseInt(bathroomsInput.value);
        const stories   = parseInt(storiesInput.value);
        const parking   = parseInt(parkingInput.value);

        // Validate — check that all fields have valid values
        // isNaN() means "is Not a Number" — returns true if value is missing or invalid
        const isValid = validateInputs({
            area, bedrooms, bathrooms, stories, parking
        });

        if (!isValid) {
            return;  // Stop here if validation failed
        }

        // ----------------------------------------------------------
        // STEP 2: Show loading state
        // ----------------------------------------------------------
        // While waiting for the API response, we show a spinner
        // and disable the button so the user can't click it again.

        setLoadingState(true);
        
        // Hide any previous result or error
        resultSection.hidden = true;
        errorSection.hidden  = true;

        // ----------------------------------------------------------
        // STEP 3: Build the request data (payload)
        // ----------------------------------------------------------
        // This is the JSON data we'll send to FastAPI.
        // It must match the HouseFeatures Pydantic model in main.py.

        const requestData = {
            area:      area,
            bedrooms:  bedrooms,
            bathrooms: bathrooms,
            stories:   stories,
            parking:   parking
        };

        // ----------------------------------------------------------
        // STEP 4: Send POST request to FastAPI using fetch()
        // ----------------------------------------------------------
        // fetch() sends an HTTP request and returns a Promise.
        // 'await' pauses execution until the response comes back.
        // 'try...catch' handles any errors that might occur.

        try {
            // -------------------------------------------------------
            // THE FETCH REQUEST
            // -------------------------------------------------------
            // fetch(url, options) sends a request to the given URL.
            //
            // URL: "/predict"
            //   This is a relative URL — the same host as the page.
            //   Since the page is at http://localhost:8000,
            //   this calls http://localhost:8000/predict
            //
            // Options:
            //   method: "POST"
            //     → Send a POST request (we're sending data, not just reading)
            //
            //   headers: { "Content-Type": "application/json" }
            //     → Tell the server we're sending JSON data
            //
            //   body: JSON.stringify(requestData)
            //     → Convert our JavaScript object to a JSON string
            //     → { area: 7420, bedrooms: 4, ... }
            //     → becomes '{"area":7420,"bedrooms":4,...}'

            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });

            // -------------------------------------------------------
            // STEP 5: Parse the response
            // -------------------------------------------------------
            // response.json() converts the JSON response body to a JS object.
            // We await it because it's also asynchronous.

            const data = await response.json();

            // -------------------------------------------------------
            // STEP 6: Check if the request was successful
            // -------------------------------------------------------
            // response.ok is true for HTTP status codes 200-299 (success)
            // response.ok is false for 400, 422, 500, etc. (errors)

            if (response.ok) {
                // SUCCESS — show the prediction
                displayResult(data);
            } else {
                // SERVER ERROR — API returned an error response
                // FastAPI puts error details in data.detail
                const errMessage = data.detail || "Prediction failed. Please try again.";
                displayError(errMessage);
            }

        } catch (networkError) {
            // NETWORK ERROR — couldn't reach the server at all
            // This happens if the server isn't running
            console.error("Network error:", networkError);
            displayError(
                "Could not connect to the server. Make sure FastAPI is running " +
                "with: uvicorn app.main:app --reload"
            );
        } finally {
            // 'finally' always runs, whether there was an error or not
            // Turn off the loading state
            setLoadingState(false);
        }
    }

    // ============================================================
    // HELPER FUNCTION: validateInputs()
    // ============================================================
    // Checks that all input values are valid before sending to API.
    // Returns true if valid, false if not.

    function validateInputs({ area, bedrooms, bathrooms, stories, parking }) {
        let isValid = true;
        
        // Define validation rules for each field
        const validations = [
            {
                element: areaInput,
                value: area,
                check: (v) => !isNaN(v) && v > 0,
                message: "Area must be a positive number."
            },
            {
                element: bedroomsInput,
                value: bedrooms,
                check: (v) => !isNaN(v) && v >= 1 && v <= 20,
                message: "Bedrooms must be between 1 and 20."
            },
            {
                element: bathroomsInput,
                value: bathrooms,
                check: (v) => !isNaN(v) && v >= 1 && v <= 10,
                message: "Bathrooms must be between 1 and 10."
            },
            {
                element: storiesInput,
                value: stories,
                check: (v) => !isNaN(v) && v >= 1 && v <= 10,
                message: "Stories must be between 1 and 10."
            },
            {
                element: parkingInput,
                value: parking,
                check: (v) => !isNaN(v) && v >= 0 && v <= 10,
                message: "Parking must be between 0 and 10."
            }
        ];

        // First, collect all errors
        const errors = [];
        
        validations.forEach(function({ element, value, check, message }) {
            if (!check(value)) {
                element.classList.add("is-invalid");  // red border
                errors.push(message);
                isValid = false;
            }
        });

        // Show the first error if any
        if (!isValid) {
            displayError("Please fix: " + errors[0]);
        }

        return isValid;
    }

    // ============================================================
    // HELPER FUNCTION: setLoadingState()
    // ============================================================
    // Switches button between normal and loading state.

    function setLoadingState(isLoading) {
        if (isLoading) {
            predictBtn.disabled = true;          // disable the button
            predictBtn.classList.add("loading"); // add loading CSS class
        } else {
            predictBtn.disabled = false;
            predictBtn.classList.remove("loading");
        }
    }

    // ============================================================
    // HELPER FUNCTION: displayResult()
    // ============================================================
    // Shows the prediction result on the page.
    // 'data' is the JSON object from the API response.

    function displayResult(data) {
        // data.formatted_price → e.g., "$4,850,000"
        // data.predicted_price → e.g., 4850000.0
        
        // Animate the price counting up
        animatePrice(data.predicted_price);
        
        // Update the meta text
        resultMetaText.textContent = "Powered by Random Forest · Trained on historical housing data";
        
        // Show the result section
        resultSection.hidden = false;
        
        // Hide any error
        errorSection.hidden = true;
        
        // Scroll to the result smoothly
        resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
        
        console.log("Prediction successful:", data);
    }

    // ============================================================
    // HELPER FUNCTION: animatePrice()
    // ============================================================
    // Animates the price from 0 up to the final value.
    // This creates a nice counting effect.

    function animatePrice(finalPrice) {
        const duration = 800;            // animation duration in ms
        const startTime = Date.now();    // when animation started
        const startValue = 0;

        function updatePrice() {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1); // 0 to 1
            
            // Ease-out curve: fast at start, slow at end
            const easedProgress = 1 - Math.pow(1 - progress, 3);
            
            const currentValue = Math.floor(startValue + (finalPrice - startValue) * easedProgress);
            
            // Format number with commas and dollar sign
            // toLocaleString("en-US") formats: 4850000 → "4,850,000"
            predictedPrice.textContent = "$" + currentValue.toLocaleString("en-US");
            
            // Continue animation until done
            if (progress < 1) {
                requestAnimationFrame(updatePrice);
            }
        }

        requestAnimationFrame(updatePrice);
    }

    // ============================================================
    // HELPER FUNCTION: displayError()
    // ============================================================
    // Shows an error message on the page.

    function displayError(message) {
        errorText.textContent = message;
        errorSection.hidden = false;
        resultSection.hidden = true;
        
        // Scroll to error
        errorSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
        
        console.error("Error:", message);
    }

    // ============================================================
    // LOG THAT SCRIPT LOADED SUCCESSFULLY
    // ============================================================
    console.log("✅ House Price Prediction script loaded successfully");
    console.log("📡 API endpoint: POST /predict");

});  // end DOMContentLoaded

// ============================================================
// HOW THE DATA FLOWS (for your reference):
// ============================================================
//
//  1. User types values into the form fields
//  2. User clicks "Predict Price"
//  3. handlePrediction() runs
//  4. validateInputs() checks all values are valid
//  5. fetch() sends POST request to http://localhost:8000/predict
//     with body: {"area": 7420, "bedrooms": 4, ...}
//  6. FastAPI receives the request in main.py
//  7. Pydantic validates the input
//  8. FastAPI loads model.pkl and calls model.predict()
//  9. FastAPI returns: {"predicted_price": 4850000, ...}
// 10. fetch() receives the response
// 11. displayResult() shows the price on the page
// 12. User sees: "$4,850,000"
// ============================================================
