// ============================================================
// static/script.js  —  House Price Prediction (v2 — Indore)
// ============================================================
//
// WHAT CHANGED FROM THE OLD VERSION:
//
//  1. Location dropdown populated dynamically
//     OLD: No location field at all.
//     NEW: On page load, fetch("/locations") is called.
//          The API returns all unique locality names from housing.csv.
//          These are inserted as <option> elements inside the
//          #location <select>. Nothing is hardcoded.
//
//  2. Form fields changed
//     OLD: area, bedrooms, bathrooms, stories, parking (all number inputs)
//     NEW: location (select), size (number), bhk (select),
//          floors (select), furnishing (select)
//
//  3. Request body changed
//     OLD: { area, bedrooms, bathrooms, stories, parking }
//     NEW: { location, size, bhk, floors, furnishing }
//
//  4. Price formatting changed
//     OLD: Displayed as "$4,850,000"  (US Dollar format)
//     NEW: Displayed as "₹1,25,00,000" (Indian Rupee format)
//          Indian numbering: last 3 digits, then groups of 2.
//          The server sends formatted_price in this format already.
//          We animate from ₹0 to the final value using the raw number.
//
//  5. Result section updated
//     OLD: Just showed the price.
//     NEW: Shows price + location + size, BHK, floors, furnishing breakdown.
//
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

  // ── Get references to all HTML elements ────────────────

  // Input fields
  const locationSelect  = document.getElementById("location");
  const sizeInput       = document.getElementById("size");
  const bhkSelect       = document.getElementById("bhk");
  const floorsSelect    = document.getElementById("floors");
  const furnishingSelect= document.getElementById("furnishing");

  // Button
  const predictBtn      = document.getElementById("predict-btn");

  // Result elements
  const resultSection   = document.getElementById("result-section");
  const predictedPrice  = document.getElementById("predicted-price");

  // Result detail spans
  const resLocation     = document.getElementById("res-location");
  const resSize         = document.getElementById("res-size");
  const resBhk          = document.getElementById("res-bhk");
  const resFloors       = document.getElementById("res-floors");
  const resFurnishing   = document.getElementById("res-furnishing");

  // Error
  const errorSection    = document.getElementById("error-section");
  const errorText       = document.getElementById("error-text");

  // ── Step 1: Populate Location dropdown on page load ────
  //
  // This is the key new feature in v2.
  // We call GET /locations which returns:
  //   { "locations": ["Amli Kheda", "Anurag Nagar", ... ] }
  // We then create an <option> element for each location and
  // append it to the #location <select>.
  // No location names are hardcoded in HTML or JS.

  async function loadLocations() {
    try {
      const response = await fetch("/locations");

      if (!response.ok) {
        throw new Error("Could not load locations from server.");
      }

      const data = await response.json();
      // data.locations = ["Amli Kheda", "Anurag Nagar", ...]

      // Clear the "Loading localities…" placeholder
      locationSelect.innerHTML = "";

      // Add a blank disabled option as the default prompt
      const defaultOpt = document.createElement("option");
      defaultOpt.value    = "";
      defaultOpt.disabled = true;
      defaultOpt.selected = true;
      defaultOpt.textContent = "Select a locality…";
      locationSelect.appendChild(defaultOpt);

      // Add one <option> for each location from the API
      data.locations.forEach(function(loc) {
        const opt = document.createElement("option");
        opt.value       = loc;   // the value sent to the API
        opt.textContent = loc;   // what the user sees in the dropdown
        locationSelect.appendChild(opt);
      });

      console.log(`✅ Loaded ${data.locations.length} localities into dropdown`);

    } catch (err) {
      // If locations can't be loaded, show a fallback message
      locationSelect.innerHTML = "";
      const errOpt = document.createElement("option");
      errOpt.value    = "";
      errOpt.disabled = true;
      errOpt.selected = true;
      errOpt.textContent = "⚠️ Could not load localities — is the server running?";
      locationSelect.appendChild(errOpt);
      console.error("Failed to load locations:", err);
    }
  }

  // Call it immediately when the page loads
  loadLocations();

  // ── Step 2: Clear validation state when user interacts ─

  // Remove red border from selects and inputs when user changes them
  [locationSelect, bhkSelect, floorsSelect, furnishingSelect].forEach(function(el) {
    el.addEventListener("change", function() {
      this.classList.remove("is-invalid");
    });
  });

  sizeInput.addEventListener("input", function() {
    this.classList.remove("is-invalid");
  });

  // ── Step 3: Predict button click handler ───────────────

  predictBtn.addEventListener("click", handlePrediction);

  // Also allow pressing Enter in the size field to trigger prediction
  sizeInput.addEventListener("keypress", function(e) {
    if (e.key === "Enter") handlePrediction();
  });

  // ── Main prediction function ───────────────────────────

  async function handlePrediction() {

    // Read all values from the form
    const location   = locationSelect.value;
    const size       = parseInt(sizeInput.value);
    const bhk        = parseInt(bhkSelect.value);
    const floors     = floorsSelect.value;
    const furnishing = furnishingSelect.value;

    // Validate before sending to API
    if (!validateInputs({ location, size, bhk, floors, furnishing })) {
      return; // Stop if validation failed
    }

    // Show loading state
    setLoadingState(true);
    resultSection.hidden = true;
    errorSection.hidden  = true;

    // Build request payload
    // Keys must match exactly what HouseFeatures in main.py expects
    const requestData = {
      location:   location,
      size:       size,
      bhk:        bhk,
      floors:     floors,       // String: "Ground", "1", "2", "3", "4"
      furnishing: furnishing    // String: "Unfurnished", "Semi Furnished", "Furnished"
    };

    try {
      // ── Send POST request to /predict ─────────────────
      const response = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(requestData)
      });

      const data = await response.json();

      if (response.ok) {
        // Success — display result
        displayResult(data);
      } else {
        // API returned an error (e.g. 422 validation error, 503 model not ready)
        const msg = data.detail || "Prediction failed. Please try again.";
        displayError(typeof msg === "object" ? JSON.stringify(msg) : msg);
      }

    } catch (networkErr) {
      // Network error — server not running or unreachable
      displayError(
        "Cannot reach the server. Make sure FastAPI is running: " +
        "uvicorn app.main:app --reload"
      );
      console.error("Network error:", networkErr);
    } finally {
      setLoadingState(false);
    }
  }

  // ── Validation ─────────────────────────────────────────

  function validateInputs({ location, size, bhk, floors, furnishing }) {
    let valid = true;
    const errors = [];

    if (!location) {
      locationSelect.classList.add("is-invalid");
      errors.push("Please select a location.");
      valid = false;
    }
    if (isNaN(size) || size < 100) {
      sizeInput.classList.add("is-invalid");
      errors.push("Size must be a positive number (e.g. 1200).");
      valid = false;
    }
    if (!bhk || isNaN(bhk)) {
      bhkSelect.classList.add("is-invalid");
      errors.push("Please select BHK.");
      valid = false;
    }
    if (!floors) {
      floorsSelect.classList.add("is-invalid");
      errors.push("Please select the number of floors.");
      valid = false;
    }
    if (!furnishing) {
      furnishingSelect.classList.add("is-invalid");
      errors.push("Please select furnishing status.");
      valid = false;
    }

    if (!valid) {
      displayError(errors[0]); // Show the first error
    }
    return valid;
  }

  // ── Loading state helpers ──────────────────────────────

  function setLoadingState(isLoading) {
    predictBtn.disabled = isLoading;
    if (isLoading) {
      predictBtn.classList.add("loading");
    } else {
      predictBtn.classList.remove("loading");
    }
  }

  // ── Display prediction result ──────────────────────────

  function displayResult(data) {
    // data structure (from PredictionResponse in main.py):
    //   data.predicted_price  → 19940000  (raw number)
    //   data.formatted_price  → "₹1,99,40,000"  (Indian format string)
    //   data.location         → "Vijay Nagar"
    //   data.size             → 1200
    //   data.bhk              → 3
    //   data.floors           → "2"
    //   data.furnishing       → "Semi Furnished"

    // Animate the price counting up from 0 to the final value
    // The server already has Indian formatting; we recreate it client-side
    // for the animation, then show the server's formatted string at the end.
    animatePriceINR(data.predicted_price, data.formatted_price);

    // Fill in the detail breakdown
    resLocation.textContent   = data.location;
    resSize.textContent       = data.size + " sq ft";
    resBhk.textContent        = data.bhk + " BHK";

    // Make floors label human-friendly
    const floorLabel = {
      "Ground": "Ground floor only",
      "1": "G+1 (1 floor above)",
      "2": "G+2 (2 floors above)",
      "3": "G+3 (3 floors above)",
      "4": "G+4 (4 floors above)"
    };
    resFloors.textContent     = floorLabel[data.floors] || data.floors;
    resFurnishing.textContent = data.furnishing;

    // Show the result card
    resultSection.hidden = false;
    errorSection.hidden  = true;

    // Smoothly scroll to show the result
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ── Indian Rupee price animation ───────────────────────
  //
  // Animates the price from ₹0 to the final value.
  // Uses the Indian number formatting logic (same as server-side):
  //   Last 3 digits as one group, then groups of 2 from the right.
  //   e.g. 19940000 → "₹1,99,40,000"
  //
  // At the very end, we snap to the server's pre-formatted string
  // to guarantee pixel-perfect formatting.

  function animatePriceINR(finalPrice, finalFormattedPrice) {
    const duration  = 900;   // ms
    const startTime = Date.now();

    function formatINR(n) {
      // Convert integer n to Indian formatted string
      n = Math.floor(n);
      const s = String(n);
      if (s.length <= 3) return "₹" + s;
      const last3 = s.slice(-3);
      let rest = s.slice(0, -3);
      const pairs = [];
      while (rest.length > 2) {
        pairs.unshift(rest.slice(-2));
        rest = rest.slice(0, -2);
      }
      if (rest) pairs.unshift(rest);
      return "₹" + pairs.join(",") + "," + last3;
    }

    function tick() {
      const elapsed  = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic: fast at start, slow at end
      const eased    = 1 - Math.pow(1 - progress, 3);
      const current  = Math.floor(finalPrice * eased);

      predictedPrice.textContent = formatINR(current);

      if (progress < 1) {
        requestAnimationFrame(tick);
      } else {
        // Snap to server-side formatted value at the very end
        predictedPrice.textContent = finalFormattedPrice;
      }
    }

    requestAnimationFrame(tick);
  }

  // ── Display error ──────────────────────────────────────

  function displayError(message) {
    errorText.textContent  = message;
    errorSection.hidden    = false;
    resultSection.hidden   = true;
    errorSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    console.error("Error:", message);
  }

  console.log("✅ House Price Prediction v2 script loaded");
  console.log("📍 Locality list: GET /locations");
  console.log("📡 Predict API : POST /predict");

}); // end DOMContentLoaded

// ============================================================
// COMPLETE DATA FLOW (v2):
//
//  Page load
//    → script.js calls GET /locations
//    → main.py reads housing.csv, returns sorted locality list
//    → script.js fills the Location dropdown
//
//  User fills form → clicks Predict
//    → script.js validates all 5 fields
//    → fetch POST /predict with:
//       { location, size, bhk, floors, furnishing }
//    → main.py validates with Pydantic
//    → main.py builds pd.DataFrame([{...}])
//    → pipeline.predict(df) runs:
//         ColumnTransformer: OneHotEncode location/floors/furnishing
//                            passthrough size/bhk
//         RandomForestRegressor: returns price
//    → main.py formats price as ₹X,XX,XX,XXX
//    → main.py returns PredictionResponse JSON
//    → script.js animates price counter
//    → script.js fills in location + details breakdown
//    → User sees ₹1,99,40,000 with full breakdown
// ============================================================
