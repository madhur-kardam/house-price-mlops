# ============================================================
# app/main.py  —  FastAPI backend (updated for new dataset)
# ============================================================
#
# WHAT CHANGED FROM THE OLD VERSION:
#
#   OLD HouseFeatures model had:
#       area (float), bedrooms (int), bathrooms (int),
#       stories (int), parking (int)
#
#   NEW HouseFeatures model has:
#       location (str), size (int), bhk (int),
#       floors (str), furnishing (str)
#
#   OLD prediction used numpy array:
#       features = np.array([[area, bedrooms, ...]])
#       model.predict(features)
#
#   NEW prediction uses Pandas DataFrame:
#       df = pd.DataFrame([{...}])
#       model.predict(df)
#   → Required because the pipeline's ColumnTransformer
#     needs column NAMES to know which columns to encode.
#     A plain numpy array has no column names.
#
#   NEW /locations endpoint:
#       Returns the sorted list of all unique locations
#       read directly from housing.csv.
#       The frontend dropdown is populated from this endpoint
#       automatically — no hardcoding needed.
#
# HOW TO RUN:
#   uvicorn app.main:app --reload
# ============================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Create FastAPI app ─────────────────────────────────────

app = FastAPI(
    title="House Price Prediction API — Indore",
    description="Predict house prices across Indore localities using Random Forest",
    version="2.0.0"
)

# ── CORS middleware ────────────────────────────────────────
# Allows the browser frontend to call this API without security blocks.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files and HTML templates ───────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Constants — must match train.py exactly ───────────────

DATA_PATH  = "data/housing.csv"
MODEL_PATH = "app/model.pkl"

# The exact column names the pipeline was trained with.
# Order matters — must match the order used in train.py.
FEATURE_COLUMNS = ["location", "size", "bhk", "floors", "furnishing"]

# Valid categorical values — used for API validation
VALID_FLOORS      = {"Ground", "1", "2", "3", "4"}
VALID_FURNISHINGS = {"Unfurnished", "Semi Furnished", "Furnished"}

# ── Load model on startup ──────────────────────────────────

model        = None
model_loaded = False

def load_model() -> bool:
    global model, model_loaded
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Run 'python train.py' first.")
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model_loaded = True
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

load_model()

# ── Load unique locations from CSV ────────────────────────
# This runs once at startup and builds the sorted list of locations
# that the /locations endpoint will return to the frontend dropdown.

def load_locations() -> list[str]:
    if not os.path.exists(DATA_PATH):
        logger.warning(f"Dataset not found at {DATA_PATH}; location list will be empty.")
        return []
    try:
        df = pd.read_csv(DATA_PATH, usecols=["location"])
        locs = sorted(df["location"].dropna().unique().tolist())
        logger.info(f"✅ Loaded {len(locs)} unique locations from {DATA_PATH}")
        return locs
    except Exception as e:
        logger.error(f"❌ Could not load locations: {e}")
        return []

LOCATIONS: list[str] = load_locations()

# ── Pydantic models ────────────────────────────────────────
#
# HouseFeatures defines exactly what JSON the /predict endpoint expects.
# Pydantic automatically validates types and raises a 422 error with a
# helpful message if the client sends wrong data.

class HouseFeatures(BaseModel):
    """Input schema for house price prediction."""

    location: str = Field(
        ...,
        description="Locality name, e.g. 'Vijay Nagar'",
        example="Vijay Nagar"
    )
    size: int = Field(
        ...,
        ge=100,
        le=10000,
        description="Plot / house size in square feet",
        example=1200
    )
    bhk: int = Field(
        ...,
        ge=1,
        le=5,
        description="Number of BHK (bedrooms, hall, kitchen)",
        example=3
    )
    floors: str = Field(
        ...,
        description="Floor level: Ground, 1, 2, 3, or 4",
        example="2"
    )
    furnishing: str = Field(
        ...,
        description="Furnishing status: Unfurnished, Semi Furnished, or Furnished",
        example="Semi Furnished"
    )


class PredictionResponse(BaseModel):
    """Response schema returned by /predict."""
    predicted_price:   float  = Field(description="Predicted price in INR (raw number)")
    formatted_price:   str    = Field(description="Price in Indian format, e.g. ₹32,50,000")
    location:          str    = Field(description="Locality name")
    size:              int    = Field(description="Size in sq ft")
    bhk:               int    = Field(description="BHK count")
    floors:            str    = Field(description="Floor level")
    furnishing:        str    = Field(description="Furnishing status")
    model_version:     str    = Field(description="Model identifier")


# ── Helper: Indian currency formatter ─────────────────────
#
# Indian numbering groups digits differently from the Western system:
#   Western : 12,345,678  (groups of 3 from the right)
#   Indian  :  1,23,45,678  (last 3 digits, then groups of 2)
#
# Examples:
#   3250000  → ₹32,50,000
#   12500000 → ₹1,25,00,000

def format_inr(amount: float) -> str:
    amount = int(round(amount))
    s = str(amount)
    # Split into last 3 digits and the rest
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest  = s[:-3]
    # Group the rest in pairs from the right
    pairs = []
    while len(rest) > 2:
        pairs.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        pairs.append(rest)
    pairs.reverse()
    return "₹" + ",".join(pairs) + "," + last3


# ── API Endpoints ──────────────────────────────────────────

# ── GET /  →  Serve the HTML frontend ─────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web page."""
    return templates.TemplateResponse(
    request=request,
    name="index.html"
)


# ── GET /health  →  Health check ──────────────────────────

@app.get("/health")
async def health_check():
    """
    Returns API status and whether the model is loaded.

    Example response:
    {
        "status": "healthy",
        "model_loaded": true,
        "locations_count": 142
    }
    """
    return {
        "status":          "healthy" if model_loaded else "degraded",
        "model_loaded":    model_loaded,
        "locations_count": len(LOCATIONS),
        "message": (
            "API ready"
            if model_loaded
            else "Model not loaded — run python train.py"
        )
    }


# ── GET /locations  →  Return all unique locations ────────
#
# The frontend JavaScript calls this endpoint on page load
# to populate the Location dropdown automatically.
# No locations are hardcoded in HTML or JS.

@app.get("/locations")
async def get_locations():
    """
    Returns sorted list of all unique localities from housing.csv.

    Example response:
    {
        "locations": ["Anurag Nagar", "Bhicholi Mardana", "Vijay Nagar", ...]
    }
    """
    return {"locations": LOCATIONS}


# ── POST /predict  →  Main prediction endpoint ────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(house: HouseFeatures):
    """
    Predicts house price for the given property details.

    KEY DIFFERENCE FROM OLD VERSION:
    The old version built a numpy array for prediction:
        features = np.array([[area, bedrooms, bathrooms, stories, parking]])
        model.predict(features)

    The new version builds a Pandas DataFrame:
        df = pd.DataFrame([{...}])
        model.predict(df)

    This is REQUIRED because the ColumnTransformer inside the pipeline
    needs column names (like "location", "floors") to know which
    transformer to apply to which column.
    A numpy array has no column names — it would cause an error.

    Example request body:
    {
        "location": "Vijay Nagar",
        "size": 1200,
        "bhk": 3,
        "floors": "2",
        "furnishing": "Semi Furnished"
    }

    Example response:
    {
        "predicted_price": 19940000,
        "formatted_price": "₹1,99,40,000",
        "location": "Vijay Nagar",
        ...
    }
    """

    # Guard: model must be loaded
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Please run 'python train.py' first."
        )

    # Validate categorical values
    if house.floors not in VALID_FLOORS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid floor '{house.floors}'. Must be one of: {sorted(VALID_FLOORS)}"
        )
    if house.furnishing not in VALID_FURNISHINGS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid furnishing '{house.furnishing}'. Must be one of: {sorted(VALID_FURNISHINGS)}"
        )

    logger.info(
        f"Predict request — location={house.location}, size={house.size}, "
        f"bhk={house.bhk}, floors={house.floors}, furnishing={house.furnishing}"
    )

    # ── Build a single-row DataFrame ──────────────────────
    # Column names must EXACTLY match what the pipeline was trained on.
    # The order of keys here must match FEATURE_COLUMNS in train.py.
    input_df = pd.DataFrame([{
        "location":   house.location,
        "size":       house.size,
        "bhk":        house.bhk,
        "floors":     house.floors,
        "furnishing": house.furnishing
    }])

    # ── Get prediction ────────────────────────────────────
    predicted_price = float(model.predict(input_df)[0])
    fmt_price       = format_inr(predicted_price)

    logger.info(f"Prediction: {fmt_price} for {house.location}")

    return PredictionResponse(
        predicted_price = predicted_price,
        formatted_price = fmt_price,
        location        = house.location,
        size            = house.size,
        bhk             = house.bhk,
        floors          = house.floors,
        furnishing      = house.furnishing,
        model_version   = "RandomForestRegressor-v2"
    )


# ── GET /features  →  Feature metadata ────────────────────

@app.get("/features")
async def get_feature_info():
    """
    Returns metadata about accepted input features.
    Useful for frontend validation and documentation.
    """
    return {
        "features": [
            {
                "name":        "location",
                "type":        "categorical (string)",
                "description": "Indore locality name",
                "options":     "See /locations endpoint"
            },
            {
                "name":        "size",
                "type":        "integer",
                "description": "Plot/house size in square feet",
                "allowed":     [500, 600, 750, 1000, 1200, 1500]
            },
            {
                "name":        "bhk",
                "type":        "integer",
                "description": "BHK count",
                "allowed":     [1, 2, 3]
            },
            {
                "name":        "floors",
                "type":        "categorical (string)",
                "description": "Floor level",
                "allowed":     ["Ground", "1", "2", "3", "4"]
            },
            {
                "name":        "furnishing",
                "type":        "categorical (string)",
                "description": "Furnishing status",
                "allowed":     ["Unfurnished", "Semi Furnished", "Furnished"]
            }
        ],
        "target": "price (INR)"
    }
