# ============================================================
# app/main.py — This is the FastAPI backend (the server)
# ============================================================
# What this file does:
#   1. Loads the trained ML model from model.pkl
#   2. Creates API endpoints (URLs) that the frontend can call
#   3. Receives house feature data from the frontend
#   4. Passes data to the model to get a price prediction
#   5. Returns the prediction back to the frontend
#
# How to run:
#   uvicorn app.main:app --reload
#
# After running, open:
#   http://localhost:8000       → the web frontend
#   http://localhost:8000/docs  → interactive API documentation
# ============================================================

# --- IMPORTS ---

from fastapi import FastAPI, Request    # FastAPI: creates the web server and API
from fastapi.responses import HTMLResponse  # sends HTML pages as responses
from fastapi.staticfiles import StaticFiles  # serves static files (CSS, JS)
from fastapi.templating import Jinja2Templates  # serves HTML templates
from fastapi.middleware.cors import CORSMiddleware  # handles cross-origin requests
from pydantic import BaseModel, Field   # validates and parses request data
import pickle                           # loads saved model files
import numpy as np                     # for array operations
import os                              # for file path operations
import logging                         # for logging messages

# Set up logging so we can see what's happening in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# STEP 1: CREATE THE FASTAPI APPLICATION
# ============================================================

# FastAPI() creates the main application object
# title and description appear in the auto-generated docs at /docs
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a trained Random Forest model",
    version="1.0.0"
)

# ============================================================
# STEP 2: SET UP CORS (Cross-Origin Resource Sharing)
# ============================================================
# CORS is a browser security feature.
# Without this, your browser might block requests from the frontend to the API.
# By allowing all origins (*), we let any webpage call our API.
# For production apps, you'd restrict this to specific domains.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (any website can call this API)
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # Allow all headers
)

# ============================================================
# STEP 3: SERVE STATIC FILES AND TEMPLATES
# ============================================================

# Mount the static folder so FastAPI serves CSS and JS files
# When browser requests /static/style.css, FastAPI serves static/style.css
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates — this tells FastAPI where HTML files are
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

# ============================================================
# STEP 4: LOAD THE TRAINED MODEL
# ============================================================

# This runs once when the server starts
# It loads the model from disk into memory

MODEL_PATH = "app/model.pkl"
model = None  # We'll set this below

def load_model():
    """Load the trained ML model from disk."""
    global model  # 'global' means we're modifying the variable outside this function
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        logger.error("Please run 'python train.py' first to train the model!")
        return False
    
    try:
        with open(MODEL_PATH, "rb") as f:  # "rb" = read binary mode
            model = pickle.load(f)         # Load the model from disk
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

# Load the model when the server starts
model_loaded = load_model()

# ============================================================
# STEP 5: DEFINE THE DATA MODEL (Pydantic Schema)
# ============================================================
# Pydantic is a library that validates data.
# HouseFeatures defines exactly what data the /predict endpoint expects.
# If someone sends wrong data types, FastAPI automatically returns an error.

class HouseFeatures(BaseModel):
    """
    This class defines the input data structure for predictions.
    Each field has a type (float/int) and validation constraints.
    """
    area: float = Field(
        ...,                    # ... means this field is required
        gt=0,                   # gt = greater than (area must be > 0)
        description="House area in square feet",
        example=7420
    )
    bedrooms: int = Field(
        ...,
        ge=1,                   # ge = greater than or equal to (at least 1 bedroom)
        le=20,                  # le = less than or equal to (max 20 bedrooms)
        description="Number of bedrooms",
        example=4
    )
    bathrooms: int = Field(
        ...,
        ge=1,
        le=10,
        description="Number of bathrooms",
        example=2
    )
    stories: int = Field(
        ...,
        ge=1,
        le=10,
        description="Number of stories/floors",
        example=3
    )
    parking: int = Field(
        ...,
        ge=0,                   # parking can be 0 (no parking)
        le=10,
        description="Number of parking spots",
        example=2
    )

# Response model — defines what our API sends back
class PredictionResponse(BaseModel):
    """Defines the structure of the prediction response."""
    predicted_price: float = Field(description="Predicted house price in USD")
    currency: str = Field(default="USD", description="Currency of the prediction")
    model_version: str = Field(description="Name of the model used")
    formatted_price: str = Field(description="Price formatted with commas and dollar sign")

# ============================================================
# STEP 6: DEFINE API ENDPOINTS (Routes)
# ============================================================

# --- ENDPOINT 1: Serve the Frontend ---
# GET / → returns the main HTML page
# When user opens http://localhost:8000, this function runs

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML frontend page.
    
    'request: Request' is required by Jinja2 templates — it contains
    information about the HTTP request (headers, client IP, etc.)
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# --- ENDPOINT 2: Health Check ---
# GET /health → tells you if the API is running and model is loaded

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Returns whether the API is running and model is loaded.
    
    Example response:
    {
        "status": "healthy",
        "model_loaded": true,
        "message": "API is running successfully"
    }
    """
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "message": "API is running successfully" if model_loaded else "Model not loaded. Run train.py first."
    }


# --- ENDPOINT 3: Predict House Price ---
# POST /predict → receives house features, returns predicted price
# This is the main endpoint our frontend calls

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(house: HouseFeatures):
    """
    Predicts house price based on input features.
    
    This endpoint:
    1. Receives house features as JSON in the request body
    2. Validates the data (Pydantic does this automatically)
    3. Creates a numpy array from the features
    4. Passes it to the trained model
    5. Returns the predicted price
    
    Example request body:
    {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "parking": 2
    }
    
    Example response:
    {
        "predicted_price": 4850000.0,
        "currency": "USD",
        "model_version": "RandomForestRegressor",
        "formatted_price": "$4,850,000"
    }
    """
    
    # Check if model is loaded
    if model is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,  # 503 = Service Unavailable
            detail="Model not loaded. Please run 'python train.py' first."
        )
    
    # Log the incoming request (so we can see it in the console)
    logger.info(f"Prediction request: area={house.area}, bedrooms={house.bedrooms}, "
                f"bathrooms={house.bathrooms}, stories={house.stories}, parking={house.parking}")
    
    # Create a numpy array from the input features
    # The model expects a 2D array: [[feature1, feature2, feature3, feature4, feature5]]
    # shape = (1, 5) → 1 sample, 5 features
    features = np.array([[
        house.area,
        house.bedrooms,
        house.bathrooms,
        house.stories,
        house.parking
    ]])
    
    # Get prediction from the model
    # .predict() returns an array like [4850000.0]
    # [0] gets the first (and only) element
    predicted_price = float(model.predict(features)[0])
    
    # Format the price nicely (e.g., 4850000 → "$4,850,000")
    formatted_price = f"${predicted_price:,.0f}"
    
    logger.info(f"Predicted price: {formatted_price}")
    
    # Return the response
    return PredictionResponse(
        predicted_price=predicted_price,
        currency="USD",
        model_version="RandomForestRegressor",
        formatted_price=formatted_price
    )


# --- ENDPOINT 4: Get Feature Info ---
# GET /features → returns info about what features the model accepts

@app.get("/features")
async def get_feature_info():
    """
    Returns information about the features the model accepts.
    Useful for frontend validation.
    """
    return {
        "features": [
            {"name": "area", "type": "float", "description": "House area in sq ft", "min": 1},
            {"name": "bedrooms", "type": "int", "description": "Number of bedrooms", "min": 1, "max": 20},
            {"name": "bathrooms", "type": "int", "description": "Number of bathrooms", "min": 1, "max": 10},
            {"name": "stories", "type": "int", "description": "Number of floors", "min": 1, "max": 10},
            {"name": "parking", "type": "int", "description": "Number of parking spots", "min": 0, "max": 10},
        ],
        "target": "price (USD)"
    }
