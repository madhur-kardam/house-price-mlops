🏠 House Price Prediction — Indore MLOps Project (v2)
Predicts house prices across 142 Indore localities using a Random Forest model
trained on real locality land rates.
Built with: Python · scikit-learn · FastAPI · MLflow · Docker · HTML/CSS/JS
---
📊 Dataset — `data/housing.csv`
Column Reference
Column	Type	Allowed Values	Description
`location`	string	142 Indore localities	Locality / area name
`size`	integer	500, 600, 750, 1000, 1200, 1500	Plot size in square feet
`bhk`	integer	1, 2, 3	Bedrooms + Hall + Kitchen
`floors`	string	Ground, 1, 2, 3, 4	Number of floors built above ground
`furnishing`	string	Unfurnished, Semi Furnished, Furnished	Interior furnishing level
`price`	integer	—	House price in INR (target variable)
Sample rows
```
location,size,bhk,floors,furnishing,price
Vijay Nagar,1200,3,2,Semi Furnished,19940000
Nipania,1000,2,Ground,Unfurnished,8748000
Sukhliya,1500,3,4,Furnished,56430000
```
Price formula used to generate the dataset
```
Land Cost        = size × locality rate per sq ft
Built-up Area    = size × floor multiplier
                   (Ground=1, 1=2, 2=3, 3=4, 4=5)
Construction Cost= built-up area × furnishing rate
                   (Unfurnished=1500, Semi=2000, Furnished=3200)
Total            = Land Cost + Construction Cost
BHK premium      = ×1.00 / ×1.05 / ×1.10  (1/2/3 BHK)
Final Price      = Total × BHK premium  (rounded to ₹10,000)
```
BHK constraints
Size (sq ft)	Allowed BHK
500, 600	1 only
750	1 or 2
1000	1, 2, or 3
1200, 1500	2 or 3
---
🗂️ Project Structure
```
house-price-mlops/
│
├── app/
│   ├── __init__.py      ← Makes app/ a Python package
│   ├── main.py          ← FastAPI server (all API endpoints)
│   └── model.pkl        ← Trained pipeline (created by train.py)
│
├── templates/
│   └── index.html       ← Web frontend (served by FastAPI)
│
├── static/
│   ├── style.css        ← Dark luxury UI styling
│   └── script.js        ← JavaScript (fetch API, dropdown, INR format)
│
├── data/
│   └── housing.csv      ← Indore locality dataset (4,544 rows)
│
├── train.py             ← Model training script
├── requirements.txt     ← Python dependencies
├── Dockerfile           ← Docker container definition
└── README.md
```
---
💻 Setup — Windows 11
Required software
Software	Download	Verify with
Python 3.11	https://python.org/downloads	`python --version`
VS Code	https://code.visualstudio.com	—
Git	https://git-scm.com/download/win	`git --version`
Docker Desktop	https://docker.com/products/docker-desktop	`docker --version`
> ⚠️ When installing Python, check **"Add Python to PATH"** before clicking Install.
---
Step 1 — Create project folder
```cmd
cd Desktop
mkdir house-price-mlops
cd house-price-mlops
```
Step 2 — Create subfolder structure
```cmd
mkdir app templates static data
```
Step 3 — Place files
Copy all project files into the correct folders.
Place `housing.csv` inside the `data/` folder.
Step 4 — Create and activate virtual environment
```cmd
python -m venv venv
venv\Scripts\activate
```
You will see `(venv)` at the start of the command line.
> 💡 Run `venv\Scripts\activate` every time you open a new terminal.
Step 5 — Install dependencies
```cmd
pip install -r requirements.txt
```
---
🤖 Training the Model
```cmd
python train.py
```
What train.py does step by step:
Reads `data/housing.csv`
Splits columns:
Features: `location, size, bhk, floors, furnishing`
Target: `price`
Builds a `Pipeline`:
```
   ColumnTransformer
     ├── OneHotEncoder → location, floors, furnishing
     └── passthrough   → size, bhk
   ↓
   RandomForestRegressor (100 trees, max_depth=15)
   ```
Trains on 80% of data, evaluates on 20%
Logs R², RMSE, MAE to MLflow
Saves trained pipeline as `app/model.pkl`
Why OneHotEncoder?
Columns like `location` ("Vijay Nagar") and `floors` ("Ground") are text.
ML models need numbers. OneHotEncoder converts each category into a binary
(0 or 1) column. For example:
```
floors = "Ground" → [1, 0, 0, 0, 0]
floors = "1"      → [0, 1, 0, 0, 0]
floors = "2"      → [0, 0, 1, 0, 0]
```
Expected output:
```
TRAINING COMPLETE
  R²   : 0.9800+
  RMSE : ₹X,XX,XXX
  Model: app/model.pkl
```
---
📊 View MLflow Dashboard
```cmd
mlflow ui
```
Open browser → http://localhost:5000
You will see your training run with R², RMSE, MAE metrics.
Press `Ctrl+C` to stop MLflow.
---
🚀 Run the FastAPI Server
```cmd
uvicorn app.main:app --reload
```
Open browser → http://localhost:8000
---
🧪 Testing
Via Browser
Open http://localhost:8000
Select a locality from the dropdown
Enter Size, BHK, Floors, Furnishing
Click Predict Price
See the predicted price in Indian Rupee format
Via Swagger UI (FastAPI's built-in docs)
Open http://localhost:8000/docs
Click `POST /predict` → Try it out
Enter:
```json
{
  "location": "Vijay Nagar",
  "size": 1200,
  "bhk": 3,
  "floors": "2",
  "furnishing": "Semi Furnished"
}
```
Click Execute
Expected response:
```json
{
  "predicted_price": 19940000.0,
  "formatted_price": "₹1,99,40,000",
  "location": "Vijay Nagar",
  "size": 1200,
  "bhk": 3,
  "floors": "2",
  "furnishing": "Semi Furnished",
  "model_version": "RandomForestRegressor-v2"
}
```
Via Postman
Method: `POST`
URL: `http://localhost:8000/predict`
Body → raw → JSON:
```json
{
  "location": "Nipania",
  "size": 1000,
  "bhk": 2,
  "floors": "Ground",
  "furnishing": "Unfurnished"
}
```
---
📡 API Reference
`GET /`
Serves the HTML frontend.
`GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "locations_count": 142
}
```
`GET /locations`
Returns all unique locality names (used to populate the frontend dropdown).
```json
{
  "locations": ["Amli Kheda", "Anurag Nagar", "Arandia", "...", "Vijay Nagar"]
}
```
`POST /predict`
Request:
```json
{
  "location": "Vijay Nagar",
  "size": 1200,
  "bhk": 3,
  "floors": "2",
  "furnishing": "Semi Furnished"
}
```
Response:
```json
{
  "predicted_price": 19940000.0,
  "formatted_price": "₹1,99,40,000",
  "location": "Vijay Nagar",
  "size": 1200,
  "bhk": 3,
  "floors": "2",
  "furnishing": "Semi Furnished",
  "model_version": "RandomForestRegressor-v2"
}
```
`GET /features`
Returns metadata about accepted features and their valid values.
---
🐳 Docker
Build image (make sure `data/housing.csv` exists first):
```cmd
docker build -t house-price-mlops .
```
Run container:
```cmd
docker run -p 8000:8000 house-price-mlops
```
Open browser → http://localhost:8000
---
🔄 Complete Data Flow
```
1. Browser opens http://localhost:8000
   → FastAPI serves templates/index.html

2. Page loads → script.js calls GET /locations
   → main.py reads housing.csv → returns 142 locality names
   → script.js fills the Location <select> dropdown

3. User selects: Vijay Nagar, 1200 sq ft, 3 BHK, 2 floors, Semi Furnished

4. User clicks "Predict Price"
   → script.js validates all 5 fields
   → fetch POST /predict with JSON body

5. FastAPI (main.py) receives request
   → Pydantic validates types and values
   → builds pd.DataFrame([{ location, size, bhk, floors, furnishing }])

6. pipeline.predict(df) runs:
   a. ColumnTransformer:
      - OneHotEncoder: location → 142 binary columns
                       floors   → 5 binary columns
                       furnishing → 3 binary columns
      - passthrough:   size, bhk → kept as numbers
   b. RandomForestRegressor: 100 trees vote → price prediction

7. main.py formats ₹19,94,0000 as "₹1,99,40,000"
   → returns PredictionResponse JSON

8. script.js receives response
   → animates price counter from ₹0 to ₹1,99,40,000
   → fills in location + details breakdown
   → User sees the result card
```
---
❌ Common Errors
`FileNotFoundError: app/model.pkl`
Run training first:
```cmd
python train.py
```
`FileNotFoundError: data/housing.csv`
Place `housing.csv` in the `data/` folder.
`ModuleNotFoundError: No module named 'fastapi'`
Activate virtual environment and install:
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```
`Port 8000 already in use`
Use a different port:
```cmd
uvicorn app.main:app --reload --port 8001
```
Location dropdown shows "Could not load localities"
The FastAPI server is not running. Start it first:
```cmd
uvicorn app.main:app --reload
```
---
🎓 Key Concepts
ColumnTransformer — Applies different preprocessing to different columns.
OneHotEncoder to text columns, passthrough to number columns.
OneHotEncoder — Converts text categories to binary columns.
`"Vijay Nagar"` becomes a row of 0s with a single 1.
Pipeline — Chains preprocessing + model into one object.
When you call `pipeline.predict(df)`, it runs encoding first, then prediction.
This is why we use `pd.DataFrame` not a numpy array — the encoder needs column names.
Indian number formatting — ₹1,99,40,000 means 1 crore 99 lakh 40 thousand.
Groups: last 3 digits, then groups of 2 from the right.
MLflow — Records every training run. Compare R² and RMSE across runs at http://localhost:5000.