🏠 House Price Prediction MLOps Project
A beginner-friendly MLOps project that predicts house prices using Machine Learning.
Built with Python, FastAPI, MLflow, Docker, and a modern HTML/CSS/JS frontend.
---
📚 What This Project Does
```
User enters house details (area, bedrooms, etc.)
         ↓
Beautiful web frontend (HTML/CSS/JS)
         ↓
FastAPI backend receives the data
         ↓
Trained ML model predicts the price
         ↓
Price is sent back and displayed on screen
```
---
🗂️ Complete Folder Structure
```
house-price-mlops/
│
├── app/
│   ├── main.py          ← FastAPI backend (the server)
│   └── model.pkl        ← Trained ML model (created after training)
│
├── templates/
│   └── index.html       ← Web page (what users see)
│
├── static/
│   ├── style.css        ← Visual styling of the page
│   └── script.js        ← JavaScript logic (sends data to API)
│
├── data/
│   └── housing.csv      ← Dataset (download instructions below)
│
├── train.py             ← Script to train the ML model
├── requirements.txt     ← List of Python packages to install
├── Dockerfile           ← Instructions to run app in Docker
└── README.md            ← This file!
```
---
💻 Step 1: Install Required Software (Windows 11)
1.1 Install Python
Go to: https://www.python.org/downloads/
Click Download Python 3.11.x (latest 3.11 version)
Run the installer
⚠️ IMPORTANT: Check the box that says "Add Python to PATH" before clicking Install
Click Install Now
Verify Python installed:
Open Command Prompt (press `Win + R`, type `cmd`, press Enter) and run:
```
python --version
```
You should see: `Python 3.11.x`
---
1.2 Install VS Code
Go to: https://code.visualstudio.com/
Download and install it
Open VS Code and install the Python extension (click the blocks icon on the left sidebar, search "Python", install the one by Microsoft)
---
1.3 Install Git
Go to: https://git-scm.com/download/win
Download and run the installer (click Next through all options)
Verify Git installed:
```
git --version
```
You should see: `git version 2.x.x`
---
1.4 Install Docker Desktop (Optional but Recommended)
Go to: https://www.docker.com/products/docker-desktop/
Download Docker Desktop for Windows
Run the installer, follow instructions
Restart your computer when asked
Verify Docker installed:
```
docker --version
```
You should see: `Docker version 24.x.x`
---
📁 Step 2: Get the Dataset
We will use the Housing Prices Dataset — it's simple, beginner-friendly, and free.
Download Instructions:
Go to: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
Click Download (you need a free Kaggle account)
Extract the ZIP file
Find the file named `Housing.csv`
Rename it to `housing.csv` (lowercase)
Place it inside the `data/` folder of your project
What's in this dataset?
The dataset has these columns:
Column	What it means
`area`	Size of the house in square feet
`bedrooms`	Number of bedrooms
`bathrooms`	Number of bathrooms
`stories`	Number of floors
`parking`	Number of parking spots
`price`	House price in dollars (this is what we predict!)
---
🛠️ Step 3: Set Up the Project
3.1 Create the Project Folder
Open Command Prompt and run these commands one by one:
```cmd
cd Desktop
mkdir house-price-mlops
cd house-price-mlops
```
3.2 Create the Subfolder Structure
```cmd
mkdir app
mkdir templates
mkdir static
mkdir data
```
Now copy all the files from this project into the correct folders.
3.3 Create a Virtual Environment
A virtual environment is like a clean room just for your project — it keeps packages separate.
```cmd
python -m venv venv
```
3.4 Activate the Virtual Environment
```cmd
venv\Scripts\activate
```
You'll see `(venv)` appear at the start of the command line. This means the virtual environment is active.
> 💡 Every time you open a new terminal to work on this project, run this activate command first!
3.5 Install Dependencies
```cmd
pip install -r requirements.txt
```
This installs all required Python packages. It might take a few minutes.
---
🤖 Step 4: Train the ML Model
```cmd
python train.py
```
What happens when you run this:
Python reads `data/housing.csv`
It separates the features (area, bedrooms, etc.) from the target (price)
It splits data into training set (80%) and test set (20%)
It trains a Random Forest Regressor model
It saves the trained model as `app/model.pkl`
MLflow records all the experiment details
Expected output:
```
Loading dataset...
Training model...
Model trained successfully!
R² Score: 0.65
RMSE: 1200000
Model saved to app/model.pkl
MLflow run ID: abc123...
```
---
📊 Step 5: View MLflow Dashboard
MLflow tracks your ML experiments — it logs metrics, parameters, and model versions.
```cmd
mlflow ui
```
Then open your browser and go to: http://localhost:5000
You'll see your training run with:
Model name
R² Score (how accurate the model is)
RMSE (average prediction error)
> Press `Ctrl+C` in the terminal to stop MLflow when done
---
🚀 Step 6: Run the FastAPI Server
```cmd
uvicorn app.main:app --reload
```
What this command means:
`uvicorn` — the web server
`app.main:app` — look in the `app` folder, in `main.py`, find the variable named `app`
`--reload` — automatically restart when you change code
Open your browser and go to: http://localhost:8000
You'll see the House Price Prediction web page! 🎉
---
🧪 Step 7: Testing
7.1 Test via Browser
Open http://localhost:8000
Enter house details
Click "Predict Price"
See the result!
7.2 Test via Swagger UI (FastAPI's built-in testing tool)
Open http://localhost:8000/docs
Click on `POST /predict`
Click "Try it out"
Enter this JSON:
```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "parking": 2
}
```
Click "Execute"
See the predicted price in the response
7.3 Test via Postman
Download Postman from https://www.postman.com/downloads/
Create a new request
Set method to POST
URL: `http://localhost:8000/predict`
Go to Body tab → select raw → choose JSON
Paste:
```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "parking": 2
}
```
Click Send
Expected Response:
```json
{
  "predicted_price": 4850000.0,
  "currency": "USD",
  "model_version": "RandomForestRegressor"
}
```
---
🐳 Step 8: Run with Docker
Docker packages your entire app into a container that runs the same on any computer.
Build the Docker image:
```cmd
docker build -t house-price-mlops .
```
Run the container:
```cmd
docker run -p 8000:8000 house-price-mlops
```
Open browser: http://localhost:8000
Stop the container:
Press `Ctrl+C` or run:
```cmd
docker stop $(docker ps -q)
```
---
❌ Common Errors and Fixes
Error: `ModuleNotFoundError: No module named 'fastapi'`
Fix: Make sure your virtual environment is activated:
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```
Error: `FileNotFoundError: app/model.pkl not found`
Fix: You need to train the model first:
```cmd
python train.py
```
Error: `FileNotFoundError: data/housing.csv not found`
Fix: Download the dataset and place it in the `data/` folder. See Step 2 above.
Error: `Port 8000 already in use`
Fix: Another program is using port 8000. Either stop it or use a different port:
```cmd
uvicorn app.main:app --reload --port 8001
```
Error: `CORS error` in browser console
Fix: This is already handled in `app/main.py` with `CORSMiddleware`.
Error: `uvicorn is not recognized`
Fix: Install it:
```cmd
pip install uvicorn
```
---
📡 API Reference
`GET /`
Returns the HTML frontend page.
`POST /predict`
Predicts house price.
Request body:
```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "parking": 2
}
```
Response:
```json
{
  "predicted_price": 4850000.0,
  "currency": "USD",
  "model_version": "RandomForestRegressor"
}
```
`GET /health`
Check if the API is running.
Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
---
🔄 Complete Data Flow Explained
```
1. User opens http://localhost:8000 in browser
   → FastAPI serves index.html from templates/

2. User fills in: area=7420, bedrooms=4, bathrooms=2, stories=3, parking=2
   → HTML form collects these values

3. User clicks "Predict Price"
   → script.js runs and collects all values

4. script.js sends a POST request to http://localhost:8000/predict
   → Request body: {"area": 7420, "bedrooms": 4, ...}

5. FastAPI receives the request in main.py
   → It validates the input using Pydantic

6. FastAPI loads the trained model from app/model.pkl
   → model.pkl was created by train.py

7. The Random Forest model predicts the price
   → It uses the 5 features to calculate price

8. FastAPI sends back the response
   → {"predicted_price": 4850000.0, ...}

9. script.js receives the response
   → It updates the webpage to show the price

10. User sees: "Predicted Price: $4,850,000"
```
---
🎓 Key Concepts Explained Simply
What is an API?
An API is like a waiter in a restaurant. You (frontend) give your order to the waiter (API), the waiter takes it to the kitchen (ML model), and brings back your food (prediction).
What is FastAPI?
A Python tool that makes it easy to create APIs. It also automatically creates documentation at `/docs`.
What is MLflow?
A tool that tracks your ML experiments. It remembers what parameters you used, what your model's accuracy was, and saves different versions of your model.
What is Docker?
Docker packages your entire app (code + dependencies) into a box called a "container". This box runs the same on any computer, so you never have "it works on my machine" problems.
What is a Virtual Environment?
A separate Python installation just for your project. It prevents conflicts between different projects that need different package versions.
What is Random Forest?
An ML algorithm that creates many decision trees and averages their predictions. It's like asking 100 experts and taking the average answer.