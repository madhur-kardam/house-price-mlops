# ============================================================
# train.py — This file trains our Machine Learning model
# ============================================================
# What this file does:
#   1. Reads the housing dataset from data/housing.csv
#   2. Prepares (preprocesses) the data
#   3. Trains a Random Forest model on the data
#   4. Saves the trained model as app/model.pkl
#   5. Logs everything to MLflow so we can track experiments
# ============================================================

# --- IMPORTS ---
# These are Python libraries we need. Think of them as tools.

import pandas as pd          # pandas: reads CSV files and handles data tables
import numpy as np           # numpy: math operations on arrays
import mlflow                # mlflow: tracks our ML experiments
import mlflow.sklearn        # mlflow.sklearn: saves sklearn models to MLflow
import pickle                # pickle: saves Python objects (like models) to disk
import os                    # os: interacts with files and folders

# sklearn is the main machine learning library
from sklearn.model_selection import train_test_split    # splits data into train/test
from sklearn.ensemble import RandomForestRegressor      # the ML model we'll use
from sklearn.preprocessing import StandardScaler        # normalizes numbers
from sklearn.metrics import mean_squared_error, r2_score  # measures model accuracy
from sklearn.pipeline import Pipeline                   # chains preprocessing + model together

# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================

print("=" * 50)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 50)
print("\nStep 1: Loading dataset...")

# Read the CSV file into a DataFrame (like a spreadsheet in Python)
# A DataFrame has rows and columns — just like Excel
df = pd.read_csv("data/housing.csv")

# Show basic info about the dataset
print(f"   Dataset loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns found: {list(df.columns)}")
print(f"\n   First 3 rows of dataset:")
print(df.head(3).to_string())

# ============================================================
# STEP 2: PREPARE FEATURES AND TARGET
# ============================================================

print("\nStep 2: Preparing features and target...")

# Features (X) = the input columns the model will use to make predictions
# Target (y) = the column we want to predict (price)

# We select only the 5 columns we care about
# The Kaggle Housing dataset has columns: price, area, bedrooms, bathrooms, stories, parking
# (and others we'll ignore for simplicity)

feature_columns = ["area", "bedrooms", "bathrooms", "stories", "parking"]
target_column = "price"

# Check if all required columns exist in the dataset
for col in feature_columns + [target_column]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset! Check your CSV file.")

# X = features (5 columns)
# y = target (1 column: price)
X = df[feature_columns]
y = df[target_column]

print(f"   Features (X): {feature_columns}")
print(f"   Target (y): {target_column}")
print(f"   X shape: {X.shape}  (rows, columns)")
print(f"   y shape: {y.shape}  (rows,)")

# ============================================================
# STEP 3: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================

print("\nStep 3: Splitting data into train/test sets...")

# We split data into:
#   - Training set (80%): model learns from this
#   - Test set (20%): we check model accuracy on this (model never saw it)
#
# random_state=42 means we always get the same split (for reproducibility)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to test set
    random_state=42     # fixed seed for reproducibility
)

print(f"   Training set size: {X_train.shape[0]} samples")
print(f"   Test set size: {X_test.shape[0]} samples")

# ============================================================
# STEP 4: CREATE ML PIPELINE (Preprocessing + Model)
# ============================================================

print("\nStep 4: Creating ML Pipeline...")

# A Pipeline chains multiple steps together.
# When you call pipeline.fit(), it runs all steps in order.
# When you call pipeline.predict(), it also runs all steps in order.

# Step 4a: StandardScaler
# Normalizes numbers so they're on the same scale.
# Example: "area" can be 7000, but "bedrooms" is 4.
# Without scaling, the model might think area is more important just because it's a bigger number.
# After scaling, all features are between roughly -3 and +3.
scaler = StandardScaler()

# Step 4b: RandomForestRegressor
# This is our actual ML model.
# "Forest" = many decision trees
# "Regressor" = predicts a continuous number (price), not a category
#
# Parameters:
#   n_estimators=100  → build 100 decision trees, then average their predictions
#   max_depth=10      → each tree can have at most 10 levels (prevents overfitting)
#   random_state=42   → fixed seed for reproducibility
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Create the pipeline: first scale, then train/predict
pipeline = Pipeline([
    ("scaler", scaler),   # Step 1: scale the features
    ("model", model)      # Step 2: train/predict with Random Forest
])

print("   Pipeline created: StandardScaler → RandomForestRegressor")

# ============================================================
# STEP 5: TRAIN THE MODEL WITH MLFLOW TRACKING
# ============================================================

print("\nStep 5: Starting MLflow experiment tracking...")

# MLflow tracks all details of your training run
# Think of it like a lab notebook for ML experiments

# Set the experiment name (creates it if it doesn't exist)
mlflow.set_experiment("house-price-prediction")

# Start an MLflow "run" — everything inside this block gets tracked
with mlflow.start_run() as run:
    
    print(f"   MLflow Run ID: {run.info.run_id}")
    
    # --- LOG PARAMETERS ---
    # Parameters are settings we chose for training
    # MLflow saves these so we can compare different runs later
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("features", str(feature_columns))
    
    print("\nStep 6: Training the model (this may take a moment)...")
    
    # TRAIN THE MODEL
    # .fit() is where actual learning happens
    # The model looks at X_train (features) and y_train (prices)
    # and learns the relationship between them
    pipeline.fit(X_train, y_train)
    
    print("   Model training complete!")
    
    # ============================================================
    # STEP 6: EVALUATE THE MODEL
    # ============================================================
    
    print("\nStep 7: Evaluating model performance...")
    
    # Use the trained model to predict prices on the TEST set
    # (The model hasn't seen these during training)
    y_pred = pipeline.predict(X_test)
    
    # Calculate R² Score (R-squared)
    # R² = 1.0 → perfect predictions
    # R² = 0.0 → model is as good as just guessing the average
    # R² < 0  → model is worse than guessing the average
    r2 = r2_score(y_test, y_pred)
    
    # Calculate RMSE (Root Mean Squared Error)
    # RMSE = average amount the prediction is wrong (in the same unit as price)
    # Lower RMSE is better
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"   R² Score: {r2:.4f}  (1.0 = perfect, 0.0 = no better than guessing)")
    print(f"   RMSE: ${rmse:,.0f}  (average error in price prediction)")
    
    # --- LOG METRICS ---
    # Metrics are the results of training
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mse", mse)
    
    # --- LOG THE MODEL TO MLFLOW ---
    mlflow.sklearn.log_model(pipeline, "house_price_model")
    
    print(f"\n   Metrics logged to MLflow!")
    print(f"   View at: http://localhost:5000 (after running 'mlflow ui')")

# ============================================================
# STEP 7: SAVE THE MODEL AS A .pkl FILE
# ============================================================

print("\nStep 8: Saving model to app/model.pkl...")

# Create the app/ directory if it doesn't exist
os.makedirs("app", exist_ok=True)

# Save the trained pipeline to a file using pickle
# pickle converts a Python object into bytes and saves to disk
# Later, our FastAPI backend will load this file to make predictions
with open("app/model.pkl", "wb") as f:  # "wb" = write binary mode
    pickle.dump(pipeline, f)

print("   Model saved successfully!")

# ============================================================
# DONE!
# ============================================================

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print(f"  Model: RandomForestRegressor")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: ${rmse:,.0f}")
print(f"  Model saved at: app/model.pkl")
print(f"\nNext steps:")
print("  1. Run 'mlflow ui' to view experiment dashboard")
print("  2. Run 'uvicorn app.main:app --reload' to start the API")
print("  3. Open http://localhost:8000 in your browser")
print("=" * 50)
