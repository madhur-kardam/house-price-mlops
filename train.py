# ============================================================
# train.py  —  House Price Prediction (Indore Localities)
# ============================================================
#
# WHAT CHANGED FROM THE OLD VERSION:
#   Old dataset columns : area, bedrooms, bathrooms, stories, parking
#   New dataset columns : location, size, bhk, floors, furnishing, price
#
#   Old model           : StandardScaler → RandomForestRegressor
#   New model           : ColumnTransformer (OneHotEncoder for categoricals
#                         + PassThrough for numerics) → RandomForestRegressor
#
# WHY OneHotEncoder?
#   Columns like "location" (Vijay Nagar, Nipania …), "floors" (Ground, 1, 2 …)
#   and "furnishing" (Unfurnished, Semi Furnished, Furnished) are TEXT values.
#   A machine learning model only understands NUMBERS.
#   OneHotEncoder converts each category into a separate 0/1 column so the
#   model can learn from them.
#
#   Example — "floors" column:
#     Ground → [1, 0, 0, 0, 0]
#     1      → [0, 1, 0, 0, 0]
#     2      → [0, 0, 1, 0, 0]
#     etc.
#
# HOW TO RUN:
#   python train.py
# ============================================================

# ── Imports ────────────────────────────────────────────────
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer   # applies different transforms to different columns
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ── Constants ──────────────────────────────────────────────

DATA_PATH  = "data/housing.csv"
MODEL_PATH = "app/model.pkl"

# These are the 3 text columns — they need OneHotEncoding
CATEGORICAL_FEATURES = ["location", "floors", "furnishing"]

# These are the 2 number columns — passed through unchanged
NUMERICAL_FEATURES   = ["size", "bhk"]

# All 5 feature columns together (order matters for the DataFrame)
FEATURE_COLUMNS = ["location", "size", "bhk", "floors", "furnishing"]
TARGET_COLUMN   = "price"

# ── Step 1 : Load dataset ──────────────────────────────────

print("=" * 55)
print("  HOUSE PRICE PREDICTION — MODEL TRAINING")
print("=" * 55)
print(f"\nStep 1 : Loading dataset from '{DATA_PATH}' …")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n❌  '{DATA_PATH}' not found.\n"
        "    Place housing.csv inside the data/ folder and retry."
    )

df = pd.read_csv(DATA_PATH)

# Normalise the 'floors' column: make sure "Ground" stays as the string "Ground"
# and numeric floors become strings like "1", "2", "3", "4".
# This is important because OneHotEncoder works with consistent string types.
df["floors"] = df["floors"].astype(str).str.strip()

print(f"   Rows × Columns : {df.shape}")
print(f"   Columns        : {list(df.columns)}")
print(f"\n   Sample rows:")
print(df.head(3).to_string(index=False))

# Quick sanity check — make sure all required columns exist
for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing from dataset!")

# ── Step 2 : Split features and target ────────────────────

print("\nStep 2 : Preparing features (X) and target (y) …")

X = df[FEATURE_COLUMNS]   # 5 columns, mix of text and numbers
y = df[TARGET_COLUMN]      # 1 column: price (integer)

print(f"   X shape : {X.shape}")
print(f"   y shape : {y.shape}")
print(f"   Price range : ₹{y.min():,.0f}  →  ₹{y.max():,.0f}")

# ── Step 3 : Train / test split ───────────────────────────

print("\nStep 3 : Splitting data (80 % train / 20 % test) …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train samples : {len(X_train)}")
print(f"   Test  samples : {len(X_test)}")

# ── Step 4 : Build the preprocessing + model pipeline ─────
#
# ColumnTransformer lets us apply DIFFERENT transforms to DIFFERENT columns:
#
#   ┌─────────────────────────────────────────────────────┐
#   │ CATEGORICAL columns (location, floors, furnishing)  │
#   │   → OneHotEncoder                                   │
#   │     handle_unknown="ignore" : if a new location     │
#   │     appears at prediction time that was not seen    │
#   │     during training, it is silently encoded as all  │
#   │     zeros instead of crashing.                      │
#   │     sparse_output=False : return a regular numpy    │
#   │     array, not a compressed sparse matrix.          │
#   ├─────────────────────────────────────────────────────┤
#   │ NUMERICAL columns (size, bhk)                       │
#   │   → "passthrough" : keep the numbers exactly as-is  │
#   └─────────────────────────────────────────────────────┘

print("\nStep 4 : Building ColumnTransformer + Pipeline …")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",                          # a name for this transformer (any string)
            OneHotEncoder(
                handle_unknown="ignore",    # don't crash on unseen categories
                sparse_output=False         # return dense array
            ),
            CATEGORICAL_FEATURES            # apply to these columns
        ),
        (
            "num",                          # name for the numeric transformer
            "passthrough",                  # keep numbers unchanged
            NUMERICAL_FEATURES              # apply to these columns
        ),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),        # Step 1 inside pipeline: encode
    ("model", RandomForestRegressor(        # Step 2 inside pipeline: learn
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1                           # use all CPU cores → faster
    ))
])

print("   Pipeline : ColumnTransformer → RandomForestRegressor")

# ── Step 5 : Train with MLflow tracking ───────────────────

print("\nStep 5 : Starting MLflow experiment …")

mlflow.set_experiment("house-price-prediction-indore")

with mlflow.start_run() as run:

    print(f"   MLflow Run ID : {run.info.run_id}")

    # Log what settings we used
    mlflow.log_param("model_type",           "RandomForestRegressor")
    mlflow.log_param("n_estimators",         100)
    mlflow.log_param("max_depth",            15)
    mlflow.log_param("categorical_features", str(CATEGORICAL_FEATURES))
    mlflow.log_param("numerical_features",   str(NUMERICAL_FEATURES))
    mlflow.log_param("train_size",           len(X_train))
    mlflow.log_param("test_size",            len(X_test))

    print("\nStep 6 : Training …  (may take 30–60 seconds)")

    # .fit() is where learning actually happens
    pipeline.fit(X_train, y_train)

    print("   Training complete!")

    # ── Step 6 : Evaluate ─────────────────────────────────

    print("\nStep 7 : Evaluating on test set …")

    y_pred = pipeline.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = np.mean(np.abs(y_test - y_pred))

    print(f"   R²   Score : {r2:.4f}   (1.0 = perfect)")
    print(f"   RMSE       : ₹{rmse:,.0f}  (average error magnitude)")
    print(f"   MAE        : ₹{mae:,.0f}  (mean absolute error)")

    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse",     rmse)
    mlflow.log_metric("mae",      mae)

    mlflow.sklearn.log_model(pipeline, "house_price_model")
    print("   Metrics and model logged to MLflow.")

# ── Step 7 : Save model.pkl ────────────────────────────────

print(f"\nStep 8 : Saving model to '{MODEL_PATH}' …")

os.makedirs("app", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("   Saved successfully!")

# ── Done ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print("=" * 55)
print(f"  R²   : {r2:.4f}")
print(f"  RMSE : ₹{rmse:,.0f}")
print(f"  Model: {MODEL_PATH}")
print()
print("  Next steps:")
print("  1.  mlflow ui                            → view dashboard")
print("  2.  uvicorn app.main:app --reload        → start API")
print("  3.  Open http://localhost:8000           → web UI")
print("=" * 55)
