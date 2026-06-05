# ============================================================
# Dockerfile  —  House Price Prediction MLOps (v2)
# ============================================================
# No changes needed from v1.
# The build steps are the same; train.py now produces a
# ColumnTransformer pipeline instead of a scaler pipeline,
# but that is transparent to Docker.
#
# To build:  docker build -t house-price-mlops .
# To run:    docker run -p 8000:8000 house-price-mlops
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Train the model inside the container at build time
# (housing.csv must exist in data/ before building)
RUN python train.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
