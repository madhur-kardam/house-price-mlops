# ============================================================
# Dockerfile — Instructions to package this app in Docker
# ============================================================
# A Dockerfile is like a recipe for Docker.
# Docker reads these instructions and builds a container
# that has everything needed to run your app.
#
# To build: docker build -t house-price-mlops .
# To run:   docker run -p 8000:8000 house-price-mlops
# ============================================================

# --- STEP 1: Choose the base image ---
# Think of this as the starting point — like choosing an OS.
# python:3.11-slim is a lightweight Python 3.11 image.
# "slim" means it doesn't have unnecessary packages, so it's smaller.
FROM python:3.11-slim

# --- STEP 2: Set the working directory inside the container ---
# All subsequent commands run from /app inside the container.
# It's like doing "cd /app" inside the container.
WORKDIR /app

# --- STEP 3: Copy requirements.txt first ---
# We copy requirements.txt before other files because Docker caches layers.
# If requirements.txt didn't change, Docker skips reinstalling packages.
# This makes rebuilds faster.
COPY requirements.txt .

# --- STEP 4: Install Python packages ---
# --no-cache-dir: don't cache downloaded packages (saves disk space)
RUN pip install --no-cache-dir -r requirements.txt

# --- STEP 5: Copy all project files into the container ---
# The first "." means "everything in the current directory on your computer"
# The second "." means "copy it to /app in the container"
COPY . .

# --- STEP 6: Train the model during build ---
# This runs train.py to create model.pkl inside the container
# NOTE: Make sure data/housing.csv exists before building the Docker image!
RUN python train.py

# --- STEP 7: Expose port 8000 ---
# Tell Docker that this container will listen on port 8000
# This doesn't actually open the port — you do that with -p flag when running
EXPOSE 8000

# --- STEP 8: Define the command to start the app ---
# This runs when you do "docker run ..."
# uvicorn starts the FastAPI server
# --host 0.0.0.0 means "listen on all network interfaces" (required in Docker)
# --port 8000 means listen on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
