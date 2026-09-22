# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies (fluidsynth is required for MIDI-to-audio wav synthesis)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    fluidsynth \
    fluid-soundfont-gm \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server, source libraries, model checkpoints, and dataset metadata
COPY backend/ ./backend/
COPY src/ ./src/
COPY models/ ./models/
COPY data/tokenized_sequences.pkl ./data/tokenized_sequences.pkl

# Expose the port the FastAPI backend runs on
EXPOSE 8000

# Define environment variables
ENV WORKSPACE_PATH=/app
ENV PYTHONUNBUFFERED=1

# Run uvicorn on container startup
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
