# Use official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /app 
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app folder contents
COPY app/ .

# Expose port
EXPOSE 8000

# Run uvicorn from the /app directory (matches your manual workflow)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]