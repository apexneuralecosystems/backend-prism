FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (safe default)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose backend port
EXPOSE 8000

# Start backend (FastAPI / Flask)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

