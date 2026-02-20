FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build tools, ffmpeg, and MediaPipe dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify MediaPipe installation (fail build if MediaPipe doesn't work)
RUN python -c "import mediapipe as mp; assert hasattr(mp, 'solutions'), 'MediaPipe solutions not available'; print('✅ MediaPipe verified')"

# Copy backend code
COPY . .

# Expose backend port
EXPOSE 5555

# Start backend (FastAPI / Flask)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5555", "--log-level", "debug"]

