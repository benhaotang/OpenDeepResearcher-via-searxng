FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its Chromium browser
RUN playwright install chromium && playwright install-deps

# Create temp_pdf directory as specified in research.config
RUN mkdir -p ./temp_pdf

# Copy application files
COPY main.py .
COPY research.config .

# Expose FastAPI port
EXPOSE 8000

CMD ["python", "main.py"]