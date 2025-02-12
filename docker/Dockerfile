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

# If you don't want to use your own browser: Install Playwright and its dependencies
# RUN playwright install chromium
# RUN playwright install-deps chromium

# Create temp_pdf directory as specified in research.config
RUN mkdir -p ./temp_pdf

# Copy application files
COPY main.py .
COPY research.config .

# Expose FastAPI port
EXPOSE 8000

CMD ["python", "main.py"]