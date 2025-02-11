# Docker Setup for Open Deep Researcher via Searxng

This directory contains Docker configuration for running the Deep Researcher FastAPI application with SearXNG integration.

## Prerequisites

1. Docker and Docker Compose installed on your system
2. Chrome browser running on your host machine with remote debugging enabled

## Setup Chrome for Remote Debugging

Before starting the containers, ensure Chrome is running with remote debugging enabled. You can start Chrome with:

```bash
# Linux
google-chrome --remote-debugging-port=9222

# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
```

## Running the Application

1. Build and start the containers:
```bash
docker compose up --build
```

2. Access the services:
- FastAPI application: http://localhost:8000
- SearXNG service: http://localhost:4000
- Chrome debugging: http://localhost:9222

## Configuration

The application uses the following configuration files:
- `research.config`: Main configuration file
- `requirements.txt`: Python dependencies
- `main.py`: FastAPI application code

## Volumes

- `temp_pdf`: Directory for temporary PDF storage
- `searxng-data`: Persistent storage for SearXNG configuration

## Troubleshooting

1. If you can't connect to Chrome:
   - Ensure Chrome is running with remote debugging enabled
   - Check if port 9222 is accessible
   - Verify no firewall is blocking the connection

2. If SearXNG is not accessible:
   - Check if port 4000 is available
   - Verify the container logs for any startup issues

## Notes

- The application uses host network mode to access Chrome running on the host machine
- All necessary Python packages will be installed during container build
- SearXNG configuration is persisted across container restarts