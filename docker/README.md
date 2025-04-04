# Docker Setup for Open Deep Researcher via Searxng

This directory contains Docker configuration for running an OpenAI-compatible Deep Researcher API endpoint. The setup supports multiple operation modes through configuration and GPU acceleration options. **But for most user, I still suggest using CPU version for smaller file size as GPU is only used for accelerating PDF OCR**

## Prerequisites

1. Docker and Docker Compose installed
2. For local web parsing mode: Chrome/Chromium with remote debugging
3. For local models mode: Ollama installed
4. For GPU acceleration:
   - NVIDIA users: NVIDIA drivers and nvidia-container-runtime installed
   - AMD users: ROCm installed (version 6.3.2 or later)

## Operation Modes

Configure your preferred mode in `research.config`:

### 1. Online Mode (Maximum Speed)
```ini
[Settings]
use_jina = true      # Use Jina for fast web parsing
use_ollama = false   # Use OpenRouter models
with_planning = true # Enable planning agent
default_model = anthropic/claude-3.5-haiku
reason_model = deepseek/deepseek-r1-distill-qwen-32b
```

### 2. Hybrid Mode (Speed/Privacy Balance)
```ini
[Settings]
use_jina = true     # Use Jina for web parsing
use_ollama = true   # Use local models
with_planning = true
default_model = mistral-small
reason_model = deepseek-r1:14b
```

### 3. Fully Local Mode (Maximum Privacy)
```ini
[Settings]
use_jina = false    # Use local web parsing
use_ollama = true   # Use local models
with_planning = true
default_model = mistral-small
reason_model = deepseek-r1:14b
[LocalAI]
default_model_ctx = -1 # Use -1 to load default context length settings from modelfile
reason_model_ctx = -1
```

## Configuration Reference (research.config)

### [LocalAI]
- `ollama_base_url`: Ollama API endpoint (default: http://localhost:11434)
- `default_model_ctx` : Set context length for search and writing models, -1 for default
- `reason_model_ctx` : Set context length for reasoning and planning models, -1 for default
### [API]
- `openai_compat_api_key`: Authentication key for the API endpoint
- `jina_api_key`: Required if use_jina = true
- `openai_url`: OpenRouter or other OpenAI-compatible endpoint
- `jina_base_url`: Jina parsing service URL
- `searxng_url`: Local SearXNG instance URL

### [Settings]
- `use_jina`: Enable Jina API for fast web parsing
- `use_ollama`: Use local Ollama models instead of OpenRouter
- `with_planning`: Enable research planning agent
- `default_model`: Model for search and writing
- `reason_model`: Model for planning and reasoning

### [Concurrency]
- `concurrent_limit`: Maximum concurrent operations (default: 3)
- `cool_down`: Delay between requests to same domain (default: 10.0)
- `chrome_port`: Chrome debugging port (default: 9222)
- `chrome_host_ip`: Chrome host IP address (default: http://localhost, if you are not running with `--remote-debugging-address=0.0.0.0`, you need to change this to your local IP)
- `use_embed_browser`: Use embedded Playwright browser instead of external Chrome (default: false)

### [Ratelimits]
- `request_per_minute`: Rate limit for default model (-1 to disable)
- `operation_wait_time`: Wait time in seconds between research iterations (0 to disable)
- `fallback_model`: Model to use when rate limited (e.g., google/gemini-2.0-flash-001). Should have:
  * Large context length (100k+ recommended for pure online method, 32k+ for local)
  * High tokens per minute limit

### [Parsing]
- `temp_pdf_dir`: Directory for temporary PDF storage
- `browse_lite`: Fast parsing mode without ML models (0/1)
- `pdf_max_pages`: Maximum PDF pages to process (default: 30)
- `pdf_max_filesize`: Maximum PDF file size in bytes
- `timeout_pdf`: PDF processing timeout in seconds
- `max_html_length`: Maximum HTML content length to process
- `max_eval_time`: JavaScript evaluation timeout
- `verbose_web_parse_detail`: Enable detailed parsing logs

## Setup Steps

1. Configure Operation Mode:
   - Edit `research.config` based on your preferred mode
   - Set API keys and URLs as needed

2. For Local Models (if use_ollama = true):
   ```bash
   ollama pull mistral-small    # search & writing
   ollama pull deepseek-r1:14b  # reasoning & planning
   ```

3. For Local Web Parsing (if use_jina = false):
    ```bash
    # Option 1: External Chrome (use_embed_browser = false)
    google-chrome --remote-debugging-port=9222 --remote-debugging-address=0.0.0.0
    # Optional: Start Chrome with your credentials for academic access
    google-chrome --remote-debugging-port=9222 --remote-debugging-address=0.0.0.0 --user-data-dir=/path/to/profile
    
    # Option 2: Embedded Browser (use_embed_browser = true)
    # No need to start Chrome manually, set use_embed_browser = true in research.config
    # The container will automatically manage a headless browser
    
    # Optional: Enhanced parsing capabilities
    ollama pull reader-lm:0.5b
    ```

4. Start Services:

   For CPU-only operation:
   ```bash
   docker compose up --build
   ```

   For NVIDIA GPU acceleration:
   ```bash
   docker compose -f docker-compose.cuda.yml up --build
   ```
   - Uses CUDA 12.8.0 with cuDNN
   - Requires NVIDIA Container Runtime
   - Automatically enables all available NVIDIA GPUs

   For AMD GPU acceleration:
   ```bash
   docker compose -f docker-compose.rocm.yml up --build
   ```
   - Uses ROCm 6.3.2
   - PyTorch 2.4.0 with ROCm support
   - Requires ROCm installation and compatible AMD GPU

## API Endpoint Usage

The service provides an OpenAI-compatible endpoint at http://localhost:8000/v1

### Python Example:
```python
import requests
import json

# Setup API configuration
base_url = "http://localhost:8000/v1"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-xxx"  # Default API key works for local endpoint
}

# Prepare request data
data = {
    "model": "deep_researcher",
    "messages": [
        {"role": "system", "content": "Write in a formal tone."}, # Only writing instructions are supported now
        {"role": "user", "content": "Latest developments in quantum computing"}
    ],
    "stream": True,  # Enable live updates
    "max_iterations": 10,  # Research depth (>1)
    "max_search_items": 4,  # Results per search (>1, for use_jina=false)
    "default_model": "anthropic/claude-3.5-haiku", # Optional: Override default model
    "reason_model": "deepseek/deepseek-r1-distill-qwen-32b" # Optional: Override reasoning model
}

# Make the API request
response = requests.post(
    f"{base_url}/chat/completions",
    headers=headers,
    json=data,
    stream=True
)

# Stream the response
for line in response.iter_lines():
    if not line:
        continue
    
    if line.startswith(b"data: "):
        try:
            chunk = json.loads(line[6:])  # Skip "data: " prefix
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                print(chunk["choices"][0]["delta"]["content"], end="")
        except json.JSONDecodeError:
            continue
```

### cURL Example:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deep_researcher",
    "messages": [{"role": "user", "content": "Latest developments in quantum computing"}],
    "stream": true,
    "max_iterations": 10,  # Research depth (>1)
    "max_search_items": 4,  # Results per search (>1, for use_jina=false)
    "default_model": "anthropic/claude-3.5-haiku", # Optional: Override default model
    "reason_model": "deepseek/deepseek-r1-distill-qwen-32b" # Optional: Override reasoning model
  }'
```

## Volumes

- `temp_pdf`: Temporary PDF storage
- `searxng-data`: Persistent SearXNG configuration

## Troubleshooting

1. Chrome/Browser Issues:
    - For external Chrome (use_embed_browser = false):
        - Verify Chrome is running with remote debugging
        - Check port 9222 accessibility
        - Ensure no firewall blocks the connection
    - For embedded browser (use_embed_browser = true):
        - No manual Chrome setup needed
        - Check container logs for Playwright browser installation status

2. SearXNG Issues:
   - Verify port 4000 availability
   - Check container logs for startup problems

3. API Response Issues:
   - Verify API keys in research.config
   - Check model availability if using Ollama
   - Review operation mode settings

4. GPU Issues:
   - NVIDIA:
     - Verify nvidia-smi shows your GPU
     - Check nvidia-container-runtime installation
     - Ensure NVIDIA drivers are up to date
   
   - AMD:
     - Verify ROCm installation with rocm-smi
     - Check GPU compatibility with ROCm
     - Ensure proper device permissions (/dev/kfd, /dev/dri)