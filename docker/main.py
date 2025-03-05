import nest_asyncio
nest_asyncio.apply()
import asyncio
import aiohttp
import re
import ast
import time
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse
import mimetypes
from playwright.async_api import async_playwright
from docling.document_converter import DocumentConverter
import configparser
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# FastAPI app
app = FastAPI(title="Deep Researcher API")

# API Models
class Message(BaseModel):
    role: str
    content: str

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelObject]

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deep_researcher")
    messages: List[Message]
    stream: bool = False
    max_iterations: Optional[int] = Field(default=10, ge=1, le=50)
    max_search_items: Optional[int] = Field(default=4, ge=1, le=20)
    default_model: Optional[str] = Field(default=None, description="Override the default model from config")
    reason_model: Optional[str] = Field(default=None, description="Override the reason model from config")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# Load configuration
config = configparser.ConfigParser()
config.read('research.config')

# ---------------------------
# Local AI
# ---------------------------
from ollama import AsyncClient
OLLAMA_BASE_URL = config.get('LocalAI', 'ollama_base_url')
DEFAULT_MODEL_CTX = int(config.get('LocalAI', 'default_model_ctx')) # -1 to load config from original modelfile
REASON_MODEL_CTX = int(config.get('LocalAI', 'reason_model_ctx')) # -1 to load config from original modelfile

# ---------------------------
# Configuration Constants
# ---------------------------
OPENAI_COMPAT_API_KEY = config.get('API', 'openai_compat_api_key')
JINA_API_KEY = config.get('API', 'jina_api_key')
OPENAI_URL = config.get('API', 'openai_url')
JINA_BASE_URL = config.get('API', 'jina_base_url')
BASE_SEARXNG_URL = config.get('API', 'searxng_url')

USE_OLLAMA = config.getboolean('Settings', 'use_ollama')
USE_JINA = config.getboolean('Settings', 'use_jina')
WITH_PLANNING = config.getboolean('Settings', 'with_planning')
DEFAULT_MODEL = config.get('Settings', 'default_model')
REASON_MODEL = config.get('Settings', 'reason_model')
DEFAULT_MODEL_MAX_INPUT = config.getint('Settings', 'default_model_max_input', fallback=-1) # -1 means we don't account for max input limits
REASON_MODEL_MAX_INPUT = config.getint('Settings', 'reason_model_max_input', fallback=-1) # -1 means we don't account for max input limits

# -------------------------------
# Concurrency control for browser
# -------------------------------
concurrent_limit = config.getint('Concurrency', 'concurrent_limit')
cool_down = config.getfloat('Concurrency', 'cool_down')
CHROME_PORT = config.getint('Concurrency', 'chrome_port')
CHROME_HOST_IP = config.get('Concurrency', 'chrome_host_ip')
USE_EMBED_BROWSER = config.getboolean('Concurrency', 'use_embed_browser')
global_semaphore = asyncio.Semaphore(concurrent_limit)
domain_locks = defaultdict(asyncio.Lock)  # domain -> asyncio.Lock
domain_next_allowed_time = defaultdict(lambda: 0.0)  # domain -> float (epoch time)

# ---------------------
# Parsing settings
# ---------------------
TEMP_PDF_DIR = Path(config.get('Parsing', 'temp_pdf_dir'))
BROWSE_LITE = config.getint('Parsing', 'browse_lite')
PDF_MAX_PAGES = config.getint('Parsing', 'pdf_max_pages')
PDF_MAX_FILESIZE = config.getint('Parsing', 'pdf_max_filesize')
TIMEOUT_PDF = config.getint('Parsing', 'timeout_pdf')
MAX_HTML_LENGTH = config.getint('Parsing', 'max_html_length')
MAX_EVAL_TIME = config.getint('Parsing', 'max_eval_time')
VERBOSE_WEB_PARSE = config.getboolean('Parsing', 'verbose_web_parse_detail')

# ----------------------
# Ratelimits
# ----------------------
REQUEST_PER_MINUTE = int(config.get('Ratelimits', 'request_per_minute', fallback=-1))  # -1 means no rate limiting
OPERATION_WAIT_TIME = int(config.get('Ratelimits', 'operation_wait_time', fallback=0))  # 0 means no wait time
FALLBACK_MODEL = config.get('Ratelimits', 'fallback_model', fallback=DEFAULT_MODEL)  # Use default model if no fallback specified

# ----------------------
# Token Estimation
# ----------------------
def estimate_tokens(text):
    """
    Estimate the number of tokens in a text string.
    This is a rough approximation: ~4 characters per token for English text.
    """
    return len(text) // 4  # Simple approximation

# Constants for prompt overhead estimation
JUDGE_PROMPT_OVERHEAD = 1500  # Conservative estimate for judge function prompt overhead
SEARCH_PROMPT_OVERHEAD = 1200  # Conservative estimate for search queries function prompt overhead
WRITING_PROMPT_OVERHEAD = 1800  # Conservative estimate for writing plan function prompt overhead
REPORT_PROMPT_OVERHEAD = 2000  # Conservative estimate for final report function prompt overhead

def split_contexts_into_chunks(contexts, max_tokens, prompt_overhead):
    """
    Split a list of context items into chunks that fit within the token limit.
    
    Args:
        contexts: List of context strings to split
        max_tokens: Maximum tokens allowed per chunk
        prompt_overhead: Estimated tokens used by the prompt
        
    Returns:
        List of context chunks, where each chunk is a joined string of context items
    """
    available_tokens = max_tokens - prompt_overhead
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    
    for item in contexts:
        item_tokens = estimate_tokens(item)
        if current_chunk_tokens + item_tokens > available_tokens and current_chunk:
            # Current chunk is full, start a new one
            chunks.append("\n".join(current_chunk))
            current_chunk = [item]
            current_chunk_tokens = item_tokens
        else:
            # Add to current chunk
            current_chunk.append(item)
            current_chunk_tokens += item_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks


# Rate limiting for OpenRouter/OpenAI compatible API
openrouter_last_request_times = []  # Track request timestamps in the last 60s


# ----------------------
# Openrouter
# ----------------------
async def call_openrouter_async(session, messages, model=DEFAULT_MODEL, is_fallback=False):
    """
    Asynchronously call the OpenRouter/OpenAI compatible chat completion API with the provided messages.
    Returns the content of the assistant’s reply.
    """
    global openrouter_last_request_times
    # Apply rate limiting only for DEFAULT_MODEL and when REQUEST_PER_MINUTE is set
    if model == DEFAULT_MODEL and REQUEST_PER_MINUTE > 0:
            current_time = time.time()
            # Remove requests older than 60 seconds
            openrouter_last_request_times = [t for t in openrouter_last_request_times if current_time - t < 60]
            
            if len(openrouter_last_request_times) >= REQUEST_PER_MINUTE:
                # Wait until we can make another request
                oldest_time = openrouter_last_request_times[0]
                wait_time = 60 - (current_time - oldest_time)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Add current request time
            openrouter_last_request_times.append(current_time)

    headers = {
        "Authorization": f"Bearer {OPENAI_COMPAT_API_KEY}",
        "X-Title": "OpenDeepResearcher, by Matt Shumer and Benhao Tang",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        async with session.post(OPENAI_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    content = result['choices'][0]['message']['content']
                    # If content is empty and not using fallback, retry with fallback model
                    if (not content or content.strip() == "") and not is_fallback:
                        print(f"Empty response from model, retrying with fallback model: {FALLBACK_MODEL}")
                        return await call_openrouter_async(session, messages, model=FALLBACK_MODEL, is_fallback=True)
                    return content
                except (KeyError, IndexError) as e:
                    error_msg = f"Unexpected OpenRouter/OpenAI compatible response structure: {result}"
                    print(error_msg)
                    return f"Error: {error_msg}"
            else:
                text = await resp.text()
                error_msg = f"OpenRouter/OpenAI compatible API error: {resp.status} - {text}"
                print(error_msg)
                # Check if this is a rate limit error and we're using the default model and not already a fallback
                if not is_fallback and any(phrase in text.lower() for phrase in ["rate limit", "rate limits", "ratelimit","rate_limit","rate-limit","context length", "context-length","max tokens","max_tokens"]):
                    print(f"Rate limit/Context length hit, retrying with fallback model: {FALLBACK_MODEL}")
                    # Retry with fallback model, marking as fallback to prevent recursion
                    return await call_openrouter_async(session, messages, model=FALLBACK_MODEL, is_fallback=True)
                if is_fallback and any(phrase in text.lower() for phrase in ["rate limit", "rate limits", "ratelimit","rate_limit","rate-limit","context length", "context-length","max tokens","max_tokens"]):
                    error_msg = "Rate limit hit/Context length hit even for fallback model, consider choosing a model with larger context length as fallback or other models/services."
                return f"Error: {error_msg}"
    except Exception as e:
        print("Error calling OpenRouter/OpenAI compatible API:", e)
        return None

# --------------------------
# Local AI and Browser use
# --------------------------

async def call_ollama_async(session, messages, model=DEFAULT_MODEL, max_tokens=20000, ctx=-1):
    """
    Asynchronously call the Ollama chat completion API with the provided messages.
    Returns the content of the assistant’s reply.
    """
    try:
        client = AsyncClient(host=OLLAMA_BASE_URL)
        response_content = ""

        if ctx <= 2000: # Load config from original modelfile respect how ollama does
            async for part in await client.chat(model=model, messages=messages, stream=True, options=dict(num_predict=max_tokens)):
                if 'message' in part and 'content' in part['message']:
                    response_content += part['message']['content']
        else:
            async for part in await client.chat(model=model, messages=messages, stream=True, options=dict(num_predict=max_tokens,num_ctx=ctx)):
                if 'message' in part and 'content' in part['message']:
                    response_content += part['message']['content']

        return response_content if response_content else None
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

def is_pdf_url(url):
    parsed_url = urlparse(url)
    if parsed_url.path.lower().endswith(".pdf"):
        return True
    mime_type, _ = mimetypes.guess_type(url)
    return mime_type == "application/pdf"

def get_domain(url):
    parsed = urlparse(url)
    return parsed.netloc.lower()

# Global lock to ensure only one PDF processing task runs at a time
pdf_processing_lock = asyncio.Lock()

async def process_pdf(pdf_path):
    """
    Converts a local PDF file to text using Docling.
    Ensures only one PDF processing task runs at a time to prevent GPU OutOfMemoryError.
    """
    converter = DocumentConverter()

    async def docling_task():
        return converter.convert(str(pdf_path), max_num_pages=PDF_MAX_PAGES, max_file_size=PDF_MAX_FILESIZE)

    # Ensure no other async task runs while processing the PDF
    async with pdf_processing_lock:
        try:
            return await asyncio.wait_for(docling_task(), TIMEOUT_PDF)  # Enforce timeout
        except asyncio.TimeoutError:
            return "Parser unable to parse the resource within defined time."

async def download_pdf(page, url):
    """
    Downloads a PDF from a webpage using Playwright and saves it locally.
    """
    pdf_filename = TEMP_PDF_DIR / f"{hash(url)}.pdf"

    async def intercept_request(request):
        """Intercepts request to log PDF download attempts."""
        if request.resource_type == "document":
            print(f"Downloading PDF: {request.url}")

    # Attach the request listener
    page.on("request", intercept_request)

    try:
        await page.goto(url, timeout=30000)  # 30 seconds timeout
        await page.wait_for_load_state("networkidle")
        
        # Attempt to save PDF (works for direct links)
        await page.pdf(path=str(pdf_filename))
        return pdf_filename

    except Exception as e:
        print(f"Error downloading PDF {url}: {e}")
        return None

async def get_cleaned_html(page):
    """
    Extracts cleaned HTML from a page while enforcing a timeout.
    """
    try:
        cleaned_html = await asyncio.wait_for(
            page.evaluate("""
                () => {
                    let clone = document.cloneNode(true);
                    
                    // Remove script, style, and noscript elements
                    clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                    
                    // Optionally remove navigation and footer
                    clone.querySelectorAll('nav, footer, aside').forEach(el => el.remove());
                    
                    return clone.body.innerHTML;
                }
            """),
            timeout=MAX_EVAL_TIME
        )
        return cleaned_html
    except asyncio.TimeoutError:
        return "Parser unable to extract HTML within defined time."


async def fetch_webpage_text_async(session, url):
    if USE_JINA:
        """
        Asynchronously retrieve the text content of a webpage using Jina.
        The URL is appended to the Jina endpoint.
        """
        full_url = f"{JINA_BASE_URL}{url}"
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
        try:
            async with session.get(full_url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    text = await resp.text()
                    print(f"Jina fetch error for {url}: {resp.status} - {text}")
                    return ""
        except Exception as e:
            print("Error fetching webpage text with Jina:", e)
            return ""
    else:
        """
        Fetches webpage HTML using Playwright, processes it using Ollama reader-lm:1.5b,
        or downloads and processes a PDF using Docling.
        Respects concurrency limits and per-domain cooldown.
        """
        domain = get_domain(url)

        async with global_semaphore:  # Global concurrency limit
            async with domain_locks[domain]:  # Ensure only one request per domain at a time
                now = time.time()
                if now < domain_next_allowed_time[domain]:
                    await asyncio.sleep(domain_next_allowed_time[domain] - now)  # Respect per-domain cooldown

                async with async_playwright() as p:
                    # Attempt to connect to an already running Chrome instance
                    try:
                        if USE_EMBED_BROWSER:
                            browser = await p.chromium.launch(
                                headless=True,
                                args=['--no-sandbox', '--disable-setuid-sandbox'],
                                firefox_user_prefs={
                                    "general.useragent.override": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
                                }
                            )
                        else:
                            browser = await p.chromium.connect_over_cdp(f"{CHROME_HOST_IP}:{CHROME_PORT}")
                    except Exception as e:
                        error_msg = "Failed to launch browser" if USE_EMBED_BROWSER else f"Failed to connect to Chrome on port {CHROME_PORT}"
                        print(f"Error: {error_msg} - {e}")
                        return error_msg

                    context = await browser.new_context()
                    page = await context.new_page()

                    # PDFs
                    if is_pdf_url(url):
                        if BROWSE_LITE:
                            result = "PDF parsing is disabled in lite browsing mode."
                        else:
                            pdf_path = await download_pdf(page, url)
                            if pdf_path:
                                text = await process_pdf(pdf_path)
                                result = f"# PDF Content\n{text}"
                            else:
                                result = "Failed to download or process PDF"
                    else:
                        try:
                            await page.goto(url, timeout=30000) # 30 seconds timeout to wait for loading
                            title = await page.title() or "Untitled Page"
                            if BROWSE_LITE:
                                # Extract main content using JavaScript inside Playwright
                                main_content = await page.evaluate("""
                                    () => {
                                        let mainEl = document.querySelector('main') || document.body;
                                        return mainEl.innerText.trim();
                                    }
                                """)
                                result = f"# {title}\n{main_content}"
                            else:
                                # Clean HTML before sending to reader-lm
                                cleaned_html = await get_cleaned_html(page)
                                cleaned_html = cleaned_html[:MAX_HTML_LENGTH] # Enforce a Maximum length for a webpage
                                messages = [{"role": "user", "content": cleaned_html}]
                                markdown_text = await call_ollama_async(session, messages, model="reader-lm:0.5b", max_tokens=int(1.25*MAX_HTML_LENGTH)) # Don't get stuck when exceed reader-lm ctx
                                result = f"# {title}\n{markdown_text}"

                        except Exception as e:
                            print(f"Error fetching webpage: {e}")
                            result = f"Failed to fetch {url}"

                    await browser.close()

                # Update next allowed time for this domain (cool down time per domain)
                domain_next_allowed_time[domain] = time.time() + cool_down

        return result

# ============================
# Asynchronous Helper Functions
# ============================

async def make_initial_searching_plan_async(session, user_query):
    """
    Ask the reasoning LLMs to produce a research plan based on the user’s query.
    """
    prompt = (
        "You are an advanced reasoning LLM that specializes in structuring and refining research plans. Based on the given user query,"
        "you will generate a comprehensive research plan that expands on the topic, identifies key areas of investigation, and breaks down"
        " the research process into actionable steps for a search agent to execute.\n"
        "Process:\n"
        "Expand the Query: 1. Clarify and enrich the user’s query by considering related aspects, possible interpretations,"
        "and necessary contextual details. 2.Identify any ambiguities and resolve them by assuming the most logical and useful framing of the problem.\n"
        "Identify Key Research Areas: 1. Break down the expanded query into core themes, subtopics, or dimensions of investigation."
        "2.Determine what information is necessary to provide a comprehensive answer.\n"
        "Define Research Steps: 1. Outline a structured plan with clear steps that guide the search agent on how to gather information."
        "2. Specify which sources or types of data are most relevant (e.g., academic papers, government reports, news sources, expert opinions)."
        "3. Prioritize steps based on importance and logical sequence.\n"
        "Suggest Search Strategies: 1.Recommend search terms, keywords, and boolean operators to optimize search efficiency."
        "2. Identify useful databases, journals, and sources where high-quality information can be found."
        "3. Suggest methodologies for verifying credibility and synthesizing findings.\n"
        "NO EXPLANATIONS, write plans ONLY!\n"
    )
    messages = [
        {"role": "system", "content": "You are an advanced reasoning LLM that guides a following search agent to search for relevant information for a research."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = await (call_ollama_async(session, messages, model=REASON_MODEL, ctx=REASON_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages, model=REASON_MODEL))
    if response:
        try:
            # Remove <think>...</think> tags and their content if they exist
            cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            print(f"Plan:{cleaned_response}")
            return cleaned_response
        except Exception as e:
            print(f"Error processing response: {e}")
    return []

async def judge_search_result_and_future_plan_aync(session, user_query, original_plan, context_combined, truncated=False, note_from_previous=""):
    """
    Ask the reasoning LLMs to judge the result of the search attempt and produce a plan for next interation.
    """
    base_prompt = (
    "You are an advanced reasoning LLM that specializes in evaluating research results and refining search strategies. "
    "Your task is to analyze the search agent's findings, assess their relevance and completeness, "
    "and generate a structured plan for the next search iteration. Your goal is to ensure a thorough and efficient research process "
    "that ultimately provides a comprehensive answer to the user's query. But still, if you think everything is enough, you can tell search agent to stop\n"
    "Process:\n"
    "1. **Evaluate Search Results:**\n"
    "   - Analyze the retrieved search results to determine their relevance, credibility, and completeness.\n"
    "   - Identify missing information, knowledge gaps, or weak sources.\n"
    "   - Assess whether the search results sufficiently address the key research areas from the original plan.\n"
    "   - If everything is enough, tell search agent to stop with your reason\n"
    "2. **Determine Next Steps:**\n"
    "   - Based on gaps identified, refine or expand the research focus.\n"
    "   - Suggest additional search directions or alternative sources to explore.\n"
    "   - If necessary, propose adjustments to search strategies, including keyword modifications, new sources, or filtering techniques.\n"
    "3. **Generate an Updated Research Plan:**\n"
    "   - Provide a structured step-by-step plan for the next search iteration.\n"
    "   - Clearly outline what aspects need further investigation and where the search agent should focus next.\n"
    "NO EXPLANATIONS, write plans ONLY!\n"
    )

    if truncated:
        prompt = base_prompt + (
            f"\n{note_from_previous}\n"
            f"User Query: {user_query}\n"
            f"Original Research Plan: {original_plan}\n"
            "IMPORTANT: Due to context length limit, you can only see a part of the full context at a time, and this is only a partial context." "Please add comments on how well this part addresses the query, "
            "what issues remain unresolved, and what needs further clarification. These comments will be used when "
            "finally reviewing all parts of the context together. Note: since the final review agent will ONLY see your comments, make sure to provide a clear and detailed assessment."
        )
    elif note_from_previous:
        prompt = base_prompt + (
            "\nDue to context length limit, the gathered context analysis is done by seeing a small part of the full context at a time. "
            f"\nBased on the previous analysis comments of different parts of the context:\n{note_from_previous}\n"
            f"User Query: {user_query}\n"
            f"Original Research Plan: {original_plan}\n"
            "Now, synthesize these comments to generate a comprehensive research plan for the next iteration."
        )
    else:
        prompt = base_prompt + "Now, based on the above information and instruction, evaluate the search results and generate a refined research plan for the next iteration."

    messages = [
        {"role": "system", "content": "You are an advanced reasoning LLM that guides a following search agent to search for relevant information for a research."},
        {"role": "user", "content": f"User Query: {user_query}\n\n Original Research Plan: {original_plan} \n\n" + (f"Extracted Relevant Contexts by the search agent:\n{context_combined}" if context_combined else "") + f"\n\n{prompt}"}
    ]
    response = await (call_ollama_async(session, messages, model=REASON_MODEL, ctx=REASON_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages, model=REASON_MODEL))
    if response:
        try:
            # Remove <think>...</think> tags and their content if they exist
            cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            print(f"Next Plan:{cleaned_response}")
            return cleaned_response
        except Exception as e:
            print(f"Error processing response: {e}")
    return []

async def generate_writing_plan_aync(session, user_query, aggregated_contexts, truncated=False, note_from_previous=""):
    """
    Ask the reasoning LLM to generate a structured writing plan for the final report based on the aggregated research findings.
    """
    base_prompt = (
        "You are an advanced reasoning LLM that specializes in structuring comprehensive research reports. "
        "Your task is to analyze the aggregated research findings from previous searches and generate a structured plan "
        "for writing the final report. The goal is to ensure that the report is well-organized, complete, and effectively presents the findings.\n\n"
        "Process:\n"
        "1. **Analyze Aggregated Findings:**\n"
        "   - Review the provided research findings for relevance, accuracy, and coherence.\n"
        "   - Identify key insights, supporting evidence, and significant conclusions.\n"
        "   - Highlight any inconsistencies or areas that may need further clarification.\n\n"
        "2. **Determine Report Structure:**\n"
        "   - Define a clear outline with logical sections that effectively communicate the research results.\n"
        "   - Ensure the structure follows a coherent flow, such as Introduction, Key Findings, Analysis, Conclusion, and References.\n"
        "   - Specify where different pieces of evidence should be integrated within the report.\n\n"
        "3. **Provide Writing Guidelines:**\n"
        "   - Suggest the tone and style appropriate for the report (e.g., formal, analytical, concise).\n"
        "   - Recommend how to synthesize multiple sources into a coherent narrative.\n"
        "   - If any section lacks sufficient information, indicate where additional elaboration may be needed.\n\n"
        "NO EXPLANATIONS, write plans ONLY!\n"
    )
    
    if truncated:
        if note_from_previous:
            # This is a subsequent iteration with a previous plan
            prompt = base_prompt + (
                f"User Query: {user_query}\n\n"
                f"Previous Writing Plan:\n{note_from_previous}\n\n"
                "Due to context length limit, you can only see a part of the full context at a time. "
                "This is a previous version of the writing plan generated with previous parts of the full context."
                "Now you have new part of the full context, please update and refine this plan "
                "based on the additional information provided. Generate a complete, updated writing plan.\n"
            )
        else:
            # This is the first iteration
            prompt = base_prompt + (
                f"User Query: {user_query}\n\n"
                "Due to context length limit, you can only see a part of the full context at a time. "
                "This is the first part of the context. Generate a complete writing plan based on this initial information. "
                "This plan will be refined later with new context.\n"
            )
    else:
        if note_from_previous:
            # This is a finial iteration with a previous plan
            prompt = base_prompt + (
                f"User Query: {user_query}\n\n"
                f"Previous Writing Plan:\n{note_from_previous}\n\n"
                "Due to context length limit, you can only see a part of the full context at a time. "
                "This is a previous version of the writing plan generated with previous parts of the full context."
                "Now you have the FINAL part of the full context, please update and refine this plan "
                "based on the additional information provided. Generate a complete, updated FINAL writing plan.\n"
            )
        else:
            prompt = base_prompt + ("Now, based on the above instructions, generate a structured plan for writing the final research report.\n")

    messages = [
        {"role": "system", "content": "You are an advanced reasoning LLM that structures research findings into a well-organized final report."},
        {"role": "user", "content": f"User Query: {user_query}\n\nAggregated Research Findings:\n{aggregated_contexts}\n\n{prompt}"}
    ]
    response = await (call_ollama_async(session, messages, model=REASON_MODEL, ctx=REASON_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages, model=REASON_MODEL))
    if response:
        try:
            # Remove <think>...</think> tags and their content if they exist
            cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            print(f"Writing Plan: {cleaned_response}")
            return cleaned_response
        except Exception as e:
            print(f"Error processing response: {e}")
    return []

    
async def generate_search_queries_async(session, query_plan):
    """
    Ask the LLM to produce up to four precise search queries (in Python list format)
    based on the user’s query.
    """
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Return ONLY a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"{query_plan}\n\n{prompt}"}
    ]
    response = await (call_ollama_async(session, messages,ctx=DEFAULT_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages))
    if response:
        cleaned = response.strip()
        
        # Remove triple backticks and language specifier if present
        cleaned = re.sub(r"```(?:\w+)?\n(.*?)\n```", r"\1", cleaned, flags=re.DOTALL).strip()

        # First, try to directly evaluate the cleaned string
        try:
            new_queries = ast.literal_eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
        except Exception as e:
            # Direct evaluation failed; try to extract the list part from the string.
            match = re.search(r'(\[.*\])', cleaned, re.DOTALL)
            if match:
                list_str = match.group(1)
                try:
                    new_queries = ast.literal_eval(list_str)
                    if isinstance(new_queries, list):
                        return new_queries
                except Exception as e_inner:
                    print("Error parsing extracted list:", e_inner, "\nExtracted text:", list_str)
                    return []
            print("Error parsing new search queries:", e, "\nResponse:", response)
            return []
    return []


async def perform_search_async(session, query):
    params = {
        "q": query,
        "format": "json"
    }
    try:
        async with session.get(BASE_SEARXNG_URL, params=params) as resp:
            if resp.status == 200:
                results = await resp.json()
                if "results" in results:
                    # Extract the URLs from the returned results list.
                    links = [item.get("url") for item in results["results"] if "url" in item]
                    return links
                else:
                    print("No results in SearXNG response.")
                    return []
            else:
                text = await resp.text()
                print(f"SearXNG error: {resp.status} - {text}")
                return []
    except Exception as e:
        print("Error performing SearXNG search:", e)
        return []


async def is_page_useful_async(session, user_query, page_text, page_url):
    """
    Ask the LLM if the provided webpage content is useful for answering the user's query.
    The LLM must reply with exactly "Yes" or "No".
    """
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information relevant and useful for addressing the query. "
        "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWeb URL: {page_url}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await (call_ollama_async(session, messages, ctx=DEFAULT_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages))
    if response:
        answer = response.strip()
        if answer in ["Yes", "No"]:
            return answer
        else:
            # Fallback: try to extract Yes/No from the response.
            if "Yes" in answer:
                return "Yes"
            elif "No" in answer:
                return "No"
    return "No"


async def extract_relevant_context_async(session, user_query, search_query, page_text, page_url):
    """
    Given the original query, the search query used, and the page content,
    have the LLM extract all information relevant for answering the query.
    """
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
        "Return only the relevant context as plain text without commentary."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWeb URL: {page_url}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await (call_ollama_async(session, messages, ctx=DEFAULT_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages))
    if response and not response.strip().startswith("Error:"):
        return response.strip()
    return ""


async def get_new_search_queries_async(session, user_query, new_research_plan, previous_search_queries, all_contexts, truncated=False, note_from_previous=""):
    """
    Based on the original query, the previously used search queries, and all the extracted contexts,
    ask the LLM whether additional search queries are needed. If yes, return a Python list of up to four queries;
    if the LLM thinks research is complete, it should return "<done>".
    """
    context_combined = "\n".join(all_contexts)
    
    base_prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "the next step plan by a planning agent and the extracted contexts from webpages, determine if further research is needed. "
    )
    
    if truncated:
        prompt = base_prompt + (
            f"\n{note_from_previous}\n"
            f"User Query: {user_query}\n"
            f"Previous Search Queries: {previous_search_queries}\n"
            + (f"Next Research Plan by planning agent:\n{new_research_plan}\n" if new_research_plan else "")
            + "IMPORTANT: Due to context length limit, you can only see a part of the full context at a time. "
            + "so this is only a partial context. Please add comments on how well this part addresses the query, "
            "what issues remain unresolved, and what needs further clarification. These comments will be used when "
            "reviewing all parts of the context together."
        )
    elif note_from_previous:
        prompt = base_prompt + (
            "Due to context length limit, you can only see part of the full context at a time, "
            "so you were previously asked to provide analysis comments on each parts of the full context provided. "
            + f"\nBased on those previous analysis comments of different parts of the context:\n{note_from_previous}\n"
            f"User Query: {user_query}\n"
            f"Previous Search Queries: {previous_search_queries}\n"
            + (f"Next Research Plan by planning agent:\n{new_research_plan}\n" if new_research_plan else "")
            + "Now, synthesize these comments to determine if further research is needed."
            + "If further research is needed, ONLY provide up to four new search queries as a Python list IN ONE LINE (for example, "
            + "['new query1', 'new query2']) in PLAIN text, NEVER wrap in code env. If you believe no further research is needed, respond with exactly <done>."
            + "\nREMEMBER: Output ONLY a Python list or the token <done> WITHOUT any additional text or explanations."
        )
    else:
        prompt = base_prompt + (
        "If further research is needed, ONLY provide up to four new search queries as a Python list IN ONE LINE (for example, "
        "['new query1', 'new query2']) in PLAIN text, NEVER wrap in code env. If you believe no further research is needed, respond with exactly <done>."
        "\nREMEMBER: Output ONLY a Python list or the token <done> WITHOUT any additional text or explanations."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n" + (f"\nExtracted Relevant Contexts:\n{context_combined}" if context_combined else "") + (f"\n\nNext Research Plan by planning agent:\n{new_research_plan}" if new_research_plan else "") + f"\n\n{prompt}"}
    ]
    response = await (call_ollama_async(session, messages, ctx=DEFAULT_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages))
    if response and not truncated:
        cleaned = response.strip()
        if cleaned == "<done>":
            return "<done>"
        # Remove triple backticks and language specifier if present
        cleaned = re.sub(r"```(?:\w+)?\n(.*?)\n```", r"\1", cleaned, flags=re.DOTALL).strip()
        # First, try to directly evaluate the cleaned string
        try:
            new_queries = ast.literal_eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
        except Exception as e:
            # Direct evaluation failed; try to extract the list part from the string.
            match = re.search(r'(\[.*\])', cleaned, re.DOTALL)
            if match:
                list_str = match.group(1)
                try:
                    new_queries = ast.literal_eval(list_str)
                    if isinstance(new_queries, list):
                        return new_queries
                except Exception as e_inner:
                    print("Error parsing extracted list:", e_inner, "\nExtracted text:", list_str)
                    return []
            print("Error parsing new search queries:", e, "\nResponse:", response)
            return []
    elif response and truncated:
        return response
    return []



async def generate_final_report_async(session, system_instruction, user_query, report_planning, all_contexts, truncated=False, note_from_previous=""):
    """
    Generate the final comprehensive report using all gathered contexts.
    """
    context_combined = "\n".join(all_contexts)
    base_prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts above and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all relevant insights and conclusions without extraneous commentary."
        "Math equations should use proper LaTeX syntax in markdown format, with \\(\\LaTeX{}\\) for inline, $$\\LaTeX{}$$ for block."
        "Properly cite all the VALID and REAL sources inline from 'Gathered Relevant Contexts' with [cite_number]"
        "and also summarize the corresponding bibliography list with their urls in markdown format in the end of your report."
        "Ensure that all VALID and REAL sources from 'Gathered Relevant Contexts' that you have used to write this report or back your"
        "statements are properly cited inline using the [cite_number] format (e.g., [1], [2], etc.)."
        "Then, append a complete bibliography section at the end of your report in markdown format, "
        "listing each source with its corresponding URL. Please NEVER omit the bibliography."
        "REMEMBER: NEVER make up sources or citations, only use the provided contexts, if no source used or available,"
        "just write 'No available sources'."
    )
    
    if truncated:
        if note_from_previous:
            # This is a subsequent iteration with a previous report
            prompt = base_prompt + (
                f"\nUser Query: {user_query}\n\n"
                f"Previous Report Draft:\n{note_from_previous}\n\n"
                "Due to context length limit, you can only see a part of the full context at a time. "
                "Previous Report Draft is a previous version of the report written by you based on some parts of the full context you cannot see anymore. "
                "But those parts are still valid and should be considered as correct."
                "Now you have a new part of the full context, please update and refine this report"
                "based on the additional information provided. Generate a complete, updated report that incorporates both "
                "the previous content and the new information. Ensure all citations are properly kept, maintained, and updated."
            )
        else:
            # This is the first iteration
            prompt = base_prompt + (
                f"\nUser Query: {user_query}\n\n"
                "Due to context length limit, you can only see a part of the full context at a time. "
                "This is the first part of the full context. Generate a complete report based on this initial information. "
                "This report will be refined later with further context. Ensure you also include the complete citation list, as the refinement agent will not see them any more."
            )
    else:
        if note_from_previous:
            # This is the final report generation
            prompt = base_prompt + (
                f"\nUser Query: {user_query}\n\n"
                f"Previous Report Draft:\n{note_from_previous}\n\n"
                "Due to context length limit, you can only see a part of the full context at a time. "
                "Previous Report Draft is a previous version of the report written by you based on some parts of the full context you cannot see anymore. "
                "But those parts are still valid and should be considered as correct."
                "And the context provided this time will be the final part of the full context. "
                "Based on the previous report draft and the new information provided, "
                "generate a complete, final report that integrates all relevant insights and conclusions. "
                "Ensure all citations are properly kept, maintained, and updated. "
                "Append a complete bibliography section at the end of your report in markdown format, listing each source with its corresponding URL."
            )
        else:
            prompt = base_prompt
    
    messages = [
        {"role": "system", "content": "You are a skilled report writer." + (f"There is also some extra system instructions: {system_instruction}" if system_instruction else "")},
        {"role": "user", "content": f"User Query: {user_query}\n\nGathered Relevant Contexts:\n{context_combined}" + (f"\n\nWriting plan from a planning agent:\n{report_planning}" if report_planning and not report_planning.startswith("Error:") else "") + f"\n\nWriting Instructions:{prompt}"}
    ]
    report = await (call_ollama_async(session, messages, ctx=DEFAULT_MODEL_CTX) if USE_OLLAMA else call_openrouter_async(session, messages))
    return report

async def process_link(session, link, user_query, search_query, create_chunk=None):
    """
    Process a single link with all operations running concurrently while maintaining proper status streaming.
    """
    # Initial status message
    status_msg = f"Fetching content from: {link}\n\n"
    if create_chunk:
        yield create_chunk(status_msg)
    else:
        print(status_msg)

    try:
        # Create fetch task immediately
        fetch_task = asyncio.create_task(fetch_webpage_text_async(session, link))
        
        # Wait for fetch to complete
        page_text = await fetch_task
        if not page_text:
            return

        # Create usefulness task immediately
        usefulness_task = asyncio.create_task(is_page_useful_async(session, user_query, page_text, link))
        
        # Create context task but don't await it yet
        context_task = asyncio.create_task(extract_relevant_context_async(session, user_query, search_query, page_text, link))
        
        # Wait for usefulness check and stream its result
        usefulness = await usefulness_task
        status_msg = f"Page usefulness for {link}: {usefulness}\n\n"
        if create_chunk:
            yield create_chunk(status_msg)
        else:
            print(status_msg)

        # If useful, wait for context
        if usefulness == "Yes":
            context = await context_task
            if context:
                status_msg = f"Extracted context from {link} (first 200 chars): {context[:200]}\n\n"
                if create_chunk and VERBOSE_WEB_PARSE:
                    yield create_chunk(status_msg)
                else:
                    print(status_msg)
                context = "url:" + link + "\ncontext:" + context
                yield context
        else:
            # Cancel context task if not useful
            context_task.cancel()
    except Exception as e:
        print(f"Error processing {link}: {e}")
    return


# =========================
# Main Asynchronous Routine
# =========================

async def process_research(system_instruction: str, user_query: str, max_iterations: int = 10, max_search_items: int = 4):
    """Core research processing function that returns the final report"""
    iteration_limit = max_iterations
    aggregated_contexts = []
    all_search_queries = []
    iteration = 0

    async with aiohttp.ClientSession() as session:
        # Initial research plan
        if WITH_PLANNING:
            research_plan = await make_initial_searching_plan_async(session, user_query)
            if isinstance(research_plan, list):
                research_plan = ""
            initial_query = "User Query:" + user_query + ("\n\nResearch Plan:" + str(research_plan) if research_plan and not research_plan.startswith("Error:") else "")
        else:
            research_plan = None
            initial_query = "User Query:" + user_query

        # Generate initial search queries
        new_search_queries = await generate_search_queries_async(session, initial_query)
        if not new_search_queries:
            return "No search queries could be generated."
        all_search_queries.extend(new_search_queries)

        # Main research loop
        while iteration < iteration_limit:
            # Perform searches
            iteration_contexts = []
            search_tasks = [asyncio.create_task(perform_search_async(session, query)) for query in new_search_queries]
            full_search_results = await asyncio.gather(*search_tasks)
            
            search_results = [result[:max_search_items] for result in full_search_results] if not USE_JINA else full_search_results

            # Process unique links
            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query

            # Process all links truly concurrently
            async def process_link_wrapper(link):
                results = []
                async for result in process_link(session, link, user_query, unique_links[link]):
                    if isinstance(result, str) and result.startswith("url:"):
                        results.append(result)
                return results

            # Create and run all tasks concurrently
            tasks = [process_link_wrapper(link) for link in unique_links]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for results in all_results:
                if isinstance(results, list):  # Skip any exceptions
                    iteration_contexts.extend(results)

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)

            # Check if we should continue
            if iteration + 1 < iteration_limit:
                if WITH_PLANNING:
                    # Check if we need to handle truncation for the reasoning model
                    if REASON_MODEL_MAX_INPUT > 8000:
                        # Estimate tokens in the aggregated contexts
                        context_combined = "\n".join(aggregated_contexts)
                        context_tokens = estimate_tokens(context_combined)
                        
                        # Account for prompt overhead
                        available_tokens = REASON_MODEL_MAX_INPUT - JUDGE_PROMPT_OVERHEAD
                        
                        # If context is too large for the model's capacity
                        if context_tokens > available_tokens:
                            # If we can only fit less than 25% of the context, abort
                            if context_tokens > 4 * available_tokens:
                                new_research_plan = "reasoning model has too small input token length that takes more than 4 iterations to finish judging search results, aborting..."
                            else:
                                # Split context into manageable chunks using the helper function
                                chunks = split_contexts_into_chunks(aggregated_contexts, REASON_MODEL_MAX_INPUT, JUDGE_PROMPT_OVERHEAD)
                                
                                # Process each chunk and collect comments
                                all_comments = []
                                for i, chunk in enumerate(chunks):
                                    chunk_comment = await judge_search_result_and_future_plan_aync(
                                        session,
                                        user_query,
                                        research_plan,
                                        chunk,
                                        truncated=True,
                                        note_from_previous=f"This is part {i+1} of {len(chunks)} of the context."
                                    )
                                    all_comments.append(f"Comments on part {i+1}: {chunk_comment}")
                                
                                # Make final call with all comments
                                new_research_plan = await judge_search_result_and_future_plan_aync(
                                    session,
                                    user_query,
                                    research_plan,
                                    "Context too large to process at once. See notes from processing each part.",
                                    truncated=False,
                                    note_from_previous="\n\n".join(all_comments)
                                )
                        else:
                            # Context fits within model's capacity
                            new_research_plan = await judge_search_result_and_future_plan_aync(session, user_query, research_plan, context_combined)
                    else:
                        # No input limit, proceed normally
                        new_research_plan = await judge_search_result_and_future_plan_aync(session, user_query, research_plan, "\n".join(aggregated_contexts))
                else:
                    new_research_plan = None

                # Check if we need to handle truncation for the default model
                if DEFAULT_MODEL_MAX_INPUT > 8000:
                    # Estimate tokens in the aggregated contexts
                    context_combined = "\n".join(aggregated_contexts)
                    context_tokens = estimate_tokens(context_combined)
                    
                    # Account for prompt overhead
                    available_tokens = DEFAULT_MODEL_MAX_INPUT - SEARCH_PROMPT_OVERHEAD
                    
                    # If context is too large for the model's capacity
                    if context_tokens > available_tokens:
                        # If we can only fit less than 25% of the context, abort
                        if context_tokens > 4 * available_tokens:
                            new_search_queries = "default model has too small input token length that takes more than 4 iterations to finish generating search queries, aborting..."
                        else:
                            # Split context into manageable chunks using the helper function
                            chunks = split_contexts_into_chunks(aggregated_contexts, DEFAULT_MODEL_MAX_INPUT, SEARCH_PROMPT_OVERHEAD)
                            
                            # Process each chunk and collect comments
                            all_comments = []
                            for i, chunk in enumerate(chunks):
                                chunk_comment = await get_new_search_queries_async(
                                    session,
                                    user_query,
                                    new_research_plan,
                                    all_search_queries,
                                    [chunk],  # the function expects a list
                                    truncated=True,
                                    note_from_previous=f"This is part {i+1} of {len(chunks)} of the context."
                                )
                                all_comments.append(f"Comments on part {i+1}: {chunk_comment}")
                            
                            # Make final call with all comments
                            new_search_queries = await get_new_search_queries_async(
                                session,
                                user_query,
                                new_research_plan,
                                all_search_queries,
                                [],
                                truncated=False,
                                note_from_previous="\n\n".join(all_comments)
                            )
                    else:
                        # Context fits within model's capacity
                        new_search_queries = await get_new_search_queries_async(session, user_query, new_research_plan, all_search_queries, aggregated_contexts)
                else:
                    # No input limit, proceed normally
                    new_search_queries = await get_new_search_queries_async(session, user_query, new_research_plan, all_search_queries, aggregated_contexts)
                if new_search_queries == "<done>" or not new_search_queries:
                    break
                all_search_queries.extend(new_search_queries)

            iteration += 1
            if OPERATION_WAIT_TIME > 0:
                await asyncio.sleep(OPERATION_WAIT_TIME)

        # Generate final report
        if WITH_PLANNING:
            # Check if we need to handle truncation for the writing plan
            if REASON_MODEL_MAX_INPUT > 8000:
                # Estimate tokens in the aggregated contexts
                context_combined = "\n".join(aggregated_contexts)
                context_tokens = estimate_tokens(context_combined)
                
                # Account for prompt overhead
                available_tokens = REASON_MODEL_MAX_INPUT - WRITING_PROMPT_OVERHEAD
                
                # If context is too large for the model's capacity
                if context_tokens > available_tokens:
                    # If we can only fit less than 25% of the context, abort
                    if context_tokens > 4 * available_tokens:
                        final_report_planning = "reasoning model has too small input token length that takes more than 4 iterations to finish generating writing plan, aborting..."
                    else:
                        # Split context into manageable chunks using the helper function
                        chunks = split_contexts_into_chunks(aggregated_contexts, REASON_MODEL_MAX_INPUT, WRITING_PROMPT_OVERHEAD)
                        
                        # Process each chunk and collect plans
                        last_plan = None
                        for i, chunk in enumerate(chunks):
                            chunk_text = '\n'.join(chunk)
                            current_plan = await generate_writing_plan_aync(
                                session,
                                user_query,
                                chunk_text,
                                truncated=True,
                                note_from_previous=last_plan
                            )
                            last_plan = current_plan
                        
                        final_report_planning = last_plan
                else:
                    # Context fits within model's capacity
                    final_report_planning = await generate_writing_plan_aync(session, user_query, aggregated_contexts)
            else:
                # No input limit, proceed normally
                final_report_planning = await generate_writing_plan_aync(session, user_query, aggregated_contexts)
        else:
            final_report_planning = None

        # Check if we need to handle truncation for the final report
        if DEFAULT_MODEL_MAX_INPUT > 8000:
            # Estimate tokens in the aggregated contexts
            context_combined = "\n".join(aggregated_contexts)
            context_tokens = estimate_tokens(context_combined)
            
            # Account for prompt overhead
            available_tokens = DEFAULT_MODEL_MAX_INPUT - REPORT_PROMPT_OVERHEAD
            
            # If context is too large for the model's capacity
            if context_tokens > available_tokens:
                # If we can only fit less than 25% of the context, abort
                if context_tokens > 4 * available_tokens:
                    final_report = "default model has too small input token length that takes more than 4 iterations to finish generating final report, aborting..."
                else:
                    # Split context into manageable chunks using the helper function
                    chunks = split_contexts_into_chunks(aggregated_contexts, DEFAULT_MODEL_MAX_INPUT, REPORT_PROMPT_OVERHEAD)
                    
                    # Process each chunk and refine the report iteratively
                    last_report = None
                    for i, chunk in enumerate(chunks):
                        chunk_text = '\n'.join(chunk)
                        current_report = await generate_final_report_async(
                            session,
                            system_instruction,
                            user_query,
                            final_report_planning,
                            chunk_text,
                            truncated=True,
                            note_from_previous=last_report
                        )
                        last_report = current_report
                    
                    final_report = last_report
                    final_report += "\n\nNote: This report was generated by iteratively processing the research results due to model context limitations."
            else:
                # Context fits within model's capacity
                final_report = await generate_final_report_async(session, system_instruction, user_query, final_report_planning, aggregated_contexts)
        else:
            # No input limit, proceed normally
            final_report = await generate_final_report_async(session, system_instruction, user_query, final_report_planning, aggregated_contexts)

        if not final_report or len(final_report) < 200:
            final_report = (final_report or "") + "\n" + "We may encounter an error while writing, usually due to rate limit or context length.\n These are the original writing prompt, please copy it and try again with anothor model\n" + f"User Query: {user_query}\n\nGathered Relevant Contexts:\n" + "\n".join(aggregated_contexts) + (f"\n\nWriting plan from a planning agent:\n{final_report_planning}" if final_report_planning else "") + "You are an expert researcher and report writer. Based on the gathered contexts above and the original query, write a comprehensive, well-structured, and detailed report that addresses the query thoroughly."

        return final_report
    
async def stream_research(system_instruction: str, user_query: str, max_iterations: int = 10, max_search_items: int = 4):
    """Generator function for streaming research results"""
    def create_chunk(content: str) -> str:
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{int(time.time()*1000)}",
            created=int(time.time()),
            model="deep_researcher",
            choices=[{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        )
        return f"data: {chunk.json()}\n\n"

    async with aiohttp.ClientSession() as session:
        yield create_chunk("\n<think>\n\n")
        # Initial research plan
        if WITH_PLANNING:
            research_plan = await make_initial_searching_plan_async(session, user_query)
            if isinstance(research_plan, list):
                research_plan = ""
            if research_plan:
                yield create_chunk(f"Initial Research Plan:\n{research_plan}\n\n")
            initial_query = "User Query:" + user_query + ("\n\nResearch Plan:" + str(research_plan) if research_plan and not research_plan.startswith("Error:") else "")
        else:
            research_plan = None
            initial_query = "User Query:" + user_query

        # Generate initial search queries
        new_search_queries = await generate_search_queries_async(session, initial_query)
        if not new_search_queries:
            yield create_chunk("No search queries were generated. Exiting.\n\n")
            yield "data: [DONE]\n\n"
            return

        iteration_limit = max_iterations
        aggregated_contexts = []
        all_search_queries = []
        iteration = 0
        all_search_queries.extend(new_search_queries)

        # Main research loop
        while iteration < iteration_limit:
            yield create_chunk(f"\n=== Iteration {iteration + 1} ===\n\n")
            
            # Perform searches
            iteration_contexts = []
            search_tasks = [asyncio.create_task(perform_search_async(session, query)) for query in new_search_queries]
            full_search_results = await asyncio.gather(*search_tasks)
            
            search_results = [result[:max_search_items] for result in full_search_results] if not USE_JINA else full_search_results

            # Process unique links
            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query

            yield create_chunk(f"Processing {len(unique_links)} unique links...\n\n")

            # Process all links concurrently while maintaining streaming
            async def process_link_wrapper(link):
                results = []
                status_updates = []
                async for result in process_link(session, link, user_query, unique_links[link], create_chunk):
                    if isinstance(result, str):
                        if result.startswith("url:"):
                            results.append(result)
                        else:
                            status_updates.append(result)
                return results, status_updates

            # Create tasks for all links
            tasks = [process_link_wrapper(link) for link in unique_links]
            
            # Process results as they complete
            for completed in asyncio.as_completed(tasks):
                try:
                    results, status_updates = await completed
                    # Stream status updates
                    for update in status_updates:
                        yield update
                    # Add contexts
                    iteration_contexts.extend(results)
                except Exception as e:
                    print(f"Error in link processing: {e}")

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
            else:
                yield create_chunk("No useful contexts found in this iteration.\n\n")

            # Check if we should continue
            if iteration + 1 < iteration_limit:
                if WITH_PLANNING:
                    # Check if we need to handle truncation for the reasoning model
                    if REASON_MODEL_MAX_INPUT > 8000:
                        # Estimate tokens in the aggregated contexts
                        context_combined = "\n".join(aggregated_contexts)
                        context_tokens = estimate_tokens(context_combined)
                        
                        # Account for prompt overhead
                        available_tokens = REASON_MODEL_MAX_INPUT - JUDGE_PROMPT_OVERHEAD
                        
                        # If context is too large for the model's capacity
                        if context_tokens > available_tokens:
                            # If we can only fit less than 25% of the context, abort
                            if context_tokens > 4 * available_tokens:
                                new_research_plan = "reasoning model has too small input token length that takes more than 4 iterations to finish judging search results, aborting..."
                                yield create_chunk(f"Warning: {new_research_plan}\n\n")
                            else:
                                # Split context into manageable chunks using the helper function
                                chunks = split_contexts_into_chunks(aggregated_contexts, REASON_MODEL_MAX_INPUT, JUDGE_PROMPT_OVERHEAD)
                                
                                yield create_chunk(f"Context too large for reasoning model, splitting into {len(chunks)} chunks...\n\n")
                                
                                # Process each chunk and collect comments
                                all_comments = []
                                for i, chunk in enumerate(chunks):
                                    yield create_chunk(f"Processing chunk {i+1}/{len(chunks)}...\n\n")
                                    chunk_comment = await judge_search_result_and_future_plan_aync(
                                        session,
                                        user_query,
                                        research_plan,
                                        chunk,
                                        truncated=True,
                                        note_from_previous=f"This is part {i+1} of {len(chunks)} of the context."
                                    )
                                    all_comments.append(f"Comments on part {i+1}: {chunk_comment}")
                                
                                # Make final call with all comments
                                yield create_chunk("Synthesizing comments from all chunks...\n\n")
                                new_research_plan = await judge_search_result_and_future_plan_aync(
                                    session,
                                    user_query,
                                    research_plan,
                                    "Context too large to process at once. See notes from processing each part.",
                                    truncated=False,
                                    note_from_previous="\n\n".join(all_comments)
                                )
                        else:
                            # Context fits within model's capacity
                            new_research_plan = await judge_search_result_and_future_plan_aync(session, user_query, research_plan, context_combined)
                    else:
                        # No input limit, proceed normally
                        new_research_plan = await judge_search_result_and_future_plan_aync(session, user_query, research_plan, "\n".join(aggregated_contexts))
                    
                    yield create_chunk(f"Updated Research Plan:\n{new_research_plan}\n\n")
                else:
                    new_research_plan = None

                # Check if we need to handle truncation for the default model
                if DEFAULT_MODEL_MAX_INPUT > 8000:
                    # Estimate tokens in the aggregated contexts
                    context_combined = "\n".join(aggregated_contexts)
                    context_tokens = estimate_tokens(context_combined)
                    
                    # Account for prompt overhead
                    available_tokens = DEFAULT_MODEL_MAX_INPUT - SEARCH_PROMPT_OVERHEAD
                    
                    # If context is too large for the model's capacity
                    if context_tokens > available_tokens:
                        # If we can only fit less than 25% of the context, abort
                        if context_tokens > 4 * available_tokens:
                            new_search_queries = "default model has too small input token length that takes more than 4 iterations to finish generating search queries, aborting..."
                            yield create_chunk(f"Warning: {new_search_queries}\n\n")
                        else:
                            # Split context into manageable chunks using the helper function
                            chunks = split_contexts_into_chunks(aggregated_contexts, DEFAULT_MODEL_MAX_INPUT, SEARCH_PROMPT_OVERHEAD)
                            
                            yield create_chunk(f"Context too large for default model, splitting into {len(chunks)} chunks...\n\n")
                            
                            # Process each chunk and collect comments
                            all_comments = []
                            for i, chunk in enumerate(chunks):
                                yield create_chunk(f"Processing chunk {i+1}/{len(chunks)}...\n\n")
                                chunk_comment = await get_new_search_queries_async(
                                    session,
                                    user_query,
                                    new_research_plan,
                                    all_search_queries,
                                    [chunk],  # the function expects a list
                                    truncated=True,
                                    note_from_previous=f"This is part {i+1} of {len(chunks)} of the context."
                                )
                                all_comments.append(f"Comments on part {i+1}: {chunk_comment}")
                            
                            # Make final call with all comments
                            yield create_chunk("Synthesizing comments from all chunks...\n\n")
                            new_search_queries = await get_new_search_queries_async(
                                session,
                                user_query,
                                new_research_plan,
                                all_search_queries,
                                [],
                                truncated=False,
                                note_from_previous="\n\n".join(all_comments)
                            )
                    else:
                        # Context fits within model's capacity
                        new_search_queries = await get_new_search_queries_async(session, user_query, new_research_plan, all_search_queries, aggregated_contexts)
                else:
                    # No input limit, proceed normally
                    new_search_queries = await get_new_search_queries_async(session, user_query, new_research_plan, all_search_queries, aggregated_contexts)
                if new_search_queries == "<done>":
                    yield create_chunk("Research complete. Generating final report...\n\n")
                    break
                elif new_search_queries:
                    yield create_chunk(f"New search queries generated: {new_search_queries}\n")
                    all_search_queries.extend(new_search_queries)
                else:
                    yield create_chunk("No new search queries. Completing research...\n\n")
                    break

            iteration += 1
            if OPERATION_WAIT_TIME > 0:
                await asyncio.sleep(OPERATION_WAIT_TIME)

        yield create_chunk("\nGenerating final report...\n\n")

        # Generate final report
        if WITH_PLANNING:
            # Check if we need to handle truncation for the writing plan
            if REASON_MODEL_MAX_INPUT > 8000:
                # Estimate tokens in the aggregated contexts
                context_combined = "\n".join(aggregated_contexts)
                context_tokens = estimate_tokens(context_combined)
                
                # Account for prompt overhead
                available_tokens = REASON_MODEL_MAX_INPUT - WRITING_PROMPT_OVERHEAD
                
                # If context is too large for the model's capacity
                if context_tokens > available_tokens:
                    # If we can only fit less than 25% of the context, abort
                    if context_tokens > 4 * available_tokens:
                        final_report_planning = "reasoning model has too small input token length that takes more than 4 iterations to finish generating writing plan, aborting..."
                        yield create_chunk(f"Warning: {final_report_planning}\n\n")
                    else:
                        # Split context into manageable chunks using the helper function
                        chunks = split_contexts_into_chunks(aggregated_contexts, REASON_MODEL_MAX_INPUT, WRITING_PROMPT_OVERHEAD)
                        
                        yield create_chunk(f"Context too large for reasoning model, splitting into {len(chunks)} chunks for writing plan...\n\n")
                        
                        # Process each chunk and collect plans
                        last_plan = None
                        for i, chunk in enumerate(chunks):
                            yield create_chunk(f"Processing writing plan chunk {i+1}/{len(chunks)}...\n\n")
                            current_plan = await generate_writing_plan_aync(
                                session,
                                user_query,
                                chunk,
                                truncated=True,
                                note_from_previous=last_plan
                            )
                            last_plan = current_plan
                        
                        final_report_planning = last_plan
                        yield create_chunk(f"Writing Plan:\n{final_report_planning}\n\n")
                else:
                    # Context fits within model's capacity
                    final_report_planning = await generate_writing_plan_aync(session, user_query, aggregated_contexts)
                    if final_report_planning:
                        yield create_chunk(f"Writing Plan:\n{final_report_planning}\n\n")
            else:
                # No input limit, proceed normally
                final_report_planning = await generate_writing_plan_aync(session, user_query, aggregated_contexts)
                if final_report_planning:
                    yield create_chunk(f"Writing Plan:\n{final_report_planning}\n\n")
        else:
            final_report_planning = None

        yield create_chunk("\n</think>\n\n")
        
        # Check if we need to handle truncation for the final report
        if DEFAULT_MODEL_MAX_INPUT > 8000:
            # Estimate tokens in the aggregated contexts
            context_combined = "\n".join(aggregated_contexts)
            context_tokens = estimate_tokens(context_combined)
            
            # Account for prompt overhead
            available_tokens = DEFAULT_MODEL_MAX_INPUT - REPORT_PROMPT_OVERHEAD
            
            # If context is too large for the model's capacity
            if context_tokens > available_tokens:
                # If we can only fit less than 25% of the context, abort
                if context_tokens > 4 * available_tokens:
                    final_report = "default model has too small input token length that takes more than 4 iterations to finish generating final report, aborting..."
                else:
                    # Split context into manageable chunks using the helper function
                    chunks = split_contexts_into_chunks(aggregated_contexts, DEFAULT_MODEL_MAX_INPUT, REPORT_PROMPT_OVERHEAD)
                    
                    yield create_chunk(f"Context too large for default model, splitting into {len(chunks)} chunks for final report...\n\n")
                    
                    # Process each chunk and refine the report iteratively
                    last_report = None
                    for i, chunk in enumerate(chunks):
                        yield create_chunk(f"Processing final report chunk {i+1}/{len(chunks)}...\n\n")
                        current_report = await generate_final_report_async(
                            session,
                            system_instruction,
                            user_query,
                            final_report_planning,
                            chunk,
                            truncated=True,
                            note_from_previous=last_report
                        )
                        last_report = current_report
                    
                    final_report = last_report
                    final_report += "\n\nNote: This report was generated by iteratively processing the research results due to model context limitations."
            else:
                # Context fits within model's capacity
                final_report = await generate_final_report_async(session, system_instruction, user_query, final_report_planning, aggregated_contexts)
        else:
            # No input limit, proceed normally
            final_report = await generate_final_report_async(session, system_instruction, user_query, final_report_planning, aggregated_contexts)
        if not final_report or len(final_report) < 200:
            final_report = (final_report or "") + "\n\n" + "These are the writing prompt, please copy it and try again with anothor model\n\n---\n\n---\n\n" + f"User Query: {user_query}\n\nGathered Relevant Contexts:\n" + "\n\n".join(aggregated_contexts) + (f"\n\nWriting plan from a planning agent:\n{final_report_planning}" if final_report_planning else "") + "\n\nYou are an expert researcher and report writer. Based on the gathered contexts above and the original query, write a comprehensive, well-structured, and detailed report that addresses the query thoroughly.\n\n---\n\n---"
        yield create_chunk(final_report)
        yield "data: [DONE]\n\n"

async def async_main(system_instruction: str, user_query: str, max_iterations: int = 10, max_search_items: int = 4,
                    stream: bool = False, default_model: Optional[str] = None, reason_model: Optional[str] = None):
    """Main entry point that handles both streaming and non-streaming modes"""
    # Override config models if provided
    global DEFAULT_MODEL, REASON_MODEL
    temp_default_model = DEFAULT_MODEL
    temp_reason_model = REASON_MODEL
    
    if default_model:
        DEFAULT_MODEL = default_model
    if reason_model:
        REASON_MODEL = reason_model
        
    try:
        if stream:
            return stream_research(system_instruction, user_query, max_iterations, max_search_items)
        else:
            return await process_research(system_instruction, user_query, max_iterations, max_search_items)
    finally:
        # Restore original config values
        DEFAULT_MODEL = temp_default_model
        REASON_MODEL = temp_reason_model

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Check for model max input token limit
    if DEFAULT_MODEL_MAX_INPUT > 0 and DEFAULT_MODEL_MAX_INPUT < 8000 and REASON_MODEL_MAX_INPUT > 0 and REASON_MODEL_MAX_INPUT < 8000:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "default or reason model with too small input tokens, please use a model that can take more input tokens"}}
        )
        
    # Validate model
    if request.model != "deep_researcher":
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Only 'deep_researcher' model is supported"}}
        )

    # Check for Chrome session when not using Jina
    if not USE_JINA:
        try:
            async with async_playwright() as p:
                if USE_EMBED_BROWSER:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=['--no-sandbox', '--disable-setuid-sandbox'],
                        firefox_user_prefs={
                            "general.useragent.override": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
                        }
                    )
                else:
                    browser = await p.chromium.connect_over_cdp(f"{CHROME_HOST_IP}:{CHROME_PORT}")
                await browser.close()
        except Exception as e:
            error_msg = "Failed to initialize browser" if USE_EMBED_BROWSER else "Not using Jina but no Chrome session connected"
            return JSONResponse(
                status_code=400,
                content={"error": {"message": error_msg}}
            )

    # Get the last user message
    last_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
    if not last_message:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user message found"}}
        )
    
    # Get the last system message
    last_system_message = next((msg for msg in reversed(request.messages) if msg.role == "system"), None)
    if not last_system_message:
        last_system_message = Message(role="system", content="")

    if request.stream:
        # Get the async generator from async_main
        generator = await async_main(
            last_system_message.content,
            last_message.content,
            max_iterations=request.max_iterations,
            max_search_items=request.max_search_items,
            stream=True,
            default_model=request.default_model,
            reason_model=request.reason_model
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream"
        )
    else:
        final_report = await async_main(
            last_system_message.content,
            last_message.content,
            max_iterations=request.max_iterations,
            max_search_items=request.max_search_items,
            stream=False,
            default_model=request.default_model,
            reason_model=request.reason_model
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time()*1000)}",
            created=int(time.time()),
            model="deep_researcher",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=final_report),
                    finish_reason="stop"
                )
            ]
        )

@app.get("/v1/models")
async def list_models():
    """List the available models"""
    return ModelList(
        data=[
            ModelObject(
                id="deep_researcher",
                created=int(time.time()),
                owned_by="deep_researcher"
            )
        ]
    )

if __name__ == "__main__":
    FastAPI_HOST = "0.0.0.0"
    FastAPI_PORT = 8000
    print("Set the address shown below to a chat client as an OpenAI completion endpoint, or lauch the gradio demo interface in simple_webui folder.")
    print(f"Service will be running on http://{FastAPI_HOST}:{FastAPI_PORT}/v1")
    import uvicorn
    uvicorn.run(app, host=FastAPI_HOST, port=FastAPI_PORT)