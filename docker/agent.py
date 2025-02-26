import dspy
from typing import Optional, AsyncGenerator
import aiohttp
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('docker/research.config')

USE_OLLAMA = config.getboolean('Settings', 'use_ollama')
OLLAMA_BASE_URL = config.get('LocalAI', 'ollama_base_url')
OPENAI_URL = config.get('API', 'openai_url')
OPENAI_COMPAT_API_KEY = config.get('API', 'openai_compat_api_key')
DEFAULT_MODEL = config.get('Settings', 'default_model')

# Initialize dspy based on configuration
if USE_OLLAMA:
    # Use Ollama for local inference
    lm = dspy.LM('ollama_chat/' + DEFAULT_MODEL, api_base=OLLAMA_BASE_URL, api_key='')
else:
    # Use OpenRouter/OpenAI compatible API
    lm = dspy.LM('openai/' + DEFAULT_MODEL, api_key=OPENAI_COMPAT_API_KEY, api_base=OPENAI_URL)

dspy.configure(lm=lm)

class WebpageAnalyzer(dspy.Signature):
    """Analyze webpage content for usefulness and extract relevant information."""
    
    user_query: str = dspy.InputField()
    search_query: str = dspy.InputField()
    page_url: str = dspy.InputField()
    page_content: str = dspy.InputField()
    
    is_useful: bool = dspy.OutputField()
    reason: str = dspy.OutputField(desc="Reasoning for usefulness decision")
    extracted_context: Optional[str] = dspy.OutputField(desc="Relevant extracted content if page is useful")

async def process_link_dspy(session: aiohttp.ClientSession, link: str, user_query: str, search_query: str,
                          page_text: str, create_chunk=None) -> AsyncGenerator[tuple[str, str, Optional[str]], None]:
    """
    Process a single link using dspy to determine usefulness and extract context in one pass.
    Returns a generator that yields tuples of (usefulness, reason, context).
    """
    try:
        # Initialize and run the analyzer
        analyzer = dspy.Predict(WebpageAnalyzer)
        result = analyzer(
            user_query=user_query,
            search_query=search_query,
            page_url=link,
            page_content=page_text[:20000]  # Limit content length similar to original
        )
        
        usefulness = "Yes" if result.is_useful else "No"
        reason = result.reason
        context = None

        if result.is_useful and result.extracted_context:
            context = result.extracted_context
            
        yield (usefulness, reason, context)
                
    except Exception as e:
        print(f"Error processing {link} with dspy: {e}")
        yield ("No", f"Error: {str(e)}", None)
    return

async def is_page_useful_dspy(session: aiohttp.ClientSession, user_query: str, page_text: str, page_url: str) -> tuple[str, str]:
    """
    Use dspy to determine if a page is useful and provide reasoning.
    Returns a tuple of (decision, reasoning).
    """
    analyzer = dspy.Predict(WebpageAnalyzer)
    result = analyzer(
        user_query=user_query,
        search_query="",  # Empty since we're only checking usefulness
        page_url=page_url,
        page_content=page_text[:20000]
    )
    return ("Yes" if result.is_useful else "No", result.reason)