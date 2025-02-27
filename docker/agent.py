import dspy
from typing import Optional, AsyncGenerator, List
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

class ReportInstructionExtraction(dspy.Signature):
    """
    Extract components from a natural language instruction for generating a report.

    This extraction includes:
      - writing_style: instructions for the writing style,
      - searching_instruction: guidance on how to search,
      - local_doc_dir: local directory for documents,
      - online_url_include: list of online URLs to include,
      - online_url_avoid: list of online URLs to avoid,
      - main_query: the primary query to search for.
    """
    instruction: str = dspy.InputField(desc="A natural language instruction that includes all necessary details")
    writing_style: str = dspy.OutputField(desc="Writing style instruction (e.g., formal, informal, technical)")
    searching_instruction: str = dspy.OutputField(desc="Instructions on how to perform the search (e.g., search keywords, filters)")
    local_doc_dir: str = dspy.OutputField(desc="Local document directory path to be used for reference")
    online_url_include: List[str] = dspy.OutputField(desc="List of online URLs that must be included in the search")
    online_url_avoid: List[str] = dspy.OutputField(desc="List of online URLs to avoid in the search")
    main_query: str = dspy.OutputField(desc="The primary search query")

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

async def extract_report_instructions_async(session: aiohttp.ClientSession, system_message: str) -> Optional[ReportInstructionExtraction]:
    """
    Extract report components from a system message using dspy.
    Returns the extracted ReportInstructionExtraction object or None if extraction fails.
    """
    try:
        extractor = dspy.Predict(ReportInstructionExtraction)
        result = extractor(instruction=system_message)
        return result
    except Exception as e:
        print(f"Error extracting report instructions: {e}")
        return None