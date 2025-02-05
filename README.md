# OpenDeepResearcher via Searxng 🧑‍🔬 (Ollama and PlayWright)

> [!TIP]
> - Use Searxng to reduce bias and improve privacy.
> - Report have citations📰!
> - Planning agent🤖 from reasoning models!!
> - Ollama support for local AI interface💻 for maximal privacy!
> - *[experimental]* Playwright🔗 support via Chrome/Chromium debug mode, parse webpage with local [reader-lm](https://huggingface.co/jinaai/reader-lm-1.5b), pdfs with [docling](https://github.com/DS4SD/docling)
> - Some refinement to reduce search query fail rate and token use.

## 🧑‍🏫 TL;DR

- 🌐 Online service for maximum speed?
   - **💸 Save money?** [open_deep_researcher.ipynb](open_deep_researcher.ipynb)
   - **💎 Absolute quality?** [open_deep_researcher_with_planning.ipynb](open_deep_researcher_with_planning.ipynb)
- 🧭 Balance Speed and privacy
   - **🏎️ Faster**: [local_open_deep_researcher.ipynb](local_open_deep_researcher.ipynb)
   - **🚗 Slower but higher quality**: [local_open_deep_researcher_with_planning.ipynb](local_open_deep_researcher_with_planning.ipynb)
- 💻 Want completely local?
   - **🚶 Slowest but everything happen on device** [local_open_deep_researcher_via_playwright.ipynb](local_open_deep_researcher_via_playwright.ipynb)
      - Change `BROWSE_LITE` to 1 to speed up parsing without using reader-lm and docling
      - **Planning?**:  [local_open_deep_researcher_with_planning_via_playwright.ipynb](local_open_deep_researcher_with_planning_via_playwright.ipynb) 
         - ⚠️ Note: I have not personally tested this, because my machine cannot even finish at this stage, please if you can run, share your experience with me.

## 📝 General INFO

This notebook implements an **AI researcher** that continuously searches for information based on a user query until the system is confident that it has gathered all the necessary details. It makes use of several services to do so:

- **SEARXNG**: To perform searches without bias and privately.
- **Content Parser**
   - **Jina**: To fetch and extract webpage content fast and reliable.
   - **Local solutions**:
      - [reader-lm](https://huggingface.co/jinaai/reader-lm-1.5b) by Jina via ollama for webpage parsing `ollama pull reader-lm:0.5b`
      - [docling](https://github.com/DS4SD/docling) for PDF OCRs
      - **WARNING: This is highly experimental, your instance may hang for long time due to poor compute power!! Use at your own risk.**
- **LLM Provider**: To interact with a LLM for generating search queries, evaluating page relevance, and extracting context.
   - **OpenRouter**: Paid, but fast
      - default searching and writing model: `anthropic/claude-3.5-haiku`
      - default reasoning and planning model: `deepseek/deepseek-r1-distill-qwen-32b`
   - *[NEW]* **Ollama**: Private, but a lot slower
      - default searching and writing model: `mistral-small` via `ollama pull mistral-small`
      - default reasoning and planning model: `deepseek-r1:14b` via `ollama pull deepseek-r1:14b`

## ☑️ Features

- **Iterative Research Loop:** The system refines its search queries iteratively until no further queries are required.
- **Asynchronous Processing:** Searches, webpage fetching, evaluation, and context extraction are performed concurrently to improve speed.
- **Duplicate Filtering:** Aggregates and deduplicates links within each round, ensuring that the same link isn’t processed twice.
- **LLM-Powered Decision Making:** Uses the LLM to generate new search queries, decide on page usefulness, extract relevant context, and produce a final comprehensive report, now with citations.
- **Plans made with Reasoning:** Before each iteration, a reasoning model will plan what to search, what to search more and how to write the final report to ensure robust planning strategy and good final quality. (only with [open_deep_researcher_with_planning.ipynb](open_deep_researcher_with_planning.ipynb) and [local_open_deep_researcher_with_planning.ipynb](local_open_deep_researcher_with_planning.ipynb))
- **Local models for maximum privacy**

## 🗺️ Planning via Reasoning

```mermaid
graph LR;
    A[User Query] --> B[Reasoning Model: Generate Initial Research Plan]
    B --> C[Search Agent: Conduct Search]
    C --> D[Reasoning Model: Evaluate Search Results]
    D -->|More Research Needed| C
    D -->|Search Complete or Max iterations| E[Reasoning Model: Generate Writing Plan]
    E --> F[Writing Agent: Write Final Report]
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style F fill:#ccf,stroke:#333,stroke-width:2px;
```

## 🧰 Requirements

- API access and keys for:
  - **OpenRouter API**
  - **Jina API**
- Local or public instance of **Searxng**
   - Get started locally
      - `docker run -d --name searxng --restart always -v $(pwd)/searxng:/etc/searxng:rw -p 4000:8080 docker.io/searxng/searxng:latest`
      - More at [https://docs.searxng.org/admin/installation-docker.html#installation-docker](https://docs.searxng.org/admin/installation-docker.html#installation-docker)
   - Public instance
     - [searx-instances](https://github.com/searx/searx-instances)
     - May have rate limits or usage logging
- Local Ollama: check out [ollama.com](https://ollama.com)
- A Chrome/Chromium browser running in debug mode `chromium --remote-debugging-port=9222 [--user-data-dir=/path/to/profile]`

## 💾 Setup

1. **Clone or Open the Notebook:**
   - Download the notebook file.

2. **Install `nest_asyncio`:**

   Run the first cell to set up `nest_asyncio`.

3. **Configure API Keys:**
   - Replace the placeholder values in the notebook for `OPENROUTER_API_KEY`, and `JINA_API_KEY` with your actual API keys.

4. **Set Base Searxng URL**
   - Replace the placeholder values in the notebook for `BASE_SEARXNG_URL` with the instance you like.

5. If using playwright?
   - Chrome/Chromium browser installed.
   - Launch debug mode with `chromium --remote-debugging-port=9222 [--user-data-dir=/path/to/profile]`, change the `CHROME_PORT` accordingly.
   - Full Mode(`BROWSE_LITE=0`) or Lite Mode(`BROWSE_LITE=1`)?
      - Full Mode parse html to markdown with reader-lm and OCR PDFs with docling, great quality, but can be extremely slow.
      - Lite Mode works like the reader view in browser, fast but may not get everything.
   - Modify other parameters in `Parsing settings` to better suit your machine's ability.

## 🧑‍🔬 Usage

1. **Run the Notebook Cells:**
   Execute all cells in order. The notebook will prompt you for:
   - A research query/topic.
   - An optional maximum number of iterations (default is 10).

2. **Follow the Research Process:**
   - **Initial Query & Search Generation:** The notebook uses the LLM to generate initial search queries.
   - **Asynchronous Searches & Extraction:** It performs SERPAPI searches for all queries concurrently, aggregates unique links, and processes each link in parallel to determine page usefulness and extract relevant context.
   - **Iterative Refinement:** After each round, the aggregated context is analyzed by the LLM to determine if further search queries are needed.
   - **Final Report:** Once the LLM indicates that no further research is needed (or the iteration limit is reached), a final report is generated based on all gathered context.

3. **View the Final Report:**
   The final comprehensive report will be printed in the output.

## ❓ How It Works

1. **Input & Query Generation:**  
   The user enters a research topic, and the LLM generates up to four distinct search queries.

2. **Concurrent Search & Processing:**  
   - **SEARXNG:** Each search query is sent to searxng concurrently.
   - **Deduplication:** All retrieved links are aggregated and deduplicated within the current iteration.
   - **Jina & LLM:** Each unique link is processed concurrently to fetch webpage content via Jina, evaluate its usefulness with the LLM, and extract relevant information if the page is deemed useful.

3. **Iterative Refinement:**  
   The system passes the aggregated context to the LLM to determine if further search queries are needed. New queries are generated if required; otherwise, the loop terminates.

4. **Final Report Generation:**  
   All gathered context is compiled and sent to the LLM to produce a final, comprehensive report addressing the original query. And the llm is instructed to properly cite the sources and summarize all the citations into a bibliography list.

## 🏁 Roadmap

- [x] Support Ollama
- [x] Support Playwright to bypass publisher limits with library proxy (*[TIP]* You can now just launch the browser with credentials logged in with your profile)
- [x] Use Playwright and Ollama's reader-lm to achieve 100% local service
- [ ] Refine process and reduce token usage
- [ ] Make into a pip package for easy install
- [ ] Integrate tool calling

## 💡 Troubleshooting

- **RuntimeError with asyncio:**  
  If you encounter an error like:
  ```
  RuntimeError: asyncio.run() cannot be called from a running event loop
  ```
  Ensure you have applied `nest_asyncio` as shown in the setup section.

- **API Issues:**  
  Verify that your API keys are correct and that you are not exceeding any rate limits.

- **Jina URL resolve issue**
   Wait and try again, this is usually due to high load.

---

Follow original author Matt on [X](https://x.com/mattshumer_) for updates on the base code.

Follow this repo for updates from my side for academic and local use.

OpenDeepResearcher and OpenDeepResearcher-via-searxng are released under the MIT License. See the LICENSE file for more details.
