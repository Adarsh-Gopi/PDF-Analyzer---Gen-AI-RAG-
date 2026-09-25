import os
import re
import sys
import urllib.parse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Reconfigure stdout/stderr for utf-8 if supported on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# --------------------------------------------------
# 1. Environment & API Key Setup
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# 2. Helpers for Local & Web Fallbacks
# --------------------------------------------------

def tokenize_text(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class LocalBM25Index:
    """Zero-API keyword search fallback across document chunks."""

    def __init__(self, chunks: list):
        self.chunks = chunks
        if chunks:
            tokenized_corpus = [tokenize_text(doc.page_content) for doc in chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 4) -> list:
        if not self.bm25 or not self.chunks:
            return []
        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]


def extract_website_content(url: str) -> Document:
    """Fetches and cleans readable text from any website URL."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg", "form", "button", "iframe"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    if not title:
        parsed = urllib.parse.urlparse(url)
        title = parsed.netloc + parsed.path

    main_node = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"id": "main-content"})
        or soup.find("div", {"class": "content"})
        or soup.body
    )
    raw_text = main_node.get_text(separator="\n", strip=True) if main_node else soup.get_text(separator="\n", strip=True)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    if not cleaned_text or len(cleaned_text) < 40:
        raise ValueError("Could not extract readable text content from the URL.")

    return Document(
        page_content=cleaned_text,
        metadata={"source": url, "title": title, "type": "url", "page": 1}
    )


def search_web_duckduckgo(query: str, max_results: int = 4) -> list[dict]:
    """Search DuckDuckGo without requiring an API key."""
    results = []
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=max_results)
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
    except Exception as e:
        print(f"[WARN] Web search notice: {e}")
    return results


def format_web_context(web_results: list[dict]) -> str:
    """Format web results into context text."""
    if not web_results:
        return ""
    lines = []
    for i, res in enumerate(web_results, start=1):
        lines.append(f"[{i}] {res['title']}\nURL: {res['url']}\nSnippet: {res['snippet']}")
    return "\n\n".join(lines)


# --------------------------------------------------
# 3. Safe Gemini LLM Calling
# --------------------------------------------------

def try_gemini_invoke(llm: ChatGoogleGenerativeAI, prompt: str) -> str | None:
    """Invokes Gemini LLM safely. Returns None if API fails or quota exceeded."""
    if not llm:
        return None
    try:
        response = llm.invoke(prompt)
        if isinstance(response.content, str):
            return response.content.strip()
        elif isinstance(response.content, list) and len(response.content) > 0 and isinstance(response.content[0], dict) and "text" in response.content[0]:
            return str(response.content[0]["text"]).strip()
        else:
            return str(response.content).strip()
    except Exception as e:
        print(f"\n[NOTICE] Gemini API unavailable ({type(e).__name__}: {e})")
        return None


# --------------------------------------------------
# 4. Main Application
# --------------------------------------------------

def main():
    print("=" * 60)
    print("   Resilient RAG Assistant (PDF + Web Link + Live Search)")
    print("=" * 60)

    # Initialize Gemini models if API key exists
    embedding_model = None
    llm = None

    if api_key:
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=api_key
            )
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.7-flash",
                google_api_key=api_key
            )
        except Exception as e:
            print(f"[WARN] Warning initializing Gemini client: {e}")
    else:
        print("[INFO] No GOOGLE_API_KEY detected in .env. Running in Offline/Fallback Mode.")

    chroma_dir = "./chroma_db"
    all_chunks = []
    vector_db = None
    bm25_index = None
    source_label = "Document"
    source_type = "pdf"

    # Step 1: Ingest or Load PDF or Web URL
    user_source_input = input("\nEnter path to PDF file OR Website URL (or press Enter to use existing / web): ").strip().strip('"').strip("'")

    if user_source_input:
        if user_source_input.startswith("http://") or user_source_input.startswith("https://") or ("." in user_source_input and "/" in user_source_input and not os.path.exists(user_source_input)):
            # Process Web URL
            source_type = "url"
            print(f"\n[1/3] Fetching and scraping website: {user_source_input}...")
            try:
                web_doc = extract_website_content(user_source_input)
                source_label = web_doc.metadata.get("title", user_source_input)
                print(f"      Title: {source_label}")
                print(f"      Extracted: {len(web_doc.page_content)} characters.")

                print("[2/3] Splitting into text chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                all_chunks = text_splitter.split_documents([web_doc])
                print(f"      Created {len(all_chunks)} text chunks.")
            except Exception as e:
                print(f"[ERROR] Failed to scrape website: {e}")
                sys.exit(1)
        else:
            # Process Local PDF
            source_type = "pdf"
            if not os.path.isfile(user_source_input):
                print(f"[ERROR] File not found at '{user_source_input}'")
                sys.exit(1)
            if not user_source_input.lower().endswith(".pdf"):
                print(f"[ERROR] '{user_source_input}' is not a PDF file.")
                sys.exit(1)

            print(f"\n[1/3] Loading PDF: {user_source_input}...")
            loader = PyPDFLoader(user_source_input)
            documents = loader.load()
            source_label = os.path.basename(user_source_input)
            print(f"      Loaded {len(documents)} page(s).")

            print("[2/3] Splitting into text chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_chunks = text_splitter.split_documents(documents)
            print(f"      Created {len(all_chunks)} text chunks.")

        # Build local BM25 index (always available offline)
        bm25_index = LocalBM25Index(all_chunks)

        # Build ChromaDB vector index if embedding API is available
        if embedding_model:
            print(f"[3/3] Generating embeddings and saving to '{chroma_dir}'...")
            try:
                vector_db = Chroma.from_documents(
                    documents=all_chunks,
                    embedding=embedding_model,
                    persist_directory=chroma_dir
                )
                print("      Vector database ready! [OK]")
            except Exception as e:
                print(f"      [WARN] Vector embedding failed ({e}). Using Local BM25 Search. [OK]")
        else:
            print("      [INFO] Skipped vector embeddings (API key absent/disabled). Local BM25 ready! [OK]")
    else:
        # Check for existing Chroma database
        if os.path.isdir(chroma_dir) and embedding_model:
            try:
                vector_db = Chroma(
                    persist_directory=chroma_dir,
                    embedding_function=embedding_model
                )
                print(f"Loaded existing vector database from '{chroma_dir}'.")
            except Exception as e:
                print(f"[INFO] Notice: Could not load vector database: {e}")

    # Create retriever if vector_db is ready
    retriever = None
    if vector_db:
        try:
            retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10}
            )
        except Exception:
            retriever = None

    # Step 2: Interactive Query Loop
    print("\n" + "=" * 60)
    print(" Ask questions below. Type 'exit' or 'q' to quit.")
    print("=" * 60)

    while True:
        question = input("\nAsk a question: ").strip()

        if not question:
            print("Please enter a non-empty question.")
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        retrieved_docs = []
        retrieval_source = "None"

        # --------------------------------------------------
        # Phase 1: Try Document / Web Link Retrieval (Vector or BM25)
        # --------------------------------------------------
        if retriever:
            try:
                retrieved_docs = retriever.invoke(question)
                if retrieved_docs:
                    retrieval_source = "ChromaDB (Vector Search)"
            except Exception as e:
                print(f"\n[NOTICE: Vector search failed: {e}. Switching to Local BM25...]")

        if not retrieved_docs and bm25_index:
            retrieved_docs = bm25_index.search(question, top_k=4)
            if retrieved_docs:
                retrieval_source = "Local BM25 (Keyword Search)"

        doc_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # --------------------------------------------------
        # Phase 2: Ingested Knowledge Base Q&A with Gemini
        # --------------------------------------------------
        answered_from_doc = False
        source_name = "Web Page" if source_type == "url" else "PDF Document"

        if doc_context:
            doc_prompt = f"""
You are a helpful AI assistant.

Answer the user's question clearly using the information provided in the context below from the active {source_name} ("{source_label}").
If the context does not contain enough information to answer the question, output EXACTLY:
"NOT_IN_SOURCE"

Context:
{doc_context}

User Question:
{question}
"""
            llm_response = try_gemini_invoke(llm, doc_prompt)

            if llm_response:
                if "NOT_IN_SOURCE" not in llm_response and "NOT_IN_PDF" not in llm_response:
                    print(f"\n[Source: {source_name} ({retrieval_source})]:")
                    print("-" * 60)
                    print(llm_response)
                    answered_from_doc = True
            else:
                # LLM API is down/quota exceeded -> Offline fallback: Show extracted passages
                print(f"\n[Source: {source_name} Extracted Passages (Offline Mode)]:")
                print("-" * 60)
                for i, doc in enumerate(retrieved_docs[:2], start=1):
                    page = doc.metadata.get("page", "N/A")
                    print(f"[Match {i} (Page {page})]:")
                    print(doc.page_content.strip()[:400] + "...\n")
                answered_from_doc = True

        # --------------------------------------------------
        # Phase 3: Web Search Fallback (if not found in document/url)
        # --------------------------------------------------
        if not answered_from_doc:
            print(f"\n[INFO: Not found in active {source_name.lower()}. Shifting to live Web Search (DuckDuckGo)...]")
            web_results = search_web_duckduckgo(question, max_results=4)

            if not web_results:
                print("\nAnswer:")
                print(f"Could not find relevant information in the active {source_name.lower()} or on the web.")
                continue

            web_context = format_web_context(web_results)

            web_prompt = f"""
You are an intelligent search assistant.
Answer the user's question accurately using the following live web search results.
Include source links or citations where relevant based on the URLs provided.

Web Search Results:
{web_context}

User Question:
{question}
"""
            web_llm_response = try_gemini_invoke(llm, web_prompt)

            if web_llm_response:
                print("\n[Source: Live Web Search + Gemini]:")
                print("-" * 60)
                print(web_llm_response)
            else:
                # Gemini is down -> Show top web search snippets directly
                print("\n[Source: DuckDuckGo Web Snippets (Offline Mode)]:")
                print("-" * 60)
                for res in web_results:
                    print(f"* {res['title']}")
                    print(f"  Link: {res['url']}")
                    print(f"  {res['snippet']}\n")


if __name__ == "__main__":
    main()