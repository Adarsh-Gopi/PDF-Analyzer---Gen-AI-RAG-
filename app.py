import os
import re
import base64
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# --------------------------------------------------
# Page Configuration & Styling
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Analyzer - AI Document Intelligence",
    page_icon="logo.jpg" if os.path.exists("logo.jpg") else "📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to load logo as base64
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_b64 = get_base64_image("logo.jpg")
logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" class="app-logo" alt="PDF Analyzer Logo">' if logo_b64 else '📄'

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* ChatGPT Dark Aesthetic */
    .stApp {
        background-color: #212121;
        color: #ECECF1;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2f2f2f;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #2f2f2f;
    }

    /* Hero Centered Header */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 4vh;
        margin-bottom: 3vh;
    }

    .app-logo {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.4);
        margin-bottom: 16px;
        object-fit: cover;
    }

    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ECECF1;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #9CA3AF;
        max-width: 580px;
        line-height: 1.5;
    }

    /* Source Badges */
    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
    }

    .badge-pdf-vector {
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .badge-pdf-bm25 {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }

    .badge-web-search {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .badge-offline-mode {
        background: rgba(156, 163, 175, 0.15);
        color: #d1d5db;
        border: 1px solid rgba(209, 213, 219, 0.3);
    }

    /* Chat Messages */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 1rem 0.5rem;
    }

    div[data-testid="stChatMessage"][data-test-role="assistant"] {
        background-color: #212121 !important;
    }

    div[data-testid="stChatMessage"][data-test-role="user"] {
        background-color: #2A2A2A !important;
        border-radius: 14px;
        margin-bottom: 8px;
    }

    /* Bottom Input Bar Pill */
    div[data-testid="stChatInput"] {
        border-radius: 24px !important;
        background-color: #2F2F2F !important;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Environment & Secure API Key Resolution
# --------------------------------------------------

load_dotenv(override=True)

# Securely check for server-side key (from .env or Streamlit Secrets) without exposing it
server_key = os.getenv("GOOGLE_API_KEY", "")
try:
    if not server_key and "GOOGLE_API_KEY" in st.secrets:
        server_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass


# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "active_doc_name" not in st.session_state:
    st.session_state.active_doc_name = None

if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = None

if "user_custom_key" not in st.session_state:
    st.session_state.user_custom_key = ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-3.6-flash"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2

if "top_k_chunks" not in st.session_state:
    st.session_state.top_k_chunks = 4

if "enable_web_fallback" not in st.session_state:
    st.session_state.enable_web_fallback = True


# Resolve the active API key (Custom override -> Server key)
active_api_key = st.session_state.user_custom_key.strip() if st.session_state.user_custom_key.strip() else server_key


# --------------------------------------------------
# Helpers for Ingestion, BM25, and Web Search
# --------------------------------------------------

def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class LocalBM25Index:
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


def search_web_duckduckgo(query: str, max_results: int = 4) -> list[dict]:
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
        st.warning(f"Web search note: {e}")
    return results


def format_web_context(web_results: list[dict]) -> str:
    if not web_results:
        return ""
    lines = []
    for i, res in enumerate(web_results, start=1):
        lines.append(f"[{i}] {res['title']}\nURL: {res['url']}\nSnippet: {res['snippet']}")
    return "\n\n".join(lines)


def try_gemini_invoke(llm: ChatGoogleGenerativeAI, prompt: str) -> str | None:
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
        st.warning(f"⚠️ Gemini API Notice: {e}")
        return None


def process_pdf_file(file_path: str, api_key: str, doc_label: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    st.session_state.chunks = chunks
    st.session_state.bm25_index = LocalBM25Index(chunks)
    st.session_state.active_doc_name = doc_label

    if api_key:
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=api_key
            )
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )
            st.session_state.vector_db = vector_db
        except Exception:
            st.session_state.vector_db = None
    else:
        st.session_state.vector_db = None

    return len(documents), len(chunks)


# --------------------------------------------------
# Auto-Load PDF on First Startup
# --------------------------------------------------

if st.session_state.active_doc_name is None:
    if os.path.exists("paper.pdf"):
        try:
            process_pdf_file("paper.pdf", active_api_key, "paper.pdf")
        except Exception:
            pass


# --------------------------------------------------
# Sidebar: Knowledge Base & Settings Modal
# --------------------------------------------------

with st.sidebar:
    # Branding
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        {logo_html}
        <div>
            <div style="font-weight: 700; font-size: 1.15rem; color: #ECECF1; letter-spacing: -0.01em;">PDF Analyzer</div>
            <div style="font-size: 0.78rem; color: #06b6d4; font-weight: 500;">AI Document Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Document Management
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #9CA3AF; margin-bottom: 8px;'>DOCUMENT KNOWLEDGE BASE</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("⚡ Index Uploaded PDF", use_container_width=True):
            with st.spinner("Processing & indexing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                pages, chunks = process_pdf_file(tmp_path, active_api_key, uploaded_file.name)
                os.unlink(tmp_path)
                st.toast(f"✅ Indexed {pages} pages ({chunks} chunks)!", icon="📄")

    if os.path.exists("paper.pdf"):
        if st.button("📂 Reload 'paper.pdf'", use_container_width=True):
            with st.spinner("Indexing paper.pdf..."):
                pages, chunks = process_pdf_file("paper.pdf", active_api_key, "paper.pdf")
                st.toast(f"✅ Loaded paper.pdf ({chunks} chunks)!", icon="📄")

    if st.session_state.active_doc_name:
        st.success(f"📌 **Active PDF**: `{st.session_state.active_doc_name}` ({len(st.session_state.chunks)} chunks)")

    st.markdown("---")

    # Settings Popover (Hidden & Secure)
    with st.popover("⚙️ Settings & Configuration", use_container_width=True):
        st.markdown("### ⚙️ System Settings")

        tab_api, tab_model, tab_rag = st.tabs(["🔑 API & Security", "🧠 Model Settings", "🔍 RAG & Search"])

        with tab_api:
            st.markdown("#### API Authentication")
            
            # Show API status without revealing the secret
            if active_api_key:
                st.success("🔒 **Status**: API Key Active & Verified", icon="✅")
            else:
                st.warning("⚠️ **Status**: No API Key Detected. Running in Offline/Fallback Mode.")

            custom_key_input = st.text_input(
                "Override API Key",
                type="password",
                value=st.session_state.user_custom_key,
                placeholder="Enter custom Gemini key (or leave empty to use server default)...",
                help="Your key is never exposed or logged. It overrides the default environment key for this session."
            )
            if custom_key_input != st.session_state.user_custom_key:
                st.session_state.user_custom_key = custom_key_input
                st.rerun()

        with tab_model:
            st.markdown("#### LLM Preferences")
            model_options = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.7-flash"]
            default_index = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            st.session_state.selected_model = st.selectbox(
                "Gemini Model",
                options=model_options,
                index=default_index
            )
            st.session_state.temperature = st.slider(
                "Creativity / Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.05,
                help="Lower values produce more deterministic, factual answers."
            )

        with tab_rag:
            st.markdown("#### Retrieval & Fallback Controls")
            st.session_state.top_k_chunks = st.slider(
                "Top-K Retrieved Context Chunks",
                min_value=2,
                max_value=8,
                value=st.session_state.top_k_chunks,
                step=1
            )
            st.session_state.enable_web_fallback = st.toggle(
                "Enable DuckDuckGo Live Web Search Fallback",
                value=st.session_state.enable_web_fallback,
                help="When document context lacks the answer, automatically queries the live web."
            )

    # Pipeline Status Indicators
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #9CA3AF; margin-top: 15px; margin-bottom: 8px;'>PIPELINE STATUS</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.82rem; line-height: 1.8; color: #D1D5DB;">
        • <span style="color: #38bdf8;">ChromaDB Vector Search</span>: {'Active' if st.session_state.vector_db else 'Standby'}<br/>
        • <span style="color: #c084fc;">Local BM25 Search</span>: {'Active' if st.session_state.bm25_index else 'Standby'}<br/>
        • <span style="color: #fbbf24;">DuckDuckGo Web Search</span>: {'Active' if st.session_state.enable_web_fallback else 'Disabled'}<br/>
        • <span style="color: #34d399;">Gemini Model</span>: `{st.session_state.selected_model}`
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Main Chat Area
# --------------------------------------------------

llm = None
if active_api_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.selected_model,
            google_api_key=active_api_key,
            temperature=st.session_state.temperature
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")

# If chat is empty, display Hero Screen
if not st.session_state.messages:
    st.markdown(f"""
    <div class="hero-container">
        {logo_html}
        <div class="hero-title">What can I help you analyze today?</div>
        <div class="hero-subtitle">Upload your PDF document to query its contents, or ask anything to search the live web automatically.</div>
    </div>
    """, unsafe_allow_html=True)

    # Prompt suggestions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📄 **Summarize Document**\n\nGive me a comprehensive executive summary of this document", use_container_width=True):
            st.session_state.prefill_query = "Give me a comprehensive executive summary of this document."
            st.rerun()

        if st.button("⚖️ **Key Findings & Verdict**\n\nExplain main methodology, results, and conclusions", use_container_width=True):
            st.session_state.prefill_query = "What are the main methodology, key findings, and conclusions of this paper?"
            st.rerun()

    with col_b:
        if st.button("🌐 **Search Live Web**\n\nWhat are the latest AI agent trends in 2026?", use_container_width=True):
            st.session_state.prefill_query = "What are the latest advancements in AI agentic coding in 2026?"
            st.rerun()

        if st.button("🔍 **Analyze Machine Learning Model**\n\nWhat datasets, models, and metrics were evaluated?", use_container_width=True):
            st.session_state.prefill_query = "What machine learning models, algorithms, and datasets were used?"
            st.rerun()


# Display existing messages
for msg in st.session_state.messages:
    avatar = "logo.jpg" if (msg["role"] == "assistant" and os.path.exists("logo.jpg")) else None
    with st.chat_message(msg["role"], avatar=avatar):
        if "badge_html" in msg and msg["badge_html"]:
            st.markdown(msg["badge_html"], unsafe_allow_html=True)
        st.markdown(msg["content"])
        if "context_details" in msg and msg["context_details"]:
            with st.expander("🔍 View Context / Source Citations"):
                st.markdown(msg["context_details"])


# Determine query from chat input or button prefill
chat_prompt = st.chat_input("Message PDF Analyzer...")
active_query = None

if st.session_state.prefill_query:
    active_query = st.session_state.prefill_query
    st.session_state.prefill_query = None
elif chat_prompt:
    active_query = chat_prompt


# Process query
if active_query:
    # 1. User Message
    st.chat_message("user").markdown(active_query)
    st.session_state.messages.append({"role": "user", "content": active_query})

    # 2. Assistant Response
    avatar = "logo.jpg" if os.path.exists("logo.jpg") else None
    with st.chat_message("assistant", avatar=avatar):
        with st.spinner("Thinking & analyzing knowledge base..."):
            retrieved_docs = []
            retrieval_source = "None"
            badge_html = ""
            context_details = ""
            final_answer = ""

            # Phase 1: Try Document Retrieval (ChromaDB -> BM25 fallback)
            if st.session_state.vector_db:
                try:
                    retriever = st.session_state.vector_db.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": st.session_state.top_k_chunks, "fetch_k": st.session_state.top_k_chunks * 2}
                    )
                    retrieved_docs = retriever.invoke(active_query)
                    if retrieved_docs:
                        retrieval_source = "ChromaDB (Vector Search)"
                except Exception:
                    pass

            if not retrieved_docs and st.session_state.bm25_index:
                retrieved_docs = st.session_state.bm25_index.search(active_query, top_k=st.session_state.top_k_chunks)
                if retrieved_docs:
                    retrieval_source = "Local BM25 (Keyword Search)"

            # If question is a general document query like "what is this document/paper/pdf about?", include top chunks if no specific match
            if not retrieved_docs and st.session_state.chunks:
                retrieved_docs = st.session_state.chunks[:st.session_state.top_k_chunks]
                retrieval_source = "Document Beginning Chunks"

            pdf_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            answered_from_pdf = False

            # Phase 2: PDF Question Answering with Gemini
            if pdf_context:
                pdf_prompt = f"""
You are an expert AI document assistant.
Answer the user's question clearly and accurately using the context provided below from the uploaded document.
If the context does NOT contain enough information to answer the question, output EXACTLY:
"NOT_IN_PDF"

Context:
{pdf_context}

User Question:
{active_query}
"""
                llm_response = try_gemini_invoke(llm, pdf_prompt)

                if llm_response and "NOT_IN_PDF" not in llm_response:
                    if "Vector" in retrieval_source:
                        badge_html = '<span class="source-badge badge-pdf-vector">📄 Source: PDF (ChromaDB Vector)</span>'
                    else:
                        badge_html = '<span class="source-badge badge-pdf-bm25">📄 Source: PDF (Local Search)</span>'
                    final_answer = llm_response
                    answered_from_pdf = True

                    context_lines = []
                    for i, doc in enumerate(retrieved_docs, 1):
                        p = doc.metadata.get("page", "N/A")
                        context_lines.append(f"**Chunk {i} (Page {p})**:\n> {doc.page_content.strip()}")
                    context_details = "\n\n".join(context_lines)

                elif not llm_response and retrieved_docs:
                    # Fallback when Gemini API encounters rate-limit / offline / network error
                    badge_html = '<span class="source-badge badge-offline-mode">⚡ Source: PDF Extracted Passages (Offline Fallback)</span>'
                    passages = []
                    for i, doc in enumerate(retrieved_docs[:3], 1):
                        p = doc.metadata.get("page", "N/A")
                        passages.append(f"**Passage {i} (Page {p})**:\n{doc.page_content.strip()}")
                    final_answer = "⚠️ *Gemini API temporarily reached its request limit or is offline. Here are the most relevant extracted passages directly from your document:*\n\n" + "\n\n---\n\n".join(passages)
                    answered_from_pdf = True

            # Phase 3: Web Search Fallback (if not in PDF and enabled)
            if not answered_from_pdf and st.session_state.enable_web_fallback:
                web_results = search_web_duckduckgo(active_query, max_results=4)

                if web_results:
                    web_context = format_web_context(web_results)
                    web_prompt = f"""
You are an expert AI assistant.
Answer the user's question accurately using the following live web search results.
Synthesize the information in a clean, professional markdown answer.
Include source links or citations where relevant based on the provided URLs.

Web Search Results:
{web_context}

User Question:
{active_query}
"""
                    web_llm_response = try_gemini_invoke(llm, web_prompt)

                    if web_llm_response:
                        badge_html = '<span class="source-badge badge-web-search">🌐 Source: Live Web Search + Gemini</span>'
                        final_answer = web_llm_response
                    else:
                        badge_html = '<span class="source-badge badge-offline-mode">🌐 Source: DuckDuckGo Web Snippets (Offline Mode)</span>'
                        snippets = []
                        for res in web_results:
                            snippets.append(f"**[{res['title']}]({res['url']})**\n{res['snippet']}")
                        final_answer = "\n\n".join(snippets)

                    src_lines = []
                    for r in web_results:
                        src_lines.append(f"- [{r['title']}]({r['url']})\n  > {r['snippet']}")
                    context_details = "\n".join(src_lines)
                else:
                    badge_html = '<span class="source-badge badge-offline-mode">❌ Not Found</span>'
                    final_answer = "Could not find relevant information in the PDF or on the web."
            elif not answered_from_pdf and not st.session_state.enable_web_fallback:
                badge_html = '<span class="source-badge badge-offline-mode">📄 PDF Only</span>'
                final_answer = "I could not find the answer in the uploaded PDF document."

            # Render response
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(final_answer)

            if context_details:
                with st.expander("🔍 View Context / Source Citations"):
                    st.markdown(context_details)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "badge_html": badge_html,
                "content": final_answer,
                "context_details": context_details
            })
            st.rerun()
