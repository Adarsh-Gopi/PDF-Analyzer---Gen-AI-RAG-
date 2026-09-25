# 📄🌐 PDF & Web Analyzer — Gen AI (RAG)

An intelligent knowledge analysis platform powered by **Generative AI** and **Retrieval-Augmented Generation (RAG)** that enables real-time question answering over **PDF documents** and **Live Website Links / URLs**, featuring hybrid vector retrieval, zero-API local keyword search, and automatic live web search fallback.

---

## 🌟 Key Features

- **🚀 Generative Q&A with Gemini**: Powered by Google's `gemini-3.7-flash` and `gemini-embedding-001`.
- **📄 Multi-Source Ingestion**:
  - **PDF Documents**: Upload any PDF via drag-and-drop or load existing documents with one click.
  - **🌐 Any Website Link / URL**: Paste any web article or documentation URL for automated web scraping, cleaning, chunking, and indexing.
- **🛡️ Multi-Tiered Fallback Architecture**:
  1. **ChromaDB Vector Search**: Semantic search over text chunks.
  2. **Local BM25 Keyword Search**: Seamless zero-API offline fallback if embedding services are unavailable.
  3. **Live Web Search (DuckDuckGo)**: Automatically queries the web with citations when information is outside the document/webpage.
  4. **Offline Mode**: Gracefully displays extracted text passages or web snippets if the LLM API is unreachable.
- **🎨 ChatGPT-Inspired Interface**: Sleek dark-mode aesthetic built with Streamlit, including dynamic prompt suggestion cards, dedicated PDF & URL tabs, and context expanders with source badges.
- **⚡ One-Click Startup Launcher**: Includes `start.bat` / `run.bat` for instant single-click launch on Windows.

---

## 🛠️ Tech Stack

- **Framework**: LangChain, Streamlit
- **LLM & Embeddings**: Google Gemini (`langchain-google-genai`)
- **Vector Database**: ChromaDB (`langchain-chroma`)
- **Offline Search**: Rank-BM25 (`rank-bm25`)
- **Web Search**: DuckDuckGo (`ddgs`)
- **Document & Web Processing**: PyPDF (`pypdf`, `langchain-community`), BeautifulSoup4 (`beautifulsoup4`), Requests (`requests`)

---

## 🚀 Quick Start (Local Setup)

### 1. One-Click Launch (Windows)
Double-click **`start.bat`** or **`run.bat`** in the project folder to automatically activate your environment and launch the web app in your browser!

---

### 2. Manual Terminal Launch

#### Step A: Configure API Key
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)).*

#### Step B: Activate Environment & Install Dependencies
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Step C: Launch the Web App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

*(Optional: You can also run the terminal CLI version via `python Main.py` or by double-clicking `start_cli.bat`).*

---

## ☁️ Streamlit Community Cloud Deployment

1. Fork or push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New app**.
3. Select this repository, branch `main`, and main file path `app.py`.
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key"
   ```
5. Click **Deploy**!
