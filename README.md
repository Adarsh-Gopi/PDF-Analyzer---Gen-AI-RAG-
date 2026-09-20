# 📄 PDF Analyzer — Gen AI (RAG)

An intelligent document analysis platform powered by **Generative AI** and **Retrieval-Augmented Generation (RAG)** that enables real-time question answering over PDF documents, featuring hybrid vector retrieval, zero-API local keyword search, and automatic live web search fallback.

---

## 🌟 Key Features

- **🚀 Generative Q&A with Gemini**: Powered by Google's `gemini-3.7-flash` and `gemini-embedding-001`.
- **📄 Interactive Document Ingestion**: Upload any PDF via drag-and-drop or load existing documents with one click.
- **🛡️ Multi-Tiered Fallback Architecture**:
  1. **ChromaDB Vector Search**: Semantic search over text chunks.
  2. **Local BM25 Keyword Search**: Seamless zero-API offline fallback if embedding services are unavailable.
  3. **Live Web Search (DuckDuckGo)**: Automatically queries the web with citations when information is outside the PDF.
  4. **Offline Mode**: Gracefully displays extracted text passages or web snippets if the LLM API is unreachable.
- **🎨 ChatGPT-Inspired Interface**: Sleek dark-mode aesthetic built with Streamlit, including prompt suggestion cards and context expanders.

---

## 🛠️ Tech Stack

- **Framework**: LangChain, Streamlit
- **LLM & Embeddings**: Google Gemini (`langchain-google-genai`)
- **Vector Database**: ChromaDB (`langchain-chroma`)
- **Offline Search**: Rank-BM25 (`rank-bm25`)
- **Web Search**: DuckDuckGo (`ddgs`)
- **Document Processing**: PyPDF (`pypdf`, `langchain-community`)

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/Adarsh-Gopi/PDF-Analyzer---Gen-AI-RAG-.git
cd PDF-Analyzer---Gen-AI-RAG-
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)).*

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

*(Optional: You can also run the terminal version via `python Main.py`).*

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
