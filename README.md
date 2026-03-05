# RAG OCR — Handwritten PDF Notes Q&A System

A local, offline document question-answering system that extracts text from scanned/handwritten PDFs using OCR and answers questions using a Retrieval-Augmented Generation (RAG) pipeline powered by local LLMs via Ollama.

---

## How It Works

```
PDF Upload → PDF-to-Images (Poppler) → OCR (Tesseract) → Text Chunking → Vector Embeddings (BGE) → ChromaDB → LLM Answer (Ollama)
```

1. **PDF → Images** — The uploaded PDF is converted to high-resolution PNG images using `pdf2image` + Poppler.
2. **OCR** — Each page image is pre-processed with OpenCV (grayscale) and passed to Tesseract OCR to extract text.
3. **Chunking** — The raw OCR text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
4. **Embedding + Vector DB** — Chunks are embedded using the `BAAI/bge-small-en` HuggingFace model and stored in a local ChromaDB instance.
5. **RAG Q&A** — On each question, the top-5 most relevant chunks are retrieved and passed as context to a local LLM running via Ollama.

---

## Prerequisites

### System Dependencies

These must be installed manually — they are **not** Python packages.

#### 1. Tesseract OCR

- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Make sure to install the **English language pack** during setup.

#### 2. Poppler (PDF rendering)

- Download from: https://github.com/oschwartz10612/poppler-windows/releases
- Extract the archive anywhere (e.g. `C:\poppler\`)
- You will need the path to the `Library\bin` folder inside it.

#### 3. Ollama (Local LLM runtime)

- Download from: https://ollama.com/download
- After installing, pull the model used by the app:
  ```bash
  ollama pull deepseek-v3.1:671b-cloud
  ```
- Make sure the Ollama server is running before starting the app:
  ```bash
  ollama serve
  ```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG_OCR
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -e .
```

### 4. Configure environment variables

Create a `.env` file in the project root by copying the example:

```bash
copy .env.example .env
```

Then edit `.env` and set the paths for your machine:

```env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\poppler\Library\bin
```

---

## Project Structure

```
RAG_OCR/
├── main.py            # Basic version — full-text LLM Q&A (no RAG)
├── main2.py           # Full RAG pipeline with ChromaDB (recommended)
├── pyproject.toml     # Project metadata and dependencies
├── .env               # Local config (paths) — not committed to git
├── .env.example       # Template for .env
├── data/              # Temporary folder — page images extracted from PDF
└── vector_db/         # Persistent ChromaDB storage
```

> **main.py vs main2.py** — `main.py` is a simpler prototype that stuffs the entire OCR output into the LLM prompt. It will fail on long documents that exceed the model's context window. `main2.py` uses proper RAG chunking and retrieval — **use main2.py**.

---

## Usage

```bash
streamlit run main2.py
```

Then open your browser at `http://localhost:8501`.

1. Upload a PDF (scanned document, handwritten notes, etc.)
2. Wait for the pipeline to complete (image conversion → OCR → vector DB build)
3. Type a question about the document in the text box
4. The app retrieves the most relevant passages and generates an answer

---

## Configuration

All machine-specific paths are configured via environment variables in `.env`:

| Variable         | Description                            | Example                                        |
| ---------------- | -------------------------------------- | ---------------------------------------------- |
| `TESSERACT_PATH` | Full path to the Tesseract executable  | `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `POPPLER_PATH`   | Path to Poppler's `Library\bin` folder | `C:\poppler\Library\bin`                       |

---

## Models Used

| Component       | Model                      | Source                                 |
| --------------- | -------------------------- | -------------------------------------- |
| Text Embeddings | `BAAI/bge-small-en`        | HuggingFace (downloaded automatically) |
| LLM (main2.py)  | `deepseek-v3.1:671b-cloud` | Ollama                                 |
| LLM (main.py)   | `qwen3-vl:235b-cloud`      | Ollama                                 |

To use a different Ollama model, change the `model=` argument in `ChatOllama(...)`.

---

## RAG Parameters

These can be tuned in `main2.py` to improve retrieval quality:

| Parameter       | Current Value | Description                          |
| --------------- | ------------- | ------------------------------------ |
| `chunk_size`    | `800`         | Characters per text chunk            |
| `chunk_overlap` | `100`         | Overlap between consecutive chunks   |
| `k` (retrieval) | `5`           | Number of chunks retrieved per query |
| `dpi`           | `300`         | Image resolution for PDF conversion  |

---

## Known Limitations

- **OCR quality** — Accuracy depends heavily on handwriting legibility and scan quality. Poor scans will produce noisy extracted text.
- **English only** — Tesseract is configured for English text extraction. Change the `lang` parameter in `pytesseract.image_to_string()` for other languages.
- **Local hardware** — Large Ollama models require significant RAM/VRAM. Smaller models (e.g. `llama3.2:3b`) can be substituted if resources are limited.
- **PDF type** — This pipeline is designed for scanned/image-based PDFs. For text-layer PDFs, direct text extraction (e.g. PyMuPDF) would be faster and more accurate than OCR.

---

## Tech Stack

| Library                 | Purpose                           |
| ----------------------- | --------------------------------- |
| `streamlit`             | Web UI                            |
| `pdf2image`             | PDF → PNG conversion              |
| `pytesseract`           | OCR engine wrapper                |
| `opencv-python`         | Image pre-processing              |
| `langchain`             | RAG orchestration                 |
| `langchain-chroma`      | ChromaDB vector store integration |
| `langchain-huggingface` | HuggingFace embeddings            |
| `langchain-ollama`      | Local LLM via Ollama              |
| `sentence-transformers` | Embedding model backend           |

---

## License

This project is for personal/educational use.
