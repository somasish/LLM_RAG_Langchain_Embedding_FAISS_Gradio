# LLM_RAG_Langchain_Embedding_FAISS_Gradio: Retrieval-Augmented Generation (RAG) with Gradio, LangChain, and Groq LLMs

This repository provides a modular, class-based Python project for Retrieval-Augmented Generation (RAG) using Gradio, LangChain, FAISS, HuggingFace embeddings, and Groq/OpenAI-compatible LLMs. It is designed for easy extension, best practices, and onboarding new LLM developers.

## Features
- **File & URL Support:** Upload files (PDF, DOCX, CSV, TXT, etc.) or paste a public URL for processing.
- **Model Selection:** Choose from multiple LLMs and embedding models via dropdowns in the UI. All model tokens and names are read from `config.yaml`.
- **Persistent Vectorstore:** Automatically saves and loads FAISS vectorstores for each unique file or URL, enabling fast reloading and reuse.
- **Chunking & Embedding:** Documents are split into overlapping chunks and embedded using the selected HuggingFace model for efficient retrieval.
- **Question Answering:** Ask questions about the uploaded or linked content using the selected LLM.
- **Gradio UI:** Simple, interactive web interface for document loading, model selection, and Q&A.
- **Class-based, Pythonic Design:** All core logic is encapsulated in the `RAGGradioApp` class for maintainability and extensibility.

## Project Structure
```
├── groq_mistral/
│   ├── rag_gradio_llm.py         # Main class-based RAG Gradio app (entry point)
│   ├── config.yaml               # Model/API configuration
│   ├── chunks/                   # Persistent vectorstore storage (auto-generated)
│   ├── requirements.txt          # Python dependencies
│   ├── readme.md                 # This file
│   └── ...                       # Other scripts and legacy files
```

## Setup
1. **Clone the repo:**
   ```bash
   git clone <repo-url>
   cd LLM_RAG_Langchain_Embedding_FAISS_Gradio
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare `config.yaml`:**
   Example:
   ```yaml
   llm_models:
     - name: "llama3-8b-8192"
       api_key: "YOUR_GROQ_API_KEY"
       api_base: "https://api.groq.com/openai/v1"
   embedding_models:
     - "all-MiniLM-L6-v2"
   ```
   - You can add as many LLMs and embedding models as you like.
   - For backward compatibility, the following keys are also supported:
     ```yaml
     groq_api_key: "YOUR_GROQ_API_KEY"
     groq_model: "llama3-8b-8192"
     embedding_model: "all-MiniLM-L6-v2"
     ```
4. **Run the app:**
   ```bash
   python rag_gradio_llm.py
   ```

## Usage
- **Upload a file** or **paste a URL** in the Gradio interface.
- **Select your LLM and embedding model** from the dropdowns.
- Click **Load Document(s)** to process and index the content.
- Enter your question in the textbox and click **Get Answer**.

## For New LLM Coders
- The main logic is in `rag_gradio_llm.py` inside the `RAGGradioApp` class.
- All configuration is handled via `config.yaml`.
- The code is modular, with clear docstrings and comments for each method.
- You can extend the app by subclassing `RAGGradioApp` or adding new methods.
- Debug print statements are included to help trace LLM calls and QA chain creation.
- See the code comments for guidance on each step of the RAG pipeline.

## Contributing
- Fork the repo and create a feature branch.
- Follow PEP8 and use class-based, modular design.
- Add docstrings and comments for clarity.
- Submit a pull request with a clear description of your changes.

## License
MIT 

## Screenshot
![app_screenshot](https://github.com/user-attachments/assets/fffb6322-dac9-4cb2-96ce-6b5f3d205864)
