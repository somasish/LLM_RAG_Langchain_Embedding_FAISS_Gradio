import os
import yaml
import hashlib
import pickle
from typing import List, Optional

import gradio as gr
from bs4 import BeautifulSoup
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

class RAGGradioApp:
    """
    A class-based Gradio app for Retrieval-Augmented Generation (RAG) with LLM and embedding model selection.
    Handles file/URL upload, chunking, embedding, vectorstore management, and QA via LangChain.
    """
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the app, load config, and set up model lists and chunk directory."""
        self.config = self.load_config(config_path)
        self.llm_models = self.config.get("llm_models", [
            {"name": self.config.get("groq_model", "llama-3-8b-instant"), "api_key": self.config.get("groq_api_key", ""), "api_base": "https://api.groq.com/openai/v1"}
        ])
        self.embedding_models = self.config.get("embedding_models", [self.config.get("embedding_model", "all-MiniLM-L6-v2")])
        self.chunks_dir = "chunks"
        os.makedirs(self.chunks_dir, exist_ok=True)
        # App state
        self.state = {"vectorstore": None, "qa_chain": None, "doc_summary": "", "llm": None, "embedding": None}

    @staticmethod
    def load_config(path: str) -> dict:
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get_url_hash(self, url: str) -> str:
        """Returns a unique hash for a URL to use as a filename."""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()

    def get_chunk_filename(self, source: str, is_url: bool) -> str:
        """Returns the local vectorstore filename for a file or a URL source."""
        if is_url:
            hashname = self.get_url_hash(source)
            return os.path.join(self.chunks_dir, f"url_{hashname}")
        else:
            basename = os.path.basename(source)
            return os.path.join(self.chunks_dir, f"file_{basename}")

    def extract_text_from_file(self, file_obj) -> str:
        """Extract text from uploaded file. Handles PDF, DOCX, CSV, TXT, and generic text."""
        import mimetypes
        file_name = getattr(file_obj, "name", "uploaded_file")
        ext = os.path.splitext(file_name)[1].lower()
        try:
            content = None
            if hasattr(file_obj, "read"):
                try:
                    raw = file_obj.read()
                    if isinstance(raw, bytes):
                        content = raw.decode("utf-8", errors="ignore")
                    else:
                        content = raw
                except Exception:
                    content = ""
            else:
                content = str(file_obj)
            # Special handling for common file types
            if ext == ".pdf":
                from pypdf import PdfReader
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                reader = PdfReader(file_obj)
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif ext == ".docx":
                import docx2txt
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                content = docx2txt.process(file_obj)
            elif ext == ".csv":
                import pandas as pd
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj)
                content = df.to_string()
            # If .txt, keep as content (already decoded above)
            return content
        except Exception as e:
            return f"Could not extract text from file: {str(e)}"

    def extract_text_from_url(self, url: str) -> str:
        """Fetch and extract main text from a public URL."""
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            return f"Could not extract text from URL: {str(e)}"

    def save_faiss_vectorstore(self, vectorstore: FAISS, base_path: str):
        """Save FAISS vectorstore with pickle metadata to specified basename."""
        vectorstore.save_local(base_path)
        with open(base_path + '.meta.pkl', 'wb') as f:
            pickle.dump(vectorstore.docstore._dict, f)

    def load_faiss_vectorstore(self, base_path: str, embedding_model):
        """Load FAISS vectorstore with pickle metadata from specified basename."""
        if not (os.path.exists(base_path + ".faiss") and os.path.exists(base_path + ".pkl")):
            return None
        vs = FAISS.load_local(base_path, embedding_model, allow_dangerous_deserialization=True)
        meta_path = base_path + '.meta.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                vs.docstore._dict = pickle.load(f)
        return vs

    def build_and_save_vectorstore(self, docs: List[Document], base_path: str, embedding_model):
        """Chunks the docs, embeds/chunks, saves index."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        vectorstore = FAISS.from_documents(chunks, embedder)
        self.save_faiss_vectorstore(vectorstore, base_path)
        return vectorstore

    def get_llm(self, llm_name: str, api_key: str, api_base: Optional[str] = None):
        """Instantiate the LLM (ChatOpenAI-compatible) with user-selected model and key."""
        if api_base is None and "llama" in llm_name:
            api_base = "https://api.groq.com/openai/v1"
        print(f"[LLM CALL] Model: {llm_name}, API Base: {api_base}, API Key (first 8): {api_key[:8]}...")
        kwargs = {
            "model": llm_name,
            "api_key": api_key,
            "temperature": 0.2,
            "streaming": False,
        }
        if api_base:
            kwargs["openai_api_base"] = api_base
        return ChatOpenAI(**kwargs)

    def make_qa_chain(self, vectorstore, llm_name, api_key, api_base=None):
        """Create a RetrievalQA chain using the vectorstore retriever and selected LLM."""
        print(f"[QA CHAIN] Creating QA chain with model: {llm_name}")
        retriever = vectorstore.as_retriever()
        llm = self.get_llm(llm_name, api_key, api_base)
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

    def load_docs(self, file, url, llm_name, emb_model):
        """
        Load and process a document or URL, build vectorstore, and QA chain.
        This is the main handler for the Gradio UI.
        """
        docs = []
        summary_msgs = []
        used_path = None
        # Find LLM config from the list based on user selection
        llm_cfg = next((m for m in self.llm_models if m["name"] == llm_name), self.llm_models[0])
        api_key = llm_cfg.get("api_key", "")
        api_base = llm_cfg.get("api_base", None)
        # Priority: file upload (only one file at a time)
        if file is not None:
            if hasattr(file, "seek"):
                file.seek(0)
            content = self.extract_text_from_file(file)
            docs.append(Document(page_content=content, metadata={"source": file.name}))
            summary_msgs.append(f"- File **{os.path.basename(file.name)}**: {len(content)} chars.")
            used_path = self.get_chunk_filename(file.name, is_url=False)
        elif url is not None and url.strip() != "":
            url = url.strip()
            content = self.extract_text_from_url(url)
            docs.append(Document(page_content=content, metadata={"source": url}))
            summary_msgs.append(f"- URL [{url}]: {len(content)} chars.")
            used_path = self.get_chunk_filename(url, is_url=True)
        else:
            # No input provided
            return gr.update(), gr.update(interactive=False), gr.update(interactive=False), "❌ Please upload a file or enter a URL."
        # Try to load or create vectorstore for this source
        try:
            vectorstore = self.load_faiss_vectorstore(used_path, emb_model)
            if vectorstore is not None:
                loaded_status = f"Existing chunks loaded for source [{os.path.basename(used_path)}]"
            else:
                vectorstore = self.build_and_save_vectorstore(docs, used_path, emb_model)
                loaded_status = f"Chunked and indexed new source: [{os.path.basename(used_path)}]"
            # Build the QA chain with the selected LLM and embedding
            qa_chain = self.make_qa_chain(vectorstore, llm_name, api_key, api_base)
            # Save for app session
            self.state["vectorstore"] = vectorstore
            self.state["qa_chain"] = qa_chain
            self.state["doc_summary"] = "\n".join(summary_msgs)
            self.state["llm"] = llm_name
            self.state["embedding"] = emb_model
            return gr.update(value=loaded_status + "\n" + "\n".join(summary_msgs)), gr.update(interactive=True), gr.update(interactive=True), ""
        except Exception as e:
            return gr.update(), gr.update(interactive=False), gr.update(interactive=False), f"❌ Error loading/creating vectorstore: {e}"

    def answer_func(self, question):
        """Answer a user question using the loaded QA chain."""
        if self.state["qa_chain"] is None:
            return "❌ Please load a document or URL first."
        try:
            res = self.state["qa_chain"](question)
            return res['result']
        except Exception as e:
            return f"Error in QA: {e}"

    def launch(self):
        """
        Launch the Gradio UI for the RAG app.
        Sets up all UI elements and binds event handlers.
        """
        with gr.Blocks() as demo:
            gr.Markdown("""# LangChain RAG LLM Demo\nUpload ANY file or paste a URL.\nChoose your LLM and embedding model.\nWill save/load vectorstore for each source.""")

            file_box = gr.File(label="Upload a File", file_count="single")
            url_box = gr.Textbox(label="Or paste a public URL")
            llm_dropdown = gr.Dropdown([m["name"] for m in self.llm_models], label="Choose LLM Model", value=self.llm_models[0]["name"])
            emb_dropdown = gr.Dropdown(self.embedding_models, label="Choose Embedding Model", value=self.embedding_models[0])
            load_btn = gr.Button("Load Document(s)")
            status = gr.Markdown()
            qa_box = gr.Textbox(label="Ask a question", interactive=False)
            submit_btn = gr.Button("Get Answer", interactive=False)
            answer_box = gr.Textbox(label="Answer", interactive=False)

            # Bind the load_docs handler to the Load Document(s) button.
            load_btn.click(
                self.load_docs,
                [file_box, url_box, llm_dropdown, emb_dropdown],
                [status, qa_box, submit_btn, answer_box],
            )

            # Bind the answer_func handler to the Get Answer button.
            submit_btn.click(
                self.answer_func,
                qa_box,
                answer_box
            )

            demo.launch()

if __name__ == "__main__":
    # Entry point for running the app as a script
    app = RAGGradioApp()
    app.launch() 
