# pip install fastapi uvicorn faiss-cpu python-multipart pydantic
# api/main.py

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import shutil
import os

from config import DATA_DIRS, EMBED_MODEL, VECTOR_STORE_PATH
from embeddings.embedder import embed_text
from storage.vector_store import VectorStore
from loaders.image_loader import load_images
from main import collect_text_data, process_and_store_text, process_and_store_images

from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.schema import Document

from fastapi import Request

app = FastAPI(
    title="Multimodal RAG API",
    description="Upload documents/images, embed them, and run semantic search.",
    version="1.0",
)

vector_store = VectorStore.load(VECTOR_STORE_PATH)

llm = Ollama(model="gemma:12b")  # Or "llama3" or "mistral"
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load FAISS index using LangChain wrapper
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH, embeddings=OllamaEmbeddings(model=EMBED_MODEL)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create Conversational Retrieval Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, return_source_documents=True
)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat/")
def chat_with_rag(req: ChatRequest):
    """
    Ask a question and get conversational RAG-powered answer.
    Maintains chat history using LangChain memory.
    """
    result = rag_chain.run(req.question)

    return {"question": req.question, "answer": result}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), filetype: str = Form(...)):
    """
    Uploads a file to the appropriate folder for processing.
    Supported filetypes: pdf, docx, markdown, excel, image
    """
    dest_dir = DATA_DIRS.get(filetype.lower())
    if not dest_dir:
        return {"error": f"Unsupported filetype '{filetype}'."}

    os.makedirs(dest_dir, exist_ok=True)
    file_path = os.path.join(dest_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"Uploaded {file.filename} to {filetype} folder."}


@app.post("/ingest/")
def run_ingestion():
    """
    Ingest all files from their folders into the vector store.
    """
    global vector_store
    vector_store = VectorStore(dim=768)

    text_data = collect_text_data()
    process_and_store_text(text_data, EMBED_MODEL, vector_store)
    process_and_store_images(EMBED_MODEL, vector_store)
    vector_store.save(VECTOR_STORE_PATH)

    return {"status": "Ingestion complete."}


@app.post("/search/")
def semantic_search(req: QueryRequest):
    """
    Perform semantic search using user query.
    """
    embedded = embed_text(
        req.query, model=EMBED_MODEL, metadata={"source_file": "user_query"}
    )
    if not embedded:
        return {"error": "Embedding failed."}

    results = vector_store.search(embedded["embedding"], top_k=req.top_k)
    return {"query": req.query, "results": results}


@app.get("/")
def root():
    return {
        "message": "Multimodal RAG API is running. Use /upload, /ingest, or /search."
    }


# uvicorn api.main:app --reload --port 8000
# curl -X POST http://localhost:8000/chat/ \
#   -H "Content-Type: application/json" \
#   -d '{"question": "How do I reset the control valve system?"}'
# {
#     "question": "How do I reset the control valve system?",
#     "answer": "To reset the control valve, locate the manual override knob and follow the steps from the safety procedure...",
# }
