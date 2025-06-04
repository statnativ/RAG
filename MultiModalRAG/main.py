# main.py

from config import DATA_DIRS, EMBED_MODEL, VECTOR_STORE_PATH
from loaders import (
    load_images,
    load_pdfs,
    load_docx_files,
    load_markdown_files,
    load_excel_files,
)
from embeddings.embedder import embed_text
from storage.vector_store import VectorStore
from utils.text_splitter import chunk_text
from loaders.image_loader import load_images


def collect_all_texts():
    all_texts = {}

    print("🔍 Loading documents...")
    all_texts.update(load_images(DATA_DIRS["images"]))
    all_texts.update(load_pdfs(DATA_DIRS["pdf"]))
    all_texts.update(load_docx_files(DATA_DIRS["docx"]))
    all_texts.update(load_markdown_files(DATA_DIRS["markdown"]))
    all_texts.update(load_excel_files(DATA_DIRS["excel"]))
    image_entries = load_images(DATA_DIRS["images"])

    return all_texts


def load_images(directory):
    for filename, item in image_entries.items():
        description = item["description"]
        metadata = item["metadata"]
        chunks = chunk_text(description)
        vector_store = VectorStore(dim=768)  # Adjust based on your embed model

    for i, chunk in enumerate(chunks):
        embed_result = embed_text(
            chunk,
            model=EMBED_MODEL,
            metadata={**metadata, "chunk_index": i, "text": chunk},
        )

        if embed_result:
            vector_store.add([embed_result["embedding"]], [embed_result["metadata"]])


def process_and_store(text_data: dict, embed_model: str, store_path: str):
    print("🧠 Initializing vector store...")
    vector_store = VectorStore(dim=768)  # Adjust based on your embed model

    for filename, full_text in text_data.items():
        chunks = chunk_text(full_text, chunk_size=500, overlap=100)
        for i, chunk in enumerate(chunks):
            try:
                embedding = embed_text(
                    chunk,
                    model=embed_model,
                    metadata={"source_file": filename, "chunk_index": i},
                )
                metadata = {"source_file": filename, "chunk_index": i, "text": chunk}
                vector_store.add([embedding], [metadata])
            except Exception as e:
                print(f"❌ Embedding failed for {filename}, chunk {i}: {e}")

    print("💾 Saving vector store...")
    vector_store.save(store_path)
    print("✅ All data embedded and stored!")


def main():
    all_text_data = collect_all_texts()
    process_and_store(all_text_data, EMBED_MODEL, VECTOR_STORE_PATH)


if __name__ == "__main__":
    main()
