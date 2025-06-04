# loaders/pdf_loader.py

from pathlib import Path
import fitz  # PyMuPDF


def load_pdfs(directory):
    pdf_texts = {}
    for pdf_path in Path(directory).glob("*.pdf"):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            pdf_texts[pdf_path.name] = text.strip()
        except Exception as e:
            print(f"Failed to load PDF {pdf_path}: {e}")
    return pdf_texts
