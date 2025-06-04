# loaders/docx_loader.py

from pathlib import Path
import docx


def load_docx_files(directory):
    docx_texts = {}
    for docx_path in Path(directory).glob("*.docx"):
        try:
            doc = docx.Document(docx_path)
            text = "\n".join(p.text for p in doc.paragraphs)
            docx_texts[docx_path.name] = text.strip()
        except Exception as e:
            print(f"Failed to load DOCX {docx_path}: {e}")
    return docx_texts
