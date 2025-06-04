# loaders/markdown_loader.py

from pathlib import Path


def load_markdown_files(directory):
    md_texts = {}
    for md_path in Path(directory).glob("*.md"):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
                md_texts[md_path.name] = text.strip()
        except Exception as e:
            print(f"Failed to load Markdown {md_path}: {e}")
    return md_texts
