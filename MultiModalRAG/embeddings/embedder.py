# embeddings/embedder.py

import subprocess
import json
import uuid
from datetime import datetime
import os


def get_filetype_from_name(filename):
    return os.path.splitext(filename)[1].lstrip(".").lower() or "unknown"


def embed_text(text, model="nomic-embed-text:v1.5", metadata=None):
    """
    Embeds text with optional metadata. Adds uuid, timestamp, and filetype.

    Args:
        text (str): Text to embed.
        model (str): Embedding model name.
        metadata (dict): Custom metadata (must include 'source_file').

    Returns:
        dict with keys: id, embedding, metadata
    """
    try:
        if not metadata or "source_file" not in metadata:
            raise ValueError("metadata with 'source_file' is required")

        # Get timestamp and filetype
        metadata_enriched = {
            "uuid": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "filetype": get_filetype_from_name(metadata["source_file"]),
            **metadata,
        }

        # Run Ollama command
        command = ["ollama", "run", model, text]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise RuntimeError(f"Ollama embedding failed: {result.stderr.decode()}")

        embedding = json.loads(result.stdout.decode("utf-8"))

        return {
            "id": metadata_enriched["uuid"],
            "embedding": embedding,
            "metadata": metadata_enriched,
        }

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None
