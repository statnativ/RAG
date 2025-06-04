# loaders/image_loader.py

from pathlib import Path
from PIL import Image
import pytesseract
import uuid
from datetime import datetime
from processors.gemma_interface import generate_caption_from_text


def load_images(directory):
    """
    Processes instructional images:
    - Extracts visible text using OCR
    - Uses LLM (Gemma) to generate caption
    - Returns dict: { filename: { "description": str, "metadata": dict } }
    """
    image_data = {}

    for img_path in Path(directory).glob("*.[pj][pn]g"):  # jpg, png
        try:
            img = Image.open(img_path)
            ocr_text = pytesseract.image_to_string(img).strip()

            if not ocr_text:
                continue  # skip blank OCR results

            # Generate caption from OCR using Gemma
            caption = generate_caption_from_text(ocr_text)

            image_data[img_path.name] = {
                "description": caption,
                "metadata": {
                    "uuid": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "filetype": img_path.suffix.lstrip("."),
                    "source_file": img_path.name,
                    "ocr_text": ocr_text,
                    "image_path": str(img_path.resolve()),
                },
            }

        except Exception as e:
            print(f"❌ Error processing image {img_path}: {e}")

    return image_data
