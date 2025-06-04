# processors/gemma_interface.py
import subprocess
import json


def run_ollama_gemma(prompt, model="gemma3:12b"):
    command = ["ollama", "run", model, prompt]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8")


def generate_caption_from_text(text):
    prompt = f"Describe the following instructional image based on its OCR text:\n\n{text}\n\nCaption:"
    command = ["ollama", "run", "gemma3:12b", prompt]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())

    return result.stdout.decode().strip()
