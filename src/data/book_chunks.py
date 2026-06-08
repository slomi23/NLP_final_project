import json
import re
from pathlib import Path

from pypdf import PdfReader

BOOK_PATH = Path("data/raw/Speech_and_Language_Processing.pdf")
OUTPUT_PATH = Path("data/jurafsky_chunks/chunks.jsonl")

MIN_WORDS = 200
TARGET_WORDS = 250
MAX_WORDS = 300


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages)


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text):
    paragraphs = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()

        if not p:
            continue
        word_count = len(p.split())

        if word_count < 15:
            continue
        paragraphs.append(p)
    return paragraphs


def build_chunks(paragraphs):
    chunks = []
    current_paragraphs = []
    current_word_count = 0
    chunk_id = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())
        if paragraph_words > MAX_WORDS:
            if current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append({
                    "chunk_id": chunk_id,
                    "word_count": current_word_count,
                    "text": chunk_text
                })
                chunk_id += 1
                current_paragraphs = []
                current_word_count = 0

            words = paragraph.split()
            start = 0
            while start < len(words):
                end = min(start + TARGET_WORDS, len(words))
                piece = " ".join(words[start:end])
                chunks.append({
                    "chunk_id": chunk_id,
                    "word_count": len(piece.split()),
                    "text": piece
                })
                chunk_id += 1
                start = end

            continue

        if current_word_count + paragraph_words > MAX_WORDS:
            if current_word_count >= MIN_WORDS:
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append({
                    "chunk_id": chunk_id,
                    "word_count": current_word_count,
                    "text": chunk_text
                })
                chunk_id += 1
                current_paragraphs = [paragraph]
                current_word_count = paragraph_words

            else:
                current_paragraphs.append(paragraph)
                current_word_count += paragraph_words

        else:
            current_paragraphs.append(paragraph)
            current_word_count += paragraph_words

    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunks.append({
            "chunk_id": chunk_id,
            "word_count": current_word_count,
            "text": chunk_text
        })

    return chunks


def save_chunks(chunks, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False))
            f.write("\n")

def main():
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(BOOK_PATH)
    print("Cleaning text...")
    text = clean_text(text)
    print("Splitting into paragraphs...")
    paragraphs = split_into_paragraphs(text)
    print(f"Found {len(paragraphs)} paragraphs")
    print("Building chunks...")
    chunks = build_chunks(paragraphs)
    save_chunks(chunks, OUTPUT_PATH)
    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    import os
    print("CURRENT DIR:", os.getcwd())
    main()