"""Clean Jurafsky/Martin PDF chunker for neural search.

Creates 200-300 word chunks from Speech and Language Processing PDF.
Filters out common PDF artifacts: page numbers, index, references,
bibliography, acknowledgements, appendix/tagset tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

try:
    from pypdf import PdfReader
except ImportError as e:
    raise ImportError("Install pypdf first: pip install pypdf") from e


BOOK_PATH = Path("data/raw/Speech_and_Language_Processing.pdf")
OUTPUT_PATH = Path("data/jurafsky_chunks/chunks.jsonl")

MIN_WORDS = 180
TARGET_WORDS = 250
MAX_WORDS = 300


BAD_SECTION_MARKERS = [
    "acknowledg",
    "bibliography",
    "references",
    "author index",
    "subject index",
    "index",
    "appendix",
    "tag description example",
    "figure c.",
    "ucrel",
]


def extract_text_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        # Skip very likely back matter/index pages. This is heuristic but helps a lot.
        lower_start = text[:1000].lower()
        if (
            "author index" in lower_start
            or "subject index" in lower_start
            or lower_start.strip().startswith("index")
        ):
            continue

        pages.append(text)

    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        # Remove isolated page numbers / page artifacts.
        if re.fullmatch(r"\d{1,4}", line):
            continue

        # Remove repeated PDF headers/footers when present.
        low = line.lower()
        if low in {
            "speech and language processing",
            "daniel jurafsky & james h. martin",
            "draft of january 7, 2023",
        }:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove Penn Treebank-like slash tags only when they look like standalone tags.
    text = re.sub(r"\s/[A-Z][A-Z0-9_-]*\b", " ", text)

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def looks_like_junk(text: str) -> bool:
    text = text.strip()
    words = text.split()

    if len(words) < 30:
        return True

    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars < 150:
        return True

    lower = text.lower()

    # Drop common back matter / table/list chunks.
    if any(marker in lower[:500] for marker in BAD_SECTION_MARKERS):
        return True

    # Drop index-like entries: too many commas with shortish text.
    if text.count(",") > 60 and len(words) < 260:
        return True

    # Drop table/tagset-looking chunks.
    taggy = len(re.findall(r"\b[A-Z]{2,5}\b", text))
    if taggy > 40 and len(words) < 260:
        return True

    # Drop chunks dominated by names/citations/pages.
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.20:
        return True

    return False


def split_into_paragraphs(text: str) -> List[str]:
    rough_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []

    for p in rough_paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue

        if looks_like_junk(p):
            continue

        paragraphs.append(p)

    return paragraphs


def split_long_paragraph(paragraph: str) -> List[str]:
    words = paragraph.split()
    pieces = []

    start = 0
    while start < len(words):
        end = min(start + TARGET_WORDS, len(words))
        piece = " ".join(words[start:end])

        if len(piece.split()) >= MIN_WORDS and not looks_like_junk(piece):
            pieces.append(piece)

        start = end

    return pieces


def build_chunks(paragraphs: List[str]) -> List[dict]:
    chunks = []
    current = []
    current_word_count = 0
    chunk_id = 0

    def flush_current():
        nonlocal chunk_id, current, current_word_count

        if not current:
            return

        text = "\n\n".join(current).strip()
        wc = len(text.split())

        if wc >= MIN_WORDS and not looks_like_junk(text):
            chunks.append({
                "chunk_id": chunk_id,
                "word_count": wc,
                "text": text,
            })
            chunk_id += 1

        current = []
        current_word_count = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())

        if paragraph_words > MAX_WORDS:
            flush_current()
            for piece in split_long_paragraph(paragraph):
                chunks.append({
                    "chunk_id": chunk_id,
                    "word_count": len(piece.split()),
                    "text": piece,
                })
                chunk_id += 1
            continue

        if current_word_count + paragraph_words > MAX_WORDS:
            if current_word_count >= MIN_WORDS:
                flush_current()
                current = [paragraph]
                current_word_count = paragraph_words
            else:
                current.append(paragraph)
                current_word_count += paragraph_words
        else:
            current.append(paragraph)
            current_word_count += paragraph_words

    flush_current()
    return chunks


def save_chunks(chunks: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    print("CURRENT DIR:", Path.cwd())
    print("Extracting text from PDF:", BOOK_PATH)
    raw_text = extract_text_from_pdf(BOOK_PATH)

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Splitting into paragraphs...")
    paragraphs = split_into_paragraphs(cleaned_text)
    print(f"Found {len(paragraphs)} useful paragraphs")

    print("Building chunks...")
    chunks = build_chunks(paragraphs)

    save_chunks(chunks, OUTPUT_PATH)

    print(f"Created {len(chunks)} clean chunks")
    print(f"Saved to {OUTPUT_PATH}")

    if chunks:
        print("\nSample chunk:")
        print(chunks[0]["text"][:1200])


if __name__ == "__main__":
    main()
