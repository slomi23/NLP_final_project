"""Clean Jurafsky/Martin PDF chunker for neural search.

Creates 180-300 word chunks from Speech and Language Processing PDF.
Filters common PDF artifacts: page numbers, table of contents dot leaders,
index pages, references, bibliography, acknowledgements, appendix/tagset tables,
and chunks dominated by dots/spaces/numbers.

Run from the project root:
    python src/data/book_chunks.py
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
    "contents",
    "table of contents",
    "brief contents",
    "tag description example",
    "figure c.",
    "ucrel",
]


HEADER_FOOTER_LINES = {
    "speech and language processing",
    "daniel jurafsky & james h. martin",
    "draft of january 7, 2023",
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF, skipping obvious index/back-matter pages."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        lower_start = text[:1200].lower().strip()

        # Skip very likely back matter/index pages.
        if (
            "author index" in lower_start
            or "subject index" in lower_start
            or lower_start.startswith("index")
        ):
            continue

        pages.append(text)

    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize PDF text and remove obvious line-level artifacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        low = line.lower().strip()

        # Remove isolated page numbers / page artifacts.
        if re.fullmatch(r"\d{1,4}", line):
            continue

        # Remove repeated PDF headers/footers when present.
        if low in HEADER_FOOTER_LINES:
            continue

        # Remove lines that are almost entirely dot leaders / page references.
        # Example: "Agreement . . . . . . . . . . . . . . 403"
        if looks_like_toc_line(line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove table-of-contents dot leaders inside merged text.
    text = re.sub(r"(?:\s*\.\s*){5,}\d{1,4}", " ", text)
    text = re.sub(r"\.{4,}\s*\d{1,4}", " ", text)

    # Remove Penn Treebank-like slash tags only when they look like standalone tags.
    text = re.sub(r"\s/[A-Z][A-Z0-9_-]*\b", " ", text)

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def looks_like_toc_line(line: str) -> bool:
    """Detect table-of-contents style lines with dot leaders and page numbers."""
    stripped = line.strip()

    if not stripped:
        return True

    # "Agreement . . . . . . . . . 403"
    if re.search(r"(?:\.\s*){5,}\d{1,4}\s*$", stripped):
        return True

    # "Agreement ................ 403"
    if re.search(r"\.{4,}\s*\d{1,4}\s*$", stripped):
        return True

    total_chars = max(len(stripped), 1)
    dot_ratio = stripped.count(".") / total_chars
    digit_ratio = sum(ch.isdigit() for ch in stripped) / total_chars
    alpha_ratio = sum(ch.isalpha() for ch in stripped) / total_chars

    # Dot-heavy and digit-heavy short line is almost always contents/table.
    if dot_ratio > 0.18 and digit_ratio > 0.03 and alpha_ratio < 0.70:
        return True

    return False


def looks_like_junk(text: str) -> bool:
    """Return True for chunks/paragraphs that are not useful prose."""
    text = text.strip()
    words = text.split()

    if len(words) < 30:
        return True

    lower = text.lower()

    # Drop common back matter / table/list chunks.
    if any(marker in lower[:700] for marker in BAD_SECTION_MARKERS):
        return True

    total_chars = max(len(text), 1)
    alpha_chars = sum(ch.isalpha() for ch in text)
    digit_chars = sum(ch.isdigit() for ch in text)
    dot_chars = text.count(".")
    comma_chars = text.count(",")

    alpha_ratio = alpha_chars / total_chars
    digit_ratio = digit_chars / total_chars
    dot_ratio = dot_chars / total_chars

    # Real prose should have enough alphabetic content.
    if alpha_chars < 250:
        return True

    if alpha_ratio < 0.55:
        return True

    # Drop table-of-contents chunks with dot leaders:
    # ". . . . . . . . . . . . . 401"
    dot_leader_patterns = [
        r"(?:\.\s*){5,}",      # many spaced dots
        r"\.{4,}\s*\d{1,4}",  # .... 401
        r"(?:\s\.\s){4,}",    # " . . . . "
    ]

    if any(re.search(pattern, text) for pattern in dot_leader_patterns):
        return True

    # Too many dots usually means table of contents, not prose.
    if dot_ratio > 0.08:
        return True

    # Too many digits usually means page numbers, tables, index, or references.
    if digit_ratio > 0.12:
        return True

    # Index-like chunks: many commas and short-ish text.
    if comma_chars > 60 and len(words) < 280:
        return True

    # Table/tagset-looking chunks: many all-caps short tags.
    taggy = len(re.findall(r"\b[A-Z]{2,5}\b", text))
    if taggy > 35 and len(words) < 280:
        return True

    # Drop chunks that look like a list of numbered headings.
    numbered_heading_count = len(re.findall(r"\b\d{1,2}\.\d+\b", text))
    if numbered_heading_count >= 5:
        return True

    # Drop chunks with many page-number-like tokens.
    page_ref_count = len(re.findall(r"\b\d{2,4}\b", text))
    if page_ref_count >= 25 and len(words) < 300:
        return True

    # If there are many short lines/headings, likely contents/table.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        short_lines = [line for line in lines if len(line.split()) <= 8]
        if len(short_lines) / len(lines) > 0.6:
            return True

    return False


def split_into_paragraphs(text: str) -> List[str]:
    """Split cleaned PDF text into useful paragraphs."""
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
    """Split a paragraph longer than MAX_WORDS into TARGET_WORDS pieces."""
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
    """Build 180-300 word chunks from paragraphs."""
    chunks = []
    current = []
    current_word_count = 0
    chunk_id = 0

    def flush_current() -> None:
        nonlocal chunk_id, current, current_word_count

        if not current:
            return

        chunk_text = "\n\n".join(current).strip()
        word_count = len(chunk_text.split())

        if word_count >= MIN_WORDS and not looks_like_junk(chunk_text):
            chunks.append({
                "chunk_id": chunk_id,
                "word_count": word_count,
                "text": chunk_text,
            })
            chunk_id += 1

        current = []
        current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        if paragraph_word_count > MAX_WORDS:
            flush_current()
            for piece in split_long_paragraph(paragraph):
                chunks.append({
                    "chunk_id": chunk_id,
                    "word_count": len(piece.split()),
                    "text": piece,
                })
                chunk_id += 1
            continue

        if current_word_count + paragraph_word_count > MAX_WORDS:
            if current_word_count >= MIN_WORDS:
                flush_current()
                current = [paragraph]
                current_word_count = paragraph_word_count
            else:
                current.append(paragraph)
                current_word_count += paragraph_word_count
        else:
            current.append(paragraph)
            current_word_count += paragraph_word_count

    flush_current()
    return chunks


def save_chunks(chunks: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def print_quality_report(chunks: List[dict]) -> None:
    """Print quick quality stats for generated chunks."""
    dot_leader_chunks = []
    numeric_chunks = []

    for chunk in chunks:
        text = chunk["text"]
        if re.search(r"(?:\.\s*){5,}", text) or " . . . " in text:
            dot_leader_chunks.append(chunk)

        digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
        if digit_ratio > 0.12:
            numeric_chunks.append(chunk)

    print("\nQuality report:")
    print(f"  chunks: {len(chunks)}")
    print(f"  dot-leader chunks: {len(dot_leader_chunks)}")
    print(f"  numeric-heavy chunks: {len(numeric_chunks)}")

    if dot_leader_chunks:
        print("\nExample bad dot-leader chunk:")
        print(dot_leader_chunks[0]["text"][:1000])


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
    print_quality_report(chunks)

    if chunks:
        print("\nSample chunk:")
        print(chunks[0]["text"][:1200])


if __name__ == "__main__":
    main()
