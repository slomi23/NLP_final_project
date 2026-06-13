import json
import math
import string
import collections
import os
import numpy as np

K1 = 1.5
B  = 0.75

def tokenize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

def load_chunks_from_jsonl(file_path: str) -> list[str]:
    chunks = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return chunks

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    text = data.get("text") or data.get("content") or data.get("chunk")
                    if text:
                        chunks.append(text)
                elif isinstance(data, str):
                    chunks.append(data)
            except json.JSONDecodeError:
                chunks.append(line)

    return chunks

class BM25Index:
    def __init__(self, corpus: list[str], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b  = b
        self.corpus = corpus
        N = len(corpus)
        tokenised = [tokenize(doc) for doc in corpus]

        self.doc_lengths = np.array([len(t) for t in tokenised], dtype=np.float32)
        self.avgdl = float(self.doc_lengths.mean())

        df: dict[str, int] = collections.defaultdict(int)
        self.term_freqs: list[dict[str, int]] = []

        for tokens in tokenised:
            tf = collections.Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                df[term] += 1

        self.idf: dict[str, float] = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query: str) -> np.ndarray:
        query_tokens = tokenize(query)
        scores = np.zeros(len(self.corpus), dtype=np.float64)

        for token in set(query_tokens):
            if token not in self.idf:
                continue

            idf_val = self.idf[token]
            tf_vec = np.array(
                [tf.get(token, 0) for tf in self.term_freqs], dtype=np.float64
            )
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths / self.avgdl)
            scores += idf_val * (tf_vec * (self.k1 + 1.0)) / (tf_vec + norm)

        return scores

def find_best_chunk(query: str, index: BM25Index) -> tuple[str | None, float]:
    if not index.corpus:
        return None, 0.0

    scores = index.get_scores(query)
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    return index.corpus[best_index], best_score

def main():
    file_path = r"C:\me\2025_2026\NLP\final_project\NLP_final_project\data\jurafsky_chunks\chunks.jsonl"

    print(f"Loading chunks from {file_path}...")
    chunks = load_chunks_from_jsonl(file_path)

    if not chunks:
        print("No chunks loaded. Please check the file path and format.")
        return

    print(f"Loaded {len(chunks)} chunks.")
    print("Building BM25 index...")
    index = BM25Index(chunks)
    print("=== Interactive BM25 Search ===")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Enter your query: ")
        except EOFError:
            break

        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not query.strip():
            print("Please enter a non-empty query.\n")
            continue

        best_match, score = find_best_chunk(query, index)

        if best_match:
            print(f"\n--- Result ---")
            print(f"Query:      '{query}'")
            print(f"Best Match: '{best_match}'")
            print(f"Score:       {score:.4f}")
            print("----------------\n")
        else:
            print("No match found.\n")

if __name__ == "__main__":
    main()
