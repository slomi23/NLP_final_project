import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import time

def load_chunks_from_jsonl(file_path):
    """Loads chunks from a JSONL file."""
    chunks = []
    if not file_path.exists():
        return chunks
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    text = data.get('text') or data.get('content')
                    if text: chunks.append(text)
            except: pass
    return chunks

def calculate_mrr_tfidf_fast(val_triplets, all_passages, tfidf_vectorizer, tfidf_matrix, top_k=10):
    """
    Fast MRR calculation using batch processing.
    """
    print("📊 Calculating similarities in batch...")
    start_time = time.time()
    
    # 1. Extract all queries and ground truths
    queries = [t['query'] for t in val_triplets]
    ground_truths = [t['positive'] for t in val_triplets]
    
    # 2. Transform ALL queries at once
    # This is much faster than looping
    query_tfidf_matrix = tfidf_vectorizer.transform(queries)
    
    # 3. Calculate cosine similarity for ALL queries at once
    # Result shape: (num_queries, num_passages)
    similarities = cosine_similarity(query_tfidf_matrix, tfidf_matrix)
    
    print(f"✅ Similarities calculated in {time.time() - start_time:.2f} seconds")
    
    # 4. Get top K indices for all queries
    # np.argsort returns indices that would sort each row
    # [::-1] sorts descending (highest similarity first)
    # [:, :top_k] takes the top_k columns
    top_indices = np.argsort(similarities)[:, ::-1][:, :top_k]
    
    # 5. Calculate MRR
    reciprocal_ranks = []
    
    # Create a mapping from passage text to its index for fast lookup
    # Note: This assumes unique passages. If duplicates exist, this might pick the first one.
    # For exact match MRR, we check if the ground truth text is in the top_k passages.
    
    print("📊 Calculating MRR...")
    for i, ground_truth_text in enumerate(ground_truths):
        found = False
        # Check top_k indices for this query
        for rank, idx in enumerate(top_indices[i], 1):
            idx_int = int(idx)
            if all_passages[idx_int] == ground_truth_text:
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break
        
        if not found:
            reciprocal_ranks.append(0.0)
            
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(val_triplets)} queries...")
            
    mrr = np.mean(reciprocal_ranks)
    return mrr

def main():
    all_passages = []

    # 1. Load Jurafsky Chunks
    jurafsky_file = Path(r"C:\Users\salo\NLP_final_project\data\jurafsky_chunks\chunks.jsonl")
    if jurafsky_file.exists():
        chunks = load_chunks_from_jsonl(jurafsky_file)
        all_passages.extend(chunks)
        print(f"✅ Loaded {len(chunks)} Jurafsky chunks.")

    # 2. Load ArXiv Passages
    arxiv_file = Path(r"C:\Users\salo\NLP_final_project\data\processed\arxiv_cs_papers_processed.csv")
    if arxiv_file.exists():
        arxiv_df = pd.read_csv(arxiv_file)
        arxiv_df['combined_text'] = arxiv_df.apply(
            lambda row: f"Title: {row['title']}. Abstract: {row['abstract']}" if pd.notna(row['title']) and pd.notna(row['abstract']) else "",
            axis=1
        )
        all_passages.extend(arxiv_df['combined_text'].tolist())
        print(f"✅ Loaded {len(arxiv_df)} ArXiv papers.")

    # 3. Load MS MARCO Passages
    ms_file = Path(r"C:\Users\salo\NLP_final_project\data\processed\msmarco_small_passages.json")
    if ms_file.exists():
        with open(ms_file, 'r', encoding='utf-8') as f:
            ms_passages = json.load(f)
            all_passages.extend([p['text'] for p in ms_passages])
            print(f"✅ Loaded {len(ms_passages)} MS MARCO passages.")

    print(f"📚 Total passages in index: {len(all_passages)}")

    # 4. Fit TF-IDF
    print("🔧 Fitting TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_passages)
    print("✅ TF-IDF Matrix Ready.")

    # 5. Load Validation Triplets
    val_file = Path(r"C:\Users\salo\NLP_final_project\data\processed\combined_val_triplets.jsonl")
    val_triplets = []
    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    val_triplets.append(json.loads(line.strip()))
        print(f"✅ Loaded {len(val_triplets)} validation triplets.")
    else:
        print("⚠️ Validation triplets not found.")
        return

    # 6. Calculate MRR
    mrr = calculate_mrr_tfidf_fast(val_triplets, all_passages, tfidf_vectorizer, tfidf_matrix, top_k=10)
    
    print("\n" + "="*50)
    print("🏆 TF-IDF BASELINE METRICS")
    print("="*50)
    print(f"Mean Reciprocal Rank (MRR@10): {mrr:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
