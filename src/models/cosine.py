import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_chunks_from_jsonl(file_path):
    """
    Loads chunks from a JSONL file.
    Assumes each line is either a JSON object with a 'text' key 
    or a plain string.
    """
    chunks = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return chunks

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Try to parse as JSON
                data = json.loads(line)
                # Handle common JSON structures: {"text": "..."} or {"content": "..."}
                if isinstance(data, dict):
                    text = data.get('text') or data.get('content') or data.get('chunk')
                    if text:
                        chunks.append(text)
                elif isinstance(data, str):
                    chunks.append(data)
            except json.JSONDecodeError:
                # If it's not JSON, treat the whole line as the chunk
                chunks.append(line)
                
    return chunks

def find_best_chunk(query, chunks):
    """
    Finds the chunk with the highest cosine similarity to the query.
    """
    if not chunks:
        return None, 0.0

    # Combine query and chunks for consistent vectorization
    documents = chunks + [query]
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Handle edge case where no features are found (e.g. empty docs)
        return None, 0.0
    
    # Extract query vector (last one) and chunk vectors (all others)
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    # Find best match
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    
    return chunks[best_index], best_score

def main():
    # Path to your chunks
    file_path = r"C:\Users\salo\NLP_final_project\data\jurafsky_chunks\chunks.jsonl"
    
    print(f"Loading chunks from {file_path}...")
    chunks = load_chunks_from_jsonl(file_path)
    
    if not chunks:
        print("No chunks loaded. Please check the file path and format.")
        return

    print(f"Loaded {len(chunks)} chunks.")
    print("=== Interactive Cosine Similarity Search ===")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Enter your query: ")
        except EOFError:
            break
            
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not query.strip():
            print("Please enter a non-empty query.\n")
            continue

        best_match, score = find_best_chunk(query, chunks)
        
        if best_match:
            print(f"\n--- Result ---")
            print(f"Query:     '{query}'")
            print(f"Best Match: '{best_match}'")
            print(f"Score:      {score:.4f}")
            print("----------------\n")
        else:
            print("No match found.\n")

if __name__ == "__main__":
    main()
