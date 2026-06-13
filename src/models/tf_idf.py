import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def analyze_chunk_tfidf():
    """Analyze TF-IDF scores for each book chunk"""
    
    # Load your chunks
    chunks_file = "data/jurafsky_chunks/chunks.jsonl"
    print(f"📚 Loading chunks from {chunks_file}...")
    
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Build TF-IDF model
    print("🔧 Building TF-IDF model...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b\w+\b',
        strip_accents='unicode'
    )
    
    # Extract text from chunks
    chunk_texts = [chunk['text'] for chunk in chunks]
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    
    print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Get feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    # Analyze each chunk
    print("\n📊 Analyzing TF-IDF scores for each chunk...")
    
    chunk_analysis = []
    
    for chunk_idx, chunk in enumerate(chunks):
        # Get TF-IDF scores for this chunk
        chunk_tfidf = tfidf_matrix[chunk_idx].toarray()[0]
        
        # Get top terms for this chunk
        top_indices = np.argsort(chunk_tfidf)[::-1][:10]  # Top 10 terms
        
        top_terms = []
        for idx in top_indices:
            if chunk_tfidf[idx] > 0:  # Only include terms with non-zero scores
                term = feature_names[idx]
                score = chunk_tfidf[idx]
                top_terms.append({
                    'term': term,
                    'tfidf_score': float(score),
                    'rank': len(top_terms) + 1
                })
        
        chunk_analysis.append({
            'chunk_id': chunk['chunk_id'],
            'word_count': chunk['word_count'],
            'top_tfidf_terms': top_terms,
            'max_tfidf_score': float(np.max(chunk_tfidf)) if np.max(chunk_tfidf) > 0 else 0,
            'avg_tfidf_score': float(np.mean(chunk_tfidf))
        })
        
        if (chunk_idx + 1) % 20 == 0:
            print(f"   Processed {chunk_idx + 1}/{len(chunks)} chunks...")
    
    return chunk_analysis, chunks, vectorizer, tfidf_matrix

def save_analysis_results(chunk_analysis, output_file="chunk_tfidf_analysis.json"):
    """Save analysis results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis saved to {output_file}")

def print_top_chunks_by_tfidf(chunk_analysis, top_n=10):
    """Print chunks with highest average TF-IDF scores"""
    print(f"\n🏆 Top {top_n} chunks by average TF-IDF score:")
    print("=" * 80)
    
    # Sort by average TF-IDF score
    sorted_chunks = sorted(chunk_analysis, key=lambda x: x['avg_tfidf_score'], reverse=True)
    
    for i, chunk in enumerate(sorted_chunks[:top_n], 1):
        print(f"{i:2d}. Chunk {chunk['chunk_id']:3d} | Avg TF-IDF: {chunk['avg_tfidf_score']:.4f} | Max TF-IDF: {chunk['max_tfidf_score']:.4f} | Words: {chunk['word_count']}")
        print(f"    Top terms: {', '.join([t['term'] for t in chunk['top_tfidf_terms'][:5]])}")
        print()

def find_most_distinctive_terms_across_chunks(chunk_analysis, top_n=20):
    """Find terms that appear frequently in top TF-IDF positions"""
    term_frequency = defaultdict(int)
    term_scores = defaultdict(list)
    
    for chunk in chunk_analysis:
        for term_info in chunk['top_tfidf_terms']:
            term = term_info['term']
            term_frequency[term] += 1
            term_scores[term].append(term_info['tfidf_score'])
    
    # Calculate average score and frequency
    term_stats = []
    for term, freq in term_frequency.items():
        avg_score = np.mean(term_scores[term])
        term_stats.append({
            'term': term,
            'frequency': freq,
            'avg_tfidf_score': avg_score,
            'total_occurrences': freq
        })
    
    # Sort by average score
    term_stats.sort(key=lambda x: x['avg_tfidf_score'], reverse=True)
    
    print(f"\n🎯 Most distinctive terms across all chunks:")
    print("=" * 80)
    
    for i, stat in enumerate(term_stats[:top_n], 1):
        print(f"{i:2d}. {stat['term']:<15} | Avg Score: {stat['avg_tfidf_score']:.4f} | Frequency: {stat['frequency']:2d} chunks")
        print()

def search_word_in_chunks(query, chunks, vectorizer, tfidf_matrix, top_k=5):
    """Search for a specific word and return relevant chunks"""
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 80)
    
    # Transform query using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and all chunks
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # Get top matching chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"🎯 Top {top_k} chunks containing '{query}':")
    print()
    
    for i, idx in enumerate(top_indices, 1):
        if similarities[idx] > 0:
            chunk = chunks[idx]
            print(f"{i}. Chunk {chunk['chunk_id']:3d} | Similarity: {similarities[idx]:.4f} | Words: {chunk['word_count']}")
            print(f"   Preview: {chunk['text'][:200]}...")
            
            # Show top terms in this chunk
            chunk_tfidf = tfidf_matrix[idx].toarray()[0]
            top_term_indices = np.argsort(chunk_tfidf)[::-1][:5]
            feature_names = vectorizer.get_feature_names_out()
            top_terms = [feature_names[idx] for idx in top_term_indices if chunk_tfidf[idx] > 0]
            print(f"   Top terms: {', '.join(top_terms)}")
            print()
        else:
            print(f"{i}. No chunks found with '{query}'")
            break

def interactive_search(chunks, vectorizer, tfidf_matrix):
    """Interactive search interface"""
    print("\n🎯 Interactive Word Search")
    print("=" * 50)
    print("Type a word to search for it in the book chunks.")
    print("Type 'quit' to exit the search mode.")
    print("=" * 50)
    
    while True:
        query = input("\n🔍 Enter a word to search: ").strip()
        
        if query.lower() == 'quit':
            print("👋 Exiting search mode...")
            break
            
        if not query:
            print("⚠️  Please enter a word to search.")
            continue
            
        # Search for the word
        search_word_in_chunks(query, chunks, vectorizer, tfidf_matrix, top_k=5)

def main():
    """Main analysis function"""
    print("🎯 TF-IDF Analysis for Jurafsky & Martin Book Chunks")
    print("=" * 60)
    
    # Run analysis
    chunk_analysis, chunks, vectorizer, tfidf_matrix = analyze_chunk_tfidf()
    
    # Save results
    save_analysis_results(chunk_analysis)
    
    # Print top chunks
    print_top_chunks_by_tfidf(chunk_analysis)
    
    # Find most distinctive terms
    find_most_distinctive_terms_across_chunks(chunk_analysis)
    
    print("\n🎉 TF-IDF Analysis Complete!")
    print("\n🚀 Starting interactive word search...")
    print("You can now type words to search for them in the book chunks!")
    
    # Start interactive search
    interactive_search(chunks, vectorizer, tfidf_matrix)

# Run the analysis
if __name__ == "__main__":
    chunk_analysis, chunks, vectorizer, tfidf_matrix = main()
