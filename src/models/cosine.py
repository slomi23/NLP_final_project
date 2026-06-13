import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
    
    # Build TF-IDF model (1-grams)
    print("🔧 Building TF-IDF model (1-grams)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b\w+\b',
        strip_accents='unicode'
    )
    
    # Build Bag-of-Words model (1-grams)
    print("🔧 Building Bag-of-Words model (1-grams)...")
    bow_vectorizer = CountVectorizer(
        max_features=5000,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b\w+\b',
        strip_accents='unicode'
    )
    
    # Build N-gram models for phrase search
    print("🔧 Building N-gram models (2-grams and 3-grams)...")
    bigram_vectorizer = CountVectorizer(
        ngram_range=(2, 2),  # Only 2-word phrases
        max_features=3000,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b\w+\b',
        strip_accents='unicode'
    )
    
    trigram_vectorizer = CountVectorizer(
        ngram_range=(3, 3),  # Only 3-word phrases
        max_features=2000,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b\w+\b',
        strip_accents='unicode'
    )
    
    # Extract text from chunks
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # Transform with all vectorizers
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)
    bow_matrix = bow_vectorizer.fit_transform(chunk_texts)
    bigram_matrix = bigram_vectorizer.fit_transform(chunk_texts)
    trigram_matrix = trigram_vectorizer.fit_transform(chunk_texts)
    
    print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"✅ Bag-of-Words matrix shape: {bow_matrix.shape}")
    print(f"✅ Bigram matrix shape: {bigram_matrix.shape}")
    print(f"✅ Trigram matrix shape: {trigram_matrix.shape}")
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    bigram_names = bigram_vectorizer.get_feature_names_out()
    trigram_names = trigram_vectorizer.get_feature_names_out()
    
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
    
    return (chunk_analysis, chunks, tfidf_vectorizer, bow_vectorizer, bigram_vectorizer, 
            trigram_vectorizer, tfidf_matrix, bow_matrix, bigram_matrix, trigram_matrix,
            feature_names, bigram_names, trigram_names)

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

def detect_search_type(query):
    """Detect if query is a single word or phrase"""
    words = query.split()
    if len(words) == 1:
        return 'single_word'
    elif len(words) == 2:
        return 'bigram_phrase'
    elif len(words) == 3:
        return 'trigram_phrase'
    else:
        return 'multi_word'

def search_with_all_methods(query, chunks, tfidf_vectorizer, bow_vectorizer, bigram_vectorizer, 
                           trigram_vectorizer, tfidf_matrix, bow_matrix, bigram_matrix, trigram_matrix,
                           feature_names, bigram_names, trigram_names, top_k=5):
    """Search using all methods (single words, phrases)"""
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 80)
    
    # Detect search type
    search_type = detect_search_type(query)
    print(f"🎯 Search type detected: {search_type}")
    
    # Check if query exists in appropriate vocabulary
    found = False
    
    if search_type == 'single_word':
        # Check single word vocabulary
        query_lower = query.lower()
        feature_names_lower = [name.lower() for name in feature_names]
        if query_lower in feature_names_lower:
            found = True
            search_single_word(query, chunks, tfidf_vectorizer, bow_vectorizer, 
                             tfidf_matrix, bow_matrix, feature_names, top_k)
    
    elif search_type == 'bigram_phrase':
        # Check bigram vocabulary
        query_lower = query.lower()
        bigram_names_lower = [name.lower() for name in bigram_names]
        if query_lower in bigram_names_lower:
            found = True
            search_phrase(query, chunks, bigram_vectorizer, bigram_matrix, 
                         bigram_names, 'bigram', top_k)
    
    elif search_type == 'trigram_phrase':
        # Check trigram vocabulary
        query_lower = query.lower()
        trigram_names_lower = [name.lower() for name in trigram_names]
        if query_lower in trigram_names_lower:
            found = True
            search_phrase(query, chunks, trigram_vectorizer, trigram_matrix, 
                         trigram_names, 'trigram', top_k)
    
    else:  # multi_word
        # Try to match as bigram or trigram
        words = query.split()
        
        # Try bigram combinations
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigram_lower = bigram.lower()
            bigram_names_lower = [name.lower() for name in bigram_names]
            if bigram_lower in bigram_names_lower:
                found = True
                print(f"\n🎯 Found matching bigram: '{bigram}'")
                search_phrase(bigram, chunks, bigram_vectorizer, bigram_matrix, 
                             bigram_names, 'bigram', top_k)
                break
        
        # Try trigram combinations
        if not found and len(words) >= 3:
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                trigram_lower = trigram.lower()
                trigram_names_lower = [name.lower() for name in trigram_names]
                if trigram_lower in trigram_names_lower:
                    found = True
                    print(f"\n🎯 Found matching trigram: '{trigram}'")
                    search_phrase(trigram, chunks, trigram_vectorizer, trigram_matrix, 
                                 trigram_names, 'trigram', top_k)
                    break
    
    if not found:
        print(f"❌ No matches found for '{query}'")
        print("💡 Try:")
        print("  - Single words: 'neural', 'language', 'algorithm'")
        print("  - 2-word phrases: 'machine learning', 'natural language'")
        print("  - 3-word phrases: 'speech recognition', 'deep learning'")

def search_single_word(query, chunks, tfidf_vectorizer, bow_vectorizer, tfidf_matrix, 
                      bow_matrix, feature_names, top_k):
    """Search for single word using both TF-IDF and Bag-of-Words"""
    # Transform query
    query_tfidf = tfidf_vectorizer.transform([query])
    query_bow = bow_vectorizer.transform([query])
    
    # Calculate similarities
    tfidf_similarities = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    bow_similarities = cosine_similarity(query_bow, bow_matrix)[0]
    
    # Get top indices
    tfidf_top_indices = np.argsort(tfidf_similarities)[::-1][:top_k]
    bow_top_indices = np.argsort(bow_similarities)[::-1][:top_k]
    
    print(f"\n📊 Single Word Search Results for '{query}':")
    print("📈 TF-IDF Results (emphasizes distinctive words):")
    for i, idx in enumerate(tfidf_top_indices, 1):
        if tfidf_similarities[idx] > 0:
            chunk = chunks[idx]
            print(f"  {i}. Chunk {chunk['chunk_id']:3d} | Similarity: {tfidf_similarities[idx]:.4f}")
            print(f"     Preview: {chunk['text'][:100]}...")
            break
    
    print("📦 Bag-of-Words Results (counts word frequency):")
    for i, idx in enumerate(bow_top_indices, 1):
        if bow_similarities[idx] > 0:
            chunk = chunks[idx]
            print(f"  {i}. Chunk {chunk['chunk_id']:3d} | Similarity: {bow_similarities[idx]:.4f}")
            print(f"     Preview: {chunk['text'][:100]}...")
            break

def search_phrase(query, chunks, vectorizer, matrix, feature_names, gram_type, top_k):
    """Search for phrase using n-gram Bag-of-Words"""
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, matrix)[0]
    
    # Get top indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n📊 {gram_type.title()} Search Results for '{query}':")
    
    found_results = False
    for i, idx in enumerate(top_indices, 1):
        if similarities[idx] > 0:
            found_results = True
            chunk = chunks[idx]
            print(f"  {i}. Chunk {chunk['chunk_id']:3d} | Similarity: {similarities[idx]:.4f}")
            print(f"     Preview: {chunk['text'][:150]}...")
            print()
    
    if not found_results:
        print("  No matches found")

def interactive_search(chunks, tfidf_vectorizer, bow_vectorizer, bigram_vectorizer, 
                     trigram_vectorizer, tfidf_matrix, bow_matrix, bigram_matrix, 
                     trigram_matrix, feature_names, bigram_names, trigram_names):
    """Interactive search interface with all methods"""
    print("\n🎯 Advanced Search Interface (Words + Phrases)")
    print("=" * 60)
    print("Search for single words or phrases using cosine similarity!")
    print("Supported formats:")
    print("• Single words: 'neural', 'language', 'algorithm'")
    print("• 2-word phrases: 'machine learning', 'natural language'")
    print("• 3-word phrases: 'speech recognition', 'deep learning'")
    print("• Longer queries: will try to match phrase parts")
    print("Type 'quit' to exit or 'help' for tips.")
    print("=" * 60)
    
    while True:
        query = input("\n🔍 Enter word or phrase to search: ").strip()
        
        if query.lower() == 'quit':
            print("👋 Exiting search mode...")
            break
            
        if query.lower() == 'help':
            print("\n💡 Search Tips:")
            print("• Single words: Try 'neural', 'network', 'algorithm'")
            print("• 2-word phrases: Try 'machine learning', 'natural language', 'speech recognition'")
            print("• 3-word phrases: Try 'deep learning', 'hidden markov', 'statistical methods'")
            print("• Longer queries: 'neural network architecture' will search for matching parts")
            print("• The system automatically detects single words vs phrases")
            continue
            
        if not query:
            print("⚠️  Please enter a word or phrase to search.")
            continue
            
        # Search using all methods
        search_with_all_methods(query, chunks, tfidf_vectorizer, bow_vectorizer, 
                              bigram_vectorizer, trigram_vectorizer, tfidf_matrix, 
                              bow_matrix, bigram_matrix, trigram_matrix,
                              feature_names, bigram_names, trigram_names, top_k=5)

def main():
    """Main analysis function"""
    print("🎯 Advanced NLP Search System (Words + Phrases)")
    print("=" * 80)
    
    # Run analysis
    (chunk_analysis, chunks, tfidf_vectorizer, bow_vectorizer, bigram_vectorizer, 
     trigram_vectorizer, tfidf_matrix, bow_matrix, bigram_matrix, trigram_matrix,
     feature_names, bigram_names, trigram_names) = analyze_chunk_tfidf()
    
    # Save results
    save_analysis_results(chunk_analysis)
    
    # Print top chunks
    print_top_chunks_by_tfidf(chunk_analysis)
    
    # Find most distinctive terms
    find_most_distinctive_terms_across_chunks(chunk_analysis)
    
    print("\n🎉 Analysis Complete!")
    print("\n🚀 Starting interactive search...")
    print("Now you can search for both single words AND phrases!")
    
    # Start interactive search
    interactive_search(chunks, tfidf_vectorizer, bow_vectorizer, bigram_vectorizer, 
                     trigram_vectorizer, tfidf_matrix, bow_matrix, bigram_matrix, 
                     trigram_matrix, feature_names, bigram_names, trigram_names)

# Run the analysis
if __name__ == "__main__":
    (chunk_analysis, chunks, tfidf_vectorizer, bow_vectorizer, bigram_vectorizer, 
     trigram_vectorizer, tfidf_matrix, bow_matrix, bigram_matrix, trigram_matrix,
     feature_names, bigram_names, trigram_names) = main()
