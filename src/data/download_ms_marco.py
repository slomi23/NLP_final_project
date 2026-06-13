import os
import urllib.request
import tarfile
import json
import random
from pathlib import Path

def ensure_data_directory():
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir

def create_synthetic_ms_marco_subset(msmarco_dir=None, output_size_mb=100):
    print(f"Creating larger synthetic MS MARCO subset...")
    processed_dir = ensure_data_directory()
    synthetic_passages = []
    synthetic_queries = []
    synthetic_qrels = {}
    topics = [
        "machine learning", "natural language processing", "computer vision",
        "deep learning", "neural networks", "artificial intelligence",
        "data science", "python programming", "algorithm design",
        "statistical analysis", "web development", "database systems",
        "cloud computing", "cybersecurity", "blockchain technology",
        "quantum computing", "robotics", "bioinformatics", "digital signal processing"
    ]
    num_passages = 10000
    print(f"  Generating {num_passages} passages...")
    for i in range(num_passages):
        topic = random.choice(topics)
        passage = f"""
        {topic.upper()} is a fundamental concept in modern computer science that has revolutionized how we approach complex problems. 
        The field encompasses various methodologies and techniques that enable machines to process information and make decisions 
        based on data rather than explicit programming. Recent advances in computational power and algorithmic efficiency 
        have made it possible to implement sophisticated {topic} systems that can handle large-scale data processing tasks.
        
        Key applications of {topic} span across multiple domains including healthcare diagnostics, financial modeling, 
        autonomous systems, and user experience optimization. The core principles involve understanding patterns in data, 
        developing predictive models, and continuously improving system performance through iterative training processes.
        
        Researchers continue to explore new architectures and optimization techniques that push the boundaries of what's 
        computationally feasible while maintaining accuracy and reliability in real-world scenarios.
        """.strip()
        
        synthetic_passages.append({
            'id': f'passage_{i:06d}',
            'text': passage,
            'topic': topic
        })
    num_queries = 20000
    print(f"  Generating {num_queries} queries...")
    for i in range(num_queries):
        topic = random.choice(topics)
        query_types = [
            f"What is {topic} and how does it work?",
            f"Applications of {topic} in modern systems",
            f"Latest advances in {topic} research",
            f"Best practices for implementing {topic} solutions",
            f"Challenges in {topic} development and deployment",
            f"How to learn {topic} effectively?",
            f"{topic} vs traditional methods: a comparison",
            f"The future of {topic} in industry",
            f"Key concepts in {topic} explained",
            f"Top tools for {topic} development"
        ]
        query = random.choice(query_types)
        
        synthetic_queries.append({
            'id': f'query_{i:06d}',
            'text': query,
            'topic': topic
        })
    print(f"  Generating relevance judgments...")
    for query in synthetic_queries:
        query_id = query['id']
        topic = query['topic']
        relevant_passages = [
            p for p in synthetic_passages 
            if p['topic'] == topic
        ]
        num_relevant = random.randint(1, min(3, len(relevant_passages)))
        selected_relevant = random.sample(relevant_passages, num_relevant)
        
        synthetic_qrels[query_id] = [
            (passage['id'], random.randint(1, 3))
            for passage in selected_relevant
        ]
    passages_file = processed_dir / "msmarco_small_passages.json"
    queries_file = processed_dir / "msmarco_small_queries.json"
    qrels_file = processed_dir / "msmarco_small_qrels.json"
    
    with open(passages_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_passages, f, ensure_ascii=False, indent=2)
    
    with open(queries_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_queries, f, ensure_ascii=False, indent=2)
    
    with open(qrels_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_qrels, f, indent=2)

    total_size = (
        passages_file.stat().st_size +
        queries_file.stat().st_size +
        qrels_file.stat().st_size
    ) / (1024 * 1024)
    
    print(f"Created synthetic dataset: {total_size:.1f}MB")
    print(f"- {len(synthetic_passages)} passages")
    print(f"- {len(synthetic_queries)} queries")
    print(f"- {len(synthetic_qrels)} query-passage relevance pairs")
    print(f"Files saved to: {processed_dir}")
    
    return {
        'passages': synthetic_passages,
        'queries': synthetic_queries,
        'qrels': synthetic_qrels,
        'files': {
            'passages': str(passages_file),
            'queries': str(queries_file),
            'qrels': str(qrels_file)
        }
    }

def download_real_ms_marco_subset():
    print("Downloading real MS MARCO subset...")

    processed_dir = ensure_data_directory()
    real_data_dir = Path("data/msmarco_real_data")
    real_data_dir.mkdir(parents=True, exist_ok=True)

    queries_url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.dev.tsv"
    qrels_url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv"

    queries_file = real_data_dir / "queries.dev.tsv"
    if not queries_file.exists():
        print("Downloading development queries...")
        urllib.request.urlretrieve(queries_url, queries_file)

    qrels_file = real_data_dir / "qrels.dev.small.tsv"
    if not qrels_file.exists():
        print("Downloading relevance judgments...")
        urllib.request.urlretrieve(qrels_url, qrels_file)
    print("Processing and limiting to 5K queries...")
    
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            query_id, query = line.strip().split('\t', 1)
            queries[query_id] = query

    qrels = {}
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id in queries:
                if query_id not in qrels:
                    qrels[query_id] = []
                qrels[query_id].append((doc_id, int(relevance)))

    real_queries_file = processed_dir / "msmarco_real_queries.json"
    real_qrels_file = processed_dir / "msmarco_real_qrels.json"
    
    with open(real_queries_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    
    with open(real_qrels_file, 'w', encoding='utf-8') as f:
        json.dump(qrels, f, indent=2)

    print(f"Real subset created:")
    print(f"- {len(queries)} queries")
    print(f"- {len(qrels)} queries with relevance judgments")
    print(f"Files saved to: {processed_dir}")
    
    return {
        'queries': queries,
        'qrels': qrels,
        'files': {
            'queries': str(real_queries_file),
            'qrels': str(real_qrels_file)
        }
    }

def main():
    print("=== Small MS MARCO Dataset Downloader ===")
    print("This script creates a dataset perfect for NLP projects")
    print("Files will be saved to: data/processed/")
    print()
    choice = input("Choose dataset type:\n1. Synthetic MS MARCO-like dataset (recommended)\n2. Real MS MARCO subset\n3. Both\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        dataset = create_synthetic_ms_marco_subset()
        
    elif choice == '2':
        dataset = download_real_ms_marco_subset()
        
    elif choice == '3':
        print("\nCreating synthetic dataset...")
        synthetic_data = create_synthetic_ms_marco_subset()
        
        print("\nDownloading real subset...")
        real_data = download_real_ms_marco_subset()
        
        dataset = {
            'synthetic': synthetic_data,
            'real': real_data
        }
        
    else:
        print("Invalid choice. Creating synthetic dataset...")
        dataset = create_synthetic_ms_marco_subset()
    
    print("\nDataset download complete!")
    print("\nFiles created in data/processed/:")

    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for file in processed_dir.glob("*.json"):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"- {file.name}: {size_mb:.1f}MB")
    
    print("\nYou can now use this dataset for your neural search project!")
    print("Example: data/processed/msmarco_small_passages.json")

if __name__ == "__main__":
    main()