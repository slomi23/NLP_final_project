# Neural Search Engine for NLP Textbook Retrieval

## 1. Introduction

Search is an important part of working with large technical documents. In an NLP course, students often need to find explanations of concepts such as attention, language models, parsing, embeddings, or evaluation metrics inside a long textbook. Simple keyword search can work well when the query uses exactly the same words as the relevant passage, but it can fail when the user describes the idea differently from the textbook.

Traditional retrieval methods such as TF-IDF and BM25 are strong baselines for this kind of task. They are efficient, simple to implement, and often work very well on technical text because important terms are repeated directly in the document. However, they mostly depend on lexical overlap. A neural search model tries to solve this limitation by mapping both queries and passages into dense vector embeddings, so that passages can be retrieved based on meaning and not only on exact word matches.

The goal of this project was to build a neural search engine over the Jurafsky and Martin *Speech and Language Processing* textbook. The project focuses on building a real retrieval pipeline instead of only a demo. It includes PDF preprocessing, chunking, model training, indexing, searching, baseline comparison, and evaluation.

## 2. Problem Statement and Goal

The task is: given a user query, retrieve the most relevant chunks from the Jurafsky and Martin *Speech and Language Processing* textbook. The book is divided into smaller passages of approximately 200–300 words, and the search engine ranks these passages for each query.

The main goal is not only to make a notebook that returns some results, but to train and evaluate a real search pipeline using real data. The project compares three retrieval approaches:

1. TF-IDF baseline
2. BM25 baseline
3. Custom neural encoder trained from scratch

The success criteria are:

- the final search corpus must be the real Jurafsky textbook chunks, not demo data;
- the neural encoder should produce meaningful embeddings;
- the neural system should be compared fairly against TF-IDF and BM25;
- evaluation should use retrieval metrics such as Precision@5, Recall@5, and MRR@5;
- the report should interpret the results honestly, even if the neural model does not outperform the baselines.

## 3. Data

The project uses different data sources for different purposes. A major design decision is that the final search corpus is only the Jurafsky textbook. External datasets are used for training the neural encoder, but they are not used as the final retrieval corpus.

### 3.1 Jurafsky and Martin Textbook

The final retrieval corpus is the Jurafsky and Martin *Speech and Language Processing* PDF:

```text
data/raw/Speech_and_Language_Processing.pdf
```

This PDF is the target collection that the search engine searches over. It is converted into cleaned chunks, and all final retrieval methods search through these chunks.

### 3.2 MS MARCO Small Data

The project also uses MS MARCO-style retrieval data for training. The processed files are:

```text
data/processed/msmarco_small_queries.json
data/processed/msmarco_small_passages.json
data/processed/msmarco_small_qrels.json
```

At one point, these files were checked and contained:

- queries: 1000
- passages: 5000
- qrels: 1000

This data is useful because it already has query-passage relevance structure. It can be used to form positive query-passage pairs and negative examples for contrastive training.

### 3.3 arXiv CS Papers CSV

An additional data source was an arXiv CS papers CSV:

```text
data/processed/arxiv_cs_papers_processed.csv
```


The goal of using this file was to add more computer science text during training.


### 3.4 Training Data vs Search Corpus

It is important to separate training data from the final search corpus.

MS MARCO and arXiv data are used to train or improve the neural encoder. They help the model learn text representations and query-passage matching. The Jurafsky textbook chunks are the final corpus that the system searches over and evaluates on.

This separation matters because the project is not a general web search engine. It is specifically a search engine for NLP textbook material. Therefore, the final comparison between TF-IDF, BM25, and the neural model should use the same cleaned Jurafsky chunks.

## 4. Preprocessing

Preprocessing was one of the most important parts of the project. The raw Jurafsky PDF could not be used directly as a search corpus because PDF extraction introduced a lot of noise. The extracted text included page numbers, table of contents fragments, index entries, bibliography sections, appendix tables, acknowledgements, and chunks full of dot leaders such as:

```text
. . . . . . . . 401
```

These bad chunks caused poor retrieval behavior. For example, a query such as “What is attention mechanism?” could return a table-of-contents chunk instead of an actual explanation from the textbook.

To fix this, `book_chunks.py` was rewritten to clean the extracted textbook text before chunking. The improved chunker filters out non-prose material such as:

- isolated page numbers;
- repeated headers and footers;
- table-of-contents dot leader lines;
- chunks with too many dots;
- chunks with too many numbers;
- index-like chunks with many commas;
- appendix or tagset-like chunks with many uppercase tags;
- bibliography, references, acknowledgements, index, and appendix sections.

The cleaned text is then divided into chunks of approximately 180–300 words, with a target chunk size of around 250 words. The output file is:

```text
data/jurafsky_chunks/chunks.jsonl
```

This preprocessing step is essential because retrieval quality depends strongly on the quality of the corpus. If the corpus contains noisy PDF artifacts, even a good retrieval method can return bad results.

### Tokenization and Vocabulary

The neural model uses a custom tokenizer and vocabulary. Since the project requirement is to train a model from scratch, the system does not use a pretrained tokenizer such as BERT’s tokenizer. The vocabulary must be built from training data instead of being loaded from a pretrained model.

The tokenizer converts text into token IDs, and these IDs are passed into the encoder. It is important that the same tokenizer and vocabulary are used consistently during training, indexing, and search. If the tokenizer changes between training and retrieval, the embeddings become unreliable.

Several checks were added during development to make sure the project used real files instead of demo chunks or fallback toy data. This was necessary because the notebook originally appeared to work, but some parts were silently using demo chunks or fallback synthetic examples.

## 5. Baselines

The project implements two classical retrieval baselines: TF-IDF and BM25. These baselines are important because they provide strong reference points for the neural model.

### 5.1 TF-IDF

TF-IDF ranks chunks using weighted word overlap. Words that appear often in a specific chunk but not too often across the whole corpus receive higher importance. This makes TF-IDF a simple but useful baseline for document retrieval.

### 5.2 BM25

BM25 is another lexical retrieval method and is usually stronger than plain TF-IDF. It uses term frequency, inverse document frequency, and document length normalization. BM25 is a standard baseline in search and information retrieval.

### 5.3 Fair Evaluation

The baselines should be evaluated on the same cleaned Jurafsky chunks and the same queries as the neural model. The evaluation should not use pseudo-evaluation that gives TF-IDF or BM25 an artificial advantage.

## 6. Neural Model

The neural model is a custom encoder-only Transformer trained from scratch. It does not use pretrained models such as BERT, DistilBERT, SentenceTransformers, or OpenAI embeddings.

The encoder maps both queries and textbook chunks into dense vectors. During search, the query is encoded into a vector, each chunk is also represented as a vector, and chunks are ranked using cosine similarity.

The main components of the neural model are:

- custom tokenizer and vocabulary;
- token embedding layer;
- positional encoding;
- Transformer encoder layers;
- pooling step to create one fixed-size embedding for a query or passage;
- cosine similarity search over chunk embeddings.

The project used a custom `EncoderOnlyTransformer` implementation in:

```text
src/models/encoder.py
```

A meaningful training configuration included:

| Component | Value |
|---|---:|
| Vocabulary size | 50,000 |
| Embedding size / `d_model` | 128 |
| Number of attention heads | [fill in value here] |
| Number of Transformer layers | [fill in value here] |
| Feed-forward size | [fill in value here] |
| Maximum sequence length | [fill in value here] |
| Dropout | [fill in value here] |
| Optimizer | [fill in optimizer here] |
| Learning rate | [fill in learning rate here] |
| Batch size | [fill in batch size here] |
| Epochs | 3 for the meaningful run described below |
| Device | CUDA GPU |

Earlier development versions used smaller configurations such as `d_model=64`, `n_heads=2`, `n_layers=2`, `d_ff=256`, `max_len=128`, and `dropout=0.5`. Later the model was increased to around `d_model=128`, with more heads and layers, while still keeping it small enough to train in Colab.

A pretrained DistilBERT-based approach was also tried as a comparison/reference. It performed much better, which is expected because DistilBERT already contains language knowledge learned from large-scale pretraining. However, this project’s main requirement was to train the encoder from scratch, so the final system and main analysis focus on the custom model rather than the pretrained one.

## 7. Training Objective

The neural encoder was trained using contrastive learning with InfoNCE loss. The model receives a query, one positive passage, and several negative passages. The positive passage is relevant to the query, while the negative passages are not relevant.

The training objective encourages the query embedding to be close to the positive passage embedding and farther away from the negative passage embeddings. This teaches the model to produce embeddings that are useful for retrieval.

Each training example contains:

- a query;
- one positive passage or chunk;
- several negative passages or chunks.

In the meaningful training setup, the model used one positive and four negative passages. With five candidates total, the random InfoNCE loss is approximately:

```text
log(5) = 1.609
```

The validation loss decreased below this random baseline, which suggests that the model learned to distinguish positive passages from negatives.

However, loss decreasing is not enough to prove that the search engine works well. The model must also be evaluated with retrieval metrics on the same search task as TF-IDF and BM25. A model can show lower training loss but still fail to retrieve the best textbook chunks if the embeddings are not good enough or if the evaluation setup is different from the training setup.

## 8. Evaluation Metrics

The project uses standard retrieval metrics.

### Precision@5

Precision@5 measures how many of the top 5 retrieved chunks are relevant. If there is only one known relevant chunk per query, and the system retrieves it in the top 5, then Precision@5 is:

```text
1 / 5 = 0.2
```

This means Precision@5 may look low even when the system found the correct chunk, because only one chunk is labeled as relevant.

### Recall@5

Recall@5 measures whether the system found the relevant chunk in the top 5 results. If there is one known relevant chunk and it appears in the top 5, Recall@5 is 1. If it does not appear, Recall@5 is 0.

### MRR@5

MRR@5, or Mean Reciprocal Rank at 5, measures how high the first relevant result appears. If the relevant chunk is ranked first, the reciprocal rank is 1. If it is ranked second, the reciprocal rank is 1/2. If it is ranked fifth, the reciprocal rank is 1/5.

These metrics are appropriate because a search engine should not only find relevant chunks, but should rank them near the top.

The project can also report Precision@10, Recall@10, and MRR@10.

## 9. Results

### 9.1 Training Results

A meaningful neural training run used approximately:

| Item | Value |
|---|---:|
| Training triplets | 170,482 |
| Validation triplets | 8,973 |
| Vocabulary size | 50,000 |
| Embedding size | 128 |
| Device | CUDA GPU |
| Training time | about 54 minutes |
| Epochs | 3 |

The loss values were:

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 1.5765 | 1.5182 |
| 2 | 1.5166 | 1.4973 |
| 3 | 1.4759 | 1.4133 |

Since the random loss for one positive plus four negatives is approximately 1.609, the decrease in validation loss to 1.4133 suggests that the encoder learned some useful distinction between positive and negative passages.

### 9.2 Retrieval Results

The final retrieval results should compare TF-IDF, BM25, and the neural encoder on the same cleaned Jurafsky chunks and the same evaluation queries.

| Method | Precision@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| TF-IDF | [insert Precision@5 here] | [insert Recall@5 here] | [insert MRR@5 here] |
| BM25 | [insert Precision@5 here] | [insert Recall@5 here] | [insert MRR@5 here] |
| Neural Encoder | [insert Precision@5 here] | [insert Recall@5 here] | [insert MRR@5 here] |
| DistilBERT reference attempt | [insert Precision@5 here, optional] | [insert Recall@5 here, optional] | [insert MRR@5 here, optional] |

Optional @10 metrics:

| Method | Precision@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| TF-IDF | [insert Precision@10 here] | [insert Recall@10 here] | [0.3387] |
| BM25 | [insert Precision@10 here] | [insert Recall@10 here] | [insert MRR@10 here] |
| Neural Encoder | [insert Precision@10 here] | [insert Recall@10 here] | [insert MRR@10 here] |



If the final neural model performs worse than TF-IDF or BM25, that does not necessarily mean the project failed. It means that the lexical baselines are very strong on this evaluation setup and that the neural model trained from scratch still needs better training, better negatives, or a harder semantic evaluation set.

## 10. Debugging and Important Issues

Several important implementation issues were found and fixed during the project. These issues are important because they show why building a real retrieval pipeline is different from building a small demo.


### 10.1 Training Was Too Fast

At one stage, training finished too quickly, which suggested that only a tiny dataset or demo data was being used. This was another sign that the notebook needed stronger checks for real data loading.

A later meaningful run used about 170,482 training triplets and took about 54 minutes for 3 epochs on CUDA, which is much more realistic.

### 10.2 Embedding Collapse

A serious debugging issue happened when embeddings for different sentences produced cosine similarity 1.0 for everything. For example, unrelated texts such as “machine learning algorithms,” “natural language processing,” “the cat sat on the mat,” and “quantum mechanics” all had cosine similarity 1.0 with each other.

This indicated embedding collapse or a broken encoder/evaluation pipeline. In a correct retrieval model, unrelated texts should not all produce identical or nearly identical embeddings. This issue showed that checking only whether code runs is not enough. The embeddings themselves must also be inspected.

### 10.3 Transformer and Retrieval Fixes

Several implementation fixes were made:

- fixed the attention mask direction in the Transformer encoder;
- fixed positional encoding being added incorrectly;
- fixed cosine similarity search indexing so the correct query similarity row is selected;
- fixed model save/load configuration;
- fixed InfoNCE loss implementation;
- improved tokenizer saving and loading;
- ensured final search uses only Jurafsky chunks;
- ensured training may use MS MARCO and arXiv data, while final retrieval remains Jurafsky-only.

### 10.4 PDF Cleaning

Raw PDF extraction produced many bad chunks, including page numbers, index sections, table of contents fragments, acknowledgements, bibliography pages, and appendix tables. These noisy chunks made retrieval results worse.

The chunking pipeline was improved so that non-prose chunks are filtered out before indexing. This became one of the most important lessons of the project: for retrieval systems, corpus quality is just as important as the retrieval model.


## 11. Discussion

The results show that building a neural search engine from scratch is challenging. The neural encoder did learn during training, as shown by the validation loss decreasing from 1.5182 to 1.4133. Since this is below the random InfoNCE loss of approximately 1.609, the model learned to distinguish positive passages from negative passages to some extent.

However, retrieval performance must be judged using search metrics, not only training loss. TF-IDF and BM25 are difficult baselines to beat on a textbook search task. Technical textbooks use specific terms, and student queries often contain the same terms. In this situation, lexical overlap is already a very strong signal.

The pretrained DistilBERT attempt performed much better, which is not surprising. DistilBERT has already learned language representations from large-scale pretraining, while this project’s custom encoder starts from random weights. This comparison shows why pretraining is so powerful, but it also reinforces the main challenge of the project: learning useful retrieval embeddings from scratch with limited data and compute.

The baseline results also show a limitation of automatic evaluation. If generated queries repeat important words from the positive chunk, then TF-IDF and BM25 can retrieve the correct chunk very easily. This does not fully test whether a system understands semantic similarity. For example, a harder query might describe the concept in different words than the textbook uses.

Overall, the project successfully built the main components of a retrieval system: data loading, PDF chunking, corpus cleaning, model training, indexing, search, baselines, and evaluation. Even if the neural model does not outperform TF-IDF or BM25, the project is still useful because it exposes the practical problems involved in neural retrieval.

## 12. Limitations

The project has several limitations.

First, no pretrained model was allowed for the main system. This makes the task much harder because the encoder must learn useful language representations from scratch.

Second, training from scratch requires a lot of data and compute. The model used a relatively small Transformer compared with modern retrieval models, so its semantic ability is limited.

Third, PDF extraction and chunking can introduce noise. Even after cleaning, some chunks may still contain formatting artifacts or may split explanations in unnatural places.

Fourth, the neural model can suffer from embedding collapse or other training problems if the loss, masking, pooling, or similarity calculation is implemented incorrectly.

Finally, the results depend strongly on negative sampling. If negative passages are too easy, the model may learn only shallow distinctions. If negatives are too hard too early, training may become unstable. Better negative sampling is important for improving retrieval quality.

## 13. Future Work

Several improvements could make the system stronger.

One improvement is better negative sampling. Hard negatives, such as passages that share words with the query but are not actually relevant, would force the model to learn more meaningful distinctions.

Another improvement is longer training with a larger model. A larger Transformer, more epochs, and better hyperparameter tuning could improve the learned embeddings, although this would require more compute.

The tokenizer could also be improved. A better subword tokenizer could handle rare technical terms more effectively.

The training data could be made more domain-specific. More NLP or computer science text, including real arXiv CS data and textbook-style query-passage pairs, could reduce the mismatch between training data and the Jurafsky search corpus.

The evaluation could be improved with manually written queries and human relevance judgments. This would make the benchmark less biased toward lexical overlap and would better test semantic retrieval.

A hybrid system would also be useful. TF-IDF or BM25 could retrieve an initial set of candidate chunks, and the neural model could rerank the top results. This is often more practical than using only lexical search or only neural search.

## 14. Conclusion

This project built a neural search engine for the Jurafsky and Martin *Speech and Language Processing* textbook. The system preprocesses the textbook PDF, creates cleaned 180–300 word chunks, trains a custom encoder-only Transformer from scratch, builds an embedding index, and retrieves relevant chunks using cosine similarity.

The project also implements TF-IDF and BM25 baselines and compares all methods using retrieval metrics such as Precision@5, Recall@5, and MRR@5. The neural model showed learning during training because validation loss decreased below the random InfoNCE baseline. However, the lexical baselines were extremely strong on the automatic evaluation split, probably because the generated queries had high word overlap with the target chunks.

The project demonstrates that neural retrieval is not only about writing a model. It also requires careful preprocessing, real data loading, correct training objectives, reliable indexing, fair baselines, and meaningful evaluation. Even if the custom neural encoder does not outperform BM25 or TF-IDF, the project is valuable because it builds a full retrieval pipeline and shows the practical challenges of training a neural search model from scratch.

## Missing Information to Fill Before Submission

Before submitting the final report, the following information should be filled in:

1. Final TF-IDF Precision@5, Recall@5, and MRR@5.
2. Final BM25 Precision@5, Recall@5, and MRR@5.
3. Final neural encoder Precision@5, Recall@5, and MRR@5.
4. Optional Precision@10, Recall@10, and MRR@10 for all methods.
5. Exact final model hyperparameters: number of heads, number of layers, feed-forward size, maximum sequence length, dropout, batch size, optimizer, and learning rate.
6. Whether the final arXiv CSV was successfully restored from Git LFS or regenerated.
7. Exact number of cleaned Jurafsky chunks in `data/jurafsky_chunks/chunks.jsonl`.
8. A few example queries and top retrieved chunks for qualitative analysis.
9. Exact DistilBERT reference results, if they will be mentioned in the final comparison table.
10. Any plots that will be inserted, such as training/validation loss or baseline vs neural retrieval metrics.
