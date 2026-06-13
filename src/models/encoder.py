import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Dict, Any, Optional
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Handle mask properly - convert boolean to float and expand dimensions
        if mask is not None:
            # Convert boolean mask to float mask (True -> 0, False -> -1e9)
            mask_float = mask.float()
            # Expand mask to match attention scores shape
            mask_expanded = mask_float.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            mask_expanded = mask_expanded.expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        self.attention = F.softmax(scores, dim=-1)
        self.attention = self.dropout(self.attention)
        
        x = torch.matmul(self.attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(x)

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class EncoderOnlyTransformer(nn.Module):
    """Encoder-only transformer for text encoding"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.d_ff = config.get('d_ff', self.d_model * 4)
        self.max_len = config.get('max_len', 512)
        self.dropout = config.get('dropout', 0.1)
        self.pad_token_id = config.get('pad_token_id', 0)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = PositionalEncoding(self.d_model, self.max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Output projection and pooling
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.pooling = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Device tracking
        self._device = None
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Special initialization for positional encoding
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.d_model ** -0.5)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided (boolean mask: True for padding, False for real tokens)
        if attention_mask is None:
            attention_mask = (input_ids == self.pad_token_id)
        
        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Add positional encoding
        position_embeddings = self.position_embedding(token_embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        # Output projection
        x = self.output_projection(x)
        
        # Pooling - use [CLS] token if available, otherwise mean pooling
        if input_ids.shape[1] > 0:
            cls_output = x[:, 0, :]  # [CLS] token
        else:
            cls_output = x.mean(dim=1)  # Mean pooling as fallback
        
        # Final pooling
        pooled_output = self.pooling(cls_output)
        
        return pooled_output
    
    def encode(self, texts: List[str], tokenizer) -> np.ndarray:
        """Encode a list of texts into embeddings"""
        self.eval()
        
        with torch.no_grad():
            # Tokenize texts
            encoded = tokenizer(texts, padding=True, truncation=True, 
                              max_length=self.max_len, return_tensors='pt')
            
            # Get device
            device = self.get_device()
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Convert attention mask to boolean format (True for padding, False for real tokens)
            # This is what our attention mechanism expects
            attention_mask_bool = (attention_mask == 0)  # Invert: 0 means padding
            
            # Get embeddings
            embeddings = self.forward(input_ids, attention_mask_bool)
            
            # Move to CPU and convert to numpy
            return embeddings.cpu().numpy()
    def search(self, query: str, passages: List[str], passage_embeddings: np.ndarray, tokenizer, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for a query against a list of passages.
        
        Args:
            query: The search query string.
            passages: List of passage texts.
            passage_embeddings: Pre-computed embeddings for all passages.
            tokenizer: The tokenizer used to encode the query.
            top_k: Number of top results to return.
            
        Returns:
            List of dictionaries with 'index', 'score', and 'text'.
        """
        # Encode query
        query_emb = self.encode([query], tokenizer)
        
        # Calculate similarities
        # cosine_similarity returns a 2D array (1, N). We take the first row.
        sim_matrix = cosine_similarity(query_emb, passage_embeddings)
        similarities = sim_matrix
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'score': float(similarities[idx]),
                'text': passages[idx]
            })
            
        return results
    def get_device(self) -> torch.device:
        """Get the current device the model is on"""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    def to_device(self, device: torch.device):
        """Move model to specified device"""
        self._device = device
        self.to(device)
    
    def save(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.__dict__
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'EncoderOnlyTransformer':
        """Load model from saved state"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Reconstruct model
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device:
            model.to_device(device)
        
        print(f"Model loaded from {path}")
        return model


class SimpleTokenizer:
    """Simple tokenizer for encoder training"""
    
    def __init__(self, vocab_size: int = 20000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.cls_token = '<CLS>'
        
        # Initialize special tokens
        special_tokens = [self.pad_token, self.unk_token, self.cls_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        self.next_id = len(special_tokens)
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_freq = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        
        # Sort by frequency and keep most common
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add top words to vocabulary
        for word, freq in sorted_words:
            if self.next_id < self.vocab_size:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
            else:
                break
    
    def tokenize(self, text: str, max_length: int = 512) -> List[int]:
        """Tokenize a single text"""
        words = text.lower().split()
        
        # Add CLS token
        tokens = [self.word_to_id.get(self.cls_token, 1)]
        
        # Add word tokens
        for word in words[:max_length-1]:  # Leave room for CLS
            tokens.append(self.word_to_id.get(word, self.word_to_id.get(self.unk_token, 1)))
            
            if len(tokens) >= max_length:
                break
        
        # Pad if necessary
        while len(tokens) < max_length:
            tokens.append(self.word_to_id.get(self.pad_token, 0))
        
        return tokens
    
    def __call__(self, texts: List[str], padding: bool = True, 
                 truncation: bool = True, max_length: int = 512,
                 return_tensors: str = 'pt') -> Dict[str, torch.Tensor]:
        """Tokenize multiple texts (compatible with HuggingFace interface)"""
        
        encoded_texts = []
        attention_masks = []
        
        for text in texts:
            tokens = self.tokenize(text, max_length)
            encoded_texts.append(tokens)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1 if token != self.word_to_id.get(self.pad_token, 0) else 0 
                   for token in tokens]
            attention_masks.append(mask)
        
        # Pad to maximum length
        if padding and len(encoded_texts) > 1:
            max_len = max(len(tokens) for tokens in encoded_texts)
            for tokens, mask in zip(encoded_texts, attention_masks):
                while len(tokens) < max_len:
                    tokens.append(self.word_to_id.get(self.pad_token, 0))
                    mask.append(0)
        
        # Convert to tensors
        if return_tensors == 'pt':
            input_ids = torch.tensor(encoded_texts, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            return {
                'input_ids': encoded_texts,
                'attention_mask': attention_masks
            }

# Training utilities
def contrastive_loss(embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                    labels: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Compute contrastive loss for training"""
    
    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)
    
    # Compute cosine similarity
    similarities = torch.matmul(embeddings1, embeddings2.T) / temperature
    
    # InfoNCE loss
    batch_size = embeddings1.shape[0]
    labels = torch.arange(batch_size).to(embeddings1.device)
    
    loss = F.cross_entropy(similarities, labels)
    
    return loss

def train_encoder(model: EncoderOnlyTransformer, 
                 training_data: List[Dict[str, Any]], 
                 config: Dict[str, Any],
                 device: torch.device = None) -> EncoderOnlyTransformer:
    """Train the encoder model"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to_device(device)
    model.train()
    
    # Training configuration
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    temperature = config.get('temperature', 0.05)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # For additional MSE loss
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(model.vocab_size)
    all_texts = [item['passage_text'] for item in training_data]
    tokenizer.fit(all_texts)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        batches = []
        
        # Create batches
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            batches.append(batch)
        
        random.shuffle(batches)
        
        for batch_idx, batch in enumerate(batches):
            # Extract passages
            passages = [item['passage_text'] for item in batch]
            
            # Tokenize and encode
            encoded = tokenizer(passages, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Convert attention mask to boolean format for training
            attention_mask_bool = (attention_mask == 0)  # True for padding, False for real tokens
            
            # Forward pass
            embeddings = model.forward(input_ids, attention_mask_bool)
            
            # Contrastive loss (with itself for positive pairs)
            batch_size_current = embeddings.shape[0]
            if batch_size_current > 1:
                # Create positive pairs by shuffling
                shuffled_indices = torch.randperm(batch_size_current)
                embeddings_shuffled = embeddings[shuffled_indices]
                
                contr_loss = contrastive_loss(embeddings, embeddings_shuffled, 
                                            torch.arange(batch_size_current).to(device), 
                                            temperature)
                
                # MSE loss between embeddings and their shuffled versions
                mse_loss = criterion(embeddings, embeddings_shuffled)
                
                # Combined loss
                loss = contr_loss + 0.1 * mse_loss
            else:
                # For single batch items, use simple reconstruction loss
                loss = criterion(embeddings, torch.zeros_like(embeddings))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(batches)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    model.eval()
    return model

# Example usage and testing
def test_encoder():
    """Test the encoder implementation"""
    
    # Configuration
    config = {
        'vocab_size': 10000,
        'd_model': 384,      # Smaller for faster training
        'n_heads': 8,        # Fewer heads
        'n_layers': 4,       # Fewer layers
        'd_ff': 1536,        # 4 * d_model
        'max_len': 256,      # Shorter sequences
        'dropout': 0.1,
        'pad_token_id': 0
    }
    
    # Create model
    model = EncoderOnlyTransformer(config)
    
    # Test data
    test_texts = [
        "The Markov assumption is fundamental in statistical language modeling",
        "Hidden Markov models are used for sequence prediction tasks",
        "Natural language processing involves understanding human language"
    ]
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(config['vocab_size'])
    tokenizer.fit(test_texts)
    
    # Test encoding
    print("Testing encoder...")
    embeddings = model.encode(test_texts, tokenizer)
    
    print(f"Input texts: {len(test_texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    print("\nSimilarity matrix:")
    print(similarity_matrix)
    
    return model, tokenizer

if __name__ == "__main__":
    # Run test
    model, tokenizer = test_encoder()
    
    # Save model
    model.save("encoder_transformer.pth")
    
    # Load model back
    loaded_model = EncoderOnlyTransformer.load("encoder_transformer.pth")
