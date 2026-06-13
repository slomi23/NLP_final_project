from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.attention = attn

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = dict(config)

        self.vocab_size = int(config["vocab_size"])
        self.d_model = int(config["d_model"])
        self.n_heads = int(config["n_heads"])
        self.n_layers = int(config["n_layers"])
        self.d_ff = int(config.get("d_ff", self.d_model * 4))
        self.max_len = int(config.get("max_len", 512))
        self.dropout = float(config.get("dropout", 0.1))
        self.pad_token_id = int(config.get("pad_token_id", 0))

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.d_model,
            padding_idx=self.pad_token_id,
        )
        self.position_embedding = PositionalEncoding(self.d_model, self.max_len)
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)

        self.pooling = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, self.d_model),
        )

        self._device: Optional[torch.device] = None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model ** -0.5)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

    def _make_padding_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return input_ids.eq(self.pad_token_id)

        if attention_mask.dtype == torch.bool:
            return attention_mask
        return attention_mask.eq(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        padding_mask = self._make_padding_mask(input_ids, attention_mask)

        x = self.token_embedding(input_ids)
        x = self.position_embedding(x)
        x = self.embedding_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, padding_mask)

        x = self.final_norm(x)
        real_token_mask = (~padding_mask).unsqueeze(-1).float()
        lengths = real_token_mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * real_token_mask).sum(dim=1) / lengths

        return self.pooling(pooled)

    def encode(self, texts: List[str], tokenizer) -> np.ndarray:
        self.eval()
        device = self.get_device()

        with torch.no_grad():
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            embeddings = self.forward(input_ids, attention_mask)
            return embeddings.detach().cpu().numpy()

    def search(
        self,
        query: str,
        passages: List[str],
        passage_embeddings: np.ndarray,
        tokenizer,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_emb = self.encode([query], tokenizer)
        similarities = cosine_similarity(query_emb, passage_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "index": int(idx),
                "score": float(similarities[idx]),
                "text": passages[idx],
            }
            for idx in top_indices
        ]

    def get_device(self) -> torch.device:
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def to_device(self, device: torch.device) -> None:
        self._device = device
        self.to(device)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "EncoderOnlyTransformer":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to_device(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 20000):
        self.vocab_size = int(vocab_size)
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.cls_token = "<CLS>"

        self.word_to_id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.cls_token: 2,
        }
        self.id_to_word: Dict[int, str] = {idx: tok for tok, idx in self.word_to_id.items()}
        self.next_id = len(self.word_to_id)

    @staticmethod
    def basic_tokenize(text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text)
        return re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", text.lower())

    def fit(self, texts: List[str]) -> None:
        word_freq: Dict[str, int] = {}

        for text in texts:
            for word in self.basic_tokenize(text):
                word_freq[word] = word_freq.get(word, 0) + 1

        for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if word in self.word_to_id:
                continue
            if self.next_id >= self.vocab_size:
                break
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1

    def tokenize(self, text: str, max_length: int = 512) -> List[int]:
        token_ids = [self.word_to_id[self.cls_token]]

        for word in self.basic_tokenize(text):
            if len(token_ids) >= max_length:
                break
            token_ids.append(self.word_to_id.get(word, self.word_to_id[self.unk_token]))

        return token_ids

    def __call__(
        self,
        texts: List[str] | str,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]

        encoded_texts: List[List[int]] = []
        for text in texts:
            ids = self.tokenize(text, max_length=max_length)
            if truncation:
                ids = ids[:max_length]
            encoded_texts.append(ids)

        if padding:
            target_len = max(len(ids) for ids in encoded_texts) if encoded_texts else max_length
            target_len = min(target_len, max_length)
        else:
            target_len = None

        padded_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []

        for ids in encoded_texts:
            if target_len is not None:
                ids = ids[:target_len]
                mask = [1] * len(ids)
                pad_len = target_len - len(ids)
                ids = ids + [self.word_to_id[self.pad_token]] * pad_len
                mask = mask + [0] * pad_len
            else:
                mask = [1] * len(ids)

            padded_ids.append(ids)
            attention_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

        return {"input_ids": padded_ids, "attention_mask": attention_masks}

    def save(self, path: str | Path) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "word_to_id" in data:
            vocab_size = int(data.get("vocab_size", len(data["word_to_id"])))
            word_to_id = data["word_to_id"]
        else:
            vocab_size = len(data)
            word_to_id = data

        tok = cls(vocab_size=vocab_size)
        tok.word_to_id = {word: int(idx) for word, idx in word_to_id.items()}
        tok.id_to_word = {idx: word for word, idx in tok.word_to_id.items()}
        tok.next_id = max(tok.id_to_word.keys(), default=-1) + 1
        return tok

def contrastive_loss(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)
    logits = torch.matmul(embeddings1, embeddings2.T) / temperature
    target = torch.arange(embeddings1.shape[0], device=embeddings1.device)
    return F.cross_entropy(logits, target)


def train_encoder(*args, **kwargs):
    raise NotImplementedError(
        "Use the notebook triplet training loop with InfoNCELoss instead of train_encoder()."
    )
