import numpy as np
from typing import List, Dict, Union

class CharacterURLTokenizer:
    """
    Character and Subword URL Tokenizer.
    Maps characters in canonical URLs to discrete integer IDs with vocabulary padding.
    Supports fixed sequence length representations for CNN/Embedding branches.
    """
    def __init__(self, max_length: int = 150):
        self.max_length = max_length
        # Printable ASCII standard characters commonly found in URLs
        # 0: <PAD>, 1: <UNK>, 2: <START>, 3: <END>
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.start_token_id = 2
        self.end_token_id = 3
        
        self.vocab = {
            '<PAD>': self.pad_token_id,
            '<UNK>': self.unk_token_id,
            '<START>': self.start_token_id,
            '<END>': self.end_token_id
        }
        
        # Standard ASCII characters: 32 (space) to 126 (~)
        for idx, char_code in enumerate(range(32, 127), start=4):
            char = chr(char_code)
            self.vocab[char] = idx
            
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Encodes string into fixed-length token list."""
        if not text:
            text = ""
            
        tokens = [self.vocab.get(c, self.unk_token_id) for c in text[:self.max_length]]
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
            
        return tokens

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encoding into 2D numpy array [batch_size, max_length]."""
        return np.array([self.encode(t) for t in texts], dtype=np.int64)

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to string."""
        chars = []
        for tid in token_ids:
            if tid == self.pad_token_id:
                continue
            chars.append(self.inv_vocab.get(tid, '?'))
        return "".join(chars)
