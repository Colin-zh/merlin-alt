"""
MIT License
Copyright (c) 2026 Colin-zh

Tokenizer module for the dataset package.

Phase 1: Word-Based Tokenizer
This tokenizer splits sentences into words based on whitespace.
>>> test = "I love programming."
>>> tokens = test.split()

Phase 2: Character-Based Tokenizer
This tokenizer splits sentences into individual characters.
>>> test = "I love programming."
>>> tokens = list(test)

Phase 3: Subword-Based Tokenizer (e.g., BPE for GPT, WordPiece for BERT, etc.)
This tokenizer breaks down words into subword units, which helps in handling
out-of-vocabulary words and capturing morphological patterns.
>>> from transformers import BertTokenizer
>>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
>>> test = "unhappiness"
>>> tokens = tokenizer.tokenize(test)
>>> print(tokens)
['un', '##happiness']

"""

import re
from collections import Counter, defaultdict


class BPETokenizer:
    """Byte Pair Encoding (BPE) Tokenizer for subword tokenization."""

    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_codes = {}
    
    def train(self, texts):
        """Train BPE tokenizer on the provided texts."""
        # Initialize vocabulary with character-level tokens
        token_freqs = Counter()
        for text in texts:
            tokens = list(text)
            token_freqs.update(tokens)
        
        self.vocab = {token: freq for token, freq in token_freqs.items()}
        
        # Perform BPE merges until reaching the desired vocab size
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(token_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair, token_freqs)
            self.bpe_codes[best_pair] = len(self.bpe_codes)
    
    def tokenize(self, text):
        """Tokenize input text using the trained BPE codes."""
        tokens = list(text)
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            candidate_pairs = [pair for pair in pairs if pair in self.bpe_codes]
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs, key=lambda pair: self.bpe_codes[pair])
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(''.join(best_pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def get_stats(self, token_freqs):
        """Get frequency of adjacent symbol pairs."""
        pairs = defaultdict(int)
        for token, freq in token_freqs.items():
            symbols = token.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, token_freqs):
        """Merge the most frequent pair in the vocabulary."""
        new_token = ''.join(pair)
        new_token_freqs = Counter()
        
        for token, freq in token_freqs.items():
            new_token_str = token.replace(' '.join(pair), new_token)
            new_token_freqs[new_token_str] += freq
        
        token_freqs.clear()
        token_freqs.update(new_token_freqs)
        self.vocab[new_token] = sum(freq for token, freq in new_token_freqs.items() if new_token in token)
    

class WordPieceTokenizer:
    """WordPiece Tokenizer for subword tokenization.
    WordPiece merges characters into subwords based on log-likelihood, rather than frequency like BPE.
    score(x, y) = log P(xy) - log P(x) - log P(y)

    """

    def __init__(self, vocab, unk_token="[UNK]", max_chars=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_chars = max_chars
    
    def tokenize(self, text):
        """Tokenize input text using WordPiece algorithm."""
        output_tokens = []
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_chars:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                curr_substr = None
                # Find the longest substring in the vocab from end to start
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        curr_substr = substr
                        break
                    end -= 1
                # If no substring found, mark as bad token
                if curr_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(curr_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
    
    def train(self, texts):
        """Training WordPiece tokenizer is complex and typically requires
        a large corpus and iterative optimization. This is a placeholder."""
        pass
