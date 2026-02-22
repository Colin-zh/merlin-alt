# MIT License
# Copyright (c) 2026 Colin-zh
import pickle
import tqdm
from collections import Counter


class TorchVocab:
    """Define a vocabulary object that will be used to numericalize a field.

    Arguments:
        counter: collections.Counter object holding word counts.
        max_size: the maximum size of the vocabulary.
        min_freq: the minimum frequency needed to include a token in the vocabulary.
        specials: a list of special tokens (e.g. <pad>, <oov>) that will be added to the
            vocabulary first (in that order) and will not be sorted or filtered.
        vectors: an object representing pre-trained word vectors. See torchtext.vocab.Vectors
            for more information.
        unk_init: a function that takes in a Tensor and initializes it for unknown words.
            If None, defaults to initializing to zero.
        vectors_cache: an optional directory to cache vectors in. If None, defaults to
            ~/.vector_cache.
    
    Attributes:
        freqs: A collections.Counter object holding word counts.
        stoi: A collections.defaultdict instance mapping tokens to numerical IDs.
        itos: A list mapping numerical IDs to tokens.
    """
    
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building the vocab
        # in frequency order
        for tok in specials:
            del counter[tok]
        
        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        
        # stoi is simply a reverse dict for itos
        self.stoi = {word: i for i, word in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init, vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None, \
                "If you don't provide vectors, you shouldn't provide unk_init or vectors_cache."
    
    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.itos != other.itos:
            return False
        if self.stoi != other.stoi:
            return False
        if self.vectors != other.vectors:
            return False
        return True
    
    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        """
        :param counter: collections.Counter object holding word counts
        :param max_size: maximum size of the vocabulary
        :param min_freq: minimum frequency needed to include a token in the vocabulary
        
        Attributes:
            pad_index: index of the padding token
            unk_index: index of the out-of-vocabulary token
            eos_index: index of the end-of-sequence token
            sos_index: index of the start-of-sequence token
            mask_index: index of the mask token
        """
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=['<pad>', '<oov>', '<eos>', '<sos>', '<mask>'],
                         max_size=max_size, min_freq=min_freq)
    
    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False) -> list:
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> "Vocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path: str):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building word vocab...")
        counter = Counter()
        for line in tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()
            
            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            words = sentence.split()
        
        seq = [self.stoi.get(w, self.unk_index) for w in words]

        if with_sos:
            seq = [self.sos_index] + seq # start-of-sequence token
        if with_eos:
            seq = seq + [self.eos_index] # end-of-sequence token
        
        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq = seq + [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]
            if with_eos and seq[-1] != self.eos_index:
                seq[-1] = self.eos_index
        
        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = []
        for idx in seq:
            if not with_pad or idx != self.pad_index:
                if idx < len(self.itos):
                    word = self.itos[idx]
                else:
                    word = "<%d>" % idx
                words.append(word)
        
        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> "WordVocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str,
                        help="Corpus file path")
    parser.add_argument("-o", "--output_path", required=True, type=str,
                        help="Output file path")
    parser.add_argument("-s", "--vocab_size", default=None, type=int,
                        help="Maximum vocabulary size")
    parser.add_argument("-e", "--encoding", default="utf-8", type=str,
                        help="File encoding")
    parser.add_argument("-m", "--min_freq", default=1, type=int,
                        help="Minimum frequency of words")
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print(f"Vocab size: {len(vocab)}")
    vocab.save_vocab(args.output_path)
