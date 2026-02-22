import torch.nn as nn

from merlin.charms.models.common import TokenEmbedding, PositionalEmbedding, SegmentEmbedding
from merlin.charms.models.common import TransformerEncoderBlock


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(self.token.embedding_dim)
        self.segment = SegmentEmbedding(self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
    
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


class NextSentencePrediction(nn.Module):
    """Next Sentence Prediction task for BERT. 2-class classifier: isNext / notNext
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """Masked Language Model task for BERT. Predict origin token from masked input
    sequence. n-class classifier, n is vocabulary size.
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: size of vocabulary
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTBackbone(nn.Module):
    """BERT model: Bidirectional Encoder Representations from Transformers"""

    def __init__(self,  vocab_size, hidden_size=768, num_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocabulary size of tokenizer
        :param hidden_size: embedding size and transformer hidden size
        :param num_layers: number of transformer encoder layers
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        :param max_seq_len: maximum sequence length
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = attn_heads
        self.dropout = dropout

        # paper noted that bert used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden_size * 4

        # embedding for BERT, sum of token, segment, and position embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden_size)

        # multi-layers transformer encoder blocks, deep bidirectional model
        self.transformer_blocks = TransformerEncoderBlock(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.feed_forward_hidden,
            dropout=self.dropout,
            activation="gelu",
        )

        # masked language model
        self.mlm = MaskedLanguageModel(self.hidden_size,vocab_size)
        self.nsp = NextSentencePrediction(self.hidden_size)
    
    def forward(self, x, segment_info):
        """
        :param x: input token ids, shape [batch size, seq length]
        :param segment_info: segment token ids, shape [batch size, seq length]
        :return: transformed feature, shape [batch size, seq length, hidden]
        """
        # attention masking for padded token
        # torch.ByteTensor([batch size, 1, seq_len, seq_len])
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        x = self.transformer_blocks.forward(x, mask)

        return x, self.nsp(x), self.mlm(x)
