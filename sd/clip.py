import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Embedding(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # (Batch_size, seq_len) -> (Batch_size, seq_len, dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int):

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor):
        # (batch_size, seq_len, dim)
        residue = x 

        ## Self Attention

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        ## Feedforward Layer

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residue

        return x

class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) # (no_of_words (vocab_size), dim, seq_len)
        
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens:torch.LongTensor):
        tokens = tokens.type(torch.long)

        # (Batch_size, seq_len) -> (Batch_size, seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_size, seq_len, dim)
        output = self.layernorm(state)

        return output
    