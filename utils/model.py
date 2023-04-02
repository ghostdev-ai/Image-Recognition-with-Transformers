from torch import nn
from layers import MultiHeadSelfAttention, MultilayerPerceptron

class TransformerEncoder(nn.Module):
  def __init__(self, 
               embedding_dim: int=768,
               mlp_size: int=3072,
               num_heads: int=12,
               attn_dropout: float=0.0,
               mlp_dropout: float=0.1):
    super().__init__()
    
    self.multihead_attn = MultiHeadSelfAttention(embedding_dim=embedding_dim, 
                                                 num_heads=num_heads,
                                                 attn_dropout=attn_dropout)
    
    self.mlp = MultilayerPerceptron(embedding_dim=embedding_dim,
                                    mlp_size=mlp_size,
                                    dropout=mlp_dropout)
    

  def forward(self, x):
    x = self.multihead_attn(x) + x
    x = self.mlp(x) + x
    return x