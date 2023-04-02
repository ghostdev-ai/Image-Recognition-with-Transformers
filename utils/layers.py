from torch import nn

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, 
               embedding_dim: int=768,
               num_heads: int=12,
               attn_dropout: float=0):
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=768)
     
    self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                           num_heads=num_heads,
                                           dropout=attn_dropout,
                                           batch_first=True)
    
  def forward(self, xp):
    xp = self.layer_norm(xp)
    attn_out, _ = self.multihead_attn(query=xp, 
                                      key=xp, 
                                      value=xp,
                                      need_weights=False)
    return attn_out


class MultilayerPerceptron(nn.Module):
  def __init__(self, 
               embedding_dim: int=768, 
               mlp_size: int=3072, 
               dropout: float=0.1):
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout, 
                   inplace=True),
        nn.Linear(in_features=mlp_size,
                  out_features=embedding_dim),
        nn.Dropout(p=dropout,
                   inplace=True)
    )


  def forward(self, x):
    return self.mlp(self.layer_norm(x))
  
