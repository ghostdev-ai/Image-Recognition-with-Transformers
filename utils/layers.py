from torch import nn

class PatchEmbedding(nn.Module):
  """Turns a 2D input image into a 1D sequence learnable embedding vector.

  Args: 
    in_channels (int): Number of color channels for the input images. Defaults to 3.
    patch_size (int): Size of patches to convert input image into. Defaults to 16.
    embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
  """
  def __init__(self, 
               in_channels: int=3, 
               patch_size: int=16, 
               embedding_dim: int=768):
    super().__init__()

    self.conv = nn.Conv2d(in_channels=in_channels,
                      out_channels=embedding_dim,
                      kernel_size=patch_size,
                      stride=patch_size,
                      padding=0)
                      
    self.flatten = nn.Flatten(start_dim=2,
                              end_dim=3)
    
  def forward(self, x):
    return self.flatten(self.conv(x)).permute(0, 2, 1)


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
  
