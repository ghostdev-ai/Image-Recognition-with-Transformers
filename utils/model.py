

import torch
from torch import nn
from layers import PatchEmbedding, TransformerEncoder


class ViT(nn.Module):
    """
    Creates the ViT architecture.

    Replicates the ViT architecture from the paper, An Image is Worth 16x16 words:
    Transformers For Image Recognition At Scale in PyTorch. See the paper here: https://arxiv.org/pdf/2010.11929.pdf.

    Args:
      height: An integer indicating the image height.
      width: An integer indicating the image width.
      color_channels: An integer indicating the number of color channels per image. 
      patch_size: An integer indicating the number of image patches.
      batch_size: An integer indicating the number of images in a batch.
      num_layers: An integer indicating the number of Transformer Encoder layers.
      embedding_dim: An integer indicating the size of the 1D token embedding.
      mlp_size: An integer indicating the size of the MLP hidden layer.
      num_heads: An integer indicating the number of attention heads in the Multi-head self attention layer. 
      dropout: A float indicating the probability of an element to be zeroed in a dropout layer. 
      num_classes: An integer indicating the number of classes. 
    """

    def __init__(self,
                 height: int = 224,
                 width: int = 224,
                 color_channels: int = 3,
                 patch_size: int = 16,
                 batch_size: int = 1,
                 num_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 num_classes: int = 1000):
        super().__init__()

        self.num_patches = int((height * width) / patch_size**2)
        self.class_embedding = nn.Parameter(data=torch.randn(batch_size, 1, embedding_dim),
                                            requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(batch_size, self.num_patches + 1, embedding_dim),
                                               requires_grad=True)
        self.dropout = nn.Dropout(p=dropout,
                                  inplace=True)

        self.embedded_patches = PatchEmbedding(in_channels=color_channels,
                                               patch_size=patch_size,
                                               embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(embedding_dim=embedding_dim,
                                                                      mlp_size=mlp_size,
                                                                      num_heads=num_heads,
                                                                      attn_dropout=0.0,
                                                                      mlp_dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        xp = self.embedded_patches(x)

        xp = torch.cat((self.class_embedding, xp), dim=1)

        xp += self.position_embedding

        xp = self.dropout(xp)

        out = self.transformer_encoder(xp)

        return self.mlp_head(out[:, 0, :])
