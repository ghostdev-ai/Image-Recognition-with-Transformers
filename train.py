if __name__ == "__main__":
    import os
    import zipfile 
    import requests
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    # Download a custom dataset by providing a url
    parser.add_argument("--dataset_url", type=str)
    # Parameters for ViT model
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--mlp_size", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    # ViT set batch_size=4096, due to hardware limitations we'll default 
    # to a batch_size=32
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--hidden_units', type=int)
    # Parameters for Optimizer
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--betas", nargs='+', type=float, default=(0.9, 0.999))
    parser.add_argument("--weight_decay", type=float, default=0.1)
    args = parser.parse_args()

    


