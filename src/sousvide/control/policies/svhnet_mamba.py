import torch
import torch.nn as nn
from transformers import MambaModel, MambaConfig, AutoProcessor
from typing import List
import torchvision.transforms as T

class MambaVisionEncoder(nn.Module):
    def __init__(self, input_dim=3*224*224, hidden_dim=128, depth=4):
        super().__init__()
        config = MambaConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=depth,
            vocab_size=1,  # dummy
            intermediate_size=4 * hidden_dim,
            use_cache=False
        )
        self.mamba = MambaModel(config)
        self.flatten = nn.Flatten()
        self.project = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):  # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.flatten(x)                        # [B*T, 3*224*224]
        x = self.project(x).view(B, T, -1)         # [B, T, hidden]
        return self.mamba(inputs_embeds=x).last_hidden_state  # [B, T, hidden]



class MambaTextEncoder(nn.Module):
    def __init__(self, hidden_dim=128, depth=4):
        super().__init__()
        config = MambaConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=depth,
            vocab_size=30522,  # BERT vocab size
            use_cache=False
        )
        self.mamba = MambaModel(config)
        self.tokenizer = AutoProcessor.from_pretrained("bert-base-uncased")

    def forward(self, texts: List[str]):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        emb = self.mamba(**tokens).last_hidden_state[:, 0]  # [B, hidden]
        return emb



class HNetVisualEncoder(nn.Module):
    def __init__(self, vision_dim=128, num_chunks=6):
        super().__init__()
        self.image_encoder = MambaVisionEncoder(hidden_dim=vision_dim)
        self.text_encoder = MambaTextEncoder(hidden_dim=vision_dim)
        self.output_proj = nn.Linear(2 * vision_dim, vision_dim)

    def forward(self, img_seq: torch.Tensor, query: List[str]):
        vision_latent = self.image_encoder(img_seq)[:, -1]   # [B, D]
        query_latent = self.text_encoder(query)              # [B, D]
        fused = torch.cat([vision_latent, query_latent], dim=-1)
        return self.output_proj(fused)                       # [B, D]
