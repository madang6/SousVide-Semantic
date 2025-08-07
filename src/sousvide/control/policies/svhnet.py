import torch
import torch.nn as nn
import timm
from typing import List
from transformers import BertModel, BertTokenizer

# --- Chunker ---
class Chunker(nn.Module):
    def __init__(self, cnn_dim=512, chunk_dim=128, num_chunks=6):
        super().__init__()
        self.encoder = timm.create_model("resnet18", pretrained=True, features_only=True)
        self.proj = nn.Linear(self.encoder.feature_info[-1]['num_chs'], chunk_dim)
        self.num_chunks = num_chunks

    def forward(self, x):  # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)[-1]  # final conv layer, shape [B*T, C, h, w]
        feats = feats.mean(dim=[2, 3])  # global average pool → [B*T, C]
        feats = self.proj(feats)       # [B*T, chunk_dim]
        feats = feats.view(B, T, -1)

        # Soft chunking: fixed linear weights for now (can be upgraded to learned gates)
        weights = torch.softmax(torch.linspace(-1, 1, T).repeat(B, 1), dim=1).unsqueeze(-1)
        pooled = torch.einsum("btd,btn->bnd", feats, weights)  # [B, num_chunks, D]
        return pooled

# --- Mamba-style sequence encoder (placeholder) ---
class MambaEncoder(nn.Module):
    def __init__(self, dim=128, depth=2):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
            for _ in range(depth)
        ])

    def forward(self, x):  # [B, T, D]
        return self.layers(x)

# --- Text Encoder ---
class TextEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, queries: List[str]):
        with torch.no_grad():  # Freeze for now
            inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
            out = self.bert(**inputs).last_hidden_state[:, 0]  # CLS token
        return self.proj(out)


class HNetVisualEncoder(nn.Module):
    def __init__(self, vision_dim=128, num_chunks=6):
        super().__init__()
        self.chunker = Chunker(chunk_dim=vision_dim, num_chunks=num_chunks)
        self.text_encoder = TextEncoder(out_dim=vision_dim)
        self.temporal_model = MambaEncoder(dim=vision_dim)

    def forward(self, image_seq: torch.Tensor, query: List[str]) -> torch.Tensor:
        """
        Args:
            image_seq: [B, T, 3, 224, 224]
            query: List of instruction strings

        Returns:
            vision_latent: [B, vision_dim]
        """
        B = image_seq.shape[0]
        vision_chunks = self.chunker(image_seq)          # [B, num_chunks, D]
        query_embed = self.text_encoder(query)           # [B, D]
        query_expanded = query_embed.unsqueeze(1).expand_as(vision_chunks)

        fused = vision_chunks + query_expanded           # [B, num_chunks, D]
        encoded = self.temporal_model(fused)             # [B, num_chunks, D]
        final = encoded[:, -1]                           # [B, D]
        return final
