"""
fusion_model.py - Task 3: GNN-BERT Cross-Attention Multi-Task Fusion.
Part of the GNN-BERT Music Context Understanding project (CSE425).

Mathematical Formulation (PDF Section 4.3):
  Cross-Attention Fusion:
    Q = g * W_Q             (Query from GNN graph readout g in R^d)
    K = H_text * W_K        (Key from BERT token sequence in R^(L x d))
    V = H_text * W_V        (Value from BERT token sequence in R^(L x d))
    A = softmax( (Q * K^T) / sqrt(d) )
    Context = A * V
    z = CONCAT(g, Context)  in R^(2d)
    y_hat = sigma(W * z)

Multi-Task Loss (PDF Equation):
  L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2
  (Joint BCE for genre/tags + MSE for continuous Valence & Arousal).

Supports All 4 Required Ablation Modes:
  1. 'cross_attention' (Full proposed hybrid model)
  2. 'early_concat'    (Concat g and BERT CLS token)
  3. 'bert_only'       (Text semantics only)
  4. 'gnn_only'        (Audio graph structure only)
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Optional
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class CrossAttentionFusionBlock(nn.Module):
    """
    Multi-Head Cross-Attention Layer between Graph Readout Query and BERT Token Sequence.
    """
    def __init__(self, embed_dim: int = 128, text_dim: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(text_dim, embed_dim)
        self.w_v = nn.Linear(text_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        g: torch.Tensor,
        H_text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g: (batch_size, embed_dim) - Graph readout vector
        H_text: (batch_size, seq_len, text_dim) - BERT contextual token representations
        attention_mask: (batch_size, seq_len) - 1 for valid tokens, 0 for pad tokens

        Returns:
          context: (batch_size, embed_dim) attended text representation
          attn_weights: (batch_size, num_heads, 1, seq_len) attention map for qualitative case studies!
        """
        batch_size = g.size(0)
        seq_len = H_text.size(1)

        # 1. Project Q from g: (batch_size, 1, embed_dim) -> (batch_size, num_heads, 1, head_dim)
        q = self.w_q(g).unsqueeze(1) # (batch, 1, embed_dim)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Project K and V from H_text: -> (batch_size, num_heads, seq_len, head_dim)
        k = self.w_k(H_text).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(H_text).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention: Q * K^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (batch, num_heads, 1, seq_len)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Multiply by Value: (batch, num_heads, 1, head_dim)
        context = torch.matmul(attn_weights, v)
        # Reshape back to (batch_size, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.embed_dim)
        context = self.out_proj(context)

        return context, attn_weights


class GNNBertFusionModel(nn.Module):
    """
    End-to-End Multi-Modal Fusion Network supporting Cross-Attention, Multi-Task Prediction,
    and all 4 ablation variants.
    """
    def __init__(
        self,
        gnn_dim: int = 128,
        text_dim: int = 768,
        fusion_dim: int = 128,
        num_classes: int = 10,
        fusion_type: str = "cross_attention",
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.gnn_dim = gnn_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        # Setup fusion mechanism
        if fusion_type == "cross_attention":
            self.cross_attn = CrossAttentionFusionBlock(
                embed_dim=gnn_dim, text_dim=text_dim, num_heads=num_heads, dropout=dropout
            )
            fused_dim = gnn_dim + gnn_dim # [g, Context] = 2d
        elif fusion_type == "early_concat":
            # Direct concat of g and CLS text vector
            self.text_proj = nn.Linear(text_dim, gnn_dim)
            fused_dim = gnn_dim + gnn_dim
        elif fusion_type == "bert_only":
            self.text_proj = nn.Linear(text_dim, gnn_dim)
            fused_dim = gnn_dim
        elif fusion_type == "gnn_only":
            fused_dim = gnn_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.dropout = nn.Dropout(dropout)

        # Latent bottleneck representation z (for t-SNE visualization deliverable)
        self.bottleneck = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-Task Prediction Heads:
        # Head 1: Multi-label genre / mood tag classifier
        self.tag_head = nn.Linear(fusion_dim, num_classes)
        # Head 2: Continuous Emotion Regressor (Valence and Arousal in [0, 1])
        self.emotion_head = nn.Linear(fusion_dim, 2)

    def forward(
        self,
        g: torch.Tensor,
        H_text: Optional[torch.Tensor] = None,
        t_cls: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          tag_logits: (batch_size, num_classes)
          emotion_preds: (batch_size, 2) -> [valence, arousal]
          z: (batch_size, fusion_dim) -> Dense fused latent representation for t-SNE
          attn_map: (batch_size, seq_len) attention alignment weights (for case studies)
        """
        attn_map = None

        if self.fusion_type == "cross_attention":
            context, attn_weights = self.cross_attn(g, H_text, attention_mask)
            fused = torch.cat([g, context], dim=-1)
            attn_map = attn_weights.mean(dim=1).squeeze(1) # Average over heads
        elif self.fusion_type == "early_concat":
            t_proj = F.relu(self.text_proj(t_cls))
            fused = torch.cat([g, t_proj], dim=-1)
        elif self.fusion_type == "bert_only":
            fused = F.relu(self.text_proj(t_cls))
        elif self.fusion_type == "gnn_only":
            fused = g

        # Latent representation z
        z = self.bottleneck(fused)

        # Prediction outputs
        tag_logits = self.tag_head(z)
        emotion_preds = torch.sigmoid(self.emotion_head(z)) # Clamped to [0, 1]

        return tag_logits, emotion_preds, z, attn_map


class MultiTaskLoss(nn.Module):
    """
    Joint Multi-Task Loss Function (PDF Section 4.3):
      L = BCE(y_tags, y_hat_tags) + alpha * MSE(valence) + beta * MSE(arousal)
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        tag_logits: torch.Tensor,
        emotion_preds: torch.Tensor,
        target_tags: torch.Tensor,
        target_emotions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_tags = self.bce(tag_logits, target_tags)
        total_loss = loss_tags
        loss_dict = {"loss_tags": float(loss_tags.item())}

        if target_emotions is not None:
            # target_emotions: (batch_size, 2) -> [valence, arousal]
            loss_valence = self.mse(emotion_preds[:, 0], target_emotions[:, 0])
            loss_arousal = self.mse(emotion_preds[:, 1], target_emotions[:, 1])
            total_loss = total_loss + self.alpha * loss_valence + self.beta * loss_arousal
            loss_dict["loss_valence"] = float(loss_valence.item())
            loss_dict["loss_arousal"] = float(loss_arousal.item())

        loss_dict["total_loss"] = float(total_loss.item())
        return total_loss, loss_dict


if __name__ == "__main__":
    print("=== Testing Task 3: GNN-BERT Multi-Task Cross-Attention Fusion ===")

    batch_size = 4
    gnn_dim = 128
    text_dim = 768
    seq_len = 32
    num_classes = 10

    # Dummy inputs
    dummy_g = torch.randn(batch_size, gnn_dim)
    dummy_H_text = torch.randn(batch_size, seq_len, text_dim)
    dummy_t_cls = dummy_H_text[:, 0, :]
    dummy_mask = torch.ones(batch_size, seq_len)

    # Targets
    target_tags = torch.randint(0, 2, (batch_size, num_classes)).float()
    target_emotions = torch.rand(batch_size, 2) # [valence, arousal] in [0, 1]

    criterion = MultiTaskLoss(alpha=0.5, beta=0.5)

    # Test all 4 ablation configurations
    for ftype in ["cross_attention", "early_concat", "bert_only", "gnn_only"]:
        model = GNNBertFusionModel(
            gnn_dim=gnn_dim,
            text_dim=text_dim,
            fusion_dim=128,
            num_classes=num_classes,
            fusion_type=ftype
        )
        tag_logits, emotion_preds, z, attn_map = model(dummy_g, dummy_H_text, dummy_t_cls, dummy_mask)
        loss, loss_dict = criterion(tag_logits, emotion_preds, target_tags, target_emotions)

        print(f"[{ftype:<16}] z shape: {z.shape} | Tag Logits: {tag_logits.shape} | Loss: {loss.item():.4f}")
        assert z.shape == (batch_size, 128)
        assert tag_logits.shape == (batch_size, num_classes)
        assert emotion_preds.shape == (batch_size, 2)

    print("SUCCESS: All 4 Task 3 Fusion ablation variants passed verification!")
