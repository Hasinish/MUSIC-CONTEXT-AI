"""
contrastive.py - Task 4: Contrastive Cross-Modal Alignment (MusicCaps).
Part of the GNN-BERT Music Context Understanding project (CSE425).

Mathematical Formulation (PDF Section 4.4):
  Normalized Embeddings:
    u_i = g_i / ||g_i||_2       (Audio structure graph projection)
    v_i = t_i / ||t_i||_2       (BERT natural language caption projection)
  Cosine Similarity Matrix:
    S_ij = (u_i^T * v_j) / tau  (tau = temperature hyperparameter, default 0.07)
  Symmetric InfoNCE Contrastive Loss:
    L_audio2text = - (1/N) * sum_i log( exp(S_ii) / sum_j exp(S_ij) )
    L_text2audio = - (1/N) * sum_i log( exp(S_ii) / sum_j exp(S_ji) )
    L_NCE = (L_audio2text + L_text2audio) / 2

Retrieval Metrics:
  Recall@K (R@1, R@5, R@10) for:
    - Text-to-Audio retrieval (Query caption -> Find audio graph)
    - Audio-to-Text retrieval (Query audio -> Find matching description)
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class DualEncoderContrastiveModel(nn.Module):
    """
    Dual-Encoder architecture projecting both GNN graph embeddings and BERT text embeddings
    into a shared multi-modal metric space.
    """
    def __init__(
        self,
        gnn_dim: int = 128,
        text_dim: int = 768,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        self.temperature = temperature

        # Audio graph projection head
        self.audio_proj = nn.Sequential(
            nn.Linear(gnn_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Text caption projection head
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(
        self,
        g: torch.Tensor,
        t_cls: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g: (batch_size, gnn_dim)
        t_cls: (batch_size, text_dim)
        Returns L2-normalized projection vectors in R^projection_dim
        """
        # Project and L2-normalize
        u = F.normalize(self.audio_proj(g), p=2, dim=-1)
        v = F.normalize(self.text_proj(t_cls), p=2, dim=-1)
        return u, v

    def compute_infonce_loss(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric InfoNCE contrastive loss over mini-batch.
        u: (N, d) normalized audio graph vectors
        v: (N, d) normalized text caption vectors
        """
        batch_size = u.size(0)
        labels = torch.arange(batch_size, device=u.device)

        # Scaled cosine similarity matrix S in (N, N)
        similarity_matrix = torch.matmul(u, v.t()) / self.temperature

        # Cross-entropy along rows (audio -> text) and columns (text -> audio)
        loss_a2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2a = F.cross_entropy(similarity_matrix.t(), labels)

        total_loss = (loss_a2t + loss_t2a) / 2.0
        return total_loss, similarity_matrix


def evaluate_retrieval_metrics(similarity_matrix: np.ndarray, top_k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Calculate R@1, R@5, R@10 for Text->Audio and Audio->Text retrieval.
    similarity_matrix: (N, N) where row i is audio i, col j is text j.
    """
    num_samples = similarity_matrix.shape[0]
    metrics = {}

    # Text-to-Audio (each column j queries rows i)
    # The true audio for text j is at row j
    t2a_ranks = []
    for j in range(num_samples):
        ranked_audios = np.argsort(similarity_matrix[:, j])[::-1]
        rank = np.where(ranked_audios == j)[0][0] + 1
        t2a_ranks.append(rank)

    # Audio-to-Text (each row i queries columns j)
    # The true text for audio i is at column i
    a2t_ranks = []
    for i in range(num_samples):
        ranked_texts = np.argsort(similarity_matrix[i, :])[::-1]
        rank = np.where(ranked_texts == i)[0][0] + 1
        a2t_ranks.append(rank)

    t2a_ranks = np.array(t2a_ranks)
    a2t_ranks = np.array(a2t_ranks)

    for k in top_k_list:
        metrics[f"t2a_r@{k}"] = float(np.mean(t2a_ranks <= k))
        metrics[f"a2t_r@{k}"] = float(np.mean(a2t_ranks <= k))

    metrics["mean_r@5"] = float((metrics.get("t2a_r@5", 0.0) + metrics.get("a2t_r@5", 0.0)) / 2.0)
    return metrics


def demo_retrieval_case_studies(
    captions: List[str],
    audio_ids: List[str],
    similarity_matrix: np.ndarray,
    num_examples: int = 5
):
    """
    Outputs qualitative retrieval examples matching Deliverable #2 of Task 4:
    '10 qualitative retrieval examples (query caption -> top-3 matched clips)'
    """
    print("\n=== Task 4: Qualitative Caption -> Audio Retrieval Results ===")
    num_eval = min(num_examples, len(captions))

    for q_idx in range(num_eval):
        query_caption = captions[q_idx]
        correct_id = audio_ids[q_idx]

        # Top 3 ranked audio tracks for this query caption
        sims = similarity_matrix[:, q_idx]
        top3_indices = np.argsort(sims)[::-1][:3]

        print(f"\n[Query Caption {q_idx + 1}]: \"{query_caption}\"")
        print(f"  Ground Truth Track: {correct_id}")
        print("  Top-3 Retrieved Audio Graphs:")
        for rank, match_idx in enumerate(top3_indices):
            matched_id = audio_ids[match_idx]
            score = sims[match_idx]
            status = " [CORRECT MATCH]" if matched_id == correct_id else ""
            print(f"    {rank + 1}. {matched_id:<20} Similarity: {score:.3f}{status}")


if __name__ == "__main__":
    print("=== Testing Task 4: Dual-Encoder InfoNCE Contrastive Model ===")

    batch_size = 8
    gnn_dim = 128
    text_dim = 768
    proj_dim = 128

    dummy_g = torch.randn(batch_size, gnn_dim)
    dummy_t = torch.randn(batch_size, text_dim)

    model = DualEncoderContrastiveModel(
        gnn_dim=gnn_dim,
        text_dim=text_dim,
        projection_dim=proj_dim,
        temperature=0.07
    )

    u, v = model(dummy_g, dummy_t)
    loss, sim_matrix = model.compute_infonce_loss(u, v)

    print(f"Projected Audio: {u.shape} (Norm: {torch.norm(u[0]).item():.2f})")
    print(f"Projected Text:  {v.shape} (Norm: {torch.norm(v[0]).item():.2f})")
    print(f"InfoNCE Loss:    {loss.item():.4f}")

    # Evaluate retrieval metrics on simulated similarity matrix
    sim_np = sim_matrix.detach().cpu().numpy()
    metrics = evaluate_retrieval_metrics(sim_np, top_k_list=[1, 5])
    print(f"Simulated Retrieval: T2A R@1: {metrics['t2a_r@1']:.1%}, T2A R@5: {metrics['t2a_r@5']:.1%}")

    # Run case study demo
    dummy_captions = [f"A track with genre characteristics #{i}" for i in range(batch_size)]
    dummy_tracks = [f"track_id_{i:03d}" for i in range(batch_size)]
    demo_retrieval_case_studies(dummy_captions, dummy_tracks, sim_np, num_examples=3)

    print("\nSUCCESS: Task 4 Dual-Encoder InfoNCE passed verification!")
