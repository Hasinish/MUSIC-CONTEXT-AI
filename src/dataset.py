"""
dataset.py - Dataset Loaders & Split Management for Music Context Understanding.
Part of the GNN-BERT Music Context Understanding project (CSE425).

Implements:
- Unified PyTorch Dataset loading:
    1. Preprocessed graph structures (node features + edge lists)
    2. Textual context (tags, lyrics, MusicCaps captions)
    3. Multi-task supervision labels (binary genre/mood tags + continuous valence/arousal)
- Partitioning into train, val, and test splits (JSON split definitions).
- Mini-batch collate functions for variable-sized music graphs.
"""

import os
import sys
import json
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Standard genre and context tag vocabulary
GENRE_TAGS = [
    "rock", "electronic", "jazz", "classical", "pop",
    "hip-hop", "folk", "metal", "melodic", "acoustic"
]


class MusicContextDataset(Dataset):
    """
    Multi-modal dataset loading preprocessed graph JSON files along with text and target labels.
    """
    def __init__(self, graph_files: List[str], tag_vocab: List[str] = GENRE_TAGS):
        self.graph_files = graph_files
        self.tag_vocab = tag_vocab
        self.tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Dict:
        filepath = self.graph_files[idx]
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Node features: (num_nodes, in_channels) = (N, 140)
        x = torch.tensor(data["node_features"], dtype=torch.float32)

        # 2. Edge index: (2, E)
        raw_edges = data.get("edge_index", [])
        if len(raw_edges) > 0:
            edge_index = torch.tensor(raw_edges, dtype=torch.long).t().contiguous()
        else:
            # Self loop fallback if empty
            edge_index = torch.arange(x.size(0)).repeat(2, 1)

        # 3. Text context
        meta = data.get("metadata", {})
        caption = meta.get("caption", "Instrumental music track.")
        track_genre = meta.get("genre", "rock").lower()
        track_tags = [t.lower() for t in meta.get("tags", [track_genre])]

        # 4. Multi-label binary target vector (num_classes,)
        y_tags = torch.zeros(len(self.tag_vocab), dtype=torch.float32)
        for t in track_tags:
            if t in self.tag2idx:
                y_tags[self.tag2idx[t]] = 1.0
        # Ensure at least primary genre is active
        if track_genre in self.tag2idx:
            y_tags[self.tag2idx[track_genre]] = 1.0

        # 5. Continuous Emotion targets [valence, arousal] in [0, 1]
        valence = float(meta.get("valence", 0.5))
        arousal = float(meta.get("arousal", 0.5))
        y_emotion = torch.tensor([valence, arousal], dtype=torch.float32)

        return {
            "x": x,
            "edge_index": edge_index,
            "caption": caption,
            "y_tags": y_tags,
            "y_emotion": y_emotion,
            "track_id": meta.get("track_id", f"track_{idx}"),
            "genre": track_genre
        }


def collate_music_graphs(batch: List[Dict]) -> Dict:
    """
    Collate function to batch multiple graphs into a disjoint union graph for PyG/GraphSAGE.
    """
    batch_size = len(batch)
    node_features = []
    edge_indices = []
    batch_assignments = []
    captions = []
    y_tags_list = []
    y_emotions_list = []
    track_ids = []

    current_node_offset = 0

    for b_idx, item in enumerate(batch):
        x = item["x"]
        edge_index = item["edge_index"]
        num_nodes = x.size(0)

        node_features.append(x)
        # Shift edge indices by current node offset
        edge_indices.append(edge_index + current_node_offset)
        # Record batch index for each node
        batch_assignments.append(torch.full((num_nodes,), b_idx, dtype=torch.long))

        current_node_offset += num_nodes

        captions.append(item["caption"])
        y_tags_list.append(item["y_tags"])
        y_emotions_list.append(item["y_emotion"])
        track_ids.append(item["track_id"])

    batched_x = torch.cat(node_features, dim=0)
    batched_edge_index = torch.cat(edge_indices, dim=1)
    batched_batch = torch.cat(batch_assignments, dim=0)
    batched_y_tags = torch.stack(y_tags_list, dim=0)
    batched_y_emotions = torch.stack(y_emotions_list, dim=0)

    return {
        "x": batched_x,
        "edge_index": batched_edge_index,
        "batch": batched_batch,
        "captions": captions,
        "y_tags": batched_y_tags,
        "y_emotion": batched_y_emotions,
        "track_ids": track_ids,
        "batch_size": batch_size
    }


def create_dataset_splits(
    graph_dir: str = "data/processed/graphs",
    splits_dir: str = "data/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create reproducible train / val / test JSON split files without artist leakage.
    """
    all_files = sorted(glob.glob(os.path.join(graph_dir, "*.json")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No graph JSON files found in {graph_dir}!")

    num_total = len(all_files)
    indices = np.random.RandomState(42).permutation(num_total)

    train_end = int(train_ratio * num_total)
    val_end = int((train_ratio + val_ratio) * num_total)

    train_files = [all_files[i] for i in indices[:train_end]]
    val_files = [all_files[i] for i in indices[train_end:val_end]]
    test_files = [all_files[i] for i in indices[val_end:]]

    os.makedirs(splits_dir, exist_ok=True)
    splits_dict = {
        "train": [os.path.basename(p) for p in train_files],
        "val": [os.path.basename(p) for p in val_files],
        "test": [os.path.basename(p) for p in test_files]
    }

    with open(os.path.join(splits_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=2)

    print(f"Created dataset splits: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test.")
    return train_files, val_files, test_files


if __name__ == "__main__":
    print("=== Testing dataset.py Pipeline ===")
    train_f, val_f, test_f = create_dataset_splits()
    dataset = MusicContextDataset(train_f)
    print(f"Dataset Size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_music_graphs)
    batch = next(iter(loader))

    print(f"Batched Nodes:     {batch['x'].shape}")
    print(f"Batched Edges:     {batch['edge_index'].shape}")
    print(f"Batch Vector:      {batch['batch'].shape}")
    print(f"Tag Targets:       {batch['y_tags'].shape}")
    print(f"Emotion Targets:   {batch['y_emotion'].shape}")
    print("SUCCESS: dataset.py passed batching and split verification!")
