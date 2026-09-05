# GNN-Based BERT for Understanding Context from Music

[![Course](https://img.shields.io/badge/Course-CSE425%20Neural%20Networks-blue.svg)](https://github.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20Supported-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official repository for the CSE425 / EEE474 / CSE715 Neural Networks Course Project: **GNN-Based BERT for Understanding Context from Music**.

---

## 🎵 Project Overview
Traditional music classification paradigms rely on 2D CNNs or RNNs operating over flat spectrograms. While effective at extracting local frequency motifs, they miss global relational structure: how repeating verse-chorus sections connect, how harmonic chord syntax evolves, and how lyrical semantics anchor emotional perception.

This project implements a hybrid **BERT + Graph Neural Network (GNN)** multi-modal architecture:
1. **BERT**: Extracts deep contextual language embeddings from lyrics, tags, and natural-language music descriptions (MusicCaps).
2. **GNN (GraphSAGE / GAT)**: Message-passes over music structure graphs (nodes = 5s segments, edges = temporal transitions + harmonic similarity).
3. **Cross-Attention Multi-Task Fusion**: Dynamically aligns audio graph queries with lexical key/value tokens to predict multi-label tags and continuous emotional valence/arousal.
4. **Contrastive Dual-Encoder (InfoNCE)**: Maps audio graphs and text captions into a shared metric space for zero-shot text-to-music search.

---

## 📂 Repository Architecture
```text
gnn-bert-music-context/
├── config.yaml                    # Global hyperparameters & architecture settings
├── requirements.txt               # Pinned dependencies
├── README.md                      # Complete setup & replication guide
├── pyrightconfig.json             # IDE type-checker configuration
├── data/
│   ├── raw/                       # Downloaded audio datasets (GTZAN)
│   ├── processed/graphs/          # Preprocessed .json / .pt structural graphs
│   └── splits/splits.json         # Train / Val / Test split definitions
├── notebooks/
│   ├── eda.ipynb                  # Dataset distributions & audio visualization
│   └── demo_context.ipynb         # Interactive end-to-end inference demo
├── src/
│   ├── audio_features.py          # Audio DSP: 22kHz resampling, mel, chroma, 5s windowing
│   ├── graph_builder.py           # Music structure graph builder (NetworkX / PyG)
│   ├── baseline_cnn.py            # Baseline B2: 2D Mel-Spectrogram CNN
│   ├── bert_encoder.py            # Task 1: DistilBERT Multi-label Tag Classifier
│   ├── gnn_model.py               # Task 2: GraphSAGE / GAT audio structure encoder
│   ├── fusion_model.py            # Task 3: Cross-Attention Multi-Task Fusion Network
│   ├── contrastive.py             # Task 4: Dual-Encoder InfoNCE Contrastive Model
│   ├── dataset.py                 # PyTorch multi-modal Dataset & graph batch collation
│   ├── train.py                   # Master training pipeline for all models
│   ├── evaluate.py                # Table 3 benchmark evaluator & t-SNE generator
│   └── download_dataset.py        # Automated dataset fetcher (Hugging Face CDN)
├── results/
│   ├── checkpoints/               # Trained model weights (.pt)
│   ├── metrics.json               # Exported benchmark numbers (Table 3)
│   └── plots/
│       └── tsne_latent_space.png  # Fused latent space visualization
└── report/
    └── final_report.tex           # 6-10 Page NeurIPS 2024 LaTeX Paper
```

---

## 🚀 Step-by-Step Reproduction Runbook (For Any Machine)

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gnn-bert-music-context.git
cd gnn-bert-music-context

# Install core dependencies
pip install -r requirements.txt

# (Optional) If running on an NVIDIA GPU (e.g. GTX 1050 Ti or RTX 4080):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 2. Download the Dataset (1 Command)
Downloads all 1,000 GTZAN audio tracks (~1.2 GB) via Hugging Face's global CDN and extracts them into `data/raw/gtzan/genres/`:
```bash
python src/download_dataset.py --dataset gtzan
```

---

### 3. Build Music Structure Graphs
Convert audio tracks into graph structures (nodes = 5s segments, edges = temporal transitions + cosine similarity > 0.65):

```bash
# Convert a single WAV file to a graph JSON:
python src/graph_builder.py --wav data/raw/gtzan/genres/blues/blues.00000.wav --out data/processed/graphs/blues_000.json

# Or generate/verify preprocessed graph samples:
python src/graph_builder.py --num_samples 20
```

---

### 4. Train Models Across All Tasks

```bash
# Task 1: BERT Text Classifier Baseline (18 Marks)
python src/train.py --task 1 --epochs 10 --batch_size 16

# Task 2: GraphSAGE Audio Structure Classifier (22 Marks)
python src/train.py --task 2 --epochs 20 --batch_size 16

# Task 3: End-to-End Cross-Attention Multi-Task Fusion (22 Marks)
python src/train.py --task 3 --epochs 30 --batch_size 16

# Task 4: InfoNCE Contrastive Dual-Encoder Retrieval (18 Marks)
python src/train.py --task 4 --epochs 20 --batch_size 16
```

---

### 5. Benchmark Evaluation & Metric Export
Evaluate all models against the test split, reproduce **Table 3 from the project specification**, and generate the 2D t-SNE projection plot:
```bash
python src/evaluate.py
```
Outputs:
- Benchmark table saved to `results/metrics.json`
- 2D t-SNE plot saved to `results/plots/tsne_latent_space.png`

---

## 📊 Benchmark Results (Table 3 Comparison)

| Model Architecture | Macro-F1 | AUC-PR | MAE (Emotion) | R@5 (Retrieval) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Baseline (B1)** | 0.147 | 0.142 | — | 0.020 |
| **2D CNN Mel-Spectrogram (B2)** | 0.412 | 0.385 | 1.250 | — |
| **Task 1: BERT-only (Text)** | 0.485 | 0.442 | — | — |
| **Task 2: GraphSAGE (Audio Graph)** | 0.521 | 0.473 | 1.100 | — |
| **Task 3: GNN–BERT Cross-Attention** | **0.618** | **0.554** | **0.170** | — |
| **Task 4: Contrastive Dual-Encoder** | 0.552 | 0.501 | — | **0.382** |

---

## 💻 Interactive Demo
Launch the interactive Jupyter notebook to perform one-click forward-pass inference on any song:
```bash
jupyter notebook notebooks/demo_context.ipynb
```
Features:
- Waveform & log-mel spectrogram visualization
- Interactive NetworkX graph plot distinguishing temporal vs. similarity edges
- Multi-modal prediction: predicted genre, top-5 tag probabilities, and valence/arousal scores

---

## 📝 Final Report Paper
The complete 6–10 page project report is formatted in the official **NeurIPS 2024 LaTeX template** at [`report/final_report.tex`](report/final_report.tex), containing:
- Formal mathematical formulations of GraphSAGE message passing and Cross-Attention
- Complete multi-task loss derivation ($L_{\text{tags}} + \alpha L_{\text{valence}} + \beta L_{\text{arousal}}$)
- Full 4-way ablation analysis and qualitative retrieval case studies

Compile using `pdflatex report/final_report.tex` or upload directly to Overleaf!

---

## 📜 License & Citation
Developed for CSE425/EEE474/CSE715 Neural Networks at BRAC University. Released under the MIT License.
