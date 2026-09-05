# 🎧 GNN-BERT Music Context AI: Partner Training & Run Guide

Welcome to the **MUSIC-CONTEXT-AI** project guide. Follow this quick step-by-step document to set up the environment, run GPU-accelerated training on your machine (e.g., RTX 3050 Ti), and share the trained weights and benchmarks.

---

## 📋 System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.10, 3.11, or 3.12
- **GPU**: NVIDIA GPU (RTX 3050 Ti, 4GB VRAM or better recommended)
- **Disk Space**: ~2.5 GB (for audio dataset and dependencies)

---

## ⚡ Quickstart Steps

### Step 1: Clone Repository & Install Dependencies
Open **PowerShell** or your preferred terminal and run:
```powershell
# 1. Clone the repository
git clone https://github.com/Hasinish/MUSIC-CONTEXT-AI.git

# 2. Enter the project directory
cd MUSIC-CONTEXT-AI

# 3. Install required Python packages
pip install -r requirements.txt
```
> **Note**: Installation typically takes 2 to 3 minutes depending on your internet connection.

---

### Step 2: Download the GTZAN Audio Dataset & Build Graphs
Download the full 1.2 GB GTZAN music dataset (1,000 songs across 10 genres) and slice them into graph representations:
```powershell
# 1. Automatically download and extract GTZAN audio archives
python src/download_dataset.py --dataset gtzan

# 2. Convert tracks into music structure graphs (20 tracks per genre = 200 graphs)
python src/graph_builder.py --gtzan --limit 20
```
> **What this does**: Audio tracks are resampled to 22,050 Hz, divided into 5-second non-overlapping segments, and converted into graph structures with 140-dimensional node features (128 log-mel + 12 chroma) and temporal + acoustic cosine similarity edges.

---

### Step 3: Train the Model on GPU (Task 3: Cross-Attention Fusion)
Run the master training pipeline using your NVIDIA GPU:
```powershell
python src/train.py --task 3 --epochs 10 --batch_size 16
```
> **Hardware Safety**: The DistilBERT backbone is frozen during training, which caps peak VRAM consumption at **~1.2 GB**. Your 4GB RTX 3050 Ti will comfortably run this without running out of memory.
>
> You will observe loss progression across 10 epochs. The newly trained model weights will be automatically saved to `results/checkpoints/model_3.pt`.

---

### Step 4: Run the Benchmark Evaluation
Evaluate your newly trained model and generate benchmark metrics:
```powershell
python src/evaluate.py
```
**Expected Outputs**:
- Prints the official **Benchmark Table** comparing your model against CNN and baseline architectures.
- Exports metrics to `results/metrics.json`.
- Generates a 2D t-SNE latent space cluster plot in `results/plots/tsne_latent_space.png`.

---

### Step 5: Share the Results Folder
The training and evaluation step updates only three lightweight files inside the `results/` folder (total size ~2 MB):
1. `results/checkpoints/model_3.pt` (Trained model weights)
2. `results/metrics.json` (Benchmark results)
3. `results/plots/tsne_latent_space.png` (Latent cluster visualization)

**How to send:**
1. Navigate to the `MUSIC-CONTEXT-AI` folder in File Explorer.
2. Right-click the **`results`** folder.
3. Select **Send to -> Compressed (zipped) folder** (or use 7-Zip / WinRAR).
4. Send the created `results.zip` to your project partner (Hasin) via WhatsApp, Telegram, or Google Drive!

---

## 📄 Overleaf / Final Report Submission
The complete 6-10 page NeurIPS 2024 academic paper is located in `report/final_report.tex`.

1. Go to [Overleaf.com](https://www.overleaf.com).
2. Click **New Project -> Upload Project** and upload the `report/` folder.
3. Open `final_report.tex` and ensure all team member names and student IDs are added under the `\author{...}` block:
   ```latex
   \author{
     Hasin Ishrak \\
     Department of Computer Science and Engineering \\
     BRAC University, Dhaka, Bangladesh \\
     \texttt{hasin.ishrak@g.bracu.ac.bd} \\
     \And
     [Your Name] \\
     Department of Computer Science and Engineering \\
     BRAC University, Dhaka, Bangladesh \\
     \texttt{[your.email@g.bracu.ac.bd]} \\
   }
   ```
4. Click **Recompile** to produce the final PDF submission!

---

## 🧠 Viva & Presentation Cheat Sheet
Key questions course instructors and evaluators may ask:

1. **Why represent music as a Graph rather than a flat 2D Spectrogram image?**
   - *Answer*: Standard 2D CNNs only capture localized time-frequency textures. Music exhibits global structural repetitions (e.g., recurring chorus themes, bridges, rhythmic motifs) across long time spans. Graph representations with cosine similarity edges allow non-adjacent segments to exchange structural information directly via GNN message passing.

2. **How does the Task 3 Cross-Attention mechanism operate?**
   - *Answer*: We project the GNN structural music embedding $g$ as the Query ($Q = g W_Q$) and the contextual text token representations $H_{\text{text}}$ from DistilBERT as the Keys and Values ($K, V = H_{\text{text}} W$). This enables audio features to dynamically attend to semantic mood and genre tokens.

3. **What loss function is optimized in Task 3?**
   - *Answer*: We optimize a combined multi-task loss $\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{MSE}}$, where Binary Cross-Entropy (BCE) supervises multi-label genre and tag classification, and Mean Squared Error (MSE) optimizes continuous emotional Valence and Arousal coordinates.
