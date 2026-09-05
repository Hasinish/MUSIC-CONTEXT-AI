# 🚀 Quick GPU Run Guide (3050 Ti)

Just run these commands in **PowerShell** one by one:

---

### Step 1: Setup
```powershell
git clone https://github.com/Hasinish/MUSIC-CONTEXT-AI.git
cd MUSIC-CONTEXT-AI
pip install -r requirements.txt
```

---

### Step 2: Prepare Audio Data
```powershell
python src/download_dataset.py --dataset gtzan
python src/graph_builder.py --gtzan --limit 20
```

---

### Step 3: Run Training on your GPU
```powershell
python src/train.py --task 3 --epochs 10 --batch_size 16
```
*(Takes about 5–10 minutes. DistilBERT is frozen, so it only uses ~1.2 GB VRAM — perfectly safe for your 4GB 3050 Ti).*

---

### Step 4: Run Evaluation
```powershell
python src/evaluate.py
```

---

### Step 5: Send Me The Files
1. Go to the `MUSIC-CONTEXT-AI` folder.
2. Right-click the **`results`** folder -> **Send to -> Compressed (zipped) folder**.
3. Send me **`results.zip`** on WhatsApp / Telegram / Drive!
