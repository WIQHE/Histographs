# 🧠 Weakly Supervised Graph Attention Network for Breast Cancer Subtype Classification

This repository implements a **weakly supervised graph neural network (GNN)** pipeline for **breast cancer subtype classification** using **H&E histopathology images**. The method focuses on modeling **cell-cell interactions** via nuclei-based graphs and leverages **attention-based subgraph selection** using GAT.

---

## 📌 Overview

The pipeline addresses:
- The challenge of learning from graph-level labels with instance-level (nucleus-level) supervision.
- The need to capture biologically meaningful tissue structures and interactions.
- The problem of scale and redundancy in large tissue graphs.

---

## 🧬 Methodology

### 🔹 Nuclei Detection
- Performed using **HoVer-Net** via [TIAToolbox](https://github.com/TIA-Lab/tiatoolbox)
- Extracted features per nucleus:
  - Centroid coordinates (x, y)
  - Nucleus type (e.g., tumor, lymphocyte, stroma)
  - Morphological features: area, perimeter, eccentricity, solidity, circularity

### 🔹 Graph Construction
- **Nodes**: Each nucleus with concatenated spatial, categorical, and morph features.
- **Edges**:
  - Built using **Delaunay triangulation** with a spatial threshold (e.g., 90 µm).
  - Edge attributes include inverse distance and nucleus-type interactions.
- Graphs are saved in `.pt` format using `torch_geometric.data.Data`.

### 🔹 Subgraph Sampling (Weak Supervision)
- Localized subgraphs are extracted around **type-1 (tumor)** nuclei.
- Each subgraph inherits the graph-level label (no nucleus-level labels used).
- During training and evaluation:
  - All subgraphs are scored using **GAT attention weights**.
  - The **top 8 subgraphs** (by mean attention score) are selected for learning.
  - Logits from these are averaged to form the graph-level prediction.

---

## 🔗 Model

- Architecture: 2-layer **Graph Attention Network (GAT)** using PyTorch Geometric.
- Heads: 8 in first layer, 1 in second layer (collapsed).
- Final representation is pooled and passed to a classifier.
- Attention weights are used for interpretability and subgraph selection.

---

## 📊 Evaluation

- Train/Test split using `metadata.csv` (80/20).
- Evaluation metrics (via `scikit-learn`):
  - Accuracy
  - Classification report (Precision, Recall, F1-score)
  - Confusion matrix
- Top-𝑘 subgraph predictions are aggregated to compute graph-level logits.

---

## 🛠️ Tools and Libraries

- Segmentation: [TIAToolbox](https://github.com/TIA-Lab/tiatoolbox)
- Graph processing: `networkx`, `scipy`, `torch_geometric`
- Modeling: `torch`, `torch_geometric`
- Evaluation: `scikit-learn`, `matplotlib`, `seaborn`


