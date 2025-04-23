# 🧠 Graph-Based GNN for Breast Cancer Subtype Classification

## 🔍 Overview  
We propose an interpretable deep learning framework combining **CNNs and GNNs** for breast cancer subtype classification using **H&E histopathology images**. Our approach addresses:
- **Imbalanced datasets**
- **Lack of biological interpretability**
- **Inability to model cell-cell interactions in tissue context**

---

## 🧬 Method Summary

### 🧠 Nuclei Detection
- **HoVer-Net (via TIAToolbox)** for segmenting nuclei  
- Extracted features: **Centroids (x, y)** and **Nucleus type** (e.g., tumor, lymphocyte)

### 🕸️ Graph Construction
- **Nodes**: Nuclei with spatial and type attributes  
- **Edges**:  
  - **Spatial** via Delaunay triangulation (≤12.5 µm)  
  - **Functional** between tumor and immune cells (≤25 µm)

### 🧩 Graph Design
- Built with `networkx.MultiGraph`
- Nodes: `(x, y), type`
- Edges: `type`, `distance`, `interaction_type`

---

## 🔗 Learning Model
- **CNN for visual feature extraction**
- **GNN for tissue interaction modeling**:
  - GraphSAGE / GCN / GAT
- **Fusion**: Combine CNN and GNN embeddings for classification

---

## 📊 Evaluation
- Metrics: **Accuracy**, **F1-score**, **PR-AUC**, **Node/Edge metrics**
- Comparisons:
  - CNN vs. CNN+GNN
  - Graph construction methods
  - Handling class imbalance

---

## 🛠️ Tools
- Segmentation: TIAToolbox
- Graphs: `networkx`, `scipy`
- Models: PyTorch, PyTorch Geometric
- Visualization: `matplotlib`, `seaborn`, t-SNE
