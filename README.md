# 🧠 Weakly Supervised Graph Attention Network for Breast Cancer Subtype Classification

This repository implements a **weakly supervised GNN** pipeline for classifying breast cancer subtypes using the **BACH dataset**. The model uses **nuclei graphs constructed from H&E-stained whole slide images (WSIs)** and trains on **attention-weighted subgraphs** selected from larger tissue graphs.

---

## 📌 Overview

- **Dataset**: [ICIAR 2018 BACH Challenge](https://iciar2018-challenge.grand-challenge.org/)
- **Segmentation model**: HoVer-Net (`hovernet_fast-monusac` and `hovernet_fast_pannuke`)
- **Graph model**: 2-layer GAT with attention-based top-𝑘 subgraph selection
- **Task**: 4-class subtype classification using weakly supervised graph labels

---

## 🧬 Pipeline Summary

### 🧠 Nuclei Detection
- HoVer-Net is applied using **TIAToolbox**.
- Two detection variants used:
  - `hovernet_fast-monusac` → `n_detected/`
  - `hovernet_fast-pannuke` → `n_detected_pannuke/`
- Outputs are `.dat` files containing per-nucleus centroids and types.

### 🕸️ Graph Construction
- Full graphs are constructed using `networkx.MultiGraph`.
- Nodes: nuclei with (x, y), type, and morph features.
- Edges: constructed using Delaunay triangulation with distance and interaction filters.
- Output saved in `.graphml` or `.pt`.

### 🧩 Subgraph Construction
- Local subgraphs created with a **sliding window**:
  - Window size = 100
  - Step size = 50
- Subgraphs inherit graph labels for **weak supervision**.
- Top-8 subgraphs selected using **GAT attention weights** for training.

---

## 🔗 Learning Model

- **Graph Model**: GAT (Graph Attention Network)
  - 8-head attention in the first layer, 1-head in the second
- **Training**:
  - Average logits from top-8 attentive subgraphs
  - Graph-level labels used (no nucleus-level supervision)
- **Evaluation**:
  - Accuracy, precision, recall, F1-score, confusion matrix

---

## 📁 Directory Structure and File Descriptions

| Path / File                       | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `dataset/`                       | Contains BACH images, metadata, and generated graphs (2018 ICAIR Challenge)                       |
| `n_detected/`                    | HoVer-Net `hovernet_fast-monusac` output (.dat) for BACH nuclei detection  |
| `n_detected_pannuke/`           | HoVer-Net `hovernet_fast-pannuke` output (.dat) on the same BACH data      |
| `n_detect.ipynb`                | Notebook to run HoVer-Net segmentation and generate `.dat` files            |
| `graphs_construction.ipynb`     | Creates large tissue graphs from nuclei features and saves as `.pt`         |
| `subgraphs_construction.ipynb`  | Builds subgraphs using sliding window and trains GAT model (with eval)      |
| `metadata.csv`                  | Master CSV with `graph_path`, `label` used for training/testing split       |
| `requirements.txt`              | List of dependencies (Torch, PyG, TIAToolbox, etc.)                         |

---
