# RND: Optimised and More Informative GNN for Breast Cancer Diagnosis And Analysis
## 🏥 Breast Cancer Subtype Classification Using Graph-Based Methods on H&E Images 🔬📊

## 📌 1. Introduction & Problem Definition
Breast cancer classification using histopathological images is crucial for accurate diagnosis and treatment. Traditional deep learning methods have demonstrated remarkable performance in classifying cancer subtypes. However, they lack interpretability and an ability to capture biological interactions explicitly.

### ⚠️ Challenges in Current Approaches:
- **⚖️ Imbalanced Data:** Medical datasets often contain an unequal distribution of subtypes, affecting recall and model generalization.
- **🧐 Lack of Interpretability:** Deep learning models act as black boxes, making it difficult to derive insights into biological interactions.
- **🧬 Complexity of Biological Interactions:** Tumor microenvironments involve complex interactions between nuclei, tissues, and other biological structures, which CNNs and standard ML models fail to capture.

### ✅ Proposed Solution
1. **🚀 Improve classification accuracy and recall** by leveraging advanced deep learning models that can be trained on imbalanced data.
2. **🔗 Incorporate graph-based techniques** to model biological entity interactions, improving interpretability and enabling link prediction in tissue structures.
3. **🕵️ Utilize state-of-the-art models for graph learning** to better understand cancer progression and cellular organization.

---

## 📚 2. Literature Review & Related Works
### 🤖 Deep Learning in Breast Cancer Classification
- **📝 BreakHis dataset analysis and CNN architectures:**
  - *Spanhol, F.A., et al. (2016), “A dataset for breast cancer histopathological image classification,” IEEE Transactions on Biomedical Engineering.*
  - Analyzes the BreakHis dataset with deep CNNs like AlexNet, VGG, and ResNet.

- **⚖️ Handling Imbalanced Datasets in Medical Imaging:**
  - *Buda, M., Maki, A., & Mazurowski, M.A. (2018), “A systematic study of the class imbalance problem in convolutional neural networks,” Neural Networks.*
  - Explores techniques such as weighted loss functions, data augmentation, and oversampling.

- **📈 SOTA CNN Models for Medical Image Classification:**
  - *Tan, M., & Le, Q. (2019), “EfficientNet: Rethinking model scaling for convolutional neural networks,” ICML.*
  - Proposes EfficientNet, which improves performance with fewer parameters.

### 📊 Graph-based Learning in Histopathology
- **🧠 Graph Representation Learning for Histopathological Images:**
  - *Lu, M.Y., et al. (2020), “AI-powered computational pathology using weakly supervised learning,” Nature Medicine.*
  - Introduces Graph Neural Networks (GNNs) for histopathology.

- **🕵️ Graph Construction from H&E Images:**
  - *Zhou, Y., et al. (2019), “Graph-based representation for histopathological images,” Medical Image Analysis.*
  - Describes methods to convert H&E images into graphs for better insights.

- **🔗 Link Prediction & Graph-based Feature Learning in Cancer Diagnosis:**
  - *Chen, R.J., et al. (2021), “Pathomic fusion: Combining deep features and graphs for histopathology analysis,” CVPR.*
  - Proposes multimodal fusion of graph-based features and CNN embeddings.

---

## 🔬 3. Methodology
### 🛠️ Step 1: Preprocessing and Graph Construction
- **🧹 Preprocessing:**
  - Convert images to grayscale or RGB standardization.
  - Normalize pixel intensities.
  - Data augmentation (rotation, flipping, scaling) to handle class imbalance.

- **📌 Graph Representation of Images:**
  - Extract nuclei and cellular structures using **CellViT** (or a nuclei segmentation tool like HoVer-Net).
  - Construct graphs where:
    - **🟢 Nodes** represent detected nuclei or tissue regions.
    - **🔵 Edges** represent spatial relationships, morphological similarities, or distance-based connections.

### ⚡ Step 2: Deep Learning Baseline Model
- **🖼️ Backbone Architectures:** ResNet-50, EfficientNet-B3, Vision Transformers (ViTs).
- **⚖️ Class Imbalance Handling:**
  - Weighted Cross-Entropy loss 🏋️
  - Focal Loss 🎯
  - SMOTE (Synthetic Minority Over-sampling) 🔄

### 🏗️ Step 3: Graph Neural Network (GNN) Model
1. **📍 Graph Construction Approaches:**
   - **🔺 Delaunay Triangulation:** Connects nearest nuclei to capture tissue structure.
   - **📌 K-Nearest Neighbors (KNN) Graphs:** Defines connectivity based on spatial proximity.
   - **🔬 Graph Attention Networks (GAT):** Learns node importance for better feature extraction.

2. **🧠 Graph Learning Models:**
   - **📡 Graph Convolutional Networks (GCNs):** Captures global structure.
   - **🎯 Graph Attention Networks (GATs):** Assigns importance to different cellular interactions.
   - **📢 GraphSAGE:** Enables inductive learning from large-scale graphs.

### 🔄 Step 4: Fusion of CNN and GNN Features
- Extract **CNN-based feature vectors** from an intermediate layer.
- Use these features as **node embeddings** for the GNN.
- Combine **graph representations** with CNN embeddings for better classification.

### 🧐 Step 5: Graph-based Analysis & Interpretability
- Use **📌 Node Classification** to identify key cellular structures contributing to malignancy.
- Apply **🧩 Graph Clustering** to discover tumor subregions with high similarity.
- Perform **🔗 Link Prediction** to infer potential interactions between biological entities.

---

## 📊 4. Model Training & Evaluation
### ✅ Evaluation Metrics:
- **🏆 For Classification:** Accuracy, F1-score, Precision-Recall AUC (for imbalanced data).
- **🔬 For Graph-based Learning:** Node classification accuracy, clustering metrics, edge reconstruction error.

### 🔍 Experiments to Perform:
- **CNN-only vs. CNN + GNN models.**
- **Effect of different graph construction techniques on performance.**
- **Impact of imbalanced learning techniques on recall improvement.**

---

## 🔧 5. Tools & Implementation
- **🤖 Deep Learning:** TensorFlow / PyTorch
- **🕸️ Graph Learning:** PyTorch Geometric (PyG), DGL (Deep Graph Library)
- **🖼️ Image Processing:** OpenCV, scikit-image
- **🔄 Data Augmentation:** Albumentations
- **📊 Evaluation & Visualization:** Matplotlib, seaborn, t-SNE for embedding visualization

---

## 🎯 6. Expected Outcome
- **🚀 High recall and robustness to imbalanced data.**
- **🔗 Graph-based model for histopathological image analysis.**
- **🕵️ Insights into tumor progression and cellular relationships.**

---

## 🔮 7. Future Work & Extensions
- **🔍 Explainability:** Use SHAP/Grad-CAM for interpretability.
- **🔬 Multi-modal learning:** Integrate genomic data for a comprehensive analysis.
- **🧠 Weakly-supervised learning:** Utilize weak labels to improve dataset annotation.

---

## 📌 8. Conclusion
This project integrates **deep learning with graph-based learning** to improve breast cancer classification while enhancing **biological insights**. The fusion of **CNNs for feature extraction** and **GNNs for biological interaction modeling** is expected to produce a **clinically relevant and interpretable model**. 🚀
