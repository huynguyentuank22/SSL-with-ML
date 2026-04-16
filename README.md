# 📄 Research Design: Semi-Supervised Learning for Tabular Model Prediction with Explainability

## 1. Problem Definition

# numeric (11 columns)
- priority
- cpus_req
- mem_req (imputer by median)
- time_limit (imputer by median)
- time_submit x3 (hour/dayofweek/month) x2 (sin/cos) = 6 columns
- eligible_delay = time_eligible - time_submit
# category (one-hot encoding) (20 columns)
- partition (5 types)
- constraint (7 types)
- flag (4 types)
- job_type (4 types)

Trong dataset hiện tại, target ban đầu là:

* `model` (26 classes)

Tuy nhiên, việc dự đoán chính xác từng model cụ thể:

* khó (multi-class nhiều lớp)
* không thực sự cần thiết về mặt ứng dụng

👉 Do đó, bài toán được **tái định nghĩa theo hướng phân cấp (hierarchical prediction)**:

### Target Levels:

1. **Fine-grained**:

   * `model` (26 classes)

2. **Mid-level (quan trọng nhất)**:

   * `model_group` (vision, language, graph)

3. **Coarse-grained**:

   * `base_model` (U-Net, VGG, Inception, Other)

---

## 2. Research Objectives

Mục tiêu chính:

* Đánh giá hiệu quả của **Self-Supervised Learning (SSL)** trên dữ liệu tabular
* So sánh với các **strong baselines (tree-based models)**
* Phân tích **representation learning**
* Đánh giá **explainability (XAI)**

---

## 3. Model Groups

### 3.1 Non-SSL Baselines

* XGBoost
* CatBoost

### 3.2 Deep Learning (No SSL)

* MLP (trained directly on raw features)

### 3.3 SSL-based Models

Pretraining + embedding:

* VIME
* SCARF
* SAINT
* SubTab
* Denoising Autoencoder
* TabNet

Pipeline:

```
SSL → embedding z → downstream classifier (MLP / Logistic)
```

---

## 4. Task Formulation

### Task 1 — Coarse Classification

* Predict: `model_group`

👉 Dễ hơn, giúp đánh giá:

* khả năng học **high-level semantics**

---

### Task 2 — Intermediate Classification

* Predict: `base_model`

👉 Đánh giá:

* khả năng học **structural similarity giữa models**

---

### Task 3 — Fine-grained Classification

* Predict: `model`

👉 Khó nhất:

* dùng để kiểm tra giới hạn của SSL

---

## 5. Research Questions (RQ)

### RQ1 — Performance

SSL-based models có outperform non-SSL baselines không?

* So sánh:

  * XGBoost / CatBoost
  * MLP (raw)
  * SSL embeddings

---

### RQ2 — When does SSL help?

SSL có hiệu quả trong điều kiện nào?

* Vary:

  * % labeled data (1%, 5%, 10%, 20%)
  * noise
  * missing features

---

### RQ3 — Representation Quality

SSL embeddings có tốt hơn raw features không?

Đánh giá qua:

* Linear probing
* Clustering (KMeans, silhouette)
* Intrinsic dimension (PCA)

---

### RQ4 — Explainability

SSL có giúp model dễ giải thích hơn không?

So sánh:

* Feature importance (tree vs SHAP)
* Noise sensitivity

---

### RQ5 — Explanation Stability

Explanation có ổn định không?

* Train nhiều seed
* So sánh similarity giữa SHAP vectors

---

### RQ6 — Representation vs Explainability

Embedding tốt hơn có dẫn đến explanation tốt hơn không?

---

## 6. Experimental Setup

### 6.1 Data Scenarios

* Semi-supervised:

  * 1%, 5%, 10%, 20% labeled

* Noise injection:

  * Gaussian noise
  * Feature corruption

* Missing data:

  * Random masking

---

### 6.2 Evaluation Metrics

#### Performance:

* Accuracy
* F1-score

#### Representation:

* Linear probing accuracy
* Clustering score

#### Explainability:

* SHAP importance
* Stability (cosine similarity)
* Sparsity (#important features)

---

## 7. XAI Analysis Design

### 7.1 Feature-level Explanation

* SHAP on raw input

So sánh:

* Tree-based vs SSL-based

---

### 7.2 Latent-level Explanation

* SHAP on embedding z

Phân tích:

* dimension nào quan trọng
* embedding có sparse không

---

### 7.3 Feature → Latent Mapping

Phương pháp:

* perturb từng feature
* đo Δz

Output:

* heatmap feature → latent dimension

---

### 7.4 Stability Analysis

* Train nhiều lần
* So sánh SHAP vectors

---

## 8. Expected Contributions

* So sánh toàn diện SSL vs non-SSL cho tabular data
* Phân tích representation beyond accuracy
* Nghiên cứu ảnh hưởng của SSL đến explainability
* Xác định điều kiện SSL thực sự hiệu quả

---

## 9. Key Insights to Validate

* SSL giúp nhiều khi label ít

* Tree-based vẫn mạnh khi label đủ

* SSL embeddings:

  * smooth hơn
  * ít phụ thuộc noise

* Explanation:

  * ổn định hơn
  * dễ interpret hơn

---

## 10. Conclusion

Bài toán không chỉ là:

> "Model nào tốt nhất?"

Mà là:

> "SSL thay đổi cách model hiểu dữ liệu như thế nào?"

---

## 11. Future Work

* Graph-based SSL
* Hybrid SSL + tree models
* Explainability-aware training

---

