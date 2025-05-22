# 🧠 ANN for Clustering Customer Segments

**Course:** DA221 - Introduction to Artificial Intelligence  
**Team Name:** *The Upside Down*

## 👥 Team Members
- M. Abhiram (230150015)
- K. Ashmita (230150014)
- B. Cherish (230150007)

---

## 📌 Overview

This project explores **unsupervised clustering using Artificial Neural Networks (ANNs)** by leveraging an internal loss function tailored for clustering quality, specifically applied to customer segmentation.

Traditionally, clustering is done using methods like K-Means or DBSCAN. However, we propose a **novel approach where the ANN is trained using a custom-defined clustering-based loss function**—the **Min-Max-Jump (MMJ) K-Means Loss**—enabling it to act as a fully unsupervised clustering system.

---

## 📊 Dataset

**Mall Customers Segmentation** ([Kaggle Link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python))

- Contains 200 customer records with the following features:
  - `CustomerID` (ignored)
  - `Gender` (One-hot encoded)
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

---

## 🔧 Methodology

### 1. Exploratory Data Analysis (EDA)
- Plotted distributions and scatterplots
- Observed weak correlations; e.g., age negatively correlates with spending score
- Identified potential cluster formations based on visual inspection

### 2. Dimensionality Reduction
Implemented **Autoencoders** using PyTorch:
- Compress high-dimensional features into a latent representation
- Compared three architectures (shallow to deep) to prevent under/overfitting
- Selected the middle-depth autoencoder (4→3→2→3→4) based on reconstruction loss and latent structure

### 3. Clustering via ANN
Designed 4 different ANN architectures:
- Each architecture had 7 output neurons (max cluster assumption)
- Applied **softmax** to output for fuzzy clustering
- Implemented **MMJ K-Means Loss** as training objective

### 4. Evaluation Metric
Used **Silhouette Coefficient** to assess internal cluster quality:
- Values range from -1 to 1
- > 0.5: optimal clusters
- 0–0.5: moderate clusters
- < 0: poor clusters

The best-performing ANN achieved a **Silhouette Score of 0.46**.

---

## 📈 Results & Insights

### ANN divided data into 4 major clusters:

| Cluster | Dominant Gender | Avg. Age | Avg. Income (k$) | Avg. Spending Score | Insight |
|--------|------------------|----------|------------------|----------------------|--------|
| 1 (Purple) | Male | 55 | 49.4 | 42.2 | Target vintage products |
| 2 (Lilac) | Male | 27 | 32.4 | 59.5 | Offer budget-friendly deals |
| 3 (Green) | Male | 30.6 | 78.6 | 71.8 | Focus on premium, trendy items |
| 4 (Yellow) | Female | 41.4 | 85.5 | 21.0 | Upsell premium items |

---

## 🛠️ Technologies & Libraries

- **Language:** Python  
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `torch` (PyTorch)

---

## 📂 File Structure

```
├── ANN_Clustering_Code.ipynb     # Jupyter notebook with full implementation
├── Mall_Customers.csv            # Dataset file
├── README.md                     # You are here
```

---

## 💻 How to Run

### 1. Setup
Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

### 2. Execution
- Open `ANN_Clustering_Code.ipynb` in Jupyter Notebook
- Ensure `Mall_Customers.csv` is in the same directory
- Run the notebook top to bottom

---

## 📚 References

- [Clustering with Neural Network and Index (arXiv)](https://arxiv.org/pdf/2212.03853)
- [Kaggle Dataset: Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org)
- [Matplotlib Documentation](https://matplotlib.org)

---

## 🤝 Contribution

All three team members collaborated equally on:
- Data preprocessing
- Model design and implementation
- Visualizations and evaluation
- Documentation and analysis

---

## 📝 License

This project is developed as a course assignment and is intended for educational purposes.

---
