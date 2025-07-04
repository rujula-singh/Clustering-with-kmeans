
# 🛍️ Mall Customer Segmentation using K-Means Clustering

This project applies **K-Means clustering** to segment customers based on their **Annual Income** and **Spending Score** from the popular Mall Customers dataset. It includes data preprocessing, visualization, optimal cluster selection using the **Elbow Method** and **Silhouette Score**, and final cluster interpretation.

---

##  Dataset Overview

- **File**: `Mall_Customers.csv`
- **Source**: Mall Customer data (unsupervised learning dataset)
- **Important Features** used:
  - `Annual Income (k$)`
  - `Spending Score (1-100)`
- **Optional**: `Gender` (used only for interpreting clusters)

---

##  Preprocessing Steps

- Extracted features: `Annual Income` and `Spending Score`
- Standardized features using `StandardScaler`
- Applied **PCA** (if necessary) for 2D visualization

---

##  Visualizations

###  Raw Customer Data
- A simple scatter plot before applying clustering to visualize data spread.

###  Elbow Method & Silhouette Scores
- **Elbow Method**: Helps identify the optimal number of clusters (K) by plotting WCSS (Within-Cluster Sum of Squares).
- **Silhouette Score**: Measures how well each sample lies within its cluster (closer to 1 is better).

###  Final Cluster Visualization
- Plots the clusters with distinct colors.
- Yellow stars (`*`) represent **centroids** of each cluster.

---

##  Plots

###  Customer Data Before Clustering
![Customer Data Before Clustering](customer%20data%20before%20clustering.png)

###  Elbow Method and Silhouette Scores
![Elbow Method and Silhouette Scores](elbow%20method%20and%20silhouette%20scores.png)

###  Final Customer Clusters
![Customer Clusters](customer%20clusters.png)

---

##  Optimal K and Evaluation

- **Chosen K**: `5`
- **Silhouette Score**: ~`0.55`  
  (Closer to `1` indicates well-separated clusters)

---

##  Cluster Analysis (If Gender Column Exists)

If the `Gender` column is available in the dataset:

- The code groups data by `Cluster` and reports:
  - Mean and median `Annual Income`
  - Mean and median `Spending Score`
  - Most common gender in each cluster

---

##  Libraries Used

- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`

---
