
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs  

   
df = pd.read_csv('./Mall_Customers.csv')
    
# Display basic info
("\nDataset Info:")
print(df.info())
    
X = df.iloc[:, [3, 4]].values  # Using columns 3 (Annual Income) and 4 (Spending Score)
    

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional PCA for 2D visualization (if more than 2 features)
if X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_scaled)
else:
    X_vis = X_scaled

# Visualize the raw data
plt.figure(figsize=(10, 6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], s=50, alpha=0.7)
plt.title("Customer Data Before Clustering")
plt.xlabel("Feature 1 (or PC1)")
plt.ylabel("Feature 2 (or PC2)")
plt.grid(True)
plt.savefig("customer data before clustering")
plt.show()

# Determine optimal K using Elbow Method
wcss = [] 
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette score
    if k > 1: 
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    else:
        silhouette_scores.append(0)

# Plot the Elbow Method graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal K')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range[1:], silhouette_scores[1:], 'go-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different K Values')
plt.tight_layout()
plt.savefig("elbow method and silhouette scores")
plt.show()


optimal_k = 5  

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

for i in range(optimal_k):
    plt.scatter(X_vis[y_kmeans == i, 0], X_vis[y_kmeans == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='yellow', marker='*', label='Centroids')

plt.title('Customer Clusters')
plt.xlabel("Feature 1 (or PC1)")
plt.ylabel("Feature 2 (or PC2)")
plt.legend()
plt.grid(True)
plt.savefig("customer clusters")
plt.show()

# Evaluate clustering
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"\nClustering Evaluation for K={optimal_k}:")
print(f"Silhouette Score: {silhouette_avg:.3f}")
print("(Closer to 1 indicates better defined clusters)")

# Interpretation of clusters (if using mall data)
if 'df' in locals() and 'Gender' in df.columns:
    print("\nCluster Interpretation:")
    df['Cluster'] = y_kmeans
    cluster_stats = df.groupby('Cluster').agg({
        'Annual Income (k$)': ['mean', 'median'],
        'Spending Score (1-100)': ['mean', 'median'],
        'Gender': lambda x: x.mode()[0]
    })
    print(cluster_stats)