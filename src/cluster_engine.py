from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
import numpy as np

def discover_clusters(embeddings, algorithm='kmeans', num_clusters=5, min_cluster_size=5):
    """
    Run Clustering (K-Means or HDBSCAN) and PCA for visualization.
    
    Returns:
        labels: Cluster IDs for each document
        coords: 2D PCA coordinates (x, y) for plotting
    """
    # 1. Dimensionality Reduction (PCA) for Visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    # 2. Clustering Algorithm
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(embeddings)
        
    elif algorithm == 'hdbscan':
        model = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
        labels = model.fit_predict(embeddings)
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return labels, coords