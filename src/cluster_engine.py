from sklearn.cluster import KMeans

def discover_clusters(embeddings, num_clusters=5):
    """
    Run K-Means clustering on embeddings.
    
    Minimizes the within-cluster sum of squares (WCSS) to group similar semantic vectors:
    $$J = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2$$
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels