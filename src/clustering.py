from sklearn.cluster import KMeans


def apply_clustering(df, X_cluster_scaled):
    df = df.copy()

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['customer_segment'] = kmeans.fit_predict(X_cluster_scaled)

    return df