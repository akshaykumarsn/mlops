# clustering.py
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

def train_clustering():
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    model = KMeans(n_clusters=4)
    model.fit(X)
    return model

def predict_cluster(model, data):
    return model.predict(np.array([data]))
