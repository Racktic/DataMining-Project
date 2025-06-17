import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ClusterVisualizer import ClusterVisualizer


class KMeansClassifier:
    def __init__(self, n_clusters=10, max_iters=30, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.cluster_labels = None
        self.visualizer = None

    def initialize_centroids(self, X) -> np.ndarray:
        """
        Randomly select initial centroids from the data
        """
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X) -> np.ndarray:
        """
        Assign clusters to the data based on the closest centroid
        """
        nums = X.shape[0]
        distances = np.empty((nums, self.n_clusters))  # Pre-allocate memory
        # Compute distance one by one [but slow:(]
        # Otherwise error givens in WSL
        # distances = np.linalg.norm(images[:, np.newaxis] - self.centroids, axis=2)
        for i in range(nums):
            distances[i] = np.linalg.norm(X[i] - self.centroids, axis=1)

        return np.argmin(distances, axis=1)

    def update_centroids(self, X, clusters_labels):
        """
        Update centroids based on the mean of the assigned images
        """
        new_centroids = np.array([X[clusters_labels == i].mean(axis=0) if len(X[clusters_labels == i]) > 0 else self.centroids[i]
                                  for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X, y, model_path=None, load_from_path=False):
        """
        Train the KMeans classifier on the dataset
        - Initialize centroids
        - do until convergence or max_iters:
            - Assign clusters
            - Update centroids
        - Assign labels to clusters based on the most frequent label in each cluster
        """
        # load the trained model
        if load_from_path and model_path and os.path.exists(model_path):
            # Load the model from the saved file
            with open(model_path, 'rb') as f:
                saved_model = pickle.load(f)
                self.centroids = saved_model['centroids']
                self.cluster_labels = saved_model['cluster_labels']
            print(f"Model loaded from {model_path}")
            return

        X = np.array(X).reshape(len(X), -1)  # Flatten input
        y = np.array(y)

        self.centroids = self.initialize_centroids(X)
        for _ in tqdm(range(self.max_iters)):
            cluster_assignments = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, cluster_assignments)

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:  # converge
                break
            self.centroids = new_centroids

        # map each cluster to the most frequent label
        self.cluster_labels = {}
        for i in range(self.n_clusters):
            cluster_members = y[np.where(cluster_assignments == i)]
            # assign the most frequent label to that cluster
            self.cluster_labels[i] = np.bincount(cluster_members.astype(
                int)).argmax() if len(cluster_members) > 0 else -1

        # Save the trained model
        if model_path:
            model_data = {
                'centroids': self.centroids,
                'cluster_labels': self.cluster_labels
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {model_path}")

    def predict(self, X):
        """
        predict the given X to the label
        """
        X = X.flatten().reshape(1, -1)  # Flatten input image
        distances = np.linalg.norm(X - self.centroids, axis=1)
        cluster = np.argmin(distances)
        return self.cluster_labels.get(cluster, -1)  # Return assigned digit

    def visualize_clusters(self, X, y, n_components):
        """
        Visualizes clusters by first reducing dimensions with PCA to 30,
        then with t-SNE to the final n_components (e.g., 2).
        """
        X = np.array(X).reshape(len(X), -1)  # Flatten X
        all_data = np.vstack([X, self.centroids])

        print(f"Reducing dims from {len(X[0])} to 30 with PCA...")
        pca = PCA(n_components=30, random_state=self.random_state)
        pca_reduced_all = pca.fit_transform(all_data)
        print("Reducing dims to 2 with TSNE")
        reduced_all = TSNE(n_components=n_components, verbose=1,
                           random_state=self.random_state).fit_transform(pca_reduced_all)
        reduced_X, reduced_centroids = reduced_all[:-len(self.centroids)], reduced_all[-len(self.centroids):]

        print("Plotting graph")
        unique_labels = np.unique(y)
        color_map = {label: plt.cm.get_cmap('tab10')(i / len(unique_labels))
                     for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in y]

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_X[:, 0], reduced_X[:, 1],
                    c=point_colors, alpha=0.3, s=2, label='Data Points')

        for i, centroid in enumerate(reduced_centroids):
            plt.scatter(centroid[0], centroid[1], c=[color_map.get(self.cluster_labels.get(i), 'black')],
                        marker='o', s=75, linewidths=1, edgecolors='black', facecolors='white', label=f'Centroid {i}')

        legend_labels = {color_map[label]: f'Label {label}' for label in unique_labels}
        legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label)
                          for color, label in legend_labels.items()]
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f'K-Means Clustering of MNIST Data in 2D with PCA using {self.n_clusters} clusters')
        plt.legend(handles=legend_patches, title="True Labels")
        plt.show()


if __name__ == '__main__':
    with open('ts_train.pkl', 'rb') as f:
        train_data = pickle.load(f)

    print(train_data['label'].value_counts())
    ratio = 0.8
    X = train_data.drop('label', axis=1)
    y = train_data['label']
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=ratio,
        stratify=y,  # This is the key parameter to ensure proportional splitting
        random_state=42  # For reproducible results
    )

    clf = KMeansClassifier(n_clusters=10)
    clf.fit(X_train, y_train, model_path='kmeans_model_10.pkl', load_from_path=True)

    clf.visualize_clusters(X_train, y_train, n_components=2)

    score = balanced_accuracy_score(y_val, [clf.predict(x) for x in X_val.values])
    acc = accuracy_score(y_val, [clf.predict(x) for x in X_val.values])
    print(f"Balanced accuracy score: {score:.4f}")
    print(f"Accuracy score: {acc:.4f}")
