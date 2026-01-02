import numpy as np
import matplotlib.pyplot as plt
import umap
from ripser import ripser
from persim import plot_diagrams
from sentence_transformers import SentenceTransformer
import os
import time
from typing import List, Tuple, Dict, Any, Optional

class VoidScanner:
    """
    Topological Data Analysis engine to detect semantic voids in text corpora.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Lazy loading to avoid overhead if tool is not used
            print(f"Loading SentenceTransformer: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def scan(self,
             texts: List[str],
             titles: Optional[List[str]] = None,
             output_dir: str = "outputs",
             max_points: int = 1000,
             persistence_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Performs the void scan pipeline: Embed -> UMAP -> TDA -> Report.

        Args:
            texts: List of text content to analyze.
            titles: Optional labels for the texts (used for landmarks).
            output_dir: Directory to save the visualization.
            max_points: Max number of data points to process (TDA is expensive).
            persistence_threshold: Min persistence to report a void.

        Returns:
            Dict containing report details and path to image.
        """

        # 1. Sampling (Volume Control)
        n_samples = len(texts)
        if n_samples > max_points:
            print(f"Dataset too large for TDA ({n_samples} > {max_points}). Sampling...")
            indices = np.random.choice(n_samples, max_points, replace=False)
            corpus = [texts[i] for i in indices]
            if titles:
                corpus_titles = [titles[i] for i in indices]
            else:
                corpus_titles = [f"Item {i}" for i in indices]
        else:
            corpus = texts
            corpus_titles = titles if titles else [f"Item {i}" for i in range(n_samples)]

        # 2. Embedding (The Map)
        print("Embedding corpus...")
        embeddings = self.model.encode(corpus)

        # 3. Manifold Projection (UMAP)
        # Reduce to 3D for checking structure and TDA processing
        print("Projecting Manifold (UMAP)...")
        # n_jobs=1 to suppress warnings and ensure stability
        reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
        input_data = reducer.fit_transform(embeddings) # (N, 3) point cloud

        # 4. TDA (The Scan)
        print("Computing Persistent Homology...")
        # maxdim=1 computes H0 (clusters) and H1 (loops/voids)
        result = ripser(input_data, maxdim=1)
        diagrams = result['dgms']

        # 5. Analysis & Visualization
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = int(time.time())
        filename = f"void_scan_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.figure(figsize=(12, 6))

        # Subplot 1: Barcode
        plt.subplot(1, 2, 1)
        plot_diagrams(diagrams, show=False)
        plt.title("Persistence Barcode")

        # Subplot 2: Manifold
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(input_data[:, 0], input_data[:, 1], input_data[:, 2],
                   s=20, c='cyan', edgecolors='k', alpha=0.7)
        ax.set_title("Semantic Manifold (3D projection)")

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        # 6. Interpretation
        report = {
            "image_path": filepath,
            "n_samples": len(corpus),
            "void_detected": False,
            "max_persistence": 0.0,
            "landmarks": []
        }

        # Analyze H1 (loops)
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            h1 = diagrams[1]
            persistence = h1[:, 1] - h1[:, 0]
            max_p = np.max(persistence)
            report["max_persistence"] = float(max_p)

            if max_p > persistence_threshold:
                report["void_detected"] = True

        # Find Landmarks (Cluster Centers)
        from sklearn.cluster import KMeans
        # Use simple K-Means to find 'representative' points in the manifold
        n_clusters = min(3, len(corpus))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(input_data)

        for i in range(n_clusters):
            center = kmeans.cluster_centers_[i]
            dists = np.linalg.norm(input_data - center, axis=1)
            idx = np.argmin(dists)
            report["landmarks"].append(corpus_titles[idx])

        return report
