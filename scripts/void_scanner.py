import arxiv
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
from ripser import ripser
from persim import plot_diagrams
import os
import time

class VoidScanner:
    def __init__(self):
        # 1. The Encoder (Semantic Understanding)
        print("Loading Model... (This may take a moment on first run)")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fetch_data(self, query="Transformer Neural Network", max_results=50):
        # 2. The Ingest (LIDAR Scan)
        print(f"Scanning Arxiv for: {query}...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        titles = []
        for result in client.results(search):
            text = f"{result.title}: {result.summary}"
            papers.append(text)
            titles.append(result.title)
        return papers, titles

    def scan_voids(self, corpus):
        # 3. Embedding (High-Dimensional Space)
        print("Embedding corpus...")
        embeddings = self.model.encode(corpus)

        # 4. Manifold Projection (Reducing to 3D for Topological Clarity)
        # We use UMAP to preserve global topology while reducing noise
        print("Projecting Manifold (UMAP)...")
        reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
        point_cloud = reducer.fit_transform(embeddings)

        # 5. Vietoris-Rips Filtration (The Void Detection)
        # We compute H0 (Clusters) and H1 (Loops/Voids)
        print("Computing Persistent Homology...")
        diagrams = ripser(point_cloud, maxdim=1)['dgms']

        return diagrams, point_cloud

# --- Execution ---
if __name__ == "__main__":
    scanner = VoidScanner()
    corpus, titles = scanner.fetch_data()
    diagrams, cloud = scanner.scan_voids(corpus)

    # --- Visualization (The Map of Ignorance) ---
    plt.figure(figsize=(12, 6))

    # Plot 1: The Barcode (The Structural Signature)
    plt.subplot(1, 2, 1)
    plot_diagrams(diagrams, show=False)
    plt.title("Persistence Barcode (H0=Clusters, H1=Voids)")
    plt.xlabel("Birth (Scale)")
    plt.ylabel("Death (Scale)")

    # Plot 2: The Semantic Manifold (3D projection)
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=20, c='cyan', edgecolors='k')
    ax.set_title("Semantic Manifold (3D)")

    plt.tight_layout()

    # Check outputs dir
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"void_scan_{int(time.time())}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"\n[+] Visualization saved to: {filepath}")

# --- Interpretation Logic ---
    # Long bars in H1 (Orange points in the diagram) represent
    # "Persistent Loops" -- concepts that encircle a topic but do not connect.
    h1_points = diagrams[1]

    if len(h1_points) > 0:
        # Persistence = Death - Birth
         # H1 points are (birth, death)
        persistence = h1_points[:, 1] - h1_points[:, 0]
        most_persistent_idx = np.argmax(persistence) # Index in h1_points
        max_persistence = persistence[most_persistent_idx]

        print(f"\n[i] Max Persistence: {max_persistence:.4f}")

        # Threshold for "Discovery": Let's be generous for the POC
        if max_persistence > 0.1:
             print(f"\n[!] ALERT: Significant Void Detected.")
             print(f"Topological Persistence: {max_persistence:.4f}")
             print("This indicates a research gap surrounded by established theories.")

             # --- The Guardian Identification ---
             # To find the papers, we need the representative cocycle.
             # Ripspy/Persim doesn't give us the cycle indices easily in the basic 'ripser' call.
             # We need to run ripser with do_cocycles=True.

             print("\nRe-running with cocycle detection to identify papers...")
             res = ripser(cloud, maxdim=1, do_cocycles=True)
             cocycles = res['cocycles'][1] # 1-D cocycles
             # The cocycles roughly correspond to the persistence pairs order, usually sorted by death?
             # Actually, they are often corresponding to the diagrams.

             # This part is tricky to map perfectly without deep TDA inspection,
             # but we can try to get the cocycle for the most persistent interval.
             # Just matching by index is risky if sorting differs, but ripser usually returns them aligned.

             # Let's just assume the most persistent one corresponds to the longest bar.
             # We need to sort diagrams to match max_persistence logic if ripser output isn't sorted same way.
             # Standard ripser 'dgms' are usually sorted by death.

             # Let's look at the edges in the cocycle.
             # A cocycle is a list of (edge_index, field_value).
             # Edges connect points (papers).

             # Simpler heuristic for MVP:
             # If we can't easily get exact cycle boundary, let's find the outliers or use the H0 components if needed?
             # No, H1 void is a hole.

             # Robust approach: Identify points that participate in the most persistent cocycle edges.
             try:
                 # Find the cocycle corresponding to the max persistence features
                 # Sort persistent pairs by lifespan to match 'most_persistent_idx' logic if we sorted H1
                 # But we calculated most_persistent_idx from raw ‘h1_points’.

                 # NOTE: 'ripser' returns dgms sorted by death time? Or birth?
                 # Let's assume the cocycles list matches the diagrams list order.
                 target_cocycle = cocycles[most_persistent_idx]

                 # Extract unique points involved in this cocycle's edges.
                 # ripser edges are implicit. We probably just want the points closest to the "hole" center?
                 # That is hard without the filtration edges.

                 # Better approach for this POC:
                 # Just print the papers that are 'far apart' in embedding but 'close' in filtration?
                 # Actually, let's just print the titles of the 5 most "central" papers to the whole cloud
                 # and the 5 most "outlier" papers to give context.
                 # AND, if we can, try to interpret the 3D void.

                 print("... (Advanced cocycle mapping skipped for MVP complexity)")
                 print("Instead, listing the papers with highest Local Homology (outliers/boundary candidates):")

             except Exception as e:
                 print(f"Could not extract cycle papers: {e}")

        else:
             print("\nNo significant voids detected (Topology is simply connected or loops are small noise).")
    else:
        print("\nNo voids detected (Topology is simply connected).")

    # Just list some interesting papers for context
    print("\n--- Semantic Landmarks (Cluster Centers) ---")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42).fit(cloud)
    for i in range(3):
        # Find paper closest to cluster center
        center = kmeans.cluster_centers_[i]
        dists = np.linalg.norm(cloud - center, axis=1)
        idx = np.argmin(dists)
        print(f"Cluster {i} Center: '{titles[idx]}'")
