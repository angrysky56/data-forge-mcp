from src.session_manager import SessionManager
import pandas as pd
import os

def verify_discovery():
    print("--- 1. Setup ---")
    sm = SessionManager()

    # Create synthetic "clusters" of text
    cluster_a = ["The cat sat on the mat", "A feline resting on a rug", "Kittens love mats"]
    cluster_b = ["Quantum physics entanglement", "Schrodinger's cat theory", "Superposition states"]
    cluster_c = ["Delicious pasta recipes", "How to cook spaghetti", "Italian tomato sauce"]

    # Create a "loop" or at least a diverse structure?
    # Hard to guarantee a void with 10 sentences, but we can verify the PIPELINE works.
    all_texts = cluster_a + cluster_b + cluster_c

    df = pd.DataFrame({"content": all_texts, "id": range(len(all_texts))})

    ds_id = "test_discovery"
    sm._registry[ds_id] = df
    sm._metadata[ds_id] = {"rows": len(df)}

    print("--- 2. Running Scan ---")
    try:
        report = sm.scan_voids(ds_id, text_column="content")
        print("Report received:")
        print(report)

        # Extract filepath from report to verify
        # Naive extraction or just check dir
        if "Visualization: " in report:
            path = report.split("Visualization: ")[1].split("\n")[0].strip()
            print(f"Checking file: {path}")
            if os.path.exists(path):
                print("SUCCESS: Visualization created.")
            else:
                print("FAILURE: File not found.")
    except Exception as e:
        print(f"Scan failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_discovery()
