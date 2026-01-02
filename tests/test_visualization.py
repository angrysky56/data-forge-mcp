import pytest
import pandas as pd
import os
from src.session_manager import SessionManager

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 25, 30, 40],
        "C": ["cat", "dog", "cat", "dog", "bird"]
    })
    path = tmp_path / "sample_viz.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_generate_scatter(sample_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(sample_csv)

    path = sm.generate_chart(ds_id, x="A", y="B", chart_type="scatter", title="Test Scatter")
    assert os.path.exists(path)
    assert path.endswith(".png")
    # Clean up optional but handled by tempfile usually
    os.remove(path)

def test_generate_hist(sample_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(sample_csv)

    path = sm.generate_chart(ds_id, x="B", chart_type="hist", title="Test Hist")
    assert os.path.exists(path)
    os.remove(path)

def test_generate_error_missing_col(sample_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(sample_csv)

    with pytest.raises(ValueError):
        sm.generate_chart(ds_id, x="Z", y="B", chart_type="scatter") # Z likely doesn't exist, seaborn might catch specific error or ValueError if we checked cols.
        # Actually our implementation relies on seaborn raising error or simple check.
        # Current implementation does not explicitly check col existence against schema, but seaborn will fail.
        # Let's adjust test expectation: our wrapper catches general exceptions in server, but here direct call raises.
