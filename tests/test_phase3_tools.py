import pytest
import polars as pl
import pandas as pd
from src.session_manager import SessionManager

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["a", "b", "c", "d", "e"],
        "C": [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_load_polars(sample_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(sample_csv, engine="polars")

    df = sm.get_dataset(ds_id)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 5
    assert "A" in df.columns

def test_profiling(sample_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(sample_csv) # Default pandas

    profile = sm.get_profile(ds_id)
    assert "description" in profile
    assert "variables" in profile
    assert "alerts" in profile
    assert "A" in profile["variables"]

def test_profiling_polars_conversion(sample_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(sample_csv, engine="polars")

    # Should automatically convert to pandas for profiling
    profile = sm.get_profile(ds_id)
    assert "variables" in profile
    assert "A" in profile["variables"]
