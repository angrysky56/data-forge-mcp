import pytest
import pandas as pd
from src.session_manager import SessionManager

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_load_dataset(sample_csv):
    sm = SessionManager()
    dataset_id = sm.load_dataset(sample_csv, alias="test_data")
    assert dataset_id.startswith("df_")

    df = sm.get_dataset(dataset_id)
    assert len(df) == 3
    assert list(df.columns) == ["a", "b"]

def test_list_datasets(sample_csv):
    sm = SessionManager()
    sm.load_dataset(sample_csv, alias="test_data")
    datasets = sm.list_datasets()
    assert len(datasets) == 1
    assert datasets[0]["alias"] == "test_data"
    assert datasets[0]["engine"] == "pandas"

def test_get_info(sample_csv):
    sm = SessionManager()
    dataset_id = sm.load_dataset(sample_csv)
    info = sm.get_dataset_info(dataset_id)
    assert "RangeIndex: 3 entries" in info
    assert "Data columns (total 2 columns):" in info

def test_invalid_engine():
    sm = SessionManager()
    with pytest.raises(NotImplementedError):
        sm.load_dataset("foo.csv", engine="vaex")
