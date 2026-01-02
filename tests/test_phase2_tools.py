import pytest
import pandas as pd
import json
from src.session_manager import SessionManager

@pytest.fixture
def messy_csv(tmp_path):
    # A messy dataframe: columns with spaces, caps, empty rows?
    df = pd.DataFrame({
        "Col A": [1, 2, 3, 4],
        "Col B": ["foo", "bar", None, "baz"],  # None will be NaN
        "Risk Value": [0.1, 0.2, -0.5, 0.9]
    })
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_validation_logic(messy_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(messy_csv)

    # Define a schema that expects positive Risk Value
    schema = {
        "columns": {
            "Risk Value": {
                "type": "float",
                "checks": {"ge": 0.0}
            }
        }
    }

    # Should fail because of -0.5
    result = sm.validate_dataset(ds_id, schema)
    assert result["valid"] is False
    assert len(result["errors"]) > 0
    assert result["errors"][0]["failure_case"] == -0.5

def test_cleaning_logic(messy_csv):
    sm = SessionManager()
    ds_id = sm.load_dataset(messy_csv)

    # Original columns: "Col A", "Col B", "Risk Value"
    # pyjanitor's clean_names should make them snake_case: "col_a", "col_b", "risk_value"

    summary = sm.clean_dataset(ds_id, ["clean_names"])

    df = sm.get_dataset(ds_id)
    assert "col_a" in df.columns
    assert "risk_value" in df.columns
    assert "Col A" not in df.columns
    assert "Cleaned dataset" in summary
