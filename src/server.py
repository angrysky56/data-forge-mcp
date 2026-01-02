from mcp.server.fastmcp import FastMCP
from src.session_manager import SessionManager
from typing import Literal, Optional, Dict, List, Any, Union
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Server
mcp = FastMCP("data-forge")

# Global State
session_manager = SessionManager()

@mcp.tool()
def load_data(file_path: str, alias: Optional[str] = None, engine: Literal["pandas", "polars"] = "pandas") -> str:
    """
    Loads a dataset file (CSV/Parquet) into the server's working memory.

    Args:
        file_path: Absolute path to the data file.
        alias: Optional human-readable name for the dataset.
        engine: The processing engine to use. Supports 'pandas' (default) and 'polars' (high performance).

    Returns:
        A unique dataset_id (e.g., 'df_a1b2c3d4') to be used in subsequent tool calls.
    """
    try:
        dataset_id = session_manager.load_dataset(file_path, alias, engine)
        return f"Successfully loaded dataset '{alias or file_path}' using {engine}. ID: {dataset_id}"
    except Exception as e:
        return f"Error loading data: {str(e)}"

@mcp.tool()
def get_dataset_info(dataset_id: str) -> str:
    """
    Returns the schema and summary (df.info()) of a loaded dataset.

    Args:
        dataset_id: The ID returned by load_data.
    """
    try:
        return session_manager.get_dataset_info(dataset_id)
    except Exception as e:
        return f"Error retrieving info: {str(e)}"

@mcp.tool()
def list_active_datasets() -> str:
    """
    Lists all currently loaded datasets and their IDs.
    """
    datasets = session_manager.list_datasets()
    if not datasets:
        return "No datasets currently loaded."

    report = "Active Datasets:\n"
    for d in datasets:
        # Include engine in report
        engine_info = f" [{d.get('engine', 'unknown')}]"
        report += f"- [{d['id']}] {d['alias']} ({d['rows']} rows, {len(d['columns'])} cols){engine_info}\n"
    return report


@mcp.tool()
def validate_dataset(dataset_id: str, schema: Dict[str, Any]) -> str:
    """
    Validates the dataset against a provided schema using Pandera.

    Args:
        dataset_id: The ID of the dataset to validate.
        schema: A dictionary defining the schema (columns, checks, coercion).
                Example: {"columns": {"col_a": {"type": "int", "checks": {"ge": 0}}}}
    """
    try:
        result = session_manager.validate_dataset(dataset_id, schema)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error validating data: {str(e)}"

@mcp.tool()
def clean_dataset(dataset_id: str, operations: List[Union[str, Dict[str, Any]]]) -> str:
    """
    Applies a sequence of Pyjanitor cleaning functions to the dataset.

    Args:
        dataset_id: The ID of the dataset.
        operations: List of operations. Can be a string (e.g. "clean_names") or a dict
                    (e.g. {"method": "currency_column_to_numeric", "args": ["Price"]})
    """
    try:
        return session_manager.clean_dataset(dataset_id, operations)
    except Exception as e:
        return f"Error cleaning data: {str(e)}"

@mcp.tool()
def get_dataset_profile(dataset_id: str) -> str:
    """
    Generates a statistical profile of the dataset using YData Profiling.
    Returns a JSON summary of key insights (alerts, variable list).

    Args:
        dataset_id: The ID of the dataset.
    """
    try:
        result = session_manager.get_profile(dataset_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error profiling data: {str(e)}"

@mcp.tool()
def generate_chart(dataset_id: str,
                   chart_type: Literal["scatter", "line", "bar", "hist", "box"],
                   x: Optional[str] = None,
                   y: Optional[str] = None,
                   title: Optional[str] = None) -> str:
    """
    Generates a chart from the dataset and saves it as an image.

    Args:
        dataset_id: The ID of the dataset.
        chart_type: Type of chart to generate.
        x: Column name for X axis.
        y: Column name for Y axis (optional for some charts).
        title: Title of the chart.

    Returns:
        Absolute path to the generated PNG image.
    """
    try:
        image_path = session_manager.generate_chart(dataset_id, x, y, chart_type, title)
        return f"Chart generated successfully. Image saved to: {image_path}"
    except Exception as e:
        return f"Error generating chart: {str(e)}"

@mcp.tool()
def scan_semantic_voids(dataset_id: str, text_column: str) -> str:
    """
    Performs Topological Data Analysis to find "semantic voids" or gaps in the dataset's text column.
    Useful for identifying missing research topics, unaddressed customer complaints, or concept holes.
    Generates a persistence barcode and 3D manifold plot.
    """
    return session_manager.scan_voids(dataset_id, text_column)

@mcp.tool()
def run_sql_query(query: str, dataset_id: Optional[str] = None) -> str:
    """
    Executes a SQL query on your datasets using DuckDB.
    Use this to filter, aggregate, join, or reshape data.

    You can reference datasets by their ID (e.g., 'SELECT * FROM df_a1b2').
    If you provide a 'dataset_id' argument, you can refer to it as table 'this'.
    """
    result = session_manager.query_data(query, dataset_id)
    if isinstance(result, dict):
        return f"Query Result ({result['rows_returned']} rows):\n{result['preview']}"
    return str(result)

@mcp.tool()
def extract_signals(dataset_id: str, value_column: str, id_column: Optional[str] = None, sort_column: Optional[str] = None) -> str:
    """
    Extracts time-series signals (features) using tsfresh.

    Args:
        dataset_id: The ID of the dataset.
        value_column: Column containing the time-series values.
        id_column: Column identifying separate time series (e.g. 'symbol').
        sort_column: Column to sort by (usually time/date).

    Returns:
        JSON summary of the extraction result.
    """
    try:
        result = session_manager.extract_signals(dataset_id, value_column, id_column, sort_column)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error extracting signals: {str(e)}"

@mcp.tool()
def load_hf_dataset(dataset_name: str, split: Optional[str] = "train", config_name: Optional[str] = None) -> str:
    """
    Loads a dataset from the Hugging Face Hub (requires 'datasets' library).

    Args:
        dataset_name: Name of the dataset on HF Hub (e.g., 'mnist', 'glue').
        split: The split to load (default: 'train').
        config_name: Optional configuration name (subset) of the dataset.

    Returns:
        The new dataset_id.
    """
    try:
        dataset_id = session_manager.load_hf_dataset(dataset_name, split, config_name)
        return f"Success. Dataset loaded with ID: {dataset_id}"
    except Exception as e:
        return f"Error loading HF dataset: {str(e)}"

@mcp.tool()
def extract_tables(url: str) -> str:
    """
    Extracts tables from a web URL using pandas (requires 'lxml' or 'html5lib').

    Args:
        url: The URL to scrape tables from.

    Returns:
        A JSON summary of extracted tables and their IDs.
    """
    try:
        tables = session_manager.extract_tables_from_url(url)
        if not tables:
            return "No tables found at the provided URL."
        return json.dumps(tables, indent=2)
    except Exception as e:
        return f"Error extracting tables: {str(e)}"

@mcp.tool()
def generate_map(dataset_id: str, lat_col: str, lon_col: str) -> str:
    """
    Generates a geospatial map (scatter plot) from a dataset.

    Args:
        dataset_id: The ID of the dataset.
        lat_col: Column name for Latitude.
        lon_col: Column name for Longitude.

    Returns:
        Absolute path to the generated map image (PNG).
    """
    try:
        path = session_manager.generate_map(dataset_id, lat_col, lon_col)
        return f"Success. Map generated at: {path}"
    except Exception as e:
        return f"Error generating map: {str(e)}"

@mcp.tool()
def start_explorer(dataset_id: str) -> str:
    """
    Launches an interactive D-Tale explorer for the dataset.

    Args:
        dataset_id: The ID of the dataset.

    Returns:
        The URL to access the explorer.
    """
    try:
        url = session_manager.start_explorer(dataset_id)
        return f"Success. Explorer running at: {url}"
    except Exception as e:
        return f"Error starting explorer: {str(e)}"

if __name__ == "__main__":
    mcp.run()
