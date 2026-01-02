import uuid
import os
import time
import json
from datetime import datetime
from src.discovery import VoidScanner
from typing import Dict, Any, List, Optional, Union
import pandas as pd

# Optional dependencies
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore
import pandera as pa
import janitor  # noqa: F401
import duckdb
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from tsfresh import extract_features

class SessionManager:
    """
    Manages the lifecycle of loaded DataFrames in memory.
    """
    def __init__(self) -> None:
        # In-memory registry for datasets: {dataset_id: DataFrame}
        self._registry: Dict[str, Union[pd.DataFrame, Any]] = {}
        # Metadata storage: {dataset_id: {info...}}
        self._metadata: Dict[str, Any] = {}
        # Outputs directory
        self._outputs_dir = os.path.join(os.getcwd(), "outputs")
        if not os.path.exists(self._outputs_dir):
            os.makedirs(self._outputs_dir)

        # Initialize sub-systems lazily or on demand?
        # For now, just keep a reference to be initialized when used to save startup time
        self.scanner = VoidScanner()

    def _to_pandas_safe(self, df: Union[pd.DataFrame, 'pl.DataFrame'], limit: int = 100000, context: str = "operation") -> pd.DataFrame:
        """
        Safely converts a DataFrame (Pandas or Polars) to Pandas, sampling if necessary
        to avoid OOM on large datasets.
        """
        if isinstance(df, pd.DataFrame):
            # Already Pandas. If huge, we technically could sample, but user didn't flag this path.
            # Let's trust Pandas users know their RAM or just sample if huge anyway?
            # For consistency with the Polars safety, let's limit it too if requested.
            if len(df) > limit:
                print(f"[{context}] Dataset too large ({len(df)} rows). Sampling {limit} rows for safety.")
                return df.sample(n=limit, random_state=42)
            return df

        if pl is not None and isinstance(df, pl.DataFrame):
            if len(df) > limit:
                print(f"[{context}] Polars Dataset too large ({len(df)} rows). Sampling {limit} rows for safely converting to Pandas.")
                return df.sample(n=limit, with_replacement=False, seed=42).to_pandas()
            return df.to_pandas()

        return df # Should be unreachable if types align

    def extract_signals(self, dataset_id: str, value_column: str, id_column: Optional[str] = None, sort_column: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Extracts time-series features using tsfresh.
        Args:
            dataset_id: ID of the dataset.
            value_column: Column containing the time-series values.
            id_column: Column identifying separate time series (e.g. 'symbol'). If None, treats whole DF as one series.
            sort_column: Column to sort by (usually time/date).
        """
        df = self.get_dataset(dataset_id)

        # tsfresh works with pandas. Use safe conversion (limit 50k, tsfresh is slow)
        df = self._to_pandas_safe(df, limit=50000, context="extract_signals")

        # Input validation
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found.")

        try:
            # Prepare arguments for extract_features
            kwargs = {"column_value": value_column}
            if id_column:
                if id_column not in df.columns:
                    raise ValueError(f"ID column '{id_column}' not found.")
                kwargs["column_id"] = id_column
            else:
                # If no ID provided, create a dummy ID to treat as one series
                df["_id_dummy"] = 1
                kwargs["column_id"] = "_id_dummy"

            if sort_column:
                if sort_column not in df.columns:
                    raise ValueError(f"Sort column '{sort_column}' not found.")
                kwargs["column_sort"] = sort_column

            # Extract features
            # Using MinimalFCParameters for speed in this context, or Efficient?
            # Let's stick to default (comprehensive) for now, or maybe make it configurable later.
            # Warning: Default extracts HUNDREDS of features which can be slow.
            # Let's use string-based coercion for proper types before extraction if needed (tsfresh is picky).

            # Run extraction
            extracted_features = extract_features(df, **kwargs)

            # Register the new features as a new dataset? Or merge?
            # For now, let's register as a new dataset.
            new_id = f"{dataset_id}_features"
            self._registry[new_id] = extracted_features
            self._metadata[new_id] = {
                "source": "tsfresh_extraction",
                "parent_id": dataset_id,
                "rows": len(extracted_features),
                "columns": list(extracted_features.columns)
            }

            return {
                "message": f"Features extracted successfully. New dataset created: {new_id}",
                "new_dataset_id": new_id,
                "n_features": len(extracted_features.columns),
                "preview_columns": list(extracted_features.columns)[:10]
            }

        except Exception as e:
            raise RuntimeError(f"Signal extraction failed: {str(e)}")

    def load_dataset(self, file_path: str, alias: Optional[str] = None, engine: str = "pandas") -> str:
        """
        Loads a dataset into memory and returns a unique ID.

        Args:
            file_path: Absolute path to the file.
            alias: Human-readable name for the dataset.
            engine: 'pandas' or 'polars'.

        Returns:
            dataset_id: A UUID string reference to the loaded DataFrame.
        """
        if engine not in ["pandas", "polars"]:
             raise NotImplementedError(f"Engine '{engine}' not supported.")

        try:
            dataset_id = f"df_{uuid.uuid4().hex[:8]}"
            alias = alias or file_path.split("/")[-1]
            rows: int = 0
            cols: List[str] = []

            if engine == "pandas":
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                else:
                    raise ValueError("Unsupported file format. Please use .csv or .parquet")

                self._registry[dataset_id] = df
                rows = len(df)
                cols = list(df.columns)

            elif engine == "polars":
                if file_path.endswith(".csv"):
                    df_pl = pl.read_csv(file_path)
                elif file_path.endswith(".parquet"):
                    df_pl = pl.read_parquet(file_path)
                else:
                    raise ValueError("Unsupported file format. Please use .csv or .parquet")

                self._registry[dataset_id] = df_pl
                rows = len(df_pl)
                # Polars columns is already a list of strings usually, but strictly it returns List[str]
                cols = df_pl.columns

            self._metadata[dataset_id] = {
                "path": file_path,
                "alias": alias,
                "rows": rows,
                "columns": cols,
                "engine": engine
            }

            return dataset_id

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

    def get_dataset(self, dataset_id: str) -> Union[pd.DataFrame, pl.DataFrame]:
        """Retrieves a DataFrame by ID."""
        if dataset_id not in self._registry:
            raise KeyError(f"Dataset ID '{dataset_id}' not found in registry.")
        return self._registry[dataset_id]

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Returns metadata for all loaded datasets."""
        return [
            {"id": k, **v} for k, v in self._metadata.items()
        ]

    def get_dataset_info(self, dataset_id: str) -> str:
        """Returns string representation of df.info() or polars equivalent."""
        df = self.get_dataset(dataset_id)

        if isinstance(df, pd.DataFrame):
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            return buffer.getvalue()
        elif pl is not None and isinstance(df, pl.DataFrame):
            return str(df)
        return "Unknown DataFrame type."

    def validate_dataset(self, dataset_id: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the dataset against a provided schema using Pandera.
        Note: Currently converts Polars to Pandas for validation as Pandera Polars support is experimental/different.
        """
        df = self.get_dataset(dataset_id)

        # Temporary: convert Polars to Pandas for validation if manageable size
        # TODO: Implement native Polars validation when Pandera support matures or use simple Polars checks
        if pl is not None and isinstance(df, pl.DataFrame):
            # Warning: this might be expensive
            df = df.to_pandas()

        try:
            columns = {}
            for col_name, rules in schema.get("columns", {}).items():
                col_type = rules.get("type", None)
                checks = []
                for check_name, check_val in rules.get("checks", {}).items():
                    if hasattr(pa.Check, check_name):
                         checks.append(getattr(pa.Check, check_name)(check_val))

                columns[col_name] = pa.Column(col_type, checks=checks, coerce=True, nullable=rules.get("nullable", False))

            pd_schema = pa.DataFrameSchema(columns=columns)
            pd_schema.validate(df, lazy=True)
            return {"valid": True, "errors": []}

        except pa.errors.SchemaErrors as err:
            return {"valid": False, "errors": err.failure_cases.to_dict(orient="records")}
        except Exception as e:
             return {"valid": False, "errors": [str(e)]}

    def clean_dataset(self, dataset_id: str, operations: List[Union[str, Dict[str, Any]]]) -> str:
        """
        Applies a sequence of cleaning functions.
        Operations can be a string (method name) or a dict {"method": "name", "args": [], "kwargs": {}}.
        """
        df = self.get_dataset(dataset_id)

        if pl is not None and isinstance(df, pl.DataFrame):
            return "Cleaning not yet fully implemented for Polars engine."

        for op in operations:
            method_name = ""
            args = []
            kwargs = {}

            if isinstance(op, str):
                method_name = op
            elif isinstance(op, dict):
                method_name = op.get("method", "")
                args = op.get("args", [])
                kwargs = op.get("kwargs", {})

            if hasattr(df, method_name):
                method = getattr(df, method_name)
                if callable(method):
                    df = method(*args, **kwargs)
                else:
                    pass
            elif method_name == "to_numeric":
                # Custom handler for robust numeric conversion
                # Usage: {"method": "to_numeric", "args": ["col_name"]}
                if not args:
                    raise ValueError("to_numeric requires a column name argument")
                col = args[0]
                if col in df.columns:
                    # Clean currency symbols first if simple replace didn't work,
                    # but actually to_numeric(coerce) will turn "$100" into NaN which isn't ideal for data preservation.
                    # Ideally we strip symbols then coerce.
                    # But for "automated" feeling, let's try to strip common non-numeric chars first?
                    # or just rely on the user to run 'clean_names' then this.
                    # Let's keep it simple: pd.to_numeric(errors='coerce')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    raise ValueError(f"Column '{col}' not found.")
            else:
                 raise ValueError(f"Operation '{method_name}' not supported or not found on DataFrame.")

        self._registry[dataset_id] = df
        self._metadata[dataset_id]["rows"] = len(df)
        self._metadata[dataset_id]["columns"] = list(df.columns)

        return "Cleaned dataset"

    def get_profile(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generates a profile report using ydata-profiling.
        Returns a JSON summary of the profile.
        """
        df = self.get_dataset(dataset_id)

        # YData Profiling works best with Pandas
        # Limit to 100k for profiling to ensure responsiveness
        df = self._to_pandas_safe(df, limit=100000, context="get_dataset_profile")

        profile = ProfileReport(df, minimal=True)
        json_data = profile.to_json()

        # Parse JSON to return key insights (avoid returning massive raw JSON)
        data = json.loads(json_data)

        # Extract warnings/alerts specifically
        alerts = data.get("alerts", [])
        description = data.get("table", {})

        return {
            "description": description,
            "alerts": alerts,
            "variables": list(data.get("variables", {}).keys())
        }

    def load_hf_dataset(self, dataset_name: str, split: Optional[str] = "train", config_name: Optional[str] = None) -> str:
        """
        Loads a dataset from the Hugging Face Hub.
        """
        try:
            import datasets

            # Load from Hub
            ds = datasets.load_dataset(dataset_name, name=config_name, split=split)

            # Convert to Pandas
            df = ds.to_pandas()

            # Register
            dataset_id = f"df_hf_{uuid.uuid4().hex[:8]}"
            self._registry[dataset_id] = df
            self._metadata[dataset_id] = {
                "source": f"huggingface: {dataset_name}",
                "config": config_name,
                "split": split,
                "rows": len(df),
                "columns": list(df.columns)
            }
            return dataset_id
        except Exception as e:
            raise RuntimeError(f"Failed to load HF dataset: {str(e)}")

    def extract_tables_from_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrapes tables from a URL using pd.read_html.
        Returns a list of metadata for extracted tables.
        """
        try:
            import urllib.request

            # Use urllib to fetch with User-Agent to avoid 403 Forbidden
            # Wikipedia and others often block requests without a User-Agent
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req) as response:
                html_content = response.read()

            # Pandas read_html returns a list of DataFrames
            # Requires lxml or html5lib
            dfs = pd.read_html(html_content)

            if not dfs:
                return []

            results = []
            base_id = f"web_{uuid.uuid4().hex[:4]}"

            for i, df in enumerate(dfs):
                if df.empty:
                     continue

                table_id = f"{base_id}_t{i}"
                self._registry[table_id] = df
                self._metadata[table_id] = {
                    "source": f"web: {url}",
                    "table_index": i,
                    "rows": len(df),
                    "columns": list(df.columns)
                }

                results.append({
                    "table_id": table_id,
                    "rows": len(df),
                    "columns": list(df.columns)[:5] # Preview
                })

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from URL: {str(e)}")

    def generate_map(self, dataset_id: str, lat_col: str, lon_col: str, title: Optional[str] = None) -> str:
        """
        Generates a geospatial scatter plot (map) using GeoPandas/Matplotlib.
        Saves the image to the outputs directory.
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            df = self.get_dataset(dataset_id)
            # Limit map points to 50k to avoid cluttered map and slow mpl
            df = self._to_pandas_safe(df, limit=50000, context="generate_map")

            if lat_col not in df.columns or lon_col not in df.columns:
                raise ValueError(f"Columns '{lat_col}' and '{lon_col}' required for mapping.")

            # Create GeoDataFrame
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)

            # Simple world map background (optional, but good for context)
            # For MVP, just plot points on a blank canvas or try to load a low-res base map if possible.
            # actually, let's keep it simple: just the points for now, maybe add a border.

            fig, ax = plt.subplots(figsize=(10, 6))

            # If we had a world map, we'd plot it here.
            # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            # world.plot(ax=ax, color='lightgrey')

            gdf.plot(ax=ax, markersize=5, color='red', alpha=0.6)

            if title:
                plt.title(title)
            else:
                plt.title(f"Map: {dataset_id}")

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

            filename = f"map_{dataset_id}_{int(time.time())}.png"
            output_path = os.path.join(self._outputs_dir, filename) # Fixed attribute name
            plt.savefig(output_path)
            plt.close()

            return os.path.abspath(output_path)

        except Exception as e:
            raise RuntimeError(f"Map generation failed: {str(e)}")
    def scan_voids(self, dataset_id: str, text_column: str) -> str:
        """
        Scans a text column for semantic voids using Topological Data Analysis.
        """
        df = self.get_dataset(dataset_id)

        # Standardize to List[str]
        texts = []
        if isinstance(df, pd.DataFrame):
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found.")
            texts = df[text_column].astype(str).tolist()
        elif pl is not None and isinstance(df, pl.DataFrame):
             if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found.")
             texts = df[text_column].cast(pl.Utf8).to_list()

        if not texts:
            return "No text data found to scan."

        print(f"Scanning {len(texts)} items in '{text_column}'...")
        report = self.scanner.scan(texts, output_dir=self._outputs_dir)

        # Format the output for the user
        status = "DETECTED" if report["void_detected"] else "NOT DETECTED"
        msg = "Scan Complete.\n"
        msg += f"Void Status: {status} (Max Persistence: {report['max_persistence']:.4f})\n"
        msg += f"Visualization: {report['image_path']}\n"
        msg += "Semantic Landmarks:\n"
        for i, lm in enumerate(report["landmarks"]):
            msg += f"  - Cluster {i}: {lm[:100]}...\n" # Truncate long texts

        return msg
    def start_explorer(self, dataset_id: str) -> str:
        """
        Starts a D-Tale instance for the given dataset and returns the URL.
        """
        try:
            import dtale

            df = self.get_dataset(dataset_id)
            # D-Tale can handle more, but 5GB pandas is risky. Let's cap at 500k for safety?
            # Or let user decide? For now, standard safe limit.
            df = self._to_pandas_safe(df, limit=500000, context="start_explorer")

            # D-Tale manages its own global state.
            # startup(data_id=...) returns an instance.
            d = dtale.show(df, data_id=dataset_id, ignore_duplicate=True)

            # Open browser is handled by user clicking the link
            return d.main_url()

        except Exception as e:
            raise RuntimeError(f"Failed to start D-Tale explorer: {str(e)}")

    def generate_chart(self, dataset_id: str, x: Optional[str] = None, y: Optional[str] = None, chart_type: str = "scatter", title: Optional[str] = None) -> str:
        """
        Generates a chart using Seaborn/Matplotlib and saves it to a temp file.
        Returns the absolute path to the generated image.
        """
        df = self.get_dataset(dataset_id)

        # Plotting millions of points is slow. Limit to 50k.
        df = self._to_pandas_safe(df, limit=50000, context="generate_chart")

        plt.figure(figsize=(10, 6))

        if chart_type == "scatter":
            if not x or not y:
                raise ValueError("Scatter plot requires x and y columns.")
            sns.scatterplot(data=df, x=x, y=y)
        elif chart_type == "line":
            if not x or not y:
                raise ValueError("Line plot requires x and y columns.")
            sns.lineplot(data=df, x=x, y=y)
        elif chart_type == "bar":
            if not x or not y:
                raise ValueError("Bar plot requires x and y columns.")
            sns.barplot(data=df, x=x, y=y)
        elif chart_type == "hist":
            if not x:
                raise ValueError("Histogram requires x column.")
            sns.histplot(data=df, x=x)
        elif chart_type == "box":
            if not x:
                raise ValueError("Box plot requires x column.")
            sns.boxplot(data=df, x=x, y=y)  # y is optional for box plot grouping
        else:
            raise ValueError(f"Chart type '{chart_type}' not supported.")

        if title:
            plt.title(title)

        # Save to outputs folder
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Consistent filename: chart_{timestamp}_{dataset_id}.png
        timestamp = int(time.time())
        filename = f"chart_{dataset_id}_{timestamp}.png"
        path = os.path.join(output_dir, filename)

        plt.savefig(path)
        plt.close()

        return path

    def query_data(self, query: str, dataset_id: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Executes a SQL query on the loaded datasets using DuckDB.
        """
        try:
            # 1. Setup DuckDB connection
            # duckdb.connect() creates an in-memory database
            con = duckdb.connect(database=':memory:')

            # 2. Register all active datasets as virtual tables
            for ds_id, df in self._registry.items():
                try:
                    # Polars/Pandas are both supported by DuckDB
                    # However, for Polars, we might need to be careful with versions.
                    # DuckDB python client handles pandas natively.
                    con.register(ds_id, df)
                except Exception:
                    pass # Skip non-compatible objects

            # 3. Handle 'this' alias if dataset_id provided
            if dataset_id and dataset_id in self._registry:
                con.register('this', self._registry[dataset_id])

            # 4. Execute Query
            # We limit to 50 rows to prevent context flooding if user forgets limit
            # But we should rely on the user's query mainly.
            # Let's run it and then head() the result dataframe.
            result_df = con.execute(query).df()

            row_count = len(result_df)
            # Preview top 20 rows as markdown
            preview = result_df.head(20).to_markdown(index=False)

            return {
                "status": "success",
                "rows_returned": row_count,
                "preview": preview,
                "note": "Query executed. Top 20 rows shown."
            }

        except Exception as e:
            return f"SQL Error: {str(e)}"
