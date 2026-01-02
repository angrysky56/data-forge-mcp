# Data-Forge MCP Server

**Data-Forge** is a Model Context Protocol (MCP) server that transforms any LLM into a powerful Data Science Assistant. It provides a suite of high-performance tools for data loading, cleaning, validation, profiling, feature engineering, and visualization.

## 🚀 Features

### 1. The Core (Data Management)

- **`load_data`**: Load CSV or Parquet files into memory using **Pandas** or **Polars** (high-performance).
- **`get_dataset_info`**: Inspect schema, columns, and data types.
- **`list_active_datasets`**: Manage multiple loaded datasets in a session.

### 2. Quality Gatekeeper

- **`validate_dataset`**: Run schema checks using **Pandera** (e.g., "price must be > 0").
- **`clean_dataset`**: Apply cleaning pipelines using **PyJanitor** (e.g., `clean_names`, `remove_empty`, `to_numeric`).

### 3. Profiling & Insights

- **`get_dataset_profile`**: Generate comprehensive statistical reports using **YData Profiling** (distribution, correlations, alerts).

### 4. Interpretation & Visualization

- **`generate_chart`**: Create **Seaborn/Matplotlib** charts (scatter, line, bar, hist, box) and save them as images for the LLM to see.

### 5. Feature Engineering

- **`extract_signals`**: Automatically extract hundreds of time-series features (peaks, entropy, volatility) using **tsfresh**.

### 6. Acquisition

- **`load_hf_dataset`**: Download datasets directly from the **Hugging Face Hub**.
- **`extract_tables`**: Scrape structured tables from any website URL (e.g., Wikipedia).

### 7. Visualizer (Render)

- **`generate_map`**: Create geospatial plots from lat/lon data (GeoPandas).
- **`start_explorer`**: Launch an interactive **D-Tale** dashboard for deep data exploration in your browser.

### 8. Discovery (Probes)

- **`scan_semantic_voids`**: Use Topological Data Analysis (TDA) to find "holes" or missing concepts in text datasets.

### 9. Headless Control (DuckDB)

- **`run_sql_query`**: "God Mode" for the agent. Execute complex SQL (Joins, Aggregates, Filters) across all loaded datasets using an in-memory DuckDB engine.

---

## 🛠️ Installation

### Prerequisites

- Python 3.13+
- `uv` (recommended) or `pip`

### Deployment

1. Clone the repository:

   ```bash
   git clone https://github.com/angrysky56/data-forge-mcp.git
   cd data-forge-mcp
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Configure Environment if not set in path (Optional):
   Create a `.env` file if you need to access private Hugging Face datasets:
   ```bash
   HF_TOKEN=your_hugging_face_token
   ```

---

## 🔌 Configuration

To use Data-Forge with your MCP Client (e.g., Claude Desktop), add the following to your config:

```json
{
  "mcpServers": {
    "data-forge": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/data-forge-mcp",
        "run",
        "-m",
        "src.server"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "LOG_LEVEL": "WARNING",
        "NUMBA_DISABLE_CUDA": "1",
        "NUMBA_ENABLE_CUDASIM": "1"
      }
    }
  }
}
```

---

## 💡 Usage Examples

### 1. Load and Profile

```python
# LLM Action
use_tool("load_data", file_path="/path/to/data.csv")
use_tool("get_dataset_profile", dataset_id="df_123")
```

### 2. Clean and Visualize

```python
# LLM Action
use_tool("clean_dataset",
    dataset_id="df_123",
    operations=[
        "clean_names",
        {"method": "to_numeric", "args": ["price"]},
        "dropna"
    ]
)
use_tool("generate_chart",
    dataset_id="df_123",
    chart_type="scatter",
    x="market_cap",
    y="price"
)
```

### 3. Fetch from Web

```python
# LLM Action
use_tool("extract_tables", url="https://en.wikipedia.org/wiki/S&P_500")
```

---

## 🏗️ Tech Stack

- **Server**: FastMCP
- **Data Engine**: Pandas, Polars
- **Validation**: Pandera
- **Cleaning**: PyJanitor
- **Visualization**: Seaborn, Matplotlib
- **Profiling**: YData Profiling
- **Features**: tsfresh
- **Acquisition**: Hugging Face Datasets, lxml

License: MIT

By: angrysky56 and Gemini 3 Pro
