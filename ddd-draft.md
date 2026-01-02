This is a **Developer Design Document (DDD)** for the **"Data-Forge" MCP Server**.

This architecture solves the primary constraint of LLM data science: **Context limits vs. Data volume.** By treating the Python environment as an "external processor," the LLM retains high-level reasoning while offloading heavy memory and compute tasks to optimized libraries.

---

# Design Doc: Data-Forge MCP Server

**Version:** 1.0 (Draft)
**Author:** AI Collaborator
**Status:** Request for Comment (RFC)

## 1. Problem Statement

Current LLM data analysis relies on uploading raw files into the context window. This fails when:

1. **Data Volume:** Datasets exceed token limits (e.g., >2MB CSVs).
2. **Hallucination:** LLMs "guess" statistical properties rather than calculating them.
3. **Compute:** LLMs cannot perform vector operations or complex joins efficiently.

## 2. Proposed Solution

Build a stateful **Model Context Protocol (MCP) Server** that exposes high-performance Python libraries as discrete tools. The server maintains a "Working Memory" (state) of loaded DataFrames, allowing the LLM to query and manipulate data by reference (ID) rather than by value (raw text).

---

## 3. System Architecture

The system follows a **Hub-and-Spoke** architecture. The MCP Server acts as the central router, managing a lifecycle of data objects.

### 3.1 Component Diagram

```mermaid
graph TD
    Client[LLM / MCP Client] <-->|JSON-RPC| Server[Data-Forge MCP Server]

    subgraph "Working Memory (State Manager)"
        Registry[DataFrame Registry (Dict)]
    end

    Server --> Registry

    subgraph "Compute Modules"
        Mod_Val[Validator Engine] -->|Pandera/Pyjanitor| Registry
        Mod_Perf[Big Data Engine] -->|Vaex/cuDF| Registry
        Mod_Prof[Profile Engine] -->|Sweetviz/YData/tsfresh| Registry
        Mod_Vis[Visual Engine] -->|GeoPandas/D-Tale| Registry
    end

    Mod_Val -->|Read/Write| Storage[(Local File System)]
    Mod_Perf -->|Lazy Load| Storage

```

### 3.2 Key Technical Decisions

1. **State Management (The Registry):**

- **Problem:** MCP is stateless (mostly), but data science requires persistence.
- **Solution:** A global `SessionManager` class.
- **Mechanism:** When `load_dataset(path)` is called, the server loads the file, assigns a UUID (e.g., `df_8a2b`), and returns the UUID to the LLM. All subsequent calls (e.g., `clean_dataset(df_id=df_8a2b)`) reference this ID.

2. **Backend Abstraction (Strategy Pattern):**

- To handle the choice between `pandas`, `vaex`, and `cudf`, the system will use a `DataFrameWrapper` abstract base class.
- The LLM can specify `mode="fast"` (Pandas/cuDF) or `mode="big"` (Vaex) at load time.

---

## 4. Module Specifications

### Module A: The Gatekeeper (Ingest & Validate)

**Libraries:** `Pandera`, `Pyjanitor`

- **Tools:**
- `load_data(path, engine)`: Loads data into the Registry. Returns `dataset_id`.
- `validate_schema(dataset_id, schema_dict)`: Uses **Pandera** to check types/ranges. Returns `{"status": "failed", "errors": [...]}`.
- `janitor_clean(dataset_id, operations)`: Uses **Pyjanitor** to chain cleaning verbs (clean_names, remove_empty). Updates the state in Registry.

### Module B: The Scout (Profile & Signal)

**Libraries:** `ydata-profiling`, `Sweetviz`, `tsfresh`

- **Tools:**
- `get_profile_summary(dataset_id)`: Uses **ydata-profiling** to generate a JSON summary (skipping the HTML generation for speed) to feed "meta-knowledge" to the LLM.
- `extract_features(dataset_id, time_col, value_col)`: Uses **tsfresh** to return a list of statistically significant features, reducing noise.

### Module C: The Heavy Lifter (Query & Scale)

**Libraries:** `Vaex`, `cuDF`

- **Tools:**
- `run_sql_query(dataset_id, query)`: Since **Vaex** allows expression evaluation, we expose a simplified SQL-like or Pandas-query interface.
- **Logic:** If `engine='vaex'`, this executes lazily. If `engine='cudf'`, it executes on GPU.

### Module D: The Visualizer (Render)

**Libraries:** `GeoPandas`, `D-Tale`

- **Tools:**
- `generate_map(dataset_id, lat_col, lon_col)`: Uses **GeoPandas** to plot points/polygons. Returns a path to a generated PNG.
- `start_explorer(dataset_id)`: Launches a **D-Tale** server on a random port and returns the URL (`http://localhost:40000...`) for the user to click.

---

## 5. API Interface (MCP Tool Definitions)

Here is the JSON schema signature for the critical `load_data` tool, which establishes the session context.

```json
{
  "name": "load_data",
  "description": "Loads a file into the server's working memory. Select 'vaex' for datasets >2GB, 'pandas' for general use, or 'cudf' if GPU is available.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": { "type": "string" },
      "alias": {
        "type": "string",
        "description": "Human readable name for the dataset"
      },
      "engine": {
        "type": "string",
        "enum": ["pandas", "vaex", "cudf"],
        "default": "pandas"
      }
    },
    "required": ["file_path"]
  }
}
```

---

## 6. Implementation Plan

### Phase 1: The Core (Skeleton)

- Setup `FastMCP` server.
- Implement `SessionManager` (Dictionary-based registry).
- Implement `load_data` (Pandas only) and `get_info` (df.info() output).

### Phase 2: Validation & Cleaning

- Integrate `Pandera` for validation logic.
- Integrate `Pyjanitor` for cleaning functions.

### Phase 3: Scale & Profiling

- Implement `Vaex` backend for out-of-core loading.
- Add `ydata-profiling` tool (configured to output JSON summaries for the LLM).

### Phase 4: Visualization

- Add `GeoPandas` plotting.
- Add `D-Tale` launcher (requires thread management to keep the server alive).

---

## 7. Security & Sandbox Constraints

- **Path Traversal:** The server must have a `ROOT_DATA_DIR` env variable. Any `file_path` requests outside this directory must be rejected.
- **Resource Limits:** The `SessionManager` must implement a Least Recently Used (LRU) eviction policy if memory usage exceeds X GB.

---

I have broken down these 10 libraries into specific **MCP Tool Use Cases**.

The core philosophy here is **Context Economy**: An LLM cannot (and should not) ingest a 10GB CSV into its context window. Instead, it should use MCP tools to query, profile, and validate data remotely, receiving only the necessary signals (schema errors, correlations, aggregations) back.

Here is how to architect these libraries into an MCP Data Processing Server.

---

### 1. The "Gatekeeper" Tools (Validation & Quality)

**Role:** These libraries serve as the first line of defense. The LLM uses them to verify data integrity before attempting complex reasoning or code generation.

#### **Pandera**

- **MCP Tool Name:** `validate_schema`
- **Use Case:** The LLM inspects a dataset and hypothesizes a schema (e.g., "Columns A and B should be floats, C is categorical"). It calls this tool to enforce that contract.
- **LLM Workflow:**

1. LLM receives a file path.
2. LLM generates a Pandera schema definition based on user intent.
3. **Tool Action:** Applies schema to the DataFrame.
4. **Return to LLM:** Instead of the full data, it returns a structured **error report** (e.g., "Row 405: value -1 violates `check_positive`").

- **Why for MCP:** Enables "Self-Healing" data pipelines where the Agent fixes data errors iteratively.

#### **Pyjanitor**

- **MCP Tool Name:** `clean_dataframe`
- **Use Case:** Providing a "verbs" interface for cleaning. Instead of the LLM writing complex, error-prone Pandas code, it sends high-level instructions.
- **LLM Workflow:**

1. LLM identifies messy column names and nulls.
2. **Tool Action:** Executes method chains: `.clean_names().remove_empty()`.
3. **Return to LLM:** A boolean success flag and the `.head()` of the cleaned data to verify structure.

---

### 2. The "Heavy Lifter" Tools (Scale & Performance)

**Role:** Handling data that exceeds the LLM's context window (and the host machine's RAM).

#### **Vaex** & **cuDF**

- **MCP Tool Name:** `query_large_dataset`
- **Use Case:** Performing "Lazy" analysis. The LLM acts as a query engine, not a storage engine.
- **LLM Workflow:**

1. User asks: "What is the average transaction value for User X in this 50GB file?"
2. **Tool Action:** Uses **Vaex** (CPU lazy loading) or **cuDF** (GPU acceleration) to perform the filter and aggregation without loading the file into memory.
3. **Return to LLM:** A single float value or a tiny summary table.

- **Why for MCP:** This is the only way an LLM can interact with production-scale data. It decouples the _reasoning_ (LLM) from the _compute_ (MCP Server).

---

### 3. The "Scout" Tools (Profiling & EDA)

**Role:** Providing high-bandwidth information compression. The LLM needs to "understand" the shape of the data without reading the rows.

#### **Sweetviz** & **ydata-profiling**

- **MCP Tool Name:** `generate_data_profile`
- **Use Case:** "Dual-Stream" Output. One stream for the LLM (metadata), one for the Human (visuals).
- **LLM Workflow:**

1. LLM calls the tool on a new dataset.
2. **Tool Action:** Generates the HTML report. Crucially, it parses the _JSON summary_ of the report (correlations, missing value % per column).
3. **Return to LLM:** The JSON summary (so the LLM knows "Column A is highly correlated with B") and a `file://` or `localhost` URL to the HTML report.
4. **Next Step:** The LLM presents the URL to the user: "I've analyzed the data. Column A drives the target variable. You can view the full visual report here."

#### **tsfresh**

- **MCP Tool Name:** `extract_timeseries_features`
- **Use Case:** Semantic compression of time-series.
- **LLM Workflow:**

1. User uploads sensor log data.
2. **Tool Action:** `tsfresh` extracts hundreds of features (entropy, peaks, mean).
3. **Return to LLM:** A list of the top 5 most significant features. The LLM can now reason: "The _entropy_ of the signal is the strongest predictor of failure," rather than trying to analyze raw signal noise.

---

### 4. The "Visualizer" Tools (Interactive GUIs)

**Role:** Bridging the gap between the chat interface and specialized UI.

#### **D-Tale** & **ITables**

- **MCP Tool Name:** `launch_interactive_explorer`
- **Use Case:** When the user asks to "explore" the data manually, the LLM acts as an app launcher.
- **LLM Workflow:**

1. User: "Let me play with this data."
2. **Tool Action:** Spins up a D-Tale server instance or renders an ITables HTML snippet.
3. **Return to LLM:** A local URL (e.g., `http://localhost:40000/dtale/main/1`).
4. **Agent Response:** "I've loaded the data into a D-Tale instance. You can manipulate it directly at [Link]."

#### **GeoPandas**

- **MCP Tool Name:** `spatial_query`
- **Use Case:** Answering "Where" questions.
- **LLM Workflow:**

1. User: "How many stores are within 5km of the warehouse?"
2. **Tool Action:** Performs the geometric buffer and intersection logic.
3. **Return to LLM:** The count (integer) and a generated static map image path (which the LLM can display using markdown).

---

### Summary Architecture

The `list_tools` endpoint could look like this:

| Tool Name             | Python Library      | Description for LLM                                                   |
| --------------------- | ------------------- | --------------------------------------------------------------------- |
| `get_dataset_health`  | **Pandera**         | Validates types and statistical checks; returns error logs.           |
| `get_dataset_profile` | **ydata-profiling** | Returns JSON statistical summary and generates HTML report link.      |
| `clean_dataset`       | **Pyjanitor**       | Applies standard cleaning (snake_case, remove_nulls) via instruction. |
| `query_big_data`      | **Vaex**            | Executes SQL-like filters on datasets > RAM; returns aggregations.    |
| `extract_signals`     | **tsfresh**         | Converts time-series columns into named statistical features.         |
| `render_map`          | **GeoPandas**       | Generates a static map image from geospatial queries.                 |

### Next Step

Would you like me to draft the **`server.py` code for the `query_big_data` tool using Vaex**, so you can test how an LLM handles out-of-core data querying?

### Next Step

Generate the **Python code for Phase 1 (The Core & Session Manager)** so you have a working skeleton to start importing these libraries into.
