from src.server import load_data, get_dataset_info, generate_chart, get_dataset_profile
import os

def main():
    file_path = os.path.abspath("data/CryptocurrencyData.csv")
    print(f"--- 1. Loading Data: {file_path} (Engine: Polars) ---")

    # Simulate MCP Call: load_data
    result_load = load_data(file_path, alias="CryptoData", engine="polars")
    print(result_load)

    # Extract ID from result string (simple parse for verification)
    # Output format: "Successfully loaded dataset '...'. ID: <id>"
    dataset_id = result_load.split("ID: ")[1].strip()

    print(f"\n--- 2. Getting Info for ID: {dataset_id} ---")
    info = get_dataset_info(dataset_id)
    print(info)

    print(f"\n--- 3. Generating Chart (Scatter: Market Cap vs Price) ---")
    # Columns in CSV likely: 'Market Cap', 'Price' (need to check exact names in info output if this fails, but guessing based on filename)
    # Let's peek at columns first from info or use a safer bet if column names are unknown?
    # Actually, let's try 'Market Cap' and 'Price' based on typical data. If they fail, we'll see the error.
    try:
        chart_path = generate_chart(dataset_id, x="Market Cap", y="Price", chart_type="scatter", title="Crypto Market Cap vs Price")
        print(chart_path)
    except Exception as e:
        print(f"Chart generation failed (maybe column names mismatch?): {e}")

    print(f"\n--- 4. Generating Profile ---")
    profile = get_dataset_profile(dataset_id)
    print(f"Profile generated (length: {len(profile)} chars)")
    print("Snippet:", profile[:200])

if __name__ == "__main__":
    main()
