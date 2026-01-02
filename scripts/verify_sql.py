from src.session_manager import SessionManager
import pandas as pd
import duckdb

def verify_sql():
    print("--- 1. Setup ---")
    sm = SessionManager()

    # Create Users DF
    users = pd.DataFrame({
        "uid": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "region": ["West", "East", "West", "North"]
    })
    sm._registry["df_users"] = users

    # Create Orders DF
    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "uid": [1, 1, 2, 3, 5], # 5 is orphan
        "amount": [12000, 500, 300, 15000, 20]
    })
    sm._registry["df_orders"] = orders

    print("--- 2. Executing SQL Join ---")
    # Join on uid, filter for amount > 10000
    query = """
    SELECT
        u.name,
        u.region,
        o.amount
    FROM df_users u
    JOIN df_orders o ON u.uid = o.uid
    WHERE o.amount > 10000
    ORDER BY o.amount DESC
    """

    result = sm.query_data(query) # No dataset_id needed for global join

    print("Result:")
    print(result)

    if isinstance(result, dict) and result["rows_returned"] == 2:
        print("\nSUCCESS: Found 2 high-value orders (Alice and Charlie).")
    else:
        print("\nFAILURE: Unexpected result count.")

    print("\n--- 3. Testing 'this' alias ---")
    # Using dataset_id context
    q2 = "SELECT count(*) as cnt FROM this WHERE region = 'West'"
    res2 = sm.query_data(q2, dataset_id="df_users")
    print(res2)

if __name__ == "__main__":
    verify_sql()
