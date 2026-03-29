from dotenv import load_dotenv
import json
import sys
import os

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# Add the current directory to sys.path to ensure crew.py can be imported
sys.path.append(os.path.dirname(__file__))
from crew import crew

# ── Path to test data ─────────────────────────────────────────────────────────
data_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "test_review_subset.json"
)

# ── Prepare Input Data ───────────────────────────────────────────────────────
test_data = []
if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]
else:
    # Fallback to a default case if file is missing
    test_data = [{"user_id": "gfQqQYI5_hCAGEHlHXIz2Q", "item_id": "WMkiheTT-8kRslImVLWMVw"}]

# Get the verified case from test data (index 11)
first_case = test_data[11] if len(test_data) > 11 else test_data[0]
user_id = first_case["user_id"]
item_id = first_case["item_id"]

def run():
    """
    Run the crew for the calculated user/item pair.
    """
    print(f"\n{'='*50}")
    print(f"  STARTING CREWAI LOCAL ENGINE")
    print(f"  user_id = {user_id}")
    print(f"  item_id = {item_id}")
    print(f"{'='*50}\n")

    inputs = {
        "user_id": user_id,
        "item_id": item_id
    }
    result = crew.kickoff(inputs=inputs)
    print("\n=== Final Output ===")
    print(result)

if __name__ == "__main__":
    run()
