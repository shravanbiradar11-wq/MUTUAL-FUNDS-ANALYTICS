git status"""
Day 1: Data Ingestion
Loads all provided CSV datasets, inspects them, validates AMFI codes.
"""

import pandas as pd
import os
from pathlib import Path

# ---- Setup paths ----
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
# ---- 1. Load all 10 CSVs ----
csv_files = {
    "nav_history": "nav_history.csv",
    "fund_master": "fund_master.csv",
    # ... add all 10 filenames as per what you were given
}

dataframes = {}

for name, filename in csv_files.items():
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)
    dataframes[name] = df

    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nHead:\n{df.head()}")

    # anomaly checks
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

# ---- 2. Explore fund master ----
fund_master = dataframes["fund_master"]

print("\nUnique fund houses:", fund_master["fund_house"].unique())
print("Unique categories:", fund_master["category"].unique())
print("Unique sub-categories:", fund_master["sub_category"].unique())
print(fund_master.columns.tolist())
print("Unique risk grades:", fund_master["risk_category"].unique())

# ---- 3. Validate AMFI codes ----
nav_history = dataframes["nav_history"]

master_codes = set(fund_master["amfi_code"].unique())
nav_codes = set(nav_history["amfi_code"].unique())

missing_in_nav = master_codes - nav_codes
missing_in_master = nav_codes - master_codes

print(f"\nAMFI codes in fund_master but missing in nav_history: {len(missing_in_nav)}")
print(missing_in_nav)
print(f"AMFI codes in nav_history but missing in fund_master: {len(missing_in_master)}")
print(missing_in_master)

# ---- 4. Data quality summary ----
summary = f"""
DATA QUALITY SUMMARY
---------------------
Total datasets loaded: {len(dataframes)}
Fund master records: {len(fund_master)}
NAV history records: {len(nav_history)}
AMFI codes unmatched (master->nav): {len(missing_in_nav)}
AMFI codes unmatched (nav->master): {len(missing_in_master)}
"""
print(summary)
os.makedirs(PROCESSED_DIR, exist_ok=True)
with open(os.path.join(PROCESSED_DIR, "data_quality_summary.txt"), "w") as f:
    f.write(summary)

# ---- Save cleaned/validated data ----
for name, df in dataframes.items():
    df.to_csv(os.path.join(PROCESSED_DIR, f"{name}_clean.csv"), index=False)

print("\nData ingestion complete.")