import os
import pandas as pd
from sqlalchemy import create_engine

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

processed_dir = os.path.join(BASE_DIR, "data", "processed")
database_dir = os.path.join(BASE_DIR, "database")

os.makedirs(database_dir, exist_ok=True)

engine = create_engine(f"sqlite:///{os.path.join(database_dir, 'bluestock_mf.db')}")

for file in os.listdir(processed_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(processed_dir, file))

        table_name = file.replace(".csv", "")

        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False
        )

        print(f"Loaded {table_name} ({len(df)} rows)")

print("All datasets loaded successfully!")