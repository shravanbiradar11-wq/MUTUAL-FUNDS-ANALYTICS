import os
import logging
import requests
import pandas as pd

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

#BASE_URL = "https://api.mfapi.in/mf/search"
NAV_URL = "https://api.mfapi.in/mf/{}"

RAW_DIR = "data/raw"

scheme_codes = {
    "SBI Small Cap Fund - Direct Plan - Growth": 125497,
    "Aditya Birla Sun Life Banking & PSU Debt Fund  - DIRECT - IDCW": 119551,
    "Axis ELSS Tax Saver Fund - Direct Plan - Growth Option": 120503,
    "Nippon India Large Cap Fund - Direct Plan Growth Plan - Growth Option": 118632,
    "HDFC Money Market Fund - Growth Option - Direct Plan": 119092,
    "quant Mid Cap Fund - Growth Option - Direct Plan": 120841,
}
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------

def fetch_nav(scheme_code: int, scheme_name: str) -> pd.DataFrame:
    """
    Fetch NAV history for a mutual fund scheme.

    Returns:
        pd.DataFrame
    """

    url = NAV_URL.format(scheme_code)

    response = requests.get(url,timeout=30,headers={"User-Agent": "MutualFundAnalytics/1.0"})
    response.raise_for_status()

    data = response.json()
    
    
    
    if "data" not in data:
        raise ValueError(f"No NAV data found for {scheme_name}")

    nav_df = pd.DataFrame(data["data"])

    meta = data.get("meta", {})
    logging.info(f"Requested : {scheme_name}")
    logging.info(f"Returned  : {meta.get('scheme_name')}")
    logging.info(f"AMFI Code : {meta.get('scheme_code')}")
    logging.info("-" * 40)
    nav_df["scheme_code"] = meta.get("scheme_code")
    nav_df["scheme_name"] = meta.get("scheme_name")
    nav_df["fund_house"] = meta.get("fund_house")

    nav_df["date"] = pd.to_datetime(
        nav_df["date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    nav_df["nav"] = pd.to_numeric(
        nav_df["nav"],
        errors="coerce"
    )

    return nav_df


def save_nav(df: pd.DataFrame, filename: str) -> None:
    """
    Save NAV dataframe as CSV.
    """

    os.makedirs(RAW_DIR, exist_ok=True)

    path = os.path.join(RAW_DIR, filename)

    df.to_csv(path, index=False)

    logging.info(f"Saved -> {path}")


def print_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics.
    """

    latest = df.sort_values("date", ascending=False).iloc[0]

    print("\n--------------------------------")
    print(f"Scheme Name       : {latest['scheme_name']}")
    print(f"Fund House        : {latest['fund_house']}")
    print(f"Total Records     : {len(df)}")
    print(f"Latest NAV Date   : {latest['date'].date()}")
    print(f"Latest NAV Value  : {latest['nav']}")
    print("--------------------------------")


def main():

    os.makedirs(RAW_DIR, exist_ok=True)

    all_nav_frames = []

    for scheme_name, scheme_code in scheme_codes.items():

        logging.info(f"Fetching NAV for {scheme_name}")

        try:

            df = fetch_nav(scheme_code, scheme_name)

            save_nav(df, f"{scheme_name}.csv")

            print_summary(df)

            all_nav_frames.append(df)

        except Exception as e:

            logging.error(f"{scheme_name} failed : {e}")

    if all_nav_frames:

        combined = pd.concat(all_nav_frames, ignore_index=True)

        combined_path = os.path.join(
            RAW_DIR,
            "live_nav_fetched.csv"
        )

        combined.to_csv(combined_path, index=False)

        logging.info(f"Combined CSV saved -> {combined_path}")

    logging.info("All schemes processed successfully.")


if __name__ == "__main__":
    main()