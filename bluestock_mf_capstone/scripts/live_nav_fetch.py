from pathlib import Path
import requests
import pandas as pd
import logging
import sys
import time


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# MFAPI endpoint
API_URL = "https://api.mfapi.in/mf/{scheme_code}"

# Output file
OUTPUT_FILE = RAW_DIR / "live_nav.csv"

# Request timeout
REQUEST_TIMEOUT = 30

# Delay between API requests
REQUEST_DELAY = 1


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# GET SCHEME CODES
# ============================================================

def get_scheme_codes():
    """
    Get AMFI scheme codes.

    First tries to read scheme codes from an existing
    NAV CSV file. If unavailable, uses the predefined
    list below.
    """

    possible_files = [
        RAW_DIR / "nav.csv",
        RAW_DIR / "NAV.csv",
        RAW_DIR / "nav_data.csv",
        RAW_DIR / "scheme_master.csv"
    ]

    for file_path in possible_files:

        if file_path.exists():

            try:

                df = pd.read_csv(
                    file_path,
                    nrows=100000
                )

                df.columns = (
                    df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(
                        " ",
                        "_",
                        regex=False
                    )
                )

                if "amfi_code" in df.columns:

                    codes = (
                        df["amfi_code"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .unique()
                        .tolist()
                    )

                    if codes:

                        logger.info(
                            f"Found {len(codes)} "
                            f"AMFI scheme codes from "
                            f"{file_path.name}"
                        )

                        return codes

            except Exception as exc:

                logger.warning(
                    f"Could not read {file_path.name}: {exc}"
                )

    # --------------------------------------------------------
    # Fallback scheme codes
    # --------------------------------------------------------

    logger.warning(
        "No existing AMFI code file found."
    )

    logger.warning(
        "Using predefined scheme codes."
    )

    return [
        "120001",
        "120015",
        "120027",
        "120034",
        "120040"
    ]


# ============================================================
# FETCH NAV FOR ONE SCHEME
# ============================================================

def fetch_scheme_nav(scheme_code):
    """
    Fetch NAV history for one mutual fund scheme.
    """

    url = API_URL.format(
        scheme_code=scheme_code
    )

    try:

        logger.info(
            f"Fetching NAV for scheme {scheme_code}"
        )

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if "data" not in data:

            raise ValueError(
                f"No NAV data returned for "
                f"scheme {scheme_code}"
            )

        records = []

        for item in data["data"]:

            records.append(
                {
                    "amfi_code": scheme_code,
                    "date": item.get("date"),
                    "nav": item.get("nav")
                }
            )

        if not records:

            logger.warning(
                f"No records found for "
                f"scheme {scheme_code}"
            )

            return pd.DataFrame(
                columns=[
                    "amfi_code",
                    "date",
                    "nav"
                ]
            )

        df = pd.DataFrame(
            records
        )

        return df

    except requests.exceptions.Timeout:

        logger.error(
            f"Timeout while fetching "
            f"scheme {scheme_code}"
        )

        return pd.DataFrame()

    except requests.exceptions.RequestException as exc:

        logger.error(
            f"API request failed for "
            f"scheme {scheme_code}: {exc}"
        )

        return pd.DataFrame()

    except Exception as exc:

        logger.error(
            f"Error processing scheme "
            f"{scheme_code}: {exc}"
        )

        return pd.DataFrame()


# ============================================================
# CLEAN FETCHED NAV
# ============================================================

def clean_nav_data(df):
    """
    Clean NAV data returned by MFAPI.
    """

    if df.empty:

        return df

    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce"
    )

    # Convert NAV
    df["nav"] = pd.to_numeric(
        df["nav"],
        errors="coerce"
    )

    # Remove invalid records
    df = df.dropna(
        subset=[
            "amfi_code",
            "date",
            "nav"
        ]
    )

    # NAV must be positive
    df = df[
        df["nav"] > 0
    ]

    # Remove duplicate scheme/date
    df = df.drop_duplicates(
        subset=[
            "amfi_code",
            "date"
        ],
        keep="last"
    )

    # Sort
    df = df.sort_values(
        [
            "amfi_code",
            "date"
        ]
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "================================================"
    )

    logger.info(
        "STARTING LIVE NAV FETCH"
    )

    logger.info(
        "================================================"
    )

    try:

        # ----------------------------------------------------
        # Get scheme codes
        # ----------------------------------------------------

        scheme_codes = get_scheme_codes()

        if not scheme_codes:

            raise ValueError(
                "No AMFI scheme codes available."
            )

        logger.info(
            f"Fetching NAV for "
            f"{len(scheme_codes)} schemes"
        )

        all_data = []

        # ----------------------------------------------------
        # Fetch each scheme
        # ----------------------------------------------------

        for index, scheme_code in enumerate(
            scheme_codes,
            start=1
        ):

            logger.info(
                f"[{index}/{len(scheme_codes)}] "
                f"Scheme: {scheme_code}"
            )

            df = fetch_scheme_nav(
                scheme_code
            )

            if not df.empty:

                all_data.append(
                    df
                )

            # Avoid excessive API requests
            time.sleep(
                REQUEST_DELAY
            )

        # ----------------------------------------------------
        # Check results
        # ----------------------------------------------------

        if not all_data:

            raise RuntimeError(
                "No NAV data was successfully fetched."
            )

        # ----------------------------------------------------
        # Combine all schemes
        # ----------------------------------------------------

        final_df = pd.concat(
            all_data,
            ignore_index=True
        )

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        final_df = clean_nav_data(
            final_df
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        final_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        logger.info(
            f"Saved {len(final_df):,} NAV records"
        )

        logger.info(
            f"Output: {OUTPUT_FILE}"
        )

        logger.info(
            "================================================"
        )

        logger.info(
            "LIVE NAV FETCH COMPLETED SUCCESSFULLY"
        )

        logger.info(
            "================================================"
        )

    except Exception as exc:

        logger.exception(
            f"Live NAV fetch failed: {exc}"
        )

        sys.exit(1)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()