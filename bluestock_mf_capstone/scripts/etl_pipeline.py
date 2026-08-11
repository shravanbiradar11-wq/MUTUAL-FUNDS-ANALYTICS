from pathlib import Path
import pandas as pd
import logging
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(file_path):
    """Load a CSV file safely."""

    try:
        logger.info(f"Loading: {file_path.name}")

        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(
                f"{file_path.name} is empty."
            )

        logger.info(
            f"Loaded {file_path.name}: "
            f"{len(df):,} rows x {len(df.columns)} columns"
        )

        return df

    except FileNotFoundError:
        logger.error(
            f"File not found: {file_path}"
        )
        raise

    except pd.errors.EmptyDataError:
        logger.error(
            f"{file_path.name} contains no data."
        )
        raise

    except Exception as exc:
        logger.error(
            f"Error loading {file_path.name}: {exc}"
        )
        raise


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """Standardize column names."""

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """Remove completely duplicated records."""

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    logger.info(
        f"Removed {removed:,} exact duplicate rows"
    )

    return df


# ============================================================
# GENERIC MISSING VALUE HANDLING
# ============================================================

def handle_missing_values(df):
    """Handle basic missing values for non-NAV datasets."""

    df = df.copy()

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    for col in numeric_cols:
        missing = df[col].isna().sum()

        if missing > 0:
            logger.info(
                f"Filling {missing:,} missing values "
                f"in numeric column '{col}' with 0"
            )

            df[col] = df[col].fillna(0)

    return df


# ============================================================
# NAV-SPECIFIC CLEANING
# ============================================================

def clean_nav_data(df, filename):
    """
    Clean mutual fund NAV data.

    Handles:
    - Required columns
    - Date conversion
    - AMFI code cleaning
    - NAV numeric conversion
    - Invalid NAV values
    - Duplicate AMFI code + date
    - Missing calendar dates
    - Weekend/holiday forward filling
    """

    logger.info(
        f"Applying NAV-specific cleaning: {filename}"
    )

    df = df.copy()

    # --------------------------------------------------------
    # 1. Required columns
    # --------------------------------------------------------

    required_columns = {
        "amfi_code",
        "date",
        "nav"
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{filename} is missing required NAV columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # 2. Convert date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:

        logger.warning(
            f"{filename}: "
            f"{invalid_dates:,} invalid dates found."
        )

        df = df.dropna(
            subset=["date"]
        )

    # --------------------------------------------------------
    # 3. Clean AMFI code
    # --------------------------------------------------------

    df["amfi_code"] = (
        df["amfi_code"]
        .astype(str)
        .str.strip()
    )

    invalid_codes = (
        (df["amfi_code"] == "") |
        (df["amfi_code"] == "nan")
    )

    invalid_code_count = invalid_codes.sum()

    if invalid_code_count > 0:

        logger.warning(
            f"{filename}: "
            f"{invalid_code_count:,} invalid AMFI codes found."
        )

        df = df[
            ~invalid_codes
        ]

    # --------------------------------------------------------
    # 4. Convert NAV to numeric
    # --------------------------------------------------------

    df["nav"] = pd.to_numeric(
        df["nav"],
        errors="coerce"
    )

    invalid_nav = df["nav"].isna().sum()

    if invalid_nav > 0:

        logger.warning(
            f"{filename}: "
            f"{invalid_nav:,} invalid NAV values found."
        )

        df = df.dropna(
            subset=["nav"]
        )

    # --------------------------------------------------------
    # 5. NAV must be positive
    # --------------------------------------------------------

    invalid_nav_values = (
        df["nav"] <= 0
    ).sum()

    if invalid_nav_values > 0:

        logger.warning(
            f"{filename}: "
            f"{invalid_nav_values:,} non-positive NAV values found."
        )

        df = df[
            df["nav"] > 0
        ]

    # --------------------------------------------------------
    # 6. Remove duplicate AMFI code + date records
    # --------------------------------------------------------

    duplicate_keys = df.duplicated(
        subset=[
            "amfi_code",
            "date"
        ]
    ).sum()

    if duplicate_keys > 0:

        logger.warning(
            f"{filename}: "
            f"{duplicate_keys:,} duplicate "
            f"AMFI code + date records found."
        )

        df = df.drop_duplicates(
            subset=[
                "amfi_code",
                "date"
            ],
            keep="last"
        )

    # --------------------------------------------------------
    # 7. Sort before reindexing
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "amfi_code",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # 8. Reindex each fund to full daily date range
    # --------------------------------------------------------

    all_funds = []

    for amfi_code, group in df.groupby(
        "amfi_code",
        sort=False
    ):

        group = group.set_index(
            "date"
        )

        # Full calendar date range
        full_dates = pd.date_range(
            start=group.index.min(),
            end=group.index.max(),
            freq="D"
        )

        group = group.reindex(
            full_dates
        )

        # Restore AMFI code
        group["amfi_code"] = amfi_code

        # Forward-fill NAV for
        # weekends and market holidays
        group["nav"] = group["nav"].ffill()

        group.index.name = "date"

        all_funds.append(
            group.reset_index()
        )

    if not all_funds:

        raise ValueError(
            f"{filename}: "
            f"No valid NAV records remain after cleaning."
        )

    df = pd.concat(
        all_funds,
        ignore_index=True
    )

    # --------------------------------------------------------
    # 9. Final sorting
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "amfi_code",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    logger.info(
        f"NAV cleaning completed: "
        f"{len(df):,} rows"
    )

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df, filename):
    """Validate cleaned dataframe."""

    logger.info(
        f"Validating: {filename}"
    )

    # --------------------------------------------------------
    # 1. Dataframe should not be empty
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            f"Validation failed: "
            f"{filename} is empty."
        )

    # --------------------------------------------------------
    # 2. Columns should exist
    # --------------------------------------------------------

    if len(df.columns) == 0:

        raise ValueError(
            f"Validation failed: "
            f"{filename} has no columns."
        )

    # --------------------------------------------------------
    # 3. Check duplicate rows
    # --------------------------------------------------------

    duplicate_count = (
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        logger.warning(
            f"{filename}: "
            f"{duplicate_count:,} duplicate rows remain."
        )

    # --------------------------------------------------------
    # 4. Check missing values
    # --------------------------------------------------------

    missing_values = (
        df.isna().sum().sum()
    )

    if missing_values > 0:

        logger.warning(
            f"{filename}: "
            f"{missing_values:,} missing values remain."
        )

    # --------------------------------------------------------
    # 5. Check infinite numeric values
    # --------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    for col in numeric_cols:

        has_infinite = df[col].isin(
            [
                float("inf"),
                float("-inf")
            ]
        ).any()

        if has_infinite:

            raise ValueError(
                f"Validation failed: "
                f"{filename} contains infinite "
                f"values in column '{col}'."
            )

    # --------------------------------------------------------
    # 6. NAV-specific validation
    # --------------------------------------------------------

    if {
        "amfi_code",
        "date",
        "nav"
    }.issubset(df.columns):

        # NAV must be positive
        invalid_nav = (
            df["nav"] <= 0
        ).sum()

        if invalid_nav > 0:

            raise ValueError(
                f"Validation failed: "
                f"{filename} contains "
                f"{invalid_nav:,} non-positive NAV values."
            )

        # AMFI code + date should be unique
        duplicate_nav_keys = (
            df.duplicated(
                subset=[
                    "amfi_code",
                    "date"
                ]
            ).sum()
        )

        if duplicate_nav_keys > 0:

            raise ValueError(
                f"Validation failed: "
                f"{filename} contains "
                f"{duplicate_nav_keys:,} duplicate "
                f"AMFI code + date records."
            )

    logger.info(
        f"Validation passed: {filename}"
    )

    return True


# ============================================================
# SAVE CLEANED CSV
# ============================================================

def save_csv(df, filename):
    """Save cleaned dataframe."""

    output_path = (
        PROCESSED_DIR / filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN ETL PIPELINE
# ============================================================

def main():
    """Run the complete ETL pipeline."""

    logger.info(
        "================================================"
    )

    logger.info(
        "Starting ETL pipeline"
    )

    logger.info(
        "================================================"
    )

    try:

        # ----------------------------------------------------
        # 1. Check raw data directory
        # ----------------------------------------------------

        if not RAW_DIR.exists():

            raise FileNotFoundError(
                f"Raw data directory not found: "
                f"{RAW_DIR}"
            )

        # ----------------------------------------------------
        # 2. Find CSV files
        # ----------------------------------------------------

        csv_files = list(
            RAW_DIR.glob("*.csv")
        )

        if not csv_files:

            raise FileNotFoundError(
                f"No CSV files found in: "
                f"{RAW_DIR}"
            )

        logger.info(
            f"Found {len(csv_files)} raw CSV files."
        )

        # ----------------------------------------------------
        # 3. Process every CSV
        # ----------------------------------------------------

        for file_path in csv_files:

            logger.info(
                "------------------------------------------------"
            )

            # Extract
            df = load_csv(
                file_path
            )

            # Transform: column names
            df = clean_column_names(
                df
            )

            # Transform: exact duplicates
            df = remove_duplicates(
                df
            )

            # ------------------------------------------------
            # NAV-specific transformation
            # ------------------------------------------------

            if "nav" in file_path.stem.lower():

                df = clean_nav_data(
                    df,
                    file_path.name
                )

            else:

                # Generic transformation
                df = handle_missing_values(
                    df
                )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            validate_dataframe(
                df,
                file_path.name
            )

            # ------------------------------------------------
            # Load
            # ------------------------------------------------

            save_csv(
                df,
                file_path.name
            )

        # ----------------------------------------------------
        # 4. Verify output files
        # ----------------------------------------------------

        output_files = list(
            PROCESSED_DIR.glob("*.csv")
        )

        logger.info(
            f"Processed output files: "
            f"{len(output_files)}"
        )

        if len(output_files) != len(csv_files):

            raise RuntimeError(
                "Output file count does not match "
                "input file count."
            )

        # ----------------------------------------------------
        # 5. Successful completion
        # ----------------------------------------------------

        logger.info(
            "================================================"
        )

        logger.info(
            "ETL PIPELINE COMPLETED SUCCESSFULLY"
        )

        logger.info(
            "================================================"
        )

    except Exception as exc:

        logger.exception(
            f"ETL pipeline failed: {exc}"
        )

        sys.exit(1)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()