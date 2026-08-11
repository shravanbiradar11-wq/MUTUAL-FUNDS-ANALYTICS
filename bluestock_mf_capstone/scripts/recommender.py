from pathlib import Path
import pandas as pd
import numpy as np
import logging
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    BASE_DIR
    / "outputs"
    / "performance_metrics"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "recommendations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Default investor risk profile
DEFAULT_RISK_PROFILE = "moderate"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics():
    """Load calculated fund performance metrics."""

    metrics_file = (
        METRICS_DIR
        / "performance_metrics.csv"
    )

    if not metrics_file.exists():

        raise FileNotFoundError(
            f"Performance metrics file not found: "
            f"{metrics_file}"
        )

    df = pd.read_csv(
        metrics_file
    )

    if df.empty:

        raise ValueError(
            "Performance metrics file is empty."
        )

    logger.info(
        f"Loaded {len(df):,} funds."
    )

    return df


# ============================================================
# CLEAN METRICS
# ============================================================

def clean_metrics(df):
    """Clean metrics required by recommender."""

    df = df.copy()

    required_columns = [
        "amfi_code",
        "cagr_1y_pct",
        "cagr_3y_pct",
        "sharpe_ratio",
        "annualized_volatility_pct",
        "beta_nifty50",
        "historical_var_95_pct",
        "max_drawdown_pct"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "cagr_1y_pct",
        "cagr_3y_pct",
        "sharpe_ratio",
        "annualized_volatility_pct",
        "beta_nifty50",
        "historical_var_95_pct",
        "max_drawdown_pct"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid funds
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "cagr_3y_pct",
            "sharpe_ratio",
            "annualized_volatility_pct",
            "max_drawdown_pct"
        ]
    )

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_score(series, higher_is_better=True):
    """
    Convert values to a 0-100 score.

    Higher value = better if higher_is_better=True.
    """

    series = series.copy()

    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value):

        return pd.Series(
            50,
            index=series.index
        )

    if max_value == min_value:

        return pd.Series(
            50,
            index=series.index
        )

    score = (
        (series - min_value)
        / (max_value - min_value)
        * 100
    )

    if not higher_is_better:

        score = 100 - score

    return score


# ============================================================
# CALCULATE COMPONENT SCORES
# ============================================================

def calculate_component_scores(df):
    """Calculate individual fund quality scores."""

    df = df.copy()

    # --------------------------------------------------------
    # Return score
    # --------------------------------------------------------

    df["return_score"] = min_max_score(
        df["cagr_3y_pct"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Sharpe score
    # --------------------------------------------------------

    df["sharpe_score"] = min_max_score(
        df["sharpe_ratio"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Volatility score
    # Lower volatility = better
    # --------------------------------------------------------

    df["volatility_score"] = min_max_score(
        df["annualized_volatility_pct"],
        higher_is_better=False
    )

    # --------------------------------------------------------
    # Drawdown score
    #
    # Example:
    # -10% is better than -30%
    #
    # Higher value is better.
    # --------------------------------------------------------

    df["drawdown_score"] = min_max_score(
        df["max_drawdown_pct"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Beta score
    #
    # Beta close to 1 is treated as preferable.
    # --------------------------------------------------------

    beta_distance = (
        df["beta_nifty50"] - 1
    ).abs()

    df["beta_score"] = min_max_score(
        beta_distance,
        higher_is_better=False
    )

    return df


# ============================================================
# RISK PROFILE WEIGHTS
# ============================================================

def get_weights(risk_profile):
    """Return scoring weights for investor risk profile."""

    risk_profile = (
        risk_profile
        .lower()
        .strip()
    )

    if risk_profile == "conservative":

        return {
            "return": 0.15,
            "sharpe": 0.30,
            "volatility": 0.25,
            "drawdown": 0.25,
            "beta": 0.05
        }

    if risk_profile == "moderate":

        return {
            "return": 0.25,
            "sharpe": 0.25,
            "volatility": 0.20,
            "drawdown": 0.20,
            "beta": 0.10
        }

    if risk_profile == "aggressive":

        return {
            "return": 0.40,
            "sharpe": 0.20,
            "volatility": 0.10,
            "drawdown": 0.10,
            "beta": 0.20
        }

    raise ValueError(
        "Invalid risk profile. "
        "Use conservative, moderate, or aggressive."
    )


# ============================================================
# CALCULATE FINAL SCORE
# ============================================================

def calculate_final_score(
    df,
    risk_profile
):
    """Calculate weighted recommendation score."""

    df = df.copy()

    weights = get_weights(
        risk_profile
    )

    df["recommendation_score"] = (
        df["return_score"]
        * weights["return"]

        + df["sharpe_score"]
        * weights["sharpe"]

        + df["volatility_score"]
        * weights["volatility"]

        + df["drawdown_score"]
        * weights["drawdown"]

        + df["beta_score"]
        * weights["beta"]
    )

    return df


# ============================================================
# ASSIGN RECOMMENDATION
# ============================================================

def assign_recommendation(score):
    """Convert numerical score into recommendation label."""

    if score >= 80:

        return "Strong Buy"

    if score >= 65:

        return "Buy"

    if score >= 50:

        return "Hold"

    if score >= 35:

        return "Watch"

    return "Avoid"


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    df,
    risk_profile
):
    """Generate ranked fund recommendations."""

    df = calculate_component_scores(
        df
    )

    df = calculate_final_score(
        df,
        risk_profile
    )

    df["recommendation"] = (
        df["recommendation_score"]
        .apply(
            assign_recommendation
        )
    )

    # --------------------------------------------------------
    # Rank funds
    # --------------------------------------------------------

    df = df.sort_values(
        "recommendation_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    df["rank"] = (
        df.index + 1
    )

    # --------------------------------------------------------
    # Add risk profile
    # --------------------------------------------------------

    df["risk_profile"] = (
        risk_profile
    )

    return df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_recommendations(df):
    """Save recommendation results."""

    output_file = (
        OUTPUT_DIR
        / "fund_recommendations.csv"
    )

    columns = [
        "rank",
        "amfi_code",
        "risk_profile",
        "recommendation_score",
        "recommendation",
        "cagr_1y_pct",
        "cagr_3y_pct",
        "sharpe_ratio",
        "annualized_volatility_pct",
        "beta_nifty50",
        "historical_var_95_pct",
        "max_drawdown_pct"
    ]

    output = df[
        [
            col
            for col in columns
            if col in df.columns
        ]
    ]

    output.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Saved recommendations: "
        f"{output_file}"
    )

    return output_file


# ============================================================
# SAVE TOP 5
# ============================================================

def save_top_five(df):
    """Save top five recommendations."""

    top_five = df.head(5)

    output_file = (
        OUTPUT_DIR
        / "top_5_funds.csv"
    )

    top_five.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Saved top 5 funds: "
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run fund recommendation engine."""

    logger.info(
        "================================================"
    )

    logger.info(
        "STARTING FUND RECOMMENDER"
    )

    logger.info(
        "================================================"
    )

    try:

        # ----------------------------------------------------
        # Risk profile
        # ----------------------------------------------------

        risk_profile = (
            sys.argv[1]
            if len(sys.argv) > 1
            else DEFAULT_RISK_PROFILE
        )

        risk_profile = (
            risk_profile
            .lower()
            .strip()
        )

        logger.info(
            f"Risk profile: {risk_profile}"
        )

        # Validate risk profile
        get_weights(
            risk_profile
        )

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        df = load_metrics()

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        df = clean_metrics(
            df
        )

        # ----------------------------------------------------
        # Generate recommendations
        # ----------------------------------------------------

        recommendations = (
            generate_recommendations(
                df,
                risk_profile
            )
        )

        if recommendations.empty:

            raise ValueError(
                "No recommendations generated."
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_recommendations(
            recommendations
        )

        save_top_five(
            recommendations
        )

        # ----------------------------------------------------
        # Display top 5
        # ----------------------------------------------------

        print(
            "\nTOP 5 RECOMMENDED FUNDS"
        )

        print(
            "=============================================="
        )

        print(
            recommendations[
                [
                    "rank",
                    "amfi_code",
                    "recommendation_score",
                    "recommendation"
                ]
            ].head(5).to_string(
                index=False
            )
        )

        print(
            "=============================================="
        )

        logger.info(
            "FUND RECOMMENDER COMPLETED SUCCESSFULLY"
        )

    except Exception as exc:

        logger.exception(
            f"Fund recommender failed: {exc}"
        )

        sys.exit(1)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    