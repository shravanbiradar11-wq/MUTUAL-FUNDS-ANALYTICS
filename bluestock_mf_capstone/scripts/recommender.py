from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    BASE_DIR /
    "data" /
    "processed"
)

SCORECARD_FILE = (
    PROCESSED_DIR /
    "fund_scorecard.csv"
)


# ============================================================
# LOAD SCORECARD
# ============================================================

def load_scorecard():
    """Load the fund scorecard."""

    if not SCORECARD_FILE.exists():
        raise FileNotFoundError(
            f"Scorecard not found: {SCORECARD_FILE}"
        )

    return pd.read_csv(
        SCORECARD_FILE
    )


# ============================================================
# FUND RECOMMENDER
# ============================================================

def recommend_funds(
    risk_appetite,
    top_n=3
):
    """
    Return top funds by Sharpe Ratio
    within the selected risk grade.

    Parameters
    ----------
    risk_appetite : str
        Low, Moderate or High.

    top_n : int
        Number of funds to recommend.

    Returns
    -------
    pandas.DataFrame
    """

    scorecard = load_scorecard()

    risk_appetite = (
        str(risk_appetite)
        .strip()
        .title()
    )

    valid_risks = [
        "Low",
        "Moderate",
        "High"
    ]

    if risk_appetite not in valid_risks:
        raise ValueError(
            "Risk appetite must be "
            "'Low', 'Moderate' or 'High'."
        )

    filtered = scorecard[
        scorecard["risk_grade"] ==
        risk_appetite
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    filtered["sharpe_ratio"] = pd.to_numeric(
        filtered["sharpe_ratio"],
        errors="coerce"
    )

    recommendations = (
        filtered
        .dropna(
            subset=["sharpe_ratio"]
        )
        .sort_values(
            "sharpe_ratio",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    recommendations.insert(
        0,
        "rank",
        range(
            1,
            len(recommendations) + 1
        )
    )

    recommendations[
        "recommendation"
    ] = "Recommended"

    return recommendations


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BLUESTOCK MUTUAL FUND RECOMMENDER")
    print("=" * 60)

    risk = input(
        "\nEnter risk appetite "
        "(Low / Moderate / High): "
    )

    result = recommend_funds(
        risk
    )

    if result.empty:

        print(
            "\nNo matching funds found."
        )

    else:

        print(
            f"\nTop {len(result)} "
            f"recommendations for "
            f"{risk.title()} risk:"
        )

        print()

        print(
            result.to_string(
                index=False
            )
        )