from pathlib import Path
import pandas as pd
import numpy as np
import logging
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "performance_metrics"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RISK_FREE_RATE = 0.065
TRADING_DAYS = 252
VAR_CONFIDENCE = 0.95


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# FIND FILE
# ============================================================

def find_file(possible_names):
    """Find the first matching CSV file."""

    for name in possible_names:

        path = PROCESSED_DIR / name

        if path.exists():

            logger.info(
                f"Using file: {path}"
            )

            return path

    raise FileNotFoundError(
        f"Could not find any of: {possible_names}"
    )


# ============================================================
# LOAD NAV DATA
# ============================================================

def load_nav_data():
    """Load cleaned mutual fund NAV data."""

    nav_file = find_file(
        [
            "nav.csv",
            "NAV.csv",
            "nav_data.csv",
            "live_nav.csv"
        ]
    )

    df = pd.read_csv(
        nav_file
    )

    # Standardize column names
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

    required = {
        "amfi_code",
        "date",
        "nav"
    }

    missing = (
        required - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"NAV file missing columns: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["nav"] = pd.to_numeric(
        df["nav"],
        errors="coerce"
    )

    df["amfi_code"] = (
        df["amfi_code"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "amfi_code",
            "date",
            "nav"
        ]
    )

    df = df[
        df["nav"] > 0
    ]

    df = df.drop_duplicates(
        subset=[
            "amfi_code",
            "date"
        ]
    )

    df = df.sort_values(
        [
            "amfi_code",
            "date"
        ]
    )

    logger.info(
        f"NAV records loaded: {len(df):,}"
    )

    return df


# ============================================================
# LOAD BENCHMARK
# ============================================================

def load_benchmark_data():
    """Load benchmark data."""

    benchmark_file = find_file(
        [
            "benchmark.csv",
            "benchmarks.csv",
            "benchmark_data.csv"
        ]
    )

    df = pd.read_csv(
        benchmark_file
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

    logger.info(
        f"Benchmark columns: "
        f"{list(df.columns)}"
    )

    return df


# ============================================================
# PREPARE BENCHMARK
# ============================================================

def prepare_benchmark(df):
    """
    Prepare NIFTY 50 benchmark.

    Handles common column names automatically.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Find date column
    # --------------------------------------------------------

    date_candidates = [
        "date",
        "nav_date",
        "index_date"
    ]

    date_col = next(
        (
            col
            for col in date_candidates
            if col in df.columns
        ),
        None
    )

    if date_col is None:

        raise ValueError(
            "Could not identify benchmark date column."
        )

    # --------------------------------------------------------
    # Find NIFTY 50 column
    # --------------------------------------------------------

    nifty_candidates = [
        "nifty_50",
        "nifty50",
        "nifty_50_close",
        "nifty50_close",
        "nifty_50_close_value",
        "close_value"
    ]

    nifty_col = next(
        (
            col
            for col in nifty_candidates
            if col in df.columns
        ),
        None
    )

    # --------------------------------------------------------
    # If benchmark has index_name + close_value
    # --------------------------------------------------------

    if (
        nifty_col is None
        and "index_name" in df.columns
        and "close_value" in df.columns
    ):

        nifty = df[
            df["index_name"]
            .astype(str)
            .str.upper()
            .str.strip()
            == "NIFTY 50"
        ].copy()

        nifty_col = "close_value"

    else:

        nifty = df.copy()

    if nifty_col is None:

        raise ValueError(
            "Could not identify NIFTY 50 value column."
        )

    nifty["date"] = pd.to_datetime(
        nifty[date_col],
        errors="coerce"
    )

    nifty["nifty50"] = pd.to_numeric(
        nifty[nifty_col],
        errors="coerce"
    )

    nifty = nifty[
        [
            "date",
            "nifty50"
        ]
    ]

    nifty = nifty.dropna()

    nifty = nifty[
        nifty["nifty50"] > 0
    ]

    nifty = nifty.drop_duplicates(
        subset=["date"]
    )

    nifty = nifty.sort_values(
        "date"
    )

    logger.info(
        f"NIFTY 50 records loaded: "
        f"{len(nifty):,}"
    )

    return nifty


# ============================================================
# DAILY RETURNS
# ============================================================

def calculate_daily_returns(df):
    """Calculate daily NAV returns."""

    df = df.copy()

    df["daily_return"] = (
        df.groupby("amfi_code")["nav"]
        .pct_change()
    )

    return df


# ============================================================
# CAGR
# ============================================================

def calculate_cagr(group, years):
    """
    Calculate CAGR using trading-day annualisation.

    CAGR =
    (Ending NAV / Beginning NAV) ^ (252 / N) - 1

    where N = number of observed trading days.
    """

    group = group.sort_values(
        "date"
    )

    end_date = group["date"].max()

    start_target = (
        end_date
        - pd.DateOffset(years=years)
    )

    period = group[
        group["date"] >= start_target
    ].copy()

    if len(period) < 2:

        return np.nan

    start_nav = period.iloc[0]["nav"]
    end_nav = period.iloc[-1]["nav"]

    n_trading_days = len(period) - 1

    if (
        start_nav <= 0
        or end_nav <= 0
        or n_trading_days <= 0
    ):

        return np.nan

    cagr = (
        (end_nav / start_nav)
        ** (TRADING_DAYS / n_trading_days)
        - 1
    )

    return cagr * 100


# ============================================================
# SHARPE RATIO
# ============================================================

def calculate_sharpe(returns):
    """
    Calculate annualised Sharpe ratio.

    Risk-free rate = 6.5% annually.
    """

    returns = returns.dropna()

    if len(returns) < 2:

        return np.nan

    daily_rf = (
        (1 + RISK_FREE_RATE)
        ** (1 / TRADING_DAYS)
        - 1
    )

    excess_returns = (
        returns - daily_rf
    )

    volatility = excess_returns.std(
        ddof=1
    )

    if volatility == 0:

        return np.nan

    sharpe = (
        excess_returns.mean()
        / volatility
        * np.sqrt(TRADING_DAYS)
    )

    return sharpe


# ============================================================
# BETA
# ============================================================

def calculate_beta(
    fund_returns,
    benchmark_returns
):
    """Calculate beta against NIFTY 50."""

    combined = pd.concat(
        [
            fund_returns,
            benchmark_returns
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(combined) < 2:

        return np.nan

    fund = combined.iloc[:, 0]
    benchmark = combined.iloc[:, 1]

    benchmark_variance = (
        benchmark.var(
            ddof=1
        )
    )

    if benchmark_variance == 0:

        return np.nan

    covariance = np.cov(
        fund,
        benchmark,
        ddof=1
    )[0, 1]

    beta = (
        covariance
        / benchmark_variance
    )

    return beta


# ============================================================
# HISTORICAL VaR
# ============================================================

def calculate_var(returns):
    """
    Calculate historical 95% daily VaR.

    VaR is reported as a positive loss percentage.
    """

    returns = returns.dropna()

    if len(returns) < 2:

        return np.nan

    percentile = (
        1 - VAR_CONFIDENCE
    ) * 100

    quantile = np.percentile(
        returns,
        percentile
    )

    var = -quantile * 100

    return var


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(group):
    """Calculate maximum drawdown."""

    group = group.sort_values(
        "date"
    ).copy()

    running_max = (
        group["nav"]
        .cummax()
    )

    drawdown = (
        group["nav"]
        / running_max
        - 1
    )

    return drawdown.min() * 100


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    nav_df,
    benchmark_df
):
    """Calculate all fund performance metrics."""

    logger.info(
        "Calculating performance metrics..."
    )

    # --------------------------------------------------------
    # Benchmark returns
    # --------------------------------------------------------

    benchmark_df = benchmark_df.copy()

    benchmark_df["benchmark_return"] = (
        benchmark_df["nifty50"]
        .pct_change()
    )

    benchmark_df = benchmark_df[
        [
            "date",
            "benchmark_return"
        ]
    ].dropna()

    benchmark_series = (
        benchmark_df
        .set_index("date")["benchmark_return"]
    )

    # --------------------------------------------------------
    # Calculate daily fund returns
    # --------------------------------------------------------

    nav_df = calculate_daily_returns(
        nav_df
    )

    results = []

    # --------------------------------------------------------
    # Process each fund
    # --------------------------------------------------------

    for amfi_code, group in nav_df.groupby(
        "amfi_code"
    ):

        group = group.sort_values(
            "date"
        ).copy()

        returns = group[
            "daily_return"
        ].dropna()

        return_series = (
            group
            .set_index("date")["daily_return"]
            .dropna()
        )

        # ----------------------------------------------------
        # CAGR
        # ----------------------------------------------------

        cagr_1y = calculate_cagr(
            group,
            1
        )

        cagr_3y = calculate_cagr(
            group,
            3
        )

        cagr_5y = calculate_cagr(
            group,
            5
        )

        # ----------------------------------------------------
        # Sharpe
        # ----------------------------------------------------

        sharpe = calculate_sharpe(
            returns
        )

        # ----------------------------------------------------
        # Beta
        # ----------------------------------------------------

        beta = calculate_beta(
            return_series,
            benchmark_series
        )

        # ----------------------------------------------------
        # VaR
        # ----------------------------------------------------

        var_95 = calculate_var(
            returns
        )

        # ----------------------------------------------------
        # Maximum drawdown
        # ----------------------------------------------------

        max_drawdown = (
            calculate_max_drawdown(
                group
            )
        )

        # ----------------------------------------------------
        # Basic statistics
        # ----------------------------------------------------

        volatility = (
            returns.std()
            * np.sqrt(TRADING_DAYS)
            * 100
        )

        average_daily_return = (
            returns.mean() * 100
        )

        results.append(
            {
                "amfi_code": amfi_code,
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "observations": len(group),
                "cagr_1y_pct": cagr_1y,
                "cagr_3y_pct": cagr_3y,
                "cagr_5y_pct": cagr_5y,
                "annualized_volatility_pct": volatility,
                "sharpe_ratio": sharpe,
                "beta_nifty50": beta,
                "historical_var_95_pct": var_95,
                "max_drawdown_pct": max_drawdown,
                "average_daily_return_pct": average_daily_return
            }
        )

    metrics = pd.DataFrame(
        results
    )

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(metrics):
    """Save performance metrics."""

    output_file = (
        OUTPUT_DIR
        / "performance_metrics.csv"
    )

    metrics.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Saved metrics: {output_file}"
    )

    # --------------------------------------------------------
    # Save separate CAGR comparison
    # --------------------------------------------------------

    cagr_file = (
        OUTPUT_DIR
        / "cagr_comparison.csv"
    )

    metrics[
        [
            "amfi_code",
            "cagr_1y_pct",
            "cagr_3y_pct",
            "cagr_5y_pct"
        ]
    ].to_csv(
        cagr_file,
        index=False
    )

    # --------------------------------------------------------
    # Save risk metrics
    # --------------------------------------------------------

    risk_file = (
        OUTPUT_DIR
        / "risk_metrics.csv"
    )

    metrics[
        [
            "amfi_code",
            "annualized_volatility_pct",
            "sharpe_ratio",
            "beta_nifty50",
            "historical_var_95_pct",
            "max_drawdown_pct"
        ]
    ].to_csv(
        risk_file,
        index=False
    )

    logger.info(
        f"Saved CAGR comparison: {cagr_file}"
    )

    logger.info(
        f"Saved risk metrics: {risk_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete performance metrics pipeline."""

    logger.info(
        "================================================"
    )

    logger.info(
        "STARTING PERFORMANCE METRICS CALCULATION"
    )

    logger.info(
        "================================================"
    )

    try:

        # ----------------------------------------------------
        # Load NAV
        # ----------------------------------------------------

        nav_df = load_nav_data()

        # ----------------------------------------------------
        # Load benchmark
        # ----------------------------------------------------

        benchmark_raw = load_benchmark_data()

        benchmark_df = prepare_benchmark(
            benchmark_raw
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            nav_df,
            benchmark_df
        )

        if metrics.empty:

            raise ValueError(
                "No performance metrics were calculated."
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_results(
            metrics
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        logger.info(
            f"Funds processed: "
            f"{len(metrics):,}"
        )

        logger.info(
            "================================================"
        )

        logger.info(
            "PERFORMANCE METRICS COMPLETED SUCCESSFULLY"
        )

        logger.info(
            "================================================"
        )

    except Exception as exc:

        logger.exception(
            f"Performance metrics failed: {exc}"
        )

        sys.exit(1)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()