import polars as pl

def preprocess_data(
    filepath: str = "data/5-min-all.csv",
    min_samples_per_ticker: int = 1000,
    infer_schema_length: int = 10_000,
) -> pl.LazyFrame:
    """
    Preprocess 5-minute financial data for modeling.

    Steps:
    1. Load CSV and apply schema overrides
    2. Build unified TIMESTAMP column
    3. Filter tickers with enough data
    4. Keep only strict 5-minute intervals
    5. Add targets: CLOSE_FWD_1, RET_FWD_1, UP_NEXT

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    min_samples_per_ticker : int
        Minimum number of rows per ticker to keep.
    infer_schema_length : int
        Number of rows to infer schema types.

    Returns
    -------
    pl.LazyFrame
        Preprocessed Polars LazyFrame (can call `.collect()` to materialize).
    """

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    df = pl.scan_csv(
        filepath,
        schema_overrides={"VOL": pl.Float64},  # VOL often has float values
        infer_schema_length=infer_schema_length,
    )

    # -------------------------------------------------------------------------
    # 2. BUILD TIMESTAMP
    # -------------------------------------------------------------------------
    df = (
        df.with_columns(
            (
                pl.col("DATE").cast(pl.Utf8) + " " +
                pl.col("TIME").cast(pl.Utf8).str.zfill(6)
            )
            .str.strptime(pl.Datetime, format="%Y%m%d %H%M%S")
            .alias("TIMESTAMP")
        )
        .drop(["PER", "OPENINT", "DATE", "TIME"])
    )

    # -------------------------------------------------------------------------
    # 3. FILTER TICKERS WITH ENOUGH DATA
    # -------------------------------------------------------------------------
    ticker_counts = (
        df.group_by("TICKER")
          .agg(pl.len().alias("n_rows"))
          .collect()
    )

    valid_tickers = (
        ticker_counts
        .filter(pl.col("n_rows") > min_samples_per_ticker)
        ["TICKER"]
        .to_list()
    )

    df = df.filter(pl.col("TICKER").is_in(valid_tickers))

    # -------------------------------------------------------------------------
    # 4. SORT + DELTA CHECK
    # -------------------------------------------------------------------------
    df = (
        df.sort(["TICKER", "TIMESTAMP"])
          .with_columns(
              pl.col("TIMESTAMP")
                .diff()
                .over("TICKER")
                .alias("DELTA")
          )
    )

    # Keep only strict 5-minute steps
    df = df.filter(pl.col("DELTA") == pl.duration(minutes=5)).drop("DELTA")

    # -------------------------------------------------------------------------
    # 5. ADD TARGETS (NEXT CLOSE, FORWARD RETURN, DIRECTION)
    # -------------------------------------------------------------------------
    df = (
        df.with_columns(
            # next bar's close, per ticker
            pl.col("CLOSE")
              .shift(-1)
              .over("TICKER")
              .alias("CLOSE_FWD_1")
        )
        .with_columns([
            # forward 5-min return
            ((pl.col("CLOSE_FWD_1") - pl.col("CLOSE")) / pl.col("CLOSE"))
                .alias("RET_FWD_1"),
            # classification label: 1 if next close is higher
            (pl.col("CLOSE_FWD_1") > pl.col("CLOSE"))
                .cast(pl.Int8)
                .alias("UP_NEXT"),
        ])
        .drop_nulls(subset=["CLOSE_FWD_1"])
    )

    # Return lazy frame (materialize later with .collect())
    return df
