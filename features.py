import polars as pl
import pandas as pd

def add_sma(df: pl.LazyFrame | pl.DataFrame, column: str, window: int, group_col: str = "TICKER") -> pl.LazyFrame:
    sma_col = f"SMA_{window}"

    return df.with_columns(
        pl.col(column)
          .rolling_mean(window_size=window, min_periods=window)
          .over(group_col)
          .alias(sma_col)
    )

def add_ema(df: pl.DataFrame, column: str, window: int) -> pl.DataFrame:
    ema_col = f"EMA_{window}"

    df = df.sort(["TICKER", "TIMESTAMP"]).with_columns(
        pl.col(column)
          .ewm_mean(span=window, adjust=False)
          .over("TICKER")
          .alias(ema_col)
    )

    return df

# ================================================================
# MACD: Moving Average Convergence Divergence
# ================================================================
def add_macd(
    df: pl.DataFrame,
    column: str = "CLOSE",
    short_span: int = 12,
    long_span: int = 26,
    signal_span: int = 9
) -> pl.DataFrame:
    """
    Add MACD (Moving Average Convergence Divergence) features to a DataFrame.

    Computes:
        - MACD = EMA(short_span) - EMA(long_span)
        - MACD_SIGNAL = EMA(signal_span) of MACD
        - MACD_HIST = MACD - MACD_SIGNAL

    Parameters
    ----------
    df : pl.DataFrame
        Must include 'TICKER', 'TIMESTAMP', and a numeric column (default: 'CLOSE').
    column : str
        Column to use for MACD calculation.
    short_span : int
        Window for short EMA (default 12).
    long_span : int
        Window for long EMA (default 26).
    signal_span : int
        Window for signal EMA (default 9).

    Returns
    -------
    pl.DataFrame
        DataFrame with added columns: MACD, MACD_SIGNAL, MACD_HIST.
    """
    df = df.sort(["TICKER", "TIMESTAMP"])

    # Step 1: Compute the short and long EMAs, then MACD line
    # Must materialize MACD before applying another window function to it
    df = df.with_columns([
        pl.col(column).ewm_mean(span=short_span, adjust=False).over("TICKER").alias("_short_ema"),
        pl.col(column).ewm_mean(span=long_span, adjust=False).over("TICKER").alias("_long_ema")
    ])
    
    # Step 2: Compute MACD from the materialized EMAs
    df = df.with_columns([
        (pl.col("_short_ema") - pl.col("_long_ema")).alias("MACD")
    ])
    
    # Step 3: Compute signal line (EMA of MACD) - now MACD is a real column
    df = df.with_columns([
        pl.col("MACD").ewm_mean(span=signal_span, adjust=False).over("TICKER").alias("MACD_SIGNAL")
    ])
    
    # Step 4: Compute histogram (MACD - Signal)
    df = df.with_columns([
        (pl.col("MACD") - pl.col("MACD_SIGNAL")).alias("MACD_HIST")
    ])
    
    # Clean up temporary columns
    df = df.drop(["_short_ema", "_long_ema"])

    return df

def add_bollinger_bands(
    df: pl.DataFrame,
    price_col: str = "CLOSE",
    window: int = 20,
    k: float = 2.0,
) -> pl.DataFrame:
    """
    Add Bollinger Bands to DataFrame per ticker.
    Produces: BB_MID, BB_UPPER, BB_LOWER
    """
    df = df.sort(["TICKER", "TIMESTAMP"])

    sma = pl.col(price_col).rolling_mean(window_size=window).over("TICKER")
    std = pl.col(price_col).rolling_std(window_size=window).over("TICKER")

    upper = (sma + k * std).alias(f"BB_UPPER_{window}")
    lower = (sma - k * std).alias(f"BB_LOWER_{window}")
    mid = sma.alias(f"BB_MID_{window}")

    return df.with_columns([mid, upper, lower])

def add_rsi(df: pl.LazyFrame | pl.DataFrame, price_col: str = "CLOSE", window: int = 14) -> pl.DataFrame:
    """
    Add RSI (Relative Strength Index) per ticker using pandas internally.
    Produces: RSI_{window}
    """
    import pandas as pd

    # Collect if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    def _rsi_per_ticker(batch: pl.DataFrame) -> pl.DataFrame:
        pdf = batch.to_pandas()
        delta = pdf[price_col].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=window, adjust=False).mean()
        avg_loss = loss.ewm(span=window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        pdf[f"RSI_{window}"] = rsi

        return pl.from_pandas(pdf)

    out = []
    for group in df.partition_by("TICKER", maintain_order=True):
        out.append(_rsi_per_ticker(group))
    return pl.concat(out)
