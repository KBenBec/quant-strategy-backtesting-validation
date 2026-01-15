from __future__ import annotations
import pandas as pd


def load_prices(csv_path: str) -> pd.DataFrame:
    """
    CSV format:
      date, asset, close
    Returns wide dataframe: index=date, columns=assets, values=close
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    px = df.pivot(index="date", columns="asset", values="close").astype(float)
    return px


def align(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = df1.index.intersection(df2.index)
    return df1.loc[idx], df2.loc[idx]
