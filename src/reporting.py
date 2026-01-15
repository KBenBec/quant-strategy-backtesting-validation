from __future__ import annotations
import os
import pandas as pd
from .risk_metrics import summary


def export_report(bt: pd.DataFrame, weights: pd.DataFrame, out_xlsx: str = "reports/report.xlsx") -> None:
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        bt.to_excel(w, sheet_name="backtest")
        weights.to_excel(w, sheet_name="weights")
        pd.DataFrame([summary(bt)]).to_excel(w, sheet_name="metrics", index=False)
    print(f"Saved: {out_xlsx}")
