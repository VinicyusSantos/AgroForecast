from datetime import date
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/macro")
URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"


def _format_bcb_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def collect_bcb_series(
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    end_date = end_date or date.today()
    start_date = start_date or date(end_date.year - 10, end_date.month, end_date.day)

    params = {
        "formato": "json",
        "dataInicial": _format_bcb_date(start_date),
        "dataFinal": _format_bcb_date(end_date),
    }

    response = requests.get(URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Erro da API do BCB: {data.get('message', data['error'])}")

    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"])
    return df


if __name__ == "__main__":
    df = collect_bcb_series()
    df.to_csv(RAW_DIR / "dolar_ptax_raw.csv", index=False, encoding="utf-8")
    print(df.head())
