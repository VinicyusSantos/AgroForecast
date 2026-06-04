from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/trade/")
OUT_PATH = Path("data/processed/fact_exports_monthly.parquet")

NCM_SOJA = 12019000
all_years = []

for year in range(2016, 2027):
    path = RAW_DIR / f"EXP_{year}.csv"

    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        continue

    chunks = []

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        chunksize=500_000,
        usecols=["CO_ANO", "CO_MES", "CO_NCM", "KG_LIQUIDO", "VL_FOB"]
    ):
        soja = chunk[chunk["CO_NCM"] == NCM_SOJA]

        if not soja.empty:
            soja_monthly = (
                soja
                .groupby(["CO_ANO", "CO_MES"], as_index=False)
                .agg(
                    volume_kg=("KG_LIQUIDO", "sum"),
                    valor_fob_usd=("VL_FOB", "sum")
                )
            )
            chunks.append(soja_monthly)

    if chunks:
        year_df = pd.concat(chunks, ignore_index=True)
        all_years.append(year_df)

df_exports = pd.concat(all_years, ignore_index=True)

df_exports = (
    df_exports
    .groupby(["CO_ANO", "CO_MES"], as_index=False)
    .agg(
        volume_kg=("volume_kg", "sum"),
        valor_fob_usd=("valor_fob_usd", "sum")
    )
)

df_exports["ano_mes"] = pd.to_datetime(
    df_exports["CO_ANO"].astype(str)
    + "-"
    + df_exports["CO_MES"].astype(str).str.zfill(2)
    + "-01"
)

df_exports["volume_ton"] = df_exports["volume_kg"] / 1000
df_exports["preco_implicito_usd_ton"] = (
    df_exports["valor_fob_usd"] / df_exports["volume_ton"]
)

df_exports = df_exports[
    [
        "ano_mes",
        "volume_kg",
        "volume_ton",
        "valor_fob_usd",
        "preco_implicito_usd_ton"
    ]
].sort_values("ano_mes")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_exports.to_parquet(OUT_PATH, index=False)

df_exports.head()