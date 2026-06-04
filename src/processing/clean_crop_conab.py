import pandas as pd
from pathlib import Path
import re

RAW_PATH = Path("data/raw/crop/SojaSerieHist.xls")
OUT_PATH = Path("data/processed/fact_supply_safra.parquet")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SHEETS = {
    "Área": "area_mil_ha",
    "Produtividade": "produtividade_kg_ha",
    "Produção": "producao_mil_ton",
}


def clean_safra(value):
    value = str(value).strip()
    match = re.search(r"\d{4}/\d{2}", value)
    if match:
        return match.group(0)
    return None


def extract_brasil_row(sheet_name, value_col):
    df = pd.read_excel(
        RAW_PATH,
        sheet_name=sheet_name,
        header=None
    )

    safra_headers = df.iloc[5].tolist()

    brasil_mask = df[0].astype(str).str.strip().str.upper().eq("BRASIL")

    if not brasil_mask.any():
        raise ValueError(f"Linha BRASIL não encontrada na aba {sheet_name}")

    brasil_row = df.loc[brasil_mask].iloc[0].tolist()

    records = []

    for col_idx in range(1, len(safra_headers)):
        safra = clean_safra(safra_headers[col_idx])

        if safra is None:
            continue

        value = brasil_row[col_idx]

        records.append({
            "safra": safra,
            value_col: value
        })

    return pd.DataFrame(records)

dfs = []

for sheet_name, value_col in SHEETS.items():
    temp = extract_brasil_row(sheet_name, value_col)
    dfs.append(temp)

df_supply = dfs[0]

for temp in dfs[1:]:
    df_supply = df_supply.merge(
        temp,
        on="safra",
        how="outer"
    )

df_supply["safra_inicio"] = df_supply["safra"].str[:4].astype(int)
df_supply["safra_fim"] = df_supply["safra_inicio"] + 1

df_supply = df_supply.sort_values("safra_inicio").reset_index(drop=True)

df_supply.to_parquet(OUT_PATH, index=False)

df_supply.head()