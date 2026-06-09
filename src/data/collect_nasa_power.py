from pathlib import Path
import time
import requests
import pandas as pd


START_DATE = "20160101"
END_DATE = "20260531"

RAW_DIR = Path("data/raw/weather/nasa_power")
PROCESSED_DIR = Path("data/processed/weather")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = [
    {
        "uf": "MT",
        "local": "sorriso",
        "latitude": -12.545,
        "longitude": -55.711,
        "peso": 0.30,
    },
    {
        "uf": "MT",
        "local": "cuiaba",
        "latitude": -15.601,
        "longitude": -56.097,
        "peso": 0.10,
    },
    {
        "uf": "MS",
        "local": "dourados",
        "latitude": -22.221,
        "longitude": -54.806,
        "peso": 0.10,
    },
    {
        "uf": "GO",
        "local": "rio_verde",
        "latitude": -17.792,
        "longitude": -50.919,
        "peso": 0.12,
    },
    {
        "uf": "PR",
        "local": "londrina",
        "latitude": -23.304,
        "longitude": -51.169,
        "peso": 0.15,
    },
    {
        "uf": "RS",
        "local": "passo_fundo",
        "latitude": -28.262,
        "longitude": -52.409,
        "peso": 0.13,
    },
    {
        "uf": "BA",
        "local": "luis_eduardo_magalhaes",
        "latitude": -12.095,
        "longitude": -45.786,
        "peso": 0.10,
    },
]

PARAMETERS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
]

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_location(location: dict) -> dict:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": location["longitude"],
        "latitude": location["latitude"],
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }

    response = requests.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()

    return response.json()


def save_raw_json(location: dict, data: dict):
    path = RAW_DIR / f"{location['uf']}_{location['local']}.json"
    pd.Series(data).to_json(path, force_ascii=False)


def parse_power_json(location: dict, data: dict) -> pd.DataFrame:
    parameters = data["properties"]["parameter"]

    frames = []

    for param_name, values in parameters.items():
        temp = pd.DataFrame(
            {
                "data": list(values.keys()),
                param_name: list(values.values()),
            }
        )
        frames.append(temp)

    df = frames[0]

    for temp in frames[1:]:
        df = df.merge(temp, on="data", how="outer")

    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d", errors="coerce")

    df = df.rename(
        columns={
            "T2M": "temp_media",
            "T2M_MAX": "temp_max",
            "T2M_MIN": "temp_min",
            "PRECTOTCORR": "chuva_mm",
            "RH2M": "umidade_media",
        }
    )

    df["uf"] = location["uf"]
    df["local"] = location["local"]
    df["latitude"] = location["latitude"]
    df["longitude"] = location["longitude"]
    df["peso"] = location["peso"]

    cols = [
        "data",
        "uf",
        "local",
        "latitude",
        "longitude",
        "peso",
        "chuva_mm",
        "temp_media",
        "temp_max",
        "temp_min",
        "umidade_media",
    ]

    df = df[cols]

    for col in ["chuva_mm", "temp_media", "temp_max", "temp_min", "umidade_media"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-999, pd.NA)

    return df


def collect_daily() -> pd.DataFrame:
    dfs = []

    for location in LOCATIONS:
        print(f"Coletando {location['uf']} - {location['local']}")

        data = fetch_location(location)
        save_raw_json(location, data)

        df = parse_power_json(location, data)
        dfs.append(df)

        time.sleep(1)

    df_daily = pd.concat(dfs, ignore_index=True)

    return df_daily


def build_weekly_by_location(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df["data"] = pd.to_datetime(df["data"])

    df_weekly = (
        df
        .set_index("data")
        .groupby(["uf", "local", "latitude", "longitude", "peso"])
        .resample("W-FRI")
        .agg(
            chuva_mm_1w=("chuva_mm", "sum"),
            temp_media_1w=("temp_media", "mean"),
            temp_max_1w=("temp_max", "max"),
            temp_min_1w=("temp_min", "min"),
            umidade_media_1w=("umidade_media", "mean"),
            dias_secos_1w=("chuva_mm", lambda x: (x < 1).sum()),
            dias_calor_extremo_1w=("temp_max", lambda x: (x >= 35).sum()),
        )
        .reset_index()
        .rename(columns={"data": "semana"})
    )

    return df_weekly


def build_weekly_soy_index(df_weekly_location: pd.DataFrame) -> pd.DataFrame:
    df = df_weekly_location.copy()

    # Normaliza pesos por semana, para evitar problema caso alguma localidade falhe
    df["peso_norm"] = df.groupby("semana")["peso"].transform(lambda x: x / x.sum())

    weighted_cols = [
        "chuva_mm_1w",
        "temp_media_1w",
        "temp_max_1w",
        "temp_min_1w",
        "umidade_media_1w",
        "dias_secos_1w",
        "dias_calor_extremo_1w",
    ]

    for col in weighted_cols:
        df[f"{col}_pond"] = df[col] * df["peso_norm"]

    df_index = (
        df
        .groupby("semana", as_index=False)
        .agg(
            chuva_soja_1w=("chuva_mm_1w_pond", "sum"),
            temp_media_soja_1w=("temp_media_1w_pond", "sum"),
            temp_max_soja_1w=("temp_max_1w_pond", "sum"),
            temp_min_soja_1w=("temp_min_1w_pond", "sum"),
            umidade_soja_1w=("umidade_media_1w_pond", "sum"),
            dias_secos_soja_1w=("dias_secos_1w_pond", "sum"),
            dias_calor_extremo_soja_1w=("dias_calor_extremo_1w_pond", "sum"),
            n_pontos=("local", "nunique"),
        )
    )

    return df_index


def add_climate_stress(df_index: pd.DataFrame) -> pd.DataFrame:
    df = df_index.copy()

    # Percentis simples dentro da própria série histórica
    df["stress_chuva"] = 1 - df["chuva_soja_1w"].rank(pct=True)
    df["stress_calor"] = df["temp_max_soja_1w"].rank(pct=True)
    df["stress_dias_secos"] = df["dias_secos_soja_1w"].rank(pct=True)

    df["stress_climatico_soja"] = (
        0.45 * df["stress_chuva"]
        + 0.35 * df["stress_dias_secos"]
        + 0.20 * df["stress_calor"]
    )

    return df


def validate(df_daily: pd.DataFrame, df_weekly_index: pd.DataFrame):
    print("\nValidação:")
    print("Daily shape:", df_daily.shape)
    print("Daily período:", df_daily["data"].min(), "até", df_daily["data"].max())
    print("Locais:", sorted(df_daily["local"].unique()))

    print("Weekly index shape:", df_weekly_index.shape)
    print("Weekly período:", df_weekly_index["semana"].min(), "até", df_weekly_index["semana"].max())

    print("\nNulos daily:")
    print(df_daily.isna().mean().sort_values(ascending=False))

    print("\nNulos weekly index:")
    print(df_weekly_index.isna().mean().sort_values(ascending=False))


def run():
    df_daily = collect_daily()

    df_weekly_location = build_weekly_by_location(df_daily)

    df_weekly_index = build_weekly_soy_index(df_weekly_location)
    df_weekly_index = add_climate_stress(df_weekly_index)

    daily_path = PROCESSED_DIR / "fact_weather_daily_nasa_power.parquet"
    weekly_location_path = PROCESSED_DIR / "fact_weather_weekly_location_nasa_power.parquet"
    weekly_index_path = PROCESSED_DIR / "fact_weather_soy_index_weekly.parquet"

    df_daily.to_parquet(daily_path, index=False)
    df_weekly_location.to_parquet(weekly_location_path, index=False)
    df_weekly_index.to_parquet(weekly_index_path, index=False)

    validate(df_daily, df_weekly_index)

    print("\nArquivos gerados:")
    print(daily_path)
    print(weekly_location_path)
    print(weekly_index_path)


if __name__ == "__main__":
    run()