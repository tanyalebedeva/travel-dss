"""Выгрузка витрин данных для BI-дашборда (Apache Superset) в SQLite: superset/data/tdss.db."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .data import MONTHS_RU, REPORTS, ROOT, load_destinations, load_price_history

DB_PATH = ROOT / "superset" / "data" / "tdss.db"


def build(ranking: pd.DataFrame | None = None, plans: pd.DataFrame | None = None) -> dict:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    dest = load_destinations()
    dim = dest[["id", "name", "country", "is_domestic", "lat", "lon"]].rename(columns={"id": "dest_id"})
    dim["region"] = np.where(dim.is_domestic == 1, "Россия", "Зарубежье")
    names = dict(zip(dim.dest_id, dim.name))
    region = dict(zip(dim.dest_id, dim.region))

    ph = load_price_history().dropna().copy()
    hist = pd.DataFrame({"dest_id": ph.dest_id, "month": ph.month.dt.strftime("%Y-%m-%d"),
                         "value": ph.cost_7d_rub, "series": "история", "lo80": np.nan, "hi80": np.nan})
    fc = pd.read_csv(REPORTS / "forecast.csv")
    fct = pd.DataFrame({"dest_id": fc.dest_id, "month": fc.month.str[:10], "value": fc.forecast,
                        "series": "прогноз", "lo80": fc.lo80, "hi80": fc.hi80})
    series = pd.concat([hist, fct], ignore_index=True)
    series["name"] = series.dest_id.map(names)
    series["region"] = series.dest_id.map(region)

    met = pd.read_csv(REPORTS / "p6_metrics_by_dest.csv").rename(columns={"Покрытие 80% ПИ": "coverage80"})
    met["name"] = met.dest_id.map(names)

    h = ph[ph.month >= "2023-01-01"].copy()
    h["year_mean"] = h.groupby(["dest_id", h.month.dt.year]).cost_7d_rub.transform("mean")
    h["idx"] = h.cost_7d_rub / h.year_mean
    season = h.groupby(["dest_id", h.month.dt.month]).idx.mean().reset_index().rename(columns={"month": "month_num", "idx": "season_index"})
    season["month_name"] = season.month_num.map(lambda m: f"{m:02d} {MONTHS_RU[m - 1]}")
    season["name"] = season.dest_id.map(names)
    season["season_index"] = season.season_index.round(3)

    tables = {"dim_destination": dim, "price_series": series, "model_metrics": met, "season_index": season}
    if ranking is not None:
        tables["dss_ranking"] = ranking
    if plans is not None:
        tables["route_plan"] = plans
    with sqlite3.connect(DB_PATH) as con:
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="replace", index=False)
    return {k: len(v) for k, v in tables.items()}
