"""Генерация синтетических исторических данных для СППР.

Создаёт:
  data/price_history.csv - помесячная стоимость 7-дневной поездки из Москвы (перелёт + 7 дней
                           проживания/питания/транспорта), руб./чел., 2019-01 … 2026-09;
  data/users.csv         - профили 600 пользователей (вкусовые предпочтения, чувствительность к цене);
  data/ratings.csv       - оценки направлений пользователями (1–5).

Данные синтетические, но построены по реалистичной схеме: тренд внутренней инфляции, курс валюты
для зарубежных направлений, сезонность конкретного направления, ковидный провал 2020 г.
(для зарубежья - пропуски, границы закрыты), ценовой шок марта 2022 г. и AR(1)-шум.
Запуск:  python scripts/generate_data.py   (seed фиксирован - результат воспроизводим)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = np.random.default_rng(42)

FEATURES = ["culture", "nature", "beach", "gastronomy", "nightlife", "shopping", "family"]

# Сезонные коэффициенты (янв…дек) - пиковые и низкие сезоны направлений
SEASON = {
    "sochi":       [0.85, 0.90, 0.80, 0.80, 0.95, 1.15, 1.40, 1.45, 1.15, 0.90, 0.75, 0.95],
    "spb":         [0.85, 0.82, 0.85, 0.90, 1.05, 1.30, 1.30, 1.20, 1.00, 0.90, 0.85, 1.05],
    "kazan":       [1.05, 0.90, 0.90, 0.95, 1.05, 1.10, 1.15, 1.10, 0.95, 0.90, 0.85, 1.05],
    "kaliningrad": [0.85, 0.85, 0.85, 0.90, 1.00, 1.20, 1.35, 1.35, 1.05, 0.90, 0.85, 0.95],
    "altai":       [0.80, 0.80, 0.80, 0.85, 0.95, 1.25, 1.45, 1.40, 1.10, 0.85, 0.75, 0.85],
    "istanbul":    [0.85, 0.85, 0.95, 1.10, 1.15, 1.10, 1.10, 1.10, 1.10, 1.10, 0.90, 0.95],
    "antalya":     [0.70, 0.70, 0.80, 0.95, 1.10, 1.30, 1.45, 1.45, 1.25, 1.05, 0.75, 0.70],
    "dubai":       [1.35, 1.20, 1.15, 1.00, 0.85, 0.75, 0.70, 0.70, 0.85, 1.10, 1.25, 1.40],
    "tbilisi":     [0.85, 0.85, 0.90, 1.00, 1.10, 1.10, 1.15, 1.15, 1.10, 1.05, 0.90, 0.95],
    "rome":        [0.80, 0.80, 0.95, 1.15, 1.20, 1.20, 1.10, 1.10, 1.15, 1.10, 0.85, 0.95],
}

# Среднемесячный рост цен внутреннего туризма по годам
DOM_GROWTH = {2019: .004, 2020: .003, 2021: .012, 2022: .008, 2023: .010, 2024: .010, 2025: .006, 2026: .005}
# Опорные точки условного курса (руб. за ед. валюты) для зарубежных направлений
FX_POINTS = {
    "2019-01": 66, "2019-12": 63, "2020-03": 73, "2020-12": 74, "2021-12": 73, "2022-02": 77,
    "2022-03": 105, "2022-05": 65, "2022-07": 58, "2022-12": 68, "2023-06": 82, "2023-10": 97,
    "2024-01": 89, "2024-06": 87, "2024-11": 102, "2025-02": 90, "2025-05": 80, "2025-09": 83,
    "2026-03": 80, "2026-09": 84,
}
QUALITY_BIAS = {"sochi": -0.10, "spb": 0.20, "kazan": 0.10, "kaliningrad": 0.0, "altai": 0.10,
                "istanbul": 0.20, "antalya": 0.0, "dubai": 0.0, "tbilisi": 0.25, "rome": 0.30}
POPULARITY = {"sochi": 1.6, "spb": 1.6, "kazan": 1.0, "kaliningrad": 0.8, "altai": 0.7,
              "istanbul": 1.3, "antalya": 1.5, "dubai": 1.1, "tbilisi": 0.9, "rome": 0.6}
ARCHETYPES = {
    "культурный": [0.95, 0.30, 0.10, 0.70, 0.40, 0.40, 0.30],
    "пляжный":    [0.20, 0.40, 0.95, 0.50, 0.50, 0.40, 0.60],
    "природа":    [0.25, 0.95, 0.30, 0.40, 0.10, 0.10, 0.50],
    "гурман":     [0.60, 0.30, 0.20, 0.95, 0.60, 0.40, 0.30],
    "тусовщик":   [0.40, 0.20, 0.60, 0.60, 0.95, 0.70, 0.10],
    "шопинг":     [0.40, 0.10, 0.50, 0.60, 0.60, 0.95, 0.40],
    "семья":      [0.40, 0.50, 0.80, 0.40, 0.10, 0.30, 0.95],
}


def _fx_series(months: pd.DatetimeIndex) -> np.ndarray:
    pts = pd.Series({pd.Timestamp(k + "-01"): v for k, v in FX_POINTS.items()})
    full = pts.reindex(pts.index.union(months)).interpolate(method="time").ffill().bfill()
    return full.reindex(months).to_numpy()


def make_price_history(dest: pd.DataFrame) -> pd.DataFrame:
    months = pd.date_range("2019-01-01", "2026-09-01", freq="MS")
    n = len(months)
    dom = np.ones(n)
    for k in range(1, n):
        dom[k] = dom[k - 1] * (1 + DOM_GROWTH[months[k].year])
    world = (1.003) ** np.arange(n)
    fx = _fx_series(months)
    t0 = list(months).index(pd.Timestamp("2022-03-01"))
    shock = np.where(np.arange(n) >= t0, 1 + 0.25 * np.exp(-(np.arange(n) - t0) / 14), 1.0)

    rows = []
    for _, d in dest.iterrows():
        season = np.array(SEASON[d.id])
        season = season / season.mean()
        s = season[months.month - 1]
        if d.is_domestic:
            level = dom.copy()
        else:
            level = 0.75 * (fx / fx[-12:].mean()) * world * shock + 0.25 * dom
        eps = np.zeros(n)
        for k in range(1, n):
            eps[k] = 0.5 * eps[k - 1] + RNG.normal(0, 0.035)
        price = level * s * np.exp(eps)
        covid = (months >= "2020-04-01") & (months <= "2020-12-01")
        if d.is_domestic:
            price = np.where((months >= "2020-04-01") & (months <= "2020-06-01"), price * 0.55, price)
        else:
            price = np.where(covid, np.nan, price)
        base = d.flight_rub + 7 * d.daily_rub
        last12 = price[-12:]
        price = price / np.nanmean(last12) * base
        for m, p in zip(months, price):
            rows.append({"dest_id": d.id, "month": m.strftime("%Y-%m-%d"),
                         "cost_7d_rub": None if np.isnan(p) else round(float(p), -1)})
    return pd.DataFrame(rows)


def make_users_ratings(dest: pd.DataFrame, n_users: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = dest[FEATURES].to_numpy(dtype=float)
    base_cost = (dest.flight_rub + 7 * dest.daily_rub).to_numpy(dtype=float)
    cost_level = (base_cost - base_cost.min()) / (base_cost.max() - base_cost.min())
    names = list(ARCHETYPES)
    users, ratings = [], []
    pop = np.array([POPULARITY[i] for i in dest.id])
    for u in range(n_users):
        arch = names[RNG.integers(len(names))]
        p = np.clip(np.array(ARCHETYPES[arch]) + RNG.normal(0, 0.12, len(FEATURES)), 0, 1)
        budget_sens = float(RNG.uniform(0, 1))
        bias_u = float(RNG.normal(0, 0.3))
        users.append({"user_id": f"u{u:04d}", "archetype": arch, "budget_sensitivity": round(budget_sens, 3),
                      **{f: round(float(v), 3) for f, v in zip(FEATURES, p)}})
        k = int(RNG.integers(3, 8))
        chosen = RNG.choice(len(dest), size=k, replace=False, p=pop / pop.sum())
        for i in chosen:
            f = feats[i]
            cos = float(p @ f / (np.linalg.norm(p) * np.linalg.norm(f) + 1e-9))
            z = (8.0 * (cos - 0.72) + bias_u + QUALITY_BIAS[dest.id.iloc[i]]
                 - 1.2 * budget_sens * (cost_level[i] - 0.5) + RNG.normal(0, 0.35))
            r = int(np.clip(np.round(1 + 4 / (1 + np.exp(-z))), 1, 5))
            ratings.append({"user_id": f"u{u:04d}", "dest_id": dest.id.iloc[i], "rating": r})
    return pd.DataFrame(users), pd.DataFrame(ratings)


def main() -> None:
    dest = pd.read_csv(DATA / "destinations.csv")
    ph = make_price_history(dest)
    ph.to_csv(DATA / "price_history.csv", index=False)
    users, ratings = make_users_ratings(dest)
    users.to_csv(DATA / "users.csv", index=False)
    ratings.to_csv(DATA / "ratings.csv", index=False)
    print(f"price_history: {len(ph)} строк, пропусков {ph.cost_7d_rub.isna().sum()}")
    print(f"users: {len(users)}, ratings: {len(ratings)}, средняя оценка {ratings.rating.mean():.2f}")
    print(ratings.rating.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
