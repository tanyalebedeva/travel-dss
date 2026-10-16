"""Загрузка данных и общие структуры СППР выбора места для путешествия."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
KB_DIR = ROOT / "kb"

FEATURES = ["culture", "nature", "beach", "gastronomy", "nightlife", "shopping", "family"]
FEATURES_RU = {
    "culture": "культура и история", "nature": "природа", "beach": "пляжный отдых",
    "gastronomy": "гастрономия", "nightlife": "ночная жизнь", "shopping": "шопинг",
    "family": "отдых с детьми",
}
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
             "сентябрь", "октябрь", "ноябрь", "декабрь"]
MONTHS_RU_GEN = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе",
                 "сентябре", "октябре", "ноябре", "декабре"]

# Дата «сейчас» для расчётов (последний месяц истории - сентябрь 2026; планирование - октябрь 2026)
PLANNING_DATE = date(2026, 10, 12)


@dataclass
class TripRequest:
    """Параметры поездки, которые задаёт ЛПР."""
    year: int = 2027
    month: int = 5
    days: int = 5
    budget_per_person: float = 70_000.0
    pace_hours: float = 8.0          # темп: часов активностей в полный день (6 - спокойный, 10 - активный)
    travelers: int = 1
    with_children: bool = False
    origin: str = "Москва"
    start_day: int = 1
    # Предпочтения 0…1 по признакам FEATURES + чувствительность к цене
    profile: dict = field(default_factory=lambda: {
        "culture": 0.9, "nature": 0.5, "beach": 0.2, "gastronomy": 0.8,
        "nightlife": 0.3, "shopping": 0.2, "family": 0.1,
    })
    budget_sensitivity: float = 0.6
    must_visit: list = field(default_factory=list)

    @property
    def start_date(self) -> date:
        return date(self.year, self.month, self.start_day)

    @property
    def days_until_departure(self) -> int:
        return (self.start_date - PLANNING_DATE).days

    @property
    def months_ahead(self) -> int:
        return (self.year - PLANNING_DATE.year) * 12 + self.month - PLANNING_DATE.month

    def label(self) -> str:
        budget = f"{self.budget_per_person:,.0f}".replace(",", " ")
        return f"{MONTHS_RU[self.month - 1].capitalize()} {self.year}, {self.days} дн., бюджет {budget} ₽/чел."


@lru_cache(maxsize=None)
def load_destinations() -> pd.DataFrame:
    return pd.read_csv(DATA / "destinations.csv")


@lru_cache(maxsize=None)
def load_climate() -> pd.DataFrame:
    return pd.read_csv(DATA / "climate.csv")


@lru_cache(maxsize=None)
def load_pois() -> pd.DataFrame:
    return pd.read_csv(DATA / "pois.csv")


@lru_cache(maxsize=None)
def load_price_history() -> pd.DataFrame:
    df = pd.read_csv(DATA / "price_history.csv", parse_dates=["month"])
    return df


@lru_cache(maxsize=None)
def load_users() -> pd.DataFrame:
    return pd.read_csv(DATA / "users.csv")


@lru_cache(maxsize=None)
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA / "ratings.csv")


def climate_for(dest_id: str, month: int) -> tuple[float, float]:
    c = load_climate()
    row = c[(c.dest_id == dest_id) & (c.month == month)].iloc[0]
    return float(row.temp_c), float(row.rain_days)


def dest_name(dest_id: str) -> str:
    d = load_destinations()
    return str(d.loc[d.id == dest_id, "name"].iloc[0])


def ensure_dirs() -> None:
    REPORTS.mkdir(exist_ok=True)
    KB_DIR.mkdir(exist_ok=True)
