"""П2. Многокритериальный анализ методом анализа иерархий (АНР, Т. Саати).

Иерархия: цель «выбрать направление поездки» → 6 критериев → 10 альтернатив-направлений.
Веса критериев - по матрице парных сравнений ЛПР (собственный вектор), согласованность - ИС/ОС.
Локальные приоритеты альтернатив - по матрицам парных сравнений, построенным из количественных
показателей (разность показателей переводится в шкалу Саати 1…9).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import trip_cost
from .data import TripRequest, climate_for, load_destinations
from .knowledge import ontology_facts

# Случайный индекс согласованности (Саати) для n = 1…12
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
      11: 1.51, 12: 1.48}

CRITERIA = [
    ("K1", "Стоимость поездки", "min"),
    ("K2", "Климат в месяц поездки", "max"),
    ("K3", "Культура и достопримечательности", "max"),
    ("K4", "Природа и пляжный отдых", "max"),
    ("K5", "Транспортная и организационная доступность", "max"),
    ("K6", "Безопасность", "max"),
]
CRIT_IDS = [c[0] for c in CRITERIA]
CRIT_NAMES = {c[0]: c[1] for c in CRITERIA}

# Матрица парных сравнений критериев, заполненная ЛПР (базовый сценарий: ограниченный бюджет,
# интерес к культуре и гастрономии, пляж второстепенен)
DEFAULT_CRITERIA_MATRIX = np.array([
    #  K1    K2    K3    K4    K5    K6
    [1,    3,    2,    5,    3,    2],     # K1 Стоимость
    [1/3,  1,    1/3,  2,    1,    1/2],   # K2 Климат
    [1/2,  3,    1,    3,    2,    2],     # K3 Культура
    [1/5,  1/2,  1/3,  1,    1/2,  1/3],   # K4 Природа и пляж
    [1/3,  1,    1/2,  2,    1,    1/2],   # K5 Доступность
    [1/2,  2,    1/2,  3,    2,    1],     # K6 Безопасность
], dtype=float)

SAATY_VALUES = np.array([1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9])


@dataclass
class Priority:
    weights: np.ndarray
    lambda_max: float
    ci: float
    cr: float
    gm_weights: np.ndarray       # контроль: метод геометрического среднего

    @property
    def consistent(self) -> bool:
        return self.cr < 0.10


def priority_vector(a: np.ndarray) -> Priority:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    vals, vecs = np.linalg.eig(a)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    cr = ci / RI[n] if RI[n] > 0 else 0.0
    gm = np.prod(a, axis=1) ** (1 / n)
    return Priority(w, lam, ci, cr, gm / gm.sum())


def snap_saaty(x: float) -> float:
    return float(SAATY_VALUES[np.argmin(np.abs(np.log(SAATY_VALUES) - np.log(x)))])


def matrix_from_importance(importance: dict[str, float]) -> np.ndarray:
    """Быстрый режим: ЛПР оценивает важность каждого критерия по шкале 1…9,
    отношения важностей округляются до ближайшего значения шкалы Саати."""
    v = np.array([max(importance[c], 1) for c in CRIT_IDS], dtype=float)
    n = len(v)
    a = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = snap_saaty(v[i] / v[j])
            a[j, i] = 1 / a[i, j]
    return a


def matrix_from_values(values: np.ndarray, sense: str = "max") -> np.ndarray:
    """Матрица парных сравнений альтернатив по количественному показателю:
    a_ij = 1 + 8·|Δ|/размах (округление до целого шкалы Саати), обратная - для отрицательной разности."""
    v = np.asarray(values, dtype=float)
    if sense == "min":
        v = -v
    rng = v.max() - v.min()
    n = len(v)
    a = np.ones((n, n))
    if rng == 0:
        return a
    for i in range(n):
        for j in range(n):
            d = (v[i] - v[j]) / rng
            s = float(np.round(1 + 8 * abs(d)))
            a[i, j] = s if d >= 0 else 1 / s
    return a


# ---------------- показатели альтернатив ----------------

def climate_comfort(temp: float, rain_days: float) -> float:
    t_score = np.exp(-((temp - 22) / 8) ** 2)
    r_score = np.clip(1 - rain_days / 25, 0, 1)
    return float(0.65 * t_score + 0.35 * r_score)


def indicators(trip: TripRequest) -> pd.DataFrame:
    dest = load_destinations()
    onto = ontology_facts().set_index("dest_id")
    rows = []
    for _, d in dest.iterrows():
        t, rain = climate_for(d.id, trip.month)
        cost = trip_cost(d.id, trip)
        beach_ok = trip.month in onto.loc[d.id, "beach_season_months"]
        nature_beach = max(d.nature, d.beach * (1.0 if beach_ok else 0.2))
        needs_cash = bool(onto.loc[d.id, "needs_cash"])
        access = (1 - 0.07 * d.flight_hours - 0.15 * (1 - d.direct_flight)
                  - 0.15 * d.visa_required - 0.10 * needs_cash)
        rows.append({
            "dest_id": d.id, "name": d["name"],
            "K1": cost["total"], "K2": round(climate_comfort(t, rain), 3), "K3": d.culture,
            "K4": nature_beach, "K5": round(max(access, 0.0), 3), "K6": d.safety_index,
            "temp": t, "rain_days": rain,
        })
    return pd.DataFrame(rows)


@dataclass
class AHPResult:
    criteria: Priority
    criteria_matrix: np.ndarray
    alt_matrices: dict           # K -> матрица
    local: pd.DataFrame          # альтернативы × критерии (локальные приоритеты)
    local_cr: dict               # K -> ОС
    scores: pd.DataFrame         # итог: dest_id, name, score, rank
    indicators: pd.DataFrame


def run(trip: TripRequest, criteria_matrix: np.ndarray | None = None) -> AHPResult:
    cm = DEFAULT_CRITERIA_MATRIX if criteria_matrix is None else np.asarray(criteria_matrix, float)
    cp = priority_vector(cm)
    ind = indicators(trip)
    local, mats, crs = {}, {}, {}
    for cid, _, sense in CRITERIA:
        a = matrix_from_values(ind[cid].to_numpy(), sense)
        p = priority_vector(a)
        local[cid], mats[cid], crs[cid] = p.weights, a, p.cr
    local_df = pd.DataFrame(local, index=ind.dest_id)
    score = local_df.to_numpy() @ cp.weights
    scores = pd.DataFrame({"dest_id": ind.dest_id, "name": ind.name, "score": score})
    scores["rank"] = scores.score.rank(ascending=False, method="min").astype(int)
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    return AHPResult(cp, cm, mats, local_df, crs, scores, ind)


def sensitivity(res: AHPResult, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Изменение итоговых оценок при варьировании веса одного критерия
    (остальные веса пересчитываются пропорционально)."""
    grid = np.linspace(0.0, 0.8, 41) if grid is None else grid
    w0 = res.criteria.weights
    rows = []
    for k, cid in enumerate(CRIT_IDS):
        for wk in grid:
            w = w0 * (1 - wk) / (1 - w0[k])
            w[k] = wk
            s = res.local.to_numpy() @ w
            leader = res.local.index[int(np.argmax(s))]
            for dest_id, val in zip(res.local.index, s):
                rows.append({"criterion": cid, "weight": round(float(wk), 3), "dest_id": dest_id,
                             "score": float(val), "leader": leader})
    return pd.DataFrame(rows)


def leader_switches(sens: pd.DataFrame, res: "AHPResult | None" = None) -> pd.DataFrame:
    """Точки смены лидера рейтинга при изменении веса каждого критерия
    (уточняются методом Брента, если передан результат АНР)."""
    from scipy.optimize import brentq
    out = []
    for cid, g in sens.groupby("criterion"):
        lead = g.drop_duplicates("weight")[["weight", "leader"]].reset_index(drop=True)
        for i in range(1, len(lead)):
            if lead.leader[i] != lead.leader[i - 1]:
                w_exact = lead.weight[i]
                if res is not None:
                    k = CRIT_IDS.index(cid)
                    w0, L = res.criteria.weights, res.local
                    a_, b_ = lead.leader[i - 1], lead.leader[i]

                    def diff(wk):
                        w = w0 * (1 - wk) / (1 - w0[k])
                        w[k] = wk
                        return float((L.loc[a_] - L.loc[b_]).to_numpy() @ w)
                    w_exact = brentq(diff, lead.weight[i - 1], lead.weight[i])
                out.append({"criterion": cid, "weight": round(float(w_exact), 4), "from": lead.leader[i - 1],
                            "to": lead.leader[i]})
    return pd.DataFrame(out)
