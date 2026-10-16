"""П4. Оптимизационная модель маршрута (смешанное целочисленное ЛП, PuLP).

Задача: для выбранного направления составить маршрут по дням (какие объекты посетить, в какой
день и в каком порядке) и выбрать класс проживания так, чтобы максимизировать суммарную
полезность поездки при ограничениях по времени каждого дня и общему бюджету.
Модель - вариант задачи маршрутизации с призами (team orienteering problem) с бюджетом:
  y[i,t] ∈ {0,1}  - объект i посещается в день t;
  x[i,j,t] ∈ {0,1} - в день t переезд i → j (узел 0 - отель);
  u[i,t] ≥ 0      - порядковый номер объекта в дне (устранение подциклов, MTZ);
  z[k] ∈ {0,1}    - выбран класс проживания k; w[t] ∈ {0,1} - в день t есть экскурсии.
Решатель: HiGHS (по умолчанию в PuLP ≥ 4) или CBC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from .costs import HOTEL_CLASSES
from .data import load_destinations

WALK_KM = 1.2            # пешком, если ближе
WALK_KMH = 4.5
ROAD_FACTOR = 1.3        # извилистость дорог относительно прямой
TRANSFER_OVERHEAD_H = 0.15
TAXI_BASE_RUB = 150
EVENING_START = 18.5     # ужин - не раньше 18:30


def haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def travel_matrices(dest_id: str, pois: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    d = load_destinations().set_index("id").loc[dest_id]
    pts = [(d.lat, d.lon)] + list(zip(pois.lat, pois.lon))
    n = len(pts)
    tt, tc = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = haversine(*pts[i], *pts[j])
            if km <= WALK_KM:
                tt[i, j] = km / WALK_KMH
            else:
                road = km * ROAD_FACTOR
                tt[i, j] = road / d.speed_kmh + TRANSFER_OVERHEAD_H
                tc[i, j] = TAXI_BASE_RUB + road * d.taxi_rub_km
    return tt, tc, pts


def day_hours(days: int, pace: float = 8.0) -> list[float]:
    """Доступное время: день прилёта - 5/8 темпа, полные дни - темп ЛПР, день вылета - 3/8 темпа."""
    first, last = round(pace * 5 / 8, 1), round(pace * 3 / 8, 1)
    if days == 1:
        return [first]
    return [first] + [float(pace)] * (days - 2) + [last]


@dataclass
class RouteResult:
    status: str
    objective: float
    hotel_class: str
    plan: pd.DataFrame          # день, порядок, объект, прибытие, длительность, стоимость
    totals: dict
    solve_time: float


def optimize_route(dest_id: str, pois: pd.DataFrame, days: int, budget: float, fixed_cost: float,
                   daily_cost: float, cost_weight: float = 0.0, time_weight: float = 0.5,
                   max_museums_per_day: int = 2, must_visit=(), hotel_classes=None,
                   pace: float = 8.0, time_limit: int = 60) -> RouteResult:
    """pois - объекты с колонками poi_id, name, category, duration_h, cost_rub, utility.
    fixed_cost - перелёт + виза; daily_cost - суточные расходы при классе «стандарт».
    cost_weight - штраф целевой функции за 1000 руб. (0 - безразличие к затратам в пределах бюджета)."""
    hotel_classes = hotel_classes or HOTEL_CLASSES
    P = list(range(1, len(pois) + 1))
    N = [0] + P
    T = list(range(days))
    H = day_hours(days, pace)
    tt, tc, _ = travel_matrices(dest_id, pois)
    util = dict(zip(P, pois.utility))
    dur = dict(zip(P, pois.duration_h))
    cost = dict(zip(P, pois.cost_rub))
    cat = dict(zip(P, pois.category))
    pid = dict(zip(P, pois.poi_id))
    evening = {i for i, nm in zip(P, pois["name"]) if "ужин" in nm.lower()}
    n = len(P)

    m = pulp.LpProblem("route", pulp.LpMaximize)
    y = _vdict(m, "y", (P, T), cat="Binary")
    x = _vdict(m, "x", (N, N, T), cat="Binary")
    u = _vdict(m, "u", (P, T), lowBound=0, upBound=n)
    w = _vdict(m, "w", T, cat="Binary")
    z = _vdict(m, "z", list(hotel_classes), cat="Binary")

    travel_h = pulp.lpSum(tt[i, j] * x[i][j][t] for i in N for j in N if i != j for t in T)
    taxi = pulp.lpSum(tc[i, j] * x[i][j][t] for i in N for j in N if i != j for t in T)
    tickets = pulp.lpSum(cost[i] * y[i][t] for i in P for t in T)
    stay = pulp.lpSum(days * daily_cost * hotel_classes[k]["mult"] * z[k] for k in hotel_classes)
    total_cost = fixed_cost + stay + tickets + taxi
    m += (pulp.lpSum(util[i] * y[i][t] for i in P for t in T)
          + pulp.lpSum(hotel_classes[k]["comfort"] * z[k] for k in hotel_classes)
          - time_weight * travel_h - cost_weight * (stay + tickets + taxi) / 1000.0)

    m += pulp.lpSum(z[k] for k in hotel_classes) == 1
    m += total_cost <= budget, "budget"
    for i in P:
        m += pulp.lpSum(y[i][t] for t in T) <= 1
        if pid[i] in must_visit:
            m += pulp.lpSum(y[i][t] for t in T) == 1
    for t in T:
        for i in N:
            m += x[i][i][t] == 0
        for i in P:
            m += pulp.lpSum(x[i][j][t] for j in N if j != i) == y[i][t]
            m += pulp.lpSum(x[j][i][t] for j in N if j != i) == y[i][t]
            m += y[i][t] <= w[t]
        m += pulp.lpSum(x[0][j][t] for j in P) == w[t]
        m += pulp.lpSum(x[j][0][t] for j in P) == w[t]
        # ужин - вечерний слот: его длительность не входит в дневной темп, переезды входят
        m += (pulp.lpSum(dur[i] * y[i][t] for i in P if i not in evening)
              + pulp.lpSum(tt[i, j] * x[i][j][t] for i in N for j in N if i != j)) <= H[t], f"time_{t}"
        m += pulp.lpSum(y[i][t] for i in evening) <= 1
        m += pulp.lpSum(y[i][t] for i in P if cat[i] == "museum") <= max_museums_per_day
        for i in evening:                       # ужин - последний пункт дня, после него - в отель
            m += x[i][0][t] == y[i][t]
        for i in P:
            for j in P:
                if i != j:
                    m += u[i][t] - u[j][t] + n * x[i][j][t] <= n - 1
            m += u[i][t] >= y[i][t]

    # отсечение симметрии: полные дни (2…D-1) взаимозаменяемы - упорядочиваем по загрузке
    for t in T[1:-2]:
        m += (pulp.lpSum(dur[i] * y[i][t] for i in P) >= pulp.lpSum(dur[i] * y[i][t + 1] for i in P))
    # переезды, которые не помещаются в день, запрещены
    for t in T:
        for i in N:
            for j in N:
                if i != j and tt[i, j] + (0 if i in evening else dur.get(i, 0)) \
                        + (0 if j in evening else dur.get(j, 0)) > H[t]:
                    m += x[i][j][t] == 0

    status, solve_time = _solve(m, time_limit)
    if status not in ("Optimal", "TimeLimit", "GapLimit") or pulp.value(m.objective) is None:
        return RouteResult(status, float("nan"), "", pd.DataFrame(), {}, solve_time)

    hotel = max(hotel_classes, key=lambda k: z[k].value() or 0)
    rows = []
    for t in T:
        if (w[t].value() or 0) < 0.5:
            continue
        node, order, clock = 0, 0, 14.0 if t == 0 else 9.0
        visited = set()
        while True:
            nxt = [j for j in N if j != node and (x[node][j][t].value() or 0) > 0.5]
            if not nxt or nxt[0] == 0 or nxt[0] in visited:
                break
            j = nxt[0]
            clock += tt[node, j]
            if j in evening:
                clock = max(clock, EVENING_START)
            order += 1
            r = pois.iloc[j - 1]
            rows.append({"day": t + 1, "order": order, "poi_id": r.poi_id, "name": r["name"],
                         "category": r.category, "arrival": _hhmm(clock), "duration_h": r.duration_h,
                         "travel_h": round(tt[node, j], 2), "ticket_rub": r.cost_rub,
                         "transfer_rub": round(tc[node, j], -1), "utility": r.utility,
                         "lat": r.lat, "lon": r.lon})
            clock += r.duration_h
            visited.add(j)
            node = j
        if rows and rows[-1]["day"] == t + 1:
            rows[-1]["return_h"] = round(tt[node, 0], 2)
            rows[-1]["return_rub"] = round(tc[node, 0], -1)
    plan = pd.DataFrame(rows)
    totals = {
        "flight_visa": round(fixed_cost, -1),
        "stay": round(pulp.value(stay), -1),
        "tickets": round(pulp.value(tickets), -1),
        "transfers": round(pulp.value(taxi), -1),
        "total": round(pulp.value(total_cost), -1),
        "budget_slack": round(budget - pulp.value(total_cost), -1),
        "utility": round(sum(util[i] * (y[i][t].value() or 0) for i in P for t in T), 2),
        "travel_h": round(pulp.value(travel_h), 2),
        "n_pois": int(round(sum((y[i][t].value() or 0) for i in P for t in T))),
        "hours_used": [round(sum(dur[i] * (y[i][t].value() or 0) for i in P if i not in evening)
                             + sum(tt[i, j] * (x[i][j][t].value() or 0) for i in N for j in N if i != j), 2)
                       for t in T],
        "hours_available": H,
    }
    return RouteResult(status, float(pulp.value(m.objective)), hotel, plan, totals, solve_time)


def greedy_route(dest_id: str, pois: pd.DataFrame, days: int, pace: float = 8.0) -> dict:
    """Базовая эвристика для сравнения: каждый день добавлять ближайший по «полезность/время» объект."""
    tt, tc, _ = travel_matrices(dest_id, pois)
    H = day_hours(days, pace)
    left = set(range(1, len(pois) + 1))
    util = dict(zip(range(1, len(pois) + 1), pois.utility))
    dur = dict(zip(range(1, len(pois) + 1), pois.duration_h))
    total_u, total_t, n = 0.0, 0.0, 0
    for t in range(days):
        node, used = 0, 0.0
        while True:
            cand = [(util[j] / (dur[j] + tt[node, j]), j) for j in left
                    if used + tt[node, j] + dur[j] + tt[j, 0] <= H[t]]
            if not cand:
                break
            _, j = max(cand)
            used += tt[node, j] + dur[j]
            total_t += tt[node, j]
            total_u += util[j]
            left.discard(j)
            node, n = j, n + 1
        total_t += tt[node, 0] if node else 0
    return {"utility": round(total_u, 2), "travel_h": round(total_t, 2), "n_pois": n}


def _vdict(m, name, idx, **kw):
    """Совместимость PuLP 3.x (LpVariable.dicts) и 4.x (LpProblem.add_variable_dicts)."""
    if hasattr(m, "add_variable_dicts"):
        return m.add_variable_dicts(name, idx, **kw)
    return pulp.LpVariable.dicts(name, idx, **kw)


def _solve(m, time_limit: int, gap: float = 0.005) -> tuple[str, float]:
    """Решение с HiGHS (если есть) или CBC; совместимо с PuLP 3.x и 4.x. Возвращает (статус, время, с)."""
    import time
    if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=gap)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap)
    t0 = time.time()
    res = m.solve(solver)
    elapsed = time.time() - t0
    if hasattr(res, "status") and hasattr(res.status, "name"):          # PuLP 4.x
        return res.status.name, elapsed
    return {1: "Optimal", 0: "NotSolved", -1: "Infeasible", -2: "Unbounded"}.get(int(res), str(res)), elapsed


def _hhmm(h: float) -> str:
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh:02d}:{mm:02d}"
