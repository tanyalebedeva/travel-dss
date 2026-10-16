"""П3. Принятие решения в условиях риска и неопределённости.

Неопределённые факторы: (1) цена поездки к моменту поздней покупки (из истории цен, П6),
(2) погода в месяц поездки (из климатических норм), (3) срыв поездки (отмена рейса, болезнь,
отказ в визе - вероятности из базы знаний, П5).
Решения ЛПР: направление (3 лучших по АНР) × стратегия бронирования:
  S1 - купить сейчас невозвратный тариф; S2 - купить сейчас возвратный (+12%);
  S3 - отложить покупку до месяца перед поездкой.
Выигрыш = субъективная ценность поездки − затраты (руб.).
Методы: дерево решений (networkx, свёртка «назад»), критерии Байеса, Лапласа, Вальда,
максимакса, Гурвица, Сэвиджа; пороговый анализ (scipy.optimize.brentq);
Монте-Карло по стоимости (scipy.stats).
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.tsa.seasonal import seasonal_decompose

from .data import climate_for, load_destinations, load_price_history

STRATEGIES = {
    "S1": "Купить сейчас (невозвратный)",
    "S2": "Купить сейчас (возвратный, +12%)",
    "S3": "Отложить покупку",
}
REFUND_PREMIUM = 0.12        # наценка возвратного тарифа
NONREF_LOSS = 0.85           # доля потерь невозвратного тарифа при срыве
REF_LOSS = 0.15              # доля потерь возвратного тарифа при срыве (сборы)
LATE_PREMIUM = 0.08          # средняя наценка при покупке за месяц до вылета (кривая бронирования)
LATE_DISRUPT_AFTER = 0.40    # доля срывов, которые происходят уже после поздней покупки
VALUE_FACTOR = 1.30          # ценность лучшего варианта для ЛПР = 1.3 × бюджет
PRICE_STATES = [("низкая", -1), ("ожидаемая", 0), ("высокая", 1)]
PRICE_PROBS = np.array([1 / 6, 2 / 3, 1 / 6])   # 3-точечная квадратура Гаусса–Эрмита


def price_volatility(dest_id: str, horizon: int) -> float:
    """σ логарифмического отклонения цены (без сезонности и тренда) на горизонте h мес."""
    ph = load_price_history()
    s = ph[(ph.dest_id == dest_id) & (ph.month >= "2021-01-01")].set_index("month").cost_7d_rub
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    dec = seasonal_decompose(np.log(s), model="additive", period=12)
    sa = (np.log(s) - dec.seasonal)
    d = (sa.shift(-horizon) - sa).dropna()
    return float(np.std(d - d.mean(), ddof=1))


def weather_risk(dest_id: str, month: int) -> float:
    t, rain = climate_for(dest_id, month)
    p = rain / 30 + (0.25 if t > 30 else 0) + (0.15 if t < 10 else 0)
    return float(np.clip(p, 0.05, 0.7))


@dataclass
class AltInput:
    dest_id: str
    name: str
    cost: float              # ожидаемая стоимость «сейчас», руб.
    value: float             # субъективная ценность поездки, руб.
    q: float                 # вероятность срыва
    p_bad: float             # вероятность неблагоприятной погоды
    ws: float                # чувствительность ценности к погоде
    sigma: float             # волатильность цены до поздней покупки


def build_inputs(alts: list[dict], budget: float, month: int, months_until_late: int) -> list[AltInput]:
    """alts: [{'dest_id','name','score','cost','disruption_risk'}], score - интегральная оценка."""
    dest = load_destinations().set_index("id")
    s_max = max(a["score"] for a in alts)
    out = []
    for a in alts:
        d = dest.loc[a["dest_id"]]
        out.append(AltInput(
            a["dest_id"], a["name"], float(a["cost"]),
            VALUE_FACTOR * budget * a["score"] / s_max,
            float(a["disruption_risk"]), weather_risk(a["dest_id"], month),
            0.15 + 0.30 * max(d.nature, d.beach), price_volatility(a["dest_id"], max(months_until_late, 1)),
        ))
    return out


def _cost_late(a: AltInput, k: int, premium: float = LATE_PREMIUM) -> float:
    return a.cost * np.exp(premium + k * np.sqrt(3) * a.sigma)


def payoff(a: AltInput, strategy: str, k: int, disrupted: bool, bad: bool, premium: float = LATE_PREMIUM) -> float:
    v = a.value * (1 - a.ws * bad)
    if strategy == "S1":
        return -NONREF_LOSS * a.cost if disrupted else v - a.cost
    if strategy == "S2":
        c = a.cost * (1 + REFUND_PREMIUM)
        return -REF_LOSS * c if disrupted else v - c
    c = _cost_late(a, k, premium)
    return -LATE_DISRUPT_AFTER * NONREF_LOSS * c if disrupted else v - c


STATE_LABELS = [f"цена {pl}; {'срыв' if dis else 'без срыва'}; {'плохая' if bad else 'хорошая'} погода"
                for pl, _ in PRICE_STATES for dis in (False, True) for bad in (False, True)]


def payoff_tables(inputs: list[AltInput]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Матрица выигрышей (решения × состояния природы) и матрица вероятностей состояний."""
    pay, prob = {}, {}
    for a in inputs:
        for s in STRATEGIES:
            row_p, row_q = [], []
            for (pl, k), pk in zip(PRICE_STATES, PRICE_PROBS):
                for dis in (False, True):
                    for bad in (False, True):
                        row_p.append(payoff(a, s, k, dis, bad))
                        row_q.append(pk * (a.q if dis else 1 - a.q) * (a.p_bad if bad else 1 - a.p_bad))
            key = f"{a.name} | {STRATEGIES[s]}"
            pay[key], prob[key] = row_p, row_q
    return (pd.DataFrame(pay, index=STATE_LABELS).T, pd.DataFrame(prob, index=STATE_LABELS).T)


def classic_criteria(pay: pd.DataFrame, prob: pd.DataFrame, alpha: float = 0.6) -> pd.DataFrame:
    regret = pay.max(axis=0) - pay
    crit = pd.DataFrame({
        "Байес (ожидаемый выигрыш)": (pay * prob).sum(axis=1),
        "Лаплас (равновероятные)": pay.mean(axis=1),
        "Вальд (максимин)": pay.min(axis=1),
        "Максимакс": pay.max(axis=1),
        f"Гурвиц (α={alpha})": alpha * pay.min(axis=1) + (1 - alpha) * pay.max(axis=1),
        "Сэвидж (мин. макс. сожаление)": -regret.max(axis=1),
    })
    return crit


def best_by_criteria(crit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in crit.columns:
        k = crit[c].idxmax()
        val = crit[c].max()
        rows.append({"Критерий": c, "Лучшее решение": k,
                     "Значение": -val if c.startswith("Сэвидж") else val})
    return pd.DataFrame(rows)


# ---------------- дерево решений ----------------

def build_tree(inputs: list[AltInput]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("root", kind="decision", label="Выбор направления")
    for a in inputs:
        na = f"A:{a.dest_id}"
        g.add_node(na, kind="decision", label=a.name)
        g.add_edge("root", na, label=a.name)
        for s, sname in STRATEGIES.items():
            ns = f"{na}/{s}"
            g.add_node(ns, kind="chance", label=sname)
            g.add_edge(na, ns, label=sname)
            for (pl, k), pk in zip(PRICE_STATES, PRICE_PROBS):
                npr = f"{ns}/p{k}"
                g.add_node(npr, kind="chance", label=f"цена {pl}")
                g.add_edge(ns, npr, prob=float(pk), label=f"цена {pl} ({pk:.2f})")
                for dis in (False, True):
                    nd = f"{npr}/d{int(dis)}"
                    pd_ = a.q if dis else 1 - a.q
                    if dis:
                        g.add_node(nd, kind="terminal", label="срыв", payoff=payoff(a, s, k, True, False))
                        g.add_edge(npr, nd, prob=pd_, label=f"срыв ({pd_:.3f})")
                        continue
                    g.add_node(nd, kind="chance", label="поездка состоялась")
                    g.add_edge(npr, nd, prob=pd_, label=f"без срыва ({pd_:.3f})")
                    for bad in (False, True):
                        nw = f"{nd}/w{int(bad)}"
                        pw = a.p_bad if bad else 1 - a.p_bad
                        g.add_node(nw, kind="terminal", label="плохая погода" if bad else "хорошая погода",
                                   payoff=payoff(a, s, k, False, bad))
                        g.add_edge(nd, nw, prob=pw, label=f"{'плохая' if bad else 'хорошая'} погода ({pw:.2f})")
    return g


def rollback(g: nx.DiGraph, node: str = "root") -> float:
    """Свёртка дерева: в узлах случая - математическое ожидание, в узлах решения - максимум."""
    kind = g.nodes[node]["kind"]
    if kind == "terminal":
        v = g.nodes[node]["payoff"]
    elif kind == "chance":
        v = sum(g.edges[node, c]["prob"] * rollback(g, c) for c in g.successors(node))
    else:
        vals = {c: rollback(g, c) for c in g.successors(node)}
        best = max(vals, key=vals.get)
        g.nodes[node]["best"] = best
        v = vals[best]
    g.nodes[node]["ev"] = v
    return v


def optimal_path(g: nx.DiGraph) -> list[str]:
    path, node = [], "root"
    while g.nodes[node]["kind"] == "decision":
        node = g.nodes[node]["best"]
        path.append(g.nodes[node]["label"])
    return path


# ---------------- пороговый анализ (scipy.optimize) ----------------

def ev(a: AltInput, s: str, q: float | None = None, premium: float = LATE_PREMIUM) -> float:
    q = a.q if q is None else q
    total = 0.0
    for (pl, k), pk in zip(PRICE_STATES, PRICE_PROBS):
        good = (1 - a.p_bad) * payoff(a, s, k, False, False, premium) + a.p_bad * payoff(a, s, k, False, True, premium)
        total += pk * ((1 - q) * good + q * payoff(a, s, k, True, False, premium))
    return total


def thresholds(a: AltInput) -> dict:
    """q* - вероятность срыва, при которой возвратный тариф (S2) сравнивается с невозвратным (S1);
    q** - при которой отложенная покупка (S3) сравнивается с S1;
    π* - наценка поздней покупки, при которой S3 сравнивается с S1."""
    f12 = lambda q: ev(a, "S1", q) - ev(a, "S2", q)
    f13 = lambda q: ev(a, "S1", q) - ev(a, "S3", q)
    fpi = lambda pi: ev(a, "S1", premium=pi) - ev(a, "S3", premium=pi)
    out = {}
    out["q_S1_S2"] = optimize.brentq(f12, 0.0, 0.99) if f12(0) * f12(0.99) < 0 else np.nan
    out["q_S1_S3"] = optimize.brentq(f13, 0.0, 0.99) if f13(0) * f13(0.99) < 0 else np.nan
    out["premium_S1_S3"] = optimize.brentq(fpi, -0.5, 0.5) if fpi(-0.5) * fpi(0.5) < 0 else np.nan
    return out


def hurwicz_switches(pay: pd.DataFrame) -> list[tuple[float, str]]:
    """Значения α (коэффициент пессимизма), при которых меняется лучшее решение по Гурвицу."""
    mn, mx = pay.min(axis=1), pay.max(axis=1)
    best = lambda al: (al * mn + (1 - al) * mx).idxmax()
    grid = np.linspace(0, 1, 201)
    out, prev = [], best(0.0)
    for al in grid[1:]:
        cur = best(al)
        if cur != prev:
            h = lambda x: (x * mn[prev] + (1 - x) * mx[prev]) - (x * mn[cur] + (1 - x) * mx[cur])
            root = optimize.brentq(h, al - 0.005, al)
            out.append((round(float(root), 4), cur))
            prev = cur
    return [(0.0, best(0.0))] + out


# ---------------- Монте-Карло ----------------

def monte_carlo(alts: list[dict], budget: float, n: int = 20000, seed: int = 1,
                extra=(0.03, 0.08, 0.15)) -> pd.DataFrame:
    """Стоимость поездки ~ логнормальная (σ из 80%-го прогнозного интервала П6) + траты на месте
    ~ треугольное распределение (доли бюджета: мин, мода, макс)."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in alts:
        sigma = np.log(a["cost_hi"] / a["cost_lo"]) / (2 * 1.2816)
        base = stats.lognorm(s=sigma, scale=a["cost"]).rvs(size=n, random_state=rng)
        extra_ = rng.triangular(*extra, n) * budget   # экскурсии, билеты, непредвиденное
        total = base * (1 + a.get("fees_rate", 0)) + extra_
        var95 = float(np.quantile(total, 0.95))
        rows.append({"dest_id": a["dest_id"], "name": a["name"], "mean": float(total.mean()),
                     "VaR95": var95, "CVaR95": float(total[total >= var95].mean()),
                     "P(превышение бюджета)": float(np.mean(total > budget))})
    return pd.DataFrame(rows)
