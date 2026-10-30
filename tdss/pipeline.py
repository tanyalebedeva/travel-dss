"""П8. Интеграция компонентов СППР: от параметров поездки ЛПР до трёх обоснованных вариантов.

Поток данных:
  ЛПР (месяц, дни, бюджет, интересы, матрица важности критериев)
   → П6 прогноз цен → ожидаемая стоимость по направлениям
   → П5 база знаний → допустимость, предупреждения, риски, исключаемые категории объектов
   → П2 АНР → оценка направлений по 6 критериям
   → П7 рекомендации → персональное соответствие + опыт похожих путешественников
   → интегральная оценка → кандидаты
   → П3 риски → стратегия бронирования, вероятность перерасхода
   → П4 оптимизация → маршрут по дням и класс проживания
   → три варианта: «Оптимальный», «Экономный», «Надёжный».
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import ahp, recommender, route_opt, uncertainty
from .costs import HOTEL_CLASSES, trip_cost
from .data import TripRequest, load_destinations
from .knowledge import evaluate

W_AHP, W_REC = 0.7, 0.3
# резерв бюджета на непредвиденные расходы, который маршрут не может тратить
RESERVE = {"Оптимальный": 0.05, "Экономный": 0.05, "Надёжный": 0.15}


@dataclass
class Variant:
    kind: str
    dest_id: str
    name: str
    score: float
    rationale: list
    strategy: str
    route: route_opt.RouteResult
    cost: dict
    risk: dict
    warnings: list
    pois: pd.DataFrame = field(repr=False, default=None)


@dataclass
class DSSResult:
    trip: TripRequest
    table: pd.DataFrame            # сводная таблица по всем направлениям
    ahp: ahp.AHPResult
    rec: recommender.DestRec
    kb: dict                       # dest_id -> Inference
    payoff: pd.DataFrame
    prob: pd.DataFrame
    criteria: pd.DataFrame
    best_by_criteria: pd.DataFrame
    tree_ev: float
    tree_path: list
    thresholds: dict
    mc: pd.DataFrame
    variants: list


def run(trip: TripRequest, criteria_matrix=None, rec_weights=recommender.DEFAULT_WEIGHTS,
        route_time_limit: int = 30, with_routes: bool = True) -> DSSResult:
    dest = load_destinations()
    costs = {d: trip_cost(d, trip) for d in dest.id}
    kb = {d: evaluate(d, trip, costs[d]) for d in dest.id}
    a = ahp.run(trip, criteria_matrix)
    r = recommender.recommend_destinations(trip, rec_weights)

    t = a.scores[["dest_id", "name", "score"]].rename(columns={"score": "ahp"})
    t = t.merge(r.table[["dest_id", "score", "explanation"]].rename(columns={"score": "rec"}), on="dest_id")
    t["rec"] = t["rec"].clip(lower=0)
    t["integral"] = W_AHP * t.ahp / t.ahp.max() + W_REC * t.rec / t.rec.max()
    t["cost"] = t.dest_id.map(lambda d: kb[d].facts["total_with_fees"])
    t["cost_lo"] = t.dest_id.map(lambda d: costs[d]["total_lo"])
    t["cost_hi"] = t.dest_id.map(lambda d: costs[d]["total_hi"])
    t["status"] = t.dest_id.map(lambda d: kb[d].facts["status"])
    t["excluded"] = t.dest_id.map(lambda d: kb[d].excluded or kb[d].facts["status"] != "допустимо")
    t["reasons"] = t.dest_id.map(lambda d: "; ".join(kb[d].facts["reasons"]))
    t["disruption_risk"] = t.dest_id.map(lambda d: kb[d].facts["disruption_risk"])
    t["fees_rate"] = t.dest_id.map(lambda d: kb[d].facts["fees_rate"])
    t = t.sort_values("integral", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)

    feas = t[~t.excluded].reset_index(drop=True)
    if feas.empty:
        raise ValueError("Нет допустимых направлений: увеличьте бюджет или измените сроки поездки")

    # ---- кандидаты для анализа рисков: 4 лучших + самый дешёвый из близких по оценке ----
    v1 = feas.iloc[0]
    close = feas[(feas.dest_id != v1.dest_id) & (feas.integral >= 0.85 * v1.integral)]
    v2 = (close if len(close) else feas[feas.dest_id != v1.dest_id]).sort_values("cost").iloc[0] \
        if len(feas) > 1 else v1
    cand = pd.concat([feas.head(4), feas[feas.dest_id == v2.dest_id]]).drop_duplicates("dest_id")
    alts = [{"dest_id": x.dest_id, "name": x["name"], "score": x.integral, "cost": x.cost,
             "cost_lo": x.cost_lo, "cost_hi": x.cost_hi, "disruption_risk": x.disruption_risk,
             "fees_rate": x.fees_rate} for _, x in cand.iterrows()]
    inputs = uncertainty.build_inputs(alts, trip.budget_per_person, trip.month, max(trip.months_ahead - 1, 1))
    pay, prob = uncertainty.payoff_tables(inputs)
    crit = uncertainty.classic_criteria(pay, prob)
    bbc = uncertainty.best_by_criteria(crit)
    tree = uncertainty.build_tree(inputs)
    tree_ev = uncertainty.rollback(tree)
    path = uncertainty.optimal_path(tree)
    thr = {i.dest_id: uncertainty.thresholds(i) for i in inputs}
    mc = uncertainty.monte_carlo(alts, trip.budget_per_person)

    # ---- «Надёжный»: лучший по Гурвицу (α = 0.6) среди оставшихся кандидатов ----
    hz_col = [c for c in crit.columns if c.startswith("Гурвиц")][0]
    by_dest = crit[hz_col].groupby(lambda k: k.split(" | ")[0]).max()
    name_to_id = dict(zip(cand["name"], cand.dest_id))
    order = [name_to_id[n] for n in by_dest.sort_values(ascending=False).index]
    v3_id = next((d for d in order if d not in (v1.dest_id, v2.dest_id)), order[0])
    v3 = feas[feas.dest_id == v3_id].iloc[0]

    variants = []
    bayes = crit["Байес (ожидаемый выигрыш)"]
    hurwicz = crit[hz_col]
    base_cw = 0.8 * trip.budget_sensitivity
    # «Экономный»: только эконом-проживание и умеренный штраф за траты, чтобы маршрут
    # оставался полноценным (при сильном штрафе оптимизатор оставлял 2–3 объекта на неделю)
    econ_hotels = {"эконом": HOTEL_CLASSES["эконом"]}
    for kind, row, cw, hotels, rule in (("Оптимальный", v1, base_cw, None, bayes),
                                        ("Экономный", v2, max(base_cw, 1.0), econ_hotels, bayes),
                                        ("Надёжный", v3, base_cw, None, hurwicz)):
        did = row.dest_id
        inf = kb[did]
        # стратегия бронирования выбирается тем же критерием, что и сам вариант:
        # для «Надёжного» (осторожный ЛПР) - по Гурвицу, для остальных - по Байесу
        rows_s = rule[[k for k in rule.index if k.startswith(row["name"] + " |")]]
        strategy = rows_s.idxmax().split(" | ")[1]
        mc_row = mc[mc.dest_id == did].iloc[0].to_dict()
        pois = recommender.recommend_pois(did, trip, inf.facts["exclude_categories"],
                                          inf.facts["outdoor_factor"], inf.facts.get("beach_bonus", False))
        c = costs[did]
        route = None
        if with_routes:
            route_budget = trip.budget_per_person * (1 - RESERVE[kind]) / (1 + inf.facts["fees_rate"])
            route = route_opt.optimize_route(
                did, pois, trip.days, route_budget,
                c["flight"] + c["visa"], c["daily"], cost_weight=cw, pace=trip.pace_hours,
                must_visit=trip.must_visit, hotel_classes=hotels, time_limit=route_time_limit)
            if route.totals:
                tot = route.totals["total"]
                fixed = route.totals["flight_visa"] + route.totals["stay"]
                vr = uncertainty.monte_carlo([{
                    "dest_id": did, "name": row["name"], "fees_rate": inf.facts["fees_rate"],
                    "cost": fixed, "cost_lo": fixed * c["total_lo"] / c["total"],
                    "cost_hi": fixed * c["total_hi"] / c["total"]}],
                    trip.budget_per_person - (tot - fixed) * (1 + inf.facts["fees_rate"]),
                    extra=(0.0, 0.03, 0.08)).iloc[0].to_dict()
                mc_row = {**mc_row, "P(перерасход маршрута)": vr["P(превышение бюджета)"],
                          "route_total_with_fees": round(tot * (1 + inf.facts["fees_rate"]), -1)}
        warns = list(inf.warnings)
        if mc_row.get("P(перерасход маршрута)", 0) > 0.2:
            warns.append(f"Высокий риск перерасхода по маршруту ({mc_row['P(перерасход маршрута)']*100:.0f}%): "
                         "маршрут почти исчерпывает бюджет - проверьте реальные цены перед покупкой "
                         "или выберите вариант «Экономный»")
        variants.append(Variant(kind, did, row["name"], float(row.integral),
                                _rationale(kind, row, t, a, mc_row, thr[did], strategy, bbc,
                                           rule_name="Гурвица (α = 0,6)" if rule is hurwicz else "Байеса"),
                                strategy, route, c, mc_row, warns, pois))
    return DSSResult(trip, t, a, r, kb, pay, prob, crit, bbc, tree_ev, path, thr, mc, variants)


def _rationale(kind, row, t, a, mc_row, thr, strategy, bbc=None, rule_name="Байеса") -> list[str]:
    loc = a.local.loc[row.dest_id]
    w = a.criteria.weights
    contrib = loc.to_numpy() * w
    top = np.argsort(contrib)[::-1][:2]
    crit_txt = " и ".join(f"«{ahp.CRIT_NAMES[ahp.CRIT_IDS[i]].lower()}»" for i in top)
    out = [f"Интегральная оценка {row.integral:.3f} (место {int(row['rank'])} из {len(t)}); "
           f"наибольший вклад в АНР - критерии {crit_txt}.",
           row.explanation,
           f"Прогноз стоимости перелёта и проживания {row.cost:,.0f} ₽ (80%-й интервал {row.cost_lo:,.0f}–"
           f"{row.cost_hi:,.0f} ₽).".replace(",", " "),
           _overrun_text(mc_row),
           f"Стратегия бронирования по критерию {rule_name}: «{strategy}». По ожидаемому выигрышу возвратный "
           f"тариф выгоднее, если вероятность срыва выше {thr['q_S1_S2']*100:.0f}% "
           f"(сейчас {row.disruption_risk*100:.1f}%)."]
    if bbc is not None:
        wins = [c.split(" (")[0] for c, d in zip(bbc["Критерий"], bbc["Лучшее решение"])
                if d.startswith(row["name"] + " |")]
        if wins:
            out.append("Лучшее решение в условиях неопределённости по критериям: " + ", ".join(wins) + ".")
    if kind == "Экономный":
        out.insert(0, "Самый дешёвый вариант среди направлений с оценкой не ниже 85% от лучшей.")
    if kind == "Надёжный":
        out.insert(0, "Лучший по критерию Гурвица (α = 0,6, осторожный ЛПР) среди оставшихся кандидатов.")
    return out


def _overrun_text(mc_row: dict) -> str:
    """Две разные вероятности перерасхода подписываются явно, чтобы ЛПР не путал их."""
    base = mc_row["P(превышение бюджета)"] * 100
    txt = f"Вероятность выйти за бюджет по типовой смете направления - {base:.1f}%"
    if "P(перерасход маршрута)" in mc_row:
        txt += (f"; для построенного маршрута (с билетами, такси и выбранным отелем) - "
                f"{mc_row['P(перерасход маршрута)'] * 100:.1f}%")
    return txt + "."
