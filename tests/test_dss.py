"""Проверки ключевых свойств модулей СППР.  Запуск: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tdss import ahp, knowledge, pipeline, recommender, route_opt, uncertainty  # noqa: E402
from tdss.costs import trip_cost  # noqa: E402
from tdss.data import TripRequest  # noqa: E402


@pytest.fixture(scope="module")
def trip():
    return TripRequest()


@pytest.fixture(scope="module")
def result(trip):
    return pipeline.run(trip, route_time_limit=30)


def test_ahp_consistency_and_weights(trip):
    r = ahp.run(trip)
    assert r.criteria.cr < 0.10
    assert np.isclose(r.criteria.weights.sum(), 1.0)
    assert all(cr < 0.10 for cr in r.local_cr.values())
    assert np.isclose(r.scores.score.sum(), 1.0)


def test_saaty_reciprocity():
    a = ahp.matrix_from_values(np.array([1.0, 3.0, 2.0, 5.0]))
    assert np.allclose(a * a.T, 1.0)


def test_kb_excludes_over_budget(trip):
    inf = knowledge.evaluate("rome", trip, trip_cost("rome", trip))
    assert inf.excluded and "ожидаемая стоимость выше бюджета" in inf.facts["reasons"]
    inf = knowledge.evaluate("kazan", trip, trip_cost("kazan", trip))
    assert not inf.excluded


def test_kb_heat_with_children():
    t = TripRequest(month=7, with_children=True, budget_per_person=300_000)
    inf = knowledge.evaluate("dubai", t, trip_cost("dubai", t))
    assert inf.excluded


def test_three_distinct_feasible_variants(result):
    ids = [v.dest_id for v in result.variants]
    assert len(set(ids)) == 3
    feasible = set(result.table[~result.table.excluded].dest_id)
    assert set(ids) <= feasible


def test_routes_respect_time_and_budget(result, trip):
    for v in result.variants:
        t = v.route.totals
        assert v.route.status in ("Optimal", "TimeLimit", "GapLimit")
        assert all(u <= h + 1e-6 for u, h in zip(t["hours_used"], t["hours_available"]))
        assert t["total"] <= trip.budget_per_person + 1e-6
        assert v.route.plan.poi_id.is_unique


def test_dinner_is_last_stop(result):
    for v in result.variants:
        for _, g in v.route.plan.groupby("day"):
            names = g.sort_values("order").name.tolist()
            for k, n in enumerate(names):
                if "ужин" in n.lower():
                    assert k == len(names) - 1


def test_optimizer_not_worse_than_greedy(result, trip):
    for v in result.variants:
        g = route_opt.greedy_route(v.dest_id, v.pois, trip.days, trip.pace_hours)
        assert v.route.totals["utility"] >= g["utility"] - 1e-6 or v.kind == "Экономный"


def test_tree_rollback_matches_bayes(result):
    assert np.isclose(result.tree_ev, result.criteria["Байес (ожидаемый выигрыш)"].max())


def test_probabilities_sum_to_one(result):
    assert np.allclose(result.prob.sum(axis=1), 1.0)


def test_recommendations_explained(trip):
    r = recommender.recommend_destinations(trip)
    assert r.table.explanation.str.len().min() > 20
    assert r.table["rank"].tolist() == list(range(1, 11))


def test_threshold_is_root():
    a = uncertainty.AltInput("x", "X", 50_000, 90_000, 0.03, 0.3, 0.2, 0.06)
    q = uncertainty.thresholds(a)["q_S1_S2"]
    assert abs(uncertainty.ev(a, "S1", q) - uncertainty.ev(a, "S2", q)) < 1e-6


def test_reliable_variant_strategy_follows_hurwicz(result):
    v = next(x for x in result.variants if x.kind == "Надёжный")
    hz = [c for c in result.criteria.columns if c.startswith("Гурвиц")][0]
    rows = result.criteria[hz][[k for k in result.criteria.index if k.startswith(v.name + " |")]]
    assert v.strategy == rows.idxmax().split(" | ")[1]


def test_economy_variant_uses_economy_hotel(result):
    v = next(x for x in result.variants if x.kind == "Экономный")
    assert v.route.hotel_class == "эконом"


def test_route_totals_are_plain_floats(result):
    for v in result.variants:
        assert all(type(h) is float for h in v.route.totals["hours_used"])


def test_profile_matrix_follows_interests():
    beach = TripRequest(month=7, with_children=True,
                        profile={"culture": 0.2, "nature": 0.5, "beach": 0.95, "gastronomy": 0.4,
                                 "nightlife": 0.1, "shopping": 0.3, "family": 0.9})
    for t in (TripRequest(), beach):
        assert ahp.priority_vector(ahp.matrix_from_profile(t)).cr < 0.10
    w_culture = ahp.priority_vector(ahp.matrix_from_profile(TripRequest())).weights
    w_beach = ahp.priority_vector(ahp.matrix_from_profile(beach)).weights
    k3, k4 = ahp.CRIT_IDS.index("K3"), ahp.CRIT_IDS.index("K4")
    assert w_culture[k3] > w_culture[k4] and w_beach[k4] > w_beach[k3]


def test_baltic_beach_season_in_summer():
    facts = knowledge._python_inference().set_index("dest_id")
    assert {7, 8} <= set(facts.loc["kaliningrad", "beach_season_months"])
    assert 5 not in facts.loc["kaliningrad", "beach_season_months"]
