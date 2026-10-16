"""П7. Рекомендательный модуль.

Объекты рекомендаций: (1) направления, (2) объекты показа (POI) внутри направления.
Сигналы: явный профиль интересов ЛПР (7 признаков), чувствительность к цене, оценки 600
пользователей (1–5), сезонность (купальный сезон из базы знаний), рейтинги POI.
Алгоритм для направлений - гибрид:
  • контентная фильтрация: косинусная близость профиля ЛПР и признаков направления;
  • коллаборативная фильтрация (user-based kNN по профилям - решает «холодный старт»
    нового пользователя без истории оценок);
  • популярность: байесовская средняя оценка направления.
Веса гибрида подбираются по сетке на кросс-валидации (NDCG@3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURES, FEATURES_RU, TripRequest, load_destinations, load_pois, load_ratings, load_users
from .knowledge import ontology_facts

POI_FEATURE = {"museum": {"culture": 1.0}, "landmark": {"culture": 0.8, "family": 0.2},
               "nature": {"nature": 1.0}, "beach": {"beach": 1.0}, "gastronomy": {"gastronomy": 1.0},
               "entertainment": {"nightlife": 0.5, "family": 0.5}, "shopping": {"shopping": 1.0}}
DEFAULT_WEIGHTS = (0.6, 0.3, 0.1)   # контент, CF, популярность - подобраны в evaluate() по NDCG@3
K_NEIGHBORS = 40


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1) + 1e-9)


def destination_features(month: int | None = None) -> pd.DataFrame:
    """Признаки направлений; вне купального сезона пляжный признак ослабляется (знание из П5)."""
    d = load_destinations().set_index("id")[FEATURES].astype(float).copy()
    if month is not None:
        onto = ontology_facts().set_index("dest_id")
        for did in d.index:
            if d.loc[did, "beach"] > 0 and month not in onto.loc[did, "beach_season_months"]:
                d.loc[did, "beach"] *= 0.2
    return d


def bayes_popularity(ratings: pd.DataFrame, m: float = 20.0) -> pd.Series:
    g = ratings.groupby("dest_id").rating.agg(["mean", "count"])
    mu = ratings.rating.mean()
    return (g["count"] * g["mean"] + m * mu) / (g["count"] + m)


def cf_predict(profile: np.ndarray, users: pd.DataFrame, ratings: pd.DataFrame, k: int = K_NEIGHBORS,
               exclude_user: str | None = None) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """User-based kNN по профилям: оценка = средняя пользователя-соседа + взвешенные отклонения."""
    u = users if exclude_user is None else users[users.user_id != exclude_user]
    sims = _cos(profile, u[FEATURES].to_numpy())[0]
    top = u.assign(sim=sims).nlargest(k, "sim")[["user_id", "sim"]]
    r = ratings[ratings.user_id.isin(top.user_id)].merge(top, on="user_id")
    umean = r.groupby("user_id").rating.transform("mean")
    r = r.assign(dev=r.rating - umean)
    global_mean = ratings.rating.mean()
    pred = r.groupby("dest_id").apply(
        lambda g: global_mean + (g.sim * g.dev).sum() / (g.sim.abs().sum() + 1e-9), include_groups=False)
    support = r.groupby("dest_id").rating.agg(["mean", "count"])
    return pred, support["count"], support


@dataclass
class DestRec:
    table: pd.DataFrame       # dest_id, name, content, cf, popularity, score, rank, explanation
    weights: tuple


def _travelers(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "похожий путешественник оценил"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "похожих путешественника оценили"
    return "похожих путешественников оценили"


def _norm(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def recommend_destinations(trip: TripRequest, weights: tuple = DEFAULT_WEIGHTS) -> DestRec:
    users, ratings = load_users(), load_ratings()
    feats = destination_features(trip.month)
    prof = np.array([trip.profile[f] for f in FEATURES])
    content = pd.Series(_cos(prof, feats.to_numpy())[0], index=feats.index)
    cf, cnt, support = cf_predict(np.r_[prof], users, ratings)
    cf = cf.reindex(feats.index).fillna(ratings.rating.mean())
    pop = bayes_popularity(ratings).reindex(feats.index)
    # учёт чувствительности к цене: дорогие направления штрафуются
    base_cost = load_destinations().set_index("id").eval("flight_rub + 7*daily_rub")
    price_pen = trip.budget_sensitivity * _norm(base_cost) * 0.15
    wc, wcf, wp = weights
    score = wc * _norm(content) + wcf * _norm(cf) + wp * _norm(pop) - price_pen
    names = load_destinations().set_index("id")["name"]
    rows = []
    for did in feats.index:
        contrib = prof * feats.loc[did].to_numpy()
        top2 = np.argsort(contrib)[::-1][:2]
        share = contrib / (contrib.sum() + 1e-9)
        feat_txt = ", ".join(f"{FEATURES_RU[FEATURES[i]]} ({share[i]*100:.0f}%)" for i in top2)
        n_nb = int(cnt.get(did, 0))
        nb_mean = support["mean"].get(did, np.nan)
        cf_txt = (f"{n_nb} {_travelers(n_nb)} в среднем на {nb_mean:.1f}".replace(".", ",")
                  if n_nb else "нет оценок похожих путешественников")
        rows.append({"dest_id": did, "name": names[did], "content": round(float(content[did]), 3),
                     "cf_pred": round(float(cf[did]), 2), "popularity": round(float(pop[did]), 2),
                     "score": round(float(score[did]), 4),
                     "explanation": f"Совпадение интересов: {feat_txt}; {cf_txt}."})
    t = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return DestRec(t, weights)


# ---------------- оценка качества ----------------

def _ndcg_at_k(rel_sorted: np.ndarray, k: int = 3) -> float:
    rel = rel_sorted[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel_sorted)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate(n_folds: int = 5, seed: int = 7, grid_step: float = 0.1) -> dict:
    """Кросс-валидация по пользователям: пользователь тестовой части считается «новым» (оценки скрыты),
    известен только профиль. Ранжируем оценённые им направления, релевантные - оценка ≥ 4."""
    users, ratings = load_users(), load_ratings()
    feats = destination_features(None)
    F = feats.to_numpy()
    rng = np.random.default_rng(seed)
    uid = users.user_id.to_numpy().copy()
    rng.shuffle(uid)
    folds = np.array_split(uid, n_folds)
    base_cost = load_destinations().set_index("id").eval("flight_rub + 7*daily_rub").reindex(feats.index)
    per_user = []   # компоненты скоринга для каждого тестового пользователя
    for f in folds:
        test = set(f)
        tr_users = users[~users.user_id.isin(test)]
        tr_r = ratings[ratings.user_id.isin(tr_users.user_id)]
        pop = bayes_popularity(tr_r).reindex(feats.index).fillna(tr_r.rating.mean())
        for _, u in users[users.user_id.isin(test)].iterrows():
            p = u[FEATURES].to_numpy(dtype=float)
            ur = ratings[ratings.user_id == u.user_id].set_index("dest_id").rating
            if ur.nunique() < 2:
                continue
            content = pd.Series(_cos(p, F)[0], index=feats.index)
            cf, _, _ = cf_predict(p, tr_users, tr_r)
            cf = cf.reindex(feats.index).fillna(tr_r.rating.mean())
            pen = u.budget_sensitivity * _norm(base_cost) * 0.15
            per_user.append((ur, _norm(content), _norm(cf), _norm(pop), pen, cf))

    def metrics_for(wc, wcf, wp, use_pen=True):
        nd, prec, auc, rmse_e = [], [], [], []
        for ur, c, cf, pop, pen, cf_raw in per_user:
            s = (wc * c + wcf * cf + wp * pop - (pen if use_pen else 0)).reindex(ur.index)
            order = s.sort_values(ascending=False).index
            rel = ur.reindex(order).to_numpy()
            nd.append(_ndcg_at_k(rel - 1, 3))
            prec.append(np.mean(rel[:3] >= 4))
            pos, neg = s[ur >= 4], s[ur < 4]
            if len(pos) and len(neg):
                auc.append(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
        return {"NDCG@3": float(np.mean(nd)), "Precision@3": float(np.mean(prec)), "AUC": float(np.mean(auc))}

    rows = [
        {"model": "Случайный порядок", **_random_baseline(per_user, seed)},
        {"model": "Популярность", **metrics_for(0, 0, 1, False)},
        {"model": "Контентная фильтрация", **metrics_for(1, 0, 0)},
        {"model": "Коллаборативная (kNN)", **metrics_for(0, 1, 0)},
    ]
    best, best_w = -1, DEFAULT_WEIGHTS
    grid = []
    steps = np.round(np.arange(0, 1 + 1e-9, grid_step), 2)
    for wc in steps:
        for wcf in steps:
            wp = round(1 - wc - wcf, 2)
            if wp < -1e-9:
                continue
            m = metrics_for(wc, wcf, wp)
            grid.append({"w_content": wc, "w_cf": wcf, "w_pop": wp, **m})
            if m["NDCG@3"] > best:
                best, best_w = m["NDCG@3"], (wc, wcf, wp)
    rows.append({"model": f"Гибрид (веса {best_w[0]:.1f}/{best_w[1]:.1f}/{best_w[2]:.1f})", **metrics_for(*best_w)})
    # точность предсказания оценки CF
    errs = []
    for ur, *_, cf_raw in per_user:
        errs.extend((cf_raw.reindex(ur.index) - ur).to_numpy())
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    mean_errs = []
    for ur, *_ in per_user:
        mean_errs.extend((ratings.rating.mean() - ur).to_numpy())
    return {"table": pd.DataFrame(rows), "grid": pd.DataFrame(grid), "best_weights": best_w,
            "cf_rmse": rmse, "baseline_rmse": float(np.sqrt(np.mean(np.square(mean_errs)))),
            "n_test_users": len(per_user)}


def _random_baseline(per_user, seed):
    rng = np.random.default_rng(seed)
    nd, prec, auc = [], [], []
    for ur, *_ in per_user:
        for _ in range(20):
            order = rng.permutation(ur.index)
            rel = ur.reindex(order).to_numpy()
            nd.append(_ndcg_at_k(rel - 1, 3))
            prec.append(np.mean(rel[:3] >= 4))
        auc.append(0.5)
    return {"NDCG@3": float(np.mean(nd)), "Precision@3": float(np.mean(prec)), "AUC": 0.5}


# ---------------- рекомендации POI ----------------

def recommend_pois(dest_id: str, trip: TripRequest, exclude_categories=(), outdoor_factor: float = 1.0,
                   beach_bonus: bool = False) -> pd.DataFrame:
    """Полезность объекта = рейтинг × соответствие интересам ЛПР (используется в П4)."""
    pois = load_pois()
    p = pois[pois.dest_id == dest_id].copy()
    p = p[~p.category.isin(exclude_categories)]
    rows = []
    for _, x in p.iterrows():
        fit = sum(w * trip.profile[f] for f, w in POI_FEATURE[x.category].items())
        rating_norm = (x.rating - 4.0) / 1.0            # 4.0…5.0 → 0…1
        util = 10 * (0.35 + 0.65 * fit) * (0.6 + 0.4 * rating_norm)
        if x.outdoor:
            util *= outdoor_factor
        if beach_bonus and x.category == "beach":
            util *= 1.2
        main = max(POI_FEATURE[x.category], key=POI_FEATURE[x.category].get)
        rows.append({**x.to_dict(), "utility": round(util, 2), "fit": round(fit, 2),
                     "why": f"интерес «{FEATURES_RU[main]}» {trip.profile[main]:.1f}, рейтинг {x.rating}"})
    return pd.DataFrame(rows).sort_values("utility", ascending=False).reset_index(drop=True)
