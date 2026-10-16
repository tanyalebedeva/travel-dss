"""П6. Предиктивная аналитика: прогноз стоимости 7-дневной поездки по направлениям.

Прогнозируемая величина: средняя стоимость 7-дневной поездки из Москвы на 1 чел. (перелёт
+ проживание/питание/местный транспорт), руб., помесячно.
Модели: сезонный наивный прогноз с дрейфом (базовая), Хольт–Винтерс (ETS, лог-шкала),
SARIMA(1,1,0)(0,1,1)12 (лог-шкала). Валидация - отложенные последние 12 месяцев.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import REPORTS, ensure_dirs, load_destinations, load_price_history

TRAIN_START = "2021-01-01"   # ковидный период 2020 г. исключён (зарубежье - пропуски, РФ - аномалия)
HORIZON_TEST = 12
FORECAST_END = "2027-12-01"
Z80 = 1.2816

warnings.filterwarnings("ignore")


def series_matrix() -> pd.DataFrame:
    ph = load_price_history()
    wide = ph.pivot(index="month", columns="dest_id", values="cost_7d_rub").sort_index()
    wide = wide.loc[TRAIN_START:]
    wide.index = pd.DatetimeIndex(wide.index, freq="MS")
    return wide


# ---------- модели: fit(y_train) -> (mean, lo, hi) на h шагов ----------

def snaive_drift(y: pd.Series, h: int):
    log = np.log(y.to_numpy())
    drift = (log[-12:].mean() - log[-24:-12].mean()) / 12.0   # среднемесячный рост за последний год
    base = log[-12:]
    steps = np.arange(1, h + 1)
    mean = np.array([base[(k - 1) % 12] + drift * (12 * ((k - 1) // 12 + 1)) for k in steps])
    resid = log[12:] - log[:-12] - 12 * drift
    sigma = resid.std(ddof=1) * np.sqrt(np.ceil(steps / 12))
    return np.exp(mean), np.exp(mean - Z80 * sigma), np.exp(mean + Z80 * sigma)


def holt_winters(y: pd.Series, h: int):
    log = np.log(y)
    model = ExponentialSmoothing(log, trend="add", damped_trend=True, seasonal="add",
                                 seasonal_periods=12, initialization_method="estimated").fit()
    mean = model.forecast(h).to_numpy()
    sims = model.simulate(h, repetitions=500, error="add", anchor="end", random_state=1).to_numpy()
    lo, hi = np.quantile(sims, [0.1, 0.9], axis=1)
    return np.exp(mean), np.exp(lo), np.exp(hi)


def sarima(y: pd.Series, h: int):
    log = np.log(y)
    model = SARIMAX(log, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(h)
    mean = fc.predicted_mean.to_numpy()
    ci = fc.conf_int(alpha=0.2).to_numpy()
    return np.exp(mean), np.exp(ci[:, 0]), np.exp(ci[:, 1])


MODELS = {"Сезонный наивный + дрейф": snaive_drift, "Хольт–Винтерс (ETS)": holt_winters,
          "SARIMA(1,1,0)(0,1,1)12": sarima}


def _metrics(y_true, mean, lo, hi) -> dict:
    err = y_true - mean
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
        "Покрытие 80% ПИ": float(np.mean((y_true >= lo) & (y_true <= hi)) * 100),
    }


def validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Обучение на 2021-01…2025-09, проверка на 2025-10…2026-09."""
    wide = series_matrix()
    train, test = wide.iloc[:-HORIZON_TEST], wide.iloc[-HORIZON_TEST:]
    rows, preds = [], []
    for dest in wide.columns:
        for name, fn in MODELS.items():
            mean, lo, hi = fn(train[dest], HORIZON_TEST)
            m = _metrics(test[dest].to_numpy(), mean, lo, hi)
            rows.append({"dest_id": dest, "model": name, **m})
            for t, a, p, l_, h_ in zip(test.index, test[dest], mean, lo, hi):
                preds.append({"dest_id": dest, "model": name, "month": t, "actual": a,
                              "pred": p, "lo80": l_, "hi80": h_})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def best_model(metrics: pd.DataFrame) -> str:
    return metrics.groupby("model")["MAPE"].mean().idxmin()


def forecast_all(model_name: str) -> pd.DataFrame:
    wide = series_matrix()
    last = wide.index[-1]
    future = pd.date_range(last + pd.offsets.MonthBegin(1), FORECAST_END, freq="MS")
    fn = MODELS[model_name]
    rows = []
    for dest in wide.columns:
        mean, lo, hi = fn(wide[dest], len(future))
        for t, p, l_, h_ in zip(future, mean, lo, hi):
            rows.append({"dest_id": dest, "month": t, "forecast": round(float(p), -1),
                         "lo80": round(float(l_), -1), "hi80": round(float(h_), -1), "model": model_name})
    return pd.DataFrame(rows)


def run(save: bool = True) -> dict:
    ensure_dirs()
    metrics, preds = validate()
    best = best_model(metrics)
    fc = forecast_all(best)
    summary = metrics.groupby("model")[["MAE", "RMSE", "MAPE", "Покрытие 80% ПИ"]].mean().round(2)
    if save:
        metrics.round(3).to_csv(REPORTS / "p6_metrics_by_dest.csv", index=False)
        summary.to_csv(REPORTS / "p6_metrics_summary.csv")
        preds.to_csv(REPORTS / "p6_validation_predictions.csv", index=False)
        fc.to_csv(REPORTS / "forecast.csv", index=False)
    return {"metrics": metrics, "summary": summary, "best": best, "forecast": fc, "preds": preds}


def load_forecast() -> pd.DataFrame:
    path = REPORTS / "forecast.csv"
    if not path.exists():
        run(save=True)
    return pd.read_csv(path, parse_dates=["month"])


def base_cost(dest_id: str) -> float:
    d = load_destinations().set_index("id").loc[dest_id]
    return float(d.flight_rub + 7 * d.daily_rub)


def price_factor(dest_id: str, year: int, month: int) -> tuple[float, float, float]:
    """Множитель к базовой (средней за последние 12 мес.) цене: (точечный, нижний 80%, верхний 80%)."""
    fc = load_forecast()
    t = pd.Timestamp(year=year, month=month, day=1)
    row = fc[(fc.dest_id == dest_id) & (fc.month == t)]
    b = base_cost(dest_id)
    if len(row):
        r = row.iloc[0]
        return r.forecast / b, r.lo80 / b, r.hi80 / b
    # вне горизонта прогноза - средний исторический сезонный множитель за 2023–2026
    ph = load_price_history()
    h = ph[(ph.dest_id == dest_id) & (ph.month.dt.month == month) & (ph.month >= "2023-01-01")]
    f = float(h.cost_7d_rub.mean() / b)
    return f, f * 0.85, f * 1.15
