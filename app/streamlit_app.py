"""СППР выбора места для путешествия - интерфейс ЛПР (Streamlit).

Запуск:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from tdss import ahp, pipeline, uncertainty  # noqa: E402
from tdss.data import FEATURES, FEATURES_RU, MONTHS_RU, TripRequest, load_destinations  # noqa: E402
from tdss.forecast import load_forecast, series_matrix  # noqa: E402
from tdss.knowledge import RULES, SWRL_RULES, ontology_facts  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SAATY_LABELS = [f"1/{k}" for k in range(9, 1, -1)] + [str(k) for k in range(1, 10)]
SAATY_VALUES = [1 / k for k in range(9, 1, -1)] + list(range(1, 10))
VARIANT_ICON = {"Оптимальный": "★", "Экономный": "₽", "Надёжный": "⛨"}

st.set_page_config(page_title="СППР: выбор места для путешествия", page_icon="🧭", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1280px;}
  .vcard {border: 1px solid rgba(11,11,11,.12); border-radius: 10px; padding: 14px 16px; height: 100%;}
  .vcard h4 {margin: 0 0 2px 0; font-size: 1.05rem;}
  .vkind {font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; opacity: .7;}
  .vbig {font-size: 1.5rem; font-weight: 600; margin: 6px 0 2px;}
  .vmuted {opacity: .72; font-size: .86rem;}
  .warn {font-size: .84rem; margin: 2px 0;}
</style>""", unsafe_allow_html=True)


def rub(x: float) -> str:
    return f"{x:,.0f} ₽".replace(",", " ")


# ------------------------------------------------------------------ боковая панель
with st.sidebar:
    st.header("Параметры поездки")
    c1, c2 = st.columns(2)
    month = c1.selectbox("Месяц", list(range(1, 13)), index=4, format_func=lambda m: MONTHS_RU[m - 1].capitalize())
    year = c2.selectbox("Год", [2027], index=0)
    days = st.slider("Длительность, дней", 2, 10, 5)
    budget = st.number_input("Бюджет на человека, ₽", 20_000, 400_000, 70_000, step=5_000)
    pace = st.select_slider("Темп", options=[6, 8, 10], value=8,
                            format_func=lambda h: {6: "спокойный (6 ч)", 8: "средний (8 ч)", 10: "активный (10 ч)"}[h])
    with_children = st.checkbox("Еду с детьми", value=False)

    st.subheader("Интересы")
    defaults = TripRequest().profile
    profile = {f: st.slider(FEATURES_RU[f].capitalize(), 0.0, 1.0, float(defaults[f]), 0.05) for f in FEATURES}
    budget_sens = st.slider("Чувствительность к цене", 0.0, 1.0, 0.6, 0.05)

    st.subheader("Важность критериев (АНР)")
    mode = st.radio("Режим", ["Матрица ЛПР по умолчанию", "Быстрый: оценки 1–9", "Экспертный: парные сравнения"],
                    label_visibility="collapsed")
    if mode == "Быстрый: оценки 1–9":
        imp = {c: st.slider(f"{c} {ahp.CRIT_NAMES[c]}", 1, 9, v)
               for c, v in zip(ahp.CRIT_IDS, [7, 3, 6, 2, 3, 5])}
        matrix = ahp.matrix_from_importance(imp)
    elif mode == "Экспертный: парные сравнения":
        matrix = np.ones((6, 6))
        st.caption("Во сколько раз критерий слева важнее критерия справа")
        for i in range(6):
            for j in range(i + 1, 6):
                d0 = ahp.DEFAULT_CRITERIA_MATRIX[i, j]
                idx = int(np.argmin([abs(np.log(v) - np.log(d0)) for v in SAATY_VALUES]))
                lab = st.select_slider(f"{ahp.CRIT_IDS[i]} vs {ahp.CRIT_IDS[j]}", SAATY_LABELS, value=SAATY_LABELS[idx],
                                       key=f"pc{i}{j}")
                matrix[i, j] = SAATY_VALUES[SAATY_LABELS.index(lab)]
                matrix[j, i] = 1 / matrix[i, j]
    else:
        matrix = ahp.DEFAULT_CRITERIA_MATRIX
    pr = ahp.priority_vector(matrix)
    st.caption(f"Отношение согласованности ОС = {pr.cr:.3f} " + ("✓ (< 0,10)" if pr.consistent else "✗ пересмотрите оценки"))
    run_btn = st.button("Подобрать варианты", type="primary", width="stretch")


@st.cache_data(show_spinner=False)
def run_dss(params: str, matrix_key: tuple):
    p = json.loads(params)
    trip = TripRequest(**p)
    m = np.array(matrix_key).reshape(6, 6)
    return pipeline.run(trip, criteria_matrix=m, route_time_limit=30)


params = dict(year=year, month=month, days=days, budget_per_person=float(budget), pace_hours=float(pace),
              with_children=with_children, profile=profile, budget_sensitivity=budget_sens)
key = (json.dumps(params, sort_keys=True), tuple(np.round(np.asarray(matrix, float).ravel(), 6)))
if run_btn or "last_key" not in st.session_state:
    st.session_state["last_key"] = key
st.title("🧭 СППР выбора места для путешествия")
trip = TripRequest(**json.loads(st.session_state["last_key"][0]))
if st.session_state["last_key"] != key:
    st.info("Параметры изменены - нажмите «Подобрать варианты», чтобы пересчитать.")
st.caption(f"Сценарий: {trip.label()}; темп {trip.pace_hours:.0f} ч/день"
           + ("; с детьми" if trip.with_children else "") + ". Вылет из Москвы, расчёт на 1 человека.")

try:
    with st.spinner("Прогноз цен → база знаний → АНР → рекомендации → риски → оптимизация маршрутов…"):
        res = run_dss(*st.session_state["last_key"])
except ValueError as e:
    st.error(str(e))
    st.stop()

tabs = st.tabs(["Итог: 3 варианта", "Рейтинг направлений", "Риски и неопределённость", "Прогноз цен",
                "База знаний", "О системе"])

# ------------------------------------------------------------------ 1. Итог
with tabs[0]:
    cols = st.columns(3)
    for col, v in zip(cols, res.variants):
        t = v.route.totals
        with col:
            st.markdown(f"""<div class="vcard">
              <div class="vkind">{VARIANT_ICON[v.kind]} {v.kind}</div>
              <h4>{v.name}</h4>
              <div class="vbig">{rub(v.risk.get('route_total_with_fees', t['total']))}</div>
              <div class="vmuted">проживание: {v.route.hotel_class} · объектов: {t['n_pois']} ·
              риск перерасхода по маршруту: {v.risk.get('P(перерасход маршрута)', 0)*100:.0f}%</div>
              <div class="vmuted">бронирование: {v.strategy}</div></div>""", unsafe_allow_html=True)
    st.write("")
    pick = st.radio("Подробности варианта", [v.kind for v in res.variants], horizontal=True)
    v = next(x for x in res.variants if x.kind == pick)
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader(f"{v.kind}: {v.name}")
        st.markdown("**Почему этот вариант**")
        for r in v.rationale:
            st.markdown(f"- {r}")
        if v.warnings:
            st.markdown("**Предупреждения базы знаний**")
            for w in v.warnings:
                st.markdown(f"<div class='warn'>⚠️ {w}</div>", unsafe_allow_html=True)
        t = v.route.totals
        st.markdown("**Смета, ₽**")
        st.dataframe(pd.DataFrame({"Статья": ["Перелёт и виза", f"Проживание и питание ({v.route.hotel_class})",
                                              "Билеты в объекты", "Трансферы", "Итого (без комиссий)", "Запас бюджета"],
                                   "Сумма": [t["flight_visa"], t["stay"], t["tickets"], t["transfers"], t["total"],
                                             t["budget_slack"]]}), hide_index=True, width="stretch")
    with right:
        plan = v.route.plan
        d = load_destinations().set_index("id").loc[v.dest_id]
        fig = go.Figure()
        for k, (day, g) in enumerate(plan.groupby("day")):
            fig.add_trace(go.Scattermap(lat=[d.lat] + g.lat.tolist() + [d.lat], lon=[d.lon] + g.lon.tolist() + [d.lon],
                                        mode="lines+markers", name=f"День {day}",
                                        line=dict(width=3, color=SERIES[k % 8]), marker=dict(size=10),
                                        text=["Отель"] + g.name.tolist() + ["Отель"], hoverinfo="text"))
        lats, lons = list(plan.lat) + [d.lat], list(plan.lon) + [d.lon]
        span = max(max(lons) - min(lons), (max(lats) - min(lats)) * 1.6, 0.01)
        zoom = float(np.clip(np.log2(360 / span) - 1.3, 3, 14))
        fig.update_layout(map=dict(style="open-street-map", zoom=zoom,
                                   center=dict(lat=(max(lats) + min(lats)) / 2, lon=(max(lons) + min(lons)) / 2)),
                          margin=dict(l=0, r=0, t=0, b=0), height=430, legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Маршрут по дням**")
    show = plan[["day", "order", "arrival", "name", "category", "duration_h", "ticket_rub", "transfer_rub"]].rename(columns={
        "day": "День", "order": "№", "arrival": "Прибытие", "name": "Объект", "category": "Категория",
        "duration_h": "Длит., ч", "ticket_rub": "Билет, ₽", "transfer_rub": "Трансфер, ₽"})
    st.dataframe(show, hide_index=True, width="stretch")
    hrs = pd.DataFrame({"День": range(1, trip.days + 1), "Занято, ч": t["hours_used"], "Доступно, ч": t["hours_available"]})
    st.caption("Загрузка дней: " + " · ".join(f"д{r['День']}: {r['Занято, ч']:.1f}/{r['Доступно, ч']:.0f} ч" for _, r in hrs.iterrows()))
    export = {"сценарий": trip.label(), "варианты": [{
        "вариант": x.kind, "направление": x.name, "стоимость": x.route.totals["total"], "класс": x.route.hotel_class,
        "стратегия": x.strategy, "обоснование": x.rationale, "предупреждения": x.warnings,
        "маршрут": x.route.plan[["day", "arrival", "name"]].to_dict("records")} for x in res.variants]}
    st.download_button("Скачать варианты (JSON)", json.dumps(export, ensure_ascii=False, indent=2, default=float),
                       file_name="travel_variants.json", mime="application/json")

# ------------------------------------------------------------------ 2. Рейтинг
with tabs[1]:
    t = res.table.copy()
    st.markdown(f"Интегральная оценка = {pipeline.W_AHP:.1f} · АНР (норм.) + {pipeline.W_REC:.1f} · рекомендации (норм.). "
                "Недопустимые по базе знаний направления не участвуют в выборе вариантов.")
    view = t[["rank", "name", "ahp", "rec", "integral", "cost", "status", "reasons"]].rename(columns={
        "rank": "Место", "name": "Направление", "ahp": "АНР", "rec": "Рекоменд.", "integral": "Интегр.",
        "cost": "Стоимость, ₽", "status": "Статус", "reasons": "Причина"})
    st.dataframe(view.style.format({"АНР": "{:.4f}", "Рекоменд.": "{:.3f}", "Интегр.": "{:.3f}", "Стоимость, ₽": "{:,.0f}"}),
                 hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        w = res.ahp.criteria
        st.markdown(f"**Веса критериев** (λmax = {w.lambda_max:.3f}, ИС = {w.ci:.4f}, ОС = {w.cr:.4f})")
        st.bar_chart(pd.DataFrame({"вес": w.weights}, index=[f"{c} {ahp.CRIT_NAMES[c]}" for c in ahp.CRIT_IDS]),
                     horizontal=True, color=SERIES[0])
    with c2:
        st.markdown("**Локальные приоритеты альтернатив по критериям**")
        loc = res.ahp.local.copy()
        loc.index = [res.ahp.indicators.set_index("dest_id").loc[i, "name"] for i in loc.index]
        st.dataframe(loc.style.format("{:.3f}").background_gradient(cmap="Blues"), width="stretch")
    st.markdown("**Почему рекомендательный модуль предлагает эти направления**")
    for _, r in res.rec.table.head(5).iterrows():
        st.markdown(f"- **{r['name']}** - {r.explanation}")

# ------------------------------------------------------------------ 3. Риски
with tabs[2]:
    st.markdown("Решения: направление × стратегия бронирования. Состояния природы: цена к моменту поздней покупки "
                "(3 сценария по волатильности истории цен), срыв поездки, погода. Выигрыш = ценность поездки − затраты, ₽.")
    alpha = st.slider("Коэффициент пессимизма α (критерий Гурвица)", 0.0, 1.0, 0.6, 0.05)
    crit = uncertainty.classic_criteria(res.payoff, res.prob, alpha)
    best = uncertainty.best_by_criteria(crit)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("**Лучшее решение по каждому критерию**")
        st.dataframe(best.style.format({"Значение": "{:,.0f}"}), hide_index=True, width="stretch")
        st.markdown(f"**Дерево решений** (свёртка): {' → '.join(res.tree_path)}, ожидаемый выигрыш {rub(res.tree_ev)}")
        thr = pd.DataFrame(res.thresholds).T.rename(columns={
            "q_S1_S2": "Порог срыва: возвратный выгоднее", "q_S1_S3": "Порог срыва: отложить выгоднее",
            "premium_S1_S3": "Порог наценки поздней покупки"})
        thr.index = [load_destinations().set_index("id").loc[i, "name"] for i in thr.index]
        st.dataframe(thr.style.format("{:.1%}"), width="stretch")
    with c2:
        st.markdown("**Значения критериев**")
        st.dataframe(crit.style.format("{:,.0f}").highlight_max(axis=0, color="#cde2fb"), width="stretch")
        st.markdown("**Монте-Карло: полная стоимость с тратами на месте (20 000 испытаний)**")
        st.dataframe(res.mc.rename(columns={"name": "Направление", "mean": "Среднее", "VaR95": "VaR 95%",
                                            "CVaR95": "CVaR 95%"}).drop(columns="dest_id")
                     .style.format({"Среднее": "{:,.0f}", "VaR 95%": "{:,.0f}", "CVaR 95%": "{:,.0f}",
                                    "P(превышение бюджета)": "{:.1%}"}), hide_index=True, width="stretch")
    with st.expander("Матрица выигрышей (решения × состояния природы)"):
        st.dataframe(res.payoff.style.format("{:,.0f}"), width="stretch")

# ------------------------------------------------------------------ 4. Прогноз
with tabs[3]:
    fc = load_forecast()
    wide = series_matrix()
    names = load_destinations().set_index("id")["name"]
    sel = st.multiselect("Направления", list(names.index), default=[v.dest_id for v in res.variants],
                         format_func=lambda i: names[i])
    fig = go.Figure()
    for k, d in enumerate(sel):
        c = SERIES[k % 8]
        f = fc[fc.dest_id == d]
        fig.add_trace(go.Scatter(x=wide.index, y=wide[d], name=f"{names[d]} - история", line=dict(color=c, width=1.5),
                                 opacity=0.55))
        fig.add_trace(go.Scatter(x=pd.concat([f.month, f.month[::-1]]), y=pd.concat([f.hi80, f.lo80[::-1]]),
                                 fill="toself", fillcolor=c, opacity=0.15, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f.month, y=f.forecast, name=f"{names[d]} - прогноз", line=dict(color=c, width=2.5)))
    fig.add_vline(x=pd.Timestamp(year=trip.year, month=trip.month, day=1), line_dash="dot", line_color="#898781")
    fig.update_layout(height=430, hovermode="x unified", yaxis_title="Стоимость 7-дневной поездки, ₽",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    met = pd.read_csv(ROOT / "reports" / "p6_metrics_summary.csv", index_col=0)
    st.markdown("**Качество моделей на отложенных 12 месяцах** (лучшая по MAPE используется для прогноза)")
    st.dataframe(met.style.format("{:.2f}"), width="stretch")

# ------------------------------------------------------------------ 5. База знаний
with tabs[4]:
    of = ontology_facts().copy()
    of["comfortable_months"] = of.comfortable_months.map(lambda m: ", ".join(MONTHS_RU[i - 1][:3] for i in m) or "-")
    of["beach_season_months"] = of.beach_season_months.map(lambda m: ", ".join(MONTHS_RU[i - 1][:3] for i in m) or "-")
    st.markdown("**Факты, выведенные ризонером Pellet из онтологии и SWRL-правил** (файл kb/travel_dss.owl открывается в Protégé)")
    st.dataframe(of.drop(columns=["hot_months"]).rename(columns={
        "dest_id": "id", "name": "Направление", "classes": "Выведенные классы", "comfortable_months": "Комфортные месяцы",
        "beach_season_months": "Купальный сезон", "needs_visa": "Виза", "needs_cash": "Нужна наличная валюта"}),
        hide_index=True, width="stretch")
    st.markdown("**Продукционные правила: журнал прямого вывода для текущей поездки**")
    for d, inf in res.kb.items():
        name = load_destinations().set_index("id").loc[d, "name"]
        status = inf.facts["status"]
        with st.expander(f"{name} - {status}" + (f" ({'; '.join(inf.facts['reasons'])})" if inf.facts["reasons"] else "")):
            st.dataframe(pd.DataFrame(inf.fired, columns=["Правило", "Название", "Вывод"]), hide_index=True,
                         width="stretch")

# ------------------------------------------------------------------ 6. О системе
with tabs[5]:
    st.markdown(f"""
**Архитектура СППР** - конвейер из семи модулей поверх общего слоя данных:

1. **Данные** (`data/`): направления, климат, 100 объектов показа, история цен 2019–2026, оценки 600 пользователей.
2. **П6 Прогноз** (`tdss/forecast.py`): Хольт–Винтерс/SARIMA/сезонный наивный → ожидаемая стоимость в месяц поездки.
3. **П5 База знаний** (`tdss/knowledge.py`): онтология OWL + {len(SWRL_RULES)} SWRL-правил (Pellet) и {len(RULES)} продукционных правил → допустимость, риски, предупреждения.
4. **П2 АНР** (`tdss/ahp.py`): 6 критериев, матрица ЛПР, локальные приоритеты по данным → глобальные оценки.
5. **П7 Рекомендации** (`tdss/recommender.py`): контент + kNN по профилям + популярность → персональное соответствие и объяснения.
6. **П3 Риски** (`tdss/uncertainty.py`): дерево решений (networkx), критерии Байеса/Вальда/Гурвица/Сэвиджа, пороги (scipy), Монте-Карло.
7. **П4 Оптимизация** (`tdss/route_opt.py`): MILP (PuLP) - маршрут по дням и класс проживания при ограничениях времени и бюджета.
8. **П8 Интеграция** (`tdss/pipeline.py`) и этот интерфейс; BI-дашборд - Apache Superset (`superset/`).

**Ограничения прототипа:** цены и оценки пользователей синтетические; климат - средние нормы, а не прогноз погоды;
стоимость рассчитывается на одного человека; время в пути - по прямой с коэффициентом извилистости.
""")
