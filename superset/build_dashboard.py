"""Сборка дашборда СППР в Apache Superset через REST API и выгрузка в архив для импорта.

Использовался для создания superset/dashboard_export.zip. Можно запустить против любого Superset 4.x:
    python superset/build_dashboard.py --url http://localhost:8088 --user admin --password admin \
        --db-uri sqlite:////app/tdss_data/tdss.db --export superset/dashboard_export.zip
"""
from __future__ import annotations

import argparse
import io
import json
import re
import uuid
import zipfile

import requests

VERBOSE = {
    "rank": "Место", "name": "Название", "ahp": "АНР", "rec": "Рекомендации", "integral": "Интегральная оценка",
    "cost": "Стоимость, ₽", "status": "Статус", "reasons": "Причина", "destination": "Направление", "day": "День",
    "order": "№", "arrival": "Прибытие", "category": "Категория", "duration_h": "Длительность, ч",
    "ticket_rub": "Билет, ₽", "model": "Модель", "month_name": "Месяц", "series": "Ряд", "region": "Регион",
    "month": "Месяц", "value": "Стоимость, ₽", "variant": "Вариант",
}


class Superset:
    def __init__(self, url, user, password):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        r = self.s.post(f"{self.url}/api/v1/security/login",
                        json={"username": user, "password": password, "provider": "db", "refresh": True})
        r.raise_for_status()
        self.s.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        csrf = self.s.get(f"{self.url}/api/v1/security/csrf_token/")
        if csrf.ok:
            self.s.headers["X-CSRFToken"] = csrf.json()["result"]
            self.s.headers["Referer"] = self.url

    def req(self, method, path, **kw):
        r = self.s.request(method, f"{self.url}/api/v1/{path}", **kw)
        if not r.ok:
            raise RuntimeError(f"{method} {path}: {r.status_code} {r.text[:500]}")
        return r


def metric(expr, label):
    return {"expressionType": "SQL", "sqlExpression": expr, "label": label}


def sql_filter(clause):
    return {"expressionType": "SQL", "clause": "WHERE", "sqlExpression": clause}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8088")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--db-uri", required=True)
    ap.add_argument("--export", default="superset/dashboard_export.zip")
    a = ap.parse_args()
    ss = Superset(a.url, a.user, a.password)

    db_id = ss.req("POST", "database/", json={"database_name": "СППР путешествий (SQLite)",
                                              "sqlalchemy_uri": a.db_uri, "expose_in_sqllab": True}).json()["id"]
    ds = {}
    for table, dttm in [("price_series", "month"), ("dss_ranking", None), ("season_index", None),
                        ("model_metrics", None), ("route_plan", None), ("dim_destination", None)]:
        did = ss.req("POST", "dataset/", json={"database": db_id, "table_name": table}).json()["id"]
        cols = ss.req("GET", f"dataset/{did}").json()["result"]["columns"]
        upd = [{"id": c["id"], "column_name": c["column_name"], "is_dttm": c["column_name"] == dttm,
                "type": c.get("type"), "verbose_name": VERBOSE.get(c["column_name"]),
                "python_date_format": "%Y-%m-%d" if c["column_name"] == dttm else None} for c in cols]
        body = {"columns": upd, **({"main_dttm_col": dttm} if dttm else {})}
        ss.req("PUT", f"dataset/{did}", json=body)
        ds[table] = did

    def chart(name, table, viz, params):
        p = {"datasource": f"{ds[table]}__table", "viz_type": viz, **params}
        return ss.req("POST", "chart/", json={"slice_name": name, "viz_type": viz, "datasource_id": ds[table],
                                              "datasource_type": "table", "params": json.dumps(p)}).json()["id"]

    c = {}
    c["kpi_feasible"] = chart("Допустимых направлений", "dss_ranking", "big_number_total", {
        "metric": metric("SUM(CASE WHEN status = 'допустимо' THEN 1 ELSE 0 END)", "Допустимо"),
        "subheader": "из 10 при бюджете 70 тыс. ₽", "y_axis_format": "SMART_NUMBER"})
    c["kpi_best"] = chart("Минимальная стоимость среди допустимых, ₽", "dss_ranking", "big_number_total", {
        "metric": metric("MIN(CASE WHEN status = 'допустимо' THEN cost END)", "Мин. стоимость"),
        "subheader": "перелёт + проживание, 5 дней, май 2027", "y_axis_format": ",d"})
    c["kpi_mape"] = chart("Точность прогноза (MAPE лучшей модели), %", "model_metrics", "big_number_total", {
        "metric": metric("AVG(MAPE)", "MAPE"), "adhoc_filters": [sql_filter("model = 'Хольт–Винтерс (ETS)'")],
        "subheader": "Хольт–Винтерс, отложенные 12 мес.", "y_axis_format": ".2f"})
    c["line"] = chart("Стоимость 7-дневной поездки: история и прогноз (Хольт–Винтерс), ₽", "price_series", "echarts_timeseries_line", {
        "x_axis": "month", "time_grain_sqla": "P1M", "metrics": [metric("MAX(value)", "Стоимость")],
        "groupby": ["name", "series"], "adhoc_filters": [sql_filter("month >= '2021-01-01'")],
        "row_limit": 10000, "truncate_metric": True,
        "show_legend": True, "legendOrientation": "top", "rich_tooltip": True, "y_axis_format": ",d",
        "x_axis_time_format": "%m.%Y", "markerEnabled": False, "zoomable": True, "color_scheme": "supersetColors"})
    c["may"] = chart("Прогноз стоимости на май 2027, ₽", "price_series", "echarts_timeseries_bar", {
        "x_axis": "name", "metrics": [metric("MAX(value)", "Прогноз")], "groupby": [],
        "adhoc_filters": [sql_filter("series = 'прогноз' AND month = '2027-05-01'")],
        "orientation": "horizontal", "x_axis_sort": "Прогноз", "x_axis_sort_asc": True, "show_value": True,
        "y_axis_format": ",d", "show_legend": False, "row_limit": 100, "color_scheme": "supersetColors"})
    c["rank"] = chart("Рейтинг направлений СППР (май 2027, 5 дней, 70 тыс. ₽)", "dss_ranking", "table", {
        "query_mode": "raw", "all_columns": ["rank", "name", "ahp", "rec", "integral", "cost", "status", "reasons"],
        "order_by_cols": [json.dumps(["rank", True])], "row_limit": 100, "table_timestamp_format": "%Y-%m-%d",
        "show_cell_bars": True, "column_config": {"ahp": {"d3NumberFormat": ".4f"}, "rec": {"d3NumberFormat": ".3f"},
                                                   "integral": {"d3NumberFormat": ".3f"}, "cost": {"d3NumberFormat": ",d"}}})
    c["season"] = chart("Сезонность цен: индекс месяца (1 = среднегодовой уровень)", "season_index", "heatmap_v2", {
        "x_axis": "month_name", "groupby": "name", "metric": metric("MAX(season_index)", "Индекс"),
        "linear_color_scheme": "blue_white_yellow", "normalize_across": "heatmap", "show_values": True,
        "value_bounds": [None, None], "y_axis_format": ".2f", "x_axis_sort": "alpha_asc", "row_limit": 1000,
        "sort_x_axis": "alpha_asc", "sort_y_axis": "alpha_asc", "legend_type": "continuous", "show_legend": True})
    c["models"] = chart("Качество моделей прогноза по направлениям (MAPE, %)", "model_metrics", "table", {
        "query_mode": "aggregate", "groupby": ["name", "model"],
        "metrics": [metric("AVG(MAPE)", "MAPE, %"), metric("AVG(MAE)", "MAE, ₽"), metric("AVG(coverage80)", "Покрытие 80% ПИ, %")],
        "row_limit": 100, "order_desc": False, "timeseries_limit_metric": metric("MIN(name)", "sort"),
        "column_config": {"MAPE, %": {"d3NumberFormat": ".2f"}, "MAE, ₽": {"d3NumberFormat": ",d"},
                          "Покрытие 80% ПИ, %": {"d3NumberFormat": ".0f"}}})
    c["route"] = chart("Маршрут варианта «Оптимальный»", "route_plan", "table", {
        "query_mode": "raw", "all_columns": ["destination", "day", "order", "arrival", "name", "category", "duration_h", "ticket_rub"],
        "adhoc_filters": [sql_filter("variant = 'Оптимальный'")],
        "order_by_cols": [json.dumps(["day", True]), json.dumps(["order", True])], "row_limit": 100})

    # uuid графиков нужен в разметке: по нему экспорт/импорт сопоставляет графики с ячейками сетки
    ids = ",".join(str(v) for v in c.values())
    zbytes = ss.req("GET", f"chart/export/?q=!({ids})").content
    uuids = {}
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        for fn in z.namelist():
            m = re.search(r"/charts/.*?(\d+)\.yaml$", fn)
            if m:
                u = re.search(r"^uuid: (\S+)", z.read(fn).decode(), re.M)
                uuids[int(m.group(1))] = u.group(1)

    def node(cid, w, h=50):
        return {"type": "CHART", "id": f"CHART-{cid}", "children": [],
                "meta": {"chartId": cid, "width": w, "height": h, "uuid": uuids[cid]}}

    rows = [[(c["kpi_feasible"], 4, 26), (c["kpi_best"], 4, 26), (c["kpi_mape"], 4, 26)],
            [(c["line"], 8, 60), (c["may"], 4, 60)],
            [(c["rank"], 12, 50)],
            [(c["season"], 7, 62), (c["models"], 5, 62)],
            [(c["route"], 12, 44)]]
    pos = {"DASHBOARD_VERSION_KEY": "v2", "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
           "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]},
           "HEADER_ID": {"id": "HEADER_ID", "type": "HEADER", "meta": {"text": "СППР: выбор места для путешествия"}}}
    for k, row in enumerate(rows):
        rid = f"ROW-{k}"
        pos[rid] = {"type": "ROW", "id": rid, "children": [], "parents": ["ROOT_ID", "GRID_ID"],
                    "meta": {"background": "BACKGROUND_TRANSPARENT"}}
        pos["GRID_ID"]["children"].append(rid)
        for cid, w, h in row:
            n = node(cid, w, h)
            n["parents"] = ["ROOT_ID", "GRID_ID", rid]
            pos[n["id"]] = n
            pos[rid]["children"].append(n["id"])

    in_scope = [c["line"], c["season"], c["models"]]
    excluded = [v for v in c.values() if v not in in_scope]
    filt = [{
        "id": f"NATIVE_FILTER-{uuid.uuid4().hex[:10]}", "name": "Направление", "filterType": "filter_select",
        "type": "NATIVE_FILTER", "description": "", "cascadeParentIds": [],
        "targets": [{"datasetId": ds["price_series"], "column": {"name": "name"}}],
        "controlValues": {"enableEmptyFilter": False, "defaultToFirstItem": False, "multiSelect": True,
                          "searchAllOptions": False, "inverseSelection": False},
        "defaultDataMask": {"extraFormData": {"filters": [{"col": "name", "op": "IN",
                                                           "val": ["Санкт-Петербург", "Казань", "Калининград"]}]},
                            "filterState": {"value": ["Санкт-Петербург", "Казань", "Калининград"]}, "ownState": {}},
        "scope": {"rootPath": ["ROOT_ID"], "excluded": excluded}, "chartsInScope": in_scope, "tabsInScope": []},
        {
        "id": f"NATIVE_FILTER-{uuid.uuid4().hex[:10]}", "name": "Регион", "filterType": "filter_select",
        "type": "NATIVE_FILTER", "description": "", "cascadeParentIds": [],
        "targets": [{"datasetId": ds["price_series"], "column": {"name": "region"}}],
        "controlValues": {"enableEmptyFilter": False, "defaultToFirstItem": False, "multiSelect": True,
                          "searchAllOptions": False, "inverseSelection": False},
        "defaultDataMask": {"extraFormData": {}, "filterState": {}, "ownState": {}},
        "scope": {"rootPath": ["ROOT_ID"], "excluded": [x for x in c.values() if x not in (c["line"], c["may"])]},
        "chartsInScope": [c["line"], c["may"]], "tabsInScope": []}]
    meta = {"native_filter_configuration": filt, "color_scheme": "supersetColors", "refresh_frequency": 0,
            "timed_refresh_immune_slices": [], "expanded_slices": {}, "label_colors": {}, "shared_label_colors": {},
            "cross_filters_enabled": True}
    dash = ss.req("POST", "dashboard/", json={"dashboard_title": "СППР: выбор места для путешествия",
                                               "slug": "travel-dss", "published": True,
                                               "position_json": json.dumps(pos), "json_metadata": json.dumps(meta)}).json()["id"]
    for cid in c.values():
        ss.req("PUT", f"chart/{cid}", json={"dashboards": [dash]})
    z = ss.req("GET", f"dashboard/export/?q=!({dash})")
    with open(a.export, "wb") as f:
        f.write(z.content)
    print(json.dumps({"database": db_id, "datasets": ds, "charts": c, "dashboard": dash, "export": a.export},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
