"""Прогон всех практик паспорта СППР по порядку зависимостей; результаты - в reports/, kb/, superset/data/.

    python run_all.py
"""
import runpy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = [
    ("П6 Прогноз цен и витрины BI", "p6_forecast.py"),      # прогноз нужен остальным модулям
    ("П5 База знаний", "p5_knowledge.py"),                   # выводы Pellet нужны АНР и конвейеру
    ("П1 Постановка задачи", "p1_problem.py"),
    ("П2 АНР", "p2_ahp.py"),
    ("П3 Риски и неопределённость", "p3_uncertainty.py"),
    ("П4 Оптимизация маршрута", "p4_route.py"),
    ("П7 Рекомендательный модуль", "p7_recommender.py"),
    ("П8 Итоговая СППР", "p8_dss.py"),
]

if __name__ == "__main__":
    for title, script in ORDER:
        t0 = time.time()
        print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
        runpy.run_path(str(ROOT / "practices" / script), run_name="__main__")
        print(f"[{title}: {time.time() - t0:.1f} с]")
