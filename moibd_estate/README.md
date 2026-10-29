# Проект: Интеллектуальное прогнозирование стоимости недвижимости (MOIBD Estate)

**Студент:** Nezire  
**Группа:** МОИБД  
**Версия:** 2.0 Pro Release  

Данный репозиторий содержит законченное решение для двух взаимосвязанных задач:
1. **Групповой проект** («Прогнозирование стоимости недвижимости» по полному регламенту на 100 баллов):
   - Глубокий исследовательский анализ данных (EDA) с 6 типами графиков.
   - Продвинутая предобработка данных (Feature Engineering), устранение пропусков, кодирование и масштабирование без утечки данных.
   - Обучение и сравнительное тестирование 4 моделей разной природы:
     * **Ridge (Linear Model)**
     * **Decision Tree Regressor**
     * **Random Forest Regressor (Bagging)**
     * **LightGBM Regressor (Gradient Boosting)**
   - Подбор гиперпараметров, расчет метрик ($R^2$, MAE, RMSE, MAPE, Adjusted $R^2$), анализ остатков и важности признаков.
   - Полнофункциональное веб-приложение **Streamlit** с формой прогноза с доверительными интервалами (±MAE), встроенным интерактивным дашбордом **Plotly** и разделом справки.
2. **Сольное задание** (по файлу `test_real_estate_samples.csv`):
   - Проверка лучшей модели на контрольной выборке из 20 объектов.
   - Ручной математический расчет метрик MAE, RMSE, $R^2$ по формулам со всеми промежуточными слагаемыми.
   - Программный расчет через `scikit-learn` с доказательством 100% совпадения результатов.
   - Глубокий экономический и технический анализ ошибок и специфики объектов.
   - Итоговый файл `result_real_estate_samples.csv` с колонками `predicted_price`, `ФИО` и `Группа`.
   - Полный отчет в формате Word (`SOLO_REPORT_REAL_ESTATE.docx`) и Markdown (`SOLO_REPORT_REAL_ESTATE.md`).

---

## Структура проекта

```text
moibd_estate/
├── app/
│   ├── app.py                     # Веб-приложение Streamlit (Форма, Дашборд, Справка)
│   └── estate_utils.py            # Модуль парсинга, Feature Engineering и инференса
├── data/
│   ├── real_estate_data.csv       # Исходный датасет портала Zingat (~400k записей)
│   ├── df_dashboard_interactive.csv # Оптимизированный датасет для интерактивного дашборда
│   ├── best_model.pkl             # Сохраненная финальная модель (Random Forest Pipeline)
│   ├── model_meta.json            # Метаданные обучения и метрики всех моделей
│   └── zingat_data_info.docx      # Документация признаков исходного датасета
├── notebook/
│   └── main.ipynb                 # Полноценный Jupyter Notebook с 4 этапами проекта
├── reports_and_charts/            # Экспортированные визуализации высокого качества
│   ├── eda_1_price_distribution.png
│   ├── eda_2_size_vs_price.png
│   ├── eda_3_boxplot_subtypes.png
│   ├── eda_4_correlation_heatmap.png
│   ├── eda_5_bar_top_cities.png
│   ├── eda_6_missing_values.png
│   ├── model_comparison_metrics.png
│   ├── actual_vs_predicted_best.png
│   ├── residuals_plot_best.png
│   └── feature_importance_best.png
├── solo_assignment/               # Все материалы сольного задания
│   ├── result_real_estate_samples.csv     # Результирующий CSV с ФИО и группой
│   ├── manual_calculation_breakdown.csv   # Таблица пошаговых слагаемых
│   ├── solo_metrics_summary.json          # Итоги ручного и библиотечного расчета
│   ├── SOLO_REPORT_REAL_ESTATE.docx       # Отчет в формате Microsoft Word
│   └── SOLO_REPORT_REAL_ESTATE.md         # Отчет в формате Markdown
├── requirements.txt               # Список внешних зависимостей
├── run_app.bat                    # Батник запуска приложения в 1 клик для Windows
└── README.md                      # Данный файл описания
```

---

## Как запустить веб-приложение

1. **В один клик (Windows):**
   Дважды щелкните по файлу `run_app.bat` в папке `moibd_estate`.
2. **Через командную строку:**
   ```bash
   cd C:\Users\Nezire\Desktop\moibd_estate\app
   streamlit run app.py
   ```
   Приложение откроется в браузере по адресу `http://localhost:8501`.

---

## Сводные метрики моделей на тестовой выборке (80 135 объектов)

| Модель | $R^2$ | MAE (TRY) | RMSE (TRY) | MAPE (%) | Adjusted $R^2$ |
|---|---|---|---|---|---|
| **Ridge (Linear)** | 0.0610 | 365 785.66 | 2 789 285.72 | 2569.96 | 0.0608 |
| **Decision Tree** | 0.0535 | 263 838.85 | 2 800 406.10 | 246.35 | 0.0533 |
| **LightGBM (Boosting)** | 0.0964 | 274 901.41 | 2 736 240.61 | 1149.79 | 0.0962 |
| **Random Forest (Bagging) 🏆** | **0.1546** | **246 803.61** | **2 646 646.63** | **331.11** | **0.1544** |

---

## Сольное задание: Ручной vs Scikit-Learn расчет

На контрольной выборке из 20 объектов (`test_real_estate_samples.csv`):
- **MAE:** Ручной = `393 747.05 TRY`, Scikit-Learn = `393 747.05 TRY` (Расхождение: `0.00`)
- **RMSE:** Ручной = `577 294.57 TRY`, Scikit-Learn = `577 294.57 TRY` (Расхождение: `0.00`)
- **$R^2$:** Ручной = `0.1444`, Scikit-Learn = `0.1444` (Расхождение: `0.0000`)
