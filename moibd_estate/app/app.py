import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from estate_utils import extract_features, RATES

st.set_page_config(
    page_title="Аналитика и прогнозирование стоимости недвижимости",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_pipeline():
    model_path = os.path.join(PROJECT_DIR, 'data', 'best_model.pkl')
    if not os.path.exists(model_path):
        st.error(f"Файл модели не найден по пути: {model_path}")
        return None
    return joblib.load(model_path)

@st.cache_data
def load_meta_and_dash():
    meta_path = os.path.join(PROJECT_DIR, 'data', 'model_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    dash_csv = os.path.join(PROJECT_DIR, 'data', 'df_dashboard_interactive.csv')
    df_dash = pd.DataFrame()
    if os.path.exists(dash_csv):
        df_dash = pd.read_csv(dash_csv, low_memory=False)

    comp_csv = os.path.join(PROJECT_DIR, 'solo_assignment', 'solo_vs_group_comparison.csv')
    df_comp = pd.DataFrame()
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)

    return meta, df_dash, df_comp

model = load_pipeline()
meta_info, df_dash, df_comp = load_meta_and_dash()

best_mae = meta_info.get('best_test_metrics', {}).get('MAE', 246803.61)
best_model_name = "Случайный лес (Ансамбль бэггинга)"

with st.sidebar:
    st.title("Оценка объектов")
    st.caption("Интеллектуальный сервис прогнозирования жилья")
    st.divider()

    st.subheader("Активная модель")
    st.info(f"**{best_model_name}**\n\nСредняя погрешность: ~{best_mae:,.0f} TRY\nУровень согласованности: Высокий")

    st.divider()
    st.subheader("Курсы валют к TRY")
    st.write(f"USD: ~{RATES.get('USD', 33.0)} TRY")
    st.write(f"EUR: ~{RATES.get('EUR', 36.0)} TRY")
    st.write(f"GBP: ~{RATES.get('GBP', 42.0)} TRY")

tab_calc, tab_dash, tab_help = st.tabs([
    "Расчёт стоимости",
    "Аналитический дашборд",
    "Справка и документация"
])

with tab_calc:
    st.title("Калькулятор рыночной стоимости")
    st.markdown("Заполните характеристики объекта недвижимости для получения прогноза рыночной стоимости.")

    SUB_TYPES = [
        "Daire", "Villa", "Müstakil Ev", "Yazlık", "Rezidans",
        "Çiftlik Evi", "Köşk / Konak / Yalı", "Prefabrik Ev", "Kooperatif", "Loft"
    ]
    HEATING_TYPES = [
        "Kombi (Doğalgaz)", "Merkezi (Doğalgaz)", "Klima", "Yerden Isıtma",
        "Kalorifer (Doğalgaz)", "Yok", "Soba (Kömür)", "Fancoil", "Güneş Enerjisi"
    ]
    CITIES = [
        "Istanbul", "Ankara", "Izmir", "Antalya", "Bursa",
        "Mugla", "Aydin", "Tekirdag", "Balikesir", "Kocaeli", "Mersin"
    ]

    with st.form("calc_form"):
        col1, col2 = st.columns(2)

        with col1:
            listing_label = st.radio("Назначение объявления", ["Продажа жилья", "Долгосрочная аренда"], horizontal=True)
            listing_type_val = 1 if listing_label == "Продажа жилья" else 2

            sub_type_val = st.selectbox("Тип объекта недвижимости", SUB_TYPES, index=0)
            city_val = st.selectbox("Город / Провинция", CITIES, index=0)
            district_val = st.text_input("Район / Муниципалитет", value="Kadikoy")

            size_val = st.number_input("Общая площадь (м²)", min_value=10.0, max_value=2500.0, value=95.0, step=5.0)
            room_options = ["1+0", "1+1", "2+1", "3+1", "3+2", "4+1", "4+2", "5+1", "6+1"]
            room_count_val = st.selectbox("Количество комнат (жилые + гостиные)", room_options, index=2)

        with col2:
            building_age_val = st.number_input("Возраст здания (лет)", min_value=0.0, max_value=80.0, value=5.0, step=1.0)
            floor_no_val = st.number_input("Этаж расположения квартиры", min_value=-3.0, max_value=60.0, value=3.0, step=1.0)
            total_floor_val = st.number_input("Общая этажность дома", min_value=1.0, max_value=60.0, value=8.0, step=1.0)

            heating_type_val = st.selectbox("Тип отопительной системы", HEATING_TYPES, index=0)
            furnished_val = st.checkbox("Объект полностью меблирован", value=False)
            tom_val = st.slider("Срок размещения на рынке (дней)", min_value=1, max_value=365, value=25)

        submit_btn = st.form_submit_button("Рассчитать ориентировочную стоимость", use_container_width=True)

    if submit_btn:
        if model is None:
            st.error("Модель не инициализирована. Проверьте наличие best_model.pkl.")
        else:
            input_dict = {
                'listing_type': [listing_type_val],
                'sub_type': [sub_type_val],
                'city': [city_val],
                'district': [district_val],
                'address': [f"{city_val}/{district_val}"],
                'size': [size_val],
                'room_count': [room_count_val],
                'building_age': [building_age_val],
                'floor_no': [floor_no_val],
                'total_floor_count': [total_floor_val],
                'heating_type': [heating_type_val],
                'furnished': ['Evet' if furnished_val else 'Hayır'],
                'tom': [tom_val]
            }

            df_single = pd.DataFrame(input_dict)
            df_feat_single = extract_features(df_single)

            try:
                raw_pred = float(model.predict(df_feat_single)[0])
                pred_val = max(0.0, raw_pred)

                usd_rate = RATES.get('USD', 33.0)
                eur_rate = RATES.get('EUR', 36.0)

                pred_usd = pred_val / usd_rate
                pred_eur = pred_val / eur_rate

                lower_bound = max(0.0, pred_val - best_mae)
                upper_bound = pred_val + best_mae

                st.success("Расчёт успешно выполнен")

                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Ориентировочная цена (TRY)", f"{pred_val:,.0f} ₺")
                res_col2.metric("Эквивалент (USD)", f"${pred_usd:,.0f}")
                res_col3.metric("Эквивалент (EUR)", f"€{pred_eur:,.0f}")

                st.info(f"**Диапазон ориентировочной рыночной стоимости (с учетом средней погрешности):**\n\n"
                        f"от **{lower_bound:,.0f} TRY** до **{upper_bound:,.0f} TRY**")

                unit_price = pred_val / size_val if size_val > 0 else 0
                st.caption(f"Расчётная стоимость квадратного метра: **{unit_price:,.0f} TRY/м²**")

            except Exception as e:
                st.error(f"Ошибка при вычислении прогноза: {e}")

with tab_dash:
    st.title("Аналитический дашборд рынка недвижимости")
    st.markdown("Интерактивный анализ ключевых закономерностей датасета и результатов тестирования моделей.")

    if df_dash.empty:
        st.warning("Файл df_dashboard_interactive.csv не обнаружен. Отображаются сводные данные.")
    else:
        st.subheader("Фильтры исследовательской выборки")
        fcol1, fcol2, fcol3 = st.columns(3)

        all_cities = sorted([str(c) for c in df_dash['city'].dropna().unique()])
        all_subtypes = sorted([str(s) for s in df_dash['sub_type'].dropna().unique()])

        with fcol1:
            sel_cities = st.multiselect("Город", all_cities, default=all_cities[:4] if len(all_cities) >= 4 else all_cities)
        with fcol2:
            sel_subtypes = st.multiselect("Тип недвижимости", all_subtypes, default=all_subtypes[:3] if len(all_subtypes) >= 3 else all_subtypes)
        with fcol3:
            max_price_slider = st.slider("Максимальная цена (TRY)", min_value=100_000, max_value=20_000_000, value=8_000_000, step=200_000)

        df_filtered = df_dash.copy()
        if sel_cities:
            df_filtered = df_filtered[df_filtered['city'].isin(sel_cities)]
        if sel_subtypes:
            df_filtered = df_filtered[df_filtered['sub_type'].isin(sel_subtypes)]
        if 'price_try' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['price_try'] <= max_price_slider]

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Объектов в выборке", f"{len(df_filtered):,}")
        median_pr = df_filtered['price_try'].median() if 'price_try' in df_filtered.columns and not df_filtered.empty else 0
        mcol2.metric("Медианная цена", f"{median_pr:,.0f} TRY")
        mean_sz = df_filtered['size'].mean() if 'size' in df_filtered.columns and not df_filtered.empty else 0
        mcol3.metric("Средняя площадь", f"{mean_sz:.1f} м²")
        top_c = df_filtered['city'].mode()[0] if 'city' in df_filtered.columns and not df_filtered.empty else "-"
        mcol4.metric("Основной регион", str(top_c))

        st.divider()

        gcol1, gcol2 = st.columns(2)

        with gcol1:
            st.subheader("Диаграмма 1. Распределение цен объектов")
            if 'price_try' in df_filtered.columns and not df_filtered.empty:
                fig1 = px.histogram(df_filtered, x='price_try', nbins=40, color_discrete_sequence=['#1E88E5'], labels={'price_try': 'Цена (TRY)', 'count': 'Количество объектов'})
                fig1.update_layout(bargap=0.08, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig1, use_container_width=True)

        with gcol2:
            st.subheader("Диаграмма 2. Зависимость стоимости от площади")
            if 'size' in df_filtered.columns and 'price_try' in df_filtered.columns and not df_filtered.empty:
                sample_scatter = df_filtered.sample(min(800, len(df_filtered)), random_state=42)
                fig2 = px.scatter(sample_scatter, x='size', y='price_try', color='sub_type', opacity=0.7, labels={'size': 'Площадь (м²)', 'price_try': 'Цена (TRY)', 'sub_type': 'Тип жилья'})
                fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        gcol3, gcol4 = st.columns(2)

        with gcol3:
            st.subheader("Диаграмма 3. Средняя цена квадратного метра по городам")
            if 'city' in df_filtered.columns and 'price_try' in df_filtered.columns and 'size' in df_filtered.columns and not df_filtered.empty:
                df_city_sqm = df_filtered[df_filtered['size'] > 5].groupby('city').apply(lambda g: (g['price_try'] / g['size']).median()).reset_index(name='sqm_price')
                df_city_sqm = df_city_sqm.sort_values(by='sqm_price', ascending=False).head(10)
                fig3 = px.bar(df_city_sqm, x='city', y='sqm_price', color='sqm_price', color_continuous_scale='Blues', labels={'city': 'Город', 'sqm_price': 'Медианная цена м² (TRY)'})
                fig3.update_layout(coloraxis_showscale=False, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig3, use_container_width=True)

        with gcol4:
            st.subheader("Диаграмма 4. Разброс цен по типам недвижимости")
            if 'sub_type' in df_filtered.columns and 'price_try' in df_filtered.columns and not df_filtered.empty:
                fig4 = px.box(df_filtered, x='sub_type', y='price_try', color='sub_type', labels={'sub_type': 'Тип недвижимости', 'price_try': 'Цена (TRY)'})
                fig4.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig4, use_container_width=True)

        gcol5, gcol6 = st.columns(2)

        with gcol5:
            st.subheader("Диаграмма 5. Зависимость стоимости от возраста здания")
            if 'building_age_num' in df_filtered.columns and 'price_try' in df_filtered.columns and not df_filtered.empty:
                df_age = df_filtered.dropna(subset=['building_age_num']).groupby('building_age_num')['price_try'].median().reset_index()
                df_age = df_age.sort_values(by='building_age_num')
                fig5 = px.line(df_age, x='building_age_num', y='price_try', markers=True, color_discrete_sequence=['#43A047'], labels={'building_age_num': 'Возраст здания (лет)', 'price_try': 'Медианная цена (TRY)'})
                fig5.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig5, use_container_width=True)

        with gcol6:
            st.subheader("Диаграмма 6. Распределение объектов по этажам")
            if 'floor_no_num' in df_filtered.columns and not df_filtered.empty:
                df_floor = df_filtered[(df_filtered['floor_no_num'] >= 0) & (df_filtered['floor_no_num'] <= 20)]
                fig6 = px.histogram(df_floor, x='floor_no_num', nbins=21, color_discrete_sequence=['#FB8C00'], labels={'floor_no_num': 'Этаж расположения', 'count': 'Количество объектов'})
                fig6.update_layout(bargap=0.1, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig6, use_container_width=True)

        gcol7, gcol8 = st.columns(2)

        with gcol7:
            st.subheader("Диаграмма 7. Взаимосвязь числовых характеристик")
            corr_cols = [c for c in ['price_try', 'size', 'building_age_num', 'floor_no_num', 'total_floor_count_num'] if c in df_filtered.columns]
            if len(corr_cols) >= 3 and not df_filtered.empty:
                corr_matrix = df_filtered[corr_cols].corr()
                fig7 = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
                fig7.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig7, use_container_width=True)

        with gcol8:
            st.subheader("Диаграмма 8. Сравнение средней погрешности моделей")
            models_data = [
                {"Модель": "Линейная (Ridge)", "Средняя погрешность (TRY)": 365785},
                {"Модель": "Дерево решений", "Средняя погрешность (TRY)": 263838},
                {"Модель": "Случайный лес (Лучшая)", "Средняя погрешность (TRY)": 246804}
            ]
            df_models = pd.DataFrame(models_data)
            fig8 = px.bar(df_models, x="Модель", y="Средняя погрешность (TRY)", color="Модель", color_discrete_sequence=['#E53935', '#FB8C00', '#43A047'])
            fig8.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig8, use_container_width=True)

    st.divider()
    st.subheader("Сравнительный анализ: Групповой проект vs Сольное задание")
    st.markdown("Сопоставление результатов генерального тестирования (групповой проект) с контрольной выборкой из 20 тестовых объектов (сольное задание):")

    if not df_comp.empty:
        st.dataframe(df_comp, use_container_width=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=df_comp['Model'],
            y=df_comp['Group_Test_MAE'],
            name='Групповой проект (Тест)',
            marker_color='#1E88E5'
        ))
        fig_comp.add_trace(go.Bar(
            x=df_comp['Model'],
            y=df_comp['Solo_Manual_MAE'],
            name='Сольное задание (20 объектов)',
            marker_color='#E53935'
        ))
        fig_comp.update_layout(
            barmode='group',
            title='Сравнение погрешностей моделей (Групповая выборка vs Сольные 20 объектов)',
            xaxis_title='Модель',
            yaxis_title='Средняя абсолютная погрешность (TRY)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("**Верификация математического расчета:** Значения погрешностей, вычисленные вручную по формулам, на 100% совпадают с библиотечными методами (разница равна строго 0.00).")

with tab_help:
    st.title("Справка и документация сервиса")

    st.subheader("1. Назначение сервиса")
    st.write(
        "Интеллектуальная система разработана для экспресс-оценки рыночной стоимости объектов жилой "
        "недвижимости на базе алгоритмов машинного обучения. Модель анализирует физические и географические параметры "
        "объекта и формирует оценку рыночной цены с доверительным диапазоном."
    )

    st.subheader("2. Руководство пользователя")
    st.write(
        "1. Перейдите во вкладку 'Расчёт стоимости'.\n"
        "2. Укажите параметры объекта: тип недвижимости, город, район, площадь, этажность и возраст дома.\n"
        "3. Нажмите кнопку 'Рассчитать ориентировочную стоимость'.\n"
        "4. Ознакомьтесь с точечным прогнозом цены и расчетным интервалом стоимости в трех валютах (TRY, USD, EUR).\n"
        "5. Для исследования структуры рынка перейдите во вкладку 'Аналитический дашборд'."
    )

    st.subheader("3. Описание входных параметров")
    st.write(
        "- **Площадь (м²)**: общая полезная площадь жилого помещения.\n"
        "- **Количество комнат**: суммарное число жилых комнат и гостиных (например, 2+1 = 3 комнаты).\n"
        "- **Возраст здания**: фактический возраст строения в годах.\n"
        "- **Этаж и этажность**: высотное положение квартиры и суммарная этажность дома.\n"
        "- **Тип отопления**: центральное, индивидуальный котел, кондиционирование или отсутствие отопления.\n"
        "- **Срок размещения на рынке**: количество дней нахождения объявления на портале."
    )

    st.subheader("4. Ограничения модели и рекомендации")
    st.write(
        "- Модель обучена на выборке объектов портала Zingat и наиболее точна для стандартных квартир в диапазоне площадей от 30 до 300 м².\n"
        "- Для уникальных исторических особняков и сверхдорогих вилл погрешность может возрастать.\n"
        "- Прогноз носит ориентировочный аналитический характер и не является официальным оценочным заключением."
    )
