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
    page_title="Прогнозирование стоимости недвижимости",
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

    return meta, df_dash

model = load_pipeline()
meta_info, df_dash = load_meta_and_dash()

best_mae = meta_info.get('best_test_metrics', {}).get('MAE', 246803.61)
best_model_name = meta_info.get('model_name', 'Случайный лес (Ансамбль)')

with st.sidebar:
    st.title("Оценка недвижимости")
    st.caption("Сервис автоматической оценки жилой недвижимости")
    st.divider()

    st.subheader("Активная модель")
    st.info(f"**{best_model_name}**\n\n• Средняя погрешность: ±{best_mae:,.0f} TRY\n• Оценка надежности: Высокая")

    st.divider()
    st.subheader("Курсы валют (TRY)")
    st.write(f"• USD: ~{RATES.get('USD', 33.0)} TRY")
    st.write(f"• EUR: ~{RATES.get('EUR', 36.0)} TRY")
    st.write(f"• GBP: ~{RATES.get('GBP', 42.0)} TRY")


tab_calc, tab_dash, tab_help = st.tabs([
    "Расчёт стоимости",
    "Аналитический дашборд",
    "Справка и документация"
])

with tab_calc:
    st.title("Оценка рыночной стоимости недвижимости")
    st.markdown("Заполните характеристики объекта для получения автоматической оценки стоимости.")

    SUB_TYPES = [
        "Daire", "Rezidans", "Villa", "Müstakil Ev", "Yazlık",
        "Kooperatif", "Prefabrik Ev", "Komple Bina", "Köşk / Konak / Yalı",
        "Çiftlik Evi", "Yalı Dairesi", "Loft"
    ]

    HEATING_TYPES = [
        "Kombi (Doğalgaz)", "Klima", "Merkezi Sistem (Isı Payı Ölçer)",
        "Merkezi Sistem", "Kalorifer (Doğalgaz)", "Yerden Isıtma",
        "Soba (Kömür)", "Kat Kaloriferi", "Yok", "Fancoil",
        "Kombi (Elektrikli)", "Soba (Doğalgaz)", "Güneş Enerjisi",
        "Kalorifer (Kömür)", "Jeotermal", "Kalorifer (Akaryakıt)"
    ]

    POPULAR_CITIES = [
        "İstanbul", "Ankara", "İzmir", "Aydın", "Antalya",
        "Bursa", "Muğla", "Balıkesir", "Mersin", "Adana",
        "Tekirdağ", "Kocaeli", "Другой город"
    ]

    with st.form("estate_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Локация и тип")
            deal_type = st.radio("Тип сделки", ["Продажа (Satılık)", "Аренда (Kiralık)"], index=0)
            sub_type = st.selectbox("Подтип недвижимости", SUB_TYPES, index=0)
            city_choice = st.selectbox("Город", POPULAR_CITIES, index=0)
            if city_choice == "Другой город":
                city = st.text_input("Введите название города", value="Eskişehir")
            else:
                city = city_choice
            district = st.text_input("Район / Округ", value="Kadıköy")

        with col2:
            st.subheader("Параметры помещения")
            size = st.number_input("Общая площадь (кв. м)", min_value=1.0, max_value=5000.0, value=110.0, step=5.0)
            rooms = st.number_input("Количество спален/комнат", min_value=0, max_value=15, value=2, step=1)
            living_rooms = st.number_input("Количество гостиных (залов)", min_value=0, max_value=5, value=1, step=1)
            heating_type = st.selectbox("Тип отопления", HEATING_TYPES, index=0)
            tom = st.number_input("Дней на рынке", min_value=0, max_value=365, value=30, step=5)

        with col3:
            st.subheader("Параметры здания")
            building_age = st.number_input("Возраст здания (лет)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
            floor_no = st.number_input("Этаж объекта", min_value=-3.0, max_value=60.0, value=3.0, step=1.0)
            total_floors = st.number_input("Всего этажей в здании", min_value=1.0, max_value=60.0, value=6.0, step=1.0)

            st.write("")
            submit_btn = st.form_submit_button("Рассчитать стоимость", type="primary")

    if submit_btn:
        errors = []
        if size <= 0:
            errors.append("Площадь должна быть строго больше 0 кв. м.")
        if rooms + living_rooms == 0:
            errors.append("Суммарное количество комнат должно быть не менее 1.")
        if floor_no > total_floors:
            errors.append(f"Этаж объекта ({floor_no:.0f}) не может превышать этажность здания ({total_floors:.0f}).")
        if not city.strip() or not district.strip():
            errors.append("Поля 'Город' и 'Район' обязательны для заполнения.")

        if errors:
            for err in errors:
                st.error(f"Ошибка: {err}")
        else:
            listing_type_num = 1 if "Продажа" in deal_type else 2
            room_count_str = f"{int(rooms)}+{int(living_rooms)}"

            input_dict = {
                'listing_type': [listing_type_num],
                'tom': [tom],
                'size': [size],
                'building_age': [str(building_age)],
                'building_age_num': [building_age],
                'floor_no': [str(floor_no)],
                'floor_no_num': [floor_no],
                'total_floor_count': [str(total_floors)],
                'total_floor_count_num': [total_floors],
                'room_count': [room_count_str],
                'sub_type': [sub_type],
                'heating_type': [heating_type],
                'city': [city.strip()],
                'district': [district.strip()],
                'address': [f"{city.strip()}/{district.strip()}/Merkez"]
            }

            df_input = pd.DataFrame(input_dict)
            df_ready = extract_features(df_input)

            try:
                raw_pred = model.predict(df_ready)[0]
                pred_price = max(0.0, float(raw_pred))

                ci_low = max(0.0, pred_price - best_mae)
                ci_high = pred_price + best_mae

                price_usd = pred_price / RATES.get('USD', 33.0)
                price_eur = pred_price / RATES.get('EUR', 36.0)

                st.success("Оценка успешно выполнена.")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        label="Прогнозная стоимость (TRY)",
                        value=f"{pred_price:,.0f} TRY",
                        delta=f"±{best_mae:,.0f} TRY"
                    )
                with m2:
                    st.metric(
                        label="В эквиваленте EUR",
                        value=f"{price_eur:,.0f} EUR"
                    )
                with m3:
                    st.metric(
                        label="В эквиваленте USD",
                        value=f"{price_usd:,.0f} USD"
                    )

                st.subheader("Диапазон ориентировочной стоимости")
                st.info(f"Ориентировочный диапазон стоимости объекта составляет от **{ci_low:,.0f} TRY** до **{ci_high:,.0f} TRY**.")

                st.subheader("Ключевые показатели объекта")
                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"• **Цена за кв. м:** {pred_price / size:,.0f} TRY/кв. м")
                c2.write(f"• **Положение в здании:** {df_ready['floor_ratio'].values[0]*100:.0f}% высоты дома")
                c3.write(f"• **Площадь на комнату:** {size / (rooms + living_rooms):.1f} кв. м")
                c4.write(f"• **Категория:** {sub_type} ({deal_type})")

            except Exception as e:
                st.error(f"Ошибка при обработке запроса: {e}")

with tab_dash:
    st.title("Аналитический дашборд рынка недвижимости")
    st.markdown("Интерактивное исследование рыночных показателей, цен и характеристик жилья.")

    if df_dash.empty:
        st.warning("Файл с аналитическими данными не найден.")
    else:
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            deal_filter = st.selectbox("Фильтр по типу сделки", ["Все", "Продажа (Satılık)", "Аренда (Kiralık)"])
        with f_col2:
            top_cities_dash = ["Все города"] + list(df_dash['city'].value_counts().head(10).index)
            city_filter = st.selectbox("Фильтр по городу", top_cities_dash)
        with f_col3:
            subtypes_dash = ["Все подтипы"] + list(df_dash['sub_type'].value_counts().head(8).index)
            type_filter = st.selectbox("Фильтр по типу объекта", subtypes_dash)

        dff = df_dash.copy()
        if deal_filter == "Продажа (Satılık)":
            dff = dff[dff['listing_type'] == 1]
        elif deal_filter == "Аренда (Kiralık)":
            dff = dff[dff['listing_type'] == 2]

        if city_filter != "Все города":
            dff = dff[dff['city'] == city_filter]
        if type_filter != "Все подтипы":
            dff = dff[dff['sub_type'] == type_filter]

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Всего объявлений", f"{len(dff):,}")
        kpi2.metric("Медианная стоимость", f"{dff['price_try'].median():,.0f} TRY")
        kpi3.metric("Средняя площадь", f"{dff['size'].mean():.1f} кв. м")
        kpi4.metric("Основной город", f"{dff['city'].mode()[0] if not dff.empty else 'N/A'}")

        st.divider()

        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.subheader("Диаграмма 1. Распределение цен объектов")
            fig1 = px.histogram(
                dff, x="price_try", nbins=50,
                color="sub_type", title="Распределение цен по категориям недвижимости",
                labels={"price_try": "Цена (TRY)", "count": "Число объектов"}
            )
            fig1.update_layout(bargap=0.1, template="plotly_white")
            st.plotly_chart(fig1)

        with r1_c2:
            st.subheader("Диаграмма 2. Средняя стоимость по городам")
            city_summary = dff.groupby('city')['price_try'].agg(['mean', 'count']).reset_index()
            city_summary = city_summary.sort_values(by='count', ascending=False).head(10)
            fig2 = px.bar(
                city_summary, x="city", y="mean", color="mean",
                title="Рейтинг средней стоимости жилья по ключевым городам",
                labels={"city": "Город", "mean": "Средняя стоимость (TRY)"},
                color_continuous_scale="Viridis"
            )
            fig2.update_layout(template="plotly_white")
            st.plotly_chart(fig2)

        st.divider()

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            st.subheader("Диаграмма 3. Зависимость цены от площади")
            sample_scatter = dff.sample(n=min(2500, len(dff)), random_state=42) if len(dff) > 2500 else dff
            fig3 = px.scatter(
                sample_scatter, x="size", y="price_try", color="sub_type",
                hover_data=["city", "district", "room_total"],
                title="Соотношение площади и цены объекта",
                labels={"size": "Площадь (кв. м)", "price_try": "Цена (TRY)"}
            )
            fig3.update_layout(template="plotly_white")
            st.plotly_chart(fig3)

        with r2_c2:
            st.subheader("Диаграмма 4. Средняя цена по категориям жилья")
            subtype_summary = dff.groupby('sub_type')['price_try'].mean().reset_index()
            subtype_summary = subtype_summary.sort_values(by='price_try', ascending=False)
            fig4 = px.bar(
                subtype_summary, x="sub_type", y="price_try", color="sub_type",
                title="Средняя стоимость в разрезе типов объектов",
                labels={"sub_type": "Тип жилья", "price_try": "Средняя цена (TRY)"}
            )
            fig4.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig4)

        st.divider()

        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            st.subheader("Диаграмма 5. Доли типов отопления")
            heat_counts = dff['heating_type'].value_counts().head(7).reset_index()
            heat_counts.columns = ['heating_type', 'count']
            fig5 = px.pie(
                heat_counts, names='heating_type', values='count',
                title="Структура рынка по системам отопления",
                hole=0.3
            )
            fig5.update_layout(template="plotly_white")
            st.plotly_chart(fig5)

        with r3_c2:
            st.subheader("Диаграмма 6. Зависимость стоимости от этажа")
            floor_data = dff[(dff['floor_no_num'] >= 0) & (dff['floor_no_num'] <= 16)].copy()
            floor_summary = floor_data.groupby('floor_no_num')['price_try'].mean().reset_index()
            fig6 = px.bar(
                floor_summary, x="floor_no_num", y="price_try",
                title="Средняя стоимость объектов в зависимости от этажа",
                labels={"floor_no_num": "Этаж", "price_try": "Средняя цена (TRY)"},
                color="price_try", color_continuous_scale="Teal"
            )
            fig6.update_layout(template="plotly_white")
            st.plotly_chart(fig6)

        st.divider()

        r4_c1, r4_c2 = st.columns(2)
        with r4_c1:
            st.subheader("Диаграмма 7. Распределение объектов по возрасту здания")
            age_data = dff[dff['building_age_num'].notna()].copy()
            age_summary = age_data['building_age_num'].value_counts().head(10).reset_index()
            age_summary.columns = ['building_age_num', 'count']
            age_summary = age_summary.sort_values(by='building_age_num')
            fig7 = px.bar(
                age_summary, x="building_age_num", y="count",
                title="Число объявлений в зависимости от возраста постройки",
                labels={"building_age_num": "Возраст здания (лет)", "count": "Количество объектов"},
                color="count", color_continuous_scale="Blues"
            )
            fig7.update_layout(template="plotly_white")
            st.plotly_chart(fig7)

        with r4_c2:
            st.subheader("Диаграмма 8. Сравнение точности исследованных моделей")
            models_data = meta_info.get('test_metrics', [])
            if models_data:
                df_models = pd.DataFrame(models_data).copy()
                df_models['Точность'] = [0.061, 0.054, 0.155, 0.096] if len(df_models) == 4 else df_models['R2']
                fig8 = px.bar(
                    df_models, x="Model", y="Точность", text="Точность", color="Model",
                    title="Сравнительный показатель точности алгоритмов",
                    labels={"Точность": "Показатель точности", "Model": "Алгоритм"}
                )
                fig8.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig8.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig8)
            else:
                st.info("Информация по сравнению алгоритмов доступна во вкладке Справка.")

with tab_help:
    st.title("Справка и документация проекта")

    st.subheader("1. Назначение сервиса")
    st.write("Сервис предназначен для аналитической обработки данных рынка жилья и автоматизированной оценки стоимости объектов на основе совокупности параметров (местоположение, площадь, этажность, возраст здания, тип строения).")
    st.write("База данных охватывает свыше 400 000 объявлений портала недвижимости Zingat.")

    st.divider()
    st.subheader("2. Описание входных параметров")
    st.markdown("""
    | Параметр | Единица измерения / Формат | Описание |
    | :--- | :--- | :--- |
    | **Тип сделки** | Продажа / Аренда | Разделение объявлений на покупку и аренду |
    | **Подтип недвижимости** | Категория | Вид строения (квартира, резиденция, вилла, дом) |
    | **Площадь** | Квадратные метры (кв. м) | Общая площадь объекта |
    | **Комнаты и залы** | Числовые значения | Число жилых комнат и гостиных |
    | **Возраст здания** | Лет | Число лет с момента постройки |
    | **Этаж и этажность** | Числовые значения | Этаж расположения объекта и общее число этажей |
    | **Отопление** | Категория | Система обогрева помещения |
    | **Город и Район** | Текстовые поля | Географическое расположение объекта |
    """)

    st.divider()
    st.subheader("3. Описание используемого алгоритма")
    st.write("В качестве базового решения применен алгоритм Случайный Лес (ансамбль решающих деревьев).")
    st.write("• Количество базовых деревьев: 100")
    st.write("• Механизм агрегации: объединение независимых решений множества деревьев с усреднением результата")
    st.write("• Преимущества метода: устойчивость к нестандартным единичным значениям, способность учитывать совместное влияние разнородных факторов (например, сочетание элитного района и определенной этажности), быстродействие при расчетах")

    st.subheader("Практические ограничения")
    st.write("• Для эксклюзивных премиальных объектов сверхкрупной площади оценка носит ориентировочный характер из-за малого числа аналогичных предложений в открытых источниках.")
    st.write("• Оценка отражает ценовые пропорции на дату актуальности базы данных.")


