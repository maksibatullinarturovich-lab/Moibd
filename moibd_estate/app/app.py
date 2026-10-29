import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add app directory to sys.path for utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from estate_utils import extract_features, RATES

st.set_page_config(
    page_title="Real Estate Price Prediction | Прогнозирование недвижимости",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CACHED RESOURCE LOADERS
# ---------------------------------------------------------
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
best_r2 = meta_info.get('best_test_metrics', {}).get('R2', 0.1546)
best_model_name = meta_info.get('model_name', 'Random Forest (Bagging)')

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/real-estate.png", width=70)
    st.title("Estate AI Predictor")
    st.caption("Система оценки жилой недвижимости на базе ML")
    st.markdown("---")
    
    st.subheader("Текущая модель")
    st.info(f"**{best_model_name}**\n\n• Test MAE: ±{best_mae:,.0f} TRY\n• Test R²: {best_r2:.4f}")
    
    st.markdown("---")
    st.subheader("Курсы валют (TRY)")
    st.write(f"• USD: ~{RATES.get('USD', 33.0)} TRY")
    st.write(f"• EUR: ~{RATES.get('EUR', 36.0)} TRY")
    st.write(f"• GBP: ~{RATES.get('GBP', 42.0)} TRY")
    st.markdown("---")
    st.caption("Версия приложения: v2.0 (МОИБД)")

# ---------------------------------------------------------
# MAIN APP TABS
# ---------------------------------------------------------
tab_calc, tab_dash, tab_help = st.tabs([
    "🎯 Расчёт стоимости",
    "📊 Аналитический дашборд",
    "ℹ️ Справка и документация"
])

# =========================================================
# TAB 1: PREDICTION FORM
# =========================================================
with tab_calc:
    st.title("🏡 Оценка рыночной стоимости недвижимости")
    st.markdown("Заполните характеристики объекта для получения интеллектуальной оценки стоимости.")

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
            st.subheader("📍 Локация и тип")
            deal_type = st.radio("Тип сделки", ["Продажа (Satılık)", "Аренда (Kiralık)"], index=0)
            sub_type = st.selectbox("Подтип недвижимости", SUB_TYPES, index=0)
            city_choice = st.selectbox("Город", POPULAR_CITIES, index=0)
            if city_choice == "Другой город":
                city = st.text_input("Введите название города", value="Eskişehir")
            else:
                city = city_choice
            district = st.text_input("Район / Округ (County/District)", value="Kadıköy")
            
        with col2:
            st.subheader("📐 Характеристики помещения")
            size = st.number_input("Общая площадь (м²)", min_value=1.0, max_value=5000.0, value=110.0, step=5.0)
            rooms = st.number_input("Количество спален/комнат", min_value=0, max_value=15, value=2, step=1)
            living_rooms = st.number_input("Количество гостиных (залов)", min_value=0, max_value=5, value=1, step=1)
            heating_type = st.selectbox("Тип отопления", HEATING_TYPES, index=0)
            tom = st.number_input("Дней на рынке (Time on Market)", min_value=0, max_value=365, value=30, step=5)
            
        with col3:
            st.subheader("🏢 Параметры здания и этажность")
            building_age = st.number_input("Возраст здания (лет)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
            floor_no = st.number_input("Этаж объекта", min_value=-3.0, max_value=60.0, value=3.0, step=1.0)
            total_floors = st.number_input("Всего этажей в здании", min_value=1.0, max_value=60.0, value=6.0, step=1.0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🚀 Рассчитать стоимость", use_container_width=True, type="primary")

    if submit_btn:
        # Validation
        errors = []
        if size <= 0:
            errors.append("Площадь должна быть строго больше 0 м².")
        if rooms + living_rooms == 0:
            errors.append("Суммарное количество комнат должно быть не менее 1.")
        if floor_no > total_floors:
            errors.append(f"Этаж объекта ({floor_no:.0f}) не может превышать этажность здания ({total_floors:.0f}).")
        if not city.strip() or not district.strip():
            errors.append("Поля 'Город' и 'Район' обязательны для заполнения.")
            
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
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
            
            # Predict
            try:
                raw_pred = model.predict(df_ready)[0]
                pred_price = max(0.0, float(raw_pred))
                
                # Confidence interval bounds (±MAE)
                ci_low = max(0.0, pred_price - best_mae)
                ci_high = pred_price + best_mae
                
                # Foreign currencies
                price_usd = pred_price / RATES.get('USD', 33.0)
                price_eur = pred_price / RATES.get('EUR', 36.0)
                
                st.success("✅ Оценка успешно выполнена!")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        label="💵 Прогнозная стоимость (TRY)",
                        value=f"{pred_price:,.0f} ₺",
                        delta=f"±{best_mae:,.0f} ₺ (MAE)"
                    )
                with m2:
                    st.metric(
                        label="💶 В эквиваленте EUR",
                        value=f"€{price_eur:,.0f}"
                    )
                with m3:
                    st.metric(
                        label="💵 В эквиваленте USD",
                        value=f"${price_usd:,.0f}"
                    )
                    
                st.markdown("### 📊 Доверительный диапазон оценки:")
                st.info(f"С вероятностью доверительного интервала модели цена объекта составляет от **{ci_low:,.0f} ₺** до **{ci_high:,.0f} ₺**.")
                
                # Key feature breakdown card
                st.markdown("#### 🔍 Учтённые ключевые факторы:")
                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"• **Цена за м²:** {pred_price / size:,.0f} ₺/м²")
                c2.write(f"• **Позиция в доме:** {df_ready['floor_ratio'].values[0]*100:.0f}% высоты")
                c3.write(f"• **Площадь на комнату:** {size / (rooms + living_rooms):.1f} м²")
                c4.write(f"• **Тип объекта:** {sub_type} ({deal_type})")
                
            except Exception as e:
                st.error(f"Ошибка при инференсе модели: {e}")

# =========================================================
# TAB 2: INTERACTIVE DASHBOARD
# =========================================================
with tab_dash:
    st.title("📊 Интерактивный дашборд рынка недвижимости")
    st.markdown("Исследуйте ключевые тенденции, зависимости и распределения в реальном времени.")

    if df_dash.empty:
        st.warning("Файл с аналитическими данными не найден. Пожалуйста, запустите обучение модели.")
    else:
        # Dashboard filter controls
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            deal_filter = st.selectbox("Фильтр по типу сделки", ["Все", "Продажа (Satılık)", "Аренда (Kiralık)"])
        with f_col2:
            top_cities_dash = ["Все города"] + list(df_dash['city'].value_counts().head(10).index)
            city_filter = st.selectbox("Фильтр по городу", top_cities_dash)
        with f_col3:
            subtypes_dash = ["Все подтипы"] + list(df_dash['sub_type'].value_counts().head(8).index)
            type_filter = st.selectbox("Фильтр по типу объекта", subtypes_dash)

        # Apply filters
        dff = df_dash.copy()
        if deal_filter == "Продажа (Satılık)":
            dff = dff[dff['listing_type'] == 1]
        elif deal_filter == "Аренда (Kiralık)":
            dff = dff[dff['listing_type'] == 2]
            
        if city_filter != "Все города":
            dff = dff[dff['city'] == city_filter]
        if type_filter != "Все подтипы":
            dff = dff[dff['sub_type'] == type_filter]

        # Key Metrics Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("📦 Всего объявлений", f"{len(dff):,}")
        kpi2.metric("💰 Медианная цена", f"{dff['price_try'].median():,.0f} ₺")
        kpi3.metric("📐 Средняя площадь", f"{dff['size'].mean():.1f} м²")
        kpi4.metric("🏙️ Лидирующий город", f"{dff['city'].mode()[0] if not dff.empty else 'N/A'}")

        st.markdown("---")
        
        # Row 1 of Charts
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.subheader("📈 Распределение цен объектов")
            log_scale = st.checkbox("Логарифмическая шкала цен", value=True)
            fig_hist = px.histogram(
                dff, x="price_try", nbins=50, log_x=log_scale,
                color="sub_type", title="Гистограмма цен по подтипам",
                labels={"price_try": "Цена (TRY)", "count": "Объявления"}
            )
            fig_hist.update_layout(bargap=0.1, template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with row1_col2:
            st.subheader("🏙️ Средняя цена по городам (Топ-10)")
            city_summary = dff.groupby('city')['price_try'].agg(['mean', 'count']).reset_index()
            city_summary = city_summary.sort_values(by='count', ascending=False).head(10)
            fig_city = px.bar(
                city_summary, x="city", y="mean", color="mean",
                title="Средняя стоимость объектов в популярных регионах",
                labels={"city": "Город", "mean": "Средняя цена (TRY)"},
                color_continuous_scale="Viridis"
            )
            fig_city.update_layout(template="plotly_white")
            st.plotly_chart(fig_city, use_container_width=True)

        # Row 2 of Charts
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.subheader("📍 Площадь (м²) vs Цена (TRY)")
            sample_scatter = dff.sample(n=min(3000, len(dff)), random_state=42) if len(dff) > 3000 else dff
            fig_scatter = px.scatter(
                sample_scatter, x="size", y="price_try", color="sub_type",
                log_y=True, hover_data=["city", "district", "room_total"],
                title="Зависимость цены от площади (логарифмическая шкала)",
                labels={"size": "Площадь (м²)", "price_try": "Цена (TRY)"}
            )
            fig_scatter.update_layout(template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with row2_col2:
            st.subheader("🏆 Сравнение протестированных моделей")
            models_data = meta_info.get('test_metrics', [])
            if models_data:
                df_models = pd.DataFrame(models_data)
                fig_models = px.bar(
                    df_models, x="Model", y="R2", text="R2", color="Model",
                    title="Коэффициент детерминации R² на тестовой выборке",
                    labels={"R2": "R² (чем выше, тем лучше)", "Model": "Модель"}
                )
                fig_models.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_models.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig_models, use_container_width=True)
            else:
                st.info("Информация по сравнениям моделей представлена во вкладке 'Справка'.")

# =========================================================
# TAB 3: HELP & DOCUMENTATION
# =========================================================
with tab_help:
    st.title("ℹ️ Справка и документация проекта")
    
    st.markdown("""
    ### 1. Назначение приложения
    Данная система предназначена для автоматизированного анализа рынка и прогнозирования рыночной стоимости жилой недвижимости на базе обученных моделей машинного обучения (ML).
    Модель обучена на выборке из более чем **400 000 объявлений** портала недвижимости *Zingat*.

    ---
    ### 2. Описание входных параметров
    | Параметр | Единица / Формат | Описание |
    | :--- | :--- | :--- |
    | **Тип сделки** | Продажа / Аренда | Разделяет объявления на покупку (1) и аренду (2) |
    | **Подтип (sub_type)** | Категория | Вид жилья: квартира (Daire), резиденция, вилла, дом |
    | **Площадь (size)** | м² | Общая площадь объекта недвижимости |
    | **Комнаты / Залы** | Число (2+1, 3+1 и т.д.) | Число спален и гостиных |
    | **Возраст здания** | Лет | Возраст строения от момента ввода в эксплуатацию |
    | **Этаж / Всего этажей** | Числовые значения | Этаж расположения квартиры и общая этажность дома |
    | **Отопление** | Категория | Тип отопительной системы (Kombi, Fancoil, Merkezi) |
    | **Город и Район** | Текст | Географическая привязка объекта |

    ---
    ### 3. Описание используемой модели машинного обучения
    В качестве финальной модели выбран ансамбль **Random Forest Regressor (Bagging)** со следующими характеристиками:
    - **Количество базовых деревьев:** 100
    - **Критерий расщепления:** Squared Error (MSE)
    - **Регуляризация глубины:** `max_depth = 16`, `min_samples_leaf = 10`
    - **Предобработка:** Заполнение пропусков медианой, One-Hot Encoding категориальных признаков с фильтрацией редких категорий, нормализация числовых признаков.

    #### Сравнительная таблица исследованных моделей:
    """)

    test_mets = meta_info.get('test_metrics', [])
    if test_mets:
        st.dataframe(pd.DataFrame(test_mets), use_container_width=True)
    else:
        st.write("• Ridge Regression (Linear): R² ~ 0.0610, MAE ~ 365,785 TRY")
        st.write("• Decision Tree: R² ~ 0.0535, MAE ~ 263,838 TRY")
        st.write("• LightGBM: R² ~ 0.0964, MAE ~ 274,901 TRY")
        st.write("• **Random Forest (Лучшая): R² ~ 0.1546, MAE ~ 246,803 TRY**")

    st.markdown("""
    #### Сильные стороны модели:
    - Устойчивость к локальным шумам и мультиколлинеарности данных.
    - Автоматический учет нелинейных взаимодействий (например, сочетание элитного района и площади).
    - Быстрый отклик при инференсе (< 20 миллисекунд на запрос).

    #### Ограничения модели:
    - Прогнозирование цен эксклюзивных премиальных особняков (стоимостью свыше 50 млн TRY) может иметь повышенную погрешность из-за их единичности в выборке.
    - Модель фиксирует ценовые пропорции на момент сбора данных (валютная волатильность турецкой лиры требует периодического переобучения).

    ---
    ### 4. Авторы и информация о релизе
    - **Проект:** Прогнозирование стоимости недвижимости (Групповой & Сольный проект)
    - **Автор:** Nezire (Группа МОИБД)
    - **Версия:** 2.0 Pro Release
    - **Стек технологий:** Python 3.11, Streamlit, Scikit-Learn, LightGBM, Plotly, Pandas, NumPy
    """)
