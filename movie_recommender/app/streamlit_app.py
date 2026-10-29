import os
import sys
import requests
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.components.movie_card import render_movie_card
from app.ml_visualizations import (
    render_dataset_overview_tab,
    render_distribution_charts,
    render_model_evaluation_chart,
    render_pca_projection, 
    render_similarity_heatmap
)

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Рекомендательная система фильмов",
    layout="wide",
    initial_sidebar_state="expanded"
)


def call_api(endpoint: str, method: str = "GET", params: dict = None, json_data: dict = None):
    url = f"{API_BASE_URL.rstrip('/')}{endpoint}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, params=params, timeout=10)
        elif method.upper() == "POST":
            resp = requests.post(url, json=json_data, timeout=10)
        else:
            return None, f"Неподдерживаемый метод {method}"

        if resp.status_code in (200, 201):
            return resp.json(), None
        else:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            return None, f"Ошибка {resp.status_code}: {err_detail}"
    except requests.exceptions.ConnectionError:
        return None, "Не удалось подключиться к серверу API."
    except Exception as e:
        return None, str(e)


# Боковая панель
st.sidebar.subheader("Пользователь")

users_data, _ = call_api("/api/users")
user_list = users_data if users_data else []
user_options = {u["id"]: f"{u['username']} ({u['ratings_count']} оценок)" for u in user_list}

selected_user_id = None
if user_options:
    selected_user_id = st.sidebar.selectbox(
        "Выберите пользователя:",
        options=list(user_options.keys()),
        format_func=lambda x: user_options[x],
        index=0
    )
    st.session_state["active_user_id"] = selected_user_id
else:
    st.sidebar.info("Загрузка пользователей...")

st.sidebar.markdown("---")
st.sidebar.subheader("Разделы")
nav_choice = st.sidebar.radio(
    "Выберите раздел:",
    options=[
        "1. Топ-10 популярных фильмов",
        "2. Коллаборативная фильтрация",
        "3. Рекомендации по контенту",
        "4. Персональные рекомендации",
        "5. Оценить фильм (БД MySQL)",
        "6. Графики и аналитика",
        "7. Справка по командам и параметрам"
    ],
    index=0
)

st.title("Рекомендательная система фильмов")

# 1. Топ-10 популярных фильмов
if nav_choice.startswith("1."):
    st.header("1. Список Топ-10 популярных фильмов")
    st.write("Список популярных фильмов с наибольшим рейтингом.")

    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        genres_data, _ = call_api("/api/movies/genres")
        genre_options = ["Все жанры"] + (genres_data if genres_data else [])
        chosen_genre = st.selectbox("Фильтр по жанру:", options=genre_options)
    with col_filter2:
        limit_val = st.slider("Число фильмов (N):", min_value=5, max_value=30, value=10, step=1)

    genre_param = chosen_genre if chosen_genre != "Все жанры" else None
    params = {"limit": limit_val}
    if genre_param:
        params["genre"] = genre_param

    with st.spinner("Загрузка фильмов..."):
        data, err = call_api("/api/recommendations/popular", params=params)

    if err:
        st.error(err)
    elif data:
        recs = data.get("recommendations", [])
        st.success(f"Найдено фильмов: {len(recs)}")
        for movie in recs:
            render_movie_card(movie, rank=movie.get("rank"), show_score="weighted")

# 2. Коллаборативная фильтрация
elif nav_choice.startswith("2."):
    st.header("2. Коллаборативная фильтрация")
    st.write("Рекомендации на основе матрицы оценок пользователей и фильмов.")

    cf_tab1, cf_tab2 = st.tabs(["По пользователям", "По фильмам"])

    with cf_tab1:
        st.subheader("Рекомендации по пользователям")
        st.write("Поиск похожих пользователей и прогноз оценок.")

        if selected_user_id:
            c_u1, c_u2 = st.columns([3, 1])
            with c_u1:
                st.info(f"Пользователь: {user_options.get(selected_user_id, 'User')} (ID: {selected_user_id})")
            with c_u2:
                ubcf_limit = st.slider("Лимит рекомендаций:", min_value=5, max_value=20, value=10, key="ubcf_lim")

            with st.spinner("Расчет рекомендаций..."):
                ubcf_data, ubcf_err = call_api(
                    "/api/recommendations/collaborative/user-based",
                    params={"user_id": selected_user_id, "limit": ubcf_limit}
                )

            if ubcf_err:
                st.error(ubcf_err)
            elif ubcf_data:
                recs = ubcf_data.get("recommendations", [])
                st.success(f"Сформировано рекомендаций: {len(recs)}")
                for m in recs:
                    render_movie_card(m, rank=m.get("rank"), show_score="predicted")
        else:
            st.warning("Выберите пользователя в боковой панели.")

    with cf_tab2:
        st.subheader("Рекомендации по фильмам")
        st.write("Поиск похожих фильмов по совпадению оценок.")

        movies_data, _ = call_api("/api/movies", params={"limit": 100})
        movie_dict = {m["id"]: f"{m['title']} ({m.get('release_year', '—')}) [{m['genres']}]" for m in (movies_data or [])}

        if movie_dict:
            target_movie_id = st.selectbox(
                "Выберите фильм:",
                options=list(movie_dict.keys()),
                format_func=lambda x: movie_dict[x],
                index=0
            )
            ibcf_limit = st.slider("Количество похожих фильмов:", min_value=3, max_value=15, value=10, key="ibcf_lim")

            with st.spinner("Поиск похожих фильмов..."):
                ibcf_data, ibcf_err = call_api(
                    "/api/recommendations/collaborative/item-based",
                    params={"movie_id": target_movie_id, "limit": ibcf_limit}
                )

            if ibcf_err:
                st.error(ibcf_err)
            elif ibcf_data:
                recs = ibcf_data.get("recommendations", [])
                st.success(f"Найдено похожих фильмов: {len(recs)}")
                for m in recs:
                    render_movie_card(m, rank=m.get("rank"), show_score="similarity")
        else:
            st.warning("Фильмы не найдены.")

# 3. Рекомендации по контенту
elif nav_choice.startswith("3."):
    st.header("3. Рекомендации по контенту")
    st.write("Поиск похожих фильмов по жанрам, описанию и создателям.")

    movies_data, _ = call_api("/api/movies", params={"limit": 100})
    movie_dict = {m["id"]: f"{m['title']} ({m.get('release_year', '—')}) — {m['genres']}" for m in (movies_data or [])}

    if movie_dict:
        col_cb1, col_cb2 = st.columns([3, 1])
        with col_cb1:
            chosen_movie_id = st.selectbox(
                "Выберите фильм:",
                options=list(movie_dict.keys()),
                format_func=lambda x: movie_dict[x],
                index=0
            )
        with col_cb2:
            cb_limit = st.slider("Число рекомендаций:", min_value=5, max_value=20, value=10, key="cb_lim")

        with st.spinner("Расчет рекомендаций..."):
            cb_data, cb_err = call_api(
                "/api/recommendations/content-based",
                params={"movie_id": chosen_movie_id, "limit": cb_limit}
            )

        if cb_err:
            st.error(cb_err)
        elif cb_data:
            recs = cb_data.get("recommendations", [])
            st.success(f"Сформировано рекомендаций: {len(recs)}")
            for m in recs:
                render_movie_card(m, rank=m.get("rank"), show_score="similarity")
    else:
        st.warning("Фильмы не загружены.")

# 4. Персональные рекомендации
elif nav_choice.startswith("4."):
    st.header("4. Персональные рекомендации")
    st.write("Персональный список рекомендаций для выбранного пользователя.")

    if selected_user_id:
        st.info(f"Пользователь: {user_options.get(selected_user_id, 'User')}")

        c_p1, c_p2 = st.columns([2, 1])
        with c_p1:
            alpha_val = st.slider(
                "Баланс модели (по оценкам / по описанию):",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05
            )
        with c_p2:
            p_limit = st.slider("Число рекомендаций:", min_value=5, max_value=20, value=10, key="p_lim")

        with st.spinner("Построение рекомендаций..."):
            p_data, p_err = call_api(
                "/api/recommendations/personal",
                params={"user_id": selected_user_id, "limit": p_limit, "alpha": alpha_val}
            )

        if p_err:
            st.error(p_err)
        elif p_data:
            recs = p_data.get("recommendations", [])
            st.success(f"Сформировано рекомендаций: {len(recs)}")
            for m in recs:
                render_movie_card(m, rank=m.get("rank"), show_score="predicted")
    else:
        st.warning("Выберите пользователя в боковой панели.")

# 5. Оценка фильма и сохранение в MySQL
elif nav_choice.startswith("5."):
    st.header("5. Оценка фильма (БД MySQL)")
    st.write("Выбор фильма и выставление оценки с сохранением в базу данных.")

    if not selected_user_id:
        st.warning("Сначала выберите пользователя в левой панели.")
    else:
        st.info(f"Пользователь: {user_options.get(selected_user_id)}")

        movies_data, _ = call_api("/api/movies", params={"limit": 100})
        movie_dict = {m["id"]: f"{m['title']} ({m.get('release_year', '—')}) — {m['genres']}" for m in (movies_data or [])}

        if movie_dict:
            with st.form("rating_form"):
                chosen_movie_id = st.selectbox(
                    "Выберите фильм:",
                    options=list(movie_dict.keys()),
                    format_func=lambda x: movie_dict[x]
                )

                score_input = st.select_slider(
                    "Оценка:",
                    options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                    value=5.0
                )

                submit_btn = st.form_submit_button("Сохранить оценку")

            if submit_btn:
                payload = {
                    "user_id": selected_user_id,
                    "movie_id": chosen_movie_id,
                    "rating": float(score_input)
                }
                with st.spinner("Запись в базу данных..."):
                    res, err = call_api("/api/ratings", method="POST", json_data=payload)

                if err:
                    st.error(f"Ошибка сохранения: {err}")
                elif res:
                    st.success(
                        f"Оценка сохранена. Фильм: {res.get('movie_title')}, "
                        f"Балл: {res.get('rating')}"
                    )

            st.markdown("---")
            st.subheader("История оценок пользователя")
            user_ratings, ur_err = call_api(f"/api/ratings/user/{selected_user_id}")

            if user_ratings:
                df_history = pd.DataFrame([
                    {
                        "ID фильма": r["movie_id"],
                        "Название": r["movie_title"],
                        "Оценка": r["rating"],
                        "Дата": r.get("timestamp", "—")
                    }
                    for r in user_ratings
                ])
                st.dataframe(df_history, use_container_width=True)
            else:
                st.info("У этого пользователя пока нет сохраненных оценок.")

# 6. Графики и аналитика
elif nav_choice.startswith("6."):
    st.header("6. Графики и аналитика")

    summary_data, _ = call_api("/api/analytics/summary")
    if summary_data:
        render_dataset_overview_tab(summary_data)

    st.markdown("---")

    r_dist, _ = call_api("/api/analytics/ratings-distribution")
    g_dist, _ = call_api("/api/analytics/genres-distribution")
    render_distribution_charts(r_dist, g_dist)

    st.markdown("---")

    eval_data, _ = call_api("/api/analytics/model-metrics")
    if eval_data:
        render_model_evaluation_chart(eval_data)

    st.markdown("---")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        pca_data, _ = call_api("/api/analytics/pca-projection")
        if pca_data:
            render_pca_projection(pca_data)

    with col_v2:
        sim_data, _ = call_api("/api/analytics/similarity-matrix")
        if sim_data:
            render_similarity_heatmap(sim_data)

# 7. Справка по командам и параметрам
elif nav_choice.startswith("7."):
    st.header("7. Справка по командам и их параметрам")
    st.write("Справочная информация по эндпоинтам и параметрам API.")

    endpoints_documentation = [
        {
            "Команда / Эндпоинт": "GET /api/recommendations/popular",
            "Назначение": "Топ-10 популярных фильмов",
            "Параметры": "limit: int (по умолчанию 10), genre: str (фильтр по жанру)"
        },
        {
            "Команда / Эндпоинт": "GET /api/recommendations/collaborative/user-based",
            "Назначение": "Рекомендации по похожим пользователям",
            "Параметры": "user_id: int (обязательный), limit: int (по умолчанию 10)"
        },
        {
            "Команда / Эндпоинт": "GET /api/recommendations/collaborative/item-based",
            "Назначение": "Рекомендации похожих фильмов",
            "Параметры": "movie_id: int (обязательный), limit: int (по умолчанию 10)"
        },
        {
            "Команда / Эндпоинт": "GET /api/recommendations/collaborative/item-based-for-user",
            "Назначение": "Рекомендации по фильмам для пользователя",
            "Параметры": "user_id: int (обязательный), limit: int (по умолчанию 10)"
        },
        {
            "Команда / Эндпоинт": "GET /api/recommendations/content-based",
            "Назначение": "Рекомендации 10 фильмов по контенту",
            "Параметры": "movie_id: int (обязательный), limit: int (по умолчанию 10)"
        },
        {
            "Команда / Эндпоинт": "GET /api/recommendations/personal",
            "Назначение": "Персональные рекомендации для пользователя",
            "Параметры": "user_id: int (обязательный), limit: int (по умолчанию 10), alpha: float (баланс)"
        },
        {
            "Команда / Эндпоинт": "POST /api/ratings",
            "Назначение": "Оценить фильм с сохранением в БД MySQL",
            "Параметры": "user_id (int), movie_id (int), rating (float от 1.0 до 5.0)"
        },
        {
            "Команда / Эндпоинт": "GET /api/ratings/user/{user_id}",
            "Назначение": "Получить все оценки пользователя из БД",
            "Параметры": "user_id: int"
        },
        {
            "Команда / Эндпоинт": "GET /api/movies",
            "Назначение": "Каталог фильмов с поиском и фильтром по жанру",
            "Параметры": "query: str, genre: str, limit: int"
        },
        {
            "Команда / Эндпоинт": "GET /api/analytics/model-metrics",
            "Назначение": "Метрики качества моделей (RMSE, MAE, Precision, Recall)",
            "Параметры": "Без параметров"
        }
    ]

    df_endpoints = pd.DataFrame(endpoints_documentation)
    st.table(df_endpoints)

    st.markdown("---")
    st.subheader("Интерактивная документация Swagger UI")
    st.markdown(f"Документация доступна по адресу: [{API_BASE_URL}/docs]({API_BASE_URL}/docs)")
