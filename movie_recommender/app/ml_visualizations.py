import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_dataset_overview_tab(summary: dict):
    st.subheader("Сводка по данным")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Пользователей", summary.get("total_users", 0))
    with c2:
        st.metric("Фильмов", summary.get("total_movies", 0))
    with c3:
        st.metric("Оценок", summary.get("total_ratings", 0))
    with c4:
        st.metric("Разреженность", f"{summary.get('sparsity_pct', 0.0)}%")

    m1, m2 = st.columns(2)
    with m1:
        st.info(f"Размерность: {summary.get('matrix_dimensions', '0x0')}")
    with m2:
        st.info(f"Средняя оценка: {summary.get('mean_rating', 0.0)} из 5.0")


def render_distribution_charts(rating_dist: list, genre_dist: list):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение оценок")
        if rating_dist:
            df_r = pd.DataFrame(rating_dist)
            fig_r = px.bar(
                df_r,
                x="rating",
                y="count",
                labels={"rating": "Оценка", "count": "Количество"},
                text="count"
            )
            fig_r.update_traces(marker_color="#4A90E2")
            fig_r.update_layout(
                xaxis=dict(tickmode="linear", tick0=1, dtick=0.5),
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.write("Нет данных")

    with col2:
        st.subheader("Фильмы по жанрам")
        if genre_dist:
            df_g = pd.DataFrame(genre_dist).head(10)
            fig_g = px.bar(
                df_g,
                x="genre",
                y="count",
                labels={"genre": "Жанр", "count": "Количество"},
                text="count"
            )
            fig_g.update_traces(marker_color="#5DADE2")
            fig_g.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.write("Нет данных")


def render_model_evaluation_chart(eval_data: dict):
    st.subheader("Сравнение моделей")

    models_data = eval_data.get("models", {})
    if not models_data:
        st.write("Недостаточно данных для оценки.")
        return

    rows = []
    for model_name, metrics in models_data.items():
        rows.append({
            "Модель": model_name,
            "RMSE": metrics.get("rmse", 0.0),
            "MAE": metrics.get("mae", 0.0),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
        })
    df_eval = pd.DataFrame(rows)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df_eval, use_container_width=True)

    with c2:
        fig = go.Figure(data=[
            go.Bar(name="RMSE", x=df_eval["Модель"], y=df_eval["RMSE"], marker_color="#E74C3C"),
            go.Bar(name="MAE", x=df_eval["Модель"], y=df_eval["MAE"], marker_color="#F39C12")
        ])
        fig.update_layout(
            barmode="group",
            yaxis_title="Ошибка",
            xaxis_tickangle=-15,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


def render_pca_projection(pca_data: list):
    st.subheader("2D проекция фильмов (PCA)")

    if not pca_data:
        st.write("Нет данных для PCA")
        return

    df_pca = pd.DataFrame(pca_data)

    fig = px.scatter(
        df_pca,
        x="pc1",
        y="pc2",
        color="genre",
        hover_name="title",
        labels={"pc1": "PC1", "pc2": "PC2", "genre": "Жанр"},
    )
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_similarity_heatmap(sim_data: dict):
    st.subheader("Матрица сходства фильмов")

    titles = sim_data.get("titles", [])
    matrix = sim_data.get("matrix", [])

    if not titles or not matrix:
        st.write("Нет данных для матрицы сходства")
        return

    short_titles = [t[:15] + ".." if len(t) > 15 else t for t in titles]

    fig = px.imshow(
        matrix,
        x=short_titles,
        y=short_titles,
        color_continuous_scale="Blues",
        labels=dict(x="Фильм", y="Фильм", color="Сходство"),
        text_auto=".2f"
    )
    fig.update_layout(
        height=450,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
