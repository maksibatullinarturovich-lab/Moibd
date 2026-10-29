import os
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def resolve_poster_path(movie: dict):
    poster = movie.get("poster_url")
    if poster:
        if os.path.isabs(poster) and os.path.exists(poster):
            return poster
        rel = os.path.join(PROJECT_ROOT, poster)
        if os.path.exists(rel):
            return rel
        if poster.startswith(("http://", "https://")):
            return poster

    m_id = movie.get("id")
    if m_id:
        for ext in (".jpg", ".png", ".jpeg"):
            fallback = os.path.join(PROJECT_ROOT, "app", "static", "posters", f"{m_id}{ext}")
            if os.path.exists(fallback):
                return fallback

    title = movie.get("title")
    if title:
        safe_title = title.replace(":", "").replace("/", "_").replace("«", "").replace("»", "").strip()
        for ext in (".jpg", ".png", ".jpeg"):
            fallback = os.path.join(PROJECT_ROOT, "app", "static", "posters", f"{safe_title}{ext}")
            if os.path.exists(fallback):
                return fallback

    return None

def render_movie_card(movie: dict, rank: int = None, show_score: str = None, key_prefix: str = ""):
    with st.container():
        cols = st.columns([1, 4])

        with cols[0]:
            poster_path = resolve_poster_path(movie)
            if poster_path:
                st.image(poster_path, use_container_width=True)
            else:
                st.write("Нет постера")

        with cols[1]:
            year_str = f" ({movie.get('release_year')})" if movie.get("release_year") else ""
            rank_str = f"{rank}. " if rank else ""
            st.subheader(f"{rank_str}{movie.get('title')}{year_str}")

            badge_cols = st.columns(4)
            with badge_cols[0]:
                avg = movie.get("avg_rating", 0.0)
                st.caption(f"Средний балл: {avg:.1f}")
            with badge_cols[1]:
                cnt = movie.get("ratings_count", 0)
                st.caption(f"Оценок: {cnt}")

            with badge_cols[2]:
                if show_score == "weighted" and movie.get("weighted_score") is not None:
                    st.caption(f"Рейтинг: {movie.get('weighted_score'):.2f}")
                elif show_score == "predicted" and movie.get("predicted_rating") is not None:
                    st.caption(f"Прогноз: {movie.get('predicted_rating'):.2f} / 5")
                elif show_score == "similarity" and movie.get("similarity_score") is not None:
                    st.caption(f"Сходство: {movie.get('similarity_score')*100:.1f}%")

            with badge_cols[3]:
                genres = movie.get("genres", "")
                st.caption(f"Жанры: {genres}")

            if movie.get("director") or movie.get("cast"):
                st.caption(f"Режиссер: {movie.get('director', 'н/д')} | В ролях: {movie.get('cast', 'н/д')}")

            if movie.get("description"):
                st.write(movie.get("description"))

            if movie.get("explanation"):
                st.info(movie.get("explanation"))

        st.divider()
