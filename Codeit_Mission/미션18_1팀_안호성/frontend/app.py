"""미션 18용 Streamlit 프론트엔드."""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"


def api_get(base_url: str, path: str, params: dict | None = None) -> dict:
    """백엔드로 GET 요청을 보낸다.

    Args:
        base_url: 백엔드 기본 URL.
        path: API 경로.
        params: 선택적 쿼리 파라미터.

    Returns:
        dict: JSON 응답 데이터.
    """

    response = requests.get(f"{base_url}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(base_url: str, path: str, payload: dict) -> dict:
    """백엔드로 POST 요청을 보낸다.

    Args:
        base_url: 백엔드 기본 URL.
        path: API 경로.
        payload: JSON 요청 바디.

    Returns:
        dict: JSON 응답 데이터.
    """

    response = requests.post(f"{base_url}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def load_movies(base_url: str) -> list[dict]:
    """백엔드에서 영화 목록을 가져온다.

    Args:
        base_url: 백엔드 기본 URL.

    Returns:
        list[dict]: 영화 목록.
    """

    return api_get(base_url, "/movies").get("movies", [])


def load_recent_reviews(base_url: str, limit: int = 10) -> list[dict]:
    """백엔드에서 최근 리뷰를 가져온다.

    Args:
        base_url: 백엔드 기본 URL.
        limit: 최대 리뷰 개수.

    Returns:
        list[dict]: 리뷰 목록.
    """

    return api_get(base_url, "/reviews", params={"limit": limit}).get("reviews", [])


def render_movie_list(movies: list[dict]) -> None:
    """영화 카드 목록을 화면에 그린다.

    Args:
        movies: 영화 목록.
    """

    st.subheader("영화 목록")
    if not movies:
        st.info("등록된 영화가 없습니다.")
        return

    columns = st.columns(3)
    for idx, movie in enumerate(movies):
        with columns[idx % 3]:
            st.image(movie["poster_url"], use_container_width=True)
            st.markdown(f"### {movie['title']}")
            st.caption(f"감독: {movie['director']}")
            st.caption(f"장르: {movie['genre']}")
            st.caption(f"개봉일: {movie['release_date']}")
            avg_rating = movie.get("average_rating")
            if avg_rating is not None:
                st.metric("평균 평점(5점 환산)", f"{avg_rating:.2f}")
            st.caption(f"리뷰 수: {movie['review_count']}")


def render_recent_reviews(reviews: list[dict]) -> None:
    """최근 리뷰 표를 화면에 그린다.

    Args:
        reviews: 리뷰 목록.
    """

    st.subheader("최근 10개 리뷰")
    if not reviews:
        st.info("등록된 리뷰가 없습니다.")
        return

    rows = [
        {
            "리뷰 ID": review["id"],
            "영화 ID": review["movie_id"],
            "영화 제목": review["movie_title"],
            "작성자": review["author"],
            "등록일": review["created_at"],
            "리뷰 내용": review["content"],
            "감성 결과": review["sentiment_label"],
            "감성 점수": review["sentiment_score"],
        }
        for review in reviews
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


st.set_page_config(page_title="Mission 18 Movie Review App", page_icon="🎬", layout="wide")
st.title("🎬 영화 리뷰 감성 분석 서비스")
st.caption("프론트엔드: Streamlit / 백엔드: FastAPI")

if "review_success_message" not in st.session_state:
    st.session_state["review_success_message"] = ""

with st.sidebar:
    st.header("백엔드 연결")
    api_base_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL).rstrip("/")
    if st.button("연결 확인", use_container_width=True):
        try:
            health = api_get(api_base_url, "/health")
            st.success(f"연결 성공: {health['status']}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"연결 실패: {exc}")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    with st.form("movie_form", clear_on_submit=True):
        st.subheader("영화 추가")
        title = st.text_input("제목")
        release_date = st.date_input("개봉일", value=date.today())
        director = st.text_input("감독")
        genre = st.text_input("장르")
        poster_url = st.text_input("포스터 URL")
        submitted_movie = st.form_submit_button("영화 등록", use_container_width=True)

    if submitted_movie:
        try:
            result = api_post(
                api_base_url,
                "/movies",
                {
                    "title": title,
                    "release_date": str(release_date),
                    "director": director,
                    "genre": genre,
                    "poster_url": poster_url,
                },
            )
            st.success(f"영화 등록 완료: {result['title']}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"영화 등록 실패: {exc}")

    try:
        movies = load_movies(api_base_url)
        render_movie_list(movies)
    except Exception as exc:  # noqa: BLE001
        movies = []
        st.error(f"영화 목록 조회 실패: {exc}")

with right_col:
    st.subheader("리뷰 등록")
    if st.session_state["review_success_message"]:
        st.success(st.session_state["review_success_message"])
        st.session_state["review_success_message"] = ""
    if movies:
        movie_options = {f"{movie['id']} | {movie['title']}": movie["id"] for movie in movies}
        with st.form("review_form", clear_on_submit=True):
            selected_movie_key = st.selectbox("영화 선택", list(movie_options.keys()))
            author = st.text_input("작성자 이름")
            content = st.text_area("리뷰 내용", height=180)
            submitted_review = st.form_submit_button("리뷰 등록 및 감성 분석", use_container_width=True)

        if submitted_review:
            try:
                review_result = api_post(
                    api_base_url,
                    "/reviews",
                    {
                        "movie_id": movie_options[selected_movie_key],
                        "author": author,
                        "content": content,
                    },
                )
                sentiment = review_result["sentiment"]
                st.session_state["review_success_message"] = (
                    "리뷰 등록 및 감성 분석이 완료되었습니다. 영화 평점과 최근 리뷰를 갱신했습니다."
                )
                st.session_state["last_sentiment"] = sentiment
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"리뷰 등록 실패: {exc}")
        if "last_sentiment" in st.session_state:
            sentiment = st.session_state["last_sentiment"]
            st.write("가장 최근 감성 분석 결과")
            metrics = st.columns(4)
            metrics[0].metric("예측 라벨", sentiment["label"])
            metrics[1].metric("감성 점수", f"{sentiment['score']:.4f}")
            metrics[2].metric("긍정 확률", f"{sentiment['positive_probability']:.4f}")
            metrics[3].metric("부정 확률", f"{sentiment['negative_probability']:.4f}")
    else:
        st.info("리뷰를 등록하려면 먼저 영화를 1개 이상 추가해야 합니다.")

    try:
        recent_reviews = load_recent_reviews(api_base_url, limit=10)
        render_recent_reviews(recent_reviews)
    except Exception as exc:  # noqa: BLE001
        st.error(f"리뷰 조회 실패: {exc}")
