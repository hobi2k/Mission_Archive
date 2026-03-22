"""영화별 샘플 리뷰를 등록하는 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import crud, schemas  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.sentiment import get_sentiment_analyzer  # noqa: E402


REVIEW_BANK = {
    "기생충": [
        "계층 구조를 이렇게 날카롭게 보여준 영화는 오랜만이었다.",
        "연출과 배우 연기가 모두 훌륭해서 몰입감이 강했다.",
        "생각할 거리가 많고 결말도 강하게 남는다.",
        "중반 이후 긴장감이 엄청나서 눈을 떼기 어려웠다.",
        "블랙코미디와 스릴러가 자연스럽게 섞여서 좋았다.",
        "재관람해도 디테일이 계속 보일 것 같은 작품이다.",
        "전개가 촘촘하고 메시지도 선명해서 만족스러웠다.",
        "다소 불편한 장면도 있었지만 그만큼 힘이 있었다.",
        "상징이 많아서 보고 나서 이야기할 거리가 풍부하다.",
        "왜 상을 많이 받았는지 이해되는 완성도였다.",
    ],
    "올드보이": [
        "강렬한 설정과 연출이 아직도 인상적으로 남아 있다.",
        "분위기가 독특하고 서사가 충격적이다.",
        "무거운 감정선이 끝까지 이어져서 쉽게 잊히지 않는다.",
        "호불호는 갈릴 수 있지만 확실히 개성이 강하다.",
        "미장센과 음악이 영화의 감정을 잘 끌어올린다.",
        "결말이 너무 세서 보고 나면 한동안 멍해진다.",
        "불편한 요소가 있지만 영화적 밀도는 높다.",
        "복수극의 형태를 빌려 인간의 집착을 잘 보여준다.",
        "배우의 에너지가 대단해서 장면마다 압도된다.",
        "편하게 볼 작품은 아니지만 인상적인 작품임은 분명하다.",
    ],
    "인터스텔라": [
        "우주를 배경으로 한 감정 드라마라는 점이 좋았다.",
        "영상미와 음악이 웅장해서 극장에서 보기 좋은 영화다.",
        "과학 설정이 어려운 부분도 있지만 감정선은 명확했다.",
        "가족 이야기가 중심이라 더 몰입해서 볼 수 있었다.",
        "러닝타임이 길지만 생각보다 지루하지 않았다.",
        "몇몇 설명은 어렵지만 장면의 힘으로 끝까지 끌고 간다.",
        "우주 장면이 아름답고 스케일이 크다.",
        "후반부 감정 폭발이 좋아서 다시 보고 싶어졌다.",
        "사운드트랙이 정말 뛰어나서 장면마다 더 인상적이었다.",
        "SF를 좋아하지 않아도 충분히 즐길 만한 영화였다.",
    ],
}


def main() -> None:
    """영화별로 리뷰가 부족할 때 샘플 리뷰 10개를 채운다."""

    analyzer = get_sentiment_analyzer()
    db = SessionLocal()
    try:
        movies = crud.list_movies(db)
        if not movies:
            print("먼저 seed_sample_data.py로 영화를 등록하세요.")
            return

        movie_map = {movie.title: movie.id for movie in movies}
        for title, reviews in REVIEW_BANK.items():
            movie_id = movie_map.get(title)
            if movie_id is None:
                continue
            existing_count = len(crud.list_reviews(db, movie_id=movie_id, limit=1000))
            if existing_count >= 10:
                print(f"{title}: 이미 리뷰가 {existing_count}개 이상 존재합니다.")
                continue

            for idx, content in enumerate(reviews, start=1):
                sentiment = analyzer.predict(content)
                crud.create_review(
                    db=db,
                    payload=schemas.ReviewCreate(
                        movie_id=movie_id,
                        author=f"sample_user_{idx}",
                        content=content,
                    ),
                    sentiment_label=sentiment.label,
                    sentiment_score=sentiment.score,
                )
            print(f"{title}: 리뷰 10개 등록 완료")
    finally:
        db.close()


if __name__ == "__main__":
    main()
