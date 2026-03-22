"""영화와 리뷰에 대한 CRUD 보조 함수."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from . import models, schemas


def _movie_stats_subquery():
    """리뷰 수와 평균 점수를 계산하는 서브쿼리를 만든다.

    Returns:
        SQLAlchemy 서브쿼리.
    """

    return (
        select(
            models.Review.movie_id.label("movie_id"),
            func.count(models.Review.id).label("review_count"),
            func.avg(models.Review.sentiment_score).label("average_sentiment_score"),
        )
        .group_by(models.Review.movie_id)
        .subquery()
    )


def _to_movie_response(movie: models.Movie, review_count: int, average_score: float | None) -> schemas.MovieResponse:
    """영화 ORM 객체를 API 응답 스키마로 변환한다.

    Args:
        movie: 영화 ORM 객체.
        review_count: 리뷰 개수.
        average_score: 감성 점수 평균.

    Returns:
        schemas.MovieResponse: API 응답용 영화 스키마.
    """

    normalized_average = round(float(average_score), 4) if average_score is not None else None
    return schemas.MovieResponse(
        id=movie.id,
        title=movie.title,
        release_date=movie.release_date,
        director=movie.director,
        genre=movie.genre,
        poster_url=movie.poster_url,
        created_at=movie.created_at,
        review_count=int(review_count or 0),
        average_sentiment_score=normalized_average,
        average_rating=round(normalized_average * 5, 2) if normalized_average is not None else None,
    )


def list_movies(db: Session) -> list[schemas.MovieResponse]:
    """리뷰 통계를 포함한 전체 영화 목록을 반환한다.

    Args:
        db: 데이터베이스 세션.

    Returns:
        list[schemas.MovieResponse]: 영화 목록.
    """

    stats = _movie_stats_subquery()
    stmt = (
        select(
            models.Movie,
            func.coalesce(stats.c.review_count, 0),
            stats.c.average_sentiment_score,
        )
        .outerjoin(stats, models.Movie.id == stats.c.movie_id)
        .order_by(models.Movie.created_at.desc())
    )
    rows = db.execute(stmt).all()
    return [_to_movie_response(movie, review_count, average_score) for movie, review_count, average_score in rows]


def get_movie(db: Session, movie_id: int) -> schemas.MovieResponse | None:
    """리뷰 통계를 포함한 특정 영화 한 건을 반환한다.

    Args:
        db: 데이터베이스 세션.
        movie_id: 영화 ID.

    Returns:
        schemas.MovieResponse | None: 영화가 있으면 응답 스키마를 반환한다.
    """

    stats = _movie_stats_subquery()
    stmt = (
        select(
            models.Movie,
            func.coalesce(stats.c.review_count, 0),
            stats.c.average_sentiment_score,
        )
        .outerjoin(stats, models.Movie.id == stats.c.movie_id)
        .where(models.Movie.id == movie_id)
    )
    row = db.execute(stmt).first()
    if row is None:
        return None
    movie, review_count, average_score = row
    return _to_movie_response(movie, review_count, average_score)


def create_movie(db: Session, payload: schemas.MovieCreate) -> schemas.MovieResponse:
    """영화 행을 생성한다.

    Args:
        db: 데이터베이스 세션.
        payload: 영화 생성 요청 데이터.

    Returns:
        schemas.MovieResponse: 생성된 영화 응답.
    """

    movie_data = payload.model_dump()
    movie_data["poster_url"] = str(movie_data["poster_url"])
    movie = models.Movie(**movie_data)
    db.add(movie)
    db.commit()
    db.refresh(movie)
    return _to_movie_response(movie, 0, None)


def delete_movie(db: Session, movie_id: int) -> bool:
    """영화 ID로 영화를 삭제한다.

    Args:
        db: 데이터베이스 세션.
        movie_id: 영화 ID.

    Returns:
        bool: 삭제 성공 여부.
    """

    movie = db.get(models.Movie, movie_id)
    if movie is None:
        return False
    db.delete(movie)
    db.commit()
    return True


def list_reviews(db: Session, movie_id: int | None = None, limit: int | None = None) -> list[schemas.ReviewResponse]:
    """조건에 맞는 리뷰 목록을 반환한다.

    Args:
        db: 데이터베이스 세션.
        movie_id: 영화 ID 필터.
        limit: 최대 조회 개수.

    Returns:
        list[schemas.ReviewResponse]: 리뷰 목록.
    """

    stmt = (
        select(models.Review)
        .options(joinedload(models.Review.movie))
        .order_by(models.Review.created_at.desc())
    )
    if movie_id is not None:
        stmt = stmt.where(models.Review.movie_id == movie_id)
    if limit is not None:
        stmt = stmt.limit(limit)

    reviews = db.scalars(stmt).all()
    return [
        schemas.ReviewResponse(
            id=review.id,
            movie_id=review.movie_id,
            movie_title=review.movie.title,
            author=review.author,
            content=review.content,
            sentiment_label=review.sentiment_label,
            sentiment_score=review.sentiment_score,
            created_at=review.created_at,
        )
        for review in reviews
    ]


def create_review(
    db: Session,
    payload: schemas.ReviewCreate,
    sentiment_label: str,
    sentiment_score: float,
) -> schemas.ReviewResponse:
    """리뷰 행을 생성한다.

    Args:
        db: 데이터베이스 세션.
        payload: 리뷰 생성 요청 데이터.
        sentiment_label: 예측된 감성 라벨.
        sentiment_score: 정규화된 감성 점수.

    Returns:
        schemas.ReviewResponse: 생성된 리뷰 응답.
    """

    review = models.Review(
        movie_id=payload.movie_id,
        author=payload.author,
        content=payload.content,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    db.refresh(review, attribute_names=["movie"])
    return schemas.ReviewResponse(
        id=review.id,
        movie_id=review.movie_id,
        movie_title=review.movie.title,
        author=review.author,
        content=review.content,
        sentiment_label=review.sentiment_label,
        sentiment_score=review.sentiment_score,
        created_at=review.created_at,
    )


def delete_review(db: Session, review_id: int) -> bool:
    """리뷰 ID로 리뷰를 삭제한다.

    Args:
        db: 데이터베이스 세션.
        review_id: 리뷰 ID.

    Returns:
        bool: 삭제 성공 여부.
    """

    review = db.get(models.Review, review_id)
    if review is None:
        return False
    db.delete(review)
    db.commit()
    return True


def get_average_rating(db: Session, movie_id: int) -> schemas.AverageRatingResponse | None:
    """특정 영화의 평균 감성 점수와 환산 평점을 반환한다.

    Args:
        db: 데이터베이스 세션.
        movie_id: 영화 ID.

    Returns:
        schemas.AverageRatingResponse | None: 평점 요약 정보.
    """

    movie = db.get(models.Movie, movie_id)
    if movie is None:
        return None

    stmt = select(
        func.count(models.Review.id),
        func.avg(models.Review.sentiment_score),
    ).where(models.Review.movie_id == movie_id)
    review_count, average_score = db.execute(stmt).one()
    normalized_average = round(float(average_score), 4) if average_score is not None else None
    return schemas.AverageRatingResponse(
        movie_id=movie.id,
        movie_title=movie.title,
        review_count=int(review_count or 0),
        average_sentiment_score=normalized_average,
        average_rating=round(normalized_average * 5, 2) if normalized_average is not None else None,
    )
