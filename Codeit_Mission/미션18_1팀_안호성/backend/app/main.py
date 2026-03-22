"""미션 18용 FastAPI 애플리케이션."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import Base, SessionLocal, engine
from .sentiment import get_sentiment_analyzer


@asynccontextmanager
async def lifespan(_: FastAPI):
    """테이블을 생성하고 앱 시작 준비를 수행한다.

    Args:
        _: FastAPI 앱 객체 자리표시자.
    """

    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Mission 18 Movie Review API",
    description=(
        "영화 정보 등록, 리뷰 등록, 감성 분석, 평균 평점 조회를 담당하는 "
        "FastAPI 백엔드입니다."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    """요청 단위 데이터베이스 세션을 반환한다.

    Returns:
        Session: SQLAlchemy 세션.
    """

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    """기본 서비스 메시지를 반환한다.

    Returns:
        dict[str, str]: 서비스 상태 메시지.
    """

    return {"message": "Mission 18 Movie Review API is running."}


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """헬스체크 결과를 반환한다.

    Returns:
        dict[str, str]: 상태 확인 결과.
    """

    return {"status": "ok"}


@app.get("/movies", response_model=schemas.MovieListResponse, tags=["Movies"], summary="영화 전체 조회")
def list_movies(db: Session = Depends(get_db)) -> schemas.MovieListResponse:
    """등록된 전체 영화 목록을 조회한다.

    Args:
        db: 데이터베이스 세션.

    Returns:
        schemas.MovieListResponse: 영화 목록 응답.
    """

    return schemas.MovieListResponse(movies=crud.list_movies(db))


@app.post(
    "/movies",
    response_model=schemas.MovieResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Movies"],
    summary="영화 등록",
)
def create_movie(payload: schemas.MovieCreate, db: Session = Depends(get_db)) -> schemas.MovieResponse:
    """영화를 등록한다.

    Args:
        payload: 영화 등록 데이터.
        db: 데이터베이스 세션.

    Returns:
        schemas.MovieResponse: 생성된 영화 응답.
    """

    try:
        return crud.create_movie(db, payload)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="이미 등록된 영화 제목입니다.") from exc


@app.get("/movies/{movie_id}", response_model=schemas.MovieResponse, tags=["Movies"], summary="특정 영화 조회")
def get_movie(movie_id: int, db: Session = Depends(get_db)) -> schemas.MovieResponse:
    """특정 영화 한 건을 조회한다.

    Args:
        movie_id: 영화 ID.
        db: 데이터베이스 세션.

    Returns:
        schemas.MovieResponse: 영화 응답 데이터.
    """

    movie = crud.get_movie(db, movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")
    return movie


@app.delete("/movies/{movie_id}", response_model=schemas.DeleteResponse, tags=["Movies"], summary="영화 삭제")
def delete_movie(movie_id: int, db: Session = Depends(get_db)) -> schemas.DeleteResponse:
    """특정 영화를 삭제한다.

    Args:
        movie_id: 영화 ID.
        db: 데이터베이스 세션.

    Returns:
        schemas.DeleteResponse: 삭제 결과 메시지.
    """

    deleted = crud.delete_movie(db, movie_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")
    return schemas.DeleteResponse(message="영화가 삭제되었습니다.")


@app.get(
    "/movies/{movie_id}/reviews",
    response_model=schemas.ReviewListResponse,
    tags=["Reviews"],
    summary="특정 영화 리뷰 조회",
)
def get_movie_reviews(
    movie_id: int,
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> schemas.ReviewListResponse:
    """특정 영화의 리뷰 목록을 조회한다.

    Args:
        movie_id: 영화 ID.
        limit: 최대 리뷰 개수.
        db: 데이터베이스 세션.

    Returns:
        schemas.ReviewListResponse: 리뷰 목록 응답.
    """

    if crud.get_movie(db, movie_id) is None:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")
    return schemas.ReviewListResponse(reviews=crud.list_reviews(db, movie_id=movie_id, limit=limit))


@app.get(
    "/movies/{movie_id}/rating",
    response_model=schemas.AverageRatingResponse,
    tags=["Movies"],
    summary="평균 평점 조회",
)
def get_movie_rating(movie_id: int, db: Session = Depends(get_db)) -> schemas.AverageRatingResponse:
    """특정 영화의 감성 기반 평균 평점을 조회한다.

    Args:
        movie_id: 영화 ID.
        db: 데이터베이스 세션.

    Returns:
        schemas.AverageRatingResponse: 평점 요약 정보.
    """

    result = crud.get_average_rating(db, movie_id)
    if result is None:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")
    return result


@app.get("/reviews", response_model=schemas.ReviewListResponse, tags=["Reviews"], summary="리뷰 전체 조회")
def list_reviews(
    limit: int = Query(default=10, ge=1, le=100, description="가져올 리뷰 개수"),
    db: Session = Depends(get_db),
) -> schemas.ReviewListResponse:
    """최근 리뷰 목록을 전체 기준으로 조회한다.

    Args:
        limit: 최대 리뷰 개수.
        db: 데이터베이스 세션.

    Returns:
        schemas.ReviewListResponse: 리뷰 목록 응답.
    """

    return schemas.ReviewListResponse(reviews=crud.list_reviews(db, limit=limit))


@app.post(
    "/reviews",
    response_model=schemas.ReviewCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Reviews"],
    summary="리뷰 등록 및 감성 분석",
)
def create_review(payload: schemas.ReviewCreate, db: Session = Depends(get_db)) -> schemas.ReviewCreateResponse:
    """리뷰를 등록하고 감성 분석을 함께 수행한다.

    Args:
        payload: 리뷰 등록 데이터.
        db: 데이터베이스 세션.

    Returns:
        schemas.ReviewCreateResponse: 생성된 리뷰와 감성 분석 결과.
    """

    if crud.get_movie(db, payload.movie_id) is None:
        raise HTTPException(status_code=404, detail="리뷰를 등록할 영화가 존재하지 않습니다.")

    sentiment = get_sentiment_analyzer().predict(payload.content)
    review = crud.create_review(
        db=db,
        payload=payload,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
    )
    return schemas.ReviewCreateResponse(
        review=review,
        sentiment=schemas.SentimentResult(
            label=sentiment.label,
            score=sentiment.score,
            positive_probability=sentiment.positive_probability,
            neutral_probability=sentiment.neutral_probability,
            negative_probability=sentiment.negative_probability,
        ),
    )


@app.delete("/reviews/{review_id}", response_model=schemas.DeleteResponse, tags=["Reviews"], summary="리뷰 삭제")
def delete_review(review_id: int, db: Session = Depends(get_db)) -> schemas.DeleteResponse:
    """특정 리뷰를 삭제한다.

    Args:
        review_id: 리뷰 ID.
        db: 데이터베이스 세션.

    Returns:
        schemas.DeleteResponse: 삭제 결과 메시지.
    """

    deleted = crud.delete_review(db, review_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="리뷰를 찾을 수 없습니다.")
    return schemas.DeleteResponse(message="리뷰가 삭제되었습니다.")
