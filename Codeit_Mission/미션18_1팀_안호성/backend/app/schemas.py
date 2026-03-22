"""API 입출력용 Pydantic 스키마."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class MovieBase(BaseModel):
    """영화 공통 필드."""

    title: str = Field(..., min_length=1, max_length=200, description="영화 제목")
    release_date: date = Field(..., description="개봉일")
    director: str = Field(..., min_length=1, max_length=100, description="감독")
    genre: str = Field(..., min_length=1, max_length=100, description="장르")
    poster_url: HttpUrl = Field(..., description="포스터 이미지 URL")


class MovieCreate(MovieBase):
    """영화 등록 요청 바디."""


class MovieResponse(MovieBase):
    """영화 응답 스키마."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    review_count: int = Field(..., description="등록된 리뷰 수")
    average_sentiment_score: float | None = Field(
        default=None,
        description="감성 점수 평균(0~1)",
    )
    average_rating: float | None = Field(
        default=None,
        description="감성 점수 평균을 5점 만점으로 환산한 값",
    )


class SentimentResult(BaseModel):
    """감성 분석 결과 스키마."""

    label: str = Field(..., description="positive, neutral, negative 중 하나")
    score: float = Field(..., ge=0.0, le=1.0, description="0~1 범위 감성 점수")
    positive_probability: float = Field(..., ge=0.0, le=1.0)
    neutral_probability: float = Field(..., ge=0.0, le=1.0)
    negative_probability: float = Field(..., ge=0.0, le=1.0)


class ReviewCreate(BaseModel):
    """리뷰 등록 요청 바디."""

    movie_id: int = Field(..., ge=1, description="리뷰를 등록할 영화 ID")
    author: str = Field(..., min_length=1, max_length=100, description="작성자 이름")
    content: str = Field(..., min_length=3, max_length=3000, description="리뷰 내용")


class ReviewResponse(BaseModel):
    """리뷰 응답 스키마."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    movie_id: int
    movie_title: str
    author: str
    content: str
    sentiment_label: str
    sentiment_score: float
    created_at: datetime


class ReviewCreateResponse(BaseModel):
    """리뷰 생성 후 응답 스키마."""

    review: ReviewResponse
    sentiment: SentimentResult


class ReviewListResponse(BaseModel):
    """리뷰 목록 응답 스키마."""

    reviews: list[ReviewResponse]


class MovieListResponse(BaseModel):
    """영화 목록 응답 스키마."""

    movies: list[MovieResponse]


class AverageRatingResponse(BaseModel):
    """영화별 평균 평점 응답 스키마."""

    movie_id: int
    movie_title: str
    review_count: int
    average_sentiment_score: float | None
    average_rating: float | None


class DeleteResponse(BaseModel):
    """삭제 성공 메시지 스키마."""

    message: str
