# Mission 18 Backend

FastAPI 기반 영화 / 리뷰 / 감성 분석 백엔드입니다.  
영화 정보와 리뷰 데이터는 SQLite에 저장하고, 리뷰 감성 분석은 Hugging Face Transformers 기반 한국어 분류 모델로 수행합니다.

## 1. 폴더 구조

```text
backend/
├─ app/
│  ├─ main.py          # FastAPI 엔트리포인트
│  ├─ database.py      # SQLite / SQLAlchemy 설정
│  ├─ models.py        # ORM 모델
│  ├─ schemas.py       # 요청/응답 스키마
│  ├─ crud.py          # DB 처리 로직
│  └─ sentiment.py     # 감성 분석 로직
├─ scripts/
│  ├─ seed_sample_data.py
│  └─ seed_reviews.py
├─ data/
│  └─ movies.db        # 실행 후 생성되는 SQLite 파일
└─ requirements.txt
```

## 2. 실행 방법

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run uvicorn app.main:app --reload
```

기본 실행 주소:

- API 서버: `http://127.0.0.1:8000`

## 3. API 문서 접속 방법

FastAPI 실행 후 아래 주소로 접속하면 문서를 볼 수 있습니다.

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

제출용 캡처는 보통 Swagger UI(`docs`) 기준으로 진행하면 됩니다.

## 4. 헬스체크

브라우저 또는 터미널에서 아래 주소로 백엔드 정상 동작 여부를 확인할 수 있습니다.

- `GET http://127.0.0.1:8000/health`

정상 응답 예시:

```json
{
  "status": "ok"
}
```

## 5. 샘플 데이터 입력

영화 3개와 영화별 리뷰 10개 이상을 빠르게 채우려면 아래 순서로 실행합니다.

```bash
cd backend
uv run python scripts/seed_sample_data.py
uv run python scripts/seed_reviews.py
```

샘플 데이터 구성:

- 영화 3개
  - 기생충
  - 올드보이
  - 인터스텔라
- 각 영화별 리뷰 10개

## 6. 데이터베이스 파일 위치

SQLite 파일은 아래 경로에 생성됩니다.

```text
backend/data/movies.db
```

데이터를 초기화하고 다시 시드하려면, 백엔드를 중지한 뒤 이 파일을 삭제하고 시드 스크립트를 다시 실행하면 됩니다.

## 7. 감성 분석 모델

- 모델: `Copycats/koelectra-base-v3-generalized-sentiment-analysis`
- 사용 이유:
  - 한국어 텍스트 분류용으로 적합
  - FastAPI 내부에서 바로 호출 가능
  - 리뷰 감성 분석에 필요한 positive / neutral / negative 분류에 적합
  - 대형 생성 모델보다 가볍게 시연 가능

감성 분석 결과는 다음 형태로 저장됩니다.

- `sentiment_label`
  - `positive`
  - `neutral`
  - `negative`
- `sentiment_score`
  - 0~1 범위 점수

평균 평점은 이 `sentiment_score` 평균을 5점 만점으로 환산해 계산합니다.

## 8. 주요 API

### 영화 API

- `GET /movies`
  - 전체 영화 목록 조회
- `POST /movies`
  - 영화 등록
- `GET /movies/{movie_id}`
  - 특정 영화 조회
- `DELETE /movies/{movie_id}`
  - 특정 영화 삭제
- `GET /movies/{movie_id}/rating`
  - 특정 영화 평균 평점 조회

### 리뷰 API

- `GET /reviews`
  - 전체 리뷰 조회
- `POST /reviews`
  - 리뷰 등록 및 감성 분석
- `DELETE /reviews/{review_id}`
  - 특정 리뷰 삭제
- `GET /movies/{movie_id}/reviews`
  - 특정 영화의 리뷰 조회

## 9. 제출용 확인 포인트

Swagger UI 캡처 전 아래 항목을 확인하면 좋습니다.

1. `/docs` 접속이 정상인지 확인
2. `POST /movies`로 영화 등록이 되는지 확인
3. `POST /reviews`로 리뷰 등록 시 감성 분석이 같이 수행되는지 확인
4. `GET /movies`에서 리뷰 수와 평균 평점이 반영되는지 확인
5. `GET /reviews`에서 최근 리뷰가 내려오는지 확인

## 10. 프론트엔드와 연결

Streamlit 프론트엔드는 기본적으로 아래 백엔드 주소를 사용합니다.

```text
http://127.0.0.1:8000
```

백엔드가 다른 포트에서 실행되면 Streamlit 사이드바의 `FastAPI URL` 입력창에서 주소를 바꾸면 됩니다.
