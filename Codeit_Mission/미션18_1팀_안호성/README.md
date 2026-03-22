# 미션 18 - 영화 리뷰 감성 분석 웹 애플리케이션

미션 18 제출 폴더입니다.  
영화 정보 등록, 사용자 리뷰 저장, 리뷰 감성 분석, 평균 평점 표시 기능을 포함한 웹 애플리케이션을 구현했습니다. 프론트엔드는 Streamlit, 백엔드는 FastAPI로 구성했고, 모든 데이터는 백엔드의 SQLite 데이터베이스에서 관리합니다.

## 1. 제출 폴더 구성

```text
미션18_1팀_안호성/
├─ backend/                        # FastAPI 백엔드
├─ frontend/                       # Streamlit 프론트엔드
├─ 미션18_1팀_안호성_요약보고서.md  # 제출용 보고서 본문
├─ mission18_architecture_and_erd.md
├─ mission18_capture_checklist.md
├─ backend01.png ~ backend03.png   # FastAPI Docs 및 백엔드 캡처
├─ frontend01.png ~ frontend02.png # 서비스 동작 캡처
└─ README.md
```

## 2. 구현 범위

### 프론트엔드

- 영화 목록 표시
- 영화 등록
- 리뷰 등록
- 감성 분석 결과 표시
- 최근 10개 리뷰 표시
- 영화별 평균 평점(5점 환산) 표시

### 백엔드

- 영화 등록 / 전체 조회 / 특정 조회 / 삭제
- 리뷰 등록 / 전체 조회 / 특정 영화 리뷰 조회 / 삭제
- 리뷰 등록 시 감성 분석 자동 수행
- 영화별 평균 평점 조회
- SQLite 기반 데이터 저장

## 3. 기술 스택

- 프론트엔드: Streamlit
- 백엔드: FastAPI
- 데이터베이스: SQLite
- ORM: SQLAlchemy
- 감성 분석: Hugging Face Transformers
- 실행 환경 관리: uv

## 4. 실행 순서

### 4-1. 백엔드 실행

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run uvicorn app.main:app --reload
```

백엔드 기본 주소:

- `http://127.0.0.1:8000`

Swagger UI:

- `http://127.0.0.1:8000/docs`

ReDoc:

- `http://127.0.0.1:8000/redoc`

### 4-2. 샘플 데이터 입력

```bash
cd backend
uv run python scripts/seed_sample_data.py
uv run python scripts/seed_reviews.py
```

샘플 데이터 구성:

- 영화 3개 등록
- 영화별 리뷰 10개 등록

### 4-3. 프론트엔드 실행

```bash
cd frontend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run streamlit run app.py
```

Streamlit 기본 주소:

- `http://localhost:8501`

## 5. 감성 분석 모델

- 모델: `Copycats/koelectra-base-v3-generalized-sentiment-analysis`

선택 이유:

- 한국어 리뷰 감성 분류에 적합
- 대형 생성 모델 서빙 없이 FastAPI 내부에서 직접 호출 가능
- 분류 모델이라 시연용 서비스에 필요한 추론 비용이 비교적 낮음
- 리뷰를 `positive / neutral / negative`로 정규화하기 쉬움

## 6. 포스터 URL 검증

샘플 데이터에 사용한 포스터 링크는 실제 접근 가능한 URL로 검증 후 반영했다.  
초기 Wikimedia 한국어 경로는 일부 404였고, 최종적으로 English Wikipedia 페이지의 `og:image` 값을 기준으로 교체했다.

검증 완료 링크:

- 기생충: `https://upload.wikimedia.org/wikipedia/en/5/53/Parasite_%282019_film%29.png`
- 올드보이: `https://upload.wikimedia.org/wikipedia/en/6/67/Oldboykoreanposter.jpg`
- 인터스텔라: `https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg`

## 7. 제출 문서 작성용 파일

- [미션18_1팀_안호성_요약보고서.md](/home/hosung/pytorch-demo/Mission_Archive/Codeit_Mission/미션18_1팀_안호성/미션18_1팀_안호성_요약보고서.md)
  - 제출용 보고서 본문
- [mission18_architecture_and_erd.md](/home/hosung/pytorch-demo/Mission_Archive/Codeit_Mission/미션18_1팀_안호성/mission18_architecture_and_erd.md)
  - 서비스 구조도, 백엔드 구조, ERD 정리
- [mission18_capture_checklist.md](/home/hosung/pytorch-demo/Mission_Archive/Codeit_Mission/미션18_1팀_안호성/mission18_capture_checklist.md)
  - 제출용 캡처 체크리스트

## 8. 캡처 파일 정리

현재 제출 폴더에는 아래 캡처 파일이 함께 정리되어 있습니다.

- `backend01.png`
- `backend02.png`
- `backend03.png`
- `frontend01.png`
- `frontend02.png`

위 파일들은 보고서 PDF에 삽입하거나 제출 검수용 이미지로 활용할 수 있습니다.

## 9. 제출 전 최종 체크

- [ ] `미션18_1팀_안호성_요약보고서.md`를 PDF로 변환했는지 확인
- [ ] FastAPI `/docs` 캡처를 보고서에 삽입했는지 확인
- [ ] 서비스 동작 화면 캡처를 보고서에 삽입했는지 확인
- [ ] 영화 3개 이상 등록 확인
- [ ] 각 영화당 리뷰 10개 이상 등록 확인
- [ ] `backend`, `frontend` 폴더가 모두 포함되어 있는지 확인
