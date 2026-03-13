# 미션 17. 손글씨 숫자 인식 Streamlit 서비스

이 폴더는 사용자가 캔버스에 손으로 숫자를 그리면, MNIST ONNX 모델로 숫자를 예측하는 Streamlit 웹 서비스를 담고 있습니다.

## 구성 파일

- `app.py`: Streamlit 앱 본체
- `assets/style.css`: 화면 카드, 배경, metric 스타일 정의
- `.streamlit/config.toml`: Streamlit 테마 및 실행 설정
- `requirements.txt`: 파이썬 의존성
- `Dockerfile`: 컨테이너 빌드 설정
- `.dockerignore`: Docker 이미지 제외 파일 목록
- `미션17_1팀_안호성_요약보고서.md`: 보고서 원문
- `미션17_1팀_안호성_요약보고서.pdf`: 제출용 보고서 PDF

## 주요 기능

- 입력 캔버스: `streamlit-drawable-canvas` 기반 숫자 입력
- 전처리 이미지 표시: 28x28 MNIST 입력 형태로 변환된 이미지 시각화
- 추론 결과 표시: 0~9 확률 막대 차트
- 이미지 저장소: 저장한 손그림과 예측 결과를 아래 갤러리에서 확인
- 모델 관리: `app.py` 내부에서 ONNX 모델 자동 다운로드 + `st.cache_resource` 캐싱
- UI 구성: `.streamlit/config.toml` 테마 + `assets/style.css` 카드형 레이아웃 적용

## 로컬 실행

```bash
cd Mission_Archive/Codeit_Mission/미션17_1팀_안호성
uv venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 됩니다.

## 화면 구성

- 상단 요약 카드: 모델 형식, 추론 엔진, 입력 규격 표시
- 좌측 대형 캔버스: 손글씨 입력과 브러시 두께 조절
- 우측 결과 패널: 예측 숫자, 최대 확률, 확률 차트, 전처리 이미지, 사용 안내
- 하단 저장소 갤러리: 저장된 손그림과 예측 결과 확인

## Docker 실행

```bash
cd Mission_Archive/Codeit_Mission/미션17_1팀_안호성
docker build -t mnist-onnx-streamlit:latest .
docker run --rm -p 8501:8501 mnist-onnx-streamlit:latest
```

## Docker Hub 배포

아래 예시는 Docker Hub 계정 ID가 `your_dockerhub_id`인 경우 기준입니다.

```bash
cd Mission_Archive/Codeit_Mission/미션17_1팀_안호성

# 1. Docker Hub 로그인
docker login

# 2. 로컬 이미지를 Docker Hub용 이름으로 태깅
docker tag mnist-onnx-streamlit:latest your_dockerhub_id/mnist-onnx-streamlit:latest

# 3. Docker Hub로 업로드
docker push your_dockerhub_id/mnist-onnx-streamlit:latest
```

업로드가 끝나면 Docker Hub에서 아래 형식의 URL로 이미지를 확인할 수 있습니다.

```text
https://hub.docker.com/r/your_dockerhub_id/mnist-onnx-streamlit
```

## Docker Hub URL

- 업로드 전 placeholder: `https://hub.docker.com/r/<dockerhub_id>/mnist-onnx-streamlit`
- 실제 업로드 시 위 값을 본인 Docker Hub 경로로 교체하면 됩니다.
