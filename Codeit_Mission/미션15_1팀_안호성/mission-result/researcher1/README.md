# 학생 성적 회귀 모델

## 1) uv 환경 준비

```bash
cd /home/hosung/pytorch-demo/Mission_Archive/Codeit_Mission/미션15_1팀_안호성/mission-result/researcher1
uv python pin 3.11
uv sync
source .venv/bin/activate
python --version
```

## 2) Jupyter 커널 등록

```bash
uv run ipykernel install --user --name m15 --display-name "m15"
```

노트북에서 `Kernel -> Change kernel -> Python (mission15-r1)`를 선택합니다.

## 3) 학습 스크립트 실행

```bash
uv run train.py
```

모델 출력:

- `/workspace/models/model.pkl` (컨테이너 경로)
- 로컬 실행 시 `./models/model.pkl`

## 4) Docker 이미지 빌드/배포

```bash
docker build -t your-dockerhub-id/mission15-trainer:1.0 .
docker push your-dockerhub-id/mission15-trainer:1.0
```

## 5) Docker 이미지 사용

```bash
docker run --rm your-dockerhub-id/mission15-trainer:1.0
```

컨테이너에서 학습이 끝나면 `model.pkl`과 `test.csv`를 researcher2에서 공유 볼륨 또는 `docker cp`로 가져와 추론합니다.
