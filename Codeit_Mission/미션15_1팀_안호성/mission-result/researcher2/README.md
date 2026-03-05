# Researcher 2 Guide

## 1) 사전 준비

1. `docker-compose.yml`의 `your-dockerhub-id/mission15-trainer:1.0`를 실제 이미지로 수정합니다.
2. 작업 경로로 이동합니다.

```bash
cd /home/hosung/pytorch-demo/Mission_Archive/Codeit_Mission/미션15_1팀_안호성/mission-result/researcher2
```

## 2) 모델/테스트 파일 전달 (볼륨 공유 방식)

`trainer_export` 컨테이너가 researcher1 이미지를 실행하고 `model.pkl`, `test.csv`를 `shared/`로 복사합니다.

```bash
docker compose run --rm trainer_export
```

복사 결과 확인:

```bash
ls -lh shared
```

## 3) Jupyter Notebook 컨테이너 실행

```bash
docker compose up notebook
```

브라우저에서 `http://localhost:8888` 접속 후 `inference.ipynb`를 실행합니다.

## 4) 결과 확인

노트북 실행이 끝나면 아래 파일이 생성됩니다.

- `shared/result.csv`

```bash
ls -lh shared/result.csv
```

## 5) Researcher 1과 결과 공유

Researcher 2가 만든 `shared/result.csv`는 다음 방식으로 Researcher 1과 공유할 수 있습니다.

1. 파일 전달: `shared/result.csv`를 직접 전달
2. 공통 경로 마운트: Researcher 1이 동일한 `shared` 폴더를 볼륨 마운트해 확인

예시(Researcher 1 측 확인 명령):

```bash
docker run --rm \
  -v /path/to/mission-result/researcher2/shared:/shared \
  your-dockerhub-id/mission15-trainer:1.0 \
  python -c "import pandas as pd; print(pd.read_csv('/shared/result.csv').head())"
```

## 6) 대안: docker cp 방식

볼륨 공유 대신 컨테이너에서 호스트로 파일을 직접 복사할 수 있습니다.

```bash
docker create --name trainer_tmp your-dockerhub-id/mission15-trainer:1.0
docker start -a trainer_tmp
docker cp trainer_tmp:/workspace/models/model.pkl ./shared/model.pkl
docker cp trainer_tmp:/workspace/data/test.csv ./shared/test.csv
docker rm trainer_tmp
```

노트북 컨테이너에서 결과 파일만 꺼낼 때:

```bash
docker cp researcher2-notebook-1:/shared/result.csv ./shared/result.csv
```
