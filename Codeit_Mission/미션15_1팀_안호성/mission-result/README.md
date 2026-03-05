# Mission 15

## Directory

- `researcher1`: 학습, EDA, 모델 저장용 코드
- `researcher2`: 추론, docker-compose, Jupyter Notebook 코드

## Researcher 1 (local + Docker Hub)

1. 로컬 개발 환경 동기화

```bash
cd researcher1
uv python pin 3.11
uv sync
```

2. Jupyter 커널 등록

```bash
source .venv/bin/activate
python -m ipykernel install --user --name mission15-r1 --display-name "Python (mission15-r1)"
```

3. Docker 이미지 빌드/푸시

```bash
docker build -t your-dockerhub-id/mission15-trainer:1.0 .
docker push your-dockerhub-id/mission15-trainer:1.0
```

4. 학습 결과 확인(선택)

```bash
docker run --rm your-dockerhub-id/mission15-trainer:1.0
```

`researcher1`는 `pyproject.toml` + `uv.lock` 기준으로 로컬/컨테이너 의존성을 동일하게 유지합니다.
모델 파일은 컨테이너 내부 `/workspace/models/model.pkl`에 생성됩니다.

## Researcher 2 (inference)

1. `researcher2/docker-compose.yml`의 이미지명을 실제 Docker Hub 주소로 수정합니다.
2. 학습 산출물을 공유 볼륨 방식으로 복사합니다.

```bash
cd researcher2
docker compose run --rm trainer_export
docker compose up notebook
```

3. 브라우저에서 `http://localhost:8888`에 접속해 `inference.ipynb`를 실행합니다.
4. 추론 결과는 `researcher2/shared/result.csv`에 저장됩니다.

위 방식은 `trainer_export`와 `notebook`이 호스트의 `./shared`를 같이 마운트해 파일을 전달합니다.

## Researcher 1 <-> Researcher 2 협업/공유 방식

1. Researcher 1이 Docker Hub에 학습 이미지를 배포합니다.
2. Researcher 2는 해당 이미지를 `trainer_export`로 실행해 `model.pkl`, `test.csv`를 `researcher2/shared/`로 가져옵니다.
3. Researcher 2는 `notebook` 컨테이너에서 추론 후 `researcher2/shared/result.csv`를 생성합니다.
4. 결과를 Researcher 1이 확인하려면 `shared/result.csv` 파일을 전달하거나, 같은 폴더를 볼륨으로 마운트해 확인합니다.

예시(Researcher 1이 shared 폴더를 마운트해서 결과 확인):

```bash
docker run --rm \
  -v /path/to/mission-result/researcher2/shared:/shared \
  your-dockerhub-id/mission15-trainer:1.0 \
  python -c "import pandas as pd; print(pd.read_csv('/shared/result.csv').head())"
```

## Optional: docker cp workflow

```bash
docker create --name trainer_tmp your-dockerhub-id/mission15-trainer:1.0
docker start -a trainer_tmp
docker cp trainer_tmp:/workspace/models/model.pkl ./researcher2/shared/model.pkl
docker cp trainer_tmp:/workspace/data/test.csv ./researcher2/shared/test.csv
docker rm trainer_tmp
```

이 방식은 볼륨 공유 없이도 `model.pkl`/`test.csv`를 호스트로 가져올 때 사용할 수 있습니다.

반대로 Researcher 2 결과를 Researcher 1에게 전달할 때도 `docker cp`를 사용할 수 있습니다.

```bash
docker cp researcher2-notebook-1:/shared/result.csv ./researcher2/shared/result.csv
```
