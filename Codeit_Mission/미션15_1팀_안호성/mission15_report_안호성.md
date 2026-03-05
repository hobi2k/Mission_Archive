# 미션 15 결과 보고서

팀: 1팀  
이름: 안호성  
작성일: 2026-03-05  
과제 주제: Docker 기반 협업 워크플로우 설계 및 모델 학습/추론 분업 구현

## 1. 미션 개요

본 과제는 연구자 1(학습)과 연구자 2(추론)의 역할을 분리하여, Docker 환경에서 모델 산출물을 안전하게 전달하고 재현 가능한 추론 파이프라인을 구축하는 것을 목표로 한다.  
연구자 1은 `train.csv` 기반 전처리·EDA·회귀 모델 학습을 수행하고 `model.pkl`을 생성한다. 연구자 2는 researcher1의 이미지를 실행하여 `model.pkl`과 `test.csv`를 확보한 뒤, Jupyter 환경에서 추론해 `result.csv`를 생성한다.

## 2. Docker Hub URL

- 저장소: https://hub.docker.com/r/ahnhs2k/mission15-trainer
- 사용 태그: `ahnhs2k/mission15-trainer:1.0`
- researcher2 `docker-compose.yml`에서 `trainer_export` 이미지로 위 태그 사용

## 3. 연구자 1의 데이터 전처리 및 모델링 결과 요약

### 3.1 데이터 및 목표

- 학습 데이터: `train.csv` (7000행)
- 입력 변수:
  - Hours Studied
  - Previous Scores
  - Extracurricular Activities
  - Sleep Hours
  - Sample Question Papers Practiced
- 타깃 변수: `Performance Index`

### 3.2 전처리 설계

- `ColumnTransformer`를 사용하여 컬럼 타입별 전처리를 분리 적용
- 범주형 컬럼:
  - `Extracurricular Activities` -> `OneHotEncoder(handle_unknown="ignore")`
- 수치형 컬럼:
  - `Hours Studied`, `Previous Scores`, `Sleep Hours`, `Sample Question Papers Practiced` -> `passthrough`
- 전처리기와 회귀 모델을 `Pipeline`으로 묶어 학습/추론 일관성 확보

### 3.3 모델 및 검증

- 모델: `RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)`
- 검증 방식: `train_test_split(test_size=0.2, random_state=42)`
- 평가 지표: RMSE
- 검증 결과: **RMSE = 2.2456**
- 해석:
  - `Performance Index` 범위를 고려할 때 평균 약 2~3점 수준 오차로, 미션 기준에서 양호한 성능을 확인함

### 3.4 최종 산출물

- 검증 후 전체 학습 데이터로 재학습하여 배포용 모델 생성
- 모델 저장 경로: `/workspace/models/model.pkl` (컨테이너 기준)
- 연구자 2 전달 파일: `model.pkl`, `test.csv`

## 4. 코드 아키텍처 도식 및 설명

### 4.1 아키텍처 도식

```text
Researcher 1 Docker Image (train.py, data/train.csv, data/test.csv)
        -> trainer_export 서비스 실행
        -> /shared/model.pkl, /shared/test.csv 복사

Host shared 볼륨 (researcher2/shared)
        -> notebook 서비스(Jupyter)에서 동일 경로 접근
        -> inference.ipynb 실행 후 /shared/result.csv 저장
```

### 4.2 구성 요소

- `researcher1`
  - `train.py`, `Dockerfile`, `pyproject.toml`, `uv.lock`
- `researcher2`
  - `docker-compose.yml`, `Dockerfile`, `inference.ipynb`
- 공유 지점
  - `researcher2/shared` (model/test/result 파일 교환)

### 4.3 실행 절차

1. 연구자 1이 trainer 이미지를 Docker Hub에 빌드/푸시
2. 연구자 2가 `trainer_export` 실행으로 `model.pkl`, `test.csv`를 `shared`로 복사
3. 연구자 2가 `notebook` 컨테이너에서 `inference.ipynb` 실행 후 `result.csv` 생성
4. `shared` 폴더를 통해 결과 파일 확인/공유

### 4.4 파일 전달 전략

- 기본 전략: `docker-compose` 공용 볼륨(`shared`) 사용
- 대안 전략: `docker cp`로 컨테이너 내부 파일 직접 복사
- 예시:
  - `docker cp trainer_tmp:/workspace/models/model.pkl ./shared/model.pkl`

## 5. 환경 일관성 및 재현성 관리

- Python 버전: 연구자 컨테이너 모두 3.11 기준 통일
- 의존성 관리: `pyproject.toml + uv.lock` 기반 동기화
- researcher2는 `/opt/venv` 사용:
  - bind mount(`.:/workspace`)로 `.venv`가 덮어써지는 문제 방지
- 위 구성으로 동일 명령 실행 시 동일 결과 재현 가능

## 6. 결론

본 과제에서는 역할 분리(학습/추론), 컨테이너 기반 환경 통일, 파일 전달 자동화를 통해 Docker 협업 워크플로우를 구현하였다.  
학습 산출물(`model.pkl`)과 테스트 데이터(`test.csv`)를 researcher2에서 안정적으로 활용해 최종 추론 결과(`result.csv`)를 생성했으며, 공유 볼륨 및 `docker cp` 전략을 모두 적용 가능한 구조로 설계하였다.
