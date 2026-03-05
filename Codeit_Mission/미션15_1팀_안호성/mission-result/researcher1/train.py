from pathlib import Path

import joblib
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Config 클래스 정의
@dataclass
class Config:
    base_dir: Path = Path.cwd()
    data_path: Path = base_dir / "data"
    model_path: Path = base_dir / "models"
    random_state: int = 42

    def __post_init__(self):
        self.data_path.mkdir(exist_ok=True)
        self.model_path.mkdir(exist_ok=True)

# 훈련 데이터 로드
cfg = Config()
train_df = pd.read_csv(cfg.data_path / 'train.csv')
print(train_df.head())

# 훈련 데이터 정보 확인
print(train_df.describe(include='all'))

# 훈련 데이터 열 타입 확인
print(train_df.info())

# 콜롬별 분류
target_col = 'Performance Index'
categorical_features = ['Extracurricular Activities']
numeric_features = [
    'Hours Studied',
    'Previous Scores',
    'Sleep Hours',
    'Sample Question Papers Practiced',
]

# 훈련 데이터 라벨 분리
train_processed = train_df.drop(columns=[target_col])
label_processed = train_df[target_col]

# 열 전처리기
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('numeric', 'passthrough', numeric_features),
    ]
)

# 앙상블 회귀 모델 파이프라인
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ]
)

# 훈련/검증 데이터 분리 및 모델 훈련
x_train, x_valid, y_train, y_valid = train_test_split(train_processed, label_processed, test_size=0.2, random_state=42)
model.fit(x_train, y_train)

# 검증 및 RMSE 계산
pred = model.predict(x_valid)
rmse = root_mean_squared_error(y_valid, pred)
print(rmse)

# 배포용 모델 훈련 및 저장
model.fit(train_processed, label_processed)
joblib.dump(model, cfg.model_path / 'model.pkl')
print(cfg.model_path / 'model.pkl')