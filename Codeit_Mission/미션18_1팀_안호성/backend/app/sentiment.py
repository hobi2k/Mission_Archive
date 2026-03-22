"""Hugging Face Transformers 기반 감성 분석 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from transformers import pipeline


MODEL_NAME = "Copycats/koelectra-base-v3-generalized-sentiment-analysis"


@dataclass(slots=True)
class SentimentPrediction:
    """정규화된 감성 분석 결과."""

    label: str
    score: float
    positive_probability: float
    neutral_probability: float
    negative_probability: float


class SentimentAnalyzer:
    """텍스트 분류 파이프라인을 감싸는 감성 분석기.

    Args:
        model_name: 사용할 Hugging Face 모델 ID.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._classifier = None

    def _load(self):
        """분류 모델을 필요할 때만 지연 로드한다.

        Returns:
            transformers pipeline 인스턴스.
        """

        if self._classifier is None:
            self._classifier = pipeline(
                task="text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,
            )
        return self._classifier

    def predict(self, text: str) -> SentimentPrediction:
        """입력 텍스트를 감성 분석하고 결과를 정규화한다.

        Args:
            text: 리뷰 본문.

        Returns:
            SentimentPrediction: 정규화된 라벨과 확률 정보.
        """

        classifier = self._load()
        raw_results = classifier(text, truncation=True, max_length=512)[0]
        id2label = getattr(getattr(classifier, "model", None), "config", None)
        id2label = getattr(id2label, "id2label", {}) if id2label is not None else {}
        probs = self._normalize(raw_results, id2label)
        label = max(probs, key=probs.get)
        score = probs["positive"] * 1.0 + probs["neutral"] * 0.5 + probs["negative"] * 0.0
        return SentimentPrediction(
            label=label,
            score=round(score, 4),
            positive_probability=round(probs["positive"], 4),
            neutral_probability=round(probs["neutral"], 4),
            negative_probability=round(probs["negative"], 4),
        )

    @staticmethod
    def _normalize(results: list[dict], id2label: dict[int, str] | dict[str, str] | None = None) -> dict[str, float]:
        """모델 출력 라벨을 positive/neutral/negative로 정규화한다.

        Args:
            results: 파이프라인 원본 출력.
            id2label: 모델 설정에 들어 있는 라벨 매핑 정보.

        Returns:
            dict[str, float]: 정규화된 확률 딕셔너리.
        """

        mapped = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for item in results:
            raw_label = str(item["label"]).lower()
            score = float(item["score"])

            normalized_label = raw_label
            if raw_label.startswith("label_"):
                label_index = raw_label.split("_")[-1]
                mapped_label = (
                    id2label.get(int(label_index))
                    if isinstance(id2label, dict) and label_index.isdigit() and int(label_index) in id2label
                    else id2label.get(label_index)
                    if isinstance(id2label, dict)
                    else None
                )
                if mapped_label is not None:
                    normalized_label = str(mapped_label).lower()

            if "pos" in normalized_label or "긍정" in normalized_label:
                mapped["positive"] += score
            elif "neu" in normalized_label or "중립" in normalized_label:
                mapped["neutral"] += score
            elif "neg" in normalized_label or "부정" in normalized_label:
                mapped["negative"] += score

        if sum(mapped.values()) == 0.0 and len(results) == 2:
            sorted_results = sorted(results, key=lambda item: str(item["label"]))
            first_score = float(sorted_results[0]["score"])
            second_score = float(sorted_results[1]["score"])
            mapped["negative"] = first_score
            mapped["positive"] = second_score

        total = sum(mapped.values()) or 1.0
        return {key: value / total for key, value in mapped.items()}


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> SentimentAnalyzer:
    """캐시된 감성 분석기 인스턴스를 반환한다.

    Returns:
        SentimentAnalyzer: 싱글턴 분석기 인스턴스.
    """

    return SentimentAnalyzer()
