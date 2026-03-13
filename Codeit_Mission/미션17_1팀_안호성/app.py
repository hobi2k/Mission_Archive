from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
GALLERY_DIR = APP_DIR / "saved_images"
GALLERY_INDEX_PATH = GALLERY_DIR / "gallery_index.json"
MODEL_PATH = MODEL_DIR / "mnist-8.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
STYLE_PATH = APP_DIR / "assets" / "style.css"


def inject_styles() -> None:
    """페이지 전용 스타일을 주입한다."""
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def ensure_directories() -> None:
    """앱 실행에 필요한 디렉터리를 미리 생성한다."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    if not GALLERY_INDEX_PATH.exists():
        GALLERY_INDEX_PATH.write_text("[]", encoding="utf-8")


def download_model_if_needed(model_url: str, target_path: Path) -> Path:
    """ONNX MNIST 모델이 없으면 다운로드하고, 있으면 기존 파일을 재사용한다."""
    ensure_directories()
    if target_path.exists():
        return target_path

    with requests.get(model_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return target_path


@st.cache_resource(show_spinner="ONNX MNIST 모델을 로드하는 중입니다...")
def load_onnx_session() -> ort.InferenceSession:
    """모델 다운로드와 ONNX Runtime 세션 생성을 캐싱한다."""
    model_path = download_model_if_needed(MODEL_URL, MODEL_PATH)
    return ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])


def preprocess_canvas_image(image_rgba: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """캔버스 이미지를 MNIST 입력 형태([1, 1, 28, 28])로 전처리한다.

    처리 순서:
    1. RGBA 이미지를 grayscale로 변환
    2. 실제로 그려진 영역의 bounding box 추출
    3. 정사각형 비율로 패딩 후 20x20으로 축소
    4. 28x28 검은 배경 중앙에 배치
    5. float32 / [0, 1] 범위 텐서로 변환
    """
    pil_image = Image.fromarray(image_rgba.astype(np.uint8), mode="RGBA").convert("L")
    gray = np.array(pil_image, dtype=np.uint8)

    # 캔버스는 검은 배경 + 흰색 선이므로, 밝은 픽셀만 숫자 영역으로 간주한다.
    ys, xs = np.where(gray > 10)
    if len(xs) == 0 or len(ys) == 0:
        empty = Image.new("L", (28, 28), color=0)
        tensor = np.zeros((1, 1, 28, 28), dtype=np.float32)
        return empty, tensor

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = pil_image.crop((x_min, y_min, x_max + 1, y_max + 1))

    # 숫자 가장자리 손실을 막기 위해 약간의 여백을 추가한다.
    crop_w, crop_h = cropped.size
    side = max(crop_w, crop_h) + 8
    square = Image.new("L", (side, side), color=0)
    paste_x = (side - crop_w) // 2
    paste_y = (side - crop_h) // 2
    square.paste(cropped, (paste_x, paste_y))

    resized = square.resize((20, 20), Image.Resampling.LANCZOS)
    canvas_28 = Image.new("L", (28, 28), color=0)
    canvas_28.paste(resized, (4, 4))

    tensor = np.asarray(canvas_28, dtype=np.float32) / 255.0
    tensor = tensor[np.newaxis, np.newaxis, :, :]
    return canvas_28, tensor


def softmax(logits: np.ndarray) -> np.ndarray:
    """로짓을 확률 분포로 변환한다."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def run_inference(session: ort.InferenceSession, input_tensor: np.ndarray) -> tuple[int, np.ndarray]:
    """전처리된 입력으로 추론하고 예측 클래스와 확률을 반환한다."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: input_tensor})[0]
    probs = softmax(logits[0])
    pred = int(np.argmax(probs))
    return pred, probs


def load_gallery_entries() -> list[dict]:
    """저장된 이미지 메타데이터를 읽어 최신순으로 반환한다."""
    ensure_directories()
    try:
        data = json.loads(GALLERY_INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(reversed(data))
    except json.JSONDecodeError:
        pass
    return []


def save_gallery_entry(raw_rgba: np.ndarray, processed_image: Image.Image, pred: int, probs: np.ndarray) -> None:
    """현재 손그림과 전처리 결과, 예측 결과를 로컬 저장소에 저장한다."""
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"mnist_draw_{timestamp}"
    raw_path = GALLERY_DIR / f"{base_name}_raw.png"
    processed_path = GALLERY_DIR / f"{base_name}_processed.png"

    Image.fromarray(raw_rgba.astype(np.uint8), mode="RGBA").save(raw_path)
    processed_image.save(processed_path)

    entries = []
    try:
        entries = json.loads(GALLERY_INDEX_PATH.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            entries = []
    except json.JSONDecodeError:
        entries = []

    entries.append(
        {
            "timestamp": timestamp,
            "raw_image": raw_path.name,
            "processed_image": processed_path.name,
            "pred_label": pred,
            "confidence": round(float(np.max(probs)), 4),
            "probabilities": [round(float(v), 4) for v in probs.tolist()],
        }
    )
    GALLERY_INDEX_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def render_gallery(entries: list[dict]) -> None:
    """저장된 이미지와 예측 결과를 카드 형태로 출력한다."""
    st.markdown('<div class="gallery-title">이미지 저장소</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gallery-caption">저장한 손그림, 전처리 결과, 예측 확률을 최신순으로 다시 확인할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )
    if not entries:
        st.info("아직 저장된 손그림이 없습니다.")
        return

    for entry in entries:
        with st.container(border=True):
            cols = st.columns([1, 1, 2])
            raw_path = GALLERY_DIR / entry["raw_image"]
            processed_path = GALLERY_DIR / entry["processed_image"]

            if raw_path.exists():
                cols[0].image(raw_path.as_posix(), caption="원본 손그림", use_container_width=True)
            if processed_path.exists():
                cols[1].image(processed_path.as_posix(), caption="전처리 결과", use_container_width=True)

            cols[2].markdown(
                "\n".join(
                    [
                        f"**저장 시각**: {entry['timestamp']}",
                        f"**예측 숫자**: {entry['pred_label']}",
                        f"**최대 확률**: {entry['confidence']:.2%}",
                    ]
                )
            )
            prob_df = pd.DataFrame(
                {
                    "digit": list(range(10)),
                    "probability": entry["probabilities"],
                }
            ).set_index("digit")
            cols[2].bar_chart(prob_df)


def render_intro() -> None:
    """페이지 상단 소개 영역을 렌더링한다."""
    st.markdown(
        """
        <div class="hero-card">
            <h1>손글씨 숫자 인식 데모</h1>
            <p>
                왼쪽 캔버스에 숫자를 그리면 ONNX Runtime 기반 MNIST 모델이 즉시 예측을 수행합니다.
                전처리된 28x28 입력 이미지와 0~9 확률 분포를 함께 보여주므로,
                입력과 모델 반응을 한 화면에서 자연스럽게 읽을 수 있습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_cols = st.columns(3)
    top_cols[0].markdown(
        """
        <div class="mini-stat">
            <div class="label">모델 형식</div>
            <div class="value">MNIST ONNX</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_cols[1].markdown(
        """
        <div class="mini-stat">
            <div class="label">추론 엔진</div>
            <div class="value">ONNX Runtime</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_cols[2].markdown(
        """
        <div class="mini-stat">
            <div class="label">입력 규격</div>
            <div class="value">1×1×28×28</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guide_panel() -> None:
    """사용 방법과 처리 흐름을 요약한 안내 패널을 출력한다."""
    st.markdown('<div class="section-label">사용 안내</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h3>그리기 요령</h3>
            <p>검은 배경 위에 흰색으로 숫자를 크게 써주세요. 숫자가 너무 작거나 끊기면 확률이 분산될 수 있습니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h3>처리 흐름</h3>
            <p>캔버스 입력 → 숫자 영역 crop/pad → 28x28 변환 → ONNX 추론 → 확률 분포 시각화</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit 앱 메인 진입점."""
    st.set_page_config(page_title="Mission17 MNIST ONNX Demo", layout="wide")
    inject_styles()
    ensure_directories()
    session = load_onnx_session()

    render_intro()

    col_canvas, col_result = st.columns([1.05, 0.95], gap="large")

    with col_canvas:
        st.markdown('<div class="section-label">입력 영역</div>', unsafe_allow_html=True)
        stroke_width = st.slider("브러시 두께", 10, 40, 22, step=2)
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=520,
            width=520,
            drawing_mode="freedraw",
            key="mnist_canvas",
            update_streamlit=True,
        )

    image_rgba = canvas.image_data if canvas.image_data is not None else np.zeros((520, 520, 4), dtype=np.uint8)
    processed_image, input_tensor = preprocess_canvas_image(image_rgba)
    pred, probs = run_inference(session, input_tensor)

    with col_result:
        st.markdown('<div class="section-label">추론 결과</div>', unsafe_allow_html=True)
        metric_cols = st.columns(2)
        metric_cols[0].metric("예측 숫자", pred)
        metric_cols[1].metric("최대 확률", f"{float(np.max(probs)):.2%}")

        result_tab, processed_tab, guide_tab = st.tabs(["확률 차트", "전처리 이미지", "사용 안내"])
        with result_tab:
            prob_df = pd.DataFrame({"digit": list(range(10)), "probability": probs}).set_index("digit")
            st.bar_chart(prob_df, use_container_width=True)
        with processed_tab:
            st.image(processed_image, caption="모델 입력으로 사용된 28x28 이미지", width=240)
            st.code(f"input shape: {tuple(input_tensor.shape)}", language="text")
        with guide_tab:
            render_guide_panel()

        st.markdown('<div class="section-label">저장 작업</div>', unsafe_allow_html=True)
        save_disabled = bool(np.allclose(input_tensor, 0.0))
        if st.button("현재 손그림 저장", use_container_width=True, type="primary", disabled=save_disabled):
            save_gallery_entry(image_rgba, processed_image, pred, probs)
            st.success("현재 손그림과 예측 결과를 저장했습니다.")

        if st.button("저장소 새로고침", use_container_width=True):
            st.rerun()

    st.divider()
    render_gallery(load_gallery_entries())


if __name__ == "__main__":
    main()
