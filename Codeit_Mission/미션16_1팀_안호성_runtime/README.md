# Mission16 ONNX Runtime Samples (C++ / JavaScript)

미션16에서 생성한 ONNX 모델(`mission_16_qtranslator_1.7b_v2.onnx`)을 C++/JavaScript에서 실행하는 샘플입니다.

사용 경로:
- ONNX: `Mission_Archive/Codeit_Mission/models/m16/mission_16_qtranslator_1.7b_v2.onnx`
- ONNX external data: `Mission_Archive/Codeit_Mission/models/m16/mission_16_qtranslator_1.7b_v2.onnx.data`
- Tokenizer/model dir: `Mission_Archive/Codeit_Mission/models/qtranslator_1.7b_v2`

## Files
- `onnx_infer.js`: Node.js + `onnxruntime-node`
- `onnx_infer.cpp`: C++ + ONNX Runtime
- `tokenizer_bridge.py`: HF tokenizer를 재사용하는 브릿지 (encode/decode/eos)
- `package.json`: JS 의존성
- `CMakeLists.txt`: C++ 빌드 설정

## Shared behavior
- 프롬프트 형식
  - `### Instruction:` / `### Input:` / `### Output:`
- 후처리 함수
  - `extract_output_text` 로 `### Output:` 이후 텍스트만 추출

## Prerequisites
- Python: `transformers`, `torch` 설치
- ONNX Runtime
  - JS: `npm i`로 `onnxruntime-node` 설치
  - C++: ONNX Runtime C++ SDK 설치 (`include/`, `lib/` 필요)

## JavaScript run
```bash
cd Mission_Archive/Codeit_Mission/미션16_1팀_안호성_runtime
npm install
npm install onnxruntime-node
node onnx_infer.js \
  ../models/qtranslator_1.7b_v2 \
  ../models/m16/mission_16_qtranslator_1.7b_v2.onnx \
  ./tokenizer_bridge.py \
  "오늘은 왠지 상태가 안 좋아."
```

Arguments:
1. `model_dir` (tokenizer dir, 권장: `../models/qtranslator_1.7b_v2`)
2. `onnx_path` (권장: `../models/m16/mission_16_qtranslator_1.7b_v2.onnx`)
3. `bridge_py_path` (default: `./tokenizer_bridge.py`)
4. `user_input` (default sample)


Example:
```bash
node onnx_infer.js \
  ../models/qtranslator_1.7b_v2 \
  ../models/m16/mission_16_qtranslator_1.7b_v2.onnx \
  ./tokenizer_bridge.py \
  "오늘은 왠지 상태가 안 좋아."
The tokenizer you are loading from '../models/qtranslator_1.7b_v2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
The tokenizer you are loading from '../models/qtranslator_1.7b_v2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
The tokenizer you are loading from '../models/qtranslator_1.7b_v2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
=== JS ONNX RESULT ===
今日はなんだか体調が悪い。
elapsed_ms: 7825
```

## C++ build/run

이 폴더 내부에 ONNX Runtime C++ SDK를 두고 빌드하는 방식을 기준으로 설명합니다.

권장 폴더 구조:
```bash
Mission_Archive/Codeit_Mission/미션16_1팀_안호성_runtime/
├─ CMakeLists.txt
├─ onnx_infer.cpp
├─ tokenizer_bridge.py
├─ build/
└─ third_party/
   └─ onnxruntime-linux-x64-<version>/
      ├─ include/
      │  └─ onnxruntime_cxx_api.h
      └─ lib/
         └─ libonnxruntime.so
```

### 1. ONNX Runtime C++ SDK 준비

- Linux용 ONNX Runtime C++ SDK 압축파일을 받아 `third_party/` 아래에 압축 해제합니다.
- `ONNXRUNTIME_DIR`는 반드시 `include/`, `lib/`를 바로 아래에 가진 폴더여야 합니다.

공식 릴리스 기준 예시 버전은 `1.22.0`입니다.

다운로드 + 압축 해제 명령:

```bash
cd Mission_Archive/Codeit_Mission/미션16_1팀_안호성_runtime
mkdir -p third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz \
  -O third_party/onnxruntime-linux-x64-1.22.0.tgz
tar -xzf third_party/onnxruntime-linux-x64-1.22.0.tgz -C third_party
```

압축 해제 후 아래 파일이 실제로 있는지 확인합니다.

```bash
ls third_party/onnxruntime-linux-x64-1.22.0/include/onnxruntime_cxx_api.h
ls third_party/onnxruntime-linux-x64-1.22.0/lib
```

### 2. CMake configure / build

기존 `build/`가 예시 경로(`/path/to/onnxruntime`)로 configure되어 있으면 그대로는 빌드가 실패합니다.  
안전하게 `build/`를 지우고 다시 configure합니다.

```bash
cd Mission_Archive/Codeit_Mission/미션16_1팀_안호성_runtime
rm -rf build
cmake -S . -B build \
  -DONNXRUNTIME_DIR=$PWD/third_party/onnxruntime-linux-x64-1.22.0
cmake --build build -j
```

다른 버전을 쓰고 싶으면 위 URL과 폴더명/파일명의 버전 번호를 함께 바꾸면 됩니다.

만약 아래 에러가 나오면:
```bash
fatal error: onnxruntime_cxx_api.h: No such file or directory
```

대부분 `ONNXRUNTIME_DIR`가 잘못된 것입니다.  
즉, `include/onnxruntime_cxx_api.h`가 실제로 존재하는 폴더를 `-DONNXRUNTIME_DIR=...`로 지정해야 합니다.

### 3. Run

```bash
./build/onnx_infer_cpp \
  ../models/m16/mission_16_qtranslator_1.7b_v2.onnx \
  ../models/qtranslator_1.7b_v2 \
  ./tokenizer_bridge.py \
  "오늘은 왠지 상태가 안 좋아." \
  64
```

Arguments:
1. `onnx_path`
2. `tokenizer_dir`
3. `bridge_py_path`
4. `korean_input`
5. `max_new_tokens` (선택, 기본값 64)

Example:
```bash
./build/onnx_infer_cpp \
  ../models/m16/mission_16_qtranslator_1.7b_v2.onnx \ 
  ../models/qtranslator_1.7b_v2 \
  ./tokenizer_bridge.py \
  "오늘은 왠지 상태가 안 좋아." \
  64
The tokenizer you are loading from '../models/qtranslator_1.7b_v2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
The tokenizer you are loading from '../models/qtranslator_1.7b_v2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
The tokenizer you are loading from '../models/qtranslator_1.7b_v2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
=== C++ ONNX RESULT ===
今日はなんだか体調が悪い。
elapsed_ms: 2006
```

## Notes
- JS/C++ 모두 토크나이저 일관성을 위해 `tokenizer_bridge.py`를 호출합니다.
- 대용량 모델 특성상 CPU 추론은 느릴 수 있습니다.
