cmake_minimum_required(VERSION 3.14)
project(mission16_onnx_cpp CXX)

# C++17 표준으로 빌드
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 사용 예:
#   cmake -S . -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime
# ONNXRUNTIME_DIR에는 include/와 lib/가 있어야 한다.
if(NOT ONNXRUNTIME_DIR)
  message(FATAL_ERROR "ONNXRUNTIME_DIR를 설정하세요. (include/와 lib/ 포함)")
endif()

# ONNX Runtime 헤더/라이브러리 경로 등록
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 실행 파일 생성 및 onnxruntime 링크
add_executable(onnx_infer_cpp onnx_infer.cpp)
target_link_libraries(onnx_infer_cpp onnxruntime)
