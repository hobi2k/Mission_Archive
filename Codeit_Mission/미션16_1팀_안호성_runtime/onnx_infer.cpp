// 미션16 ONNX C++ 추론 예제
//
// 동작 방식:
// 1. 한국어 입력을 미션 프롬프트 형식으로 구성
// 2. tokenizer_bridge.py를 호출해 프롬프트를 토큰 ID로 인코딩
// 3. ONNX 런타임 세션으로 매 단계 최대값 토큰 선택 생성
// 4. 생성 토큰을 다시 문자열로 디코딩하고 "### Output:" 이후만 출력

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

// 셸 인자로 문자열을 안전하게 넘기기 위한 작은 이스케이프 함수.
static std::string shellEscape(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// 셸 명령 실행 후 stdout 문자열을 반환한다.
// 실패(비정상 종료 코드) 시 예외를 던진다.
static std::string runCommand(const std::string& cmd) {
    std::array<char, 4096> buf{};
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("Failed to run command: " + cmd);
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) output += buf.data();
    int rc = pclose(pipe);
    if (rc != 0) throw std::runtime_error("Command failed: " + cmd);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    return output;
}

// "1,2,3" 같은 콤마 구분 문자열을 int64 벡터로 변환한다.
static std::vector<int64_t> splitCsvInt64(const std::string& csv) {
    std::vector<int64_t> out;
    if (csv.empty()) return out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoll(tok));
    }
    return out;
}

// int64 벡터를 CSV 문자열로 변환한다.
static std::string joinCsvInt64(const std::vector<int64_t>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << v[i];
    }
    return os.str();
}

// 미션 요구 프롬프트 포맷을 그대로 구성한다.
static std::string buildPrompt(const std::string& instruction, const std::string& userInput) {
    return "### Instruction:\n" + instruction + "\n\n" +
           "### Input:\n" + userInput + "\n\n" +
           "### Output:\n";
}

// 모델이 프롬프트를 echo해도 실제 출력 본문만 추출한다.
static std::string extractOutputText(const std::string& generatedText) {
    const std::string marker = "### Output:\n";
    const std::string marker2 = "### Output:";

    auto pos = generatedText.find(marker);
    if (pos != std::string::npos) return generatedText.substr(pos + marker.size());

    pos = generatedText.find(marker2);
    if (pos != std::string::npos) return generatedText.substr(pos + marker2.size());

    return generatedText;
}

// tokenizer_bridge.py encode 명령을 통해 텍스트 -> 토큰 ID 변환.
static std::vector<int64_t> encodeWithBridge(
    const std::string& bridgePath,
    const std::string& tokenizerDir,
    const std::string& text
) {
    const std::string tmp = "/tmp/mission16_prompt_cpp.txt";
    {
        std::ofstream ofs(tmp);
        ofs << text;
    }
    std::string cmd = "python3 " + shellEscape(bridgePath) +
                      " encode --tokenizer_dir " + shellEscape(tokenizerDir) +
                      " --text_file " + shellEscape(tmp);
    std::string csv = runCommand(cmd);
    return splitCsvInt64(csv);
}

// tokenizer_bridge.py eos 명령으로 문장 종료 토큰 ID를 가져온다.
static int64_t eosWithBridge(const std::string& bridgePath, const std::string& tokenizerDir) {
    std::string cmd = "python3 " + shellEscape(bridgePath) +
                      " eos --tokenizer_dir " + shellEscape(tokenizerDir);
    return std::stoll(runCommand(cmd));
}

// tokenizer_bridge.py decode 명령으로 토큰 ID -> 텍스트 변환.
static std::string decodeWithBridge(
    const std::string& bridgePath,
    const std::string& tokenizerDir,
    const std::vector<int64_t>& ids
) {
    std::string cmd = "python3 " + shellEscape(bridgePath) +
                      " decode --tokenizer_dir " + shellEscape(tokenizerDir) +
                      " --ids " + shellEscape(joinCsvInt64(ids));
    return runCommand(cmd);
}

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::cerr << "사용법:\n"
                      << "  " << argv[0]
                      << " <onnx_path> <tokenizer_dir> <bridge_py_path> <korean_input> [max_new_tokens]\n";
            return 1;
        }

        const std::string onnxPath = argv[1];
        const std::string tokenizerDir = argv[2];
        const std::string bridgePath = argv[3];
        const std::string userInput = argv[4];
        const int maxNewTokens = (argc >= 6) ? std::stoi(argv[5]) : 64;
        const std::string instruction = "다음 한국어 문장을 자연스러운 일본어로 번역하시오.";

        // ONNX 런타임 환경/세션 초기화.
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mission16-cpp");
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        Ort::Session session(env, onnxPath.c_str(), so);

        std::string prompt = buildPrompt(instruction, userInput);
        int64_t eosId = eosWithBridge(bridgePath, tokenizerDir);
        std::vector<int64_t> ids = encodeWithBridge(bridgePath, tokenizerDir, prompt);

        Ort::AllocatorWithDefaultOptions alloc;
        const char* inputNames[] = {"input_ids", "attention_mask"};
        const char* outputNames[] = {"logits"};

        auto t0 = std::chrono::steady_clock::now();

        for (int step = 0; step < maxNewTokens; ++step) {
            const int64_t seqLen = static_cast<int64_t>(ids.size());
            // 현재 길이 기준 어텐션 마스크를 모두 1로 생성.
            std::vector<int64_t> attention(static_cast<size_t>(seqLen), 1);
            std::array<int64_t, 2> shape{1, seqLen};

            // 입력 텐서 모양: [배치=1, 토큰길이]
            Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputIds = Ort::Value::CreateTensor<int64_t>(
                mi, ids.data(), ids.size(), shape.data(), shape.size());
            Ort::Value attentionMask = Ort::Value::CreateTensor<int64_t>(
                mi, attention.data(), attention.size(), shape.data(), shape.size());

            std::array<Ort::Value, 2> inputs{std::move(inputIds), std::move(attentionMask)};
            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                inputNames,
                inputs.data(),
                inputs.size(),
                outputNames,
                1
            );

            // 로짓 텐서 모양: [배치, 토큰길이, 어휘크기]
            float* logits = outputs[0].GetTensorMutableData<float>();
            auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (outShape.size() != 3) throw std::runtime_error("Unexpected logits shape.");

            const int64_t outSeq = outShape[1];
            const int64_t vocab = outShape[2];
            // 마지막 토큰 위치의 어휘 점수에서 최대값 인덱스를 뽑는다.
            const int64_t offset = (outSeq - 1) * vocab;

            int64_t bestId = 0;
            float bestVal = logits[offset];
            for (int64_t i = 1; i < vocab; ++i) {
                float v = logits[offset + i];
                if (v > bestVal) {
                    bestVal = v;
                    bestId = i;
                }
            }

            ids.push_back(bestId);
            if (eosId >= 0 && bestId == eosId) break;
        }

        auto t1 = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::string generated = decodeWithBridge(bridgePath, tokenizerDir, ids);
        std::string output = extractOutputText(generated);

        std::cout << "=== C++ ONNX RESULT ===\n";
        std::cout << output << "\n";
        std::cout << "elapsed_ms: " << elapsedMs << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
