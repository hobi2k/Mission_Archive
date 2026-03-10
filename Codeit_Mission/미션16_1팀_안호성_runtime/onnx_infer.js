/**
 * 미션16 ONNX 자바스크립트(Node.js) 추론 예제.
 *
 * 동작 개요:
 * 1. 미션 프롬프트 생성
 * 2. tokenizer_bridge.py로 인코딩/디코딩
 * 3. ONNX 런타임으로 그리디 생성
 * 4. "### Output:" 이후 텍스트만 출력
 */
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';
import * as ort from 'onnxruntime-node';

/**
 * 미션 요구 프롬프트 형식으로 텍스트를 구성한다.
 * @param {string} instruction - 모델 지시문
 * @param {string} userInput - 번역 대상 입력
 * @returns {string}
 */
function buildPrompt(instruction, userInput) {
  instruction = (instruction || '').trim();
  userInput = (userInput || '').trim();
  return (
    '### Instruction:\n' +
    `${instruction}\n\n` +
    '### Input:\n' +
    `${userInput}\n\n` +
    '### Output:\n'
  );
}

/**
 * 생성 결과에서 실제 출력 본문만 잘라낸다.
 * @param {string} generatedText
 * @returns {string}
 */
function extractOutputText(generatedText) {
  const marker = '### Output:\n';
  if (generatedText.includes(marker)) {
    return generatedText.split(marker, 2)[1].trim();
  }
  const marker2 = '### Output:';
  if (generatedText.includes(marker2)) {
    return generatedText.split(marker2, 2)[1].trim();
  }
  return generatedText.trim();
}

/**
 * tokenizer_bridge.py 서브커맨드를 동기 실행한다.
 * @param {string} bridgePath
 * @param {string[]} args
 * @returns {string}
 */
function runBridge(bridgePath, args) {
  return execFileSync('python3', [bridgePath, ...args], { encoding: 'utf-8' }).trim();
}

/**
 * 텍스트 파일을 통해 bridge encode를 호출하고 토큰 ID 배열을 얻는다.
 * @param {string} bridgePath
 * @param {string} tokenizerDir
 * @param {string} text
 * @returns {number[]}
 */
function encodeText(bridgePath, tokenizerDir, text) {
  const tmpPath = path.join(os.tmpdir(), `mission16_prompt_${Date.now()}.txt`);
  fs.writeFileSync(tmpPath, text, 'utf-8');
  try {
    const output = runBridge(bridgePath, ['encode', '--tokenizer_dir', tokenizerDir, '--text_file', tmpPath]);
    if (!output) return [];
    return output.split(',').map((x) => Number(x));
  } finally {
    fs.rmSync(tmpPath, { force: true });
  }
}

/**
 * 토큰 ID 배열을 문자열로 디코딩한다.
 * @param {string} bridgePath
 * @param {string} tokenizerDir
 * @param {number[]} ids
 * @returns {string}
 */
function decodeIds(bridgePath, tokenizerDir, ids) {
  const csv = ids.join(',');
  return runBridge(bridgePath, ['decode', '--tokenizer_dir', tokenizerDir, '--ids', csv]);
}

/**
 * EOS 토큰 ID를 bridge에서 조회한다.
 * @param {string} bridgePath
 * @param {string} tokenizerDir
 * @returns {number}
 */
function getEosId(bridgePath, tokenizerDir) {
  const raw = runBridge(bridgePath, ['eos', '--tokenizer_dir', tokenizerDir]);
  return Number(raw);
}

/**
 * 연속 배열의 일부 구간(start~start+length)에서 argmax 인덱스를 찾는다.
 * @param {Float32Array|number[]} arr
 * @param {number} start
 * @param {number} length
 * @returns {number}
 */
function argmax(arr, start, length) {
  let bestIdx = 0;
  let bestVal = arr[start];
  for (let i = 1; i < length; i += 1) {
    const v = arr[start + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * ONNX 런타임 그리디 생성 반복문.
 * 매 단계에서 마지막 토큰 위치 로짓의 최대값 토큰을 붙인다.
 *
 * @param {ort.InferenceSession} session
 * @param {string} bridgePath
 * @param {string} tokenizerDir
 * @param {string} prompt
 * @param {number} eosId
 * @param {number} maxNewTokens
 * @returns {Promise<string>}
 */
async function onnxGreedyGenerate(session, bridgePath, tokenizerDir, prompt, eosId, maxNewTokens = 64) {
  let ids = encodeText(bridgePath, tokenizerDir, prompt);

  for (let step = 0; step < maxNewTokens; step += 1) {
    const seqLen = ids.length;
    // ONNX int64 입력이므로 JS bigint 배열을 사용한다.
    const inputIds = new ort.Tensor('int64', BigInt64Array.from(ids.map((x) => BigInt(x))), [1, seqLen]);
    const attentionMask = new ort.Tensor('int64', BigInt64Array.from(Array(seqLen).fill(1n)), [1, seqLen]);

    const out = await session.run({
      input_ids: inputIds,
      attention_mask: attentionMask,
    });

    const logits = out.logits.data;
    // 로짓 모양: [1, seqLen, vocab] -> 1차원 배열로 평탄화되어 들어온다.
    const vocabSize = logits.length / seqLen;
    const lastOffset = (seqLen - 1) * vocabSize;
    const nextToken = argmax(logits, lastOffset, vocabSize);

    ids.push(nextToken);
    if (eosId >= 0 && nextToken === eosId) break;
  }

  const generated = decodeIds(bridgePath, tokenizerDir, ids);
  return extractOutputText(generated);
}

/**
 * 명령행 진입점.
 *
 * 인자:
 *   argv[2] modelDir
 *   argv[3] onnxPath
 *   argv[4] bridgePath
 *   argv[5] userInput
 */
async function main() {
  const runtimeDir = path.resolve(path.dirname(new URL(import.meta.url).pathname));
  const repoRoot = path.resolve(runtimeDir, '..', '..');

  const modelDir = process.argv[2] || path.join(repoRoot, 'models', 'qtranslator_1.7b_v2');
  const onnxPath = process.argv[3] || path.join(repoRoot, 'models', 'm16', 'mission_16_qtranslator_1.7b_v2.onnx');
  const bridgePath = process.argv[4] || path.join(runtimeDir, 'tokenizer_bridge.py');

  const instruction = '다음 한국어 문장을 자연스러운 일본어로 번역하시오.';
  const userInput = process.argv[5] || '오늘은 왠지 상태가 안 좋아.';
  const prompt = buildPrompt(instruction, userInput);

  const session = await ort.InferenceSession.create(onnxPath, {
    executionProviders: ['cpu'],
  });

  const eosId = getEosId(bridgePath, modelDir);
  const start = Date.now();
  const text = await onnxGreedyGenerate(session, bridgePath, modelDir, prompt, eosId, 64);
  const elapsedMs = Date.now() - start;

  console.log('=== JS ONNX RESULT ===');
  console.log(text);
  console.log(`elapsed_ms: ${elapsedMs}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
