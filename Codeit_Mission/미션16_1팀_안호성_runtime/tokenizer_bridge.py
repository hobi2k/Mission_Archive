#!/usr/bin/env python3
"""미션16 ONNX 런타임 예제를 위한 토크나이저 연결 스크립트.

C++/JS 예제에서 동일한 허깅페이스 토크나이저를 재사용할 수 있도록
명령행 기능(encode/decode/eos)을 제공한다.
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer


def _load_text(args) -> str:
    """명령행 인자에서 입력 텍스트를 읽는다.

    우선순위:
    1) --text_file 경로가 있으면 파일 내용을 사용
    2) 없으면 --text 문자열 사용
    """
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    return args.text or ""


def _normalize_eos_token(raw):
    """토크나이저 eos_token 표현을 문자열로 정규화한다."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("content")
    return str(raw)


def cmd_encode(args):
    """encode 서브커맨드.

    입력 텍스트를 허깅페이스 토크나이저로 인코딩해 `id,id,id` 형태로 출력한다.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    text = _load_text(args)
    ids = tokenizer(text, return_tensors=None, add_special_tokens=True)["input_ids"]
    print(",".join(str(int(x)) for x in ids))


def cmd_decode(args):
    """decode 서브커맨드.

    CSV 토큰 ID를 문자열로 디코딩해 stdout으로 출력한다.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    print(text)


def cmd_eos(args):
    """eos 서브커맨드.

    eos_token_id를 출력하고, 없으면 eos_token -> id 변환을 시도한다.
    둘 다 실패하면 -1을 반환한다.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_token = _normalize_eos_token(tokenizer.eos_token)
        if eos_token is not None:
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_id is None:
        eos_id = -1
    print(int(eos_id))


def main():
    """명령행 진입점.

    사용 예:
      python tokenizer_bridge.py encode --tokenizer_dir <dir> --text "안녕하세요"
      python tokenizer_bridge.py decode --tokenizer_dir <dir> --ids "1,2,3"
      python tokenizer_bridge.py eos --tokenizer_dir <dir>
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encode")
    p_enc.add_argument("--tokenizer_dir", required=True)
    p_enc.add_argument("--text", default="")
    p_enc.add_argument("--text_file", default="")
    p_enc.set_defaults(func=cmd_encode)

    p_dec = sub.add_parser("decode")
    p_dec.add_argument("--tokenizer_dir", required=True)
    p_dec.add_argument("--ids", required=True)
    p_dec.set_defaults(func=cmd_decode)

    p_eos = sub.add_parser("eos")
    p_eos.add_argument("--tokenizer_dir", required=True)
    p_eos.set_defaults(func=cmd_eos)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
