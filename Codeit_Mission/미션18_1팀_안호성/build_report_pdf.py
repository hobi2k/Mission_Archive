"""미션 18 요약보고서를 보기 좋은 PDF로 렌더링한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE_MD = ROOT / "미션18_1팀_안호성_요약보고서.md"
OUTPUT_PDF = ROOT / "미션18_1팀_안호성_요약보고서.pdf"

FONT_REGULAR = "/mnt/c/Windows/Fonts/malgun.ttf"
FONT_BOLD = "/mnt/c/Windows/Fonts/malgunbd.ttf"
FONT_MONO = "/mnt/c/Windows/Fonts/gulim.ttc"

PAGE_W = 2480
PAGE_H = 3508
MARGIN_X = 180
MARGIN_Y = 180
CONTENT_W = PAGE_W - (MARGIN_X * 2)

NAVY = "#102542"
BLUE = "#2F80ED"
SKY = "#EAF3FF"
SOFT = "#F6F8FC"
TEXT = "#1F2937"
GRAY = "#6B7280"
LINE = "#D7DFEA"
WHITE = "#FFFFFF"


@dataclass
class Fonts:
    """PDF 렌더링에 사용하는 폰트 묶음."""

    title: ImageFont.FreeTypeFont
    section: ImageFont.FreeTypeFont
    heading: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont


def load_fonts() -> Fonts:
    """보고서에 사용할 폰트를 로드한다.

    Returns:
        Fonts: 렌더링용 폰트 객체 모음.
    """

    return Fonts(
        title=ImageFont.truetype(FONT_BOLD, 116),
        section=ImageFont.truetype(FONT_BOLD, 72),
        heading=ImageFont.truetype(FONT_BOLD, 56),
        body=ImageFont.truetype(FONT_REGULAR, 54),
        small=ImageFont.truetype(FONT_REGULAR, 40),
        mono=ImageFont.truetype(FONT_MONO, 46),
    )


def parse_markdown(path: Path) -> list[tuple[str, str]]:
    """간단한 마크다운 구조를 렌더링용 토큰으로 변환한다.

    Args:
        path: 보고서 마크다운 경로.

    Returns:
        list[tuple[str, str]]: 렌더링 토큰 목록.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    parsed: list[tuple[str, str]] = []
    in_code = False
    code_lines: list[str] = []

    for raw in lines:
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                parsed.append(("codeblock", "\n".join(code_lines)))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line if line else " ")
            continue
        if not line.strip():
            parsed.append(("blank", ""))
            continue
        cleaned = line.replace("`", "")
        if line.startswith("# "):
            parsed.append(("title", cleaned[2:].strip()))
        elif line.startswith("## "):
            parsed.append(("section", cleaned[3:].strip()))
        elif line.startswith("- "):
            parsed.append(("bullet", cleaned[2:].strip()))
        else:
            parsed.append(("text", cleaned))

    if code_lines:
        parsed.append(("codeblock", "\n".join(code_lines)))

    return parsed


def text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    """텍스트 가로 길이를 계산한다.

    Args:
        font: 측정에 사용할 폰트.
        text: 측정할 문자열.

    Returns:
        int: 텍스트 픽셀 너비.
    """

    box = font.getbbox(text)
    return box[2] - box[0]


def line_height(font: ImageFont.FreeTypeFont, extra: int = 0) -> int:
    """폰트 기준 줄 높이를 반환한다.

    Args:
        font: 기준 폰트.
        extra: 추가 여백.

    Returns:
        int: 줄 높이.
    """

    box = font.getbbox("가Ag")
    return (box[3] - box[1]) + extra


def wrap_text(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """문자열을 주어진 폭에 맞게 줄바꿈한다.

    Args:
        text: 원본 문자열.
        font: 기준 폰트.
        width: 최대 폭.

    Returns:
        list[str]: 줄바꿈된 문자열 목록.
    """

    if not text:
        return [""]

    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if text_width(font, candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


class PdfComposer:
    """PIL 이미지를 이용해 보고서 PDF 페이지를 구성한다."""

    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts
        self.pages: list[Image.Image] = []
        self.page_no = 0
        self.page = Image.new("RGB", (PAGE_W, PAGE_H), WHITE)
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_Y
        self.current_section = ""

    def _finish_page(self) -> None:
        """현재 페이지 하단에 페이지 번호를 찍고 저장한다."""

        self.page_no += 1
        footer = str(self.page_no)
        fw = text_width(self.fonts.small, footer)
        self.draw.text(
            ((PAGE_W - fw) // 2, PAGE_H - 52),
            footer,
            font=self.fonts.small,
            fill=GRAY,
        )
        self.pages.append(self.page)

    def new_page(self, with_header: bool = True) -> None:
        """새 페이지를 시작한다.

        Args:
            with_header: 상단 헤더를 그릴지 여부.
        """

        self._finish_page()
        self.page = Image.new("RGB", (PAGE_W, PAGE_H), WHITE)
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_Y
        if with_header:
            self._draw_header()

    def _draw_header(self) -> None:
        """상단 헤더를 그린다."""

        self.draw.rounded_rectangle((120, 80, PAGE_W - 120, 240), radius=56, fill=SOFT)
        self.draw.text((180, 124), "Mission 18 Report", font=self.fonts.heading, fill=NAVY)
        if self.current_section:
            self.draw.text((PAGE_W - 840, 128), self.current_section, font=self.fonts.small, fill=GRAY)
        self.y = 320

    def draw_cover(self) -> None:
        """표지 페이지를 그린다."""

        self.draw.rectangle((0, 0, PAGE_W, PAGE_H), fill=NAVY)
        self.draw.rounded_rectangle((140, 140, PAGE_W - 140, PAGE_H - 140), radius=80, outline="#3B82F6", width=6)
        self.draw.rounded_rectangle((180, 240, 720, 380), radius=52, fill=BLUE)
        self.draw.text((252, 274), "Codeit Mission 18", font=self.fonts.heading, fill=WHITE)
        self.draw.text((180, 560), "영화 리뷰 감성 분석", font=self.fonts.title, fill=WHITE)
        self.draw.text((180, 720), "웹 애플리케이션 보고서", font=self.fonts.title, fill=WHITE)
        self.draw.text((184, 940), "Streamlit + FastAPI + SQLite + Transformers", font=self.fonts.heading, fill="#DCEBFF")

        info_box = (180, 1220, PAGE_W - 180, 1720)
        self.draw.rounded_rectangle(info_box, radius=32, fill="#142E52")
        infos = [
            ("팀", "1팀"),
            ("이름", "안호성"),
            ("구성", "프론트엔드 / 백엔드 분리 구조"),
            ("핵심 기능", "영화 등록, 리뷰 등록, 감성 분석, 평균 평점"),
        ]
        y = 1320
        for label, value in infos:
            self.draw.text((260, y), label, font=self.fonts.heading, fill="#8FB7FF")
            self.draw.text((600, y), value, font=self.fonts.heading, fill=WHITE)
            y += 124

        chips = ["Streamlit", "FastAPI", "SQLite", "SQLAlchemy", "KoELECTRA"]
        x = 90
        y = 1960
        for chip in chips:
            chip_w = text_width(self.fonts.small, chip) + 108
            self.draw.rounded_rectangle((x, y, x + chip_w, y + 104), radius=48, fill=WHITE)
            self.draw.text((x + 52, y + 28), chip, font=self.fonts.small, fill=NAVY)
            x += chip_w + 32

        self.draw.text((180, 3120), "제출용 요약 보고서", font=self.fonts.heading, fill="#DCEBFF")
        self.draw.text((180, 3220), "Movie Review Sentiment Analysis Service", font=self.fonts.small, fill="#AFC8E8")

    def draw_summary_page(self) -> None:
        """프로젝트 요약 페이지를 그린다."""

        self.page = Image.new("RGB", (PAGE_W, PAGE_H), WHITE)
        self.draw = ImageDraw.Draw(self.page)
        self.current_section = "Project Summary"
        self._draw_header()

        self.draw.text((MARGIN_X, self.y), "프로젝트 한눈에 보기", font=self.fonts.section, fill=NAVY)
        self.y += 140

        cards = [
            ("서비스 목표", "영화 정보와 리뷰를 관리하고, 리뷰 감성 분석 결과를 평균 평점으로 시각화하는 웹 서비스를 구현했다."),
            ("구현 방식", "프론트는 Streamlit, 백엔드는 FastAPI로 분리하고, 데이터는 SQLite에 저장했다."),
            ("감성 분석", "KoELECTRA 기반 분류 모델을 FastAPI 내부에서 직접 호출해 라벨과 점수를 생성했다."),
            ("시연 준비", "영화 3개와 영화별 리뷰 10개 이상을 시드 스크립트로 재현 가능하게 구성했다."),
        ]

        card_w = (CONTENT_W - 24) // 2
        card_h = 500
        positions = [
            (MARGIN_X, self.y),
            (MARGIN_X + card_w + 24, self.y),
            (MARGIN_X, self.y + card_h + 24),
            (MARGIN_X + card_w + 24, self.y + card_h + 24),
        ]
        for (title, body), (x, y) in zip(cards, positions, strict=True):
            self.draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=60, fill=SOFT, outline=LINE)
            self.draw.rounded_rectangle((x + 36, y + 36, x + 340, y + 116), radius=36, fill=SKY)
            self.draw.text((x + 68, y + 54), title, font=self.fonts.small, fill=BLUE)
            wrapped = wrap_text(body, self.fonts.body, card_w - 50)
            ty = y + 172
            for line in wrapped:
                self.draw.text((x + 48, ty), line, font=self.fonts.body, fill=TEXT)
                ty += line_height(self.fonts.body, 20)

        self.y = self.y + (card_h * 2) + 160
        stack_lines = [
            "Frontend: Streamlit",
            "Backend: FastAPI",
            "Database: SQLite / SQLAlchemy",
            "Model: Copycats/koelectra-base-v3-generalized-sentiment-analysis",
            "Environment: uv",
        ]
        wrapped_stack: list[str] = []
        for line in stack_lines:
            wrapped = wrap_text(f"• {line}", self.fonts.body, CONTENT_W - 180)
            wrapped_stack.extend(wrapped)
        stack_box_h = 120 + len(wrapped_stack) * line_height(self.fonts.body, 20) + 60
        self.draw.rounded_rectangle((MARGIN_X, self.y, PAGE_W - MARGIN_X, self.y + stack_box_h), radius=60, fill="#F8FBFF", outline=LINE)
        self.draw.text((MARGIN_X + 60, self.y + 52), "기술 스택", font=self.fonts.heading, fill=NAVY)
        ty = self.y + 152
        for wrapped_line in wrapped_stack:
            self.draw.text((MARGIN_X + 72, ty), wrapped_line, font=self.fonts.body, fill=TEXT)
            ty += line_height(self.fonts.body, 20)

    def _ensure_space(self, needed: int) -> None:
        """현재 페이지에 필요한 공간이 없으면 새 페이지를 연다.

        Args:
            needed: 필요한 높이.
        """

        if self.y + needed > PAGE_H - MARGIN_Y - 40:
            self.new_page(with_header=True)

    def render_body(self, items: list[tuple[str, str]]) -> None:
        """본문 섹션을 렌더링한다.

        Args:
            items: 파싱된 보고서 토큰 목록.
        """

        for kind, text in items:
            if kind == "title":
                continue
            if kind == "blank":
                self.y += 10
                continue
            if kind == "section":
                self.current_section = text
                self.new_page(with_header=True)
                section_lines = wrap_text(text, self.fonts.section, CONTENT_W - 120)
                section_h = 48 + len(section_lines) * line_height(self.fonts.section, 12) + 36
                self.draw.rounded_rectangle((MARGIN_X, self.y, PAGE_W - MARGIN_X, self.y + section_h), radius=24, fill=SKY)
                ty = self.y + 26
                for section_line in section_lines:
                    self.draw.text((MARGIN_X + 32, ty), section_line, font=self.fonts.section, fill=NAVY)
                    ty += line_height(self.fonts.section, 12)
                self.y += section_h + 36
                continue

            if kind == "codeblock":
                raw_lines = text.splitlines() or [""]
                wrapped: list[str] = []
                for raw_line in raw_lines:
                    parts = wrap_text(raw_line, self.fonts.mono, CONTENT_W - 60)
                    wrapped.extend(parts if parts else [" "])
                needed = len(wrapped) * line_height(self.fonts.mono, 16) + 92
                self._ensure_space(needed)
                self.draw.rounded_rectangle((MARGIN_X, self.y, PAGE_W - MARGIN_X, self.y + needed), radius=40, fill="#F3F4F6")
                ty = self.y + 40
                for line in wrapped:
                    self.draw.text((MARGIN_X + 40, ty), line, font=self.fonts.mono, fill=TEXT)
                    ty += line_height(self.fonts.mono, 16)
                self.y += needed + 40
                continue

            font = self.fonts.body
            prefix = ""
            x_offset = 0
            if kind == "bullet":
                prefix = "• "
                x_offset = 18

            wrapped = wrap_text(text, font, CONTENT_W - x_offset)
            if kind == "bullet" and wrapped:
                wrapped[0] = prefix + wrapped[0]
                wrapped[1:] = ["  " + line for line in wrapped[1:]]

            needed = len(wrapped) * line_height(font, 20) + 20
            self._ensure_space(needed)
            for line in wrapped:
                self.draw.text((MARGIN_X, self.y), line, font=font, fill=TEXT)
                self.y += line_height(font, 20)
            self.y += 12

    def append_image_page(self, title: str, image_paths: list[Path]) -> None:
        """캡처 이미지를 보기 좋게 배치한 페이지를 추가한다.

        Args:
            title: 페이지 제목.
            image_paths: 배치할 이미지 경로 목록.
        """

        self.new_page(with_header=True)
        self.draw.text((MARGIN_X, self.y), title, font=self.fonts.section, fill=NAVY)
        self.y += 140

        slots = [
            (MARGIN_X, self.y, CONTENT_W, 1120),
            (MARGIN_X, self.y + 1220, CONTENT_W, 1120),
        ]
        for path, (x, y, w, h) in zip(image_paths, slots, strict=False):
            if not path.exists():
                continue
            self.draw.rounded_rectangle((x, y, x + w, y + h), radius=48, fill=SOFT, outline=LINE)
            caption = path.name
            self.draw.text((x + 48, y + 36), caption, font=self.fonts.small, fill=GRAY)
            src = Image.open(path).convert("RGB")
            ratio = min((w - 80) / src.width, (h - 160) / src.height)
            new_size = (max(1, int(src.width * ratio)), max(1, int(src.height * ratio)))
            src = src.resize(new_size, Image.LANCZOS)
            px = x + (w - src.width) // 2
            py = y + 112 + ((h - 152 - src.height) // 2)
            self.page.paste(src, (px, py))

    def save(self, output: Path) -> None:
        """페이지 목록을 PDF 파일로 저장한다.

        Args:
            output: 저장 경로.
        """

        self._finish_page()
        self.pages[0].save(output, "PDF", resolution=300.0, save_all=True, append_images=self.pages[1:])


def main() -> None:
    """스타일이 적용된 최종 PDF를 생성한다."""

    fonts = load_fonts()
    items = parse_markdown(SOURCE_MD)
    composer = PdfComposer(fonts)

    composer.draw_cover()
    composer.draw_summary_page()
    composer.render_body(items)
    composer.append_image_page("FastAPI Docs 및 백엔드 캡처", [ROOT / "backend01.png", ROOT / "backend02.png"])
    composer.append_image_page("서비스 동작 캡처", [ROOT / "backend03.png", ROOT / "frontend01.png"])
    composer.append_image_page("프론트엔드 화면 캡처", [ROOT / "frontend02.png"])
    composer.save(OUTPUT_PDF)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
