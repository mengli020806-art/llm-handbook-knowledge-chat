from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from .cleaner import normalize_text


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WIDE_IMAGE_RATIO = 1.18
SUPPLEMENTAL_HINT_RE = re.compile(r"(\d|[一二三四五六七八九十百两]|学分|会议|报告|不少于|不低于|不超过|个月|次)")


class OcrUnavailableError(RuntimeError):
    """当前环境没有可用 OCR 引擎。"""


def extract_pages_from_image(path: str | Path) -> list[str]:
    """从图片中按页提取文字，双页拍照会优先拆成左右两页分别识别。"""

    image_path = Path(path)
    errors: list[str] = []
    for reader in (_read_pages_by_rapidocr, _read_pages_by_paddleocr, _read_pages_by_tesseract):
        try:
            pages = [normalize_text(text) for text in reader(image_path)]
            pages = [text for text in pages if text.strip()]
            if pages:
                return pages
        except Exception as exc:
            errors.append(str(exc))

    detail = "；".join(item for item in errors if item)
    raise OcrUnavailableError(
        "当前 meng 环境没有可用 OCR 引擎。请安装 rapidocr-onnxruntime、paddleocr 或 pytesseract 并配置 Tesseract 后重试。"
        + (f" 详细信息：{detail}" if detail else "")
    )


def extract_text_from_image(path: str | Path) -> str:
    """从图片中识别文字，按可用性依次尝试 RapidOCR、PaddleOCR、Tesseract。"""

    return "\n".join(extract_pages_from_image(path))


def _read_by_rapidocr(path: Path) -> str:
    return "\n".join(_read_pages_by_rapidocr(path))


def _read_pages_by_rapidocr(path: Path) -> list[str]:
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        raise RuntimeError("未安装 rapidocr-onnxruntime") from exc

    def decode_image(file_path: Path):
        # 对双页拍照的知识库图片，直接按原始像素方向读取更稳定，
        # 否则 EXIF 自动旋转可能把左右页顺序打乱，导致 OCR 串页。
        with Image.open(file_path) as raw_image:
            rgb = np.asarray(raw_image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    engine = RapidOCR()
    image = decode_image(path)
    page_images = _split_wide_image(image)
    return [_read_page_with_rapidocr(engine, page) for page in page_images]


def _read_by_paddleocr(path: Path) -> str:
    return "\n".join(_read_pages_by_paddleocr(path))


def _read_pages_by_paddleocr(path: Path) -> list[str]:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError("未安装 paddleocr") from exc

    engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = engine.ocr(str(path), cls=True)
    lines: list[str] = []
    for page in result or []:
        for item in page or []:
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1][0]))
    return ["\n".join(lines)] if lines else []


def _read_by_tesseract(path: Path) -> str:
    return "\n".join(_read_pages_by_tesseract(path))


def _read_pages_by_tesseract(path: Path) -> list[str]:
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("未安装 pytesseract 或 Pillow") from exc

    with Image.open(path) as image:
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
    return [text] if text.strip() else []


def _split_wide_image(image):
    height, width = image.shape[:2]
    if width < height * WIDE_IMAGE_RATIO:
        return [image]

    mid = width // 2
    overlap = max(8, width // 120)
    left = image[:, : min(width, mid + overlap)]
    right = image[:, max(0, mid - overlap) :]
    return [left, right]


def _read_page_with_rapidocr(engine, image) -> str:
    primary_lines = _rapidocr_lines(engine, image)
    if not primary_lines:
        return ""

    supplemental_lines = _collect_supplemental_lines(engine, image)
    if not supplemental_lines:
        return normalize_text("\n".join(primary_lines))

    merged_lines = _merge_supplemental_lines(primary_lines, supplemental_lines)
    return normalize_text("\n".join(merged_lines))


def _rapidocr_lines(engine, image) -> list[str]:
    result, _ = engine(image)
    if not result:
        return []
    return [str(item[1]).strip() for item in result if len(item) >= 2 and str(item[1]).strip()]


def _collect_supplemental_lines(engine, image) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for crop in _make_focus_crops(image):
        for candidate in _iter_ocr_variants(crop):
            for line in _rapidocr_lines(engine, candidate):
                key = _line_key(line)
                if not key or key in seen:
                    continue
                seen.add(key)
                collected.append(line)
    return collected


def _make_focus_crops(image) -> list:
    height, width = image.shape[:2]
    candidates = [
        (0.05, 0.97, 0.34, 0.86),
        (0.09, 0.94, 0.41, 0.86),
        (0.06, 0.96, 0.36, 0.78),
        (0.06, 0.96, 0.50, 0.86),
    ]
    crops: list = []
    for left_ratio, right_ratio, top_ratio, bottom_ratio in candidates:
        left = int(width * left_ratio)
        right = int(width * right_ratio)
        top = int(height * top_ratio)
        bottom = int(height * bottom_ratio)
        if bottom - top < 240 or right - left < 240:
            continue
        crops.append(image[top:bottom, left:right])
    return crops


def _iter_ocr_variants(image):
    try:
        import cv2
    except Exception:
        return [image]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_large = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray_large = cv2.normalize(gray_large, None, 0, 255, cv2.NORM_MINMAX)
    sharpened = cv2.GaussianBlur(gray_large, (0, 0), 1.2)
    sharpened = cv2.addWeighted(gray_large, 1.45, sharpened, -0.45, 0)
    _, binary = cv2.threshold(gray_large, 175, 255, cv2.THRESH_BINARY)
    return [image, gray_large, sharpened, binary]


def _merge_supplemental_lines(primary_lines: list[str], supplemental_lines: list[str]) -> list[str]:
    if not supplemental_lines:
        return primary_lines

    insert_after: dict[int, list[str]] = {}
    cursor = len(primary_lines) - 1
    seen_keys = {_line_key(line) for line in primary_lines}

    for line in supplemental_lines:
        matched_index = _find_similar_line_index(primary_lines, line)
        if matched_index is not None:
            cursor = matched_index
            continue

        key = _line_key(line)
        if not key or key in seen_keys or not _should_keep_supplemental_line(line):
            continue

        insert_after.setdefault(cursor, []).append(line)
        seen_keys.add(key)

    merged: list[str] = []
    leading = insert_after.get(-1, [])
    if leading:
        merged.extend(leading)
    for index, line in enumerate(primary_lines):
        merged.append(line)
        merged.extend(insert_after.get(index, []))
    return merged


def _find_similar_line_index(lines: list[str], target: str) -> int | None:
    target_key = _line_key(target)
    if len(target_key) < 4:
        return None

    best_index: int | None = None
    best_score = 0.0
    for index, line in enumerate(lines):
        line_key = _line_key(line)
        if len(line_key) < 4:
            continue
        if target_key in line_key or line_key in target_key:
            return index
        score = SequenceMatcher(None, target_key, line_key).ratio()
        if score > best_score:
            best_score = score
            best_index = index
    if best_score >= 0.74:
        return best_index
    return None


def _should_keep_supplemental_line(line: str) -> bool:
    key = _line_key(line)
    if len(key) < 6:
        return False
    return bool(SUPPLEMENTAL_HINT_RE.search(line))


def _line_key(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9A-Za-z]", "", text)
