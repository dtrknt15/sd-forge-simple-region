import json
import re

_TEXT_PREFIX_RE = re.compile(
    r'^\s*text\s*:\s*(?:"([^"]*)"|\'([^\']*)\')\s*(.*)$',
    re.DOTALL | re.IGNORECASE,
)
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(character|char|object|obj)\s*:\s*(.+)$",
    re.DOTALL | re.IGNORECASE,
)


def _parse_region_item(raw_prompt: str) -> dict:
    """Parse optional label prefix from a region prompt for MOSS-Image."""
    text_match = _TEXT_PREFIX_RE.match(raw_prompt)
    if text_match:
        rendered_text = (text_match.group(1) or text_match.group(2) or "").strip()
        caption = (text_match.group(3) or "").strip()
        if not caption:
            caption = f'Text "{rendered_text}"'
        return {
            "label": "text",
            "caption": caption,
            "text": rendered_text,
        }

    label_match = _LABEL_PREFIX_RE.match(raw_prompt)
    if label_match:
        prefix = label_match.group(1).lower()
        label = "character" if prefix in ("character", "char") else "object"
        return {
            "label": label,
            "caption": label_match.group(2).strip(),
        }

    return {
        "label": "object",
        "caption": raw_prompt.strip(),
    }


def build_moss_layout_json(
    base_prompt: str,
    regions: list[dict],
    width: int,
    height: int,
) -> str:
    """Convert Simple Region boxes into MOSS-Image native layout JSON."""
    half_w = width / 2.0
    half_h = height / 2.0

    items: list[dict] = []
    for reg in regions:
        xmin = round(reg["x1"] * width)
        ymin = round(reg["y1"] * height)
        xmax = round(reg["x2"] * width)
        ymax = round(reg["y2"] * height)

        item = _parse_region_item(reg["prompt"])
        item["bbox"] = [
            int(ymin - half_h),
            int(xmin - half_w),
            int(ymax - half_h),
            int(xmax - half_w),
        ]
        items.append(item)

    caption = (base_prompt or "").strip()
    if not caption:
        caption = ", ".join(it["caption"] for it in items if it.get("caption"))

    payload = {
        "summary": {
            "canvas": "single",
            "caption": caption,
        },
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(payload, ensure_ascii=False)
