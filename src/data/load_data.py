"""Load the fictional company's policy numbers and handbook, and split the handbook into numbered sections."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from src.utils.paths import DATA_RAW


@lru_cache
def load_policy() -> dict:
    return json.loads((DATA_RAW / "policy.json").read_text())


@lru_cache
def load_handbook() -> str:
    return (DATA_RAW / "handbook.md").read_text(encoding="utf-8")


def sections(text: str | None = None) -> list[dict]:
    """One dict per numbered section: id ('4.2'), title, level, body (text up to the next heading)."""
    text = text or load_handbook()
    heads = list(re.finditer(r"^(#{2,3}) (\d+(?:\.\d+)?)\.? (.+)$", text, re.M))
    out = []
    if heads and heads[0].start() > 0:           # title and preamble before section 1
        out.append(dict(id="0", title="Introduction", level=1, body=text[:heads[0].start()].strip(),
                        start=0, end=heads[0].start()))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append(dict(id=h.group(2), title=h.group(3).strip(), level=len(h.group(1)),
                        body=text[h.end():end].strip(), start=h.start(), end=end))
    return out
