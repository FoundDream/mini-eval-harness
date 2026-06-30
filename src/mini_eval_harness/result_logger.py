from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultLogger:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
