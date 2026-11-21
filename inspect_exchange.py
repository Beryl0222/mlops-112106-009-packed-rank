"""检查交换资料的形状和索引。"""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    case_dir = path.parent
    validate(data, case_dir)
    return {
        "资料": path.name,
        "字段": sorted(data),
        "字节": path.stat().st_size,
        "数值环境": np.__version__,
    }


def validate(data: dict, case_dir: Path) -> None:
    recordings = {chunk for chunks in data["recordings"].values() for chunk in chunks}
    coverage = {chunk: set() for chunk in recordings}
    for block in data["blocks"].values():
        assert set(block["rows"]) <= recordings
        assert len(block["bits"]) <= block["width"]
        size = case_dir.joinpath(block["file"]).stat().st_size
        assert size == len(block["rows"]) * block["width"] // 8
        for chunk in block["rows"]:
            assert not coverage[chunk].intersection(block["bits"])
            coverage[chunk].update(block["bits"])
    assert all(len(bits) == data["width"] for bits in coverage.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="检查本地交换资料")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.scene), ensure_ascii=False))


if __name__ == "__main__":
    main()
