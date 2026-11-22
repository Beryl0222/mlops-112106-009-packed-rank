"""按交换资料的时间线驱动查询, 输出每一步的交付与解释。

时间线事件:
  ["open", 版本, 额度]      打开一次查询并自动推进
  ["continue", 版本, 追加]  给同版在办查询追加额度并自动推进
  ["replace", 版本, 块]     上游替换档案: 指定块换新版载荷, 旧版读取能力移除

用法: python3 run_exchange.py samples/scene.json
"""

import argparse
import json
from pathlib import Path

from engine import Archive, ContractError, ExchangeError


def _snapshot(session) -> dict:
    return {
        "result": session.result(),
        "explanation": {"plan": session.plan(), "status": session.status()},
    }


def run(scene_path: Path) -> dict:
    archive = Archive.from_scene(scene_path)
    data = json.loads(scene_path.read_text(encoding="utf-8"))
    sessions = []
    events = []
    for step in data.get("timeline", []):
        record = {"event": step}
        try:
            action = step[0]
            if action == "open":
                _, version, quota = step
                session = archive.open_query(version=version, quota=quota)
                sessions.append(session)
                session.advance()
                record["session"] = len(sessions) - 1
                record.update(_snapshot(session))
            elif action == "continue":
                _, version, quota = step
                session = next((s for s in reversed(sessions)
                                if s.version == version), None)
                if session is None:
                    raise ContractError(f"没有版本 {version} 的在办查询可续取")
                session.add_quota(quota, version=version)
                session.advance()
                record["session"] = sessions.index(session)
                record.update(_snapshot(session))
            elif action == "replace":
                _, version, block = step
                archive.replace(version, block)
                record["replaced"] = {"block": block, "version": version}
            else:
                raise ContractError(f"未知时间线事件: {step}")
            record["ok"] = True
        except ExchangeError as exc:
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
        events.append(record)
    return {
        "events": events,
        "final": {str(i): session.result() for i, session in enumerate(sessions)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按时间线驱动位打包 top-k 查询")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.scene), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
