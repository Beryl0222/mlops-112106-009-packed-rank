"""查询会话语义: 确认、见证、配额、版本、工作区与交接。"""

import json
import tempfile
import unittest
from pathlib import Path

from engine import (Archive, ContractError, IrrelevantBlockError,
                    QuotaExceededError, QuerySession, VersionLostError,
                    WorkspaceError)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def make_archive(blocks, rows_payloads, **scene):
    """在临时目录里合成一份档案。rows_payloads: {文件: [行字节...]}"""
    tmp = tempfile.TemporaryDirectory()
    case_dir = Path(tmp.name)
    for name, rows in rows_payloads.items():
        (case_dir / name).write_bytes(bytes(rows))
    data = {
        "width": 4, "query": [0, 0, 0, 0], "weights": [1, 1, 1, 1],
        "k": 1, "recordings": {}, "unavailable": {}, "blocks": blocks,
        "budget_bytes": 1 << 16,
    }
    data.update(scene)
    archive = Archive(data, case_dir)
    archive._tmp = tmp  # 防止临时目录提前回收
    return archive


def block(bits, cost, rows, file, width=8, endian="little", order="lsb"):
    return {"bits": bits, "width": width, "endian": endian, "order": order,
            "cost": cost, "rows": rows, "file": file}


class WitnessAmbiguityTests(unittest.TestCase):
    """名次已定但见证未定时, 不得让自动关联展开原片。"""

    def setUp(self):
        self.archive = make_archive(
            blocks={
                "head": block([0, 1], 0, ["s0", "s1", "t0", "u0"], "head.bin"),
                "tail": block([2, 3], 1, ["s0", "s1", "t0", "u0"], "tail.bin"),
            },
            rows_payloads={
                "head.bin": [0x00, 0x00, 0x03, 0x00],
                "tail.bin": [0x00, 0x04, 0x00, 0x00],
            },
            recordings={"r0": ["s0", "s1"], "r1": ["t0"], "r2": ["u0"]},
            unavailable={"u0": [0, 1, 2, 3]},
        )

    def test_rank_confirmed_but_witness_pending(self):
        session = self.archive.open_query("v1", quota=0)
        session.advance()
        result = session.result()
        self.assertFalse(result["complete"])
        self.assertEqual(result["status"], "blocked_quota")
        self.assertEqual(len(result["confirmed"]), 1)
        top = result["confirmed"][0]
        self.assertEqual(top["recording"], "r0")
        self.assertIsNone(top["witness"])
        # 只确认了录音身份: 自动关联不得据以展开原片
        self.assertFalse(top["associable"])

    def test_zero_denominator_slice_excluded(self):
        session = self.archive.open_query("v1", quota=0)
        session.advance()
        self.assertEqual(session.status()["excluded"], ["u0"])

    def test_witness_resolved_after_read(self):
        session = self.archive.open_query("v1", quota=1)
        session.advance()
        result = session.result()
        self.assertTrue(result["complete"])
        top = result["confirmed"][0]
        self.assertEqual(top["witness"], "s0")
        self.assertEqual(top["distance"], {"exact": [0, 1]})
        self.assertTrue(top["associable"])


class EarlyConfirmTests(unittest.TestCase):
    """剩余有效位无法改变排名时, 未读完整个码即确认, 且无关块不可再读。"""

    def setUp(self):
        self.archive = make_archive(
            blocks={
                "head": block([0, 1], 0, ["a0", "b0"], "head.bin"),
                "tail": block([2, 3], 1, ["a0", "b0"], "tail.bin"),
            },
            rows_payloads={
                "head.bin": [0x00, 0x03],
                "tail.bin": [0x00, 0x00],
            },
            recordings={"r0": ["a0"], "r1": ["b0"]},
        )

    def test_confirmed_without_full_code(self):
        session = self.archive.open_query("v1", quota=1)
        session.advance()
        result = session.result()
        # tail 未读, 但 r0 上界 1/2 与 r1 下界 1/2 相接且编号在前, 名次已定。
        self.assertTrue(result["complete"])
        self.assertEqual(result["quota"]["spent"], 0)
        self.assertEqual(result["confirmed"][0]["recording"], "r0")

    def test_irrelevant_block_rejected(self):
        session = self.archive.open_query("v1", quota=1)
        session.advance()
        with self.assertRaises(IrrelevantBlockError):
            session.read("tail")
        # 免费重读已读块不受此限
        receipt = session.read("head")
        self.assertEqual(receipt["charged"], 0)


class QuotaAndTieTests(unittest.TestCase):
    def setUp(self):
        self.archive = make_archive(
            blocks={"all": block([0, 1], 1, ["a0", "b0", "c0"], "all.bin")},
            rows_payloads={"all.bin": [0x01, 0x01, 0x03]},
            width=2,
            query=[0, 0],
            weights=[1, 1],
            k=2,
            recordings={"r0": ["a0"], "r1": ["b0"], "r2": ["c0"]},
        )

    def test_quota_enforced(self):
        session = self.archive.open_query("v1", quota=0)
        with self.assertRaises(QuotaExceededError):
            session.read("all")

    def test_recording_tie_broken_by_stable_id(self):
        session = self.archive.open_query("v1", quota=1)
        session.advance()
        result = session.result()
        self.assertTrue(result["complete"])
        got = [(c["rank"], c["recording"]) for c in result["confirmed"]]
        # a0 与 b0 距离并列 1/2, 录音编号小者在先
        self.assertEqual(got, [(1, "r0"), (2, "r1")])

    def test_continue_version_mismatch_rejected(self):
        session = self.archive.open_query("v1", quota=0)
        with self.assertRaises(ContractError):
            session.add_quota(1, version="v2")


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((SAMPLES / "archival-gap.json").read_text())

    def test_lost_block_requires_new_query(self):
        archive = Archive(self.data, SAMPLES)
        session = archive.open_query("v1", quota=0)
        session.advance()
        archive.replace("v2", "tail-b")
        session.add_quota(2, version="v1")
        with self.assertRaises(VersionLostError) as ctx:
            session.read("tail-b")
        self.assertIn("重新发起查询", str(ctx.exception))

    def test_other_blocks_of_old_version_still_readable(self):
        archive = Archive(self.data, SAMPLES)
        session = archive.open_query("v1", quota=0)
        session.advance()
        archive.replace("v2", "tail-b")
        session.add_quota(2, version="v1")
        receipt = session.read("tail-a")
        self.assertEqual(receipt["charged"], 2)

    def test_new_version_uses_own_payload(self):
        archive = Archive(self.data, SAMPLES)
        archive.open_query("v1", quota=0)
        archive.replace("v2", "tail-b")
        session = archive.open_query("v2", quota=99)
        session.advance()
        result = session.result()
        self.assertTrue(result["complete"])
        # v2 的 tail-b 载荷里 b1 第 9 位为 1, 与查询一致, 距离 5/23
        self.assertEqual(session.status()["settled"]["b1"]["distance"], [5, 23])

    def test_read_before_replace_remains_fact(self):
        archive = Archive(self.data, SAMPLES)
        session = archive.open_query("v1", quota=2)
        session.advance()  # 头部 + tail-b 已在替换前读入
        archive.replace("v2", "tail-b")
        session.add_quota(6, version="v1")
        session.advance()
        result = session.result()
        self.assertTrue(result["complete"])
        # v1 的 tail-b 里 b1 第 9 位为 0, 距离 8/23, 不借用新版载荷
        self.assertEqual(session.status()["settled"]["b1"]["distance"], [8, 23])


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((SAMPLES / "archival-gap.json").read_text())

    def test_state_must_fit_budget(self):
        self.data["budget_bytes"] = 100
        archive = Archive(self.data, SAMPLES)
        with self.assertRaises(WorkspaceError):
            archive.open_query("v1", quota=0)

    def test_cache_must_fit_budget(self):
        # 状态 6*64 + 4*64 = 640, 容得下状态但容不下任何块缓存。
        self.data["budget_bytes"] = 640
        archive = Archive(self.data, SAMPLES)
        session = archive.open_query("v1", quota=99)
        with self.assertRaises(WorkspaceError):
            session.advance()


class HandoffTests(unittest.TestCase):
    """未决查询可交给值班端, 在同一档案上继续。"""

    def setUp(self):
        self.data = json.loads((SAMPLES / "archival-gap.json").read_text())

    def test_roundtrip_continues_identically(self):
        archive = Archive(self.data, SAMPLES)
        session = archive.open_query("v1", quota=2)
        session.advance()
        paused = session.result()
        self.assertFalse(paused["complete"])

        payload = json.loads(json.dumps(session.to_dict()))
        restored = QuerySession.from_dict(archive, payload)
        self.assertEqual(restored.result(), paused)

        # 双方各自续取同版额度, 结论一致
        for s in (session, restored):
            s.add_quota(2, version="v1")
            s.advance()
        self.assertEqual(session.result(), restored.result())

    def test_sealed_ranks_survive_handoff(self):
        archive = Archive(self.data, SAMPLES)
        session = archive.open_query("v1", quota=99)
        session.advance()
        sealed_before = [(c["rank"], c["recording"])
                         for c in session.result()["confirmed"]]
        restored = QuerySession.from_dict(
            archive, json.loads(json.dumps(session.to_dict())))
        sealed_after = [(c["rank"], c["recording"])
                        for c in restored.result()["confirmed"]]
        self.assertEqual(sealed_before, sealed_after)


class WeightOverrideTests(unittest.TestCase):
    """同一档案可接不同权重的查询。"""

    def test_custom_weights(self):
        data = json.loads((SAMPLES / "scene.json").read_text())
        archive = Archive(data, SAMPLES)
        session = archive.open_query("v1", quota=99, weights=[1] * 11)
        session.advance()
        # 全 1 权重下 a0 的可比较位 10 个, 差异位 4 个
        self.assertEqual(session.status()["settled"]["a0"]["distance"], [2, 5])


if __name__ == "__main__":
    unittest.main()
