"""按样本时间线端到端校验: 替换档案后的旧版查询与新版查询。"""

import unittest
from pathlib import Path

from run_exchange import run

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class SceneTimelineTests(unittest.TestCase):
    """scene.json: v1 查询的必要块在替换后失去, 必须要求从新查询开始。"""

    @classmethod
    def setUpClass(cls):
        cls.outcome = run(SAMPLES / "scene.json")

    def test_all_events_ok(self):
        for record in self.outcome["events"]:
            self.assertTrue(record["ok"], record)

    def test_top2_settled_but_order_disputed_before_replace(self):
        record = self.outcome["events"][1]  # continue v1 4
        result = record["result"]
        self.assertEqual(result["confirmed"], [])
        self.assertEqual(len(result["disputed"]), 1)
        dispute = result["disputed"][0]
        self.assertEqual(dispute["ranks"], [1, 2])
        self.assertEqual({c["recording"] for c in dispute["candidates"]},
                         {"r0", "r1"})

    def test_old_query_blocked_by_lost_block(self):
        final = self.outcome["final"]["0"]
        self.assertEqual(final["status"], "blocked_version")
        self.assertTrue(final["requires_new_query"])
        self.assertEqual(final["lost_blocks"], ["tail-b"])
        self.assertFalse(final["complete"])


class ArchivalGapTimelineTests(unittest.TestCase):
    """archival-gap.json: 替换前已读的块仍是事实; 新版另开查询。"""

    @classmethod
    def setUpClass(cls):
        cls.outcome = run(SAMPLES / "archival-gap.json")

    def test_v1_query_completes_with_exact_ratios(self):
        final = self.outcome["final"]["0"]
        self.assertEqual(final["status"], "complete")
        self.assertTrue(final["complete"])
        got = [(c["rank"], c["recording"], c["witness"],
                c["distance"].get("exact"))
               for c in final["confirmed"]]
        self.assertEqual(got, [
            (1, "r1", "b0", [2, 17]),
            (2, "r0", "a1", [3, 23]),
            (3, "r2", "c0", [5, 21]),
        ])
        self.assertTrue(all(c["associable"] for c in final["confirmed"]))

    def test_v2_query_partial_with_dispute_basis(self):
        final = self.outcome["final"]["1"]
        self.assertEqual(final["version"], "v2")
        self.assertEqual(final["status"], "blocked_quota")
        self.assertFalse(final["requires_new_query"])
        # 仍有争议的名次范围与依据(待读块)一并交付
        disputes = final["disputed"]
        self.assertEqual([d["ranks"] for d in disputes], [[1, 2], [3]])
        rank3 = disputes[1]["candidates"]
        self.assertEqual({c["recording"] for c in rank3}, {"r2", "r3"})
        for candidate in rank3:
            self.assertTrue(candidate["pending_blocks"])


if __name__ == "__main__":
    unittest.main()
