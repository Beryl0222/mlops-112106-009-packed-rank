"""位打包原语: 解码、投影应用与公开距离定义的校验。"""

import json
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from bitpack import BlockProjection, BlockSpec, decode_block, exact_distance

SAMPLES = Path(__file__).resolve().parents[1] / "samples"

# 手工解码的各切片逻辑位(与 samples/*.bin 逐位核对过)。
SLICE_BITS = {
    "a0": (1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1),
    "a1": (1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1),
    "b0": (1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    "b1": (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1),
    "c0": (0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1),
    "d0": (0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1),
}


def load_scene():
    return json.loads((SAMPLES / "scene.json").read_text(encoding="utf-8"))


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.scene = load_scene()

    def decode_row_bits(self, block_name, row_index):
        spec = BlockSpec.from_dict(self.scene["blocks"][block_name])
        payload = (SAMPLES / spec.file).read_bytes()
        word = int(decode_block(payload, spec)[row_index])
        return tuple((word >> pos) & 1 for pos in spec.positions)

    def test_all_blocks_match_hand_decode(self):
        for block_name, block in self.scene["blocks"].items():
            for row_index, slice_id in enumerate(block["rows"]):
                got = self.decode_row_bits(block_name, row_index)
                expect = tuple(SLICE_BITS[slice_id][b] for b in block["bits"])
                self.assertEqual(got, expect, f"{block_name} 行 {slice_id}")

    def test_padding_bits_are_not_read_as_data(self):
        # tail-a 为 32 位字只放 3 个逻辑位, 且填补位非零(载荷里有 0x60000001)。
        spec = BlockSpec.from_dict(self.scene["blocks"]["tail-a"])
        payload = (SAMPLES / spec.file).read_bytes()
        words = decode_block(payload, spec)
        # 字 0 的低位填补非零, 但 positions 只取高三位。
        self.assertEqual(int(words[0]) & 1, 1)
        self.assertEqual(spec.positions, (31, 30, 29))

    def test_bad_payload_length_rejected(self):
        spec = BlockSpec.from_dict(self.scene["blocks"]["header"])
        with self.assertRaises(ValueError):
            decode_block(b"\x00", spec)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.scene = load_scene()
        self.query = tuple(self.scene["query"])
        self.weights = tuple(self.scene["weights"])
        self.q_unavail = frozenset(self.scene["query_unavailable"])
        self.unavailable = {s: frozenset(b) for s, b in
                            self.scene["unavailable"].items()}

    def naive_contrib(self, block, slice_id):
        """逐位参考实现: 本块为该片贡献的带权异或和。"""
        excluded = self.unavailable.get(slice_id, frozenset())
        total = 0
        for bit in block["bits"]:
            if bit in self.q_unavail or bit in excluded:
                continue
            if self.query[bit] != SLICE_BITS[slice_id][bit]:
                total += self.weights[bit]
        return total

    def test_apply_matches_naive_per_bit(self):
        for block_name, block in self.scene["blocks"].items():
            spec = BlockSpec.from_dict(block)
            proj = BlockProjection(spec, self.query, self.weights,
                                   self.q_unavail, self.unavailable)
            payload = (SAMPLES / spec.file).read_bytes()
            contrib = proj.apply(decode_block(payload, spec))
            for row_index, slice_id in enumerate(block["rows"]):
                self.assertEqual(int(contrib[row_index]),
                                 self.naive_contrib(block, slice_id),
                                 f"{block_name} 行 {slice_id}")

    def test_exact_distance_public_definition(self):
        cases = {
            "a0": Fraction(8, 23), "a1": Fraction(3, 23),
            "b0": Fraction(2, 17), "b1": Fraction(8, 23),
            "c0": Fraction(5, 21), "d0": Fraction(8, 15),
        }
        for slice_id, expect in cases.items():
            got = exact_distance(self.query, self.weights, self.q_unavail,
                                 SLICE_BITS[slice_id],
                                 self.unavailable.get(slice_id, frozenset()))
            self.assertEqual(got, expect, slice_id)

    def test_zero_denominator_excluded(self):
        got = exact_distance(self.query, self.weights, self.q_unavail,
                             SLICE_BITS["a0"], frozenset(range(11)))
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main()
