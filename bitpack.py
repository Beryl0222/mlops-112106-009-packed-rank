"""位打包原语:按块解码、按查询权重投影、带权异或分子累计。

公开距离定义(见 exchange.md):
  原始距离 = 可比较位上的带权异或总和
  最终距离 = 原始距离 / 实际参与的权重总和
  可比较位 = 查询可用 且 候选未声明不可用 的位;零权重位不参与。
  尚未交付的候选位只要没有被排除仍计入分母。

本模块只做"打包"运算:候选行始终以机器字(uint64)形式存在,
逐块应用,绝不展开成 切片 x 位 的完整矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

WORD_WIDTHS = (8, 16, 32, 64)
ENDIANS = ("little", "big")
ORDERS = ("lsb", "msb")


@dataclass(frozen=True)
class BlockSpec:
    """一个设备分片包的目录描述。"""

    bits: tuple[int, ...]   # 本块覆盖的逻辑位,按字内顺序
    width: int              # 字宽: 8/16/32/64
    endian: str             # 字节序
    order: str              # 字内位序: lsb=低位在前, msb=高位在前
    cost: int               # 目录成本(整块 IO)
    rows: tuple[str, ...]   # 每一行所属切片
    file: str               # 载荷文件(相对资料目录)

    def __post_init__(self) -> None:
        if self.width not in WORD_WIDTHS:
            raise ValueError(f"字宽必须是 {WORD_WIDTHS}: {self.width}")
        if self.endian not in ENDIANS:
            raise ValueError(f"未知字节序: {self.endian}")
        if self.order not in ORDERS:
            raise ValueError(f"未知位序: {self.order}")
        if len(self.bits) > self.width:
            raise ValueError("块内逻辑位多于字宽")
        if len(set(self.bits)) != len(self.bits):
            raise ValueError("块内逻辑位重复")

    @property
    def row_bytes(self) -> int:
        return self.width // 8

    @property
    def positions(self) -> tuple[int, ...]:
        """第 j 个逻辑位在字内的位置(0=最低位)。"""
        if self.order == "lsb":
            return tuple(range(len(self.bits)))
        return tuple(self.width - 1 - j for j in range(len(self.bits)))

    @classmethod
    def from_dict(cls, data: dict) -> "BlockSpec":
        return cls(
            bits=tuple(data["bits"]),
            width=data["width"],
            endian=data["endian"],
            order=data["order"],
            cost=data["cost"],
            rows=tuple(data["rows"]),
            file=data["file"],
        )


def decode_block(payload: bytes, spec: BlockSpec) -> np.ndarray:
    """把载荷解码成每行一个 uint64 字(含填补位,调用方负责掩掉)。"""
    expect = len(spec.rows) * spec.row_bytes
    if len(payload) != expect:
        raise ValueError(f"载荷长度 {len(payload)} 与目录不符 {expect}")
    if spec.row_bytes == 1:
        dtype = np.dtype("u1")
    else:
        dtype = np.dtype("<u%d" % spec.row_bytes if spec.endian == "little"
                         else ">u%d" % spec.row_bytes)
    return np.frombuffer(payload, dtype=dtype).astype("<u8")


def _bits_to_word(positions: tuple[int, ...], chosen: set[int]) -> int:
    """把逻辑位集合按字内位置折进一个机器字。"""
    word = 0
    for j, pos in enumerate(positions):
        if j in chosen:
            word |= 1 << pos
    return word


class BlockProjection:
    """一次查询对某个块的打包投影。

    在字位空间中预计算:查询字、每行可比较掩码、带权查表。
    之后每读一次块只需 xor + and + 查表求和,全程不展开位。
    """

    def __init__(
        self,
        spec: BlockSpec,
        query_bits: tuple[int, ...],
        weights: tuple[int, ...],
        query_unavailable: frozenset[int],
        slice_unavailable: dict[str, frozenset[int]],
    ) -> None:
        self.spec = spec
        positions = spec.positions
        # 查询字: 查询位放到字内位置上; 查询不可用位留 0 并由掩码排除。
        self.query_word = np.uint64(
            _bits_to_word(positions, {j for j, b in enumerate(spec.bits)
                                      if query_bits[b]})
        )
        # 每行可比较掩码(字位空间): 查询可用 且 该切片未排除 且 权重非零。
        comparable_rows = []
        self.row_logical: list[int] = []   # 逻辑位空间的可比较掩码(切片状态用)
        self.row_weight: list[int] = []    # 本块为该行交付的可比较权重
        for slice_id in spec.rows:
            excluded = slice_unavailable.get(slice_id, frozenset())
            chosen = {
                j for j, bit in enumerate(spec.bits)
                if bit not in query_unavailable
                and bit not in excluded
                and weights[bit] > 0
            }
            comparable_rows.append(_bits_to_word(positions, chosen))
            logical = 0
            weight = 0
            for j in chosen:
                logical |= 1 << spec.bits[j]
                weight += weights[spec.bits[j]]
            self.row_logical.append(logical)
            self.row_weight.append(weight)
        self.row_mask = np.array(comparable_rows, dtype="<u8")
        # 带权查表: 字节的每个取值对应落在其中的可比较权重和。
        self.luts = self._build_luts(spec, positions, weights, query_unavailable)

    @staticmethod
    def _build_luts(spec: BlockSpec, positions: tuple[int, ...],
                    weights: tuple[int, ...],
                    query_unavailable: frozenset[int]) -> np.ndarray:
        luts = np.zeros((8, 256), dtype=np.int64)
        for j, bit in enumerate(spec.bits):
            if bit in query_unavailable or weights[bit] == 0:
                continue
            pos = positions[j]
            byte_pos, bit_in_byte = divmod(pos, 8)
            # 该字位对所有含此比特的字节取值累加权重。
            idx = np.arange(256, dtype=np.int64)
            luts[byte_pos] += ((idx >> bit_in_byte) & 1).astype(np.int64) * weights[bit]
        return luts

    def apply(self, words: np.ndarray) -> np.ndarray:
        """对解码出的每行字计算带权异或分子贡献(int64 每行)。"""
        if words.shape[0] != len(self.spec.rows):
            raise ValueError("行数与目录不符")
        diff = (words ^ self.query_word) & self.row_mask
        view = diff.astype("<u8").view(np.uint8).reshape(len(diff), 8)
        out = np.zeros(len(diff), dtype=np.int64)
        for bp in range(8):
            out += self.luts[bp][view[:, bp]]
        return out


def weighted_popcount(value: int, weights: tuple[int, ...]) -> int:
    """逻辑位空间的带权 popcount(状态簿记用, 非热路径)。"""
    total = 0
    while value:
        low = value & -value
        total += weights[low.bit_length() - 1]
        value ^= low
    return total


def exact_distance(
    query_bits: tuple[int, ...],
    weights: tuple[int, ...],
    query_unavailable: frozenset[int],
    slice_bits: tuple[int, ...],
    slice_unavailable: frozenset[int],
) -> Fraction | None:
    """全量已知时的公开距离; 分母为零返回 None(不进入排名)。"""
    num = 0
    den = 0
    for i, (q, c) in enumerate(zip(query_bits, slice_bits)):
        if i in query_unavailable or i in slice_unavailable or weights[i] == 0:
            continue
        den += weights[i]
        if q != c:
            num += weights[i]
    if den == 0:
        return None
    return Fraction(num, den)
