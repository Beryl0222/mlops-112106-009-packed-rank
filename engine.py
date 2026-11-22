"""精确 top-k 张量查询引擎。

职责:
  * 区间距离: 未交付的可比较位只影响分子上界, 分母按公开定义一次算清;
  * 名次确认: 录音间按 (距离, 录音编号) 的确定先后关系做偏序,
    剩余有效位无法改变排名时即签出, 已签出的名次不撤回;
  * 见证: 同录音切片按 (距离, 切片编号) 确定先后, 唯一胜出者为见证;
    只确认录音身份而未确认见证时 associable=False, 自动关联不得展开;
  * 配额与缓存: 一次查询共享读块配额, 同块重读不重复收费;
  * 版本: 会话绑定打开时的版本, 不借用其他版本的载荷;
  * 工作区: 候选状态与读块缓存按交换契约的预算计量, 不完整解包候选矩阵。
"""

from __future__ import annotations

import base64
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from bitpack import BlockProjection, BlockSpec, decode_block, weighted_popcount
from inspect_exchange import validate


class ExchangeError(Exception):
    """交换契约相关错误的基类。"""


class ContractError(ExchangeError):
    """调用顺序或参数违反交换契约。"""


class VersionLostError(ExchangeError):
    """本版必要块已被上游移除, 必须从新查询开始。"""


class QuotaExceededError(ExchangeError):
    """本次查询的读块配额不足。"""


class IrrelevantBlockError(ExchangeError):
    """块只触及已无关的候选, 不允许为之消耗配额。"""


class WorkspaceError(ExchangeError):
    """候选状态与读块缓存超出工作区预算。"""


# 工作区计量模型(字节): 每切片/每录音的固定状态 + 每块缓存的解码字。
STATE_BYTES_PER_SLICE = 64
STATE_BYTES_PER_RECORDING = 64
CACHE_BASE_PER_BLOCK = 64
CACHE_BYTES_PER_ROW = 24


def _ratio(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _interval_json(lo: Fraction, hi: Fraction) -> dict:
    if lo == hi:
        return {"exact": _ratio(lo)}
    return {"interval": [_ratio(lo), _ratio(hi)]}


class _SliceState:
    __slots__ = ("slice_id", "recording", "comparable", "denom",
                 "num", "unknown_w", "delivered")

    def __init__(self, slice_id: str, recording: str, comparable: int, denom: int):
        self.slice_id = slice_id
        self.recording = recording
        self.comparable = comparable   # 逻辑位空间的可比较掩码
        self.denom = denom             # 分母: 可比较权重总和(含未交付位)
        self.num = 0                   # 已交付位的带权异或和
        self.unknown_w = denom         # 未交付可比较位的权重
        self.delivered = 0             # 已交付逻辑位掩码(含不可比较位)


class Archive:
    """候选档案: 分块目录、版本沿革与工作区预算。"""

    def __init__(self, data: dict, case_dir: Path):
        validate(data, case_dir)
        self.case_dir = Path(case_dir)
        self.width = data["width"]
        self.query_bits = tuple(data["query"])
        self.weights = tuple(data["weights"])
        self.query_unavailable = frozenset(data.get("query_unavailable", ()))
        self.k = data["k"]
        self.recordings = {r: tuple(s) for r, s in data["recordings"].items()}
        self.unavailable = {s: frozenset(b) for s, b in data.get("unavailable", {}).items()}
        self.blocks = {name: BlockSpec.from_dict(b) for name, b in data["blocks"].items()}
        self.budget_bytes = data.get("budget_bytes", 1 << 20)
        self.current_version: str | None = None
        self._removed: set[tuple[str, str]] = set()
        self._overrides: dict[tuple[str, str], Path] = {}

    @classmethod
    def from_scene(cls, path: str | Path) -> "Archive":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data, path.parent)

    # ---- 版本沿革 ----

    def replace(self, new_version: str, block_id: str) -> None:
        """上游替换档案: 指定块换新版载荷, 旧版读取能力被移除。"""
        if self.current_version is None:
            raise ContractError("尚无打开的版本, 不能替换")
        if block_id not in self.blocks:
            raise KeyError(f"未知块: {block_id}")
        spec = self.blocks[block_id]
        stem = Path(spec.file).stem
        suffix = Path(spec.file).suffix
        override = self.case_dir / f"{stem}-{new_version}{suffix}"
        if not override.exists():
            raise ContractError(f"新版载荷缺失: {override.name}")
        self._removed.add((self.current_version, block_id))
        self._overrides[(new_version, block_id)] = override
        self.current_version = new_version

    def has_payload(self, version: str, block_id: str) -> bool:
        return (version, block_id) not in self._removed

    def payload(self, version: str, block_id: str) -> bytes:
        if not self.has_payload(version, block_id):
            raise VersionLostError(
                f"块 {block_id} 在版本 {version} 的载荷已被上游移除; "
                "旧证据不能沿用到新档案, 请从当前版本重新发起查询"
            )
        path = self._overrides.get((version, block_id),
                                   self.case_dir / self.blocks[block_id].file)
        return path.read_bytes()

    # ---- 打开查询 ----

    def open_query(self, version: str | None = None, quota: int = 0,
                   query=None, weights=None, query_unavailable=None,
                   k: int | None = None) -> "QuerySession":
        """打开一次查询。查询可自带有效位、权重与版本, 缺省沿用档案场景。"""
        version = version or self.current_version or "v1"
        if self.current_version is None:
            self.current_version = version
        return QuerySession(
            self, version, quota,
            query_bits=tuple(query) if query is not None else self.query_bits,
            weights=tuple(weights) if weights is not None else self.weights,
            query_unavailable=(frozenset(query_unavailable)
                               if query_unavailable is not None
                               else self.query_unavailable),
            k=self.k if k is None else k,
        )


class QuerySession:
    """一次绑定版本的精确 top-k 查询。"""

    def __init__(self, archive: Archive, version: str, quota: int,
                 query_bits, weights, query_unavailable, k: int):
        self.archive = archive
        self.version = version
        self.k = k
        self.query_bits = query_bits
        self.weights = weights
        self.query_unavailable = query_unavailable
        self.quota_remaining = quota
        self.quota_spent = 0
        self._slices: dict[str, _SliceState] = {}
        for recording, slices in archive.recordings.items():
            for slice_id in slices:
                excluded = archive.unavailable.get(slice_id, frozenset())
                comparable = 0
                denom = 0
                for bit in range(archive.width):
                    if (bit in query_unavailable or bit in excluded
                            or weights[bit] == 0):
                        continue
                    comparable |= 1 << bit
                    denom += weights[bit]
                self._slices[slice_id] = _SliceState(slice_id, recording,
                                                     comparable, denom)
        self._recordings = archive.recordings
        self._projections = {
            name: BlockProjection(spec, query_bits, weights, query_unavailable,
                                  archive.unavailable)
            for name, spec in archive.blocks.items()
        }
        self._block_logical = {
            name: sum(1 << b for b in spec.bits)
            for name, spec in archive.blocks.items()
        }
        self._read_blocks: set[str] = set()
        self._cache: dict[str, np.ndarray] = {}
        self._sealed: list[tuple[int, str]] = []
        self._workspace_used = (STATE_BYTES_PER_SLICE * len(self._slices)
                                + STATE_BYTES_PER_RECORDING * len(self._recordings))
        if self._workspace_used > archive.budget_bytes:
            raise WorkspaceError(
                f"候选状态 {self._workspace_used} 超出工作区预算 "
                f"{archive.budget_bytes}"
            )

    # ---- 距离与排名(内部) ----

    @staticmethod
    def _precedes(a_hi: Fraction, a_id: str, b_lo: Fraction, b_id: str) -> bool:
        """a 确定排在 b 前: 上界更小, 或上界相接且编号更小(平局稳定编号)。"""
        return a_hi < b_lo or (a_hi == b_lo and a_id < b_id)

    def _slice_interval(self, slice_id: str) -> tuple[Fraction, Fraction] | None:
        st = self._slices[slice_id]
        if st.denom == 0:
            return None
        return (Fraction(st.num, st.denom),
                Fraction(st.num + st.unknown_w, st.denom))

    def _recording_interval(self, recording: str) -> tuple[Fraction, Fraction] | None:
        los, his = [], []
        for slice_id in self._recordings[recording]:
            interval = self._slice_interval(slice_id)
            if interval is not None:
                los.append(interval[0])
                his.append(interval[1])
        if not los:
            return None
        return (min(los), min(his))

    def _analyze(self):
        ranked = [r for r in self._recordings
                  if self._recording_interval(r) is not None]
        intervals = {r: self._recording_interval(r) for r in ranked}
        total = len(ranked)
        lo_rank: dict[str, int] = {}
        hi_rank: dict[str, int] = {}
        for r in ranked:
            r_lo, r_hi = intervals[r]
            before = sum(1 for a in ranked if a != r
                         and self._precedes(intervals[a][1], a, r_lo, r))
            after = sum(1 for b in ranked if b != r
                        and self._precedes(r_hi, r, intervals[b][0], b))
            lo_rank[r] = before + 1
            hi_rank[r] = total - after
        return ranked, intervals, lo_rank, hi_rank

    def _witness(self, recording: str) -> str | None:
        """唯一确定击败全部同组切片的见证; 尚未唯一时为 None。"""
        winner = None
        for s in self._recordings[recording]:
            s_iv = self._slice_interval(s)
            if s_iv is None:
                continue
            beats_all = True
            for t in self._recordings[recording]:
                if t == s:
                    continue
                t_iv = self._slice_interval(t)
                if t_iv is None:
                    continue
                if not self._precedes(s_iv[1], s, t_iv[0], t):
                    beats_all = False
                    break
            if beats_all:
                if winner is not None:
                    return None  # 理论上不可能, 保守处理
                winner = s
        return winner

    def _k_eff(self, ranked) -> int:
        return min(self.k, len(ranked))

    def _recording_decided(self, r, lo_rank, hi_rank, k_eff) -> bool:
        if lo_rank[r] > k_eff:
            return True  # 确定出局
        return lo_rank[r] == hi_rank[r] and self._witness(r) is not None

    def _beaten_by(self, slice_id: str) -> str | None:
        """同组中确定压制本切片的切片(取编号最小者)。"""
        st = self._slices[slice_id]
        s_iv = self._slice_interval(slice_id)
        if s_iv is None:
            return None
        for t in self._recordings[st.recording]:
            if t == slice_id:
                continue
            t_iv = self._slice_interval(t)
            if t_iv is not None and self._precedes(t_iv[1], t, s_iv[0], slice_id):
                return t
        return None

    def _live_slices(self) -> set[str]:
        """仍可能改变未决名次或未决见证的切片。"""
        ranked, _, lo_rank, hi_rank = self._analyze()
        k_eff = self._k_eff(ranked)
        live: set[str] = set()
        for r in ranked:
            if self._recording_decided(r, lo_rank, hi_rank, k_eff):
                continue
            for s in self._recordings[r]:
                st = self._slices[s]
                if st.denom == 0 or st.unknown_w == 0:
                    continue
                if self._beaten_by(s) is not None:
                    continue
                live.add(s)
        return live

    # ---- 读块 ----

    def _block_live_rows(self, block_id: str, live: set[str]) -> list[int]:
        proj = self._projections[block_id]
        return [i for i, s in enumerate(proj.spec.rows)
                if s in live and proj.row_weight[i] > 0]

    def read(self, block_id: str) -> dict:
        """读取一个块: 计费、解码、应用。同块重读免费且幂等。"""
        if block_id not in self.archive.blocks:
            raise KeyError(f"未知块: {block_id}")
        if block_id in self._read_blocks:
            return {"block": block_id, "charged": 0, "cached": True,
                    "quota_remaining": self.quota_remaining}
        live = self._live_slices()
        proj = self._projections[block_id]
        if not self._block_live_rows(block_id, live):
            raise IrrelevantBlockError(
                f"块 {block_id} 只触及已无关的候选, 不能继续为其消耗配额"
            )
        spec = proj.spec
        # 版本失去是永久条件, 先于配额(暂时性)报告。
        if not self.archive.has_payload(self.version, block_id):
            raise VersionLostError(
                f"块 {block_id} 在版本 {self.version} 的载荷已被上游移除; "
                "旧证据不能沿用到新档案, 请从当前版本重新发起查询"
            )
        if spec.cost > self.quota_remaining:
            raise QuotaExceededError(
                f"块 {block_id} 成本 {spec.cost} 超出剩余配额 {self.quota_remaining}"
            )
        entry_bytes = CACHE_BASE_PER_BLOCK + CACHE_BYTES_PER_ROW * len(spec.rows)
        if self._workspace_used + entry_bytes > self.archive.budget_bytes:
            raise WorkspaceError(
                f"缓存块 {block_id} 需要 {entry_bytes} 字节, 超出工作区预算"
            )
        payload = self.archive.payload(self.version, block_id)
        words = decode_block(payload, spec)
        contrib = proj.apply(words)
        for i, slice_id in enumerate(spec.rows):
            st = self._slices[slice_id]
            st.num += int(contrib[i])
            st.unknown_w -= proj.row_weight[i]
            st.delivered |= self._block_logical[block_id]
        self._read_blocks.add(block_id)
        self._cache[block_id] = words
        self._workspace_used += entry_bytes
        self.quota_remaining -= spec.cost
        self.quota_spent += spec.cost
        return {"block": block_id, "charged": spec.cost, "cached": False,
                "updated": list(spec.rows),
                "quota_remaining": self.quota_remaining}

    def add_quota(self, amount: int, version: str | None = None) -> None:
        """值班端追加配额; 续取必须与会话同版。"""
        if version is not None and version != self.version:
            raise ContractError(
                f"续取版本 {version} 与会话版本 {self.version} 不符"
            )
        if amount < 0:
            raise ContractError("追加额度不能为负")
        self.quota_remaining += amount

    def advance(self) -> dict:
        """按解释顺序自动读取可消除歧义的块, 直到定案或配额/版本受阻。"""
        while True:
            nxt = None
            for entry in self.plan():
                if not entry["available"]:
                    continue
                if entry["cost"] > self.quota_remaining:
                    continue
                nxt = entry["block"]
                break
            if nxt is None:
                return self.result()
            self.read(nxt)

    # ---- 解释 ----

    def plan(self) -> list[dict]:
        """为什么某个设备块值得继续读: 触及的候选、可消除的歧义、成本。"""
        live = self._live_slices()
        ranked, _, lo_rank, hi_rank = self._analyze()
        entries = []
        for name, spec in self.archive.blocks.items():
            if name in self._read_blocks:
                continue
            proj = self._projections[name]
            helps = []
            for i, slice_id in enumerate(spec.rows):
                if slice_id not in live or proj.row_weight[i] == 0:
                    continue
                st = self._slices[slice_id]
                r = st.recording
                if lo_rank[r] == hi_rank[r]:
                    note = f"录音 {r} 第 {lo_rank[r]} 名的见证切片"
                else:
                    note = f"录音 {r} 的名次归属({lo_rank[r]}..{hi_rank[r]}名)"
                helps.append({"slice": slice_id, "recording": r,
                              "pending_weight": proj.row_weight[i],
                              "resolves": note})
            if not helps:
                continue
            entries.append({
                "block": name,
                "cost": spec.cost,
                "available": self.archive.has_payload(self.version, name),
                "worth": sum(h["pending_weight"] for h in helps),
                "live_bits": helps,
            })
        entries.sort(key=lambda e: (not e["available"], -e["worth"],
                                    e["cost"], e["block"]))
        return entries

    def status(self) -> dict:
        """哪些距离确定后不必再读, 哪些仍可能改变结果。"""
        live = self._live_slices()
        ranked, _, lo_rank, hi_rank = self._analyze()
        k_eff = self._k_eff(ranked)
        settled, irrelevant, pending, excluded = {}, {}, {}, []
        for s, st in self._slices.items():
            if st.denom == 0:
                excluded.append(s)
                continue
            if st.unknown_w == 0:
                settled[s] = {"distance": _ratio(Fraction(st.num, st.denom))}
            elif s in live:
                pending[s] = {"recording": st.recording,
                              "pending_weight": st.unknown_w}
            else:
                beaten = self._beaten_by(s)
                if beaten is not None:
                    reason = f"被同组切片 {beaten} 压制"
                elif st.recording not in lo_rank:
                    reason = "录音没有可排名切片"
                elif lo_rank[st.recording] > k_eff:
                    reason = "录音已确定出前 k"
                else:
                    reason = "录音名次与见证已定"
                irrelevant[s] = reason
        return {"settled": settled, "irrelevant": irrelevant,
                "live": pending, "excluded": sorted(excluded)}

    # ---- 交付 ----

    def _delivery(self, rank: int, recording: str) -> dict:
        witness = self._witness(recording)
        lo, hi = self._recording_interval(recording)
        return {
            "rank": rank,
            "recording": recording,
            "witness": witness,
            "distance": _interval_json(lo, hi),
            # 见证未定时自动关联不得据以展开原片
            "associable": witness is not None,
        }

    def result(self) -> dict:
        """当前可交付结果: 已获证前缀 + 仍有争议的名次范围和依据。"""
        ranked, intervals, lo_rank, hi_rank = self._analyze()
        k_eff = self._k_eff(ranked)
        confirmed: list[tuple[int, str]] = []
        for rank in range(1, k_eff + 1):
            holders = [r for r in ranked
                       if lo_rank[r] == hi_rank[r] == rank]
            if len(holders) != 1:
                break
            confirmed.append((rank, holders[0]))
        # 已签出的确定名次不能撤回: 只许延伸, 不许改判。
        if len(confirmed) >= len(self._sealed):
            if confirmed[: len(self._sealed)] != self._sealed:
                raise AssertionError("内部错误: 已签出名次被改判")
            self._sealed = confirmed
        sealed = self._sealed

        live = self._live_slices()
        # 争议分组: 名次区间有交叠的录音才可能互换, 取连通分量。
        contending = [r for r in ranked
                      if lo_rank[r] <= k_eff
                      and not (lo_rank[r] == hi_rank[r])]
        contending.sort(key=lambda r: (lo_rank[r], hi_rank[r], r))
        components: list[list[str]] = []
        for r in contending:
            if components:
                last = components[-1]
                if lo_rank[r] <= max(hi_rank[x] for x in last):
                    last.append(r)
                    continue
            components.append([r])
        disputed_out = []
        for group in components:
            lo = min(lo_rank[r] for r in group)
            hi = min(max(hi_rank[r] for r in group), k_eff)
            candidates = []
            for r in group:
                pending_blocks = [
                    name for name in self.archive.blocks
                    if name not in self._read_blocks
                    and any(self._slices[self._projections[name].spec.rows[i]].recording == r
                            for i in self._block_live_rows(name, live))
                ]
                candidates.append({
                    "recording": r,
                    "rank_range": [lo_rank[r], hi_rank[r]],
                    "distance": _interval_json(*intervals[r]),
                    "pending_blocks": sorted(pending_blocks),
                })
            disputed_out.append({"ranks": [lo, hi] if lo != hi else [lo],
                                 "candidates": candidates})

        deliveries = [self._delivery(rank, r) for rank, r in sealed]
        complete = (len(sealed) == k_eff
                    and all(d["associable"] for d in deliveries))
        plans = self.plan()
        available = [p for p in plans if p["available"]]
        lost = [p["block"] for p in plans if not p["available"]]
        affordable = [p for p in available
                      if p["cost"] <= self.quota_remaining]
        if complete:
            state = "complete"
        elif not available and lost:
            # 必要块已失去: 旧证据不能沿用到新档案, 必须从新查询开始
            state = "blocked_version"
        elif available and not affordable:
            state = "blocked_quota"
        else:
            state = "partial"
        return {
            "version": self.version,
            "k": self.k,
            "status": state,
            "complete": complete,
            "confirmed": deliveries,
            "disputed": disputed_out,
            "requires_new_query": state == "blocked_version",
            "lost_blocks": lost,
            "quota": {"spent": self.quota_spent,
                      "remaining": self.quota_remaining},
            "workspace_bytes": self._workspace_used,
            "workspace_budget": self.archive.budget_bytes,
        }

    # ---- 交接 ----

    def to_dict(self) -> dict:
        """把未决查询交给值班端继续的可序列化形式。"""
        return {
            "version": self.version,
            "k": self.k,
            "quota_remaining": self.quota_remaining,
            "quota_spent": self.quota_spent,
            "slices": {s: {"num": st.num, "delivered": st.delivered}
                       for s, st in self._slices.items()},
            "blocks_read": sorted(self._read_blocks),
            "cache": {b: base64.b64encode(words.astype("<u8").tobytes()).decode("ascii")
                      for b, words in self._cache.items()},
            "sealed": [[rank, r] for rank, r in self._sealed],
        }

    @classmethod
    def from_dict(cls, archive: Archive, data: dict) -> "QuerySession":
        """在同一档案上恢复值班端交回的查询。"""
        session = cls(archive, data["version"], data["quota_remaining"],
                      query_bits=archive.query_bits, weights=archive.weights,
                      query_unavailable=archive.query_unavailable, k=data["k"])
        session.quota_spent = data["quota_spent"]
        for slice_id, saved in data["slices"].items():
            st = session._slices[slice_id]
            st.num = saved["num"]
            st.delivered = saved["delivered"]
            st.unknown_w = weighted_popcount(
                st.comparable & ~st.delivered, session.weights
            )
        for block_id in data["blocks_read"]:
            session._read_blocks.add(block_id)
            words = np.frombuffer(base64.b64decode(data["cache"][block_id]),
                                  dtype="<u8")
            session._cache[block_id] = words
            spec = archive.blocks[block_id]
            session._workspace_used += (CACHE_BASE_PER_BLOCK
                                        + CACHE_BYTES_PER_ROW * len(spec.rows))
        if session._workspace_used > archive.budget_bytes:
            raise WorkspaceError("恢复的查询超出工作区预算")
        session._sealed = [tuple(item) for item in data["sealed"]]
        return session
