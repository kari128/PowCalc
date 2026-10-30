#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "rich>=13.7", "numpy>=1.26", "scipy>=1.11", "psutil>=5.9",
#     "cupy-cuda12x", "gmpy2", "numba",
# ]
# ///
"""
PowCalc — x^y を (できるだけ) 速く厳密計算する CPU/GPU ハイブリッド電卓

■ アルゴリズム
  1. x の末尾の 0 を除去 (x = m·10^t → x^y = m^y · 10^(t·y))
  2. 左→右 二進法 (square-and-multiply)。先頭の小さい部分は GMP で厳密計算
  3. 大きい部分は「10進リム (10^3 進) 配列 + FFT 畳み込み」で二乗
       - GPU : CuPy (cuFFT)             … 自動検出
       - CPU : scipy.fft (全コア並列)   … GPU が無い/足りない時
     10進のまま演算するため、最後の 10進変換 (最大のボトルネック) が不要
  4. 結果は mod (B-1), mod (B+1) の検算 (casting-out) をしてから txt へ出力

■ 使い方 (uv 推奨)
  uv run powcalc.py 3 100000000           # CuPy / gmpy2 / numba も自動で導入される
  uv run powcalc.py 7 "10**8"
  (uv が PATH に無い場合は  python -m uv run powcalc.py ...  )
  uv run powcalc.py --selftest            # 自己テスト (GPU 経路の検証にも)

  オプション: --backend auto|gpu|cpu|gmp   --out DIR   --base 1000|100|10
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import math
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np
import psutil
import scipy.fft as sfft
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

sys.set_int_max_str_digits(0)

try:
    import gmpy2

    HAVE_GMP = True
except Exception:  # pragma: no cover
    gmpy2, HAVE_GMP = None, False
try:
    import cupy as cp

    HAVE_CUPY = True
except Exception:  # pragma: no cover
    cp, HAVE_CUPY = None, False
try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:

    @njit(parallel=True, cache=True)
    def _nb_digsum(x, B, K, out):
        for i in prange(out.shape[0]):
            s = np.int64(0)
            p = np.int64(1)
            for k in range(K):
                j = i - k
                if j < 0:
                    break
                v = np.int64(x[j])
                s += (v // p) % B
                p *= B
            out[i] = s

    @njit(parallel=True, cache=True)
    def _nb_round(t, B, out):
        for i in prange(out.shape[0]):
            v = t[i] % B
            if i > 0:
                v += t[i - 1] // B
            out[i] = v


# ───────────────────────────── 基本ユーティリティ ─────────────────────────────
class Cancelled(Exception): ...


class PrecisionError(Exception): ...


BASES = (1000, 100, 10)
PRECISION_BITS = 45.0  # 2·log2(B-1) + log2(n) がこれ以下になる基数を選ぶ
ERR_LIMIT = 0.25  # FFT 丸め誤差の許容上限 (0.5 で破綻)
PHASES = [
    "準備・自己検証",
    "CPU前処理 (先頭部を厳密計算)",
    "本計算 (FFT二乗ループ)",
    "検算 (mod B±1)",
    "結果出力 (txt)",
]


def kdig(B: int) -> int:
    return {10: 1, 100: 2, 1000: 3}[B]


def ndigits(v: int, B: int) -> int:
    """v を B 進で表した桁数 (v=0 でも 1)"""
    k, p = 1, B
    while p <= v:
        p *= B
        k += 1
    return k


@lru_cache(maxsize=4096)
def fast_len(n: int) -> int:
    """n 以上で 2^a·3^b·5^c (偶数) の最小値 = FFT 長"""
    best = 1 << max(1, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        v = p5
        while v < best:
            q = max(2, -(-n // v))
            t = v * (1 << (q - 1).bit_length())
            if t < best:
                best = t
            v *= 3
        p5 *= 5
    return best


def fmt_time(s: float | None) -> str:
    if s is None or not math.isfinite(s):
        return "--:--"
    s = max(0.0, s)
    if s < 60:
        return f"{s:.2f}秒" if s < 10 else f"{s:.1f}秒"
    h, r = divmod(int(s + 0.5), 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.2f} {u}"
        n /= 1024
    return str(n)


def short(v: int) -> str:
    s = str(abs(v))
    sign = "-" if v < 0 else ""
    return sign + s if len(s) <= 18 else f"{sign}{s[:6]}…({len(s)}桁)"


def parse_int(text: str) -> int:
    """整数 / 1e9 / 10**9 / 2^64 / 1,000,000 などを安全にパース"""
    t = text.strip().replace(",", "").replace("_", "").replace("^", "**").replace("×", "*")
    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)

    def ev(n: ast.AST) -> int:
        if isinstance(n, ast.Constant):
            v = n.value
            if isinstance(v, bool):
                raise ValueError
            if isinstance(v, int):
                return v
            if isinstance(v, float) and v.is_integer():
                return int(Decimal(repr(v)))
            raise ValueError
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if b < 0 or (abs(a) > 1 and b * a.bit_length() > 100_000_000):
                raise ValueError("式が大きすぎます")
            return a**b
        raise ValueError

    try:
        return ev(ast.parse(t, mode="eval").body)
    except Exception as e:
        raise ValueError(f"整数として解釈できません: {text!r}") from e


# ─────────────────────────────── ジョブ定義 ───────────────────────────────
@dataclass
class Job:
    x: int
    y: int
    m: int = 0  # |x| から末尾の 10 を除いた値
    zeros: int = 0  # 末尾に付く 0 の個数 = t·y
    sign: str = ""
    prefix_digits: int = 100_000  # CPU(GMP)で先に厳密計算する桁数の目安
    est_digits: int = 0  # m^y の桁数 (推定)
    trivial: str | None = None  # 自明な結果 (文字列)

    def __post_init__(self):
        x, y = self.x, self.y
        if y < 0:
            raise ValueError("y は 0 以上の整数にしてください (負の指数は有理数になります)")
        if y == 0:
            self.trivial = "1"
            return
        if x == 0:
            self.trivial = "0"
            return
        self.sign = "-" if (x < 0 and y % 2 == 1) else ""
        m, t = abs(x), 0
        while m % 10 == 0:
            m //= 10
            t += 1
        self.m, self.zeros = m, t * y
        lg = math.log10(m) if m > 1 else 0.0
        self.est_digits = int(y * lg) + 1


def digits_to_limbs(s: str, B: int) -> np.ndarray:
    """10進文字列 → リトルエンディアン int16 リム配列"""
    k = kdig(B)
    a = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 48
    pad = (-len(a)) % k
    if pad:
        a = np.concatenate([np.zeros(pad, np.uint8), a])
    w = (10 ** np.arange(k - 1, -1, -1)).astype(np.int16)
    limbs = (a.reshape(-1, k).astype(np.int16) * w).sum(axis=1, dtype=np.int16)
    return np.ascontiguousarray(limbs[::-1])


# ─────────────────────────────── FFT エンジン ───────────────────────────────
class FFTEngine:
    """リム配列 (int16, little-endian, 10^k 進) 上の二乗/乗算。xp = cupy | numpy"""

    def __init__(self, kind: str):
        self.kind = kind
        self.fast = False
        self.last_L = 0
        self.max_L = 0
        self.n_fft = 0
        if kind == "gpu":
            self.xp = cp
            props = cp.cuda.runtime.getDeviceProperties(0)
            self.name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            self.rfft = lambda a, n: cp.fft.rfft(a, n=n)
            self.irfft = lambda a, n: cp.fft.irfft(a, n=n)
            self._k_digsum = cp.ElementwiseKernel(
                "raw T x, int64 B, int32 K",
                "int64 y",
                """
                long long s = 0; long long p = 1;
                for (int k = 0; k < K; k++) {
                    long long j = (long long)i - k;
                    if (j < 0) break;
                    long long v = (long long)x[j];
                    s += (v / p) % B;
                    p *= B;
                }
                y = s;
                """,
                "powcalc_digsum",
            )
            self._k_round = cp.ElementwiseKernel(
                "raw int64 x, int64 B",
                "int64 y",
                """
                long long v = x[i] % B;
                if (i > 0) v += x[i - 1] / B;
                y = v;
                """,
                "powcalc_round",
            )
            self._fast_digsum = lambda r, B, K, t: self._k_digsum(r, B, K, t)
            self._fast_round = lambda t, B, o: self._k_round(t, B, o)
            self.fast_name = "CUDA融合カーネル"
            self.threads = props.get("multiProcessorCount", 0)
        else:
            self.xp = np
            self.name = f"CPU {os.cpu_count()}スレッド (scipy.fft)"
            w = os.cpu_count() or 1
            self.rfft = lambda a, n: sfft.rfft(a, n=n, workers=w)
            self.irfft = lambda a, n: sfft.irfft(a, n=n, workers=w, overwrite_x=True)
            if HAVE_NUMBA:
                self._fast_digsum = lambda r, B, K, t: _nb_digsum(r, B, K, t)
                self._fast_round = lambda t, B, o: _nb_round(t, B, o)
                self.fast_name = "numba並列カーネル"
            else:
                self._fast_digsum = None
                self.fast_name = "NumPy汎用"

    # ── デバイス/メモリ ──
    def to_dev(self, a):
        return self.xp.asarray(a)

    def to_host(self, a):
        return a.get() if self.kind == "gpu" else a

    def sync(self):
        if self.kind == "gpu":
            cp.cuda.Stream.null.synchronize()

    def mem(self) -> tuple[int, int]:
        """(使用中, 合計) バイト"""
        if self.kind == "gpu":
            free, total = cp.cuda.runtime.memGetInfo()
            return total - free, total
        vm = psutil.virtual_memory()
        return vm.total - vm.available, vm.total

    def free_bytes(self) -> int:
        used, total = self.mem()
        return total - used

    def release(self):
        if self.kind == "gpu":
            cp.get_default_memory_pool().free_all_blocks()
            try:
                cp.fft.config.get_plan_cache().clear()
            except Exception:
                pass

    # ── 繰り上がり処理 ──
    def _digsum_ref(self, r, B, K):
        xp = self.xp
        q = r.astype(xp.int64)
        t = xp.zeros_like(q)
        for k in range(K):
            if k < K - 1:
                qq = q // B
                d = qq * B
                xp.subtract(q, d, out=d)
            else:
                qq, d = None, q
            if k == 0:
                t += d
            else:
                t[k:] += d[:-k]
            q = qq
        return t

    def _lookahead(self, t, B):
        """各桁 ≤ 2B-2 のとき、繰り上がりを並列プレフィックスで一括解決"""
        xp = self.xp
        nonp = t != (B - 1)
        rid = xp.cumsum(nonp, dtype=xp.int64)
        gen = (t >= B)[nonp]
        gx = xp.concatenate([xp.zeros(1, dtype=xp.bool_), gen])
        cout = gx[rid]
        del rid, gen, gx, nonp
        t[1:] += cout[:-1]
        t -= (t >= B) * B
        return t.astype(xp.int16)

    def normalize(self, r, B, bound, fast=None):
        """係数配列 r (非負整数値, 長さ=出力リム数) → 正規化リム (int16)"""
        xp = self.xp
        use_fast = self.fast if fast is None else fast
        K = ndigits(bound, B)
        if use_fast:
            t = xp.empty(r.shape[0], dtype=xp.int64)
            self._fast_digsum(r, B, K, t)
        else:
            t = self._digsum_ref(r, B, K)
        bt = K * (B - 1)
        while bt > 2 * B - 2:
            if use_fast:
                t2 = xp.empty_like(t)
                self._fast_round(t, B, t2)
                t = t2
            else:
                h = t // B
                t -= h * B
                t[1:] += h[:-1]
            bt = (B - 1) + bt // B
        return self._lookahead(t, B)

    def trim(self, a):
        n = a.shape[0]
        while n > 1 and int(a[n - 1]) == 0:
            n -= 1
        return a if n == a.shape[0] else a[:n].copy()

    # ── 演算 ──
    def mul(self, a, b, B):
        """a*b (b=None なら二乗)。FFT 畳み込み → 丸め誤差検査 → 繰り上がり"""
        xp = self.xp
        na = a.shape[0]
        nb = na if b is None else b.shape[0]
        m = na + nb - 1
        L = fast_len(m)
        self.last_L = L
        self.max_L = max(self.max_L, L)
        fa = self.rfft(a.astype(xp.float64), L)
        if b is None:
            fa *= fa
            self.n_fft += 2
        else:
            fb = self.rfft(b.astype(xp.float64), L)
            fa *= fb
            del fb
            self.n_fft += 3
        c = self.irfft(fa, L)
        del fa
        r = xp.zeros(na + nb, dtype=xp.float64)
        xp.rint(c[:m], out=r[:m])
        c = c[:m]
        c -= r[:m]
        xp.abs(c, out=c)
        err = float(c.max())
        del c
        if err > ERR_LIMIT:
            raise PrecisionError(f"FFT丸め誤差 {err:.3f} > {ERR_LIMIT}")
        self.max_err = max(getattr(self, "max_err", 0.0), err)
        bound = min(na, nb) * (B - 1) ** 2
        out = self.normalize(r, B, bound)
        del r
        return self.trim(out)

    def mul_small(self, a, x: int, B):
        """a * x (x < 2^40 のスカラー)。FFT 不要で O(n)"""
        xp = self.xp
        na = a.shape[0]
        n_out = na + ndigits(x, B)
        ci = xp.zeros(n_out, dtype=xp.float64 if self.fast else xp.int64)
        ci[:na] = a
        ci *= x  # float64 でも (B-1)·x < 2^53 なので厳密
        self.last_L = n_out
        out = self.normalize(ci, B, (B - 1) * x)
        del ci
        return self.trim(out)

    # ── 自己検証 / 校正 ──
    def selftest(self, log) -> None:
        """融合カーネルを汎用実装・Python 整数と突き合わせ。不一致なら無効化"""
        rng = np.random.default_rng(7)
        for B in BASES:
            n = 1500
            bound = 2**40
            c = np.zeros(n, dtype=np.float64)
            c[: n - 16] = rng.integers(0, bound, size=n - 16).astype(np.float64)
            truth = sum(int(v) * B**i for i, v in enumerate(c))
            xr = self.xp.asarray(c)
            ref = self.to_host(self.normalize(xr, B, bound, fast=False))
            got_ref = sum(int(v) * B**i for i, v in enumerate(ref))
            if got_ref != truth:
                raise RuntimeError("汎用繰り上がり処理が不正 (内部エラー)")
            if self._fast_digsum is not None:
                try:
                    fast = self.to_host(self.normalize(xr, B, bound, fast=True))
                    ok = sum(int(v) * B**i for i, v in enumerate(fast)) == truth
                except Exception as e:  # コンパイル失敗など
                    ok, msg = False, str(e).splitlines()[0][:80]
                    log(f"融合カーネル失敗: {msg}")
                if not ok:
                    self.fast = False
                    log("融合カーネルの結果が一致せず → 汎用実装に切替")
                    return
        self.fast = self._fast_digsum is not None

    def calibrate(self, B: int) -> float:
        """二乗 1 回の実測 → 1コスト単位あたりの秒数 (ウォームアップ兼ねる)"""
        rng = np.random.default_rng(1)
        n = 1 << 16
        a = rng.integers(0, B, size=n).astype(np.int16)
        a[-1] = max(1, int(a[-1]))
        a = self.to_dev(a)
        dt = 0.0
        for _ in range(2):
            self.sync()
            t0 = time.perf_counter()
            self.mul(a, None, B)
            self.sync()
            dt = time.perf_counter() - t0
        self.release()
        self.n_fft = 0
        self.max_L = 0
        return dt / cost_sq(n)


def cost_sq(n: int) -> float:
    L = fast_len(2 * n - 1)
    return L * (math.log2(L) + 30.0)


def cost_mul(n: int, scalar: bool) -> float:
    return n * 18.0 if scalar else 1.5 * cost_sq(n)


@dataclass
class Op:
    kind: str  # 'sq' | 'mul'
    n: int  # 入力リム数 (推定)
    cost: float
    L: int  # 想定 FFT 長 / 作業配列長


# ─────────────────────────────── 進捗状態 ───────────────────────────────
@dataclass
class State:
    x_disp: str = ""
    y_disp: str = ""
    backend_desc: str = ""
    accel: str = ""
    target_digits: int = 0
    base: int = 0
    phase: int = -1
    phase_t: dict = field(default_factory=dict)
    skipped: set = field(default_factory=set)
    ops: list = field(default_factory=list)
    cum: list = field(default_factory=list)
    total_cost: float = 1.0
    op_i: int = 0
    op_t0: float = 0.0
    op_active: bool = False
    rates: dict = field(default_factory=dict)
    cur_digits: int = 0
    fft_len: int = 0
    mem_used: int = 0
    mem_total: int = 0
    mem_peak: int = 0
    n_sq: int = 0
    n_mul: int = 0
    note: str = ""
    logs: list = field(default_factory=list)
    write_done: int = 0
    write_total: int = 1
    cancel: bool = False
    done: bool = False
    error: str | None = None
    t_compute0: float = 0.0

    def log(self, msg: str):
        self.logs.append(msg)
        self.logs = self.logs[-4:]

    def begin_phase(self, i: int):
        self.phase = i
        self.phase_t[i] = [time.perf_counter(), None]

    def end_phase(self, i: int):
        if i in self.phase_t:
            self.phase_t[i][1] = time.perf_counter()

    def phase_time(self, i: int) -> float | None:
        if i not in self.phase_t:
            return None
        s, e = self.phase_t[i]
        return (e or time.perf_counter()) - s

    def set_ops(self, ops: list[Op]):
        self.ops = ops
        self.cum, c = [], 0.0
        for o in ops:
            self.cum.append(c)
            c += o.cost
        self.total_cost = max(c, 1.0)
        self.op_i, self.op_active = 0, False

    def pred(self, op: Op) -> float:
        r = self.rates.get(op.kind) or self.rates.get("sq") or 0.0
        return r * op.cost

    def progress(self) -> tuple[float, float | None]:
        """(0..1, 推定残り秒)。実行中オペレーションは前回実測から所要時間を予測して補間"""
        if not self.ops:
            return (1.0 if self.phase > 2 else 0.0), (0.0 if self.phase > 2 else None)
        i = self.op_i
        if i >= len(self.ops):
            return 1.0, 0.0
        cur = self.ops[i]
        frac, rem_cur = 0.0, None
        pr = self.pred(cur)
        if self.op_active and pr > 0:
            el = time.perf_counter() - self.op_t0
            frac = min(el / pr, 0.97)
            rem_cur = max(pr - el, 0.03 * pr)
        elif pr > 0:
            rem_cur = pr
        p = (self.cum[i] + frac * cur.cost) / self.total_cost
        if rem_cur is None:
            return p, None
        return p, rem_cur + sum(self.pred(o) for o in self.ops[i + 1 :])

    def sample_mem(self, eng: FFTEngine | None):
        if eng is None:
            self.mem_used = psutil.Process().memory_info().rss
            self.mem_total = psutil.virtual_memory().total
        else:
            self.mem_used, self.mem_total = eng.mem()
        self.mem_peak = max(self.mem_peak, self.mem_used)


# ─────────────────────────────── 計算本体 ───────────────────────────────
@dataclass
class Rep:
    s: str | None = None  # 10進文字列 (GMP/自明ケース)
    limbs: object = None  # デバイス上のリム配列 (FFT)
    B: int = 0
    eng: FFTEngine | None = None


def choose_base(est_digits: int, forced: int | None) -> int:
    if forced:
        return forced
    for B in BASES:
        n_op = est_digits // kdig(B) // 2 + 2
        if 2 * math.log2(B - 1) + math.log2(max(n_op, 2)) <= PRECISION_BITS:
            return B
    return 10


def plan_prefix(job: Job) -> tuple[int, str]:
    """先頭 j ビット分の指数 e と、残りビット列"""
    bits = bin(job.y)[2:]
    lg = math.log10(job.m) if job.m > 1 else 0.0
    e, j = 1, 1
    while j < len(bits):
        e2 = (e << 1) | int(bits[j])
        if e2 * lg > job.prefix_digits:
            break
        e, j = e2, j + 1
    return e, bits[j:]


def build_ops(job: Job, rem_bits: str, e: int, B: int, scalar: bool) -> list[Op]:
    kd = kdig(B)
    lg = math.log10(job.m) if job.m > 1 else 0.0
    d = e * lg + 1
    ops: list[Op] = []
    for bit in rem_bits:
        n = max(1, math.ceil(d / kd))
        ops.append(Op("sq", n, cost_sq(n), fast_len(2 * n - 1)))
        d *= 2
        if bit == "1":
            n2 = max(1, math.ceil(d / kd))
            ops.append(Op("mul", n2, cost_mul(n2, scalar), n2 if scalar else fast_len(2 * n2)))
            d += lg
    return ops


def compute_fft(eng: FFTEngine, job: Job, B: int, st: State, prefix_str: str, rem_bits: str, e: int) -> Rep:
    st.base = B
    a = eng.to_dev(digits_to_limbs(prefix_str, B))
    scalar = job.m < (1 << 40)
    xl = None if scalar else eng.to_dev(digits_to_limbs(str(job.m), B))
    ops = build_ops(job, rem_bits, e, B, scalar)
    st.set_ops(ops)
    need = int(36 * max((o.L for o in ops), default=0))
    free = eng.free_bytes()
    if need > free * 0.9:
        raise MemoryError(f"必要メモリ ≈ {fmt_bytes(need)} / 空き {fmt_bytes(free)}")
    st.begin_phase(2)
    st.t_compute0 = time.perf_counter()
    st.cur_digits = len(prefix_str)
    st.note = "FFT畳み込み + 繰り上がり並列処理"
    for i, op in enumerate(ops):
        if st.cancel:
            raise Cancelled
        st.op_i, st.op_t0, st.op_active = i, time.perf_counter(), True
        st.fft_len = op.L
        if op.kind == "sq":
            a = eng.mul(a, None, B)
            st.n_sq += 1
        else:
            a = eng.mul_small(a, job.m, B) if scalar else eng.mul(a, xl, B)
            st.n_mul += 1
        eng.sync()
        dt = time.perf_counter() - st.op_t0
        st.rates[op.kind] = dt / op.cost
        st.op_active = False
        st.cur_digits = (a.shape[0] - 1) * kdig(B) + len(str(int(a[-1])))
        eng.release()
        st.sample_mem(eng)
    st.op_i = len(ops)
    st.end_phase(2)
    return Rep(limbs=a, B=B, eng=eng)


def compute_gmp(job: Job, st: State, pv, rem_bits: str, e: int) -> Rep:
    mpz = gmpy2.mpz if HAVE_GMP else int
    m = mpz(job.m)
    lg = math.log10(job.m) if job.m > 1 else 0.0
    d, ops = e * lg + 1, []
    for bit in rem_bits:
        ops.append(Op("sq", int(d), d * math.log2(d + 2), int(d)))
        d *= 2
        if bit == "1":
            ops.append(Op("mul", int(d), d * 2.0, int(d)))
            d += lg
    st.set_ops(ops)
    st.begin_phase(2)
    st.t_compute0 = time.perf_counter()
    st.note = "GMP 二乗ループ (シングルスレッド)"
    r = pv
    for i, op in enumerate(ops):
        if st.cancel:
            raise Cancelled
        st.op_i, st.op_t0, st.op_active = i, time.perf_counter(), True
        if op.kind == "sq":
            r = r * r
            st.n_sq += 1
        else:
            r = r * m
            st.n_mul += 1
        st.rates[op.kind] = (time.perf_counter() - st.op_t0) / op.cost
        st.op_active = False
        st.cur_digits = int(op.n * (2 if op.kind == "sq" else 1))
        st.sample_mem(None)
    st.op_i = len(ops)
    st.note = "10進変換中 (GMP, シングルスレッド・進捗不明)"
    st.op_t0, st.op_active = time.perf_counter(), False
    s = r.digits(10) if HAVE_GMP else str(r)
    st.cur_digits = len(s)
    st.end_phase(2)
    return Rep(s=s, B=1000)


def compute(job: Job, kind: str, st: State, forced_base: int | None = None) -> tuple[Rep, int]:
    """kind: gpu | cpu | gmp。戻り値: (Rep, 使用した基数)"""
    # ── 準備 ──
    e, rem_bits = plan_prefix(job)
    st.begin_phase(0)
    eng = None
    if kind in ("gpu", "cpu"):
        st.note = "エンジン初期化 / 自己検証"
        eng = FFTEngine(kind)
        st.backend_desc = eng.name + ("  [cuFFT]" if kind == "gpu" else "")
        eng.selftest(st.log)
        st.accel = f"繰り上がり: {eng.fast_name if eng.fast else 'NumPy汎用'}"
    else:
        st.backend_desc = "GMP (gmpy2, 単一スレッド)" if HAVE_GMP else "Python int (gmpy2 なし)"
        st.accel = ""
    st.end_phase(0)

    # ── CPU 前処理 ──
    st.begin_phase(1)
    st.note = "GMP で先頭部分 m^e を厳密計算"
    pv = (gmpy2.mpz(job.m) ** e) if HAVE_GMP else (job.m**e)
    if not rem_bits:
        st.skipped.add(2)
        st.end_phase(1)
        s = pv.digits(10) if HAVE_GMP else str(pv)
        st.cur_digits = len(s)
        return Rep(s=s, B=1000), 1000
    if kind == "gmp":
        st.end_phase(1)
        return compute_gmp(job, st, pv, rem_bits, e), 1000

    B = choose_base(job.est_digits, forced_base)
    while True:
        try:
            st.begin_phase(1)
            prefix_str = pv.digits(10) if HAVE_GMP else str(pv)
            st.end_phase(1)
            # 校正 (rate 初期値)
            st.begin_phase(0)
            st.note = f"速度校正 (基数 10^{kdig(B)})"
            rate = eng.calibrate(B)
            st.rates = {"sq": rate}
            st.end_phase(0)
            rep = compute_fft(eng, job, B, st, prefix_str, rem_bits, e)
            return rep, B
        except PrecisionError as ex:
            nxt = [b for b in BASES if b < B]
            if not nxt:
                raise
            st.log(f"{ex} → 基数 10^{kdig(B)} → 10^{kdig(nxt[0])} で再計算")
            B = nxt[0]
            st.n_sq = st.n_mul = 0
            st.rates = {}
            eng.max_err = 0.0
            eng.release()


# ─────────────────────── 検算 / 出力 ───────────────────────
def verify(job: Job, rep: Rep) -> tuple[bool, str]:
    B = rep.B
    if rep.s is not None:
        mods = (B - 1, B + 1)
        n = gmpy2.mpz(rep.s) if HAVE_GMP else int(rep.s)
        res = [int(n % q) for q in mods]
    else:
        xp, a = rep.eng.xp, rep.limbs
        s_all = int(xp.sum(a, dtype=xp.int64))
        s_alt = int(xp.sum(a[0::2], dtype=xp.int64)) - int(xp.sum(a[1::2], dtype=xp.int64))
        res = [s_all % (B - 1), s_alt % (B + 1)]
    exp = [pow(job.m, job.y, B - 1), pow(job.m, job.y, B + 1)]
    ok = res == exp
    # 桁数の整合 (±1 を許容)
    nd = rep_digits(rep)
    ok_len = abs(nd - job.est_digits) <= 1 if job.est_digits else True
    msg = f"mod {B-1}: {'OK' if res[0]==exp[0] else 'NG'} / mod {B+1}: {'OK' if res[1]==exp[1] else 'NG'} / 桁数: {'OK' if ok_len else 'NG'}"
    return ok and ok_len, msg


def rep_digits(rep: Rep) -> int:
    if rep.s is not None:
        return len(rep.s)
    kd = kdig(rep.B)
    a = rep.limbs
    return (a.shape[0] - 1) * kd + len(str(int(a[-1])))


def head_tail(rep: Rep, job: Job, n: int = 24) -> tuple[str, str]:
    if rep.s is not None:
        s = rep.s
    else:
        kd, a, eng = kdig(rep.B), rep.limbs, rep.eng
        cnt = min(a.shape[0], n // kd + 3)
        top = eng.to_host(a[-cnt:])[::-1]
        s_top = str(int(top[0])) + "".join(f"{int(v):0{kd}d}" for v in top[1:])
        bot = eng.to_host(a[:cnt])[::-1]
        s_bot = "".join(f"{int(v):0{kd}d}" for v in bot)
        if a.shape[0] <= cnt:
            s = s_top
        else:
            return job.sign + s_top[:n], (s_bot + "0" * job.zeros)[-n:] if job.zeros < n else "0" * n
    head = job.sign + s[:n]
    tail = (s + "0" * min(job.zeros, n))[-n:]
    return head, tail


def iter_ascii(rep: Rep, chunk_limbs: int = 1 << 24):
    """結果を (bytes, 進捗0..1) のチャンクで先頭から返す"""
    if rep.s is not None:
        s, n, step = rep.s, len(rep.s), 1 << 26
        for i in range(0, n, step):
            yield s[i : i + step].encode("ascii"), min(1.0, (i + step) / n)
        return
    eng, xp, a, kd = rep.eng, rep.eng.xp, rep.limbs, kdig(rep.B)
    n = a.shape[0]
    lead = kd - len(str(int(a[n - 1])))
    hi, first = n, True
    while hi > 0:
        lo = max(0, hi - chunk_limbs)
        part = a[lo:hi][::-1].astype(xp.int32)
        digs = xp.empty((part.shape[0], kd), dtype=xp.uint8)
        for pos in range(kd - 1, -1, -1):
            q = part // 10
            digs[:, pos] = (part - q * 10 + 48).astype(xp.uint8)
            part = q
        buf = eng.to_host(digs.reshape(-1)).tobytes()
        if first:
            buf, first = buf[lead:], False
        yield buf, (n - lo) / n
        hi = lo


def rep_to_str(rep: Rep) -> str:
    return b"".join(b for b, _ in iter_ascii(rep)).decode("ascii")


def write_result(job: Job, rep: Rep, path: Path, st: State) -> dict:
    total_digits = rep_digits(rep) + job.zeros
    st.write_total = max(1, total_digits + len(job.sign))
    st.write_done = 0
    h = hashlib.sha256()
    with open(path, "wb", buffering=1 << 22) as f:
        def put(b: bytes):
            f.write(b)
            h.update(b)
            st.write_done += len(b)

        if job.sign:
            put(job.sign.encode())
        for buf, _ in iter_ascii(rep):
            if st.cancel:
                raise Cancelled
            put(buf)
        z = job.zeros
        blk = b"0" * (1 << 24)
        while z > 0:
            k = min(z, len(blk))
            put(blk[:k])
            z -= k
    return {"sha256": h.hexdigest(), "size": path.stat().st_size, "digits": total_digits}


# ─────────────────────────────── パイプライン ───────────────────────────────
@dataclass
class Report:
    path: Path | None = None
    info: dict = field(default_factory=dict)
    kind: str = ""
    base: int = 0
    ok: bool = True
    verify_msg: str = ""
    head: str = ""
    tail: str = ""
    total: float = 0.0
    max_fft: int = 0
    n_fft: int = 0
    max_err: float = 0.0


def detect_backends(requested: str) -> list[str]:
    if requested != "auto":
        return [requested]
    order = []
    if HAVE_CUPY:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                cp.fft.rfft(cp.ones(64))  # cuFFT が実際に動くか
                order.append("gpu")
        except Exception:
            pass
    order.append("cpu")
    return order


def pipeline(job: Job, args, st: State) -> Report:
    t0 = time.perf_counter()
    rep_out = Report()
    st.x_disp, st.y_disp = short(job.x), short(job.y)
    st.target_digits = (job.est_digits + job.zeros) if not job.trivial else 1

    if job.trivial is not None:
        st.backend_desc = "自明なケース"
        for i in (0, 1, 2, 3):
            st.skipped.add(i)
        rep, B = Rep(s=job.trivial, B=1000), 1000
        st.cur_digits = 1
    else:
        rep, B, last_err = None, 0, None
        for kind in detect_backends(args.backend):
            try:
                rep, B = compute(job, kind, st, args.base)
                rep_out.kind = kind
                break
            except (MemoryError, Exception) as ex:
                if isinstance(ex, Cancelled):
                    raise
                oom = isinstance(ex, MemoryError) or ex.__class__.__name__ == "OutOfMemoryError"
                if not oom:
                    raise
                last_err = ex
                st.log(f"{kind.upper()} メモリ不足 → 次のバックエンドへ ({ex})")
                st.ops, st.n_sq, st.n_mul, st.rates, st.cur_digits = [], 0, 0, {}, 0
                st.phase_t.pop(2, None)
        if rep is None:
            raise MemoryError(f"どのバックエンドでもメモリが足りません: {last_err}")
        st.base = B
        if rep.eng is not None:
            rep_out.max_fft, rep_out.n_fft = rep.eng.max_L, rep.eng.n_fft
            rep_out.max_err = getattr(rep.eng, "max_err", 0.0)

    rep_out.base = B
    if not rep_out.kind:
        rep_out.kind = "trivial"

    # 検算
    st.begin_phase(3)
    st.note = "casting-out 検算"
    if job.trivial is None:
        rep_out.ok, rep_out.verify_msg = verify(job, rep)
    else:
        rep_out.ok, rep_out.verify_msg = True, "自明"
    st.end_phase(3)
    rep_out.head, rep_out.tail = head_tail(rep, job)
    if not rep_out.ok:
        st.log("⚠ 検算に失敗しました (結果は信用できません)")

    # 出力
    st.begin_phase(4)
    st.note = "txt へ書き出し (SHA-256 計算込み)"
    outdir = Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = (outdir / f"pow_{short(job.x).replace('…', '_').replace('(', '').replace(')', '')}"
            f"_pow_{short(job.y).replace('…', '_').replace('(', '').replace(')', '')}_{stamp}.txt").resolve()
    rep_out.info = write_result(job, rep, path, st)
    rep_out.path = path
    st.end_phase(4)
    rep_out.total = time.perf_counter() - t0
    st.sample_mem(rep.eng)
    return rep_out


# ─────────────────────────────── TUI ───────────────────────────────
class UI:
    def __init__(self, st: State, console: Console):
        self.st, self.console = st, console
        self.t0 = time.perf_counter()
        psutil.cpu_percent(None)

    def _bar(self, p: float, style="cyan", width=44):
        return ProgressBar(total=1000, completed=int(max(0, min(1, p)) * 1000), width=width,
                           complete_style=style, finished_style="green", pulse=False)

    def render(self):
        st, now = self.st, time.perf_counter()
        # ヘッダ
        head = Table.grid(padding=(0, 2))
        head.add_column(style="dim", justify="right")
        head.add_column()
        head.add_row("計算式", f"[bold cyan]{st.x_disp}[/] ^ [bold magenta]{st.y_disp}[/]")
        head.add_row("予想桁数", f"約 [bold]{st.target_digits:,}[/] 桁")
        head.add_row("実行環境", f"{st.backend_desc or '検出中…'}")
        if st.base:
            head.add_row("方式", f"10^{kdig(st.base)} 進リム FFT乗算 / {st.accel}")

        # フェーズ
        ph = Table.grid(padding=(0, 1))
        for i, name in enumerate(PHASES):
            if i in st.skipped:
                ph.add_row("[dim]–[/]", f"[dim]{name} (省略)[/]", "")
            elif i in st.phase_t and st.phase_t[i][1] is not None:
                ph.add_row("[green]✔[/]", f"[green]{name}[/]", f"[dim]{fmt_time(st.phase_time(i))}[/]")
            elif i == st.phase:
                ph.add_row(Spinner("dots", style="cyan"), f"[bold]{name}[/]", f"[dim]{fmt_time(st.phase_time(i))}[/]")
            else:
                ph.add_row("[dim]○[/]", f"[dim]{name}[/]", "")

        # 進捗バー
        p, eta = st.progress()
        if st.phase == 4:
            wp = st.write_done / st.write_total
            bar_p, label = wp, "出力"
            rem = None
            if st.phase_time(4) and wp > 0.02:
                rem = st.phase_time(4) * (1 - wp) / wp
            eta = rem
        elif st.phase >= 3:
            bar_p, label = 1.0, "計算"
        else:
            bar_p, label = p, "計算"
        grid = Table.grid(padding=(0, 1))
        grid.add_row(f"[bold]{label}[/]", self._bar(bar_p), f"[bold]{bar_p*100:5.1f}%[/]")
        elapsed = now - self.t0

        # 統計
        stats = Table.grid(padding=(0, 3))
        stats.add_column(style="dim")
        stats.add_column(justify="right")
        stats.add_column(style="dim")
        stats.add_column(justify="right")
        nops = len(st.ops)
        stats.add_row("経過時間", fmt_time(elapsed), "推定残り", f"[bold yellow]{fmt_time(eta)}[/]" if eta is not None else "[dim]算出中…[/]")
        stats.add_row("現在の桁数", f"{st.cur_digits:,}", "ステップ", f"{min(st.op_i + (1 if st.op_active else 0), nops)}/{nops}  (二乗 {st.n_sq} / 乗算 {st.n_mul})")
        comp_t = (now - st.t_compute0) if st.t_compute0 and st.phase <= 2 else None
        thr = f"{st.cur_digits / comp_t:,.0f} 桁/秒" if comp_t and comp_t > 0.5 and st.cur_digits else "—"
        stats.add_row("FFT長", f"{st.fft_len:,}" if st.fft_len else "—", "平均スループット", thr)
        mem = f"{fmt_bytes(st.mem_used)} / {fmt_bytes(st.mem_total)}" if st.mem_total else "—"
        stats.add_row("メモリ(VRAM/RAM)", mem, "ピーク", fmt_bytes(st.mem_peak) if st.mem_peak else "—")
        cpu = psutil.cpu_percent(None)
        stats.add_row("CPU使用率", f"{cpu:.0f}%", "状態", f"[dim]{st.note}[/]")

        logs = Text("\n".join(st.logs[-3:]), style="yellow") if st.logs else Text("")
        body = Group(head, Text(""), ph, Text(""), grid, Text(""), stats, logs)
        return Panel(body, title="[bold]⚡ PowCalc[/]  x^y 高速計算機", subtitle="[dim]Ctrl+C で中断[/]",
                     border_style="cyan", box=box.ROUNDED, padding=(1, 2))


def summary_panel(job: Job, r: Report, st: State) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    ok = "[bold green]✔ 検算OK[/]" if r.ok else "[bold red]✘ 検算NG[/]"
    t.add_row("計算式", f"[cyan]{short(job.x)}[/] ^ [magenta]{short(job.y)}[/]")
    t.add_row("出力先", f"[bold underline]{r.path}[/]")
    t.add_row("ファイルサイズ", f"{fmt_bytes(r.info['size'])}  ({r.info['size']:,} bytes)")
    t.add_row("総桁数", f"[bold]{r.info['digits']:,}[/] 桁" + (f"  (末尾の 0 を {job.zeros:,} 個含む)" if job.zeros else ""))
    t.add_row("先頭 / 末尾", f"{r.head} … {r.tail}")
    t.add_row("SHA-256", r.info["sha256"])
    t.add_row("検算", f"{ok}  [dim]{r.verify_msg}[/]")
    t.add_row("", "")
    comp = st.phase_time(2) or 0.0
    digits = r.info["digits"]
    t.add_row("実行環境", st.backend_desc)
    t.add_row("基数 / 方式", f"10^{kdig(r.base)} 進  " + (f"FFT最大長 {r.max_fft:,} · FFT {r.n_fft} 回 · 最大丸め誤差 {r.max_err:.4f}" if r.max_fft else "GMP"))
    t.add_row("ステップ", f"二乗 {st.n_sq} 回 / 乗算 {st.n_mul} 回")
    t.add_row("ピークメモリ", fmt_bytes(st.mem_peak) if st.mem_peak else "—")
    t.add_row("", "")
    tm = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    tm.add_column("フェーズ")
    tm.add_column("時間", justify="right")
    tm.add_column("割合", justify="right")
    for i, name in enumerate(PHASES):
        pt = st.phase_time(i)
        if i in st.skipped or pt is None:
            continue
        tm.add_row(name, fmt_time(pt), f"{pt / max(r.total, 1e-9) * 100:4.1f}%")
    tm.add_row("[bold]合計[/]", f"[bold]{fmt_time(r.total)}[/]", "100%")
    rate = f"{digits / comp:,.0f} 桁/秒" if comp > 0.01 else "—"
    perf = Table.grid(padding=(0, 2))
    perf.add_row("[dim]本計算スループット[/]", rate, "[dim]全体スループット[/]", f"{digits / max(r.total, 1e-9):,.0f} 桁/秒")
    return Panel(Group(t, tm, perf), title="[bold green]✔ 計算完了[/]", border_style="green", box=box.ROUNDED, padding=(1, 2))


# ─────────────────────────────── セルフテスト ───────────────────────────────
def run_selftest(args, console: Console) -> int:
    import random

    rng = random.Random(2024)
    cases = [(2, 1000), (3, 777), (10, 123), (12000, 57), (99999, 41), (-7, 301), (-6, 300), (10**40 + 12345, 29),
             (2**70 + 1, 33), (5, 5000), (999, 400), (10**20 - 1, 31), (1, 999), (0, 5), (7, 0)]
    cases += [(rng.randrange(2, 10 ** rng.randrange(1, 30)), rng.randrange(1, 400)) for _ in range(25)]
    kinds = detect_backends(args.backend)
    bad = 0
    tb = Table(title="自己テスト (Python の pow と全桁照合)", box=box.SIMPLE_HEAD)
    tb.add_column("backend")
    tb.add_column("base")
    tb.add_column("OK/全件", justify="right")
    for kind in kinds:
        for B in ([args.base] if args.base else list(BASES)):
            ok = 0
            for x, y in cases:
                job = Job(x, y, prefix_digits=25)
                st = State()
                try:
                    rep, _ = (Rep(s=job.trivial, B=1000), 0) if job.trivial else compute(job, kind, st, B)
                    got = job.sign + rep_to_str(rep) + "0" * job.zeros
                    want = str(x**y)
                    good = got == want
                    ok += good
                    if not good:
                        console.print(f"[red]不一致 {kind} B={B} x={x} y={y}[/]")
                except Exception as e:
                    console.print(f"[red]{kind} B={B} x={x} y={y}: {e!r}[/]")
            bad += len(cases) - ok
            tb.add_row(kind, f"10^{kdig(B)}", f"{'[green]' if ok == len(cases) else '[red]'}{ok}/{len(cases)}[/]")
    console.print(tb)
    console.print("[bold green]すべて一致[/]" if not bad else f"[bold red]{bad} 件不一致[/]")
    return 1 if bad else 0


# ─────────────────────────────── main ───────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="x^y を CPU/GPU で高速に厳密計算し txt に出力")
    ap.add_argument("x", nargs="?", help="底 (例: 3, -7, 2**64)")
    ap.add_argument("y", nargs="?", help="指数 (例: 100000000, 10**8, 1e8)")
    ap.add_argument("-o", "--out", default="./pow_results", help="出力ディレクトリ (既定: ./pow_results)")
    ap.add_argument("--backend", choices=["auto", "gpu", "cpu", "gmp"], default="auto")
    ap.add_argument("--base", type=int, choices=[10, 100, 1000], help="リム基数を固定 (既定: 自動)")
    ap.add_argument("--selftest", action="store_true", help="自己テストを実行して終了")
    args = ap.parse_args()
    console = Console()

    if args.selftest:
        return run_selftest(args, console)

    try:
        x = parse_int(args.x if args.x is not None else Prompt.ask("[cyan]x[/] (底)"))
        y = parse_int(args.y if args.y is not None else Prompt.ask("[magenta]y[/] (指数)"))
        job = Job(x, y)
    except ValueError as e:
        console.print(f"[red]入力エラー:[/] {e}")
        return 2
    if job.est_digits > 3_000_000_000:
        console.print("[red]結果が 30 億桁を超えるため中止します。[/]")
        return 2

    st = State()
    result: dict = {}

    def worker():
        try:
            result["r"] = pipeline(job, args, st)
        except Cancelled:
            st.error = "ユーザーにより中断されました"
        except BaseException as e:  # noqa: BLE001
            st.error = f"{e.__class__.__name__}: {e}"
            result["tb"] = traceback.format_exc()
        finally:
            st.done = True

    th = threading.Thread(target=worker, daemon=True)
    ui = UI(st, console)
    th.start()
    try:
        with Live(get_renderable=ui.render, console=console, refresh_per_second=12, transient=False):
            while th.is_alive():
                time.sleep(0.05)
    except KeyboardInterrupt:
        st.cancel = True
        console.print("\n[yellow]中断要求を受け付けました。現在のステップ完了後に停止します…[/]")
        th.join()
    if st.error:
        console.print(Panel(f"[red]{st.error}[/]", title="エラー / 中断", border_style="red"))
        if "tb" in result and os.environ.get("POWCALC_DEBUG"):
            console.print(result["tb"])
        return 1
    console.print(summary_panel(job, result["r"], st))
    return 0 if result["r"].ok else 3


if __name__ == "__main__":
    sys.exit(main())
