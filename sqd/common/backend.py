"""量子计算后端统一配置（单一可信源）。

策略（auto）：
- 有 NVIDIA GPU 且 jax 可用 → jax + GPU（电路态矢量在 GPU 上算）。
- 否则 → numpy（无 GPU 时 jax 的 XLA 编译开销大于收益）。

覆盖方式：
- 环境变量 ``TC_BACKEND`` = ``jax`` / ``numpy`` / ``auto``（默认 auto）。
- 或 ``init_backend(prefer="jax")``。

所有模块只需 ``import common.backend``（导入即初始化），``tc.set_backend``
集中于此。后端信息见 ``BACKEND`` / ``HAS_GPU`` / ``DEVICE`` / ``N_CPUS``。
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

# numpy 2.0 移除了顶层别名（np.ComplexWarning 等），老版 tensorcircuit/openfermion
# 仍引用——在导入任何依赖 numpy 的包之前补齐别名，保证 numpy 2.x 兼容。
if not hasattr(np, "ComplexWarning"):
    np.ComplexWarning = np.exceptions.ComplexWarning
for _alias, _target in [("float", "float64"), ("int", "int_"),
                         ("complex", "complex128"), ("bool", "bool_")]:
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(np, _target))

# openfermion 的 SymbolicOperator 在 numpy 2.x 下用 isinstance(coeff, int/float)
# 拒绝 numpy 标量（numpy 2.0 不再把 numpy.int64 注册为 int 子类）。patch 其
# __init__ 把 numpy 标量转原生 Python 类型，使 jw/parity/bk 变换不报错。
try:
    from openfermion.ops.operators.symbolic_operator import \
        SymbolicOperator as _SymOp
    _orig_sym_init = _SymOp.__init__

    def _np_compat_sym_init(self, *args, **kwargs):
        # openfermion 签名 (term, coefficient)；coefficient 可能在任意位置或 kwargs
        args = tuple(a.item() if isinstance(a, np.generic) else a for a in args)
        if "coefficient" in kwargs and isinstance(kwargs["coefficient"], np.generic):
            kwargs["coefficient"] = kwargs["coefficient"].item()
        _orig_sym_init(self, *args, **kwargs)

    _SymOp.__init__ = _np_compat_sym_init
except Exception:
    pass  # openfermion 未安装时跳过

BACKEND = "numpy"
HAS_GPU = False
DEVICE = "cpu"
N_CPUS = os.cpu_count() or 4


def _has_nvidia_gpu() -> bool:
    """用 nvidia-smi 探测 GPU（不触发 jax 的 cuda 插件，避免无 GPU 时的噪声）。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _jax_has_gpu() -> bool:
    try:
        import jax  # noqa
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _apply(name: str, gpu: bool, device: str) -> None:
    global BACKEND, HAS_GPU, DEVICE
    import tensorcircuit as tc
    tc.set_backend(name)
    BACKEND, HAS_GPU, DEVICE = name, gpu, device


def init_backend(prefer: str | None = None) -> str:
    """初始化 TC 后端，返回后端名。导入 common.backend 时自动调用一次。"""
    prefer = (prefer or os.environ.get("TC_BACKEND", "auto")).lower()

    if prefer in ("numpy", "np"):
        _apply("numpy", False, "cpu")
        return BACKEND
    if prefer in ("jax",):
        gpu = _jax_has_gpu()
        _apply("jax", gpu, "gpu" if gpu else "cpu")
        return BACKEND

    # auto：有 GPU 才上 jax，否则 numpy
    if _has_nvidia_gpu() and _jax_has_gpu():
        _apply("jax", True, "gpu")
    else:
        # 无 GPU：抑制 jax cuda 插件的报错噪声（若后续有代码 import jax）
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        _apply("numpy", False, "cpu")
    return BACKEND


# 导入即初始化
init_backend()
