"""每题结果存档与一致性检查 helper。"""
from __future__ import annotations

import json
import os
from typing import Dict


def save_results(qid: str, exact: Dict, quantum: Dict, meta: Dict = None,
                 tol: float = 1e-3, out_dir: str = None):
    """把精确解与量子版结果写入 results.json，并打印对比。"""
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "..", qid)
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(qid=qid, exact=exact, quantum=quantum, meta=meta or {},
                   tol=tol)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_default)
    print(f"[{qid}] exact  = {_short(exact)}")
    print(f"[{qid}] quantum= {_short(quantum)}")
    return payload


def check_close(qid: str, exact_val, quantum_val, tol: float = 1e-3,
                label: str = "E"):
    if exact_val is None or quantum_val is None:
        print(f"[{qid}] {label}: 跳过一致性检查（无参照）")
        return
    diff = abs(exact_val - quantum_val)
    ok = diff <= tol
    print(f"[{qid}] {label}: |Δ|={diff:.3e} tol={tol} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _short(d: Dict, n: int = 4):
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in list(d.items())[:n]}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


import numpy as np  # noqa: E402
