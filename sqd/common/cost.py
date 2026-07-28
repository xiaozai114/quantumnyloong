"""代价汇总工具：把每题的代价指标聚合成表格并写出 cost_summary.md。"""
from __future__ import annotations

import os
from typing import Dict, List


def cost_row(qid: str, title: str, metrics: Dict) -> Dict:
    row = {"qid": qid, "title": title}
    row.update(metrics)
    return row


def render_summary(rows: List[Dict], out_path: str):
    if not rows:
        return
    keys = ["qid", "title"] + [k for k in rows[0].keys()
                                if k not in ("qid", "title")]
    lines = ["# SQD Practice (TensorCircuit) 成本汇总", "",
             "| 题目 | 主题 | " +
             " | ".join(k for k in keys if k not in ("qid", "title")) + " |",
             "|------|------|" +
             "------|" * len([k for k in keys if k not in ("qid", "title")])]
    for r in rows:
        cells = [r.get("qid", ""), r.get("title", "")]
        for k in keys:
            if k in ("qid", "title"):
                continue
            v = r.get(k, "")
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
