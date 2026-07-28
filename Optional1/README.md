# Optional 1：LUCJ-SQD `ccsd_scale` 研究材料

## 目录结构

```text
optional1_lucj_sqd_package/
├── README.md
├── report/
│   └── optional1_lucj_sqd_report.tex
├── figures/
│   ├── controlled_experiments.png
│   ├── coverage_saturated.png
│   ├── large_lambda_peaks.png
│   └── h2_sampling_validation.png
├── code/
│   ├── lucj_sqd_framework.py
│   ├── turnaround_study.py
│   ├── controlled_optional1_experiments.py
│   ├── coverage_saturated_experiments.py
│   ├── large_lambda_peak_scan.py
│   └── h2_sampling_validation.py
├── data/
│   ├── controlled_experiments.json
│   ├── coverage_saturated.json
│   ├── large_lambda_peaks.json
│   └── h2_sampling_validation.json
└── environment/
    ├── requirements-lock.txt
    └── sqd_env/                 # 桌面交付副本中包含
```

## 报告

主报告为：

```text
report/optional1_lucj_sqd_report.tex
```

报告使用 `ctexart`，公式均使用原生 LaTeX 数学环境。按要求没有编译 PDF。

如需自行编译，建议从包根目录执行：

```bash
xelatex report/optional1_lucj_sqd_report.tex
xelatex report/optional1_lucj_sqd_report.tex
```

从包根目录编译可以保证 `figures/` 的相对路径正确。需要本机安装 TeX Live/MacTeX 及 `ctex`。

## Python 环境

桌面交付副本包含完整虚拟环境：

```text
environment/sqd_env/
```

使用方式：

```bash
source environment/sqd_env/bin/activate
python -c "import pyscf, ffsim, qiskit_aer; print('environment OK')"
```

若复制后的 venv 激活脚本因路径变化失效，可直接调用：

```bash
environment/sqd_env/bin/python code/h2_sampling_validation.py
```

或者使用锁定依赖重建：

```bash
python3 -m venv environment/sqd_env_rebuilt
environment/sqd_env_rebuilt/bin/python -m pip install -r environment/requirements-lock.txt
```

核心版本：Python 3.13.12、PySCF 2.14.0、ffsim 0.0.83、Qiskit 2.5.1、Qiskit Addon SQD 0.12.1、Qiskit Aer 0.17.2。

## 主要实验代码

- `lucj_sqd_framework.py`：统一分子、LUCJ、采样、噪声和 SQD 接口。
- `turnaround_study.py`：分子、shots、键长、噪声、层数及 λ 参数扫描。
- `controlled_optional1_experiments.py`：固定 K、关键组态覆盖和受控噪声实验。
- `coverage_saturated_experiments.py`：不固定 K 的覆盖饱和实验。
- `large_lambda_peak_scan.py`：H₂/H₂O 大范围 λ 节点自动探测。
- `h2_sampling_validation.py`：H₂ 显式有限-shot 采样和子空间对角化验证。

## 数据说明

`data/` 中 JSON 保存报告图所对应的原始数值。报告中的主要结论可以由这些数据和 `code/` 中脚本独立复查。

## 重要结论边界

- 报告中的有限-shot “实验”是理想模拟器采样，不是物理量子硬件实验。
- H₂ 图同时给出解析期望和显式 ffsim 抽样验证，两者已区分标注。
- SQD 对振幅/相位误差较鲁棒，但不对关键组态零概率或支持缺失鲁棒。
