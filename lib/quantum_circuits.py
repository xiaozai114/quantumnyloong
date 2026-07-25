"""
量子电路类 - 存储本项目用到的可逆/量子电路。
"""


class QuantumCircuits:
    """可逆与量子电路的集合。

    使用 Toffoli (CCNOT) 和 NOT 门构建可逆经典电路。
    """

    @staticmethod
    def build_s3(c, inputs=(0, 1, 2), result=4, ancilla=3):
        """
        在已有电路 c 上构建 S3(x) 可逆电路。

        布尔函数: S3(x) = not x1 and not x2 and x3
        (可由 5 子句 CNF 化简得到)

        电路结构:
          1. X(x1), X(x2)            -- 翻转控制位
          2. CCNOT(x1, x2 -> anc)     -- anc = not x1 and not x2
          3. CCNOT(anc, x3 -> result) -- result = anc and x3
          4. CCNOT(x1, x2 -> anc)     -- 恢复 anc (uncompute)
          5. X(x1), X(x2)            -- 恢复 x1, x2

        资源统计:
          Gates:  4 NOT + 3 Toffoli = 7
          Qubits: 3 (input) + 1 (ancilla) + 1 (result) = 5
          Product: 7 x 5 = 35
        """
        x1, x2, x3 = inputs

        # Step 1
        c.x(x1)
        c.x(x2)

        # Step 2
        c.toffoli(x1, x2, ancilla)

        # Step 3
        c.toffoli(ancilla, x3, result)

        # Step 4 (uncompute ancilla)
        c.toffoli(x1, x2, ancilla)

        # Step 5 (uncompute inputs)
        c.x(x1)
        c.x(x2)
