"""Q03 (a): Configuration recovery algorithm."""


def configuration_recovery(b, n_bar, N_e):
    """
    输入: b (观测比特串), n̄ (平均占据数), N_e (目标电子数)
    输出: d (恢复的合法配置)

    1. d = b
    2. WHILE sum(d) != N_e:
    3.   IF sum(d) > N_e:
    4.     候选 = {i : d_i == 1}
    5.     选 i = argmax (d_i - n̄_i)  // n̄_i 最小的占据比特
    6.     d_i = 0
    7.   ELSE:
    8.     候选 = {i : d_i == 0}
    9.     选 i = argmax (n̄_i - d_i)  // n̄_i 最大的空比特
    10.    d_i = 1
    11. RETURN d
    """
    d = list(b)
    while sum(d) != N_e:
        if sum(d) > N_e:
            candidates = [i for i in range(len(d)) if d[i] == 1]
            i = max(candidates, key=lambda i: d[i] - n_bar[i])
            d[i] = 0
        else:
            candidates = [i for i in range(len(d)) if d[i] == 0]
            i = max(candidates, key=lambda i: n_bar[i] - d[i])
            d[i] = 1
    return d
