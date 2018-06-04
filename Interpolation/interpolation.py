import numpy as np


def newton_interpolation(base: list, val: list, xs: list) -> list:
    diffs = val[:]
    for t in range(len(base) - 1):
        for s in range(len(base) - 1, t, -1):
            diffs[s] = (diffs[s] - diffs[s - 1]) / (base[s] - base[s - t - 1])
    ret = []
    for x in xs:
        predict = 0
        for i in range(len(base)):
            base_predict = diffs[i]
            for j in range(i):
                base_predict *= (x - base[j])
            predict += base_predict
        ret.append(predict)
    return ret


def lagrange_interpolation(base: list, val: list, xs: list) -> list:
    ret = []
    for x in xs:
        predict = 0
        for i in range(len(base)):
            base_predict = val[i]
            for j in range(len(base)):
                if j != i:
                    base_predict *= (x - base[j]) / (base[i] - base[j])
            predict += base_predict
        ret.append(predict)
    return ret


def piecewise_linear_interpolation(base: list, val: list, xs: list) -> list:
    ob, ov = zip(*(sorted(zip(base, val), key=lambda x: x[0])))
    ret = []
    for x in xs:
        if x < ob[0] or x > ob[-1]:
            ret.append(0)
        else:
            for i in range(len(ob) - 1):
                if ob[i] <= x <= ob[i + 1]:
                    val_l = (x - ob[i + 1]) / (ob[i] - ob[i + 1])
                    val_r = (x - ob[i]) / (ob[i + 1] - ob[i])
                    ret.append(ov[i] * val_l + ov[i + 1] * val_r)
                    break
    assert len(ret) == len(xs)
    return ret


def natural_cubic_spline_interpolation(base: list, val: list, xs: list) -> list:
    ob, ov = zip(*(sorted(zip(base, val), key=lambda x: x[0])))
    hs = []
    mus = [None]
    lambdas = [None]
    ds = [None]
    for i in range(len(ob) - 1):
        hs.append(ob[i + 1] - ob[i])
    for i in range(1, len(ob) - 1):
        mus.append(hs[i - 1] / (hs[i - 1] + hs[i]))
        lambdas.append(1 - mus[i])
        diff1 = (ov[i + 1] - ov[i]) / hs[i]
        diff2 = (ov[i] - ov[i - 1]) / hs[i - 1]
        ds.append(6 * (diff1 - diff2) / (ob[i + 1] - ob[i - 1]))
    mat = np.diag([2.0] * (len(ob) - 2))
    mat[0, 1] = lambdas[1]
    mat[-1, -2] = mus[-1]
    for i in range(1, len(ob) - 3):
        mat[i, i - 1] = mus[i + 1]
        mat[i, i + 1] = lambdas[i + 1]
    ms = np.linalg.solve(mat, ds[1:])
    all_ms = [0]
    all_ms.extend(ms)
    all_ms.append(0)
    ms = all_ms

    ret = []
    for x in xs:
        if x < ob[0] or x > ob[-1]:
            ret.append(0)
        else:
            for i in range(len(ob) - 1):
                if ob[i] <= x <= ob[i + 1]:
                    s = 0
                    s += ms[i] * ((ob[i + 1] - x) ** 3) / (6 * hs[i])
                    s += ms[i + 1] * ((x - ob[i]) ** 3) / (6 * hs[i])
                    s += (ov[i] - ms[i] * (hs[i] ** 2) / 6) * \
                        ((ob[i + 1] - x) / hs[i])
                    s += (ov[i + 1] - ms[i + 1] * (hs[i] ** 2) / 6) * \
                        ((x - ob[i]) / hs[i])
                    ret.append(s)
                    break
    assert len(ret) == len(xs)
    return ret
