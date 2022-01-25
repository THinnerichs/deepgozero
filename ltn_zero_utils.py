import torch


# see https://bmxitalia.github.io/LTNtorch/fuzzy_ops.html#ltn.fuzzy_ops.AggregPMeanError for more information
def Forall(mat, dim, p=2):
    out = 1 - ((1-mat ** p).sum(dim=dim) ** (1 / p))  # \Forall x
    return out


def Exists(mat, dim, p=2):
    out = ((mat ** p).sum(dim=dim) ** (1 / p))  # \Exists x
    return out


def And(mat_c, mat_d, tnorm='prod'):
    if mat_c.size() != mat_d.size():
        raise ValueError("mat_c and mat_d have incompatible shapes.")
    if tnorm == 'prod':
        return mat_c * mat_d  # element-wise multiplication
    else:
        raise ValueError("Invalid t-norm selected in And.")


def Implies(mat_c, mat_d, method='reichenbach'):
    if mat_c.size() != mat_d.size():
        raise ValueError("mat_c and mat_d have incompatible shapes.")
    if method == 'reichenbach':
        return 1 - mat_c + mat_c * mat_d  # Reichenbach Fuzzy Implication
    else:
        raise ValueError("Invalid fuzzy implication selected in Implies.")


def sparse_dense_mul(s, d):
    """
    Implement element-wise multiplication of a sparse matrix and a dense one
    :param s:  sparse matrix
    :param d:  dense matrix
    :return:   element-wise product as sparse tensor
    """
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.Tensor(i, v * dv, s.size())


def sparse_repeat(s, reps, dim=0):
    return torch.stack([s for _ in range(reps)], dim=dim)
