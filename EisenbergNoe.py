import torch
import numpy as np
from scipy.sparse import csr_matrix
from torch import Tensor


def get_clearing_vector_iter_from_batch(batch_of_liab:Tensor, batch_of_assets:Tensor, batch_of_outgoing_liab:Tensor, n_iter: int = 50):
    cv = batch_of_assets * 0
    batch_of_pi_trp = torch.nan_to_num(batch_of_liab / batch_of_outgoing_liab, nan=0, posinf=0, neginf=0).transpose(-2,-1)

    for i in range(n_iter):
        cv = torch.baddbmm(batch_of_assets,batch_of_pi_trp,cv)
        cv = torch.minimum(cv, batch_of_outgoing_liab)
    return cv

