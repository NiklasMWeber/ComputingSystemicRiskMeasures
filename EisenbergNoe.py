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


# def get_clearing_vector_gnn_iter_from_batch_graph():
#     pass


# def get_liability_sum_sparse(g, weight_name='weight', type='out'):
#     numNodes = g.num_nodes()
#     if type == 'in':
#         axis = 0
#     else:
#         axis = 1
#     # get edge source and destination
#     src, dst = g.edges()
#     # get weights
#     weights = g.edata[weight_name]
#     # build sparse matrix
#     liab = csr_matrix((weights.detach().numpy(), (src.detach().numpy(), dst.detach().numpy())), shape = (numNodes, numNodes))
#     # sum along axis
#     sum = np.asarray(liab.sum(axis=axis))
#     sum = sum.squeeze()
#
#     return torch.Tensor(sum)


# class EisenbergNoeFast:
#     def __init__(self, alpha: float, beta: float, bailout_capital: torch.Tensor, device='cpu'):
#         self.alpha = alpha
#         self.beta = beta
#         self.bailout_capital = bailout_capital
#         self.device = device
#
#     def clear_cv(self, g: dgl.DGLGraph, bailout_ratio: torch.Tensor) -> torch.Tensor:
#         return self.clear(g, bailout_ratio, return_type='cv')
#
#     def clear_nv(self, g: dgl.DGLGraph, bailout_ratio: torch.Tensor) -> torch.Tensor:
#         return self.clear(g, bailout_ratio, return_type='nv')
#
#     def clear(self, g: dgl.DGLGraph, bailout_ratio: torch.Tensor, return_type='cv') -> torch.Tensor:
#         L = get_liability_matrix(g, device=self.device)
#         L_hat = torch.sum(L, dim=1)
#         pi = torch.stack(
#             [L[i, :] / L_hat[i] if L_hat[i] != 0 else torch.zeros(len(L[i, :]), device=self.device) for i in
#              range(len(L_hat))])
#         a = g.ndata['assets']
#
#         a_bail = self.bailout_capital * bailout_ratio
#
#         insolvent_banks = [torch.LongTensor([]).to(self.device)]
#         sets_are_equal = False
#         mu = 0
#         clearing_vector = L_hat.clone().detach()
#         while (not sets_are_equal) and mu < 100:
#             v = a + a_bail - L_hat + torch.matmul(torch.transpose(pi, 0, 1), clearing_vector)
#             insol = ((v < 1e-10).nonzero(as_tuple=True))[0]  # Tol for insol to account for calc inaccuracies
#             sol = ((v >= 1e-10).nonzero(as_tuple=True))[0]
#
#             sets_are_equal = torch.equal(insolvent_banks[mu], insol)
#             if sets_are_equal:
#                 break
#
#             insolvent_banks.append(insol)
#
#             E = torch.eye(len(insol), device=self.device)
#             A = E - self.beta * (pi[insol][:, insol]).transpose(0, 1)
#             B = self.alpha * a[insol] + a_bail[insol] + self.beta * torch.matmul((pi[sol][:, insol]).transpose(0, 1),
#                                                                                  L_hat[sol])
#             x = torch.linalg.solve(A, B)
#             clearing_vector[insol] = x
#
#             mu += 1
#         node_value = a + a_bail + torch.matmul(pi.transpose(0, 1), clearing_vector) - L_hat
#         if return_type == 'cv':
#             return clearing_vector
#         else:
#             return node_value
#
#
# class EisenbergNoe:
#     def __init__(self, alpha=1.0, beta=1.0):
#         with torch.no_grad():
#             self.clearingVector = torch.Tensor()
#             self.nodeValue = torch.Tensor()
#             self.outgoingLiabilities = torch.Tensor()
#             self.incomingLiabilities = torch.Tensor()
#             self.insolvencyLevel = None
#         self.alpha = alpha
#         self.beta = beta
#         self.insolventNodes = []
#
#     def clear(self, g):
#
#         with torch.no_grad():
#             L = get_liability_matrix(g)
#             L_hat = torch.sum(L, dim=1)
#             pi = torch.stack(
#                 [L[i, :] / L_hat[i] if L_hat[i] != 0 else torch.zeros(len(L[i, :])) for i in
#                  range(len(L_hat))])
#             a = g.ndata['assets']
#
#             insolvent_banks = [torch.LongTensor([])]
#             # solvent_banks = []
#             sets_are_equal = False
#             mu = 0
#             # counter = 0
#             self.clearingVector = L_hat.clone().detach()
#             while not sets_are_equal and mu < 100:
#                 # counter += 1 # will exit while loop in case of oscillatijng insolvency sets
#                 v = a - L_hat + torch.matmul(torch.transpose(pi, 0, 1), self.clearingVector)
#                 insol = ((v < 1e-10).nonzero(as_tuple=True))[0]
#                 sol = ((v >= 1e-10).nonzero(as_tuple=True))[0]
#
#                 # insol_new = set(insol.numpy()).union(set(insolvent_banks[mu].numpy()))
#                 sets_are_equal = torch.equal(insolvent_banks[mu], insol)
#                 if sets_are_equal: break
#
#                 insolvent_banks.append(insol)
#                 # solvent_banks.append(sol)
#
#                 E = torch.eye(len(insol))
#                 A = E - self.beta * (pi[insol][:, insol]).transpose(0, 1)
#                 B = self.alpha * a[insol] + self.beta * torch.matmul((pi[sol][:, insol]).transpose(0, 1), L_hat[sol])
#                 x = torch.linalg.solve(A, B)
#
#                 self.clearingVector[insol] = x
#
#                 mu += 1
#         # test1 = torch.matmul(pi.transpose(0, 1), self.clearingVector)
#         self.nodeValue = a + torch.matmul(pi.transpose(0, 1), self.clearingVector) - L_hat
#         self.insolventNodes = insolvent_banks[1:]
#         self.outgoingLiabilities = get_liability_sum_sparse(g, 'weight', 'out')
#         self.incomingLiabilities = get_liability_sum_sparse(g, 'weight', 'in')
#
#         self.insolvencyLevel = torch.zeros(len(self.clearingVector))
#         count = len(insolvent_banks[1:])
#         for i in range(count):
#             self.insolvencyLevel[insolvent_banks[-(i+1)]] = count - (i) * torch.ones(len(insolvent_banks[-(i+1)]))
#             # 0: solvent, 1: insolvent from beginning, 2: insolvent after 1 round of default propagation, 3: ...
#
#         return self.clearingVector


# if __name__=='__main__':
#
#     ## 1. basic functionality of get_clearing_vector_iter_from_batch
#     print('1. Test basic functionality:')
#     list_of_graphs,  useful_tensor = DataGeneration.get_er_graphs(2,  5, 0.4,10)
#     (batch_of_liab, batch_of_assets, batch_of_outgoing_liab, batch_of_incoming_liab) = useful_tensor
#     cv = get_clearing_vector_iter_from_batch(batch_of_liab, batch_of_assets, batch_of_outgoing_liab, 50)
#     print('No errors')
#
#     ## 2. correctness of get_clearing_vector_iter_from_batch and EisenbergNoeGNN
#     print('2. Test general correctness in deterministic test case')
#     batch_of_liab = torch.Tensor([[[0,0,0],[2,0,2],[3,2,0]],
#                                   [[0,0,0],[2,0,2],[3,2,0]]])
#
#     batch_of_outgoing_liab = batch_of_liab.sum(dim=-1, keepdim=True)
#     # batch_of_pi = torch.nan_to_num(batch_of_liab / batch_of_outgoing_liab, nan=0, posinf=0, neginf=0)
#     # batch_of_pi_trp = batch_of_pi.transpose(-2, -1)
#     batch_of_assets = torch.Tensor([[[0],[2],[2]],
#                                   [[0],[2],[2]]])
#
#     cv = get_clearing_vector_iter_from_batch(batch_of_liab, batch_of_assets, batch_of_outgoing_liab, n_iter=50)
#
#     en = EisenbergNoeFast(1.0,1.0, torch.Tensor([0]))
#     src, dst = torch.nonzero(batch_of_liab[0], as_tuple = True)
#     weight = batch_of_liab[0,src,dst].unsqueeze(-1)
#     g = dgl.graph((src, dst), num_nodes=3)
#     g.ndata['assets'] = batch_of_assets[0].squeeze()
#     g.edata['weight'] = torch.FloatTensor(weight)
#
#     cv_old = en.clear(g, batch_of_assets[0].squeeze()*0)
#
#     # en_gnn = EisenbergNoeGNN()
#     # batched_graph = dgl.batch(list_of_graphs)
#     # all_assets = batch_of_assets.reshape(-1, list_of_graphs[0].num_nodes())
#     # cv_gnn = en_gnn.clear()
#
#     print(f"Summed diff between cv and cv_old for n_iter=50 is: {(cv[0] - cv_old).sum()}")
#
#     ## 3. convergence of get_cv_... and speed
#     print('3. Test convergence and duration for random batch')
#     list_of_graphs, useful_tensor = DataGeneration.get_er_graphs(100, 50, 0.4, 10)
#     (batch_of_liab, batch_of_assets, batch_of_outgoing_liab, batch_of_incoming_liab) = useful_tensor
#
#     for n_iter in [1,5,10,20,50,100]:
#
#         start_cv = time.time()
#         cv = get_clearing_vector_iter_from_batch(batch_of_liab, batch_of_assets, batch_of_outgoing_liab, n_iter)
#         end_cv = time.time()
#         d_cv = end_cv - start_cv
#
#         start_cv_old = time.time()
#         en = EisenbergNoeFast(1.0,1.0, torch.Tensor([0]))
#         cv_old_list = []
#         for g in list_of_graphs:
#             cv_old = en.clear(g, g.ndata['assets']*0)
#             cv_old_list.append(cv_old)
#         cv_old = torch.stack(cv_old_list, dim=0).unsqueeze(-1)
#         end_cv_old = time.time()
#         d_cv_old = end_cv_old - start_cv_old
#
#         delta_value = (cv - cv_old).abs().sum()
#
#         print(f"n_iter: {n_iter}\t d_cv: {d_cv}\t d_cv_old: {d_cv_old}\t diff: {delta_value}")
#
#
#     ## grad or not, does it make a difference?
#     print('4. Test convergence and duration with gradient for random batch')
#     list_of_graphs, useful_tensor = DataGeneration.get_er_graphs(100, 50, 0.4, 10)
#     (batch_of_liab, batch_of_assets, batch_of_outgoing_liab, batch_of_incoming_liab) = useful_tensor
#
#     for g in list_of_graphs:
#         g.ndata['assets'].requires_grad = True
#
#     batch_of_assets.requires_grad = True
#
#     for n_iter in [1, 5, 10, 20, 50, 100]:
#
#         start_cv = time.time()
#         cv = get_clearing_vector_iter_from_batch(batch_of_liab, batch_of_assets, batch_of_outgoing_liab, n_iter)
#         end_cv = time.time()
#         d_cv = end_cv - start_cv
#
#         start_cv_old = time.time()
#         en = EisenbergNoeFast(1.0, 1.0, torch.Tensor([0]))
#         cv_old_list = []
#         for g in list_of_graphs:
#             cv_old = en.clear(g, g.ndata['assets'] * 0)
#             cv_old_list.append(cv_old)
#         cv_old = torch.stack(cv_old_list, dim=0).unsqueeze(-1)
#         end_cv_old = time.time()
#         d_cv_old = end_cv_old - start_cv_old
#
#         delta_value = (cv - cv_old).abs().sum()
#
#         print(f"With grad: n_iter: {n_iter}\t d_cv: {d_cv}\t d_cv_old: {d_cv_old}\t diff: {delta_value}")

    ## timing
    ## gnn



    # en = EisenbergNoe(1., 1.)

    # # Example 1
    # g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]), num_nodes=3)
    #
    # g1 = g
    #
    # sum_in = get_liability_sum_sparse(g, 'weight', type = 'in')
    # print(f'Liability sum in: {sum_in}')
    # sum_out = get_liability_sum_sparse(g, 'weight', type='out')
    # print(f'Liability sum out: {sum_out}')
    #
    # cv1 = en.clear(g1)
    # print(f'Clearing Vector: {cv1}')
    # nv1 = en.nodeValue
    # print(f'Node Values: {nv1}')
    # print(f'Insolvent Nodes: {en.insolventNodes}')
    # print(f'Insolvency levels: {en.insolvencyLevel}')
    # print(f'Outgoing Liabilities: {en.outgoingLiabilities}')
    # print(f'Incoming Liabilities: {en.incomingLiabilities}')
    # # Example 2
    # g = dgl.graph(([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]))
    # g.ndata['assets'] = torch.FloatTensor([1, 2, 3, 100])
    # g.edata['weight'] = torch.FloatTensor([3, 0, 6, 2, 8, 1])
    # g2 = g
    #
    # sum_in = get_liability_sum_sparse(g, 'weight', type='in')
    # print(f'Liability sum in: {sum_in}')
    # sum_out = get_liability_sum_sparse(g, 'weight', type='out')
    # print(f'Liability sum out: {sum_out}')
    #
    # cv2 = en.clear(g2)
    # print(f'Clearing Vector: {cv2}')
    # nv2 = en.nodeValue
    # print(f'Node Values: {nv2}')
    # print(f'Insolvent Nodes: {en.insolventNodes}')
    # print(f'Insolvency levels: {en.insolvencyLevel}')
    # print(f'Outgoing Liabilities: {en.outgoingLiabilities}')
    # print(f'Incoming Liabilities: {en.incomingLiabilities}')

    # print('Done')
