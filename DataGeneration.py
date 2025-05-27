import dgl
import scipy.stats
import numpy as np
import torch
import os


class DataGenerator:

    def __init__(self, datatype, path_to_fixed_liability_graph=None):
        self.path_to_fixed_liability_graph = path_to_fixed_liability_graph
        self.datatype = datatype
        if self.datatype == 'CPf':
            g = dgl.load_graphs(path_to_fixed_liability_graph)[0][0]
            self.liab = get_liability_matrix(g).numpy()
        else:
            self.liab = None

    def get_graphs(self, num_graphs):
        if self.datatype == 'ER':
            return get_er_graphs(num_graphs)
        elif self.datatype == 'CP':
            return get_cp_graphs(num_graphs)
        elif self.datatype == 'CPf':
            return get_cpf_graphs(num_graphs, liab = self.liab)


### Erdos Renyi
def create_er_adj_matrix(n, p):
    '''
    Creates an n*n adjacency matrix from input dimension n and edge probability p.
    :param n:
    :param p:
    :return:
    '''
    adj = np.random.uniform(0, 1, size=(n, n))
    adj = (adj <= p).astype(int)
    np.fill_diagonal(adj, 0)
    edges = np.nonzero(adj)
    return adj, edges

def get_er_graphs(num_graphs: int = 100):
    num_nodes = 100
    p_link = 0.4
    asset_factor = 10
    list_of_graphs = []
    list_of_assets = []
    list_of_liab = []
    list_of_outgoing_liab = []
    for i in range(num_graphs):
        adj, [src, dst] = create_er_adj_matrix(n=num_nodes, p=p_link)
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['weight'] = torch.FloatTensor(np.full(len(src), fill_value=1.0))
        factor = torch.full((num_nodes,), asset_factor)
        assets = torch.Tensor(get_gaussian_copula_beta_assets(num_nodes=num_nodes, correlation=0)) * factor
        liab_tensor = torch.FloatTensor(adj)
        incoming_liabilities = liab_tensor.sum(dim=0)
        outgoing_liabilities = liab_tensor.sum(dim=1)
        g.ndata['assets'] = assets
        g.ndata['outgoingLiabilities'] = outgoing_liabilities
        g.ndata['incomingLiabilities'] = incoming_liabilities

        list_of_graphs.append(g)
        list_of_liab.append(liab_tensor)  # all liabilities are of size 1, hence can use adj directly
        list_of_outgoing_liab.append(outgoing_liabilities.unsqueeze(-1))
        list_of_assets.append(assets.unsqueeze(-1))

        useful_tensor = (torch.stack(list_of_liab, dim=0),
                          torch.stack( list_of_assets, dim=0),
                          torch.stack(list_of_outgoing_liab, dim=0))

    return list_of_graphs, useful_tensor

### Core Periphery
def create_cp_liab_matrix(n):
    ###
    group_sizes = [10, 90]
    group_link_probability = np.array([[0.7, 0.3],[0.3, 0.1]])
    group_liabilities = np.array([[10, 2],[2, 1]])
    adj = np.random.uniform(0, 1, size=(n, n))
    liab = adj.copy()

    indices = np.insert(np.cumsum(group_sizes), 0, 0.)
    for i in range(len(group_sizes)):
        for j in range(len(group_sizes)):
            liab[indices[i]:indices[i + 1], indices[j]:indices[j + 1]] = group_liabilities[i,j] * (
                adj[indices[i]:indices[i + 1], indices[j]:indices[j + 1]] <= group_link_probability[i, j])

    np.fill_diagonal(liab, 0)
    return liab

def get_cp_graphs(num_graphs):
    num_nodes = 100
    list_of_graphs = []
    list_of_assets = []
    list_of_liab = []
    list_of_outgoing_liab = []
    for i in range(num_graphs):
        liab = create_cp_liab_matrix(n=num_nodes)
        factor = torch.cat((torch.full((10,), 50), torch.full((90,), 10)))
        assets = factor * torch.Tensor(get_gaussian_copula_beta_assets(num_nodes=num_nodes, correlation=0.5))

        idx = np.arange(num_nodes)
        idx_perm = np.arange(num_nodes)
        np.random.shuffle(idx_perm)
        assets[idx_perm] = assets[idx]
        liab[idx_perm,:] = liab[idx,:]
        liab[:, idx_perm] = liab[:, idx]

        liab_tensor = torch.FloatTensor(liab)
        outgoing_liabilities = liab_tensor.sum(dim=1)
        incoming_liabilities = liab_tensor.sum(dim=0)

        adj = (liab > 0).astype(int)
        src, dst = np.nonzero(adj)
        g = dgl.graph((src, dst), num_nodes=100)
        g.edata['weight'] = torch.FloatTensor(liab[src,dst])
        g.ndata['assets'] = assets  # add in- and out-liab later after permutation
        g.ndata['outgoingLiabilities'] = outgoing_liabilities
        g.ndata['incomingLiabilities'] = incoming_liabilities

        list_of_graphs.append(g)
        list_of_liab.append(liab_tensor)  # all liabilities are of size 1, hence can use adj directly
        list_of_assets.append(assets.unsqueeze(-1))
        list_of_outgoing_liab.append(outgoing_liabilities.unsqueeze(-1))

        useful_tensor = (torch.stack(list_of_liab, dim=0), torch.stack(list_of_assets, dim=0),
                         torch.stack(list_of_outgoing_liab, dim=0))

    return list_of_graphs, useful_tensor



### Core Periphery fixed
def get_cpf_graphs(num_graphs, liab):
    num_nodes = 100
    list_of_graphs = []
    list_of_assets = []
    list_of_liab = []
    list_of_outgoing_liab = []
    for i in range(num_graphs):
        factor = torch.cat((torch.full((10,), 50), torch.full((90,), 10)))
        assets = factor * torch.Tensor(get_gaussian_copula_beta_assets(num_nodes=num_nodes, correlation=0.5))

        adj = (liab > 0).astype(int)
        src, dst = np.nonzero(adj)
        g = dgl.graph((src, dst), num_nodes=100)
        g.edata['weight'] = torch.FloatTensor(liab[src,dst])
        g.ndata['assets'] = assets  # add in- and out-liab later after permutation

        liab_tensor = torch.from_numpy(liab)
        outgoing_liabilities = liab_tensor.sum(dim=1)
        incoming_liabilities = liab_tensor.sum(dim=0)

        # complete graph ndata
        g.ndata['outgoingLiabilities'] = outgoing_liabilities
        g.ndata['incomingLiabilities'] = incoming_liabilities

        list_of_graphs.append(g)
        list_of_liab.append(liab_tensor)  # all liabilities are of size 1, hence can use adj directly
        list_of_assets.append(assets.unsqueeze(-1))
        list_of_outgoing_liab.append(outgoing_liabilities.unsqueeze(-1))

        useful_tensor = (torch.stack(list_of_liab, dim=0),
                         torch.stack(list_of_assets, dim=0),
                         torch.stack(list_of_outgoing_liab, dim=0))

    return list_of_graphs, useful_tensor

### General Methods
def get_gaussian_copula_beta_assets(num_nodes, correlation):
    cov = np.full((num_nodes, num_nodes), correlation)
    np.fill_diagonal(cov, 1)
    gauss_sample = np.random.multivariate_normal(mean=np.zeros(shape=num_nodes), cov=cov)
    corr_unif = scipy.stats.multivariate_normal.cdf(gauss_sample)
    assets = scipy.stats.beta.ppf(corr_unif, 2.0, 5.0)
    return assets

def get_liability_matrix(g, weight_name='weight'):
    # get edge source and destination
    src, dst = g.edges()
    # initialize zeros matrix
    liab = torch.zeros(g.num_src_nodes(), g.num_dst_nodes())
    # fill entries according to edges with corresponding edata
    liab[src, dst] = g.edata[weight_name].squeeze(-1)
    return liab

def get_graphs_from_list(list_of_graphs:list):
    list_of_assets = []
    list_of_liab = []
    list_of_outgoing_liab = []
    for g in list_of_graphs:
        liab_tensor = get_liability_matrix(g)
        assets = g.ndata['assets']
        outgoing_liabilities = liab_tensor.sum(dim=1)
        list_of_liab.append(liab_tensor)
        list_of_outgoing_liab.append(outgoing_liabilities.unsqueeze(-1))
        list_of_assets.append(assets.unsqueeze(-1))
    useful_tensor = (torch.stack(list_of_liab, dim=0),
                     torch.stack(list_of_assets, dim=0),
                     torch.stack(list_of_outgoing_liab, dim=0))
    return list_of_graphs, useful_tensor

def get_list_of_batches_from_list_of_graphs(number_of_grad_steps, list_of_all_graphs):
    batch_size = round(len(list_of_all_graphs)/number_of_grad_steps)
    list_of_batches = [list_of_all_graphs[i*batch_size:(i+1)*batch_size] for i in range(len(list_of_all_graphs)//batch_size)]
    return list_of_batches

def permute_graph(g):
    arr = np.arange(g.num_nodes())
    np.random.shuffle(arr)
    g_permuted =  dgl.reorder_graph(g, node_permute_algo='custom', store_ids=False, permute_config={'nodes_perm': arr})
    return g_permuted


if __name__== "__main__":
    ### create validation and test set for experiments

    # get path
    path_to_package = os.path.abspath(os.getcwd()).split('ComputingSystemicRiskMeasures')[0] + 'ComputingSystemicRiskMeasures/'

    # ER data
    list_of_er_graphs, _ = get_er_graphs(5)
    dgl.save_graphs(path_to_package + "Data/5000_beta_assetFactor_erdos_renyi_10_04.bin", list_of_er_graphs)

    # CP data
    list_of_cp_graphs, _ = get_cp_graphs(5)
    dgl.save_graphs(path_to_package + "Data/5000_beta_assetFactor_50_10_shuffled.bin'", list_of_cp_graphs)

    # CPf data
    data_generator = DataGenerator('CPf',
                                   path_to_package + 'Data/5000_beta_assetFactor_50_10_fixed_idx0_unshuffled[0].bin')
    list_of_cpf_graphs, _ = data_generator.get_graphs(5)
    dgl.save_graphs(path_to_package + "Data/5000_beta_assetFactor_50_10_fixed_idx0_unshuffled.bin", list_of_cpf_graphs)


    print('Done')